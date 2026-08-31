#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fusion_model.py

Main Model: TimesNet Chart Encoder + News Transformer Encoder + Time-wise Fusion Transformer

핵심 구조:
1. Chart
   X_chart [B, 72, 12]
   -> TimesNetEncoder
   -> chart_seq [B, 72, chart_hidden]

2. News
   X_news [B, 72, 9]
   -> NewsTransformerSequenceEncoder
   -> news_seq [B, 72, news_hidden]

3. Time-wise Fusion
   chart_seq[t] + news_seq[t]
   -> projection + gated fusion
   -> fusion_tokens [B, 72, fusion_hidden]

4. Fusion Transformer
   [CLS] + fusion_tokens
   -> Fusion Transformer
   -> cls + mean_pool + last_pool
   -> classifier
   -> logits [B, 3]

Label:
0 = SHORT
1 = HOLD
2 = LONG
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn


# ======================================================
# Import chart encoder
# ======================================================
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from chart_only.timesnet_encoder import TimesNetEncoder
except ImportError as e:
    raise ImportError(
        "chart_only/timesnet_encoder.py 안에 TimesNetEncoder 클래스가 있는지 확인해줘.\n"
        "예상 경로: chart_only/timesnet_encoder.py"
    ) from e


# ======================================================
# News Transformer Sequence Encoder
# ======================================================
class NewsTransformerSequenceEncoder(nn.Module):
    """
    뉴스 feature 시계열을 sequence embedding으로 변환.

    Input:
        news_window: [B, 72, 9]

    Output:
        news_seq: [B, 72, hidden_size]
    """

    def __init__(
        self,
        input_size: int = 72,
        input_dim: int = 9,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.30,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_dim, hidden_size)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, input_size, hidden_size)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, news_window):
        if news_window.dim() != 3:
            raise ValueError(
                f"news_window는 [B, T, F] 형태여야 합니다. 현재: {news_window.shape}"
            )

        B, T, F_dim = news_window.shape

        if T != self.input_size:
            raise ValueError(
                f"news input length 불일치: expected {self.input_size}, got {T}"
            )

        if F_dim != self.input_dim:
            raise ValueError(
                f"news input dim 불일치: expected {self.input_dim}, got {F_dim}"
            )

        x = self.input_proj(news_window)
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout(x)

        x = self.encoder(x)
        x = self.final_norm(x)

        return x


# ======================================================
# Chart TimesNet Sequence Encoder Wrapper
# ======================================================
class ChartTimesNetSequenceEncoder(nn.Module):
    """
    TimesNetEncoder wrapper.

    Input:
        chart_window: [B, 72, 12]

    Output:
        chart_seq: [B, 72, hidden_size]
    """

    def __init__(
        self,
        input_size: int = 72,
        input_dim: int = 12,
        prediction_horizon: int = 24,
        hidden_size: int = 32,
        dropout: float = 0.30,
        conv_hidden_size: int = 64,
        top_k: int = 2,
        num_kernels: int = 4,
        encoder_layers: int = 1,
        expand_to_future: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.encoder = TimesNetEncoder(
            input_size=input_size,
            input_dim=input_dim,
            h=prediction_horizon,
            hidden_size=hidden_size,
            dropout=dropout,
            conv_hidden_size=conv_hidden_size,
            top_k=top_k,
            num_kernels=num_kernels,
            encoder_layers=encoder_layers,
            expand_to_future=expand_to_future,
            pre_norm=pre_norm,
        )

    def forward(self, chart_window):
        if chart_window.dim() != 3:
            raise ValueError(
                f"chart_window는 [B, T, F] 형태여야 합니다. 현재: {chart_window.shape}"
            )

        B, T, F_dim = chart_window.shape

        if T != self.input_size:
            raise ValueError(
                f"chart input length 불일치: expected {self.input_size}, got {T}"
            )

        if F_dim != self.input_dim:
            raise ValueError(
                f"chart input dim 불일치: expected {self.input_dim}, got {F_dim}"
            )

        chart_seq = self.encoder(chart_window)  # [B, T, H]

        return chart_seq


# ======================================================
# Main Fusion Model
# ======================================================
class ChartNewsTimeFusionTransformerClassifier(nn.Module):
    """
    TimesNet chart encoder + News Transformer encoder + Time-wise Fusion Transformer.

    Input:
        chart_window: [B, 72, 12]
        news_window : [B, 72, 9]

    Output:
        logits: [B, 3]
    """

    def __init__(
        self,
        input_size: int = 72,
        chart_input_dim: int = 12,
        news_input_dim: int = 9,
        num_classes: int = 3,
        prediction_horizon: int = 24,

        # chart encoder
        chart_hidden_size: int = 32,
        chart_conv_hidden_size: int = 64,
        chart_top_k: int = 2,
        chart_num_kernels: int = 4,
        chart_encoder_layers: int = 1,

        # news encoder
        news_hidden_size: int = 32,
        news_num_heads: int = 4,
        news_num_layers: int = 1,

        # fusion
        fusion_hidden_size: int = 64,
        fusion_num_heads: int = 4,
        fusion_num_layers: int = 1,

        # classifier
        classifier_hidden_size: int = 64,

        # common
        dropout: float = 0.30,
    ):
        super().__init__()

        self.input_size = input_size
        self.chart_input_dim = chart_input_dim
        self.news_input_dim = news_input_dim
        self.num_classes = num_classes

        self.chart_encoder = ChartTimesNetSequenceEncoder(
            input_size=input_size,
            input_dim=chart_input_dim,
            prediction_horizon=prediction_horizon,
            hidden_size=chart_hidden_size,
            dropout=dropout,
            conv_hidden_size=chart_conv_hidden_size,
            top_k=chart_top_k,
            num_kernels=chart_num_kernels,
            encoder_layers=chart_encoder_layers,
            expand_to_future=False,
            pre_norm=True,
        )

        self.news_encoder = NewsTransformerSequenceEncoder(
            input_size=input_size,
            input_dim=news_input_dim,
            hidden_size=news_hidden_size,
            num_heads=news_num_heads,
            num_layers=news_num_layers,
            dropout=dropout,
        )

        # chart/news hidden이 달라도 fusion_hidden으로 맞춤
        self.chart_proj = nn.Linear(chart_hidden_size, fusion_hidden_size)
        self.news_proj = nn.Linear(news_hidden_size, fusion_hidden_size)

        # 시간별 chart-news concat 후 nonlinear fusion
        self.time_fusion_proj = nn.Sequential(
            nn.LayerNorm(fusion_hidden_size * 2),
            nn.Linear(fusion_hidden_size * 2, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # chart/news 중요도 gate
        self.modality_gate = nn.Sequential(
            nn.LayerNorm(fusion_hidden_size * 2),
            nn.Linear(fusion_hidden_size * 2, fusion_hidden_size),
            nn.Sigmoid(),
        )

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, fusion_hidden_size)
        )

        self.fusion_pos_embedding = nn.Parameter(
            torch.zeros(1, input_size + 1, fusion_hidden_size)
        )

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=fusion_hidden_size,
            nhead=fusion_num_heads,
            dim_feedforward=fusion_hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.fusion_transformer = nn.TransformerEncoder(
            fusion_layer,
            num_layers=fusion_num_layers,
        )

        self.fusion_norm = nn.LayerNorm(fusion_hidden_size)
        self.dropout = nn.Dropout(dropout)

        # CLS + mean_pool + last_pool 사용
        classifier_input_dim = fusion_hidden_size * 3

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Linear(classifier_input_dim, classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_classes),
        )

        # auxiliary heads는 multi-head loss에서 사용 가능
        self.chart_aux_head = nn.Sequential(
            nn.LayerNorm(fusion_hidden_size * 2),
            nn.Linear(fusion_hidden_size * 2, classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_classes),
        )

        self.news_aux_head = nn.Sequential(
            nn.LayerNorm(fusion_hidden_size * 2),
            nn.Linear(fusion_hidden_size * 2, classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.fusion_pos_embedding, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def encode_modal_sequences(self, chart_window, news_window):
        """
        Returns:
            chart_proj_seq: [B, 72, fusion_hidden]
            news_proj_seq : [B, 72, fusion_hidden]
        """

        chart_seq = self.chart_encoder(chart_window)  # [B, T, Hc]
        news_seq = self.news_encoder(news_window)     # [B, T, Hn]

        chart_proj_seq = self.chart_proj(chart_seq)
        news_proj_seq = self.news_proj(news_seq)

        return chart_proj_seq, news_proj_seq

    def encode_fusion(self, chart_window, news_window):
        """
        Main fusion encoding.

        Returns:
            fusion_embedding: [B, fusion_hidden * 3]
            chart_summary   : [B, fusion_hidden * 2]
            news_summary    : [B, fusion_hidden * 2]
        """

        chart_proj_seq, news_proj_seq = self.encode_modal_sequences(
            chart_window,
            news_window,
        )

        B, T, H = chart_proj_seq.shape

        if T != self.input_size:
            raise ValueError(f"fusion input length 불일치: expected {self.input_size}, got {T}")

        # 시간별 concat
        pair = torch.cat([chart_proj_seq, news_proj_seq], dim=-1)  # [B, T, 2H]

        # nonlinear concat fusion
        mixed = self.time_fusion_proj(pair)  # [B, T, H]

        # gated fusion
        gate = self.modality_gate(pair)      # [B, T, H]
        gated = gate * chart_proj_seq + (1.0 - gate) * news_proj_seq

        # 두 fusion 방식을 합쳐 안정화
        fusion_tokens = mixed + gated        # [B, T, H]
        fusion_tokens = self.dropout(fusion_tokens)

        # CLS 추가
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, H]
        tokens = torch.cat([cls, fusion_tokens], dim=1)  # [B, T+1, H]

        tokens = tokens + self.fusion_pos_embedding[:, : T + 1, :]
        tokens = self.dropout(tokens)

        fusion_out = self.fusion_transformer(tokens)
        fusion_out = self.fusion_norm(fusion_out)

        cls_pool = fusion_out[:, 0, :]       # [B, H]
        seq_out = fusion_out[:, 1:, :]       # [B, T, H]
        mean_pool = seq_out.mean(dim=1)      # [B, H]
        last_pool = seq_out[:, -1, :]        # [B, H]

        fusion_embedding = torch.cat(
            [cls_pool, mean_pool, last_pool],
            dim=-1,
        )  # [B, 3H]

        # auxiliary summaries
        chart_mean = chart_proj_seq.mean(dim=1)
        chart_last = chart_proj_seq[:, -1, :]
        chart_summary = torch.cat([chart_mean, chart_last], dim=-1)

        news_mean = news_proj_seq.mean(dim=1)
        news_last = news_proj_seq[:, -1, :]
        news_summary = torch.cat([news_mean, news_last], dim=-1)

        return fusion_embedding, chart_summary, news_summary

    def forward(self, chart_window, news_window, return_aux: bool = False):
        """
        기본:
            logits만 반환

        return_aux=True:
            {
                "signal_logits": ...,
                "chart_logits": ...,
                "news_logits": ...
            }
        """

        fusion_embedding, chart_summary, news_summary = self.encode_fusion(
            chart_window,
            news_window,
        )

        signal_logits = self.classifier(fusion_embedding)

        if not return_aux:
            return signal_logits

        chart_logits = self.chart_aux_head(chart_summary)
        news_logits = self.news_aux_head(news_summary)

        return {
            "signal_logits": signal_logits,
            "chart_logits": chart_logits,
            "news_logits": news_logits,
        }


# ======================================================
# Backward-compatible alias
# ======================================================
ChartNewsFusionTransformerClassifier = ChartNewsTimeFusionTransformerClassifier


# ======================================================
# Quick shape test
# ======================================================
if __name__ == "__main__":
    model = ChartNewsTimeFusionTransformerClassifier(
        input_size=72,
        chart_input_dim=12,
        news_input_dim=9,
        num_classes=3,
        prediction_horizon=24,

        chart_hidden_size=32,
        chart_conv_hidden_size=64,
        chart_top_k=2,
        chart_num_kernels=4,
        chart_encoder_layers=1,

        news_hidden_size=32,
        news_num_heads=4,
        news_num_layers=1,

        fusion_hidden_size=64,
        fusion_num_heads=4,
        fusion_num_layers=1,

        classifier_hidden_size=64,
        dropout=0.30,
    )

    dummy_chart = torch.randn(4, 72, 12)
    dummy_news = torch.randn(4, 72, 9)

    logits = model(dummy_chart, dummy_news)
    aux = model(dummy_chart, dummy_news, return_aux=True)

    fusion_embedding, chart_summary, news_summary = model.encode_fusion(
        dummy_chart,
        dummy_news,
    )

    print("dummy_chart:", dummy_chart.shape)
    print("dummy_news :", dummy_news.shape)
    print("fusion_embedding:", fusion_embedding.shape)
    print("chart_summary   :", chart_summary.shape)
    print("news_summary    :", news_summary.shape)
    print("logits          :", logits.shape)

    print("signal_logits:", aux["signal_logits"].shape)
    print("chart_logits :", aux["chart_logits"].shape)
    print("news_logits  :", aux["news_logits"].shape)

    assert fusion_embedding.shape == (4, 192)
    assert chart_summary.shape == (4, 128)
    assert news_summary.shape == (4, 128)
    assert logits.shape == (4, 3)
    assert aux["signal_logits"].shape == (4, 3)
    assert aux["chart_logits"].shape == (4, 3)
    assert aux["news_logits"].shape == (4, 3)

    print("✅ ChartNewsTimeFusionTransformerClassifier shape test passed")