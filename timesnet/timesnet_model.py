"""Official-core TimesNet baselines for the common financial benchmark.

The FFT period discovery, 2-D Inception convolution, adaptive aggregation,
residual connection, and post-block LayerNorm follow THUML's TimesNet design.
Only the final task head is adapted to the benchmark's direct targets:
one future-24h return or three SHORT/HOLD/LONG logits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TimesNetConfig:
    """Architecture fixed before evaluation.

    The capacity matches Cryptova's chart encoder settings so the standalone
    TimesNet and the multimodal system receive a comparable periodic backbone.
    The block ordering itself follows the official TimesNet implementation.
    """

    sequence_length: int = 72
    input_size: int = 12
    hidden_size: int = 32
    conv_hidden_size: int = 64
    top_k: int = 2
    num_kernels: int = 4
    encoder_layers: int = 1
    dropout: float = 0.30
    num_classes: int = 3

    def __post_init__(self) -> None:
        integer_fields = (
            self.sequence_length,
            self.input_size,
            self.hidden_size,
            self.conv_hidden_size,
            self.top_k,
            self.num_kernels,
            self.encoder_layers,
            self.num_classes,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("All TimesNet dimensions and counts must be positive.")
        max_non_dc_frequencies = self.sequence_length // 2
        if self.top_k > max_non_dc_frequencies:
            raise ValueError(
                f"top_k={self.top_k} exceeds {max_non_dc_frequencies} non-DC frequencies."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict:
        return asdict(self)


class PositionalEmbedding(nn.Module):
    """Fixed sinusoidal position encoding used by the official embedding."""

    def __init__(self, hidden_size: int, max_length: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-(math.log(10000.0) / hidden_size))
        )
        encoding = torch.zeros(max_length, hidden_size, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * divisor)
        if hidden_size > 1:
            encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.encoding[:, : sequence.shape[1]]


class TokenEmbedding(nn.Module):
    """Circular Conv1d value embedding from the official Time-Series-Library."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.token_conv(sequence.transpose(1, 2)).transpose(1, 2)


class DataEmbedding(nn.Module):
    """Value plus positional embedding; no calendar markers are supplied."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(input_size, hidden_size)
        self.position_embedding = PositionalEmbedding(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.value_embedding(sequence) + self.position_embedding(sequence)
        )


class InceptionBlockV1(nn.Module):
    """Average parallel odd-sized 2-D convolution kernels."""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2 * index + 1,
                    padding=index,
                )
                for index in range(num_kernels)
            ]
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack([kernel(tensor) for kernel in self.kernels], dim=-1).mean(-1)


def fft_for_period(sequence: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batch-global periods and per-sample amplitudes for those periods."""

    if sequence.ndim != 3:
        raise ValueError(f"Expected (batch, time, hidden), got {tuple(sequence.shape)}")
    spectrum = torch.fft.rfft(sequence, dim=1)
    frequency_strength = spectrum.abs().mean(0).mean(-1).clone()
    frequency_strength[0] = 0
    top_indices = torch.topk(frequency_strength, top_k).indices
    periods = torch.div(
        sequence.shape[1], top_indices.clamp_min(1), rounding_mode="floor"
    ).clamp_min(1)
    period_weight = spectrum.abs().mean(-1).index_select(1, top_indices)
    return periods, period_weight


class TimesBlock(nn.Module):
    """Official TimesNet periodic 2-D variation block."""

    def __init__(self, config: TimesNetConfig) -> None:
        super().__init__()
        self.top_k = config.top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(
                config.hidden_size, config.conv_hidden_size, config.num_kernels
            ),
            nn.GELU(),
            InceptionBlockV1(
                config.conv_hidden_size, config.hidden_size, config.num_kernels
            ),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, time, hidden = sequence.shape
        periods, period_weight = fft_for_period(sequence, self.top_k)
        periodic_outputs: list[torch.Tensor] = []

        for period_tensor in periods:
            period = int(period_tensor.item())
            padded_length = ((time + period - 1) // period) * period
            if padded_length > time:
                padding = torch.zeros(
                    batch,
                    padded_length - time,
                    hidden,
                    dtype=sequence.dtype,
                    device=sequence.device,
                )
                periodic = torch.cat([sequence, padding], dim=1)
            else:
                periodic = sequence

            periodic = periodic.reshape(batch, padded_length // period, period, hidden)
            periodic = periodic.permute(0, 3, 1, 2).contiguous()
            periodic = self.conv(periodic)
            periodic = periodic.permute(0, 2, 3, 1).reshape(batch, padded_length, hidden)
            periodic_outputs.append(periodic[:, :time])

        stacked = torch.stack(periodic_outputs, dim=-1)
        weights = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)
        aggregated = torch.sum(stacked * weights, dim=-1)
        return aggregated + sequence


class TimesNetEncoder(nn.Module):
    """Encode `(batch, 72, 12)` into `(batch, 72, hidden_size)`."""

    def __init__(self, config: TimesNetConfig = TimesNetConfig()) -> None:
        super().__init__()
        self.config = config
        self.embedding = DataEmbedding(
            config.input_size, config.hidden_size, config.dropout
        )
        self.blocks = nn.ModuleList(
            [TimesBlock(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, chart: torch.Tensor) -> torch.Tensor:
        if chart.ndim != 3:
            raise ValueError(
                f"chart must have shape (batch, sequence, feature), got {tuple(chart.shape)}"
            )
        if tuple(chart.shape[1:]) != (
            self.config.sequence_length,
            self.config.input_size,
        ):
            raise ValueError(
                "Expected chart shape (*, "
                f"{self.config.sequence_length}, {self.config.input_size}), got {tuple(chart.shape)}"
            )
        encoded = self.embedding(chart)
        for block in self.blocks:
            encoded = self.layer_norm(block(encoded))
        return encoded


class TimesNetTaskHead(nn.Module):
    """Official classification-style GELU/dropout/flatten projection."""

    def __init__(self, config: TimesNetConfig, output_size: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.projection = nn.Linear(
            config.sequence_length * config.hidden_size, output_size
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        flattened = self.dropout(self.activation(encoded)).reshape(encoded.shape[0], -1)
        return self.projection(flattened)


class TimesNetRegressor(nn.Module):
    def __init__(self, config: TimesNetConfig = TimesNetConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = TimesNetEncoder(config)
        self.regression_head = TimesNetTaskHead(config, 1)

    def forward(self, chart: torch.Tensor) -> torch.Tensor:
        return self.regression_head(self.encoder(chart)).squeeze(-1)


class TimesNetClassifier(nn.Module):
    def __init__(self, config: TimesNetConfig = TimesNetConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = TimesNetEncoder(config)
        self.classification_head = TimesNetTaskHead(config, config.num_classes)

    def forward(self, chart: torch.Tensor) -> torch.Tensor:
        return self.classification_head(self.encoder(chart))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
