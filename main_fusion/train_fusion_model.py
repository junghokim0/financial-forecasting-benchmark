#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_fusion_model.py

Main Model:
TimesNet Chart Encoder + News Transformer Encoder + Time-wise Fusion Transformer

적용 내용:
- Fusion multi-head 구조 유지
- total_loss = 0.6 * signal_loss + 0.2 * chart_loss + 0.2 * news_loss
- class weights = False
- label smoothing = 0.03
- validation/test backtest를 non-overlap 24h cooldown 기준으로 평가

Label:
0 = SHORT
1 = HOLD
2 = LONG
"""

import json
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from fusion_model import ChartNewsTimeFusionTransformerClassifier


# ======================================================
# Config
# ======================================================
SEED = 42

# 필요에 맞게 여기만 바꾸면 됨
DATA_BASE_DIR = Path("/content/drive/MyDrive/trading/data/dataset/rolling_threshold_0012")
#DATA_BASE_DIR = Path(
#    "/content/drive/MyDrive/trading12/trading/data/dataset/rolling_threshold_0012_deriv14"
#)

#DATA_BASE_DIR = Path(
#    "/content/drive/MyDrive/trading12/trading/data/dataset/rolling_threshold_0012_funding13"
#)

OUTPUT_BASE_DIR = Path("outputs/main_fusion_multihead_nonoverlap_th_0012")
#OUTPUT_BASE_DIR = Path(
#    "outputs/main_fusion_multihead_nonoverlap_th_0012_deriv14"
#)
#OUTPUT_BASE_DIR = Path(
#    "outputs/main_fusion_multihead_nonoverlap_th_0012_funding13"
#)


ROLLING_NAMES = ["rolling_1", "rolling_2", "rolling_3"]

LABEL_NAMES = ["SHORT", "HOLD", "LONG"]
NUM_CLASSES = 3

INPUT_SIZE = 72
CHART_INPUT_DIM = 12
NEWS_INPUT_DIM = 9
PREDICTION_HORIZON = 24

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 8

LR = 1e-4
WEIGHT_DECAY = 1e-4

# Chart Encoder
CHART_HIDDEN_SIZE = 32
CHART_CONV_HIDDEN_SIZE = 64
CHART_TOP_K = 2
CHART_NUM_KERNELS = 4
CHART_ENCODER_LAYERS = 1

# News Encoder
NEWS_HIDDEN_SIZE = 32
NEWS_NUM_HEADS = 4
NEWS_NUM_LAYERS = 1

# Fusion Transformer
FUSION_HIDDEN_SIZE = 64
FUSION_NUM_HEADS = 4
FUSION_NUM_LAYERS = 1

CLASSIFIER_HIDDEN_SIZE = 64
DROPOUT = 0.30

USE_CLASS_WEIGHTS = False
LABEL_SMOOTHING = 0.03

USE_MULTI_HEAD_LOSS = True
SIGNAL_LOSS_WEIGHT = 0.60
CHART_LOSS_WEIGHT = 0.20
NEWS_LOSS_WEIGHT = 0.20

CONFIDENCE_THRESHOLDS = [
    0.34, 0.36, 0.38, 0.40, 0.42,
    0.44, 0.46, 0.48, 0.50, 0.52,
    0.55, 0.60, 0.65, 0.70, 0.75,
]

# non-overlap에서는 trade ratio가 낮게 나올 수 있어서 최소값을 조금 낮춤
MIN_TRADE_RATIO = 0.01
MAX_TRADE_RATIO = 0.30

FEE = 0.001
SLIPPAGE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# Seed
# ======================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ======================================================
# Dataset
# ======================================================
class ChartNewsDataset(Dataset):
    def __init__(self, chart_path, news_path, y_path):
        chart_path = Path(chart_path)
        news_path = Path(news_path)
        y_path = Path(y_path)

        if not chart_path.exists():
            raise FileNotFoundError(f"X_chart 파일이 없습니다: {chart_path}")

        if not news_path.exists():
            raise FileNotFoundError(f"X_news 파일이 없습니다: {news_path}")

        if not y_path.exists():
            raise FileNotFoundError(f"y 파일이 없습니다: {y_path}")

        self.X_chart = np.load(chart_path).astype(np.float32)
        self.X_news = np.load(news_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)

        if len(self.X_chart) != len(self.X_news):
            raise ValueError("X_chart와 X_news sample 수가 다릅니다.")

        if len(self.X_chart) != len(self.y):
            raise ValueError("X와 y sample 수가 다릅니다.")

        if self.X_chart.ndim != 3:
            raise ValueError(f"X_chart는 [N,T,F] 형태여야 합니다: {self.X_chart.shape}")

        if self.X_news.ndim != 3:
            raise ValueError(f"X_news는 [N,T,F] 형태여야 합니다: {self.X_news.shape}")

        if self.X_chart.shape[1] != INPUT_SIZE:
            raise ValueError(f"chart window size 불일치: {self.X_chart.shape}")

        if self.X_news.shape[1] != INPUT_SIZE:
            raise ValueError(f"news window size 불일치: {self.X_news.shape}")

        if self.X_chart.shape[2] != CHART_INPUT_DIM:
            raise ValueError(
                f"chart feature dim 불일치: expected {CHART_INPUT_DIM}, got {self.X_chart.shape[2]}"
            )

        if self.X_news.shape[2] != NEWS_INPUT_DIM:
            raise ValueError(
                f"news feature dim 불일치: expected {NEWS_INPUT_DIM}, got {self.X_news.shape[2]}"
            )

        if not np.isfinite(self.X_chart).all():
            raise ValueError(f"X_chart에 NaN 또는 inf가 있습니다: {chart_path}")

        if not np.isfinite(self.X_news).all():
            raise ValueError(f"X_news에 NaN 또는 inf가 있습니다: {news_path}")

        if not np.isfinite(self.y).all():
            raise ValueError(f"y에 NaN 또는 inf가 있습니다: {y_path}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_chart = torch.from_numpy(self.X_chart[idx])
        x_news = torch.from_numpy(self.X_news[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return x_chart, x_news, y


# ======================================================
# Utils
# ======================================================
def compute_class_weights(y):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    total = counts.sum()
    weights = total / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def prediction_distribution(preds):
    total = len(preds)
    result = {}

    for label_id, label_name in enumerate(LABEL_NAMES):
        count = int((preds == label_id).sum())
        result[label_name] = {
            "count": count,
            "ratio": float(count / total) if total > 0 else 0.0,
        }

    return result


def get_time_columns(df: pd.DataFrame):
    possible_sample_cols = [
        "sample_time",
        "hour",
        "timestamp",
    ]

    possible_target_cols = [
        "target_time",
    ]

    sample_col = None
    target_col = None

    for col in possible_sample_cols:
        if col in df.columns:
            sample_col = col
            break

    for col in possible_target_cols:
        if col in df.columns:
            target_col = col
            break

    if sample_col is None:
        raise ValueError(
            f"sample_time 역할을 하는 컬럼이 없습니다. 현재 columns: {list(df.columns)}"
        )

    if target_col is None:
        raise ValueError(
            f"target_time 컬럼이 없습니다. 현재 columns: {list(df.columns)}"
        )

    return sample_col, target_col


def non_overlapping_backtest(
    meta_df,
    preds,
    fee=0.001,
    slippage=0.001,
):
    """
    non-overlap 24h cooldown backtest.

    규칙:
    - pred == 2: LONG 진입
    - pred == 0: SHORT 진입
    - pred == 1: HOLD
    - 한 번 진입하면 해당 row의 target_time까지 추가 진입 금지
    """

    if "raw_future_return" not in meta_df.columns:
        raise ValueError("meta에 raw_future_return 컬럼이 없습니다.")

    if len(meta_df) != len(preds):
        raise ValueError("meta row 수와 prediction 수가 다릅니다.")

    sample_col, target_col = get_time_columns(meta_df)

    df = meta_df.copy()
    df["pred"] = preds

    df[sample_col] = pd.to_datetime(df[sample_col], utc=True, errors="coerce")
    df[target_col] = pd.to_datetime(df[target_col], utc=True, errors="coerce")

    if df[sample_col].isna().sum() > 0:
        raise ValueError(f"{sample_col}에 NaN 시간이 있습니다.")

    if df[target_col].isna().sum() > 0:
        raise ValueError(f"{target_col}에 NaN 시간이 있습니다.")

    df = df.sort_values(sample_col).reset_index(drop=True)

    cost = fee + slippage
    total_rows = len(df)

    next_available_time = pd.Timestamp.min.tz_localize("UTC")

    all_strategy_returns = []
    trade_returns = []

    trade_records = []

    for _, row in df.iterrows():
        sample_time = row[sample_col]
        target_time = row[target_col]
        pred = int(row["pred"])
        raw_return = float(row["raw_future_return"])

        # 이전 포지션이 아직 24h 보유 중이면 이번 신호 무시
        if sample_time < next_available_time:
            all_strategy_returns.append(0.0)
            continue

        # HOLD면 거래 안 함
        if pred == 1:
            all_strategy_returns.append(0.0)
            continue

        # LONG / SHORT
        if pred == 2:
            position = 1.0
        elif pred == 0:
            position = -1.0
        else:
            raise ValueError(f"알 수 없는 pred label: {pred}")

        strategy_return = position * raw_return - cost

        all_strategy_returns.append(strategy_return)
        trade_returns.append(strategy_return)

        trade_records.append({
            "sample_time": sample_time,
            "target_time": target_time,
            "pred": pred,
            "pred_label": LABEL_NAMES[pred],
            "raw_future_return": raw_return,
            "strategy_return": strategy_return,
        })

        # 핵심: 24h target_time까지 다음 진입 금지
        next_available_time = target_time

    all_strategy_returns = np.array(all_strategy_returns, dtype=np.float64)
    trade_returns = np.array(trade_returns, dtype=np.float64)

    if len(all_strategy_returns) == 0:
        return {
            "cumulative_return": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "trade_ratio": 0.0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
        }, pd.DataFrame(trade_records)

    cumulative_return = float(np.prod(1.0 + all_strategy_returns) - 1.0)

    if all_strategy_returns.std() > 0:
        sharpe_like = float(
            all_strategy_returns.mean()
            / all_strategy_returns.std()
            * np.sqrt(365 * 24)
        )
    else:
        sharpe_like = 0.0

    equity = np.cumprod(1.0 + all_strategy_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    trade_count = len(trade_returns)

    if trade_count > 0:
        win_rate = float((trade_returns > 0).mean())
        avg_trade_return = float(trade_returns.mean())
    else:
        win_rate = 0.0
        avg_trade_return = 0.0

    metrics = {
        "cumulative_return": cumulative_return,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "trade_count": int(trade_count),
        "trade_ratio": float(trade_count / total_rows) if total_rows > 0 else 0.0,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
    }

    return metrics, pd.DataFrame(trade_records)


def apply_confidence_threshold(probs, threshold, hold_label=1):
    preds = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)

    filtered_preds = preds.copy()
    filtered_preds[confidence < threshold] = hold_label

    return filtered_preds, confidence


def tune_confidence_threshold_non_overlap(
    val_labels,
    val_probs,
    val_meta,
    thresholds,
    fee=0.001,
    slippage=0.001,
):
    """
    validation non-overlap backtest 기준으로 confidence threshold 선택.
    """

    results = []

    for th in thresholds:
        preds_th, _ = apply_confidence_threshold(
            val_probs,
            threshold=th,
            hold_label=1,
        )

        acc = accuracy_score(val_labels, preds_th)
        macro_f1 = f1_score(
            val_labels,
            preds_th,
            average="macro",
            zero_division=0,
        )

        backtest, _ = non_overlapping_backtest(
            val_meta,
            preds_th,
            fee=fee,
            slippage=slippage,
        )

        results.append({
            "threshold": float(th),
            "val_acc": float(acc),
            "val_macro_f1": float(macro_f1),
            "trade_ratio": float(backtest["trade_ratio"]),
            "cumulative_return": float(backtest["cumulative_return"]),
            "sharpe_like": float(backtest["sharpe_like"]),
            "max_drawdown": float(backtest["max_drawdown"]),
            "win_rate": float(backtest["win_rate"]),
            "avg_trade_return": float(backtest["avg_trade_return"]),
            "prediction_distribution": prediction_distribution(preds_th),
        })

    valid_results = [
        r for r in results
        if MIN_TRADE_RATIO <= r["trade_ratio"] <= MAX_TRADE_RATIO
    ]

    if len(valid_results) == 0:
        print(
            f"⚠️ non-overlap trade_ratio 조건을 만족하는 threshold가 없습니다. "
            f"전체 후보에서 선택합니다. "
            f"조건: {MIN_TRADE_RATIO:.2f} <= trade_ratio <= {MAX_TRADE_RATIO:.2f}"
        )
        valid_results = results

    best = sorted(
        valid_results,
        key=lambda x: (
            x["avg_trade_return"],
            x["cumulative_return"],
            x["sharpe_like"],
            x["win_rate"],
            x["val_macro_f1"],
            -abs(x["trade_ratio"] - 0.05),
        ),
        reverse=True,
    )[0]

    return best, results


def build_criterion(train_y):
    class_weights = compute_class_weights(train_y).to(DEVICE)
    print("class_weights reference:", class_weights.detach().cpu().numpy())

    if USE_CLASS_WEIGHTS:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=LABEL_SMOOTHING,
        )
        print("loss: CrossEntropyLoss with class weights")
    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=LABEL_SMOOTHING,
        )
        print("loss: CrossEntropyLoss without class weights")

    return criterion


def compute_multihead_loss(outputs, y, criterion):
    signal_loss = criterion(outputs["signal_logits"], y)
    chart_loss = criterion(outputs["chart_logits"], y)
    news_loss = criterion(outputs["news_logits"], y)

    total_loss = (
        SIGNAL_LOSS_WEIGHT * signal_loss
        + CHART_LOSS_WEIGHT * chart_loss
        + NEWS_LOSS_WEIGHT * news_loss
    )

    return total_loss, {
        "signal_loss": float(signal_loss.detach().cpu().item()),
        "chart_loss": float(chart_loss.detach().cpu().item()),
        "news_loss": float(news_loss.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
    }


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_signal_loss = 0.0
    total_chart_loss = 0.0
    total_news_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_chart, X_news, y in loader:
            X_chart = X_chart.to(DEVICE)
            X_news = X_news.to(DEVICE)
            y = y.to(DEVICE)

            if USE_MULTI_HEAD_LOSS:
                outputs = model(X_chart, X_news, return_aux=True)
                loss, loss_dict = compute_multihead_loss(outputs, y, criterion)
                logits = outputs["signal_logits"]

                total_signal_loss += loss_dict["signal_loss"] * len(y)
                total_chart_loss += loss_dict["chart_loss"] * len(y)
                total_news_loss += loss_dict["news_loss"] * len(y)
            else:
                logits = model(X_chart, X_news)
                loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * len(y)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    loss_detail = {
        "loss": float(avg_loss),
        "signal_loss": float(total_signal_loss / len(all_labels)) if USE_MULTI_HEAD_LOSS else None,
        "chart_loss": float(total_chart_loss / len(all_labels)) if USE_MULTI_HEAD_LOSS else None,
        "news_loss": float(total_news_loss / len(all_labels)) if USE_MULTI_HEAD_LOSS else None,
    }

    return avg_loss, acc, macro_f1, all_preds, all_labels, all_probs, loss_detail


# ======================================================
# Train One Rolling
# ======================================================
def train_one_rolling(rolling_name):
    print("\n" + "=" * 80)
    print(f"🚀 Main Fusion Multi-head 학습 시작: {rolling_name}")
    print("=" * 80)

    rolling_dir = DATA_BASE_DIR / rolling_name
    save_dir = OUTPUT_BASE_DIR / rolling_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if not rolling_dir.exists():
        raise FileNotFoundError(f"rolling 데이터 폴더가 없습니다: {rolling_dir}")

    train_dataset = ChartNewsDataset(
        rolling_dir / "X_chart_train.npy",
        rolling_dir / "X_news_train.npy",
        rolling_dir / "y_train.npy",
    )

    val_dataset = ChartNewsDataset(
        rolling_dir / "X_chart_val.npy",
        rolling_dir / "X_news_val.npy",
        rolling_dir / "y_val.npy",
    )

    test_dataset = ChartNewsDataset(
        rolling_dir / "X_chart_test.npy",
        rolling_dir / "X_news_test.npy",
        rolling_dir / "y_test.npy",
    )

    val_meta_path = rolling_dir / "sample_meta_val.csv"
    test_meta_path = rolling_dir / "sample_meta_test.csv"

    if not val_meta_path.exists():
        raise FileNotFoundError(f"val meta 파일이 없습니다: {val_meta_path}")

    if not test_meta_path.exists():
        raise FileNotFoundError(f"test meta 파일이 없습니다: {test_meta_path}")

    val_meta = pd.read_csv(val_meta_path)
    test_meta = pd.read_csv(test_meta_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("X_chart_train:", train_dataset.X_chart.shape)
    print("X_news_train :", train_dataset.X_news.shape)
    print("X_chart_val  :", val_dataset.X_chart.shape)
    print("X_news_val   :", val_dataset.X_news.shape)
    print("X_chart_test :", test_dataset.X_chart.shape)
    print("X_news_test  :", test_dataset.X_news.shape)

    print("train label count:", np.bincount(train_dataset.y, minlength=NUM_CLASSES))
    print("val label count  :", np.bincount(val_dataset.y, minlength=NUM_CLASSES))
    print("test label count :", np.bincount(test_dataset.y, minlength=NUM_CLASSES))

    model = ChartNewsTimeFusionTransformerClassifier(
        input_size=INPUT_SIZE,
        chart_input_dim=CHART_INPUT_DIM,
        news_input_dim=NEWS_INPUT_DIM,
        num_classes=NUM_CLASSES,
        prediction_horizon=PREDICTION_HORIZON,

        chart_hidden_size=CHART_HIDDEN_SIZE,
        chart_conv_hidden_size=CHART_CONV_HIDDEN_SIZE,
        chart_top_k=CHART_TOP_K,
        chart_num_kernels=CHART_NUM_KERNELS,
        chart_encoder_layers=CHART_ENCODER_LAYERS,

        news_hidden_size=NEWS_HIDDEN_SIZE,
        news_num_heads=NEWS_NUM_HEADS,
        news_num_layers=NEWS_NUM_LAYERS,

        fusion_hidden_size=FUSION_HIDDEN_SIZE,
        fusion_num_heads=FUSION_NUM_HEADS,
        fusion_num_layers=FUSION_NUM_LAYERS,

        classifier_hidden_size=CLASSIFIER_HIDDEN_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = build_criterion(train_dataset.y)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda"),
    )

    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_signal_loss_sum = 0.0
        train_chart_loss_sum = 0.0
        train_news_loss_sum = 0.0

        train_preds = []
        train_labels = []

        for X_chart, X_news, y in train_loader:
            X_chart = X_chart.to(DEVICE)
            X_news = X_news.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(DEVICE.type == "cuda"),
            ):
                if USE_MULTI_HEAD_LOSS:
                    outputs = model(X_chart, X_news, return_aux=True)
                    loss, loss_dict = compute_multihead_loss(outputs, y, criterion)
                    logits = outputs["signal_logits"]
                else:
                    logits = model(X_chart, X_news)
                    loss = criterion(logits, y)
                    loss_dict = None

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(logits.detach(), dim=1)

            train_loss_sum += loss.item() * len(y)

            if USE_MULTI_HEAD_LOSS:
                train_signal_loss_sum += loss_dict["signal_loss"] * len(y)
                train_chart_loss_sum += loss_dict["chart_loss"] * len(y)
                train_news_loss_sum += loss_dict["news_loss"] * len(y)

            train_preds.append(preds.cpu().numpy())
            train_labels.append(y.cpu().numpy())

        train_preds = np.concatenate(train_preds)
        train_labels = np.concatenate(train_labels)

        train_loss = train_loss_sum / len(train_labels)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(
            train_labels,
            train_preds,
            average="macro",
            zero_division=0,
        )

        val_loss, val_acc, val_f1, _, _, _, val_loss_detail = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_log = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_macro_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "lr": float(current_lr),
        }

        if USE_MULTI_HEAD_LOSS:
            epoch_log.update({
                "train_signal_loss": float(train_signal_loss_sum / len(train_labels)),
                "train_chart_loss": float(train_chart_loss_sum / len(train_labels)),
                "train_news_loss": float(train_news_loss_sum / len(train_labels)),
                "val_signal_loss": val_loss_detail["signal_loss"],
                "val_chart_loss": val_loss_detail["chart_loss"],
                "val_news_loss": val_loss_detail["news_loss"],
            })

        history.append(epoch_log)

        if USE_MULTI_HEAD_LOSS:
            print(
                f"Epoch [{epoch:02d}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} "
                f"Val F1: {val_f1:.4f} "
                f"Val Signal/Chart/News: "
                f"{val_loss_detail['signal_loss']:.4f}/"
                f"{val_loss_detail['chart_loss']:.4f}/"
                f"{val_loss_detail['news_loss']:.4f} "
                f"LR: {current_lr:.6f}"
            )
        else:
            print(
                f"Epoch [{epoch:02d}/{EPOCHS}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.4f} "
                f"Train F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_acc:.4f} "
                f"Val F1: {val_f1:.4f} "
                f"LR: {current_lr:.6f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_count = 0
            print("✅ Best fusion model updated")
        else:
            patience_count += 1
            print(f"patience: {patience_count}/{PATIENCE}")

        if patience_count >= PATIENCE:
            print("⏹ Early stopping")
            break

    if best_state is None:
        raise RuntimeError("best_state가 저장되지 않았습니다.")

    model.load_state_dict(best_state)

    # --------------------------------------------------
    # validation threshold tuning using non-overlap
    # --------------------------------------------------
    val_loss, val_acc_argmax, val_f1_argmax, val_preds_argmax, val_labels, val_probs, val_loss_detail = evaluate(
        model,
        val_loader,
        criterion,
    )

    best_threshold_result, threshold_tuning_results = tune_confidence_threshold_non_overlap(
        val_labels=val_labels,
        val_probs=val_probs,
        val_meta=val_meta,
        thresholds=CONFIDENCE_THRESHOLDS,
        fee=FEE,
        slippage=SLIPPAGE,
    )

    best_conf_threshold = best_threshold_result["threshold"]

    print("\n[Confidence Threshold Tuning on Validation - Non-overlap 24h]")
    print("best threshold:", best_conf_threshold)
    print("best val sharpe_like:", best_threshold_result["sharpe_like"])
    print("best val cumulative_return:", best_threshold_result["cumulative_return"])
    print("best val max_drawdown:", best_threshold_result["max_drawdown"])
    print("best val trade_ratio:", best_threshold_result["trade_ratio"])
    print("best val avg_trade_return:", best_threshold_result["avg_trade_return"])
    print("best val macro_f1:", best_threshold_result["val_macro_f1"])

    # --------------------------------------------------
    # test evaluation
    # --------------------------------------------------
    test_loss, test_acc_argmax, test_f1_argmax, test_preds_argmax, test_labels, test_probs, test_loss_detail = evaluate(
        model,
        test_loader,
        criterion,
    )

    test_preds, test_confidence = apply_confidence_threshold(
        test_probs,
        threshold=best_conf_threshold,
        hold_label=1,
    )

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(
        test_labels,
        test_preds,
        average="macro",
        zero_division=0,
    )

    pred_dist_argmax = prediction_distribution(test_preds_argmax)
    pred_dist_threshold = prediction_distribution(test_preds)

    cm_argmax = confusion_matrix(
        test_labels,
        test_preds_argmax,
        labels=[0, 1, 2],
    ).tolist()

    cm_threshold = confusion_matrix(
        test_labels,
        test_preds,
        labels=[0, 1, 2],
    ).tolist()

    cls_report_argmax = classification_report(
        test_labels,
        test_preds_argmax,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    cls_report_threshold = classification_report(
        test_labels,
        test_preds,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    backtest_argmax, argmax_trades = non_overlapping_backtest(
        test_meta,
        test_preds_argmax,
        fee=FEE,
        slippage=SLIPPAGE,
    )

    backtest_threshold, threshold_trades = non_overlapping_backtest(
        test_meta,
        test_preds,
        fee=FEE,
        slippage=SLIPPAGE,
    )

    print("\n[Test Result - Argmax / Non-overlap]")
    print("test_loss:", round(test_loss, 4))
    print("test_acc :", round(test_acc_argmax, 4))
    print("test_f1  :", round(test_f1_argmax, 4))
    print("pred_dist:", pred_dist_argmax)
    print("backtest :", backtest_argmax)

    print("\n[Test Result - Confidence Threshold / Non-overlap]")
    print("threshold:", best_conf_threshold)
    print("test_acc :", round(test_acc, 4))
    print("test_f1  :", round(test_f1, 4))
    print("pred_dist:", pred_dist_threshold)
    print("backtest :", backtest_threshold)

    print("\n[Confusion Matrix - Confidence Threshold]")
    print(confusion_matrix(test_labels, test_preds, labels=[0, 1, 2]))

    torch.save(best_state, save_dir / "best_model.pt")

    argmax_trades.to_csv(
        save_dir / "non_overlap_trades_argmax.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold_trades.to_csv(
        save_dir / "non_overlap_trades_threshold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pred_df = test_meta.copy()
    pred_df["y_true"] = test_labels
    pred_df["y_pred_argmax"] = test_preds_argmax
    pred_df["y_pred"] = test_preds

    pred_df["y_true_label"] = [LABEL_NAMES[i] for i in test_labels]
    pred_df["y_pred_argmax_label"] = [LABEL_NAMES[i] for i in test_preds_argmax]
    pred_df["y_pred_label"] = [LABEL_NAMES[i] for i in test_preds]

    pred_df["prob_short"] = test_probs[:, 0]
    pred_df["prob_hold"] = test_probs[:, 1]
    pred_df["prob_long"] = test_probs[:, 2]
    pred_df["confidence"] = test_confidence
    pred_df["confidence_threshold"] = best_conf_threshold

    pred_df.to_csv(
        save_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "rolling": rolling_name,
        "model": "ChartNewsTimeFusionTransformerClassifier",
        "best_epoch": int(best_epoch),
        "best_val_macro_f1_argmax": float(best_val_f1),
        "val_argmax": {
            "loss": float(val_loss),
            "acc": float(val_acc_argmax),
            "macro_f1": float(val_f1_argmax),
            "loss_detail": val_loss_detail,
        },
        "confidence_threshold": float(best_conf_threshold),
        "threshold_tuning_on_val_non_overlap": best_threshold_result,
        "threshold_tuning_candidates": threshold_tuning_results,
        "test_argmax_non_overlap": {
            "loss": float(test_loss),
            "acc": float(test_acc_argmax),
            "macro_f1": float(test_f1_argmax),
            "prediction_distribution": pred_dist_argmax,
            "confusion_matrix": cm_argmax,
            "classification_report": cls_report_argmax,
            "backtest": backtest_argmax,
            "loss_detail": test_loss_detail,
        },
        "test_confidence_threshold_non_overlap": {
            "loss": float(test_loss),
            "acc": float(test_acc),
            "macro_f1": float(test_f1),
            "prediction_distribution": pred_dist_threshold,
            "confusion_matrix": cm_threshold,
            "classification_report": cls_report_threshold,
            "backtest": backtest_threshold,
        },
        "config": {
            "seed": SEED,
            "input_size": INPUT_SIZE,
            "chart_input_dim": CHART_INPUT_DIM,
            "news_input_dim": NEWS_INPUT_DIM,
            "prediction_horizon": PREDICTION_HORIZON,

            "chart_hidden_size": CHART_HIDDEN_SIZE,
            "chart_conv_hidden_size": CHART_CONV_HIDDEN_SIZE,
            "chart_top_k": CHART_TOP_K,
            "chart_num_kernels": CHART_NUM_KERNELS,
            "chart_encoder_layers": CHART_ENCODER_LAYERS,

            "news_hidden_size": NEWS_HIDDEN_SIZE,
            "news_num_heads": NEWS_NUM_HEADS,
            "news_num_layers": NEWS_NUM_LAYERS,

            "fusion_hidden_size": FUSION_HIDDEN_SIZE,
            "fusion_num_heads": FUSION_NUM_HEADS,
            "fusion_num_layers": FUSION_NUM_LAYERS,

            "dropout": DROPOUT,
            "classifier_hidden_size": CLASSIFIER_HIDDEN_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "fee": FEE,
            "slippage": SLIPPAGE,
            "use_class_weights": USE_CLASS_WEIGHTS,
            "label_smoothing": LABEL_SMOOTHING,
            "use_multi_head_loss": USE_MULTI_HEAD_LOSS,
            "signal_loss_weight": SIGNAL_LOSS_WEIGHT,
            "chart_loss_weight": CHART_LOSS_WEIGHT,
            "news_loss_weight": NEWS_LOSS_WEIGHT,
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
            "min_trade_ratio": MIN_TRADE_RATIO,
            "max_trade_ratio": MAX_TRADE_RATIO,
            "best_selection_metric": "validation_macro_f1_argmax",
            "threshold_selection": "validation_non_overlap_backtest",
            "test_prediction_rule": "argmax_or_confidence_threshold_with_non_overlap_24h_cooldown",
            "label_mapping": {
                "0": "SHORT",
                "1": "HOLD",
                "2": "LONG",
            },
            "data_dir": str(rolling_dir),
            "output_dir": str(save_dir),
        },
        "history": history,
    }

    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nSaved:", save_dir)

    return result


# ======================================================
# Main
# ======================================================
def main():
    set_seed(SEED)

    print("DEVICE:", DEVICE)
    print("DATA_BASE_DIR:", DATA_BASE_DIR)
    print("OUTPUT_BASE_DIR:", OUTPUT_BASE_DIR)
    print("ROLLING_NAMES:", ROLLING_NAMES)
    print("USE_CLASS_WEIGHTS:", USE_CLASS_WEIGHTS)
    print("LABEL_SMOOTHING:", LABEL_SMOOTHING)
    print("USE_MULTI_HEAD_LOSS:", USE_MULTI_HEAD_LOSS)
    print("LOSS WEIGHTS:", {
        "signal": SIGNAL_LOSS_WEIGHT,
        "chart": CHART_LOSS_WEIGHT,
        "news": NEWS_LOSS_WEIGHT,
    })
    print("BACKTEST:", "non-overlap 24h cooldown")
    print("CONFIDENCE_THRESHOLDS:", CONFIDENCE_THRESHOLDS)

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for rolling_name in ROLLING_NAMES:
        result = train_one_rolling(rolling_name)
        all_results.append(result)

    summary = []

    for result in all_results:
        argmax = result["test_argmax_non_overlap"]
        th = result["test_confidence_threshold_non_overlap"]

        summary.append({
            "rolling": result["rolling"],
            "best_epoch": result["best_epoch"],
            "best_val_macro_f1_argmax": result["best_val_macro_f1_argmax"],
            "confidence_threshold": result["confidence_threshold"],

            "test_acc_argmax": argmax["acc"],
            "test_macro_f1_argmax": argmax["macro_f1"],
            "cum_return_argmax": argmax["backtest"]["cumulative_return"],
            "sharpe_argmax": argmax["backtest"]["sharpe_like"],
            "mdd_argmax": argmax["backtest"]["max_drawdown"],
            "trade_ratio_argmax": argmax["backtest"]["trade_ratio"],
            "win_rate_argmax": argmax["backtest"]["win_rate"],

            "test_acc_threshold": th["acc"],
            "test_macro_f1_threshold": th["macro_f1"],
            "cum_return_threshold": th["backtest"]["cumulative_return"],
            "sharpe_threshold": th["backtest"]["sharpe_like"],
            "mdd_threshold": th["backtest"]["max_drawdown"],
            "trade_ratio_threshold": th["backtest"]["trade_ratio"],
            "win_rate_threshold": th["backtest"]["win_rate"],
        })

    summary_path = OUTPUT_BASE_DIR / "fusion_multihead_nonoverlap_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("📌 Main Fusion Multi-head Non-overlap Rolling Summary")
    print("=" * 80)

    for s in summary:
        print(
            f"{s['rolling']} | "
            f"best_val_f1={s['best_val_macro_f1_argmax']:.4f} | "
            f"th={s['confidence_threshold']:.2f} | "
            f"argmax_f1={s['test_macro_f1_argmax']:.4f} | "
            f"th_f1={s['test_macro_f1_threshold']:.4f} | "
            f"argmax_ret={s['cum_return_argmax']:.4f} | "
            f"argmax_trade={s['trade_ratio_argmax']:.4f} | "
            f"th_ret={s['cum_return_threshold']:.4f} | "
            f"th_trade={s['trade_ratio_threshold']:.4f}"
        )

    print("\nSummary saved:", summary_path)


if __name__ == "__main__":
    main()