#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_original12_predictions.py

기존 original12 best_model.pt를 로드해서
val_predictions.csv / test_predictions.csv 저장.

저장 컬럼:
- y_pred_argmax: softmax argmax
- y_pred: 기존 confidence threshold 적용 prediction
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from fusion_model import ChartNewsTimeFusionTransformerClassifier


ROLLING_NAMES = ["rolling_1", "rolling_2", "rolling_3"]
LABEL_NAMES = ["SHORT", "HOLD", "LONG"]

BEST_THRESHOLDS = {
    "rolling_1": 0.40,
    "rolling_2": 0.36,
    "rolling_3": 0.46,
}

NUM_CLASSES = 3
INPUT_SIZE = 72
CHART_INPUT_DIM = 12
NEWS_INPUT_DIM = 9
PREDICTION_HORIZON = 24
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_BASE_DIR = Path(
    "/content/drive/MyDrive/trading12/trading/data/dataset/rolling_threshold_0012"
)

OUTPUT_BASE_DIR = Path(
    "outputs/main_fusion_multihead_nonoverlap_th_0012"
)

CHART_HIDDEN_SIZE = 32
CHART_CONV_HIDDEN_SIZE = 64
CHART_TOP_K = 2
CHART_NUM_KERNELS = 4
CHART_ENCODER_LAYERS = 1

NEWS_HIDDEN_SIZE = 32
NEWS_NUM_HEADS = 4
NEWS_NUM_LAYERS = 1

FUSION_HIDDEN_SIZE = 64
FUSION_NUM_HEADS = 4
FUSION_NUM_LAYERS = 1

CLASSIFIER_HIDDEN_SIZE = 64
DROPOUT = 0.30


class ChartNewsDataset(Dataset):
    def __init__(self, chart_path, news_path, y_path):
        self.X_chart = np.load(chart_path).astype(np.float32)
        self.X_news = np.load(news_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_chart[idx]),
            torch.from_numpy(self.X_news[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


def apply_confidence_threshold(probs, threshold, hold_label=1):
    preds_argmax = np.argmax(probs, axis=1)
    confidence = probs.max(axis=1)

    preds_threshold = preds_argmax.copy()
    preds_threshold[confidence < threshold] = hold_label

    return preds_threshold, confidence


def evaluate(model, loader):
    model.eval()

    all_preds_argmax = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_chart, X_news, y in loader:
            X_chart = X_chart.to(DEVICE)
            X_news = X_news.to(DEVICE)

            outputs = model(X_chart, X_news, return_aux=True)
            logits = outputs["signal_logits"]

            probs = torch.softmax(logits, dim=1)
            preds_argmax = torch.argmax(probs, dim=1)

            all_preds_argmax.append(preds_argmax.cpu().numpy())
            all_labels.append(y.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds_argmax = np.concatenate(all_preds_argmax)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    acc = accuracy_score(all_labels, all_preds_argmax)
    macro_f1 = f1_score(
        all_labels,
        all_preds_argmax,
        average="macro",
        zero_division=0,
    )

    return acc, macro_f1, all_preds_argmax, all_labels, all_probs


def build_model():
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

    return model


def make_prediction_df(meta, labels, preds_argmax, probs, threshold):
    preds_threshold, confidence = apply_confidence_threshold(
        probs,
        threshold=threshold,
        hold_label=1,
    )

    df = meta.copy()

    df["y_true"] = labels
    df["y_pred_argmax"] = preds_argmax
    df["y_pred"] = preds_threshold

    df["y_true_label"] = [LABEL_NAMES[i] for i in labels]
    df["y_pred_argmax_label"] = [LABEL_NAMES[i] for i in preds_argmax]
    df["y_pred_label"] = [LABEL_NAMES[i] for i in preds_threshold]

    df["prob_short"] = probs[:, 0]
    df["prob_hold"] = probs[:, 1]
    df["prob_long"] = probs[:, 2]
    df["confidence"] = confidence
    df["confidence_threshold"] = threshold

    return df


def export_one_rolling(rolling_name):
    print("\n" + "=" * 80)
    print(f"🚀 Export Prediction: {rolling_name}")
    print("=" * 80)

    rolling_dir = DATA_BASE_DIR / rolling_name
    output_dir = OUTPUT_BASE_DIR / rolling_name
    model_path = output_dir / "best_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    threshold = BEST_THRESHOLDS[rolling_name]
    print("confidence threshold:", threshold)

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

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_meta = pd.read_csv(rolling_dir / "sample_meta_val.csv")
    test_meta = pd.read_csv(rolling_dir / "sample_meta_test.csv")

    model = build_model()

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    print("Loaded:", model_path)

    val_acc, val_f1, val_preds_argmax, val_labels, val_probs = evaluate(
        model,
        val_loader,
    )

    val_pred_df = make_prediction_df(
        meta=val_meta,
        labels=val_labels,
        preds_argmax=val_preds_argmax,
        probs=val_probs,
        threshold=threshold,
    )

    val_save_path = output_dir / "val_predictions.csv"
    val_pred_df.to_csv(val_save_path, index=False, encoding="utf-8-sig")

    print("\n[Validation argmax]")
    print("acc:", round(val_acc, 4))
    print("f1 :", round(val_f1, 4))
    print("argmax != threshold count:", int((val_pred_df["y_pred_argmax"] != val_pred_df["y_pred"]).sum()))
    print("saved:", val_save_path)

    test_acc, test_f1, test_preds_argmax, test_labels, test_probs = evaluate(
        model,
        test_loader,
    )

    test_pred_df = make_prediction_df(
        meta=test_meta,
        labels=test_labels,
        preds_argmax=test_preds_argmax,
        probs=test_probs,
        threshold=threshold,
    )

    test_save_path = output_dir / "test_predictions.csv"
    test_pred_df.to_csv(test_save_path, index=False, encoding="utf-8-sig")

    print("\n[Test argmax]")
    print("acc:", round(test_acc, 4))
    print("f1 :", round(test_f1, 4))
    print("argmax != threshold count:", int((test_pred_df["y_pred_argmax"] != test_pred_df["y_pred"]).sum()))
    print("saved:", test_save_path)

    print("\n✅ Done:", rolling_name)


def main():
    print("DEVICE:", DEVICE)
    print("DATA_BASE_DIR:", DATA_BASE_DIR)
    print("OUTPUT_BASE_DIR:", OUTPUT_BASE_DIR)

    for rolling_name in ROLLING_NAMES:
        export_one_rolling(rolling_name)


if __name__ == "__main__":
    main()