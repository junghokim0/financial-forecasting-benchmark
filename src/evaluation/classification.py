"""Classification metrics extracted from Cryptova's fusion training script."""

from __future__ import annotations

from typing import Final

import numpy as np


LABEL_IDS: Final[list[int]] = [0, 1, 2]
LABEL_NAMES: Final[list[str]] = ["SHORT", "HOLD", "LONG"]


def prediction_distribution(predictions: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Reproduce Cryptova's per-class count and ratio output."""
    predictions = np.asarray(predictions, dtype=np.int64)
    total = len(predictions)
    result: dict[str, dict[str, float | int]] = {}
    for label_id, label_name in enumerate(LABEL_NAMES):
        count = int((predictions == label_id).sum())
        result[label_name] = {
            "count": count,
            "ratio": float(count / total) if total > 0 else 0.0,
        }
    return result


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate Cryptova metrics plus non-breaking supplementary metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true.shape} != {y_pred.shape}")
    if y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError("y_true and y_pred must be non-empty one-dimensional arrays.")

    matrix = np.zeros((3, 3), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[true_label, predicted_label] += 1

    support = matrix.sum(axis=1).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(3, dtype=np.float64),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(3, dtype=np.float64),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(3, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    accuracy = float(true_positive.sum() / len(y_true))
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.average(f1, weights=support))
    report = {
        label_name: {
            "precision": float(precision[label_id]),
            "recall": float(recall[label_id]),
            "f1-score": float(f1[label_id]),
            "support": float(support[label_id]),
        }
        for label_id, label_name in enumerate(LABEL_NAMES)
    }
    report.update(
        {
            "accuracy": accuracy,
            "macro avg": {
                "precision": float(precision.mean()),
                "recall": float(recall.mean()),
                "f1-score": macro_f1,
                "support": float(support.sum()),
            },
            "weighted avg": {
                "precision": float(np.average(precision, weights=support)),
                "recall": float(np.average(recall, weights=support)),
                "f1-score": weighted_f1,
                "support": float(support.sum()),
            },
        }
    )

    return {
        # Original Cryptova outputs
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "prediction_distribution": prediction_distribution(y_pred),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        # Supplementary metrics; these do not alter the original calculations.
        "balanced_accuracy": float(recall.mean()),
        "weighted_f1": weighted_f1,
    }
