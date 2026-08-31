"""Regression metrics for models that output predicted future returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate finite regression metrics in raw return units."""
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {actual.shape} != {predicted.shape}")
    if actual.ndim != 1 or len(actual) == 0:
        raise ValueError("y_true and y_pred must be non-empty one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Regression inputs must contain only finite values.")

    error = predicted - actual
    actual_rank = pd.Series(actual).rank(method="average").to_numpy(dtype=np.float64)
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy(dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
        "pearson_correlation": _correlation(actual, predicted),
        "spearman_correlation": _correlation(actual_rank, predicted_rank),
    }
