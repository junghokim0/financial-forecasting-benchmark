"""NumPy implementation of Ridge Regression for flattened time-series windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def flatten_chart_windows(windows: np.ndarray) -> np.ndarray:
    """Convert (samples, time, features) chart windows to (samples, time*features)."""
    values = np.asarray(windows)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3-D chart tensor, received shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Chart tensor contains NaN or infinite values.")
    return values.reshape(values.shape[0], -1).astype(np.float64, copy=False)


@dataclass(frozen=True)
class RidgeModel:
    """Fitted Ridge model with an unregularized intercept."""

    alpha: float
    coefficient: np.ndarray
    intercept: float
    input_shape: tuple[int, int]

    def predict(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.coefficient.shape[0]:
            raise ValueError(
                f"Expected (*, {self.coefficient.shape[0]}) features, got {matrix.shape}."
            )
        predictions = matrix @ self.coefficient + self.intercept
        if not np.isfinite(predictions).all():
            raise ValueError("Model produced non-finite predictions.")
        return predictions

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            alpha=np.asarray(self.alpha, dtype=np.float64),
            coefficient=self.coefficient.astype(np.float64),
            intercept=np.asarray(self.intercept, dtype=np.float64),
            input_shape=np.asarray(self.input_shape, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Path) -> "RidgeModel":
        with np.load(path) as artifact:
            return cls(
                alpha=float(artifact["alpha"]),
                coefficient=np.asarray(artifact["coefficient"], dtype=np.float64),
                intercept=float(artifact["intercept"]),
                input_shape=tuple(int(v) for v in artifact["input_shape"]),
            )


class RidgeSVDPath:
    """Fit one SVD and reuse it to evaluate multiple Ridge alpha values.

    The intercept is handled by centering X and y and is not regularized. The
    Ridge solution is computed as V diag(s / (s^2 + alpha)) U^T y, avoiding an
    explicit inverse of X^T X.
    """

    def __init__(self, features: np.ndarray, target: np.ndarray, input_shape: tuple[int, int]):
        matrix = np.asarray(features, dtype=np.float64)
        values = np.asarray(target, dtype=np.float64)
        if matrix.ndim != 2 or values.ndim != 1:
            raise ValueError("features must be 2-D and target must be 1-D.")
        if matrix.shape[0] != values.shape[0] or matrix.shape[0] == 0:
            raise ValueError("features and target must contain the same non-zero sample count.")
        if not np.isfinite(matrix).all() or not np.isfinite(values).all():
            raise ValueError("Training data contains NaN or infinite values.")

        self.feature_mean = matrix.mean(axis=0)
        self.target_mean = float(values.mean())
        centered_features = matrix - self.feature_mean
        centered_target = values - self.target_mean
        _, singular_values, right_vectors = np.linalg.svd(
            centered_features, full_matrices=False
        )

        # Recompute U^T y without retaining the large U matrix:
        # X^T y = V diag(s) U^T y, hence U^T y = (V^T X^T y) / s.
        projected_xy = right_vectors @ (centered_features.T @ centered_target)
        tolerance = np.finfo(np.float64).eps * max(centered_features.shape) * (
            singular_values.max() if singular_values.size else 0.0
        )
        ut_y = np.divide(
            projected_xy,
            singular_values,
            out=np.zeros_like(projected_xy),
            where=singular_values > tolerance,
        )

        self.singular_values = singular_values
        self.right_vectors = right_vectors
        self.ut_y = ut_y
        self.input_shape = input_shape

    def model(self, alpha: float) -> RidgeModel:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Ridge alpha must be a finite positive number.")
        factors = self.singular_values / (self.singular_values**2 + float(alpha))
        coefficient = self.right_vectors.T @ (factors * self.ut_y)
        intercept = self.target_mean - float(self.feature_mean @ coefficient)
        return RidgeModel(
            alpha=float(alpha),
            coefficient=coefficient,
            intercept=intercept,
            input_shape=self.input_shape,
        )
