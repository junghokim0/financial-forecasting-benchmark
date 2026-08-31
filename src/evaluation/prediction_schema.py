"""Canonical prediction-table contract for the financial benchmark.

Every model writes validation/test predictions in this format. Evaluation and
backtesting consume this table without importing model-specific code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


SCHEMA_VERSION: Final[str] = "1.0"
LABELS: Final[set[int]] = {0, 1, 2}
SPLITS: Final[set[str]] = {"validation", "test"}

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "schema_version",
    "model",
    "model_version",
    "rolling",
    "split",
    "seed",
    "sample_time",
    "target_time",
    "y_true",
    "raw_future_return",
    "y_pred",
)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "predicted_return",
    "prob_short",
    "prob_hold",
    "prob_long",
    "confidence",
)


@dataclass(frozen=True)
class PredictionSchemaConfig:
    short_threshold: float = -0.012
    long_threshold: float = 0.012
    probability_tolerance: float = 1e-5


def returns_to_classes(
    predicted_return: pd.Series | np.ndarray,
    config: PredictionSchemaConfig = PredictionSchemaConfig(),
) -> np.ndarray:
    """Convert predicted 24-hour returns to SHORT/HOLD/LONG labels."""
    values = np.asarray(predicted_return, dtype=np.float64)
    labels = np.full(values.shape, 1, dtype=np.int64)
    labels[values <= config.short_threshold] = 0
    labels[values >= config.long_threshold] = 2
    return labels


def validate_prediction_frame(
    frame: pd.DataFrame,
    config: PredictionSchemaConfig = PredictionSchemaConfig(),
) -> pd.DataFrame:
    """Validate and normalize a model prediction table.

    Returns a copy with UTC timestamps and, when probabilities are supplied,
    a checked/derived confidence column. Raises ValueError on contract errors.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    result = frame.copy()
    if result.empty:
        raise ValueError("Prediction table is empty.")

    if not (result["schema_version"].astype(str) == SCHEMA_VERSION).all():
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}.")

    for column in ("model", "model_version", "rolling"):
        if result[column].isna().any() or (result[column].astype(str).str.len() == 0).any():
            raise ValueError(f"{column} must contain non-empty values.")

    invalid_splits = set(result["split"].astype(str)) - SPLITS
    if invalid_splits:
        raise ValueError(f"Invalid split values: {sorted(invalid_splits)}")

    for column in ("sample_time", "target_time"):
        result[column] = pd.to_datetime(result[column], utc=True, errors="raise")

    if (result["target_time"] <= result["sample_time"]).any():
        raise ValueError("Every target_time must be later than sample_time.")

    duplicate_key = ["model", "rolling", "split", "seed", "sample_time"]
    if result.duplicated(duplicate_key).any():
        raise ValueError(f"Duplicate prediction keys found: {duplicate_key}")

    for column in ("y_true", "y_pred"):
        numeric = pd.to_numeric(result[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain integer labels.")
        result[column] = numeric.astype(np.int64)
        invalid = set(result[column].unique()) - LABELS
        if invalid:
            raise ValueError(f"{column} contains invalid labels: {sorted(invalid)}")

    result["raw_future_return"] = pd.to_numeric(
        result["raw_future_return"], errors="raise"
    ).astype(np.float64)
    if not np.isfinite(result["raw_future_return"]).all():
        raise ValueError("raw_future_return must contain finite values.")

    if "predicted_return" in result.columns:
        non_null = result["predicted_return"].notna()
        values = pd.to_numeric(result.loc[non_null, "predicted_return"], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError("Non-null predicted_return values must be finite.")

    probability_columns = ["prob_short", "prob_hold", "prob_long"]
    present_probability_columns = [c for c in probability_columns if c in result.columns]
    if present_probability_columns and len(present_probability_columns) != 3:
        raise ValueError("Provide all three probability columns or none of them.")

    if len(present_probability_columns) == 3:
        probabilities = result[probability_columns].apply(pd.to_numeric, errors="raise")
        if probabilities.isna().any().any():
            raise ValueError("Probability columns cannot contain NaN.")
        if ((probabilities < 0) | (probabilities > 1)).any().any():
            raise ValueError("Probabilities must be between 0 and 1.")
        sums = probabilities.sum(axis=1).to_numpy()
        if not np.allclose(sums, 1.0, atol=config.probability_tolerance, rtol=0.0):
            raise ValueError("Class probabilities must sum to 1 for every row.")
        expected_prediction = probabilities.to_numpy().argmax(axis=1)
        if not np.array_equal(expected_prediction, result["y_pred"].to_numpy()):
            raise ValueError("y_pred must equal argmax of class probabilities.")
        expected_confidence = probabilities.max(axis=1).to_numpy()
        if "confidence" in result.columns:
            supplied = pd.to_numeric(result["confidence"], errors="raise").to_numpy()
            if not np.allclose(
                supplied,
                expected_confidence,
                atol=config.probability_tolerance,
                rtol=0.0,
            ):
                raise ValueError("confidence must equal max class probability.")
        else:
            result["confidence"] = expected_confidence

    result = result.sort_values(
        ["model", "rolling", "split", "seed", "sample_time"]
    ).reset_index(drop=True)
    return result


def empty_prediction_frame() -> pd.DataFrame:
    """Return an empty table containing the complete canonical column order."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
