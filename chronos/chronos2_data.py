"""Data adapters for the Cryptova Chronos-2 benchmark.

Chronos-2 is trained on a continuous target series, while benchmark inference
must use the exact 72-hour windows and timestamps used by the other models.
This module keeps those two representations consistent and validates every
time boundary before a model is called.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import joblib
import numpy as np
import pandas as pd


WINDOW_SIZE = 72
PREDICTION_HORIZON = 24

CHART_FEATURES = (
    "log_return",
    "return_6h",
    "return_24h",
    "std_24h",
    "close_ma24_gap",
    "close_ma72_gap",
    "volume_ratio_24",
    "spread_ratio",
    "macd_hist",
    "hour_sin",
    "hour_cos",
    "is_missing_candle",
)

@dataclass(frozen=True)
class SplitBounds:
    start: pd.Timestamp
    end: pd.Timestamp


def load_master(benchmark_root: Path) -> pd.DataFrame:
    path = benchmark_root / "data" / "master" / "merged_with_future_return.csv"
    frame = pd.read_csv(path)
    required = {"hour", "close", *CHART_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["hour"] = pd.to_datetime(frame["hour"], utc=True, errors="raise")
    frame = frame.sort_values("hour").drop_duplicates("hour", keep="last").reset_index(drop=True)
    if not frame["hour"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Master data must be a continuous 1-hour time series.")
    numeric = frame[["close", *CHART_FEATURES]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("Master target/features contain NaN or infinity.")
    if (numeric["close"] <= 0).any():
        raise ValueError("Chronos target close must be positive.")
    frame[["close", *CHART_FEATURES]] = numeric
    return frame


def rolling_dir(benchmark_root: Path, rolling: int) -> Path:
    path = (
        benchmark_root
        / "data"
        / "dataset"
        / "rolling_threshold_0012"
        / f"rolling_{rolling}"
    )
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_bounds(rolling_path: Path, split: str) -> SplitBounds:
    config = json.loads((rolling_path / "rolling_config.json").read_text(encoding="utf-8"))
    start, end = config["split"][split]
    return SplitBounds(pd.Timestamp(start), pd.Timestamp(end))


def load_metadata(rolling_path: Path, split: str) -> pd.DataFrame:
    path = rolling_path / f"sample_meta_{split}.csv"
    frame = pd.read_csv(path)
    required = {
        "input_start_time",
        "input_end_time",
        "sample_time",
        "target_time",
        "raw_future_return",
        "label_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    for column in ("input_start_time", "input_end_time", "sample_time", "target_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not frame["input_end_time"].equals(frame["sample_time"]):
        raise ValueError(f"{path}: input_end_time must equal sample_time.")
    if not (
        frame["input_start_time"]
        == frame["sample_time"] - pd.Timedelta(hours=WINDOW_SIZE - 1)
    ).all():
        raise ValueError(f"{path}: invalid 72-hour input window.")
    if not (
        frame["target_time"]
        == frame["sample_time"] + pd.Timedelta(hours=PREDICTION_HORIZON)
    ).all():
        raise ValueError(f"{path}: invalid 24-hour prediction horizon.")
    return frame


def _load_scaled_chart_features(
    master: pd.DataFrame,
    rolling_path: Path,
) -> np.ndarray:
    chart_scaler = joblib.load(rolling_path / "chart_scaler.pkl")
    return chart_scaler.transform(master.loc[:, CHART_FEATURES]).astype(np.float32)


def make_continuous_fit_input(
    master: pd.DataFrame,
    rolling_path: Path,
    split: str,
) -> list[dict]:
    """Build one continuous Chronos task fully contained inside a split.

    The target is raw BTC close. Covariates use the rolling Train-fitted
    StandardScaler already produced by the canonical dataset pipeline.
    Chronos-2 additionally applies its model-native instance normalization.
    """
    bounds = load_bounds(rolling_path, split)
    mask = (master["hour"] >= bounds.start) & (master["hour"] < bounds.end)
    split_frame = master.loc[mask].copy()
    if len(split_frame) < WINDOW_SIZE + PREDICTION_HORIZON:
        raise ValueError(f"{rolling_path.name}/{split}: not enough continuous rows.")
    expected_start = bounds.start
    expected_end = bounds.end - pd.Timedelta(hours=1)
    if split_frame["hour"].iloc[0] != expected_start or split_frame["hour"].iloc[-1] != expected_end:
        raise ValueError(f"{rolling_path.name}/{split}: split does not cover its declared bounds.")

    scaled = _load_scaled_chart_features(master, rolling_path)
    split_positions = np.flatnonzero(mask.to_numpy())
    covariates = {
        name: scaled[split_positions, index].astype(np.float32, copy=False)
        for index, name in enumerate(CHART_FEATURES)
    }
    return [
        {
            "target": split_frame["close"].to_numpy(dtype=np.float32),
            "past_covariates": covariates,
        }
    ]


def _close_windows(master: pd.DataFrame, metadata: pd.DataFrame) -> np.ndarray:
    hours = master["hour"].tolist()
    position = {timestamp: index for index, timestamp in enumerate(hours)}
    close = master["close"].to_numpy(dtype=np.float32)
    windows: list[np.ndarray] = []
    for row in metadata.itertuples(index=False):
        try:
            end = position[row.sample_time]
        except KeyError as exc:
            raise ValueError(f"sample_time is absent from master: {row.sample_time}") from exc
        start = end - WINDOW_SIZE + 1
        if start < 0:
            raise ValueError(f"Not enough close history for {row.sample_time}.")
        if hours[start] != row.input_start_time:
            raise ValueError(f"Window boundary mismatch at {row.sample_time}.")
        window = close[start : end + 1]
        if window.shape != (WINDOW_SIZE,):
            raise ValueError(f"Invalid close window at {row.sample_time}: {window.shape}")
        windows.append(window)
    return np.stack(windows).astype(np.float32, copy=False)


def load_inference_windows(
    benchmark_root: Path,
    rolling: int,
    split: str,
    master: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], pd.DataFrame]:
    rolling_path = rolling_dir(benchmark_root, rolling)
    metadata = load_metadata(rolling_path, split)
    master = load_master(benchmark_root) if master is None else master
    close_windows = _close_windows(master, metadata)

    chart = np.load(rolling_path / f"X_chart_{split}.npy").astype(np.float32, copy=False)
    if chart.shape != (len(metadata), WINDOW_SIZE, len(CHART_FEATURES)):
        raise ValueError(f"Unexpected chart tensor shape: {chart.shape}")
    if not np.isfinite(chart).all() or not np.isfinite(close_windows).all():
        raise ValueError("Inference windows contain NaN or infinity.")
    return close_windows, chart, CHART_FEATURES, metadata


def iter_chronos_input_batches(
    close_windows: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    chunk_size: int,
) -> Iterator[tuple[int, int, list[dict]]]:
    if close_windows.shape[:2] != features.shape[:2]:
        raise ValueError("Target and covariate windows are not aligned.")
    if features.shape[-1] != len(feature_names):
        raise ValueError("Feature name count does not match tensor width.")
    for start in range(0, len(close_windows), chunk_size):
        end = min(start + chunk_size, len(close_windows))
        inputs = []
        for row in range(start, end):
            inputs.append(
                {
                    "target": close_windows[row],
                    "past_covariates": {
                        name: features[row, :, column]
                        for column, name in enumerate(feature_names)
                    },
                }
            )
        yield start, end, inputs
