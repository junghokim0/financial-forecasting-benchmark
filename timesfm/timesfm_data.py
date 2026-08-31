"""Leakage-safe data adapter for the TimesFM 2.5 benchmark.

TimesFM 2.5 is univariate and uses 32-step patches.  The benchmark's canonical
window is 72 hours, so this adapter uses the most recent 64 hours: the largest
model-native context contained entirely inside the common 72-hour window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


CANONICAL_WINDOW_SIZE = 72
MODEL_CONTEXT_LENGTH = 64
PREDICTION_HORIZON = 24


def rolling_dir(benchmark_root: Path, rolling: int) -> Path:
    path = benchmark_root / "data" / "dataset" / "rolling_threshold_0012" / f"rolling_{rolling}"
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_master(benchmark_root: Path) -> pd.DataFrame:
    path = benchmark_root / "data" / "master" / "merged_with_future_return.csv"
    frame = pd.read_csv(path)
    required = {"hour", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["hour"] = pd.to_datetime(frame["hour"], utc=True, errors="raise")
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    frame = frame.sort_values("hour").drop_duplicates("hour", keep="last").reset_index(drop=True)
    if not frame["hour"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Master data must be a continuous 1-hour time series.")
    close = frame["close"].to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("TimesFM close target must be finite and positive.")
    return frame


def load_metadata(rolling_path: Path, split: str) -> pd.DataFrame:
    path = rolling_path / f"sample_meta_{split}.csv"
    frame = pd.read_csv(path)
    required = {
        "input_start_time", "input_end_time", "sample_time", "target_time",
        "raw_future_return", "label_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    for column in ("input_start_time", "input_end_time", "sample_time", "target_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if not frame["input_end_time"].equals(frame["sample_time"]):
        raise ValueError(f"{path}: input_end_time must equal sample_time.")
    if not (frame["input_start_time"] == frame["sample_time"] - pd.Timedelta(hours=71)).all():
        raise ValueError(f"{path}: invalid canonical 72-hour input window.")
    if not (frame["target_time"] == frame["sample_time"] + pd.Timedelta(hours=24)).all():
        raise ValueError(f"{path}: invalid 24-hour target horizon.")
    return frame


def _build_windows(master: pd.DataFrame, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    hours = master["hour"].tolist()
    positions = {timestamp: index for index, timestamp in enumerate(hours)}
    close = master["close"].to_numpy(dtype=np.float32)
    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    for row in metadata.itertuples(index=False):
        if row.sample_time not in positions:
            raise ValueError(f"sample_time is absent from master: {row.sample_time}")
        end = positions[row.sample_time]
        canonical_start = end - CANONICAL_WINDOW_SIZE + 1
        future_end = end + PREDICTION_HORIZON
        if canonical_start < 0 or future_end >= len(close):
            raise ValueError(f"Insufficient close history/future at {row.sample_time}")
        if hours[canonical_start] != row.input_start_time or hours[future_end] != row.target_time:
            raise ValueError(f"Window boundary mismatch at {row.sample_time}")
        canonical = close[canonical_start : end + 1]
        future = close[end + 1 : future_end + 1]
        context = canonical[-MODEL_CONTEXT_LENGTH:]
        if context.shape != (MODEL_CONTEXT_LENGTH,) or future.shape != (PREDICTION_HORIZON,):
            raise ValueError(f"Unexpected window shape at {row.sample_time}")
        calculated_return = float(future[-1] / context[-1] - 1.0)
        if not np.isclose(calculated_return, float(row.raw_future_return), rtol=1e-5, atol=1e-7):
            raise ValueError(f"raw_future_return mismatch at {row.sample_time}")
        contexts.append(context)
        futures.append(future)
    context_array = np.stack(contexts).astype(np.float32, copy=False)
    future_array = np.stack(futures).astype(np.float32, copy=False)
    if not np.isfinite(context_array).all() or not np.isfinite(future_array).all():
        raise ValueError("TimesFM windows contain NaN or infinity.")
    return context_array, future_array


def load_split(
    benchmark_root: Path,
    rolling: int,
    split: str,
    master: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    master = load_master(benchmark_root) if master is None else master
    metadata = load_metadata(rolling_dir(benchmark_root, rolling), split)
    contexts, futures = _build_windows(master, metadata)
    return contexts, futures, metadata


class TimesFMDataset(Dataset):
    def __init__(self, contexts: np.ndarray, futures: np.ndarray):
        if contexts.shape != (len(futures), MODEL_CONTEXT_LENGTH):
            raise ValueError(f"Invalid context shape: {contexts.shape}")
        if futures.shape != (len(contexts), PREDICTION_HORIZON):
            raise ValueError(f"Invalid future shape: {futures.shape}")
        self.contexts = contexts
        self.futures = futures

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, index: int):
        return self.contexts[index], self.futures[index]


def read_split_bounds(benchmark_root: Path, rolling: int) -> dict:
    path = rolling_dir(benchmark_root, rolling) / "rolling_config.json"
    return json.loads(path.read_text(encoding="utf-8"))["split"]
