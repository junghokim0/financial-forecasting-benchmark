"""Validate all TimesFM windows without downloading or executing the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from timesfm_data import (
    CANONICAL_WINDOW_SIZE,
    MODEL_CONTEXT_LENGTH,
    PREDICTION_HORIZON,
    load_master,
    load_split,
    read_split_bounds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    master = load_master(root)
    report = []
    for rolling in (1, 2, 3):
        row = {
            "rolling": rolling,
            "input_mode": "close_only",
            "canonical_window": CANONICAL_WINDOW_SIZE,
            "model_context": MODEL_CONTEXT_LENGTH,
            "horizon": PREDICTION_HORIZON,
            "split_bounds": read_split_bounds(root, rolling),
        }
        for split in ("train", "val", "test"):
            context, future, metadata = load_split(root, rolling, split, master)
            row[f"{split}_rows"] = len(metadata)
            row[f"{split}_context_shape"] = list(context.shape)
            row[f"{split}_future_shape"] = list(future.shape)
        report.append(row)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
