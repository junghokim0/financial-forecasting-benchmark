"""Validate Chronos-2 inputs without downloading or executing the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chronos2_data import (
    load_inference_windows,
    load_master,
    make_continuous_fit_input,
    rolling_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.benchmark_root.resolve()
    master = load_master(root)
    report = []
    for rolling in (1, 2, 3):
        roll_path = rolling_dir(root, rolling)
        train = make_continuous_fit_input(master, roll_path, "train")[0]
        validation = make_continuous_fit_input(master, roll_path, "val")[0]
        row = {
            "rolling": rolling,
            "input_mode": "chart_12",
            "train_target_length": len(train["target"]),
            "validation_target_length": len(validation["target"]),
            "covariate_count": len(train["past_covariates"]),
        }
        for split in ("val", "test"):
            close, features, names, metadata = load_inference_windows(
                root, rolling, split, master
            )
            row[f"{split}_close_shape"] = list(close.shape)
            row[f"{split}_feature_shape"] = list(features.shape)
            row[f"{split}_feature_count"] = len(names)
            row[f"{split}_rows"] = len(metadata)
        report.append(row)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
