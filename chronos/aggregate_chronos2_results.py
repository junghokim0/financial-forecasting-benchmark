"""Connect Rolling 1-3 Chronos-2 test predictions and evaluate OOS results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parent.parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.resolve()
    evaluation_path = root / "src" / "evaluation"
    if str(evaluation_path) not in sys.path:
        sys.path.insert(0, str(evaluation_path))
    from evaluate_predictions import evaluate_prediction_frame

    output_root = root / "outputs" / "chronos2_lora_chart"
    paths = [output_root / f"rolling_{rolling}" / "predictions_test.csv" for rolling in (1, 2, 3)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run all three rolling models first. Missing: {missing}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    frame = frame.sort_values("sample_time").reset_index(drop=True)
    frame["rolling"] = "connected_oos"
    frame["split"] = "test"

    destination = output_root / "connected_oos"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "predictions_test.csv", index=False, encoding="utf-8-sig")
    metrics, trades = evaluate_prediction_frame(frame, fee=0.001, slippage=0.001)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trades.to_csv(destination / "trades.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
