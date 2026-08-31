"""Aggregate rolling 1-3 TimesNet test predictions into connected OOS results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
EVALUATION_DIR = BENCHMARK_ROOT / "src" / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from evaluate_predictions import evaluate_prediction_frame  # noqa: E402
from regression import regression_metrics  # noqa: E402


def connect_predictions(paths: list[Path], identity: dict) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], utc=True)
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True)
    frame = frame.sort_values("sample_time").reset_index(drop=True)
    for column, value in identity.items():
        frame[column] = value
    return frame


def save_regression(output_root: Path) -> dict:
    paths = [
        output_root / f"rolling_{index}" / "rmse_selected" / "predictions_test.csv"
        for index in (1, 2, 3)
    ]
    frame = connect_predictions(
        paths,
        {
            "model": "timesnet_official_core_regressor_rmse_selected",
            "model_version": "timesnet_official_core_direct_target_v1",
            "rolling": "connected_oos",
            "split": "test",
            "seed": 42,
        },
    )
    destination = output_root / "connected_oos"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "predictions_test.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "schema_version": "1.0",
        "evaluation_variant": "connected_oos_regression",
        "identity": {
            "model": "timesnet_official_core_regressor_rmse_selected",
            "model_version": "timesnet_official_core_direct_target_v1",
            "rolling": "connected_oos",
            "split": "test",
            "seed": 42,
        },
        "num_rows": int(len(frame)),
        "regression": regression_metrics(
            frame["raw_future_return"].to_numpy(dtype=float),
            frame["predicted_return"].to_numpy(dtype=float),
        ),
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def save_classifier(output_root: Path) -> dict:
    paths = [output_root / f"rolling_{index}" / "predictions_test.csv" for index in (1, 2, 3)]
    frame = connect_predictions(
        paths,
        {
            "model": "timesnet_official_core_classifier",
            "model_version": "timesnet_official_core_direct_target_v1",
            "rolling": "connected_oos",
            "split": "test",
            "seed": 42,
        },
    )
    destination = output_root / "connected_oos"
    destination.mkdir(parents=True, exist_ok=True)
    prediction_path = destination / "predictions_test.csv"
    frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    metrics, trades = evaluate_prediction_frame(frame, fee=0.001, slippage=0.001)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trades.to_csv(destination / "trades.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> None:
    regression = save_regression(BENCHMARK_ROOT / "outputs" / "timesnet_regression")
    classifier = save_classifier(BENCHMARK_ROOT / "outputs" / "timesnet_classifier")
    print(json.dumps({"regression": regression, "classifier": classifier}, indent=2))


if __name__ == "__main__":
    main()
