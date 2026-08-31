"""Re-evaluate preserved Cryptova-Base (Fusion + confidence) predictions.

No model is loaded. The preserved ``pred_base`` column is evaluated while the
funding-rate/std_24h risk-filter output is ignored. Original files are read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from evaluate_existing_cryptova_full import (
    SEED,
    canonical_metadata_path,
    load_and_validate_source,
    sha256,
    source_prediction_path,
)


MODEL = "cryptova_base"
MODEL_VERSION = "original12_fusion_confidence_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-name", default="cryptova_base_re_evaluation")
    return parser.parse_args()


def base_frame(source_path: Path, metadata_path: Path, rolling: int) -> pd.DataFrame:
    frame = load_and_validate_source(source_path, metadata_path, rolling)
    frame["model"] = MODEL
    frame["model_version"] = MODEL_VERSION
    frame["y_pred"] = frame["base_y_pred"].astype("int64")
    return frame


def evaluate_and_save(frame: pd.DataFrame, destination: Path, evaluator) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    frame.to_csv(destination / "predictions_test.csv", index=False, encoding="utf-8-sig")
    metrics, trades = evaluator(frame, fee=0.001, slippage=0.001)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trades.to_csv(destination / "trades.csv", index=False, encoding="utf-8-sig")
    return metrics


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    source_root = args.source_root.resolve()
    output_root = benchmark_root / "outputs" / args.output_name
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing re-evaluation: {output_root}")
    evaluation_path = benchmark_root / "src" / "evaluation"
    if str(evaluation_path) not in sys.path:
        sys.path.insert(0, str(evaluation_path))
    from evaluate_predictions import evaluate_prediction_frame

    frames = []
    rolling_metrics = {}
    sources = []
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        for rolling in (1, 2, 3):
            source_path = source_prediction_path(source_root, rolling)
            metadata_path = canonical_metadata_path(benchmark_root, rolling)
            frame = base_frame(source_path, metadata_path, rolling)
            rolling_metrics[f"rolling_{rolling}"] = evaluate_and_save(
                frame, output_root / f"rolling_{rolling}", evaluate_prediction_frame
            )
            frames.append(frame)
            sources.append(
                {
                    "rolling": f"rolling_{rolling}",
                    "prediction_path": str(source_path),
                    "prediction_sha256": sha256(source_path),
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": sha256(metadata_path),
                    "rows": len(frame),
                }
            )

        connected = pd.concat(frames, ignore_index=True).sort_values("sample_time").reset_index(drop=True)
        if connected["sample_time"].duplicated().any():
            raise ValueError("Connected OOS contains duplicate sample_time values")
        connected["rolling"] = "connected_oos"
        connected_metrics = evaluate_and_save(
            connected, output_root / "connected_oos", evaluate_prediction_frame
        )
        manifest = {
            "model": MODEL,
            "model_version": MODEL_VERSION,
            "evaluation": "existing predictions; no training and no inference",
            "source_variant": "original12_funding_vol_filter/threshold",
            "final_prediction_column": "pred_base",
            "definition": "Fusion model plus validation-selected confidence threshold; no risk filter",
            "original_files_modified": False,
            "seed": SEED,
            "fee": 0.001,
            "slippage": 0.001,
            "sources": sources,
            "rolling_metrics": rolling_metrics,
            "connected_oos_metrics": connected_metrics,
        }
        (output_root / "re_evaluation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        (output_root / "FAILED.txt").write_text(
            "Re-evaluation did not complete. Inspect the exception output.\n", encoding="utf-8"
        )
        raise
    print(json.dumps({"output": str(output_root), "connected_oos": connected_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
