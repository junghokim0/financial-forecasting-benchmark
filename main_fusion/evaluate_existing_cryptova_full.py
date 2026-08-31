"""Re-evaluate preserved Cryptova-Full predictions with the common evaluator.

This script never loads a model and never modifies the original experiment.
It converts the preserved confidence + funding-rate + std_24h filtered test
predictions into the benchmark schema, evaluates Rolling 1-3 independently,
then evaluates their connected OOS sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODEL = "cryptova_full"
MODEL_VERSION = "original12_fusion_confidence_funding_vol_filter_v1"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output-name",
        default="cryptova_full_re_evaluation",
        help="New directory under benchmark/outputs. Existing directories are refused.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_prediction_path(source_root: Path, rolling: int) -> Path:
    return (
        source_root
        / "main_fusion"
        / "outputs"
        / "outputs"
        / "original12_funding_vol_filter"
        / "threshold"
        / f"rolling_{rolling}"
        / "test_predictions_with_funding_vol_filter.csv"
    )


def canonical_metadata_path(benchmark_root: Path, rolling: int) -> Path:
    return (
        benchmark_root
        / "data"
        / "dataset"
        / "rolling_threshold_0012"
        / f"rolling_{rolling}"
        / "sample_meta_test.csv"
    )


def load_and_validate_source(
    source_path: Path, metadata_path: Path, rolling: int
) -> pd.DataFrame:
    source = pd.read_csv(source_path)
    metadata = pd.read_csv(metadata_path)
    required = {
        "sample_time", "target_time", "raw_future_return", "y_true",
        "pred_base", "pred_filtered", "y_pred_argmax", "prob_short",
        "prob_hold", "prob_long", "confidence", "confidence_threshold",
        "funding_rate", "std_24h", "funding_filter_threshold",
        "vol_filter_threshold",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{source_path} is missing columns: {missing}")
    if len(source) != len(metadata):
        raise ValueError(f"rolling_{rolling}: source/benchmark row count mismatch")

    for frame in (source, metadata):
        frame["sample_time"] = pd.to_datetime(frame["sample_time"], utc=True, errors="raise")
        frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True, errors="raise")
    source = source.sort_values("sample_time").reset_index(drop=True)
    metadata = metadata.sort_values("sample_time").reset_index(drop=True)
    if not source["sample_time"].equals(metadata["sample_time"]):
        raise ValueError(f"rolling_{rolling}: sample_time mismatch")
    if not source["target_time"].equals(metadata["target_time"]):
        raise ValueError(f"rolling_{rolling}: target_time mismatch")
    if not np.array_equal(
        source["y_true"].to_numpy(dtype=np.int64),
        metadata["label_id"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(f"rolling_{rolling}: y_true mismatch")
    if not np.allclose(
        source["raw_future_return"].to_numpy(dtype=np.float64),
        metadata["raw_future_return"].to_numpy(dtype=np.float64),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"rolling_{rolling}: raw_future_return mismatch")

    final_prediction = source["pred_filtered"].to_numpy(dtype=np.int64)
    if not set(np.unique(final_prediction)).issubset({0, 1, 2}):
        raise ValueError(f"rolling_{rolling}: invalid pred_filtered labels")
    probabilities = source[["prob_short", "prob_hold", "prob_long"]].to_numpy(dtype=np.float64)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"rolling_{rolling}: base probabilities do not sum to one")

    # base_prob_* names intentionally distinguish pre-filter model probabilities
    # from final post-filter predictions required by the common evaluator.
    return pd.DataFrame(
        {
            "schema_version": "1.0",
            "model": MODEL,
            "model_version": MODEL_VERSION,
            "rolling": f"rolling_{rolling}",
            "split": "test",
            "seed": SEED,
            "sample_time": source["sample_time"],
            "target_time": source["target_time"],
            "y_true": source["y_true"].astype(np.int64),
            "raw_future_return": source["raw_future_return"].astype(np.float64),
            "y_pred": final_prediction,
            "base_y_pred": source["pred_base"].astype(np.int64),
            "base_y_pred_argmax": source["y_pred_argmax"].astype(np.int64),
            "base_prob_short": source["prob_short"].astype(np.float64),
            "base_prob_hold": source["prob_hold"].astype(np.float64),
            "base_prob_long": source["prob_long"].astype(np.float64),
            "base_confidence": source["confidence"].astype(np.float64),
            "confidence_threshold": source["confidence_threshold"].astype(np.float64),
            "funding_rate": source["funding_rate"].astype(np.float64),
            "std_24h": source["std_24h"].astype(np.float64),
            "funding_filter_threshold": source["funding_filter_threshold"].astype(np.float64),
            "vol_filter_threshold": source["vol_filter_threshold"].astype(np.float64),
        }
    )


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
        raise FileExistsError(
            f"Refusing to overwrite existing re-evaluation: {output_root}"
        )
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
            if not source_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(source_path if not source_path.is_file() else metadata_path)
            frame = load_and_validate_source(source_path, metadata_path, rolling)
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
            "final_prediction_column": "pred_filtered",
            "original_files_modified": False,
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
        # Keep a failed run visible for diagnosis; never touch source experiment files.
        (output_root / "FAILED.txt").write_text(
            "Re-evaluation did not complete. Inspect the exception output.\n", encoding="utf-8"
        )
        raise
    print(json.dumps({"output": str(output_root), "connected_oos": connected_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
