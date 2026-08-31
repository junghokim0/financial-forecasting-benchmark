"""Train and evaluate Ridge-Flat on all benchmark rolling splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
EVALUATION_DIR = BENCHMARK_ROOT / "src" / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from evaluate_predictions import evaluate_prediction_csv  # noqa: E402
from classification import classification_metrics  # noqa: E402
from prediction_schema import SCHEMA_VERSION, returns_to_classes  # noqa: E402
from regression import regression_metrics  # noqa: E402
from ridge_flat import RidgeSVDPath, flatten_chart_windows  # noqa: E402


DEFAULT_ALPHAS = (
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
    10000000000.0,
)
MODEL_NAME = "ridge_flat"
MODEL_VERSION = "ridge_flat_svd_v1"


def load_split(rolling_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    tensor_path = rolling_dir / f"X_chart_{split}.npy"
    metadata_path = rolling_dir / f"sample_meta_{split}.csv"
    tensor = np.load(tensor_path)
    metadata = pd.read_csv(metadata_path)
    if len(tensor) != len(metadata):
        raise ValueError(
            f"{rolling_dir.name}/{split}: tensor rows {len(tensor)} != metadata rows {len(metadata)}"
        )
    required = {"sample_time", "target_time", "raw_future_return", "label_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"{metadata_path} is missing columns: {missing}")
    return tensor, metadata


def build_prediction_frame(
    metadata: pd.DataFrame,
    predicted_return: np.ndarray,
    rolling: str,
    split: str,
    seed: int,
    model_name: str,
) -> pd.DataFrame:
    predicted_return = np.asarray(predicted_return, dtype=np.float64)
    if len(metadata) != len(predicted_return):
        raise ValueError("Metadata/prediction length mismatch.")
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "model": model_name,
            "model_version": MODEL_VERSION,
            "rolling": rolling,
            "split": "validation" if split == "val" else split,
            "seed": int(seed),
            "sample_time": metadata["sample_time"],
            "target_time": metadata["target_time"],
            "y_true": metadata["label_id"].astype(np.int64),
            "raw_future_return": metadata["raw_future_return"].astype(np.float64),
            "y_pred": returns_to_classes(predicted_return),
            "predicted_return": predicted_return,
        }
    )


def save_selected_variant(
    model,
    variant_name: str,
    selection_metric: str,
    selection_value: float,
    rolling_output: Path,
    rolling: str,
    seed: int,
    x_val: np.ndarray,
    val_meta: pd.DataFrame,
    x_test: np.ndarray,
    test_meta: pd.DataFrame,
    fee: float,
    slippage: float,
) -> dict:
    """Save one validation-selected Ridge variant and evaluate val/test."""
    variant_output = rolling_output / variant_name
    variant_output.mkdir(parents=True, exist_ok=True)
    model.save(variant_output / "model.npz")
    schema_model_name = f"ridge_flat_{variant_name}"
    summary = {
        "model": schema_model_name,
        "model_version": MODEL_VERSION,
        "rolling": rolling,
        "seed": int(seed),
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
        "selected_alpha": float(model.alpha),
        "solver": "numpy_svd_closed_form_no_explicit_inverse",
        "refit_on_train_plus_validation": False,
        "input_scaling": "existing_rolling_train_fitted_chart_scaler; no additional refit",
    }
    for split, matrix, metadata in (
        ("val", x_val, val_meta),
        ("test", x_test, test_meta),
    ):
        prediction = model.predict(matrix)
        frame = build_prediction_frame(
            metadata, prediction, rolling, split, seed, schema_model_name
        )
        csv_path = variant_output / f"predictions_{split}.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        metrics_dir = variant_output / f"evaluation_{split}"
        metrics_path, _ = evaluate_prediction_csv(
            csv_path, metrics_dir, fee=fee, slippage=slippage
        )
        summary[f"{split}_prediction_csv"] = str(csv_path)
        summary[f"{split}_metrics_json"] = str(metrics_path)
    (variant_output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def train_one_rolling(
    rolling_dir: Path,
    output_root: Path,
    alphas: tuple[float, ...],
    seed: int,
    fee: float,
    slippage: float,
) -> dict:
    rolling = rolling_dir.name
    print(f"[{rolling}] Loading data...", flush=True)
    train_tensor, train_meta = load_split(rolling_dir, "train")
    val_tensor, val_meta = load_split(rolling_dir, "val")
    test_tensor, test_meta = load_split(rolling_dir, "test")

    expected_shape = (72, 12)
    for split_name, tensor in (("train", train_tensor), ("val", val_tensor), ("test", test_tensor)):
        if tuple(tensor.shape[1:]) != expected_shape:
            raise ValueError(
                f"{rolling}/{split_name}: expected window shape {expected_shape}, got {tensor.shape[1:]}"
            )

    x_train = flatten_chart_windows(train_tensor)
    x_val = flatten_chart_windows(val_tensor)
    x_test = flatten_chart_windows(test_tensor)
    y_train = train_meta["raw_future_return"].to_numpy(dtype=np.float64)
    y_val = val_meta["raw_future_return"].to_numpy(dtype=np.float64)

    print(f"[{rolling}] Computing one SVD for {x_train.shape}...", flush=True)
    path = RidgeSVDPath(x_train, y_train, input_shape=expected_shape)
    search_rows: list[dict] = []
    selected_rmse_model = None
    selected_macro_f1_model = None
    selected_rmse = float("inf")
    selected_macro_f1 = float("-inf")
    selected_macro_f1_tiebreak_rmse = float("inf")

    for alpha in alphas:
        model = path.model(alpha)
        val_prediction = model.predict(x_val)
        regression = regression_metrics(y_val, val_prediction)
        predicted_class = returns_to_classes(val_prediction)
        classification = classification_metrics(
            val_meta["label_id"].to_numpy(dtype=np.int64), predicted_class
        )
        macro_f1 = float(classification["macro_f1"])
        search_rows.append(
            {
                "alpha": float(alpha),
                **regression,
                "macro_f1": macro_f1,
                "accuracy": float(classification["accuracy"]),
                "pred_short_count": int((predicted_class == 0).sum()),
                "pred_hold_count": int((predicted_class == 1).sum()),
                "pred_long_count": int((predicted_class == 2).sum()),
            }
        )
        print(
            f"[{rolling}] alpha={alpha:g}, val_rmse={regression['rmse']:.8f}, "
            f"val_macro_f1={macro_f1:.6f}",
            flush=True,
        )
        if regression["rmse"] < selected_rmse:
            selected_rmse = regression["rmse"]
            selected_rmse_model = model
        if (
            macro_f1 > selected_macro_f1
            or (
                np.isclose(macro_f1, selected_macro_f1, atol=1e-12, rtol=0.0)
                and regression["rmse"] < selected_macro_f1_tiebreak_rmse
            )
        ):
            selected_macro_f1 = macro_f1
            selected_macro_f1_tiebreak_rmse = regression["rmse"]
            selected_macro_f1_model = model

    if selected_rmse_model is None or selected_macro_f1_model is None:
        raise RuntimeError("Both Ridge selection variants must produce a model.")

    rolling_output = output_root / rolling
    rolling_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search_rows).to_csv(rolling_output / "alpha_search.csv", index=False)
    common_summary = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "rolling": rolling,
        "seed": int(seed),
        "input_tensor_shape": list(expected_shape),
        "flattened_feature_count": int(x_train.shape[1]),
        "target": "raw_future_return",
        "solver": "numpy_svd_closed_form_no_explicit_inverse",
        "refit_on_train_plus_validation": False,
        "input_scaling": "existing_rolling_train_fitted_chart_scaler; no additional refit",
    }
    variants = {
        "rmse_selected": save_selected_variant(
            selected_rmse_model,
            "rmse_selected",
            "validation_rmse_min",
            selected_rmse,
            rolling_output,
            rolling,
            seed,
            x_val,
            val_meta,
            x_test,
            test_meta,
            fee,
            slippage,
        ),
        "macro_f1_selected": save_selected_variant(
            selected_macro_f1_model,
            "macro_f1_selected",
            "validation_macro_f1_max",
            selected_macro_f1,
            rolling_output,
            rolling,
            seed,
            x_val,
            val_meta,
            x_test,
            test_meta,
            fee,
            slippage,
        ),
    }
    summary = {**common_summary, "variants": variants}

    (rolling_output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{rolling}] RMSE-selected alpha={selected_rmse_model.alpha:g}; "
        f"Macro-F1-selected alpha={selected_macro_f1_model.alpha:g}",
        flush=True,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Ridge-Flat benchmark baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BENCHMARK_ROOT / "data" / "dataset" / "rolling_threshold_0012",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_ROOT / "outputs" / "ridge_flat",
    )
    parser.add_argument("--rollings", nargs="+", default=["rolling_1", "rolling_2", "rolling_3"])
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alphas = tuple(sorted(set(float(alpha) for alpha in args.alphas)))
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("All Ridge alpha values must be finite and positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for rolling in args.rollings:
        rolling_dir = args.data_dir / rolling
        if not rolling_dir.is_dir():
            raise FileNotFoundError(f"Rolling directory not found: {rolling_dir}")
        summaries.append(
            train_one_rolling(
                rolling_dir,
                args.output_dir,
                alphas,
                args.seed,
                args.fee,
                args.slippage,
            )
        )

    run_index = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "alpha_candidates": list(alphas),
        "rollings": summaries,
    }
    (args.output_dir / "run_index.json").write_text(
        json.dumps(run_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved Ridge-Flat run index: {args.output_dir / 'run_index.json'}")


if __name__ == "__main__":
    main()
