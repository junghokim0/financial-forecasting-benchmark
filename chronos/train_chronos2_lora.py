"""Fine-tune and evaluate Chart-only Chronos-2 with LoRA on Colab GPU.

This script intentionally requires ``--execute-training`` and a CUDA device.
It is designed for Google Colab; importing it never downloads or trains a model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT_DEFAULT = SCRIPT_DIR.parent
EVALUATION_DIR_DEFAULT = BENCHMARK_ROOT_DEFAULT / "src" / "evaluation"
if str(EVALUATION_DIR_DEFAULT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR_DEFAULT))

from chronos2_data import (  # noqa: E402
    PREDICTION_HORIZON,
    WINDOW_SIZE,
    iter_chronos_input_batches,
    load_inference_windows,
    load_master,
    make_continuous_fit_input,
    rolling_dir,
)


MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "95a9710e2596287d08352589f42634fa5abdf0a7"
MODEL_DISPLAY_NAME = "Chronos-2 LoRA Fine-tuned"
MODEL_VERSION = "amazon_chronos2_lora_qforecast_v1"
SHORT_THRESHOLD = -0.012
LONG_THRESHOLD = 0.012
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT_DEFAULT)
    parser.add_argument("--rolling", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    parser.add_argument("--prediction-chunk-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--execute-training",
        action="store_true",
        help="Required safety flag. Without it no model is downloaded or trained.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def returns_to_classes(values: np.ndarray) -> np.ndarray:
    labels = np.full(values.shape, 1, dtype=np.int64)
    labels[values <= SHORT_THRESHOLD] = 0
    labels[values >= LONG_THRESHOLD] = 2
    return labels


def model_name() -> str:
    return "chronos2_lora_chart"


def build_prediction_frame(
    metadata: pd.DataFrame,
    predicted_return: np.ndarray,
    q10_return: np.ndarray,
    q90_return: np.ndarray,
    rolling: int,
    split: str,
    seed: int,
) -> pd.DataFrame:
    y_pred = returns_to_classes(predicted_return)
    return pd.DataFrame(
        {
            "schema_version": "1.0",
            "model": model_name(),
            "model_version": MODEL_VERSION,
            "rolling": f"rolling_{rolling}",
            "split": "validation" if split == "val" else split,
            "seed": seed,
            "sample_time": metadata["sample_time"],
            "target_time": metadata["target_time"],
            "y_true": metadata["label_id"].astype(np.int64),
            "raw_future_return": metadata["raw_future_return"].astype(np.float64),
            "y_pred": y_pred,
            "predicted_return": predicted_return,
            "predicted_return_q10": q10_return,
            "predicted_return_q90": q90_return,
        }
    )


def predict_returns(
    pipeline,
    close_windows: np.ndarray,
    features: np.ndarray,
    feature_names: tuple[str, ...],
    prediction_batch_size: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median_values: list[float] = []
    q10_values: list[float] = []
    q90_values: list[float] = []
    for start, end, inputs in iter_chronos_input_batches(
        close_windows, features, feature_names, chunk_size
    ):
        quantiles, medians = pipeline.predict_quantiles(
            inputs,
            prediction_length=PREDICTION_HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=prediction_batch_size,
            context_length=WINDOW_SIZE,
            cross_learning=False,
        )
        for offset, (quantile_tensor, median_tensor) in enumerate(zip(quantiles, medians)):
            quantile = quantile_tensor.detach().float().cpu().numpy()
            median = median_tensor.detach().float().cpu().numpy()
            if quantile.shape != (1, PREDICTION_HORIZON, 3):
                raise ValueError(f"Unexpected quantile output shape: {quantile.shape}")
            if median.shape != (1, PREDICTION_HORIZON):
                raise ValueError(f"Unexpected median output shape: {median.shape}")
            current_close = float(close_windows[start + offset, -1])
            median_values.append(float(median[0, -1] / current_close - 1.0))
            q10_values.append(float(quantile[0, -1, 0] / current_close - 1.0))
            q90_values.append(float(quantile[0, -1, 2] / current_close - 1.0))
        if len(medians) != end - start:
            raise ValueError("Chronos returned a different number of forecasts than requested.")
    arrays = tuple(np.asarray(values, dtype=np.float64) for values in (median_values, q10_values, q90_values))
    if any(len(values) != len(close_windows) for values in arrays):
        raise ValueError("Incomplete Chronos prediction output.")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Chronos produced NaN or infinity.")
    return arrays


def evaluate_and_save(frame: pd.DataFrame, destination: Path) -> dict:
    evaluation_dir = destination / f"evaluation_{frame['split'].iloc[0]}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = destination.parents[2]
    evaluation_path = benchmark_root / "src" / "evaluation"
    if str(evaluation_path) not in sys.path:
        sys.path.insert(0, str(evaluation_path))
    from evaluate_predictions import evaluate_prediction_frame

    metrics, trades = evaluate_prediction_frame(frame, fee=0.001, slippage=0.001)
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trades.to_csv(evaluation_dir / "trades.csv", index=False, encoding="utf-8-sig")
    return metrics


def load_base_pipeline():
    from chronos import Chronos2Pipeline

    major = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if major >= 8 else torch.float32
    return Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="cuda",
        torch_dtype=dtype,
    )


def train_one_rolling(args: argparse.Namespace, rolling: int, master: pd.DataFrame) -> dict:
    root = args.benchmark_root.resolve()
    roll_path = rolling_dir(root, rolling)
    output_root = root / "outputs" / "chronos2_lora_chart"
    output_dir = output_root / f"rolling_{rolling}"
    trainer_dir = output_dir / "trainer"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_inputs = make_continuous_fit_input(master, roll_path, "train")
    validation_inputs = make_continuous_fit_input(master, roll_path, "val")

    set_seed(args.seed)
    base_pipeline = load_base_pipeline()
    started = time.perf_counter()
    finetuned_pipeline = base_pipeline.fit(
        train_inputs,
        prediction_length=PREDICTION_HORIZON,
        validation_inputs=validation_inputs,
        finetune_mode="lora",
        lora_config=None,
        context_length=WINDOW_SIZE,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        batch_size=args.train_batch_size,
        output_dir=trainer_dir,
        min_past=WINDOW_SIZE,
        finetuned_ckpt_name="finetuned-ckpt",
        logging_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        seed=args.seed,
        data_seed=args.seed,
    )
    elapsed = time.perf_counter() - started

    split_metrics: dict[str, dict] = {}
    for split in ("val", "test"):
        close_windows, features, feature_names, metadata = load_inference_windows(
            root, rolling, split, master
        )
        predicted, q10, q90 = predict_returns(
            finetuned_pipeline,
            close_windows,
            features,
            feature_names,
            args.prediction_batch_size,
            args.prediction_chunk_size,
        )
        frame = build_prediction_frame(
            metadata, predicted, q10, q90, rolling, split, args.seed
        )
        frame.to_csv(output_dir / f"predictions_{split}.csv", index=False, encoding="utf-8-sig")
        split_metrics[split] = evaluate_and_save(frame, output_dir)

    trainable = sum(
        parameter.numel()
        for parameter in finetuned_pipeline.model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in finetuned_pipeline.model.parameters())
    summary = {
        "display_name": MODEL_DISPLAY_NAME,
        "model": model_name(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_version": MODEL_VERSION,
        "finetune_mode": "lora",
        "lora_configuration": "Chronos-2 official default: r=8, alpha=16, q/k/v/o and output head",
        "rolling": f"rolling_{rolling}",
        "input_mode": "chart_12",
        "target": "raw BTC close path; q=0.5 at t+24 converted to return",
        "context_length": WINDOW_SIZE,
        "prediction_length": PREDICTION_HORIZON,
        "selection_split": "validation",
        "selection_metric": "official Chronos-2 validation quantile loss (eval_loss)",
        "short_threshold": SHORT_THRESHOLD,
        "long_threshold": LONG_THRESHOLD,
        "num_steps": args.num_steps,
        "train_batch_size_in_variates": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "trainable_parameters": trainable,
        "total_parameters_including_adapter": total,
        "elapsed_seconds": elapsed,
        "checkpoint": str(trainer_dir / "finetuned-ckpt"),
        "validation_metrics": split_metrics["val"],
        "test_metrics": split_metrics["test"],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del finetuned_pipeline, base_pipeline
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    if not args.execute_training:
        raise SystemExit(
            "Safety stop: add --execute-training to explicitly download and fine-tune Chronos-2."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required. Run this script in a GPU-enabled Colab runtime.")
    if args.num_steps <= 0 or args.train_batch_size <= 0:
        raise SystemExit("num_steps and train_batch_size must be positive.")
    root = args.benchmark_root.resolve()
    evaluation_path = root / "src" / "evaluation"
    if str(evaluation_path) not in sys.path:
        sys.path.insert(0, str(evaluation_path))
    master = load_master(root)
    summaries = [train_one_rolling(args, rolling, master) for rolling in args.rolling]
    index_path = root / "outputs" / "chronos2_lora_chart" / "run_index.json"
    index_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_index": str(index_path), "completed": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
