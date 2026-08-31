"""Train and evaluate the official-core TimesNet return regressor."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
EVALUATION_DIR = BENCHMARK_ROOT / "src" / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from regression import regression_metrics  # noqa: E402
from timesnet_model import (  # noqa: E402
    TimesNetConfig,
    TimesNetRegressor,
    count_trainable_parameters,
)


MODEL_NAME = "timesnet_official_core_regressor"
MODEL_VERSION = "timesnet_official_core_direct_target_v1"
EXPECTED_WINDOW_SHAPE = (72, 12)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 42


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(rolling_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    tensor_path = rolling_dir / f"X_chart_{split}.npy"
    metadata_path = rolling_dir / f"sample_meta_{split}.csv"
    tensor = np.load(tensor_path)
    metadata = pd.read_csv(metadata_path)
    if len(tensor) != len(metadata):
        raise ValueError(
            f"{rolling_dir.name}/{split}: tensor rows {len(tensor)} != metadata rows {len(metadata)}"
        )
    if tuple(tensor.shape[1:]) != EXPECTED_WINDOW_SHAPE:
        raise ValueError(
            f"{rolling_dir.name}/{split}: expected {EXPECTED_WINDOW_SHAPE}, got {tensor.shape[1:]}"
        )
    required = {"sample_time", "target_time", "raw_future_return", "label_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"{metadata_path} is missing columns: {missing}")
    if not np.isfinite(tensor).all():
        raise ValueError(f"{tensor_path} contains non-finite values.")
    if not np.isfinite(metadata["raw_future_return"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{metadata_path} contains non-finite raw_future_return values.")
    return np.asarray(tensor, dtype=np.float32), metadata


def make_loader(
    chart: np.ndarray,
    metadata: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    target = metadata["raw_future_return"].to_numpy(dtype=np.float32)
    dataset = TensorDataset(torch.from_numpy(chart), torch.from_numpy(target))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
    )


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for chart, _ in loader:
        prediction = model(chart.to(device, non_blocking=True))
        chunks.append(prediction.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64, copy=False)


def build_prediction_frame(
    metadata: pd.DataFrame,
    predicted_return: np.ndarray,
    rolling: str,
    split: str,
    seed: int,
    model_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "schema_version": "1.0",
            "model": model_name,
            "model_version": MODEL_VERSION,
            "rolling": rolling,
            "split": "validation" if split == "val" else split,
            "seed": int(seed),
            "sample_time": metadata["sample_time"],
            "target_time": metadata["target_time"],
            "raw_future_return": metadata["raw_future_return"].astype(np.float64),
            "predicted_return": np.asarray(predicted_return, dtype=np.float64),
        }
    )


def validation_metrics(metadata: pd.DataFrame, prediction: np.ndarray) -> dict:
    return regression_metrics(
        metadata["raw_future_return"].to_numpy(dtype=np.float64), prediction
    )


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_variant(
    variant_name: str,
    checkpoint: dict,
    model_config: TimesNetConfig,
    training_config: TrainingConfig,
    rolling_output: Path,
    rolling: str,
    loaders: dict[str, DataLoader],
    metadata: dict[str, pd.DataFrame],
    device: torch.device,
) -> dict:
    variant_output = rolling_output / variant_name
    variant_output.mkdir(parents=True, exist_ok=True)
    model = TimesNetRegressor(model_config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    schema_model_name = f"{MODEL_NAME}_{variant_name}"

    torch.save(
        {
            "model_state_dict": checkpoint["state_dict"],
            "model_config": model_config.to_dict(),
            "training_config": asdict(training_config),
            "epoch": checkpoint["epoch"],
            "selection_metric": checkpoint["selection_metric"],
            "selection_value": checkpoint["selection_value"],
        },
        variant_output / "model.pt",
    )

    summary = {
        "model": schema_model_name,
        "model_version": MODEL_VERSION,
        "rolling": rolling,
        "seed": training_config.seed,
        "selection_metric": checkpoint["selection_metric"],
        "selection_value": checkpoint["selection_value"],
        "selected_epoch": checkpoint["epoch"],
        "refit_on_train_plus_validation": False,
        "input_scaling": "existing_rolling_train_fitted_chart_scaler; no additional refit",
    }
    for split in ("val", "test"):
        predicted_return = predict(model, loaders[split], device)
        frame = build_prediction_frame(
            metadata[split], predicted_return, rolling, split, training_config.seed, schema_model_name
        )
        csv_path = variant_output / f"predictions_{split}.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        metrics_dir = variant_output / f"evaluation_{split}"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "metrics.json"
        metrics = {
            "schema_version": "1.0",
            "evaluation_variant": "regression_only",
            "identity": {
                "model": schema_model_name,
                "model_version": MODEL_VERSION,
                "rolling": rolling,
                "split": "validation" if split == "val" else split,
                "seed": training_config.seed,
            },
            "num_rows": int(len(frame)),
            "regression": regression_metrics(
                frame["raw_future_return"].to_numpy(dtype=np.float64),
                frame["predicted_return"].to_numpy(dtype=np.float64),
            ),
        }
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary[f"{split}_prediction_csv"] = str(csv_path)
        summary[f"{split}_metrics_json"] = str(metrics_path)

    (variant_output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def train_variant(
    variant_name: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, pd.DataFrame],
    model_config: TimesNetConfig,
    training_config: TrainingConfig,
    device: torch.device,
) -> tuple[dict, list[dict], float, int, dict[str, DataLoader]]:
    """Train one independently early-stopped validation selection track."""
    if variant_name != "rmse_selected":
        raise ValueError(f"Unknown variant: {variant_name}")

    set_reproducible_seed(training_config.seed)
    pin_memory = device.type == "cuda"
    loaders = {
        split: make_loader(
            arrays[split],
            metadata[split],
            training_config.batch_size,
            shuffle=split == "train",
            seed=training_config.seed,
            pin_memory=pin_memory,
        )
        for split in ("train", "val", "test")
    }
    model = TimesNetRegressor(model_config).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    parameter_count = count_trainable_parameters(model)
    history: list[dict] = []
    best_checkpoint = None
    best_primary = float("inf")
    last_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for chart, target in loaders["train"]:
            chart = chart.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted_return = model(chart)
            loss = loss_function(predicted_return, target)
            loss.backward()
            clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            batch_count = chart.shape[0]
            loss_sum += float(loss.detach()) * batch_count
            sample_count += batch_count

        val_prediction = predict(model, loaders["val"], device)
        regression = validation_metrics(metadata["val"], val_prediction)
        val_rmse = float(regression["rmse"])
        row = {
            "epoch": epoch,
            "train_mse": loss_sum / sample_count,
            "val_rmse": val_rmse,
            "val_mae": float(regression["mae"]),
        }
        history.append(row)
        print(
            f"[{variant_name}] epoch={epoch:02d} train_mse={row['train_mse']:.8f} "
            f"val_rmse={val_rmse:.8f}",
            flush=True,
        )

        improved = val_rmse < best_primary
        selection_metric = "validation_rmse_min"
        selection_value = val_rmse

        if improved:
            best_primary = selection_value
            last_improvement = epoch
            best_checkpoint = {
                "state_dict": clone_state_dict(model),
                "epoch": epoch,
                "selection_metric": selection_metric,
                "selection_value": selection_value,
                "validation_rmse": val_rmse,
            }

        if epoch - last_improvement >= training_config.patience:
            print(f"[{variant_name}] early stopping at epoch {epoch}", flush=True)
            break

    if best_checkpoint is None:
        raise RuntimeError(f"{variant_name} did not produce a checkpoint.")
    return (
        best_checkpoint,
        history,
        time.perf_counter() - started,
        parameter_count,
        loaders,
    )


def train_one_rolling(
    rolling_dir: Path,
    output_root: Path,
    model_config: TimesNetConfig,
    training_config: TrainingConfig,
    device: torch.device,
) -> dict:
    rolling = rolling_dir.name
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        arrays[split], metadata[split] = load_split(rolling_dir, split)

    rolling_output = output_root / rolling
    rolling_output.mkdir(parents=True, exist_ok=True)
    variants = {}
    training_runs = {}
    parameter_count = None
    for variant_name in ("rmse_selected",):
        print(f"[{rolling}] starting independent {variant_name} run", flush=True)
        checkpoint, history, elapsed, current_parameter_count, loaders = train_variant(
            variant_name,
            arrays,
            metadata,
            model_config,
            training_config,
            device,
        )
        parameter_count = current_parameter_count
        variant_output = rolling_output / variant_name
        variant_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(variant_output / "training_history.csv", index=False)
        variants[variant_name] = save_variant(
            variant_name,
            checkpoint,
            model_config,
            training_config,
            rolling_output,
            rolling,
            loaders,
            metadata,
            device,
        )
        training_runs[variant_name] = {
            "epochs_completed": len(history),
            "training_seconds": elapsed,
            "early_stopping_metric": checkpoint["selection_metric"],
        }

    summary = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "rolling": rolling,
        "input_tensor_shape": list(EXPECTED_WINDOW_SHAPE),
        "target": "raw_future_return",
        "architecture": model_config.to_dict(),
        "training": asdict(training_config),
        "optimizer": "AdamW",
        "loss": "MSELoss",
        "device": str(device),
        "trainable_parameters": parameter_count,
        "training_runs": training_runs,
        "early_stopping_rule": "independent run per selection metric; patience applies only to that metric",
        "variants": variants,
    }
    (rolling_output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the official-core TimesNet regressor.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BENCHMARK_ROOT / "data" / "dataset" / "rolling_threshold_0012",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BENCHMARK_ROOT / "outputs" / "timesnet_regression"
    )
    parser.add_argument("--rollings", nargs="+", default=["rolling_1", "rolling_2", "rolling_3"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Rebuild run_index.json from completed rolling run_summary.json files without training.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def validate_training_config(config: TrainingConfig) -> None:
    if config.batch_size <= 0 or config.max_epochs <= 0 or config.patience <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive.")
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.gradient_clip <= 0:
        raise ValueError("Invalid optimizer or gradient clipping setting.")


def rebuild_run_index(output_dir: Path, device: torch.device) -> Path:
    """Index every completed rolling result so partial CLI runs cannot hide earlier runs."""
    summaries = []
    for summary_path in sorted(output_dir.glob("rolling_*/run_summary.json")):
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not summaries:
        raise FileNotFoundError(f"No completed rolling summaries found under {output_dir}")
    run_index = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "device": str(device),
        "rollings": summaries,
    }
    index_path = output_dir / "run_index.json"
    index_path.write_text(json.dumps(run_index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
    )
    validate_training_config(training_config)
    model_config = TimesNetConfig()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        index_path = rebuild_run_index(args.output_dir, device)
        print(f"Rebuilt TimesNet regression run index: {index_path}")
        return

    for rolling in args.rollings:
        rolling_dir = args.data_dir / rolling
        if not rolling_dir.is_dir():
            raise FileNotFoundError(f"Rolling directory not found: {rolling_dir}")
        train_one_rolling(
            rolling_dir,
            args.output_dir,
            model_config,
            training_config,
            device,
        )
    index_path = rebuild_run_index(args.output_dir, device)
    print(f"Saved TimesNet regression run index: {index_path}")


if __name__ == "__main__":
    main()
