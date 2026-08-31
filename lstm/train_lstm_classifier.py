"""Train and evaluate the direct SHORT/HOLD/LONG LSTM classifier."""

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

from classification import classification_metrics  # noqa: E402
from evaluate_predictions import evaluate_prediction_csv  # noqa: E402
from prediction_schema import SCHEMA_VERSION  # noqa: E402
from lstm_classifier import (  # noqa: E402
    ManyToOneLSTMClassifier,
    count_trainable_parameters,
)
from lstm_model import LSTMConfig  # noqa: E402


MODEL_NAME = "lstm_many_to_one_classifier"
MODEL_VERSION = "lstm_many_to_one_classifier_v1"
EXPECTED_WINDOW_SHAPE = (72, 12)
CLASS_NAMES = ("SHORT", "HOLD", "LONG")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    label_smoothing: float = 0.03
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
    labels = np.array(metadata["label_id"], dtype=np.int64, copy=True)
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError(f"{metadata_path} contains labels outside 0, 1, 2.")
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
    labels = np.array(metadata["label_id"], dtype=np.int64, copy=True)
    dataset = TensorDataset(torch.from_numpy(chart), torch.from_numpy(labels))
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
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for chart, _ in loader:
        logits = model(chart.to(device, non_blocking=True))
        chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probabilities = np.concatenate(chunks).astype(np.float64, copy=False)
    if probabilities.shape[1] != 3:
        raise ValueError(f"Expected three probability columns, got {probabilities.shape}")
    return probabilities


@torch.inference_mode()
def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    for chart, labels in loader:
        chart = chart.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss = loss_function(model(chart), labels)
        batch_count = chart.shape[0]
        loss_sum += float(loss) * batch_count
        sample_count += batch_count
    return loss_sum / sample_count


def build_prediction_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    rolling: str,
    split: str,
    seed: int,
) -> pd.DataFrame:
    predicted_class = probabilities.argmax(axis=1).astype(np.int64)
    return pd.DataFrame(
        {
            "schema_version": SCHEMA_VERSION,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "rolling": rolling,
            "split": "validation" if split == "val" else split,
            "seed": int(seed),
            "sample_time": metadata["sample_time"],
            "target_time": metadata["target_time"],
            "y_true": metadata["label_id"].astype(np.int64),
            "raw_future_return": metadata["raw_future_return"].astype(np.float64),
            "y_pred": predicted_class,
            "prob_short": probabilities[:, 0],
            "prob_hold": probabilities[:, 1],
            "prob_long": probabilities[:, 2],
            "confidence": probabilities.max(axis=1),
        }
    )


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_one_rolling(
    rolling_dir: Path,
    output_root: Path,
    model_config: LSTMConfig,
    training_config: TrainingConfig,
    device: torch.device,
    fee: float,
    slippage: float,
) -> dict:
    rolling = rolling_dir.name
    set_reproducible_seed(training_config.seed)
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        arrays[split], metadata[split] = load_split(rolling_dir, split)

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

    model = ManyToOneLSTMClassifier(model_config).to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    parameter_count = count_trainable_parameters(model)
    best_checkpoint = None
    best_macro_f1 = float("-inf")
    best_tiebreak_loss = float("inf")
    last_improvement = 0
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        sample_count = 0
        for chart, labels in loaders["train"]:
            chart = chart.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(chart)
            loss = loss_function(logits, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            batch_count = chart.shape[0]
            train_loss_sum += float(loss.detach()) * batch_count
            sample_count += batch_count

        val_loss = validation_loss(model, loaders["val"], device, loss_function)
        val_probabilities = predict_probabilities(model, loaders["val"], device)
        val_prediction = val_probabilities.argmax(axis=1)
        classification = classification_metrics(
            metadata["val"]["label_id"].to_numpy(dtype=np.int64), val_prediction
        )
        val_macro_f1 = float(classification["macro_f1"])
        row = {
            "epoch": epoch,
            "train_cross_entropy": train_loss_sum / sample_count,
            "val_cross_entropy": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_balanced_accuracy": float(classification["balanced_accuracy"]),
            "val_accuracy": float(classification["accuracy"]),
            "pred_short_count": int((val_prediction == 0).sum()),
            "pred_hold_count": int((val_prediction == 1).sum()),
            "pred_long_count": int((val_prediction == 2).sum()),
        }
        history.append(row)
        print(
            f"[{rolling}] epoch={epoch:02d} train_ce={row['train_cross_entropy']:.6f} "
            f"val_ce={val_loss:.6f} val_macro_f1={val_macro_f1:.6f}",
            flush=True,
        )

        improved = val_macro_f1 > best_macro_f1 or (
            np.isclose(val_macro_f1, best_macro_f1, atol=1e-12, rtol=0.0)
            and val_loss < best_tiebreak_loss
        )
        if improved:
            best_macro_f1 = val_macro_f1
            best_tiebreak_loss = val_loss
            last_improvement = epoch
            best_checkpoint = {
                "state_dict": clone_state_dict(model),
                "epoch": epoch,
                "validation_macro_f1": val_macro_f1,
                "validation_cross_entropy": val_loss,
            }
        if epoch - last_improvement >= training_config.patience:
            print(f"[{rolling}] early stopping at epoch {epoch}", flush=True)
            break

    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")

    elapsed_seconds = time.perf_counter() - started
    output_dir = output_root / rolling
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    model.load_state_dict(best_checkpoint["state_dict"])
    torch.save(
        {
            "model_state_dict": best_checkpoint["state_dict"],
            "model_config": model_config.to_dict(),
            "training_config": asdict(training_config),
            "epoch": best_checkpoint["epoch"],
            "selection_metric": "validation_macro_f1_max",
            "selection_value": best_checkpoint["validation_macro_f1"],
        },
        output_dir / "model.pt",
    )

    summary = {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "rolling": rolling,
        "seed": training_config.seed,
        "input_tensor_shape": list(EXPECTED_WINDOW_SHAPE),
        "target": "label_id",
        "label_definition": {"0": "SHORT", "1": "HOLD", "2": "LONG"},
        "architecture": model_config.to_dict(),
        "num_classes": 3,
        "training": asdict(training_config),
        "optimizer": "AdamW",
        "loss": "CrossEntropyLoss",
        "class_weights": None,
        "selection_metric": "validation_macro_f1_max",
        "selected_epoch": best_checkpoint["epoch"],
        "validation_macro_f1": best_checkpoint["validation_macro_f1"],
        "validation_cross_entropy": best_checkpoint["validation_cross_entropy"],
        "device": str(device),
        "trainable_parameters": parameter_count,
        "epochs_completed": len(history),
        "training_seconds": elapsed_seconds,
        "refit_on_train_plus_validation": False,
        "input_scaling": "existing_rolling_train_fitted_chart_scaler; no additional refit",
        "confidence_filter_applied": False,
        "risk_filter_applied": False,
    }
    for split in ("val", "test"):
        probabilities = predict_probabilities(model, loaders[split], device)
        frame = build_prediction_frame(
            metadata[split], probabilities, rolling, split, training_config.seed
        )
        csv_path = output_dir / f"predictions_{split}.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        metrics_path, _ = evaluate_prediction_csv(
            csv_path,
            output_dir / f"evaluation_{split}",
            fee=fee,
            slippage=slippage,
        )
        summary[f"{split}_prediction_csv"] = str(csv_path)
        summary[f"{split}_metrics_json"] = str(metrics_path)

    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def rebuild_run_index(output_dir: Path, device: torch.device) -> Path:
    summaries = []
    for summary_path in sorted(output_dir.glob("rolling_*/run_summary.json")):
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not summaries:
        raise FileNotFoundError(f"No completed rolling summaries found under {output_dir}")
    index_path = output_dir / "run_index.json"
    index_path.write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "device": str(device),
                "rollings": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return index_path


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the direct LSTM classifier benchmark.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=BENCHMARK_ROOT / "data" / "dataset" / "rolling_threshold_0012",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BENCHMARK_ROOT / "outputs" / "lstm_classifier"
    )
    parser.add_argument("--rollings", nargs="+", default=["rolling_1", "rolling_2", "rolling_3"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--index-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
    )
    if training_config.batch_size <= 0 or training_config.max_epochs <= 0:
        raise ValueError("batch_size and max_epochs must be positive.")
    if training_config.patience <= 0 or training_config.gradient_clip <= 0:
        raise ValueError("patience and gradient_clip must be positive.")
    if not 0.0 <= training_config.label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")

    model_config = LSTMConfig()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        print(f"Rebuilt LSTM classifier run index: {rebuild_run_index(args.output_dir, device)}")
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
            args.fee,
            args.slippage,
        )
    print(f"Saved LSTM classifier run index: {rebuild_run_index(args.output_dir, device)}")


if __name__ == "__main__":
    main()
