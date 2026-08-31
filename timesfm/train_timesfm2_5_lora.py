"""Fine-tune and evaluate close-only TimesFM 2.5 with LoRA on Colab GPU.

The script requires both CUDA and ``--execute-training``. Importing it never
downloads a checkpoint or starts training.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from timesfm_data import (
    CANONICAL_WINDOW_SIZE,
    MODEL_CONTEXT_LENGTH,
    PREDICTION_HORIZON,
    TimesFMDataset,
    load_master,
    load_split,
)


MODEL_ID = "google/timesfm-2.5-200m-transformers"
MODEL_DISPLAY_NAME = "TimesFM 2.5 LoRA Fine-tuned"
MODEL_VERSION = "timesfm2_5_lora_close_v1"
SHORT_THRESHOLD = -0.012
LONG_THRESHOLD = 0.012
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--rolling", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--prediction-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--execute-training", action="store_true")
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


def resolve_revision() -> str:
    from huggingface_hub import model_info

    return model_info(MODEL_ID).sha


def environment_metadata() -> dict:
    packages = {}
    for name in ("transformers", "peft", "accelerate", "huggingface-hub", "numpy", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "packages": packages,
    }


def load_base_model(revision: str):
    from transformers import TimesFm2_5ModelForPrediction

    major = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if major >= 8 else torch.float32
    return TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID,
        revision=revision,
        torch_dtype=dtype,
        device_map="cuda",
    )


def make_loader(contexts, futures, batch_size, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TimesFMDataset(contexts, futures),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator if shuffle else None,
        pin_memory=True,
    )


def forward_loss(model, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    # Iterating over a [B, 64] tensor produces the official Sequence[Tensor] input.
    outputs = model(
        past_values=tuple(context),
        future_values=future,
        forecast_context_len=MODEL_CONTEXT_LENGTH,
        truncate_negative=False,
    )
    return outputs.loss


@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    model.eval()
    total = 0.0
    rows = 0
    for context, future in loader:
        context = context.to(device, non_blocking=True)
        future = future.to(device, non_blocking=True)
        loss = forward_loss(model, context, future)
        total += float(loss.item()) * len(context)
        rows += len(context)
    return total / max(rows, 1)


def train_adapter(model, train_loader, val_loader, args, adapter_dir: Path) -> dict:
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules="all-linear",
            lora_dropout=args.lora_dropout,
            bias="none",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = int(np.ceil(len(train_loader) / args.gradient_accumulation))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * updates_per_epoch)
    )
    device = next(model.parameters()).device
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_rows = 0
        for step, (context, future) in enumerate(train_loader, start=1):
            context = context.to(device, non_blocking=True)
            future = future.to(device, non_blocking=True)
            loss = forward_loss(model, context, future)
            (loss / args.gradient_accumulation).backward()
            train_total += float(loss.item()) * len(context)
            train_rows += len(context)
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        train_loss = train_total / max(train_rows, 1)
        val_loss = validation_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(adapter_dir)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    return {
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "history": history,
        "trainable_parameters": trainable,
        "total_parameters_including_adapter": total,
    }


def quantile_index(config, quantile: float) -> int:
    values = [float(q) for q in config.quantiles]
    matches = [index for index, value in enumerate(values) if np.isclose(value, quantile)]
    if len(matches) != 1:
        raise ValueError(f"Quantile {quantile} is not uniquely present in {values}")
    return matches[0] + 1  # channel 0 is the additional point-forecast channel


@torch.no_grad()
def predict_returns(model, contexts: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    median_values, q10_values, q90_values = [], [], []
    q10_index = quantile_index(model.config, 0.1)
    q90_index = quantile_index(model.config, 0.9)
    model.eval()
    for start in range(0, len(contexts), batch_size):
        batch_np = contexts[start : start + batch_size]
        batch = torch.from_numpy(batch_np).to(device)
        outputs = model(
            past_values=tuple(batch),
            forecast_context_len=MODEL_CONTEXT_LENGTH,
            truncate_negative=False,
        )
        median_close = outputs.mean_predictions[:, PREDICTION_HORIZON - 1].float().cpu().numpy()
        full = outputs.full_predictions[:, PREDICTION_HORIZON - 1].float().cpu().numpy()
        current_close = batch_np[:, -1].astype(np.float64)
        median_values.extend(median_close / current_close - 1.0)
        q10_values.extend(full[:, q10_index] / current_close - 1.0)
        q90_values.extend(full[:, q90_index] / current_close - 1.0)
    arrays = tuple(np.asarray(x, dtype=np.float64) for x in (median_values, q10_values, q90_values))
    if any(len(x) != len(contexts) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("TimesFM produced incomplete or non-finite predictions.")
    return arrays


def build_prediction_frame(metadata, predicted, q10, q90, rolling, split, seed):
    return pd.DataFrame(
        {
            "schema_version": "1.0",
            "model": "timesfm2_5_lora_close",
            "model_version": MODEL_VERSION,
            "rolling": f"rolling_{rolling}",
            "split": "validation" if split == "val" else split,
            "seed": seed,
            "sample_time": metadata["sample_time"],
            "target_time": metadata["target_time"],
            "y_true": metadata["label_id"].astype(np.int64),
            "raw_future_return": metadata["raw_future_return"].astype(np.float64),
            "y_pred": returns_to_classes(predicted),
            "predicted_return": predicted,
            "predicted_return_q10": q10,
            "predicted_return_q90": q90,
        }
    )


def evaluate_and_save(frame: pd.DataFrame, destination: Path, benchmark_root: Path) -> dict:
    evaluation_path = benchmark_root / "src" / "evaluation"
    if str(evaluation_path) not in sys.path:
        sys.path.insert(0, str(evaluation_path))
    from evaluate_predictions import evaluate_prediction_frame

    evaluation_dir = destination / f"evaluation_{frame['split'].iloc[0]}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics, trades = evaluate_prediction_frame(frame, fee=0.001, slippage=0.001)
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trades.to_csv(evaluation_dir / "trades.csv", index=False, encoding="utf-8-sig")
    return metrics


def train_one_rolling(args, rolling: int, master: pd.DataFrame, revision: str) -> dict:
    from peft import PeftModel

    root = args.benchmark_root.resolve()
    output_dir = root / "outputs" / "timesfm2_5_lora_close" / f"rolling_{rolling}"
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_context, train_future, _ = load_split(root, rolling, "train", master)
    val_context, val_future, val_metadata = load_split(root, rolling, "val", master)
    train_loader = make_loader(train_context, train_future, args.batch_size, True, args.seed)
    val_loader = make_loader(val_context, val_future, args.prediction_batch_size, False, args.seed)

    set_seed(args.seed)
    started = time.perf_counter()
    base_model = load_base_model(revision)
    training = train_adapter(base_model, train_loader, val_loader, args, adapter_dir)
    del base_model
    torch.cuda.empty_cache()

    best_base = load_base_model(revision)
    model = PeftModel.from_pretrained(best_base, adapter_dir)
    elapsed = time.perf_counter() - started
    split_metrics = {}
    cached = {"val": (val_context, val_metadata)}
    for split in ("val", "test"):
        if split in cached:
            contexts, metadata = cached[split]
        else:
            contexts, _, metadata = load_split(root, rolling, split, master)
        predicted, q10, q90 = predict_returns(model, contexts, args.prediction_batch_size)
        frame = build_prediction_frame(metadata, predicted, q10, q90, rolling, split, args.seed)
        frame.to_csv(output_dir / f"predictions_{split}.csv", index=False, encoding="utf-8-sig")
        split_metrics[split] = evaluate_and_save(frame, output_dir, root)

    summary = {
        "display_name": MODEL_DISPLAY_NAME,
        "model": "timesfm2_5_lora_close",
        "model_id": MODEL_ID,
        "resolved_revision": revision,
        "model_version": MODEL_VERSION,
        "rolling": f"rolling_{rolling}",
        "input_mode": "close_only",
        "canonical_available_context": CANONICAL_WINDOW_SIZE,
        "model_context_length": MODEL_CONTEXT_LENGTH,
        "context_policy": "latest_64_of_canonical_72_due_to_patch_length_32",
        "prediction_length": PREDICTION_HORIZON,
        "target": "raw BTC close path; TimesFM point forecast at t+24 converted to return",
        "selection_split": "validation",
        "selection_metric": "official TimesFM normalized MSE plus quantile loss",
        "finetune_mode": "lora",
        "full_finetuning": False,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": "all-linear",
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "short_threshold": SHORT_THRESHOLD,
        "long_threshold": LONG_THRESHOLD,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "adapter_path": str(adapter_dir),
        "environment": environment_metadata(),
        "validation_metrics": split_metrics["val"],
        "test_metrics": split_metrics["test"],
        **training,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model, best_base
    torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    if not args.execute_training:
        raise SystemExit("Safety stop: add --execute-training to download and fine-tune TimesFM 2.5.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required. Run this script in a GPU-enabled Colab runtime.")
    if min(args.epochs, args.patience, args.batch_size, args.gradient_accumulation) <= 0:
        raise SystemExit("Training counts must be positive.")
    if MODEL_CONTEXT_LENGTH % 32 != 0:
        raise SystemExit("TimesFM 2.5 context length must be divisible by patch length 32.")
    root = args.benchmark_root.resolve()
    master = load_master(root)
    revision = resolve_revision()
    summaries = [train_one_rolling(args, rolling, master, revision) for rolling in args.rolling]
    index_path = root / "outputs" / "timesfm2_5_lora_close" / "run_index.json"
    index_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_index": str(index_path), "completed": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
