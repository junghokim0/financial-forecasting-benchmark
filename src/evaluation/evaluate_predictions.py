"""CLI and library entry point for base benchmark evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .backtest import BacktestConfig, non_overlapping_backtest
    from .classification import classification_metrics
    from .prediction_schema import validate_prediction_frame
    from .regression import regression_metrics
except ImportError:  # Direct execution: python evaluate_predictions.py ...
    from backtest import BacktestConfig, non_overlapping_backtest
    from classification import classification_metrics
    from prediction_schema import validate_prediction_frame
    from regression import regression_metrics


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    fee: float = 0.001,
    slippage: float = 0.001,
) -> tuple[dict, pd.DataFrame]:
    """Validate a single model/rolling/split table and evaluate its base signal."""
    validated = validate_prediction_frame(frame)
    identity_columns = ["model", "model_version", "rolling", "split", "seed"]
    identities = validated[identity_columns].drop_duplicates()
    if len(identities) != 1:
        raise ValueError(
            "One prediction file must contain exactly one model/version/rolling/split/seed."
        )

    classification = classification_metrics(
        validated["y_true"].to_numpy(), validated["y_pred"].to_numpy()
    )
    backtest, trades = non_overlapping_backtest(
        validated,
        BacktestConfig(fee=fee, slippage=slippage),
    )
    identity = identities.iloc[0].to_dict()
    identity["seed"] = int(identity["seed"])
    result = {
        "schema_version": "1.0",
        "evaluation_variant": "base_prediction",
        "identity": identity,
        "num_rows": int(len(validated)),
        "cost": {
            "fee": float(fee),
            "slippage": float(slippage),
            "total_per_selected_trade": float(fee + slippage),
            "interpretation": "identical_to_existing_cryptova_code",
        },
        "classification": classification,
        "backtest": backtest,
    }

    if "predicted_return" in validated.columns:
        predicted = validated["predicted_return"]
        if predicted.notna().any() and not predicted.notna().all():
            raise ValueError("predicted_return must be complete when regression metrics are used.")
        if predicted.notna().all():
            result["regression"] = regression_metrics(
                validated["raw_future_return"].to_numpy(),
                predicted.to_numpy(dtype=float),
            )

    return result, trades


def evaluate_prediction_csv(
    input_path: Path,
    output_dir: Path,
    fee: float = 0.001,
    slippage: float = 0.001,
) -> tuple[Path, Path]:
    frame = pd.read_csv(input_path)
    result, trades = evaluate_prediction_frame(frame, fee=fee, slippage=slippage)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    trades_path = output_dir / "trades.csv"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    return metrics_path, trades_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one canonical prediction CSV using Cryptova's base protocol."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path, trades_path = evaluate_prediction_csv(
        args.input_csv,
        args.output_dir,
        fee=args.fee,
        slippage=args.slippage,
    )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved trades: {trades_path}")


if __name__ == "__main__":
    main()
