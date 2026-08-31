"""Evaluate existing benchmark predictions by market regime without retraining.

Regimes use only information available at ``sample_time``:

* trend: trailing 72-hour return split by each rolling Train Q33/Q67;
* volatility: trailing ``std_24h`` split by each rolling Train median.

The script never uses ``future_return_24h`` to define a regime.  Backtest trades
are selected once on the full connected OOS sequence and then attributed to the
regime observed at entry, preserving the benchmark's 24-hour non-overlap rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SIGNAL_SHORT_THRESHOLD = -0.012
SIGNAL_LONG_THRESHOLD = 0.012

ROLLING_BOUNDS = {
    "rolling_1": {
        "train": ("2024-01-01T00:00:00Z", "2025-04-01T00:00:00Z"),
        "test": ("2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
    },
    "rolling_2": {
        "train": ("2024-04-01T00:00:00Z", "2025-07-01T00:00:00Z"),
        "test": ("2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    },
    "rolling_3": {
        "train": ("2024-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
        "test": ("2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"),
    },
}


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    track: str
    paths: tuple[str, ...]


MODEL_SPECS = (
    ModelSpec(
        "Ridge-Flat",
        "regression",
        tuple(
            f"outputs/ridge_flat/rolling_{i}/rmse_selected/predictions_test.csv"
            for i in (1, 2, 3)
        ),
    ),
    ModelSpec(
        "LSTM Regression",
        "regression",
        tuple(
            f"outputs/lstm_regression/rolling_{i}/rmse_selected/predictions_test.csv"
            for i in (1, 2, 3)
        ),
    ),
    ModelSpec(
        "TimesNet Regression",
        "regression",
        ("outputs/timesnet_regression/connected_oos/predictions_test.csv",),
    ),
    ModelSpec(
        "Chronos-2 LoRA",
        "regression",
        ("outputs/chronos2_lora_chart/connected_oos/predictions_test.csv",),
    ),
    ModelSpec(
        "TimesFM 2.5 LoRA",
        "regression",
        ("outputs/timesfm2_5_lora_close/connected_oos/predictions_test.csv",),
    ),
    ModelSpec(
        "Ridge-Flat",
        "classification",
        tuple(
            f"outputs/ridge_flat/rolling_{i}/macro_f1_selected/predictions_test.csv"
            for i in (1, 2, 3)
        ),
    ),
    ModelSpec(
        "LSTM Classifier",
        "classification",
        tuple(
            f"outputs/lstm_classifier/rolling_{i}/predictions_test.csv"
            for i in (1, 2, 3)
        ),
    ),
    ModelSpec(
        "TimesNet Classifier",
        "classification",
        ("outputs/timesnet_classifier/connected_oos/predictions_test.csv",),
    ),
    ModelSpec(
        "Chronos-2 LoRA",
        "classification",
        ("outputs/chronos2_lora_chart/connected_oos/predictions_test.csv",),
    ),
    ModelSpec(
        "TimesFM 2.5 LoRA",
        "classification",
        ("outputs/timesfm2_5_lora_close/connected_oos/predictions_test.csv",),
    ),
    ModelSpec(
        "Cryptova-Raw",
        "classification",
        ("outputs/cryptova_raw_re_evaluation/connected_oos/predictions_test.csv",),
    ),
    ModelSpec(
        "Cryptova-Base",
        "classification",
        ("outputs/cryptova_base_re_evaluation/connected_oos/predictions_test.csv",),
    ),
    ModelSpec(
        "Cryptova-Full",
        "classification",
        ("outputs/cryptova_full_re_evaluation/connected_oos/predictions_test.csv",),
    ),
)

REGIME_AXES = (
    ("trend", "trend_regime", ("UP", "DOWN", "SIDEWAYS")),
    ("volatility", "volatility_regime", ("HIGH", "LOW")),
    (
        "combined",
        "market_regime",
        (
            "UP_HIGH",
            "UP_LOW",
            "DOWN_HIGH",
            "DOWN_LOW",
            "SIDEWAYS_HIGH",
            "SIDEWAYS_LOW",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <benchmark-root>/outputs/regime_analysis",
    )
    return parser.parse_args()


def returns_to_classes(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=np.float64)
    return np.where(
        array <= SIGNAL_SHORT_THRESHOLD,
        0,
        np.where(array >= SIGNAL_LONG_THRESHOLD, 2, 1),
    )


def infer_rolling(sample_time: pd.Series) -> pd.Series:
    result = pd.Series(index=sample_time.index, dtype="object")
    for rolling, bounds in ROLLING_BOUNDS.items():
        start, end = (pd.Timestamp(value) for value in bounds["test"])
        mask = (sample_time >= start) & (sample_time < end)
        result.loc[mask] = rolling
    if result.isna().any():
        examples = sample_time[result.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Could not infer rolling for sample_time values: {examples}")
    return result


def load_master(root: Path) -> pd.DataFrame:
    path = root / "data" / "master" / "merged_with_future_return.csv"
    frame = pd.read_csv(path, usecols=["sample_time", "close", "std_24h"])
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], utc=True, errors="raise")
    if frame["sample_time"].duplicated().any():
        raise ValueError("Master data contains duplicate sample_time values.")
    frame = frame.sort_values("sample_time").reset_index(drop=True)
    frame["return_72h"] = frame["close"].pct_change(72, fill_method=None)
    return frame


def trend_thresholds(master: pd.DataFrame) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for rolling, bounds in ROLLING_BOUNDS.items():
        start, end = (pd.Timestamp(value) for value in bounds["train"])
        train = master.loc[
            (master["sample_time"] >= start) & (master["sample_time"] < end),
            "return_72h",
        ].dropna()
        if train.empty:
            raise ValueError(f"No Train rows available for {rolling} trend threshold.")
        q33 = float(train.quantile(1.0 / 3.0))
        q67 = float(train.quantile(2.0 / 3.0))
        thresholds[rolling] = {
            "q33": q33,
            "q67": q67,
        }
    return thresholds


def volatility_thresholds(master: pd.DataFrame) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for rolling, bounds in ROLLING_BOUNDS.items():
        start, end = (pd.Timestamp(value) for value in bounds["train"])
        train = master.loc[
            (master["sample_time"] >= start) & (master["sample_time"] < end), "std_24h"
        ]
        if train.empty:
            raise ValueError(f"No Train rows available for {rolling} volatility threshold.")
        thresholds[rolling] = float(train.median())
    return thresholds


def load_predictions(root: Path, spec: ModelSpec) -> pd.DataFrame:
    frames = []
    for relative in spec.paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction file for {spec.display_name}: {path}")
        frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], utc=True, errors="raise")
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True, errors="raise")
    frame = frame.sort_values("sample_time").reset_index(drop=True)
    if frame["sample_time"].duplicated().any():
        raise ValueError(f"{spec.display_name}/{spec.track} contains duplicate sample_time values.")
    frame["source_rolling"] = infer_rolling(frame["sample_time"])
    if "y_true" not in frame:
        frame["y_true"] = returns_to_classes(frame["raw_future_return"])
    if "y_pred" not in frame and "predicted_return" in frame:
        frame["y_pred"] = returns_to_classes(frame["predicted_return"])
    required = {"sample_time", "target_time", "raw_future_return", "y_true", "y_pred"}
    if spec.track == "regression":
        required.add("predicted_return")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{spec.display_name}/{spec.track} missing columns: {missing}")
    return frame


def attach_regimes(
    frame: pd.DataFrame,
    master: pd.DataFrame,
    trend_limits: dict[str, dict[str, float]],
    vol_thresholds: dict[str, float],
) -> pd.DataFrame:
    # Regime variables have one canonical source: the common master data.  Some
    # prediction artifacts also retain feature columns such as std_24h; remove
    # those copies to avoid suffixes and accidental model-specific provenance.
    frame = frame.drop(
        columns=["close", "return_72h", "std_24h"], errors="ignore"
    )
    merged = frame.merge(master, on="sample_time", how="left", validate="one_to_one")
    if merged[["return_72h", "std_24h"]].isna().any().any():
        raise ValueError("Some predictions could not be matched to master regime data.")
    down_threshold = merged["source_rolling"].map(
        {rolling: values["q33"] for rolling, values in trend_limits.items()}
    )
    up_threshold = merged["source_rolling"].map(
        {rolling: values["q67"] for rolling, values in trend_limits.items()}
    )
    if down_threshold.isna().any() or up_threshold.isna().any():
        raise ValueError("Missing rolling trend threshold.")
    merged["trend_down_threshold"] = down_threshold.astype(float)
    merged["trend_up_threshold"] = up_threshold.astype(float)
    merged["trend_regime"] = np.select(
        [
            (merged["return_72h"] <= merged["trend_down_threshold"])
            & (merged["return_72h"] < 0.0),
            (merged["return_72h"] >= merged["trend_up_threshold"])
            & (merged["return_72h"] > 0.0),
        ],
        ["DOWN", "UP"],
        default="SIDEWAYS",
    )
    threshold = merged["source_rolling"].map(vol_thresholds)
    if threshold.isna().any():
        raise ValueError("Missing rolling volatility threshold.")
    merged["volatility_threshold"] = threshold.astype(float)
    merged["volatility_regime"] = np.where(
        merged["std_24h"] > merged["volatility_threshold"], "HIGH", "LOW"
    )
    merged["market_regime"] = (
        merged["trend_regime"] + "_" + merged["volatility_regime"]
    )
    return merged


def max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1.0))


def trade_summary(trades: pd.DataFrame, regime_rows: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "trade_count": 0,
            "trade_ratio": 0.0,
            "cumulative_return": 0.0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "max_drawdown": 0.0,
        }
    values = trades["strategy_return"].to_numpy(dtype=np.float64)
    return {
        "trade_count": int(len(trades)),
        "trade_ratio": float(len(trades) / len(regime_rows)) if len(regime_rows) else 0.0,
        "cumulative_return": float(np.prod(1.0 + values) - 1.0),
        "win_rate": float(np.mean(values > 0.0)),
        "avg_trade_return": float(np.mean(values)),
        "max_drawdown": max_drawdown(values),
    }


def main() -> None:
    args = parse_args()
    root = args.benchmark_root.resolve()
    output = (args.output_dir or root / "outputs" / "regime_analysis").resolve()
    evaluation_dir = root / "src" / "evaluation"
    if str(evaluation_dir) not in sys.path:
        sys.path.insert(0, str(evaluation_dir))
    from backtest import non_overlapping_backtest
    from classification import classification_metrics
    from regression import regression_metrics

    master = load_master(root)
    trend_limits = trend_thresholds(master)
    vol_thresholds = volatility_thresholds(master)
    regression_rows: list[dict] = []
    classification_rows: list[dict] = []
    trade_rows: list[dict] = []
    count_reference: pd.DataFrame | None = None
    expected_times: pd.Series | None = None

    for spec in MODEL_SPECS:
        frame = attach_regimes(
            load_predictions(root, spec), master, trend_limits, vol_thresholds
        )
        if expected_times is None:
            expected_times = frame["sample_time"]
        elif not frame["sample_time"].reset_index(drop=True).equals(expected_times.reset_index(drop=True)):
            raise ValueError(f"Timestamp mismatch for {spec.display_name}/{spec.track}.")
        if count_reference is None:
            count_reference = frame[
                [
                    "sample_time",
                    "source_rolling",
                    "trend_regime",
                    "volatility_regime",
                    "market_regime",
                ]
            ].copy()

        for axis, column, order in REGIME_AXES:
            for regime in order:
                subset = frame.loc[frame[column] == regime].copy()
                if subset.empty:
                    raise ValueError(f"Empty {axis}/{regime} subset.")
                common = {
                    "regime_axis": axis,
                    "regime": regime,
                    "model": spec.display_name,
                    "samples": int(len(subset)),
                }
                if spec.track == "regression":
                    regression_rows.append(
                        {
                            **common,
                            **regression_metrics(
                                subset["raw_future_return"].to_numpy(),
                                subset["predicted_return"].to_numpy(),
                            ),
                        }
                    )
                else:
                    metrics = classification_metrics(
                        subset["y_true"].to_numpy(), subset["y_pred"].to_numpy()
                    )
                    report = metrics["classification_report"]
                    classification_rows.append(
                        {
                            **common,
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "short_precision": report["SHORT"]["precision"],
                            "short_recall": report["SHORT"]["recall"],
                            "hold_precision": report["HOLD"]["precision"],
                            "hold_recall": report["HOLD"]["recall"],
                            "long_precision": report["LONG"]["precision"],
                            "long_recall": report["LONG"]["recall"],
                            "short_support": int(report["SHORT"]["support"]),
                            "hold_support": int(report["HOLD"]["support"]),
                            "long_support": int(report["LONG"]["support"]),
                        }
                    )

        if spec.track == "classification":
            _, trades = non_overlapping_backtest(frame)
            trades["sample_time"] = pd.to_datetime(trades["sample_time"], utc=True)
            trade_regimes = frame[
                ["sample_time", "trend_regime", "volatility_regime", "market_regime"]
            ].drop_duplicates()
            trades = trades.merge(trade_regimes, on="sample_time", how="left", validate="one_to_one")
            for axis, column, order in REGIME_AXES:
                for regime in order:
                    regime_samples = frame.loc[frame[column] == regime]
                    regime_trades = trades.loc[trades[column] == regime]
                    trade_rows.append(
                        {
                            "regime_axis": axis,
                            "regime": regime,
                            "model": spec.display_name,
                            "samples": int(len(regime_samples)),
                            **trade_summary(regime_trades, regime_samples),
                        }
                    )

    if count_reference is None:
        raise RuntimeError("No model predictions were processed.")
    count_rows = []
    count_by_rolling_rows = []
    for axis, column, order in REGIME_AXES:
        for regime in order:
            subset = count_reference.loc[count_reference[column] == regime]
            count_rows.append(
                {
                    "regime_axis": axis,
                    "regime": regime,
                    "samples": int(len(subset)),
                    "ratio": float(len(subset) / len(count_reference)),
                }
            )
            for rolling in ROLLING_BOUNDS:
                rolling_rows = count_reference.loc[
                    count_reference["source_rolling"] == rolling
                ]
                rolling_subset = rolling_rows.loc[rolling_rows[column] == regime]
                count_by_rolling_rows.append(
                    {
                        "regime_axis": axis,
                        "regime": regime,
                        "rolling": rolling,
                        "samples": int(len(rolling_subset)),
                        "ratio": float(len(rolling_subset) / len(rolling_rows)),
                    }
                )

    regression = pd.DataFrame(regression_rows)
    classification = pd.DataFrame(classification_rows)
    trades = pd.DataFrame(trade_rows)
    counts = pd.DataFrame(count_rows)
    counts_by_rolling = pd.DataFrame(count_by_rolling_rows)

    winners = {
        "regression_lowest_rmse": regression.loc[
            regression.groupby(["regime_axis", "regime"])["rmse"].idxmin(),
            ["regime_axis", "regime", "model", "rmse"],
        ].to_dict("records"),
        "classification_highest_macro_f1": classification.loc[
            classification.groupby(["regime_axis", "regime"])["macro_f1"].idxmax(),
            ["regime_axis", "regime", "model", "macro_f1"],
        ].to_dict("records"),
        "trading_highest_cumulative_return": trades.loc[
            trades.groupby(["regime_axis", "regime"])["cumulative_return"].idxmax(),
            ["regime_axis", "regime", "model", "cumulative_return", "trade_count"],
        ].to_dict("records"),
    }

    output.mkdir(parents=True, exist_ok=True)
    regression.to_csv(output / "regression_by_regime.csv", index=False, encoding="utf-8-sig")
    classification.to_csv(output / "classification_by_regime.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output / "trades_by_entry_regime.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(output / "regime_sample_counts.csv", index=False, encoding="utf-8-sig")
    counts_by_rolling.to_csv(
        output / "regime_sample_counts_by_rolling.csv", index=False, encoding="utf-8-sig"
    )
    protocol = {
        "schema_version": "1.0",
        "created_after_global_results": True,
        "frozen_before_regime_conditioned_metrics": True,
        "regime_information_time": "sample_time",
        "future_information_used_for_regime": False,
        "trend": {
            "source": "trailing return_72h computed from close at sample_time",
            "threshold_source": "each rolling Train Q33/Q67 only",
            "sign_guard": "DOWN additionally requires return_72h < 0; UP additionally requires return_72h > 0",
            "down": "return_72h <= rolling Train Q33 AND return_72h < 0",
            "sideways": "all rows that satisfy neither DOWN nor UP",
            "up": "return_72h >= rolling Train Q67 AND return_72h > 0",
            "thresholds": trend_limits,
        },
        "volatility": {
            "source": "trailing std_24h",
            "threshold_source": "each rolling Train median only",
            "high": "std_24h > rolling Train median",
            "low": "std_24h <= rolling Train median",
            "thresholds": vol_thresholds,
        },
        "backtest_attribution": (
            "Run the canonical connected OOS non-overlap backtest once per model, "
            "then attribute selected trades to the entry-time regime."
        ),
        "combined_regime": "Cartesian combination of trend and volatility labels",
        "models": [
            {"display_name": spec.display_name, "track": spec.track, "paths": list(spec.paths)}
            for spec in MODEL_SPECS
        ],
    }
    (output / "regime_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "regime_winners.json").write_text(
        json.dumps(winners, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "regression_rows": len(regression),
                "classification_rows": len(classification),
                "trade_rows": len(trades),
                "sample_counts": counts.to_dict("records"),
                "trend_thresholds": trend_limits,
                "volatility_thresholds": vol_thresholds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
