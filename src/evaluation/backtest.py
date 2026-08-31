"""Cryptova-compatible 24-hour non-overlapping backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


LABEL_NAMES = ["SHORT", "HOLD", "LONG"]


@dataclass(frozen=True)
class BacktestConfig:
    fee: float = 0.001
    slippage: float = 0.001

    @property
    def cost(self) -> float:
        # Intentionally identical to the existing Cryptova implementation.
        return self.fee + self.slippage


def non_overlapping_backtest(
    prediction_frame: pd.DataFrame,
    config: BacktestConfig = BacktestConfig(),
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Run the original Cryptova backtest without importing model code.

    A selected LONG/SHORT trade blocks new entries until that row's target_time.
    HOLD and blocked rows contribute zero to the hourly strategy-return series.
    """
    required = {"sample_time", "target_time", "raw_future_return", "y_pred"}
    missing = sorted(required - set(prediction_frame.columns))
    if missing:
        raise ValueError(f"Missing backtest columns: {missing}")

    frame = prediction_frame.copy()
    frame["sample_time"] = pd.to_datetime(frame["sample_time"], utc=True, errors="raise")
    frame["target_time"] = pd.to_datetime(frame["target_time"], utc=True, errors="raise")
    frame = frame.sort_values("sample_time").reset_index(drop=True)

    total_rows = len(frame)
    next_available_time = pd.Timestamp.min.tz_localize("UTC")
    all_strategy_returns: list[float] = []
    trade_returns: list[float] = []
    trade_records: list[dict] = []

    for row in frame.itertuples(index=False):
        sample_time = row.sample_time
        target_time = row.target_time
        prediction = int(row.y_pred)
        raw_return = float(row.raw_future_return)

        if sample_time < next_available_time:
            all_strategy_returns.append(0.0)
            continue
        if prediction == 1:
            all_strategy_returns.append(0.0)
            continue
        if prediction == 2:
            position = 1.0
        elif prediction == 0:
            position = -1.0
        else:
            raise ValueError(f"Unsupported prediction label: {prediction}")

        strategy_return = position * raw_return - config.cost
        all_strategy_returns.append(strategy_return)
        trade_returns.append(strategy_return)
        trade_records.append(
            {
                "sample_time": sample_time,
                "target_time": target_time,
                "pred": prediction,
                "pred_label": LABEL_NAMES[prediction],
                "raw_future_return": raw_return,
                "strategy_return": strategy_return,
            }
        )
        next_available_time = target_time

    all_returns = np.asarray(all_strategy_returns, dtype=np.float64)
    trades = np.asarray(trade_returns, dtype=np.float64)

    if len(all_returns) == 0:
        metrics = {
            "cumulative_return": 0.0,
            "sharpe_like": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "trade_ratio": 0.0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
        }
        return metrics, pd.DataFrame(trade_records)

    cumulative_return = float(np.prod(1.0 + all_returns) - 1.0)
    sharpe_like = (
        float(all_returns.mean() / all_returns.std() * np.sqrt(365 * 24))
        if all_returns.std() > 0
        else 0.0
    )
    equity = np.cumprod(1.0 + all_returns)
    running_max = np.maximum.accumulate(equity)
    max_drawdown = float((equity / running_max - 1.0).min())
    trade_count = len(trades)

    metrics = {
        "cumulative_return": cumulative_return,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "trade_count": int(trade_count),
        "trade_ratio": float(trade_count / total_rows) if total_rows > 0 else 0.0,
        "win_rate": float((trades > 0).mean()) if trade_count > 0 else 0.0,
        "avg_trade_return": float(trades.mean()) if trade_count > 0 else 0.0,
    }
    return metrics, pd.DataFrame(trade_records)
