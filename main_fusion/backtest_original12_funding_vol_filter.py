#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
original12 prediction + funding + volatility risk filter backtest

목적:
- original12 모델 prediction은 그대로 사용
- funding_rate, std_24h를 모델 입력이 아니라 외부 risk filter로 사용
- validation에서 funding_threshold × vol_threshold 조합 선택
- test에 동일 threshold 고정 적용

Rule:
if pred == LONG and funding_rate > funding_threshold and std_24h < vol_threshold:
    pred = HOLD
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================
# Config
# ======================================================

PRED_BASE_DIR = Path("outputs/main_fusion_multihead_nonoverlap_th_0012")

FUNDING_PATH = Path(
    "/content/drive/MyDrive/trading12/trading/data/chart/raw/bybit_funding_rate_raw.csv"
)

FEATURE_PATH = Path(
    "/content/drive/MyDrive/trading12/trading/data/merged/merged_with_future_return.csv"
)

OUT_DIR = Path("outputs/original12_funding_vol_filter")

ROLLING_NAMES = ["rolling_1", "rolling_2", "rolling_3"]

LABEL_NAMES = ["SHORT", "HOLD", "LONG"]

SHORT = 0
HOLD = 1
LONG = 2

FEE = 0.001
SLIPPAGE = 0.001

FUNDING_THRESHOLDS = [
    -0.0002,
    -0.0001,
    0.0,
    0.00005,
    0.0001,
    0.00015,
    0.0002,
    0.0003,
    0.0005,
    0.0008,
    0.0010,
]

VOL_THRESHOLDS = [
    0.008,
    0.010,
    0.012,
    0.015,
    0.020,
    0.025,
]


# ======================================================
# Feature load
# ======================================================

def load_funding_hourly():
    funding = pd.read_csv(FUNDING_PATH)

    if "ts" not in funding.columns:
        raise ValueError("funding csv에 ts 컬럼이 없습니다.")

    if "funding_rate" not in funding.columns:
        raise ValueError("funding csv에 funding_rate 컬럼이 없습니다.")

    funding["ts"] = pd.to_datetime(
        funding["ts"],
        utc=True,
        errors="coerce",
    )

    funding["funding_rate"] = pd.to_numeric(
        funding["funding_rate"],
        errors="coerce",
    )

    if funding["ts"].isna().sum() > 0:
        raise ValueError("funding ts에 NaN이 있습니다.")

    if funding["funding_rate"].isna().sum() > 0:
        raise ValueError("funding_rate에 NaN이 있습니다.")

    funding = (
        funding
        .sort_values("ts")
        .drop_duplicates("ts")
        .set_index("ts")[["funding_rate"]]
        .resample("1h")
        .ffill()
        .reset_index()
        .rename(columns={"ts": "sample_time"})
    )

    print("[funding]")
    print("rows :", len(funding))
    print("range:", funding["sample_time"].min(), "~", funding["sample_time"].max())

    return funding


def load_volatility_feature():
    df = pd.read_csv(FEATURE_PATH)

    if "sample_time" in df.columns:
        time_col = "sample_time"
    elif "hour" in df.columns:
        time_col = "hour"
    elif "ts" in df.columns:
        time_col = "ts"
    else:
        raise ValueError("feature 파일에 sample_time/hour/ts 컬럼이 없습니다.")

    if "std_24h" not in df.columns:
        raise ValueError("feature 파일에 std_24h 컬럼이 없습니다.")

    vol = df[[time_col, "std_24h"]].copy()

    vol[time_col] = pd.to_datetime(
        vol[time_col],
        utc=True,
        errors="coerce",
    )

    vol["std_24h"] = pd.to_numeric(
        vol["std_24h"],
        errors="coerce",
    )

    if vol[time_col].isna().sum() > 0:
        raise ValueError("vol time NaN")

    if vol["std_24h"].isna().sum() > 0:
        raise ValueError("std_24h NaN")

    vol = (
        vol
        .drop_duplicates(time_col)
        .sort_values(time_col)
        .rename(columns={time_col: "sample_time"})
        .reset_index(drop=True)
    )

    print("[volatility]")
    print("rows :", len(vol))
    print("range:", vol["sample_time"].min(), "~", vol["sample_time"].max())
    print(vol["std_24h"].describe())

    return vol


# ======================================================
# Utils
# ======================================================

def get_pred_col(df: pd.DataFrame, mode: str):
    if mode == "argmax":
        if "y_pred_argmax" in df.columns:
            return "y_pred_argmax"
        if "y_pred" in df.columns:
            return "y_pred"
        raise ValueError("argmax prediction 컬럼이 없습니다.")

    if mode == "threshold":
        if "y_pred" not in df.columns:
            raise ValueError("threshold prediction 컬럼 y_pred가 없습니다.")
        return "y_pred"

    raise ValueError(f"unknown mode: {mode}")


def merge_features(
    pred_df: pd.DataFrame,
    funding: pd.DataFrame,
    vol: pd.DataFrame,
):
    df = pred_df.copy()

    if "sample_time" not in df.columns:
        raise ValueError("prediction csv에 sample_time 컬럼이 없습니다.")

    df["sample_time"] = pd.to_datetime(
        df["sample_time"],
        utc=True,
        errors="coerce",
    )

    if df["sample_time"].isna().sum() > 0:
        raise ValueError("prediction sample_time에 NaN이 있습니다.")

    merged = df.merge(
        funding,
        on="sample_time",
        how="left",
        validate="many_to_one",
    )

    merged = merged.merge(
        vol,
        on="sample_time",
        how="left",
        validate="many_to_one",
    )

    merged["funding_rate"] = merged["funding_rate"].ffill()
    merged["std_24h"] = merged["std_24h"].ffill()

    if merged["funding_rate"].isna().sum() > 0:
        raise ValueError("merge 후 funding_rate에 NaN이 남아 있습니다.")

    if merged["std_24h"].isna().sum() > 0:
        raise ValueError("merge 후 std_24h에 NaN이 남아 있습니다.")

    return merged


def apply_funding_vol_filter(
    df: pd.DataFrame,
    funding_threshold: float,
    vol_threshold: float,
    pred_col: str,
):
    filtered = df[pred_col].astype(int).to_numpy().copy()

    funding = df["funding_rate"].to_numpy(dtype=np.float64)
    vol = df["std_24h"].to_numpy(dtype=np.float64)

    mask = (
        (filtered == LONG)
        & (funding > funding_threshold)
        & (vol < vol_threshold)
    )

    filtered[mask] = HOLD

    return filtered


def non_overlapping_backtest(
    meta_df: pd.DataFrame,
    preds: np.ndarray,
    fee: float = 0.001,
    slippage: float = 0.001,
):
    df = meta_df.copy()

    if len(df) != len(preds):
        raise ValueError("df와 preds 길이가 다릅니다.")

    required = ["sample_time", "target_time", "raw_future_return"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing)}")

    df["sample_time"] = pd.to_datetime(
        df["sample_time"],
        utc=True,
        errors="coerce",
    )

    df["target_time"] = pd.to_datetime(
        df["target_time"],
        utc=True,
        errors="coerce",
    )

    df["raw_future_return"] = pd.to_numeric(
        df["raw_future_return"],
        errors="coerce",
    )

    if df["sample_time"].isna().sum() > 0:
        raise ValueError("sample_time NaN")

    if df["target_time"].isna().sum() > 0:
        raise ValueError("target_time NaN")

    if df["raw_future_return"].isna().sum() > 0:
        raise ValueError("raw_future_return NaN")

    df["pred"] = preds.astype(int)
    df = df.sort_values("sample_time").reset_index(drop=True)

    cost = fee + slippage
    next_available_time = pd.Timestamp.min.tz_localize("UTC")

    all_strategy_returns = []
    trade_returns = []
    trade_records = []

    for _, row in df.iterrows():
        sample_time = row["sample_time"]
        target_time = row["target_time"]
        pred = int(row["pred"])
        raw_return = float(row["raw_future_return"])

        if sample_time < next_available_time:
            all_strategy_returns.append(0.0)
            continue

        if pred == HOLD:
            all_strategy_returns.append(0.0)
            continue

        if pred == LONG:
            position = 1.0
        elif pred == SHORT:
            position = -1.0
        else:
            raise ValueError(f"unknown pred: {pred}")

        strategy_return = position * raw_return - cost

        all_strategy_returns.append(strategy_return)
        trade_returns.append(strategy_return)

        trade_records.append({
            "sample_time": sample_time,
            "target_time": target_time,
            "pred": pred,
            "pred_label": LABEL_NAMES[pred],
            "raw_future_return": raw_return,
            "strategy_return": strategy_return,
            "funding_rate": row.get("funding_rate", np.nan),
            "std_24h": row.get("std_24h", np.nan),
        })

        next_available_time = target_time

    all_strategy_returns = np.array(all_strategy_returns, dtype=np.float64)
    trade_returns = np.array(trade_returns, dtype=np.float64)

    cumulative_return = float(np.prod(1.0 + all_strategy_returns) - 1.0)

    if all_strategy_returns.std() > 0:
        sharpe_like = float(
            all_strategy_returns.mean()
            / all_strategy_returns.std()
            * np.sqrt(365 * 24)
        )
    else:
        sharpe_like = 0.0

    equity = np.cumprod(1.0 + all_strategy_returns)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    trade_count = len(trade_returns)
    total_rows = len(df)

    if trade_count > 0:
        win_rate = float((trade_returns > 0).mean())
        avg_trade_return = float(trade_returns.mean())
    else:
        win_rate = 0.0
        avg_trade_return = 0.0

    metrics = {
        "cumulative_return": cumulative_return,
        "sharpe_like": sharpe_like,
        "max_drawdown": float(drawdown.min()),
        "trade_count": int(trade_count),
        "trade_ratio": float(trade_count / total_rows),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
    }

    return metrics, pd.DataFrame(trade_records)


def score_for_selection(metrics: dict):
    return (
        metrics["avg_trade_return"],
        metrics["cumulative_return"],
        metrics["sharpe_like"],
        metrics["max_drawdown"],
    )


# ======================================================
# Rolling process
# ======================================================

def run_one_rolling(
    rolling_name: str,
    funding: pd.DataFrame,
    vol: pd.DataFrame,
    pred_mode: str = "threshold",
):
    print("\n" + "=" * 80)
    print(f"🚀 Funding + Vol filter backtest: {rolling_name} | mode={pred_mode}")
    print("=" * 80)

    rolling_dir = PRED_BASE_DIR / rolling_name

    val_path = rolling_dir / "val_predictions.csv"
    test_path = rolling_dir / "test_predictions.csv"

    if not val_path.exists():
        raise FileNotFoundError(val_path)

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    pred_col_val = get_pred_col(val_df, pred_mode)
    pred_col_test = get_pred_col(test_df, pred_mode)

    val_df = merge_features(val_df, funding, vol)
    test_df = merge_features(test_df, funding, vol)

    print("[merge check]")
    print("val funding NaN :", int(val_df["funding_rate"].isna().sum()))
    print("test funding NaN:", int(test_df["funding_rate"].isna().sum()))
    print("val std_24h NaN :", int(val_df["std_24h"].isna().sum()))
    print("test std_24h NaN:", int(test_df["std_24h"].isna().sum()))

    val_base_preds = val_df[pred_col_val].astype(int).to_numpy()
    test_base_preds = test_df[pred_col_test].astype(int).to_numpy()

    val_base_metrics, _ = non_overlapping_backtest(
        val_df,
        val_base_preds,
        fee=FEE,
        slippage=SLIPPAGE,
    )

    test_base_metrics, test_base_trades = non_overlapping_backtest(
        test_df,
        test_base_preds,
        fee=FEE,
        slippage=SLIPPAGE,
    )

    val_results = []

    for funding_th in FUNDING_THRESHOLDS:
        for vol_th in VOL_THRESHOLDS:
            val_filtered_preds = apply_funding_vol_filter(
                val_df,
                funding_threshold=funding_th,
                vol_threshold=vol_th,
                pred_col=pred_col_val,
            )

            metrics, _ = non_overlapping_backtest(
                val_df,
                val_filtered_preds,
                fee=FEE,
                slippage=SLIPPAGE,
            )

            removed_long_count = int(
                (
                    (val_df[pred_col_val].astype(int).to_numpy() == LONG)
                    & (val_filtered_preds == HOLD)
                ).sum()
            )

            val_results.append({
                "funding_threshold": float(funding_th),
                "vol_threshold": float(vol_th),
                "removed_long_count": removed_long_count,
                **metrics,
            })

    best_val = sorted(
        val_results,
        key=lambda r: score_for_selection(r),
        reverse=True,
    )[0]

    best_funding_th = best_val["funding_threshold"]
    best_vol_th = best_val["vol_threshold"]

    print("\n[validation selection]")
    print("base val:", val_base_metrics)
    print("best funding threshold:", best_funding_th)
    print("best vol threshold:", best_vol_th)
    print("best val:", best_val)

    test_filtered_preds = apply_funding_vol_filter(
        test_df,
        funding_threshold=best_funding_th,
        vol_threshold=best_vol_th,
        pred_col=pred_col_test,
    )

    test_filter_metrics, test_filter_trades = non_overlapping_backtest(
        test_df,
        test_filtered_preds,
        fee=FEE,
        slippage=SLIPPAGE,
    )

    removed_long_test = int(
        (
            (test_df[pred_col_test].astype(int).to_numpy() == LONG)
            & (test_filtered_preds == HOLD)
        ).sum()
    )

    print("\n[test result]")
    print("base test  :", test_base_metrics)
    print("filter test:", test_filter_metrics)
    print("removed long test:", removed_long_test)

    out_dir = OUT_DIR / pred_mode / rolling_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(val_results).to_csv(
        out_dir / "val_threshold_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_df["pred_base"] = test_base_preds
    test_df["pred_filtered"] = test_filtered_preds
    test_df["funding_filter_threshold"] = best_funding_th
    test_df["vol_filter_threshold"] = best_vol_th

    test_df.to_csv(
        out_dir / "test_predictions_with_funding_vol_filter.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_base_trades.to_csv(
        out_dir / "test_trades_base.csv",
        index=False,
        encoding="utf-8-sig",
    )

    test_filter_trades.to_csv(
        out_dir / "test_trades_funding_vol_filter.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "rolling": rolling_name,
        "pred_mode": pred_mode,
        "selected_funding_threshold": best_funding_th,
        "selected_vol_threshold": best_vol_th,
        "validation_base": val_base_metrics,
        "validation_best_filter": best_val,
        "test_base": test_base_metrics,
        "test_funding_vol_filter": test_filter_metrics,
        "removed_long_test": removed_long_test,
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ======================================================
# Main
# ======================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PRED_BASE_DIR:", PRED_BASE_DIR)
    print("FUNDING_PATH :", FUNDING_PATH)
    print("FEATURE_PATH :", FEATURE_PATH)
    print("OUT_DIR      :", OUT_DIR)

    funding = load_funding_hourly()
    vol = load_volatility_feature()

    all_results = []

    for rolling_name in ROLLING_NAMES:
        result = run_one_rolling(
            rolling_name=rolling_name,
            funding=funding,
            vol=vol,
            pred_mode="threshold",
        )
        all_results.append(result)

    summary = []

    for r in all_results:
        summary.append({
            "rolling": r["rolling"],
            "selected_funding_threshold": r["selected_funding_threshold"],
            "selected_vol_threshold": r["selected_vol_threshold"],

            "base_return": r["test_base"]["cumulative_return"],
            "filter_return": r["test_funding_vol_filter"]["cumulative_return"],

            "base_sharpe": r["test_base"]["sharpe_like"],
            "filter_sharpe": r["test_funding_vol_filter"]["sharpe_like"],

            "base_mdd": r["test_base"]["max_drawdown"],
            "filter_mdd": r["test_funding_vol_filter"]["max_drawdown"],

            "base_trade_ratio": r["test_base"]["trade_ratio"],
            "filter_trade_ratio": r["test_funding_vol_filter"]["trade_ratio"],

            "base_win_rate": r["test_base"]["win_rate"],
            "filter_win_rate": r["test_funding_vol_filter"]["win_rate"],

            "removed_long_test": r["removed_long_test"],
        })

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUT_DIR / "funding_vol_filter_summary_threshold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("📌 Funding + Vol Filter Summary - Threshold")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    print("\nSaved:", OUT_DIR / "funding_vol_filter_summary_threshold.csv")


if __name__ == "__main__":
    main()