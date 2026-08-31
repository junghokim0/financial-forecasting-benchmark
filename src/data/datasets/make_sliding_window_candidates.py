#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_sliding_window_candidates.py

역할:
- merged_with_future_return.csv를 읽어서
- 전체 데이터에서 sliding window 후보를 생성한다.
- 아직 train/val/test split은 하지 않는다.
- 아직 LONG/SHORT/HOLD label도 만들지 않는다.
- 아직 scaling도 하지 않는다.

입력:
data/merged/merged_with_future_return.csv

출력:
data/dataset/candidates/window_72/
    - X_chart_candidates.npy
    - X_news_candidates.npy
    - sample_meta.csv
    - candidate_config.json
    - chart_cols.csv
    - news_cols.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 설정
# =========================
INPUT_PATH = Path("data/merged/merged_with_future_return.csv")
#INPUT_PATH = Path("data/merged/merged_with_future_return_deriv14.csv")
#INPUT_PATH = Path("data/merged/merged_with_future_return_funding13.csv")

OUT_BASE_DIR = Path("data/dataset/candidates")
#OUT_BASE_DIR = Path("data/dataset/candidates_deriv14")
#OUT_BASE_DIR = Path("data/dataset/candidates_funding13")

WINDOW_SIZE = 72
PREDICTION_HORIZON = 24


CHART_FEATURES = [
    "log_return",
    "return_6h",
    "return_24h",
    "std_24h",
    "close_ma24_gap",
    "close_ma72_gap",
    "volume_ratio_24",
    "spread_ratio",
    "macd_hist",
    "hour_sin",
    "hour_cos",
    "is_missing_candle",
    #"funding_rate",   #추가
    #"open_interest",  #추가
]


NEWS_FEATURES = [
    "news_presence",
    "news_count_log1p",
    "finbert_mean",
    "finbert_sq_mean",
    "finbert_pos_sum",
    "finbert_neg_sum",
    "pos_neg_count_imbalance",
    "finbert_mean_ma_24h",
    "news_count_sum_24h",
]


REQUIRED_META_COLS = [
    "sample_time",
    "target_time",
    "future_return_24h",
]


def check_required_columns(df: pd.DataFrame):
    required_cols = (
        ["hour"]
        + CHART_FEATURES
        + NEWS_FEATURES
        + REQUIRED_META_COLS
    )

    missing = set(required_cols) - set(df.columns)

    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")

    print("\n[feature column check]")
    print("chart feature count:", len(CHART_FEATURES))
    print("news feature count :", len(NEWS_FEATURES))
    print("required columns OK")


def parse_and_check_time(df: pd.DataFrame) -> pd.DataFrame:
    time_cols = ["hour", "sample_time", "target_time"]

    print("\n[time parse check]")

    for col in time_cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        nan_count = int(df[col].isna().sum())
        print(f"{col} NaN:", nan_count)

        if nan_count > 0:
            raise ValueError(f"{col} 컬럼에 NaN이 있습니다.")

    df = df.sort_values("hour").reset_index(drop=True)

    # hour와 sample_time은 같은 값이어야 함
    mismatch = int((df["hour"] != df["sample_time"]).sum())
    print("hour != sample_time count:", mismatch)

    if mismatch > 0:
        print(df.loc[df["hour"] != df["sample_time"], ["hour", "sample_time"]].head(10))
        raise ValueError("hour와 sample_time이 일치하지 않습니다.")

    # target_time = sample_time + 24h 이어야 함
    expected_target_time = df["sample_time"] + pd.Timedelta(hours=PREDICTION_HORIZON)

    target_mismatch = int((df["target_time"] != expected_target_time).sum())
    print("target_time mismatch count:", target_mismatch)

    if target_mismatch > 0:
        bad = df.loc[
            df["target_time"] != expected_target_time,
            ["sample_time", "target_time"],
        ].head(10).copy()

        bad["expected_target_time"] = expected_target_time.loc[bad.index]
        print(bad)
        raise ValueError("target_time이 sample_time + 24h와 일치하지 않습니다.")

    return df


def check_1h_interval(df: pd.DataFrame):
    diff = df["hour"].diff()

    bad_mask = (
        diff.notna()
        & (diff != pd.Timedelta(hours=1))
    )

    bad_diff_count = int(bad_mask.sum())

    print("\n[1h interval check]")
    print("bad 1h interval count:", bad_diff_count)

    if bad_diff_count > 0:
        bad_rows = df.loc[bad_mask, ["hour"]].copy()
        bad_rows["prev_hour"] = df["hour"].shift(1).loc[bad_mask]
        bad_rows["diff"] = diff.loc[bad_mask]

        print(bad_rows.head(20))
        raise ValueError("데이터가 1시간 간격으로 연속되어 있지 않습니다.")

    print("1h interval check: OK")


def check_numeric_values(df: pd.DataFrame):
    numeric_cols = CHART_FEATURES + NEWS_FEATURES + ["future_return_24h"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    values = df[numeric_cols].to_numpy(dtype=np.float64)

    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())

    print("\n[numeric value check]")
    print("NaN count:", nan_count)
    print("inf count:", inf_count)

    if nan_count > 0:
        print("\n[NaN columns]")
        print(df[numeric_cols].isna().sum()[df[numeric_cols].isna().sum() > 0])
        raise ValueError("숫자 feature에 NaN이 있습니다.")

    if inf_count > 0:
        raise ValueError("숫자 feature에 inf 값이 있습니다.")


def make_candidates(df: pd.DataFrame):
    chart_values = df[CHART_FEATURES].to_numpy(dtype=np.float32)
    news_values = df[NEWS_FEATURES].to_numpy(dtype=np.float32)

    X_chart_list = []
    X_news_list = []
    meta_rows = []

    # i는 sample_time의 row index
    # window는 i-71 ~ i
    start_i = WINDOW_SIZE - 1
    end_i = len(df)

    print("\n[sliding window generation]")
    print("window_size:", WINDOW_SIZE)
    print("prediction_horizon:", PREDICTION_HORIZON)
    print("start index:", start_i)
    print("end index:", end_i - 1)

    for i in range(start_i, end_i):
        input_start_idx = i - WINDOW_SIZE + 1
        input_end_idx = i

        x_chart = chart_values[input_start_idx: input_end_idx + 1]
        x_news = news_values[input_start_idx: input_end_idx + 1]

        if x_chart.shape != (WINDOW_SIZE, len(CHART_FEATURES)):
            raise ValueError(f"X_chart shape 오류 at i={i}: {x_chart.shape}")

        if x_news.shape != (WINDOW_SIZE, len(NEWS_FEATURES)):
            raise ValueError(f"X_news shape 오류 at i={i}: {x_news.shape}")

        input_start_time = df.loc[input_start_idx, "hour"]
        input_end_time = df.loc[input_end_idx, "hour"]

        sample_time = df.loc[i, "sample_time"]
        target_time = df.loc[i, "target_time"]
        raw_future_return = float(df.loc[i, "future_return_24h"])

        # input_start_time = sample_time - 71h 검증
        expected_input_start_time = sample_time - pd.Timedelta(hours=WINDOW_SIZE - 1)

        if input_start_time != expected_input_start_time:
            raise ValueError(
                f"input_start_time 불일치 at i={i}: "
                f"{input_start_time} != {expected_input_start_time}"
            )

        # input_end_time과 sample_time은 같아야 함
        if input_end_time != sample_time:
            raise ValueError(
                f"input_end_time 불일치 at i={i}: "
                f"{input_end_time} != {sample_time}"
            )

        X_chart_list.append(x_chart)
        X_news_list.append(x_news)

        meta_rows.append({
            "candidate_index": len(meta_rows),
            "input_start_time": input_start_time,
            "input_end_time": input_end_time,
            "sample_time": sample_time,
            "target_time": target_time,
            "raw_future_return": raw_future_return,
            "window_size": WINDOW_SIZE,
            "prediction_horizon": PREDICTION_HORIZON,
        })

    X_chart = np.stack(X_chart_list).astype(np.float32)
    X_news = np.stack(X_news_list).astype(np.float32)
    meta = pd.DataFrame(meta_rows)

    return X_chart, X_news, meta


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}")

    out_dir = OUT_BASE_DIR / f"window_{WINDOW_SIZE}"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_chart_path = out_dir / "X_chart_candidates.npy"
    x_news_path = out_dir / "X_news_candidates.npy"
    meta_path = out_dir / "sample_meta.csv"
    config_path = out_dir / "candidate_config.json"
    chart_cols_path = out_dir / "chart_cols.csv"
    news_cols_path = out_dir / "news_cols.csv"

    print("🚀 Sliding Window 후보 생성 시작")
    print("input :", INPUT_PATH)
    print("out_dir:", out_dir)

    df = pd.read_csv(INPUT_PATH)

    print("\n[load check]")
    print("rows:", len(df))
    print("columns:", df.columns.tolist())

    check_required_columns(df)
    df = parse_and_check_time(df)

    print("\n[time range]")
    print("hour       :", df["hour"].min(), "~", df["hour"].max())
    print("target_time:", df["target_time"].min(), "~", df["target_time"].max())

    check_1h_interval(df)
    check_numeric_values(df)

    X_chart, X_news, meta = make_candidates(df)

    print("\n[candidate result]")
    print("X_chart shape:", X_chart.shape)
    print("X_news shape :", X_news.shape)
    print("meta rows    :", len(meta))

    expected_candidates = len(df) - WINDOW_SIZE + 1
    print("expected candidates:", expected_candidates)

    if len(meta) != expected_candidates:
        raise ValueError(
            f"candidate 수가 예상과 다릅니다. "
            f"expected={expected_candidates}, actual={len(meta)}"
        )

    if X_chart.shape != (len(meta), WINDOW_SIZE, len(CHART_FEATURES)):
        raise ValueError(f"X_chart 최종 shape 오류: {X_chart.shape}")

    if X_news.shape != (len(meta), WINDOW_SIZE, len(NEWS_FEATURES)):
        raise ValueError(f"X_news 최종 shape 오류: {X_news.shape}")

    # 저장
    np.save(x_chart_path, X_chart)
    np.save(x_news_path, X_news)
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

    pd.Series(CHART_FEATURES).to_csv(
        chart_cols_path,
        index=False,
        header=False,
        encoding="utf-8-sig",
    )

    pd.Series(NEWS_FEATURES).to_csv(
        news_cols_path,
        index=False,
        header=False,
        encoding="utf-8-sig",
    )

    config = {
        "input_file": str(INPUT_PATH),
        "output_dir": str(out_dir),
        "window_size": WINDOW_SIZE,
        "prediction_horizon": PREDICTION_HORIZON,
        "chart_features": CHART_FEATURES,
        "news_features": NEWS_FEATURES,
        "chart_feature_count": len(CHART_FEATURES),
        "news_feature_count": len(NEWS_FEATURES),
        "num_candidates": int(len(meta)),
        "X_chart_shape": list(X_chart.shape),
        "X_news_shape": list(X_news.shape),
        "sample_time_range": [
            str(meta["sample_time"].min()),
            str(meta["sample_time"].max()),
        ],
        "target_time_range": [
            str(meta["target_time"].min()),
            str(meta["target_time"].max()),
        ],
        "note": (
            "This file contains full sliding window candidates only. "
            "No train/val/test split, no scaling, and no LONG/SHORT/HOLD labels are applied here. "
            "Rolling split and percentile threshold labeling must be done in the next step."
        ),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n✅ Sliding Window 후보 생성 완료")
    print("saved X_chart:", x_chart_path)
    print("saved X_news :", x_news_path)
    print("saved meta   :", meta_path)
    print("saved config :", config_path)
    print("saved chart cols:", chart_cols_path)
    print("saved news cols :", news_cols_path)

    print("\n[meta head]")
    print(meta.head().to_string(index=False))

    print("\n[meta tail]")
    print(meta.tail().to_string(index=False))


if __name__ == "__main__":
    main()