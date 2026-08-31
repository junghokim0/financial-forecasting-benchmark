#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_merged_future_return.py

역할:
- merged_hourly_features.csv를 읽어서
- 24시간 뒤 close 가격을 기준으로 future_return_24h 생성
- sample_time, target_time 저장
- 이후 rolling별 percentile label 생성을 위한 기본 파일 생성

입력:
data/merged/merged_hourly_features.csv

출력:
data/merged/merged_with_future_return.csv

중요:
- 여기서는 label을 만들지 않음
- label threshold는 rolling train sample에서만 계산해야 함
- 이 파일은 raw future_return을 준비하는 단계
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 설정
# =========================
INPUT_PATH = Path("data/merged/merged_hourly_features.csv")
OUTPUT_PATH = Path("data/merged/merged_with_future_return_checked.csv")

PREDICTION_HORIZON = 24  # 24시간 뒤 수익률


def main():
    # =========================
    # 0. 입력 파일 확인
    # =========================
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_PATH}")

    # =========================
    # 1. 데이터 로드
    # =========================
    df = pd.read_csv(INPUT_PATH)

    print("🚀 Future return 생성 시작")
    print("input :", INPUT_PATH)
    print("output:", OUTPUT_PATH)

    print("\n===== Load Check =====")
    print("rows:", len(df))
    print("columns:", df.columns.tolist())

    # =========================
    # 2. 시간 컬럼 확인
    # =========================
    if "hour" in df.columns:
        time_col = "hour"
    elif "timestamp" in df.columns:
        time_col = "timestamp"
    else:
        raise ValueError("'hour' 또는 'timestamp' 시간 컬럼이 없습니다.")

    print("time_col:", time_col)

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    time_nan = int(df[time_col].isna().sum())
    print("time parse NaN:", time_nan)

    if time_nan > 0:
        raise ValueError("시간 컬럼 파싱 실패 값이 있습니다.")

    df = df.sort_values(time_col).reset_index(drop=True)

    print("time range:", df[time_col].min(), "~", df[time_col].max())

    # =========================
    # 3. 필수 컬럼 확인
    # =========================
    if "close" not in df.columns:
        raise ValueError("'close' 컬럼이 없습니다. future_return 계산 불가.")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    close_nan = int(df["close"].isna().sum())
    print("close NaN:", close_nan)

    if close_nan > 0:
        raise ValueError("close 컬럼에 NaN이 있습니다.")

    non_positive_close = int((df["close"] <= 0).sum())
    print("close <= 0 count:", non_positive_close)

    if non_positive_close > 0:
        bad_rows = df.loc[df["close"] <= 0, [time_col, "close"]].head(10)
        print("\n[close <= 0 sample]")
        print(bad_rows)
        raise ValueError("close가 0 이하인 row가 있습니다.")

    # =========================
    # 4. 중복 시간 확인
    # =========================
    dup_count = int(df[time_col].duplicated().sum())
    print("duplicated time count:", dup_count)

    if dup_count > 0:
        bad_dup = df.loc[df[time_col].duplicated(keep=False), [time_col]].head(20)
        print("\n[duplicated time sample]")
        print(bad_dup)
        raise ValueError("중복된 시간 row가 있습니다.")

    # =========================
    # 5. 1시간 간격 재검증
    # =========================
    time_diff = df[time_col].diff()

    bad_diff_mask = (
        time_diff.notna()
        & (time_diff != pd.Timedelta(hours=1))
    )

    bad_diff_count = int(bad_diff_mask.sum())

    print("\n===== 1h Interval Check =====")
    print("bad 1h interval count:", bad_diff_count)

    if bad_diff_count > 0:
        bad_diff_rows = df.loc[bad_diff_mask, [time_col]].copy()
        bad_diff_rows["prev_time"] = df[time_col].shift(1).loc[bad_diff_mask]
        bad_diff_rows["diff"] = time_diff.loc[bad_diff_mask]

        print("\n[bad interval sample]")
        print(bad_diff_rows.head(20))

        raise ValueError(
            "1시간 간격이 아닌 구간이 있습니다. "
            "shift(-24)를 사용할 수 없습니다."
        )

    print("1h interval check: OK")

    # =========================
    # 6. future target 생성
    # =========================
    df["sample_time"] = df[time_col]
    df["target_time"] = df[time_col].shift(-PREDICTION_HORIZON)

    df["future_close_24h"] = df["close"].shift(-PREDICTION_HORIZON)

    df["future_return_24h"] = (
        df["future_close_24h"] / df["close"] - 1
    )

    # =========================
    # 7. target_time 검증
    # =========================
    expected_target_time = (
        df["sample_time"] + pd.Timedelta(hours=PREDICTION_HORIZON)
    )

    target_time_mismatch_mask = (
        df["target_time"].notna()
        & (df["target_time"] != expected_target_time)
    )

    target_time_mismatch_count = int(target_time_mismatch_mask.sum())

    print("\n===== Target Time Check =====")
    print("target_time mismatch count:", target_time_mismatch_count)

    if target_time_mismatch_count > 0:
        mismatch_rows = df.loc[
            target_time_mismatch_mask,
            ["sample_time", "target_time"]
        ].copy()

        mismatch_rows["expected_target_time"] = expected_target_time.loc[
            target_time_mismatch_mask
        ]

        print("\n[target_time mismatch sample]")
        print(mismatch_rows.head(20))

        raise ValueError("target_time이 sample_time + 24h와 일치하지 않습니다.")

    print("target_time check: OK")

    # =========================
    # 8. inf 처리 및 future 값 검증
    # =========================
    df = df.replace([np.inf, -np.inf], np.nan)

    future_cols = [
        "target_time",
        "future_close_24h",
        "future_return_24h",
    ]

    print("\n===== Future Columns NaN Check Before Drop =====")
    print(df[future_cols].isna().sum())

    # 마지막 horizon개는 future 값이 없는 것이 정상
    expected_tail_nan = PREDICTION_HORIZON

    actual_future_return_nan = int(df["future_return_24h"].isna().sum())
    actual_future_close_nan = int(df["future_close_24h"].isna().sum())
    actual_target_time_nan = int(df["target_time"].isna().sum())

    print("\nexpected tail NaN:", expected_tail_nan)
    print("target_time NaN:", actual_target_time_nan)
    print("future_close_24h NaN:", actual_future_close_nan)
    print("future_return_24h NaN:", actual_future_return_nan)

    if actual_target_time_nan != expected_tail_nan:
        raise ValueError(
            f"target_time NaN 개수가 예상과 다릅니다. "
            f"expected={expected_tail_nan}, actual={actual_target_time_nan}"
        )

    if actual_future_close_nan != expected_tail_nan:
        raise ValueError(
            f"future_close_24h NaN 개수가 예상과 다릅니다. "
            f"expected={expected_tail_nan}, actual={actual_future_close_nan}"
        )

    if actual_future_return_nan != expected_tail_nan:
        raise ValueError(
            f"future_return_24h NaN 개수가 예상과 다릅니다. "
            f"expected={expected_tail_nan}, actual={actual_future_return_nan}"
        )

    # =========================
    # 9. 마지막 24개 제거
    # =========================
    before_rows = len(df)

    df = df.dropna(
        subset=[
            "target_time",
            "future_close_24h",
            "future_return_24h",
        ]
    ).reset_index(drop=True)

    after_rows = len(df)
    removed_rows = before_rows - after_rows

    print("\n===== Drop Tail Rows Check =====")
    print("before rows:", before_rows)
    print("after rows :", after_rows)
    print("removed    :", removed_rows)

    if removed_rows != PREDICTION_HORIZON:
        raise ValueError(
            f"제거된 row 수가 prediction horizon과 다릅니다. "
            f"expected={PREDICTION_HORIZON}, actual={removed_rows}"
        )

    # =========================
    # 10. 최종 검증
    # =========================
    final_nan = int(df[["sample_time", "target_time", "close", "future_close_24h", "future_return_24h"]].isna().sum().sum())
    print("\n===== Final NaN Check =====")
    print("final target-related NaN:", final_nan)

    if final_nan > 0:
        raise ValueError("최종 future return 관련 컬럼에 NaN이 남아 있습니다.")

    # 수익률 분포 확인
    print("\n===== Future Return Describe =====")
    print(df["future_return_24h"].describe())

    print("\nfuture_return_24h quantiles:")
    print(df["future_return_24h"].quantile([0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]))

    # 너무 비정상적인 수익률이 있는지 단순 확인
    extreme_count = int((df["future_return_24h"].abs() > 0.5).sum())
    print("\nabs(future_return_24h) > 50% count:", extreme_count)

    if extreme_count > 0:
        print("\n[extreme future_return sample]")
        print(
            df.loc[
                df["future_return_24h"].abs() > 0.5,
                ["sample_time", "target_time", "close", "future_close_24h", "future_return_24h"]
            ].head(20)
        )

    # =========================
    # 11. 샘플 출력
    # =========================
    show_cols = [
        time_col,
        "sample_time",
        "target_time",
        "close",
        "future_close_24h",
        "future_return_24h",
    ]

    print("\n===== Head Sample =====")
    print(df[show_cols].head().to_string(index=False))

    print("\n===== Tail Sample =====")
    print(df[show_cols].tail().to_string(index=False))

    # =========================
    # 12. 저장
    # =========================
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n✅ Future return 생성 완료")
    print("saved:", OUTPUT_PATH)
    print("rows:", len(df))
    print("start:", df[time_col].min())
    print("end:", df[time_col].max())
    print("target start:", df["target_time"].min())
    print("target end:", df["target_time"].max())


if __name__ == "__main__":
    main()