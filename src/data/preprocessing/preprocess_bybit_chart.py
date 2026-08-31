#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chart 전처리 코드

시간 기준:
- candle_start: 1시간봉 시작 시간
- hour: 모델이 해당 feature를 사용할 수 있는 시간

예:
- 10:00~10:59 캔들 → candle_start = 10:00
- 해당 캔들은 11:00부터 확정 사용 가능 → hour = 11:00

뉴스 feature도 hour 기준으로 merge한다.
"""

import os
import numpy as np
import pandas as pd


IN_CSV = "data/chart/raw/chart_raw_bybit.csv"
OUT_DIR = "data/chart/processed"
OUT_CSV = f"{OUT_DIR}/chart_hourly_features.csv"

FREQ = "1h"

VOLUME_RATIO_CLIP_MAX = 20.0

USE_SPREAD_RATIO_CLIP = True
SPREAD_RATIO_CLIP_MAX = 0.2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(IN_CSV)

    print("현재 작업 위치:", os.getcwd())
    print("입력 파일:", os.path.abspath(IN_CSV))
    print("raw rows:", len(df))

    # 1. 필수 컬럼 확인
    need = {"ts", "open", "high", "low", "close", "volume"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"필수 컬럼 없음: {sorted(miss)}")

    print("raw ts min:", df["ts"].min())
    print("raw ts max:", df["ts"].max())

    # 2. 시간 변환
    df["ts"] = pd.to_datetime(
        df["ts"],
        utc=True,
        errors="coerce",
        format="mixed"
    )

    print("timestamp 변환 실패 개수:", df["ts"].isna().sum())

    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    print("parsed ts min:", df["ts"].min())
    print("parsed ts max:", df["ts"].max())

    # 3. 숫자형 변환
    numeric_cols = ["open", "high", "low", "close", "volume"]

    has_turnover = "turnover" in df.columns
    if has_turnover:
        numeric_cols.append("turnover")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_numeric_drop = len(df)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    print("숫자형 변환 후 OHLCV 결측 제거 rows:", before_numeric_drop - len(df))

    # 4. 1시간 간격 검증 및 보정
    df = df.set_index("ts").asfreq(FREQ)

    # ffill 전에 결측 캔들 여부 기록
    df["is_missing_candle"] = df["close"].isna().astype(float)

    missing_count = int(df["is_missing_candle"].sum())
    print("asfreq 이후 생성된 결측 캔들 수:", missing_count)

    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].ffill()

    df["volume"] = df["volume"].fillna(0)

    if has_turnover:
        df["turnover"] = df["turnover"].fillna(0)

    print("ffill 이후 close 결측 개수:", int(df["close"].isna().sum()))

    # 5. Feature 생성
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["return_6h"] = df["close"].pct_change(6)
    df["return_24h"] = df["close"].pct_change(24)

    df["std_24h"] = df["log_return"].rolling(24, min_periods=24).std()

    ma_24 = df["close"].rolling(24, min_periods=24).mean()
    ma_72 = df["close"].rolling(72, min_periods=72).mean()

    df["close_ma24_gap"] = (df["close"] - ma_24) / (ma_24 + 1e-9)
    df["close_ma72_gap"] = (df["close"] - ma_72) / (ma_72 + 1e-9)

    volume_ma_24 = df["volume"].rolling(24, min_periods=24).mean()
    df["volume_ratio_24"] = df["volume"] / (volume_ma_24 + 1e-9)
    df["volume_ratio_24"] = df["volume_ratio_24"].clip(
        lower=0,
        upper=VOLUME_RATIO_CLIP_MAX
    )

    df["spread_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)

    if USE_SPREAD_RATIO_CLIP:
        df["spread_ratio"] = df["spread_ratio"].clip(
            lower=0,
            upper=SPREAD_RATIO_CLIP_MAX
        )

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal

    utc_hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * utc_hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * utc_hour / 24)

    # 6. 이상값 및 NaN 처리
    df = df.replace([np.inf, -np.inf], np.nan)

    before_dropna = len(df)
    df = df.dropna()
    after_dropna = len(df)

    print("dropna removed rows:", before_dropna - after_dropna)

    # 7. 컬럼 정리
    chart_feature_cols = [
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
    ]

    raw_save_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if has_turnover:
        raw_save_cols.append("turnover")

    # 8. candle_start / hour 분리
    df = df.reset_index()
    df = df.rename(columns={"ts": "candle_start"})

    df["hour"] = df["candle_start"] + pd.Timedelta(hours=1)

    save_cols = [
        "hour",
        "candle_start",
    ] + raw_save_cols + chart_feature_cols

    df = df[save_cols]

    # 9. 현재 chart 단계에서 가능한 검증
    print("\n[volume_ratio_24 distribution]")
    print(
        df["volume_ratio_24"].describe(
            percentiles=[0.90, 0.95, 0.99, 0.999]
        )
    )

    print(
        "\nvolume_ratio_24 >= 20 개수:",
        int((df["volume_ratio_24"] >= VOLUME_RATIO_CLIP_MAX).sum())
    )

    print("\n[is_missing_candle]")
    print(df["is_missing_candle"].value_counts())

    print("\n[chart feature correlation]")
    corr = df[chart_feature_cols].corr()
    print(corr.round(3))

    # 10. 저장
    df.to_csv(OUT_CSV, index=False)

    print("\n✅ Chart preprocessing done")
    print("saved:", OUT_CSV)
    print("rows:", len(df))
    print("hour start:", df["hour"].min())
    print("hour end:", df["hour"].max())
    print("candle_start start:", df["candle_start"].min())
    print("candle_start end:", df["candle_start"].max())

    print("\n[모델 입력용 chart features]")
    print(chart_feature_cols)

    print("\n[주의]")
    print("원본 OHLCV와 turnover는 저장만 하고, 초기 모델 입력 feature에서는 제외 권장.")
    print("모델은 1시간봉 마감 후 확정된 feature만 사용한다고 가정.")
    print("뉴스 feature와는 hour 컬럼 기준으로 merge.")


if __name__ == "__main__":
    main()