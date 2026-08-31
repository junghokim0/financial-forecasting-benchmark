#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHART_CSV = PROJECT_ROOT / "data/chart/processed/chart_hourly_features.csv"
#CHART_CSV = PROJECT_ROOT / "data/chart/processed/chart_hourly_features_deriv14.csv"
#CHART_CSV = PROJECT_ROOT / "data/chart/processed/chart_hourly_features_funding13.csv"
NEWS_CSV = PROJECT_ROOT / "data/news/processed/news_hourly_features.csv"

OUT_CSV = PROJECT_ROOT / "data/merged/merged_hourly_features.csv"
#OUT_CSV = PROJECT_ROOT / "data/merged/merged_hourly_features_deriv14.csv"
#OUT_CSV = PROJECT_ROOT / "data/merged/merged_hourly_features_funding13.csv"

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
    #"funding_rate", #추가
    #"open_interest", #추가
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


def main():
    print("🚀 Merge 시작")
    print("chart:", CHART_CSV)
    print("news :", NEWS_CSV)

    if not CHART_CSV.exists():
        raise FileNotFoundError(f"chart 파일 없음: {CHART_CSV}")

    if not NEWS_CSV.exists():
        raise FileNotFoundError(f"news 파일 없음: {NEWS_CSV}")

    chart = pd.read_csv(CHART_CSV)
    news = pd.read_csv(NEWS_CSV)

    print("\n[raw rows]")
    print("chart rows:", len(chart))
    print("news rows :", len(news))

    # 1. hour 컬럼 확인
    if "hour" not in chart.columns:
        raise ValueError("chart 데이터에 hour 컬럼이 없습니다.")

    if "hour" not in news.columns:
        raise ValueError("news 데이터에 hour 컬럼이 없습니다.")

    # 2. 모델 feature 컬럼 명시적 검증
    missing_chart_features = set(CHART_FEATURES) - set(chart.columns)
    missing_news_features = set(NEWS_FEATURES) - set(news.columns)

    if missing_chart_features:
        raise ValueError(f"차트 feature 없음: {sorted(missing_chart_features)}")

    if missing_news_features:
        raise ValueError(f"뉴스 feature 없음: {sorted(missing_news_features)}")

    print("\n[feature check]")
    print("chart feature count:", len(CHART_FEATURES))
    print("news feature count :", len(NEWS_FEATURES))
    print("feature columns OK")

    # 3. 시간 파싱
    chart["hour"] = pd.to_datetime(chart["hour"], utc=True, errors="coerce")
    news["hour"] = pd.to_datetime(news["hour"], utc=True, errors="coerce")

    print("\n[time parse failed]")
    print("chart hour NaN:", int(chart["hour"].isna().sum()))
    print("news hour NaN :", int(news["hour"].isna().sum()))

    chart = chart.dropna(subset=["hour"])
    news = news.dropna(subset=["hour"])

    # candle_start도 있으면 검증용으로 파싱
    if "candle_start" in chart.columns:
        chart["candle_start"] = pd.to_datetime(
            chart["candle_start"],
            utc=True,
            errors="coerce",
        )

    # 4. 중복 hour 검증
    chart_dup = int(chart["hour"].duplicated().sum())
    news_dup = int(news["hour"].duplicated().sum())

    print("\n[duplicate hour]")
    print("chart duplicated hour:", chart_dup)
    print("news duplicated hour :", news_dup)

    if chart_dup > 0:
        raise ValueError("chart 데이터에 중복 hour가 있습니다.")

    if news_dup > 0:
        raise ValueError("news 데이터에 중복 hour가 있습니다.")

    # 5. 시간 범위 및 overlap 검증
    chart_start = chart["hour"].min()
    chart_end = chart["hour"].max()
    news_start = news["hour"].min()
    news_end = news["hour"].max()

    print("\n[time range]")
    print("chart:", chart_start, "~", chart_end)
    print("news :", news_start, "~", news_end)

    overlap_start = max(chart_start, news_start)
    overlap_end = min(chart_end, news_end)

    print("\n[overlap range]")
    print("overlap:", overlap_start, "~", overlap_end)

    if overlap_start > overlap_end:
        raise ValueError("chart와 news의 시간 범위가 겹치지 않습니다.")

    # 6. chart의 hour = candle_start + 1h 검증
    if "candle_start" in chart.columns:
        hour_gap = chart["hour"] - chart["candle_start"]
        bad_gap = int((hour_gap != pd.Timedelta(hours=1)).sum())

        print("\n[chart hour alignment]")
        print("bad candle_start -> hour gap count:", bad_gap)

        if bad_gap > 0:
            raise ValueError("chart의 hour가 candle_start + 1시간 구조가 아닙니다.")

    # 7. merge
    merged = chart.merge(
        news,
        on="hour",
        how="left",
        validate="one_to_one",
    )

    print("\n[merge rows]")
    print("chart rows :", len(chart))
    print("merged rows:", len(merged))

    if len(merged) != len(chart):
        raise ValueError("merge 후 row 수가 chart와 다릅니다.")

    # 8. news feature NaN 확인 후 fill
    print("\n[news NaN ratio before fill]")
    print(merged[NEWS_FEATURES].isna().mean().sort_values(ascending=False))

    merged[NEWS_FEATURES] = merged[NEWS_FEATURES].fillna(0.0)

    print("\n[news NaN after fill]")
    print(int(merged[NEWS_FEATURES].isna().sum().sum()))

    # 9. news_presence 비율 확인
    print("\n[news presence check]")
    print("news_presence mean:", float(merged["news_presence"].mean()))
    print("news_presence sum :", float(merged["news_presence"].sum()))
    print("news_presence min :", float(merged["news_presence"].min()))
    print("news_presence max :", float(merged["news_presence"].max()))

    # 10. 1시간 간격 검증
    merged = merged.sort_values("hour").reset_index(drop=True)

    diff = merged["hour"].diff().dropna()
    bad_diff = int((diff != pd.Timedelta(hours=1)).sum())

    print("\n[1h interval check]")
    print("bad 1h interval count:", bad_diff)

    if bad_diff > 0:
        bad_rows = merged.loc[
            merged["hour"].diff() != pd.Timedelta(hours=1),
            ["hour"],
        ].head(10)

        print("\n[bad interval sample]")
        print(bad_rows)
        raise ValueError("merged 데이터에 1시간 간격이 아닌 구간이 있습니다.")

    # 11. 전체 NaN / inf 검증
    merged = merged.replace([np.inf, -np.inf], np.nan)

    total_nan = int(merged.isna().sum().sum())

    print("\n[total NaN after merge]")
    print(total_nan)

    if total_nan > 0:
        print("\n[NaN columns]")
        print(merged.isna().sum()[merged.isna().sum() > 0])
        raise ValueError("merge 후 NaN이 남아 있습니다.")

    # 12. 저장
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n✅ Merge 완료")
    print("saved:", OUT_CSV)
    print("rows:", len(merged))
    print("start:", merged["hour"].min())
    print("end:", merged["hour"].max())

    print("\n[preview]")
    preview_cols = [
        "hour",
        "candle_start",
        "log_return",
        "return_24h",
        "news_presence",
        "news_count_log1p",
        "finbert_mean",
        "finbert_sq_mean",
    ]

    preview_cols = [col for col in preview_cols if col in merged.columns]
    print(merged[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()