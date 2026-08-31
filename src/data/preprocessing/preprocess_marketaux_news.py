#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess_marketaux_news.py

역할:
1. MarketAux raw 뉴스 데이터 로드
2. 중복 뉴스 제거
3. FinBERT로 positive / negative / neutral 확률 계산
4. 설명 AI용 뉴스 context 저장
5. 모델 학습용 1시간 단위 뉴스 feature 저장

입력:
- data/news/raw/news_raw_marketaux.csv

출력:
- data/news/processed/news_explain_context.csv
- data/news/processed/news_hourly_features.csv
- data/news/processed/news_preprocess_metadata.json

최종 NEWS_FEATURES 9개:
- news_presence
- news_count_log1p
- finbert_mean
- finbert_sq_mean
- finbert_pos_sum
- finbert_neg_sum
- pos_neg_count_imbalance
- finbert_mean_ma_24h
- news_count_sum_24h

중요:
- scaling은 여기서 하지 않음
- scaling은 train/val/test split 이후 train 기준으로 fit 해야 데이터 누수를 막을 수 있음
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import BertTokenizer, BertForSequenceClassification


# ============================================================
# 경로 설정
# 현재 파일 위치:
# trading/data/news/processed/preprocess_marketaux_news.py
# parents[3] = trading
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_IN = PROJECT_ROOT / "data/news/raw/news_raw_marketaux.csv"
PROC_DIR = PROJECT_ROOT / "data/news/processed"

EXPLAIN_OUT = PROC_DIR / "news_explain_context.csv"
HOURLY_OUT = PROC_DIR / "news_hourly_features.csv"
META_OUT = PROC_DIR / "news_preprocess_metadata.json"


# ============================================================
# 최종 모델용 뉴스 feature
# ============================================================
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


# ============================================================
# 1. raw 뉴스 기본 정리
# ============================================================
def normalize_title(title: str) -> str:
    """
    제목 중복 제거용 정규화.
    완전히 같은 제목이 다른 URL로 들어오는 경우를 줄이기 위함.
    """

    if pd.isna(title):
        return ""

    s = str(title).lower()
    s = pd.Series([s]).str.replace(r"[^a-z0-9\s]", " ", regex=True).iloc[0]
    s = pd.Series([s]).str.replace(r"\s+", " ", regex=True).iloc[0]
    return s.strip()


def clean_raw_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    raw 뉴스 데이터 기본 정리:
    - 필요한 컬럼 없으면 생성
    - published_at datetime 변환
    - 시간 결측 제거
    - title/summary 결측 처리
    - URL 기준 중복 제거
    - title_norm 기준 중복 제거
    - 시간순 정렬
    """

    required_cols = [
        "source",
        "title",
        "summary",
        "url",
        "published_at",
        "origin",
        "entities",
        "symbols",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])

    text_cols = ["source", "title", "summary", "url", "origin", "entities", "symbols"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    before = len(df)

    # 1차: URL 기준 중복 제거
    has_url = df["url"].str.len() > 0
    df_url = df[has_url].drop_duplicates(subset=["url"], keep="first")
    df_no_url = df[~has_url].drop_duplicates(subset=["title", "published_at"], keep="first")
    df = pd.concat([df_url, df_no_url], ignore_index=True)

    after_url = len(df)

    # 2차: 제목 정규화 기준 완전 동일 제목 제거
    # 같은 이슈의 재보도까지 모두 제거하지는 않고, 제목이 거의 완전히 같은 경우만 제거
    df["title_norm"] = df["title"].apply(normalize_title)
    df = df.drop_duplicates(subset=["title_norm"], keep="first")

    after_title = len(df)

    df = df.sort_values("published_at").reset_index(drop=True)

    print("🧹 중복 제거 결과")
    print(f"   원본 rows: {before}")
    print(f"   URL 중복 제거 후 rows: {after_url}")
    print(f"   title_norm 중복 제거 후 rows: {after_title}")
    print(f"   총 제거 rows: {before - after_title}")

    return df


# ============================================================
# 2. FinBERT 입력 텍스트 생성
# ============================================================
def build_text_for_finbert(df: pd.DataFrame) -> list[str]:
    """
    FinBERT 입력은 title + summary를 사용한다.
    문자 기준으로 자르지 않고 tokenizer의 truncation에 맡긴다.
    """

    texts = (
        df["title"].fillna("").astype(str).str.strip()
        + ". "
        + df["summary"].fillna("").astype(str).str.strip()
    )

    texts = texts.str.replace(r"\s+", " ", regex=True).str.strip()

    return texts.tolist()


# ============================================================
# 3. BTC 관련성 feature
# ============================================================
def add_btc_relevance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    설명 AI용 BTC 관련성 점수 생성.
    모델 학습용 9개 feature에는 사용하지 않음.
    """

    text = (
        df["title"].fillna("").astype(str)
        + " "
        + df["summary"].fillna("").astype(str)
        + " "
        + df["entities"].fillna("").astype(str)
        + " "
        + df["symbols"].fillna("").astype(str)
    ).str.lower()

    # 너무 넓히면 노이즈가 늘 수 있으므로 BTC 직접 관련 키워드 중심
    btc_patterns = [
        r"\bbitcoin\b",
        r"\bbtc\b",
        r"bitcoin etf",
        r"spot bitcoin etf",
        r"halving",
    ]

    df["has_bitcoin_keyword"] = text.str.contains(r"\bbitcoin\b", regex=True).astype(int)
    df["has_btc_keyword"] = text.str.contains(r"\bbtc\b", regex=True).astype(int)

    relevance = pd.Series(0.3, index=df.index, dtype=float)

    for pattern in btc_patterns:
        relevance += text.str.contains(pattern, regex=True).astype(float) * 0.2

    df["btc_relevance_score"] = relevance.clip(upper=1.0)

    return df


# ============================================================
# 4. FinBERT 모델 로드
# ============================================================
def init_finbert():
    """
    금융 뉴스 감성 분석용 FinBERT 로드.

    모델:
    - yiyanghkust/finbert-tone

    AutoTokenizer / AutoModel이 로컬 환경에서 config를 자동 인식하지 못하는 문제를 피하기 위해
    BertTokenizer와 BertForSequenceClassification을 직접 사용한다.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🔎 FinBERT 모델 로딩 중...")
    print("   model: yiyanghkust/finbert-tone")
    print(f"   device: {device}")

    model_name = "yiyanghkust/finbert-tone"

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # label 확인
    print("   id2label:", model.config.id2label)

    return tokenizer, model, device


def resolve_label_mapping(model) -> dict:
    """
    모델 config의 id2label을 읽어서 positive/negative/neutral index를 찾는다.
    만약 LABEL_0 형식으로만 나오면 yiyanghkust/finbert-tone의 일반적인 label 순서를 fallback으로 사용한다.

    fallback:
    0: neutral
    1: positive
    2: negative
    """

    id2label = model.config.id2label

    mapping = {
        "positive": None,
        "negative": None,
        "neutral": None,
    }

    for idx, label in id2label.items():
        label_lower = str(label).lower()

        if "positive" in label_lower:
            mapping["positive"] = int(idx)
        elif "negative" in label_lower:
            mapping["negative"] = int(idx)
        elif "neutral" in label_lower:
            mapping["neutral"] = int(idx)

    # config에서 못 찾으면 fallback
    if any(v is None for v in mapping.values()):
        print("⚠️ id2label에서 positive/negative/neutral을 명확히 찾지 못했습니다.")
        print("   fallback mapping 사용: 0=neutral, 1=positive, 2=negative")
        mapping = {
            "neutral": 0,
            "positive": 1,
            "negative": 2,
        }

    print("   resolved label mapping:", mapping)

    return mapping


# ============================================================
# 5. FinBERT 감성 확률 생성
# ============================================================
def run_finbert(df: pd.DataFrame, batch_size: int | None = None) -> pd.DataFrame:
    """
    뉴스별 FinBERT 감성 확률 생성.

    기존 방식:
    - positive면 +score
    - negative면 -score
    - neutral이면 0

    개선 방식:
    - positive / negative / neutral 확률을 직접 계산
    - finbert_score = positive_prob - negative_prob
    - finbert_sq_score = finbert_score ** 2
    """

    tokenizer, model, device = init_finbert()
    label_map = resolve_label_mapping(model)

    texts = build_text_for_finbert(df)

    if batch_size is None:
        batch_size = 128 if torch.cuda.is_available() else 32

    print("🔎 FinBERT 감성 분석 시작")
    print(f"   뉴스 개수: {len(texts)}")
    print(f"   batch_size: {batch_size}")
    print("   truncation: tokenizer 기준 max_length=512")

    pos_probs = []
    neg_probs = []
    neu_probs = []
    labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT"):
            batch = texts[i:i + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

            pos_idx = label_map["positive"]
            neg_idx = label_map["negative"]
            neu_idx = label_map["neutral"]

            batch_pos = probs[:, pos_idx]
            batch_neg = probs[:, neg_idx]
            batch_neu = probs[:, neu_idx]

            pos_probs.extend(batch_pos.tolist())
            neg_probs.extend(batch_neg.tolist())
            neu_probs.extend(batch_neu.tolist())

            pred_idx = probs.argmax(axis=1)

            reverse_map = {
                pos_idx: "positive",
                neg_idx: "negative",
                neu_idx: "neutral",
            }

            labels.extend([reverse_map.get(int(idx), "neutral") for idx in pred_idx])

    df["finbert_label"] = labels

    df["finbert_pos_prob"] = pos_probs
    df["finbert_neg_prob"] = neg_probs
    df["finbert_neu_prob"] = neu_probs

    # 방향 score: 긍정 확률 - 부정 확률
    df["finbert_score"] = df["finbert_pos_prob"] - df["finbert_neg_prob"]

    # 방향 분리
    df["finbert_pos_score"] = df["finbert_pos_prob"]
    df["finbert_neg_score"] = -df["finbert_neg_prob"]

    # 강도 feature
    df["finbert_abs_score"] = df["finbert_score"].abs()
    df["finbert_sq_score"] = df["finbert_score"] ** 2

    # count feature
    df["is_positive"] = (df["finbert_label"] == "positive").astype(int)
    df["is_negative"] = (df["finbert_label"] == "negative").astype(int)
    df["is_neutral"] = (df["finbert_label"] == "neutral").astype(int)

    return df


# ============================================================
# 6. 설명 AI용 데이터 생성
# ============================================================
def build_explain_context(df: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """
    설명 AI용 데이터 생성.

    설명 AI용은 기사 단위 정보를 보존한다.

    available_hour:
    모델 feature에 실제로 반영되는 시간.
    예:
    10:37 뉴스, lag_hours=1
    → published_hour = 10:00
    → available_hour = 11:00
    """

    explain = df.copy()

    explain["published_hour"] = explain["published_at"].dt.floor("h")
    explain["available_hour"] = explain["published_hour"] + pd.Timedelta(hours=lag_hours)

    # 설명 AI용 중요도 점수
    # BTC 관련성이 높고, 감성 강도가 강할수록 중요하게 본다.
    explain["importance_score"] = (
        explain["btc_relevance_score"] * (0.5 + 0.5 * explain["finbert_sq_score"])
    )

    final_cols = [
        "published_at",
        "published_hour",
        "available_hour",

        "source",
        "origin",
        "title",
        "summary",
        "url",
        "entities",
        "symbols",

        "finbert_label",
        "finbert_pos_prob",
        "finbert_neg_prob",
        "finbert_neu_prob",
        "finbert_score",
        "finbert_abs_score",
        "finbert_sq_score",
        "finbert_pos_score",
        "finbert_neg_score",

        "is_positive",
        "is_negative",
        "is_neutral",

        "has_bitcoin_keyword",
        "has_btc_keyword",
        "btc_relevance_score",
        "importance_score",
    ]

    explain = explain[final_cols]
    explain = explain.sort_values("published_at").reset_index(drop=True)

    return explain


# ============================================================
# 7. 모델 학습용 1시간 feature 생성
# ============================================================
def build_hourly_features(
    explain: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    모델 학습용 1시간 단위 뉴스 feature 생성.

    기준 시간:
    - published_at이 아니라 available_hour 기준으로 집계
    - 그래야 미래 뉴스가 현재 입력에 섞이지 않음

    출력:
    - 1시간마다 한 row
    - 뉴스 없는 시간은 0으로 채움
    """

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    df = explain.copy()
    df["hour"] = pd.to_datetime(df["available_hour"], utc=True, errors="coerce")

    # 1시간 단위 집계
    grouped = df.groupby("hour").agg(
        news_count=("finbert_score", "count"),

        # 현재 시간 감성 방향/강도
        finbert_mean=("finbert_score", "mean"),
        finbert_sq_mean=("finbert_sq_score", "mean"),

        # 긍정/부정 감성 총량
        # pos는 양수, neg는 음수로 유지
        finbert_pos_sum=("finbert_pos_score", "sum"),
        finbert_neg_sum=("finbert_neg_score", "sum"),

        # 긍정/부정 개수
        positive_count=("is_positive", "sum"),
        negative_count=("is_negative", "sum"),
    ).reset_index()

    # 전체 1시간 grid 생성
    # end는 포함하지 않기 위해 end - 1시간까지 생성
    full_hours = pd.date_range(
        start=start_ts,
        end=end_ts - pd.Timedelta(hours=1),
        freq="h",
        tz="UTC",
    )

    grid = pd.DataFrame({"hour": full_hours})

    hourly = grid.merge(grouped, on="hour", how="left")

    fill_zero_cols = [
        "news_count",
        "finbert_mean",
        "finbert_sq_mean",
        "finbert_pos_sum",
        "finbert_neg_sum",
        "positive_count",
        "negative_count",
    ]

    for col in fill_zero_cols:
        hourly[col] = hourly[col].fillna(0.0)

    # 뉴스량 feature
    hourly["news_presence"] = (hourly["news_count"] > 0).astype(float)
    hourly["news_count_log1p"] = np.log1p(hourly["news_count"])

    # 긍정/부정 개수 기반 방향성
    hourly["pos_neg_count_imbalance"] = np.where(
        hourly["news_count"] > 0,
        (hourly["positive_count"] - hourly["negative_count"]) / hourly["news_count"],
        0.0,
    )

    # 최근 24시간 rolling feature
    # 중요 보강:
    # finbert_mean_ma_24h는 뉴스 없는 시간의 0을 평균에 넣지 않고,
    # 뉴스가 있었던 시간만 기준으로 평균을 계산한다.
    hourly["finbert_mean_news_only"] = hourly["finbert_mean"].where(
        hourly["news_presence"] == 1,
        np.nan,
    )

    hourly["finbert_mean_ma_24h"] = (
        hourly["finbert_mean_news_only"]
        .rolling(window=24, min_periods=1)
        .mean()
        .fillna(0.0)
    )

    # 뉴스량은 0도 의미가 있으므로 0 포함 sum이 맞다.
    hourly["news_count_sum_24h"] = (
        hourly["news_count"]
        .rolling(window=24, min_periods=1)
        .sum()
    )

    # 최종 모델용 NEWS_FEATURES 9개
    final_cols = ["hour"] + NEWS_FEATURES

    hourly = hourly[final_cols]

    feature_cols = [col for col in hourly.columns if col != "hour"]
    hourly[feature_cols] = hourly[feature_cols].fillna(0.0)

    return hourly


# ============================================================
# 8. 메타데이터 저장
# ============================================================
def save_metadata(
    raw_df: pd.DataFrame,
    explain_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    args,
):
    meta = {
        "input_file": str(args.raw),
        "explain_output_file": str(EXPLAIN_OUT),
        "hourly_output_file": str(HOURLY_OUT),
        "start": args.start,
        "end": args.end,
        "lag_hours": args.lag_hours,

        "raw_rows_after_cleaning": int(len(raw_df)),
        "explain_rows": int(len(explain_df)),
        "hourly_rows": int(len(hourly_df)),

        "raw_time_range": [
            str(raw_df["published_at"].min()) if len(raw_df) else None,
            str(raw_df["published_at"].max()) if len(raw_df) else None,
        ],

        "hourly_time_range": [
            str(hourly_df["hour"].min()) if len(hourly_df) else None,
            str(hourly_df["hour"].max()) if len(hourly_df) else None,
        ],

        "model_feature_columns": NEWS_FEATURES,
        "feature_count": len(NEWS_FEATURES),

        "finbert_model": "yiyanghkust/finbert-tone",

        "note": (
            "news_hourly_features.csv is for model input. "
            "news_explain_context.csv is for explanation AI. "
            "FinBERT model is yiyanghkust/finbert-tone. "
            "FinBERT is called directly with tokenizer truncation max_length=512. "
            "finbert_score = positive_probability - negative_probability. "
            "Model uses finbert_sq_mean as sentiment intensity. "
            "Hourly aggregation is based on available_hour to avoid future leakage. "
            "finbert_mean_ma_24h is calculated only over hours with news_presence=1. "
            "Scaling is not applied here to avoid data leakage."
        ),
    }

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"🧾 metadata 저장 완료: {META_OUT}")


# ============================================================
# 9. main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw",
        default=str(RAW_IN),
        help="raw 뉴스 CSV 경로"
    )

    parser.add_argument(
        "--start",
        default="2023-05-01 00:00",
        help="1시간 feature 시작 시각 UTC"
    )

    parser.add_argument(
        "--end",
        default="2026-05-01 00:00",
        help="1시간 feature 종료 시각 UTC. 해당 시각은 포함하지 않음"
    )

    parser.add_argument(
        "--lag-hours",
        type=int,
        default=1,
        help="뉴스가 모델 feature에 반영되기까지의 지연 시간. 기본값 1시간"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="FinBERT batch size. 미지정 시 GPU 128, CPU 32"
    )

    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 뉴스 전처리 시작")
    print(f"raw 입력 파일: {raw_path}")
    print(f"시작 시각: {args.start}")
    print(f"종료 시각: {args.end}")
    print(f"lag_hours: {args.lag_hours}")

    if not raw_path.exists():
        raise FileNotFoundError(f"raw 뉴스 파일을 찾을 수 없습니다: {raw_path}")

    # 1. raw 로드
    raw_df = pd.read_csv(raw_path)
    print(f"📥 raw 로드 완료: {len(raw_df)} rows")

    # 2. 기본 정리 + 중복 제거
    raw_df = clean_raw_news(raw_df)
    print(f"🧹 기본 정리 완료: {len(raw_df)} rows")
    print(f"   기간: {raw_df['published_at'].min()} ~ {raw_df['published_at'].max()}")

    # 3. BTC 관련성 feature
    raw_df = add_btc_relevance_features(raw_df)

    # 4. FinBERT 감성 분석
    scored_df = run_finbert(raw_df, batch_size=args.batch_size)

    # 5. 설명 AI용 데이터 생성
    explain_df = build_explain_context(scored_df, lag_hours=args.lag_hours)

    # 6. 모델 학습용 1시간 feature 생성
    hourly_df = build_hourly_features(
        explain=explain_df,
        start=args.start,
        end=args.end,
    )

    # 7. 저장
    explain_df.to_csv(EXPLAIN_OUT, index=False, encoding="utf-8-sig")
    hourly_df.to_csv(HOURLY_OUT, index=False, encoding="utf-8-sig")

    print(f"✅ 설명 AI용 데이터 저장 완료: {EXPLAIN_OUT}")
    print(f"   rows: {len(explain_df)}")

    print(f"✅ 모델 학습용 1시간 feature 저장 완료: {HOURLY_OUT}")
    print(f"   rows: {len(hourly_df)}")
    print(f"   기간: {hourly_df['hour'].min()} ~ {hourly_df['hour'].max()}")
    print(f"   NEWS_FEATURES: {NEWS_FEATURES}")

    # 8. 메타데이터 저장
    save_metadata(raw_df, explain_df, hourly_df, args)

    print("\n✨ 뉴스 전처리 완료")

    print("\n[모델용 feature 미리보기]")
    print(hourly_df.head(5).to_string(index=False))

    print("\n[설명용 뉴스 미리보기]")
    preview_cols = [
        "published_at",
        "available_hour",
        "title",
        "finbert_label",
        "finbert_pos_prob",
        "finbert_neg_prob",
        "finbert_neu_prob",
        "finbert_score",
        "finbert_sq_score",
        "importance_score",
    ]
    print(explain_df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()