#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_marketaux_news.py

역할:
- MarketAux API로 비트코인 관련 뉴스 원본 데이터만 수집
- 전처리, FinBERT, 1시간 집계, train/test split 안 함
- 중간 저장 지원
- 429 rate limit 발생 시 지금까지 저장한 뒤 안전 종료
- 다음 실행 시 완료된 chunk는 건너뜀

api_key.txt 형식:
marketaux_api_key=여기에_키_값

저장 결과:
data/news/raw/news_raw_marketaux.csv
data/news/raw/news_fetch_progress.json
"""

import sys
import time
import json
import random
import argparse
from pathlib import Path

import requests
import pandas as pd
from tqdm.auto import tqdm


# ============================================================
# 프로젝트 경로 설정
# 현재 파일 위치:
# TRADING/data/news/raw/fetch_marketaux_news.py
# parents[3] = TRADING
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

API_KEY_PATH = PROJECT_ROOT / "api_key.txt"
DEFAULT_RAW_OUT = PROJECT_ROOT / "data/news/raw/news_raw_marketaux.csv"
DEFAULT_PROGRESS_OUT = PROJECT_ROOT / "data/news/raw/news_fetch_progress.json"


# ============================================================
# requests / tqdm 설정
# ============================================================
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "fetch-marketaux-bitcoin-news/1.0 (+research use)"
})

TQDM_KW = dict(
    dynamic_ncols=True,
    ncols=100,
    mininterval=0.3,
    file=sys.stdout,
    disable=False,
)


# ============================================================
# API Key 읽기
# ============================================================
def load_marketaux_api_key(path: Path = API_KEY_PATH) -> str:
    """
    api_key.txt에서 marketaux_api_key= 뒤의 값을 읽는다.

    예:
    marketaux_api_key=abc123
    """

    if not path.exists():
        raise FileNotFoundError(
            f"API key 파일을 찾을 수 없습니다: {path}\n"
            f"프로젝트 최상위 폴더에 api_key.txt를 만들고 "
            f"marketaux_api_key=키값 형태로 저장하세요."
        )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("marketaux_api_key="):
                key = line.split("=", 1)[1].strip()

                if not key:
                    raise ValueError("marketaux_api_key= 뒤에 API 키 값이 비어 있습니다.")

                return key

    raise ValueError(
        "api_key.txt 안에서 marketaux_api_key= 형식을 찾지 못했습니다.\n"
        "예: marketaux_api_key=여기에_키_값"
    )


# ============================================================
# 시간 처리 함수
# ============================================================
def to_utc_ts(value: str) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def iso(dt: pd.Timestamp) -> str:
    """
    MarketAux API에 넣을 ISO 문자열로 변환
    """
    d = pd.to_datetime(dt, utc=True)
    d = d.tz_convert(None)
    return d.strftime("%Y-%m-%dT%H:%M:%S")


def daterange_utc(start: pd.Timestamp, end: pd.Timestamp, hours: int):
    """
    start ~ end 기간을 hours 단위로 쪼갠다.
    """
    cur = start

    while cur < end:
        nxt = min(cur + pd.Timedelta(hours=hours), end)
        yield cur, nxt
        cur = nxt


def chunk_id(a: pd.Timestamp, b: pd.Timestamp) -> str:
    """
    progress.json에 저장할 chunk 식별자
    """
    return f"{iso(a)}__{iso(b)}"


# ============================================================
# progress 저장 / 로드
# ============================================================
def load_progress(path: Path) -> dict:
    if not path.exists():
        return {"completed_chunks": []}

    try:
        with open(path, "r", encoding="utf-8") as f:
            progress = json.load(f)

        if "completed_chunks" not in progress:
            progress["completed_chunks"] = []

        return progress

    except Exception as e:
        print(f"⚠️ progress 파일 읽기 실패: {e}")
        return {"completed_chunks": []}


def save_progress(progress: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ============================================================
# 기존 CSV 로드 / 저장
# ============================================================
def load_existing_csv(path: Path) -> pd.DataFrame:
    """
    이미 저장된 raw CSV가 있으면 읽어온다.
    없으면 빈 DataFrame 반환.
    """

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

        print(f"♻️ 기존 파일 로드: {path}")
        print(f"   기존 뉴스 개수: {len(df)}")

        return df

    except Exception as e:
        print(f"⚠️ 기존 CSV를 읽는 중 오류 발생: {e}")
        print("   안전을 위해 새 DataFrame으로 시작합니다.")
        return pd.DataFrame()


def clean_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    published_at 정리 + 중복 제거.
    URL이 있으면 URL 기준 중복 제거.
    URL이 없으면 title + published_at 기준 중복 제거.
    """

    if df.empty:
        return df

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_at"])
        df = df.sort_values("published_at")

    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")
    elif {"title", "published_at"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["title", "published_at"], keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    df = df.reset_index(drop=True)

    return df


def save_csv_safely(df: pd.DataFrame, path: Path):
    """
    CSV 저장.
    폴더가 없으면 생성.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    df = clean_and_deduplicate(df)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"💾 저장 완료: {path}")
    print(f"   저장된 뉴스 개수: {len(df)}")

    if not df.empty and "published_at" in df.columns:
        print(f"   저장 범위: {df['published_at'].min()} ~ {df['published_at'].max()}")


# ============================================================
# HTTP 요청 함수
# ============================================================
def get_json(url: str, params: dict, tries: int = 3):
    """
    MarketAux API 요청.

    반환:
    - 성공: dict
    - rate limit: "RATE_LIMIT"
    - 기타 실패: {}
    """

    last_status = None
    last_text = None

    for k in range(tries):
        try:
            response = SESSION.get(url, params=params, timeout=30)
            last_status = response.status_code
            last_text = response.text[:500]

            if response.status_code == 429:
                wait_sec = 2 * (k + 1) + random.random()
                print(f"[MarketAux] 429 rate limit 발생. {wait_sec:.1f}초 후 재시도...", flush=True)
                time.sleep(wait_sec)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            print(f"[MarketAux] HTTP 에러 발생 ({k + 1}/{tries}): {e}", flush=True)

            if last_status is not None:
                print(f"[MarketAux] status={last_status}, body={last_text}", flush=True)

            wait_sec = 2 * (k + 1) + random.random()
            time.sleep(wait_sec)

    if last_status == 429:
        print("[MarketAux] rate limit으로 인해 수집을 중단합니다.", flush=True)
        print(f"status={last_status}", flush=True)
        print(f"body={last_text}", flush=True)
        return "RATE_LIMIT"

    print("[MarketAux] 모든 재시도 실패", flush=True)

    if last_status is not None:
        print(f"status={last_status}", flush=True)
        print(f"body={last_text}", flush=True)

    return {}


# ============================================================
# MarketAux 뉴스 수집
# ============================================================
def fetch_marketaux_news_safe(
    token: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out_path: Path,
    progress_path: Path,
    query: str = "(bitcoin OR BTC)",
    language: str = "en",
    chunk_hours: int = 168,
    limit: int = 20,
    sleep: float = 0.5,
    save_every_chunks: int = 1,
    max_pages_per_chunk: int = 5,
) -> pd.DataFrame:
    """
    안전 수집 함수.

    핵심:
    - 기존 CSV를 먼저 읽는다.
    - 이미 완료된 chunk는 progress.json 기준으로 건너뛴다.
    - chunk 하나 처리할 때마다 저장한다.
    - 429가 나오면 지금까지 받은 것 저장하고 종료한다.
    - 다음 실행하면 완료된 구간은 skip한다.
    """

    base_url = "https://api.marketaux.com/v1/news/all"

    existing_df = load_existing_csv(out_path)
    all_rows = []

    if not existing_df.empty:
        all_rows.extend(existing_df.to_dict("records"))

    progress = load_progress(progress_path)
    completed_chunks = set(progress.get("completed_chunks", []))

    ranges = list(daterange_utc(start, end, hours=chunk_hours))

    print("🚀 MarketAux 비트코인 뉴스 raw 데이터 안전 수집 시작", flush=True)
    print(f"수집 기간: {start} ~ {end}", flush=True)
    print(f"검색어: {query}", flush=True)
    print(f"요청 구간 단위: {chunk_hours}시간", flush=True)
    print(f"limit: {limit}", flush=True)
    print(f"max_pages_per_chunk: {max_pages_per_chunk}", flush=True)
    print(f"CSV 저장 경로: {out_path}", flush=True)
    print(f"진행상황 저장 경로: {progress_path}", flush=True)
    print(f"이미 완료된 chunk 수: {len(completed_chunks)}", flush=True)

    processed_chunks = 0

    try:
        for a, b in tqdm(ranges, desc="MarketAux BTC 뉴스 수집", **TQDM_KW):
            cid = chunk_id(a, b)

            if cid in completed_chunks:
                continue

            chunk_rows = []
            page = 1

            while True:
                if page > max_pages_per_chunk:
                    print(
                        f"⚠️ {a} ~ {b} 구간에서 page {max_pages_per_chunk}까지만 수집하고 다음 구간으로 넘어갑니다.",
                        flush=True
                    )
                    break

                params = {
                    "api_token": token,
                    "search": query,
                    "language": language,
                    "published_after": iso(a),
                    "published_before": iso(b),
                    "group_similar": "true",
                    "limit": limit,
                    "page": page,
                }

                js = get_json(base_url, params)

                if js == "RATE_LIMIT":
                    print("🛑 Rate limit에 걸렸습니다. 지금까지 받은 데이터를 저장하고 종료합니다.", flush=True)

                    if chunk_rows:
                        all_rows.extend(chunk_rows)

                    current_df = pd.DataFrame(all_rows)
                    save_csv_safely(current_df, out_path)
                    save_progress(progress, progress_path)

                    return clean_and_deduplicate(current_df)

                data = js.get("data", js.get("news", [])) if isinstance(js, dict) else []
                data = data or []

                if not data:
                    break

                for item in data:
                    chunk_rows.append({
                        "source": "marketaux",
                        "title": item.get("title"),
                        "summary": item.get("description") or item.get("snippet"),
                        "url": item.get("url"),
                        "published_at": item.get("published_at"),
                        "origin": item.get("source"),
                        "entities": item.get("entities"),
                        "symbols": item.get("symbols"),
                    })

                if len(data) < limit:
                    break

                page += 1
                time.sleep(sleep)

            if chunk_rows:
                all_rows.extend(chunk_rows)

            # 여기까지 왔다는 건 이 chunk는 정상 처리 완료
            completed_chunks.add(cid)
            progress["completed_chunks"] = sorted(list(completed_chunks))
            progress["last_completed_chunk"] = cid
            progress["last_completed_at"] = pd.Timestamp.utcnow().isoformat()

            processed_chunks += 1

            if processed_chunks % save_every_chunks == 0:
                current_df = pd.DataFrame(all_rows)
                save_csv_safely(current_df, out_path)
                save_progress(progress, progress_path)

            time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n🛑 사용자가 실행을 중단했습니다. 지금까지 받은 데이터를 저장합니다.", flush=True)

        current_df = pd.DataFrame(all_rows)
        save_csv_safely(current_df, out_path)
        save_progress(progress, progress_path)

        return clean_and_deduplicate(current_df)

    final_df = pd.DataFrame(all_rows)
    save_csv_safely(final_df, out_path)
    save_progress(progress, progress_path)

    return clean_and_deduplicate(final_df)


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2023-05-01 00:00",
        help="뉴스 수집 시작 시각 UTC. 예: 2023-05-01 00:00"
    )

    parser.add_argument(
        "--end",
        default="2026-05-01 00:00",
        help="뉴스 수집 종료 시각 UTC. 예: 2026-05-01 00:00"
    )

    parser.add_argument(
        "--query",
        default="(bitcoin OR BTC)",
        help="MarketAux 검색어. 비트코인 전용 기본값: (bitcoin OR BTC)"
    )

    parser.add_argument(
        "--language",
        default="en",
        help="뉴스 언어. 기본값 en"
    )

    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=168,
        help="API 요청을 몇 시간 단위로 쪼갤지 설정. 기본값 168시간 = 7일"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="한 페이지당 가져올 뉴스 수. Basic 플랜이면 보통 20 권장"
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="요청 사이 대기 시간"
    )

    parser.add_argument(
        "--save-every-chunks",
        type=int,
        default=1,
        help="몇 개 chunk마다 저장할지 설정. 기본값 1이면 매 chunk마다 저장"
    )

    parser.add_argument(
        "--max-pages-per-chunk",
        type=int,
        default=5,
        help="한 chunk에서 최대 몇 page까지 가져올지 설정. 요청 폭주 방지용"
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_RAW_OUT),
        help="저장할 CSV 경로"
    )

    parser.add_argument(
        "--progress-out",
        default=str(DEFAULT_PROGRESS_OUT),
        help="진행상황 JSON 저장 경로"
    )

    args = parser.parse_args()

    token = load_marketaux_api_key(API_KEY_PATH)

    start = to_utc_ts(args.start)
    end = to_utc_ts(args.end)

    out_path = Path(args.out)
    progress_path = Path(args.progress_out)

    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    if not progress_path.is_absolute():
        progress_path = PROJECT_ROOT / progress_path

    df = fetch_marketaux_news_safe(
        token=token,
        start=start,
        end=end,
        out_path=out_path,
        progress_path=progress_path,
        query=args.query,
        language=args.language,
        chunk_hours=args.chunk_hours,
        limit=args.limit,
        sleep=args.sleep,
        save_every_chunks=args.save_every_chunks,
        max_pages_per_chunk=args.max_pages_per_chunk,
    )

    print("\n✅ 수집 프로세스 종료", flush=True)
    print(f"최종 저장 파일: {out_path}", flush=True)
    print(f"최종 뉴스 개수: {len(df)}", flush=True)

    if not df.empty:
        print(f"최종 기간: {df['published_at'].min()} ~ {df['published_at'].max()}", flush=True)

        print("\n미리보기:")
        preview_cols = [col for col in ["published_at", "title", "url"] if col in df.columns]
        print(df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()