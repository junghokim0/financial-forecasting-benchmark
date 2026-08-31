#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_bybit_derivatives.py

Bybit BTCUSDT derivatives raw data 수집 코드

수집:
1. funding_rate
2. open_interest

저장 위치:
Desktop/trading12/trading/data/chart/raw/bybit_funding_rate_raw.csv
Desktop/trading12/trading/data/chart/raw/bybit_open_interest_raw.csv

기본 수집 기간:
2023-05-01 ~ 2026-05-01
"""

import os
import time
import argparse
import random
import requests
import pandas as pd

from datetime import datetime, UTC
from dateutil import parser as dparser
from tqdm import tqdm


BYBIT_BASE = os.getenv("BYBIT_BASE", "https://api.bybit.com")


def to_ms(ts: str) -> int:
    """YYYY-MM-DD 또는 ISO 문자열을 UTC epoch millisecond로 변환"""
    return int(dparser.parse(ts).astimezone(UTC).timestamp() * 1000)


def iso_utc_from_ms(ms: int) -> str:
    """epoch millisecond를 UTC ISO 문자열로 변환"""
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


def safe_get_json(url, params, timeout=30):
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None, -1

    ret_code = data.get("retCode", -2)
    if ret_code != 0:
        return data, ret_code

    return data, 0


# ======================================================
# Funding Rate
# ======================================================
def fetch_funding_chunk(base, category, symbol, start_ms, end_ms, limit=200):
    """
    Bybit v5 /market/funding/history 호출
    정상: (list, 0)
    오류: ([], retCode)
    """
    url = f"{base}/v5/market/funding/history"

    params = {
        "category": category,
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    data, ret_code = safe_get_json(url, params)

    if ret_code != 0 or data is None:
        return [], ret_code

    return data.get("result", {}).get("list", []), 0


def collect_funding_rate(args):
    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)

    # funding은 일반적으로 8시간 단위.
    # limit=200 기준 약 66일치씩 요청.
    chunk_ms = args.funding_chunk_hours * 60 * 60 * 1000
    total_chunks = max(1, (end_ms - start_ms + chunk_ms - 1) // chunk_ms)

    cursor = start_ms
    rows = []
    backoff = 1.0

    print("\n[Bybit Funding Rate Fetch]")
    print(f"symbol   : {args.symbol}")
    print(f"category : {args.category}")
    print(f"period   : {args.start} → {args.end}")
    print(f"output   : {args.funding_out}")

    pbar = tqdm(total=total_chunks, desc="Fetching funding", ncols=100)

    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)

        data_list, ret_code = fetch_funding_chunk(
            base=args.base,
            category=args.category,
            symbol=args.symbol,
            start_ms=cursor,
            end_ms=chunk_end,
            limit=args.funding_limit,
        )

        if ret_code != 0:
            tag = "rate_limit" if ret_code == 10006 else f"err{ret_code}"
            pbar.set_postfix_str(tag)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, args.max_backoff)
            continue

        backoff = 1.0

        if data_list:
            data_list.sort(key=lambda x: int(x["fundingRateTimestamp"]))

            for item in data_list:
                ts_ms = int(item["fundingRateTimestamp"])

                rows.append({
                    "ts": iso_utc_from_ms(ts_ms),
                    "funding_rate": float(item["fundingRate"]),
                    "symbol": args.symbol,
                    "category": args.category,
                })

            pbar.set_postfix_str(f"last={rows[-1]['ts']}")

        cursor = chunk_end + 1
        pbar.update(1)

        time.sleep(args.sleep_base + random.random() * 0.2)

    pbar.close()

    if not rows:
        print("⚠️ funding_rate 수집 데이터가 없습니다.")
        return pd.DataFrame(columns=["ts", "funding_rate", "symbol", "category"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df.to_csv(args.funding_out, index=False, encoding="utf-8-sig")

    print("\n✅ Bybit funding rate saved")
    print(f"path  : {args.funding_out}")
    print(f"rows  : {len(df)}")
    print(f"range : {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}")
    print("columns:", list(df.columns))

    return df


# ======================================================
# Open Interest
# ======================================================
def fetch_open_interest_chunk(base, category, symbol, interval_time, start_ms, end_ms, limit=200):
    """
    Bybit v5 /market/open-interest 호출
    정상: (list, 0)
    오류: ([], retCode)
    """
    url = f"{base}/v5/market/open-interest"

    params = {
        "category": category,
        "symbol": symbol,
        "intervalTime": interval_time,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    data, ret_code = safe_get_json(url, params)

    if ret_code != 0 or data is None:
        return [], ret_code

    return data.get("result", {}).get("list", []), 0


def collect_open_interest(args):
    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)

    # 1h * 200개 = 약 8.3일치씩 요청
    chunk_ms = args.oi_chunk_hours * 60 * 60 * 1000
    total_chunks = max(1, (end_ms - start_ms + chunk_ms - 1) // chunk_ms)

    cursor = start_ms
    rows = []
    backoff = 1.0

    print("\n[Bybit Open Interest Fetch]")
    print(f"symbol        : {args.symbol}")
    print(f"category      : {args.category}")
    print(f"interval_time : {args.oi_interval_time}")
    print(f"period        : {args.start} → {args.end}")
    print(f"output        : {args.open_interest_out}")

    pbar = tqdm(total=total_chunks, desc="Fetching open interest", ncols=100)

    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)

        data_list, ret_code = fetch_open_interest_chunk(
            base=args.base,
            category=args.category,
            symbol=args.symbol,
            interval_time=args.oi_interval_time,
            start_ms=cursor,
            end_ms=chunk_end,
            limit=args.oi_limit,
        )

        if ret_code != 0:
            tag = "rate_limit" if ret_code == 10006 else f"err{ret_code}"
            pbar.set_postfix_str(tag)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, args.max_backoff)
            continue

        backoff = 1.0

        if data_list:
            # Bybit 응답 필드: openInterest, timestamp
            data_list.sort(key=lambda x: int(x["timestamp"]))

            for item in data_list:
                ts_ms = int(item["timestamp"])

                rows.append({
                    "ts": iso_utc_from_ms(ts_ms),
                    "open_interest": float(item["openInterest"]),
                    "symbol": args.symbol,
                    "category": args.category,
                    "interval_time": args.oi_interval_time,
                })

            pbar.set_postfix_str(f"last={rows[-1]['ts']}")

        cursor = chunk_end + 1
        pbar.update(1)

        time.sleep(args.sleep_base + random.random() * 0.2)

    pbar.close()

    if not rows:
        print("⚠️ open_interest 수집 데이터가 없습니다.")
        return pd.DataFrame(columns=["ts", "open_interest", "symbol", "category", "interval_time"])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df.to_csv(args.open_interest_out, index=False, encoding="utf-8-sig")

    diff = df["ts"].diff().dropna()
    irregular_count = int((diff != pd.Timedelta(hours=1)).sum())

    print("\n✅ Bybit open interest saved")
    print(f"path  : {args.open_interest_out}")
    print(f"rows  : {len(df)}")
    print(f"range : {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}")
    print(f"irregular 1h intervals: {irregular_count}")
    print("columns:", list(df.columns))

    return df


# ======================================================
# Main
# ======================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2023-05-01")
    ap.add_argument("--end", default="2026-05-01")
    ap.add_argument("--category", default="linear", choices=["linear", "inverse"])
    ap.add_argument("--base", default=BYBIT_BASE)

    ap.add_argument(
        "--funding-out",
        default=os.path.expanduser(
            "~/Desktop/trading12/trading/data/chart/raw/bybit_funding_rate_raw.csv"
        ),
    )

    ap.add_argument(
        "--open-interest-out",
        default=os.path.expanduser(
            "~/Desktop/trading12/trading/data/chart/raw/bybit_open_interest_raw.csv"
        ),
    )

    # Bybit API limit은 보수적으로 200 사용
    ap.add_argument("--funding-limit", type=int, default=200)
    ap.add_argument("--oi-limit", type=int, default=200)

    # funding: 8h * 200 = 1600h
    ap.add_argument("--funding-chunk-hours", type=int, default=1600)

    # open interest: 1h * 200 = 200h
    ap.add_argument("--oi-chunk-hours", type=int, default=200)
    ap.add_argument("--oi-interval-time", default="1h")

    ap.add_argument("--sleep-base", type=float, default=0.5)
    ap.add_argument("--max-backoff", type=float, default=20.0)

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.funding_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.open_interest_out), exist_ok=True)

    print("[Bybit Derivatives Fetch]")
    print(f"symbol   : {args.symbol}")
    print(f"category : {args.category}")
    print(f"period   : {args.start} → {args.end}")
    print(f"base     : {args.base}")

    funding_df = collect_funding_rate(args)
    oi_df = collect_open_interest(args)

    print("\n" + "=" * 80)
    print("📌 Bybit derivatives raw fetch summary")
    print("=" * 80)
    print("funding rows      :", len(funding_df))
    print("open interest rows:", len(oi_df))

    if len(funding_df) > 0:
        print("funding range      :", funding_df["ts"].iloc[0], "→", funding_df["ts"].iloc[-1])

    if len(oi_df) > 0:
        print("open interest range:", oi_df["ts"].iloc[0], "→", oi_df["ts"].iloc[-1])

    print("\n✅ Done")


if __name__ == "__main__":
    main()