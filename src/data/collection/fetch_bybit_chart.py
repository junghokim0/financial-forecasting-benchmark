#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_bybit_chart.py

Bybit BTCUSDT 선물 1시간봉 차트 데이터 수집 코드

저장 위치:
Desktop/trading/data/chart/raw/chart_raw_bybit.csv

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


def fetch_chunk(base, category, symbol, interval, start_ms, end_ms, limit=1000):
    """
    Bybit v5 /market/kline 호출
    정상: (list, 0)
    오류: ([], retCode)
    """
    url = f"{base}/v5/market/kline"

    params = {
        "category": category,
        "symbol": symbol,
        "interval": str(interval),
        "start": start_ms,
        "end": end_ms,
        "limit": limit,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return [], -1

    ret_code = data.get("retCode", -2)

    if ret_code != 0:
        return [], ret_code

    return data["result"].get("list", []), 0


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2023-05-01")
    ap.add_argument("--end", default="2026-05-01")
    ap.add_argument("--interval", type=int, default=60)  # 60분 = 1시간봉
    ap.add_argument("--category", default="linear", choices=["linear", "spot", "inverse", "option"])
    ap.add_argument("--base", default=BYBIT_BASE)

    ap.add_argument(
        "--out",
        default=os.path.expanduser(
            "~/Desktop/trading/data/chart/raw/chart_raw_bybit.csv"
        ),
    )

    ap.add_argument("--sleep-base", type=float, default=0.5)
    ap.add_argument("--max-backoff", type=float, default=20.0)
    ap.add_argument("--checkpoint-every", type=int, default=5000)

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)

    cursor = start_ms
    rows = []

    total_minutes = (end_ms - start_ms) // (1000 * 60)
    total_candles = total_minutes // args.interval
    total_chunks = (total_candles + 999) // 1000

    if total_chunks <= 0:
        total_chunks = 1

    print(f"[Bybit Chart Fetch]")
    print(f"symbol   : {args.symbol}")
    print(f"category : {args.category}")
    print(f"interval : {args.interval}m")
    print(f"period   : {args.start} → {args.end}")
    print(f"output   : {args.out}")

    pbar = tqdm(total=total_chunks, desc="Fetching chunks", ncols=100)

    backoff = 1.0

    while cursor < end_ms:
        chunk_end = min(
            cursor + 1000 * 60 * args.interval,
            end_ms
        )

        data_list, ret_code = fetch_chunk(
            base=args.base,
            category=args.category,
            symbol=args.symbol,
            interval=args.interval,
            start_ms=cursor,
            end_ms=chunk_end,
        )

        if ret_code != 0 or not data_list:
            tag = "rate_limit" if ret_code == 10006 else f"err{ret_code}"
            pbar.set_postfix_str(tag)

            time.sleep(backoff)
            backoff = min(backoff * 2.0, args.max_backoff)
            continue

        backoff = 1.0

        data_list.sort(key=lambda x: int(x[0]))

        for item in data_list:
            ts, open_, high, low, close, volume, turnover = item[:7]

            rows.append({
                "ts": iso_utc_from_ms(int(ts)),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
                "turnover": float(turnover),
                "symbol": args.symbol,
                "interval": str(args.interval),
                "category": args.category,
            })

        cursor = int(data_list[-1][0]) + args.interval * 60 * 1000

        pbar.set_postfix_str(f"last={rows[-1]['ts']}")
        pbar.update(1)

        if args.checkpoint_every and len(rows) % args.checkpoint_every == 0:
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(args.out, index=False)

        time.sleep(args.sleep_base + random.random() * 0.2)

    pbar.close()

    if not rows:
        print("수집된 데이터가 없습니다. 기간, 심볼, category를 확인하세요.")
        return

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    diff = df["ts"].diff().dropna()
    irregular_count = (diff != pd.Timedelta(hours=1)).sum()

    df.to_csv(args.out, index=False)

    print("\n✅ Bybit chart data saved")
    print(f"path  : {args.out}")
    print(f"rows  : {len(df)}")
    print(f"range : {df['ts'].iloc[0]} → {df['ts'].iloc[-1]}")
    print(f"irregular intervals: {irregular_count}")
    print("\ncolumns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()