import pandas as pd
from pathlib import Path

# =========================
# 설정
# =========================
INPUT_PATH = Path("data/merged/merged_hourly_features.csv")
#INPUT_PATH = Path("data/merged/merged_hourly_features_deriv14.csv")
#INPUT_PATH = Path("data/merged/merged_hourly_features_funding13.csv")

OUTPUT_PATH = Path("data/merged/merged_with_future_return.csv")
#OUTPUT_PATH = Path("data/merged/merged_with_future_return_deriv14.csv")
#OUTPUT_PATH = Path("data/merged/merged_with_future_return_funding13.csv")

PREDICTION_HORIZON = 24  # 24시간 뒤 수익률


# =========================
# 1. 데이터 로드
# =========================
df = pd.read_csv(INPUT_PATH)

time_col = "hour" if "hour" in df.columns else "timestamp"

df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
df = df.sort_values(time_col).reset_index(drop=True)

print("===== Load Check =====")
print("rows:", len(df))
print("time range:", df[time_col].min(), "~", df[time_col].max())


# =========================
# 2. 필수 컬럼 확인
# =========================
if "close" not in df.columns:
    raise ValueError("'close' 컬럼이 없습니다.")

if df[time_col].isna().sum() > 0:
    raise ValueError("시간 컬럼에 NaN이 있습니다.")

if df["close"].isna().sum() > 0:
    raise ValueError("close 컬럼에 NaN이 있습니다.")


# =========================
# 3. 1시간 간격 재검증
# =========================
time_diff = df[time_col].diff()

bad_diff = df[
    (time_diff.notna()) &
    (time_diff != pd.Timedelta(hours=1))
]

if len(bad_diff) > 0:
    print(bad_diff[[time_col]].head(20))
    raise ValueError("1시간 간격이 아닌 구간이 있습니다. shift(-24)를 사용할 수 없습니다.")

print("1h interval check: OK")


# =========================
# 4. future target 생성
# =========================
df["sample_time"] = df[time_col]
df["target_time"] = df[time_col].shift(-PREDICTION_HORIZON)

df["future_close_24h"] = df["close"].shift(-PREDICTION_HORIZON)

df["future_return_24h"] = (
    df["future_close_24h"] / df["close"] - 1
)


# =========================
# 5. 마지막 24개 제거
# =========================
before_rows = len(df)

df = df.dropna(
    subset=["target_time", "future_close_24h", "future_return_24h"]
).reset_index(drop=True)

after_rows = len(df)

print("\n===== Future Return Check =====")
print("before rows:", before_rows)
print("after rows :", after_rows)
print("removed    :", before_rows - after_rows)

print("\nfuture_return_24h describe:")
print(df["future_return_24h"].describe())

print("\nSample:")
print(
    df[
        [time_col, "sample_time", "target_time", "close", "future_close_24h", "future_return_24h"]
    ].head()
)

print("\nTail:")
print(
    df[
        [time_col, "sample_time", "target_time", "close", "future_close_24h", "future_return_24h"]
    ].tail()
)


# =========================
# 6. 저장
# =========================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n✅ 3단계 완료")
print("saved:", OUTPUT_PATH)