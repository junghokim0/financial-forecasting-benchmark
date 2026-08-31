import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/merged/merged_hourly_features.csv")

df = pd.read_csv(INPUT_PATH)

print("===== Basic Info =====")
print("rows:", len(df))
print("columns:", len(df.columns))
print(df.columns.tolist())

# 시간 컬럼 자동 탐색
if "hour" in df.columns:
    time_col = "hour"
elif "timestamp" in df.columns:
    time_col = "timestamp"
else:
    raise ValueError("시간 컬럼이 없습니다. 'hour' 또는 'timestamp' 컬럼명을 확인하세요.")

print("\nTime column:", time_col)

# datetime 변환
df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

print("\n===== Time Parse Check =====")
print("time NaN:", df[time_col].isna().sum())

if df[time_col].isna().sum() > 0:
    print(df[df[time_col].isna()].head())
    raise ValueError("시간 변환 실패 row가 있습니다.")

# 시간 정렬
df = df.sort_values(time_col).reset_index(drop=True)

print("\n===== Time Range =====")
print("start:", df[time_col].min())
print("end  :", df[time_col].max())

# 중복 시간 확인
dup_count = df[time_col].duplicated().sum()

print("\n===== Duplicate Time Check =====")
print("duplicate time count:", dup_count)

if dup_count > 0:
    print(df[df[time_col].duplicated(keep=False)][[time_col]].head(20))
    raise ValueError("중복 시간이 있습니다.")

# 1시간 간격 검증
df["time_diff"] = df[time_col].diff()

print("\n===== Time Interval Check =====")
print(df["time_diff"].value_counts().head(10))

bad_diff = df[
    (df["time_diff"].notna()) &
    (df["time_diff"] != pd.Timedelta(hours=1))
]

print("\nnon-1h interval count:", len(bad_diff))

if len(bad_diff) > 0:
    print("\n===== Non-1h Interval Examples =====")
    print(bad_diff[[time_col, "time_diff"]].head(20))
    raise ValueError("1시간 간격이 아닌 구간이 있습니다. shift(-24) 사용 전 보정이 필요합니다.")

# close 컬럼 확인
if "close" not in df.columns:
    raise ValueError("'close' 컬럼이 없습니다. future_return_24h 계산에 필요합니다.")

print("\n===== Close Check =====")
print("close NaN:", df["close"].isna().sum())
print("close <= 0:", (df["close"] <= 0).sum())

if df["close"].isna().sum() > 0:
    raise ValueError("close에 NaN이 있습니다.")

if (df["close"] <= 0).sum() > 0:
    raise ValueError("close에 0 이하 값이 있습니다.")

print("\n✅ 1, 2단계 검증 완료")
print("merged_hourly_features.csv 로드 정상")
print("시간 정렬 정상")
print("1시간 간격 정상")
print("close 컬럼 정상")