#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_rolling_datasets.py

역할:
- 전체 sliding window candidates를 rolling train/val/test로 분리
- strict split 조건 적용
- cost-aware threshold 기준으로 SHORT/HOLD/LONG label 부여
- rolling train 데이터 기준으로만 scaling fit
- train/val/test X_chart, X_news, y, meta 저장

Label:
- raw_future_return <= -0.012 → SHORT
- -0.012 < raw_future_return < 0.012 → HOLD
- raw_future_return >= 0.012 → LONG
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# 입력/출력 설정
# ============================================================
WINDOW_SIZE = 72
PREDICTION_HORIZON = 24



CANDIDATE_DIR = Path(f"data/dataset/candidates/window_{WINDOW_SIZE}")
#CANDIDATE_DIR = Path(f"data/dataset/candidates_deriv14/window_{WINDOW_SIZE}")
#CANDIDATE_DIR = Path(f"data/dataset/candidates_funding13/window_{WINDOW_SIZE}")
X_CHART_PATH = CANDIDATE_DIR / "X_chart_candidates.npy"
X_NEWS_PATH = CANDIDATE_DIR / "X_news_candidates.npy"
META_PATH = CANDIDATE_DIR / "sample_meta.csv"


# ============================================================
# Cost-aware label 설정
# ============================================================
FEE = 0.001
SLIPPAGE = 0.001
MARGIN = 0.010

RETURN_THRESHOLD = FEE + SLIPPAGE + MARGIN  # 0.012



THRESHOLD_TAG = str(RETURN_THRESHOLD).replace(".", "")

OUTPUT_BASE_DIR = Path(
    f"data/dataset/rolling_threshold_{THRESHOLD_TAG}"
)

#OUTPUT_BASE_DIR = Path(
#    f"data/dataset/rolling_threshold_{THRESHOLD_TAG}_deriv14"
#)

#OUTPUT_BASE_DIR = Path(
#    f"data/dataset/rolling_threshold_{THRESHOLD_TAG}_funding13"
#)
LABEL_MAP = {
    "SHORT": 0,
    "HOLD": 1,
    "LONG": 2,
}

ID_TO_LABEL = {
    0: "SHORT",
    1: "HOLD",
    2: "LONG",
}


ROLLING_SPLITS = [
    {
        "name": "rolling_1",
        "train": ("2024-01-01 00:00:00+00:00", "2025-04-01 00:00:00+00:00"),
        "val":   ("2025-04-01 00:00:00+00:00", "2025-07-01 00:00:00+00:00"),
        "test":  ("2025-07-01 00:00:00+00:00", "2025-10-01 00:00:00+00:00"),
    },
    {
        "name": "rolling_2",
        "train": ("2024-04-01 00:00:00+00:00", "2025-07-01 00:00:00+00:00"),
        "val":   ("2025-07-01 00:00:00+00:00", "2025-10-01 00:00:00+00:00"),
        "test":  ("2025-10-01 00:00:00+00:00", "2026-01-01 00:00:00+00:00"),
    },
    {
        "name": "rolling_3",
        "train": ("2024-07-01 00:00:00+00:00", "2025-10-01 00:00:00+00:00"),
        "val":   ("2025-10-01 00:00:00+00:00", "2026-01-01 00:00:00+00:00"),
        "test":  ("2026-01-01 00:00:00+00:00", "2026-04-01 00:00:00+00:00"),
    },
]


# ============================================================
# 유틸 함수
# ============================================================
def to_utc(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, utc=True)


def load_candidates():
    if not X_CHART_PATH.exists():
        raise FileNotFoundError(f"X_chart 후보 파일 없음: {X_CHART_PATH}")

    if not X_NEWS_PATH.exists():
        raise FileNotFoundError(f"X_news 후보 파일 없음: {X_NEWS_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"meta 파일 없음: {META_PATH}")

    X_chart = np.load(X_CHART_PATH)
    X_news = np.load(X_NEWS_PATH)
    meta = pd.read_csv(META_PATH)

    required_meta_cols = [
        "candidate_index",
        "input_start_time",
        "input_end_time",
        "sample_time",
        "target_time",
        "raw_future_return",
    ]

    missing = set(required_meta_cols) - set(meta.columns)
    if missing:
        raise ValueError(f"meta 필수 컬럼 없음: {sorted(missing)}")

    for col in ["input_start_time", "input_end_time", "sample_time", "target_time"]:
        meta[col] = pd.to_datetime(meta[col], utc=True, errors="coerce")
        nan_count = int(meta[col].isna().sum())
        print(f"{col} NaN:", nan_count)

        if nan_count > 0:
            raise ValueError(f"meta {col} 컬럼에 시간 파싱 실패가 있습니다.")

    meta["raw_future_return"] = pd.to_numeric(
        meta["raw_future_return"],
        errors="coerce",
    )

    if meta["raw_future_return"].isna().sum() > 0:
        raise ValueError("raw_future_return에 NaN이 있습니다.")

    if X_chart.ndim != 3:
        raise ValueError(f"X_chart는 3차원이어야 합니다: {X_chart.shape}")

    if X_news.ndim != 3:
        raise ValueError(f"X_news는 3차원이어야 합니다: {X_news.shape}")

    if len(meta) != X_chart.shape[0]:
        raise ValueError("meta row 수와 X_chart sample 수가 다릅니다.")

    if len(meta) != X_news.shape[0]:
        raise ValueError("meta row 수와 X_news sample 수가 다릅니다.")

    if X_chart.shape[1] != WINDOW_SIZE:
        raise ValueError(f"X_chart window size가 {WINDOW_SIZE}가 아닙니다: {X_chart.shape}")

    if X_news.shape[1] != WINDOW_SIZE:
        raise ValueError(f"X_news window size가 {WINDOW_SIZE}가 아닙니다: {X_news.shape}")

    if not np.isfinite(X_chart).all():
        raise ValueError("X_chart candidates에 NaN 또는 inf가 있습니다.")

    if not np.isfinite(X_news).all():
        raise ValueError("X_news candidates에 NaN 또는 inf가 있습니다.")

    print("\n===== Candidate Load Check =====")
    print("X_chart shape:", X_chart.shape)
    print("X_news shape :", X_news.shape)
    print("meta rows    :", len(meta))
    print("sample_time range:", meta["sample_time"].min(), "~", meta["sample_time"].max())
    print("target_time range:", meta["target_time"].min(), "~", meta["target_time"].max())

    return X_chart, X_news, meta


def make_strict_mask(meta: pd.DataFrame, start: str, end: str) -> pd.Series:
    start = to_utc(start)
    end = to_utc(end)

    mask = (
        (meta["input_start_time"] >= start)
        & (meta["input_end_time"] < end)
        & (meta["sample_time"] >= start)
        & (meta["sample_time"] < end)
        & (meta["target_time"] >= start)
        & (meta["target_time"] < end)
    )

    return mask


def check_split_overlap(train_idx, val_idx, test_idx, rolling_name: str):
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    train_val_overlap = len(train_set & val_set)
    train_test_overlap = len(train_set & test_set)
    val_test_overlap = len(val_set & test_set)

    print("\n[split overlap check]")
    print("train ∩ val :", train_val_overlap)
    print("train ∩ test:", train_test_overlap)
    print("val ∩ test  :", val_test_overlap)

    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        raise ValueError(f"{rolling_name}: train/val/test index overlap이 있습니다.")


def summarize_split(meta: pd.DataFrame, idx: np.ndarray, split_name: str):
    split_meta = meta.loc[idx]

    print(f"\n[{split_name} split]")
    print("sample count:", len(split_meta))

    if len(split_meta) == 0:
        return

    print("input_start:", split_meta["input_start_time"].min(), "~", split_meta["input_start_time"].max())
    print("input_end  :", split_meta["input_end_time"].min(), "~", split_meta["input_end_time"].max())
    print("sample_time:", split_meta["sample_time"].min(), "~", split_meta["sample_time"].max())
    print("target_time:", split_meta["target_time"].min(), "~", split_meta["target_time"].max())
    print("return describe:")
    print(split_meta["raw_future_return"].describe())


def compute_thresholds():
    short_th = -RETURN_THRESHOLD
    long_th = RETURN_THRESHOLD

    return float(short_th), float(long_th)


def make_labels(returns: np.ndarray, short_th: float, long_th: float):
    labels = np.full(len(returns), "HOLD", dtype=object)

    labels[returns <= short_th] = "SHORT"
    labels[returns >= long_th] = "LONG"

    label_ids = np.array([LABEL_MAP[x] for x in labels], dtype=np.int64)

    return labels, label_ids


def print_label_distribution(y: np.ndarray, split_name: str):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    print(f"\n[{split_name} label distribution]")

    for label_id, count in zip(unique, counts):
        label_name = ID_TO_LABEL[int(label_id)]
        ratio = count / total if total > 0 else 0
        print(f"{label_name:5s} ({int(label_id)}): {int(count):6d} ({ratio:.4f})")


def fit_transform_3d_scaler(X_train, X_val, X_test):
    if X_train.ndim != 3:
        raise ValueError(f"X_train은 3차원이어야 합니다: {X_train.shape}")

    _, t, f = X_train.shape

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, f))

    def transform(X):
        if X.ndim != 3:
            raise ValueError(f"X는 3차원이어야 합니다: {X.shape}")

        n = X.shape[0]
        transformed = scaler.transform(X.reshape(-1, f)).reshape(n, t, f)
        transformed = transformed.astype(np.float32)

        if not np.isfinite(transformed).all():
            raise ValueError("scaling 결과에 NaN 또는 inf가 있습니다.")

        return transformed

    return transform(X_train), transform(X_val), transform(X_test), scaler


def save_split(out_dir, split_name, X_chart, X_news, y, meta_split):
    np.save(out_dir / f"X_chart_{split_name}.npy", X_chart)
    np.save(out_dir / f"X_news_{split_name}.npy", X_news)
    np.save(out_dir / f"y_{split_name}.npy", y)

    meta_split.to_csv(
        out_dir / f"sample_meta_{split_name}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def process_one_rolling(X_chart, X_news, meta, split):
    name = split["name"]

    print("\n" + "=" * 80)
    print(f"🚀 {name} 처리 시작")
    print("=" * 80)

    out_dir = OUTPUT_BASE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[rolling period]")
    print("train:", split["train"][0], "~", split["train"][1])
    print("val  :", split["val"][0], "~", split["val"][1])
    print("test :", split["test"][0], "~", split["test"][1])

    train_mask = make_strict_mask(meta, *split["train"])
    val_mask = make_strict_mask(meta, *split["val"])
    test_mask = make_strict_mask(meta, *split["test"])

    train_idx = np.where(train_mask.values)[0]
    val_idx = np.where(val_mask.values)[0]
    test_idx = np.where(test_mask.values)[0]

    summarize_split(meta, train_idx, "train")
    summarize_split(meta, val_idx, "val")
    summarize_split(meta, test_idx, "test")

    if len(train_idx) == 0:
        raise ValueError(f"{name}: train sample이 0개입니다.")

    if len(val_idx) == 0:
        raise ValueError(f"{name}: val sample이 0개입니다.")

    if len(test_idx) == 0:
        raise ValueError(f"{name}: test sample이 0개입니다.")

    check_split_overlap(train_idx, val_idx, test_idx, name)

    short_th, long_th = compute_thresholds()

    print("\n[cost-aware threshold]")
    print(f"fee              : {FEE:.6f}")
    print(f"slippage         : {SLIPPAGE:.6f}")
    print(f"margin           : {MARGIN:.6f}")
    print(f"return threshold : {RETURN_THRESHOLD:.6f}")
    print(f"short threshold  : {short_th:.8f}")
    print(f"long threshold   : {long_th:.8f}")

    y_train_label, y_train = make_labels(
        meta.loc[train_idx, "raw_future_return"].to_numpy(dtype=np.float64),
        short_th,
        long_th,
    )

    y_val_label, y_val = make_labels(
        meta.loc[val_idx, "raw_future_return"].to_numpy(dtype=np.float64),
        short_th,
        long_th,
    )

    y_test_label, y_test = make_labels(
        meta.loc[test_idx, "raw_future_return"].to_numpy(dtype=np.float64),
        short_th,
        long_th,
    )

    print_label_distribution(y_train, "train")
    print_label_distribution(y_val, "val")
    print_label_distribution(y_test, "test")

    Xc_train_raw = X_chart[train_idx]
    Xc_val_raw = X_chart[val_idx]
    Xc_test_raw = X_chart[test_idx]

    Xn_train_raw = X_news[train_idx]
    Xn_val_raw = X_news[val_idx]
    Xn_test_raw = X_news[test_idx]

    print("\n[raw split shape]")
    print("X_chart_train:", Xc_train_raw.shape)
    print("X_chart_val  :", Xc_val_raw.shape)
    print("X_chart_test :", Xc_test_raw.shape)
    print("X_news_train :", Xn_train_raw.shape)
    print("X_news_val   :", Xn_val_raw.shape)
    print("X_news_test  :", Xn_test_raw.shape)

    Xc_train, Xc_val, Xc_test, chart_scaler = fit_transform_3d_scaler(
        Xc_train_raw,
        Xc_val_raw,
        Xc_test_raw,
    )

    Xn_train, Xn_val, Xn_test, news_scaler = fit_transform_3d_scaler(
        Xn_train_raw,
        Xn_val_raw,
        Xn_test_raw,
    )

    print("\n[scaled check]")
    print("chart train mean approx:", float(Xc_train.mean()))
    print("chart train std approx :", float(Xc_train.std()))
    print("news train mean approx :", float(Xn_train.mean()))
    print("news train std approx  :", float(Xn_train.std()))

    meta_train = meta.loc[train_idx].copy().reset_index(drop=True)
    meta_val = meta.loc[val_idx].copy().reset_index(drop=True)
    meta_test = meta.loc[test_idx].copy().reset_index(drop=True)

    meta_train["label"] = y_train_label
    meta_val["label"] = y_val_label
    meta_test["label"] = y_test_label

    meta_train["label_id"] = y_train
    meta_val["label_id"] = y_val
    meta_test["label_id"] = y_test

    save_split(out_dir, "train", Xc_train, Xn_train, y_train, meta_train)
    save_split(out_dir, "val", Xc_val, Xn_val, y_val, meta_val)
    save_split(out_dir, "test", Xc_test, Xn_test, y_test, meta_test)

    joblib.dump(chart_scaler, out_dir / "chart_scaler.pkl")
    joblib.dump(news_scaler, out_dir / "news_scaler.pkl")

    config = {
        "name": name,
        "window_size": WINDOW_SIZE,
        "prediction_horizon": PREDICTION_HORIZON,
        "labeling_method": "cost_aware_fixed_threshold",
        "fee": FEE,
        "slippage": SLIPPAGE,
        "margin": MARGIN,
        "return_threshold": RETURN_THRESHOLD,
        "short_threshold": short_th,
        "long_threshold": long_th,
        "label_map": LABEL_MAP,
        "id_to_label": ID_TO_LABEL,
        "split": split,
        "strict_policy": (
            "input_start_time, input_end_time, sample_time, target_time "
            "must all be inside each split."
        ),
        "scaling": (
            "StandardScaler fitted only on rolling train data. "
            "Same scaler used for train/val/test."
        ),
        "num_samples": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "X_chart_shape": {
            "train": list(Xc_train.shape),
            "val": list(Xc_val.shape),
            "test": list(Xc_test.shape),
        },
        "X_news_shape": {
            "train": list(Xn_train.shape),
            "val": list(Xn_val.shape),
            "test": list(Xn_test.shape),
        },
    }

    with open(out_dir / "rolling_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n✅ saved: {out_dir}")


def main():
    print("🚀 Rolling dataset 생성 시작")

    X_chart, X_news, meta = load_candidates()

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for split in ROLLING_SPLITS:
        process_one_rolling(X_chart, X_news, meta, split)

    print("\n🎉 모든 rolling dataset 생성 완료")
    print("saved base dir:", OUTPUT_BASE_DIR)


if __name__ == "__main__":
    main()