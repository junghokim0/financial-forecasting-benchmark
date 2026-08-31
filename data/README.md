# Data availability and reconstruction

This directory intentionally contains metadata only. No raw news, processed market data, NumPy windows, fitted scalers, or model-ready rolling datasets are distributed through Git.

## Canonical research artifacts

The benchmark was evaluated from the following local data groups:

```text
data/master/merged_with_future_return.csv
data/risk/bybit_funding_rate_raw.csv
data/cryptova_reference/candidates/window_72/
data/cryptova_reference/rolling_threshold_0012/rolling_1/
data/cryptova_reference/rolling_threshold_0012/rolling_2/
data/cryptova_reference/rolling_threshold_0012/rolling_3/
```

Their expected paths, byte sizes, roles, and SHA-256 hashes are listed in `data_manifest.csv`. `checksums.sha256` provides a standard integrity list.

## Sources

| Data | Source | Frequency | Use |
|---|---|---:|---|
| BTC market candles | Bybit public market API | 1 hour | Chart features and target construction |
| Derivatives/Funding Rate | Bybit public market API | provider timestamps aligned hourly | Cryptova risk-filter input only |
| Financial news | MarketAux API | event time aggregated hourly | Nine Cryptova news features |

News content is not redistributed. Researchers must obtain their own provider access and comply with the applicable terms.

## Pipeline code

```text
src/data/collection/fetch_bybit_chart.py
src/data/collection/fetch_bybit_derivatives.py
src/data/collection/fetch_marketaux_news.py
src/data/preprocessing/preprocess_bybit_chart.py
src/data/preprocessing/preprocess_marketaux_news.py
src/data/preprocessing/merge_chart_news.py
src/data/preprocessing/make_merged_future_return.py
src/data/datasets/make_sliding_window_candidates.py
src/data/datasets/make_rolling_datasets.py
src/data/validation/check_merged_data.py
src/data/validation/check_future_return.py
```

Run collection and preprocessing chronologically. API credentials must be stored locally and must never be committed.

## Leakage-prevention rules

- Scalers are fitted separately on each rolling Train period.
- Validation and Test are transformed using the corresponding Train-fitted scaler.
- Splits preserve chronological order.
- Future 24-hour returns are targets and are never used as model inputs or regime features.
- Regime thresholds are estimated from the corresponding rolling Train period only.

## Rolling periods

| Rolling | Train | Validation | Test |
|---|---|---|---|
| 1 | 2024-01-01 to 2025-04-01 | 2025-04-01 to 2025-07-01 | 2025-07-01 to 2025-10-01 |
| 2 | 2024-04-01 to 2025-07-01 | 2025-07-01 to 2025-10-01 | 2025-10-01 to 2026-01-01 |
| 3 | 2024-07-01 to 2025-10-01 | 2025-10-01 to 2026-01-01 | 2026-01-01 to 2026-04-01 |
