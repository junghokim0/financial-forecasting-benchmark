# Benchmarking Foundation Models and Specialized Models for Financial Forecasting

This repository compares traditional, specialized, and pretrained time-series models on a common Bitcoin forecasting protocol. The primary goal is to determine whether foundation models are competitive with simpler baselines and the Chart+News Cryptova system under identical out-of-sample evaluation rules.

## Research questions

1. Can time-series foundation models outperform traditional forecasting models?
2. Can foundation models outperform Cryptova?
3. In which short-term trend and volatility regimes does each model perform best?

## Common protocol

- Market frequency: hourly BTC data
- Canonical input: observations available during the previous 72 hours
- Target: future 24-hour return or its `SHORT/HOLD/LONG` class
- TimesFM exception: the most recent 64 hours, aligned to its 32-step patch size
- Splits: three chronological rolling Train/Validation/Test windows
- Model selection: Validation only
- Final evaluation: 6,291 connected out-of-sample Test timestamps
- Trading evaluation: 24-hour non-overlapping positions
- Cost per selected trade: 0.1% fee + 0.1% slippage

### Rolling Train/Validation/Test periods

All timestamps are UTC. Each period is written as `start (inclusive) → end (exclusive)`.

| Rolling | Train | Validation | Test |
|---|---|---|---|
| 1 | 2024-01-01 00:00 to 2025-04-01 00:00 | 2025-04-01 00:00 to 2025-07-01 00:00 | 2025-07-01 00:00 to 2025-10-01 00:00 |
| 2 | 2024-04-01 00:00 to 2025-07-01 00:00 | 2025-07-01 00:00 to 2025-10-01 00:00 | 2025-10-01 00:00 to 2026-01-01 00:00 |
| 3 | 2024-07-01 00:00 to 2025-10-01 00:00 | 2025-10-01 00:00 to 2026-01-01 00:00 | 2026-01-01 00:00 to 2026-04-01 00:00 |

For the complete frozen protocol—including split construction, leakage-prevention rules, model-selection criteria, backtest assumptions, metric definitions, model settings, and limitations—see [`result/fair_comparison_protocol_and_results.md`](result/fair_comparison_protocol_and_results.md).

## Models

| Family | Model | Input | Output |
|---|---|---|---|
| Linear baseline | Ridge-Flat | 72 x 12 chart features | Future return |
| Recurrent baseline | LSTM | 72 x 12 chart features | Return / direct class |
| Specialized model | TimesNet | 72 x 12 chart features | Return / direct class |
| Foundation model | Chronos-2 LoRA | Close + 12 past chart covariates | Future return |
| Foundation model | TimesFM 2.5 LoRA | Recent 64-hour close series | Future return |
| Proposed system | Cryptova | 12 chart + 9 news features | Direct class + confidence/risk filters |

## Main connected OOS results

### Regression track

| Model | RMSE | MAE |
|---|---:|---:|
| **Zero-return baseline** | **0.022948** | **0.016422** |
| Ridge-Flat | 0.023178 | 0.016620 |
| LSTM | 0.023790 | 0.017203 |
| TimesNet | 0.023916 | 0.017317 |
| Chronos-2 LoRA | 0.024300 | 0.017148 |
| TimesFM 2.5 LoRA | 0.024992 | 0.017804 |

The zero-return baseline always predicts a future return of 0% and requires no training. It achieved the lowest RMSE and MAE; Ridge-Flat was the best-performing trained regression model but did not outperform this naive baseline.

### Classification and trading track

| Model | Macro F1 | Balanced Accuracy | Cost-adjusted return |
|---|---:|---:|---:|
| Ridge-Flat | 0.284691 | 0.343804 | -0.005% |
| LSTM Classifier | 0.318197 | 0.356864 | -39.67% |
| TimesNet Classifier | 0.364654 | 0.366543 | -16.21% |
| Chronos-2 LoRA | 0.252000 | 0.338928 | -24.64% |
| TimesFM 2.5 LoRA | 0.287013 | 0.345735 | -30.96% |
| **Cryptova-Raw** | **0.381875** | **0.393802** | -18.11% |
| Cryptova-Base | 0.376506 | 0.389647 | +7.42% |
| **Cryptova-Full** | 0.350898 | 0.368649 | **+27.46%** |

These are exploratory research results, not evidence of guaranteed profitability. Neural models were primarily evaluated with seed 42; repeated-seed uncertainty and a newly untouched holdout remain future work.

## Market-regime analysis

The direction regime describes whether price was relatively rising or falling over the 72 hours immediately before each prediction. Thresholds are calculated from each rolling Train period only:

```text
UP:       return_72h >= Train Q67 and return_72h > 0
DOWN:     return_72h <= Train Q33 and return_72h < 0
SIDEWAYS: otherwise
```

Volatility is split using the corresponding rolling Train median of `std_24h`. Detailed conditional metrics and the six combined regimes are available in [`outputs/regime_analysis`](outputs/regime_analysis) and Section 6 of the result document.

## Repository layout

```text
configs/           Common configuration and prediction schema
src/data/          Collection, preprocessing, dataset, and validation code
src/evaluation/    Common regression, classification, backtest, and regime evaluators
src/models/        Shared model components
ridge_regression/  Ridge-Flat baseline
lstm/              LSTM regression and classifier
timesnet/          TimesNet regression and classifier
chronos/           Chronos-2 LoRA Colab workflow
timesfm/           TimesFM 2.5 LoRA Colab workflow
main_fusion/       Cryptova model and reference evaluation code
outputs/           Selected final CSV/JSON evaluation artifacts only
result/            Consolidated protocol and research conclusions
data/              Metadata and hashes only; no dataset files
```

## Data availability

Raw, processed, windowed, and rolling dataset files are intentionally not distributed through Git. The repository contains collection and preprocessing code plus:

- [`data/README.md`](data/README.md): sources, expected layout, and reconstruction notes
- [`data/data_manifest.csv`](data/data_manifest.csv): paths, sizes, roles, and SHA-256 hashes
- [`data/checksums.sha256`](data/checksums.sha256): integrity checks for the research artifacts

MarketAux news content is not redistributed. Users must obtain their own API access and comply with the provider's terms.

## Installation

Install common dependencies:

```bash
python -m pip install -r requirements.txt
```

Chronos and TimesFM use separate Colab environments:

```text
chronos/requirements-colab.txt
timesfm/requirements-colab.txt
```

See the README inside each model directory for model-specific validation, training, aggregation, and evaluation commands.

## Reproducing evaluation

The repository publishes final prediction CSV files but not fitted weights. This allows the common evaluators and result calculations to be audited without distributing `*.pt`, LoRA adapters, scalers, or foundation-model checkpoints.

Key evaluation code:

```text
src/evaluation/classification.py
src/evaluation/regression.py
src/evaluation/backtest.py
src/evaluation/analyze_market_regimes.py
```

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License and disclaimer

No repository-wide open-source license has been granted at this time. Contact the author before reuse or redistribution. This project is for research and educational purposes and does not constitute financial advice.
