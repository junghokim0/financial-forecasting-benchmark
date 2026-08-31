# Ridge-Flat baseline

This folder contains the benchmark's regulated linear baseline. It preserves
the original Cryptova source and consumes the already-created rolling chart
tensors without changing them.

## Protocol

- Input: `(N, 72, 12)` chart tensor
- Transformation: row-major flatten to `(N, 864)`
- Target: unscaled `raw_future_return` from `sample_meta_*.csv`
- Model: Ridge Regression with an unregularized intercept
- Solver: NumPy SVD closed-form solution; no explicit matrix inverse
- Alpha selection variants:
  - `rmse_selected`: lowest validation RMSE (forecasting result)
  - `macro_f1_selected`: highest validation Macro F1 after the fixed class conversion
- Refit after selection: no
- Test use: final evaluation only
- Class conversion: `<= -0.012` SHORT, `>= 0.012` LONG, otherwise HOLD
- Backtest: common Cryptova-compatible 24-hour non-overlapping evaluator

The existing Chart tensors were already transformed by a scaler fitted on each
rolling train period. This implementation does not fit another scaler.

## Run

```powershell
python ridge_regression/train_ridge_flat.py
```

Run one rolling split:

```powershell
python ridge_regression/train_ridge_flat.py --rollings rolling_1
```

Outputs are written to `outputs/ridge_flat/rolling_N/`. `alpha_search.csv`
contains both regression and classification validation metrics. The two
selection variants are stored separately under `rmse_selected/` and
`macro_f1_selected/`, each with its selected model, validation/test prediction
tables, common evaluation metrics, trades, and run metadata. Macro-F1 ties are
resolved by lower validation RMSE; test results never affect selection.
