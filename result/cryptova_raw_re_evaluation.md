# Cryptova-Raw 기존 예측 재평가

## 1. 정의 및 평가 범위

Cryptova-Raw는 Chart 12 + News 9 Fusion 모델의 softmax 출력에 단순 argmax만 적용한
결과다. Confidence threshold와 Funding rate/`std_24h` Risk Filter를 모두 적용하지 않는다.

- 최종 신호: 기존 prediction CSV의 `y_pred_argmax`
- 거래 비용: fee 0.1% + slippage 0.1%
- Backtest: 24시간 non-overlap
- 재학습 및 재추론: 수행하지 않음

## 2. Rolling별 결과

| Rolling | Rows | Accuracy | Balanced Accuracy | Macro F1 | Return | Sharpe-like | MDD | Trades | Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 2,113 | 0.621391 | 0.350653 | 0.309426 | -9.95% | -1.949 | -12.70% | 38 | 44.74% |
| rolling_2 | 2,113 | 0.380028 | 0.347500 | 0.293626 | -24.90% | -2.649 | -33.41% | 68 | 42.65% |
| rolling_3 | 2,065 | 0.379177 | 0.390435 | 0.382108 | +21.10% | +1.980 | -12.44% | 72 | 55.56% |

## 3. Connected Out-of-Sample 결과

| Rows | Accuracy | Balanced Accuracy | Macro F1 | SHORT Recall | HOLD Recall | LONG Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 6,291 | 0.460817 | 0.393802 | 0.381875 | 0.1571 | 0.6716 | 0.3527 |

| Return | Sharpe-like | MDD | Trades | Trade Ratio | Win Rate | Avg Trade Return |
|---:|---:|---:|---:|---:|---:|---:|
| -18.11% | -0.545 | -36.74% | 178 | 2.83% | 48.31% | -0.083% |

## 4. Raw · Base · Full Ablation

| 지표 | Raw: Argmax | Base: +Confidence | Full: +Risk |
|---|---:|---:|---:|
| Macro F1 | **0.381875** | 0.376506 | 0.350898 |
| Balanced Accuracy | **0.393802** | 0.389647 | 0.368649 |
| SHORT Recall | **0.1571** | 0.1426 | 0.1426 |
| LONG Recall | **0.3527** | 0.3062 | 0.1724 |
| Return | -18.11% | +7.42% | **+27.46%** |
| Sharpe-like | -0.545 | +0.446 | **+1.143** |
| MDD | -36.74% | -37.38% | **-24.40%** |
| Trades | 178 | 158 | 119 |
| Win Rate | 48.31% | 48.10% | **55.46%** |

후처리를 추가할수록 Macro F1과 Balanced Accuracy는 낮아졌다. 반면 Confidence threshold는
저신뢰 신호를 HOLD로 바꾸면서 연결 수익률을 `-18.11%`에서 `+7.42%`로 전환했다.
Risk Filter는 LONG 신호를 추가로 제거하여 Full 수익률을 `+27.46%`, Sharpe-like를
`+1.143`으로 높이고 MDD를 `-24.40%`로 줄였다.

따라서 Fusion 모델 자체의 분류 성능은 Raw에서 가장 높고, 실제 거래성과는 Full에서 가장
높다. 이는 분류 정확도와 거래의 경제적 가치가 같은 목적함수가 아니라는 것을 보여준다.
