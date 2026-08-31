# Cryptova-Base 기존 예측 재평가

## 1. 정의 및 평가 범위

Cryptova-Base는 Chart 12 + News 9 Fusion 모델의 출력에 Rolling Validation에서 선택한
Confidence threshold까지 적용한 시스템이다. Confidence가 threshold보다 낮은 예측은
HOLD로 변경한다. Funding rate와 `std_24h` Risk Filter는 적용하지 않는다.

- 최종 신호: 기존 prediction CSV의 `pred_base`
- Confidence threshold: Rolling 1/2/3 = `0.40 / 0.36 / 0.46`
- 거래 비용: fee 0.1% + slippage 0.1%
- Backtest: 24시간 non-overlap
- 재학습 및 재추론: 수행하지 않음

## 2. Rolling별 결과

| Rolling | Rows | Accuracy | Balanced Accuracy | Macro F1 | Return | Sharpe-like | MDD | Trades | Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 2,113 | 0.633696 | 0.338434 | 0.278670 | -2.73% | -0.595 | -6.61% | 20 | 45.00% |
| rolling_2 | 2,113 | 0.386654 | 0.350358 | 0.294778 | -28.24% | -3.213 | -31.15% | 68 | 39.71% |
| rolling_3 | 2,065 | 0.391768 | 0.391279 | 0.390477 | +53.88% | +3.695 | -11.40% | 70 | 57.14% |

## 3. Connected Out-of-Sample 결과

| Rows | Accuracy | Balanced Accuracy | Macro F1 | SHORT Recall | HOLD Recall | LONG Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 6,291 | 0.471308 | 0.389647 | 0.376506 | 0.1426 | 0.7202 | 0.3062 |

| Return | Sharpe-like | MDD | Trades | Trade Ratio | Win Rate | Avg Trade Return |
|---:|---:|---:|---:|---:|---:|---:|
| +7.42% | +0.446 | -37.38% | 158 | 2.51% | 48.10% | +0.080% |

## 4. Cryptova-Base와 Full 비교

| 지표 | Base: Confidence | Full: Confidence + Risk | 변화 |
|---|---:|---:|---:|
| Macro F1 | **0.376506** | 0.350898 | -0.025608 |
| Balanced Accuracy | **0.389647** | 0.368649 | -0.020998 |
| LONG Recall | **0.3062** | 0.1724 | -0.1338 |
| Return | +7.42% | **+27.46%** | +20.04%p |
| Sharpe-like | +0.446 | **+1.143** | +0.697 |
| MDD | -37.38% | **-24.40%** | +12.98%p 개선 |
| Trades | 158 | 119 | -39 |

Risk Filter는 일부 LONG을 HOLD로 변경하면서 분류 성능과 LONG recall을 낮췄다. 반면
거래 수와 손실 노출을 줄여 Connected OOS 수익률, Sharpe-like 및 MDD를 개선했다.

Base도 전체 연결 구간에서는 양의 수익률이지만 rolling 2에서 `-28.24%`, rolling 3에서
`+53.88%`를 기록해 regime별 편차가 크다. 따라서 Base와 Full 모두 구간 안정성 분석이
필요하며, Full의 개선은 분류 정확도 향상이 아니라 거래 선택과 위험 관리의 효과로
해석해야 한다.
