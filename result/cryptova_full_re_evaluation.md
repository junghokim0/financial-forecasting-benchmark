# Cryptova-Full 기존 예측 재평가

## 1. 평가 범위

- 모델: 기존 Chart 12 + News 9 Fusion
- 기반 예측: Rolling별 Validation에서 선택된 confidence threshold 적용 결과
- Risk Filter: raw `funding_rate` + `std_24h`
- 최종 신호: 기존 `pred_filtered`
- 평가 데이터: 기존 `rolling_threshold_0012` Test 구간
- 거래 비용: fee 0.1% + slippage 0.1%
- Backtest: 24시간 non-overlap
- 재학습 및 재추론: 수행하지 않음

기존 실험 폴더는 수정하지 않았다. 원본 prediction CSV를 benchmark 공통 schema로
변환한 뒤 공통 evaluator로 다시 평가했으며, 각 원본의 SHA-256은
`re_evaluation_manifest.json`에 기록했다.

## 2. Rolling별 결과

| Rolling | Rows | Accuracy | Balanced Accuracy | Macro F1 | Return | Sharpe-like | MDD | Trades | Win Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 2,113 | 0.639375 | 0.334182 | 0.261728 | -0.13% | -0.107 | -1.53% | 2 | 50.00% |
| rolling_2 | 2,113 | 0.409844 | 0.347294 | 0.294403 | -18.04% | -1.919 | -24.40% | 61 | 49.18% |
| rolling_3 | 2,065 | 0.382567 | 0.353066 | 0.349761 | +55.72% | +4.321 | -9.36% | 56 | 62.50% |

기존 `result.json`에 기록된 Risk Filter Backtest 지표와 재평가 결과가 일치했다.

## 3. Connected Out-of-Sample 결과

| Rows | Accuracy | Balanced Accuracy | Macro F1 | SHORT Recall | HOLD Recall | LONG Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 6,291 | 0.477984 | 0.368649 | 0.350898 | 0.1426 | 0.7910 | 0.1724 |

| Return | Sharpe-like | MDD | Trades | Trade Ratio | Win Rate | Avg Trade Return |
|---:|---:|---:|---:|---:|---:|---:|
| +27.46% | +1.143 | -24.40% | 119 | 1.89% | 55.46% | +0.240% |

## 4. 해석

Cryptova-Full은 연결 OOS에서 비용 차감 후 양의 수익률과 양의 Sharpe-like를 기록했다.
평균 거래수익률 `+0.240%`는 적용한 거래당 비용 `0.20%`를 소폭 상회했다.

그러나 수익은 rolling별로 일관되지 않았다. rolling 1은 거래가 2건뿐이었고 rolling 2는
`-18.04%` 손실과 `-24.40%` MDD를 기록했으며, 전체 양의 성과는 rolling 3의 `+55.72%`에
크게 의존했다. 따라서 연결 수익률만으로 안정적인 수익모델이라고 결론 내리기보다
시장 regime별 성능과 추가 OOS 안정성 분석이 필요하다.

분류 관점에서 Macro F1은 `0.350898`이고 HOLD recall은 `0.7910`으로 높았지만 SHORT와
LONG recall은 각각 `0.1426`, `0.1724`였다. Risk Filter는 위험한 거래를 HOLD로 바꾸므로
최종 전략의 분류 성능과 투자성과를 함께 보되, Fusion base와 filter의 효과는 ablation으로
분리해서 해석해야 한다.
