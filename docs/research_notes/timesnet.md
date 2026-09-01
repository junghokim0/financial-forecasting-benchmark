# TimesNet 연구 노트

## 1. 정의와 실험 목적

TimesNet은 FFT로 시계열의 주요 주기를 찾고, 각 주기에 따라 1차원 시계열을 2차원으로 재배열한 뒤 Inception CNN으로 주기 내부와 주기 사이 변화를 학습하는 모델이다. 해당 실험에서는 TimesNet의 주기 모델링이 Ridge의 선형 결합과 LSTM의 순차 상태 모델링보다 추가적인 성능을 제공하는지 확인한다.

주 benchmark에는 Cryptova 내부의 수정된 TimesNet이 아니라 THUML 공식 구조를 기준으로 만든 독립 Chart 모델을 사용한다. 이를 통해 `TimesNet vs Cryptova`와 `Cryptova Chart-only vs Chart+News` 질문을 분리한다.

## 2. 모델 구조

공통 입력은 `(B,72,12)`이고 hidden dimension은 32다. `top_k=2`, convolution hidden 64, Inception kernel 4개, TimesNet layer 1개, dropout 0.30을 test 결과 확인 전에 고정했다. 이 용량은 Cryptova Chart encoder와 맞춰 backbone capacity 차이를 줄였지만, residual·normalization·embedding 순서는 공식 TimesNet core를 따른다.

```text
(B,72,12)
  ↓ DataEmbedding
(B,72,32)
  ↓ FFT top-k=2
  ↓ period별 2D reshape + Inception CNN
  ↓ amplitude 기반 가중합 + residual + LayerNorm
(B,72,32)
  ↓ GELU + Dropout + Flatten
(B,2304)
  ├─ Linear(2304,1) → predicted_return
  └─ Linear(2304,3) → SHORT/HOLD/LONG logits
```

Regression과 Classifier는 동일한 구조에서 마지막 Linear 출력 차원과 loss만 다르며, 서로 독립적으로 초기화하고 학습했다.

## 3. 공정 비교 조건

- Ridge·LSTM과 동일한 chart tensor, timestamp, rolling 1~3 및 train-only scaler
- Regression: MSE 학습, Validation RMSE 최소 checkpoint
- Classification: Cross-Entropy와 label smoothing 0.03, Validation Macro F1 최대 checkpoint
- batch 64, max epoch 50, patience 8, AdamW `1e-4`, weight decay `1e-4`
- gradient clipping 1.0, seed 42
- confidence/risk filter와 class weight 사용 안 함
- fee 0.1%, slippage 0.1%, 24시간 non-overlap backtest

## 4. Regression 결과

| Rolling | Selected epoch | Test RMSE | Test MAE |
|---|---:|---:|---:|
| rolling_1 | 12 | 0.016193 | 0.012278 |
| rolling_2 | 23 | 0.025021 | 0.018312 |
| rolling_3 | 10 | 0.028873 | 0.021455 |

연결 OOS 6,291개 결과는 다음과 같다.

| RMSE | MAE | Pearson | Spearman | Directional Accuracy |
|---:|---:|---:|---:|---:|
| 0.023916 | 0.017317 | -0.0223 | -0.0350 | 48.24% |

Ridge RMSE `0.023178`, LSTM RMSE `0.023790`보다 TimesNet RMSE가 조금 높았다. 방향 정확도는 세 모델 중 가장 높았지만 50%를 넘지 못했고 상관계수도 0에 가까워, 주기 구조가 안정적인 미래 수익률 예측력으로 이어졌다고 보기는 어렵다.

## 5. Classification 결과

| Rolling | Selected epoch | Macro F1 | Balanced Accuracy | Return | Trades |
|---|---:|---:|---:|---:|---:|
| rolling_1 | 8 | 0.300942 | 0.321828 | -14.46% | 56 |
| rolling_2 | 41 | 0.380404 | 0.380921 | +1.08% | 72 |
| rolling_3 | 7 | 0.335989 | 0.374361 | -3.10% | 59 |

연결 OOS 결과는 다음과 같다.

| Accuracy | Macro F1 | Balanced Accuracy | SHORT Recall | HOLD Recall | LONG Recall |
|---:|---:|---:|---:|---:|---:|
| 0.431728 | 0.364654 | 0.366543 | 0.2075 | 0.6230 | 0.2691 |

TimesNet은 Ridge(`0.284691`)와 LSTM Classifier(`0.318197`)보다 높은 Macro F1을 기록했다. HOLD Recall은 낮아졌지만 SHORT와 LONG Recall이 크게 개선되어 세 class를 더 균형 있게 예측했다. 예측 분포도 SHORT 21.75%, HOLD 59.20%, LONG 19.06%로 LSTM의 HOLD 82.99% 편향보다 완화됐다.

## 6. Backtest 결과

| Return | Sharpe-like | MDD | Trades | Trade Ratio | Win Rate | Avg Trade Return |
|---:|---:|---:|---:|---:|---:|---:|
| -16.21% | -0.449 | -23.75% | 187 | 2.97% | 48.13% | -0.066% |

TimesNet은 LSTM Classifier의 `-39.67%`보다 손실과 MDD가 작았지만 Ridge의 사실상 0% 수익률보다 나빴다. 비용 차감 평균 거래수익률은 `-0.066%`이고 거래당 비용이 0.20%이므로, 비용 차감 전 평균은 약 `+0.134%`다. 즉, 약한 gross edge는 있었지만 현재 거래비용을 넘을 정도로 강하지 않았다.

## 7. 종합 비교

```text
Regression 숫자 오차       Ridge > LSTM > TimesNet
Classification 균형 성능  TimesNet > LSTM > Ridge
비용 반영 Backtest         Ridge > TimesNet > LSTM
```

TimesNet의 핵심 성과는 수익률 숫자를 더 정확히 맞힌 것이 아니라 SHORT와 LONG을 더 적극적이고 균형 있게 구분한 것이다. 그러나 더 많은 거래가 발생하면서 작은 gross edge가 비용에 소진되어 최종 수익률은 음수가 됐다. 따라서 Classification 성능 개선과 거래전략 수익성은 별도로 판단해야 한다.

## 8. 현재 단계의 결론

> Official-core TimesNet Classifier는 Ridge와 LSTM보다 높은 Macro F1과 class별 균형을 보였지만, 미래 수익률 회귀에서는 단순 모델을 넘지 못했고 비용 반영 backtest도 음수였다. 주기 모델링은 신호 분류에 추가 가치를 보였으나 안정적인 순수익 모델을 완성하지는 못했다.

이 결론은 seed 42와 세 rolling에 대한 현재 benchmark 결과다. 최종 논문 결론은 Chronos, TimesFM, Cryptova 및 사전에 고정한 regime 분석이 완료된 뒤 내린다.
