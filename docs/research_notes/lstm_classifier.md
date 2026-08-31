# LSTM Classifier 연구 노트

## 1. 정의

LSTM Classifier는 과거 72시간의 Chart Feature 12개를 순서대로 처리하고 미래 24시간의 `SHORT/HOLD/LONG`을 직접 예측하는 Many-to-One Classification 모델이다.

```text
Chart sequence (B,72,12)
     ↓
LSTM Encoder
     ↓
3-class logits
     ↓
SHORT / HOLD / LONG
```

LSTM Regression이 `raw_future_return`을 MSE로 학습하는 것과 달리, LSTM Classifier는 처음부터 `label_id`를 Cross-Entropy로 학습한다. 따라서 Cryptova와의 주 Classification 비교에 사용한다.

## 2. 모델 구조

```text
Chart Input (B,72,12)
      ↓
Unidirectional LSTM
input_size=12, hidden_size=32, num_layers=1
      ↓
Final Hidden State (B,32)
      ↓
Dropout(0.30)
      ↓
Linear(32 → 3)
      ↓
Logits (B,3)
      ↓
Softmax
[P(SHORT), P(HOLD), P(LONG)]
      ↓
Argmax → SHORT / HOLD / LONG
```

| 항목 | 값 |
|---|---:|
| Input size | 12 |
| Sequence length | 72 |
| Hidden size | 32 |
| LSTM layers | 1 |
| Bidirectional | False |
| Output dropout | 0.30 |
| Classification head | `Linear(32,3)` |
| Trainable parameters | 5,987 |

## 3. Label 정의

Dataset 생성 단계에서 실제 미래 24시간 수익률에 고정 threshold를 적용해 `label_id`를 만들었다.

```text
raw_future_return <= -0.012 → SHORT, label 0
-0.012 < return < +0.012   → HOLD,  label 1
raw_future_return >= 0.012 → LONG,  label 2
```

LSTM Classifier는 `raw_future_return`을 예측한 뒤 threshold를 적용하지 않는다. 이미 생성된 `label_id`를 직접 target으로 사용한다.

## 4. 실험 가정과 공정성

- 입력은 LSTM Regression과 동일한 `(N,72,12)` Chart tensor다.
- Scaler는 각 rolling의 Train에서만 fit된 기존 결과를 사용한다.
- Train에서만 모델 파라미터를 학습한다.
- Validation Macro F1으로 checkpoint를 선택한다.
- Test는 최종 평가에만 사용한다.
- Confidence filter와 funding/std risk filter는 사용하지 않는다.
- Cryptova와 동일하게 class weight는 사용하지 않고 label smoothing `0.03`을 적용한다.

Cryptova는 Chart+News를 사용하고 auxiliary head loss를 포함하므로 순수 architecture-only 비교는 아니다. 이 결과는 각 모델을 실제 목적에 맞게 구성한 end-to-end Classification 비교로 해석한다.

## 5. 학습 설정

| 항목 | 값 |
|---|---:|
| Target | `label_id` |
| Loss | Cross-Entropy |
| Label smoothing | 0.03 |
| Class weights | 없음 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Batch size | 64 |
| Maximum epochs | 50 |
| Patience | 8 |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Device | CPU |

매 epoch Validation Macro F1을 계산하고 가장 높은 checkpoint를 저장하였다. Macro F1이 같으면 Validation Cross-Entropy가 낮은 checkpoint를 선택하였다. 8 epoch 동안 선택 기준이 개선되지 않으면 학습을 종료하였다.

## 6. 학습과 추론 과정

```text
X_chart_train + label_id
          ↓
TensorDataset / DataLoader
          ↓
LSTM Classifier logits (B,3)
          ↓
Cross-Entropy Loss → Backpropagation → Gradient clipping → AdamW
          ↓
Validation Macro F1
          ↓
Best checkpoint
          ↓
Test softmax probabilities → Argmax class
          ↓
공통 Classification / Backtest evaluator
```

Prediction CSV에는 `prob_short`, `prob_hold`, `prob_long`, `confidence`를 저장하지만 주 비교에서 confidence filtering은 적용하지 않는다.

## 7. 결과

### 7.1 Rolling별 결과

| Rolling | 선택 epoch | Validation Macro F1 | Test Macro F1 | Balanced Accuracy | Return | Trades |
|---|---:|---:|---:|---:|---:|---:|
| rolling_1 | 35 | 0.382521 | 0.325391 | 0.351624 | -14.46% | 45 |
| rolling_2 | 1 | 0.285237 | 0.258837 | 0.335706 | -31.43% | 54 |
| rolling_3 | 16 | 0.281623 | 0.325513 | 0.397635 | +2.86% | 39 |

rolling 2는 첫 epoch 이후 Validation Macro F1이 개선되지 않아 epoch 1이 선택되었다. rolling 1은 Validation에서 0.3825였지만 Test에서는 0.3254로 하락했고, 구간에 따라 일반화 성능이 달라졌다.

### 7.2 연결 Classification 결과

| Test samples | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---:|---:|---:|---:|---:|
| 6,291 | 0.483389 | 0.356864 | 0.318197 | 0.400513 |

| Class | Precision | Recall | F1 | 예측 비율 |
|---|---:|---:|---:|---:|
| SHORT | 0.2405 | 0.0551 | 0.0896 | 6.28% |
| HOLD | 0.5215 | 0.8533 | 0.6474 | 82.99% |
| LONG | 0.3304 | 0.1622 | 0.2176 | 10.73% |

Macro F1은 always-HOLD와 Ridge threshold baseline보다 rolling 1~3에서 모두 높았지만, 실제 SHORT의 5.51%와 LONG의 16.22%만 찾아냈다. 예측의 82.99%가 HOLD에 집중되어 class 균형 성능은 여전히 낮다.

### 7.3 연결 Backtest 결과

| Return | Sharpe-like | MDD | Trades | Trade Ratio | Win Rate | Avg Trade Return |
|---:|---:|---:|---:|---:|---:|---:|
| -39.67% | -1.890 | -51.54% | 138 | 2.19% | 42.03% | -0.335% |

rolling 3에서는 +2.86%를 기록했지만 rolling 1과 2의 손실이 컸다. 138개의 non-overlap 거래에서 승률은 42.03%였고 거래당 평균수익률도 음수였다. 따라서 직접 Classification이 Ridge보다 Macro F1을 개선했지만 안정적인 경제적 가치로 이어지지는 않았다.

## 8. 결과 해석

LSTM Classifier는 MSE가 아니라 Cross-Entropy로 class를 직접 학습했음에도 HOLD 집중과 낮은 SHORT/LONG recall이 나타났다. 이는 이전 회귀 학습 목적만이 성능 문제의 유일한 원인은 아니라는 증거다.

가능한 원인은 Chart-only 입력의 낮은 signal-to-noise ratio, label 불균형, 시장 regime 변화, 중첩 window로 인한 낮은 실질 독립 표본 수, 마지막 hidden state의 정보 병목 및 단일 seed 불확실성이다.

Class weight를 즉시 추가하면 Cryptova와 학습 조건이 달라지므로 주 실험 완료 전에 LSTM만 변경하지 않는다. 모든 모델 결과를 확보한 뒤 공통 ablation 규칙으로 검증해야 한다.

## 9. 보고서 핵심 문장

> 본 연구는 Cryptova와의 직접 신호 비교를 위해 과거 72시간의 12개 Chart Feature를 입력받고 미래 24시간의 SHORT/HOLD/LONG을 Cross-Entropy로 직접 학습하는 단방향 Many-to-One LSTM Classifier를 구성하였다.

> LSTM Classifier는 연결된 6,291개 out-of-sample 표본에서 Macro F1 0.3182와 Balanced Accuracy 0.3569를 기록하였다.

> LSTM Classifier는 Ridge threshold baseline보다 높은 Macro F1을 기록했지만 예측의 82.99%가 HOLD에 집중되었고 SHORT와 LONG recall은 각각 5.51%와 16.22%에 그쳤다.

> 동일한 거래비용과 24시간 non-overlap 규칙을 적용한 연결 Backtest 수익률은 -39.67%로, 직접 Classification 성능의 개선이 안정적인 투자성과로 이어지지는 않았다.
