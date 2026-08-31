# LSTM Regression 연구 노트

## 1. 정의

LSTM(Long Short-Term Memory)은 시계열을 한 시점씩 순서대로 처리하면서 hidden state와 cell state에 과거 정보를 전달하는 순환신경망이다. 입력·망각·출력 gate를 이용해 일반 RNN의 장기 의존성 및 gradient 소실 문제를 완화한다.

본 실험은 과거 72시간의 Chart Feature 12개를 입력받아 미래 24시간 수익률 하나를 출력하는 **단방향 Many-to-One LSTM Regression**이다.

```text
x(t-71) → x(t-70) → ... → x(t-1) → x(t)
                                      ↓
                             predicted_return 1개
```

이 모델의 공식 역할은 Regression Track이다. SHORT/HOLD/LONG Classification에는 별도의 `LSTM Classifier`를 사용한다.

## 2. 특징

- 72개 시점을 시간 순서대로 처리한다.
- 동일한 LSTM cell 파라미터를 모든 시점에 공유한다.
- gate와 비선형 활성함수를 통해 순차적·비선형 관계를 표현할 수 있다.
- 마지막 hidden state에 sequence 정보를 압축해 하나의 수익률을 예측한다.
- Ridge-Flat보다 표현력이 높지만 금융 noise나 특정 Train 구간의 패턴까지 학습할 위험이 있다.

LSTM의 표현력이 높다는 사실은 Test 성능이 Ridge보다 반드시 높다는 의미가 아니다. 반복 가능한 시간 패턴, 충분한 독립 표본 및 Train 이후에도 유지되는 시장 관계가 필요하다.

## 3. 문제 제기

Ridge-Flat은 `(72,12)` 입력을 864개의 열로 펼치고 고정된 선형계수를 적용한다. 시간 위치는 구분하지만 시점이 연결되며 형성하는 동적인 상태 변화를 별도 구조로 처리하지 않는다.

```text
Ridge-Flat
864개 값 → 고정 선형결합 → 미래 수익률

LSTM Regression
시점별 입력 → 상태 갱신 → 상태 갱신 → 마지막 상태 → 미래 수익률
```

LSTM 실험은 하락 후 반등, 거래량 증가 후 추세 변화처럼 여러 시점의 순서가 함께 의미를 갖는 관계가 Ridge보다 높은 out-of-sample 예측력으로 이어지는지 확인하기 위해 수행하였다.

## 4. 모델 구조

```text
Chart Input
(B, 72, 12)
      ↓
Unidirectional LSTM
input_size=12, hidden_size=32, num_layers=1
      ↓
Final Hidden State
(B, 32)
      ↓
Dropout(0.30)
(B, 32)
      ↓
Linear(32 → 1)
      ↓
Predicted Return
(B,)
```

| 항목 | 값 |
|---|---:|
| Sequence length | 72 |
| Input size | 12 |
| Hidden size | 32 |
| LSTM layers | 1 |
| Bidirectional | False |
| Output dropout | 0.30 |
| Regression head | `Linear(32,1)` |
| Trainable parameters | 5,921 |

마지막에 사용하는 것은 원본 입력의 마지막 12개 feature가 아니다. LSTM이 72시간을 순서대로 처리한 뒤 만든 마지막 hidden state `(B,32)`다. 다만 72시간 전체를 32차원으로 압축하므로 오래된 정보나 국소 사건이 약화될 수 있는 병목은 존재한다.

## 5. 실험 가정

### 5.1 데이터와 Target

- 입력: 기존 rolling dataset의 `X_chart_*.npy`, shape `(N,72,12)`
- target: `sample_meta_*.csv`의 `raw_future_return`
- 예측 horizon: 미래 24시간
- 추가 scaling: 사용하지 않음
- 이유: 각 rolling의 Train에서 fit된 scaler가 tensor에 이미 적용됨
- News Feature: 사용하지 않음

### 5.2 누수 방지

- 시간 순서를 유지한 rolling split을 사용한다.
- Train에서만 파라미터를 학습한다.
- Validation RMSE로 checkpoint를 선택한다.
- Test는 선택 완료 후 최종 평가에만 사용한다.
- Validation/Test scaler refit을 수행하지 않는다.

### 5.3 학습 설정

| 항목 | 값 |
|---|---:|
| Loss | MSE |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Batch size | 64 |
| Maximum epochs | 50 |
| Patience | 8 |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Device | CPU |

학습은 MSE를 최소화하지만 결과는 수익률과 같은 단위로 해석할 수 있도록 RMSE와 MAE로 보고한다. 같은 Validation 표본에서 MSE와 RMSE는 동일한 checkpoint 순서를 만든다.

## 6. 실험 과정

```text
X_chart_train + raw_future_return
              ↓
TensorDataset / DataLoader
              ↓
Many-to-One LSTM Regression
              ↓
MSE Loss → Backpropagation → Gradient clipping → AdamW
              ↓
Validation RMSE checkpoint 선택
              ↓
Test predicted_return
              ↓
RMSE·MAE·Correlation·Directional Accuracy
```

Regression 모델은 확률, softmax, class logits, confidence 또는 risk filter를 사용하지 않는다.

## 7. 결과

### 7.1 Rolling별 결과

| Rolling | 선택 epoch | Zero-return RMSE | Ridge RMSE | LSTM RMSE |
|---|---:|---:|---:|---:|
| rolling_1 | 14 | 0.015358 | 0.015569 | 0.016097 |
| rolling_2 | 24 | 0.023728 | 0.023963 | 0.025003 |
| rolling_3 | 30 | 0.028052 | 0.028301 | 0.028625 |

세 rolling 모두 `0%`를 항상 예측하는 zero-return baseline이 가장 낮은 RMSE를 기록했고 Ridge, LSTM 순으로 나타났다. LSTM은 zero-return baseline보다 rolling별로 약 4.81%, 5.37%, 2.04% 높은 RMSE를 기록했다.

### 7.2 연결 Out-of-Sample 결과

| Test samples | RMSE | MAE | Pearson | Spearman | Directional Accuracy |
|---:|---:|---:|---:|---:|---:|
| 6,291 | 0.023790 | 0.017203 | -0.0041 | -0.0478 | 46.97% |

연결 zero-return baseline의 RMSE는 `0.022948`이었다. LSTM RMSE는 약 3.67% 더 높았다. Pearson correlation이 거의 0이고 방향 정확도가 50%보다 낮아 예측값과 실제 미래 수익률 사이의 안정적인 관계는 확인되지 않았다.

### 7.3 해석

LSTM은 Ridge보다 높은 순차적·비선형 표현력을 갖지만 추가 표현력이 out-of-sample 수익률 예측 개선으로 이어지지 않았다. 가능한 원인으로는 Chart-only 신호의 낮은 signal-to-noise ratio, 24시간 horizon의 외부 충격, 1시간 간격으로 중첩된 window와 target, 시장 regime 변화 및 마지막 hidden state의 정보 병목이 있다.

이 결과는 모든 LSTM 구조가 실패한다는 증거가 아니다. 현재 고정한 hidden size 32, 1 layer, MSE, Seed 42 조건에서 단순 기준선을 넘지 못했다는 의미다.

## 8. 공식 비교에서의 역할

LSTM Regression은 Ridge Regression, TimesNet Regression, Chronos 및 TimesFM과 미래 수익률 예측력을 비교한다. 이 모델의 결과는 SHORT/HOLD/LONG 주 Classification 표나 Backtest에 사용하지 않는다. Cryptova와의 신호 비교에는 별도 `LSTM Classifier` 결과를 사용한다.

## 9. 보고서 핵심 문장

> 본 연구는 과거 72시간의 12개 Chart Feature를 순차적으로 처리하고 마지막 hidden state를 통해 미래 24시간 수익률 하나를 예측하는 단방향 Many-to-One LSTM Regression을 전통 시계열 forecasting baseline으로 사용하였다.

> LSTM Regression은 연결된 6,291개 out-of-sample 표본에서 RMSE 0.02379, MAE 0.01720 및 방향 정확도 46.97%를 기록하였다.

> 세 rolling 모두 LSTM의 Test RMSE가 zero-return baseline과 Ridge보다 높았으며, 순차적 비선형 표현력의 증가가 현재 Chart 데이터에서 안정적인 미래 수익률 예측 개선으로 이어지지는 않았다.

> LSTM Regression은 Regression Track 전용으로 사용하며, Cryptova와의 SHORT/HOLD/LONG 비교에는 직접 Cross-Entropy로 학습한 별도의 LSTM Classifier를 사용하였다.
