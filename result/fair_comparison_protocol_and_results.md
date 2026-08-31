# Financial Forecasting Benchmark 공정 비교 기준 및 결과 원장

## 문서 정보

- 문서 목적: 모든 비교 모델에 적용할 공통 실험 조건을 고정하고, 모델별 설정과 결과를 같은 형식으로 누적하여 최종 연구 결론을 도출한다.
- Protocol version: `1.18`
- 최초 고정일: `2026-08-27`
- 공통 예측 설정: 원칙적으로 각 모델은 관측 시점 기준 과거 72시간의 정보를 바탕으로 미래 24시간 수익률 또는 SHORT/HOLD/LONG 신호를 예측한다. 단, TimesFM 2.5는 모델의 32-step patch 제약으로 공통 72시간 범위 내 최근 64시간을 입력받아 미래 24시간을 예측한다.
- 주 연구 질문: **Cryptova는 기존 시계열 모델 및 foundation model과 비교하여 경쟁력이 있는가?**
- 보조 연구 질문:
  - 각 모델은 미래 24시간 수익률을 얼마나 정확히 예측하는가?
  - 각 모델은 SHORT/HOLD/LONG을 얼마나 잘 구분하는가?
  - 각 모델은 어떤 시장 regime에서 강하거나 약한가?

---

## 1. 비교의 범위와 해석

본 연구의 주 비교는 각 모델을 실제 사용 방식으로 구성한 뒤 동일한 시점, target 및 evaluator에서 평가하는 **end-to-end system comparison**이다.

```text
Regression Task
Ridge / LSTM-Reg / TimesNet-Reg / Chronos / TimesFM
→ predicted_return → RMSE·MAE·Correlation

Classification Task
Ridge / Chronos / TimesFM → return → fixed threshold ─┐
LSTM-Cls / TimesNet-Cls / Cryptova → direct class ─────┤
                                                       └→ 공통 분류·Backtest
```

### 1.1 Task 1 — 미래 수익률 예측

연구 질문은 다음과 같다.

> 각 forecasting model은 과거 72시간의 정보를 바탕으로 미래 24시간 수익률을 얼마나 정확하게 예측하는가?

Task 1의 공통 target과 출력은 연속값인 `raw_future_return`이다.

```text
Model Input
    ↓
Forecasting Model
    ↓
predicted_return
    ↓
RMSE / MAE / Pearson / Spearman / Directional Accuracy
```

| 모델 | 입력 | Task 1 출력 | 학습·사용 방식 | 참여 여부 |
|---|---|---|---|---|
| Ridge Regression | Chart `(72,12)` flatten | `predicted_return` | Ridge 회귀 | 참여 |
| LSTM Regression | Chart `(72,12)` | `predicted_return` | MSE 회귀학습 | 참여 |
| TimesNet Regression | Chart `(72,12)` | `predicted_return` | 회귀학습 | 참여 |
| Chronos-2 LoRA Fine-tuned | BTC close + Chart 12 covariates | forecast/return | 사전학습 모델 LoRA 적응 | 완료 |
| TimesFM 2.5 LoRA Fine-tuned | BTC close, 최근 64시간 | forecast/return | 사전학습 모델 LoRA 적응 | 완료 |
| LSTM Classifier | Chart `(72,12)` | 3-class logits | 직접 Classification | 제외 |
| TimesNet Classifier | Chart `(72,12)` | 3-class logits | 직접 Classification | 제외 |
| Cryptova | Chart+News | 3-class probability | 직접 Classification | 제외 |

Train 가능한 회귀모델은 Validation RMSE로 checkpoint 또는 hyperparameter를 선택한다. Cryptova와 직접 Classifier는 연속적인 수익률을 출력하지 않으므로 이 Task에 포함하지 않는다.

### 1.2 Task 2 — SHORT/HOLD/LONG Classification

연구 질문은 다음과 같다.

> 각 모델은 미래 24시간의 SHORT/HOLD/LONG 신호를 얼마나 정확하게 예측하며, 동일한 거래 조건에서 어떤 투자성과를 보이는가?

Task 2에는 두 가지 신호 생성 경로가 존재한다.

```text
Regression/Forecasting model
predicted_return → fixed threshold → SHORT/HOLD/LONG ─┐
                                                       ├→ 공통 Classification·Backtest
Direct Classifier                                     │
class logits/probability → argmax → SHORT/HOLD/LONG ──┘
```

| 모델 | Task 2 신호 생성 방식 | 학습 target | 참여 여부 |
|---|---|---|---|
| Ridge Regression | `predicted_return` → 고정 threshold | `raw_future_return` | 참여 |
| LSTM Classifier | 3-class logits → argmax | `label_id` | 참여 |
| TimesNet Classifier | 3-class logits → argmax | `label_id` | 참여 |
| Chronos-2 LoRA Fine-tuned | forecast → return → 고정 threshold | Validation quantile loss 기반 LoRA | 완료 |
| TimesFM 2.5 LoRA Fine-tuned | forecast → return → 고정 threshold | Validation 공식 loss 기반 LoRA | 완료 |
| Cryptova-Full | class probability → confidence/risk filter | `label_id` | 주 비교 참여 |
| Cryptova-Base | class probability → confidence threshold | `label_id` | Ablation 참여 |
| Cryptova-Raw | class probability → argmax | `label_id` | 순수 출력 ablation 참여 |
| LSTM Regression | 수익률 회귀 전용 | `raw_future_return` | 주 Classification에서 제외 |
| TimesNet Regression | 수익률 회귀 전용 | `raw_future_return` | 주 Classification에서 제외 |

고정 threshold는 모든 return 기반 모델에 동일하게 적용한다.

```text
predicted_return <= -0.012 → SHORT
-0.012 < predicted_return < +0.012 → HOLD
predicted_return >= +0.012 → LONG
```

직접 Classifier에는 예측 수익률 threshold를 적용하지 않는다. Dataset 생성 단계에서 동일한 ±1.2% 기준으로 만들어진 `label_id`를 직접 학습하고, 출력 확률의 argmax로 class를 결정한다.

공통 평가 지표는 Accuracy, Balanced Accuracy, Macro F1, Weighted F1, class별 Precision/Recall/F1, Confusion Matrix 및 동일한 non-overlap Backtest 결과다. Cryptova의 경쟁력에 관한 주 연구 질문은 이 Task의 Macro F1, class별 성능 및 Backtest를 중심으로 판단한다.

### 1.3 모델 역할 고정 원칙

1. LSTM Regression과 TimesNet Regression은 Task 1의 수익률 예측 전용 모델로 사용한다.
2. LSTM Classifier와 TimesNet Classifier는 Task 2의 직접 Classification 모델로 사용한다.
3. Ridge Regression, Chronos 및 TimesFM처럼 return을 출력하는 모델은 Task 2에서만 고정 threshold 변환을 허용한다.
4. Cryptova는 수익률 출력이 없으므로 Task 1에서 제외하고 Task 2에서 직접 Classification 모델로 평가한다.
5. Cryptova의 실제 최종 시스템은 Validation에서 선택한 confidence threshold와 funding/std risk filter까지 포함한 `Cryptova-Full`로 정의한다.
6. Funding/std Risk Filter를 제거하고 Validation confidence threshold까지만 적용한 `Cryptova-Base`는 risk 후처리 효과를 분리하기 위한 ablation으로 보고한다. 순수 argmax는 `Cryptova-Raw`로 구분한다.
7. 모델별 신호 생성 방식이 다르다는 사실을 결과표에 표시하고, 주 비교를 architecture-only가 아닌 end-to-end system comparison으로 해석한다.

Cryptova는 핵심 구조인 Chart+News Fusion을 유지한다. 따라서 주 비교 결과는 완성된 시스템의 경쟁력을 의미하며, 모델 구조만의 우수성이나 News의 단독 효과를 의미하지 않는다. 구조 및 입력 modality의 효과는 별도 ablation으로 분석한다.

### 가능한 결론

> 동일한 시장 기간, target 및 거래 조건에서 Cryptova가 다른 모델보다 높은 최종 신호 성능 또는 투자성과를 보였다.

### 주 비교만으로 단정할 수 없는 결론

- Cryptova의 신경망 구조 자체가 다른 구조보다 우수하다.
- News가 성능 차이의 유일한 원인이다.
- 직접 Classification이 Regression-to-Classification보다 항상 우수하다.

---

## 2. 공통으로 동일하게 설정한 조건

### 2.1 데이터 및 Target

| 항목 | 공통 설정 | 상태 |
|---|---|---|
| 시장 데이터 빈도 | 1시간 | 고정 |
| Chart 입력 window | 과거 72시간 | 고정 |
| 예측 horizon | 미래 24시간 | 고정 |
| Chart feature | 동일한 12개 | 고정 |
| 회귀 target | `raw_future_return` | 고정 |
| SHORT 기준 | 실제/예측 수익률 `<= -0.012` | 고정 |
| HOLD 기준 | `-0.012 < return < +0.012` | 고정 |
| LONG 기준 | 실제/예측 수익률 `>= +0.012` | 고정 |
| 시간대 | UTC | 고정 |

Chart 기반 supervised 모델에는 동일한 `(N, 72, 12)` tensor를 제공한다. Cryptova는 주 비교에서 모델의 핵심인 `(N, 72, 9)` News tensor를 추가로 사용한다. Foundation model이 공식 구조상 단변량 가격 입력만 지원하는 경우에는 공식 입력 형식을 사용하고 입력 차이를 결과 해석에 명시한다.

### 2.2 Rolling split

| Rolling | Train | Validation | Test |
|---|---|---|---|
| rolling_1 | 2024-01-01 ~ 2025-04-01 | 2025-04-01 ~ 2025-07-01 | 2025-07-01 ~ 2025-10-01 |
| rolling_2 | 2024-04-01 ~ 2025-07-01 | 2025-07-01 ~ 2025-10-01 | 2025-10-01 ~ 2026-01-01 |
| rolling_3 | 2024-07-01 ~ 2025-10-01 | 2025-10-01 ~ 2026-01-01 | 2026-01-01 ~ 2026-04-01 |

모든 모델은 동일한 sample timestamp를 사용한다. Input 시작·종료, sample time 및 target time이 해당 split 안에 포함되는 strict split 규칙을 유지한다.

### 2.3 전처리 및 누수 방지

| 항목 | 공통 규칙 |
|---|---|
| Scaler fit | 각 rolling의 Train 데이터만 사용 |
| Validation/Test | Train에서 fit된 scaler로만 transform |
| Validation/Test refit | 금지 |
| Hyperparameter 선택 | Validation만 사용 |
| Test 사용 | 최종 평가에만 사용 |
| 무작위 split | 사용하지 않음 |

Foundation model은 필요한 경우 공식 preprocessing을 사용하지만 Validation/Test 전체 통계에 fit하는 처리는 금지한다.

### 2.4 Trainable neural model의 공통 학습 budget

적용 대상: LSTM, TimesNet, Cryptova.

| 항목 | 공통 기준 |
|---|---:|
| Batch size | 64 |
| Maximum epochs | 50 |
| Early-stopping patience | 8 |
| Gradient clipping | 1.0 |
| 기본 재현 seed | 42 |
| Checkpoint 선택 | Validation only |

추가 안정성 분석에서 여러 seed를 사용한다면 LSTM, TimesNet 및 Cryptova에 동일한 seed 집합을 적용한다. 권장 seed 집합은 `42, 123, 2026`이다. Ridge는 결정적 해를 사용하므로 seed 변화에 따른 모델 결과가 없다.

### 2.5 공통 평가

#### Regression Track

적용 대상: Ridge, LSTM Regression, TimesNet Regression, Chronos, TimesFM.

- RMSE
- MAE
- Pearson correlation
- Spearman correlation
- Directional accuracy

Train 가능한 forecasting 모델은 Validation RMSE로 선택한 checkpoint 또는 hyperparameter 결과를 사용한다. Cryptova는 수익률 값을 출력하지 않으므로 Regression Track에서 제외한다.

#### Classification Track — Cryptova 주 비교

적용 대상: Ridge Regression, LSTM Classifier, TimesNet Classifier, Chronos, TimesFM, Cryptova.

- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1
- Class별 Precision/Recall/F1
- Confusion Matrix

LSTM Classifier, TimesNet Classifier 및 Cryptova는 `label_id`를 직접 학습하고 Validation Macro F1으로 checkpoint를 선택한다. Ridge Regression, Chronos 및 TimesFM은 예측 수익률에 고정 threshold를 적용해 class를 생성한다. LSTM Regression과 TimesNet Regression의 threshold 결과는 주 Classification 비교에서 제외한다. Test 결과는 선택에 사용하지 않는다.

#### Backtest Track

| 항목 | 공통 설정 |
|---|---:|
| SHORT position | -1 |
| HOLD position | 0 |
| LONG position | +1 |
| Fee | 0.001 |
| Slippage | 0.001 |
| 선택 거래당 총비용 | 0.002 |
| Position overlap | 금지 |
| Cooldown | 24시간 |

공통 지표:

- 누적수익률
- Sharpe-like
- Maximum Drawdown
- 거래 수 및 거래 비율
- 승률
- 평균 거래수익률

Cryptova의 주 비교 결과에는 실제 최종 시스템인 confidence filter와 funding/std 기반 risk filter를 적용한다. Risk Filter가 없고 confidence threshold까지만 적용된 출력은 `Cryptova-Base` ablation으로 분리한다. Confidence도 없는 단순 argmax는 `Cryptova-Raw`로 구분한다.

### 2.6 평가지표 용어 사전

이 절은 결과표의 숫자가 본 프로젝트에서 무엇을 의미하는지 설명한다. Classification 지표는
미래 24시간 수익률을 기준으로 만든 `SHORT/HOLD/LONG`을 평가하고, Regression 지표는 미래
24시간 실제 수익률과 예측 수익률을 직접 비교한다.

#### Classification 결과표 읽는 법

실제 class는 다음 기준으로 정의한다.

```text
raw_future_return <= -1.2% → SHORT
-1.2% < raw_future_return < +1.2% → HOLD
raw_future_return >= +1.2% → LONG
```

| 용어 | 이 프로젝트에서의 의미 | 값 해석 |
|---|---|---|
| Precision | 모델이 특정 신호라고 예측한 것 중 실제로 그 신호였던 비율 | 높을수록 해당 예측 신호의 신뢰도가 높음 |
| Recall | 실제 특정 신호 중 모델이 그 신호를 찾아낸 비율 | 높을수록 실제 기회를 적게 놓침 |
| F1 | Precision과 Recall의 조화평균 | 정확하게 예측하는 능력과 놓치지 않는 능력을 함께 평가 |
| Macro F1 | SHORT·HOLD·LONG의 F1을 동일한 비중으로 평균 | class 개수와 관계없이 세 신호의 종합 분류력을 평가; 높을수록 좋음 |
| Balanced Accuracy | SHORT·HOLD·LONG Recall의 단순 평균 | 각 실제 class를 얼마나 찾아내는지 동일 비중으로 평가; 높을수록 좋음 |
| SHORT Recall | 실제 미래 수익률이 -1.2% 이하였던 표본 중 SHORT로 맞힌 비율 | `0.2075`는 실제 SHORT 100개 중 약 21개를 탐지했다는 뜻 |
| HOLD Recall | 실제 미래 수익률이 -1.2%와 +1.2% 사이였던 표본 중 HOLD로 맞힌 비율 | 높기만 하면 좋은 것이 아니며 SHORT/LONG을 모두 HOLD로 보내지 않았는지 함께 확인 |
| LONG Recall | 실제 미래 수익률이 +1.2% 이상이었던 표본 중 LONG으로 맞힌 비율 | `0.3527`은 실제 LONG 100개 중 약 35개를 탐지했다는 뜻 |

Macro F1이 높다는 것은 세 신호를 같은 개수로 예측했다는 뜻이 아니다. **세 class의 F1을
동일한 중요도로 평가했을 때 평균 성능이 높다**는 뜻이다. Balanced Accuracy도 예측 비율의
균등함이 아니라 실제 SHORT/HOLD/LONG 각각의 Recall을 동일한 비중으로 평균한 값이다.

```text
Macro F1
= (SHORT F1 + HOLD F1 + LONG F1) / 3

Balanced Accuracy
= (SHORT Recall + HOLD Recall + LONG Recall) / 3
```

#### Regression 결과표 읽는 법

Regression 모델은 미래 24시간 수익률 `raw_future_return`을 연속적인 숫자로 예측한다.

```text
실제 수익률: +2.0%
예측 수익률: +0.3%

방향은 둘 다 양수이므로 Directional Accuracy에서는 정답
수익률 크기는 1.7%p 차이이므로 RMSE·MAE에는 오차로 반영
±1.2% Classification에서는 실제 LONG, 예측 HOLD이므로 오답
```

| 용어 | 이 프로젝트에서의 의미 | 값 해석 |
|---|---|---|
| Zero-return baseline | 입력이나 학습 없이 모든 표본의 미래 24시간 수익률을 `0%`로 예측하는 naive Regression 기준선 | 연결 OOS RMSE `0.022948`, MAE `0.016422`; 상수 예측이므로 Pearson·Spearman과 상승·하락 방향 예측은 `N/A` |
| Selection | 여러 checkpoint 또는 hyperparameter 중 최종 모델을 고른 Validation 기준 | Test 성능이 아니라 Validation만 사용; 낮은 loss/RMSE 또는 높은 Macro F1을 선택 |
| RMSE | 예측 수익률 오차를 제곱·평균한 뒤 제곱근을 계산 | 낮을수록 좋고 큰 오차에 더 큰 penalty; `0.023178`은 약 2.32%p 규모의 RMSE |
| MAE | 실제 수익률과 예측 수익률 차이의 절댓값 평균 | 낮을수록 좋음; `0.016620`은 평균적으로 약 1.66%p 차이 |
| Pearson | 실제 수익률과 예측 수익률 사이의 선형 상관계수 | `+1`에 가까울수록 같은 방향의 선형관계, `0`은 선형관계가 거의 없음, 음수는 반대 관계 |
| Spearman | 실제·예측 수익률의 크기 순위 사이 상관계수 | 높은 수익 구간을 상대적으로 높게 순위화하는지 평가; `+1`에 가까울수록 좋음 |
| Directional Accuracy | 예측 수익률과 실제 수익률의 부호가 같은 표본 비율 | 미래 24시간이 상승인지 하락인지 맞힌 비율; 수익률 크기와 ±1.2% 신호 정답 여부는 평가하지 않음 |

`Selection`은 평가 지표가 아니라 **최종 후보를 고른 규칙**이다.

| Selection 표기 | 선택 방법 |
|---|---|
| Validation RMSE | Validation RMSE가 가장 낮은 checkpoint 또는 alpha 선택 |
| Validation Macro F1 | Validation Macro F1이 가장 높은 checkpoint 또는 alpha 선택 |
| Validation quantile loss | Chronos-2 공식 Validation quantile loss가 가장 낮은 adapter 선택 |
| Validation official loss | TimesFM 공식 normalized MSE + quantile loss가 가장 낮은 adapter 선택 |

Pearson과 Spearman은 오차의 크기를 측정하지 않는다. 예측값이 실제값보다 작더라도 함께
오르내리거나 순서를 잘 맞히면 높은 상관을 가질 수 있다. 반대로 RMSE가 낮아도 예측값이
0% 근처로 수축하면 상관관계와 SHORT/LONG 신호 성능은 낮을 수 있다. 따라서 본 연구에서는
RMSE·MAE, correlation, Directional Accuracy 및 Classification을 함께 해석한다.

---

## 3. 모델마다 달라도 되는 조건

모델 구조가 다르므로 다음 값을 억지로 동일하게 만들지 않는다. 대신 실제 설정과 trainable parameter 수를 모두 보고한다.

| 항목 | 동일하게 강제하지 않는 이유 |
|---|---|
| Hidden dimension | 구조별 hidden state의 역할이 다름 |
| Layer 수 | LSTM, TimesNet, Transformer layer의 계산 의미가 다름 |
| Attention head | Attention 기반 모델에만 존재 |
| FFT top-k | TimesNet에만 존재 |
| Kernel 수 | CNN/TimesNet에만 존재 |
| Optimizer | 학습 모델에만 존재하며 구조별 최적 조건이 다름 |
| Learning rate | 구조별 최적 scale이 다름 |
| Loss | Regression과 직접 Classification의 task가 다름 |
| Foundation preprocessing | 사전학습 모델의 공식 입력 규칙을 따라야 함 |

### 공정성 원칙

1. 모델마다 다른 hyperparameter는 Test가 아닌 Validation으로만 선택한다.
2. 한 모델에만 과도한 탐색 기회를 제공하지 않는다.
3. 모델별 탐색 후보 수와 선택 기준을 사전에 기록한다.
4. 동일 숫자의 hidden size가 동일한 모델 capacity를 의미하지 않으므로 전체 trainable parameter 수도 함께 보고한다.
5. 입력 modality가 다른 경우 이를 숨기지 않고 결과표에 명시한다.

---

## 4. 모델별 설정 및 결과

모든 모델은 아래 형식을 동일하게 사용한다.

### 4.1 Ridge-Flat

#### 설정

| 항목 | 값 |
|---|---|
| 상태 | 완료 |
| 입력 | Chart `(N, 72, 12)` |
| 변환 | Flatten `(N, 864)` |
| Target | `raw_future_return` |
| 모델 | Ridge Regression |
| Solver | NumPy SVD, explicit inverse 미사용 |
| Alpha 후보 | `1e-8` ~ `1e10`, 총 19개 |
| Regression 선택 | Validation RMSE 최소 |
| Classification 선택 | Validation Macro F1 최대 |
| Macro F1 동점 처리 | Validation RMSE가 낮은 alpha |
| 추가 scaling | 없음; 기존 rolling train-fitted Chart scaler 사용 |

#### Regression 결과 — RMSE-selected(수익률을 얼마나 잘 맞췄는가)

| Rolling | 선택 alpha | Test RMSE | Test Macro F1 | 거래 수 |
|---|---:|---:|---:|---:|
| rolling_1 | 10,000 | 0.015569 | 0.259890 | 0 |
| rolling_2 | 10,000,000,000 | 0.023963 | 0.212844 | 0 |
| rolling_3 | 10,000,000,000 | 0.028301 | 0.194397 | 0 |

RMSE-selected는 강한 규제를 선택하여 예측 수익률이 평균 근처로 축소됐고 세 Test 구간 모두 HOLD만 예측했다. 이는 수익률 숫자 오차를 줄이는 목적과 거래 신호 구분 목적이 동일하지 않음을 보여준다.

#### Classification 및 Backtest 결과 — Macro-F1-selected(신호를 얼마나 잘 구분했는가)

| Rolling | 선택 alpha | Test Macro F1 | Test RMSE | 누적수익률 | 거래 수 |
|---|---:|---:|---:|---:|---:|
| rolling_1 | 0.001 | 0.297379 | 0.016537 | -0.007684 | 17 |
| rolling_2 | 0.1 | 0.245693 | 0.025633 | -0.086828 | 32 |
| rolling_3 | 0.0001 | 0.283922 | 0.029549 | +0.103506 | 37 |

#### 연결 Out-of-Sample Backtest

| 지표 | 결과 |
|---|---:|
| 기간 | 2025-07 ~ 2026-03 |
| 누적수익률 | -0.005% |
| Sharpe-like | 0.125 |
| MDD | -14.42% |
| 거래 수 | 86 |
| 승률 | 50.0% |
| 평균 거래수익률 | +0.026% |

#### 해석

Ridge-Flat은 항상 HOLD인 기준선보다 Macro F1이 소폭 높았지만 SHORT와 LONG recall이 낮았다. 예측 수익률과 실제 수익률의 correlation 및 방향 정확도도 0과 50% 근처였고, rolling별 투자성과가 일관되지 않았다. 따라서 안정적인 선형 예측 신호나 수익성은 확인되지 않았지만 이후 모델이 넘어야 할 규제 선형 baseline으로는 유효하다.

### 4.2 LSTM Regression

#### 설정

| 항목 | 값 |
|---|---|
| 상태 | Seed 42, rolling 1~3 실행 완료 |
| 입력 | Chart `(N, 72, 12)` |
| Input size | 12 |
| Sequence length | 72 |
| 구조 | 단방향 Many-to-One LSTM |
| Hidden size | 32 |
| LSTM layers | 1 |
| Output dropout | 0.30 |
| Output | `predicted_return` 1개 |
| Regression head | `Linear(32, 1)` |
| Trainable parameters | 5,921 |
| Loss | MSE |
| Optimizer | AdamW |
| Learning rate / weight decay | `1e-4` / `1e-4` |
| Maximum epochs | 50 |
| Patience | 8 |
| Batch size | 64 |
| Gradient clipping | 1.0 |
| Checkpoint 선택 | Validation RMSE 최소 |
| 공식 역할 | Regression Track 전용 |

#### Regression 결과 — RMSE-selected

| Rolling | Seed | Best epoch | LSTM Test RMSE |
|---|---:|---:|---:|
| rolling_1 | 42 | 14 | 0.016097 |
| rolling_2 | 42 | 24 | 0.025003 |
| rolling_3 | 42 | 30 | 0.028625 |

#### 연결 Out-of-Sample Regression 결과

| Test rows | RMSE | MAE | Pearson | Spearman | Directional Accuracy |
|---:|---:|---:|---:|---:|---:|
| 6,291 | 0.023790 | 0.017203 | -0.0041 | -0.0478 | 46.97% |

#### 해석

LSTM Regression은 세 rolling 모두 zero-return baseline과 Ridge보다 높은 Test RMSE를 기록했다. 연결 Test Pearson correlation은 -0.0041이고 방향 정확도는 46.97%로, Seed 42 기준에서 안정적인 미래 24시간 수익률 예측 신호는 확인되지 않았다. 이 모델의 threshold Classification 결과는 공식 비교에서 사용하지 않는다.

### 4.3 LSTM Classifier

#### 설정

| 항목 | 값 |
|---|---|
| 상태 | Seed 42, rolling 1~3 실행 완료 |
| 입력 | Chart `(N, 72, 12)` |
| 구조 | 단방향 Many-to-One LSTM |
| Hidden size / layers | 32 / 1 |
| Output dropout | 0.30 |
| Classification head | `Linear(32, 3)` |
| Trainable parameters | 5,987 |
| Target | `label_id` |
| Loss | Cross-Entropy, label smoothing 0.03 |
| Class weights | 사용하지 않음 |
| Optimizer | AdamW, `lr=1e-4`, `weight_decay=1e-4` |
| Maximum epochs / patience | 50 / 8 |
| Batch size / gradient clipping | 64 / 1.0 |
| Checkpoint 선택 | Validation Macro F1 최대; 동점 시 Validation Cross-Entropy 최소 |
| Confidence/risk filter | 사용하지 않음 |

#### Rolling별 Test 결과

| Rolling | Best epoch | Test Macro F1 | Balanced Accuracy | Return | Sharpe-like | MDD | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 35 | 0.325391 | 0.351624 | -0.144594 | -3.139 | -0.150669 | 45 |
| rolling_2 | 1 | 0.258837 | 0.335706 | -0.314331 | -3.835 | -0.339820 | 54 |
| rolling_3 | 16 | 0.325513 | 0.397635 | +0.028584 | +0.500 | -0.197746 | 39 |

#### 연결 Out-of-Sample 결과

| Test rows | Accuracy | Balanced Accuracy | Macro F1 | SHORT Recall | HOLD Recall | LONG Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 6,291 | 0.483389 | 0.356864 | 0.318197 | 0.0551 | 0.8533 | 0.1622 |

| Return | Sharpe-like | MDD | Trades | Trade Ratio | Win Rate | Avg Trade Return |
|---:|---:|---:|---:|---:|---:|---:|
| -39.67% | -1.890 | -51.54% | 138 | 2.19% | 42.03% | -0.335% |

#### 해석

LSTM Classifier는 Ridge threshold 결과보다 세 rolling 모두 높은 Macro F1을 기록했지만 SHORT recall은 5.51%, LONG recall은 16.22%에 그쳤고 예측의 82.99%가 HOLD였다. rolling 3만 양의 수익률을 보였으며 연결 Backtest는 -39.67%로, 직접 Classification 학습이 안정적인 투자성과로 이어지지는 않았다.

### 4.4 TimesNet Regression

#### 설정

| 항목 | 값 |
|---|---|
| 상태 | 구현·rolling 1~3 평가 완료 |
| 입력 | Chart `(N, 72, 12)` |
| Output | `predicted_return` 1개 |
| TimesNet core | THUML 공식 FFT·Inception 2D CNN·adaptive aggregation·residual/post-LayerNorm |
| Head | GELU → Dropout → Flatten → Linear(`72×32`, 1) |
| Loss | MSE |
| TimesNet layer | 1 |
| `top_k` | 2 |
| Hidden / convolution hidden | 32 / 64 |
| Inception kernel 수 | 4 |
| Dropout | 0.30 |
| Maximum epochs | 50 |
| Patience | 8 |
| Batch size | 64 |
| Optimizer | AdamW (`lr=1e-4`, `weight_decay=1e-4`) |
| Regression 선택 | Validation RMSE 최소 checkpoint |
| 공식 역할 | Regression Track 전용 |

#### 결과

| Rolling | Seed | Best epoch | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| rolling_1 | 42 | 12 | 0.016193 | 0.012278 |
| rolling_2 | 42 | 23 | 0.025021 | 0.018312 |
| rolling_3 | 42 | 10 | 0.028873 | 0.021455 |

#### 해석

연결 OOS RMSE는 `0.023916`으로 Ridge-Flat(`0.023178`)과 LSTM Regression(`0.023790`)보다 높았다. 반면 방향 정확도는 `48.24%`로 세 모델 중 가장 높았지만 여전히 50%를 넘지 못했고 Pearson과 Spearman도 각각 `-0.0223`, `-0.0350`으로 안정적인 수익률 예측 관계를 보여주지 못했다.

### 4.5 TimesNet Classifier

#### 설정

| 항목 | 값 |
|---|---|
| 상태 | 구현·rolling 1~3 평가 완료 |
| 입력 | Chart `(N, 72, 12)` |
| Output | SHORT/HOLD/LONG logits |
| Encoder | Regression과 동일한 official-core TimesNet |
| Head | GELU → Dropout → Flatten → Linear(`72×32`, 3) |
| Loss | Cross-Entropy (`label_smoothing=0.03`, class weight 없음) |
| Checkpoint 선택 | Validation Macro F1 최대 |
| 공식 역할 | Classification Track |

#### 결과

| Rolling | Best epoch | Test Macro F1 | Balanced Accuracy | Return | Sharpe-like | MDD | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 8 | 0.300942 | 0.321828 | -14.46% | -2.364 | -15.63% | 56 |
| rolling_2 | 41 | 0.380404 | 0.380921 | +1.08% | +0.324 | -14.69% | 72 |
| rolling_3 | 7 | 0.335989 | 0.374361 | -3.10% | -0.099 | -22.93% | 59 |

연결 OOS 6,291개에서 Macro F1 `0.364654`, Balanced Accuracy `0.366543`을 기록해 현재까지 Ridge와 LSTM보다 가장 높은 분류 성능을 보였다. SHORT/HOLD/LONG recall은 `0.2075 / 0.6230 / 0.2691`로 HOLD 편향도 완화됐다. 그러나 비용 반영 연결수익률은 `-16.21%`였으며, 평균 거래수익률은 비용 차감 후 `-0.066%`로 분류 개선이 순수익으로 이어지지는 않았다.

### 4.6 Chronos-2 LoRA Fine-tuned

#### 설정

| 항목 | 값 |
|---|---|
| 상태 | Colab 학습 및 Rolling 1~3·Connected OOS 평가 완료 |
| Checkpoint | `amazon/chronos-2` |
| Revision | `95a9710e2596287d08352589f42634fa5abdf0a7` |
| Target | raw BTC `close` 시계열 |
| Past covariates | 기존 Chart feature 12개 |
| Context | 72시간 |
| Forecast horizon | 24시간 |
| Fine-tuning | 공식 `finetune_mode="lora"`; Full fine-tuning 아님 |
| LoRA | 공식 기본값 `r=8`, `alpha=16`, q/k/v/o 및 output head |
| Learning rate / steps | `1e-5` / 최대 `1000` steps |
| Batch size | 32 variates; 일반 sample batch와 의미가 다름 |
| Checkpoint 선택 | Rolling Validation quantile loss |
| Output 변환 | `q=0.5`의 `t+24` close → 24시간 예측 수익률 |
| Classification 변환 | 고정 ±1.2% threshold |
| 실행 환경 | Google Colab GPU |
| Seed | 42 |
| Trainable / Total parameter | `1,206,912 / 120,684,576` |
| 저장 모델 | Rolling별 `finetuned-ckpt/adapter_model.safetensors` |

#### 결과

| Rolling | Model version | Test RMSE | Test Macro F1 | Return | Sharpe-like | MDD | Trades |
|---|---|---:|---:|---:|---:|---:|---:|
| rolling_1 | `amazon_chronos2_lora_qforecast_v1` | 0.015879 | 0.259189 | -0.66% | -0.349 | -2.61% | 6 |
| rolling_2 | `amazon_chronos2_lora_qforecast_v1` | 0.024393 | 0.267978 | -8.70% | -1.504 | -18.20% | 28 |
| rolling_3 | `amazon_chronos2_lora_qforecast_v1` | 0.030530 | 0.213475 | -16.90% | -1.913 | -17.03% | 25 |

#### 해석

Chart feature 12개를 past covariate로 사용하는 Chart-only Track이다. `close`는 예측 target이며
12개 feature에 추가된 feature로 세지 않는다. 기존 Rolling Train에만 fit된 scaler로
Chart covariate를 변환하고, Chronos-2의 공식 instance normalization도 유지했다.
LoRA는 전체 `120,684,576`개 parameter 중 `1,206,912`개만 학습해 계산·메모리 비용과
제한된 데이터에서의 과적합 위험을 줄이고 사전학습 표현을 보존하기 위해 선택했다.

Connected OOS 6,291개에서 RMSE `0.024300`, MAE `0.017148`, Macro F1 `0.252000`,
Balanced Accuracy `0.338928`을 기록했다. 예측의 `95.74%`가 HOLD였고 SHORT/LONG recall은
각각 `0.0220 / 0.0233`에 그쳤다. 따라서 고정 ±1.2% threshold 아래에서 방향성 신호를
거의 만들지 못했다. 비용 반영 연결수익률은 `-24.64%`, Sharpe-like는 `-1.406`,
MDD는 `-24.75%`였다. Rolling 2에서만 방향 정확도 `52.30%`와 양의 Pearson `0.0455`를
보였지만 Rolling 3에서는 Pearson `-0.2415`로 악화되어 구간 간 안정성도 확인되지 않았다.

### 4.7 TimesFM 2.5 LoRA Fine-tuned

#### 설정

| 항목 | 값 |
|---|---|
| 상태 | Colab 학습 및 Rolling 1~3·Connected OOS 평가 완료 |
| Checkpoint | `google/timesfm-2.5-200m-transformers` |
| Revision | `5a9806b9b291fad9233b5249d88263f1846304d3` |
| 입력 | raw BTC `close` 단변량 |
| 공통 가용 context | 72시간 |
| 실제 모델 context | 최근 64시간; patch length 32 제약 |
| Forecast horizon | 24시간 |
| Fine-tuning | LoRA `r=4`, `alpha=8`, dropout `0.05`, all-linear |
| Full fine-tuning | 아님 |
| Epoch / patience | 최대 10 / 3 |
| Batch | micro 16 × accumulation 2 = effective 32 |
| Optimizer | AdamW, `lr=1e-4`, `weight_decay=0.01` |
| Checkpoint 선택 | Rolling Validation 공식 normalized MSE + quantile loss |
| Output 변환 | `mean_predictions`의 t+24 close → 24시간 예측 수익률 |
| Classification 변환 | 고정 ±1.2% threshold |
| 실행 환경 | Google Colab, NVIDIA A100-SXM4-40GB |
| Seed | 42 |
| Trainable / Total parameter | `1,382,912 / 232,672,192` |
| Best epoch | Rolling 1/2/3 모두 1 |
| 저장 모델 | Rolling별 `adapter/adapter_model.safetensors` |

#### 결과

| Rolling | Model version | Test RMSE | Test Macro F1 | Return | Sharpe-like | MDD | Trades |
|---|---|---:|---:|---:|---:|---:|---:|
| rolling_1 | `timesfm2_5_lora_close_v1` | 0.016446 | 0.278344 | +1.04% | +0.400 | -5.27% | 14 |
| rolling_2 | `timesfm2_5_lora_close_v1` | 0.025565 | 0.293757 | -4.17% | -0.313 | -20.27% | 46 |
| rolling_3 | `timesfm2_5_lora_close_v1` | 0.030939 | 0.260042 | -28.70% | -2.944 | -29.39% | 44 |

#### 해석

TimesFM 2.5의 공식 최대 context는 16,384 시점이지만 32-step patch를 사용하므로 공통
72시간을 그대로 사용하지 않고 그 범위 안의 최근 64시간을 입력했다. 96시간은 다른 모델보다
추가 과거 정보를 사용하고, 0-padding은 내부 normalization 통계를 왜곡할 수 있기 때문이다.
따라서 64시간은 모델 한계가 아니라 현재 공정 비교 protocol의 model-native 입력 선택이다.

Connected OOS 6,291개에서 RMSE `0.024992`, MAE `0.017804`, 방향 정확도 `49.83%`를
기록했다. 방향 정확도는 Regression 모델 중 수치상 가장 높지만 50% 미만이며, RMSE와 MAE는
비교 회귀모델 중 가장 컸다. 예측의 `89.38%`가 HOLD였고 SHORT/LONG recall은 각각
`0.0574 / 0.0625`에 그쳐 Macro F1은 `0.287013`이었다. 비용 반영 연결수익률은
`-30.96%`, Sharpe-like는 `-1.346`, MDD는 `-41.22%`였다.

세 Rolling 모두 첫 epoch에서 Validation loss가 가장 낮고 이후 Train loss 감소와 함께
Validation loss가 증가했다. LoRA가 parameter 수를 약 0.59%로 제한했음에도 현재 BTC
close-only 데이터에서는 매우 빠른 과적합이 나타났다. 특히 Rolling 3의 RMSE `0.030939`,
Pearson `-0.1825`, 수익률 `-28.70%`가 Connected OOS 악화의 주요 원인이었다.

### 4.8 Cryptova-Full

#### 설정

| 항목 | 값 |
|---|---|
| 상태 | 기존 예측 보존·공통 schema 재평가 완료 |
| Chart 입력 | `(N, 72, 12)` |
| News 입력 | `(N, 72, 9)` |
| Chart hidden | 32 |
| Chart convolution hidden | 64 |
| Chart encoder layers | 1 |
| News hidden | 32 |
| News heads/layers | 4 / 1 |
| Fusion hidden | 64 |
| Fusion heads/layers | 4 / 1 |
| Classifier hidden | 64 |
| Dropout | 0.30 |
| Batch size | 64 |
| Maximum epochs | 50 |
| Patience | 8 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Gradient clipping | 1.0 |
| Checkpoint 선택 | Validation Macro F1 최대 |
| Confidence threshold | Rolling 1/2/3: `0.40 / 0.36 / 0.46` |
| Risk filter | raw `funding_rate` + `std_24h` |
| Funding threshold | Rolling 1/2/3: `0 / 0.00005 / 0` |
| Volatility threshold | Rolling 1/2/3: `0.01 / 0.008 / 0.01` |
| 최종 prediction | 기존 `pred_filtered` |
| 재평가 방식 | 기존 prediction 재사용; 재학습·재추론 없음 |

#### 결과

| Rolling | Seed | Best epoch | Test Accuracy | Test Macro F1 | Return | Sharpe-like | MDD | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 42 | 1 | 0.639375 | 0.261728 | -0.13% | -0.107 | -1.53% | 2 |
| rolling_2 | 42 | 1 | 0.409844 | 0.294403 | -18.04% | -1.919 | -24.40% | 61 |
| rolling_3 | 42 | 42 | 0.382567 | 0.349761 | +55.72% | +4.321 | -9.36% | 56 |

#### 해석

기존 Risk Filter `result.json`의 Rolling별 Backtest 결과와 공통 evaluator 재평가가 일치했다. Connected OOS에서는 Macro F1 `0.350898`, 수익률 `+27.46%`, Sharpe-like `+1.143`, MDD `-24.40%`를 기록했다. 다만 전체 양의 수익은 rolling 3의 `+55.72%`에 크게 의존하며 rolling 2에서는 `-18.04%` 손실이 발생했다. 따라서 수익성은 확인됐지만 rolling 간 안정성은 추가 regime 분석에서 검증해야 한다.

### 4.9 Cryptova-Base — Ablation

| 항목 | 내용 |
|---|---|
| Base model | Cryptova Chart+News Fusion |
| Confidence threshold | Rolling 1/2/3: `0.40 / 0.36 / 0.46` |
| Risk filter | 적용하지 않음 |
| 최종 prediction | 기존 `pred_base` |
| 주 비교 포함 여부 | 제외; Risk Filter ablation으로 사용 |
| 재평가 방식 | 기존 prediction 재사용; 재학습·재추론 없음 |

| Rolling | Accuracy | Balanced Accuracy | Macro F1 | Return | Sharpe-like | MDD | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 0.633696 | 0.338434 | 0.278670 | -2.73% | -0.595 | -6.61% | 20 |
| rolling_2 | 0.386654 | 0.350358 | 0.294778 | -28.24% | -3.213 | -31.15% | 68 |
| rolling_3 | 0.391768 | 0.391279 | 0.390477 | +53.88% | +3.695 | -11.40% | 70 |

Connected OOS Macro F1은 `0.376506`, Balanced Accuracy는 `0.389647`였으며 비용 반영
수익률은 `+7.42%`, MDD는 `-37.38%`였다.
Full Risk Filter는 Base보다 분류 성능과 LONG recall을 낮추는 대신 수익률, Sharpe-like와
MDD를 개선했다. 따라서 Full 개선은 분류 정확도보다 거래 선택 및 위험 관리 효과다.

### 4.10 Cryptova-Raw — Argmax Ablation

| 항목 | 내용 |
|---|---|
| Base model | Cryptova Chart+News Fusion |
| 최종 prediction | 기존 `y_pred_argmax` |
| Confidence threshold | 적용하지 않음 |
| Risk filter | 적용하지 않음 |
| 재평가 방식 | 기존 prediction 재사용; 재학습·재추론 없음 |

| Rolling | Accuracy | Balanced Accuracy | Macro F1 | Return | Sharpe-like | MDD | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 0.621391 | 0.350653 | 0.309426 | -9.95% | -1.949 | -12.70% | 38 |
| rolling_2 | 0.380028 | 0.347500 | 0.293626 | -24.90% | -2.649 | -33.41% | 68 |
| rolling_3 | 0.379177 | 0.390435 | 0.382108 | +21.10% | +1.980 | -12.44% | 72 |

Connected OOS Macro F1은 `0.381875`, Balanced Accuracy는 `0.393802`로 현재 완료 모델 중
가장 높았다. 그러나 Backtest는 수익률 `-18.11%`, Sharpe-like `-0.545`였다. Raw는
분류 성능이 가장 높지만 신호의 경제적 가치는 비용 반영 후 음수였다.

---

## 5. 최종 통합 결과표

### 5.1 Regression Track

| Model | Input | Selection | RMSE | MAE | Pearson | Spearman | Directional Accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| Zero-return baseline | No input | None; always predicts `0%` | **0.022948** | **0.016422** | N/A | N/A | N/A |
| Ridge-Flat | Chart | Validation RMSE | 0.023178 | 0.016620 | -0.0471 | -0.0454 | 46.84% |
| LSTM | Chart | Validation RMSE | 0.023790 | 0.017203 | -0.0041 | -0.0478 | 46.97% |
| TimesNet | Chart | Validation RMSE | 0.023916 | 0.017317 | -0.0223 | -0.0350 | 48.24% |
| Chronos-2 LoRA Fine-tuned | Close target + Chart 12 past covariates | Validation quantile loss | 0.024300 | 0.017148 | -0.1095 | -0.0305 | 49.09% |
| TimesFM 2.5 LoRA Fine-tuned | Close, latest 64h | Validation official loss | 0.024992 | 0.017804 | -0.0973 | -0.0501 | 49.83% |

Zero-return baseline은 연결 OOS의 모든 실제 수익률에 대해 `0%`를 예측해 계산했다. 전체
Regression 비교에서 RMSE와 MAE가 가장 낮았으며, Ridge-Flat은 **학습된 모델 중** 가장 낮은
오차를 기록했지만 이 naive baseline을 능가하지 못했다. 상수 예측에는 수익률의 순위·선형관계나
상승·하락 방향 예측이 존재하지 않으므로 Pearson, Spearman 및 Directional Accuracy를 `N/A`로
표기한다.

### 5.2 Classification Track — Cryptova 주 비교

| Model | Input | Signal generation | Macro F1 | Balanced Accuracy | SHORT Recall | HOLD Recall | LONG Recall |
|---|---|---|---:|---:|---:|---:|---:|
| Ridge-Flat | Chart | Return → threshold | 0.284691 | 0.343804 | 0.0342 | 0.9063 | 0.0909 |
| LSTM Classifier | Chart | Direct classification | 0.318197 | 0.356864 | 0.0551 | 0.8533 | 0.1622 |
| TimesNet Classifier | Chart | Direct classification | 0.364654 | 0.366543 | 0.2075 | 0.6230 | 0.2691 |
| Chronos-2 LoRA Fine-tuned | Close target + Chart 12 | Forecast return → threshold | 0.252000 | 0.338928 | 0.0220 | 0.9715 | 0.0233 |
| TimesFM 2.5 LoRA Fine-tuned | Close, latest 64h | Forecast return → threshold | 0.287013 | 0.345735 | 0.0574 | 0.9173 | 0.0625 |
| Cryptova-Full | Chart+News | Direct class → confidence + funding/std filter | 0.350898 | 0.368649 | 0.1426 | 0.7910 | 0.1724 |
| Cryptova-Base | Chart+News | Direct class → confidence threshold | 0.376506 | 0.389647 | 0.1426 | 0.7202 | 0.3062 |
| Cryptova-Raw | Chart+News | Direct classification argmax | **0.381875** | **0.393802** | 0.1571 | 0.6716 | **0.3527** |

### 5.3 연결 Out-of-Sample Backtest

| Model | Return | Sharpe-like | MDD | Trades | Trade Ratio | Win Rate | Avg Trade Return |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge-Flat | -0.005% | 0.125 | -14.42% | 86 | 1.37% | 50.0% | +0.026% |
| LSTM Classifier | -39.67% | -1.890 | -51.54% | 138 | 2.19% | 42.03% | -0.335% |
| TimesNet Classifier | -16.21% | -0.449 | -23.75% | 187 | 2.97% | 48.13% | -0.066% |
| Chronos-2 LoRA Fine-tuned | -24.64% | -1.406 | -24.75% | 59 | 0.94% | 35.59% | -0.439% |
| TimesFM 2.5 LoRA Fine-tuned | -30.96% | -1.346 | -41.22% | 104 | 1.65% | 50.00% | -0.316% |
| Cryptova-Full | +27.46% | +1.143 | -24.40% | 119 | 1.89% | 55.46% | +0.240% |
| Cryptova-Base | +7.42% | +0.446 | -37.38% | 158 | 2.51% | 48.10% | +0.080% |
| Cryptova-Raw | -18.11% | -0.545 | -36.74% | 178 | 2.83% | 48.31% | -0.083% |

Ridge-Flat의 Regression 지표는 각 rolling에서 **Validation RMSE로 선택한 모델**의 test prediction 6,291개를 연결해 계산했다. Classification 및 Backtest 지표는 **Validation Macro F1으로 선택한 모델**의 동일한 연결 OOS 구간을 사용했다. 따라서 두 표의 Ridge-Flat은 같은 학습 방식이지만 연구 목적에 따라 선택된 `alpha` variant가 다르다.

### 5.4 Model size 및 계산량

| Model | Trainable Parameters | Training Time | Inference Time | Peak Memory | Seed 수 |
|---|---:|---:|---:|---:|---:|
| Ridge-Flat | 865 수준(864 coefficients+intercept) |  |  |  | 결정적 |
| LSTM Regression | 5,921 | 156.6초(RMSE 전용 재실행, CPU) | 미측정 | 미측정 | 1 |
| LSTM Classifier | 5,987 | 145.3초(CPU) | 미측정 | 미측정 | 1 |
| TimesNet Regression | 347,969 | 5,080.7초(CPU, Classifier와 병렬 실행) | 미측정 | 미측정 | 1 |
| TimesNet Classifier | 352,579 | 5,897.6초(CPU, Regression과 병렬 실행) | 미측정 | 미측정 | 1 |
| Chronos-2 LoRA Fine-tuned | 1,206,912 / 전체 120,684,576 | Rolling별 153.5~153.9초, 총 461.3초(Colab A100) | 미측정 | 미측정 | 1 |
| TimesFM 2.5 LoRA Fine-tuned | 1,382,912 / 전체 232,672,192 | Rolling별 1,011.8~1,016.7초, 총 3,041.8초(Colab A100) | 미측정 | 미측정 | 1 |
| Cryptova-Full | 기존 checkpoint 보존 | 기존 실험 재사용 | 재평가만 수행 | 미측정 | 1 |
| Cryptova-Base | Cryptova-Full과 동일 checkpoint | 추가 학습 없음 | 재평가만 수행 | 미측정 | 1 |
| Cryptova-Raw | Cryptova-Full과 동일 checkpoint | 추가 학습 없음 | 재평가만 수행 | 미측정 | 1 |

TimesNet 두 task는 CPU에서 동시에 실행했으므로 위 elapsed time은 CPU 자원 경합의 영향을 받는다. 모델 파라미터 수는 직접 비교할 수 있지만, LSTM과의 순수 학습속도 비교에는 동일한 단독 실행 환경에서 별도 측정이 필요하다.

### 5.5 현재까지의 비교 분석

| 관점 | 현재 우세 모델 | 해석 |
|---|---|---|
| Regression RMSE·MAE | Zero-return baseline; 학습 모델 중 Ridge-Flat | 모든 학습 모델이 항상 `0%`를 예측하는 naive baseline의 오차를 낮추지 못함 |
| Regression Directional Accuracy | TimesFM 2.5 | 49.83%로 수치상 가장 높지만 50% 미만 |
| Chronos-2 Regression | 기존 세 baseline보다 열세 | RMSE 0.024300, 음의 Pearson -0.1095로 안정적인 수익률 예측 관계를 확인하지 못함 |
| TimesFM 2.5 Regression | 현재 Regression 모델 중 RMSE·MAE 최하위 | 방향 정확도는 상대적으로 높지만 수익률 크기와 상관관계 예측은 불안정 |
| Classification Macro F1·class 균형 | Cryptova-Raw | Macro F1 0.3819, Balanced Accuracy 0.3938로 현재 완료 모델 중 가장 높음 |
| 비용 반영 Backtest | Cryptova-Full | 현재 완료 모델 중 가장 높은 연결 OOS 수익률(+27.46%)과 Sharpe-like(+1.143) 기록 |
| 적극적 신호 중 손실 규모 | TimesNet Classifier | LSTM보다 손실과 MDD가 작지만 최종 수익률은 여전히 음수 |

현재 결과는 예측 지표와 투자성과를 분리해서 해석해야 함을 보여준다. TimesNet은 Macro F1을 개선했지만 거래 수가 증가했고, 거래당 비용 차감 전 평균수익 약 `+0.134%`가 총비용 `0.20%`보다 작아 연결 수익률은 `-16.21%`가 됐다. 따라서 높은 분류 성능이 자동으로 높은 순수익을 의미하지 않는다.

Cryptova-Full은 TimesNet Classifier보다 Macro F1은 낮았지만 비용 반영 Connected OOS에서 `+27.46%`를 기록했다. 반면 rolling 2 손실과 rolling 3 수익 의존성이 커서, 현재 결과는 최종 시스템의 잠재적 수익성을 보여주지만 안정적인 regime 일반화를 입증한 것은 아니다. Fusion base와 Risk Filter의 기여도는 Cryptova-Base ablation 결과를 추가해 분리한다.

Cryptova-Base는 Macro F1 `0.376506`과 Balanced Accuracy `0.389647`로 TimesNet Classifier를 모두 상회했고 Connected OOS 수익률도 `+7.42%`였다. Risk Filter를 추가한 Full은 Macro F1을 낮췄지만 수익률을 `+27.46%`로 높이고 MDD를 `-37.38%`에서 `-24.40%`로 줄였다. 이는 모델 분류 성능과 최종 위험조정 거래성과를 별도로 평가해야 한다는 근거다.

Cryptova-Raw는 Macro F1 `0.381875`, Balanced Accuracy `0.393802`로 후처리 없는 출력에서도 TimesNet Classifier의 `0.364654 / 0.366543`을 상회했다. 그러나 Raw Backtest는 `-18.11%`였다. Confidence와 Risk Filter는 분류 점수를 높인 것이 아니라 신호 수와 거래 선택을 조절해 경제적 성과를 개선했다.

Chronos-2 LoRA Fine-tuned는 RMSE `0.024300`으로 Ridge-Flat, LSTM Regression 및 TimesNet Regression보다 낮은 순위를 기록했다. 또한 예측 신호의 `95.74%`가 HOLD에 집중되어 Macro F1은 `0.252000`, SHORT/LONG recall은 각각 `0.0220 / 0.0233`에 그쳤다. 이는 사전학습 foundation model에 Chart 12 covariate를 제공하고 LoRA로 적응했더라도 현재 데이터·target·고정 threshold 조건에서 기존 baseline이나 Cryptova보다 자동으로 우수해지지 않았음을 보여준다. 특히 Connected OOS 수익률 `-24.64%`와 Rolling 3의 음의 상관은 현재 설정의 regime 일반화가 약하다는 근거다.

TimesFM 2.5 LoRA Fine-tuned는 RMSE `0.024992`로 현재 Regression 모델 중 가장 큰 오차를
기록했다. 방향 정확도 `49.83%`는 상대적으로 가장 높지만 50% 미만이며 Pearson도
`-0.0973`이므로 안정적인 예측력으로 해석할 수 없다. Macro F1은 `0.287013`으로 Chronos-2와
Ridge-Flat보다 높지만 LSTM Classifier, TimesNet Classifier 및 Cryptova보다 낮았다. 예측의
`89.38%`가 HOLD에 편중됐고 Connected OOS 수익률은 `-30.96%`였다. 따라서 현재
close-only 64시간 조건에서는 대규모 사전학습과 LoRA가 기존 baseline 또는 Cryptova 대비
우위로 이어지지 않았다.

### 5.6 모델별 최적 활용 목적

여기서 `최적`은 모델의 보편적인 우수성을 의미하지 않는다. **현재 BTC 데이터, 동일 Rolling
Test 기간, 미래 24시간 target, 고정 threshold, 거래비용 및 evaluator 안에서 확인된 상대적
역할**을 의미한다.

#### 목적별 최고 모델 요약

| 목적 | 가장 좋은 모델 | 근거 |
|---|---|---|
| 수익률 수치 오차 최소 | **Zero-return baseline** | RMSE `0.022948`, MAE `0.016422`로 전체 최저; 학습 모델 중에는 Ridge-Flat이 가장 낮음 |
| 상승·하락 부호 예측 | TimesFM 2.5 | 방향 정확도 `49.83%`로 상대적으로 가장 높지만 50% 미만이라 실질적인 강점으로 단정하기 어려움 |
| 전체 신호 분류 | **Cryptova-Raw** | Macro F1 `0.381875`, Balanced Accuracy `0.393802`로 가장 높음 |
| Chart-only 신호 분류 | **TimesNet Classifier** | Macro F1 `0.364654`로 Chart-only 모델 중 가장 높음 |
| SHORT 탐지 | **TimesNet Classifier** | SHORT Recall `0.2075`로 가장 높음 |
| LONG 탐지 | **Cryptova-Raw** | LONG Recall `0.3527`로 가장 높음 |
| 실제 비용 반영 수익 | **Cryptova-Full** | 누적수익률 `+27.46%`로 가장 높음 |
| 위험조정 거래성과 | **Cryptova-Full** | Sharpe-like `+1.143`으로 가장 높음 |
| 학습 가능한 선형 baseline | **Ridge-Flat** | 학습 모델 중 가장 낮은 예측 오차, 빠른 학습, 비교적 작은 MDD |
| 사전학습 모델 비교 | Chronos-2·TimesFM | 둘 다 현재 조건에서는 기존 모델과 Cryptova를 능가하지 못함 |

#### 모델별 활용 목적 상세

| 모델 | 현재 가장 적합한 활용 목적 | 실험 근거 | 주의점 |
|---|---|---|---|
| Ridge-Flat | 학습 가능한 수익률 수치 예측 baseline | 학습 모델 중 최저 RMSE `0.023178`, 최저 MAE `0.016620` | Zero-return RMSE `0.022948`, MAE `0.016422`를 능가하지 못했고 강한 규제에서는 HOLD로 수축할 수 있음 |
| LSTM Regression | 순차 구조를 반영한 기본 neural regression 비교군 | 5,921 parameter로 72시간 순서를 직접 처리 | RMSE `0.023790`으로 Ridge보다 낮은 성능이며 방향 정확도도 50% 미만 |
| LSTM Classifier | 기본 recurrent direct-classification 비교군 | 수익률을 거치지 않고 SHORT/HOLD/LONG을 직접 학습 | Macro F1 `0.318197`, 수익률 `-39.67%`로 최종 성과가 낮음 |
| TimesNet Regression | 주기 구조가 수익률 회귀에 기여하는지 검증 | FFT·2D convolution 기반 주기 modeling을 동일 데이터에서 시험 | RMSE `0.023916`으로 Ridge/LSTM보다 개선되지 않음 |
| TimesNet Classifier | **Chart-only 신호 분류 및 SHORT 탐지** | Chart-only 최고 Macro F1 `0.364654`, 전체 모델 중 최고 SHORT recall `0.2075` | 비용 반영 수익률은 `-16.21%`로 신호 개선이 순수익으로 연결되지 않음 |
| Chronos-2 LoRA | 다변량 covariate foundation model·확률예측 비교 | Close target과 Chart 12 past covariate, quantile forecast 제공 | HOLD `95.74%`, Macro F1 `0.252000`, 수익률 `-24.64%` |
| TimesFM 2.5 LoRA | Close-only·긴-context foundation model 비교 | Patch 기반 효율적 예측, Regression 방향 정확도 `49.83%`로 수치상 최고 | 50% 미만이며 RMSE·MAE 최하위, 수익률 `-30.96%` |
| Cryptova-Raw | **모델 자체의 순수 신호 분류** | 전체 최고 Macro F1 `0.381875`, Balanced Accuracy `0.393802`, LONG recall `0.3527` | 비용 반영 수익률은 `-18.11%`로 좋은 분류가 수익성을 보장하지 않음 |
| Cryptova-Base | Confidence threshold의 신호 선택 효과 분석 | 높은 Macro F1 `0.376506`을 유지하며 Raw 수익률 `-18.11%`를 `+7.42%`로 개선 | MDD `-37.38%`이고 Rolling 간 편차가 큼 |
| Cryptova-Full | **실제 비용 반영 최종 거래 시스템** | 최고 수익률 `+27.46%`, 최고 Sharpe-like `+1.143`, 승률 `55.46%` | Macro F1은 Raw/Base보다 낮고 성과가 Rolling 3에 크게 의존 |

목적에 따른 핵심 선택은 다음과 같다.

```text
수익률 숫자를 가장 가깝게 예측
→ Zero-return baseline; 학습 모델 중 Ridge-Flat

Chart만 사용해 SHORT/HOLD/LONG을 직접 분류
→ TimesNet Classifier

Chart+News 모델 자체의 순수 신호 분류
→ Cryptova-Raw

Confidence만 적용한 신호 선택 효과
→ Cryptova-Base

비용과 Risk Filter까지 포함한 최종 투자성과
→ Cryptova-Full

Foundation model의 전이 성능과 확률예측 연구
→ Chronos-2 / TimesFM 2.5
```

### 5.7 모델별 장점과 단점

구조적 장점은 모델 설계상 가능한 능력이고, 실험상 장점은 이번 Connected OOS에서 실제로
확인된 결과다. 구조적 장점이 이번 데이터에서 반드시 성능 향상으로 나타나는 것은 아니다.

| 모델 | 구조적 장점 | 이번 실험에서 확인된 장점 | 단점·한계 |
|---|---|---|---|
| Ridge-Flat | 단순하고 빠르며 계수 해석 가능; L2 규제로 공선성 완화 | 학습 모델 중 가장 낮은 RMSE·MAE | Zero-return baseline을 능가하지 못했으며 비선형 상호작용, 동적 순차 패턴 및 반복 주기를 직접 modeling하지 못함 |
| LSTM | 시간 순서와 비선형 순차 의존성을 hidden state로 처리 | 작은 parameter로 전체 72시간을 처리하는 neural baseline | Ridge보다 회귀 오차가 크고 Classifier의 거래손실과 MDD가 가장 큼 |
| TimesNet | FFT로 주요 주기를 찾고 2D convolution으로 주기 내·주기 간 변화를 처리 | Chart-only 분류 최고 Macro F1 및 전체 최고 SHORT recall | 주기 구조가 Regression 오차를 개선하지 못했고 잦은 신호가 비용 후 손실로 연결 |
| Chronos-2 LoRA | 대규모 사전학습, past covariate 처리, point·quantile forecast | Chart 12 covariate를 사용하는 foundation model 비교와 불확실성 출력 가능 | SHORT/LONG을 거의 탐지하지 못하고 음의 상관·수익률을 기록; 큰 base model 필요 |
| TimesFM 2.5 LoRA | Patch로 긴 context를 효율적으로 처리하고 최대 128-step output patch 예측 | 회귀모델 중 방향 정확도가 상대적으로 가장 높고 point·quantile 출력 제공 | 현재는 close-only 64시간; 방향 정확도 50% 미만, RMSE·MAE 최하위, 빠른 과적합과 Rolling 3 붕괴 |
| Cryptova-Raw | Chart+News fusion으로 가격과 텍스트 정보를 함께 반영 | 전체 최고 Macro F1·Balanced Accuracy·LONG recall | 후처리 없이 거래하면 `-18.11%`; 분류 목적과 경제적 목적이 직접 일치하지 않음 |
| Cryptova-Base | Validation confidence threshold로 불확실한 신호 제거 | Raw 대비 수익률을 양수로 전환하면서 높은 분류 성능 유지 | 큰 MDD와 Rolling별 성과 편차; risk 정보는 아직 사용하지 않음 |
| Cryptova-Full | Confidence와 funding/std Risk Filter를 결합한 완성된 의사결정 시스템 | 전체 최고 누적수익률·Sharpe-like·평균 거래수익률 | 후처리로 분류 점수가 낮아지고 Rolling 3 수익 의존성이 큼; 구조와 재현 절차가 가장 복잡 |

따라서 모델 우열은 하나의 지표로 결정하지 않는다. **수익률 수치 예측, 신호 분류, 경제적
성과 및 시장구간 안정성**을 서로 다른 목표로 나눠 판단한다. 현재 결과에서는 Ridge-Flat,
TimesNet Classifier, Cryptova-Raw 및 Cryptova-Full이 각각 다른 목적에서 가장 강하다.

---

## 6. Market Regime 분석

전체 Connected OOS 결과를 확인한 뒤 추가한 사후 분석이다. 따라서 “모든 모델 결과를 보기 전에
Regime을 고정했다”고 주장하지 않는다. 다만 아래 정의와 threshold는 **Regime별 성능을 계산하기
전에 고정**했으며, 특정 모델에 유리하도록 Test의 Regime별 결과를 보고 변경하지 않았다.

### 6.1 Regime 정의

방향 Regime은 **예측 시점 직전 과거 72시간 동안 가격이 상대적으로 상승 추세였는지,
하락 추세였는지**를 나타낸다. 장기 상승장·하락장의 정의가 아니라
`72시간 수익률 기반 단기 가격 상태`다.

\[
R_{72}(t)=\frac{Close_t}{Close_{t-72}}-1
\]

각 Rolling Train의 `R_72`에서 33% 분위수 `Q33`과 67% 분위수 `Q67`을 계산한다.

```text
단기 상승 추세: return_72h >= Train Q67 AND return_72h > 0
단기 하락 추세: return_72h <= Train Q33 AND return_72h < 0
단기 중립: 위 두 조건을 모두 만족하지 않는 경우
```

변동성 Regime은 예측 시점의 과거 정보인 `std_24h`를 각 Rolling Train 중앙값과 비교한다.

```text
고변동성: std_24h > Train std_24h 중앙값
저변동성: std_24h <= Train std_24h 중앙값
```

미래 24시간 수익률은 Regime 결정에 사용하지 않았다. 각 Rolling Train에서 계산한 기준을 해당
Validation·Test에 변경 없이 적용했다. 방향과 변동성을 결합해 `UP_HIGH`, `DOWN_LOW` 등
6개의 복합 Regime도 함께 평가했다.

### 6.2 Rolling별 Train 기준값

| Rolling | 하락 기준 Q33 | 상승 기준 Q67 | std_24h 중앙값 |
|---|---:|---:|---:|
| rolling_1 | -1.3233% | +2.1375% | 0.004667 |
| rolling_2 | -1.2927% | +1.8277% | 0.004414 |
| rolling_3 | -0.9528% | +1.7564% | 0.004095 |

### 6.3 Connected OOS Regime 분포

| 축 | Regime | Samples | 비율 |
|---|---|---:|---:|
| 방향 | 단기 상승 | 1,482 | 23.56% |
| 방향 | 단기 하락 | 2,544 | 40.44% |
| 방향 | 단기 중립 | 2,265 | 36.00% |
| 변동성 | 고변동성 | 2,653 | 42.17% |
| 변동성 | 저변동성 | 3,638 | 57.83% |

Train에서는 분위수 경계가 약 1/3씩 나누도록 계산되지만 Test 비율을 강제로 맞추지 않는다.
따라서 위 분포는 Connected OOS 기간에 단기 하락 상태가 상대적으로 많았음을 보여준다.

### 6.4 Regime별 수익률 예측 결과

| Regime | 학습 모델 중 최저 RMSE | RMSE | 해석 |
|---|---|---:|---|
| 단기 상승 | Chronos-2 LoRA | 0.021902 | 방향 Regime 중 유일하게 Ridge보다 낮은 RMSE |
| 단기 하락 | Ridge-Flat | 0.027231 | 모든 모델의 오차가 중립보다 커짐 |
| 단기 중립 | Ridge-Flat | 0.018461 | 가장 예측하기 쉬운 방향 구간 |
| 고변동성 | Ridge-Flat | 0.027982 | 모든 모델에서 저변동성보다 오차가 큼 |
| 저변동성 | Ridge-Flat | 0.018920 | Ridge의 낮은 전체 RMSE가 유지됨 |

Zero-return baseline은 현재 Regime별 승자 계산에 포함하지 않았다. 아래 순위는 학습된 모델끼리의
조건부 비교다. Ridge-Flat은 5개 구간 중 4개에서 가장 낮은 RMSE를 기록했다. Chronos-2는 단기
상승에서만 최저 RMSE였으며, 이것만으로 일반적인 우위를 주장할 수는 없다.

### 6.5 Regime별 신호 분류 결과

| Regime | 최고 Macro F1 모델 | Macro F1 |
|---|---|---:|
| 단기 상승 | TimesNet Classifier | 0.346972 |
| 단기 하락 | Cryptova-Raw | 0.409788 |
| 단기 중립 | TimesNet Classifier | 0.367104 |
| 고변동성 | Cryptova-Base | 0.408816 |
| 저변동성 | TimesNet Classifier | 0.355281 |

Cryptova의 분류 강점은 단기 하락과 고변동성 구간에서 가장 뚜렷했다. TimesNet은 단기 상승,
중립 및 저변동성에서 가장 높은 Macro F1을 기록해 Chart-only 모델로서 비교적 고른 결과를
보였다. Cryptova-Full은 거래 필터 때문에 Raw·Base보다 신호 수가 줄어 순수 분류 Macro F1의
최고 모델은 아니었다.

### 6.6 비용 반영 거래성과

Connected OOS의 24시간 non-overlap 거래를 모델별로 한 번 선택한 뒤, 거래 진입 시점의
Regime에 귀속했다. Cryptova-Full 결과는 다음과 같다.

| Regime | 거래 수 | 누적수익률 | 승률 | 평균 거래수익률 |
|---|---:|---:|---:|---:|
| 단기 상승 | 25 | +7.81% | 64.00% | +0.317% |
| 단기 하락 | 68 | +17.03% | 52.94% | +0.280% |
| 단기 중립 | 26 | +1.03% | 53.85% | +0.059% |
| 고변동성 | 78 | +48.74% | 62.82% | +0.554% |
| 저변동성 | 41 | -14.31% | 41.46% | -0.358% |

Cryptova-Full의 전체 수익은 **고변동성 구간에서 발생한 이익에 크게 의존**했다. 방향별로는
상승·하락·중립 모두 양수였지만, 저변동성에서는 손실이었다. 따라서 현재 Risk Filter는 방향
구분보다 변동성 환경에 민감하며, 안정적인 전 구간 수익 모델이라고 단정할 수 없다.

복합 Regime에서 최고 성능 모델은 다음과 같았다. 거래성과의 최고 모델은 거래 수가 적은
경우가 있으므로 탐색적 결과로만 해석한다.

| 복합 Regime | 학습 모델 중 최저 RMSE | 최고 Macro F1 | 최고 비용 반영 수익률 |
|---|---|---|---|
| 상승·고변동성 | Chronos-2 `0.023503` | Cryptova-Base `0.362986` | Cryptova-Full `+12.60%` (18회) |
| 상승·저변동성 | Ridge `0.020347` | TimesNet `0.351864` | Chronos-2 `+0.17%` (7회) |
| 하락·고변동성 | Ridge `0.032185` | Cryptova-Raw `0.431269` | Cryptova-Base `+46.60%` (56회) |
| 하락·저변동성 | Ridge `0.019742` | TimesNet `0.354073` | TimesFM `+20.69%` (18회) |
| 중립·고변동성 | Ridge `0.020766` | Cryptova-Base `0.398617` | Cryptova-Base `+9.67%` (11회) |
| 중립·저변동성 | Ridge `0.017604` | TimesNet `0.353759` | TimesFM `+7.23%` (12회) |

Regime별 최고 모델을 사후에 골라 전환하는 전략은 이번 분석에서 검증하지 않았다. 이를 실제
전략으로 사용하려면 Train·Validation만으로 전환 규칙을 정한 뒤 새로운 OOS에서 별도로 평가해야 한다.

---

## 7. 현재 결론

### 7.1 Foundation model이 전통 모델보다 우수한가?

현재 Connected OOS 기준으로는 우수하지 않았다. Chronos-2와 TimesFM 2.5의 RMSE는 각각
`0.024300`, `0.024992`로 Ridge-Flat `0.023178`, LSTM `0.023790`, TimesNet `0.023916`보다
높았다. Foundation model은 사전학습됐지만 현재 BTC target에 대한 수익률 수치 예측에서는
전통·전용 학습 모델을 능가하지 못했다. 또한 Zero-return baseline의 RMSE `0.022948`보다도
높아, 이번 Regression Track에서는 어떤 학습 모델도 naive `0%` 예측을 능가하지 못했다.

### 7.2 Foundation model이 Cryptova보다 우수한가?

현재 결과에서는 우수하지 않았다. TimesFM과 Chronos-2의 Macro F1은 각각 `0.287013`,
`0.252000`으로 Cryptova-Full `0.350898`보다 낮았다. Connected OOS 수익률도 TimesFM
`-30.96%`, Chronos-2 `-24.64%`인 반면 Cryptova-Full은 `+27.46%`였다. 단, TimesFM은
close-only 64시간, Chronos-2는 close+Chart 12, Cryptova는 Chart+News를 사용하므로 이 결론은
동일 입력 architecture-only 비교가 아니라 완성된 시스템의 end-to-end 비교다.

### 7.3 Cryptova는 경쟁력이 있는가?

현재 완료된 Connected OOS 기준으로 경쟁력이 확인됐다. Cryptova-Raw는 가장 높은 Macro F1
`0.381875`와 Balanced Accuracy `0.393802`를 기록했고, Cryptova-Full은 가장 높은 비용 반영
수익률 `+27.46%`와 Sharpe-like `+1.143`을 기록했다. 다만 Full 수익이 Rolling 3에 크게
의존하고 Rolling 2에서는 손실이 발생했으므로 안정적인 regime 일반화는 아직 입증되지 않았다.

### 7.4 어떤 시장 regime에서 각 모델이 강한가?

학습 모델 간 수익률 수치 예측은 Ridge-Flat이 단기 하락·중립 및 고·저변동성에서 가장 낮은
RMSE를 기록했고, Chronos-2는 단기 상승에서만 가장 낮았다. Zero-return baseline은 현재
Regime별 승자 계산에 포함하지 않았다. 신호 분류는 Cryptova-Raw가 단기 하락,
Cryptova-Base가 고변동성에서 가장 강했고, TimesNet Classifier가 단기 상승·중립·저변동성에서
가장 높았다. 실제 거래에서 Cryptova-Full은 고변동성 `+48.74%`, 저변동성 `-14.31%`로 성과
차이가 컸다. 따라서 Cryptova의 현재 경쟁력은 특히 **하락·고변동성 환경의 신호 처리와
고변동성 거래성과**에서 확인되며, 저변동성 안정성은 개선 과제로 남는다.

### 7.5 결과 해석 시 제한사항

- 모델별 입력 modality와 출력 방식이 다를 수 있다.
- Foundation model은 대규모 외부 데이터로 사전학습됐다.
- Cryptova는 News 정보를 추가로 사용한다.
- Rolling 수와 거래 수가 통계적 확신에 충분한지 검토해야 한다.
- 연결 Test 기간을 본 뒤 실험 규칙을 변경하면 추가 holdout 검증이 필요하다.

---
