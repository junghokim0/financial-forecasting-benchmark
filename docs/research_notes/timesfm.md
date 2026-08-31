# TimesFM 2.5 LoRA Fine-tuned 연구 노트

## 1. 정의

TimesFM(Time Series Foundation Model)은 Google Research가 개발한 시계열 예측용
foundation model이다. 여러 분야의 대규모 시계열로 사전학습한 하나의 모델을 새로운
시계열에 zero-shot 또는 fine-tuning 방식으로 적용하는 것을 목적으로 한다.

TimesFM은 언어모델이 여러 단어를 하나의 token으로 처리하듯 연속된 여러 시점을 하나의
**time-series patch token**으로 변환한다. 이후 Decoder-only Transformer가 과거 patch들의
관계를 처리하고 다음 미래 구간을 예측한다.

이번 benchmark에서 사용한 모델은 다음과 같다.

| 항목 | 값 |
|---|---|
| 공식 표기 | **TimesFM 2.5 LoRA Fine-tuned** |
| Base checkpoint | `google/timesfm-2.5-200m-transformers` |
| 고정 revision | `5a9806b9b291fad9233b5249d88263f1846304d3` |
| 전체 parameter | 232,672,192 |
| 학습 parameter | 1,382,912 |
| Fine-tuning | LoRA; Full fine-tuning 아님 |
| 입력 | raw BTC `close` 최근 64시간 |
| 출력 | 미래 24시간 close 경로와 24시간 예측 수익률 |

---

## 2. 특징

TimesFM 2.5의 주요 특징은 다음과 같다.

1. **시계열 Patch 사용**
   - 연속된 32개 시점을 하나의 입력 token으로 변환한다.
   - 한 시점씩 Attention하는 것보다 긴 context를 적은 token 수로 처리할 수 있다.

2. **Decoder-only Transformer**
   - 별도의 Encoder나 Cross-Attention 없이 causal self-attention으로 과거 patch 관계를 처리한다.
   - 미래 정보가 과거 표현에 유출되지 않도록 causal mask를 사용한다.

3. **긴 Output Patch**
   - 한 개 hidden token으로 다음 최대 128개 시점을 예측할 수 있다.
   - 장기 horizon을 한 시점씩 생성할 때 발생하는 반복 추론과 오차 누적을 줄인다.

4. **Point 및 Quantile Forecast**
   - 하나의 point forecast와 `q10`부터 `q90`까지의 quantile forecast를 출력한다.
   - 단일 예측값뿐 아니라 예측 불확실성 범위도 표현할 수 있다.

5. **대규모 사전학습**
   - 다양한 실제·합성 시계열을 통해 일반적인 추세, 계절성 및 변동 패턴을 미리 학습했다.
   - 이번 실험에서는 이 사전학습 가중치에 LoRA adapter만 추가로 학습했다.

6. **TimesFM 2.5의 구조적 개선**
   - RoPE(Rotary Position Embedding)
   - QK normalization
   - 차원별 Attention scaling
   - Continuous quantile prediction

---

## 3. 문제 제기와 실험 목적

본 실험의 핵심 질문은 다음과 같다.

> 대규모 외부 시계열로 사전학습한 TimesFM 2.5를 동일한 BTC rolling 구간에 LoRA
> fine-tuning했을 때 전통 모델, 일반 딥러닝 모델 및 Cryptova보다 경쟁력 있는가?

TimesFM은 연속적인 미래 close를 출력하므로 두 평가 Track에 참여한다.

```text
Regression Track
미래 close → predicted_return
→ RMSE / MAE / Pearson / Spearman / Directional Accuracy

Classification Track
predicted_return → 고정 ±1.2% threshold
→ SHORT / HOLD / LONG
→ Macro F1 / Balanced Accuracy / Backtest
```

TimesFM의 공식 사용 방식과 benchmark의 공정성을 동시에 유지하기 위해 다음 원칙을 적용했다.

- 공통 정보 범위는 과거 72시간이다.
- TimesFM은 그 범위 안의 최근 64시간만 사용한다.
- 미래 예측 horizon은 다른 모델과 동일한 24시간이다.
- Test를 보고 checkpoint, threshold 또는 입력 길이를 변경하지 않는다.

---

## 4. 모델 구조

### 4.1 전체 흐름

```text
BTC close 최근 64시간
[B, 64]
        │
        ▼
모델 내부 정규화
[B, 64]
        │
        ▼
32시간씩 Patch 분할
[B, 2, 32]
        │
        ▼
Residual MLP Tokenizer
각 Patch: 32 → 1,280
[B, 2, 1280]
        │
        ▼
RoPE 위치 정보
        │
        ▼
Decoder-only Transformer × 20
Causal Self-Attention: 16 heads
[B, 2, 1280]
        │
        ▼
Point/Quantile Output Projection
미래 최대 128개 시점
        │
        ▼
우리 실험에서 미래 24시간 사용
        │
        ▼
24번째 Point Forecast close
        │
        ▼
predicted_close(t+24) / close(t) - 1
        │
        ▼
predicted_return
        │
        ├─ Regression 평가
        └─ ±1.2% → SHORT/HOLD/LONG
```

### 4.2 입력 Tensor

이번 실험은 TimesFM의 단변량 입력 방식에 맞춰 raw BTC `close`만 사용한다.

```text
Input = [close(t-63), close(t-62), ..., close(t-1), close(t)]
Shape = [B, 64]
```

| 항목 | 값 |
|---|---:|
| 입력 feature | raw BTC `close` 1개 |
| 실제 입력 길이 | 64시간 |
| 공통 가용 정보 범위 | 72시간 |
| 미래 horizon | 24시간 |
| Chart 12 feature | 사용하지 않음 |
| News 9 feature | 사용하지 않음 |

TimesFM이 최대 64시간만 볼 수 있는 모델이라는 뜻은 아니다. 공식 checkpoint 설정의
최대 context는 **16,384 시점**이다. 이번 benchmark에서는 공통 72시간 범위를 넘지 않으면서
patch length 32에 맞추기 위해 최근 64시간을 사용했다.

```text
공식 최대 context       = 16,384
Benchmark 공통 범위     = 72시간
TimesFM patch length    = 32시간
72 이하의 최대 32 배수 = 64시간
```

따라서 64시간은 모델의 최대 능력이 아니라 **현재 benchmark protocol에서 선택한 입력 길이**다.

### 4.3 내부 정규화

가격의 절대 수준은 기간마다 크게 다르므로 TimesFM은 context 통계를 이용해 내부적으로
값을 정규화하고, 예측 후 원래 scale로 복원한다. 개념적으로 다음과 같다.

\[
z_t=\frac{x_t-\mu}{\sigma}
\]

```text
Raw BTC close
    ↓ context mean/std
Normalized sequence
    ↓ TimesFM forecast
Normalized future
    ↓ inverse transform
Future close scale
```

이 처리는 benchmark의 Chart scaler와 별개인 TimesFM 공식 전처리다. 현재 입력은 raw
`close`이며 Chart 12 feature용 StandardScaler를 적용하지 않는다.

### 4.4 Patch 생성

TimesFM 2.5의 input patch length는 32다. 최근 64시간은 두 개의 patch로 분할된다.

```text
Patch 1: close(t-63) ~ close(t-32)  ┐
                                     ├─ [B, 2, 32]
Patch 2: close(t-31) ~ close(t)     ┘
```

Shape 변화는 다음과 같다.

```text
[B, 64]
   ↓ reshape
[B, 2, 32]
```

Transformer는 64개의 시간을 개별 token으로 보는 것이 아니라 32시간 묶음 두 개를 token
단위로 처리한다. Patch 내부 32시간의 세부 형태는 다음 Tokenizer가 압축한다.

### 4.5 Residual MLP Tokenizer

각 32시간 patch와 mask 정보는 Residual MLP를 통과해 1,280차원 embedding으로 변환된다.

```text
32시간 Patch
[32]
   ↓ Residual MLP
Patch Token
[1280]
```

전체 shape는 다음과 같다.

```text
[B, 2, 32]
   ↓
[B, 2, 1280]
```

이 1,280차원은 사람이 추세·변동성 등의 의미를 직접 배정한 공간이 아니다. 대규모
사전학습을 통해 미래 예측에 유용한 patch 표현을 모델이 학습한 hidden space다.

### 4.6 RoPE 위치 정보

두 patch의 값만 제공하면 모델은 어느 patch가 더 오래됐는지 구분하기 어렵다. TimesFM
2.5는 RoPE를 Attention에 적용해 patch의 상대적인 시간 위치를 표현한다.

```text
Patch 1 → 더 오래된 32시간
Patch 2 → 현재에 가까운 32시간
```

RoPE는 별도의 위치값을 가격에 더하는 단순 feature가 아니라 Query와 Key의 회전을 통해
Attention이 상대 위치를 반영하도록 한다.

### 4.7 Decoder Block

TimesFM 2.5 checkpoint의 주요 구조는 다음과 같다.

| 항목 | 값 |
|---|---:|
| Transformer 종류 | Decoder-only |
| Hidden dimension | 1,280 |
| Layer 수 | 20 |
| Attention head | 16 |
| Head dimension | 80 |
| Key/Value head | 16 |
| FFN intermediate size | 1,280 |
| Activation | Swish |
| Normalization | RMSNorm |
| Attention dropout | 0.0 |

한 개 block의 개념적 흐름은 다음과 같다.

```text
Input [B, 2, 1280]
        │
        ▼
RMSNorm
        │
        ▼
Causal Self-Attention
16 heads × 80 dimensions
        │
        ▼
Residual Connection
        │
        ▼
RMSNorm
        │
        ▼
Feed Forward Network + Swish
        │
        ▼
Residual Connection
        │
        ▼
Output [B, 2, 1280]
```

이 block이 20번 반복되며 token 수와 hidden dimension은 유지된다.

### 4.8 Causal Self-Attention

Decoder-only 구조에서는 각 patch가 자신과 과거 patch만 참고할 수 있다.

```text
                 참고 가능
              Patch 1  Patch 2
현재 Patch 1     O        X
현재 Patch 2     O        O
```

따라서 최근 Patch 2는 이전 Patch 1을 참고할 수 있지만, 이전 Patch 1의 표현을 만들 때
미래 위치인 Patch 2를 미리 볼 수 없다. 이 causal mask가 학습 중 미래 정보 누수를 막는다.

16개 Attention head는 1,280차원을 `16 × 80`으로 나눠 서로 다른 관점에서 patch 관계를
분석할 수 있다. 각 head의 구체적인 의미는 코드로 지정되지 않고 학습 과정에서 결정된다.

이번 입력에는 patch token이 두 개뿐이므로 Attention이 직접 비교하는 시간 단위도 두 개다.
즉 TimesFM은 **이전 32시간과 최근 32시간의 관계**를 처리하고, 각 patch 내부의 세부 패턴은
Residual MLP embedding에 압축되어 있다.

### 4.9 Feed Forward Network와 Residual Connection

Attention이 patch 사이의 관계를 혼합한 뒤 FFN이 각 token의 표현을 비선형적으로 변환한다.
Residual connection은 원래 표현과 변환 결과를 더해 깊은 20-layer 학습을 안정화한다.

```text
Attention output + original input
              ↓
            FFN
              ↓
FFN output + Attention residual
```

Attention은 token 간 관계를 담당하고 FFN은 각 token 내부의 표현 변환을 담당한다고 이해할
수 있다.

### 4.10 Decoder-only Forecasting

언어모델은 앞의 token으로 다음 단어 token을 예측한다. TimesFM은 앞의 시계열 patch로
다음 미래 patch를 예측한다.

```text
언어모델: 앞의 단어 token → 다음 단어 token
TimesFM: 과거 시간 patch → 다음 미래 시간 patch
```

다만 TimesFM은 미래를 반드시 한 시간씩 생성하지 않는다. Output patch length가 128이므로
한 hidden token으로 다음 최대 128개 시점을 출력할 수 있다. 이번 horizon 24는 첫 output
patch 안에 포함되므로 미래 24시간을 한 시간씩 24회 반복 생성할 필요가 없다.

```text
Output patch: t+1 ... t+128
우리의 사용: t+1 ... t+24
```

128시간보다 긴 horizon이라면 생성된 output patch를 다시 조건으로 사용해 다음 patch를
autoregressive하게 생성할 수 있다.

### 4.11 Point 및 Quantile Output Head

마지막 Transformer hidden state는 Residual MLP output projection을 통해 미래 close로
변환된다.

```text
Transformer hidden [1280]
        ├─ Point projection
        └─ Continuous quantile projection
```

TimesFM 2.5는 point forecast와 다음 9개 quantile을 제공한다.

```text
q10, q20, q30, q40, q50, q60, q70, q80, q90
```

현재 코드에서는 다음 값을 저장한다.

| 저장값 | 코드상 출력 |
|---|---|
| 중심 예측 | `mean_predictions[:, 23]` |
| 하단 예측 | `full_predictions[:, 23, q10_index]` |
| 상단 예측 | `full_predictions[:, 23, q90_index]` |

코드 변수명은 `median_close`지만 실제 중심 예측으로 사용하는 값은 `q50`이 아니라
`mean_predictions`의 24번째 point forecast다.

### 4.12 최종 수익률과 신호

24시간 후 예측 close를 현재 close와 비교해 예측 수익률을 계산한다.

\[
\hat r_{t,t+24}
=\frac{\widehat{close}_{t+24}}{close_t}-1
\]

이 값은 Regression Track에서 그대로 평가하고, Classification Track에서는 고정 threshold로
변환한다.

```text
predicted_return <= -0.012 → SHORT
-0.012 < predicted_return < 0.012 → HOLD
predicted_return >= 0.012 → LONG
```

---

## 5. 실험 가정과 공정성

### 5.1 데이터와 Target

| 항목 | 설정 |
|---|---|
| 시장 | BTC |
| 빈도 | 1시간 |
| 공통 window | 72시간 |
| TimesFM 실제 context | 최근 64시간 |
| 입력 | raw `close` 단변량 |
| 학습 미래 경로 | 미래 close 24개 |
| 평가 target | `raw_future_return` |
| Classification label | 동일한 ±1.2% 기준 |

### 5.2 왜 72시간이 아니라 64시간인가

32시간 patch를 기준으로 가능한 후보는 다음과 같다.

| 후보 | 문제 |
|---|---|
| 64시간 | 공통 72시간 안에서 완전한 patch 2개 사용 |
| 72시간 | 32로 나누어떨어지지 않음 |
| 96시간 | 다른 모델보다 과거 24시간을 추가 사용 |
| 72→96 zero padding | 가격 정규화 통계와 입력 표현을 왜곡할 수 있음 |

따라서 공통 정보 범위를 넘지 않는 가장 긴 32의 배수인 최근 64시간을 선택했다. 이는
완전히 동일한 입력 길이 비교가 아니라 **공통 정보 범위 내 model-native input 비교**다.
결과 해석에서는 TimesFM이 다른 72시간 모델보다 8시간 적은 history를 사용했다는 점을
명시해야 한다.

다시 강조하면 공식 최대 context는 16,384이며, 64시간은 TimesFM의 한계가 아니다.

### 5.3 데이터 누수 방지

각 Rolling은 다음 규칙을 따른다.

```text
Rolling Train
→ LoRA gradient update

Rolling Validation
→ 공식 TimesFM loss가 가장 낮은 adapter 선택

Rolling Test
→ 선택된 adapter를 변경하지 않고 한 번 평가
```

- Test는 adapter, epoch, learning rate 또는 threshold 선택에 사용하지 않는다.
- 세 Rolling은 각각 원본 base checkpoint에서 독립적으로 시작한다.
- SHORT/HOLD/LONG threshold는 모든 return 기반 모델과 동일한 ±1.2%다.
- 평가 비용과 non-overlap 규칙은 Cryptova 공통 evaluator와 동일하다.

### 5.4 Foundation Model 비교의 의미

TimesFM은 외부 대규모 시계열을 사전학습했지만 Ridge, LSTM, TimesNet 및 Cryptova는 현재
데이터를 중심으로 처음부터 학습했다. 따라서 이번 비교는 동일 parameter budget 비교가
아니다.

정확한 해석은 다음과 같다.

> 서로 다른 사전학습 및 모델 구조를 가진 시스템을 동일 BTC Test 기간, target, threshold,
> 비용 및 evaluator 아래에서 비교하는 end-to-end 경쟁력 실험이다.

### 5.5 LoRA 선택 이유와 설정

약 2억 3천만 parameter 전체를 중첩이 큰 제한된 BTC window로 갱신하면 계산·메모리 비용과
과적합 위험이 커질 수 있다. 따라서 본체 대부분을 고정하고 작은 저랭크 adapter를 학습했다.

```text
Original Linear: y = Wx
LoRA Linear:     y = Wx + BAx
```

| 항목 | 값 |
|---|---:|
| LoRA rank | 4 |
| LoRA alpha | 8 |
| LoRA dropout | 0.05 |
| Target module | all-linear |
| Trainable parameter | 1,382,912 |
| Total parameter | 232,672,192 |
| 학습 비율 | 약 0.59% |

LoRA는 입력·출력 shape나 TimesFM의 전체 forward 구조를 바꾸지 않는다. 기존 Linear layer에
학습 가능한 작은 저랭크 경로를 추가한다.

---

## 6. 학습과 추론 과정

### 6.1 Rolling 학습

| 항목 | 설정 |
|---|---:|
| 최대 epoch | 10 |
| Early stopping patience | 3 |
| Micro batch | 16 |
| Gradient accumulation | 2 |
| Effective batch | 32 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Gradient clipping | `1.0` |
| Scheduler | CosineAnnealingLR |
| Seed | 42 |
| 최종 모델 선택 기준 | Validation 공식 loss 최소 |

학습 forward는 다음과 같다.

```text
past_values [B, 64]
future_values [B, 24]
        ↓
TimesFM 2.5 + LoRA
        ↓
Point forecast + Quantile forecast
        ↓
공식 normalized MSE + quantile loss
        ↓
Backpropagation
        ↓
LoRA parameter update
```

### 6.2 Best Adapter 저장

Validation loss가 개선될 때 다음 폴더에 LoRA adapter를 저장한다.

```text
outputs/timesfm2_5_lora_close/
└─ rolling_n/
   └─ adapter/
      ├─ adapter_model.safetensors
      ├─ adapter_config.json
      └─ README.md
```

`adapter_model.safetensors`가 각 Rolling에서 BTC 데이터로 학습된 핵심 가중치다. 추론 시에는
동일 revision의 base TimesFM checkpoint를 불러온 뒤 해당 adapter를 결합해야 한다.

### 6.3 OOS 추론과 평가

```text
Best adapter + Base TimesFM
        ↓
Validation/Test close 64시간
        ↓
mean_predictions의 t+24 close
        ↓
predicted_return
        ├─ Regression metrics
        └─ ±1.2% class → Classification/Backtest
```

Rolling 1~3 Test prediction은 시간순으로 연결해 6,291개의 Connected OOS를 만든다. Backtest는
fee 0.1%, slippage 0.1%, 선택 거래당 총비용 0.2%, 24시간 non-overlap을 적용한다.

---

## 7. 결과

### 7.1 Rolling별 Test 결과

| Rolling | Best epoch | RMSE | MAE | Direction | Macro F1 | Balanced Accuracy | Return | MDD | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 1 | 0.016446 | 0.012338 | 49.93% | 0.278344 | 0.335734 | +1.04% | -5.27% | 14 |
| rolling_2 | 1 | 0.025565 | 0.018592 | 52.82% | 0.293757 | 0.342989 | -4.17% | -20.27% | 46 |
| rolling_3 | 1 | 0.030939 | 0.022590 | 46.68% | 0.260042 | 0.342290 | -28.70% | -29.39% | 44 |

### 7.2 연결 Regression 결과

| Samples | RMSE | MAE | Pearson | Spearman | Directional Accuracy |
|---:|---:|---:|---:|---:|---:|
| 6,291 | 0.024992 | 0.017804 | -0.0973 | -0.0501 | 49.83% |

### 7.3 연결 Classification 결과

| Accuracy | Macro F1 | Balanced Accuracy | Weighted F1 |
|---:|---:|---:|---:|
| 0.494675 | 0.287013 | 0.345735 | 0.385255 |

| Class | Prediction count | Prediction ratio | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| SHORT | 339 | 5.39% | 0.2920 | 0.0574 | 0.0959 |
| HOLD | 5,623 | 89.38% | 0.5205 | 0.9173 | 0.6642 |
| LONG | 329 | 5.23% | 0.2614 | 0.0625 | 0.1009 |

### 7.4 연결 Backtest 결과

| Return | Sharpe-like | MDD | Trades | Trade ratio | Win rate | Avg trade return |
|---:|---:|---:|---:|---:|---:|---:|
| -30.96% | -1.346 | -41.22% | 104 | 1.65% | 50.00% | -0.316% |

### 7.5 학습 과정

세 Rolling 모두 첫 epoch에서 Validation loss가 가장 낮았다.

| Rolling | Epoch 1 best loss | 이후 Validation loss |
|---|---:|---|
| rolling_1 | 3.6200 | 4.3285 → 4.5737 → 4.7986 |
| rolling_2 | 4.6167 | 5.0840 → 5.4017 → 5.1899 |
| rolling_3 | 4.0073 | 4.4043 → 4.5673 → 4.5106 |

Train loss는 계속 감소했지만 Validation loss는 첫 epoch 이후 증가했다. 따라서 현재 설정에서는
추가 LoRA 학습이 Train 적합도를 높였지만 새로운 구간의 일반화를 개선하지 못했다.

---

## 8. 결과 해석

### 8.1 Regression 예측력

Connected RMSE `0.024992`와 MAE `0.017804`는 현재 Regression Track에서 가장 좋은 결과가
아니다.

| Model | RMSE | MAE | Direction |
|---|---:|---:|---:|
| Ridge-Flat | **0.023178** | **0.016620** | 46.84% |
| LSTM | 0.023790 | 0.017203 | 46.97% |
| TimesNet | 0.023916 | 0.017317 | 48.24% |
| Chronos-2 LoRA | 0.024300 | 0.017148 | 49.09% |
| TimesFM 2.5 LoRA | 0.024992 | 0.017804 | **49.83%** |

TimesFM은 방향 정확도 수치가 가장 높지만 50%보다 낮다. 따라서 방향을 안정적으로 맞혔다고
볼 수 없으며, RMSE와 MAE 기준으로는 비교 모델 중 가장 큰 오차를 기록했다.

### 8.2 HOLD 편향

전체 예측의 `89.38%`가 HOLD다. 실제 HOLD recall은 `91.73%`로 높지만 SHORT와 LONG recall은
각각 `5.74%`, `6.25%`에 그쳤다.

```text
예측값이 0% 근처에 집중
        ↓
±1.2%를 넘는 예측이 적음
        ↓
대부분 HOLD
        ↓
SHORT/LONG 탐지 실패
```

Accuracy `49.47%`도 실제 HOLD 비중 약 `50.72%`보다 낮다. 따라서 높은 HOLD recall만으로
분류 성능이 좋다고 해석할 수 없다.

### 8.3 Classification과 거래성과

Macro F1 `0.287013`은 Chronos-2와 Ridge-Flat보다 조금 높지만 LSTM Classifier, TimesNet
Classifier 및 모든 Cryptova variant보다 낮다. 직접 class를 학습한 모델과 달리 TimesFM은
close 예측을 threshold로 변환하므로 수익률 크기가 ±1.2%를 넘지 않으면 방향 부호가 맞아도
HOLD 오답이 될 수 있다.

승률은 `50%`지만 평균 거래수익률은 비용 반영 후 `-0.316%`다. 맞힌 거래와 틀린 거래 수가
같더라도 손실 거래의 크기와 총비용이 더 크면 누적수익률은 음수가 된다.

### 8.4 Rolling 안정성과 과적합

Rolling 1에서는 `+1.04%`였지만 거래가 14개뿐이다. Rolling 2는 방향 정확도 `52.82%`, 승률
`58.70%`였음에도 평균 거래수익이 비용을 넘지 못해 `-4.17%`를 기록했다. Rolling 3에서는
Pearson `-0.1825`, 수익률 `-28.70%`로 크게 악화됐다.

세 Rolling 모두 Best epoch가 1이라는 점은 LoRA가 과적합을 완전히 방지하지 못했음을 보여준다.
가능한 원인으로 close-only 입력의 제한, 높은 시장 잡음, 중첩 sample, learning rate 및 regime
변화를 고려할 수 있지만 현재 결과만으로 하나의 원인을 확정할 수는 없다.

### 8.5 64시간 입력의 해석

TimesFM의 저조한 결과를 단순히 모델 구조의 실패로만 해석해서는 안 된다. 현재 TimesFM은
다른 Chart 모델보다 8시간 적은 history를 사용하며 close 한 개만 입력받는다. 반면 Chronos-2는
72시간과 Chart 12 covariate를, Cryptova는 Chart와 News를 사용한다.

그럼에도 64시간은 임의로 모델 성능을 제한한 값이 아니라 공통 72시간 범위와 공식 32-step
patch 구조를 동시에 지키기 위한 선택이다. 공식 최대 context 16,384와 현재 실험 context 64를
구분해야 한다.

### 8.6 현재 단계의 결론

> TimesFM 2.5 LoRA Fine-tuned는 close-only 최근 64시간으로 미래 24시간을 예측한 현재
> protocol에서 Ridge, LSTM, TimesNet 및 Chronos-2보다 낮은 수익률 수치 예측 성능을
> 기록했다. 신호는 HOLD에 편중됐으며, Rolling 3의 성능 악화로 Connected OOS 거래성과도
> 음수였다. 따라서 사전학습 foundation model이라는 사실만으로 BTC downstream task에서
> 우수한 성능이 보장되지는 않았다.

---

## 9. 공식 비교에서의 역할

TimesFM은 다음 두 Track에서 사용한다.

| Track | 입력 | 출력 | 평가 방식 |
|---|---|---|---|
| Regression | Close 최근 64시간 | predicted return | RMSE·MAE·Correlation·Direction |
| Classification | 동일 | return → ±1.2% class | Macro F1·Balanced Accuracy·Backtest |

TimesFM 결과는 다음 질문에 답하는 데 사용한다.

1. 사전학습 foundation model이 처음부터 학습한 Ridge/LSTM/TimesNet보다 우수한가?
2. Foundation model이 Chart+News 전문 모델 Cryptova보다 경쟁력이 있는가?
3. TimesFM의 성능이 시장 Rolling 또는 regime에 따라 어떻게 달라지는가?
4. Point forecasting 성능과 실제 거래 신호 성능이 일치하는가?

현재 답은 다음과 같다.

- Regression RMSE와 MAE에서는 기존 baseline을 능가하지 못했다.
- 방향 정확도는 상대적으로 가장 높지만 50% 미만이다.
- Classification과 Backtest에서는 Cryptova보다 낮았다.
- Rolling 3에서 성능이 크게 악화되어 안정적인 regime 일반화가 확인되지 않았다.

---

## 10. 보고서 핵심 문장

### 모델 정의

> TimesFM 2.5는 연속된 32개 시점을 하나의 patch token으로 변환하고, Decoder-only
> Transformer의 causal self-attention으로 과거 patch 사이의 관계를 처리한 뒤 미래 구간을
> output patch 단위로 예측하는 사전학습 시계열 foundation model이다.

### 입력 길이

> TimesFM 2.5의 공식 최대 context는 16,384 시점이며, 본 실험의 64시간 입력은 모델의 최대
> 한계가 아니다. 공통 72시간 정보 범위를 초과하지 않으면서 32-step patch 구조를 유지하기
> 위해 최근 64시간을 사용했다.

### LoRA

> 전체 232,672,192개 parameter 중 1,382,912개만 학습하는 LoRA를 적용해 계산·메모리 비용과
> 제한된 BTC 데이터에서의 과적합 위험을 줄였으며, 본 실험은 Full fine-tuning이 아니다.

### Regression 결과

> TimesFM 2.5 LoRA Fine-tuned는 Connected OOS에서 RMSE 0.024992와 MAE 0.017804를 기록해
> 비교 회귀모델 중 가장 큰 수익률 예측 오차를 보였다. 방향 정확도는 49.83%로 상대적으로
> 가장 높았지만 50%를 넘지 못했다.

### Classification 결과

> 예측의 89.38%가 HOLD에 집중되었으며 SHORT와 LONG recall은 각각 5.74%와 6.25%였다.
> 이에 따라 Macro F1은 0.287013에 그쳤다.

### 거래성과

> 비용과 24시간 non-overlap 규칙을 적용한 Connected OOS Backtest에서 누적수익률 -30.96%,
> Sharpe-like -1.346, MDD -41.22%를 기록했으며 Rolling 3의 -28.70% 손실이 전체 성과 악화의
> 주요 원인이었다.

### 종합 결론

> 대규모 사전학습과 LoRA 적응에도 불구하고 TimesFM 2.5는 현재 close-only BTC forecasting
> 조건에서 기존 baseline과 Cryptova를 능가하지 못했다. 이는 foundation model의 일반적인
> 사전학습 능력이 특정 금융 downstream target의 예측력으로 자동 전이되지는 않음을 보여준다.

---

## 11. 공식 자료

- [TimesFM 논문: A Decoder-Only Foundation Model for Time-Series Forecasting](https://arxiv.org/abs/2310.10688)
- [Google Research TimesFM 설명](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [TimesFM 공식 GitHub](https://github.com/google-research/timesfm)
- [TimesFM 2.5 Hugging Face checkpoint](https://huggingface.co/google/timesfm-2.5-200m-transformers)
- [TimesFM 2.5 Transformers 문서](https://huggingface.co/docs/transformers/model_doc/timesfm2_5)
- [TimesFM 2.5 공식 config](https://huggingface.co/google/timesfm-2.5-200m-transformers/blob/main/config.json)
