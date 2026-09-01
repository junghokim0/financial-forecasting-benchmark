# Chronos-2 LoRA Fine-tuned 연구 노트

## 1. 정의

Chronos-2는 Amazon이 공개한 약 120M parameter의 pretrained time-series foundation
model이다. 연속적인 시계열을 patch로 나누고, Time Self-Attention과 Group
Self-Attention을 적용한 Encoder-only Transformer를 통해 미래 여러 시점의 quantile을
직접 예측한다.

본 benchmark는 공식 `amazon/chronos-2` checkpoint를 사용하여 과거 72시간의 Raw BTC
Close와 Chart Feature 12개로 미래 24시간의 Close 경로를 예측한다. `q=0.5` 중앙값의
마지막 시점으로 미래 24시간 수익률을 계산하고, 같은 수익률 threshold를 적용하여
`SHORT/HOLD/LONG`으로 변환한다.

```text
Raw BTC Close 72시간
+ Chart Feature 12개 × 72시간
→ Chronos-2 LoRA Fine-tuned
→ 미래 Close 24시간의 Quantile
→ t+24 q50 수익률
→ SHORT / HOLD / LONG
```

본 실험은 Full fine-tuning이 아니다. 결과와 보고서의 모델명은 반드시 다음과 같이
표기한다.

> **Chronos-2 LoRA Fine-tuned**

## 2. 특징

- 연속적인 시계열을 이산 token이 아니라 16시점 단위 patch로 처리한다.
- T5-inspired Encoder-only Transformer로 미래 여러 시점을 직접 예측한다.
- Time Attention으로 각 시계열의 patch 사이 시간 관계를 학습한다.
- Group Attention으로 같은 group에 속한 target과 covariate 사이 관계를 학습한다.
- 단변량, 다변량, past covariate 및 known future covariate forecasting을 지원한다.
- 하나의 미래값만 출력하지 않고 각 미래 시점의 21개 quantile을 직접 출력한다.
- 대규모 시계열 사전학습 표현을 사용하며, 본 실험에서는 LoRA로 BTC 데이터에 적응한다.
- 미래 가격분포 예측에 최적화된 모델이므로 SHORT/HOLD/LONG 분류나 투자수익을 직접
  최적화하는 모델은 아니다.

Chronos-2가 높은 표현력을 갖는다는 사실이 암호화폐 Test 성능을 보장하지는 않는다.
사전학습 지식이 BTC 시장에 전이되어야 하며, 미래 가격 quantile을 잘 예측하는 목적이
수익률 방향 분류 및 거래비용 차감 수익과도 연결되어야 한다.

## 3. 문제 제기와 실험 목적

Ridge, LSTM 및 TimesNet은 현재 benchmark의 Rolling Train 데이터에서 처음부터 학습한다.
반면 Chronos-2는 다양한 실제·합성 시계열로 대규모 사전학습된 foundation model이다.
따라서 다음 질문을 검증한다.

1. 범용 사전학습 시계열 표현이 제한된 BTC 데이터의 미래 수익률 예측에 도움이 되는가?
2. Group Attention이 Close와 Chart Feature 12개의 관계를 유효하게 사용할 수 있는가?
3. Quantile forecasting 성능이 SHORT/HOLD/LONG 분류력으로 연결되는가?
4. 예측력이 동일한 거래비용과 24시간 non-overlap backtest에서 경제적 가치로 이어지는가?
5. 범용 foundation model이 Chart+News 특화 모델인 Cryptova와 경쟁할 수 있는가?

Foundation model은 외부 대규모 데이터로 사전학습됐기 때문에 LSTM·TimesNet·Cryptova와
parameter 수나 사전학습 데이터가 동일하지 않다. 본 연구는 동일 parameter budget
비교가 아니라 동일한 BTC timestamp, target, rolling split 및 evaluator에서 각 모델의
**end-to-end 경쟁력**을 비교한다.

## 4. 모델 구조

### 4.1 기존 Chronos의 구조

기존 Chronos는 연속 시계열 값을 scaling하고 이산 구간으로 quantization하여 언어
token과 유사한 고정 vocabulary로 변환한다. Amazon의 공식 Chronos 모델들은 주로 T5
Encoder-Decoder 구조를 기반으로 하며 cross-entropy로 미래 token을 학습한다.

```text
시계열 값
→ Scaling
→ Quantization
→ 숫자 Token
→ T5 Encoder-Decoder
→ 미래 Token을 Autoregressive하게 생성
→ 미래 값으로 복원
```

예를 들어 scaling된 값이 다음 token으로 바뀔 수 있다.

```text
0.980 → Token 102
0.990 → Token 104
0.985 → Token 103
1.000 → Token 106
```

Decoder는 미래값을 한 시점씩 생성한다.

```text
과거 Token
→ t+1 Token 생성
→ 생성한 t+1을 사용해 t+2 Token 생성
→ 반복
→ t+24 Token 생성
```

따라서 기존 Chronos를 단순히 Decoder-only라고 부르는 것은 정확하지 않다. 공식 모델은
주로 T5 Encoder-Decoder지만, 출력 방식이 autoregressive하다는 점에서 생성 모델의
성격을 갖는다. 여러 미래 token 경로를 sampling한 후 이를 값으로 복원해 median과
quantile을 계산한다.

### 4.2 Chronos와 Chronos-2의 차이

두 모델의 가장 중요한 차이는 다음 문장으로 정리할 수 있다.

> 기존 Chronos는 시계열 값을 이산 token으로 바꾸고 미래 token을 순차적으로 생성하며,
> Chronos-2는 연속적인 시계열을 patch로 바꾸고 미래 quantile patch를 직접 예측한다.

| 구분 | 기존 Chronos | Chronos-2 |
|---|---|---|
| 입력 표현 | Scaling 후 이산 token | Scaling 후 연속값 patch |
| 기본 구조 | 주로 T5 Encoder-Decoder | T5-inspired Encoder-only |
| 미래 예측 | Autoregressive token 생성 | Direct multi-patch prediction |
| 예측 단위 | 한 시점씩 순차 생성 | 여러 미래 시점을 동시에 출력 |
| 앞 예측값 재사용 | 사용 | 사용하지 않음 |
| 확률분포 | 여러 sampled path에서 계산 | 21개 quantile 직접 출력 |
| Multivariate | 기본적으로 제한적 | 지원 |
| Past/known covariate | 기본적으로 제한적 | 공식 지원 |
| 변수 관계 | 별도 구조 없음 | Group Attention |
| 시간 관계 | 일반 Transformer attention | Time Attention |
| 입력 정보 손실 | Quantization 오차 가능 | 연속값 patch 사용 |

```text
기존 Chronos
과거 → t+1 → t+2 → t+3 → ... → t+24

Chronos-2
과거 ─────────────────────→ [t+1, t+2, ..., t+24]
```

Chronos-2가 미래 시점을 서로 완전히 독립적으로 예측한다는 뜻은 아니다. 미래 시점들은
output patch representation을 통해 함께 모델링되지만, 앞에서 예측한 값을 다음 입력으로
되먹이는 autoregressive 과정은 없다.

### 4.3 Chronos-2 전체 흐름

```text
Target + Covariates
→ Instance Normalization
→ Time Index와 Observation Mask 추가
→ 16시점 단위 Patch 생성
→ Residual Patch Embedding
→ Encoder Block × 12
   ├─ Time Self-Attention
   ├─ Group Self-Attention
   └─ Feed Forward Network
→ Final LayerNorm + Dropout
→ Multi-patch Quantile Output Head
→ 미래 여러 시점의 Quantile 예측
```

### 4.4 입력 종류와 우리 입력

Chronos-2는 다음 입력을 구분한다.

| 입력 | 의미 | 예시 |
|---|---|---|
| Target | 실제로 예측할 시계열 | BTC Close |
| Past covariate | 과거 값만 제공하는 보조 시계열 | 과거 수익률, 거래량, MACD |
| Known future covariate | 미래 값도 미리 알려진 변수 | 달력, 공휴일, 예약된 일정 |

우리 실험은 다음 입력만 사용한다.

```text
Target
└─ Raw BTC Close [72]

Past covariates
├─ log_return [72]
├─ return_6h [72]
├─ return_24h [72]
├─ std_24h [72]
├─ close_ma24_gap [72]
├─ close_ma72_gap [72]
├─ volume_ratio_24 [72]
├─ spread_ratio [72]
├─ macd_hist [72]
├─ hour_sin [72]
├─ hour_cos [72]
└─ is_missing_candle [72]

Known future covariates
└─ 사용하지 않음
```

한 sample은 개념적으로 다음 shape로 볼 수 있다.

```text
Target 1개 + Past Covariate 12개
≈ [Variate 13, Time 72]
```

`close`는 예측 대상 target이며 Chart Feature 13번째 항목으로 세지 않는다. 이번 실험은
News를 제외한 Chart-only Track이다.

### 4.5 Instance Normalization

Close, 거래량, 수익률처럼 단위와 scale이 다른 시계열을 안정적으로 처리하기 위해
Chronos-2는 모델 내부 instance normalization을 사용한다. 공식 checkpoint는 robust
scaling과 `arcsinh` 변환을 적용한다.

우리 benchmark에는 다음 두 단계가 있다.

```text
Chart Feature 12개
→ 각 Rolling Train에서 fit한 StandardScaler
→ Chronos-2 내부 Instance Normalization

Raw Close
→ Chronos-2 내부 Instance Normalization
```

첫 번째는 공통 benchmark 전처리이고 두 번째는 Chronos-2의 공식 preprocessing이다.
Validation과 Test에서 scaler를 다시 fit하지 않는다.

### 4.6 Patch 생성과 Embedding

Chronos-2는 각 시간값을 하나의 token으로 넣지 않고 연속된 16개 시점을 하나의 patch로
묶는다.

```text
input_patch_size   = 16
input_patch_stride = 16
output_patch_size  = 16
```

과거 72시간은 개념적으로 다음과 같이 구성된다.

```text
Patch 1: 과거 16시간
Patch 2: 다음 16시간
Patch 3: 다음 16시간
Patch 4: 다음 16시간
Patch 5: 나머지 최근 구간 + Padding
```

72는 16으로 나누어떨어지지 않으므로 padding과 mask로 길이를 맞춘다. Padding 위치는
attention에서 관측값으로 사용되지 않는다.

각 patch에는 다음 정보가 결합된다.

```text
정규화된 시계열 값
+ 시간 위치 정보
+ 관측값/결측값 Mask
→ Residual Patch Embedding
→ 768차원 Hidden Representation
```

따라서 개념적 shape 변화는 다음과 같다.

```text
Patch 전: [Variate 13, Time 72]
Patch 후: [Variate 13, Patch 약 5, Hidden 768]
```

실제 내부 tensor에는 batch, special token 및 mask 차원이 추가될 수 있으므로 위 shape는
구조를 이해하기 위한 개념적 표현이다.

### 4.7 Encoder Block

Chronos-2에는 12개의 Encoder Block이 있다. 공식 구현에서 각 block은 정확히 다음 순서로
구성된다.

```text
Input Hidden State
→ Time Self-Attention
→ Group Self-Attention
→ Feed Forward Network
→ 다음 Encoder Block
```

각 attention과 Feed Forward에는 normalization, dropout 및 residual connection이 포함된다.
주요 architecture 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| 전체 parameter | 약 120M |
| Encoder block | 12 |
| Hidden dimension | 768 |
| Attention head | 12 |
| Head dimension | 64 |
| Feed-forward dimension | 3,072 |
| Dropout | 0.1 |
| Input patch size/stride | 16/16 |
| Output patch size | 16 |
| Quantile | 21개 |

### 4.8 Time Self-Attention

Time Attention은 하나의 target 또는 feature 시계열 안에서 서로 다른 시간 patch의 관계를
계산한다.

```text
Close Patch 1 ↔ Close Patch 2 ↔ Close Patch 3 ↔ Close Patch 4
MACD  Patch 1 ↔ MACD  Patch 2 ↔ MACD  Patch 3 ↔ MACD  Patch 4
Volume Patch 1 ↔ Volume Patch 2 ↔ Volume Patch 3 ↔ Volume Patch 4
```

이를 통해 각 변수의 추세, 반전, 변동성 변화, 반복 패턴, 최근 구간과 과거 구간의 관계를
표현할 수 있다.

> Time Attention이 처리하는 질문: 이 변수는 시간에 따라 어떻게 변해 왔는가?

### 4.9 Group Self-Attention

Group Attention은 동일한 patch 위치에서 같은 group에 속한 target과 covariate 사이의
관계를 계산한다.

```text
동일한 16시간 Patch 위치

Close
↕
log_return
↕
std_24h
↕
volume_ratio_24
↕
macd_hist
↕
나머지 Chart Feature
```

여기서 동일 시간은 한 시간의 한 점이 아니라 동일한 16시간 patch 위치를 뜻한다.

> Group Attention이 처리하는 질문: 같은 시간 구간에서 target과 covariate는 어떤 관계를
> 보이는가?

### 4.10 Time Attention과 Group Attention의 결합

각 block에서 Time Attention 다음에 Group Attention이 실행되고 이 과정이 12개
Encoder Block에서 반복된다.

```text
Time Attention
→ 각 변수의 시간 흐름 계산
→ Group Attention
→ 같은 구간의 변수 관계 결합
→ 다음 Block의 Time Attention
→ 결합된 정보가 시간축을 따라 전달
→ 다음 Group Attention
→ 새로운 시간 구간의 변수 관계와 재결합
```

따라서 Group Attention이 직접적으로는 같은 patch 위치를 비교하더라도 여러 block을
거치면 다음과 같은 lead-lag 관계도 표현할 수 있다.

```text
과거 거래량 급증
→ 이후 변동성 확대
→ 이후 Close 하락
```

두 attention은 다음 두 방향을 번갈아 처리한다고 이해할 수 있다.

```text
                 시간축
          Patch1  Patch2  Patch3
Close        ● ───── ● ───── ●
Return       ● ───── ● ───── ●
Volume       ● ───── ● ───── ●
MACD         ● ───── ● ───── ●
             │       │       │
             │       │       │
           변수축   변수축   변수축

가로 방향: Time Attention
세로 방향: Group Attention
```

### 4.11 Encoder-only Direct Forecasting

Chronos-2는 과거 patch 뒤의 미래 위치를 masked patch로 구성하고 미래 여러 시점을 한
번의 forward pass에서 직접 예측한다.

```text
과거 Patch 1: 관측값
과거 Patch 2: 관측값
과거 Patch 3: 관측값
과거 Patch 4: 관측값
과거 Patch 5: 관측값
미래 Patch 1: MASK
미래 Patch 2: MASK
        ↓
Encoder-only Transformer
        ↓
미래 Patch 1과 2의 Quantile 직접 출력
```

우리 horizon은 24시간이므로 16개 단위 output patch 두 개가 미래 범위를 덮고, pipeline은
요청한 24시간만 반환한다. 별도의 autoregressive Decoder가 없고 앞에서 예측한 값을
다음 입력으로 사용하지 않는다.

### 4.12 Quantile Output Head

Chronos-2는 각 미래 시점에 대해 다음 21개 quantile을 직접 출력한다.

```text
0.01, 0.05, 0.10, 0.15, ..., 0.50, ..., 0.90, 0.95, 0.99
```

예를 들어 t+24 출력은 다음처럼 해석할 수 있다.

```text
q10 = 95,000  → 낮은 가격 시나리오
q50 = 101,000 → 중앙값 Point Forecast
q90 = 107,000 → 높은 가격 시나리오
```

우리 코드는 전체 quantile 중 `q10`, `q50`, `q90`을 요청한다. Output Head는 16시간
단위 patch 두 개로 최대 32시간을 출력하고, pipeline은 요청한 미래 24시간만 반환한다.
이후 우리 코드는 마지막 `t+24` 값만 수익률로 변환해 저장한다.

```text
Output Head 내부:     [Target 1, Future 32, Quantile 21]
24시간으로 자른 결과: [Target 1, Future 24, Quantile 21]
요청한 Quantile 결과: [Target 1, Future 24, Quantile 3]
표본별 최종 저장값:   t+24의 q10·q50·q90 수익률 3개
```

#### q10·q90을 저장한 이유와 향후 Risk Filter

현재 Benchmark의 Regression 평가, 신호 생성 및 Backtest에는 `t+24`의 `q50`만 사용한다.
`q10`과 `q90`은 현재 성과를 계산하는 데 사용하지 않았으며, Chronos-2의 확률적 예측 정보를
보존해 향후 불확실성 분석과 Risk Filter 실험에 사용할 수 있도록 함께 저장했다.

```text
q50 → 대표 수익률 예측과 기본 LONG/HOLD/SHORT 신호
q10·q90 → 예측 범위와 하방·상방 위험 정보
```

예를 들어 `q90_return - q10_return`이 크면 모델이 미래 범위를 넓게 예상한 것이므로 거래를
`HOLD`로 바꾸는 filter를 검토할 수 있다. LONG 후보에서는 낮은 시나리오인 `q10_return`도
양수이거나 거래비용을 넘는지, SHORT 후보에서는 높은 시나리오인 `q90_return`도 음수인지
확인하는 보수적인 규칙도 가능하다.

다만 이는 **현재 결과에 적용한 규칙이 아니라 향후 별도 실험 후보**다. 먼저 실제 t+24
수익률이 `q10~q90` 안에 들어오는 비율을 측정해 예측 구간의 calibration을 확인해야 한다.
Filter 기준은 Rolling Validation에서만 정하고, Rolling Test에는 변경 없이 적용해야 한다.

## 5. 실험 가정과 공정성

### 5.1 데이터와 Target

- Target: Raw BTC Close의 미래 24시간 경로
- 입력 context: 과거 72시간
- Past covariates: 기존 Chart Feature 12개
- News Feature: 사용하지 않음
- Known future covariate: 사용하지 않음
- Point forecast: `q=0.5` 중앙값의 t+24 Close
- Regression target: `raw_future_return`
- Classification: 예측 수익률에 고정 `±0.012` threshold 적용

```text
predicted_return = predicted_close_q50(t+24) / close(t) - 1
```

`close`는 target이므로 Chart Feature 개수에 포함하지 않는다.

### 5.2 데이터 누수 방지

- 시간 순서를 유지한 기존 Rolling 1~3 split을 사용한다.
- Chart scaler는 각 Rolling Train에서만 fit한 기존 scaler를 사용한다.
- Train과 Validation의 연속 시계열로 LoRA를 학습하고 checkpoint를 선택한다.
- Validation quantile loss만 checkpoint 선택에 사용한다.
- Test는 모든 선택이 끝난 후 OOS 평가에만 사용한다.
- Validation/Test에서 scaler를 다시 fit하지 않는다.
- 예측 시점 이후의 Chart covariate는 제공하지 않는다.

### 5.3 Foundation Model 비교의 의미

Chronos-2는 외부 데이터로 사전학습됐으므로 로컬 데이터에서 처음부터 학습한 모델과
pretraining data 또는 parameter budget이 동일하지 않다. 본 비교는 foundation model을
실제로 사용할 때의 사전학습 이점을 포함한 end-to-end 비교다.

동일하게 맞춘 조건은 다음과 같다.

- BTC timestamp와 Rolling Test 구간
- 미래 24시간 horizon
- 미래 수익률 및 분류 label 정의
- `-0.012/+0.012` 신호 threshold
- 공통 prediction schema와 evaluator
- fee 0.1%, slippage 0.1%
- 24시간 non-overlap backtest
- Test를 선택에 사용하지 않는 규칙

### 5.4 Cross-learning 설정

추론에서 `cross_learning=False`를 사용했다. 따라서 서로 다른 72시간 sample 사이의
정보는 추론 중 섞이지 않는다. 각 sample 내부에서는 Close와 Chart Feature 12개가 같은
group으로 구성되어 Group Attention을 수행한다.

### 5.5 LoRA 선택 이유와 설정

제한된 암호화폐 데이터로 전체 120M parameter를 업데이트하면 GPU·optimizer memory와
학습시간이 증가하고 과적합 및 사전학습 표현 손실 위험이 커진다. 따라서 본체 대부분을
고정하고 attention과 output layer의 LoRA adapter만 학습한다.

| 항목 | 값 |
|---|---:|
| Base checkpoint | `amazon/chronos-2` |
| Revision | `95a9710e2596287d08352589f42634fa5abdf0a7` |
| Total parameter | 120,684,576 |
| Trainable parameter | 1,206,912 |
| Trainable ratio | 약 1.0% |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA target | q/k/v/o + output patch layer |
| Learning rate | `1e-5` |
| Maximum step | 1,000 |
| Evaluation interval | 100 step |
| Seed | 42 |

이는 과적합을 완전히 방지한다는 뜻이 아니다. Trainable parameter와 optimizer state를
줄이고 사전학습 표현을 보존하는 효율적인 adaptation 방법으로 해석한다.

## 6. 학습과 추론 과정

### 6.1 Rolling 학습

각 Rolling은 동일한 공식 base checkpoint에서 독립적으로 시작한다.

```text
Rolling Train의 연속 Raw Close
+ Train scaler가 적용된 Chart Feature 12개
→ Chronos-2 공식 LoRA Fine-tuning
→ 100 step마다 Validation Quantile Loss
→ Validation Loss가 가장 낮은 Adapter 선택
→ Best Adapter 저장
```

Chronos-2의 공식 fine-tuning loss는 quantile loss다. 따라서 Ridge/LSTM/TimesNet
Regression의 MSE와 학습 objective는 다르지만, 최종 Test에서는 동일한 RMSE·MAE·상관계수
등으로 비교한다.

### 6.2 Best Model 저장

Cryptova의 `best_model.pth`에 대응하는 파일은 각 Rolling의 다음 adapter다.

```text
rolling_1/trainer/finetuned-ckpt/adapter_model.safetensors
rolling_2/trainer/finetuned-ckpt/adapter_model.safetensors
rolling_3/trainer/finetuned-ckpt/adapter_model.safetensors
```

| Rolling | Validation Best step | `finetuned-ckpt`와 동일 여부 |
|---|---:|---:|
| rolling_1 | 800 | 동일 |
| rolling_2 | 1,000 | 동일 |
| rolling_3 | 1,000 | 동일 |

파일 hash를 비교한 결과 각 `finetuned-ckpt` adapter는 해당 Validation best checkpoint와
동일하다. Adapter에는 LoRA 가중치만 들어 있으므로 추론 시 같은 revision의
`amazon/chronos-2` base checkpoint와 결합해야 한다.

### 6.3 OOS 추론과 평가

```text
OOS Raw Close Window [72]
+ OOS Scaled Chart Window [72,12]
→ Best Chronos-2 LoRA Adapter
→ 미래 Close Quantile [24,21]
→ q10/q50/q90 [24,3]
→ t+24 q50 Close
→ predicted_return
├─ Regression Metrics
└─ ±1.2% Threshold
   → SHORT/HOLD/LONG
   → Classification Metrics
   → 24시간 Non-overlap Backtest
```

신호 변환 규칙은 다음과 같다.

```text
predicted_return <= -0.012 → SHORT
-0.012 < predicted_return < 0.012 → HOLD
predicted_return >= 0.012 → LONG
```

Rolling 1~3 Test prediction을 시간순으로 연결하여 Connected OOS를 추가 평가한다.

## 7. 결과

### 7.1 Rolling별 Test 결과

| Rolling | Best step | RMSE | MAE | Pearson | Directional Accuracy | Macro F1 | Balanced Accuracy | Return | MDD | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_1 | 800 | 0.015879 | 0.011844 | -0.0228 | 49.31% | 0.259189 | 0.328308 | -0.66% | -2.61% | 6 |
| rolling_2 | 1,000 | 0.024393 | 0.017752 | +0.0455 | 52.30% | 0.267978 | 0.349631 | -8.70% | -18.20% | 28 |
| rolling_3 | 1,000 | 0.030530 | 0.021958 | -0.2415 | 45.57% | 0.213475 | 0.334220 | -16.90% | -17.03% | 25 |

Rolling 2에서는 방향 정확도가 50%를 넘었지만 수익률과 MDD는 음수였다. Rolling 3에서는
Pearson이 `-0.2415`로 크게 악화되어 regime에 따른 일반화 불안정성이 나타났다.

### 7.2 연결 Regression 결과

| Test samples | RMSE | MAE | Pearson | Spearman | Directional Accuracy |
|---:|---:|---:|---:|---:|---:|
| 6,291 | 0.024300 | 0.017148 | -0.1095 | -0.0305 | 49.09% |

미래 24시간 수익률의 RMSE는 약 2.43%다. Pearson과 Spearman이 모두 음수이고 방향
정확도도 50%보다 낮아 Connected OOS에서 안정적인 수익률 예측력을 확인하기 어렵다.

### 7.3 연결 Classification 결과

| Accuracy | Macro F1 | Balanced Accuracy | Weighted F1 |
|---:|---:|---:|---:|
| 50.39% | 0.252000 | 0.338928 | 0.361797 |

| Class | Precision | Recall | F1 | Predicted Count | Predicted Ratio |
|---|---:|---:|---:|---:|---:|
| SHORT | 0.3918 | 0.0220 | 0.0417 | 97 | 1.54% |
| HOLD | 0.5147 | 0.9715 | 0.6729 | 6,023 | 95.74% |
| LONG | 0.1871 | 0.0233 | 0.0414 | 171 | 2.72% |

실제 HOLD label은 3,191개이므로 항상 HOLD만 예측해도 Accuracy는 약 50.72%다. Chronos-2
Accuracy 50.39%는 이 기준보다 약간 낮다. Balanced Accuracy도 무작위 3-class 기준인
약 0.333과 비슷하며 실제 SHORT와 LONG의 약 2%만 찾아냈다.

### 7.4 연결 Backtest 결과

| Return | Sharpe-like | MDD | Trades | Trade Ratio | Win Rate | Avg Trade Return |
|---:|---:|---:|---:|---:|---:|---:|
| -24.64% | -1.406 | -24.75% | 59 | 0.94% | 35.59% | -0.439% |

평가는 선택된 거래당 fee 0.1%, slippage 0.1%, 총 0.2% 비용과 24시간 non-overlap 규칙을
적용했다. 거래 수는 적었지만 승률과 평균 거래수익률이 모두 낮아 손실이 수수료만으로
발생했다고 해석할 수 없다.

## 8. 결과 해석

### 8.1 Regression 예측력

Chronos-2의 Connected RMSE `0.024300`은 현재 Ridge-Flat `0.023178`, LSTM Regression
`0.023790`, TimesNet Regression `0.023916`보다 높다. MAE는 비슷한 범위지만 Pearson
`-0.1095`와 Directional Accuracy 49.09%를 함께 고려하면 foundation model의 사전학습
표현이 현재 BTC 미래수익률 예측에서 추가적인 OOS 가치를 제공했다고 보기 어렵다.

이 결과가 모든 Chronos-2 설정이 실패한다는 뜻은 아니다. 현재 고정한 72시간 context,
24시간 horizon, Chart 12 past covariates, LoRA 1,000-step 및 Seed 42 조건의 결과다.

### 8.2 HOLD 편향

Chronos-2는 미래 가격분포의 quantile loss를 최소화하지만 benchmark의 최종 분류 목적은
`SHORT/HOLD/LONG`을 구분하는 것이다. 두 목적은 동일하지 않다.

```text
미래 가격분포 예측 목적
→ 보수적인 q50 중앙값
→ 예상수익률이 0% 근처에 집중
→ ±1.2% Threshold를 거의 넘지 않음
→ HOLD 95.74%
→ SHORT/LONG Recall 약 2%
```

따라서 Accuracy 50.39%만 보면 분류를 잘한 것처럼 보일 수 있지만, 항상 HOLD Accuracy보다
낮고 Macro F1과 Balanced Accuracy도 낮다.

### 8.3 Classification과 거래성과

현재 Classification Macro F1은 Chronos-2 `0.2520`, Ridge `0.2847`, LSTM Classifier
`0.3182`, TimesNet Classifier `0.3647` 순으로 Chronos-2가 가장 낮다. Connected OOS
수익률도 `-24.64%`로 음수다.

Chronos-2는 classification을 직접 Cross-Entropy로 학습한 모델이 아니라 predicted return을
threshold로 변환한 forecasting model이라는 점을 함께 보고해야 한다. 그럼에도 동일한
신호 규칙과 evaluator에서 거래 가능한 예측력은 확인되지 않았다.

### 8.4 Covariate와 Foundation Model에 대한 해석

Group Attention이 Chart Feature 12개를 입력받았다는 사실만으로 feature가 자동으로
유용해지는 것은 아니다. 다음 가능성이 남아 있다.

- Chart Feature와 미래수익률 관계가 시장 regime마다 달라질 수 있다.
- 과거 72시간이 24시간 horizon을 설명하기에 부족하거나 부적절할 수 있다.
- Raw Close quantile loss가 작은 수익률 방향 신호와 맞지 않을 수 있다.
- News와 갑작스러운 외부 사건이 Chart-only 입력에 포함되지 않는다.
- 제한된 LoRA update만으로 BTC-specific 관계를 충분히 적응하지 못했을 수 있다.
- 사전학습의 범용 예측 능력이 금융시장의 낮은 signal-to-noise ratio를 극복하지 못했을 수
  있다.

### 8.5 현재 단계의 결론

```text
수익률 Regression 성능       → 기존 단순 모델보다 개선되지 않음
SHORT/HOLD/LONG 균형 성능    → 낮음, HOLD 95.74% 편향
비용 반영 Backtest           → -24.64%
시장 구간별 안정성           → 낮음
Foundation benchmark 역할    → 정상적으로 수행
```

정확한 결론은 다음과 같다.

> Chronos-2가 일반적으로 성능이 낮은 모델이라는 뜻이 아니라, BTC 1시간 데이터,
> 72시간 context, 24시간 horizon, Chart 12 past covariates, LoRA 1,000-step 설정에서는
> 유의미한 수익률 예측력과 거래성과를 보이지 못했다.

## 9. 공식 비교에서의 역할

Chronos-2의 공식 역할은 **Regression Track의 foundation model baseline**이다.

```text
Regression Track
→ q50의 t+24 predicted return
→ RMSE / MAE / Pearson / Spearman / Directional Accuracy

Classification 변환 Track
→ 동일 predicted return
→ ±1.2% threshold
→ Macro F1 / Balanced Accuracy / Class Recall

Trading Evaluation
→ 변환된 SHORT/HOLD/LONG
→ 동일 비용과 non-overlap 규칙
→ Return / Sharpe-like / MDD / Trades
```

Chronos-2의 주 목적은 수익률 및 가격 forecasting이며, 분류는 회귀 예측값을 공통 신호로
변환한 보조 비교다. Cryptova는 Chart+News로 class를 직접 학습하므로 동일 architecture
비교가 아니라 완성된 end-to-end 모델의 경쟁력 비교로 해석한다.

## 10. 보고서 핵심 문장

> 본 연구는 범용 시계열 사전학습 표현이 제한된 BTC 데이터에서 전통 모델과 도메인 특화
> Cryptova보다 높은 예측력을 제공하는지 검증하기 위해 Amazon의 120M parameter
> Chronos-2를 LoRA로 fine-tuning하였다.

> Chronos-2는 과거 시계열을 16시점 단위 patch로 변환하고, 각 변수의 시간 관계를 처리하는
> Time Self-Attention과 동일 patch 위치의 target-covariate 관계를 처리하는 Group
> Self-Attention을 12개 Encoder Block에서 순차적으로 적용한다.

> 기존 Chronos가 시계열 값을 이산 token으로 변환하고 T5 Encoder-Decoder로 미래 token을
> autoregressive하게 생성하는 것과 달리, Chronos-2는 연속값 patch와 Encoder-only
> Transformer를 사용하여 미래 여러 시점의 quantile을 직접 예측한다.

> 본 실험은 Raw BTC Close를 target으로, 기존 Chart Feature 12개를 past covariate로
> 사용했으며, 과거 72시간으로 미래 Close 24시간을 예측한 뒤 t+24 q50 중앙값을 미래
> 수익률로 변환하였다.

> Chronos-2 LoRA Fine-tuned는 연결된 6,291개 out-of-sample 표본에서 RMSE 0.02430,
> MAE 0.01715, Pearson correlation -0.1095 및 방향 정확도 49.09%를 기록하여 현재 설정에서
> 안정적인 미래수익률 예측력을 보이지 못했다.

> 예측의 95.74%가 HOLD에 집중되어 Macro F1은 0.2520, Balanced Accuracy는 0.3389에
> 그쳤으며, 동일 거래비용과 24시간 non-overlap 규칙을 적용한 Connected OOS 수익률은
> -24.64%였다.

> 이 결과는 Chronos-2 자체의 일반적 한계를 의미하는 것이 아니라 BTC 1시간 데이터,
> 72시간 context, 24시간 horizon, Chart 12 past covariates, LoRA 1,000-step 및 Seed 42로
> 고정한 benchmark 조건에서의 out-of-sample 결과로 해석한다.

## 11. 공식 자료

- [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821)
- [Amazon Chronos-2 Model Card](https://huggingface.co/amazon/chronos-2)
- [Amazon Chronos-2 Config](https://huggingface.co/amazon/chronos-2/blob/main/config.json)
- [Amazon Chronos-2 Official Implementation](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/model.py)
- [Amazon Science Chronos-2 Introduction](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
