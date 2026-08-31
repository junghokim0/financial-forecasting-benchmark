# Linear Regression 및 Ridge-Flat 연구 노트

## 1. Linear Regression의 정의

Linear Regression은 입력 변수의 선형결합으로 연속형 target을 예측하는 지도학습 모델이다.

\[
\hat{y}=\beta_0+\beta_1x_1+\cdots+\beta_px_p
\]

- \(x_j\): 입력 feature
- \(\beta_j\): 학습되는 회귀계수
- \(\beta_0\): 절편
- \(\hat y\): 예측값

현재 연구에서는 target이 하나의 미래 24시간 수익률 `raw_future_return`이고 입력 feature가 여러 개이므로 **다중 선형회귀(multiple linear regression)** 문제에 해당한다. 여러 target을 동시에 예측하는 다변량 회귀(multivariate regression)와는 구분한다.

선형회귀에서 말하는 선형성은 입력과 target의 관계가 반드시 단순한 직선이어야 한다는 의미보다, 모델이 파라미터 \(\beta\)에 대해 선형이라는 뜻이다. 예를 들어

\[
\hat y=\beta_0+\beta_1x+\beta_2x^2
\]

도 \(x^2\)를 별도 feature로 제공하면 파라미터들이 선형으로 결합되므로 선형회귀에 해당한다.

일반 OLS(Ordinary Least Squares)는 실제값과 예측값 사이의 제곱오차 합을 최소화한다.

\[
\mathcal L_{OLS}=\sum_{i=1}^{N}(y_i-\hat y_i)^2
\]

이론적 해는 다음과 같이 표현할 수 있다.

\[
\hat\beta=(X^TX)^{-1}X^Ty
\]

실제 구현에서는 수치적 안정성을 위해 역행렬을 직접 계산하기보다 QR, SVD 또는 안정적인 least-squares solver를 사용한다.

## 2. Linear Regression의 특징

- 구조가 단순하고 학습 속도가 빠르다.
- 회귀계수를 통해 각 입력 변수가 예측에 미치는 방향과 크기를 해석하기 쉽다.
- 입력 feature의 선형결합만 사용하므로 복잡한 비선형 관계를 스스로 학습하지 못한다.
- 시계열의 시간 순서, 인접성 및 주기를 자동으로 이해하지 못한다.
- 입력 변수 사이의 공선성이 강하면 OLS 계수가 불안정해질 수 있다.
- 복잡한 모델이 단순한 선형 신호보다 실제로 나은지 확인하기 위한 baseline으로 적합하다.

본 benchmark에서 Linear Regression의 목적은 금융 데이터가 모든 고전적 회귀 가정을 만족한다고 주장하는 것이 아니다. 동일한 데이터에서 LSTM, TimesNet, foundation model 및 Cryptova가 단순한 선형 예측 관계를 얼마나 넘어서는지 측정하는 기준선으로 사용한다.

## 3. 문제 제기: 일반적인 Linear Regression을 바로 사용할 수 없는 이유

### 3.1 입력 shape의 차이

현재 Chart 데이터에서 한 sample은 과거 72시간과 시간별 12개 feature로 구성된다.

```text
한 sample의 Chart 입력: (72, 12)
전체 입력:              (N, 72, 12)
target:                 (N,)
```

일반적인 Linear Regression은 `(N, p)` 형태의 2차원 feature matrix를 입력받는다. 따라서 `(N, 72, 12)`의 시간축을 어떤 방식으로 처리할지 먼저 결정해야 한다.

### 3.2 마지막 시점만 사용하는 방법의 정보 손실

마지막 시점만 선택하면 입력은 `(N, 12)`가 되어 일반 Linear Regression에 바로 넣을 수 있다.

```text
(N, 72, 12) → 마지막 시점 선택 → (N, 12)
```

그러나 Cryptova, LSTM 및 TimesNet은 과거 72시간 전체를 사용한다. 마지막 시점만 사용하면 나머지 71개 시점의 직접적인 값과 시간 경로를 버리게 되므로 주된 비교 baseline으로는 적합하지 않다고 판단하였다.

`return_24h`, `std_24h`, `close_ma72_gap`처럼 마지막 행 자체에 과거 요약정보가 포함된 feature도 있지만, 72시간 동안 값이 어떤 순서와 경로로 변화했는지는 보존하지 못한다.

### 3.3 Flatten 시 발생하는 공선성 문제

72시간 전체를 한 줄로 펼치면 입력은 864차원이 된다.

```text
log_return(t-71), ..., log_return(t)
return_6h(t-71),  ..., return_6h(t)
...
12개 feature × 72개 시점 = 864개 입력
```

동일한 feature 종류가 72번 들어가므로 중복이 아니냐는 문제가 제기되었다. 같은 계산식이 반복되더라도 각 lag는 서로 다른 시간의 값을 나타내므로 자동으로 정확한 중복이 되는 것은 아니다. 예를 들어 `log_return(t-1)`과 `log_return(t)`는 feature 종류는 같지만 측정 구간과 값이 다르다.

다만 다음과 같은 rolling feature는 인접 시점끼리 계산 구간 대부분을 공유한다.

- `return_6h(t)`와 `return_6h(t-1)`
- `return_24h(t)`와 `return_24h(t-1)`
- `std_24h(t)`와 `std_24h(t-1)`
- `close_ma72_gap(t)`과 `close_ma72_gap(t-1)`

따라서 정확한 복제보다 **강한 상관 또는 거의 완전한 공선성**이 발생할 가능성이 높다. 일반 OLS에서는 회귀계수가 과도하게 커지거나, 부호가 불안정해지거나, train 데이터의 작은 변화에도 계수가 크게 달라질 수 있다.

## 4. 문제 해결을 위해 검토한 방법

### 4.1 방법 A: 72시간을 1개 또는 소수의 요약값으로 압축

각 feature의 72개 값을 평균하면 다음과 같이 변환할 수 있다.

```text
(N, 72, 12) → 시간축 평균 → (N, 12)
```

평균 이외에 `mean`, `std`, `min`, `max`, `last`, `slope`를 계산하면 각 feature를 6개 통계로 요약할 수 있다.

```text
12개 feature × 6개 통계 = 72개 입력
```

장점은 차원, noise 및 공선성을 줄이고 해석을 쉽게 만든다는 것이다. 반면 단순 평균은 상승 후 하락과 하락 후 상승처럼 평균이 같고 경로가 다른 sample을 구분하지 못한다. 여러 통계를 사용하더라도 사건의 정확한 발생 시점과 72시간 내부의 순서가 일부 사라진다.

또한 사람이 선택한 요약 규칙이 추가되므로 다른 시계열 모델과 입력 정보량이 달라질 수 있다. 성능 차이가 모델 구조 때문인지 요약 과정의 정보 손실 때문인지 구분하기 어려워진다.

### 4.2 방법 B: Ridge-Flat

72시간 전체를 flatten한 뒤 Ridge Regression에 입력한다.

```text
Chart window              Flatten              Ridge Regression
(N, 72, 12)        →      (N, 864)       →     predicted_return (N,)
```

Ridge Regression의 loss는 다음과 같다.

\[
\mathcal L_{Ridge}
=
\sum_{i=1}^{N}(y_i-\hat y_i)^2
+
\lambda\sum_{j=1}^{864}\beta_j^2
\]

Ridge는 예측식을 비선형으로 바꾸는 것이 아니라 큰 회귀계수에 L2 penalty를 부과한다. 이로써 상관된 lag들의 계수가 과도하게 커지는 것을 억제하고 일반 OLS보다 안정적인 예측을 유도한다.

Ridge가 공선성을 제거하거나 중복 변수를 삭제하는 것은 아니다. 공선성이 존재하는 상태에서 계수의 크기와 variance를 줄여 모델을 안정화하는 방식이다.

### 4.3 Ridge-Flat의 구조적 한계

Flatten은 `(72, 12)`의 시간축을 864개의 입력 위치로 펼치는 **입력 변환 방식**이다. 각 lag의 값과 위치는 보존하지만 시점 간 인접성이나 반복 주기를 모델에 명시적인 구조로 제공하지 않는다.

Flatten된 입력에서도 시간 위치는 사라지지 않는다. `log_return(t-24)`, `log_return(t-1)`, `log_return(t)`는 서로 다른 열이며, Ridge는 각 lag에 별도의 선형계수를 부여한다. 따라서 최근 시점과 과거 시점의 중요도를 다르게 표현하는 등 **각 lag가 target에 미치는 개별적인 선형효과**는 학습할 수 있다.

그러나 Ridge는 864개 변수를 선형적으로 더할 뿐, 시간 순서를 처리하는 별도의 구조는 갖지 않는다. 이에 따라 다음과 같은 한계가 있다.

- **순차적 의존성:** 하락 후 반등처럼 여러 시점의 발생 순서가 함께 의미를 갖는 관계를 hidden state를 통해 학습하지 못한다.
- **시간적 인접성:** 인접한 시점들이 서로 가깝다는 관계를 공유된 규칙으로 사용하지 않는다. 같은 패턴이 시간축에서 이동하면 서로 다른 계수를 사용한다.
- **반복 주기:** `t-24` 등의 개별 효과는 반영할 수 있지만, TimesNet처럼 24시간 등의 주기를 자동으로 탐색하고 하나의 반복 구조로 학습하지 못한다.
- **비선형 상호작용:** 수익률과 거래량의 결합이나 특정 조건에서만 유효한 반등 신호를 자동으로 학습하지 못한다. 이런 관계는 사람이 interaction feature로 미리 만들어야 한다.
- **고정된 계수:** 동일한 계수가 모든 sample에 적용되므로 시장 regime이나 현재 문맥에 따라 특정 시점의 중요도를 동적으로 바꾸지 못한다.

따라서 Ridge-Flat이 시간 정보를 전혀 사용하지 못한다고 표현하는 것은 부정확하다. 정확한 해석은 다음과 같다.

> Ridge-Flat은 시간 위치별 lag의 개별적인 선형효과는 학습하지만, 시점 사이의 동적이고 비선형적인 의존관계를 시계열 구조로 학습하지 못한다.

이 한계를 기준으로 LSTM의 순차 상태 학습이나 TimesNet의 주기 모델링이 실제로 추가적인 성능을 제공하는지 평가할 수 있다. 따라서 Ridge-Flat은 최적의 시계열 모델이 아니라, 과거 72시간 전체를 사용하는 **규제 선형 baseline**으로 해석한다.

### 4.4 최종 선택

본 benchmark의 주 선형 baseline은 **Ridge-Flat**으로 결정한다.

1. Cryptova, LSTM 및 TimesNet과 동일하게 과거 72시간 전체의 Chart 정보를 제공한다.
2. 사람이 정한 요약 과정에서 발생할 수 있는 정보 손실을 피한다.
3. 성능 차이를 입력 범위보다 모델링 능력의 차이로 해석하기 쉽다.
4. Flatten으로 증가하는 lag 간 공선성과 계수 불안정성을 L2 regularization으로 완화한다.
5. 864차원은 현재 표본 규모에서 계산할 수 있다.

`Summary-Ridge`는 전체 benchmark 이후 요약에 따른 noise 감소와 정보 손실을 비교하는 보조 ablation으로 남긴다. `Linear-last`는 주 비교에서 제외하고 필요할 경우 최소 입력 기준선으로만 사용한다.

## 5. 실험

### 5.1 실험 가정

#### 선형 기준선의 역할

Ridge-Flat은 72시간 Chart feature와 미래 24시간 수익률 사이에 선형적으로 활용 가능한 신호가 있는지를 측정한다. 비선형 관계나 시계열 구조를 충분히 표현할 수 있다고 가정하지 않는다.

#### 동일한 Chart 정보 범위

Chart 기반 비교 모델에는 동일한 과거 72시간과 동일한 12개 Chart feature를 제공한다. Cryptova는 모델의 핵심인 News 입력을 추가로 사용하며, 이는 동일 입력만을 비교하는 구조 실험이 아니라 완성된 end-to-end 모델의 경쟁력 비교임을 명시한다.

#### 시간에 따른 관계의 변화

Train에서 학습한 관계가 바로 다음 validation과 test 구간에도 일부 유지될 수 있다고 전제한다. 금융시장의 관계가 시간에 따라 변할 수 있으므로 한 번의 무작위 분할이 아니라 여러 rolling 구간에서 평가한다.

#### 데이터 누수 방지

- scaler는 각 rolling의 train 데이터에만 fit한다.
- Ridge 계수는 train 데이터로 학습한다.
- 규제 강도 `alpha`는 validation 데이터에서만 선택한다.
- test 데이터는 hyperparameter 선택에 사용하지 않고 최종 평가에만 사용한다.
- 시간 순서를 유지하며 무작위 train/test split을 사용하지 않는다.

#### 동일한 평가 조건

모든 모델에 동일한 target, rolling split, 분류 threshold, 거래비용 및 24시간 non-overlap backtest 규칙을 적용한다. Non-overlap 규칙은 포지션 중첩을 막지만 학습 sample 자체를 독립적으로 만드는 규칙은 아니다.

### 5.2 실험 입력과 target

- 입력: 기존 rolling dataset의 Chart tensor `(N, 72, 12)`
- 변환: 시간 순서를 유지한 flatten `(N, 864)`
- target: 미래 24시간 실제 수익률 `raw_future_return`
- 출력: 미래 24시간 예측 수익률 `predicted_return`
- 학습 단위: 각 rolling 구간을 독립적으로 학습·검증·평가

기존 tensor가 rolling train 구간에 fit된 scaler로 이미 정규화되어 있다면 전체 데이터에 scaler를 다시 fit하지 않는다.

### 5.3 학습 및 Hyperparameter 선택

1. Train tensor를 `(N, 864)`로 flatten한다.
2. 여러 `alpha` 후보에 대해 Ridge 모델을 train에서 학습한다.
3. 각 alpha의 validation RMSE와, 예측 수익률을 고정된 ±1.2% 기준으로 변환한 Macro F1을 계산한다.
4. 최저 validation RMSE 모델을 `RMSE-selected`, 최고 validation Macro F1 모델을 `Macro-F1-selected`로 각각 선택한다.
5. Macro F1이 같으면 validation RMSE가 더 낮은 alpha를 선택한다.
6. 두 선택 과정 모두 test 데이터를 사용하지 않는다.
7. 선택된 두 모델로 test의 미래 수익률을 예측하고 공통 prediction schema로 저장한다.

`RMSE-selected`는 순수 수익률 예측의 보조 결과이고, `Macro-F1-selected`는 Cryptova의 SHORT/HOLD/LONG 성능과 비교하기 위한 주 선형 baseline이다. 두 variant는 Ridge 학습식이 다른 모델이 아니라 동일한 alpha 후보 중 어떤 validation 목적을 기준으로 선택했는지가 다르다.

### 5.4 평가

- 회귀 지표: MAE, RMSE, 실제 수익률과 예측 수익률의 상관계수
- 분류 변환: `predicted_return`에 동일한 `-0.012 / +0.012` threshold 적용
- 분류 지표: Accuracy, Macro F1
- 백테스트: 동일한 거래비용과 24시간 non-overlap 규칙 적용
- 투자 지표: 누적수익률, Sharpe-like, MDD, 거래 수, 거래 비율, 승률, 평균 거래수익률
- 진단 항목: residual, 시간에 따른 residual, residual ACF, alpha별 validation 성능, 계수 크기와 안정성

잔차의 정규성과 등분산성은 Ridge 학습 및 예측의 필수조건으로 사용하지 않는다. 필요한 경우 결과 해석과 regime별 오차 진단을 위한 보조 분석으로 확인한다.

## 6. 결과

### 6.1 방법 선택 결과

일반 OLS에 마지막 시점의 12개 feature만 넣는 방식은 나머지 71개 시점의 직접적인 정보를 버리므로 주 baseline에서 제외하였다. 72시간을 통계량으로 요약하는 방법은 차원과 noise를 줄일 수 있지만 시간 경로의 정보 손실과 연구자가 선택한 요약 규칙의 영향을 발생시킨다.

이에 따라 동일한 72시간 정보 범위를 유지하면서 공선성으로 인한 계수 불안정을 완화할 수 있는 **Ridge-Flat을 대표 선형 baseline으로 선택하였다.**

### 6.2 실제 실험 결과

Ridge-Flat을 rolling 1~3에서 실행했으며, 아래 값은 각 rolling의 validation에서 alpha를 선택한 뒤 test에서 한 번 평가한 결과다.

#### RMSE-selected

| Rolling | 선택 alpha | Test RMSE | Test Macro F1 | 누적수익률 | 거래 수 |
|---|---:|---:|---:|---:|---:|
| rolling_1 | 10,000 | 0.015569 | 0.259890 | 0.000000 | 0 |
| rolling_2 | 10,000,000,000 | 0.023963 | 0.212844 | 0.000000 | 0 |
| rolling_3 | 10,000,000,000 | 0.028301 | 0.194397 | 0.000000 | 0 |

RMSE 기준은 강한 규제를 선택하여 예측 수익률을 평균 근처로 축소했다. 세 test 구간 모두 예측값이 ±1.2% 거래 기준을 넘지 않아 전부 HOLD가 되었고 거래는 발생하지 않았다. 특히 rolling 2와 3은 매우 큰 alpha에서 validation RMSE가 평탄해져 사실상 절편 중심의 평균 예측에 가까운 결과다.

#### Macro-F1-selected

| Rolling | 선택 alpha | Test RMSE | Test Macro F1 | 누적수익률 | 거래 수 |
|---|---:|---:|---:|---:|---:|
| rolling_1 | 0.001 | 0.016537 | 0.297379 | -0.007684 | 17 |
| rolling_2 | 0.1 | 0.025633 | 0.245693 | -0.086828 | 32 |
| rolling_3 | 0.0001 | 0.029549 | 0.283922 | 0.103506 | 37 |
| 단순 평균 | — | 0.023906 | 0.275665 | 0.002998 | 28.7 |

Macro F1 기준은 더 약한 규제를 선택해 SHORT와 LONG 신호를 생성했다. 그러나 test 누적수익률은 rolling별로 음수와 양수가 혼재해 안정적인 수익 모델이라고 결론 내릴 수 없다. 이는 아직 단일 seed의 Ridge baseline 결과이며, Cryptova 및 다른 모델과의 공통 비교와 rolling 간 변동성 분석이 남아 있다.

두 결과의 차이는 수익률 숫자의 평균오차를 최소화하는 목적과 세 class를 균형 있게 구분하는 목적이 동일하지 않다는 것을 보여준다. 본 연구의 Cryptova 주 비교에는 `Macro-F1-selected`를 사용하고, `RMSE-selected`는 순수 forecasting 성능의 보조 결과로 보고한다.

### 6.3 결과에 대한 쉬운 해석

#### 두 결과는 무엇이 다른가?

두 variant 모두 과거 72시간 Chart 데이터를 flatten하고 `raw_future_return`을 target으로 Ridge Regression을 학습한다. 별도의 분류모델을 추가로 학습한 것이 아니라, 동일한 Ridge 후보 중 최종 alpha를 선택한 목적이 다르다.

```text
RMSE-selected
→ 미래 수익률 숫자를 가장 가깝게 맞히는 alpha 선택

Macro-F1-selected
→ 예측 수익률을 SHORT/HOLD/LONG으로 변환했을 때
  세 class를 가장 균형 있게 맞히는 alpha 선택
```

#### RMSE-selected가 전부 HOLD가 된 이유

암호화폐 수익률은 변동이 크기 때문에 상승이나 하락을 강하게 예측했다가 틀리면 제곱오차가 크게 증가한다. 이런 경우 Ridge는 적극적인 방향 예측보다 평균인 0% 근처를 예측하는 것이 RMSE에 유리할 수 있다.

```text
큰 alpha
→ 회귀계수 강하게 축소
→ 예측 수익률이 0% 근처로 모임
→ 예측값이 ±1.2%를 넘지 못함
→ 모두 HOLD
```

Rolling 2와 3에서 선택된 `alpha=10^10`은 feature의 영향을 거의 제거한 절편 중심의 평균 예측에 가깝다. 따라서 높은 HOLD Accuracy가 나타나더라도 방향 예측력이 높다는 의미는 아니다. 실제 HOLD 비율이 높은 데이터에서는 모든 sample을 HOLD로 예측해도 높은 Accuracy를 얻을 수 있기 때문이다.

#### Macro-F1-selected는 항상 HOLD보다 나았는가?

| Rolling | 항상 HOLD Macro F1 | Ridge Macro F1 | 개선 폭 |
|---|---:|---:|---:|
| rolling_1 | 0.259890 | 0.297379 | +0.037489 |
| rolling_2 | 0.212844 | 0.245693 | +0.032849 |
| rolling_3 | 0.194397 | 0.283922 | +0.089525 |

세 구간 모두 항상 HOLD보다 Macro F1은 높았다. 따라서 Ridge가 아무 정보도 학습하지 못한 것은 아니며 일부 SHORT/LONG 관련 선형 신호를 포착했다고 볼 수 있다. 그러나 개선 폭이 작고 예측은 여전히 HOLD에 집중되었다.

```text
rolling_1: HOLD 예측 95.7%
rolling_2: HOLD 예측 90.6%
rolling_3: HOLD 예측 81.2%
```

실제 SHORT와 LONG을 찾아낸 비율도 낮았다.

| Rolling | SHORT Recall | HOLD Recall | LONG Recall |
|---|---:|---:|---:|
| rolling_1 | 6.22% | 96.37% | 0.76% |
| rolling_2 | 1.38% | 90.11% | 6.41% |
| rolling_3 | 3.85% | 82.12% | 17.90% |

즉, HOLD를 찾는 능력은 높지만 실제 거래 신호인 SHORT와 LONG을 안정적으로 찾는 능력은 부족했다.

#### 수익률 예측값과 실제 수익률 사이의 관계

| Rolling | Test Pearson correlation | 방향 정확도 |
|---|---:|---:|
| rolling_1 | -0.014 | 50.97% |
| rolling_2 | -0.068 | 47.52% |
| rolling_3 | +0.049 | 46.20% |

Correlation은 모두 0에 가까웠고 부호도 rolling마다 달랐다. 방향 정확도 역시 약 46~51%로 50% 근처였다. 따라서 예측 수익률과 실제 수익률 사이에 안정적인 선형관계가 확인됐다고 보기 어렵다.

#### Rolling별 Backtest가 서로 달랐던 이유

Macro-F1-selected의 test 누적수익률은 `-0.77%`, `-8.68%`, `+10.35%`로 크게 달랐다. 특정 시장 구간에서 유효했던 선형관계가 다음 구간에서는 유지되지 않았을 가능성을 보여준다.

또한 Macro F1은 class를 맞혔는지만 평가하고 다음 요소는 고려하지 않는다.

- 정확히 맞힌 거래와 틀린 거래의 수익률 크기
- 큰 손실 한 번의 영향
- 거래비용
- MDD와 손익의 시간 순서

따라서 Macro F1이 항상 HOLD보다 높더라도 투자 수익이 반드시 양수가 되는 것은 아니다.

#### 세 Test 기간을 연결한 실제 결과

Rolling 1~3의 test 기간은 2025년 7월부터 2026년 3월까지 시간순으로 이어진다. 세 구간의 Macro-F1-selected prediction을 하나의 out-of-sample backtest로 연결한 결과는 다음과 같다.

| 지표 | 연결 결과 |
|---|---:|
| 누적수익률 | -0.005% |
| Sharpe-like | 0.125 |
| MDD | -14.42% |
| 거래 수 | 86 |
| 승률 | 50.0% |
| 평균 거래수익률 | +0.026% |

약 9개월 동안 최종 수익률은 사실상 0%였지만 중간에는 최고점 대비 약 14.42%의 손실을 경험했다. 따라서 감수한 위험에 비해 얻은 수익이 부족했다.

Rolling별 누적수익률의 단순 평균 `+0.30%`는 실제 투자수익률로 해석하면 안 된다. 시간순으로 이어진 기간에서는 각 구간의 수익을 복리로 연결해야 하며, 해당 결과는 약 `-0.005%`다.

#### 이 결과가 의미하는 것

현재 결과로 말할 수 있는 결론은 다음과 같다.

> 과거 72시간의 12개 Chart feature를 모두 제공하더라도, 이를 flatten하여 Ridge의 고정된 선형계수로 결합하는 방법에서는 미래 24시간 수익률과 SHORT/HOLD/LONG을 안정적으로 예측하지 못했다.

이 결과가 12개 Chart feature 전체가 쓸모없다는 의미는 아니다. 다음 가능성이 남아 있다.

- Feature와 미래 수익률의 관계가 비선형일 수 있다.
- 시점 간 순차적 관계나 반복 주기가 중요할 수 있다.
- 시장 regime에 따라 feature의 의미와 계수가 달라질 수 있다.
- Chart 정보만으로 부족하고 News 정보가 추가로 필요할 수 있다.
- 24시간 target 자체의 noise가 클 수 있다.

따라서 아직 Cryptova가 우수하다고 결론 내릴 수 없다. 같은 rolling split과 evaluator로 LSTM, TimesNet, foundation model 및 Cryptova를 평가한 뒤 비교해야 한다.

#### 현재 단계의 최종 평가

```text
수익률 회귀모델로서의 성능  → 약함
SHORT/HOLD/LONG 분류 성능   → 항상 HOLD보다 소폭 개선
거래모델로서의 안정성       → 확인되지 않음
시장 구간별 안정성          → 낮음
Benchmark baseline 역할     → 정상적으로 수행
```

Ridge-Flat은 성공적인 거래모델은 아니지만, 단순한 선형관계만으로 얻을 수 있는 성능의 기준점을 제공했다. 이후 LSTM과 TimesNet이 이 결과를 넘어서는지 확인함으로써 순차적·비선형 구조 및 주기 모델링의 추가 가치를 평가할 수 있다.

## 7. 보고서용 핵심 문장

> 본 연구에서 Linear Regression은 복잡한 딥러닝 및 foundation model이 단순한 선형 예측 관계를 넘어서는 성능을 제공하는지 검증하기 위한 기준선으로 사용한다.

> 원래 Chart 입력은 과거 72시간과 12개 feature로 구성된 3차원 tensor이므로 일반적인 선형회귀에 직접 입력할 수 없다. 마지막 시점만 선택하면 시간 정보가 손실되고, 통계적 요약을 적용하면 입력 경로가 축약된다. 이에 본 연구는 72×12 window를 864차원으로 flatten하여 전체 정보 범위를 유지한다.

> Flatten된 입력에는 동일 feature의 시차 변수 및 중첩된 rolling feature로 인해 강한 공선성이 발생할 수 있다. 이를 완화하기 위해 L2 regularization을 적용한 Ridge Regression을 사용하고, 규제 강도는 각 rolling의 validation 구간에서만 선택한다.

> Ridge-Flat은 수익률 예측과 분류 비교라는 서로 다른 목적을 구분하기 위해 validation RMSE로 선택한 forecasting variant와 validation Macro F1으로 선택한 classification variant를 함께 보고한다. Cryptova와의 주 분류 비교에는 Macro-F1-selected variant를 사용하며 test 데이터는 모델 선택에 사용하지 않는다.

> Ridge-Flat은 시간적 인접성, 반복 주기 및 비선형 상호작용을 구조적으로 학습하는 모델이 아니다. 과거 72시간 전체를 제공받는 규제 선형 baseline으로서 LSTM, TimesNet 및 Cryptova의 추가적인 모델링 가치를 측정하는 역할을 한다.

> 원래 Chart 입력은 (N, 72, 12) 형태의 3차원 Tensor이므로 일반 Linear Regression에 입력하기 위해 (N, 864)로 flatten하였다. 이 과정에서 동일 feature의 인접 lag와 중첩된 rolling feature 사이에 강한 공선성이 발생할 수 있으므로, 회귀계수에 L2 penalty를 부여하는 Ridge Regression을 사용해 계수 불안정성과 과적합을 완화하였다.
