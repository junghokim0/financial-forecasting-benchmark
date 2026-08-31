# Prediction schema v1.0

모든 모델은 모델별 평가 코드를 실행하지 않고, 먼저 이 형식의 validation/test 예측 파일을 생성한다. 공통 evaluator는 이 파일만 읽는다.

## 필수 컬럼

| 컬럼 | 의미 |
|---|---|
| `schema_version` | 현재 `1.0` |
| `model` | 모델 식별자 |
| `model_version` | checkpoint 또는 구현 버전 |
| `rolling` | `rolling_1`~`rolling_3` |
| `split` | `validation` 또는 `test` |
| `seed` | 학습 seed; 결정론적 zero-shot도 명시 |
| `sample_time` | 신호를 생성한 UTC 시점 |
| `target_time` | 실제 수익률 평가 시점 |
| `y_true` | 실제 클래스: SHORT=0, HOLD=1, LONG=2 |
| `raw_future_return` | 실제 24시간 수익률 |
| `y_pred` | 모델의 기본 예측 클래스 |

## 선택 컬럼

| 컬럼 | 적용 모델 | 의미 |
|---|---|---|
| `predicted_return` | 회귀·forecasting | 예측 24시간 수익률 |
| `prob_short` | 확률 분류 | SHORT 확률 |
| `prob_hold` | 확률 분류 | HOLD 확률 |
| `prob_long` | 확률 분류 | LONG 확률 |
| `confidence` | 확률 분류 | 세 확률의 최댓값 |

확률 컬럼은 세 개를 모두 제공하거나 모두 생략한다. 제공할 경우 각 행의 합은 1이어야 하고 `y_pred`는 확률 argmax와 일치해야 한다.

## 회귀 출력의 클래스 변환

```text
predicted_return <= -0.012  -> SHORT (0)
-0.012 < predicted_return < 0.012 -> HOLD (1)
predicted_return >= 0.012   -> LONG (2)
```

경계 포함 규칙은 Cryptova label 정의와 동일하다.

## 기본 평가와 후처리 평가

- Primary benchmark는 `y_pred` 그대로 평가한다.
- Confidence filter는 validation에서 threshold를 선택하는 secondary 결과다.
- Funding/volatility risk filter는 모델 자체 비교와 분리된 strategy 결과다.
- Test 데이터로 threshold나 모델을 선택하면 안 된다.

## 파일 위치 규칙

```text
outputs/predictions/{model}/{rolling}_{split}_seed_{seed}.csv
```

예:

```text
outputs/predictions/timesnet/rolling_1_test_seed_42.csv
```
