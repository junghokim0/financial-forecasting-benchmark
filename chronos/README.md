# Chronos-2 LoRA Fine-tuned Benchmark

이 폴더는 Cryptova benchmark의 동일한 Rolling 1~3 OOS 프로토콜에서
`amazon/chronos-2`를 LoRA로 적응시키는 코드다. 모델명은 결과와 보고서에서
반드시 **Chronos-2 LoRA Fine-tuned**로 표기한다. Full fine-tuning이 아니다.

## 현재 실험 범위

- BTC `close`: 미래 경로를 예측하기 위한 target이며 feature 개수에 포함하지 않음
- 기존 Chart feature 12개: Chronos-2의 past covariates
- context: 72시간
- horizon: 24시간
- point forecast: Chronos-2 `q=0.5` 중앙값의 24번째 시점
- classification: predicted return을 공통 `±0.012` threshold로 변환
- checkpoint: 각 Rolling의 Validation quantile loss만 사용해 선택
- test: 선택에 사용하지 않고 구간별 OOS 및 Connected OOS에만 사용

Chart covariate는 기존 Rolling dataset에서 Train에만 fit된 scaler 결과를 사용한다.
Chronos-2의 모델 내부 instance normalization은 공식 preprocessing의 일부이므로
그대로 유지한다. 미래 Chart covariate는 제공하지 않는다. 이번 구현에는 News 9개를
넣지 않으며, Chart+News Track은 후속 실험으로 분리한다.

## LoRA를 선택한 이유

Chronos-2는 120M parameter foundation model이다. 제한된 암호화폐 데이터에서 전체
가중치를 학습하면 GPU 메모리·시간 부담이 크고 과적합 및 사전학습 표현 손실 위험이
있다. 따라서 본체 대부분을 고정하고 공식 기본 LoRA adapter만 학습한다. LoRA는
계산을 완전히 없애는 방식은 아니지만 trainable parameter와 optimizer state를 줄이고
사전학습 표현을 보존하는 데 적합하다.

## 파일

- `chronos2_data.py`: 연속 학습 시계열과 정확한 72시간 OOS window 구성
- `validate_chronos2_data.py`: 모델을 다운로드하지 않는 데이터 검증
- `train_chronos2_lora.py`: Colab GPU 전용 Rolling별 LoRA 학습·추론 코드
- `aggregate_chronos2_results.py`: Rolling 1~3 Test 예측 연결 및 최종 OOS 평가
- `chronos2_colab.ipynb`: Google Drive 기반 Colab 실행 순서
- `requirements-colab.txt`: Colab 의존성

## Colab 실행

Google Drive에 `benchmark` 폴더 전체를 올리고 GPU runtime을 선택한다. 노트북의
`BENCHMARK_ROOT`만 실제 Drive 경로에 맞게 변경한다.

데이터 검증은 모델을 다운로드하거나 학습하지 않는다.

```bash
python chronos/validate_chronos2_data.py \
  --benchmark-root /content/drive/MyDrive/crypto/benchmark \
```

Chart-only Rolling 1~3:

```bash
python chronos/train_chronos2_lora.py \
  --benchmark-root /content/drive/MyDrive/crypto/benchmark \
  --rolling 1 2 3 \
  --execute-training

python chronos/aggregate_chronos2_results.py \
  --benchmark-root /content/drive/MyDrive/crypto/benchmark
```

T4 메모리가 부족하면 `--train-batch-size 16 --prediction-batch-size 32`로 낮춘다. 이 batch size는
Chronos-2에서 variate 수를 의미하므로 LSTM/TimesNet의 sample batch 64와 숫자를
억지로 같게 설정하지 않는다.

## 출력

```text
outputs/chronos2_lora_chart/
├─ rolling_1/
├─ rolling_2/
├─ rolling_3/
└─ connected_oos/
```

각 Rolling에는 validation/test 예측, 공통 regression/classification/backtest 지표,
거래 내역, LoRA checkpoint 및 실행 metadata가 저장된다.

## 재현성 및 해석 주의

- checkpoint: `amazon/chronos-2`
- revision: `95a9710e2596287d08352589f42634fa5abdf0a7`
- seed: 42
- official default LoRA: `r=8`, `alpha=16`, q/k/v/o 및 output head
- Foundation model 사전학습 데이터와 기간은 로컬 Rolling Train만 사용한 모델과
  동일하지 않으므로, 결과는 동일 parameter budget이 아닌 end-to-end 경쟁력 비교다.
- 실행 후 Colab의 실제 GPU, CUDA, PyTorch, Chronos, Transformers, PEFT 버전을
  `run_summary.json`과 함께 보존해야 한다.
