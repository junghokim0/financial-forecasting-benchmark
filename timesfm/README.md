# TimesFM 2.5 LoRA Fine-tuned Benchmark

이 폴더는 Cryptova benchmark의 동일한 Rolling 1~3 OOS 구간에서 Google의
`TimesFM 2.5`를 LoRA로 적응시키는 코드다. 결과와 보고서의 모델명은 반드시
**TimesFM 2.5 LoRA Fine-tuned**로 표기한다. Full fine-tuning이 아니다.

## 실험 범위

- 입력: raw BTC `close`만 사용하는 univariate forecast
- 공통 가용 window: 과거 72시간
- 실제 TimesFM 입력: 가장 최근 64시간
- horizon: 미래 close 24개
- point forecast: `mean_predictions`의 24번째 close
- regression: `predicted_close(t+24) / close(t) - 1`
- classification: 예측 수익률을 공통 `±0.012` threshold로 변환
- checkpoint: 각 Rolling의 Validation loss만 사용해 선택
- 평가: 공통 evaluator 및 24시간 non-overlap backtest

## 왜 72시간이 아니라 64시간인가

TimesFM 2.5의 patch length는 32라서 context가 32의 배수여야 한다. 72를 96으로
늘리면 다른 모델보다 과거 24시간을 더 보게 되고, 0-padding은 모델 내부 RevIN
통계를 바꾼다. 따라서 공통 72시간 범위 안에서 사용할 수 있는 가장 긴 model-native
context인 최근 64시간을 사용한다. 이는 숨기지 않고 결과표의 입력 차이로 기록한다.

## LoRA를 선택한 이유

TimesFM 2.5는 약 200M parameter foundation model이다. 겹침이 큰 제한된 BTC window로
전체 parameter를 업데이트하면 계산·optimizer memory가 커지고 과적합 및 사전학습
표현 손실 위험이 커진다. 공식 예제와 같은 `r=4`, `alpha=8`, dropout `0.05`,
`all-linear` LoRA를 사용해 본체 대부분을 고정한다.

## 파일

- `timesfm_data.py`: 동일 timestamp에서 close 64시간과 미래 close 24시간 구성
- `validate_timesfm_data.py`: 모델 다운로드 없이 timestamp/target/window 검증
- `train_timesfm2_5_lora.py`: Colab GPU 전용 Rolling별 LoRA 학습·추론·평가
- `aggregate_timesfm_results.py`: Rolling 1~3 Test를 연결한 OOS 평가
- `timesfm_colab.ipynb`: Google Drive 기반 Colab 실행 순서
- `requirements-colab.txt`: Colab 의존성

## Colab 실행

Google Drive에 `benchmark` 폴더 전체를 올린 뒤 GPU runtime을 선택한다.

```bash
python timesfm/validate_timesfm_data.py \
  --benchmark-root /content/drive/MyDrive/crypto/benchmark

python timesfm/train_timesfm2_5_lora.py \
  --benchmark-root /content/drive/MyDrive/crypto/benchmark \
  --rolling 1 2 3 \
  --execute-training

python timesfm/aggregate_timesfm_results.py \
  --benchmark-root /content/drive/MyDrive/crypto/benchmark
```

기본 micro-batch는 16, gradient accumulation은 2라서 effective batch는 32다.
T4에서 메모리가 부족하면 `--batch-size 8 --gradient-accumulation 4`로 바꾼다.

## 재현성

- checkpoint: `google/timesfm-2.5-200m-transformers`
- 실행 시 Hugging Face의 실제 commit SHA를 먼저 해석하고 모든 rolling에 고정
- epoch/patience: `10/3`
- optimizer: AdamW, learning rate `1e-4`, weight decay `0.01`
- LoRA: `r=4`, `alpha=8`, dropout `0.05`, `all-linear`
- seed: 42
- loss: TimesFM 공식 normalized MSE + quantile loss

Foundation model은 외부 대규모 데이터로 사전학습됐으므로 결과는 동일 parameter
budget 비교가 아니라 동일 시장 구간과 evaluator에서의 end-to-end 경쟁력 비교다.
