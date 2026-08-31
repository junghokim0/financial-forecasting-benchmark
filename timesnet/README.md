# TimesNet Benchmark

공식 THUML TimesNet의 핵심 구조를 공통 Cryptova benchmark pipeline에 연결한 독립 Chart baseline이다.

## 구조

```text
Chart (B,72,12)
  → circular Conv1d value embedding + sinusoidal position embedding
  → TimesBlock
      FFT top-k=2
      → period별 2D reshape
      → Inception 2D CNN (32→64→32, kernels=4)
      → amplitude softmax weighted sum
      → residual
  → post LayerNorm
  → GELU → Dropout(0.30) → Flatten(B,2304)
      ├─ Linear(2304,1): Regression
      └─ Linear(2304,3): Classification
```

Regression과 Classification은 encoder 가중치를 공유하지 않고 seed 42에서 독립적으로 학습한다.

## 파일

- `timesnet_model.py`: 공통 TimesNet encoder와 두 task head
- `train_timesnet.py`: MSE regression, Validation RMSE checkpoint
- `train_timesnet_classifier.py`: direct 3-class classification, Validation Macro F1 checkpoint
- `aggregate_timesnet_results.py`: rolling 1~3 test prediction 연결 평가
- `test_timesnet_model.py`: shape, FFT, validation unit tests

## 공통 학습 조건

- 입력 `(72,12)`, train-only scaler가 적용된 기존 rolling tensor
- batch 64, maximum epoch 50, patience 8
- AdamW, learning rate `1e-4`, weight decay `1e-4`
- gradient clipping 1.0, seed 42
- Classification label smoothing 0.03, class weight 없음
- Test는 checkpoint 선택에 사용하지 않음

## 실행

```powershell
python train_timesnet.py --device cpu
python train_timesnet_classifier.py --device cpu
python aggregate_timesnet_results.py
python -m unittest -v test_timesnet_model.py
```

결과는 `outputs/timesnet_regression`과 `outputs/timesnet_classifier`에 저장된다.

## 구현 해석

이 모델은 공식 TimesNet core를 사용하지만 원 논문의 24-step sequence forecasting을 그대로 재현하는 실험은 아니다. 공통 benchmark target인 미래 24시간 수익률 1개와 SHORT/HOLD/LONG 3개를 직접 출력하도록 마지막 head만 task에 맞게 변경했다. Cryptova 내부 TimesNet의 pre-norm·추가 block dropout·multimodal fusion은 사용하지 않는다.
