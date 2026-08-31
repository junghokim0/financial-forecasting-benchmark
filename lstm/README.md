# Many-to-One LSTM baseline

과거 72시간의 Chart Feature 12개를 순서대로 읽고 미래 24시간 수익률 하나를 예측하는 단방향 LSTM baseline이다.

```text
(B, 72, 12)
  -> LSTM(input=12, hidden=32, layers=1)
  -> final hidden state (B, 32)
  -> Dropout(0.30)
  -> Linear(32, 1)
  -> predicted_return (B,)
```

## 고정 설정

- Target: `sample_meta_*.csv`의 `raw_future_return`
- Loss: MSE
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- Batch size: 64
- Maximum epochs: 50
- Patience: 8
- Gradient clipping: 1.0
- Seed: 42
- 추가 scaling 없음: rolling dataset에 이미 적용된 train-fitted Chart scaler를 그대로 사용

Validation RMSE 기준으로 checkpoint를 선택하고 `patience=8`을 적용한다. Test는 checkpoint 선택에 사용하지 않는다. 이 모델은 Regression Track 전용이며 RMSE, MAE, correlation 및 directional accuracy만 공식 결과로 사용한다. SHORT/HOLD/LONG 비교에는 별도의 `LSTM Classifier`를 사용한다.

## 실행

PyTorch가 설치된 benchmark 실행 환경에서 다음 명령을 사용한다.

```powershell
python .\lstm\train_lstm.py
```

rolling_1만 먼저 검증하려면 다음과 같이 실행한다.

```powershell
python .\lstm\train_lstm.py --rollings rolling_1
```

공식 Regression 결과는 기본적으로 `outputs/lstm_regression/rolling_N/`에 저장된다. 이전 `outputs/lstm/`은 threshold Classification을 포함했던 탐색 실행 기록이며 주 결과에 사용하지 않는다.

## 파일

- `lstm_model.py`: 모델과 architecture config
- `train_lstm.py`: 데이터 로딩, 학습, checkpoint 선택, 추론 및 공통 평가
- `test_lstm_model.py`: 출력 shape, parameter 수 및 입력 검증 테스트
