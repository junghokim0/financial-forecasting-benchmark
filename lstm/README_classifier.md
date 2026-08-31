# Many-to-One LSTM Classifier

과거 72시간의 Chart Feature 12개를 순서대로 처리하고 미래 24시간의 `SHORT/HOLD/LONG`을 직접 예측하는 Classification baseline이다.

```text
(B, 72, 12)
  -> LSTM(input=12, hidden=32, layers=1)
  -> final hidden state (B, 32)
  -> Dropout(0.30)
  -> Linear(32, 3)
  -> logits (B, 3)
  -> softmax probabilities
  -> SHORT / HOLD / LONG
```

## 학습 설정

- Target: `sample_meta_*.csv`의 `label_id`
- Loss: Cross-Entropy, label smoothing `0.03`
- Class weight: 사용하지 않음
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- Batch size: 64
- Maximum epochs: 50
- Patience: 8
- Gradient clipping: 1.0
- Seed: 42
- Checkpoint: Validation Macro F1 최대, 동점이면 Validation Cross-Entropy 최소
- Confidence/risk filter: 주 비교에서 사용하지 않음

이 모델은 수익률을 예측하거나 threshold를 적용하지 않는다. `label_id`를 직접 학습하며 Cryptova와의 주 Classification 비교에 사용한다.

```powershell
python .\lstm\train_lstm_classifier.py
```

결과는 `outputs/lstm_classifier/rolling_N/`에 저장된다.
