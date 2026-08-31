"""Many-to-one LSTM classifier for SHORT/HOLD/LONG prediction."""

from __future__ import annotations

import torch
from torch import nn

from lstm_model import LSTMConfig


class ManyToOneLSTMClassifier(nn.Module):
    """Map a `(batch, 72, 12)` chart sequence to three class logits."""

    num_classes: int = 3

    def __init__(self, config: LSTMConfig = LSTMConfig()) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.output_dropout = nn.Dropout(config.output_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)

    def forward(self, chart: torch.Tensor) -> torch.Tensor:
        if chart.ndim != 3:
            raise ValueError(
                f"chart must have shape (batch, sequence, feature), got {tuple(chart.shape)}"
            )
        if chart.shape[-1] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} chart features, got {chart.shape[-1]}"
            )

        _, (hidden, _) = self.lstm(chart)
        last_hidden = hidden[-1]
        regularized = self.output_dropout(last_hidden)
        return self.classifier(regularized)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
