"""Many-to-one LSTM return forecaster used by the common benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LSTMConfig:
    """Architecture settings fixed before benchmark evaluation."""

    input_size: int = 12
    hidden_size: int = 32
    num_layers: int = 1
    output_dropout: float = 0.30
    bidirectional: bool = False

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("input_size, hidden_size, and num_layers must be positive.")
        if not 0.0 <= self.output_dropout < 1.0:
            raise ValueError("output_dropout must be in [0, 1).")
        if self.bidirectional:
            raise ValueError(
                "This benchmark class is the unidirectional LSTM baseline. "
                "Implement BiLSTM as a separately named ablation."
            )

    def to_dict(self) -> dict:
        return asdict(self)


class ManyToOneLSTM(nn.Module):
    """Map a `(batch, 72, 12)` chart sequence to one predicted return.

    PyTorch's internal LSTM dropout only operates between stacked recurrent
    layers. Because the baseline has one recurrent layer, dropout is applied
    explicitly to the final hidden state before the regression head.
    """

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
        self.regression_head = nn.Linear(config.hidden_size, 1)

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
        predicted_return = self.regression_head(regularized)
        return predicted_return.squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
