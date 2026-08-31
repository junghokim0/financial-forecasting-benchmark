"""Unit tests for the many-to-one LSTM baseline."""

from __future__ import annotations

import unittest

import torch

from lstm_model import LSTMConfig, ManyToOneLSTM, count_trainable_parameters


class ManyToOneLSTMTest(unittest.TestCase):
    def test_forward_returns_one_value_per_sample(self) -> None:
        model = ManyToOneLSTM()
        chart = torch.randn(7, 72, 12)
        prediction = model(chart)
        self.assertEqual(tuple(prediction.shape), (7,))

    def test_parameter_count_is_fixed(self) -> None:
        model = ManyToOneLSTM()
        self.assertEqual(count_trainable_parameters(model), 5921)

    def test_wrong_feature_count_is_rejected(self) -> None:
        model = ManyToOneLSTM()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 72, 13))

    def test_bidirectional_setting_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LSTMConfig(bidirectional=True)


if __name__ == "__main__":
    unittest.main()
