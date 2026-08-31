"""Unit tests for the direct many-to-one LSTM classifier."""

from __future__ import annotations

import unittest

import torch

from lstm_classifier import ManyToOneLSTMClassifier, count_trainable_parameters


class ManyToOneLSTMClassifierTest(unittest.TestCase):
    def test_forward_returns_three_logits_per_sample(self) -> None:
        model = ManyToOneLSTMClassifier()
        logits = model(torch.randn(7, 72, 12))
        self.assertEqual(tuple(logits.shape), (7, 3))

    def test_probabilities_sum_to_one(self) -> None:
        model = ManyToOneLSTMClassifier().eval()
        with torch.inference_mode():
            probabilities = torch.softmax(model(torch.randn(4, 72, 12)), dim=-1)
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4))

    def test_parameter_count_is_fixed(self) -> None:
        self.assertEqual(count_trainable_parameters(ManyToOneLSTMClassifier()), 5987)

    def test_wrong_feature_count_is_rejected(self) -> None:
        model = ManyToOneLSTMClassifier()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 72, 13))


if __name__ == "__main__":
    unittest.main()
