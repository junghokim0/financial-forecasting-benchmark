"""Unit tests for official-core TimesNet regression and classification models."""

import unittest

import torch

from timesnet_model import (
    TimesNetClassifier,
    TimesNetConfig,
    TimesNetEncoder,
    TimesNetRegressor,
    count_trainable_parameters,
    fft_for_period,
)


class TimesNetModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.chart = torch.randn(4, 72, 12)

    def test_encoder_shape(self) -> None:
        encoded = TimesNetEncoder()(self.chart)
        self.assertEqual(tuple(encoded.shape), (4, 72, 32))

    def test_regression_shape(self) -> None:
        prediction = TimesNetRegressor()(self.chart)
        self.assertEqual(tuple(prediction.shape), (4,))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_classifier_shape(self) -> None:
        logits = TimesNetClassifier()(self.chart)
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_fft_period_outputs(self) -> None:
        hidden = torch.randn(4, 72, 32)
        periods, weights = fft_for_period(hidden, top_k=2)
        self.assertEqual(tuple(periods.shape), (2,))
        self.assertEqual(tuple(weights.shape), (4, 2))
        self.assertTrue((periods >= 1).all())

    def test_invalid_input_shape(self) -> None:
        with self.assertRaises(ValueError):
            TimesNetRegressor()(torch.randn(4, 12))
        with self.assertRaises(ValueError):
            TimesNetClassifier()(torch.randn(4, 71, 12))

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            TimesNetConfig(top_k=100)

    def test_parameter_count(self) -> None:
        self.assertGreater(count_trainable_parameters(TimesNetRegressor()), 0)
        self.assertGreater(count_trainable_parameters(TimesNetClassifier()), 0)


if __name__ == "__main__":
    unittest.main()
