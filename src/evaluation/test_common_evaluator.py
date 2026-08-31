"""Small deterministic tests for the common evaluator."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from backtest import BacktestConfig, non_overlapping_backtest
from classification import classification_metrics
from evaluate_predictions import evaluate_prediction_frame
from regression import regression_metrics


def example_frame(include_predicted_return: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "schema_version": ["1.0"] * 4,
            "model": ["dummy"] * 4,
            "model_version": ["test"] * 4,
            "rolling": ["rolling_1"] * 4,
            "split": ["test"] * 4,
            "seed": [42] * 4,
            "sample_time": pd.to_datetime(
                ["2025-07-01 00:00", "2025-07-01 01:00", "2025-07-02 00:00", "2025-07-03 00:00"],
                utc=True,
            ),
            "target_time": pd.to_datetime(
                ["2025-07-02 00:00", "2025-07-02 01:00", "2025-07-03 00:00", "2025-07-04 00:00"],
                utc=True,
            ),
            "y_true": [2, 0, 0, 1],
            "raw_future_return": [0.02, -0.50, -0.03, 0.01],
            "y_pred": [2, 0, 0, 1],
        }
    )
    if include_predicted_return:
        frame["predicted_return"] = [0.02, -0.50, -0.03, 0.01]
    return frame


class CommonEvaluatorTest(unittest.TestCase):
    def test_cryptova_non_overlap_and_cost(self) -> None:
        metrics, trades = non_overlapping_backtest(
            example_frame(), BacktestConfig(fee=0.001, slippage=0.001)
        )
        self.assertEqual(metrics["trade_count"], 2)
        self.assertEqual(metrics["trade_ratio"], 0.5)
        np.testing.assert_allclose(trades["strategy_return"], [0.018, 0.028])
        self.assertAlmostEqual(metrics["cumulative_return"], 1.018 * 1.028 - 1.0)
        self.assertEqual(metrics["win_rate"], 1.0)
        self.assertAlmostEqual(metrics["avg_trade_return"], 0.023)

    def test_classification_matches_expected(self) -> None:
        metrics = classification_metrics(np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_regression_metrics(self) -> None:
        metrics = regression_metrics(np.array([1.0, -2.0]), np.array([1.0, -2.0]))
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["directional_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["pearson_correlation"], 1.0)

    def test_end_to_end_without_regression_output(self) -> None:
        result, trades = evaluate_prediction_frame(example_frame())
        self.assertNotIn("regression", result)
        self.assertEqual(result["backtest"]["trade_count"], 2)
        self.assertTrue(math.isfinite(result["backtest"]["sharpe_like"]))
        self.assertEqual(len(trades), 2)

    def test_end_to_end_with_regression_output(self) -> None:
        result, _ = evaluate_prediction_frame(example_frame(include_predicted_return=True))
        self.assertIn("regression", result)
        self.assertEqual(result["regression"]["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
