"""Unit tests for the fixed market-regime labeling rules."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyze_market_regimes import (
    SIGNAL_LONG_THRESHOLD,
    SIGNAL_SHORT_THRESHOLD,
    attach_regimes,
    returns_to_classes,
    trend_thresholds,
)


class MarketRegimeTests(unittest.TestCase):
    def test_return_class_boundaries_are_inclusive(self) -> None:
        values = pd.Series(
            [SIGNAL_SHORT_THRESHOLD, -0.0119, 0.0, 0.0119, SIGNAL_LONG_THRESHOLD]
        )
        np.testing.assert_array_equal(returns_to_classes(values), [0, 1, 1, 1, 2])

    def test_trend_thresholds_use_train_quantiles(self) -> None:
        times = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
        master = pd.DataFrame(
            {
                "sample_time": times,
                "return_72h": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            }
        )
        from analyze_market_regimes import ROLLING_BOUNDS

        original = ROLLING_BOUNDS.copy()
        try:
            ROLLING_BOUNDS.clear()
            ROLLING_BOUNDS["rolling_test"] = {
                "train": ("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"),
                "test": ("2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"),
            }
            limits = trend_thresholds(master)["rolling_test"]
        finally:
            ROLLING_BOUNDS.clear()
            ROLLING_BOUNDS.update(original)
        self.assertGreater(limits["q33"], 0.0)
        self.assertGreater(limits["q67"], limits["q33"])

    def test_regime_labels_use_trailing_columns(self) -> None:
        times = pd.to_datetime(
            ["2025-07-04T00:00:00Z", "2025-07-05T00:00:00Z", "2025-07-06T00:00:00Z"],
            utc=True,
        )
        predictions = pd.DataFrame(
            {
                "sample_time": times,
                "source_rolling": ["rolling_1"] * 3,
            }
        )
        master = pd.DataFrame(
            {
                "sample_time": times,
                "return_72h": [-0.02, 0.0, 0.03],
                "std_24h": [0.004, 0.005, 0.006],
            }
        )
        trend_limits = {
            "rolling_1": {"q33": -0.01, "q67": 0.01}
        }
        labeled = attach_regimes(
            predictions, master, trend_limits, {"rolling_1": 0.005}
        )
        self.assertEqual(labeled["trend_regime"].tolist(), ["DOWN", "SIDEWAYS", "UP"])
        self.assertEqual(labeled["volatility_regime"].tolist(), ["LOW", "LOW", "HIGH"])

    def test_zero_return_is_neutral_even_when_train_is_one_sided(self) -> None:
        times = pd.to_datetime(["2025-07-04T00:00:00Z"], utc=True)
        predictions = pd.DataFrame(
            {"sample_time": times, "source_rolling": ["rolling_1"]}
        )
        master = pd.DataFrame(
            {"sample_time": times, "return_72h": [0.0], "std_24h": [0.005]}
        )
        for q33, q67 in ((0.01, 0.03), (-0.03, -0.01)):
            labeled = attach_regimes(
                predictions,
                master,
                {"rolling_1": {"q33": q33, "q67": q67}},
                {"rolling_1": 0.005},
            )
            self.assertEqual(labeled.loc[0, "trend_regime"], "SIDEWAYS")


if __name__ == "__main__":
    unittest.main()
