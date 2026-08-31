"""Unit tests for the Ridge-Flat implementation."""

from __future__ import annotations

import unittest

import numpy as np

from ridge_flat import RidgeSVDPath, flatten_chart_windows


class RidgeFlatTest(unittest.TestCase):
    def test_flatten_preserves_row_major_time_feature_order(self) -> None:
        tensor = np.arange(2 * 3 * 2).reshape(2, 3, 2)
        flattened = flatten_chart_windows(tensor)
        np.testing.assert_array_equal(flattened[0], [0, 1, 2, 3, 4, 5])
        self.assertEqual(flattened.shape, (2, 6))

    def test_ridge_recovers_simple_linear_signal(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(300, 4))
        target = 1.5 + features @ np.array([0.8, -0.4, 0.2, 0.0])
        path = RidgeSVDPath(features, target, input_shape=(2, 2))
        model = path.model(1e-6)
        prediction = model.predict(features)
        self.assertLess(float(np.sqrt(np.mean((target - prediction) ** 2))), 1e-6)

    def test_positive_alpha_required(self) -> None:
        path = RidgeSVDPath(np.eye(3), np.arange(3.0), input_shape=(1, 3))
        with self.assertRaises(ValueError):
            path.model(0.0)


if __name__ == "__main__":
    unittest.main()
