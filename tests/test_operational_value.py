from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.evaluation.operational_value import correlation_interval


class OperationalValueTests(unittest.TestCase):
    def test_perfect_monotonic_association(self) -> None:
        x = np.arange(1.0, 9.0)
        result = correlation_interval(
            x,
            3.0 * x + 2.0,
            method="spearman",
            resamples=200,
            seed=4,
        )

        self.assertAlmostEqual(result.estimate, 1.0)
        self.assertAlmostEqual(result.lower, 1.0)
        self.assertAlmostEqual(result.upper, 1.0)

    def test_fixed_seed_is_deterministic(self) -> None:
        x = np.arange(10.0)
        y = np.asarray([2, 0, 1, 3, 5, 4, 8, 7, 6, 9], dtype=float)
        first = correlation_interval(
            x, y, method="pearson", resamples=100, seed=19
        )
        second = correlation_interval(
            x, y, method="pearson", resamples=100, seed=19
        )

        self.assertEqual(first, second)

    def test_constant_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            correlation_interval(
                np.ones(5),
                np.arange(5.0),
                method="pearson",
                resamples=10,
                seed=2,
            )


if __name__ == "__main__":
    unittest.main()
