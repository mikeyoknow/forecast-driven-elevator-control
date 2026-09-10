from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.evaluation.experiments import _select_robust_controller


class RobustControllerSelectionTests(unittest.TestCase):
    @staticmethod
    def _grid() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "distance_penalty": 0.0,
                    "minimum_score_advantage": 0.0,
                    "positioning_interval_minutes": 1,
                    "mean_wait_saved_s": 1.0,
                    "p95_wait_change_s": 1.0,
                    "movement_change_pct": 4.0,
                },
                {
                    "distance_penalty": 0.35,
                    "minimum_score_advantage": 0.5,
                    "positioning_interval_minutes": 2,
                    "mean_wait_saved_s": 0.8,
                    "p95_wait_change_s": -0.2,
                    "movement_change_pct": 3.0,
                },
                {
                    "distance_penalty": 0.6,
                    "minimum_score_advantage": 1.0,
                    "positioning_interval_minutes": 5,
                    "mean_wait_saved_s": 1.2,
                    "p95_wait_change_s": -0.1,
                    "movement_change_pct": 7.0,
                },
            ]
        )

    def test_strict_rule_enforces_movement_and_p95_constraints(self) -> None:
        selected, rule = _select_robust_controller(self._grid(), 5.0)

        self.assertEqual(float(selected["distance_penalty"]), 0.35)
        self.assertTrue(rule.startswith("strict:"))

    def test_relaxed_rule_remains_inside_movement_budget(self) -> None:
        grid = self._grid()
        grid["p95_wait_change_s"] = [1.0, 0.2, -0.1]
        selected, rule = _select_robust_controller(grid, 5.0)

        self.assertEqual(float(selected["movement_change_pct"]), 3.0)
        self.assertTrue(rule.startswith("relaxed:"))


if __name__ == "__main__":
    unittest.main()
