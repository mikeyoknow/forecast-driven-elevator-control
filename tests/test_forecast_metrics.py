from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.data.features import build_forecasting_dataset
from elevator_ml.data.generator import create_days
from elevator_ml.evaluation.forecast_metrics import (
    floor_metric_rows,
    forecast_metric_row,
    scenario_metric_rows,
    top_floor_overlap,
)


class ForecastMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        days = create_days("validation", ("morning", "lunch"), 1, 20, 10, 5)
        cls.dataset = build_forecasting_dataset(
            days, (1, 3), 2, ("morning", "lunch")
        )

    def test_perfect_predictions_have_zero_error(self) -> None:
        row = forecast_metric_row(
            self.dataset,
            self.dataset.targets.copy(),
            model_name="perfect",
            evaluation="validation",
        )

        self.assertEqual(row["floor_mae"], 0.0)
        self.assertEqual(row["floor_rmse"], 0.0)
        self.assertEqual(row["total_demand_mae"], 0.0)
        self.assertEqual(row["top3_floor_overlap"], 1.0)

    def test_scenario_and_floor_row_counts(self) -> None:
        predictions = np.zeros_like(self.dataset.targets)

        scenarios = scenario_metric_rows(
            self.dataset,
            predictions,
            model_name="zero",
            evaluation="validation",
        )
        floors = floor_metric_rows(
            self.dataset,
            predictions,
            model_name="zero",
            evaluation="validation",
        )

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(len(floors), self.dataset.n_targets)

    def test_top_floor_overlap_rejects_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            top_floor_overlap(
                self.dataset.targets,
                self.dataset.targets,
                k=self.dataset.n_targets + 1,
            )


if __name__ == "__main__":
    unittest.main()
