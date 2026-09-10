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
from elevator_ml.forecasting.baselines import (
    HistoricalMeanForecaster,
    PerFloorMeanForecaster,
    PersistenceForecaster,
)
from elevator_ml.forecasting.models import RandomForestForecaster, RidgeForecaster


class ForecasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        days = create_days(
            "train", ("morning", "lunch"), 2, 100, 12, 5
        )
        cls.dataset = build_forecasting_dataset(
            days, (1, 3, 5), 2, ("morning", "lunch")
        )

    def assert_valid_predictions(self, predictions: np.ndarray) -> None:
        self.assertEqual(predictions.shape, self.dataset.targets.shape)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions >= 0).all())

    def test_per_floor_mean_matches_training_target_mean(self) -> None:
        model = PerFloorMeanForecaster().fit(self.dataset)
        predictions = model.predict(self.dataset)

        np.testing.assert_allclose(
            predictions[0], self.dataset.targets.mean(axis=0)
        )
        self.assert_valid_predictions(predictions)

    def test_persistence_uses_last_minute_times_horizon(self) -> None:
        model = PersistenceForecaster().fit(self.dataset)
        predictions = model.predict(self.dataset)
        indices = [
            self.dataset.feature_names.index(f"origin_floor_{floor}_last_1m")
            for floor in range(self.dataset.n_targets)
        ]

        np.testing.assert_array_equal(
            predictions,
            self.dataset.features[:, indices] * self.dataset.horizon_minutes,
        )

    def test_historical_forecaster_is_finite_and_nonnegative(self) -> None:
        model = HistoricalMeanForecaster(bucket_minutes=5).fit(self.dataset)

        self.assert_valid_predictions(model.predict(self.dataset))

    def test_ridge_predictions_are_valid(self) -> None:
        model = RidgeForecaster(alpha=10.0).fit(self.dataset)

        self.assert_valid_predictions(model.predict(self.dataset))

    def test_random_forest_predictions_and_importance_are_valid(self) -> None:
        model = RandomForestForecaster(
            n_estimators=10,
            max_depth=4,
            min_samples_leaf=2,
            random_state=1,
            n_jobs=1,
        ).fit(self.dataset)

        self.assert_valid_predictions(model.predict(self.dataset))
        self.assertEqual(len(model.feature_importances_), self.dataset.n_features)
        self.assertAlmostEqual(float(model.feature_importances_.sum()), 1.0)

        mean, uncertainty = model.predict_with_uncertainty(self.dataset)
        self.assert_valid_predictions(mean)
        self.assertEqual(uncertainty.shape, self.dataset.targets.shape)
        self.assertTrue((uncertainty >= 0).all())


if __name__ == "__main__":
    unittest.main()
