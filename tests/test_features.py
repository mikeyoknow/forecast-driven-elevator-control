from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.data.features import (
    build_inference_dataset,
    build_forecasting_dataset,
    feature_names,
    make_feature_vector,
)
from elevator_ml.data.generator import aggregate_minute_counts, generate_day
from elevator_ml.simulation.entities import DayData, Passenger


SCENARIOS = ("morning", "lunch", "evening", "mixed")


def manual_day(
    day_id: str,
    passengers: tuple[Passenger, ...],
    *,
    scenario: str = "mixed",
    duration_minutes: int = 8,
) -> DayData:
    origin, destination = aggregate_minute_counts(
        passengers, duration_minutes=duration_minutes, n_floors=4
    )
    return DayData(
        day_id=day_id,
        split="train",
        scenario=scenario,
        seed=1,
        duration_minutes=duration_minutes,
        passengers=passengers,
        origin_counts=origin,
        destination_counts=destination,
    )


class FeatureTests(unittest.TestCase):
    def test_future_changes_do_not_change_current_features(self) -> None:
        shared_past = Passenger(0, arrival_time=60, origin=0, destination=1)
        first_day = manual_day(
            "train-mixed-a",
            (
                shared_past,
                Passenger(1, arrival_time=300, origin=1, destination=2),
            ),
        )
        second_day = manual_day(
            "train-mixed-b",
            (
                Passenger(0, arrival_time=60, origin=0, destination=1),
                Passenger(1, arrival_time=300, origin=3, destination=2),
            ),
        )

        first = make_feature_vector(first_day, 4, (1, 3), SCENARIOS)
        second = make_feature_vector(second_day, 4, (1, 3), SCENARIOS)

        np.testing.assert_array_equal(first, second)

    def test_dataset_windows_never_overlap_targets(self) -> None:
        day = generate_day("train", "morning", 0, 42, 12, 5)
        dataset = build_forecasting_dataset(
            (day,), (1, 3, 5), 2, SCENARIOS
        )

        self.assertTrue(
            (
                dataset.metadata["input_end_minute_exclusive"]
                <= dataset.metadata["target_start_minute"]
            ).all()
        )
        self.assertTrue(
            (
                dataset.metadata["target_end_minute_exclusive"]
                <= day.duration_minutes
            ).all()
        )

    def test_expected_sample_and_matrix_shapes(self) -> None:
        days = (
            generate_day("train", "morning", 0, 10, 12, 5),
            generate_day("train", "lunch", 0, 11, 12, 5),
        )
        windows = (1, 3, 5)
        dataset = build_forecasting_dataset(days, windows, 2, SCENARIOS)
        expected_per_day = 12 - max(windows) - 2 + 1

        self.assertEqual(dataset.n_samples, 2 * expected_per_day)
        self.assertEqual(dataset.n_features, len(dataset.feature_names))
        self.assertEqual(dataset.n_targets, 5)
        self.assertEqual(dataset.metadata["day_id"].nunique(), 2)

    def test_target_is_exact_future_origin_count(self) -> None:
        passengers = (
            Passenger(0, arrival_time=180, origin=0, destination=1),
            Passenger(1, arrival_time=240, origin=2, destination=0),
        )
        day = manual_day("train-mixed-target", passengers)
        dataset = build_forecasting_dataset((day,), (1, 3), 2, SCENARIOS)
        row = dataset.metadata.index[
            dataset.metadata["target_start_minute"] == 3
        ][0]

        np.testing.assert_array_equal(
            dataset.targets[row], np.array([1.0, 0.0, 1.0, 0.0])
        )

    def test_unseen_scenario_has_zero_training_scenario_indicators(self) -> None:
        day = generate_day("stress", "surge", 0, 55, 8, 4)
        names = feature_names(4, (1, 3), SCENARIOS)
        vector = make_feature_vector(day, 3, (1, 3), SCENARIOS)
        scenario_indices = [
            index for index, name in enumerate(names) if name.startswith("scenario_")
        ]

        np.testing.assert_array_equal(vector[scenario_indices], np.zeros(4))

    def test_samples_do_not_cross_day_boundaries(self) -> None:
        days = (
            generate_day("train", "morning", 0, 100, 8, 4),
            generate_day("train", "morning", 1, 101, 8, 4),
        )
        dataset = build_forecasting_dataset(days, (1, 3), 2, SCENARIOS)

        for _, group in dataset.metadata.groupby("day_id"):
            self.assertEqual(group["input_start_minute"].min(), 0)
            self.assertEqual(group["target_start_minute"].min(), 3)
            self.assertEqual(group["target_end_minute_exclusive"].max(), 8)

    def test_inference_dataset_covers_every_day_minute(self) -> None:
        day = generate_day("validation", "lunch", 0, 202, 8, 4)

        dataset = build_inference_dataset(day, (1, 3), 2, SCENARIOS)

        self.assertEqual(dataset.n_samples, day.duration_minutes)
        self.assertEqual(dataset.metadata["target_start_minute"].tolist(), list(range(8)))
        self.assertTrue(
            (
                dataset.metadata["input_end_minute_exclusive"]
                <= dataset.metadata["target_start_minute"]
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
