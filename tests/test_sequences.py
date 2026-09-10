from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.data.generator import generate_day
from elevator_ml.data.sequences import build_sequence_dataset
from elevator_ml.simulation.entities import DayData, Passenger


class SequenceDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.day = generate_day("train", "morning", 0, 12, 20, 5)

    def test_shape_and_day_boundaries(self) -> None:
        second = generate_day("train", "lunch", 0, 13, 20, 5)
        dataset = build_sequence_dataset(
            (self.day, second), 5, 2, ("morning", "lunch")
        )

        self.assertEqual(dataset.inputs.shape, (28, 5, 15))
        self.assertEqual(dataset.targets.shape, (28, 5))
        self.assertTrue(
            (
                dataset.metadata["input_end_minute_exclusive"]
                == dataset.metadata["target_start_minute"]
            ).all()
        )
        self.assertTrue(
            (
                dataset.metadata["input_start_minute"]
                == dataset.metadata["target_start_minute"] - 5
            ).all()
        )

    def test_target_is_exact_future_origin_count(self) -> None:
        dataset = build_sequence_dataset(
            (self.day,), 5, 2, ("morning", "lunch")
        )
        target_minute = int(dataset.metadata.iloc[0]["target_start_minute"])

        np.testing.assert_array_equal(
            dataset.targets[0],
            self.day.origin_counts[target_minute : target_minute + 2].sum(
                axis=0
            ),
        )

    def test_future_counts_do_not_appear_in_inputs(self) -> None:
        dataset = build_sequence_dataset(
            (self.day,), 5, 2, ("morning", "lunch")
        )
        modified_origins = self.day.origin_counts.copy()
        modified_destinations = self.day.destination_counts.copy()
        target_minute = int(dataset.metadata.iloc[0]["target_start_minute"])
        modified_origins[target_minute, 0] += 1
        modified_destinations[target_minute, 1] += 1
        extra = Passenger(
            passenger_id=len(self.day.passengers),
            arrival_time=target_minute * 60,
            origin=0,
            destination=1,
        )
        changed = DayData(
            day_id=self.day.day_id,
            split=self.day.split,
            scenario=self.day.scenario,
            seed=self.day.seed,
            duration_minutes=self.day.duration_minutes,
            passengers=self.day.passengers + (extra,),
            origin_counts=modified_origins,
            destination_counts=modified_destinations,
        )
        changed_dataset = build_sequence_dataset(
            (changed,), 5, 2, ("morning", "lunch")
        )
        np.testing.assert_array_equal(
            dataset.inputs[0], changed_dataset.inputs[0]
        )
        self.assertFalse(
            np.array_equal(dataset.targets[0], changed_dataset.targets[0])
        )
