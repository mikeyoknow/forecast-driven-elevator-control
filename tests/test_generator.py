from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.data.generator import (
    aggregate_minute_counts,
    create_days,
    generate_passengers,
)
from elevator_ml.simulation.entities import Passenger


def passenger_signature(
    passengers: tuple[Passenger, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (
            passenger.passenger_id,
            passenger.arrival_time,
            passenger.origin,
            passenger.destination,
        )
        for passenger in passengers
    )


class GeneratorTests(unittest.TestCase):
    def test_same_seed_reproduces_identical_passengers(self) -> None:
        first = generate_passengers("morning", 3404, 15, 10)
        second = generate_passengers("morning", 3404, 15, 10)

        self.assertEqual(passenger_signature(first), passenger_signature(second))

    def test_different_seed_changes_passenger_stream(self) -> None:
        first = generate_passengers("morning", 3404, 15, 10)
        second = generate_passengers("morning", 3405, 15, 10)

        self.assertNotEqual(passenger_signature(first), passenger_signature(second))

    def test_generated_passengers_respect_time_and_floor_bounds(self) -> None:
        duration_minutes = 20
        n_floors = 15
        passengers = generate_passengers(
            "surge", 9000, duration_minutes, n_floors
        )

        self.assertGreater(len(passengers), 0)
        self.assertEqual(
            [passenger.arrival_time for passenger in passengers],
            sorted(passenger.arrival_time for passenger in passengers),
        )
        for passenger in passengers:
            self.assertGreaterEqual(passenger.arrival_time, 0)
            self.assertLess(passenger.arrival_time, duration_minutes * 60)
            self.assertIn(passenger.origin, range(n_floors))
            self.assertIn(passenger.destination, range(n_floors))
            self.assertNotEqual(passenger.origin, passenger.destination)

    def test_aggregation_conserves_passengers(self) -> None:
        passengers = generate_passengers("lunch", 88, 12, 8)
        origin, destination = aggregate_minute_counts(passengers, 12, 8)

        self.assertEqual(origin.shape, (12, 8))
        self.assertEqual(destination.shape, (12, 8))
        self.assertEqual(int(origin.sum()), len(passengers))
        self.assertEqual(int(destination.sum()), len(passengers))
        np.testing.assert_array_equal(origin.sum(axis=1), destination.sum(axis=1))

    def test_create_days_produces_unique_ids_and_seeds(self) -> None:
        days = create_days(
            split="train",
            scenarios=("morning", "lunch"),
            days_per_scenario=3,
            seed_offset=10_000,
            duration_minutes=10,
            n_floors=6,
        )

        self.assertEqual(len(days), 6)
        self.assertEqual(len({day.day_id for day in days}), 6)
        self.assertEqual(len({day.seed for day in days}), 6)
        self.assertTrue(all(day.split == "train" for day in days))

    def test_two_floor_building_is_supported(self) -> None:
        passengers = generate_passengers("mixed", 22, 5, 2)

        self.assertGreater(len(passengers), 0)
        self.assertTrue(
            all(
                {passenger.origin, passenger.destination} == {0, 1}
                for passenger in passengers
            )
        )

    def test_unknown_scenario_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_passengers("weekend", 1, 10, 5)

    def test_demand_multiplier_increases_expected_event_count(self) -> None:
        normal = generate_passengers("mixed", 77, 30, 8, demand_multiplier=1.0)
        high = generate_passengers("mixed", 77, 30, 8, demand_multiplier=1.5)

        self.assertGreater(len(high), len(normal))

    def test_out_of_day_arrival_is_rejected_during_aggregation(self) -> None:
        passenger = Passenger(0, arrival_time=60, origin=0, destination=1)

        with self.assertRaises(ValueError):
            aggregate_minute_counts((passenger,), duration_minutes=1, n_floors=2)


if __name__ == "__main__":
    unittest.main()
