from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.config import BuildingConfig
from elevator_ml.evaluation.metrics import (
    floor_service_metrics,
    paired_bootstrap_interval,
    summarize_simulation,
)
from elevator_ml.simulation.entities import Elevator, Passenger
from elevator_ml.simulation.simulator import SimulationResult


BUILDING = BuildingConfig(
    n_floors=4,
    n_elevators=1,
    capacity=4,
    seconds_per_floor=2,
    door_seconds=4,
    maximum_drain_minutes=10,
)


def manual_result() -> SimulationResult:
    first = Passenger(0, arrival_time=0, origin=0, destination=3)
    first.assigned_elevator = 0
    first.boarding_time = 10
    first.dropoff_time = 20
    second = Passenger(1, arrival_time=0, origin=2, destination=0)
    second.assigned_elevator = 0
    second.boarding_time = 70
    second.dropoff_time = 90
    elevator = Elevator(
        elevator_id=0,
        floor=0,
        floors_travelled=8,
        stops_served=4,
        reversals=1,
        max_load=2,
    )
    return SimulationResult(
        day_id="manual-mixed-00",
        scenario="mixed",
        policy_name="nearest_car",
        generation_end_timestamp=60,
        final_timestamp=90,
        passengers=(first, second),
        elevators=(elevator,),
    )


class MetricTests(unittest.TestCase):
    def test_simulation_summary_matches_known_values(self) -> None:
        metrics = summarize_simulation(
            manual_result(), BUILDING, long_wait_seconds=60
        )

        self.assertEqual(metrics.passengers, 2)
        self.assertEqual(metrics.mean_wait_s, 40.0)
        self.assertEqual(metrics.median_wait_s, 40.0)
        self.assertEqual(metrics.p95_wait_s, 67.0)
        self.assertEqual(metrics.long_wait_rate, 0.5)
        self.assertEqual(metrics.mean_ride_s, 15.0)
        self.assertEqual(metrics.mean_journey_s, 55.0)
        self.assertEqual(metrics.floor_fairness_gap_s, 60.0)
        self.assertEqual(metrics.floors_travelled, 8)
        self.assertEqual(metrics.reversals, 1)
        self.assertEqual(metrics.capacity_utilization_peak, 0.5)
        self.assertEqual(metrics.drain_seconds, 30)

    def test_floor_metrics_include_unobserved_floors(self) -> None:
        rows = floor_service_metrics(
            manual_result(), n_floors=4, long_wait_seconds=60
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0].passengers, 1)
        self.assertEqual(rows[2].mean_wait_s, 70.0)
        self.assertEqual(rows[1].passengers, 0)
        self.assertTrue(math.isnan(rows[1].mean_wait_s))

    def test_paired_bootstrap_constant_effect_has_exact_interval(self) -> None:
        interval = paired_bootstrap_interval(
            np.array([10.0, 12.0, 14.0, 16.0]),
            np.array([8.0, 10.0, 12.0, 14.0]),
            resamples=500,
            seed=3404,
        )

        self.assertEqual(interval.estimate, 2.0)
        self.assertEqual(interval.lower, 2.0)
        self.assertEqual(interval.upper, 2.0)
        self.assertEqual(interval.paired_days, 4)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        baseline = np.array([10.0, 8.0, 14.0, 11.0])
        candidate = np.array([9.0, 9.0, 10.0, 12.0])

        first = paired_bootstrap_interval(
            baseline, candidate, resamples=250, seed=99
        )
        second = paired_bootstrap_interval(
            baseline, candidate, resamples=250, seed=99
        )

        self.assertEqual(first, second)

    def test_unpaired_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_interval(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                resamples=100,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
