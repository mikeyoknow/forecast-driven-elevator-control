from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.config import BuildingConfig
from elevator_ml.control.policies import (
    ForecastPositioningPolicy,
    NearestCarPolicy,
    StaticZoningPolicy,
    UncertaintyGatedPositioningPolicy,
)
from elevator_ml.simulation.entities import Elevator, Passenger


BUILDING = BuildingConfig(
    n_floors=10,
    n_elevators=2,
    capacity=2,
    seconds_per_floor=2,
    door_seconds=4,
    maximum_drain_minutes=10,
)


class NearestCarPolicyTests(unittest.TestCase):
    def test_selects_closest_available_car(self) -> None:
        passenger = Passenger(0, arrival_time=0, origin=7, destination=0)
        elevators = [Elevator(0, floor=0), Elevator(1, floor=6)]

        chosen = NearestCarPolicy().choose_elevator(
            passenger, elevators, [passenger], BUILDING
        )

        self.assertEqual(chosen, 1)

    def test_full_car_is_not_selected(self) -> None:
        passenger = Passenger(0, arrival_time=0, origin=7, destination=0)
        elevators = [
            Elevator(0, floor=7, onboard_ids=[10, 11]),
            Elevator(1, floor=0),
        ]

        chosen = NearestCarPolicy().choose_elevator(
            passenger, elevators, [passenger], BUILDING
        )

        self.assertEqual(chosen, 1)

    def test_returns_none_when_every_car_is_full(self) -> None:
        passenger = Passenger(0, arrival_time=0, origin=7, destination=0)
        elevators = [
            Elevator(0, floor=7, onboard_ids=[10, 11]),
            Elevator(1, floor=0, onboard_ids=[12, 13]),
        ]

        chosen = NearestCarPolicy().choose_elevator(
            passenger, elevators, [passenger], BUILDING
        )

        self.assertIsNone(chosen)

    def test_ties_are_broken_by_elevator_id(self) -> None:
        passenger = Passenger(0, arrival_time=0, origin=5, destination=0)
        elevators = [Elevator(1, floor=4), Elevator(0, floor=6)]

        chosen = NearestCarPolicy().choose_elevator(
            passenger, elevators, [passenger], BUILDING
        )

        self.assertEqual(chosen, 0)


class PositioningPolicyTests(unittest.TestCase):
    def test_static_zoning_spaces_idle_cars(self) -> None:
        elevators = [Elevator(0, floor=0), Elevator(1, floor=0)]

        StaticZoningPolicy().update(0, elevators, BUILDING)

        self.assertIsNone(elevators[0].parking_target)
        self.assertEqual(elevators[1].parking_target, 9)

    def test_forecast_policy_targets_high_demand_floor(self) -> None:
        schedule = np.zeros((2, BUILDING.n_floors))
        schedule[0, 7] = 10.0
        elevators = [Elevator(0, floor=0), Elevator(1, floor=9)]
        policy = ForecastPositioningPolicy(
            schedule, distance_penalty=0.0, name="forecast"
        )

        policy.update(0, elevators, BUILDING)

        self.assertEqual(elevators[0].parking_target, 7)
        self.assertEqual(elevators[1].parking_target, 7)

    def test_busy_elevator_is_not_repositioned(self) -> None:
        schedule = np.zeros((1, BUILDING.n_floors))
        schedule[0, 8] = 10.0
        elevator = Elevator(0, floor=2, onboard_ids=[5], stops={6})

        ForecastPositioningPolicy(
            schedule, distance_penalty=0.0, name="forecast"
        ).update(0, [elevator], BUILDING)

        self.assertIsNone(elevator.parking_target)

    def test_uncertainty_gate_blocks_prediction(self) -> None:
        schedule = np.zeros((1, BUILDING.n_floors))
        schedule[0, 8] = 10.0
        elevator = Elevator(0, floor=2)
        policy = UncertaintyGatedPositioningPolicy(
            schedule,
            uncertainty=np.array([2.0]),
            uncertainty_threshold=1.0,
            distance_penalty=0.0,
        )

        policy.update(0, [elevator], BUILDING)

        self.assertIsNone(elevator.parking_target)

    def test_minimum_advantage_can_prevent_speculative_move(self) -> None:
        schedule = np.zeros((1, BUILDING.n_floors))
        schedule[0, 5] = 1.0
        elevator = Elevator(0, floor=4)

        ForecastPositioningPolicy(
            schedule,
            distance_penalty=0.0,
            minimum_score_advantage=2.0,
            name="thresholded",
        ).update(0, [elevator], BUILDING)

        self.assertIsNone(elevator.parking_target)

    def test_positioning_interval_skips_intermediate_minutes(self) -> None:
        schedule = np.zeros((3, BUILDING.n_floors))
        schedule[:, 8] = 10.0
        elevator = Elevator(0, floor=0)
        policy = ForecastPositioningPolicy(
            schedule,
            distance_penalty=0.0,
            positioning_interval_minutes=2,
            name="interval",
        )

        policy.update(60, [elevator], BUILDING)
        self.assertIsNone(elevator.parking_target)
        policy.update(120, [elevator], BUILDING)
        self.assertEqual(elevator.parking_target, 8)


if __name__ == "__main__":
    unittest.main()
