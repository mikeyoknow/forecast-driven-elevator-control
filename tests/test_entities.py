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
from elevator_ml.simulation.entities import DayData, Elevator, Passenger


BUILDING = BuildingConfig(
    n_floors=5,
    n_elevators=2,
    capacity=2,
    seconds_per_floor=2,
    door_seconds=4,
    maximum_drain_minutes=10,
)


class PassengerTests(unittest.TestCase):
    def test_completed_passenger_metrics(self) -> None:
        passenger = Passenger(0, arrival_time=10, origin=0, destination=4)
        passenger.boarding_time = 16
        passenger.dropoff_time = 30
        passenger.validate(BUILDING.n_floors)

        self.assertEqual(passenger.wait_seconds, 6)
        self.assertEqual(passenger.ride_seconds, 14)
        self.assertEqual(passenger.journey_seconds, 20)

    def test_same_origin_and_destination_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Passenger(0, arrival_time=0, origin=2, destination=2)

    def test_boarding_before_arrival_is_rejected(self) -> None:
        passenger = Passenger(0, arrival_time=10, origin=0, destination=4)
        passenger.boarding_time = 9

        with self.assertRaises(ValueError):
            passenger.validate(BUILDING.n_floors)


class ElevatorTests(unittest.TestCase):
    def test_new_elevator_is_idle_and_valid(self) -> None:
        elevator = Elevator(elevator_id=0)

        elevator.validate(BUILDING)
        self.assertTrue(elevator.is_idle)
        self.assertEqual(elevator.load, 0)

    def test_capacity_violation_is_rejected(self) -> None:
        elevator = Elevator(elevator_id=0, onboard_ids=[1, 2, 3])

        with self.assertRaises(ValueError):
            elevator.validate(BUILDING)

    def test_passenger_cannot_be_waiting_and_onboard(self) -> None:
        elevator = Elevator(elevator_id=0, waiting_ids={7}, onboard_ids=[7])

        with self.assertRaises(ValueError):
            elevator.validate(BUILDING)


class DayDataTests(unittest.TestCase):
    def test_day_conserves_passengers(self) -> None:
        passenger = Passenger(0, arrival_time=5, origin=0, destination=3)
        origin = np.zeros((2, 5), dtype=float)
        destination = np.zeros((2, 5), dtype=float)
        origin[0, 0] = 1
        destination[0, 3] = 1

        day = DayData(
            day_id="train-morning-00",
            split="train",
            scenario="morning",
            seed=123,
            duration_minutes=2,
            passengers=(passenger,),
            origin_counts=origin,
            destination_counts=destination,
        )

        self.assertEqual(len(day.passengers), 1)

    def test_inconsistent_counts_are_rejected(self) -> None:
        passenger = Passenger(0, arrival_time=5, origin=0, destination=3)
        counts = np.zeros((2, 5), dtype=float)

        with self.assertRaises(ValueError):
            DayData(
                day_id="train-morning-00",
                split="train",
                scenario="morning",
                seed=123,
                duration_minutes=2,
                passengers=(passenger,),
                origin_counts=counts,
                destination_counts=counts.copy(),
            )


if __name__ == "__main__":
    unittest.main()
