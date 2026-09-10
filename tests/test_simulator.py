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
from elevator_ml.control.policies import ForecastPositioningPolicy
from elevator_ml.data.generator import (
    aggregate_minute_counts,
    generate_day,
)
from elevator_ml.simulation.entities import DayData, Passenger
from elevator_ml.simulation.simulator import simulate_day


class HighestIdAvailablePolicy:
    name = "highest_id_available"

    def choose_elevator(self, passenger, elevators, passengers, config):
        del passenger, passengers
        eligible = [
            elevator.elevator_id
            for elevator in elevators
            if elevator.load < config.capacity
        ]
        return max(eligible) if eligible else None


def make_building(*, capacity: int = 2, n_elevators: int = 1) -> BuildingConfig:
    return BuildingConfig(
        n_floors=5,
        n_elevators=n_elevators,
        capacity=capacity,
        seconds_per_floor=2,
        door_seconds=4,
        maximum_drain_minutes=10,
    )


def make_day(passengers: tuple[Passenger, ...], duration_minutes: int = 1) -> DayData:
    origin, destination = aggregate_minute_counts(
        passengers, duration_minutes=duration_minutes, n_floors=5
    )
    return DayData(
        day_id="manual-mixed-00",
        split="manual",
        scenario="mixed",
        seed=0,
        duration_minutes=duration_minutes,
        passengers=passengers,
        origin_counts=origin,
        destination_counts=destination,
    )


class SimulatorTests(unittest.TestCase):
    def test_one_passenger_matches_hand_calculated_timing(self) -> None:
        # At t=0 the passenger boards at floor 0. Four door seconds and two
        # two-second floor movements place the drop-off at t=10.
        day = make_day((Passenger(0, 0, 0, 2),))

        result = simulate_day(day, make_building(), validate_invariants=True)
        passenger = result.passengers[0]

        self.assertTrue(result.all_served)
        self.assertEqual(passenger.boarding_time, 0)
        self.assertEqual(passenger.dropoff_time, 10)
        self.assertEqual(passenger.wait_seconds, 0)
        self.assertEqual(passenger.ride_seconds, 10)
        self.assertEqual(result.floors_travelled, 2)
        self.assertEqual(result.stops_served, 2)

    def test_capacity_overflow_is_eventually_served(self) -> None:
        day = make_day(
            (
                Passenger(0, 0, 0, 1),
                Passenger(1, 0, 0, 2),
                Passenger(2, 0, 0, 3),
            )
        )

        result = simulate_day(
            day,
            make_building(capacity=1),
            validate_invariants=True,
        )

        self.assertTrue(result.all_served)
        self.assertEqual(result.max_elevator_load, 1)
        self.assertTrue(
            all(passenger.dropoff_time is not None for passenger in result.passengers)
        )

    def test_simulation_does_not_mutate_source_day(self) -> None:
        day = make_day((Passenger(0, 0, 0, 4),))

        simulate_day(day, make_building(), validate_invariants=True)

        self.assertIsNone(day.passengers[0].assigned_elevator)
        self.assertIsNone(day.passengers[0].boarding_time)
        self.assertIsNone(day.passengers[0].dropoff_time)

    def test_same_day_reproduces_identical_outcome(self) -> None:
        day = generate_day(
            split="test",
            scenario="morning",
            repetition=0,
            seed=1234,
            duration_minutes=5,
            n_floors=5,
        )
        building = make_building(n_elevators=2)

        first = simulate_day(day, building, validate_invariants=True)
        second = simulate_day(day, building, validate_invariants=True)

        first_times = [
            (passenger.boarding_time, passenger.dropoff_time)
            for passenger in first.passengers
        ]
        second_times = [
            (passenger.boarding_time, passenger.dropoff_time)
            for passenger in second.passengers
        ]
        self.assertEqual(first_times, second_times)
        self.assertEqual(first.floors_travelled, second.floors_travelled)

    def test_generated_day_serves_every_passenger_without_capacity_violation(self) -> None:
        day = generate_day(
            split="test",
            scenario="surge",
            repetition=0,
            seed=998,
            duration_minutes=8,
            n_floors=5,
        )
        building = make_building(capacity=3, n_elevators=2)

        result = simulate_day(day, building, validate_invariants=True)

        self.assertTrue(result.all_served)
        self.assertLessEqual(result.max_elevator_load, building.capacity)
        for elevator in result.elevators:
            elevator.validate(building)

    def test_elevator_reverses_after_serving_upper_destination(self) -> None:
        day = make_day(
            (
                Passenger(0, 0, 0, 4),
                Passenger(1, 1, 2, 0),
            )
        )

        result = simulate_day(day, make_building(), validate_invariants=True)

        self.assertTrue(result.all_served)
        self.assertGreaterEqual(result.elevators[0].reversals, 1)
        self.assertLess(
            result.passengers[0].dropoff_time,
            result.passengers[1].dropoff_time,
        )

    def test_late_arrival_is_served_during_drain_period(self) -> None:
        day = make_day((Passenger(0, 59, 0, 4),))

        result = simulate_day(day, make_building(), validate_invariants=True)

        self.assertTrue(result.all_served)
        self.assertGreater(result.final_timestamp, result.generation_end_timestamp)
        self.assertEqual(result.generation_end_timestamp, 60)

    def test_different_policies_receive_identical_passenger_demand(self) -> None:
        day = generate_day(
            split="test",
            scenario="lunch",
            repetition=0,
            seed=505,
            duration_minutes=5,
            n_floors=5,
        )
        building = make_building(n_elevators=2)

        nearest = simulate_day(day, building, validate_invariants=True)
        alternate = simulate_day(
            day,
            building,
            policy=HighestIdAvailablePolicy(),
            validate_invariants=True,
        )

        nearest_demand = [
            (p.arrival_time, p.origin, p.destination) for p in nearest.passengers
        ]
        alternate_demand = [
            (p.arrival_time, p.origin, p.destination) for p in alternate.passengers
        ]
        self.assertEqual(nearest_demand, alternate_demand)
        self.assertEqual(nearest_demand, [
            (p.arrival_time, p.origin, p.destination) for p in day.passengers
        ])

    def test_positioning_policy_moves_idle_car_without_passengers(self) -> None:
        day = make_day(())
        schedule = np.zeros((1, 5), dtype=float)
        schedule[0, 4] = 10.0

        result = simulate_day(
            day,
            make_building(),
            positioning_policy=ForecastPositioningPolicy(
                schedule,
                distance_penalty=0.0,
                name="forecast_parking",
            ),
            validate_invariants=True,
        )

        self.assertEqual(result.policy_name, "forecast_parking")
        self.assertEqual(result.elevators[0].floor, 4)
        self.assertEqual(result.floors_travelled, 4)


if __name__ == "__main__":
    unittest.main()
