"""Interchangeable elevator assignment policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np

from elevator_ml.config import BuildingConfig

if TYPE_CHECKING:
    from elevator_ml.simulation.entities import Elevator, Passenger


class AssignmentPolicy(Protocol):
    """Interface used by the simulator when assigning a passenger call."""

    name: str

    def choose_elevator(
        self,
        passenger: Passenger,
        elevators: Sequence[Elevator],
        passengers: Sequence[Passenger],
        config: BuildingConfig,
    ) -> int | None:
        """Return an elevator ID, or None if no car can currently accept a call."""


class PositioningPolicy(Protocol):
    """Interface for moving service-idle cars before hall calls arrive."""

    name: str

    def update(
        self,
        timestamp: int,
        elevators: Sequence[Elevator],
        config: BuildingConfig,
    ) -> None: ...


class NearestCarPolicy:
    """Reactive baseline using distance, route, queue, and occupancy costs."""

    name = "nearest_car"

    @staticmethod
    def _score(
        elevator: Elevator,
        passenger: Passenger,
        config: BuildingConfig,
    ) -> tuple[float, int]:
        distance_cost = (
            abs(elevator.floor - passenger.origin) * config.seconds_per_floor
        )
        route_cost = len(elevator.stops) * config.door_seconds
        queue_cost = 0.75 * len(elevator.waiting_ids)
        occupancy_cost = 0.5 * elevator.load

        moving_away = (
            elevator.direction > 0 and passenger.origin < elevator.floor
        ) or (
            elevator.direction < 0 and passenger.origin > elevator.floor
        )
        if moving_away:
            route_extent = max(
                (abs(stop - elevator.floor) for stop in elevator.stops),
                default=0,
            )
            distance_cost += 2 * route_extent * config.seconds_per_floor

        stop_bonus = (
            -float(config.door_seconds)
            if passenger.origin in elevator.stops
            else 0.0
        )
        same_floor_bonus = (
            -3.0
            if elevator.floor == passenger.origin and elevator.move_timer == 0
            else 0.0
        )
        score = (
            distance_cost
            + route_cost
            + queue_cost
            + occupancy_cost
            + stop_bonus
            + same_floor_bonus
        )
        # Elevator ID makes tie-breaking explicit and deterministic.
        return score, elevator.elevator_id

    def choose_elevator(
        self,
        passenger: Passenger,
        elevators: Sequence[Elevator],
        passengers: Sequence[Passenger],
        config: BuildingConfig,
    ) -> int | None:
        del passengers  # Reserved for richer route estimation policies.
        eligible = [
            elevator for elevator in elevators if elevator.load < config.capacity
        ]
        if not eligible:
            return None
        chosen = min(
            eligible,
            key=lambda elevator: self._score(elevator, passenger, config),
        )
        return chosen.elevator_id


def _service_idle(elevator: Elevator) -> bool:
    """True when a car has no passenger-service obligation."""
    return (
        not elevator.stops
        and not elevator.waiting_ids
        and not elevator.onboard_ids
        and elevator.door_timer == 0
    )


class StaticZoningPolicy:
    """Non-predictive baseline that spaces idle cars across the building."""

    name = "static_zoning"

    def update(
        self,
        timestamp: int,
        elevators: Sequence[Elevator],
        config: BuildingConfig,
    ) -> None:
        if timestamp % 60 != 0:
            return
        targets = np.rint(
            np.linspace(0, config.n_floors - 1, config.n_elevators)
        ).astype(int)
        for elevator in elevators:
            if not _service_idle(elevator):
                continue
            target = int(targets[elevator.elevator_id % len(targets)])
            elevator.parking_target = None if target == elevator.floor else target


class StayIdlePositioningPolicy:
    """Named no-positioning policy used inside gated controller comparisons."""

    def __init__(self, name: str = "stay_idle") -> None:
        if not name.strip():
            raise ValueError("Positioning policy name cannot be blank.")
        self.name = name

    def update(
        self,
        timestamp: int,
        elevators: Sequence[Elevator],
        config: BuildingConfig,
    ) -> None:
        del config
        if timestamp % 60 != 0:
            return
        for elevator in elevators:
            if _service_idle(elevator):
                elevator.parking_target = None


class ForecastPositioningPolicy:
    """Greedily place idle cars near predicted per-floor demand."""

    def __init__(
        self,
        schedule: np.ndarray,
        *,
        distance_penalty: float,
        name: str,
        minimum_score_advantage: float = 0.0,
        positioning_interval_minutes: int = 1,
    ) -> None:
        values = np.asarray(schedule, dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Forecast schedule must be a finite 2D array.")
        if np.any(values < 0):
            raise ValueError("Forecast demand cannot be negative.")
        if distance_penalty < 0:
            raise ValueError("Positioning distance penalty cannot be negative.")
        if minimum_score_advantage < 0:
            raise ValueError("Minimum positioning advantage cannot be negative.")
        if positioning_interval_minutes <= 0:
            raise ValueError("Positioning interval must be positive.")
        if not name.strip():
            raise ValueError("Positioning policy name cannot be blank.")
        self.schedule = values
        self.distance_penalty = float(distance_penalty)
        self.minimum_score_advantage = float(minimum_score_advantage)
        self.positioning_interval_minutes = int(positioning_interval_minutes)
        self.name = name

    def _prediction_allowed(self, minute: int) -> bool:
        return True

    def update(
        self,
        timestamp: int,
        elevators: Sequence[Elevator],
        config: BuildingConfig,
    ) -> None:
        if timestamp % (60 * self.positioning_interval_minutes) != 0:
            return
        minute = timestamp // 60
        if minute >= self.schedule.shape[0]:
            return
        if self.schedule.shape[1] != config.n_floors:
            raise ValueError("Forecast floor count differs from the building.")
        if not self._prediction_allowed(minute):
            for elevator in elevators:
                if _service_idle(elevator):
                    elevator.parking_target = None
            return
        forecast = self.schedule[minute]
        if float(forecast.sum()) < 0.5:
            return
        reserved = np.zeros(config.n_floors, dtype=float)
        for elevator in sorted(elevators, key=lambda car: car.elevator_id):
            if not _service_idle(elevator):
                continue
            distance = np.abs(np.arange(config.n_floors) - elevator.floor)
            score = forecast / (1.0 + reserved) - self.distance_penalty * distance
            target = int(np.argmax(score))
            current_score = float(score[elevator.floor])
            if float(score[target]) - current_score < self.minimum_score_advantage:
                elevator.parking_target = None
                continue
            reserved[target] += 1.0
            elevator.parking_target = None if target == elevator.floor else target


class UncertaintyGatedPositioningPolicy(ForecastPositioningPolicy):
    """Use forecast positioning only below a validation-set uncertainty limit."""

    def __init__(
        self,
        schedule: np.ndarray,
        uncertainty: np.ndarray,
        *,
        uncertainty_threshold: float,
        distance_penalty: float,
        name: str = "uncertainty_gated_random_forest",
        minimum_score_advantage: float = 0.0,
        positioning_interval_minutes: int = 1,
    ) -> None:
        super().__init__(
            schedule,
            distance_penalty=distance_penalty,
            name=name,
            minimum_score_advantage=minimum_score_advantage,
            positioning_interval_minutes=positioning_interval_minutes,
        )
        values = np.asarray(uncertainty, dtype=float)
        if values.ndim != 1 or values.shape[0] != self.schedule.shape[0]:
            raise ValueError("Uncertainty must have one value per forecast minute.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Uncertainty values must be finite and non-negative.")
        if uncertainty_threshold < 0:
            raise ValueError("Uncertainty threshold cannot be negative.")
        self.uncertainty = values
        self.uncertainty_threshold = float(uncertainty_threshold)

    def _prediction_allowed(self, minute: int) -> bool:
        return self.uncertainty[minute] <= self.uncertainty_threshold
