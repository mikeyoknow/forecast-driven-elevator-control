"""Core domain entities and their local correctness checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from elevator_ml.config import BuildingConfig


@dataclass
class Passenger:
    passenger_id: int
    arrival_time: int
    origin: int
    destination: int
    assigned_elevator: int | None = None
    boarding_time: int | None = None
    dropoff_time: int | None = None

    def __post_init__(self) -> None:
        if self.passenger_id < 0:
            raise ValueError("passenger_id cannot be negative.")
        if self.arrival_time < 0:
            raise ValueError("arrival_time cannot be negative.")
        if self.origin < 0 or self.destination < 0:
            raise ValueError("Passenger floors cannot be negative.")
        if self.origin == self.destination:
            raise ValueError("Passenger origin and destination must differ.")

    def validate(self, n_floors: int) -> None:
        if not 0 <= self.origin < n_floors:
            raise ValueError(f"Passenger origin {self.origin} is outside the building.")
        if not 0 <= self.destination < n_floors:
            raise ValueError(
                f"Passenger destination {self.destination} is outside the building."
            )
        if self.boarding_time is not None and self.boarding_time < self.arrival_time:
            raise ValueError("A passenger cannot board before arriving.")
        if self.dropoff_time is not None:
            if self.boarding_time is None:
                raise ValueError("A passenger cannot be dropped off before boarding.")
            if self.dropoff_time < self.boarding_time:
                raise ValueError("A passenger cannot be dropped off before boarding.")

    @property
    def wait_seconds(self) -> int | None:
        if self.boarding_time is None:
            return None
        return self.boarding_time - self.arrival_time

    @property
    def ride_seconds(self) -> int | None:
        if self.boarding_time is None or self.dropoff_time is None:
            return None
        return self.dropoff_time - self.boarding_time

    @property
    def journey_seconds(self) -> int | None:
        if self.dropoff_time is None:
            return None
        return self.dropoff_time - self.arrival_time


@dataclass
class Elevator:
    elevator_id: int
    floor: int = 0
    direction: int = 0
    stops: set[int] = field(default_factory=set)
    parking_target: int | None = None
    waiting_ids: set[int] = field(default_factory=set)
    onboard_ids: list[int] = field(default_factory=list)
    move_timer: int = 0
    door_timer: int = 0
    floors_travelled: int = 0
    stops_served: int = 0
    reversals: int = 0
    max_load: int = 0

    @property
    def load(self) -> int:
        return len(self.onboard_ids)

    @property
    def is_idle(self) -> bool:
        return (
            not self.stops
            and not self.waiting_ids
            and not self.onboard_ids
            and self.parking_target is None
            and self.move_timer == 0
            and self.door_timer == 0
        )

    def validate(self, config: BuildingConfig) -> None:
        if self.elevator_id < 0:
            raise ValueError("elevator_id cannot be negative.")
        if not 0 <= self.floor < config.n_floors:
            raise ValueError(f"Elevator floor {self.floor} is outside the building.")
        if self.direction not in (-1, 0, 1):
            raise ValueError("Elevator direction must be -1, 0, or 1.")
        if self.move_timer < 0 or self.door_timer < 0:
            raise ValueError("Elevator timers cannot be negative.")
        if self.load > config.capacity:
            raise ValueError(
                f"Elevator load {self.load} exceeds capacity {config.capacity}."
            )
        if len(set(self.onboard_ids)) != len(self.onboard_ids):
            raise ValueError("An onboard passenger cannot appear more than once.")
        if set(self.onboard_ids) & self.waiting_ids:
            raise ValueError("A passenger cannot be waiting and onboard simultaneously.")
        if any(not 0 <= stop < config.n_floors for stop in self.stops):
            raise ValueError("Elevator stops must remain inside the building.")
        if self.parking_target is not None and not (
            0 <= self.parking_target < config.n_floors
        ):
            raise ValueError("Elevator parking target is outside the building.")
        for counter_name in ("floors_travelled", "stops_served", "reversals", "max_load"):
            if getattr(self, counter_name) < 0:
                raise ValueError(f"{counter_name} cannot be negative.")


@dataclass(frozen=True)
class DayData:
    day_id: str
    split: str
    scenario: str
    seed: int
    duration_minutes: int
    passengers: tuple[Passenger, ...]
    origin_counts: np.ndarray
    destination_counts: np.ndarray

    def __post_init__(self) -> None:
        if not self.day_id.strip() or not self.split.strip() or not self.scenario.strip():
            raise ValueError("Day identifiers, split, and scenario cannot be blank.")
        if self.seed < 0:
            raise ValueError("Day seed cannot be negative.")
        if self.duration_minutes <= 0:
            raise ValueError("Day duration must be positive.")
        if self.origin_counts.ndim != 2 or self.destination_counts.ndim != 2:
            raise ValueError("Origin and destination counts must be 2D arrays.")
        if self.origin_counts.shape != self.destination_counts.shape:
            raise ValueError("Origin and destination count shapes must match.")
        if self.origin_counts.shape[0] != self.duration_minutes:
            raise ValueError("Count rows must equal duration_minutes.")
        if self.origin_counts.shape[1] < 2:
            raise ValueError("Count matrices must contain at least two floors.")
        if np.any(self.origin_counts < 0) or np.any(self.destination_counts < 0):
            raise ValueError("Passenger counts cannot be negative.")
        passenger_ids = [passenger.passenger_id for passenger in self.passengers]
        if len(set(passenger_ids)) != len(passenger_ids):
            raise ValueError("Passenger IDs must be unique within a day.")
        n_floors = self.origin_counts.shape[1]
        for passenger in self.passengers:
            passenger.validate(n_floors)
        if not np.isclose(self.origin_counts.sum(), len(self.passengers)):
            raise ValueError("Origin counts must conserve the number of passengers.")
        if not np.isclose(self.destination_counts.sum(), len(self.passengers)):
            raise ValueError("Destination counts must conserve the number of passengers.")
