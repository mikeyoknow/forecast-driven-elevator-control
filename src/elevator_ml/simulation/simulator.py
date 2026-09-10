"""Second-by-second elevator digital twin.

The simulator is deterministic for a fixed passenger day, configuration, and
policy. It intentionally keeps demand forecasting outside the motion engine so
forecast quality and controller quality can be studied separately.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass

from elevator_ml.config import BuildingConfig
from elevator_ml.control.policies import (
    AssignmentPolicy,
    NearestCarPolicy,
    PositioningPolicy,
)
from elevator_ml.simulation.entities import DayData, Elevator, Passenger


class SimulationError(RuntimeError):
    """Raised when a run violates invariants or fails to serve every passenger."""


@dataclass(frozen=True)
class SimulationResult:
    day_id: str
    scenario: str
    policy_name: str
    generation_end_timestamp: int
    final_timestamp: int
    passengers: tuple[Passenger, ...]
    elevators: tuple[Elevator, ...]

    @property
    def all_served(self) -> bool:
        return all(
            passenger.dropoff_time is not None for passenger in self.passengers
        )

    @property
    def floors_travelled(self) -> int:
        return sum(elevator.floors_travelled for elevator in self.elevators)

    @property
    def stops_served(self) -> int:
        return sum(elevator.stops_served for elevator in self.elevators)

    @property
    def max_elevator_load(self) -> int:
        return max((elevator.max_load for elevator in self.elevators), default=0)


def _set_direction(elevator: Elevator, direction: int) -> None:
    if direction not in (-1, 0, 1):
        raise SimulationError(f"Invalid direction {direction}.")
    if (
        elevator.direction in (-1, 1)
        and direction in (-1, 1)
        and elevator.direction != direction
    ):
        elevator.reversals += 1
    elevator.direction = direction


def _assign_passenger(
    passenger_id: int,
    elevators: list[Elevator],
    passengers: list[Passenger],
    config: BuildingConfig,
    policy: AssignmentPolicy,
) -> bool:
    passenger = passengers[passenger_id]
    elevator_id = policy.choose_elevator(
        passenger, elevators, passengers, config
    )
    if elevator_id is None:
        return False
    matching = [
        elevator for elevator in elevators if elevator.elevator_id == elevator_id
    ]
    if len(matching) != 1:
        raise SimulationError(
            f"Policy returned unknown or duplicate elevator ID {elevator_id}."
        )
    elevator = matching[0]
    if elevator.load >= config.capacity:
        raise SimulationError("Policy assigned a new call to a full elevator.")
    passenger.assigned_elevator = elevator.elevator_id
    elevator.waiting_ids.add(passenger_id)
    elevator.stops.add(passenger.origin)
    # Passenger service always takes priority over speculative parking.
    elevator.parking_target = None
    return True


def _serve_floor(
    elevator: Elevator,
    timestamp: int,
    passengers: list[Passenger],
    unassigned: set[int],
    config: BuildingConfig,
) -> None:
    elevator.stops.discard(elevator.floor)

    dropped = [
        passenger_id
        for passenger_id in elevator.onboard_ids
        if passengers[passenger_id].destination == elevator.floor
    ]
    for passenger_id in dropped:
        passengers[passenger_id].dropoff_time = timestamp
        elevator.onboard_ids.remove(passenger_id)

    pickup_candidates = sorted(
        (
            passenger_id
            for passenger_id in elevator.waiting_ids
            if passengers[passenger_id].origin == elevator.floor
            and passengers[passenger_id].arrival_time <= timestamp
        ),
        key=lambda passenger_id: (
            passengers[passenger_id].arrival_time,
            passenger_id,
        ),
    )
    available = config.capacity - elevator.load
    boarding = pickup_candidates[:available]
    overflow = pickup_candidates[available:]

    for passenger_id in boarding:
        passenger = passengers[passenger_id]
        passenger.boarding_time = timestamp
        elevator.waiting_ids.remove(passenger_id)
        elevator.onboard_ids.append(passenger_id)
        elevator.stops.add(passenger.destination)

    # Overflow calls return to the global queue so another car may accept them.
    for passenger_id in overflow:
        elevator.waiting_ids.remove(passenger_id)
        passengers[passenger_id].assigned_elevator = None
        unassigned.add(passenger_id)

    if dropped or boarding:
        elevator.stops_served += 1
        elevator.door_timer = config.door_seconds
        elevator.max_load = max(elevator.max_load, elevator.load)


def _next_target(elevator: Elevator) -> int | None:
    if elevator.stops:
        if elevator.direction > 0:
            above = sorted(stop for stop in elevator.stops if stop > elevator.floor)
            if above:
                return above[0]
            below = sorted(
                (stop for stop in elevator.stops if stop < elevator.floor),
                reverse=True,
            )
            if below:
                _set_direction(elevator, -1)
                return below[0]
        elif elevator.direction < 0:
            below = sorted(
                (stop for stop in elevator.stops if stop < elevator.floor),
                reverse=True,
            )
            if below:
                return below[0]
            above = sorted(stop for stop in elevator.stops if stop > elevator.floor)
            if above:
                _set_direction(elevator, 1)
                return above[0]

        target = min(
            elevator.stops,
            key=lambda floor: (abs(floor - elevator.floor), floor),
        )
        _set_direction(elevator, 1 if target > elevator.floor else -1)
        return target

    if elevator.parking_target is not None:
        target = elevator.parking_target
        if target == elevator.floor:
            elevator.parking_target = None
            _set_direction(elevator, 0)
            return None
        _set_direction(elevator, 1 if target > elevator.floor else -1)
        return target

    _set_direction(elevator, 0)
    return None


def _tick_elevator(
    elevator: Elevator,
    timestamp: int,
    passengers: list[Passenger],
    unassigned: set[int],
    config: BuildingConfig,
) -> None:
    if elevator.door_timer > 0:
        elevator.door_timer -= 1
        return

    if elevator.move_timer > 0:
        elevator.move_timer -= 1
        if elevator.move_timer == 0:
            if elevator.direction == 0:
                raise SimulationError("A moving elevator cannot have zero direction.")
            elevator.floor += elevator.direction
            elevator.floors_travelled += 1
            if not 0 <= elevator.floor < config.n_floors:
                raise SimulationError("An elevator moved outside the building.")
            if elevator.floor in elevator.stops:
                _serve_floor(elevator, timestamp, passengers, unassigned, config)
            elif not elevator.stops and elevator.parking_target == elevator.floor:
                elevator.parking_target = None
                _set_direction(elevator, 0)
        return

    if elevator.floor in elevator.stops:
        _serve_floor(elevator, timestamp, passengers, unassigned, config)
        return

    target = _next_target(elevator)
    if target is None:
        return
    if target == elevator.floor:
        raise SimulationError("The next elevator target cannot equal its floor.")
    elevator.move_timer = config.seconds_per_floor


def assert_system_invariants(
    timestamp: int,
    passengers: list[Passenger],
    elevators: list[Elevator],
    unassigned: set[int],
    config: BuildingConfig,
) -> None:
    """Validate global passenger conservation and mutually exclusive states."""
    elevator_ids = [elevator.elevator_id for elevator in elevators]
    if len(set(elevator_ids)) != len(elevator_ids):
        raise SimulationError("Elevator IDs must be unique.")
    for elevator in elevators:
        try:
            elevator.validate(config)
        except ValueError as error:
            raise SimulationError(str(error)) from error

    waiting_location: dict[int, int] = {}
    onboard_location: dict[int, int] = {}
    for elevator in elevators:
        for passenger_id in elevator.waiting_ids:
            if passenger_id in waiting_location:
                raise SimulationError("A passenger is waiting for multiple elevators.")
            waiting_location[passenger_id] = elevator.elevator_id
        for passenger_id in elevator.onboard_ids:
            if passenger_id in onboard_location:
                raise SimulationError("A passenger is onboard multiple elevators.")
            onboard_location[passenger_id] = elevator.elevator_id

    valid_ids = set(range(len(passengers)))
    referenced_ids = set(waiting_location) | set(onboard_location) | unassigned
    if not referenced_ids <= valid_ids:
        raise SimulationError("Elevator state references an unknown passenger ID.")

    for passenger in passengers:
        passenger_id = passenger.passenger_id
        if passenger_id not in valid_ids:
            raise SimulationError("Passenger IDs must be contiguous from zero.")
        try:
            passenger.validate(config.n_floors)
        except ValueError as error:
            raise SimulationError(str(error)) from error

        memberships = sum(
            (
                passenger_id in waiting_location,
                passenger_id in onboard_location,
                passenger_id in unassigned,
            )
        )
        if memberships > 1:
            raise SimulationError("A passenger occupies multiple system states.")
        if passenger.arrival_time > timestamp:
            if memberships or passenger.assigned_elevator is not None:
                raise SimulationError("A passenger entered the system before arrival.")
            continue
        if passenger.dropoff_time is not None:
            if memberships:
                raise SimulationError("A completed passenger remains in the system.")
            continue
        if passenger_id in onboard_location:
            if passenger.boarding_time is None:
                raise SimulationError("An onboard passenger has no boarding time.")
            if passenger.assigned_elevator != onboard_location[passenger_id]:
                raise SimulationError("Onboard assignment does not match elevator.")
            continue
        if passenger_id in waiting_location:
            if passenger.boarding_time is not None:
                raise SimulationError("A waiting passenger already has a boarding time.")
            if passenger.assigned_elevator != waiting_location[passenger_id]:
                raise SimulationError("Waiting assignment does not match elevator.")
            continue
        if passenger_id in unassigned:
            if passenger.assigned_elevator is not None:
                raise SimulationError("An unassigned passenger has an elevator ID.")
            continue
        raise SimulationError("An arrived passenger was lost from the system.")


def simulate_day(
    day: DayData,
    config: BuildingConfig,
    policy: AssignmentPolicy | None = None,
    positioning_policy: PositioningPolicy | None = None,
    *,
    validate_invariants: bool = False,
) -> SimulationResult:
    """Simulate one passenger day and fail loudly if anyone remains unserved."""
    active_policy = policy or NearestCarPolicy()
    result_policy_name = (
        positioning_policy.name if positioning_policy is not None else active_policy.name
    )
    passengers = list(copy.deepcopy(day.passengers))
    elevators = [
        Elevator(elevator_id=elevator_id)
        for elevator_id in range(config.n_elevators)
    ]
    arrivals: dict[int, list[int]] = defaultdict(list)
    for passenger in passengers:
        arrivals[passenger.arrival_time].append(passenger.passenger_id)

    unassigned: set[int] = set()
    generation_end = day.duration_minutes * 60
    hard_end = generation_end + config.maximum_drain_minutes * 60
    final_timestamp = hard_end

    for timestamp in range(hard_end + 1):
        if positioning_policy is not None and timestamp < generation_end:
            positioning_policy.update(timestamp, elevators, config)
        unassigned.update(arrivals.get(timestamp, ()))

        for passenger_id in sorted(tuple(unassigned)):
            if _assign_passenger(
                passenger_id,
                elevators,
                passengers,
                config,
                active_policy,
            ):
                unassigned.remove(passenger_id)

        for elevator in elevators:
            _tick_elevator(
                elevator,
                timestamp,
                passengers,
                unassigned,
                config,
            )

        if validate_invariants:
            assert_system_invariants(
                timestamp,
                passengers,
                elevators,
                unassigned,
                config,
            )

        if timestamp >= generation_end and all(
            passenger.dropoff_time is not None for passenger in passengers
        ):
            final_timestamp = timestamp
            break

    unserved = [
        passenger.passenger_id
        for passenger in passengers
        if passenger.dropoff_time is None
    ]
    if unserved:
        raise SimulationError(
            f"Policy {result_policy_name!r} left {len(unserved)} passengers "
            f"unserved on {day.day_id}: {unserved[:10]}."
        )

    return SimulationResult(
        day_id=day.day_id,
        scenario=day.scenario,
        policy_name=result_policy_name,
        generation_end_timestamp=generation_end,
        final_timestamp=final_timestamp,
        passengers=tuple(passengers),
        elevators=tuple(elevators),
    )
