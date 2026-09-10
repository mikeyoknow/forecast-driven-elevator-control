"""Deterministic synthetic office-building passenger traffic.

The generator creates passenger-level origin-destination events from explicit
traffic regimes. Synthetic data lets every dispatch policy receive the exact
same counterfactual passenger stream and gives us known stress scenarios. It
does not replace external validation against real building traffic.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from elevator_ml.simulation.entities import DayData, Passenger


SUPPORTED_SCENARIOS = ("morning", "lunch", "evening", "mixed", "surge")


def _validate_generation_inputs(
    scenario: str, seed: int, duration_minutes: int, n_floors: int
) -> None:
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unsupported traffic scenario {scenario!r}; "
            f"choose one of {SUPPORTED_SCENARIOS}."
        )
    if seed < 0:
        raise ValueError("Passenger-generation seed cannot be negative.")
    if duration_minutes <= 0:
        raise ValueError("Traffic duration must be positive.")
    if n_floors < 2:
        raise ValueError("Traffic generation requires at least two floors.")


def _sample_upper_floor(rng: np.random.Generator, weights: np.ndarray) -> int:
    return int(rng.choice(np.arange(1, len(weights) + 1), p=weights))


def _sample_interfloor_pair(
    rng: np.random.Generator, weights: np.ndarray
) -> tuple[int, int]:
    if len(weights) == 1:
        # A two-floor building has only one non-lobby floor.
        return (0, 1) if rng.random() < 0.5 else (1, 0)
    origin = _sample_upper_floor(rng, weights)
    destination = origin
    while destination == origin:
        destination = _sample_upper_floor(rng, weights)
    return origin, destination


def _sample_od_pair(
    scenario: str,
    minute_fraction: float,
    rng: np.random.Generator,
    floor_weights: np.ndarray,
    meeting_floor: int,
) -> tuple[int, int]:
    draw = float(rng.random())

    if scenario == "morning":
        if draw < 0.84:
            return 0, _sample_upper_floor(rng, floor_weights)
        if draw < 0.94:
            return _sample_upper_floor(rng, floor_weights), 0
        return _sample_interfloor_pair(rng, floor_weights)

    if scenario == "evening":
        if draw < 0.80:
            return _sample_upper_floor(rng, floor_weights), 0
        if draw < 0.90:
            return 0, _sample_upper_floor(rng, floor_weights)
        return _sample_interfloor_pair(rng, floor_weights)

    if scenario == "lunch":
        if minute_fraction < 0.48:
            if draw < 0.68:
                return _sample_upper_floor(rng, floor_weights), 0
            if draw < 0.82:
                return 0, _sample_upper_floor(rng, floor_weights)
        else:
            if draw < 0.68:
                return 0, _sample_upper_floor(rng, floor_weights)
            if draw < 0.82:
                return _sample_upper_floor(rng, floor_weights), 0
        return _sample_interfloor_pair(rng, floor_weights)

    if scenario == "surge":
        if 0.34 <= minute_fraction < 0.50 and draw < 0.74:
            return 0, meeting_floor
        if 0.56 <= minute_fraction < 0.72 and draw < 0.74:
            return meeting_floor, 0

    # Mixed traffic and the non-event portions of the surge scenario.
    if draw < 0.30:
        return 0, _sample_upper_floor(rng, floor_weights)
    if draw < 0.60:
        return _sample_upper_floor(rng, floor_weights), 0
    return _sample_interfloor_pair(rng, floor_weights)


def _base_arrival_rate(scenario: str, minute_fraction: float) -> float:
    if scenario == "morning":
        return 7.0 + 11.0 * math.exp(-((minute_fraction - 0.38) / 0.22) ** 2)
    if scenario == "lunch":
        return 6.0 + 9.0 * math.exp(-((minute_fraction - 0.50) / 0.25) ** 2)
    if scenario == "evening":
        return 7.0 + 11.0 * math.exp(-((minute_fraction - 0.62) / 0.22) ** 2)
    if scenario == "surge":
        pulse = 13.0 if 0.34 <= minute_fraction < 0.72 else 0.0
        return 7.0 + pulse
    return 7.0 + 3.0 * (
        0.5 + 0.5 * math.sin(2.0 * math.pi * minute_fraction)
    )


def generate_passengers(
    scenario: str,
    seed: int,
    duration_minutes: int,
    n_floors: int,
    demand_multiplier: float = 1.0,
) -> tuple[Passenger, ...]:
    """Generate a repeatable, arrival-time-sorted passenger stream."""
    _validate_generation_inputs(scenario, seed, duration_minutes, n_floors)
    if demand_multiplier <= 0:
        raise ValueError("demand_multiplier must be positive.")
    rng = np.random.default_rng(seed)
    floor_weights = rng.lognormal(mean=0.0, sigma=0.42, size=n_floors - 1)
    floor_weights /= floor_weights.sum()
    meeting_floor = int(rng.integers(max(1, n_floors // 2), n_floors))
    daily_scale = float(rng.uniform(0.80, 1.20))
    latent_intensity = 1.0
    passenger_rows: list[tuple[int, int, int]] = []

    for minute in range(duration_minutes):
        minute_fraction = minute / max(duration_minutes - 1, 1)
        innovation = float(
            rng.lognormal(mean=-0.5 * 0.18**2, sigma=0.18)
        )
        latent_intensity = float(
            np.clip(
                0.78 * latent_intensity + 0.22 * innovation,
                0.60,
                1.55,
            )
        )
        arrival_rate = (
            _base_arrival_rate(scenario, minute_fraction)
            * daily_scale
            * latent_intensity
            * demand_multiplier
        )
        arrivals = int(rng.poisson(arrival_rate))

        for _ in range(arrivals):
            arrival_time = int(minute * 60 + rng.integers(0, 60))
            origin, destination = _sample_od_pair(
                scenario,
                minute_fraction,
                rng,
                floor_weights,
                meeting_floor,
            )
            passenger_rows.append((arrival_time, origin, destination))

    passenger_rows.sort(key=lambda row: row[0])
    return tuple(
        Passenger(passenger_id, arrival_time, origin, destination)
        for passenger_id, (arrival_time, origin, destination) in enumerate(
            passenger_rows
        )
    )


def aggregate_minute_counts(
    passengers: Iterable[Passenger], duration_minutes: int, n_floors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert passenger events into per-minute origin and destination counts."""
    if duration_minutes <= 0 or n_floors < 2:
        raise ValueError("Aggregation requires positive duration and at least two floors.")
    origin_counts = np.zeros((duration_minutes, n_floors), dtype=float)
    destination_counts = np.zeros((duration_minutes, n_floors), dtype=float)

    for passenger in passengers:
        passenger.validate(n_floors)
        minute = passenger.arrival_time // 60
        if not 0 <= minute < duration_minutes:
            raise ValueError(
                f"Passenger {passenger.passenger_id} arrives outside the day."
            )
        origin_counts[minute, passenger.origin] += 1.0
        destination_counts[minute, passenger.destination] += 1.0
    return origin_counts, destination_counts


def generate_day(
    split: str,
    scenario: str,
    repetition: int,
    seed: int,
    duration_minutes: int,
    n_floors: int,
    demand_multiplier: float = 1.0,
) -> DayData:
    if not split.strip():
        raise ValueError("Split name cannot be blank.")
    if repetition < 0:
        raise ValueError("Day repetition cannot be negative.")
    passengers = generate_passengers(
        scenario=scenario,
        seed=seed,
        duration_minutes=duration_minutes,
        n_floors=n_floors,
        demand_multiplier=demand_multiplier,
    )
    origin_counts, destination_counts = aggregate_minute_counts(
        passengers, duration_minutes, n_floors
    )
    return DayData(
        day_id=f"{split}-{scenario}-{repetition:02d}",
        split=split,
        scenario=scenario,
        seed=seed,
        duration_minutes=duration_minutes,
        passengers=passengers,
        origin_counts=origin_counts,
        destination_counts=destination_counts,
    )


def create_days(
    split: str,
    scenarios: Sequence[str],
    days_per_scenario: int,
    seed_offset: int,
    duration_minutes: int,
    n_floors: int,
    demand_multiplier: float = 1.0,
) -> tuple[DayData, ...]:
    """Generate independent days with simple, auditable non-overlapping seeds."""
    if not split.strip():
        raise ValueError("Split name cannot be blank.")
    if days_per_scenario <= 0:
        raise ValueError("days_per_scenario must be positive.")
    if seed_offset < 0:
        raise ValueError("seed_offset cannot be negative.")
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("Scenario names must be unique within a split.")

    days: list[DayData] = []
    for scenario_index, scenario in enumerate(scenarios):
        for repetition in range(days_per_scenario):
            seed = seed_offset + scenario_index * days_per_scenario + repetition
            days.append(
                generate_day(
                    split=split,
                    scenario=scenario,
                    repetition=repetition,
                    seed=seed,
                    duration_minutes=duration_minutes,
                    n_floors=n_floors,
                    demand_multiplier=demand_multiplier,
                )
            )
    return tuple(days)
