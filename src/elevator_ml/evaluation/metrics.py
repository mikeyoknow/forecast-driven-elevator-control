"""Passenger-service, system-cost, fairness, and uncertainty metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from elevator_ml.config import BuildingConfig
from elevator_ml.simulation.simulator import SimulationResult


@dataclass(frozen=True)
class SimulationMetrics:
    day_id: str
    scenario: str
    policy: str
    passengers: int
    mean_wait_s: float
    median_wait_s: float
    p95_wait_s: float
    long_wait_rate: float
    mean_ride_s: float
    mean_journey_s: float
    floor_fairness_gap_s: float
    floors_travelled: int
    stops_served: int
    reversals: int
    max_elevator_load: int
    capacity_utilization_peak: float
    unserved_passengers: int
    drain_seconds: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True)
class FloorServiceMetrics:
    floor: int
    passengers: int
    mean_wait_s: float
    p95_wait_s: float
    long_wait_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    paired_days: int
    resamples: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _completed_arrays(
    result: SimulationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not result.all_served:
        raise ValueError("Metrics require a simulation in which every passenger is served.")
    if not result.passengers:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    waits = np.asarray(
        [passenger.wait_seconds for passenger in result.passengers], dtype=float
    )
    rides = np.asarray(
        [passenger.ride_seconds for passenger in result.passengers], dtype=float
    )
    journeys = np.asarray(
        [passenger.journey_seconds for passenger in result.passengers], dtype=float
    )
    if np.any(waits < 0) or np.any(rides < 0) or np.any(journeys < 0):
        raise ValueError("Passenger service times cannot be negative.")
    return waits, rides, journeys


def floor_service_metrics(
    result: SimulationResult,
    n_floors: int,
    long_wait_seconds: int,
) -> tuple[FloorServiceMetrics, ...]:
    if n_floors < 2:
        raise ValueError("n_floors must be at least two.")
    if long_wait_seconds <= 0:
        raise ValueError("long_wait_seconds must be positive.")
    waits, _, _ = _completed_arrays(result)
    by_floor: list[FloorServiceMetrics] = []
    origins = np.asarray(
        [passenger.origin for passenger in result.passengers], dtype=int
    )
    for floor in range(n_floors):
        floor_waits = waits[origins == floor]
        if floor_waits.size == 0:
            by_floor.append(
                FloorServiceMetrics(
                    floor=floor,
                    passengers=0,
                    mean_wait_s=float("nan"),
                    p95_wait_s=float("nan"),
                    long_wait_rate=float("nan"),
                )
            )
            continue
        by_floor.append(
            FloorServiceMetrics(
                floor=floor,
                passengers=int(floor_waits.size),
                mean_wait_s=float(floor_waits.mean()),
                p95_wait_s=float(np.quantile(floor_waits, 0.95)),
                long_wait_rate=float(np.mean(floor_waits > long_wait_seconds)),
            )
        )
    return tuple(by_floor)


def summarize_simulation(
    result: SimulationResult,
    building: BuildingConfig,
    long_wait_seconds: int,
) -> SimulationMetrics:
    """Create one policy-day row, the independent unit used in comparisons."""
    waits, rides, journeys = _completed_arrays(result)
    floor_rows = floor_service_metrics(
        result, building.n_floors, long_wait_seconds
    )
    observed_floor_means = np.asarray(
        [row.mean_wait_s for row in floor_rows if row.passengers > 0],
        dtype=float,
    )
    fairness_gap = (
        float(observed_floor_means.max() - observed_floor_means.min())
        if observed_floor_means.size
        else 0.0
    )
    max_load = result.max_elevator_load
    passenger_count = len(result.passengers)
    return SimulationMetrics(
        day_id=result.day_id,
        scenario=result.scenario,
        policy=result.policy_name,
        passengers=passenger_count,
        mean_wait_s=float(waits.mean()) if passenger_count else 0.0,
        median_wait_s=float(np.median(waits)) if passenger_count else 0.0,
        p95_wait_s=float(np.quantile(waits, 0.95)) if passenger_count else 0.0,
        long_wait_rate=(
            float(np.mean(waits > long_wait_seconds)) if passenger_count else 0.0
        ),
        mean_ride_s=float(rides.mean()) if passenger_count else 0.0,
        mean_journey_s=float(journeys.mean()) if passenger_count else 0.0,
        floor_fairness_gap_s=fairness_gap,
        floors_travelled=result.floors_travelled,
        stops_served=result.stops_served,
        reversals=sum(elevator.reversals for elevator in result.elevators),
        max_elevator_load=max_load,
        capacity_utilization_peak=max_load / building.capacity,
        unserved_passengers=0,
        drain_seconds=max(
            0, result.final_timestamp - result.generation_end_timestamp
        ),
    )


def paired_bootstrap_interval(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap the paired mean benefit: baseline minus candidate.

    Each input position must represent the same independent building day under
    two policies. Positive estimates therefore mean the candidate is better for
    metrics where lower is preferred, such as waiting time.
    """
    baseline = np.asarray(baseline_values, dtype=float)
    candidate = np.asarray(candidate_values, dtype=float)
    if baseline.ndim != 1 or candidate.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be one-dimensional.")
    if baseline.shape != candidate.shape:
        raise ValueError("Paired bootstrap inputs must have identical shapes.")
    if baseline.size < 2:
        raise ValueError("Paired bootstrap requires at least two independent days.")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError("Paired bootstrap inputs must be finite.")
    if resamples <= 0:
        raise ValueError("resamples must be positive.")
    if seed < 0:
        raise ValueError("Bootstrap seed cannot be negative.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one.")

    differences = baseline - candidate
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, differences.size, size=(resamples, differences.size)
    )
    bootstrap_means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        paired_days=int(differences.size),
        resamples=resamples,
    )
