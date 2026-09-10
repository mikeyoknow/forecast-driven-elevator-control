"""Leakage-safe day-bounded sequences for recurrent demand forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from elevator_ml.data.features import SCENARIO_START_HOUR
from elevator_ml.simulation.entities import DayData


def sequence_feature_names(
    n_floors: int, scenario_categories: Sequence[str]
) -> tuple[str, ...]:
    if n_floors < 2:
        raise ValueError("Sequence features require at least two floors.")
    categories = tuple(scenario_categories)
    if len(set(categories)) != len(categories):
        raise ValueError("Scenario categories must be unique.")
    names = [f"origin_floor_{floor}" for floor in range(n_floors)]
    names.extend(f"destination_floor_{floor}" for floor in range(n_floors))
    names.extend(("time_sin", "time_cos", "elapsed_fraction"))
    names.extend(f"scenario_{scenario}" for scenario in categories)
    return tuple(names)


def _minute_vector(
    day: DayData,
    minute: int,
    scenario_categories: Sequence[str],
) -> np.ndarray:
    if day.scenario not in SCENARIO_START_HOUR:
        raise ValueError(f"Unknown scenario start hour for {day.scenario!r}.")
    hour = SCENARIO_START_HOUR[day.scenario] + minute / 60.0
    angle = 2.0 * math.pi * hour / 24.0
    time_features = np.asarray(
        [
            math.sin(angle),
            math.cos(angle),
            minute / max(day.duration_minutes - 1, 1),
        ],
        dtype=float,
    )
    scenario_features = np.asarray(
        [float(day.scenario == value) for value in scenario_categories],
        dtype=float,
    )
    return np.concatenate(
        [
            day.origin_counts[minute],
            day.destination_counts[minute],
            time_features,
            scenario_features,
        ]
    )


@dataclass(frozen=True)
class SequenceDataset:
    inputs: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    sequence_length: int
    horizon_minutes: int

    def __post_init__(self) -> None:
        if self.inputs.ndim != 3 or self.targets.ndim != 2:
            raise ValueError("Sequence inputs must be 3D and targets must be 2D.")
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError("Sequence and target sample counts must match.")
        if self.inputs.shape[0] != len(self.metadata):
            raise ValueError("Sequence metadata count must match numeric samples.")
        if self.inputs.shape[1] != self.sequence_length:
            raise ValueError("Sequence tensor width differs from sequence_length.")
        if self.inputs.shape[2] != len(self.feature_names):
            raise ValueError("Sequence feature names do not match tensor width.")
        if self.targets.shape[1] != len(self.target_names):
            raise ValueError("Sequence target names do not match target width.")
        if self.sequence_length <= 0 or self.horizon_minutes <= 0:
            raise ValueError("Sequence length and forecast horizon must be positive.")
        if not np.all(np.isfinite(self.inputs)):
            raise ValueError("Sequence inputs must be finite.")
        if not np.all(np.isfinite(self.targets)) or np.any(self.targets < 0):
            raise ValueError("Sequence targets must be finite and non-negative.")

    @property
    def n_samples(self) -> int:
        return self.inputs.shape[0]

    @property
    def n_features(self) -> int:
        return self.inputs.shape[2]

    @property
    def n_targets(self) -> int:
        return self.targets.shape[1]


def build_sequence_dataset(
    days: Sequence[DayData],
    sequence_length: int,
    horizon_minutes: int,
    scenario_categories: Sequence[str],
    minimum_target_minute: int | None = None,
) -> SequenceDataset:
    """Create sequences ending strictly before each forecast target."""
    if not days:
        raise ValueError("At least one day is required.")
    if sequence_length <= 0 or horizon_minutes <= 0:
        raise ValueError("Sequence length and forecast horizon must be positive.")
    if minimum_target_minute is not None and minimum_target_minute < 0:
        raise ValueError("Minimum target minute cannot be negative.")
    n_floors = days[0].origin_counts.shape[1]
    if any(day.origin_counts.shape[1] != n_floors for day in days):
        raise ValueError("Every sequence day must have the same floor count.")
    names = sequence_feature_names(n_floors, scenario_categories)
    target_names = tuple(
        f"origin_floor_{floor}_next_{horizon_minutes}m"
        for floor in range(n_floors)
    )
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, str | int]] = []

    for day in days:
        if day.duration_minutes < sequence_length + horizon_minutes:
            raise ValueError(f"Day {day.day_id} is too short for this sequence.")
        first_target_minute = max(
            sequence_length,
            sequence_length
            if minimum_target_minute is None
            else minimum_target_minute,
        )
        for target_minute in range(
            first_target_minute,
            day.duration_minutes - horizon_minutes + 1,
        ):
            inputs.append(
                np.vstack(
                    [
                        _minute_vector(day, minute, scenario_categories)
                        for minute in range(
                            target_minute - sequence_length, target_minute
                        )
                    ]
                )
            )
            targets.append(
                day.origin_counts[
                    target_minute : target_minute + horizon_minutes
                ].sum(axis=0)
            )
            metadata.append(
                {
                    "sample_id": (
                        f"{day.day_id}-m{target_minute:03d}-h{horizon_minutes}"
                    ),
                    "day_id": day.day_id,
                    "split": day.split,
                    "scenario": day.scenario,
                    "seed": day.seed,
                    "input_start_minute": target_minute - sequence_length,
                    "input_end_minute_exclusive": target_minute,
                    "target_start_minute": target_minute,
                    "target_end_minute_exclusive": (
                        target_minute + horizon_minutes
                    ),
                    "horizon_minutes": horizon_minutes,
                }
            )

    if not inputs:
        raise ValueError("The requested sequence produced no samples.")
    return SequenceDataset(
        inputs=np.stack(inputs),
        targets=np.vstack(targets),
        metadata=pd.DataFrame(metadata),
        feature_names=names,
        target_names=target_names,
        sequence_length=sequence_length,
        horizon_minutes=horizon_minutes,
    )
