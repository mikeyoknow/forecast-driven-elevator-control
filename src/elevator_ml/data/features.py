"""Leakage-safe temporal features for per-floor demand forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from elevator_ml.simulation.entities import DayData


SCENARIO_START_HOUR = {
    "morning": 8.0,
    "lunch": 12.0,
    "mixed": 14.0,
    "surge": 15.0,
    "evening": 17.0,
}


def _validate_windows(history_windows: Sequence[int]) -> tuple[int, ...]:
    windows = tuple(int(value) for value in history_windows)
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("History windows must contain positive integers.")
    if tuple(sorted(set(windows))) != windows:
        raise ValueError("History windows must be unique and increasing.")
    return windows


def feature_names(
    n_floors: int,
    history_windows: Sequence[int],
    scenario_categories: Sequence[str],
    destination_window_minutes: int = 3,
) -> tuple[str, ...]:
    windows = _validate_windows(history_windows)
    if n_floors < 2:
        raise ValueError("Feature construction requires at least two floors.")
    if destination_window_minutes <= 0:
        raise ValueError("Destination history window must be positive.")
    categories = tuple(scenario_categories)
    if len(set(categories)) != len(categories):
        raise ValueError("Scenario feature categories must be unique.")

    names = ["time_sin", "time_cos", "elapsed_fraction"]
    names.extend(f"total_origin_last_{window}m" for window in windows)
    names.extend(f"scenario_{scenario}" for scenario in categories)
    for window in windows:
        names.extend(
            f"origin_floor_{floor}_last_{window}m" for floor in range(n_floors)
        )
    names.extend(
        f"destination_floor_{floor}_last_{destination_window_minutes}m"
        for floor in range(n_floors)
    )
    names.extend(
        f"origin_floor_{floor}_recent_trend" for floor in range(n_floors)
    )
    return tuple(names)


def _history_sum(values: np.ndarray, minute: int, width: int) -> np.ndarray:
    start = max(0, minute - width)
    if start == minute:
        return np.zeros(values.shape[1], dtype=float)
    # The slice ends strictly before `minute`; the target starts at `minute`.
    return values[start:minute].sum(axis=0)


def make_feature_vector(
    day: DayData,
    minute: int,
    history_windows: Sequence[int],
    scenario_categories: Sequence[str],
    destination_window_minutes: int = 3,
) -> np.ndarray:
    """Build features using only observations strictly before `minute`."""
    windows = _validate_windows(history_windows)
    if not 0 <= minute <= day.duration_minutes:
        raise ValueError("Feature minute is outside the day.")
    if day.scenario not in SCENARIO_START_HOUR:
        raise ValueError(f"Unknown scenario start hour for {day.scenario!r}.")
    if destination_window_minutes <= 0:
        raise ValueError("Destination history window must be positive.")

    origin_histories = [
        _history_sum(day.origin_counts, minute, window) for window in windows
    ]
    destination_history = _history_sum(
        day.destination_counts, minute, destination_window_minutes
    )
    recent_one = _history_sum(day.origin_counts, minute, 1)
    prior_two_start = max(0, minute - 3)
    prior_two_end = max(0, minute - 1)
    prior_two = day.origin_counts[prior_two_start:prior_two_end].sum(axis=0)
    prior_width = prior_two_end - prior_two_start
    prior_rate = prior_two / prior_width if prior_width else np.zeros_like(recent_one)
    trend = recent_one - prior_rate

    hour = SCENARIO_START_HOUR[day.scenario] + minute / 60.0
    angle = 2.0 * math.pi * hour / 24.0
    elapsed_fraction = minute / max(day.duration_minutes - 1, 1)
    scalars = np.asarray(
        [math.sin(angle), math.cos(angle), elapsed_fraction], dtype=float
    )
    totals = np.asarray(
        [history.sum() for history in origin_histories], dtype=float
    )
    scenario_one_hot = np.asarray(
        [float(day.scenario == scenario) for scenario in scenario_categories],
        dtype=float,
    )
    vector = np.concatenate(
        [
            scalars,
            totals,
            scenario_one_hot,
            *origin_histories,
            destination_history,
            trend,
        ]
    )
    expected = feature_names(
        day.origin_counts.shape[1],
        windows,
        scenario_categories,
        destination_window_minutes,
    )
    if vector.shape != (len(expected),):
        raise RuntimeError("Feature vector and feature-name schema disagree.")
    return vector


@dataclass(frozen=True)
class ForecastDataset:
    features: np.ndarray
    targets: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    horizon_minutes: int

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.targets.ndim != 2:
            raise ValueError("Forecast features and targets must be 2D arrays.")
        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError("Feature and target row counts must match.")
        if self.features.shape[0] != len(self.metadata):
            raise ValueError("Metadata row count must match the numeric arrays.")
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError("Feature names do not match feature columns.")
        if self.targets.shape[1] != len(self.target_names):
            raise ValueError("Target names do not match target columns.")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("Forecast features must be finite.")
        if not np.all(np.isfinite(self.targets)) or np.any(self.targets < 0):
            raise ValueError("Forecast targets must be finite and non-negative.")
        if self.horizon_minutes <= 0:
            raise ValueError("Forecast horizon must be positive.")

    @property
    def n_samples(self) -> int:
        return self.features.shape[0]

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    @property
    def n_targets(self) -> int:
        return self.targets.shape[1]


def build_forecasting_dataset(
    days: Sequence[DayData],
    history_windows: Sequence[int],
    horizon_minutes: int,
    scenario_categories: Sequence[str],
    destination_window_minutes: int = 3,
) -> ForecastDataset:
    """Create one leakage-auditable forecasting table for a fixed horizon."""
    windows = _validate_windows(history_windows)
    if not days:
        raise ValueError("At least one day is required.")
    if horizon_minutes <= 0:
        raise ValueError("Forecast horizon must be positive.")
    n_floors = days[0].origin_counts.shape[1]
    if any(day.origin_counts.shape[1] != n_floors for day in days):
        raise ValueError("All days in one forecasting dataset need equal floor counts.")
    names = feature_names(
        n_floors,
        windows,
        scenario_categories,
        destination_window_minutes,
    )
    target_names = tuple(
        f"origin_floor_{floor}_next_{horizon_minutes}m"
        for floor in range(n_floors)
    )
    max_history = max(windows)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[dict[str, str | int]] = []

    for day in days:
        if day.duration_minutes < max_history + horizon_minutes:
            raise ValueError(
                f"Day {day.day_id} is too short for the requested windows."
            )
        for minute in range(
            max_history, day.duration_minutes - horizon_minutes + 1
        ):
            rows.append(
                make_feature_vector(
                    day,
                    minute,
                    windows,
                    scenario_categories,
                    destination_window_minutes,
                )
            )
            targets.append(
                day.origin_counts[minute : minute + horizon_minutes].sum(axis=0)
            )
            metadata.append(
                {
                    "sample_id": f"{day.day_id}-m{minute:03d}-h{horizon_minutes}",
                    "day_id": day.day_id,
                    "split": day.split,
                    "scenario": day.scenario,
                    "seed": day.seed,
                    "input_start_minute": minute - max_history,
                    "input_end_minute_exclusive": minute,
                    "target_start_minute": minute,
                    "target_end_minute_exclusive": minute + horizon_minutes,
                    "horizon_minutes": horizon_minutes,
                }
            )

    if not rows:
        raise ValueError("The requested windows produced no forecasting samples.")
    return ForecastDataset(
        features=np.vstack(rows),
        targets=np.vstack(targets),
        metadata=pd.DataFrame(metadata),
        feature_names=names,
        target_names=target_names,
        horizon_minutes=horizon_minutes,
    )


def build_inference_dataset(
    day: DayData,
    history_windows: Sequence[int],
    horizon_minutes: int,
    scenario_categories: Sequence[str],
    destination_window_minutes: int = 3,
) -> ForecastDataset:
    """Build one online feature row for every minute of a simulation day.

    Targets are retained only for diagnostics and are truncated at the end of
    the day. Models never access them through their prediction interfaces.
    """
    windows = _validate_windows(history_windows)
    if horizon_minutes <= 0:
        raise ValueError("Forecast horizon must be positive.")
    n_floors = day.origin_counts.shape[1]
    names = feature_names(
        n_floors,
        windows,
        scenario_categories,
        destination_window_minutes,
    )
    target_names = tuple(
        f"origin_floor_{floor}_next_{horizon_minutes}m"
        for floor in range(n_floors)
    )
    features = []
    targets = []
    metadata = []
    max_history = max(windows)
    for minute in range(day.duration_minutes):
        features.append(
            make_feature_vector(
                day,
                minute,
                windows,
                scenario_categories,
                destination_window_minutes,
            )
        )
        targets.append(
            day.origin_counts[
                minute : min(day.duration_minutes, minute + horizon_minutes)
            ].sum(axis=0)
        )
        metadata.append(
            {
                "sample_id": f"{day.day_id}-m{minute:03d}-h{horizon_minutes}",
                "day_id": day.day_id,
                "split": day.split,
                "scenario": day.scenario,
                "seed": day.seed,
                "input_start_minute": max(0, minute - max_history),
                "input_end_minute_exclusive": minute,
                "target_start_minute": minute,
                "target_end_minute_exclusive": min(
                    day.duration_minutes, minute + horizon_minutes
                ),
                "horizon_minutes": horizon_minutes,
            }
        )
    return ForecastDataset(
        features=np.vstack(features),
        targets=np.vstack(targets),
        metadata=pd.DataFrame(metadata),
        feature_names=names,
        target_names=target_names,
        horizon_minutes=horizon_minutes,
    )
