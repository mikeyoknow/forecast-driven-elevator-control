"""Typed configuration loading and validation.

Keeping configuration outside the experiment code makes every run auditable and
prevents important assumptions from being silently changed between policies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when an experiment configuration is internally inconsistent."""


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive; received {value!r}.")


@dataclass(frozen=True)
class BuildingConfig:
    n_floors: int
    n_elevators: int
    capacity: int
    seconds_per_floor: int
    door_seconds: int
    maximum_drain_minutes: int

    def __post_init__(self) -> None:
        if self.n_floors < 2:
            raise ConfigError("building.n_floors must be at least 2.")
        for name in (
            "n_elevators",
            "capacity",
            "seconds_per_floor",
            "door_seconds",
            "maximum_drain_minutes",
        ):
            _require_positive(f"building.{name}", getattr(self, name))


@dataclass(frozen=True)
class TrafficConfig:
    duration_minutes: int
    train_scenarios: tuple[str, ...]
    stress_scenarios: tuple[str, ...]
    train_days_per_scenario: int
    validation_days_per_scenario: int
    development_test_days_per_scenario: int
    stress_days_per_scenario: int

    def __post_init__(self) -> None:
        for name in (
            "duration_minutes",
            "train_days_per_scenario",
            "validation_days_per_scenario",
            "development_test_days_per_scenario",
            "stress_days_per_scenario",
        ):
            _require_positive(f"traffic.{name}", getattr(self, name))
        if not self.train_scenarios:
            raise ConfigError("traffic.train_scenarios cannot be empty.")
        if not self.stress_scenarios:
            raise ConfigError("traffic.stress_scenarios cannot be empty.")
        all_scenarios = self.train_scenarios + self.stress_scenarios
        if any(not scenario.strip() for scenario in all_scenarios):
            raise ConfigError("Traffic scenario names cannot be blank.")
        if len(set(all_scenarios)) != len(all_scenarios):
            raise ConfigError(
                "Training and stress scenario names must be unique and disjoint."
            )


@dataclass(frozen=True)
class ForecastConfig:
    history_windows_minutes: tuple[int, ...]
    target_horizons_minutes: tuple[int, ...]
    primary_horizon_minutes: int

    def __post_init__(self) -> None:
        if not self.history_windows_minutes:
            raise ConfigError("forecast.history_windows_minutes cannot be empty.")
        if not self.target_horizons_minutes:
            raise ConfigError("forecast.target_horizons_minutes cannot be empty.")
        if any(value <= 0 for value in self.history_windows_minutes):
            raise ConfigError("Forecast history windows must be positive.")
        if any(value <= 0 for value in self.target_horizons_minutes):
            raise ConfigError("Forecast target horizons must be positive.")
        if tuple(sorted(set(self.history_windows_minutes))) != self.history_windows_minutes:
            raise ConfigError("Forecast history windows must be unique and increasing.")
        if tuple(sorted(set(self.target_horizons_minutes))) != self.target_horizons_minutes:
            raise ConfigError("Forecast target horizons must be unique and increasing.")
        if self.primary_horizon_minutes not in self.target_horizons_minutes:
            raise ConfigError(
                "forecast.primary_horizon_minutes must be one of the target horizons."
            )

    @property
    def maximum_history_minutes(self) -> int:
        return max(self.history_windows_minutes)

    @property
    def maximum_horizon_minutes(self) -> int:
        return max(self.target_horizons_minutes)


@dataclass(frozen=True)
class EvaluationConfig:
    long_wait_seconds: int
    bootstrap_resamples: int
    positioning_distance_penalties: tuple[float, ...]
    positioning_minimum_advantages: tuple[float, ...]
    positioning_intervals_minutes: tuple[int, ...]
    movement_budget_percent: float
    high_demand_multiplier: float

    def __post_init__(self) -> None:
        _require_positive("evaluation.long_wait_seconds", self.long_wait_seconds)
        _require_positive("evaluation.bootstrap_resamples", self.bootstrap_resamples)
        if not self.positioning_distance_penalties:
            raise ConfigError(
                "evaluation.positioning_distance_penalties cannot be empty."
            )
        if any(value < 0 for value in self.positioning_distance_penalties):
            raise ConfigError("Positioning distance penalties cannot be negative.")
        if not self.positioning_minimum_advantages or any(
            value < 0 for value in self.positioning_minimum_advantages
        ):
            raise ConfigError(
                "Positioning minimum advantages must be non-negative and nonempty."
            )
        if not self.positioning_intervals_minutes or any(
            value <= 0 for value in self.positioning_intervals_minutes
        ):
            raise ConfigError(
                "Positioning intervals must be positive and nonempty."
            )
        _require_positive(
            "evaluation.movement_budget_percent", self.movement_budget_percent
        )
        if self.high_demand_multiplier <= 1.0:
            raise ConfigError("High-demand multiplier must be greater than one.")


@dataclass(frozen=True)
class ModelConfig:
    historical_bucket_minutes: int
    ridge_alphas: tuple[float, ...]
    random_forest_n_estimators: int
    random_forest_max_depths: tuple[int, ...]
    random_forest_min_samples_leaf: tuple[int, ...]
    random_state: int

    def __post_init__(self) -> None:
        _require_positive(
            "models.historical_bucket_minutes", self.historical_bucket_minutes
        )
        _require_positive(
            "models.random_forest_n_estimators", self.random_forest_n_estimators
        )
        if not self.ridge_alphas or any(value <= 0 for value in self.ridge_alphas):
            raise ConfigError("Ridge alpha candidates must be positive and nonempty.")
        if not self.random_forest_max_depths or any(
            value <= 0 for value in self.random_forest_max_depths
        ):
            raise ConfigError("Random Forest max-depth candidates must be positive.")
        if not self.random_forest_min_samples_leaf or any(
            value <= 0 for value in self.random_forest_min_samples_leaf
        ):
            raise ConfigError(
                "Random Forest min-samples-leaf candidates must be positive."
            )
        if self.random_state < 0:
            raise ConfigError("models.random_state cannot be negative.")


@dataclass(frozen=True)
class SequenceModelConfig:
    sequence_lengths: tuple[int, ...]
    hidden_sizes: tuple[int, ...]
    learning_rate: float
    batch_size: int
    maximum_epochs: int
    early_stopping_patience: int
    gradient_clip_norm: float
    l2_penalty: float

    def __post_init__(self) -> None:
        if not self.sequence_lengths or any(
            value <= 0 for value in self.sequence_lengths
        ):
            raise ConfigError("Sequence lengths must be positive and nonempty.")
        if tuple(sorted(set(self.sequence_lengths))) != self.sequence_lengths:
            raise ConfigError("Sequence lengths must be unique and increasing.")
        if not self.hidden_sizes or any(value <= 0 for value in self.hidden_sizes):
            raise ConfigError("RNN hidden sizes must be positive and nonempty.")
        if tuple(sorted(set(self.hidden_sizes))) != self.hidden_sizes:
            raise ConfigError("RNN hidden sizes must be unique and increasing.")
        for name in (
            "learning_rate",
            "batch_size",
            "maximum_epochs",
            "early_stopping_patience",
            "gradient_clip_norm",
        ):
            _require_positive(f"sequence_models.{name}", getattr(self, name))
        if self.l2_penalty < 0:
            raise ConfigError("sequence_models.l2_penalty cannot be negative.")


@dataclass(frozen=True)
class SeedConfig:
    train: int
    validation: int
    development_test: int
    stress: int

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.development_test, self.stress)
        if any(value < 0 for value in values):
            raise ConfigError("Seed offsets cannot be negative.")
        if len(set(values)) != len(values):
            raise ConfigError("Every development split must use a distinct seed offset.")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    building: BuildingConfig
    traffic: TrafficConfig
    forecast: ForecastConfig
    evaluation: EvaluationConfig
    models: ModelConfig
    sequence_models: SequenceModelConfig
    seeds: SeedConfig

    def __post_init__(self) -> None:
        if not self.experiment_name.strip():
            raise ConfigError("experiment_name cannot be blank.")
        minimum_duration = (
            self.forecast.maximum_history_minutes
            + self.forecast.maximum_horizon_minutes
        )
        if self.traffic.duration_minutes <= minimum_duration:
            raise ConfigError(
                "traffic.duration_minutes must exceed history_minutes + "
                "horizon_minutes."
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        required = {
            "experiment_name",
            "building",
            "traffic",
            "forecast",
            "evaluation",
            "models",
            "sequence_models",
            "seeds",
        }
        missing = required - set(raw)
        unknown = set(raw) - required
        if missing:
            raise ConfigError(f"Missing top-level configuration keys: {sorted(missing)}")
        if unknown:
            raise ConfigError(f"Unknown top-level configuration keys: {sorted(unknown)}")

        traffic_raw = dict(raw["traffic"])
        traffic_raw["train_scenarios"] = tuple(traffic_raw["train_scenarios"])
        traffic_raw["stress_scenarios"] = tuple(traffic_raw["stress_scenarios"])
        forecast_raw = dict(raw["forecast"])
        forecast_raw["history_windows_minutes"] = tuple(
            int(value) for value in forecast_raw["history_windows_minutes"]
        )
        forecast_raw["target_horizons_minutes"] = tuple(
            int(value) for value in forecast_raw["target_horizons_minutes"]
        )
        evaluation_raw = dict(raw["evaluation"])
        evaluation_raw["positioning_distance_penalties"] = tuple(
            float(value)
            for value in evaluation_raw["positioning_distance_penalties"]
        )
        evaluation_raw["positioning_minimum_advantages"] = tuple(
            float(value)
            for value in evaluation_raw["positioning_minimum_advantages"]
        )
        evaluation_raw["positioning_intervals_minutes"] = tuple(
            int(value)
            for value in evaluation_raw["positioning_intervals_minutes"]
        )
        models_raw = dict(raw["models"])
        models_raw["ridge_alphas"] = tuple(
            float(value) for value in models_raw["ridge_alphas"]
        )
        models_raw["random_forest_max_depths"] = tuple(
            int(value) for value in models_raw["random_forest_max_depths"]
        )
        models_raw["random_forest_min_samples_leaf"] = tuple(
            int(value) for value in models_raw["random_forest_min_samples_leaf"]
        )
        sequence_raw = dict(raw["sequence_models"])
        sequence_raw["sequence_lengths"] = tuple(
            int(value) for value in sequence_raw["sequence_lengths"]
        )
        sequence_raw["hidden_sizes"] = tuple(
            int(value) for value in sequence_raw["hidden_sizes"]
        )
        return cls(
            experiment_name=str(raw["experiment_name"]),
            building=BuildingConfig(**raw["building"]),
            traffic=TrafficConfig(**traffic_raw),
            forecast=ForecastConfig(**forecast_raw),
            evaluation=EvaluationConfig(**evaluation_raw),
            models=ModelConfig(**models_raw),
            sequence_models=SequenceModelConfig(**sequence_raw),
            seeds=SeedConfig(**raw["seeds"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot suitable for a run manifest."""
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON in {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("The configuration root must be a JSON object.")
    return ExperimentConfig.from_mapping(raw)
