"""Simple demand forecasters that learned models must beat."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from elevator_ml.data.features import ForecastDataset


class DemandForecaster(Protocol):
    name: str

    def fit(self, dataset: ForecastDataset) -> "DemandForecaster": ...

    def predict(self, dataset: ForecastDataset) -> np.ndarray: ...


class PerFloorMeanForecaster:
    """Predict each floor's training-target mean for every sample."""

    name = "per_floor_mean"

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, dataset: ForecastDataset) -> "PerFloorMeanForecaster":
        self.mean_ = dataset.targets.mean(axis=0)
        return self

    def predict(self, dataset: ForecastDataset) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("PerFloorMeanForecaster must be fitted first.")
        if dataset.n_targets != len(self.mean_):
            raise ValueError("Target width differs from fitted training data.")
        return np.tile(self.mean_, (dataset.n_samples, 1))


class HistoricalMeanForecaster:
    """Training mean by known traffic regime and session-time bucket."""

    name = "historical_mean"

    def __init__(self, bucket_minutes: int = 10) -> None:
        if bucket_minutes <= 0:
            raise ValueError("Historical bucket size must be positive.")
        self.bucket_minutes = bucket_minutes
        self.by_scenario_bucket_: dict[tuple[str, int], np.ndarray] = {}
        self.by_scenario_: dict[str, np.ndarray] = {}
        self.by_bucket_: dict[int, np.ndarray] = {}
        self.global_mean_: np.ndarray | None = None

    @staticmethod
    def _bucket(dataset: ForecastDataset) -> np.ndarray:
        return (
            dataset.metadata["target_start_minute"].to_numpy(dtype=int)
            // 10
        )

    def fit(self, dataset: ForecastDataset) -> "HistoricalMeanForecaster":
        metadata = dataset.metadata.copy()
        metadata["row_index"] = np.arange(dataset.n_samples)
        metadata["time_bucket"] = (
            metadata["target_start_minute"] // self.bucket_minutes
        )
        for key, group in metadata.groupby(["scenario", "time_bucket"]):
            scenario, bucket = key
            indices = group["row_index"].to_numpy(dtype=int)
            self.by_scenario_bucket_[(str(scenario), int(bucket))] = (
                dataset.targets[indices].mean(axis=0)
            )
        for scenario, group in metadata.groupby("scenario"):
            indices = group["row_index"].to_numpy(dtype=int)
            self.by_scenario_[str(scenario)] = dataset.targets[indices].mean(axis=0)
        for bucket, group in metadata.groupby("time_bucket"):
            indices = group["row_index"].to_numpy(dtype=int)
            self.by_bucket_[int(bucket)] = dataset.targets[indices].mean(axis=0)
        self.global_mean_ = dataset.targets.mean(axis=0)
        return self

    def predict(self, dataset: ForecastDataset) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("HistoricalMeanForecaster must be fitted first.")
        predictions = []
        for row in dataset.metadata.itertuples(index=False):
            scenario = str(row.scenario)
            bucket = int(row.target_start_minute) // self.bucket_minutes
            prediction = self.by_scenario_bucket_.get((scenario, bucket))
            if prediction is None:
                prediction = self.by_scenario_.get(scenario)
            if prediction is None:
                prediction = self.by_bucket_.get(bucket)
            if prediction is None:
                prediction = self.global_mean_
            predictions.append(prediction)
        return np.vstack(predictions)


class PersistenceForecaster:
    """Assume the last minute's per-floor arrival rate persists."""

    name = "persistence"

    def fit(self, dataset: ForecastDataset) -> "PersistenceForecaster":
        required = {
            f"origin_floor_{floor}_last_1m" for floor in range(dataset.n_targets)
        }
        if not required <= set(dataset.feature_names):
            raise ValueError("Persistence requires one-minute per-floor histories.")
        return self

    def predict(self, dataset: ForecastDataset) -> np.ndarray:
        indices = [
            dataset.feature_names.index(f"origin_floor_{floor}_last_1m")
            for floor in range(dataset.n_targets)
        ]
        return np.clip(
            dataset.features[:, indices] * dataset.horizon_minutes,
            0.0,
            None,
        )
