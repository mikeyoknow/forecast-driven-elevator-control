"""Learned multi-output demand-forecasting models."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from elevator_ml.data.features import ForecastDataset


class RidgeForecaster:
    name = "ridge"

    def __init__(self, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError("Ridge alpha must be positive.")
        self.alpha = float(alpha)
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self.is_fitted_ = False

    def fit(self, dataset: ForecastDataset) -> "RidgeForecaster":
        self.model.fit(dataset.features, dataset.targets)
        self.is_fitted_ = True
        return self

    def predict(self, dataset: ForecastDataset) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("RidgeForecaster must be fitted first.")
        return np.clip(self.model.predict(dataset.features), 0.0, None)


class RandomForestForecaster:
    name = "random_forest"

    def __init__(
        self,
        *,
        n_estimators: int,
        max_depth: int,
        min_samples_leaf: int,
        random_state: int,
        n_jobs: int = -1,
    ) -> None:
        if n_estimators <= 0 or max_depth <= 0 or min_samples_leaf <= 0:
            raise ValueError("Random Forest hyperparameters must be positive.")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=0.70,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.is_fitted_ = False

    def fit(self, dataset: ForecastDataset) -> "RandomForestForecaster":
        self.model.fit(dataset.features, dataset.targets)
        self.is_fitted_ = True
        return self

    def predict(self, dataset: ForecastDataset) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("RandomForestForecaster must be fitted first.")
        return np.clip(self.model.predict(dataset.features), 0.0, None)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("RandomForestForecaster must be fitted first.")
        return self.model.feature_importances_

    def predict_with_uncertainty(
        self, dataset: ForecastDataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ensemble mean and per-output tree disagreement."""
        if not self.is_fitted_:
            raise RuntimeError("RandomForestForecaster must be fitted first.")
        tree_predictions = np.stack(
            [tree.predict(dataset.features) for tree in self.model.estimators_],
            axis=0,
        )
        mean = np.clip(tree_predictions.mean(axis=0), 0.0, None)
        uncertainty = tree_predictions.std(axis=0)
        return mean, uncertainty
