"""Forecast-level error metrics and diagnostic slices."""

from __future__ import annotations

import numpy as np
import pandas as pd

from elevator_ml.data.features import ForecastDataset


def _validate_predictions(
    dataset: ForecastDataset, predictions: np.ndarray
) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    if values.shape != dataset.targets.shape:
        raise ValueError(
            f"Prediction shape {values.shape} does not match targets "
            f"{dataset.targets.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Forecast predictions must be finite.")
    if np.any(values < 0):
        raise ValueError("Passenger-count predictions cannot be negative.")
    return values


def top_floor_overlap(
    targets: np.ndarray, predictions: np.ndarray, k: int = 3
) -> float:
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("Top-floor overlap requires equal 2D arrays.")
    if not 0 < k <= targets.shape[1]:
        raise ValueError("k must be between one and the number of floors.")
    overlaps = []
    for actual, predicted in zip(targets, predictions):
        actual_top = set(np.argsort(actual, kind="stable")[-k:])
        predicted_top = set(np.argsort(predicted, kind="stable")[-k:])
        overlaps.append(len(actual_top & predicted_top) / k)
    return float(np.mean(overlaps))


def forecast_metric_row(
    dataset: ForecastDataset,
    predictions: np.ndarray,
    *,
    model_name: str,
    evaluation: str,
) -> dict[str, str | int | float]:
    values = _validate_predictions(dataset, predictions)
    errors = values - dataset.targets
    return {
        "evaluation": evaluation,
        "horizon_minutes": dataset.horizon_minutes,
        "model": model_name,
        "samples": dataset.n_samples,
        "floor_mae": float(np.mean(np.abs(errors))),
        "floor_rmse": float(np.sqrt(np.mean(errors**2))),
        "total_demand_mae": float(
            np.mean(np.abs(values.sum(axis=1) - dataset.targets.sum(axis=1)))
        ),
        "lobby_mae": float(np.mean(np.abs(errors[:, 0]))),
        "upper_floor_mae": float(np.mean(np.abs(errors[:, 1:]))),
        "mean_bias": float(np.mean(errors)),
        "top3_floor_overlap": top_floor_overlap(dataset.targets, values, k=3),
    }


def scenario_metric_rows(
    dataset: ForecastDataset,
    predictions: np.ndarray,
    *,
    model_name: str,
    evaluation: str,
) -> list[dict[str, str | int | float]]:
    values = _validate_predictions(dataset, predictions)
    rows = []
    scenarios = dataset.metadata["scenario"].to_numpy()
    for scenario in sorted(set(scenarios)):
        indices = np.flatnonzero(scenarios == scenario)
        errors = values[indices] - dataset.targets[indices]
        rows.append(
            {
                "evaluation": evaluation,
                "horizon_minutes": dataset.horizon_minutes,
                "model": model_name,
                "scenario": str(scenario),
                "samples": len(indices),
                "floor_mae": float(np.mean(np.abs(errors))),
                "total_demand_mae": float(
                    np.mean(
                        np.abs(
                            values[indices].sum(axis=1)
                            - dataset.targets[indices].sum(axis=1)
                        )
                    )
                ),
            }
        )
    return rows


def floor_metric_rows(
    dataset: ForecastDataset,
    predictions: np.ndarray,
    *,
    model_name: str,
    evaluation: str,
) -> list[dict[str, str | int | float]]:
    values = _validate_predictions(dataset, predictions)
    rows = []
    for floor in range(dataset.n_targets):
        errors = values[:, floor] - dataset.targets[:, floor]
        rows.append(
            {
                "evaluation": evaluation,
                "horizon_minutes": dataset.horizon_minutes,
                "model": model_name,
                "floor": floor,
                "actual_mean": float(dataset.targets[:, floor].mean()),
                "predicted_mean": float(values[:, floor].mean()),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "bias": float(np.mean(errors)),
            }
        )
    return rows


def metrics_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
