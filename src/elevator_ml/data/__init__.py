"""Passenger traffic generation and feature engineering."""

from .generator import (
    SUPPORTED_SCENARIOS,
    aggregate_minute_counts,
    create_days,
    generate_day,
    generate_passengers,
)
from .features import (
    ForecastDataset,
    build_inference_dataset,
    build_forecasting_dataset,
    feature_names,
    make_feature_vector,
)

__all__ = [
    "SUPPORTED_SCENARIOS",
    "aggregate_minute_counts",
    "create_days",
    "generate_day",
    "generate_passengers",
    "ForecastDataset",
    "build_inference_dataset",
    "build_forecasting_dataset",
    "feature_names",
    "make_feature_vector",
]
