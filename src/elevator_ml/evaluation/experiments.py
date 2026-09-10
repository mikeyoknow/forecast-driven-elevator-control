"""Reproducible experiment orchestration."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import dump

from elevator_ml.config import ExperimentConfig, load_config
from elevator_ml.data.audit import (
    day_manifest,
    plot_data_audit,
    scenario_summary,
)
from elevator_ml.data.features import build_forecasting_dataset
from elevator_ml.data.features import build_inference_dataset
from elevator_ml.data.generator import create_days
from elevator_ml.data.sequences import build_sequence_dataset
from elevator_ml.evaluation.forecast_metrics import (
    floor_metric_rows,
    forecast_metric_row,
    scenario_metric_rows,
)
from elevator_ml.forecasting.baselines import (
    HistoricalMeanForecaster,
    PerFloorMeanForecaster,
    PersistenceForecaster,
)
from elevator_ml.forecasting.models import (
    RandomForestForecaster,
    RidgeForecaster,
)
from elevator_ml.forecasting.rnn import ElmanRNNForecaster
from elevator_ml.control.policies import (
    ForecastPositioningPolicy,
    StayIdlePositioningPolicy,
    StaticZoningPolicy,
    UncertaintyGatedPositioningPolicy,
)
from elevator_ml.evaluation.metrics import (
    floor_service_metrics,
    paired_bootstrap_interval,
    summarize_simulation,
)
from elevator_ml.evaluation.operational_value import correlation_interval
from elevator_ml.simulation.entities import DayData
from elevator_ml.simulation.simulator import simulate_day


def create_development_days(
    config: ExperimentConfig,
) -> dict[str, tuple[DayData, ...]]:
    """Generate train and validation only; do not access evaluation splits."""
    common = {
        "scenarios": config.traffic.train_scenarios,
        "duration_minutes": config.traffic.duration_minutes,
        "n_floors": config.building.n_floors,
    }
    train = create_days(
        split="train",
        days_per_scenario=config.traffic.train_days_per_scenario,
        seed_offset=config.seeds.train,
        **common,
    )
    validation = create_days(
        split="validation",
        days_per_scenario=config.traffic.validation_days_per_scenario,
        seed_offset=config.seeds.validation,
        **common,
    )
    train_seeds = {day.seed for day in train}
    validation_seeds = {day.seed for day in validation}
    if train_seeds & validation_seeds:
        raise RuntimeError("Training and validation day seeds overlap.")
    return {"train": train, "validation": validation}


def _environment_manifest() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }


def run_data_audit(config_path: str | Path, output_dir: str | Path) -> Path:
    """Generate leakage-safe train/validation data and its first EDA package."""
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_days = create_development_days(config)
    all_days = split_days["train"] + split_days["validation"]

    manifest = day_manifest(all_days)
    summary = scenario_summary(all_days)
    manifest.to_csv(output / "day_manifest.csv", index=False)
    summary.to_csv(output / "scenario_summary.csv", index=False)
    plot_data_audit(all_days, output / "traffic_audit.png")

    dataset_rows: list[dict[str, str | int]] = []
    primary_train_dataset = None
    primary_validation_dataset = None
    for split, days in split_days.items():
        for horizon in config.forecast.target_horizons_minutes:
            dataset = build_forecasting_dataset(
                days=days,
                history_windows=config.forecast.history_windows_minutes,
                horizon_minutes=horizon,
                scenario_categories=config.traffic.train_scenarios,
            )
            dataset_rows.append(
                {
                    "split": split,
                    "horizon_minutes": horizon,
                    "samples": dataset.n_samples,
                    "features": dataset.n_features,
                    "targets": dataset.n_targets,
                    "unique_days": dataset.metadata["day_id"].nunique(),
                }
            )
            if horizon == config.forecast.primary_horizon_minutes:
                if split == "train":
                    primary_train_dataset = dataset
                else:
                    primary_validation_dataset = dataset

    pd.DataFrame(dataset_rows).to_csv(
        output / "forecast_dataset_manifest.csv", index=False
    )
    if primary_train_dataset is None or primary_validation_dataset is None:
        raise RuntimeError("Primary-horizon datasets were not created.")
    pd.DataFrame(
        {
            "column_index": np.arange(primary_train_dataset.n_features),
            "feature_name": primary_train_dataset.feature_names,
        }
    ).to_csv(output / "feature_schema.csv", index=False)
    pd.concat(
        [
            primary_train_dataset.metadata,
            primary_validation_dataset.metadata,
        ],
        ignore_index=True,
    ).to_csv(output / "window_leakage_audit.csv", index=False)

    (output / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(_environment_manifest(), indent=2), encoding="utf-8"
    )
    return output


def _primary_datasets(
    config: ExperimentConfig,
    split_days: dict[str, tuple[DayData, ...]],
):
    horizon = config.forecast.primary_horizon_minutes
    train = build_forecasting_dataset(
        split_days["train"],
        config.forecast.history_windows_minutes,
        horizon,
        config.traffic.train_scenarios,
    )
    validation = build_forecasting_dataset(
        split_days["validation"],
        config.forecast.history_windows_minutes,
        horizon,
        config.traffic.train_scenarios,
    )
    return train, validation


def _tune_learned_models(
    config: ExperimentConfig,
    train,
    validation,
) -> tuple[dict[str, int | float], pd.DataFrame]:
    rows: list[dict[str, str | int | float]] = []
    for alpha in config.models.ridge_alphas:
        model = RidgeForecaster(alpha=alpha).fit(train)
        metric = forecast_metric_row(
            validation,
            model.predict(validation),
            model_name=model.name,
            evaluation="validation",
        )
        rows.append(
            {
                "model": model.name,
                "alpha": alpha,
                "max_depth": np.nan,
                "min_samples_leaf": np.nan,
                "validation_floor_mae": metric["floor_mae"],
                "validation_total_demand_mae": metric["total_demand_mae"],
            }
        )

    for max_depth in config.models.random_forest_max_depths:
        for min_samples_leaf in config.models.random_forest_min_samples_leaf:
            model = RandomForestForecaster(
                n_estimators=config.models.random_forest_n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=config.models.random_state,
            ).fit(train)
            metric = forecast_metric_row(
                validation,
                model.predict(validation),
                model_name=model.name,
                evaluation="validation",
            )
            rows.append(
                {
                    "model": model.name,
                    "alpha": np.nan,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "validation_floor_mae": metric["floor_mae"],
                    "validation_total_demand_mae": metric["total_demand_mae"],
                }
            )

    tuning = pd.DataFrame(rows).sort_values(
        ["model", "validation_floor_mae", "validation_total_demand_mae"]
    )
    best_ridge = tuning[tuning["model"] == "ridge"].iloc[0]
    best_forest = tuning[tuning["model"] == "random_forest"].iloc[0]
    selected: dict[str, int | float] = {
        "ridge_alpha": float(best_ridge["alpha"]),
        "random_forest_max_depth": int(best_forest["max_depth"]),
        "random_forest_min_samples_leaf": int(
            best_forest["min_samples_leaf"]
        ),
    }
    return selected, tuning


def _forecaster_set(
    config: ExperimentConfig, selected: dict[str, int | float]
):
    return [
        PerFloorMeanForecaster(),
        HistoricalMeanForecaster(config.models.historical_bucket_minutes),
        PersistenceForecaster(),
        RidgeForecaster(float(selected["ridge_alpha"])),
        RandomForestForecaster(
            n_estimators=config.models.random_forest_n_estimators,
            max_depth=int(selected["random_forest_max_depth"]),
            min_samples_leaf=int(
                selected["random_forest_min_samples_leaf"]
            ),
            random_state=config.models.random_state,
        ),
    ]


def _plot_forecasting_results(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    order = [
        "per_floor_mean",
        "historical_mean",
        "persistence",
        "ridge",
        "random_forest",
    ]
    colors = ["#6b7280", "#2563eb", "#dc2626", "#7c3aed", "#059669"]
    validation = metrics[metrics["evaluation"] == "validation"]
    primary_horizon = 2
    primary = validation[
        validation["horizon_minutes"] == primary_horizon
    ].set_index("model").reindex(order)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].bar(order, primary["floor_mae"], color=colors)
    axes[0, 0].set_title("Primary 2-minute per-floor MAE")
    axes[0, 0].set_ylabel("Passengers")
    axes[0, 0].tick_params(axis="x", rotation=20)

    for model in order:
        group = validation[validation["model"] == model].sort_values(
            "horizon_minutes"
        )
        axes[0, 1].plot(
            group["horizon_minutes"],
            group["floor_mae"],
            marker="o",
            label=model,
        )
    axes[0, 1].set_title("Validation error by forecast horizon")
    axes[0, 1].set_xlabel("Horizon (minutes)")
    axes[0, 1].set_ylabel("Per-floor MAE")
    axes[0, 1].legend(fontsize=8)

    comparison = metrics[metrics["horizon_minutes"] == primary_horizon].pivot(
        index="model", columns="evaluation", values="floor_mae"
    ).reindex(order)
    comparison.plot(kind="bar", ax=axes[1, 0], color=["#94a3b8", "#2563eb"])
    axes[1, 0].set_title("Primary-horizon generalization gap")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Per-floor MAE")
    axes[1, 0].tick_params(axis="x", rotation=20)

    axes[1, 1].bar(order, primary["total_demand_mae"], color=colors)
    axes[1, 1].set_title("Primary 2-minute total-demand MAE")
    axes[1, 1].set_ylabel("Passengers")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.suptitle(
        "Development forecasting comparison (validation only)",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_forecasting_experiment(
    config_path: str | Path, output_dir: str | Path
) -> Path:
    """Tune on validation and compare all forecasters without test access."""
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_days = create_development_days(config)
    primary_train, primary_validation = _primary_datasets(config, split_days)
    selected, tuning = _tune_learned_models(
        config, primary_train, primary_validation
    )
    tuning.to_csv(output / "hyperparameter_tuning.csv", index=False)
    (output / "selected_hyperparameters.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    metric_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    floor_rows: list[dict[str, object]] = []
    primary_models = {}
    primary_predictions: dict[str, np.ndarray] = {
        "targets": primary_validation.targets
    }
    primary_feature_names = primary_train.feature_names

    for horizon in config.forecast.target_horizons_minutes:
        train = build_forecasting_dataset(
            split_days["train"],
            config.forecast.history_windows_minutes,
            horizon,
            config.traffic.train_scenarios,
        )
        validation = build_forecasting_dataset(
            split_days["validation"],
            config.forecast.history_windows_minutes,
            horizon,
            config.traffic.train_scenarios,
        )
        for model in _forecaster_set(config, selected):
            model.fit(train)
            train_predictions = model.predict(train)
            validation_predictions = model.predict(validation)
            metric_rows.append(
                forecast_metric_row(
                    train,
                    train_predictions,
                    model_name=model.name,
                    evaluation="train",
                )
            )
            metric_rows.append(
                forecast_metric_row(
                    validation,
                    validation_predictions,
                    model_name=model.name,
                    evaluation="validation",
                )
            )
            scenario_rows.extend(
                scenario_metric_rows(
                    validation,
                    validation_predictions,
                    model_name=model.name,
                    evaluation="validation",
                )
            )
            floor_rows.extend(
                floor_metric_rows(
                    validation,
                    validation_predictions,
                    model_name=model.name,
                    evaluation="validation",
                )
            )
            if horizon == config.forecast.primary_horizon_minutes:
                primary_models[model.name] = model
                primary_predictions[model.name] = validation_predictions

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["horizon_minutes", "evaluation", "floor_mae"]
    )
    metrics.to_csv(output / "forecast_metrics.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(
        output / "scenario_forecast_metrics.csv", index=False
    )
    pd.DataFrame(floor_rows).to_csv(
        output / "floor_forecast_metrics.csv", index=False
    )
    dump(primary_models, output / "primary_models.joblib")
    np.savez_compressed(output / "primary_validation_predictions.npz", **primary_predictions)

    forest = primary_models["random_forest"]
    importance = pd.DataFrame(
        {
            "feature": primary_feature_names,
            "importance": forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output / "random_forest_feature_importance.csv", index=False)
    _plot_forecasting_results(metrics, output / "forecast_comparison.png")
    (output / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(_environment_manifest(), indent=2), encoding="utf-8"
    )
    return output


def _fit_primary_controller_models(
    config: ExperimentConfig,
    split_days: dict[str, tuple[DayData, ...]],
):
    train, validation = _primary_datasets(config, split_days)
    selected, tuning = _tune_learned_models(config, train, validation)
    models = {
        model.name: model.fit(train) for model in _forecaster_set(config, selected)
    }
    return models, selected, tuning


def _day_schedules(
    day: DayData,
    config: ExperimentConfig,
    models: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    inference = build_inference_dataset(
        day,
        config.forecast.history_windows_minutes,
        config.forecast.primary_horizon_minutes,
        config.traffic.train_scenarios,
    )
    historical = models["historical_mean"].predict(inference)
    ridge = models["ridge"].predict(inference)
    forest_model = models["random_forest"]
    forest, tree_uncertainty = forest_model.predict_with_uncertainty(inference)
    cold_start = config.forecast.maximum_history_minutes
    ridge[:cold_start] = historical[:cold_start]
    forest[:cold_start] = historical[:cold_start]
    oracle = inference.targets.copy()
    uncertainty_score = tree_uncertainty.mean(axis=1)
    schedules = {
        "historical_positioning": historical,
        "ridge_positioning": ridge,
        "random_forest_positioning": forest,
        "oracle_positioning": oracle,
    }
    return schedules, uncertainty_score


def _select_positioning_penalty(
    config: ExperimentConfig,
    validation_days: tuple[DayData, ...],
    schedules: dict[str, dict[str, np.ndarray]],
) -> tuple[float, pd.DataFrame]:
    rows = []
    for penalty in config.evaluation.positioning_distance_penalties:
        metrics = []
        for day in validation_days:
            result = simulate_day(
                day,
                config.building,
                positioning_policy=ForecastPositioningPolicy(
                    schedules[day.day_id]["random_forest_positioning"],
                    distance_penalty=penalty,
                    name="random_forest_positioning",
                ),
            )
            metrics.append(
                summarize_simulation(
                    result,
                    config.building,
                    config.evaluation.long_wait_seconds,
                )
            )
        rows.append(
            {
                "distance_penalty": penalty,
                "validation_mean_wait_s": float(
                    np.mean([metric.mean_wait_s for metric in metrics])
                ),
                "validation_p95_wait_s": float(
                    np.mean([metric.p95_wait_s for metric in metrics])
                ),
                "validation_floors_travelled": float(
                    np.mean([metric.floors_travelled for metric in metrics])
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["validation_mean_wait_s", "validation_p95_wait_s", "validation_floors_travelled"]
    )
    return float(frame.iloc[0]["distance_penalty"]), frame


def _paired_controller_effects(
    results: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    baseline = results[results["policy"] == "nearest_car"].set_index("day_id")
    rows = []
    for policy in sorted(set(results["policy"]) - {"nearest_car"}):
        candidate = results[results["policy"] == policy].set_index("day_id")
        common = baseline.index.intersection(candidate.index)
        base_wait = baseline.loc[common, "mean_wait_s"].to_numpy()
        candidate_wait = candidate.loc[common, "mean_wait_s"].to_numpy()
        base_p95 = baseline.loc[common, "p95_wait_s"].to_numpy()
        candidate_p95 = candidate.loc[common, "p95_wait_s"].to_numpy()
        wait_interval = paired_bootstrap_interval(
            base_wait,
            candidate_wait,
            resamples=config.evaluation.bootstrap_resamples,
            seed=config.models.random_state,
        )
        p95_interval = paired_bootstrap_interval(
            base_p95,
            candidate_p95,
            resamples=config.evaluation.bootstrap_resamples,
            seed=config.models.random_state + 1,
        )
        movement_change = 100.0 * (
            candidate.loc[common, "floors_travelled"].to_numpy()
            / baseline.loc[common, "floors_travelled"].to_numpy()
            - 1.0
        )
        rows.append(
            {
                "policy": policy,
                "paired_days": len(common),
                "mean_wait_saved_s": wait_interval.estimate,
                "mean_wait_saved_ci_lower_s": wait_interval.lower,
                "mean_wait_saved_ci_upper_s": wait_interval.upper,
                "mean_wait_improvement_pct": float(
                    100.0 * wait_interval.estimate / base_wait.mean()
                ),
                "p95_wait_saved_s": p95_interval.estimate,
                "p95_wait_saved_ci_lower_s": p95_interval.lower,
                "p95_wait_saved_ci_upper_s": p95_interval.upper,
                "days_with_lower_mean_wait": int(
                    np.sum(candidate_wait < base_wait)
                ),
                "movement_change_pct": float(movement_change.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_wait_saved_s", ascending=False)


def _plot_controller_results(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    order = [
        "nearest_car",
        "static_zoning",
        "historical_positioning",
        "ridge_positioning",
        "random_forest_positioning",
        "uncertainty_gated_random_forest",
        "oracle_positioning",
    ]
    colors = [
        "#6b7280",
        "#64748b",
        "#2563eb",
        "#7c3aed",
        "#059669",
        "#0f766e",
        "#f59e0b",
    ]
    indexed = summary.set_index("policy").reindex(order)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for column, axis, title, ylabel in (
        ("mean_wait_s", axes[0, 0], "Mean passenger wait", "Seconds"),
        ("p95_wait_s", axes[0, 1], "P95 passenger wait", "Seconds"),
        ("floors_travelled", axes[1, 0], "Elevator movement", "Floors travelled"),
    ):
        axis.bar(order, indexed[column], color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)

    effects_indexed = effects.set_index("policy").reindex(order[1:])
    estimates = effects_indexed["mean_wait_saved_s"].to_numpy()
    lower = effects_indexed["mean_wait_saved_ci_lower_s"].to_numpy()
    upper = effects_indexed["mean_wait_saved_ci_upper_s"].to_numpy()
    axes[1, 1].bar(order[1:], estimates, color=colors[1:])
    axes[1, 1].errorbar(
        np.arange(len(estimates)),
        estimates,
        yerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="none",
        color="black",
        capsize=4,
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set_title("Paired mean-wait savings with 95% bootstrap intervals")
    axes[1, 1].set_ylabel("Seconds saved vs nearest-car")
    axes[1, 1].tick_params(axis="x", rotation=25)

    fig.suptitle(
        "Development controller comparison (validation only)",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_controller_experiment(
    config_path: str | Path, output_dir: str | Path
) -> Path:
    """Tune and compare idle-car positioning policies on validation days."""
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_days = create_development_days(config)
    models, selected_models, model_tuning = _fit_primary_controller_models(
        config, split_days
    )
    validation_days = split_days["validation"]
    schedules: dict[str, dict[str, np.ndarray]] = {}
    uncertainty_by_day: dict[str, np.ndarray] = {}
    uncertainty_values = []
    for day in validation_days:
        day_schedules, uncertainty = _day_schedules(day, config, models)
        schedules[day.day_id] = day_schedules
        uncertainty_by_day[day.day_id] = uncertainty
        uncertainty_values.extend(
            uncertainty[config.forecast.maximum_history_minutes :].tolist()
        )
    uncertainty_threshold = float(np.quantile(uncertainty_values, 0.90))

    selected_penalty, penalty_tuning = _select_positioning_penalty(
        config, validation_days, schedules
    )
    result_rows = []
    floor_rows = []
    for day in validation_days:
        policies = [
            ("nearest_car", None),
            ("static_zoning", StaticZoningPolicy()),
            *[
                (
                    policy_name,
                    ForecastPositioningPolicy(
                        schedules[day.day_id][policy_name],
                        distance_penalty=selected_penalty,
                        name=policy_name,
                    ),
                )
                for policy_name in (
                    "historical_positioning",
                    "ridge_positioning",
                    "random_forest_positioning",
                    "oracle_positioning",
                )
            ],
            (
                "uncertainty_gated_random_forest",
                UncertaintyGatedPositioningPolicy(
                    schedules[day.day_id]["random_forest_positioning"],
                    uncertainty_by_day[day.day_id],
                    uncertainty_threshold=uncertainty_threshold,
                    distance_penalty=selected_penalty,
                ),
            ),
        ]
        for policy_name, positioning in policies:
            result = simulate_day(
                day,
                config.building,
                positioning_policy=positioning,
            )
            metric = summarize_simulation(
                result,
                config.building,
                config.evaluation.long_wait_seconds,
            )
            result_rows.append(metric.to_dict())
            for floor_metric in floor_service_metrics(
                result,
                config.building.n_floors,
                config.evaluation.long_wait_seconds,
            ):
                floor_rows.append(
                    {
                        "day_id": day.day_id,
                        "scenario": day.scenario,
                        "policy": policy_name,
                        **floor_metric.to_dict(),
                    }
                )

    results = pd.DataFrame(result_rows)
    summary = (
        results.groupby("policy", as_index=False)
        .agg(
            days=("day_id", "count"),
            mean_wait_s=("mean_wait_s", "mean"),
            p95_wait_s=("p95_wait_s", "mean"),
            long_wait_rate=("long_wait_rate", "mean"),
            mean_journey_s=("mean_journey_s", "mean"),
            floor_fairness_gap_s=("floor_fairness_gap_s", "mean"),
            floors_travelled=("floors_travelled", "mean"),
            reversals=("reversals", "mean"),
        )
        .sort_values("mean_wait_s")
    )
    effects = _paired_controller_effects(results, config)
    results.to_csv(output / "controller_metrics_by_day.csv", index=False)
    summary.to_csv(output / "controller_summary.csv", index=False)
    effects.to_csv(output / "paired_controller_effects.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(
        output / "controller_floor_metrics.csv", index=False
    )
    penalty_tuning.to_csv(output / "positioning_penalty_tuning.csv", index=False)
    model_tuning.to_csv(output / "forecast_model_tuning.csv", index=False)
    controller_parameters = {
        **selected_models,
        "positioning_distance_penalty": selected_penalty,
        "uncertainty_quantile": 0.90,
        "uncertainty_threshold": uncertainty_threshold,
        "uncertainty_gate_validation_trigger_rate": float(
            np.mean(np.asarray(uncertainty_values) > uncertainty_threshold)
        ),
        "cold_start_minutes": config.forecast.maximum_history_minutes,
    }
    (output / "selected_controller_parameters.json").write_text(
        json.dumps(controller_parameters, indent=2), encoding="utf-8"
    )
    _plot_controller_results(summary, effects, output / "controller_comparison.png")
    (output / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(_environment_manifest(), indent=2), encoding="utf-8"
    )
    return output


def _controller_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent day-level controller observations."""
    return (
        results.groupby(["evaluation", "policy"], as_index=False)
        .agg(
            days=("day_id", "count"),
            passengers=("passengers", "mean"),
            mean_wait_s=("mean_wait_s", "mean"),
            p95_wait_s=("p95_wait_s", "mean"),
            long_wait_rate=("long_wait_rate", "mean"),
            mean_journey_s=("mean_journey_s", "mean"),
            floor_fairness_gap_s=("floor_fairness_gap_s", "mean"),
            floors_travelled=("floors_travelled", "mean"),
            reversals=("reversals", "mean"),
            capacity_utilization_peak=("capacity_utilization_peak", "mean"),
            drain_seconds=("drain_seconds", "mean"),
        )
        .sort_values(["evaluation", "mean_wait_s"])
    )


def _select_robust_controller(
    grid: pd.DataFrame, movement_budget_percent: float
) -> tuple[pd.Series, str]:
    """Select a validation-tuned controller under explicit cost constraints."""
    required = {
        "mean_wait_saved_s",
        "p95_wait_change_s",
        "movement_change_pct",
        "distance_penalty",
        "minimum_score_advantage",
        "positioning_interval_minutes",
    }
    if not required <= set(grid.columns):
        missing = sorted(required - set(grid.columns))
        raise ValueError(f"Controller grid is missing columns: {missing}.")
    if grid.empty:
        raise ValueError("Controller grid cannot be empty.")
    if movement_budget_percent <= 0:
        raise ValueError("Movement budget must be positive.")

    within_budget = grid[
        grid["movement_change_pct"] <= movement_budget_percent
    ]
    strict = within_budget[
        (within_budget["mean_wait_saved_s"] > 0.0)
        & (within_budget["p95_wait_change_s"] <= 0.0)
    ]
    if not strict.empty:
        ranked = strict.sort_values(
            [
                "mean_wait_saved_s",
                "p95_wait_change_s",
                "movement_change_pct",
            ],
            ascending=[False, True, True],
        )
        return ranked.iloc[0], (
            "strict: movement within budget, mean wait improved, and P95 did "
            "not worsen; maximize mean-wait savings"
        )

    if not within_budget.empty:
        ranked = within_budget.sort_values(
            ["p95_wait_change_s", "mean_wait_saved_s", "movement_change_pct"],
            ascending=[True, False, True],
        )
        return ranked.iloc[0], (
            "relaxed: no configuration met every service constraint; remain "
            "within movement budget, then minimize P95 change"
        )

    ranked = grid.sort_values(
        ["movement_change_pct", "p95_wait_change_s", "mean_wait_saved_s"],
        ascending=[True, True, False],
    )
    return ranked.iloc[0], (
        "fallback: no configuration met the movement budget; minimize movement "
        "increase, then P95 change"
    )


def _tune_robust_controller(
    config: ExperimentConfig,
    validation_days: tuple[DayData, ...],
    schedules: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.Series, str, pd.DataFrame]:
    """Run the positioning ablation grid using validation days only."""
    baseline_by_day = {}
    for day in validation_days:
        result = simulate_day(day, config.building)
        baseline_by_day[day.day_id] = summarize_simulation(
            result,
            config.building,
            config.evaluation.long_wait_seconds,
        )

    rows: list[dict[str, float | int]] = []
    for penalty in config.evaluation.positioning_distance_penalties:
        for advantage in config.evaluation.positioning_minimum_advantages:
            for interval in config.evaluation.positioning_intervals_minutes:
                candidate_metrics = []
                for day in validation_days:
                    result = simulate_day(
                        day,
                        config.building,
                        positioning_policy=ForecastPositioningPolicy(
                            schedules[day.day_id][
                                "random_forest_positioning"
                            ],
                            distance_penalty=penalty,
                            minimum_score_advantage=advantage,
                            positioning_interval_minutes=interval,
                            name="tuned_random_forest",
                        ),
                    )
                    candidate_metrics.append(
                        summarize_simulation(
                            result,
                            config.building,
                            config.evaluation.long_wait_seconds,
                        )
                    )

                base_wait = np.asarray(
                    [baseline_by_day[day.day_id].mean_wait_s for day in validation_days]
                )
                base_p95 = np.asarray(
                    [baseline_by_day[day.day_id].p95_wait_s for day in validation_days]
                )
                base_movement = np.asarray(
                    [
                        baseline_by_day[day.day_id].floors_travelled
                        for day in validation_days
                    ],
                    dtype=float,
                )
                candidate_wait = np.asarray(
                    [metric.mean_wait_s for metric in candidate_metrics]
                )
                candidate_p95 = np.asarray(
                    [metric.p95_wait_s for metric in candidate_metrics]
                )
                candidate_movement = np.asarray(
                    [metric.floors_travelled for metric in candidate_metrics],
                    dtype=float,
                )
                rows.append(
                    {
                        "distance_penalty": float(penalty),
                        "minimum_score_advantage": float(advantage),
                        "positioning_interval_minutes": int(interval),
                        "validation_mean_wait_s": float(candidate_wait.mean()),
                        "validation_p95_wait_s": float(candidate_p95.mean()),
                        "validation_floors_travelled": float(
                            candidate_movement.mean()
                        ),
                        "mean_wait_saved_s": float(
                            (base_wait - candidate_wait).mean()
                        ),
                        "p95_wait_change_s": float(
                            (candidate_p95 - base_p95).mean()
                        ),
                        "movement_change_pct": float(
                            100.0
                            * np.mean(candidate_movement / base_movement - 1.0)
                        ),
                        "days_with_lower_mean_wait": int(
                            np.sum(candidate_wait < base_wait)
                        ),
                    }
                )

    grid = pd.DataFrame(rows).sort_values(
        ["mean_wait_saved_s", "p95_wait_change_s", "movement_change_pct"],
        ascending=[False, True, True],
    )
    selected, rule = _select_robust_controller(
        grid, config.evaluation.movement_budget_percent
    )
    return selected, rule, grid


def _development_stress_days(
    config: ExperimentConfig,
) -> dict[str, tuple[DayData, ...]]:
    """Create explicit development stress sets without opening a holdout."""
    common = {
        "duration_minutes": config.traffic.duration_minutes,
        "n_floors": config.building.n_floors,
    }
    unseen_surge = create_days(
        split="stress_surge",
        scenarios=config.traffic.stress_scenarios,
        days_per_scenario=config.traffic.stress_days_per_scenario,
        seed_offset=config.seeds.stress,
        **common,
    )
    high_demand = create_days(
        split="stress_high_demand",
        scenarios=config.traffic.train_scenarios,
        days_per_scenario=1,
        seed_offset=config.seeds.stress + 100_000,
        demand_multiplier=config.evaluation.high_demand_multiplier,
        **common,
    )
    stress_seeds = {day.seed for day in unseen_surge + high_demand}
    if len(stress_seeds) != len(unseen_surge) + len(high_demand):
        raise RuntimeError("Development stress-day seeds overlap.")
    return {
        "unseen_surge": unseen_surge,
        "high_demand": high_demand,
    }


def _plot_robustness_results(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    grid: pd.DataFrame,
    selected: pd.Series,
    movement_budget_percent: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    policies = [
        "static_zoning",
        "tuned_random_forest",
        "uncertainty_gated_random_forest",
        "regime_gated_random_forest",
        "oracle_positioning",
    ]
    colors = {
        "static_zoning": "#64748b",
        "tuned_random_forest": "#059669",
        "uncertainty_gated_random_forest": "#0f766e",
        "regime_gated_random_forest": "#2563eb",
        "oracle_positioning": "#f59e0b",
    }
    evaluations = ["validation", "unseen_surge", "high_demand"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for evaluation in evaluations:
        group = effects[effects["evaluation"] == evaluation].set_index("policy")
        values = [
            group.loc[policy, "mean_wait_saved_s"]
            if policy in group.index
            else np.nan
            for policy in policies
        ]
        axes[0, 0].plot(policies, values, marker="o", label=evaluation)
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_title("Mean-wait savings vs nearest-car")
    axes[0, 0].set_ylabel("Seconds saved")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].legend(fontsize=8)

    width = 0.25
    x = np.arange(len(policies))
    for index, evaluation in enumerate(evaluations):
        group = effects[effects["evaluation"] == evaluation].set_index("policy")
        values = [
            -group.loc[policy, "p95_wait_saved_s"]
            if policy in group.index
            else np.nan
            for policy in policies
        ]
        axes[0, 1].bar(
            x + (index - 1) * width,
            values,
            width,
            label=evaluation,
        )
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_xticks(x, policies, rotation=25)
    axes[0, 1].set_title("P95 wait change vs nearest-car")
    axes[0, 1].set_ylabel("Seconds (lower is better)")
    axes[0, 1].legend(fontsize=8)

    for evaluation in evaluations:
        group = effects[effects["evaluation"] == evaluation].set_index("policy")
        values = [
            group.loc[policy, "movement_change_pct"]
            if policy in group.index
            else np.nan
            for policy in policies
        ]
        axes[1, 0].plot(policies, values, marker="o", label=evaluation)
    axes[1, 0].axhline(
        movement_budget_percent,
        color="#dc2626",
        linestyle="--",
        label="movement budget",
    )
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_title("Movement cost vs nearest-car")
    axes[1, 0].set_ylabel("Change (%)")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].legend(fontsize=8)

    scatter = axes[1, 1].scatter(
        grid["movement_change_pct"],
        grid["mean_wait_saved_s"],
        c=grid["p95_wait_change_s"],
        cmap="coolwarm",
        alpha=0.8,
    )
    axes[1, 1].scatter(
        [selected["movement_change_pct"]],
        [selected["mean_wait_saved_s"]],
        marker="*",
        s=220,
        color="#111827",
        label="selected",
    )
    axes[1, 1].axvline(
        movement_budget_percent, color="#dc2626", linestyle="--"
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set_title("Validation ablation trade-off")
    axes[1, 1].set_xlabel("Movement change (%)")
    axes[1, 1].set_ylabel("Mean-wait savings (s)")
    axes[1, 1].legend(fontsize=8)
    fig.colorbar(scatter, ax=axes[1, 1], label="P95 wait change (s)")

    fig.suptitle(
        "Controller ablation and development stress tests",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_robustness_experiment(
    config_path: str | Path, output_dir: str | Path
) -> Path:
    """Tune controller cost controls and evaluate development stress cases."""
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_days = create_development_days(config)
    models, selected_models, model_tuning = _fit_primary_controller_models(
        config, split_days
    )

    validation_days = split_days["validation"]
    validation_schedules: dict[str, dict[str, np.ndarray]] = {}
    validation_uncertainty: dict[str, np.ndarray] = {}
    uncertainty_values: list[float] = []
    for day in validation_days:
        day_schedules, uncertainty = _day_schedules(day, config, models)
        validation_schedules[day.day_id] = day_schedules
        validation_uncertainty[day.day_id] = uncertainty
        uncertainty_values.extend(
            uncertainty[config.forecast.maximum_history_minutes :].tolist()
        )
    uncertainty_threshold = float(np.quantile(uncertainty_values, 0.90))
    selected, selection_rule, grid = _tune_robust_controller(
        config, validation_days, validation_schedules
    )
    penalty = float(selected["distance_penalty"])
    advantage = float(selected["minimum_score_advantage"])
    interval = int(selected["positioning_interval_minutes"])

    evaluations = {"validation": validation_days}
    evaluations.update(_development_stress_days(config))
    result_rows: list[dict[str, object]] = []
    floor_rows: list[dict[str, object]] = []
    for evaluation, days in evaluations.items():
        for day in days:
            if evaluation == "validation":
                schedules = validation_schedules[day.day_id]
                uncertainty = validation_uncertainty[day.day_id]
            else:
                schedules, uncertainty = _day_schedules(day, config, models)

            common_parameters = {
                "distance_penalty": penalty,
                "minimum_score_advantage": advantage,
                "positioning_interval_minutes": interval,
            }
            regime_positioning = (
                StayIdlePositioningPolicy(
                    name="regime_gated_random_forest"
                )
                if day.scenario == "mixed"
                else ForecastPositioningPolicy(
                    schedules["random_forest_positioning"],
                    name="regime_gated_random_forest",
                    **common_parameters,
                )
            )
            policies = [
                ("nearest_car", None),
                ("static_zoning", StaticZoningPolicy()),
                (
                    "tuned_random_forest",
                    ForecastPositioningPolicy(
                        schedules["random_forest_positioning"],
                        name="tuned_random_forest",
                        **common_parameters,
                    ),
                ),
                (
                    "uncertainty_gated_random_forest",
                    UncertaintyGatedPositioningPolicy(
                        schedules["random_forest_positioning"],
                        uncertainty,
                        uncertainty_threshold=uncertainty_threshold,
                        name="uncertainty_gated_random_forest",
                        **common_parameters,
                    ),
                ),
                ("regime_gated_random_forest", regime_positioning),
                (
                    "oracle_positioning",
                    ForecastPositioningPolicy(
                        schedules["oracle_positioning"],
                        name="oracle_positioning",
                        **common_parameters,
                    ),
                ),
            ]
            for policy_name, positioning in policies:
                result = simulate_day(
                    day,
                    config.building,
                    positioning_policy=positioning,
                )
                metric = summarize_simulation(
                    result,
                    config.building,
                    config.evaluation.long_wait_seconds,
                )
                result_rows.append(
                    {"evaluation": evaluation, **metric.to_dict()}
                )
                for floor_metric in floor_service_metrics(
                    result,
                    config.building.n_floors,
                    config.evaluation.long_wait_seconds,
                ):
                    floor_rows.append(
                        {
                            "evaluation": evaluation,
                            "day_id": day.day_id,
                            "scenario": day.scenario,
                            "policy": policy_name,
                            **floor_metric.to_dict(),
                        }
                    )

    results = pd.DataFrame(result_rows)
    summary = _controller_summary(results)
    effect_frames = []
    for evaluation, group in results.groupby("evaluation", sort=False):
        effects = _paired_controller_effects(group, config)
        effects.insert(0, "evaluation", evaluation)
        effect_frames.append(effects)
    paired_effects = pd.concat(effect_frames, ignore_index=True)

    grid.to_csv(output / "controller_ablation_grid.csv", index=False)
    results.to_csv(output / "robustness_metrics_by_day.csv", index=False)
    summary.to_csv(output / "robustness_summary.csv", index=False)
    paired_effects.to_csv(output / "robustness_paired_effects.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(
        output / "robustness_floor_metrics.csv", index=False
    )
    model_tuning.to_csv(output / "forecast_model_tuning.csv", index=False)
    selected_parameters = {
        **selected_models,
        "positioning_distance_penalty": penalty,
        "positioning_minimum_score_advantage": advantage,
        "positioning_interval_minutes": interval,
        "movement_budget_percent": config.evaluation.movement_budget_percent,
        "selection_rule": selection_rule,
        "validation_mean_wait_saved_s": float(selected["mean_wait_saved_s"]),
        "validation_p95_wait_change_s": float(selected["p95_wait_change_s"]),
        "validation_movement_change_pct": float(
            selected["movement_change_pct"]
        ),
        "uncertainty_quantile": 0.90,
        "uncertainty_threshold": uncertainty_threshold,
        "cold_start_minutes": config.forecast.maximum_history_minutes,
        "stress_sets_are_development_only": True,
        "final_holdout_accessed": False,
    }
    (output / "selected_robust_controller.json").write_text(
        json.dumps(selected_parameters, indent=2), encoding="utf-8"
    )
    _plot_robustness_results(
        summary,
        paired_effects,
        grid,
        selected,
        config.evaluation.movement_budget_percent,
        output / "robustness_comparison.png",
    )
    (output / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(_environment_manifest(), indent=2), encoding="utf-8"
    )
    return output


def _new_rnn(
    config: ExperimentConfig,
    hidden_size: int,
    *,
    random_state: int | None = None,
) -> ElmanRNNForecaster:
    settings = config.sequence_models
    return ElmanRNNForecaster(
        hidden_size=hidden_size,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        maximum_epochs=settings.maximum_epochs,
        patience=settings.early_stopping_patience,
        gradient_clip_norm=settings.gradient_clip_norm,
        l2_penalty=settings.l2_penalty,
        random_state=(
            config.models.random_state
            if random_state is None
            else random_state
        ),
    )


def _assert_sequence_alignment(sequence, tabular) -> None:
    sequence_ids = sequence.metadata["sample_id"].tolist()
    tabular_ids = tabular.metadata["sample_id"].tolist()
    if sequence_ids != tabular_ids:
        raise RuntimeError("Sequence and tabular sample IDs are not aligned.")
    if not np.array_equal(sequence.targets, tabular.targets):
        raise RuntimeError("Sequence and tabular targets are not identical.")


def _plot_sequence_results(
    metrics: pd.DataFrame,
    tuning: pd.DataFrame,
    history: pd.DataFrame,
    selected_parameters: dict[str, int | float | str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    order = [
        "per_floor_mean",
        "historical_mean",
        "persistence",
        "ridge",
        "random_forest",
        "elman_rnn",
    ]
    labels = [
        "Per-floor mean",
        "Historical",
        "Persistence",
        "Ridge",
        "Random Forest",
        "Elman RNN",
    ]
    colors = ["#6b7280", "#2563eb", "#dc2626", "#7c3aed", "#059669", "#f59e0b"]
    validation = metrics[metrics["evaluation"] == "validation"].set_index(
        "model"
    ).reindex(order)
    comparison = metrics.pivot(
        index="model", columns="evaluation", values="floor_mae"
    ).reindex(order)
    selected_history = history[
        (history["sequence_length"] == selected_parameters["sequence_length"])
        & (history["hidden_size"] == selected_parameters["hidden_size"])
    ]
    matrix = tuning.pivot(
        index="hidden_size",
        columns="sequence_length",
        values="validation_floor_mae",
    ).sort_index().sort_index(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(labels, validation["floor_mae"], color=colors)
    axes[0, 0].set_title("Two-minute validation per-floor MAE")
    axes[0, 0].set_ylabel("Passengers")
    axes[0, 0].tick_params(axis="x", rotation=20)

    axes[0, 1].plot(
        selected_history["epoch"],
        selected_history["train_loss"],
        label="train",
        color="#64748b",
    )
    axes[0, 1].plot(
        selected_history["epoch"],
        selected_history["validation_loss"],
        label="validation",
        color="#f59e0b",
    )
    axes[0, 1].axvline(
        selected_parameters["best_epoch"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="restored epoch",
    )
    axes[0, 1].set_title("Selected RNN learning curve")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Scaled MSE + L2")
    axes[0, 1].legend(fontsize=8)

    comparison.plot(
        kind="bar",
        ax=axes[1, 0],
        color=["#94a3b8", "#2563eb"],
    )
    axes[1, 0].set_xticklabels(labels, rotation=20)
    axes[1, 0].set_title("Train-validation generalization")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("Per-floor MAE")

    image = axes[1, 1].imshow(matrix.to_numpy(), cmap="YlGnBu_r", aspect="auto")
    axes[1, 1].set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    axes[1, 1].set_yticks(np.arange(len(matrix.index)), matrix.index)
    axes[1, 1].set_xlabel("Sequence length (minutes)")
    axes[1, 1].set_ylabel("Hidden units")
    axes[1, 1].set_title("RNN validation MAE ablation")
    for row_index in range(len(matrix.index)):
        for column_index in range(len(matrix.columns)):
            axes[1, 1].text(
                column_index,
                row_index,
                f"{matrix.iloc[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(image, ax=axes[1, 1], label="Per-floor MAE")

    fig.suptitle(
        "Development sequence-model comparison",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_sequence_experiment(
    config_path: str | Path, output_dir: str | Path
) -> Path:
    """Tune a from-scratch Elman RNN on training/validation days only."""
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_days = create_development_days(config)
    tabular_train, tabular_validation = _primary_datasets(config, split_days)
    first_target = config.forecast.maximum_history_minutes
    horizon = config.forecast.primary_horizon_minutes

    tuning_rows: list[dict[str, float | int | str]] = []
    history_frames: list[pd.DataFrame] = []
    trained_models: dict[tuple[int, int], ElmanRNNForecaster] = {}
    sequence_datasets = {}
    for sequence_length in config.sequence_models.sequence_lengths:
        train = build_sequence_dataset(
            split_days["train"],
            sequence_length,
            horizon,
            config.traffic.train_scenarios,
            minimum_target_minute=first_target,
        )
        validation = build_sequence_dataset(
            split_days["validation"],
            sequence_length,
            horizon,
            config.traffic.train_scenarios,
            minimum_target_minute=first_target,
        )
        _assert_sequence_alignment(train, tabular_train)
        _assert_sequence_alignment(validation, tabular_validation)
        sequence_datasets[sequence_length] = (train, validation)

        for hidden_size in config.sequence_models.hidden_sizes:
            model = _new_rnn(config, hidden_size)
            started = time.perf_counter()
            model.fit(train, validation)
            runtime_seconds = time.perf_counter() - started
            train_predictions = model.predict(train)
            validation_predictions = model.predict(validation)
            train_metric = forecast_metric_row(
                tabular_train,
                train_predictions,
                model_name=model.name,
                evaluation="train",
            )
            validation_metric = forecast_metric_row(
                tabular_validation,
                validation_predictions,
                model_name=model.name,
                evaluation="validation",
            )
            if model.summary_ is None or model.history_ is None:
                raise RuntimeError("Fitted RNN did not expose diagnostics.")
            tuning_rows.append(
                {
                    "sequence_length": sequence_length,
                    "hidden_size": hidden_size,
                    "train_floor_mae": train_metric["floor_mae"],
                    "validation_floor_mae": validation_metric["floor_mae"],
                    "validation_floor_rmse": validation_metric["floor_rmse"],
                    "validation_total_demand_mae": validation_metric[
                        "total_demand_mae"
                    ],
                    "epochs_trained": model.summary_.epochs_trained,
                    "best_epoch": model.summary_.best_epoch,
                    "gradient_clip_steps": model.summary_.gradient_clip_steps,
                    "parameter_count": model.summary_.parameter_count,
                    "training_runtime_seconds": runtime_seconds,
                }
            )
            history = model.history_.copy()
            history.insert(0, "hidden_size", hidden_size)
            history.insert(0, "sequence_length", sequence_length)
            history_frames.append(history)
            trained_models[(sequence_length, hidden_size)] = model

    tuning = pd.DataFrame(tuning_rows).sort_values(
        [
            "validation_floor_mae",
            "validation_total_demand_mae",
            "parameter_count",
        ]
    )
    selected_row = tuning.iloc[0]
    selected_length = int(selected_row["sequence_length"])
    selected_hidden = int(selected_row["hidden_size"])
    selected_model = trained_models[(selected_length, selected_hidden)]
    sequence_train, sequence_validation = sequence_datasets[selected_length]

    selected_tabular, model_tuning = _tune_learned_models(
        config, tabular_train, tabular_validation
    )
    comparison_models = _forecaster_set(config, selected_tabular)
    metric_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    floor_rows: list[dict[str, object]] = []
    for model in comparison_models:
        model.fit(tabular_train)
        train_predictions = model.predict(tabular_train)
        validation_predictions = model.predict(tabular_validation)
        metric_rows.extend(
            [
                forecast_metric_row(
                    tabular_train,
                    train_predictions,
                    model_name=model.name,
                    evaluation="train",
                ),
                forecast_metric_row(
                    tabular_validation,
                    validation_predictions,
                    model_name=model.name,
                    evaluation="validation",
                ),
            ]
        )
        scenario_rows.extend(
            scenario_metric_rows(
                tabular_validation,
                validation_predictions,
                model_name=model.name,
                evaluation="validation",
            )
        )
        floor_rows.extend(
            floor_metric_rows(
                tabular_validation,
                validation_predictions,
                model_name=model.name,
                evaluation="validation",
            )
        )

    rnn_train_predictions = selected_model.predict(sequence_train)
    rnn_validation_predictions = selected_model.predict(sequence_validation)
    metric_rows.extend(
        [
            forecast_metric_row(
                tabular_train,
                rnn_train_predictions,
                model_name=selected_model.name,
                evaluation="train",
            ),
            forecast_metric_row(
                tabular_validation,
                rnn_validation_predictions,
                model_name=selected_model.name,
                evaluation="validation",
            ),
        ]
    )
    scenario_rows.extend(
        scenario_metric_rows(
            tabular_validation,
            rnn_validation_predictions,
            model_name=selected_model.name,
            evaluation="validation",
        )
    )
    floor_rows.extend(
        floor_metric_rows(
            tabular_validation,
            rnn_validation_predictions,
            model_name=selected_model.name,
            evaluation="validation",
        )
    )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["evaluation", "floor_mae"]
    )
    history = pd.concat(history_frames, ignore_index=True)
    selected_parameters: dict[str, int | float | str] = {
        "model": selected_model.name,
        "framework": "numpy",
        "sequence_length": selected_length,
        "hidden_size": selected_hidden,
        "input_features_per_step": sequence_train.n_features,
        "output_floors": sequence_train.n_targets,
        "learning_rate": config.sequence_models.learning_rate,
        "batch_size": config.sequence_models.batch_size,
        "maximum_epochs": config.sequence_models.maximum_epochs,
        "early_stopping_patience": (
            config.sequence_models.early_stopping_patience
        ),
        "gradient_clip_norm": config.sequence_models.gradient_clip_norm,
        "l2_penalty": config.sequence_models.l2_penalty,
        "epochs_trained": selected_model.summary_.epochs_trained,
        "best_epoch": selected_model.summary_.best_epoch,
        "gradient_clip_steps": selected_model.summary_.gradient_clip_steps,
        "parameter_count": selected_model.summary_.parameter_count,
        "validation_floor_mae": float(selected_row["validation_floor_mae"]),
        "selection_metric": "validation per-floor MAE",
        "scalers_fitted_on": "training sequences only",
        "backpropagation": "full sequence BPTT",
        "final_holdout_accessed": False,
    }

    tuning.to_csv(output / "rnn_tuning.csv", index=False)
    history.to_csv(output / "rnn_training_history.csv", index=False)
    metrics.to_csv(output / "sequence_model_metrics.csv", index=False)
    pd.DataFrame(scenario_rows).to_csv(
        output / "sequence_scenario_metrics.csv", index=False
    )
    pd.DataFrame(floor_rows).to_csv(
        output / "sequence_floor_metrics.csv", index=False
    )
    model_tuning.to_csv(output / "tabular_model_tuning.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": split,
                "sequence_length": selected_length,
                "samples": dataset.n_samples,
                "time_steps": dataset.sequence_length,
                "features_per_step": dataset.n_features,
                "targets": dataset.n_targets,
                "unique_days": dataset.metadata["day_id"].nunique(),
            }
            for split, dataset in (
                ("train", sequence_train),
                ("validation", sequence_validation),
            )
        ]
    ).to_csv(output / "sequence_dataset_manifest.csv", index=False)
    dump(selected_model, output / "selected_rnn_model.joblib")
    np.savez_compressed(
        output / "selected_rnn_validation_predictions.npz",
        targets=sequence_validation.targets,
        predictions=rnn_validation_predictions,
    )
    (output / "selected_rnn_parameters.json").write_text(
        json.dumps(selected_parameters, indent=2), encoding="utf-8"
    )
    _plot_sequence_results(
        metrics,
        tuning,
        history,
        selected_parameters,
        output / "sequence_model_comparison.png",
    )
    (output / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(_environment_manifest(), indent=2), encoding="utf-8"
    )
    return output


def _plot_operational_value(
    by_day: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    evaluations = ["validation", "unseen_surge", "high_demand"]
    labels = ["Validation", "Unseen surge", "1.5x demand"]
    colors = {
        "validation": "#2563eb",
        "unseen_surge": "#f97316",
        "high_demand": "#16a34a",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for evaluation in evaluations:
        group = by_day[by_day["evaluation"] == evaluation]
        axes[0, 0].scatter(
            group["rf_floor_mae"],
            group["rf_mean_wait_saved_s"],
            label=evaluation.replace("_", " "),
            color=colors[evaluation],
            s=55,
            alpha=0.85,
        )
    axes[0, 0].axhline(0.0, color="black", linewidth=1)
    axes[0, 0].set_xlabel("Random Forest per-floor forecast MAE")
    axes[0, 0].set_ylabel("Mean wait saved (s)")
    axes[0, 0].set_title("Forecast error versus operational value")
    axes[0, 0].legend(fontsize=8)

    x = np.arange(len(evaluations))
    indexed = summary.set_index("evaluation").reindex(evaluations)
    width = 0.34
    axes[0, 1].bar(
        x - width / 2,
        indexed["rf_mean_wait_saved_s"],
        width,
        label="Random Forest",
        color="#059669",
    )
    axes[0, 1].bar(
        x + width / 2,
        indexed["oracle_mean_wait_saved_s"],
        width,
        label="Perfect-input greedy",
        color="#f59e0b",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Mean wait saved (s)")
    axes[0, 1].set_title("Learned versus perfect-input greedy heuristic")
    axes[0, 1].legend(fontsize=8)

    for position, evaluation in enumerate(evaluations):
        values = by_day.loc[
            by_day["evaluation"] == evaluation, "rf_floor_mae"
        ].to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(values))
        axes[1, 0].scatter(
            np.full(len(values), position) + jitter,
            values,
            color=colors[evaluation],
            s=50,
        )
        axes[1, 0].plot(
            [position - 0.18, position + 0.18],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=2,
        )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("Per-floor MAE")
    axes[1, 0].set_title("Forecast distribution shift by evaluation")

    categories = [
        "benefit_realized",
        "forecast_limited",
        "controller_limited",
    ]
    category_labels = ["Benefit realized", "Forecast limited", "Controller limited"]
    bottoms = np.zeros(len(evaluations), dtype=float)
    category_colors = ["#16a34a", "#7c3aed", "#dc2626"]
    for category, label, color in zip(
        categories, category_labels, category_colors
    ):
        counts = np.asarray(
            [
                np.sum(
                    (by_day["evaluation"] == evaluation)
                    & (by_day["failure_attribution"] == category)
                )
                for evaluation in evaluations
            ],
            dtype=float,
        )
        axes[1, 1].bar(x, counts, bottom=bottoms, label=label, color=color)
        bottoms += counts
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Independent days")
    axes[1, 1].set_title("Day-level failure attribution")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle(
        "Forecast accuracy and elevator-control value",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_operational_value_analysis(
    config_path: str | Path,
    robustness_output_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Relate day-level forecast quality to paired controller outcomes."""
    config = load_config(config_path)
    robustness_output = Path(robustness_output_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    operational_path = robustness_output / "robustness_metrics_by_day.csv"
    controller_path = robustness_output / "selected_robust_controller.json"
    if not operational_path.is_file() or not controller_path.is_file():
        raise FileNotFoundError(
            "Robustness outputs are required before operational-value analysis."
        )
    operational = pd.read_csv(operational_path)
    controller_parameters = json.loads(controller_path.read_text(encoding="utf-8"))

    split_days = create_development_days(config)
    models, selected_models, model_tuning = _fit_primary_controller_models(
        config, split_days
    )
    evaluations = {"validation": split_days["validation"]}
    evaluations.update(_development_stress_days(config))
    rows: list[dict[str, object]] = []
    required_policies = {
        "nearest_car",
        "tuned_random_forest",
        "oracle_positioning",
    }
    for evaluation, days in evaluations.items():
        for day in days:
            forecast_data = build_forecasting_dataset(
                (day,),
                config.forecast.history_windows_minutes,
                config.forecast.primary_horizon_minutes,
                config.traffic.train_scenarios,
            )
            forest_predictions = models["random_forest"].predict(forecast_data)
            historical_predictions = models["historical_mean"].predict(
                forecast_data
            )
            forest_metric = forecast_metric_row(
                forecast_data,
                forest_predictions,
                model_name="random_forest",
                evaluation=evaluation,
            )
            historical_metric = forecast_metric_row(
                forecast_data,
                historical_predictions,
                model_name="historical_mean",
                evaluation=evaluation,
            )

            day_operations = operational[
                (operational["evaluation"] == evaluation)
                & (operational["day_id"] == day.day_id)
            ].set_index("policy")
            if not required_policies <= set(day_operations.index):
                missing = sorted(required_policies - set(day_operations.index))
                raise RuntimeError(
                    f"Operational rows missing for {day.day_id}: {missing}."
                )
            nearest = day_operations.loc["nearest_car"]
            forest = day_operations.loc["tuned_random_forest"]
            oracle = day_operations.loc["oracle_positioning"]
            forest_saved = float(nearest["mean_wait_s"] - forest["mean_wait_s"])
            oracle_saved = float(nearest["mean_wait_s"] - oracle["mean_wait_s"])
            if oracle_saved <= 0.0:
                attribution = "controller_limited"
            elif forest_saved <= 0.0:
                attribution = "forecast_limited"
            else:
                attribution = "benefit_realized"
            rows.append(
                {
                    "evaluation": evaluation,
                    "day_id": day.day_id,
                    "scenario": day.scenario,
                    "seed": day.seed,
                    "passengers": len(day.passengers),
                    "rf_floor_mae": forest_metric["floor_mae"],
                    "rf_total_demand_mae": forest_metric["total_demand_mae"],
                    "rf_top3_floor_overlap": forest_metric[
                        "top3_floor_overlap"
                    ],
                    "historical_floor_mae": historical_metric["floor_mae"],
                    "nearest_mean_wait_s": nearest["mean_wait_s"],
                    "rf_mean_wait_s": forest["mean_wait_s"],
                    "oracle_mean_wait_s": oracle["mean_wait_s"],
                    "rf_mean_wait_saved_s": forest_saved,
                    "oracle_mean_wait_saved_s": oracle_saved,
                    "forecast_control_gap_s": oracle_saved - forest_saved,
                    "rf_p95_wait_saved_s": float(
                        nearest["p95_wait_s"] - forest["p95_wait_s"]
                    ),
                    "rf_movement_change_pct": float(
                        100.0
                        * (
                            forest["floors_travelled"]
                            / nearest["floors_travelled"]
                            - 1.0
                        )
                    ),
                    "failure_attribution": attribution,
                }
            )

    by_day = pd.DataFrame(rows).sort_values(["evaluation", "day_id"])
    summary = (
        by_day.groupby("evaluation", as_index=False)
        .agg(
            days=("day_id", "count"),
            passengers=("passengers", "mean"),
            rf_floor_mae=("rf_floor_mae", "mean"),
            rf_total_demand_mae=("rf_total_demand_mae", "mean"),
            rf_top3_floor_overlap=("rf_top3_floor_overlap", "mean"),
            historical_floor_mae=("historical_floor_mae", "mean"),
            rf_mean_wait_saved_s=("rf_mean_wait_saved_s", "mean"),
            oracle_mean_wait_saved_s=("oracle_mean_wait_saved_s", "mean"),
            forecast_control_gap_s=("forecast_control_gap_s", "mean"),
            rf_p95_wait_saved_s=("rf_p95_wait_saved_s", "mean"),
            rf_movement_change_pct=("rf_movement_change_pct", "mean"),
        )
        .sort_values("evaluation")
    )
    attribution_counts = (
        by_day.groupby(["evaluation", "failure_attribution"])
        .size()
        .rename("days")
        .reset_index()
    )

    associations = [
        ("rf_floor_mae", "rf_mean_wait_saved_s"),
        ("rf_total_demand_mae", "rf_mean_wait_saved_s"),
        ("rf_top3_floor_overlap", "rf_mean_wait_saved_s"),
        ("rf_floor_mae", "forecast_control_gap_s"),
    ]
    correlation_rows: list[dict[str, str | int | float]] = []
    scopes = [("all_development", by_day)] + [
        (evaluation, group)
        for evaluation, group in by_day.groupby("evaluation", sort=True)
    ]
    seed = config.models.random_state
    for scope, group in scopes:
        for predictor, outcome in associations:
            for method in ("pearson", "spearman"):
                result = correlation_interval(
                    group[predictor].to_numpy(),
                    group[outcome].to_numpy(),
                    method=method,
                    resamples=config.evaluation.bootstrap_resamples,
                    seed=seed,
                )
                seed += 1
                correlation_rows.append(
                    {
                        "scope": scope,
                        "predictor": predictor,
                        "outcome": outcome,
                        **result.to_dict(),
                    }
                )
    correlations = pd.DataFrame(correlation_rows)

    by_day.to_csv(output / "forecast_control_by_day.csv", index=False)
    summary.to_csv(output / "forecast_control_summary.csv", index=False)
    attribution_counts.to_csv(
        output / "failure_attribution_counts.csv", index=False
    )
    correlations.to_csv(
        output / "forecast_control_correlations.csv", index=False
    )
    model_tuning.to_csv(output / "forecast_model_tuning.csv", index=False)
    analysis_manifest = {
        "forecast_models": selected_models,
        "controller_parameters": controller_parameters,
        "correlation_unit": "independent building day",
        "correlation_uncertainty": "paired day bootstrap",
        "correlation_significance": "two-sided permutation test",
        "failure_attribution_rule": {
            "controller_limited": "oracle wait savings <= 0",
            "forecast_limited": "oracle savings > 0 and RF savings <= 0",
            "benefit_realized": "oracle savings > 0 and RF savings > 0",
        },
        "final_holdout_accessed": False,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2), encoding="utf-8"
    )
    _plot_operational_value(
        by_day,
        summary,
        output / "forecast_control_diagnostics.png",
    )
    (output / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(_environment_manifest(), indent=2), encoding="utf-8"
    )
    return output
