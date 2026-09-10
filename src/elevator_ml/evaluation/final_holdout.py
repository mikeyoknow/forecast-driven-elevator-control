"""Run-once final holdout pipeline executed only after protocol locking."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from joblib import dump

from elevator_ml.config import ExperimentConfig, load_config
from elevator_ml.control.policies import (
    ForecastPositioningPolicy,
    StaticZoningPolicy,
)
from elevator_ml.data.features import build_forecasting_dataset
from elevator_ml.data.generator import create_days
from elevator_ml.data.sequences import build_sequence_dataset
from elevator_ml.evaluation.experiments import (
    _day_schedules,
    _forecaster_set,
    _paired_controller_effects,
    create_development_days,
)
from elevator_ml.evaluation.forecast_metrics import (
    floor_metric_rows,
    forecast_metric_row,
    scenario_metric_rows,
)
from elevator_ml.evaluation.metrics import (
    floor_service_metrics,
    paired_bootstrap_interval,
    summarize_simulation,
)
from elevator_ml.forecasting.rnn import ElmanRNNForecaster
from elevator_ml.simulation.entities import DayData
from elevator_ml.simulation.simulator import simulate_day


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {source}.")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON root must be an object: {source}.")
    return raw


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_final_protocol(
    protocol: dict[str, Any], config: ExperimentConfig
) -> None:
    required = {
        "protocol_name",
        "status",
        "final_seed_file",
        "run_once",
        "evaluations",
        "selected_forecast_hyperparameters",
        "selected_rnn_hyperparameters",
        "selected_controller_parameters",
        "forecast_models",
        "controller_policies",
        "post_holdout_rule",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Final protocol is missing keys: {sorted(missing)}.")
    if protocol["status"] != "locked_before_seed_reveal":
        raise ValueError("Final protocol status is not locked_before_seed_reveal.")
    if protocol["run_once"] is not True:
        raise ValueError("Final protocol must enforce run_once=true.")
    expected_evaluations = {"in_distribution", "unseen_surge", "high_demand"}
    if set(protocol["evaluations"]) != expected_evaluations:
        raise ValueError("Final evaluation groups differ from the locked design.")
    for name, specification in protocol["evaluations"].items():
        if not specification.get("scenarios"):
            raise ValueError(f"Final evaluation {name} has no scenarios.")
        if int(specification.get("days_per_scenario", 0)) <= 0:
            raise ValueError(f"Final evaluation {name} needs positive day count.")
        if float(specification.get("demand_multiplier", 0.0)) <= 0:
            raise ValueError(f"Final evaluation {name} needs positive demand.")
        if int(specification.get("seed_offset_increment", -1)) < 0:
            raise ValueError(f"Final evaluation {name} has invalid seed increment.")
    forecast = protocol["selected_forecast_hyperparameters"]
    if int(forecast["random_state"]) != config.models.random_state:
        raise ValueError("Locked Random Forest seed differs from development config.")
    if int(forecast["random_forest_n_estimators"]) != (
        config.models.random_forest_n_estimators
    ):
        raise ValueError("Locked Random Forest tree count differs from config.")
    controller = protocol["selected_controller_parameters"]
    if float(controller["movement_budget_percent"]) != (
        config.evaluation.movement_budget_percent
    ):
        raise ValueError("Locked movement budget differs from development config.")


def _validate_final_seed(
    seed_manifest: dict[str, Any],
    protocol_path: str | Path,
    config: ExperimentConfig,
) -> int:
    required = {"seed_offset", "revealed_utc", "protocol_sha256"}
    missing = required - set(seed_manifest)
    if missing:
        raise ValueError(f"Final seed manifest is missing keys: {sorted(missing)}.")
    seed_offset = int(seed_manifest["seed_offset"])
    if seed_offset < 1_000_000:
        raise ValueError("Final seed offset must be at least 1,000,000.")
    development_seeds = {
        config.seeds.train,
        config.seeds.validation,
        config.seeds.development_test,
        config.seeds.stress,
    }
    if seed_offset in development_seeds:
        raise ValueError("Final seed offset overlaps a development seed offset.")
    if seed_manifest["protocol_sha256"] != _sha256(protocol_path):
        raise ValueError("Final seed manifest does not reference the locked protocol.")
    return seed_offset


def create_final_days(
    config: ExperimentConfig,
    protocol: dict[str, Any],
    seed_offset: int,
) -> dict[str, tuple[DayData, ...]]:
    evaluations: dict[str, tuple[DayData, ...]] = {}
    all_seeds: set[int] = set()
    for evaluation, specification in protocol["evaluations"].items():
        days = create_days(
            split=f"final_{evaluation}",
            scenarios=tuple(specification["scenarios"]),
            days_per_scenario=int(specification["days_per_scenario"]),
            seed_offset=(
                seed_offset + int(specification["seed_offset_increment"])
            ),
            duration_minutes=config.traffic.duration_minutes,
            n_floors=config.building.n_floors,
            demand_multiplier=float(specification["demand_multiplier"]),
        )
        day_seeds = {day.seed for day in days}
        if all_seeds & day_seeds:
            raise RuntimeError("Final evaluation seed ranges overlap.")
        all_seeds.update(day_seeds)
        evaluations[evaluation] = days
    return evaluations


def _fit_locked_models(
    config: ExperimentConfig,
    protocol: dict[str, Any],
    development_days: dict[str, tuple[DayData, ...]],
):
    forecast = protocol["selected_forecast_hyperparameters"]
    selected = {
        "ridge_alpha": float(forecast["ridge_alpha"]),
        "random_forest_max_depth": int(forecast["random_forest_max_depth"]),
        "random_forest_min_samples_leaf": int(
            forecast["random_forest_min_samples_leaf"]
        ),
    }
    horizon = config.forecast.primary_horizon_minutes
    train = build_forecasting_dataset(
        development_days["train"],
        config.forecast.history_windows_minutes,
        horizon,
        config.traffic.train_scenarios,
    )
    models = {
        model.name: model.fit(train) for model in _forecaster_set(config, selected)
    }

    rnn_settings = protocol["selected_rnn_hyperparameters"]
    sequence_length = int(rnn_settings["sequence_length"])
    train_sequence = build_sequence_dataset(
        development_days["train"],
        sequence_length,
        horizon,
        config.traffic.train_scenarios,
        minimum_target_minute=config.forecast.maximum_history_minutes,
    )
    validation_sequence = build_sequence_dataset(
        development_days["validation"],
        sequence_length,
        horizon,
        config.traffic.train_scenarios,
        minimum_target_minute=config.forecast.maximum_history_minutes,
    )
    sequence = config.sequence_models
    rnn = ElmanRNNForecaster(
        hidden_size=int(rnn_settings["hidden_size"]),
        learning_rate=sequence.learning_rate,
        batch_size=sequence.batch_size,
        maximum_epochs=sequence.maximum_epochs,
        patience=sequence.early_stopping_patience,
        gradient_clip_norm=sequence.gradient_clip_norm,
        l2_penalty=sequence.l2_penalty,
        random_state=config.models.random_state,
    ).fit(train_sequence, validation_sequence)
    if rnn.summary_ is None or rnn.summary_.best_epoch != int(
        rnn_settings["selection_epoch"]
    ):
        raise RuntimeError("Locked RNN best epoch was not reproduced.")
    return models, rnn, sequence_length


def _forecast_day_rows(
    evaluation: str,
    days: tuple[DayData, ...],
    config: ExperimentConfig,
    models: dict[str, object],
    rnn: ElmanRNNForecaster,
    sequence_length: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    aggregate_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    floor_rows: list[dict[str, object]] = []
    for day in days:
        tabular = build_forecasting_dataset(
            (day,),
            config.forecast.history_windows_minutes,
            config.forecast.primary_horizon_minutes,
            config.traffic.train_scenarios,
        )
        sequence = build_sequence_dataset(
            (day,),
            sequence_length,
            config.forecast.primary_horizon_minutes,
            config.traffic.train_scenarios,
            minimum_target_minute=config.forecast.maximum_history_minutes,
        )
        if tabular.metadata["sample_id"].tolist() != (
            sequence.metadata["sample_id"].tolist()
        ):
            raise RuntimeError("Final tabular and sequence samples are misaligned.")
        predictions = {
            name: model.predict(tabular) for name, model in models.items()
        }
        predictions[rnn.name] = rnn.predict(sequence)
        for name, values in predictions.items():
            row = forecast_metric_row(
                tabular,
                values,
                model_name=name,
                evaluation=evaluation,
            )
            aggregate_rows.append(
                {"day_id": day.day_id, "scenario": day.scenario, **row}
            )
            scenario_rows.extend(
                scenario_metric_rows(
                    tabular,
                    values,
                    model_name=name,
                    evaluation=evaluation,
                )
            )
            for floor_row in floor_metric_rows(
                tabular,
                values,
                model_name=name,
                evaluation=evaluation,
            ):
                floor_rows.append(
                    {"day_id": day.day_id, "scenario": day.scenario, **floor_row}
                )
    return aggregate_rows, scenario_rows, floor_rows


def _forecast_paired_effects(
    by_day: pd.DataFrame, config: ExperimentConfig
) -> pd.DataFrame:
    frames = []
    for evaluation, group in by_day.groupby("evaluation", sort=False):
        baseline = group[group["model"] == "historical_mean"].set_index("day_id")
        rows = []
        for model in sorted(set(group["model"]) - {"historical_mean"}):
            candidate = group[group["model"] == model].set_index("day_id")
            common = baseline.index.intersection(candidate.index)
            interval = paired_bootstrap_interval(
                baseline.loc[common, "floor_mae"].to_numpy(),
                candidate.loc[common, "floor_mae"].to_numpy(),
                resamples=config.evaluation.bootstrap_resamples,
                seed=config.models.random_state,
            )
            rows.append(
                {
                    "evaluation": evaluation,
                    "model": model,
                    "baseline": "historical_mean",
                    "paired_days": len(common),
                    "floor_mae_reduction": interval.estimate,
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                    "days_with_lower_mae": int(
                        np.sum(
                            candidate.loc[common, "floor_mae"].to_numpy()
                            < baseline.loc[common, "floor_mae"].to_numpy()
                        )
                    ),
                }
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _controller_rows(
    evaluation: str,
    days: tuple[DayData, ...],
    config: ExperimentConfig,
    models: dict[str, object],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    parameters = protocol["selected_controller_parameters"]
    common = {
        "distance_penalty": float(parameters["distance_penalty"]),
        "minimum_score_advantage": float(
            parameters["minimum_score_advantage"]
        ),
        "positioning_interval_minutes": int(
            parameters["positioning_interval_minutes"]
        ),
    }
    rows: list[dict[str, object]] = []
    floor_rows: list[dict[str, object]] = []
    for day in days:
        schedules, _ = _day_schedules(day, config, models)
        policies = [
            ("nearest_car", None),
            ("static_zoning", StaticZoningPolicy()),
            (
                "historical_positioning",
                ForecastPositioningPolicy(
                    schedules["historical_positioning"],
                    name="historical_positioning",
                    **common,
                ),
            ),
            (
                "tuned_random_forest",
                ForecastPositioningPolicy(
                    schedules["random_forest_positioning"],
                    name="tuned_random_forest",
                    **common,
                ),
            ),
            (
                "perfect_input_greedy",
                ForecastPositioningPolicy(
                    schedules["oracle_positioning"],
                    name="perfect_input_greedy",
                    **common,
                ),
            ),
        ]
        for policy_name, positioning in policies:
            result = simulate_day(
                day, config.building, positioning_policy=positioning
            )
            metric = summarize_simulation(
                result, config.building, config.evaluation.long_wait_seconds
            )
            rows.append({"evaluation": evaluation, **metric.to_dict()})
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
    return rows, floor_rows


def _controller_summary(results: pd.DataFrame) -> pd.DataFrame:
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
            capacity_utilization_peak=("capacity_utilization_peak", "mean"),
        )
        .sort_values(["evaluation", "mean_wait_s"])
    )


def _plot_final_results(
    forecast_summary: pd.DataFrame,
    controller_effects: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    evaluations = ["in_distribution", "unseen_surge", "high_demand"]
    evaluation_labels = ["In-dist.", "Surge", "1.5x demand"]
    forecast_models = ["historical_mean", "random_forest", "elman_rnn"]
    forecast_labels = ["Historical", "Random Forest", "Elman RNN"]
    policies = ["static_zoning", "historical_positioning", "tuned_random_forest"]
    policy_labels = ["Static zoning", "Historical", "Random Forest"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    width = 0.25
    x = np.arange(len(evaluations))
    for index, (model, label) in enumerate(zip(forecast_models, forecast_labels)):
        group = forecast_summary[forecast_summary["model"] == model].set_index(
            "evaluation"
        )
        axes[0, 0].bar(
            x + (index - 1) * width,
            [group.loc[value, "floor_mae"] for value in evaluations],
            width,
            label=label,
        )
    axes[0, 0].set_xticks(x, evaluation_labels)
    axes[0, 0].set_ylabel("Per-floor MAE")
    axes[0, 0].set_title("Final forecast comparison")
    axes[0, 0].legend(fontsize=8)

    for axis, metric, title, ylabel in (
        (axes[0, 1], "mean_wait_saved_s", "Mean-wait savings", "Seconds saved"),
        (axes[1, 0], "p95_wait_saved_s", "P95-wait savings", "Seconds saved"),
        (axes[1, 1], "movement_change_pct", "Movement cost", "Change (%)"),
    ):
        for index, (policy, label) in enumerate(zip(policies, policy_labels)):
            group = controller_effects[
                controller_effects["policy"] == policy
            ].set_index("evaluation")
            axis.bar(
                x + (index - 1) * width,
                [group.loc[value, metric] for value in evaluations],
                width,
                label=label,
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks(x, evaluation_labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(fontsize=8)

    fig.suptitle("Untouched final holdout", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_final_holdout(
    config_path: str | Path,
    protocol_path: str | Path,
    seed_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Execute the complete final protocol once and preserve every result."""
    config = load_config(config_path)
    protocol = _load_json(protocol_path)
    validate_final_protocol(protocol, config)
    seed_manifest = _load_json(seed_path)
    seed_offset = _validate_final_seed(
        seed_manifest, protocol_path, config
    )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Final output directory is not empty: {output}. Refusing to rerun."
        )
    output.mkdir(parents=True, exist_ok=True)

    started_utc = datetime.now(timezone.utc).isoformat()
    development_days = create_development_days(config)
    final_days = create_final_days(config, protocol, seed_offset)
    models, rnn, sequence_length = _fit_locked_models(
        config, protocol, development_days
    )

    forecast_rows: list[dict[str, object]] = []
    forecast_scenario_rows: list[dict[str, object]] = []
    forecast_floor_rows: list[dict[str, object]] = []
    controller_rows: list[dict[str, object]] = []
    controller_floor_rows: list[dict[str, object]] = []
    for evaluation, days in final_days.items():
        day_forecasts, scenario_forecasts, floor_forecasts = _forecast_day_rows(
            evaluation,
            days,
            config,
            models,
            rnn,
            sequence_length,
        )
        forecast_rows.extend(day_forecasts)
        forecast_scenario_rows.extend(scenario_forecasts)
        forecast_floor_rows.extend(floor_forecasts)
        day_controllers, floor_controllers = _controller_rows(
            evaluation, days, config, models, protocol
        )
        controller_rows.extend(day_controllers)
        controller_floor_rows.extend(floor_controllers)

    forecast_by_day = pd.DataFrame(forecast_rows)
    forecast_summary = (
        forecast_by_day.groupby(["evaluation", "model"], as_index=False)
        .agg(
            days=("day_id", "count"),
            floor_mae=("floor_mae", "mean"),
            floor_rmse=("floor_rmse", "mean"),
            total_demand_mae=("total_demand_mae", "mean"),
            top3_floor_overlap=("top3_floor_overlap", "mean"),
            mean_bias=("mean_bias", "mean"),
        )
        .sort_values(["evaluation", "floor_mae"])
    )
    forecast_effects = _forecast_paired_effects(forecast_by_day, config)
    controller_by_day = pd.DataFrame(controller_rows)
    controller_summary = _controller_summary(controller_by_day)
    effect_frames = []
    for evaluation, group in controller_by_day.groupby(
        "evaluation", sort=False
    ):
        effects = _paired_controller_effects(group, config)
        effects.insert(0, "evaluation", evaluation)
        effect_frames.append(effects)
    controller_effects = pd.concat(effect_frames, ignore_index=True)

    forecast_by_day.to_csv(output / "final_forecast_metrics_by_day.csv", index=False)
    forecast_summary.to_csv(output / "final_forecast_summary.csv", index=False)
    forecast_effects.to_csv(output / "final_forecast_paired_effects.csv", index=False)
    pd.DataFrame(forecast_scenario_rows).to_csv(
        output / "final_forecast_scenario_metrics.csv", index=False
    )
    pd.DataFrame(forecast_floor_rows).to_csv(
        output / "final_forecast_floor_metrics.csv", index=False
    )
    controller_by_day.to_csv(
        output / "final_controller_metrics_by_day.csv", index=False
    )
    controller_summary.to_csv(output / "final_controller_summary.csv", index=False)
    controller_effects.to_csv(
        output / "final_controller_paired_effects.csv", index=False
    )
    pd.DataFrame(controller_floor_rows).to_csv(
        output / "final_controller_floor_metrics.csv", index=False
    )
    dump({**models, rnn.name: rnn}, output / "final_fitted_models.joblib")
    _plot_final_results(
        forecast_summary,
        controller_effects,
        output / "final_comparison.png",
    )
    day_manifest = [
        {
            "evaluation": evaluation,
            "day_id": day.day_id,
            "scenario": day.scenario,
            "seed": day.seed,
            "passengers": len(day.passengers),
        }
        for evaluation, days in final_days.items()
        for day in days
    ]
    pd.DataFrame(day_manifest).to_csv(output / "final_day_manifest.csv", index=False)
    (output / "config_snapshot.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "protocol_snapshot.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    (output / "seed_manifest_snapshot.json").write_text(
        json.dumps(seed_manifest, indent=2), encoding="utf-8"
    )
    completed_utc = datetime.now(timezone.utc).isoformat()
    completion = {
        "protocol_name": protocol["protocol_name"],
        "protocol_sha256": _sha256(protocol_path),
        "seed_offset": seed_offset,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "final_days": len(day_manifest),
        "run_once_guard": True,
        "post_holdout_tuning_permitted": False,
    }
    (output / "FINAL_RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    return output
