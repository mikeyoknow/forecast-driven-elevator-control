"""Dataset manifests and exploratory traffic summaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elevator_ml.simulation.entities import DayData


def day_manifest(days: Sequence[DayData]) -> pd.DataFrame:
    rows = []
    for day in days:
        rows.append(
            {
                "day_id": day.day_id,
                "split": day.split,
                "scenario": day.scenario,
                "seed": day.seed,
                "duration_minutes": day.duration_minutes,
                "passengers": len(day.passengers),
                "passengers_per_minute": len(day.passengers)
                / day.duration_minutes,
                "lobby_origin_share": (
                    float(day.origin_counts[:, 0].sum()) / len(day.passengers)
                    if day.passengers
                    else 0.0
                ),
                "lobby_destination_share": (
                    float(day.destination_counts[:, 0].sum()) / len(day.passengers)
                    if day.passengers
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def minute_demand_frame(days: Sequence[DayData]) -> pd.DataFrame:
    rows = []
    for day in days:
        for minute in range(day.duration_minutes):
            rows.append(
                {
                    "day_id": day.day_id,
                    "split": day.split,
                    "scenario": day.scenario,
                    "minute": minute,
                    "total_arrivals": float(day.origin_counts[minute].sum()),
                    "lobby_arrivals": float(day.origin_counts[minute, 0]),
                    "upper_floor_arrivals": float(
                        day.origin_counts[minute, 1:].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def scenario_summary(days: Sequence[DayData]) -> pd.DataFrame:
    manifest = day_manifest(days)
    return (
        manifest.groupby(["split", "scenario"], as_index=False)
        .agg(
            days=("day_id", "count"),
            passenger_events=("passengers", "sum"),
            mean_passengers_per_day=("passengers", "mean"),
            std_passengers_per_day=("passengers", "std"),
            mean_passengers_per_minute=("passengers_per_minute", "mean"),
            mean_lobby_origin_share=("lobby_origin_share", "mean"),
            mean_lobby_destination_share=("lobby_destination_share", "mean"),
        )
        .fillna(0.0)
        .sort_values(["split", "scenario"])
    )


def plot_data_audit(days: Sequence[DayData], output_path: Path) -> None:
    """Create a four-panel traffic audit for train and validation days."""
    if not days:
        raise ValueError("Cannot plot an empty collection of days.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    minute_frame = minute_demand_frame(days)
    manifest = day_manifest(days)
    train_days = [day for day in days if day.split == "train"]
    if not train_days:
        raise ValueError("The data-audit plot requires training days.")
    n_floors = train_days[0].origin_counts.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    train_minutes = minute_frame[minute_frame["split"] == "train"]
    for scenario, group in train_minutes.groupby("scenario"):
        profile = group.groupby("minute")["total_arrivals"].mean()
        axes[0, 0].plot(profile.index, profile.values, label=scenario)
    axes[0, 0].set_title("Mean training demand over session")
    axes[0, 0].set_xlabel("Minute")
    axes[0, 0].set_ylabel("Passenger arrivals")
    axes[0, 0].legend(ncol=2)

    order = sorted(manifest["scenario"].unique())
    distributions = [
        manifest[manifest["scenario"] == scenario]["passengers"].to_numpy()
        for scenario in order
    ]
    axes[0, 1].boxplot(distributions, labels=order)
    axes[0, 1].set_title("Daily passenger-count variation")
    axes[0, 1].set_ylabel("Passengers per day")
    axes[0, 1].tick_params(axis="x", rotation=20)

    origin_counts = np.sum(
        [day.origin_counts.sum(axis=0) for day in train_days], axis=0
    )
    axes[1, 0].bar(np.arange(n_floors), origin_counts, color="#2563eb")
    axes[1, 0].set_title("Training origin-floor distribution")
    axes[1, 0].set_xlabel("Origin floor")
    axes[1, 0].set_ylabel("Passenger events")

    od_matrix = np.zeros((n_floors, n_floors), dtype=float)
    for day in train_days:
        for passenger in day.passengers:
            od_matrix[passenger.origin, passenger.destination] += 1.0
    image = axes[1, 1].imshow(od_matrix, cmap="viridis", aspect="auto")
    axes[1, 1].set_title("Training origin-destination matrix")
    axes[1, 1].set_xlabel("Destination floor")
    axes[1, 1].set_ylabel("Origin floor")
    fig.colorbar(image, ax=axes[1, 1], label="Passenger events")

    fig.suptitle(
        "Development data audit (training and validation only)",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
