"""Command-line interface for reproducible project workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from elevator_ml.evaluation.experiments import (
    run_data_audit,
    run_controller_experiment,
    run_forecasting_experiment,
    run_robustness_experiment,
    run_sequence_experiment,
    run_operational_value_analysis,
)
from elevator_ml.evaluation.final_holdout import run_final_holdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elevator-ml",
        description="Forecast-driven elevator-control experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit-data",
        help="Generate training/validation data and exploratory diagnostics.",
    )
    audit.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.json"),
    )
    audit.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/development_data_audit"),
    )
    forecast = subparsers.add_parser(
        "train-forecasters",
        help="Tune and compare demand forecasters on development data.",
    )
    forecast.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.json"),
    )
    forecast.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/development_forecasting"),
    )
    controllers = subparsers.add_parser(
        "evaluate-controllers",
        help="Tune and compare idle-car positioning policies on validation days.",
    )
    controllers.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.json"),
    )
    controllers.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/development_controllers"),
    )
    robustness = subparsers.add_parser(
        "run-robustness",
        help="Tune controller cost controls and run development stress tests.",
    )
    robustness.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.json"),
    )
    robustness.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/development_robustness"),
    )
    sequence = subparsers.add_parser(
        "train-sequence-models",
        help="Tune a from-scratch Elman RNN on development data.",
    )
    sequence.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.json"),
    )
    sequence.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/development_sequence_models"),
    )
    value = subparsers.add_parser(
        "analyze-operational-value",
        help="Relate day-level forecast errors to elevator-control outcomes.",
    )
    value.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.json"),
    )
    value.add_argument(
        "--robustness-output",
        type=Path,
        default=Path("outputs/development_robustness"),
    )
    value.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/development_operational_value"),
    )
    final = subparsers.add_parser(
        "run-final-holdout",
        help="Execute the locked, run-once final evaluation.",
    )
    final.add_argument(
        "--config",
        type=Path,
        default=Path("configs/development.json"),
    )
    final.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/final_protocol.json"),
    )
    final.add_argument(
        "--seed",
        type=Path,
        default=Path("configs/final_seed.json"),
    )
    final.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/final_holdout"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-data":
        output = run_data_audit(args.config, args.output)
        print(f"Data audit written to {output.resolve()}")
        return 0
    if args.command == "train-forecasters":
        output = run_forecasting_experiment(args.config, args.output)
        print(f"Forecasting results written to {output.resolve()}")
        return 0
    if args.command == "evaluate-controllers":
        output = run_controller_experiment(args.config, args.output)
        print(f"Controller results written to {output.resolve()}")
        return 0
    if args.command == "run-robustness":
        output = run_robustness_experiment(args.config, args.output)
        print(f"Robustness results written to {output.resolve()}")
        return 0
    if args.command == "train-sequence-models":
        output = run_sequence_experiment(args.config, args.output)
        print(f"Sequence-model results written to {output.resolve()}")
        return 0
    if args.command == "analyze-operational-value":
        output = run_operational_value_analysis(
            args.config, args.robustness_output, args.output
        )
        print(f"Operational-value analysis written to {output.resolve()}")
        return 0
    if args.command == "run-final-holdout":
        output = run_final_holdout(
            args.config, args.protocol, args.seed, args.output
        )
        print(f"Final holdout written to {output.resolve()}")
        return 0
    raise RuntimeError(f"Unhandled command {args.command!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
