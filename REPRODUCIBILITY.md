# Reproducibility Guide

## Scope

This public portfolio release contains the synthetic-data generator,
preprocessing, model training, controller evaluation, diagnostics, selected
development evidence, and selected untouched final-holdout results. No
external dataset, API key, or inference service is required.

Commands below assume the repository root and Python 3.11.

## Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The reported environment used Python 3.11.4, NumPy 1.26.2, pandas 2.1.4,
Matplotlib 3.8.2, scikit-learn 1.3.2, and joblib 1.3.2.

## Correctness Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The release state contains 74 tests. They cover configuration validation,
traffic generation, feature alignment and leakage guards, forecasting,
sequence preparation and the from-scratch RNN, policy behaviour, simulator
invariants, metrics, experiments, robustness, operational-value analysis, and
the locked final protocol.

## Development Workflows

Write new runs to fresh directories so the selected published evidence remains
unchanged:

```bash
PYTHONPATH=src python -m elevator_ml audit-data --output reproduced/data_audit
PYTHONPATH=src python -m elevator_ml train-forecasters --output reproduced/forecasting
PYTHONPATH=src python -m elevator_ml evaluate-controllers --output reproduced/controllers
PYTHONPATH=src python -m elevator_ml run-robustness --output reproduced/robustness
PYTHONPATH=src python -m elevator_ml train-sequence-models --output reproduced/sequence_models
PYTHONPATH=src python -m elevator_ml analyze-operational-value \
  --robustness-output reproduced/robustness \
  --output reproduced/operational_value
```

All workflows default to `configs/development.json`. Synthetic trips are
generated deterministically from the split-specific seeds in that file.

## Final-Holdout Reproduction

The public release preserves the locked protocol, seed configuration, selected
summary tables, day-level results, and completion snapshot. They are provided
for inspection rather than additional tuning.

For an independent clean reproduction, write to a new path:

```bash
PYTHONPATH=src python -m elevator_ml run-final-holdout \
  --config configs/development.json \
  --protocol configs/final_protocol.json \
  --seed configs/final_seed.json \
  --output reproduced/final_holdout
```

Do not treat the synthetic holdout as evidence of real-building readiness.

## Technical Report

A reviewed public report is included at
`docs/Forecast_Driven_Elevator_Control_Report.pdf`. Its LaTeX source is in
`report/` and contains only the authors' public names.

With Tectonic installed, rebuild it from the repository root:

```bash
cd report
tectonic -o ../docs main.tex
cd ..
mv docs/main.pdf docs/Forecast_Driven_Elevator_Control_Report.pdf
```

## Public Evidence Map

- `results/development/`: selected development tables.
- `results/final_holdout/`: selected final tables and locked snapshots.
- `results/figures/`: the principal plots used in the report and portfolio.
- `docs/FINAL_RESULTS.md`: complete readable interpretation of the final run.
- `docs/POST_HOLDOUT_CHANGELOG.md`: disclosed behaviour-neutral source change.
- `report/`: sanitized public LaTeX source and bibliography.
