# Development Forecasting Findings

> The later recurrent-model comparison is documented in
> [SEQUENCE_MODEL_FINDINGS.md](SEQUENCE_MODEL_FINDINGS.md). It retains Random
> Forest as the development forecast model.

## Evaluation status

These results use training and validation days only. They guide model and
controller development and are not final-test estimates.

Learned-model hyperparameters were selected using per-floor MAE on the primary
two-minute validation task. Total-demand MAE was the tie-breaker. The selected
settings are:

- Ridge alpha: `1.0`
- Random Forest trees: `120`
- Random Forest maximum depth: `20`
- Random Forest minimum leaf size: `5`
- Random Forest seed: `3404`

## Primary two-minute validation comparison

| Model | Per-floor MAE | RMSE | Total-demand MAE | Top-3 floor overlap |
|---|---:|---:|---:|---:|
| Per-floor mean | 1.283 | 2.502 | 7.125 | 0.374 |
| Historical mean | 0.893 | 1.415 | **4.511** | 0.351 |
| Persistence | 1.207 | 2.060 | 6.421 | 0.387 |
| Ridge | 0.903 | 1.437 | 4.713 | **0.411** |
| Random Forest | **0.878** | **1.346** | 4.581 | 0.385 |

Random Forest reduces per-floor MAE by approximately 31.6% relative to the
per-floor mean baseline and 1.7% relative to the strong historical baseline.
The modest improvement over historical mean is important context: recurring
traffic regime and session time explain much of the predictable signal.

No model dominates every metric:

- Random Forest has the best per-floor MAE and RMSE.
- Historical mean has the best total-demand MAE.
- Ridge has the best top-three high-demand-floor overlap.
- Persistence is substantially weaker, especially at longer horizons.

This validates the project's distinction between forecast accuracy and
operational value. The correct model for elevator positioning must be selected
through simulator outcomes, not one forecasting metric alone.

## Horizon behavior

| Horizon | Best per-floor MAE | Model |
|---:|---:|---|
| 1 minute | 0.582 | Random Forest |
| 2 minutes | 0.878 | Random Forest |
| 5 minutes | 1.578 | Random Forest |

Absolute count error grows with the number of future minutes being predicted.
Historical mean remains very competitive at all horizons, while mean and
persistence baselines degrade sharply at five minutes.

## Scenario behavior at two minutes

| Scenario | Historical | Ridge | Random Forest |
|---|---:|---:|---:|
| Morning | **0.654** | 0.655 | 0.674 |
| Lunch | 0.964 | 0.978 | **0.892** |
| Evening | 1.108 | 1.125 | **1.105** |
| Mixed | 0.847 | 0.853 | **0.840** |

Random Forest's largest useful advantage appears during lunch traffic. It does
not beat the simpler models during morning traffic, where the lobby-heavy
pattern is regular and easy for historical averages to represent.

## Generalization gap

| Model | Train MAE | Validation MAE | Gap |
|---|---:|---:|---:|
| Per-floor mean | 1.266 | 1.283 | 0.016 |
| Historical mean | 0.863 | 0.893 | 0.030 |
| Persistence | 1.226 | 1.207 | -0.019 |
| Ridge | 0.831 | 0.903 | 0.072 |
| Random Forest | 0.638 | 0.878 | 0.239 |

Random Forest has the largest train-validation gap. Its minimum-leaf tuning
limits but does not eliminate overfitting. This is a reason to preserve the
historical and Ridge models in the controller comparison rather than selecting
Random Forest automatically.

## Feature interpretation

The most influential Random Forest features are lobby histories over three,
five, and ten minutes, elapsed session time, cyclical time, and total recent
demand. This agrees with the known morning/evening regimes but also reflects the
dataset's lobby imbalance. Per-floor diagnostics remain necessary to ensure
upper floors are not ignored.

## Decision for the next phase

Carry historical mean, Ridge, and Random Forest into the elevator-positioning
experiment. Keep per-floor mean and persistence in the forecasting table as
required sanity baselines. Do not declare a winning deployable model until
paired passenger-service and elevator-movement outcomes are available.
