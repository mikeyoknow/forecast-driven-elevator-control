# Development Data Audit Findings

## Scope

This audit uses training and validation data only. Development-test, stress,
and final-holdout data were not generated or inspected by the audit command.

## Split sizes

| Split | Days | Passenger events |
|---|---:|---:|
| Training | 40 | 30,448 |
| Validation | 12 | 9,109 |

Each of the four normal traffic regimes contributes 10 training days and 3
validation days. Seed ranges are disjoint and verified programmatically.

## Forecast tables

| Split | Horizon | Samples | Features | Targets |
|---|---:|---:|---:|---:|
| Training | 1 minute | 2,600 | 101 | 15 |
| Training | 2 minutes | 2,560 | 101 | 15 |
| Training | 5 minutes | 2,440 | 101 | 15 |
| Validation | 1 minute | 780 | 101 | 15 |
| Validation | 2 minutes | 768 | 101 | 15 |
| Validation | 5 minutes | 732 | 101 | 15 |

The decreasing sample count at longer horizons is correct: more minutes near
the end of each day lack a complete future target window.

## Traffic-regime checks

- **Morning:** 83.8% of training passenger trips originate at the lobby,
  matching an up-peak arrival pattern.
- **Evening:** 79.6% terminate at the lobby, matching down-peak departures.
- **Lunch:** lobby origin and destination shares are 42.9% and 39.1%, showing
  expected bidirectional flow.
- **Mixed:** both lobby shares are approximately 30%, leaving more inter-floor
  traffic than the peak regimes.
- The time-profile plot places the morning peak earlier and evening peak later
  in their sessions, while lunch peaks near the middle.

## Train-validation comparison

Average passengers per minute remain close between training and validation for
every scenario:

| Scenario | Train | Validation |
|---|---:|---:|
| Morning | 10.808 | 11.018 |
| Lunch | 10.335 | 10.484 |
| Evening | 11.317 | 11.227 |
| Mixed | 8.137 | 7.756 |

The validation split therefore differs through independent daily randomness
without showing an unintended regime shift. Deliberate distribution shift is
reserved for separate stress experiments.

## Modeling implications

1. Lobby demand dominates the aggregate training set, so overall MAE alone may
   hide poor upper-floor forecasts. Per-floor and top-demand-floor metrics are
   required.
2. Traffic regime is strongly informative, but recent histories remain needed
   to capture within-regime variation and unexpected local bursts.
3. The 101-column feature representation is modest relative to 2,560 primary
   training samples. Regularized linear and tree baselines are appropriate.
4. Validation has only 12 independent days. Final uncertainty must be computed
   by resampling days, not the 768 correlated minute rows.
5. The synthetic OD matrix is intentionally structured. External calibration
   remains necessary before claiming real-building effectiveness.

## Leakage audit

- The longest ten-minute history begins no earlier than minute zero.
- Every input window ends at or before its target begins.
- No sequence crosses a building-day boundary.
- Scaling and learned preprocessing have not yet been fitted.
- Scenario categories are defined from the training regimes only.
- Development-test and final-test seeds are absent from this audit workflow.
