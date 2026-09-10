# Development Controller Findings

> This file records the initial always-on controller comparison. The subsequent
> cost ablation and stress tests are documented in
> [ROBUSTNESS_FINDINGS.md](ROBUSTNESS_FINDINGS.md); they supersede the controller
> selection decision below.

## Evaluation status

These results use the same 12 validation days used for development decisions.
They are diagnostic and must not be presented as final generalization results.
Every policy receives identical passenger trips and uses the identical
nearest-car hall-call assignment. Only idle-car positioning changes.

## Controller objective

At each minute, the greedy positioner assigns each service-idle elevator to the
floor with the highest score:

```text
score(floor) = predicted_demand(floor) / (1 + cars_already_reserved(floor))
               - distance_penalty * travel_distance_to_floor
```

The distance penalty was selected on validation days using mean passenger wait
as the primary criterion, then P95 wait and movement as tie-breakers. The chosen
value was `0.0`. A penalty of `0.6` used approximately 3% less movement and had
better P95 wait, but its mean wait was 0.101 seconds higher. These are two
different points on the service-versus-movement tradeoff.

## Validation comparison

| Policy | Mean wait (s) | P95 wait (s) | Long-wait rate | Floors travelled |
|---|---:|---:|---:|---:|
| Nearest-car, stay idle | 15.627 | **49.462** | 0.034 | **3,679** |
| Static zoning | 15.152 | 50.746 | 0.036 | 3,873 |
| Historical positioning | 15.295 | 51.950 | 0.037 | 3,938 |
| Ridge positioning | 15.220 | 51.554 | 0.037 | 3,963 |
| Random Forest positioning | 15.133 | 50.421 | 0.034 | 3,946 |
| Uncertainty-gated Random Forest | **15.091** | 50.254 | 0.034 | 3,946 |
| Oracle positioning | 14.808 | 49.846 | **0.033** | 3,945 |

## Paired effects relative to nearest-car

| Policy | Mean wait saved | 95% paired bootstrap interval | Wins / 12 days | Movement change |
|---|---:|---:|---:|---:|
| Static zoning | 0.475 s | -0.507 to 1.456 | 8 | +5.36% |
| Historical positioning | 0.333 s | -0.403 to 1.118 | 7 | +7.37% |
| Ridge positioning | 0.407 s | -0.376 to 1.171 | 8 | +8.04% |
| Random Forest positioning | 0.494 s | -0.181 to 1.241 | 8 | +7.56% |
| Uncertainty-gated Random Forest | 0.536 s | -0.120 to 1.249 | 8 | +7.56% |
| Oracle positioning | 0.819 s | 0.236 to 1.384 | 9 | +7.52% |

The learned-policy intervals cross zero on only 12 independent validation days.
The oracle interval does not, showing that future demand contains operationally
useful information under the current simulator, even though the learned
forecast has not yet extracted a conclusive service benefit.

## Key findings

### 1. Mean service improves, but tail service does not

Random Forest positioning reduces mean wait by 3.16%, but average daily P95
wait becomes 0.958 seconds worse. This provisionally contradicts hypothesis H2
and prevents us from describing the current controller as unconditionally
better.

### 2. Static zoning is a necessary operational baseline

Static zoning obtains a 3.04% mean improvement, almost matching Random Forest's
3.16%. The incremental mean advantage of Random Forest over static zoning is
only 0.019 seconds. A comparison against nearest-car alone would exaggerate the
value attributable specifically to ML.

### 3. Prediction quality and operational quality differ

Historical positioning has strong total-demand forecast accuracy but the worst
P95 service among the positioning policies. Random Forest has the best
per-floor forecast MAE and the best always-on learned controller mean. This is
consistent with, but does not prove, the value of spatially accurate demand.

### 4. Benefits depend strongly on traffic regime

- Morning: forecast positioning saves about 1.26 seconds because lobby demand
  is strong and predictable.
- Lunch: Random Forest saves about 1.61 seconds.
- Evening: Random Forest is approximately neutral.
- Mixed: Random Forest is about 0.93 seconds worse than nearest-car.

An always-on policy is therefore inappropriate. Traffic-aware gating or a
multi-objective positioning rule is justified.

### 5. The first uncertainty gate is directionally useful

The validation-derived gate disables predictive positioning during the 10% of
minutes with the highest Random Forest tree disagreement. It improves mean wait
from 15.133 to 15.091 seconds but does not reduce movement because the blocked
minutes seldom coincide with long speculative trips. Stress evaluation is
needed before judging its drift protection.

## Decision for the next phase

Do not lock the greedy controller yet. Run controller-focused ablations:

1. compare the four movement penalties as a Pareto frontier;
2. gate by traffic regime, especially protecting mixed traffic;
3. require a minimum forecast-demand advantage before moving a car;
4. vary the positioning interval and forecast horizon;
5. evaluate uncertainty gating under unseen surge and demand-scale shifts;
6. preserve static zoning as the serious non-ML baseline.
