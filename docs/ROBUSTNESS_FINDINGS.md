# Controller Ablation and Development Robustness Findings

## Evaluation status

These are development results. Controller parameters and gates were chosen or
diagnosed using the same 12 validation days, so validation confidence intervals
measure day-to-day variability but are not final generalization evidence. The
four unseen-surge days and four 1.5x-demand days are development stress tests.
No development-test or final-holdout seeds were accessed.

Selected public evidence is in `results/development/`; the complete result set
can be regenerated into `reproduced/robustness`. The original run
contains 120 policy-day rows, 36 ablation rows, and 20 independent building
days. Every policy received identical passengers for a given day, all
passengers were served, and peak elevator load never exceeded capacity.

## Ablation design and selection rule

The Random Forest positioner was evaluated over:

- four distance penalties: 0.00, 0.15, 0.35, and 0.60;
- three minimum score advantages: 0.0, 0.5, and 1.0; and
- three positioning intervals: 1, 2, and 5 minutes.

This produces 36 validation configurations. We imposed a maximum 5% mean
movement increase relative to nearest-car. The strict rule required positive
mean-wait savings and no increase in average daily P95 wait. No configuration
met all three conditions. The predeclared relaxed rule therefore remained
within the movement budget and minimized P95 degradation before using mean
wait as a tie-breaker.

The selected configuration is:

| Parameter | Selected value |
|---|---:|
| Distance penalty | 0.00 |
| Minimum score advantage | 0.50 |
| Positioning interval | 5 minutes |
| Movement budget | 5% |

On validation, it saved 0.382 seconds of mean wait, changed P95 wait by +0.008
seconds, and increased movement by 1.35%. This is substantially less movement
than the initial every-minute controller, which increased movement by 7.56%.
The result shows that update frequency is a meaningful systems parameter, not
an implementation detail.

## Main paired results

Positive wait savings mean the candidate is better than nearest-car. Movement
change is the mean paired percentage change across days.

| Evaluation | Policy | Mean wait saved (s) | 95% paired CI (s) | P95 wait saved (s) | Movement change |
|---|---|---:|---:|---:|---:|
| Validation | Tuned Random Forest | 0.382 | -0.023 to 0.859 | -0.008 | +1.35% |
| Validation | Regime-gated Random Forest | 0.363 | 0.071 to 0.767 | -0.633 | +0.97% |
| Unseen surge | Tuned Random Forest | -1.404 | -2.849 to -0.043 | -7.650 | +1.90% |
| Unseen surge | Regime-gated Random Forest | -1.404 | -2.849 to -0.043 | -7.650 | +1.90% |
| High demand | Tuned Random Forest | -0.122 | -1.298 to 0.929 | -0.288 | -0.51% |
| High demand | Static zoning | 1.004 | -1.126 to 3.753 | 6.350 | -0.62% |

The validation benefit of the tuned Random Forest is modest and its interval
crosses zero. The regime gate disables prediction on mixed-traffic days; it
has a positive validation interval and lower movement, but worsens P95 wait by
0.633 seconds. Because the weak mixed regime was identified on validation,
this is still a development hypothesis rather than final evidence.

## Stress-test interpretation

### Unseen surge is a demonstrated failure mode

All repositioning policies are worse than nearest-car during the unseen surge.
The tuned Random Forest adds 1.404 seconds of mean wait and 7.650 seconds of P95
wait. Its paired mean effect is negative on three of four days, and the 95%
bootstrap interval remains below zero despite the small stress sample.

Static zoning is also worse, so the problem is not unique to forecast error.
Most importantly, the oracle controller with perfect future arrival counts is
1.547 seconds worse than nearest-car. This isolates a controller limitation:
the current greedy positioner uses origin demand but does not optimize a full
route, account for destination flows, or preserve capacity for concentrated
surges. Better predictions alone cannot fix that control objective.

### High demand is inconclusive, not a claimed win

At 1.5x demand, tuned predictive positioning is approximately neutral: -0.122
seconds of mean-wait savings with a wide interval from -1.298 to 0.929 seconds.
Static zoning saves 1.004 seconds on average and improves P95 by 6.350 seconds,
but only four independent days make the interval inconclusive. This finding is
a reason to retain static zoning as a serious baseline in the final test.

### The uncertainty gate is currently ineffective

At the selected five-minute update interval, uncertainty-gated and always-on
Random Forest results are identical in every evaluated condition. The highest
uncertainty minutes rarely coincide with controller update times. Tree
disagreement alone is therefore not a sufficient fallback trigger. A useful
gate must combine regime or demand-shift evidence with the actual decision
times at which repositioning can occur.

## What this means for the project claim

The defensible development claim is not that ML universally makes elevators
faster. It is:

> Short-horizon spatial forecasts can provide small in-distribution mean-wait
> improvements at low extra movement, but a greedy forecast-driven positioner
> is brittle under abrupt traffic drift. Forecast accuracy, operational value,
> and safe control must be evaluated separately.

That conclusion directly supports the course emphasis on diagnostics,
uncertainty, failure modes, leakage-safe methodology, and engineering
reasoning. It also gives the final report a stronger story than presenting a
single favorable average.

## Decision before the final holdout

Do not open the final holdout yet. The planned sequence comparison and
forecast-to-control diagnostic are now complete; see
[SEQUENCE_MODEL_FINDINGS.md](SEQUENCE_MODEL_FINDINGS.md) and
[OPERATIONAL_VALUE_FINDINGS.md](OPERATIONAL_VALUE_FINDINGS.md). The remaining
step is to freeze the model, controller, metrics, and runner before generating
the final seeds.
