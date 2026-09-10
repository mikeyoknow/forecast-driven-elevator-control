# Forecast Accuracy Versus Operational Value

## Evaluation status

This analysis joins forecast errors to the already generated paired controller
outcomes for 20 independent development days: 12 validation days, four unseen
surge days, and four 1.5x-demand days. It reuses the locked development
controller parameters and performs no additional tuning. The final holdout is
still untouched.

Associations use an independent building day as the unit of analysis. We
report Pearson and Spearman estimates, 95% day-bootstrap intervals, and
two-sided permutation p-values. With only four days in each stress set, the
stress-specific intervals are necessarily very wide.

## Distribution shift is visible in forecast error

| Evaluation | Days | Random Forest MAE | Historical MAE | RF mean wait saved | Perfect-input heuristic saved |
|---|---:|---:|---:|---:|---:|
| Validation | 12 | 0.878 | 0.893 | 0.382 s | 0.249 s |
| Unseen surge | 4 | 1.551 | 1.581 | -1.404 s | -1.547 s |
| 1.5x demand | 4 | 1.184 | 1.251 | -0.122 s | 0.058 s |

Random Forest error rises by approximately 77% from validation to unseen
surge and 35% under 1.5x demand. This confirms a measurable forecasting
distribution shift. It does not, by itself, explain the control failure.

The perfect-input result is not a globally optimal elevator controller. It is
the same greedy positioner supplied with exact future origin counts. Random
Forest can occasionally outperform it because imperfect predictions may
accidentally suppress harmful speculative moves. We therefore describe it as
a perfect-input diagnostic rather than an oracle upper bound.

## Forecast MAE is not a dependable proxy for elevator benefit

Across all 20 heterogeneous development days, the Pearson association between
Random Forest per-floor MAE and mean-wait savings is -0.476. Its bootstrap
interval is -0.747 to 0.037 and its permutation p-value is 0.034. The rank
association is weaker at -0.358, with interval -0.741 to 0.159 and p = 0.123.

These statistics must not be simplified into a significant universal
relationship. The pooled Pearson trend is strongly influenced by grouping
validation and stress conditions together. Within the 12 validation days,
Pearson correlation is -0.313 with interval -0.722 to 0.282 and p = 0.329.
Top-three floor overlap is also unrelated to wait savings in the pooled data
(Pearson r = 0.095, p = 0.676).

The defensible interpretation is that large distribution shifts coincide with
both larger forecast errors and worse service, but ordinary day-to-day changes
in forecast MAE do not reliably predict operational value. Forecast metrics
cannot replace simulation-based controller evaluation.

## Diagnostic failure attribution

We use the following transparent diagnostic labels:

- **Benefit realized:** both Random Forest and the perfect-input greedy
  positioner improve on nearest-car.
- **Forecast limited:** perfect-input positioning improves service while the
  Random Forest controller does not.
- **Controller limited:** even the perfect-input greedy positioner fails to
  improve on nearest-car.

| Evaluation | Benefit realized | Forecast limited | Controller limited |
|---|---:|---:|---:|
| Validation | 6 | 2 | 4 |
| Unseen surge | 0 | 0 | 4 |
| 1.5x demand | 2 | 1 | 1 |
| **Total** | **8** | **3** | **9** |

These are diagnostic, not causal, categories. Nevertheless, all four surge
days being controller-limited is strong evidence that simply replacing the
forecaster will not repair the surge failure. The greedy policy lacks
route-level optimization, destination-aware capacity planning, and an explicit
tail-wait objective.

## Protocol decision

Random Forest remains the learned forecasting challenger because it has the
best development per-floor MAE. The final controller comparison will retain:

1. nearest-car with no repositioning as the primary operational baseline;
2. static zoning as the serious non-ML control baseline;
3. historical-mean positioning as a simple forecast-aware baseline;
4. the validation-selected Random Forest positioner with distance penalty
   0.0, minimum score advantage 0.5, and five-minute update interval; and
5. perfect-input greedy positioning strictly as a diagnostic.

The scenario-label gate will not be treated as deployable because its access to
the synthetic regime label would not be available in the same form in a real
building. The uncertainty gate is also excluded because it produced identical
decisions to the selected Random Forest at five-minute update times.

The final claims will report mean wait, P95 wait, long-wait rate, movement,
fairness, paired confidence intervals, and stress failures. A lower forecast
MAE alone will never be described as proof of a better elevator controller.
