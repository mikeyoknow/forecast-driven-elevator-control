# Untouched Final Holdout Results

## Chain of custody and validity

The final protocol and source were locked before seed reveal. The protocol was
locked on August 10, 2026 at `17:37:49Z`; seed offset `16575894` was revealed
on August 11 at `23:21:44Z`; and the run completed at
`23:22:16.139975Z`. The protocol SHA-256 was
`f7c13894a43efb025a6dfe88912dc6a2a9cbad702f20f61936d2923feee89f6a`.

The run contains 36 unique final days: 20 in-distribution, eight unseen-surge,
and eight 1.5x-demand days. It produced 216 forecast model-day rows and 180
controller policy-day rows. All policies saw identical passengers within a
day, all passengers were served, and no elevator exceeded capacity. The source
manifest still verifies after the run. The original research archive retained
a cryptographic manifest; this public release contains selected result tables
and snapshots in `results/final_holdout/`.

Post-run verification note: a behavior-neutral, type-only import-order fix was
later applied to `src/elevator_ml/control/policies.py`; the holdout was not
rerun. The original final-artifact manifest continued to verify, while the
pre-run source manifest reported this one disclosed mismatch. See
`docs/POST_HOLDOUT_CHANGELOG.md` for the exact scope and rationale.

## Primary forecasting result

| Final evaluation | Historical MAE | Random Forest MAE | Paired reduction | 95% CI | RF wins |
|---|---:|---:|---:|---:|---:|
| In distribution | 0.897 | **0.868** | 0.028 | 0.011 to 0.046 | 14 / 20 |
| Unseen surge | 1.657 | **1.642** | 0.015 | -0.009 to 0.041 | 4 / 8 |
| 1.5x demand | 1.240 | **1.162** | 0.078 | 0.037 to 0.126 | 7 / 8 |

The locked primary forecasting hypothesis is supported. Random Forest has a
small but statistically stable reduction in per-floor MAE on independent
in-distribution days. It also beats the historical baseline under higher
demand, although Ridge is the best high-demand model at MAE 1.094. Under the
unseen surge, Random Forest's advantage over historical mean is inconclusive;
persistence is best at MAE 1.476 and Ridge fails badly at 2.365.

The selected Elman RNN does not beat historical mean in distribution: MAE
0.905 versus 0.897, with paired difference interval -0.025 to 0.009. Its role
remains an informative negative result.

## Primary controller result

| In-distribution policy | Mean wait | P95 wait | Long-wait rate | Fairness gap | Movement |
|---|---:|---:|---:|---:|---:|
| Nearest-car | 16.023 s | 52.413 s | 4.05% | 19.203 s | 3,720 floors |
| Static zoning | **15.510 s** | **51.635 s** | **3.77%** | **17.139 s** | 3,928 floors |
| Historical positioning | 16.086 s | 53.650 s | 4.09% | 20.024 s | 3,779 floors |
| Random Forest positioning | 16.038 s | 53.580 s | 4.03% | 20.006 s | 3,780 floors |
| Perfect-input greedy | 15.651 s | 52.558 s | 3.82% | 19.065 s | 3,789 floors |

Random Forest positioning saves `-0.015` seconds relative to nearest-car: it
is 0.015 seconds worse on average. The paired 95% interval is -0.485 to 0.394
seconds, and it wins exactly 10 of 20 days. P95 wait is 1.168 seconds worse,
movement rises 1.75%, and the floor fairness gap rises 0.803 seconds.

The primary controller hypothesis is therefore not supported. Random Forest
forecasting is measurably more accurate, but feeding it to the locked greedy
positioner does not improve elevator service.

Static zoning has the best final in-distribution mean wait, saving 0.513
seconds, but its interval crosses zero (-0.148 to 1.242) and its 5.73% movement
increase exceeds the locked 5% budget. It is a strong non-ML baseline, not a
conclusive deployable winner.

The perfect-input greedy diagnostic saves 0.372 seconds, but its interval also
crosses zero and it slightly worsens P95 wait. Exact future origins are not
sufficient to make the current greedy controller reliably beneficial.

## Stress results

| Evaluation | RF mean wait saved | 95% CI | P95 saved | Movement change | Wins |
|---|---:|---:|---:|---:|---:|
| Unseen surge | 0.147 s | -1.218 to 1.726 | 1.131 s | +0.37% | 4 / 8 |
| 1.5x demand | 0.369 s | -0.001 to 0.934 | 2.650 s | +0.11% | 3 / 8 |

Neither stress result establishes robust mean-wait improvement. The unseen
surge estimate reverses the negative development stress estimate, but its wide
interval and four wins show instability rather than dependable recovery. The
high-demand mean estimate is favorable and its P95 improves, but the interval
barely includes zero and only three of eight days have lower mean wait.

Historical and Random Forest positioning produce exactly identical operational
metrics on every final surge and high-demand day. At the locked five-minute
interval and minimum-advantage threshold, their forecasts map to the same
repositioning decisions. This is direct evidence that forecast accuracy and
control action must be evaluated separately.

## Exploratory in-distribution scenario slice

Random Forest's largest forecasting advantage is lunch traffic: it reduces
MAE by 0.089 and wins all five lunch days. Yet its controller loses 0.111
seconds of mean wait on average during lunch. During morning traffic it lowers
forecast MAE slightly but worsens P95 wait by 5.50 seconds and adds 4.31%
movement. Evening is the only scenario with positive average mean-wait savings
(0.194 seconds), despite almost no forecast-MAE advantage.

These scenario slices are descriptive and were not used for further tuning.
They reinforce the main conclusion rather than defining a new policy.

## Final conclusion

The project demonstrates that successful demand prediction and successful
elevator control are different technical problems. Random Forest generalizes
as the best in-distribution forecaster, satisfying the forecasting hypothesis.
The greedy positioning layer fails to convert that accuracy into a reliable
service improvement, so the controller hypothesis is rejected.

This is not a failed project result. It is the central engineering finding:
offline prediction metrics cannot establish operational value. A future system
needs route- and destination-aware optimization, explicit tail-wait and energy
objectives, causal online drift detection, and calibration against real
building traffic before deployment.
