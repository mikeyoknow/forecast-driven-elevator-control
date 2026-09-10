# Final Evaluation Protocol Lock

## Lock status

The protocol was locked at `2026-08-10T17:37:49Z`, before creation of
`configs/final_seed.json` and before creation of the original final-holdout
archive. Selected public artifacts are retained in `results/final_holdout/`.
The seed will be revealed only after the source manifest is recorded. Final
results will be generated once and will not be used for further tuning.

The disposable smoke test used six temporary days under `/private/tmp`. It
verified the complete runner, output schemas, plots, serialization, passenger
service invariants, capacity invariants, and the second-run refusal guard. Its
seed and results are not part of the study and must never be reported.

## Locked research questions

1. Does Random Forest reduce two-minute per-floor demand MAE relative to the
   historical-mean baseline on independent in-distribution building days?
2. Does forecast-driven idle-car positioning reduce mean passenger wait
   relative to nearest-car/stay-idle control?
3. Does any service benefit remain acceptable in P95 wait, long-wait rate,
   floor fairness, and elevator movement?
4. How do forecast and controller conclusions change under an unseen surge and
   a 1.5x demand-scale shift?

## Locked data protocol

Training and validation remain exactly as used during development:

- training: 10 days per scenario for morning, lunch, evening, and mixed;
- validation: three independent days per scenario;
- models fit on the original training split;
- validation is used only for the already completed hyperparameter selection
  and deterministic RNN early stopping.

The final seed is not stored in the development configuration. After seed
reveal, the runner will generate:

| Final evaluation | Scenarios | Days | Demand scale |
|---|---|---:|---:|
| In distribution | Morning, lunch, evening, mixed | 20 | 1.0x |
| Unseen surge | Surge | 8 | 1.0x |
| High demand | Morning, lunch, evening, mixed | 8 | 1.5x |

This gives 36 independent final building days. Every policy will receive the
same passenger trips within each day.

## Locked forecasting models

The final forecast table includes per-floor mean, historical mean,
persistence, Ridge, Random Forest, and the from-scratch Elman RNN.

- Ridge alpha: 1.0
- Random Forest: 120 trees, maximum depth 20, minimum leaf size 5
- Random state: 3404
- RNN: 10 time steps, 32 hidden units, validation early stopping with the
  development best epoch required to reproduce as epoch 9
- Target: per-floor passenger origins over the next two minutes
- Primary forecasting comparison: Random Forest versus historical mean

The primary forecast effect is the paired building-day reduction in per-floor
MAE. RMSE, total-demand MAE, bias, top-three overlap, scenario slices, and
floor slices are secondary diagnostics.

## Locked controller comparison

All policies use the same nearest-car hall-call assignment. Only idle-car
positioning changes.

1. Nearest-car with no repositioning - primary operational baseline.
2. Static zoning - serious non-ML controller baseline.
3. Historical-mean positioning - simple forecast-aware baseline.
4. Tuned Random Forest positioning - learned challenger.
5. Perfect-input greedy positioning - diagnostic only, not an optimal upper
   bound.

The learned positioner is locked to distance penalty 0.0, minimum score
advantage 0.5, and a five-minute positioning interval. The development
movement budget is 5% relative to nearest-car.

The scenario-label gate is excluded because the synthetic regime label is not
a deployable real-building input. The uncertainty gate is excluded because it
made the same five-minute decisions as the ungated selected controller.

## Locked operational endpoints

The primary controller effect is paired mean passenger wait saved by Random
Forest positioning relative to nearest-car. The runner also reports:

- P95 passenger wait and long-wait rate;
- journey time;
- per-floor wait and the floor fairness gap;
- floors travelled and capacity utilization;
- day-level wins;
- 95% paired building-day bootstrap intervals with 2,000 resamples.

A favorable mean alone will not be called a safe improvement. P95 behavior and
the 5% movement budget will be reported beside it. Stress results are reported
even if they contradict the in-distribution result.

## Run-once procedure

After this lock:

1. Record SHA-256 hashes in the original private protocol-lock archive.
2. Generate one new seed offset of at least 1,000,000.
3. Store the seed with this protocol's SHA-256 hash in
   `configs/final_seed.json`.
4. Execute exactly once:

```bash
PYTHONPATH=src python3 -m elevator_ml run-final-holdout \
  --config configs/development.json \
  --protocol configs/final_protocol.json \
  --seed configs/final_seed.json \
  --output reproduced/final_holdout
```

The runner refuses a nonempty output directory and writes
`FINAL_RUN_COMPLETE.json` last. After viewing the results, no model,
hyperparameter, feature, metric, controller, threshold, scenario, or seed may
be changed. Bugs that invalidate the run must be disclosed rather than silently
rerun.
