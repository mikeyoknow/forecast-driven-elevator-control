# Development Sequence-Model Findings

## Evaluation status

These experiments use only the existing training and validation building days.
They do not access development-test or final-holdout seeds. Hyperparameters are
chosen by two-minute validation per-floor MAE, so all results in this document
are development evidence rather than final generalization estimates.

## Why implement the RNN from scratch

The project environment has no PyTorch, TensorFlow, or Keras dependency. We
implemented a many-to-one Elman RNN directly in NumPy so that the recurrent
state, forward equations, backpropagation through time, gradient control, and
optimizer behavior remain visible and reproducible. This also avoids making
the project a wrapper around a prebuilt model.

For a batch of sequences, the tensors are:

```text
input X:  (batch, time steps, 37 observable features)
hidden h: (batch, time steps + 1, hidden units)
output y: (batch, 15 floors)
```

Each time step contains per-floor origin and destination counts, cyclical time,
elapsed-session fraction, and the known traffic-regime indicator. A sequence
ends strictly before its target starts. Samples never cross building-day
boundaries. The target is the per-floor origin count over the next two minutes.

Input and target scalers are fitted on training sequences only. The model uses
a tanh recurrent state, linear multi-output head, Adam optimization, full
sequence backpropagation through time, L2 regularization, global-norm gradient
clipping, and validation-only early stopping with restoration of the best
epoch.

## Controlled architecture ablation

| Sequence | Hidden units | Parameters | Best epoch | Epochs run | Runtime (s) | Validation MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 5 minutes | 16 | 1,119 | 20 | 35 | 0.61 | 0.925 |
| 10 minutes | 16 | 1,119 | 20 | 35 | 1.04 | 0.920 |
| 5 minutes | 32 | 2,735 | 9 | 24 | 0.58 | 0.916 |
| 10 minutes | 32 | 2,735 | 9 | 24 | 1.10 | **0.909** |

The larger state and longer sequence both help slightly. The selected model is
the 10-minute, 32-unit RNN. It trained for 24 epochs and restored epoch 9 after
validation loss began rising. Its largest observed batch-gradient norm stayed
below the configured clipping threshold of 5.0, so zero clipping interventions
were required. This is evidence of stable optimization, not evidence that
clipping was unnecessary as a safeguard.

## Fair model comparison

Every model below is evaluated on the exact same 768 validation windows and
targets.

| Model | Per-floor MAE | RMSE | Total-demand MAE | Top-3 overlap |
|---|---:|---:|---:|---:|
| Per-floor mean | 1.283 | 2.502 | 7.125 | 0.374 |
| Persistence | 1.207 | 2.060 | 6.421 | 0.387 |
| Historical mean | 0.893 | 1.415 | **4.511** | 0.351 |
| Ridge | 0.903 | 1.437 | 4.713 | **0.411** |
| Random Forest | **0.878** | **1.346** | 4.581 | 0.385 |
| Elman RNN | 0.909 | 1.475 | 4.780 | 0.386 |

The RNN is much better than the naive mean and persistence baselines, showing
that it learned meaningful temporal structure. It does not beat the historical
baseline or Random Forest. Relative to Random Forest, its validation per-floor
MAE is approximately 3.6% higher. Relative to the historical baseline, it is
approximately 1.8% higher.

The RNN has a smaller train-validation MAE gap than Random Forest (0.052 versus
0.239), but this lower overfitting does not compensate for its higher absolute
validation error. Random Forest has stronger fit to the useful nonlinear
signal in this small tabular dataset.

## Scenario diagnostics

| Scenario | Historical | Random Forest | Elman RNN |
|---|---:|---:|---:|
| Morning | **0.654** | 0.674 | 0.698 |
| Lunch | 0.964 | **0.892** | 0.951 |
| Evening | 1.108 | **1.105** | 1.132 |
| Mixed | 0.847 | **0.840** | 0.857 |

The RNN does not win any traffic-regime slice. Its greatest weakness relative
to the simple historical baseline is regular morning traffic, where known
session time and regime averages already encode the dominant lobby pattern.
The small dataset provides limited benefit for learning recurrence beyond the
engineered recent-history summaries used by Random Forest.

## Decision

Retain the RNN in the report as a reproducible sequence-learning comparison and
an informative negative result. Do not add it to the final elevator controller:
it is less accurate than Random Forest, does not win a scenario slice, and has
no current operational evidence that would justify expanding the controller
search space.

This result strengthens the project methodology. Model complexity is not
treated as evidence of quality; architecture, validation behavior, runtime,
parameter count, and failure to beat strong baselines are all reported.
