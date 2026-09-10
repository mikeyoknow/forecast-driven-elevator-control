# Post-Holdout Change Log

## Purpose

The final holdout was executed once on August 11, 2026. This file records any
subsequent source change so that the pre-run source manifest, active repository,
and original immutable result archive are not confused.

## 2026-08-11 - Type-only policy import correction

After the final run, an isolated `tests.test_policies` invocation exposed a
circular import between `elevator_ml.control.policies` and the eager imports in
`elevator_ml.simulation.__init__`.

The only post-run source change in the locked manifest is in
`src/elevator_ml/control/policies.py`:

- `TYPE_CHECKING` was added to the `typing` import;
- the imports of `Elevator` and `Passenger` were moved under
  `if TYPE_CHECKING:`.

The module already uses `from __future__ import annotations`, and these two
names occur only in annotations. The change therefore affects import order and
testability, not assignment scores, positioning decisions, simulation inputs,
model predictions, metric calculations, or saved final results.

The final holdout was not rerun. In the original research archive, the pre-run
source manifest reported exactly this one disclosed mismatch, while the
separate final-artifact manifest continued to verify the saved run. This public
release retains selected tables and snapshots rather than the fitted-model
binaries. The complete 74-test suite passes with the active source.
