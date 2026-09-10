from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from elevator_ml.data.generator import create_days
from elevator_ml.data.sequences import build_sequence_dataset
from elevator_ml.forecasting.rnn import ElmanRNNForecaster


class RNNForecasterTests(unittest.TestCase):
    def _datasets(self):
        train_days = create_days("train", ("mixed",), 2, 100, 24, 5)
        validation_days = create_days("validation", ("mixed",), 1, 200, 24, 5)
        return (
            build_sequence_dataset(train_days, 4, 2, ("mixed",)),
            build_sequence_dataset(validation_days, 4, 2, ("mixed",)),
        )

    @staticmethod
    def _model(seed: int = 7) -> ElmanRNNForecaster:
        return ElmanRNNForecaster(
            hidden_size=6,
            learning_rate=0.005,
            batch_size=16,
            maximum_epochs=8,
            patience=3,
            gradient_clip_norm=2.0,
            l2_penalty=0.0001,
            random_state=seed,
        )

    def test_predictions_are_finite_nonnegative_and_shaped(self) -> None:
        train, validation = self._datasets()
        model = self._model().fit(train, validation)
        predictions = model.predict(validation)

        self.assertEqual(predictions.shape, validation.targets.shape)
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertTrue(np.all(predictions >= 0))
        self.assertIsNotNone(model.summary_)
        self.assertGreater(model.summary_.parameter_count, 0)

    def test_fixed_seed_is_deterministic(self) -> None:
        train, validation = self._datasets()
        first = self._model().fit(train, validation).predict(validation)
        second = self._model().fit(train, validation).predict(validation)

        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
