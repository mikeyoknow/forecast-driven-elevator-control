"""A compact NumPy Elman RNN with full backpropagation through time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from elevator_ml.data.sequences import SequenceDataset


@dataclass(frozen=True)
class RNNTrainingSummary:
    epochs_trained: int
    best_epoch: int
    best_validation_loss: float
    gradient_clip_steps: int
    parameter_count: int


class ElmanRNNForecaster:
    """Many-to-one tanh RNN trained with Adam and validation early stopping."""

    name = "elman_rnn"

    def __init__(
        self,
        *,
        hidden_size: int,
        learning_rate: float,
        batch_size: int,
        maximum_epochs: int,
        patience: int,
        gradient_clip_norm: float,
        l2_penalty: float,
        random_state: int,
    ) -> None:
        if hidden_size <= 0 or batch_size <= 0 or maximum_epochs <= 0:
            raise ValueError("RNN sizes and epoch count must be positive.")
        if learning_rate <= 0 or patience <= 0 or gradient_clip_norm <= 0:
            raise ValueError("RNN optimizer controls must be positive.")
        if l2_penalty < 0 or random_state < 0:
            raise ValueError("RNN L2 penalty and random seed cannot be negative.")
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.maximum_epochs = int(maximum_epochs)
        self.patience = int(patience)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.l2_penalty = float(l2_penalty)
        self.random_state = int(random_state)
        self.parameters_: dict[str, np.ndarray] = {}
        self.input_mean_: np.ndarray | None = None
        self.input_scale_: np.ndarray | None = None
        self.target_mean_: np.ndarray | None = None
        self.target_scale_: np.ndarray | None = None
        self.history_: pd.DataFrame | None = None
        self.summary_: RNNTrainingSummary | None = None

    def _initialize(self, input_size: int, output_size: int) -> None:
        rng = np.random.default_rng(self.random_state)
        input_limit = np.sqrt(6.0 / (input_size + self.hidden_size))
        output_limit = np.sqrt(6.0 / (self.hidden_size + output_size))
        recurrent = rng.normal(size=(self.hidden_size, self.hidden_size))
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        recurrent *= 0.90 / max(spectral_radius, 1e-8)
        self.parameters_ = {
            "input_weight": rng.uniform(
                -input_limit,
                input_limit,
                size=(input_size, self.hidden_size),
            ),
            "recurrent_weight": recurrent,
            "hidden_bias": np.zeros(self.hidden_size, dtype=float),
            "output_weight": rng.uniform(
                -output_limit,
                output_limit,
                size=(self.hidden_size, output_size),
            ),
            "output_bias": np.zeros(output_size, dtype=float),
        }

    def _scale_inputs(self, inputs: np.ndarray) -> np.ndarray:
        if self.input_mean_ is None or self.input_scale_ is None:
            raise RuntimeError("RNN input scaler has not been fitted.")
        return (inputs - self.input_mean_) / self.input_scale_

    def _scale_targets(self, targets: np.ndarray) -> np.ndarray:
        if self.target_mean_ is None or self.target_scale_ is None:
            raise RuntimeError("RNN target scaler has not been fitted.")
        return (targets - self.target_mean_) / self.target_scale_

    def _forward(
        self, inputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        batch, steps, _ = inputs.shape
        hidden = np.zeros(
            (batch, steps + 1, self.hidden_size), dtype=float
        )
        for step in range(steps):
            hidden[:, step + 1] = np.tanh(
                inputs[:, step] @ self.parameters_["input_weight"]
                + hidden[:, step] @ self.parameters_["recurrent_weight"]
                + self.parameters_["hidden_bias"]
            )
        outputs = (
            hidden[:, -1] @ self.parameters_["output_weight"]
            + self.parameters_["output_bias"]
        )
        return outputs, hidden

    def _loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        predictions, _ = self._forward(inputs)
        mse = float(np.mean((predictions - targets) ** 2))
        regularization = self.l2_penalty * sum(
            float(np.sum(self.parameters_[name] ** 2))
            for name in (
                "input_weight",
                "recurrent_weight",
                "output_weight",
            )
        )
        return mse + regularization

    def _gradients(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float]:
        predictions, hidden = self._forward(inputs)
        output_gradient = 2.0 * (predictions - targets) / targets.size
        gradients = {
            name: np.zeros_like(value) for name, value in self.parameters_.items()
        }
        gradients["output_weight"] = hidden[:, -1].T @ output_gradient
        gradients["output_bias"] = output_gradient.sum(axis=0)
        hidden_gradient = output_gradient @ self.parameters_["output_weight"].T

        for step in range(inputs.shape[1] - 1, -1, -1):
            activation_gradient = hidden_gradient * (
                1.0 - hidden[:, step + 1] ** 2
            )
            gradients["input_weight"] += inputs[:, step].T @ activation_gradient
            gradients["recurrent_weight"] += (
                hidden[:, step].T @ activation_gradient
            )
            gradients["hidden_bias"] += activation_gradient.sum(axis=0)
            hidden_gradient = (
                activation_gradient @ self.parameters_["recurrent_weight"].T
            )

        for name in (
            "input_weight",
            "recurrent_weight",
            "output_weight",
        ):
            gradients[name] += 2.0 * self.l2_penalty * self.parameters_[name]
        norm = float(
            np.sqrt(sum(float(np.sum(value**2)) for value in gradients.values()))
        )
        if norm > self.gradient_clip_norm:
            factor = self.gradient_clip_norm / (norm + 1e-12)
            gradients = {name: value * factor for name, value in gradients.items()}
        return gradients, norm

    def fit(
        self,
        train: SequenceDataset,
        validation: SequenceDataset,
    ) -> "ElmanRNNForecaster":
        if train.n_features != validation.n_features:
            raise ValueError("Train and validation sequence features differ.")
        if train.n_targets != validation.n_targets:
            raise ValueError("Train and validation sequence targets differ.")
        if train.sequence_length != validation.sequence_length:
            raise ValueError("Train and validation sequence lengths differ.")

        flattened = train.inputs.reshape(-1, train.n_features)
        self.input_mean_ = flattened.mean(axis=0)
        input_scale = flattened.std(axis=0)
        self.input_scale_ = np.where(input_scale < 1e-8, 1.0, input_scale)
        self.target_mean_ = train.targets.mean(axis=0)
        target_scale = train.targets.std(axis=0)
        self.target_scale_ = np.where(target_scale < 1e-8, 1.0, target_scale)
        train_inputs = self._scale_inputs(train.inputs)
        validation_inputs = self._scale_inputs(validation.inputs)
        train_targets = self._scale_targets(train.targets)
        validation_targets = self._scale_targets(validation.targets)
        self._initialize(train.n_features, train.n_targets)

        first_moment = {
            name: np.zeros_like(value) for name, value in self.parameters_.items()
        }
        second_moment = {
            name: np.zeros_like(value) for name, value in self.parameters_.items()
        }
        rng = np.random.default_rng(self.random_state)
        best_loss = float("inf")
        best_epoch = 0
        best_parameters = {
            name: value.copy() for name, value in self.parameters_.items()
        }
        epochs_without_improvement = 0
        optimizer_step = 0
        gradient_clip_steps = 0
        rows: list[dict[str, float | int]] = []

        for epoch in range(1, self.maximum_epochs + 1):
            permutation = rng.permutation(train.n_samples)
            epoch_max_gradient = 0.0
            for start in range(0, train.n_samples, self.batch_size):
                indices = permutation[start : start + self.batch_size]
                gradients, gradient_norm = self._gradients(
                    train_inputs[indices], train_targets[indices]
                )
                epoch_max_gradient = max(epoch_max_gradient, gradient_norm)
                if gradient_norm > self.gradient_clip_norm:
                    gradient_clip_steps += 1
                optimizer_step += 1
                for name, gradient in gradients.items():
                    first_moment[name] = (
                        0.9 * first_moment[name] + 0.1 * gradient
                    )
                    second_moment[name] = (
                        0.999 * second_moment[name] + 0.001 * gradient**2
                    )
                    corrected_first = first_moment[name] / (
                        1.0 - 0.9**optimizer_step
                    )
                    corrected_second = second_moment[name] / (
                        1.0 - 0.999**optimizer_step
                    )
                    self.parameters_[name] -= self.learning_rate * (
                        corrected_first / (np.sqrt(corrected_second) + 1e-8)
                    )

            train_loss = self._loss(train_inputs, train_targets)
            validation_loss = self._loss(
                validation_inputs, validation_targets
            )
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "maximum_gradient_norm_before_clip": epoch_max_gradient,
                }
            )
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_epoch = epoch
                best_parameters = {
                    name: value.copy()
                    for name, value in self.parameters_.items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.patience:
                break

        self.parameters_ = best_parameters
        self.history_ = pd.DataFrame(rows)
        parameter_count = sum(value.size for value in self.parameters_.values())
        self.summary_ = RNNTrainingSummary(
            epochs_trained=len(rows),
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            gradient_clip_steps=gradient_clip_steps,
            parameter_count=parameter_count,
        )
        return self

    def predict(self, dataset: SequenceDataset) -> np.ndarray:
        if self.summary_ is None:
            raise RuntimeError("ElmanRNNForecaster must be fitted first.")
        scaled_predictions, _ = self._forward(
            self._scale_inputs(dataset.inputs)
        )
        if self.target_mean_ is None or self.target_scale_ is None:
            raise RuntimeError("RNN target scaler has not been fitted.")
        predictions = (
            scaled_predictions * self.target_scale_ + self.target_mean_
        )
        return np.clip(predictions, 0.0, None)
