"""Uncertainty-aware association between forecast error and control value."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationInterval:
    method: str
    samples: int
    estimate: float
    lower: float
    upper: float
    permutation_p_value: float
    confidence: float
    resamples: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def _correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "spearman":
        x = _average_ranks(x)
        y = _average_ranks(y)
    elif method != "pearson":
        raise ValueError("Correlation method must be 'pearson' or 'spearman'.")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks without requiring SciPy."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def correlation_interval(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> CorrelationInterval:
    """Bootstrap a correlation CI and compute a paired permutation p-value."""
    first = np.asarray(x, dtype=float)
    second = np.asarray(y, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("Correlation inputs must be equal one-dimensional arrays.")
    if len(first) < 4:
        raise ValueError("Correlation analysis requires at least four observations.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Correlation inputs must be finite.")
    if resamples <= 0 or seed < 0:
        raise ValueError("Correlation resamples must be positive and seed non-negative.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Correlation confidence must be between zero and one.")

    estimate = _correlation(first, second, method)
    if not np.isfinite(estimate):
        raise ValueError("Correlation is undefined for a constant input.")
    rng = np.random.default_rng(seed)
    bootstrap_values = []
    for _ in range(resamples):
        indices = rng.integers(0, len(first), size=len(first))
        value = _correlation(first[indices], second[indices], method)
        if np.isfinite(value):
            bootstrap_values.append(value)
    if not bootstrap_values:
        raise RuntimeError("All bootstrap correlation samples were undefined.")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_values, [alpha, 1.0 - alpha])

    more_extreme = 0
    for _ in range(resamples):
        permuted = rng.permutation(second)
        value = _correlation(first, permuted, method)
        if np.isfinite(value) and abs(value) >= abs(estimate):
            more_extreme += 1
    p_value = (more_extreme + 1.0) / (resamples + 1.0)
    return CorrelationInterval(
        method=method,
        samples=len(first),
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        permutation_p_value=float(p_value),
        confidence=confidence,
        resamples=resamples,
    )
