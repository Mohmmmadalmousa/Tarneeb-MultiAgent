"""Statistics for paired evaluation: bootstrap CIs, Wilson intervals, variance.

Everything here is a pure function of the samples it is given, uses an explicit
seeded ``numpy.random.Generator``, and never touches global RNG state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95
STANDARD_NORMAL_95 = 1.959963984540054
"""The two-sided 95% normal quantile, hard-coded so SciPy is not needed."""


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A point estimate with a lower and upper bound."""

    mean: float
    lower_bound: float
    upper_bound: float
    sample_count: int
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def excludes_zero_above(self) -> bool:
        """True when the whole interval lies strictly above zero."""
        return self.lower_bound > 0.0

    @property
    def half_width(self) -> float:
        return (self.upper_bound - self.lower_bound) / 2.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
        }


def bootstrap_mean_confidence_interval(
    samples: np.ndarray | list[float],
    seed: int,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ConfidenceInterval:
    """Return a percentile bootstrap CI for the mean of ``samples``.

    Resampling draws with replacement from a generator seeded by ``seed``, so the
    interval is exactly reproducible.
    """
    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.ndim != 1:
        raise ValueError(f"samples must be one-dimensional, got shape {sample_array.shape}")
    if sample_array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    sample_count = sample_array.size
    resample_indices = rng.integers(0, sample_count, size=(resamples, sample_count))
    resample_means = sample_array[resample_indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower_bound, upper_bound = np.quantile(resample_means, [tail, 1.0 - tail])

    return ConfidenceInterval(
        mean=float(sample_array.mean()),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        sample_count=int(sample_count),
        confidence=confidence,
    )


def wilson_score_interval(
    win_count: int, trial_count: int, confidence: float = DEFAULT_CONFIDENCE
) -> ConfidenceInterval:
    """Return the Wilson score interval for a binomial proportion.

    Preferred over ``p +/- 1.96 * se`` because it stays inside ``[0, 1]`` and
    behaves sensibly at 0 wins, all wins, and small samples.
    """
    if trial_count < 1:
        raise ValueError(f"trial_count must be at least 1, got {trial_count}")
    if not 0 <= win_count <= trial_count:
        raise ValueError(f"win_count must be in 0..{trial_count}, got {win_count}")
    if confidence != DEFAULT_CONFIDENCE:
        raise ValueError("only the 95% Wilson interval is implemented")

    z = STANDARD_NORMAL_95
    observed_proportion = win_count / trial_count
    denominator = 1.0 + z * z / trial_count
    centre = (observed_proportion + z * z / (2 * trial_count)) / denominator
    spread = (
        z
        * math.sqrt(
            observed_proportion * (1 - observed_proportion) / trial_count
            + z * z / (4 * trial_count * trial_count)
        )
        / denominator
    )

    return ConfidenceInterval(
        mean=observed_proportion,
        lower_bound=max(0.0, centre - spread),
        upper_bound=min(1.0, centre + spread),
        sample_count=trial_count,
        confidence=confidence,
    )


def paired_score_differences(
    policy_a_scores: np.ndarray | list[float],
    policy_b_scores: np.ndarray | list[float],
) -> np.ndarray:
    """Return ``policy_a_scores - policy_b_scores`` elementwise.

    Positive means policy A outscored policy B on that pair.
    """
    first = np.asarray(policy_a_scores, dtype=np.float64)
    second = np.asarray(policy_b_scores, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(f"paired arrays must match: {first.shape} vs {second.shape}")
    return first - second


def sample_standard_deviation(samples: np.ndarray | list[float]) -> float:
    """Return the unbiased (n-1) standard deviation, the empirical ``sigma_d``."""
    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.size < 2:
        raise ValueError("need at least two samples for a standard deviation")
    return float(sample_array.std(ddof=1))


def required_sample_count(
    detectable_effect: float, sigma_d: float, confidence: float = DEFAULT_CONFIDENCE
) -> int:
    """Return the ``n`` whose 95% CI half-width equals ``detectable_effect``.

    ``n = (z * sigma_d / effect) ** 2``. Used to size later evaluations from the
    measured variance rather than from an estimate.
    """
    if detectable_effect <= 0:
        raise ValueError(f"detectable_effect must be positive, got {detectable_effect}")
    if sigma_d < 0:
        raise ValueError(f"sigma_d must be non-negative, got {sigma_d}")
    if confidence != DEFAULT_CONFIDENCE:
        raise ValueError("only the 95% sample size is implemented")
    return math.ceil((STANDARD_NORMAL_95 * sigma_d / detectable_effect) ** 2)
