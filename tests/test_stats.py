"""Statistics tests.

Every expected value is hand-computed or derived from a property that must hold,
never by calling the function under test and recording what it returned.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tarneeb.evaluation.stats import (
    STANDARD_NORMAL_95,
    ConfidenceInterval,
    bootstrap_mean_confidence_interval,
    paired_score_differences,
    required_sample_count,
    sample_standard_deviation,
    wilson_score_interval,
)

# --- bootstrap --------------------------------------------------------------


def test_bootstrap_mean_is_the_sample_mean() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean 3.0 by hand
    interval = bootstrap_mean_confidence_interval(samples, seed=1, resamples=500)
    assert interval.mean == pytest.approx(3.0)
    assert interval.sample_count == 5


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    samples = np.arange(50, dtype=np.float64)
    first = bootstrap_mean_confidence_interval(samples, seed=42, resamples=1000)
    second = bootstrap_mean_confidence_interval(samples, seed=42, resamples=1000)
    assert first == second


def test_bootstrap_differs_with_a_different_seed() -> None:
    samples = np.arange(50, dtype=np.float64)
    first = bootstrap_mean_confidence_interval(samples, seed=1, resamples=1000)
    second = bootstrap_mean_confidence_interval(samples, seed=2, resamples=1000)
    assert (first.lower_bound, first.upper_bound) != (second.lower_bound, second.upper_bound)


def test_bootstrap_bounds_are_ordered() -> None:
    samples = np.random.default_rng(0).normal(5.0, 2.0, size=200)
    interval = bootstrap_mean_confidence_interval(samples, seed=3, resamples=2000)
    assert interval.lower_bound <= interval.mean <= interval.upper_bound


def test_constant_samples_give_a_degenerate_interval() -> None:
    """Every resample of a constant sample has the same mean."""
    interval = bootstrap_mean_confidence_interval([7.0] * 30, seed=1, resamples=500)
    assert interval.mean == pytest.approx(7.0)
    assert interval.lower_bound == pytest.approx(7.0)
    assert interval.upper_bound == pytest.approx(7.0)
    assert interval.half_width == pytest.approx(0.0)


def test_negating_the_samples_negates_the_interval() -> None:
    samples = np.array([1.0, -2.0, 3.5, 4.0, -0.5, 6.0])
    positive = bootstrap_mean_confidence_interval(samples, seed=9, resamples=1000)
    negative = bootstrap_mean_confidence_interval(-samples, seed=9, resamples=1000)

    assert negative.mean == pytest.approx(-positive.mean)
    assert negative.lower_bound == pytest.approx(-positive.upper_bound)
    assert negative.upper_bound == pytest.approx(-positive.lower_bound)


def test_clearly_positive_samples_exclude_zero() -> None:
    interval = bootstrap_mean_confidence_interval([5.0] * 40, seed=1, resamples=1000)
    assert interval.excludes_zero_above


def test_clearly_negative_samples_do_not_exclude_zero_above() -> None:
    interval = bootstrap_mean_confidence_interval([-5.0] * 40, seed=1, resamples=1000)
    assert not interval.excludes_zero_above


def test_bootstrap_interval_narrows_as_the_sample_grows() -> None:
    rng = np.random.default_rng(7)
    small = bootstrap_mean_confidence_interval(rng.normal(0, 1, 50), seed=1, resamples=2000)
    large = bootstrap_mean_confidence_interval(rng.normal(0, 1, 5000), seed=1, resamples=2000)
    assert large.half_width < small.half_width


@pytest.mark.parametrize(
    ("samples", "seed", "resamples", "match"),
    [
        ([], 1, 100, "empty sample"),
        ([1.0], 1, 0, "resamples must be"),
    ],
)
def test_bootstrap_input_validation(samples, seed, resamples, match) -> None:
    with pytest.raises(ValueError, match=match):
        bootstrap_mean_confidence_interval(samples, seed=seed, resamples=resamples)


def test_bootstrap_rejects_a_two_dimensional_sample() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        bootstrap_mean_confidence_interval(np.zeros((3, 3)), seed=1)


def test_bootstrap_rejects_an_impossible_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be"):
        bootstrap_mean_confidence_interval([1.0, 2.0], seed=1, confidence=1.5)


# --- Wilson interval --------------------------------------------------------


def test_wilson_at_fifty_fifty_is_centred_on_a_half() -> None:
    interval = wilson_score_interval(50, 100)
    assert interval.mean == pytest.approx(0.5)
    assert interval.lower_bound < 0.5 < interval.upper_bound
    assert interval.lower_bound == pytest.approx(1 - interval.upper_bound, abs=1e-12)


def test_wilson_hand_computed_value() -> None:
    """n = 100, wins = 50. Computed by hand from the Wilson formula."""
    z = STANDARD_NORMAL_95
    n, wins = 100, 50
    proportion = wins / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator

    interval = wilson_score_interval(wins, n)
    assert interval.lower_bound == pytest.approx(centre - spread)
    assert interval.upper_bound == pytest.approx(centre + spread)


def test_wilson_with_zero_wins_stays_at_or_above_zero() -> None:
    interval = wilson_score_interval(0, 30)
    assert interval.mean == 0.0
    assert interval.lower_bound == 0.0
    assert 0.0 < interval.upper_bound < 1.0


def test_wilson_with_all_wins_stays_at_or_below_one() -> None:
    """At p = 1 the upper bound is mathematically exactly 1, up to float rounding."""
    interval = wilson_score_interval(30, 30)
    assert interval.mean == 1.0
    assert interval.upper_bound <= 1.0
    assert interval.upper_bound == pytest.approx(1.0)
    assert 0.0 < interval.lower_bound < 1.0


@pytest.mark.parametrize("trial_count", [1, 2, 5, 10, 100, 10_000])
@pytest.mark.parametrize("win_fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_wilson_always_stays_inside_the_unit_interval(
    trial_count: int, win_fraction: float
) -> None:
    win_count = round(win_fraction * trial_count)
    interval = wilson_score_interval(win_count, trial_count)
    assert 0.0 <= interval.lower_bound <= interval.upper_bound <= 1.0


def test_wilson_is_narrower_with_more_trials() -> None:
    small = wilson_score_interval(5, 10)
    large = wilson_score_interval(500, 1000)
    assert large.half_width < small.half_width


def test_wilson_is_deterministic() -> None:
    assert wilson_score_interval(37, 100) == wilson_score_interval(37, 100)


def test_wilson_beats_the_naive_interval_at_zero_wins() -> None:
    """The naive interval collapses to a point at 0 wins; Wilson does not."""
    interval = wilson_score_interval(0, 20)
    naive_half_width = STANDARD_NORMAL_95 * math.sqrt(0.0 * 1.0 / 20)
    assert naive_half_width == 0.0
    assert interval.upper_bound > 0.0


@pytest.mark.parametrize(
    ("win_count", "trial_count", "match"),
    [(0, 0, "trial_count must be"), (5, 3, "win_count must be"), (-1, 3, "win_count must be")],
)
def test_wilson_input_validation(win_count, trial_count, match) -> None:
    with pytest.raises(ValueError, match=match):
        wilson_score_interval(win_count, trial_count)


# --- paired differences -----------------------------------------------------


def test_paired_differences_are_elementwise() -> None:
    differences = paired_score_differences([10.0, 5.0, 0.0], [4.0, 5.0, 3.0])
    assert differences.tolist() == [6.0, 0.0, -3.0]


def test_swapping_the_arguments_flips_every_sign() -> None:
    policy_a = [10.0, 5.0, 0.0]
    policy_b = [4.0, 5.0, 3.0]
    forward = paired_score_differences(policy_a, policy_b)
    reversed_differences = paired_score_differences(policy_b, policy_a)
    assert np.array_equal(forward, -reversed_differences)


def test_paired_differences_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="paired arrays must match"):
        paired_score_differences([1.0, 2.0], [1.0])


def test_pair_order_does_not_change_the_mean_or_the_standard_deviation() -> None:
    differences = np.array([3.0, -1.0, 4.0, 0.0, -2.0, 7.0])
    shuffled = differences[::-1]
    assert differences.mean() == pytest.approx(shuffled.mean())
    assert sample_standard_deviation(differences) == pytest.approx(
        sample_standard_deviation(shuffled)
    )


# --- sigma_d and sample sizing ----------------------------------------------


def test_standard_deviation_hand_computed() -> None:
    """[2, 4, 4, 4, 5, 5, 7, 9]: mean 5, sum of squares 32, n-1 = 7."""
    samples = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert sample_standard_deviation(samples) == pytest.approx(math.sqrt(32 / 7))


def test_standard_deviation_of_a_constant_sample_is_zero() -> None:
    assert sample_standard_deviation([3.0] * 10) == pytest.approx(0.0)


def test_standard_deviation_needs_two_samples() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        sample_standard_deviation([1.0])


def test_required_sample_count_matches_the_formula() -> None:
    """n = (z * sigma / effect)^2, computed by hand for sigma = 6, effect = 0.25."""
    expected = math.ceil((STANDARD_NORMAL_95 * 6.0 / 0.25) ** 2)
    assert required_sample_count(0.25, 6.0) == expected
    assert expected == 2213  # the blueprint's ~2,214 figure, recomputed


def test_required_sample_count_grows_as_the_effect_shrinks() -> None:
    assert required_sample_count(0.1, 6.0) > required_sample_count(1.0, 6.0)


def test_required_sample_count_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="detectable_effect must be positive"):
        required_sample_count(0.0, 6.0)
    with pytest.raises(ValueError, match="sigma_d must be non-negative"):
        required_sample_count(0.5, -1.0)


# --- the interval container -------------------------------------------------


def test_confidence_interval_as_dict() -> None:
    interval = ConfidenceInterval(mean=1.0, lower_bound=0.5, upper_bound=1.5, sample_count=10)
    assert interval.as_dict() == {
        "mean": 1.0,
        "lower_bound": 0.5,
        "upper_bound": 1.5,
        "sample_count": 10,
        "confidence": 0.95,
    }
    assert interval.half_width == pytest.approx(0.5)
    assert interval.excludes_zero_above
