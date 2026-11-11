"""Held-out evaluation seeds must be disjoint from training seeds.

If an evaluation deal could also appear during training, every later "beats the
baseline" claim would be contaminated. The split is a range check, so it can be
proved rather than sampled.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.evaluation.arena import (
    HELD_OUT_EVALUATION_SEEDS,
    TRAINING_SEEDS,
    Arena,
    evaluation_match_seed,
    seed_ranges_are_disjoint,
)


def test_held_out_range_is_the_documented_one() -> None:
    assert HELD_OUT_EVALUATION_SEEDS.start == 10**9
    assert HELD_OUT_EVALUATION_SEEDS.stop == 10**9 + 100_000
    assert len(HELD_OUT_EVALUATION_SEEDS) == 100_000


def test_training_range_is_below_the_evaluation_range() -> None:
    assert TRAINING_SEEDS.start == 0
    assert TRAINING_SEEDS.stop == 10**8
    assert TRAINING_SEEDS.stop <= HELD_OUT_EVALUATION_SEEDS.start


def test_the_two_ranges_do_not_overlap() -> None:
    assert seed_ranges_are_disjoint()
    assert not set(range(TRAINING_SEEDS.start, TRAINING_SEEDS.stop, 997_913)) & set(
        HELD_OUT_EVALUATION_SEEDS
    )


def test_no_training_seed_is_an_evaluation_seed() -> None:
    """Check the boundary explicitly, plus a spread of training seeds."""
    for training_seed in (0, 1, 12345, TRAINING_SEEDS.stop - 1):
        assert training_seed not in HELD_OUT_EVALUATION_SEEDS
    for evaluation_seed in (
        HELD_OUT_EVALUATION_SEEDS.start,
        HELD_OUT_EVALUATION_SEEDS.start + 1,
        HELD_OUT_EVALUATION_SEEDS.stop - 1,
    ):
        assert evaluation_seed not in TRAINING_SEEDS


# --- derived match seeds stay inside the held-out range ---------------------


@pytest.mark.parametrize("evaluation_seed", [0, 1, 20250910, 10**7])
def test_derived_match_seeds_are_held_out(evaluation_seed: int) -> None:
    for pair_index in range(500):
        match_seed = evaluation_match_seed(evaluation_seed, pair_index)
        assert match_seed in HELD_OUT_EVALUATION_SEEDS
        assert match_seed not in TRAINING_SEEDS


def test_derived_match_seeds_are_deterministic() -> None:
    first = [evaluation_match_seed(42, index) for index in range(50)]
    second = [evaluation_match_seed(42, index) for index in range(50)]
    assert first == second


def test_derived_match_seeds_are_distinct_within_a_run() -> None:
    seeds = [evaluation_match_seed(42, index) for index in range(2000)]
    assert len(set(seeds)) == 2000


def test_different_evaluation_seeds_select_different_deals() -> None:
    first = [evaluation_match_seed(1, index) for index in range(50)]
    second = [evaluation_match_seed(2, index) for index in range(50)]
    assert first != second


def test_negative_pair_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="pair_index must be non-negative"):
        evaluation_match_seed(1, -1)


# --- an actual evaluation only touches held-out seeds -----------------------


def test_an_evaluation_run_uses_only_held_out_seeds() -> None:
    arena = Arena(replay_sample_rate=0.0)
    result = arena.evaluate(
        BasicHeuristicAgent(),
        RandomAgent(seed=1),
        paired_deal_count=20,
        evaluation_seed=20250910,
        bootstrap_resamples=200,
    )
    for pair in result.paired_outcomes:
        assert pair.match_seed in HELD_OUT_EVALUATION_SEEDS
        assert pair.first_run.match_seed in HELD_OUT_EVALUATION_SEEDS
        assert pair.second_run.match_seed in HELD_OUT_EVALUATION_SEEDS


def test_evaluation_does_not_touch_global_rng_state() -> None:
    """Evaluation must not consume a stream a future trainer might rely on."""
    np.random.seed(12345)
    random.seed(12345)
    numpy_state_before = np.random.get_state()[1].copy()
    python_state_before = random.getstate()

    arena = Arena(replay_sample_rate=0.0)
    arena.evaluate(
        BasicHeuristicAgent(),
        RandomAgent(seed=3),
        paired_deal_count=10,
        evaluation_seed=1,
        bootstrap_resamples=200,
    )

    assert np.array_equal(np.random.get_state()[1], numpy_state_before)
    assert random.getstate() == python_state_before


def test_evaluation_results_are_independent_of_global_rng_churn() -> None:
    def run_evaluation() -> list[float]:
        arena = Arena(replay_sample_rate=0.0)
        result = arena.evaluate(
            BasicHeuristicAgent(),
            RandomAgent(seed=5),
            paired_deal_count=10,
            evaluation_seed=2,
            bootstrap_resamples=200,
        )
        return [pair.paired_score_difference for pair in result.paired_outcomes]

    np.random.seed(1)
    random.seed(1)
    first_run = run_evaluation()

    np.random.seed(999)
    random.seed(999)
    for _ in range(500):
        np.random.random()
        random.random()
    second_run = run_evaluation()

    assert first_run == second_run
