"""Generalized advantage estimation, checked against hand-computed values.

Every expected number below was worked out on paper from the recursion in
docs and the blueprint, never by running the function being tested.

Worked example used throughout::

    rewards            = [0.0, 0.0, 1.0]
    values             = [0.5, 0.2, 0.1]
    next values        = [0.2, 0.1, 0.0]   (the third decision ends the deal)
    episode continues  = [1.0, 1.0, 0.0]
    gamma              = 1.0

    delta_2 = 1.0 + 0.0        - 0.1 =  0.9
    delta_1 = 0.0 + 1.0 * 0.1  - 0.2 = -0.1
    delta_0 = 0.0 + 1.0 * 0.2  - 0.5 = -0.3
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.buffers.rollout import compute_generalized_advantage_estimates

REWARDS = np.array([0.0, 0.0, 1.0])
VALUES = np.array([0.5, 0.2, 0.1])
NEXT_VALUES = np.array([0.2, 0.1, 0.0])
CONTINUES = np.array([1.0, 1.0, 0.0])

TEMPORAL_DIFFERENCE_ERRORS = np.array([-0.3, -0.1, 0.9])


def advantages(gae_lambda: float, discount_factor: float = 1.0) -> np.ndarray:
    return compute_generalized_advantage_estimates(
        rewards=REWARDS,
        state_values=VALUES,
        next_state_values=NEXT_VALUES,
        episode_continues=CONTINUES,
        discount_factor=discount_factor,
        gae_lambda=gae_lambda,
    )


def test_lambda_zero_reduces_to_the_one_step_temporal_difference_error() -> None:
    assert advantages(gae_lambda=0.0) == pytest.approx(TEMPORAL_DIFFERENCE_ERRORS)


def test_lambda_one_reduces_to_the_monte_carlo_return_minus_the_value() -> None:
    # With gamma = 1 the return from every decision in this deal is 1.0.
    monte_carlo_returns = np.array([1.0, 1.0, 1.0])
    assert advantages(gae_lambda=1.0) == pytest.approx(monte_carlo_returns - VALUES)
    assert advantages(gae_lambda=1.0) == pytest.approx([0.5, 0.8, 0.9])


def test_intermediate_lambda_matches_the_hand_computed_recursion() -> None:
    # adv_2 = 0.9
    # adv_1 = -0.1 + 0.95 * 0.9   = 0.755
    # adv_0 = -0.3 + 0.95 * 0.755 = 0.41725
    assert advantages(gae_lambda=0.95) == pytest.approx([0.41725, 0.755, 0.9])


def test_returns_are_advantages_plus_values() -> None:
    computed = advantages(gae_lambda=0.95)
    assert (computed + VALUES) == pytest.approx([0.91725, 0.955, 1.0])


def test_a_deal_boundary_stops_credit_from_flowing_backwards() -> None:
    """Two deals in one sequence must not share advantage."""
    rewards = np.array([0.0, 1.0, 0.0, -1.0])
    values = np.array([0.1, 0.2, 0.3, 0.4])
    next_values = np.array([0.2, 0.0, 0.4, 0.0])
    continues = np.array([1.0, 0.0, 1.0, 0.0])

    computed = compute_generalized_advantage_estimates(
        rewards, values, next_values, continues, discount_factor=1.0, gae_lambda=1.0
    )
    # First deal:  delta_1 = 1.0 - 0.2 = 0.8 ; delta_0 = 0.2 - 0.1 = 0.1 -> 0.9
    # Second deal: delta_3 = -1.0 - 0.4 = -1.4 ; delta_2 = 0.4 - 0.3 = 0.1 -> -1.3
    assert computed == pytest.approx([0.9, 0.8, -1.3, -1.4])


def test_discounting_is_applied_where_configured() -> None:
    computed = compute_generalized_advantage_estimates(
        rewards=REWARDS,
        state_values=VALUES,
        next_state_values=NEXT_VALUES,
        episode_continues=CONTINUES,
        discount_factor=0.5,
        gae_lambda=1.0,
    )
    # delta_2 = 0.9 ; delta_1 = 0.5*0.1 - 0.2 = -0.15 ; delta_0 = 0.5*0.2 - 0.5 = -0.4
    # adv_1 = -0.15 + 0.5 * 0.9 = 0.3 ; adv_0 = -0.4 + 0.5 * 0.3 = -0.25
    assert computed == pytest.approx([-0.25, 0.3, 0.9])


def test_a_single_terminal_decision_is_just_its_reward_minus_its_value() -> None:
    computed = compute_generalized_advantage_estimates(
        rewards=np.array([0.75]),
        state_values=np.array([0.25]),
        next_state_values=np.array([0.0]),
        episode_continues=np.array([0.0]),
        discount_factor=1.0,
        gae_lambda=0.95,
    )
    assert computed == pytest.approx([0.5])


def test_mismatched_input_shapes_are_refused() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_generalized_advantage_estimates(
            rewards=np.zeros(3),
            state_values=np.zeros(2),
            next_state_values=np.zeros(3),
            episode_continues=np.ones(3),
            discount_factor=1.0,
            gae_lambda=0.95,
        )
