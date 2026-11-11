"""The mask-safety gate: an illegal action must be impossible, not merely rare.

These tests are numerical, not structural. They check the actual probabilities
the policy would sample from, including under an adversarial logit that strongly
prefers an illegal action, and they audit a large number of real masks taken
from the frozen engine.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from tests.test_nets import random_history

from tarneeb.core.actions import NO_TRUMP_ACTION, NUM_ACTIONS
from tarneeb.env.spaces import D_STATIC
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.masking import (
    MASKED_LOGIT_FILL_VALUE,
    MaskedCategorical,
    actions_are_legal,
    apply_legal_action_mask,
)
from tarneeb.models.nets import build_actor_and_critic


def masks_from_engine(step_count: int, num_envs: int = 8, seed: int = 0) -> np.ndarray:
    """Collect real legal-action masks by playing random legal actions."""
    environments = VecTarneeb(num_envs=num_envs, root_seed=seed)
    environments.reset()
    rng = np.random.default_rng(seed)
    collected: list[np.ndarray] = []
    while len(collected) * num_envs < step_count:
        masks = environments.legal_action_masks()
        collected.append(masks.copy())
        actions = np.array([int(rng.choice(np.flatnonzero(row))) for row in masks], dtype=np.int64)
        step = environments.step(actions)
        if step.terminated.any():
            environments.reset_terminated(step.terminated)
    return np.concatenate(collected, axis=0)


def test_illegal_actions_have_exactly_zero_probability() -> None:
    legal = torch.zeros((1, NUM_ACTIONS), dtype=torch.bool)
    legal[0, [3, 7, 52]] = True
    # An adversarial logit: the network wants an illegal action very badly.
    logits = torch.full((1, NUM_ACTIONS), 50.0)
    logits[0, 3] = -50.0

    distribution = MaskedCategorical.from_logits(logits, legal)
    probabilities = distribution.probabilities

    assert float(probabilities[0, ~legal[0]].sum()) == 0.0
    assert torch.all(probabilities[0, ~legal[0]] == 0.0)
    assert pytest.approx(float(probabilities.sum()), abs=1e-6) == 1.0


def test_masked_logits_use_the_documented_fill_value() -> None:
    legal = torch.zeros((1, NUM_ACTIONS), dtype=torch.bool)
    legal[0, 0] = True
    masked = apply_legal_action_mask(torch.zeros((1, NUM_ACTIONS)), legal)
    assert float(masked[0, 1]) == MASKED_LOGIT_FILL_VALUE
    assert bool(torch.isfinite(masked).all())


def test_entropy_and_log_probabilities_stay_finite() -> None:
    """A finite fill, not -inf, is what keeps ``0 * log 0`` out of the entropy."""
    legal = torch.zeros((4, NUM_ACTIONS), dtype=torch.bool)
    legal[:, 0] = True
    legal[1, 5] = True
    distribution = MaskedCategorical.from_logits(torch.randn(4, NUM_ACTIONS), legal)

    assert bool(torch.isfinite(distribution.entropy()).all())
    assert bool(torch.isfinite(distribution.log_probabilities).all())
    # A single legal action means zero entropy.
    assert pytest.approx(float(distribution.entropy()[0]), abs=1e-6) == 0.0
    assert float(distribution.entropy()[1]) > 0.0


def test_entropy_of_a_uniform_masked_distribution_is_log_of_the_legal_count() -> None:
    legal = torch.zeros((1, NUM_ACTIONS), dtype=torch.bool)
    legal[0, :8] = True
    distribution = MaskedCategorical.from_logits(torch.zeros((1, NUM_ACTIONS)), legal)
    assert pytest.approx(float(distribution.entropy()[0]), abs=1e-5) == float(np.log(8))


def test_all_false_mask_is_refused_rather_than_smoothed() -> None:
    legal = torch.zeros((2, NUM_ACTIONS), dtype=torch.bool)
    legal[0, 1] = True
    with pytest.raises(ValueError, match="no legal action"):
        apply_legal_action_mask(torch.zeros((2, NUM_ACTIONS)), legal)


def test_a_mask_that_allows_action_64_is_refused() -> None:
    legal = torch.zeros((1, NUM_ACTIONS), dtype=torch.bool)
    legal[0, NO_TRUMP_ACTION] = True
    with pytest.raises(ValueError, match="reserved"):
        apply_legal_action_mask(torch.zeros((1, NUM_ACTIONS)), legal)


def test_mask_shape_mismatch_is_refused() -> None:
    legal = torch.ones((1, NUM_ACTIONS), dtype=torch.bool)
    legal[0, NO_TRUMP_ACTION] = False
    with pytest.raises(ValueError, match="same shape"):
        apply_legal_action_mask(torch.zeros((1, NUM_ACTIONS - 1)), legal)


def test_sampling_is_reproducible_from_a_seeded_generator() -> None:
    rng = np.random.default_rng(3)
    legal = torch.from_numpy(rng.random((64, NUM_ACTIONS)) > 0.5)
    legal[:, NO_TRUMP_ACTION] = False
    legal[:, 0] = True  # guarantee at least one legal action per row
    logits = torch.from_numpy(rng.standard_normal((64, NUM_ACTIONS)).astype(np.float32))
    distribution = MaskedCategorical.from_logits(logits, legal)

    first_generator = torch.Generator()
    first_generator.manual_seed(2024)
    second_generator = torch.Generator()
    second_generator.manual_seed(2024)

    assert torch.equal(distribution.sample(first_generator), distribution.sample(second_generator))


def test_deterministic_mode_picks_the_best_legal_action_not_the_best_action() -> None:
    legal = torch.zeros((1, NUM_ACTIONS), dtype=torch.bool)
    legal[0, [10, 20]] = True
    logits = torch.zeros((1, NUM_ACTIONS))
    logits[0, 30] = 100.0  # illegal but wildly preferred
    logits[0, 20] = 1.0
    logits[0, 10] = 0.5

    distribution = MaskedCategorical.from_logits(logits, legal)
    assert int(distribution.most_probable_action()) == 20


def test_sampled_actions_are_always_legal_over_many_real_masks() -> None:
    """Numerical audit over real engine masks and a randomly initialized policy."""
    masks = masks_from_engine(step_count=4_000, seed=5)
    legal_masks = torch.from_numpy(masks)
    rng = np.random.default_rng(9)
    actor, _ = build_actor_and_critic(seed=1)

    observations = torch.from_numpy(rng.random((masks.shape[0], D_STATIC), dtype=np.float32))
    history_ids, history_masks = random_history(masks.shape[0], rng)

    with torch.no_grad():
        logits = actor(observations, history_ids, history_masks)
    distribution = MaskedCategorical.from_logits(logits, legal_masks)

    generator = torch.Generator()
    generator.manual_seed(77)
    for _draw in range(5):
        actions = distribution.sample(generator)
        assert actions_are_legal(actions, legal_masks)
        assert int((actions == NO_TRUMP_ACTION).sum()) == 0

    greedy_actions = distribution.most_probable_action()
    assert actions_are_legal(greedy_actions, legal_masks)
    assert int((greedy_actions == NO_TRUMP_ACTION).sum()) == 0


def test_action_64_never_appears_in_an_engine_mask() -> None:
    masks = masks_from_engine(step_count=4_000, seed=6)
    assert not masks[:, NO_TRUMP_ACTION].any()


def test_probability_mass_on_illegal_actions_is_zero_across_real_masks() -> None:
    masks = masks_from_engine(step_count=2_000, seed=8)
    legal_masks = torch.from_numpy(masks)
    rng = np.random.default_rng(21)
    # Deliberately extreme logits: nothing here should rescue an illegal action.
    logits = torch.from_numpy(
        (rng.standard_normal((masks.shape[0], NUM_ACTIONS)) * 100.0).astype(np.float32)
    )
    probabilities = MaskedCategorical.from_logits(logits, legal_masks).probabilities
    illegal_mass = probabilities.masked_fill(legal_masks, 0.0)
    assert float(illegal_mass.max()) == 0.0
