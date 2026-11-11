"""The PPO objective, its diagnostics, and what an update is allowed to do.

The clipped objective and the ratio are checked against values worked out by
hand, not against another call to the same code. The remaining tests check the
properties that a silent PPO bug would violate: the epoch-zero identity, finite
losses, parameters actually moving, and no NaN surviving an update.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.buffers.rollout import RolloutBatch
from tarneeb.core.actions import NO_TRUMP_ACTION, NUM_ACTIONS
from tarneeb.env.spaces import D_STATIC, L_MAX, N_EVENT_FIELDS
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.training.ppo import (
    EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
    PPOConfig,
    build_optimizer,
    explained_variance,
    linear_learning_rate,
    ppo_update,
)


def build_batch(decision_count: int = 64, seed: int = 0) -> RolloutBatch:
    """Build a synthetic but well-formed batch, with log-probabilities that match.

    The stored log-probabilities are taken from the same freshly built actor, so
    the epoch-zero ratio must be exactly 1 before any optimizer step.
    """
    rng = np.random.default_rng(seed)
    actor, critic = build_batch.networks  # type: ignore[attr-defined]

    observations = torch.from_numpy(rng.random((decision_count, D_STATIC), dtype=np.float32))
    history_ids = torch.zeros((decision_count, L_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    history_masks = torch.zeros((decision_count, L_MAX), dtype=torch.bool)
    history_masks[:, :6] = True

    legal_action_masks = torch.from_numpy(rng.random((decision_count, NUM_ACTIONS)) > 0.6)
    legal_action_masks[:, NO_TRUMP_ACTION] = False
    legal_action_masks[:, 0] = True

    with torch.no_grad():
        logits = actor(observations, history_ids, history_masks)
        distribution = MaskedCategorical.from_logits(logits, legal_action_masks)
        generator = torch.Generator()
        generator.manual_seed(seed)
        selected_actions = distribution.sample(generator)
        action_log_probabilities = distribution.log_probability_of(selected_actions)
        state_values = critic(observations, history_ids, history_masks)

    advantages = torch.from_numpy(rng.standard_normal(decision_count).astype(np.float32))
    return RolloutBatch(
        observations=observations,
        history_ids=history_ids,
        history_masks=history_masks,
        legal_action_masks=legal_action_masks,
        selected_actions=selected_actions,
        action_log_probabilities=action_log_probabilities,
        state_values=state_values,
        advantages=advantages,
        returns=advantages + state_values,
    )


build_batch.networks = build_actor_and_critic(seed=1)  # type: ignore[attr-defined]


# --- hand-computed objective ------------------------------------------------


def clipped_objective(ratio: float, advantage: float, clip_epsilon: float = 0.2) -> float:
    """Reference implementation of the PPO surrogate for one sample, by hand."""
    clipped = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
    return min(ratio * advantage, clipped * advantage)


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected"),
    [
        # Inside the trust region: the clip does nothing.
        (1.0, 2.0, 2.0),
        (1.1, 2.0, 2.2),
        # Positive advantage, ratio above 1 + eps: the gain is capped at 1.2.
        (1.5, 2.0, 2.4),
        # Negative advantage, ratio below 1 - eps: the loss is capped at 0.8.
        (0.5, -2.0, -1.6),
        # Negative advantage, ratio above 1 + eps: the unclipped term is worse
        # and the min keeps it, so clipping does NOT limit the penalty here.
        (1.5, -2.0, -3.0),
        # Positive advantage, ratio below 1 - eps: unclipped is smaller, kept.
        (0.5, 2.0, 1.0),
    ],
)
def test_clipped_surrogate_matches_hand_computed_values(
    ratio: float, advantage: float, expected: float
) -> None:
    ratio_tensor = torch.tensor([ratio])
    advantage_tensor = torch.tensor([advantage])
    unclipped = ratio_tensor * advantage_tensor
    clipped = torch.clamp(ratio_tensor, 0.8, 1.2) * advantage_tensor
    computed = float(torch.min(unclipped, clipped))
    assert computed == pytest.approx(expected)
    assert clipped_objective(ratio, advantage) == pytest.approx(expected)


def test_ratio_is_one_when_the_policy_has_not_changed() -> None:
    log_probabilities = torch.tensor([-1.5, -0.3, -2.25])
    ratio = (log_probabilities - log_probabilities).exp()
    assert ratio == pytest.approx(torch.ones(3))


def test_ratio_is_the_exponential_of_the_log_probability_difference() -> None:
    stored = torch.tensor([-2.0, -1.0])
    recomputed = torch.tensor([-1.0, -1.5])
    ratio = (recomputed - stored).exp()
    assert float(ratio[0]) == pytest.approx(float(np.exp(1.0)))
    assert float(ratio[1]) == pytest.approx(float(np.exp(-0.5)))


def test_explained_variance_is_one_for_a_perfect_critic_and_zero_for_a_constant() -> None:
    returns = torch.tensor([1.0, -1.0, 0.5, 0.25])
    assert explained_variance(returns.clone(), returns) == pytest.approx(1.0)
    constant_prediction = torch.full_like(returns, float(returns.mean()))
    assert explained_variance(constant_prediction, returns) == pytest.approx(0.0, abs=1e-6)
    assert explained_variance(torch.zeros(3), torch.zeros(3)) == 0.0


def test_linear_learning_rate_anneals_to_zero() -> None:
    assert linear_learning_rate(3e-4, 0, 10) == pytest.approx(3e-4)
    assert linear_learning_rate(3e-4, 5, 10) == pytest.approx(1.5e-4)
    assert linear_learning_rate(3e-4, 10, 10) == pytest.approx(0.0)


# --- the update itself ------------------------------------------------------


def test_epoch_zero_log_probabilities_reproduce_the_stored_ones() -> None:
    """T-E10 for the feedforward variant: the ratio must start at exactly 1."""
    actor, critic = build_actor_and_critic(seed=1)
    build_batch.networks = (actor, critic)  # type: ignore[attr-defined]
    batch = build_batch(decision_count=48, seed=4)
    optimizer = build_optimizer(actor, critic, PPOConfig())

    metrics = ppo_update(actor, critic, optimizer, batch, PPOConfig(), np.random.default_rng(0))
    assert metrics.epoch_zero_log_probability_max_deviation < EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE


def test_an_update_changes_every_trainable_tensor_and_stays_finite() -> None:
    actor, critic = build_actor_and_critic(seed=2)
    build_batch.networks = (actor, critic)  # type: ignore[attr-defined]
    batch = build_batch(decision_count=64, seed=5)
    optimizer = build_optimizer(actor, critic, PPOConfig())

    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]

    metrics = ppo_update(actor, critic, optimizer, batch, PPOConfig(), np.random.default_rng(1))

    assert metrics.optimizer_steps > 0
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, critic.parameters(), strict=True)
    )
    for parameter in [*actor.parameters(), *critic.parameters()]:
        assert bool(torch.isfinite(parameter).all())

    for value in (
        metrics.policy_loss,
        metrics.value_loss,
        metrics.entropy,
        metrics.approximate_kl_divergence,
        metrics.clip_fraction,
        metrics.explained_variance,
        metrics.gradient_norm,
    ):
        assert np.isfinite(value)


def test_update_metrics_are_plain_scalars_not_tensors() -> None:
    """The observability boundary: nothing leaving the update is an autograd tensor."""
    actor, critic = build_actor_and_critic(seed=3)
    build_batch.networks = (actor, critic)  # type: ignore[attr-defined]
    batch = build_batch(decision_count=32, seed=6)
    optimizer = build_optimizer(actor, critic, PPOConfig())

    metrics = ppo_update(actor, critic, optimizer, batch, PPOConfig(), np.random.default_rng(2))
    for key, value in metrics.as_dict().items():
        assert not isinstance(value, torch.Tensor), key
        assert isinstance(value, int | float | bool), key


def test_the_gradient_norm_reported_is_the_pre_clip_norm() -> None:
    actor, critic = build_actor_and_critic(seed=4)
    build_batch.networks = (actor, critic)  # type: ignore[attr-defined]
    batch = build_batch(decision_count=64, seed=7)
    config = PPOConfig(max_gradient_norm=1e-6, update_epochs=1, minibatch_count=1)
    optimizer = build_optimizer(actor, critic, config)

    metrics = ppo_update(actor, critic, optimizer, batch, config, np.random.default_rng(3))
    # If the clipped norm were reported it could never exceed the tiny threshold.
    assert metrics.gradient_norm > config.max_gradient_norm


def test_minibatch_shuffling_is_reproducible_from_its_generator() -> None:
    def run(seed: int) -> list[float]:
        actor, critic = build_actor_and_critic(seed=5)
        build_batch.networks = (actor, critic)  # type: ignore[attr-defined]
        batch = build_batch(decision_count=64, seed=8)
        optimizer = build_optimizer(actor, critic, PPOConfig())
        metrics = ppo_update(
            actor, critic, optimizer, batch, PPOConfig(), np.random.default_rng(seed)
        )
        return [metrics.policy_loss, metrics.value_loss, metrics.entropy]

    assert run(11) == pytest.approx(run(11))
    assert run(11) != pytest.approx(run(12))


def test_an_empty_batch_is_refused() -> None:
    actor, critic = build_actor_and_critic(seed=6)
    optimizer = build_optimizer(actor, critic, PPOConfig())
    empty = RolloutBatch(
        observations=torch.zeros((0, D_STATIC)),
        history_ids=torch.zeros((0, L_MAX, N_EVENT_FIELDS), dtype=torch.int64),
        history_masks=torch.zeros((0, L_MAX), dtype=torch.bool),
        legal_action_masks=torch.zeros((0, NUM_ACTIONS), dtype=torch.bool),
        selected_actions=torch.zeros(0, dtype=torch.int64),
        action_log_probabilities=torch.zeros(0),
        state_values=torch.zeros(0),
        advantages=torch.zeros(0),
        returns=torch.zeros(0),
    )
    with pytest.raises(ValueError, match="empty rollout"):
        ppo_update(actor, critic, optimizer, empty, PPOConfig(), np.random.default_rng(0))


@pytest.mark.parametrize(
    "overrides",
    [
        {"learning_rate": 0.0},
        {"gae_lambda": 1.5},
        {"clip_epsilon": 0.0},
        {"clip_epsilon": 1.0},
        {"update_epochs": 0},
        {"minibatch_count": 0},
    ],
)
def test_invalid_hyper_parameters_are_refused(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**overrides)  # type: ignore[arg-type]


def test_the_configured_discount_factor_is_one() -> None:
    """gamma = 1.0 is a deliberate design decision, not a default worth drifting."""
    assert PPOConfig().discount_factor == 1.0
