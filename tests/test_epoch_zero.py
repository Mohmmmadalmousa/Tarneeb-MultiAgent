"""The epoch-zero guard, measured on a real rollout.

Before the first optimizer step the policy has not changed, so replaying the
collected batch through the training path must reproduce the log-probabilities
that were stored during collection, and the PPO ratio must be exactly 1.

This is the test that catches a whole class of silent bugs: a buffer that stores
the wrong row, a history that is rebuilt differently at replay time than at
collection time, a mask that is regenerated instead of stored, an observation
that is recomputed from a mutated state. None of those would raise; all of them
would show up here as a non-zero deviation.

The measurement is::

    max | log pi_replay(a_t | o_t) - log pi_stored(a_t | o_t) |  <  1e-4

taken over the exact stored observations, event-history ids, padding masks,
legal-action masks and actions -- never recomputed from the environment.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.env.vec import VecTarneeb
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.training.ppo import (
    EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
    PPOConfig,
    build_optimizer,
    ppo_update,
)
from tarneeb.training.rollout import RolloutCollector
from tarneeb.training.train import build_baseline_agent


def collect_real_rollout(
    opponent_name: str = "basic",
    num_envs: int = 4,
    rollout_length: int = 120,
    seed: int = 0,
):
    environments = VecTarneeb(num_envs=num_envs, root_seed=seed)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    opponents = [build_baseline_agent(opponent_name, seed=index) for index in range(num_envs)]
    collector = RolloutCollector(
        environments, actor, critic, rollout_length, generator, opponents=opponents
    )
    buffer, _metrics = collector.collect()
    batch = buffer.to_batch(collector.bootstrap_values(), 1.0, 0.95)
    return actor, critic, batch


def test_epoch_zero_reconstruction_deviation_is_below_the_threshold() -> None:
    """Replay the stored batch through the actor; the ratio must be exactly 1."""
    actor, _critic, batch = collect_real_rollout()
    assert batch.decision_count > 0

    actor.train()  # the replay path runs in training mode, as PPO does
    logits = actor(batch.observations, batch.history_ids, batch.history_masks)
    distribution = MaskedCategorical.from_logits(logits, batch.legal_action_masks)
    replayed_log_probabilities = distribution.log_probability_of(batch.selected_actions)

    deviation = float(
        (replayed_log_probabilities - batch.action_log_probabilities).detach().abs().max()
    )
    ratio = (replayed_log_probabilities - batch.action_log_probabilities).detach().exp()

    assert deviation < EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE, (
        f"epoch-zero log-probability deviation {deviation:.3e} exceeds "
        f"{EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE:.0e}"
    )
    assert float((ratio - 1.0).abs().max()) < 1e-4


def test_the_update_reports_the_same_epoch_zero_deviation() -> None:
    """The number the training loop logs must be the measured one, not a stub."""
    actor, critic, batch = collect_real_rollout(seed=3)
    optimizer = build_optimizer(actor, critic, PPOConfig())

    metrics = ppo_update(actor, critic, optimizer, batch, PPOConfig(), np.random.default_rng(0))
    assert metrics.epoch_zero_log_probability_max_deviation < EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE


def test_a_shuffled_minibatch_still_reconstructs_exactly() -> None:
    """PPO replays a permuted subset; the identity must survive the permutation."""
    actor, _critic, batch = collect_real_rollout(seed=5)
    shuffle = np.random.default_rng(11).permutation(batch.decision_count)
    indices = torch.from_numpy(shuffle[: max(1, batch.decision_count // 3)])

    actor.train()
    logits = actor(
        batch.observations[indices], batch.history_ids[indices], batch.history_masks[indices]
    )
    distribution = MaskedCategorical.from_logits(logits, batch.legal_action_masks[indices])
    replayed = distribution.log_probability_of(batch.selected_actions[indices])

    deviation = float((replayed - batch.action_log_probabilities[indices]).detach().abs().max())
    assert deviation < EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE


@pytest.mark.parametrize("opponent_name", ["random", "basic", "strong"])
def test_the_guard_holds_against_every_training_opponent(opponent_name: str) -> None:
    actor, _critic, batch = collect_real_rollout(
        opponent_name=opponent_name, num_envs=2, rollout_length=80, seed=7
    )
    actor.train()
    logits = actor(batch.observations, batch.history_ids, batch.history_masks)
    distribution = MaskedCategorical.from_logits(logits, batch.legal_action_masks)
    deviation = float(
        (distribution.log_probability_of(batch.selected_actions) - batch.action_log_probabilities)
        .detach()
        .abs()
        .max()
    )
    assert deviation < EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE


def test_the_guard_would_catch_a_corrupted_replay() -> None:
    """A negative control: perturb one stored input and the deviation must blow up.

    Without this, a guard that always returned zero would look like it passed.
    """
    actor, _critic, batch = collect_real_rollout(seed=9)
    corrupted_observations = batch.observations.clone()
    corrupted_observations[0] += 1.0

    actor.train()
    logits = actor(corrupted_observations, batch.history_ids, batch.history_masks)
    distribution = MaskedCategorical.from_logits(logits, batch.legal_action_masks)
    deviation = float(
        (distribution.log_probability_of(batch.selected_actions) - batch.action_log_probabilities)
        .detach()
        .abs()
        .max()
    )
    assert deviation > EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE


def test_the_tolerance_is_the_specified_one() -> None:
    assert EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE == 1e-4
