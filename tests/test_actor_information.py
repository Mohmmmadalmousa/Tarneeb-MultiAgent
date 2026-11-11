"""The actor's information boundary, checked on the training path itself.

The Phase 4 leakage gate proves ``static_actor`` is safe. This file proves the
separate thing Phase 6 must not break: that the *training* code only ever feeds
the actor that safe vector, and that nothing in the loop can hand it a hidden
hand.

In Phase 6 the critic is decentralized, so the strongest available statement is
also the simplest: the whole training stack never calls ``static_critic`` at all.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.spaces import D_CRITIC, D_STATIC
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.nets import ArchitectureSpec, build_actor_and_critic
from tarneeb.training.config import EvaluationSchedule, TrainingConfig
from tarneeb.training.ppo import PPOConfig
from tarneeb.training.rollout import RolloutCollector
from tarneeb.training.train import build_baseline_agent, train

HIDDEN_STATE_FIELDS = ("hands", "original_hands")


def build_collector(num_envs: int = 2, rollout_length: int = 40) -> RolloutCollector:
    environments = VecTarneeb(num_envs=num_envs, root_seed=0)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(0)
    opponents = [build_baseline_agent("basic", seed=index) for index in range(num_envs)]
    return RolloutCollector(
        environments, actor, critic, rollout_length, generator, opponents=opponents
    )


def tiny_config(checkpoint_directory: Path) -> TrainingConfig:
    return TrainingConfig(
        run_name="boundary",
        environment_count=2,
        rollout_length=24,
        total_environment_steps=96,
        environment_root_seed=3,
        checkpoint_directory=checkpoint_directory,
        checkpoint_every_updates=99,
        ppo=PPOConfig(update_epochs=1, minibatch_count=1),
        evaluation=EvaluationSchedule(
            every_updates=99, paired_deal_count=2, bootstrap_resamples=50
        ),
    )


def test_the_training_loop_never_builds_a_critic_observation(tmp_path: Path) -> None:
    """The 856-dim privileged vector is not constructed anywhere in Phase 6."""
    with patch.object(ObservationBuilder, "static_critic", autospec=True) as privileged_builder:
        train(tiny_config(tmp_path))
    privileged_builder.assert_not_called()


def test_a_rollout_never_builds_a_critic_observation() -> None:
    collector = build_collector()
    with patch.object(ObservationBuilder, "static_critic", autospec=True) as privileged_builder:
        collector.collect()
        collector.bootstrap_values()
    privileged_builder.assert_not_called()


def test_everything_the_networks_receive_is_the_actor_width() -> None:
    collector = build_collector()
    buffer, _metrics = collector.collect()
    batch = buffer.to_batch(collector.bootstrap_values(), 1.0, 0.95)

    assert batch.observations.shape[-1] == D_STATIC == 692
    assert batch.observations.shape[-1] != D_CRITIC
    assert collector.actor.spec.actor_observation_dim == D_STATIC
    assert collector.critic.spec.critic_observation_dim == D_STATIC


def test_the_stored_observation_matches_the_actor_safe_builder_exactly() -> None:
    """What the buffer holds must be the proven-safe vector, not a variant."""
    environments = VecTarneeb(num_envs=2, root_seed=7)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(7)
    opponents = [build_baseline_agent("basic", seed=index) for index in range(2)]
    collector = RolloutCollector(environments, actor, critic, 1, generator, opponents=opponents)

    acting_seats = environments.acting_seats()
    learner_flags = collector.learner_decision_flags(acting_seats)
    learner_indices = np.flatnonzero(learner_flags)
    expected = [
        environments.environments[int(index)].actor_observation() for index in learner_indices
    ]

    observations, _ids, _masks, _legal = collector._gather_learner_inputs(learner_indices)
    for row, reference in enumerate(expected):
        assert np.array_equal(observations[row], reference)


def test_the_actor_cannot_physically_accept_a_privileged_observation() -> None:
    """A miswiring must raise, not train quietly on leaked cards."""
    actor, _critic = build_actor_and_critic(seed=1)
    assert actor.trunk.static_encoder.layers[0].in_features == D_STATIC
    assert actor.trunk.static_encoder.layers[0].in_features != D_CRITIC


def test_no_hidden_state_field_is_reachable_from_the_stored_rows() -> None:
    """The buffer stores arrays, never a state object that carries hands."""
    collector = build_collector()
    buffer, _metrics = collector.collect()
    for name in HIDDEN_STATE_FIELDS:
        assert not hasattr(buffer, name)
    for array in (
        buffer.observations,
        buffer.history_ids,
        buffer.history_masks,
        buffer.legal_action_masks,
    ):
        assert isinstance(array, np.ndarray)


def test_a_widened_critic_leaves_the_actor_untouched() -> None:
    """Phase 9 may widen the critic; that must not widen the actor with it."""
    spec = ArchitectureSpec(critic_observation_dim=D_CRITIC)
    actor, critic = build_actor_and_critic(spec, seed=1)
    assert actor.trunk.static_encoder.layers[0].in_features == D_STATIC
    assert critic.trunk.static_encoder.layers[0].in_features == D_CRITIC
