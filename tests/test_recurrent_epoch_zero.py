"""The epoch-zero replay identity, padding inertness, and gradient behaviour.

Blueprint T-E10, extended to recurrent sequences. This is the hard correctness
gate of Phase 8: if replay does not reconstruct exactly the states the rollout
used, every PPO update is silently wrong and no loss curve will say so.

Every positive claim here is paired with a negative control. "The deviation is
small" means nothing unless a deliberately broken input makes it large.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.buffers.recurrent import ChunkBatch
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.recurrent import build_recurrent_actor_and_critic
from tarneeb.training.ppo import PPOConfig
from tarneeb.training.recurrent_ppo import (
    RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
    RECURRENT_EPOCH_ZERO_VALUE_TOLERANCE,
    build_recurrent_optimizer,
    gradients_are_finite,
    gru_gradient_norm,
    recurrent_ppo_update,
    replay_chunks,
)
from tarneeb.training.recurrent_rollout import RecurrentRolloutCollector

GAMMA = 1.0
LAMBDA = 0.95


def build_rollout(
    seed: int = 13, environments: int = 4, rollout_length: int = 96
) -> tuple[RecurrentRolloutCollector, ChunkBatch]:
    vector_environments = VecTarneeb(num_envs=environments, root_seed=seed)
    actor, critic = build_recurrent_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(2)
    collector = RecurrentRolloutCollector(
        environments=vector_environments,
        actor=actor,
        critic=critic,
        rollout_length=rollout_length,
        action_sampling_generator=generator,
        opponents=[BasicHeuristicAgent() for _ in range(environments)],
    )
    buffer, _metrics = collector.collect()
    batch = buffer.to_chunks(collector.bootstrap_values(), GAMMA, LAMBDA)
    return collector, batch


@pytest.fixture(scope="module")
def rollout() -> tuple[RecurrentRolloutCollector, ChunkBatch]:
    return build_rollout()


def replace_field(batch: ChunkBatch, **overrides: torch.Tensor) -> ChunkBatch:
    """Return a copy of ``batch`` with the named tensors replaced."""
    values = {field: overrides.get(field, getattr(batch, field)) for field in ChunkBatch.__slots__}
    return ChunkBatch(**values)


# --- the identity -----------------------------------------------------------


def test_replayed_log_probabilities_reproduce_the_stored_ones(rollout) -> None:
    """T-E10. Must hold before any optimizer step."""
    collector, batch = rollout
    replayed = replay_chunks(collector.actor, collector.critic, batch)

    deviation = replayed.max_log_probability_deviation(batch)

    assert deviation < RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    assert deviation < 1e-5, "this should be float32 noise, not merely within tolerance"


def test_the_replayed_ratio_is_one(rollout) -> None:
    collector, batch = rollout
    replayed = replay_chunks(collector.actor, collector.critic, batch)

    ratio = torch.exp(replayed.log_probabilities.detach() - batch.old_log_probabilities)

    assert torch.allclose(ratio[batch.valid], torch.ones(batch.valid_transition_count), atol=1e-4)


def test_replayed_values_reproduce_the_stored_ones(rollout) -> None:
    collector, batch = rollout
    replayed = replay_chunks(collector.actor, collector.critic, batch)

    assert replayed.max_value_deviation(batch) < RECURRENT_EPOCH_ZERO_VALUE_TOLERANCE


# --- negative controls ------------------------------------------------------


def test_corrupting_the_stored_initial_state_breaks_the_identity(rollout) -> None:
    """Without this, the positive result would prove nothing."""
    collector, batch = rollout
    corrupted = replace_field(batch, initial_actor_hidden=batch.initial_actor_hidden + 1.0)

    replayed = replay_chunks(collector.actor, collector.critic, corrupted)

    assert (
        replayed.max_log_probability_deviation(corrupted)
        > RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )


def test_reordering_the_stored_segments_breaks_the_identity(rollout) -> None:
    """A chunk's segments belong to specific positions, not to the chunk in bulk."""
    collector, batch = rollout
    reversed_ids = batch.previous_event_ids.flip(0)
    reversed_lengths = batch.previous_event_lengths.flip(0)
    corrupted = replace_field(
        batch,
        previous_event_ids=reversed_ids,
        previous_event_lengths=reversed_lengths,
    )

    replayed = replay_chunks(collector.actor, collector.critic, corrupted)

    assert (
        replayed.max_log_probability_deviation(corrupted)
        > RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )


def test_feeding_the_head_its_own_prior_events_breaks_the_identity(rollout) -> None:
    """The exact double-consumption bug rule R4 exists to prevent.

    If position 0 replayed the events already summarized in ``h0``, every
    reconstructed state in the chunk would be wrong.
    """
    collector, batch = rollout
    lengths = batch.previous_event_lengths.clone()
    ids = batch.previous_event_ids.clone()
    # Give every head the segment its successor stored, i.e. consume twice.
    lengths[0] = batch.previous_event_lengths[1]
    ids[0] = batch.previous_event_ids[1]
    corrupted = replace_field(batch, previous_event_ids=ids, previous_event_lengths=lengths)

    replayed = replay_chunks(collector.actor, collector.critic, corrupted)

    assert (
        replayed.max_log_probability_deviation(corrupted)
        > RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )


def test_perturbing_the_static_observation_breaks_the_identity(rollout) -> None:
    collector, batch = rollout
    corrupted = replace_field(batch, static=batch.static + 0.5)

    replayed = replay_chunks(collector.actor, collector.critic, corrupted)

    assert (
        replayed.max_log_probability_deviation(corrupted)
        > RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )


def test_the_update_refuses_to_proceed_when_the_identity_fails(rollout) -> None:
    """The gate must stop a corrupted run, not merely report a number."""
    collector, batch = rollout
    actor, critic = build_recurrent_actor_and_critic(seed=1)
    config = PPOConfig(update_epochs=1, minibatch_count=1)
    optimizer = build_recurrent_optimizer(actor, critic, config)
    corrupted = replace_field(batch, initial_actor_hidden=batch.initial_actor_hidden + 2.0)
    del collector

    with pytest.raises(AssertionError, match="epoch-zero log-probability deviation"):
        recurrent_ppo_update(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            batch=corrupted,
            config=config,
            shuffle_generator=np.random.default_rng(0),
        )


# --- padding is inert -------------------------------------------------------


def test_overwriting_every_padded_field_with_noise_changes_no_loss(rollout) -> None:
    """Padded timesteps contribute exactly zero to policy, value and entropy loss."""
    collector, batch = rollout
    padded = ~batch.valid
    assert bool(padded.any()), "this rollout must contain padding for the test to mean anything"

    generator = torch.Generator()
    generator.manual_seed(0)
    noisy_static = batch.static.clone()
    noisy_static[padded] = torch.randn(
        (int(padded.sum()), batch.static.shape[-1]), generator=generator
    )
    noisy_advantages = batch.advantages.clone()
    noisy_advantages[padded] = 99.0
    noisy_returns = batch.returns.clone()
    noisy_returns[padded] = -99.0
    noisy_log_probabilities = batch.old_log_probabilities.clone()
    noisy_log_probabilities[padded] = -50.0

    noisy = replace_field(
        batch,
        static=noisy_static,
        critic_static=noisy_static,
        advantages=noisy_advantages,
        returns=noisy_returns,
        old_log_probabilities=noisy_log_probabilities,
    )

    clean_losses = _losses(collector, batch)
    noisy_losses = _losses(collector, noisy)

    for name in clean_losses:
        assert clean_losses[name] == pytest.approx(noisy_losses[name], abs=1e-6), name


def test_overwriting_padded_fields_changes_no_gradient(rollout) -> None:
    """Stronger than the loss check: no parameter may move because of padding."""
    collector, batch = rollout
    padded = ~batch.valid
    noisy_advantages = batch.advantages.clone()
    noisy_advantages[padded] = 25.0
    noisy_returns = batch.returns.clone()
    noisy_returns[padded] = -25.0
    noisy = replace_field(batch, advantages=noisy_advantages, returns=noisy_returns)

    clean_gradients = _gradients(collector, batch)
    noisy_gradients = _gradients(collector, noisy)

    for name, clean in clean_gradients.items():
        assert torch.allclose(clean, noisy_gradients[name], atol=1e-7), name


def _losses(collector: RecurrentRolloutCollector, batch: ChunkBatch) -> dict[str, float]:
    from tarneeb.buffers.recurrent import masked_mean

    replayed = replay_chunks(collector.actor, collector.critic, batch)
    valid = batch.valid
    ratio = (replayed.log_probabilities - batch.old_log_probabilities).exp()
    return {
        "policy": float(masked_mean(ratio * batch.advantages, valid).detach()),
        "value": float(masked_mean((replayed.values - batch.returns) ** 2, valid).detach()),
        "entropy": float(masked_mean(replayed.entropies, valid).detach()),
    }


def _gradients(collector: RecurrentRolloutCollector, batch: ChunkBatch) -> dict[str, torch.Tensor]:
    from tarneeb.buffers.recurrent import masked_mean

    actor, critic = build_recurrent_actor_and_critic(seed=1)
    actor.load_state_dict(collector.actor.state_dict())
    critic.load_state_dict(collector.critic.state_dict())

    replayed = replay_chunks(actor, critic, batch)
    valid = batch.valid
    ratio = (replayed.log_probabilities - batch.old_log_probabilities).exp()
    loss = (
        -masked_mean(ratio * batch.advantages, valid)
        + 0.5 * masked_mean((replayed.values - batch.returns) ** 2, valid)
        - 0.01 * masked_mean(replayed.entropies, valid)
    )
    actor.zero_grad()
    critic.zero_grad()
    loss.backward()
    return {
        name: parameter.grad.clone()
        for name, parameter in [*actor.named_parameters(), *critic.named_parameters()]
        if parameter.grad is not None
    }


# --- gradients --------------------------------------------------------------


def test_the_gru_receives_a_finite_non_zero_gradient(rollout) -> None:
    """A recurrent update that never touches the GRU is not recurrent training."""
    collector, batch = rollout
    gradients = _gradients(collector, batch)

    gru_names = [name for name in gradients if ".gru." in name]
    assert gru_names, "the GRU parameters must appear in the gradient set"
    for name in gru_names:
        assert torch.isfinite(gradients[name]).all(), name
    assert any(bool((gradients[name] != 0).any()) for name in gru_names)


def test_the_event_embedding_tables_receive_gradient(rollout) -> None:
    collector, batch = rollout
    gradients = _gradients(collector, batch)

    table_names = [name for name in gradients if "embedding.tables" in name]
    assert table_names
    assert any(bool((gradients[name] != 0).any()) for name in table_names)


def test_actor_and_critic_gradients_stay_in_their_own_networks(rollout) -> None:
    """A policy-only loss must leave every critic parameter untouched."""
    from tarneeb.buffers.recurrent import masked_mean

    collector, batch = rollout
    actor, critic = build_recurrent_actor_and_critic(seed=1)
    actor.load_state_dict(collector.actor.state_dict())
    critic.load_state_dict(collector.critic.state_dict())

    replayed = replay_chunks(actor, critic, batch)
    policy_only_loss = -masked_mean(replayed.log_probabilities * batch.advantages, batch.valid)
    actor.zero_grad()
    critic.zero_grad()
    policy_only_loss.backward()

    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in actor.parameters()
    )
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_the_stored_initial_state_is_detached(rollout) -> None:
    """Truncated BPTT: gradients must not flow out through ``h0``."""
    collector, batch = rollout
    hidden = batch.initial_actor_hidden.clone().requires_grad_(True)
    detached_batch = replace_field(batch, initial_actor_hidden=hidden)

    replayed = replay_chunks(collector.actor, collector.critic, detached_batch)
    replayed.log_probabilities.sum().backward()

    assert hidden.grad is None or float(hidden.grad.abs().sum()) == 0.0


def test_rollout_collection_carries_no_autograd_graph(rollout) -> None:
    """Every rollout forward runs under no_grad, so nothing is retained."""
    collector, batch = rollout
    del batch
    for tensor in collector.state.actor_hidden.values():
        assert not tensor.requires_grad
        assert tensor.grad_fn is None
    for tensor in collector.state.critic_hidden.values():
        assert not tensor.requires_grad


# --- a real tiny update -----------------------------------------------------


def test_a_tiny_recurrent_ppo_update_runs_and_stays_finite(rollout) -> None:
    collector, batch = rollout
    actor, critic = build_recurrent_actor_and_critic(seed=1)
    actor.load_state_dict(collector.actor.state_dict())
    critic.load_state_dict(collector.critic.state_dict())
    config = PPOConfig(update_epochs=1, minibatch_count=2)
    optimizer = build_recurrent_optimizer(actor, critic, config)

    metrics = recurrent_ppo_update(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        batch=batch,
        config=config,
        shuffle_generator=np.random.default_rng(0),
    )

    assert metrics.optimizer_steps >= 1
    assert metrics.valid_transition_count == batch.valid_transition_count
    assert (
        metrics.epoch_zero_log_probability_max_deviation
        < RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )
    assert np.isfinite(
        [metrics.policy_loss, metrics.value_loss, metrics.entropy, metrics.gradient_norm]
    ).all()
    assert gradients_are_finite(actor)
    assert gradients_are_finite(critic)
    assert gru_gradient_norm(actor) > 0.0
    assert gru_gradient_norm(critic) > 0.0


def test_an_update_changes_the_gru_weights(rollout) -> None:
    collector, batch = rollout
    actor, critic = build_recurrent_actor_and_critic(seed=1)
    actor.load_state_dict(collector.actor.state_dict())
    critic.load_state_dict(collector.critic.state_dict())
    before = actor.trunk.history_encoder.gru.weight_hh_l0.clone()
    config = PPOConfig(update_epochs=1, minibatch_count=1)

    recurrent_ppo_update(
        actor=actor,
        critic=critic,
        optimizer=build_recurrent_optimizer(actor, critic, config),
        batch=batch,
        config=config,
        shuffle_generator=np.random.default_rng(1),
    )

    assert not torch.equal(actor.trunk.history_encoder.gru.weight_hh_l0, before)


def test_minibatches_sample_whole_chunks(rollout) -> None:
    """Splitting a chunk across minibatches would lose the state carried within it."""
    from tarneeb.training.recurrent_ppo import _chunk_minibatches

    _collector, batch = rollout
    groups = _chunk_minibatches(batch.chunk_count, 4, np.random.default_rng(0))

    covered = np.concatenate(groups)
    assert sorted(covered.tolist()) == list(range(batch.chunk_count))
    for group in groups:
        selected = batch.select(torch.from_numpy(group))
        assert selected.chunk_length == batch.chunk_length
        assert bool((selected.previous_event_lengths[0] == 0).all())


def test_advantage_normalization_ignores_padded_positions(rollout) -> None:
    from tarneeb.training.recurrent_ppo import normalize_advantages

    _collector, batch = rollout
    padded = ~batch.valid
    noisy = batch.advantages.clone()
    noisy[padded] = 1000.0

    clean_normalized = normalize_advantages(batch.advantages, batch.valid)
    noisy_normalized = normalize_advantages(noisy, batch.valid)

    assert torch.allclose(clean_normalized[batch.valid], noisy_normalized[batch.valid], atol=1e-5)
    assert float(clean_normalized[batch.valid].mean()) == pytest.approx(0.0, abs=1e-4)
