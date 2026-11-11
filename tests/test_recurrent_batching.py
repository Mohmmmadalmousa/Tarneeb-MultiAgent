"""Phase 9 Part A: the batched collector must agree with the frozen Phase 8 one.

The frozen `RecurrentRolloutCollector` is the **oracle**. Phase 9 may not edit
it, so the only way to trust an optimized replacement is to run both from
identical state and compare everything they produce.

Exact float equality is not claimed: a batch-N matmul and a packed GRU reorder
floating-point work. So hidden states, logits and values are compared to 1e-5,
while everything discrete — sampled actions, event cursors, sequence ownership,
chunk heads, segment lengths and reset boundaries — must match **exactly**.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.core.actions import NO_TRUMP_ACTION
from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.spaces import K_MAX
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.recurrent import build_recurrent_actor_and_critic
from tarneeb.training.recurrent_batched_rollout import BatchedRecurrentRolloutCollector
from tarneeb.training.recurrent_rollout import (
    RecurrentRolloutCollector,
    RecurrentRolloutMetrics,
)
from tarneeb.training.recurrent_state import LEARNER_POLICY_ID

HIDDEN_TOLERANCE = 1e-5
ENVIRONMENTS = 4
ROLLOUT = 80


def build(
    collector_class, seed: int = 11, environments: int = ENVIRONMENTS, rollout: int = ROLLOUT
):
    """Build a collector of the given class from a fixed, reproducible state."""
    vector_environments = VecTarneeb(num_envs=environments, root_seed=seed)
    actor, critic = build_recurrent_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(2)
    return collector_class(
        environments=vector_environments,
        actor=actor,
        critic=critic,
        rollout_length=rollout,
        action_sampling_generator=generator,
        opponents=[BasicHeuristicAgent() for _ in range(environments)],
    )


@pytest.fixture(scope="module")
def both():
    """Run the frozen and the batched collector over identical configuration."""
    frozen = build(RecurrentRolloutCollector)
    batched = build(BatchedRecurrentRolloutCollector)
    frozen_buffer, frozen_metrics = frozen.collect()
    batched_buffer, batched_metrics = batched.collect()
    return (frozen, frozen_buffer, frozen_metrics), (batched, batched_buffer, batched_metrics)


# --- the batched collector is the frozen one, only faster -------------------


def test_the_batched_collector_subclasses_the_frozen_one() -> None:
    """Phase 9 attaches additively; the frozen semantics remain the reference."""
    assert issubclass(BatchedRecurrentRolloutCollector, RecurrentRolloutCollector)


def test_both_collectors_store_the_same_sequences(both) -> None:
    (_f, frozen_buffer, _fm), (_b, batched_buffer, _bm) = both
    assert frozen_buffer.sequences.keys() == batched_buffer.sequences.keys()
    for key, frozen_sequence in frozen_buffer.sequences.items():
        assert len(frozen_sequence) == len(batched_buffer.sequences[key]), key


def test_sampled_actions_are_identical(both) -> None:
    """The generator must be consumed in exactly the same order and amounts."""
    (_f, frozen_buffer, _fm), (_b, batched_buffer, _bm) = both
    for key, frozen_sequence in frozen_buffer.sequences.items():
        frozen_actions = [t.action for t in frozen_sequence]
        batched_actions = [t.action for t in batched_buffer.sequences[key]]
        assert frozen_actions == batched_actions, key


def test_legal_masks_and_chunk_structure_are_identical(both) -> None:
    (_f, frozen_buffer, _fm), (_b, batched_buffer, _bm) = both
    for key, frozen_sequence in frozen_buffer.sequences.items():
        for left, right in zip(frozen_sequence, batched_buffer.sequences[key], strict=True):
            assert np.array_equal(left.legal_action_mask, right.legal_action_mask)
            assert left.is_chunk_head == right.is_chunk_head
            assert left.previous_event_length == right.previous_event_length
            assert np.array_equal(left.previous_event_ids, right.previous_event_ids)
            assert left.first_event_seq == right.first_event_seq
            assert left.seat == right.seat
            assert left.environment_index == right.environment_index
            assert left.done == right.done
            assert left.reward == pytest.approx(right.reward, abs=1e-7)


def test_observations_are_identical(both) -> None:
    """Same engine state means the same actor-safe observation, bit for bit."""
    (_f, frozen_buffer, _fm), (_b, batched_buffer, _bm) = both
    for key, frozen_sequence in frozen_buffer.sequences.items():
        for left, right in zip(frozen_sequence, batched_buffer.sequences[key], strict=True):
            assert np.array_equal(left.static, right.static)
            assert np.array_equal(left.critic_static, right.critic_static)


def test_stored_log_probabilities_and_values_agree(both) -> None:
    (_f, frozen_buffer, _fm), (_b, batched_buffer, _bm) = both
    for key, frozen_sequence in frozen_buffer.sequences.items():
        for left, right in zip(frozen_sequence, batched_buffer.sequences[key], strict=True):
            assert left.old_log_probability == pytest.approx(
                right.old_log_probability, abs=HIDDEN_TOLERANCE
            )
            assert left.value == pytest.approx(right.value, abs=HIDDEN_TOLERANCE)


def test_stored_chunk_head_states_agree(both) -> None:
    """The stored h0/hc0 are what replay reconstructs from; they must match."""
    (_f, frozen_buffer, _fm), (_b, batched_buffer, _bm) = both
    compared = 0
    for key, frozen_sequence in frozen_buffer.sequences.items():
        for left, right in zip(frozen_sequence, batched_buffer.sequences[key], strict=True):
            if not left.is_chunk_head:
                continue
            compared += 1
            assert np.allclose(
                left.actor_hidden_head, right.actor_hidden_head, atol=HIDDEN_TOLERANCE
            )
            assert np.allclose(
                left.critic_hidden_head, right.critic_hidden_head, atol=HIDDEN_TOLERANCE
            )
    assert compared > 0


def test_final_actor_and_critic_hidden_states_agree(both) -> None:
    (frozen, _fb, _fm), (batched, _bb, _bm) = both
    assert torch.allclose(
        frozen.state.actor_hidden[LEARNER_POLICY_ID],
        batched.state.actor_hidden[LEARNER_POLICY_ID],
        atol=HIDDEN_TOLERANCE,
    )
    assert torch.allclose(
        frozen.state.critic_hidden[LEARNER_POLICY_ID],
        batched.state.critic_hidden[LEARNER_POLICY_ID],
        atol=HIDDEN_TOLERANCE,
    )


def test_event_cursors_and_chunk_bookkeeping_agree_exactly(both) -> None:
    (frozen, _fb, _fm), (batched, _bb, _bm) = both
    assert np.array_equal(frozen.state.last_seq_consumed, batched.state.last_seq_consumed)
    assert np.array_equal(frozen.state.chunk_position, batched.state.chunk_position)
    assert np.array_equal(frozen.state.force_new_chunk, batched.state.force_new_chunk)
    assert np.array_equal(frozen.redeal_counts, batched.redeal_counts)


def test_pending_queues_agree_exactly(both) -> None:
    """Compared row by row rather than through ``as_padded_block``.

    An opponent-held seat is never stored, so its backlog legitimately grows
    past ``K_MAX`` -- the bound applies to a *stored* segment and is enforced at
    the point of storage. Padding such a queue would raise, correctly.
    """
    (frozen, _fb, _fm), (batched, _bb, _bm) = both
    for environment_index in range(ENVIRONMENTS):
        for seat in range(NUM_SEATS):
            left = frozen.state.pending[environment_index][seat]
            right = batched.state.pending[environment_index][seat]
            assert len(left) == len(right), (environment_index, seat)
            for left_row, right_row in zip(left.rows, right.rows, strict=True):
                assert np.array_equal(left_row, right_row)


def test_reset_boundaries_and_deal_counts_agree(both) -> None:
    (_f, _fb, frozen_metrics), (_b, _bb, batched_metrics) = both
    assert frozen_metrics.new_hand_boundaries == batched_metrics.new_hand_boundaries
    assert frozen_metrics.completed_deals == batched_metrics.completed_deals
    assert frozen_metrics.passed_out_deals == batched_metrics.passed_out_deals


def test_every_discrete_metric_agrees(both) -> None:
    (_f, _fb, frozen_metrics), (_b, _bb, batched_metrics) = both
    for field in (
        "environment_steps",
        "decision_count",
        "learner_decision_count",
        "stored_transition_count",
        "illegal_action_count",
        "no_trump_action_count",
        "auction_decision_count",
        "pass_action_count",
        "max_segment_length",
        "total_events_consumed",
    ):
        assert getattr(frozen_metrics, field) == getattr(batched_metrics, field), field


def test_reward_accounting_agrees(both) -> None:
    (_f, _fb, frozen_metrics), (_b, _bb, batched_metrics) = both
    assert frozen_metrics.total_reward_paid == pytest.approx(
        batched_metrics.total_reward_paid, abs=1e-6
    )
    assert frozen_metrics.total_learner_reward_emitted == pytest.approx(
        batched_metrics.total_learner_reward_emitted, abs=1e-6
    )


def test_bootstrap_values_agree(both) -> None:
    (frozen, _fb, _fm), (batched, _bb, _bm) = both
    assert np.allclose(frozen.bootstrap_values(), batched.bootstrap_values(), atol=HIDDEN_TOLERANCE)


def test_the_assembled_chunk_batches_agree(both) -> None:
    (frozen, frozen_buffer, _fm), (batched, batched_buffer, _bm) = both
    left = frozen_buffer.to_chunks(frozen.bootstrap_values(), 1.0, 0.95)
    right = batched_buffer.to_chunks(batched.bootstrap_values(), 1.0, 0.95)

    assert left.chunk_count == right.chunk_count
    assert left.valid_transition_count == right.valid_transition_count
    assert torch.equal(left.valid, right.valid)
    assert torch.equal(left.actions, right.actions)
    assert torch.equal(left.previous_event_lengths, right.previous_event_lengths)
    assert torch.equal(left.previous_event_ids, right.previous_event_ids)
    assert torch.equal(left.seats, right.seats)
    assert torch.equal(left.environment_indices, right.environment_indices)
    assert torch.allclose(left.advantages, right.advantages, atol=1e-4)
    assert torch.allclose(left.returns, right.returns, atol=1e-4)


# --- R1 and R2 are preserved, not merely claimed -----------------------------


def test_the_decision_forward_consumes_nothing(both) -> None:
    """R2. The batched collector asserts this internally; here it is observable."""
    (_f, _fb, _fm), (_b, batched_buffer, _bm) = both
    for sequence in batched_buffer.sequences.values():
        assert sequence[0].is_chunk_head
        assert sequence[0].previous_event_length == 0


def test_every_event_is_consumed_exactly_once_per_seat(both) -> None:
    """R1, checked against the engine after the batched run."""
    (_f, _fb, _fm), (batched, _bb, _bm) = both
    batched.assert_consumption_invariant()
    for environment_index in range(ENVIRONMENTS):
        cursors = batched.state.last_seq_consumed[environment_index]
        assert len(set(cursors.tolist())) == 1


def test_no_stored_segment_exceeds_the_bound(both) -> None:
    (_f, _fb, _fm), (_b, _bb, batched_metrics) = both
    assert 0 < batched_metrics.max_segment_length <= K_MAX
    assert batched_metrics.max_segment_length <= 14


# --- isolation still holds under batching ------------------------------------


def test_batching_never_moves_an_event_between_environments() -> None:
    """The central risk of cross-environment batching, tested head-on.

    Feed the batched advance a step in which **only environment 1** emitted
    events. Environment 1's four seats must move; environments 0 and 2 must be
    left byte-identical. This is the direct claim -- comparing whole rollouts at
    different environment counts would prove nothing, because all environments
    share one action-sampling generator.
    """
    collector = build(BatchedRecurrentRolloutCollector, seed=55, environments=3, rollout=12)
    collector.collect()

    before_actor = collector.state.actor_hidden[LEARNER_POLICY_ID].clone()
    before_critic = collector.state.critic_hidden[LEARNER_POLICY_ID].clone()
    before_cursor = collector.state.last_seq_consumed.copy()

    live = collector.environments.environments[1].state
    assert live is not None and live.events, "environment 1 needs a live event stream"
    events_by_environment = [(), tuple(live.events[-3:]), ()]

    collector._advance_every_environment(events_by_environment, RecurrentRolloutMetrics())

    actor = collector.state.actor_hidden[LEARNER_POLICY_ID]
    critic = collector.state.critic_hidden[LEARNER_POLICY_ID]
    for untouched in (0, 2):
        assert torch.equal(actor[untouched], before_actor[untouched])
        assert torch.equal(critic[untouched], before_critic[untouched])
        assert np.array_equal(
            collector.state.last_seq_consumed[untouched], before_cursor[untouched]
        )
    assert not torch.equal(actor[1], before_actor[1])
    assert not torch.equal(critic[1], before_critic[1])


def test_a_zero_length_row_is_an_exact_no_op_inside_a_batch() -> None:
    """An environment that emitted nothing must be untouched by its neighbours."""
    collector = build(BatchedRecurrentRolloutCollector, seed=57, environments=3, rollout=10)
    collector.collect()
    before = collector.state.actor_hidden[LEARNER_POLICY_ID].clone()

    live = collector.environments.environments[0].state
    assert live is not None and live.events
    collector._advance_every_environment(
        [tuple(live.events[-2:]), (), ()], RecurrentRolloutMetrics()
    )

    actor = collector.state.actor_hidden[LEARNER_POLICY_ID]
    assert torch.equal(actor[1], before[1])
    assert torch.equal(actor[2], before[2])


def test_mutating_one_environments_memory_cannot_reach_another() -> None:
    collector = build(BatchedRecurrentRolloutCollector, seed=61, environments=3, rollout=20)
    collector.collect()
    before = collector.state.actor_hidden[LEARNER_POLICY_ID].clone()

    collector.state.actor_hidden[LEARNER_POLICY_ID][1] = 4.0

    tensor = collector.state.actor_hidden[LEARNER_POLICY_ID]
    assert torch.equal(tensor[0], before[0])
    assert torch.equal(tensor[2], before[2])


def test_actor_and_critic_states_stay_separate_under_batching() -> None:
    collector = build(BatchedRecurrentRolloutCollector, seed=62, environments=2, rollout=30)
    collector.collect()
    actor_state = collector.state.actor_hidden[LEARNER_POLICY_ID]
    critic_state = collector.state.critic_hidden[LEARNER_POLICY_ID]
    assert actor_state.data_ptr() != critic_state.data_ptr()
    assert not torch.allclose(actor_state, critic_state)


# --- safety -----------------------------------------------------------------


def test_no_illegal_or_no_trump_action_under_batching(both) -> None:
    (_f, _fb, _fm), (_b, batched_buffer, batched_metrics) = both
    assert batched_metrics.illegal_action_count == 0
    assert batched_metrics.no_trump_action_count == 0
    for sequence in batched_buffer.sequences.values():
        for transition in sequence:
            assert bool(transition.legal_action_mask[transition.action])
            assert not bool(transition.legal_action_mask[NO_TRUMP_ACTION])


def test_the_batched_rollout_is_deterministic() -> None:
    first_buffer, _ = build(BatchedRecurrentRolloutCollector, seed=71, rollout=40).collect()
    second_buffer, _ = build(BatchedRecurrentRolloutCollector, seed=71, rollout=40).collect()
    for key, first_sequence in first_buffer.sequences.items():
        for left, right in zip(first_sequence, second_buffer.sequences[key], strict=True):
            assert left.action == right.action
            assert left.old_log_probability == right.old_log_probability
            assert left.value == right.value


def test_a_different_seed_diverges() -> None:
    """The negative control for the determinism test above."""
    first_buffer, _ = build(BatchedRecurrentRolloutCollector, seed=71, rollout=40).collect()
    second_buffer, _ = build(BatchedRecurrentRolloutCollector, seed=72, rollout=40).collect()
    first = [t.action for k in sorted(first_buffer.sequences) for t in first_buffer.sequences[k]]
    second = [t.action for k in sorted(second_buffer.sequences) for t in second_buffer.sequences[k]]
    assert first != second


# --- the optimized path still satisfies the Phase 8 epoch-zero gate ----------


def test_epoch_zero_holds_for_the_batched_collector() -> None:
    """The hard gate, re-run against the optimized path."""
    from tarneeb.training.recurrent_ppo import (
        RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
        replay_chunks,
    )

    collector = build(BatchedRecurrentRolloutCollector, seed=13, environments=4, rollout=96)
    buffer, _metrics = collector.collect()
    batch = buffer.to_chunks(collector.bootstrap_values(), 1.0, 0.95)

    replayed = replay_chunks(collector.actor, collector.critic, batch)

    assert (
        replayed.max_log_probability_deviation(batch)
        < RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )
    assert replayed.max_value_deviation(batch) < 1e-3


def test_corrupting_the_batched_initial_state_breaks_epoch_zero() -> None:
    """Negative control: the gate above must be capable of failing."""
    from dataclasses import replace as dataclass_replace

    from tarneeb.training.recurrent_ppo import (
        RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
        replay_chunks,
    )

    collector = build(BatchedRecurrentRolloutCollector, seed=13, environments=4, rollout=96)
    buffer, _metrics = collector.collect()
    batch = buffer.to_chunks(collector.bootstrap_values(), 1.0, 0.95)
    corrupted = dataclass_replace(batch, initial_actor_hidden=batch.initial_actor_hidden + 1.0)

    replayed = replay_chunks(collector.actor, collector.critic, corrupted)

    assert (
        replayed.max_log_probability_deviation(corrupted)
        > RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )
