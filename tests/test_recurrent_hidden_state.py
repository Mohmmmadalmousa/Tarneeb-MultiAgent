"""Hidden-state ownership, isolation, reset and consumption bookkeeping.

Every test here is written to catch one specific way recurrent state can leak or
go stale. They deliberately mutate one slot and assert another did not move,
rather than asserting that two slots merely *happen* to differ: a network is
allowed to map different inputs to similar states (guardrail G7), so only
storage-level and causal claims are meaningful.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.event_features import EVENT_FIELD_CARDINALITIES
from tarneeb.env.spaces import K_MAX, N_EVENT_FIELDS
from tarneeb.models.recurrent import (
    RecurrentArchitectureSpec,
    build_recurrent_actor_and_critic,
)
from tarneeb.training.recurrent_state import (
    LEARNER_POLICY_ID,
    NO_EVENT_CONSUMED,
    PendingEventQueue,
    RecurrentStateStore,
    recurrent_states_are_disjoint,
)

ENVIRONMENTS = 3


@pytest.fixture
def store() -> RecurrentStateStore:
    return RecurrentStateStore(ENVIRONMENTS, RecurrentArchitectureSpec())


def random_ids(count: int, generator: np.random.Generator) -> np.ndarray:
    rows = np.zeros((count, N_EVENT_FIELDS), dtype=np.int16)
    for field_index, cardinality in enumerate(EVENT_FIELD_CARDINALITIES):
        rows[:, field_index] = generator.integers(0, cardinality, size=count)
    return rows


# --- shape and ownership ----------------------------------------------------


def test_every_environment_and_seat_owns_a_separate_slot(store: RecurrentStateStore) -> None:
    spec = store.spec
    expected = (ENVIRONMENTS, NUM_SEATS, spec.gru_layer_count, spec.hidden_size)
    assert store.actor_hidden[LEARNER_POLICY_ID].shape == expected
    assert store.critic_hidden[LEARNER_POLICY_ID].shape == expected


def test_actor_and_critic_hidden_tensors_are_distinct_storage(
    store: RecurrentStateStore,
) -> None:
    """Not merely equal-valued: they must be different objects in memory."""
    assert recurrent_states_are_disjoint(store)
    actor = store.actor_hidden[LEARNER_POLICY_ID]
    critic = store.critic_hidden[LEARNER_POLICY_ID]
    assert actor.data_ptr() != critic.data_ptr()

    actor[0, 0] = 1.0
    assert float(critic[0, 0].abs().sum()) == 0.0


def test_initial_hidden_state_is_exactly_zero(store: RecurrentStateStore) -> None:
    assert float(store.actor_hidden[LEARNER_POLICY_ID].abs().sum()) == 0.0
    assert float(store.critic_hidden[LEARNER_POLICY_ID].abs().sum()) == 0.0
    assert (store.last_seq_consumed == NO_EVENT_CONSUMED).all()
    assert store.force_new_chunk.all()


def test_writing_one_seat_does_not_touch_another_seat(store: RecurrentStateStore) -> None:
    """The single most important isolation property: seats do not share memory."""
    before = store.actor_hidden[LEARNER_POLICY_ID].clone()
    store.actor_hidden[LEARNER_POLICY_ID][0, 1] = 3.5

    changed = store.actor_hidden[LEARNER_POLICY_ID]
    assert float(changed[0, 1].sum()) != 0.0
    for seat in (0, 2, 3):
        assert torch.equal(changed[0, seat], before[0, seat])


def test_writing_one_environment_does_not_touch_another_environment(
    store: RecurrentStateStore,
) -> None:
    before = store.actor_hidden[LEARNER_POLICY_ID].clone()
    store.actor_hidden[LEARNER_POLICY_ID][1] = 2.0

    changed = store.actor_hidden[LEARNER_POLICY_ID]
    assert torch.equal(changed[0], before[0])
    assert torch.equal(changed[2], before[2])


def test_gather_and_scatter_round_trip_exactly(store: RecurrentStateStore) -> None:
    environments = np.array([0, 1, 2, 0], dtype=np.int64)
    seats = np.array([0, 1, 2, 3], dtype=np.int64)
    values = torch.randn(store.spec.gru_layer_count, 4, store.spec.hidden_size)

    store.scatter_actor(environments, seats, values)

    assert torch.equal(store.gather_actor(environments, seats), values)


def test_scatter_writes_only_the_named_slots(store: RecurrentStateStore) -> None:
    environments = np.array([1], dtype=np.int64)
    seats = np.array([2], dtype=np.int64)
    store.scatter_actor(
        environments, seats, torch.ones(store.spec.gru_layer_count, 1, store.spec.hidden_size)
    )

    tensor = store.actor_hidden[LEARNER_POLICY_ID]
    assert float(tensor[1, 2].sum()) == store.spec.hidden_size
    assert float(tensor.sum()) == store.spec.hidden_size  # nothing else was written


def test_a_second_policy_gets_its_own_slots() -> None:
    """A frozen opponent must not share memory with the learner (guardrail G5)."""
    store = RecurrentStateStore(
        2, RecurrentArchitectureSpec(), policy_ids=(LEARNER_POLICY_ID, "frozen_checkpoint")
    )
    assert recurrent_states_are_disjoint(store)

    store.actor_hidden[LEARNER_POLICY_ID][0, 0] = 1.0

    assert float(store.actor_hidden["frozen_checkpoint"][0, 0].abs().sum()) == 0.0


# --- a mutated slot must not change another seat's output -------------------


def test_changing_one_seats_state_cannot_change_another_seats_policy_output() -> None:
    """Causal isolation, not merely storage isolation.

    Seat 1's memory is scribbled over; seat 0's logits and advanced state must be
    bit-identical to what they were before.
    """
    actor, _critic = build_recurrent_actor_and_critic(seed=3)
    store = RecurrentStateStore(2, actor.spec)
    generator = np.random.default_rng(0)

    observation = torch.from_numpy(
        generator.standard_normal((1, actor.spec.actor_observation_dim)).astype(np.float32)
    )
    ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    ids[0, :3] = torch.from_numpy(random_ids(3, generator).astype(np.int64))
    lengths = torch.tensor([3], dtype=torch.int64)

    with torch.no_grad():
        before_logits, before_hidden = actor(observation, ids, lengths, store.actor_slot(0, 0))

    store.actor_hidden[LEARNER_POLICY_ID][0, 1] = 9.0
    store.actor_hidden[LEARNER_POLICY_ID][1, 0] = -4.0

    with torch.no_grad():
        after_logits, after_hidden = actor(observation, ids, lengths, store.actor_slot(0, 0))

    assert torch.equal(before_logits, after_logits)
    assert torch.equal(before_hidden, after_hidden)


def test_changing_the_critic_state_cannot_change_the_actor_output() -> None:
    """The actor and critic recurrent states must be causally independent."""
    actor, _critic = build_recurrent_actor_and_critic(seed=4)
    store = RecurrentStateStore(1, actor.spec)
    observation = torch.zeros((1, actor.spec.actor_observation_dim))
    ids = torch.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
    lengths = torch.zeros(1, dtype=torch.int64)

    with torch.no_grad():
        before_logits, _ = actor(observation, ids, lengths, store.actor_slot(0, 0))
    store.critic_hidden[LEARNER_POLICY_ID][0, 0] = 7.0
    with torch.no_grad():
        after_logits, _ = actor(observation, ids, lengths, store.actor_slot(0, 0))

    assert torch.equal(before_logits, after_logits)


def test_environments_advance_independently_through_the_encoder() -> None:
    """Advancing environment A's seats must leave environment B numerically identical."""
    actor, _critic = build_recurrent_actor_and_critic(seed=5)
    store = RecurrentStateStore(2, actor.spec)
    generator = np.random.default_rng(1)

    seats = np.arange(NUM_SEATS, dtype=np.int64)
    ids = np.zeros((NUM_SEATS, K_MAX, N_EVENT_FIELDS), dtype=np.int64)
    ids[:, :4] = random_ids(4, generator).astype(np.int64)
    lengths = torch.full((NUM_SEATS,), 4, dtype=torch.int64)

    environment_b_before = store.actor_hidden[LEARNER_POLICY_ID][1].clone()

    environments = np.zeros(NUM_SEATS, dtype=np.int64)
    with torch.no_grad():
        advanced = actor.trunk.history_encoder.advance(
            torch.from_numpy(ids), lengths, store.gather_actor(environments, seats)
        )
    store.scatter_actor(environments, seats, advanced)

    assert float(store.actor_hidden[LEARNER_POLICY_ID][0].abs().sum()) > 0.0
    assert torch.equal(store.actor_hidden[LEARNER_POLICY_ID][1], environment_b_before)


# --- reset ------------------------------------------------------------------


def test_reset_returns_one_environment_to_exactly_the_defined_value(
    store: RecurrentStateStore,
) -> None:
    generator = np.random.default_rng(2)
    store.actor_hidden[LEARNER_POLICY_ID][1] = 1.5
    store.critic_hidden[LEARNER_POLICY_ID][1] = -1.5
    store.append_events(1, 0, random_ids(3, generator))
    store.last_seq_consumed[1, :] = 8
    store.chunk_position[1, :] = 5
    store.force_new_chunk[1, :] = False

    store.reset_environment(1)

    assert float(store.actor_hidden[LEARNER_POLICY_ID][1].abs().sum()) == 0.0
    assert float(store.critic_hidden[LEARNER_POLICY_ID][1].abs().sum()) == 0.0
    assert all(len(queue) == 0 for queue in store.pending[1])
    assert (store.last_seq_consumed[1] == NO_EVENT_CONSUMED).all()
    assert (store.chunk_position[1] == 0).all()
    assert store.force_new_chunk[1].all()


def test_reset_of_one_environment_leaves_the_others_alone(
    store: RecurrentStateStore,
) -> None:
    store.actor_hidden[LEARNER_POLICY_ID][:] = 1.0
    store.last_seq_consumed[:, :] = 4

    store.reset_environment(1)

    for environment_index in (0, 2):
        assert float(store.actor_hidden[LEARNER_POLICY_ID][environment_index].sum()) > 0.0
        assert (store.last_seq_consumed[environment_index] == 4).all()


def test_a_reset_deal_matches_a_freshly_built_store(store: RecurrentStateStore) -> None:
    """After a reset the state must be indistinguishable from a new environment."""
    generator = np.random.default_rng(3)
    store.actor_hidden[LEARNER_POLICY_ID][0] = 2.0
    store.append_events(0, 2, random_ids(5, generator))
    store.reset_environment(0)

    fresh = RecurrentStateStore(ENVIRONMENTS, store.spec)

    assert torch.equal(
        store.actor_hidden[LEARNER_POLICY_ID][0], fresh.actor_hidden[LEARNER_POLICY_ID][0]
    )
    assert torch.equal(
        store.critic_hidden[LEARNER_POLICY_ID][0], fresh.critic_hidden[LEARNER_POLICY_ID][0]
    )
    assert list(store.last_seq_consumed[0]) == list(fresh.last_seq_consumed[0])


def test_begin_rollout_forces_a_chunk_boundary_without_resetting_memory(
    store: RecurrentStateStore,
) -> None:
    """A rollout boundary truncates BPTT; it does not erase what a seat knows."""
    store.actor_hidden[LEARNER_POLICY_ID][0, 0] = 0.75
    store.last_seq_consumed[0, :] = 11
    store.force_new_chunk[:, :] = False
    store.chunk_position[:, :] = 7

    store.begin_rollout()

    assert store.force_new_chunk.all()
    assert (store.chunk_position == 0).all()
    assert float(store.actor_hidden[LEARNER_POLICY_ID][0, 0].sum()) == pytest.approx(
        0.75 * store.spec.hidden_size
    )
    assert (store.last_seq_consumed[0] == 11).all()


# --- pending queues and chunk positions -------------------------------------


def test_a_pending_queue_pads_to_k_max_and_reports_its_true_length() -> None:
    generator = np.random.default_rng(4)
    queue = PendingEventQueue()
    rows = random_ids(5, generator)
    queue.extend(rows)

    block, count = queue.as_padded_block()

    assert count == 5
    assert block.shape == (K_MAX, N_EVENT_FIELDS)
    assert np.array_equal(block[:5], rows)
    assert not block[5:].any()


def test_a_segment_longer_than_k_max_is_refused_rather_than_truncated() -> None:
    """Truncating would silently corrupt the auction record."""
    generator = np.random.default_rng(5)
    queue = PendingEventQueue()
    queue.extend(random_ids(K_MAX + 1, generator))

    with pytest.raises(ValueError, match="exceeds K_MAX"):
        queue.as_padded_block()


def test_a_chunk_head_stores_the_empty_segment(store: RecurrentStateStore) -> None:
    """R4: the events before a head are summarized by h0, so replaying them twice."""
    generator = np.random.default_rng(6)
    store.append_events(0, 1, random_ids(4, generator))

    head_ids, head_length = store.stored_segment(0, 1, is_chunk_head=True)
    body_ids, body_length = store.stored_segment(0, 1, is_chunk_head=False)

    assert head_length == 0
    assert not head_ids.any()
    assert body_length == 4
    assert body_ids[:4].any()


def test_chunk_position_wraps_at_the_configured_length(store: RecurrentStateStore) -> None:
    store.force_new_chunk[0, 0] = False
    for step in range(16):
        assert store.is_chunk_head(0, 0) == (step == 0)
        store.note_decision_stored(0, 0, chunk_length=16)
    assert store.is_chunk_head(0, 0)  # position wrapped back to zero


def test_storing_a_decision_clears_only_that_seats_backlog(
    store: RecurrentStateStore,
) -> None:
    generator = np.random.default_rng(7)
    for seat in range(NUM_SEATS):
        store.append_events(0, seat, random_ids(2, generator))

    store.note_decision_stored(0, 1, chunk_length=16)

    assert len(store.pending[0][1]) == 0
    for seat in (0, 2, 3):
        assert len(store.pending[0][seat]) == 2


# --- consumption accounting -------------------------------------------------


def test_consumption_assertion_accepts_a_matching_cursor(store: RecurrentStateStore) -> None:
    store.last_seq_consumed[0, :] = 6
    store.assert_consumption_matches(0, event_count=7)


def test_consumption_assertion_catches_a_dropped_event(store: RecurrentStateStore) -> None:
    """The single assertion that catches nearly every consumption bug."""
    store.last_seq_consumed[0, :] = 6
    store.last_seq_consumed[0, 2] = 5  # seat 2 missed one
    with pytest.raises(AssertionError, match="seat 2"):
        store.assert_consumption_matches(0, event_count=7)


def test_consumption_assertion_catches_a_duplicated_event(
    store: RecurrentStateStore,
) -> None:
    store.last_seq_consumed[0, :] = 7  # one ahead of the engine
    with pytest.raises(AssertionError, match="has consumed up to seq 7"):
        store.assert_consumption_matches(0, event_count=7)


# --- resume round trip ------------------------------------------------------


def test_snapshot_and_restore_reproduce_the_store_exactly(
    store: RecurrentStateStore,
) -> None:
    generator = np.random.default_rng(8)
    store.actor_hidden[LEARNER_POLICY_ID].normal_()
    store.critic_hidden[LEARNER_POLICY_ID].normal_()
    store.append_events(0, 1, random_ids(3, generator))
    store.append_events(2, 3, random_ids(1, generator))
    store.last_seq_consumed[:, :] = 4
    store.chunk_position[1, 2] = 9
    store.force_new_chunk[2, 1] = False

    snapshot = store.snapshot()
    restored = RecurrentStateStore(ENVIRONMENTS, store.spec)
    restored.restore(snapshot)

    assert torch.equal(
        restored.actor_hidden[LEARNER_POLICY_ID], store.actor_hidden[LEARNER_POLICY_ID]
    )
    assert torch.equal(
        restored.critic_hidden[LEARNER_POLICY_ID], store.critic_hidden[LEARNER_POLICY_ID]
    )
    assert np.array_equal(restored.last_seq_consumed, store.last_seq_consumed)
    assert np.array_equal(restored.chunk_position, store.chunk_position)
    assert np.array_equal(restored.force_new_chunk, store.force_new_chunk)
    assert len(restored.pending[0][1]) == 3
    assert len(restored.pending[2][3]) == 1
    assert np.array_equal(
        restored.pending[0][1].as_padded_block()[0], store.pending[0][1].as_padded_block()[0]
    )


def test_a_snapshot_does_not_alias_the_live_tensors(store: RecurrentStateStore) -> None:
    """Otherwise a later rollout step would silently rewrite the saved state."""
    snapshot = store.snapshot()
    store.actor_hidden[LEARNER_POLICY_ID][0, 0] = 5.0
    assert float(snapshot["actor_hidden"][LEARNER_POLICY_ID][0, 0].abs().sum()) == 0.0


def test_restore_refuses_a_snapshot_from_a_different_environment_count(
    store: RecurrentStateStore,
) -> None:
    snapshot = store.snapshot()
    smaller = RecurrentStateStore(2, store.spec)
    with pytest.raises(ValueError, match="snapshot actor state is"):
        smaller.restore(snapshot)
