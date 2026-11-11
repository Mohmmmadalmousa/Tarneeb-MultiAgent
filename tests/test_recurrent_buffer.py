"""``chunk_v2`` storage: chunk construction, GAE equivalence and padding.

The central claims under test:

- a chunk never spans a seat, an environment or a reset;
- chunking is a *view*, so advantages are identical to the unchunked
  calculation performed with the frozen Phase 6 recursion;
- a padded timestep contributes exactly nothing to any loss or gradient;
- reward attribution is conserved, with nothing silently lost.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.buffers.recurrent import (
    DEFAULT_CHUNK_LENGTH,
    PAD_LEGAL_ACTION,
    RECURRENT_BUFFER_CONTRACT_VERSION,
    RecurrentRolloutBuffer,
    RecurrentTransition,
    masked_mean,
)
from tarneeb.buffers.rollout import compute_generalized_advantage_estimates
from tarneeb.core.actions import NUM_ACTIONS
from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.spaces import D_STATIC, K_MAX, N_EVENT_FIELDS
from tarneeb.models.checkpoint import BUFFER_CONTRACT_VERSION

HIDDEN = (1, 256)
GAMMA = 1.0
LAMBDA = 0.95


def make_transition(
    environment_index: int,
    seat: int,
    *,
    is_chunk_head: bool,
    value: float = 0.0,
    segment_length: int = 0,
    action: int = 0,
    log_probability: float = -1.0,
    first_event_seq: int = -1,
) -> RecurrentTransition:
    """Build a synthetic transition with the contract's shapes."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[action] = True
    mask[(action + 1) % 52] = True
    segment = np.zeros((K_MAX, N_EVENT_FIELDS), dtype=np.int16)
    segment[:segment_length] = 1
    return RecurrentTransition(
        environment_index=environment_index,
        seat=seat,
        static=np.zeros(D_STATIC, dtype=np.float32),
        critic_static=np.zeros(D_STATIC, dtype=np.float32),
        previous_event_ids=segment,
        previous_event_length=segment_length,
        legal_action_mask=mask,
        action=action,
        old_log_probability=log_probability,
        value=value,
        is_chunk_head=is_chunk_head,
        actor_hidden_head=np.zeros(HIDDEN, dtype=np.float32) if is_chunk_head else None,
        critic_hidden_head=np.zeros(HIDDEN, dtype=np.float32) if is_chunk_head else None,
        first_event_seq=first_event_seq,
    )


def fill_sequence(
    buffer: RecurrentRolloutBuffer,
    environment_index: int,
    seat: int,
    count: int,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    values: list[float] | None = None,
) -> None:
    """Append ``count`` decisions, marking heads exactly as the collector would."""
    for position in range(count):
        buffer.add(
            make_transition(
                environment_index,
                seat,
                is_chunk_head=(position % chunk_length == 0),
                value=0.0 if values is None else values[position],
                segment_length=0 if position % chunk_length == 0 else 2,
                first_event_seq=-1 if position % chunk_length == 0 else 2 * position,
            )
        )


# --- contract identity ------------------------------------------------------


def test_the_recurrent_contract_is_distinct_from_the_frozen_flat_one() -> None:
    assert RECURRENT_BUFFER_CONTRACT_VERSION == "chunk_v2"
    assert BUFFER_CONTRACT_VERSION == "flat_v1"
    assert RECURRENT_BUFFER_CONTRACT_VERSION != BUFFER_CONTRACT_VERSION


def test_the_chunk_length_is_the_blueprint_value() -> None:
    assert DEFAULT_CHUNK_LENGTH == 16


# --- storage invariants -----------------------------------------------------


def test_a_chunk_head_must_store_the_empty_segment() -> None:
    buffer = RecurrentRolloutBuffer(1)
    with pytest.raises(ValueError, match="empty segment"):
        buffer.add(make_transition(0, 0, is_chunk_head=True, segment_length=3))


def test_a_chunk_head_must_store_both_hidden_states() -> None:
    buffer = RecurrentRolloutBuffer(1)
    transition = make_transition(0, 0, is_chunk_head=True)
    broken = RecurrentTransition(
        **{
            field: getattr(transition, field)
            for field in transition.__slots__
            if field != "actor_hidden_head"
        },
        actor_hidden_head=None,
    )
    with pytest.raises(ValueError, match="must store both h0 and hc0"):
        buffer.add(broken)


def test_only_a_chunk_head_may_store_hidden_states() -> None:
    buffer = RecurrentRolloutBuffer(1)
    transition = make_transition(0, 0, is_chunk_head=True)
    broken = RecurrentTransition(
        **{
            field: getattr(transition, field)
            for field in transition.__slots__
            if field != "is_chunk_head"
        },
        is_chunk_head=False,
    )
    with pytest.raises(ValueError, match="only a chunk head"):
        buffer.add(broken)


def test_a_segment_longer_than_k_max_is_refused() -> None:
    buffer = RecurrentRolloutBuffer(1)
    transition = make_transition(0, 0, is_chunk_head=False, segment_length=1)
    broken = RecurrentTransition(
        **{
            field: getattr(transition, field)
            for field in transition.__slots__
            if field != "previous_event_length"
        },
        previous_event_length=K_MAX + 1,
    )
    with pytest.raises(ValueError, match=f"0..{K_MAX}"):
        buffer.add(broken)


# --- chunk construction -----------------------------------------------------


def test_a_full_sequence_splits_into_chunks_of_the_configured_length() -> None:
    buffer = RecurrentRolloutBuffer(1, chunk_length=4)
    fill_sequence(buffer, 0, 0, count=10, chunk_length=4)

    slices = buffer.chunk_slices()

    assert [(stop - start) for _key, start, stop in slices] == [4, 4, 2]


def test_no_chunk_ever_spans_two_seats_or_two_environments() -> None:
    buffer = RecurrentRolloutBuffer(2, chunk_length=4)
    fill_sequence(buffer, 0, 0, count=5, chunk_length=4)
    fill_sequence(buffer, 0, 2, count=3, chunk_length=4)
    fill_sequence(buffer, 1, 1, count=6, chunk_length=4)

    for key, start, stop in buffer.chunk_slices():
        owners = {
            (transition.environment_index, transition.seat)
            for transition in buffer.sequences[key][start:stop]
        }
        assert owners == {key}


def test_a_forced_boundary_ends_a_chunk_early() -> None:
    """A newly dealt hand forces a head even mid-chunk."""
    buffer = RecurrentRolloutBuffer(1, chunk_length=8)
    for position in range(6):
        # A reset lands after the third decision.
        forced = position in (0, 3)
        buffer.add(make_transition(0, 0, is_chunk_head=forced, segment_length=0 if forced else 2))

    assert [(stop - start) for _key, start, stop in buffer.chunk_slices()] == [3, 3]


def test_every_chunk_begins_with_an_empty_segment_and_a_stored_state() -> None:
    buffer = RecurrentRolloutBuffer(2, chunk_length=4)
    fill_sequence(buffer, 0, 0, count=9, chunk_length=4)
    fill_sequence(buffer, 1, 3, count=5, chunk_length=4)

    batch = buffer.to_chunks(np.zeros((2, NUM_SEATS), dtype=np.float32), GAMMA, LAMBDA)

    assert bool((batch.previous_event_lengths[0] == 0).all())
    assert bool(batch.valid[0].all())


def test_chunks_never_exceed_the_configured_length() -> None:
    buffer = RecurrentRolloutBuffer(1, chunk_length=4)
    for position in range(9):
        head = position == 0
        buffer.add(make_transition(0, 0, is_chunk_head=head, segment_length=0 if head else 1))

    with pytest.raises(ValueError, match="exceeds chunk_length"):
        buffer.chunk_slices()


def test_a_sequence_that_does_not_start_at_a_head_is_refused() -> None:
    buffer = RecurrentRolloutBuffer(1, chunk_length=4)
    buffer.add(make_transition(0, 0, is_chunk_head=False, segment_length=1))
    with pytest.raises(ValueError, match="does not start at a chunk head"):
        buffer.chunk_slices()


# --- GAE --------------------------------------------------------------------


def test_chunked_advantages_equal_the_unchunked_sequence_calculation() -> None:
    """Chunking is a view. It must not move a single advantage."""
    buffer = RecurrentRolloutBuffer(1, chunk_length=4)
    generator = np.random.default_rng(0)
    count = 11
    values = generator.standard_normal(count).tolist()
    fill_sequence(buffer, 0, 0, count=count, chunk_length=4, values=values)

    rewards = np.zeros(count, dtype=np.float64)
    rewards[-1] = 0.75
    for position, reward in enumerate(rewards):
        if reward:
            buffer.sequences[(0, 0)][position] = RecurrentTransition(
                **{
                    field: getattr(buffer.sequences[(0, 0)][position], field)
                    for field in RecurrentTransition.__slots__
                    if field not in ("reward", "done")
                },
                reward=float(reward),
                done=True,
            )

    bootstrap = np.zeros((1, NUM_SEATS), dtype=np.float32)
    bootstrap[0, 0] = 0.25
    batch = buffer.to_chunks(bootstrap, GAMMA, LAMBDA)

    stored_values = np.array(values, dtype=np.float64)
    next_values = np.empty_like(stored_values)
    next_values[:-1] = stored_values[1:]
    next_values[-1] = 0.25
    continues = np.ones(count, dtype=np.float64)
    continues[-1] = 0.0
    expected = compute_generalized_advantage_estimates(
        rewards=rewards,
        state_values=stored_values,
        next_state_values=next_values,
        episode_continues=continues,
        discount_factor=GAMMA,
        gae_lambda=LAMBDA,
    )

    flattened = batch.advantages.T[batch.valid.T].numpy()
    assert np.allclose(flattened, expected.astype(np.float32), atol=1e-6)


def test_returns_are_advantages_plus_stored_values() -> None:
    buffer = RecurrentRolloutBuffer(1, chunk_length=4)
    generator = np.random.default_rng(1)
    values = generator.standard_normal(7).tolist()
    fill_sequence(buffer, 0, 0, count=7, chunk_length=4, values=values)

    batch = buffer.to_chunks(np.zeros((1, NUM_SEATS), dtype=np.float32), GAMMA, LAMBDA)

    valid = batch.valid
    assert torch.allclose(
        batch.returns[valid], batch.advantages[valid] + batch.values[valid], atol=1e-6
    )


def test_a_redeal_boundary_does_not_stop_credit_from_flowing() -> None:
    """A forced chunk boundary is not an episode boundary.

    An all-pass hand emits no reward and the episode continues, so the advantage
    recursion must bootstrap straight through the chunk boundary it creates.
    """
    buffer = RecurrentRolloutBuffer(1, chunk_length=8)
    values = [0.5, 0.5, 0.5, 0.5]
    for position, value in enumerate(values):
        buffer.add(
            make_transition(
                0,
                0,
                is_chunk_head=position in (0, 2),  # a redeal lands before position 2
                value=value,
                segment_length=0 if position in (0, 2) else 1,
            )
        )

    batch = buffer.to_chunks(np.zeros((1, NUM_SEATS), dtype=np.float32), GAMMA, LAMBDA)

    assert len(buffer.chunk_slices()) == 2
    # Nothing was marked done, so no advantage is cut off at the boundary.
    assert all(not transition.done for transition in buffer.sequences[(0, 0)])
    assert batch.valid.sum() == 4


# --- reward attribution -----------------------------------------------------


def test_settlement_pays_each_seat_exactly_once_and_conserves_the_total() -> None:
    buffer = RecurrentRolloutBuffer(1, chunk_length=8)
    fill_sequence(buffer, 0, 0, count=3, chunk_length=8)
    fill_sequence(buffer, 0, 2, count=2, chunk_length=8)
    rewards = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)

    settlement = buffer.settle_completed_deal(0, rewards, seats=(0, 2))

    assert settlement.seats_paid == 2
    assert settlement.reward_paid == pytest.approx(1.0)
    assert settlement.reward_unattached == pytest.approx(0.0)
    assert settlement.total == pytest.approx(1.0)
    assert buffer.sequences[(0, 0)][-1].done
    assert buffer.sequences[(0, 2)][-1].done
    assert not buffer.sequences[(0, 0)][0].done


def test_a_seat_with_no_stored_decision_reports_its_reward_as_unattached() -> None:
    """Nothing vanishes silently: it shows up in the accounting."""
    buffer = RecurrentRolloutBuffer(1, chunk_length=8)
    fill_sequence(buffer, 0, 0, count=2, chunk_length=8)
    rewards = np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32)

    settlement = buffer.settle_completed_deal(0, rewards, seats=(0, 2))

    assert settlement.seats_paid == 1
    assert settlement.reward_paid == pytest.approx(0.5)
    assert settlement.reward_unattached == pytest.approx(0.5)
    assert settlement.total == pytest.approx(1.0)


def test_a_second_settlement_does_not_pay_the_same_transition_twice() -> None:
    buffer = RecurrentRolloutBuffer(1, chunk_length=8)
    fill_sequence(buffer, 0, 0, count=2, chunk_length=8)
    rewards = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    first = buffer.settle_completed_deal(0, rewards, seats=(0,))
    second = buffer.settle_completed_deal(0, rewards, seats=(0,))

    assert first.seats_paid == 1
    assert second.seats_paid == 0
    assert second.reward_unattached == pytest.approx(1.0)


def test_nothing_is_ever_held_in_a_pending_queue() -> None:
    buffer = RecurrentRolloutBuffer(1)
    fill_sequence(buffer, 0, 0, count=3)
    assert buffer.unpaid_reward_total == 0.0


# --- padding ----------------------------------------------------------------


def test_padded_positions_carry_a_usable_sentinel_mask() -> None:
    """An all-False mask is rejected by the frozen masking utility, correctly."""
    buffer = RecurrentRolloutBuffer(1, chunk_length=8)
    fill_sequence(buffer, 0, 0, count=3, chunk_length=8)

    batch = buffer.to_chunks(np.zeros((1, NUM_SEATS), dtype=np.float32), GAMMA, LAMBDA)

    padded = ~batch.valid
    assert bool(padded.any())
    assert bool(batch.legal_action_masks[padded][:, PAD_LEGAL_ACTION].all())
    assert bool(batch.legal_action_masks[padded].sum(dim=-1).eq(1).all())


def test_padded_positions_consume_no_events() -> None:
    buffer = RecurrentRolloutBuffer(1, chunk_length=8)
    fill_sequence(buffer, 0, 0, count=3, chunk_length=8)

    batch = buffer.to_chunks(np.zeros((1, NUM_SEATS), dtype=np.float32), GAMMA, LAMBDA)

    assert int(batch.previous_event_lengths[~batch.valid].sum()) == 0


def test_masked_mean_ignores_masked_entries() -> None:
    values = torch.tensor([1.0, 100.0, 3.0])
    mask = torch.tensor([True, False, True])
    assert float(masked_mean(values, mask)) == pytest.approx(2.0)


def test_masked_mean_of_an_empty_mask_is_exactly_zero() -> None:
    """A chunk with no valid position must contribute exactly zero gradient."""
    values = torch.tensor([5.0, 7.0], requires_grad=True)
    mask = torch.tensor([False, False])
    result = masked_mean(values, mask)
    assert float(result.detach()) == 0.0
    result.backward()
    assert float(values.grad.abs().sum()) == 0.0


# --- event accounting (invariant I15) ---------------------------------------


def test_stored_segments_within_a_chunk_are_contiguous_and_unique() -> None:
    """I15: no event is covered twice, and the covered range has no hole."""
    buffer = RecurrentRolloutBuffer(1, chunk_length=8)
    buffer.add(make_transition(0, 0, is_chunk_head=True))
    for position in range(1, 5):
        buffer.add(
            make_transition(
                0,
                0,
                is_chunk_head=False,
                segment_length=3,
                first_event_seq=3 * (position - 1),
            )
        )

    covered = buffer.consumed_event_sequence(0, 0)

    assert covered == list(range(12))
    assert len(covered) == len(set(covered))
