"""Rollout collection: event consumption, resets, determinism and safety.

These tests run the real collector against the real frozen engine, because the
properties at stake -- that no public event is dropped or double-counted, that a
newly dealt hand resets memory, that a redeal is a chunk boundary but not an
episode boundary -- are properties of the collector *and* the engine together
and cannot be established from synthetic data.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.core.actions import NO_TRUMP_ACTION
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.state import team_of_seat
from tarneeb.env.spaces import K_MAX
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.recurrent import build_recurrent_actor_and_critic
from tarneeb.training.recurrent_rollout import (
    RecurrentRolloutCollector,
    fresh_hand_state_is_zero,
    relativized_event_ids,
)
from tarneeb.training.recurrent_state import LEARNER_POLICY_ID, NO_EVENT_CONSUMED

ENVIRONMENTS = 4
ROLLOUT = 80


def build_collector(
    seed: int = 11,
    environments: int = ENVIRONMENTS,
    rollout_length: int = ROLLOUT,
    opponent: str = "basic",
    chunk_length: int = 16,
) -> RecurrentRolloutCollector:
    vector_environments = VecTarneeb(num_envs=environments, root_seed=seed)
    actor, critic = build_recurrent_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(2)
    opponents = [
        RandomAgent(seed=index) if opponent == "random" else BasicHeuristicAgent()
        for index in range(environments)
    ]
    return RecurrentRolloutCollector(
        environments=vector_environments,
        actor=actor,
        critic=critic,
        rollout_length=rollout_length,
        action_sampling_generator=generator,
        opponents=opponents,
        chunk_length=chunk_length,
    )


@pytest.fixture(scope="module")
def collected() -> tuple[RecurrentRolloutCollector, object, object]:
    collector = build_collector()
    buffer, metrics = collector.collect()
    return collector, buffer, metrics


# --- initial state ----------------------------------------------------------


def test_a_new_collector_starts_every_environment_at_the_reset_value() -> None:
    collector = build_collector()
    for environment_index in range(ENVIRONMENTS):
        assert fresh_hand_state_is_zero(collector.state, environment_index)


def test_the_learner_holds_one_partnership_per_environment() -> None:
    collector = build_collector()
    for environment_index in range(ENVIRONMENTS):
        seats = collector.learner_seats(environment_index)
        assert len(seats) == 2
        assert team_of_seat(seats[0]) == team_of_seat(seats[1])
    # Alternating by index puts the shared policy in all four absolute seats.
    covered = {
        seat
        for environment_index in range(ENVIRONMENTS)
        for seat in collector.learner_seats(environment_index)
    }
    assert covered == set(range(NUM_SEATS))


# --- safety -----------------------------------------------------------------


def test_no_illegal_action_and_no_no_trump_action_is_ever_taken(collected) -> None:
    _collector, _buffer, metrics = collected
    assert metrics.illegal_action_count == 0
    assert metrics.no_trump_action_count == 0


def test_every_stored_action_is_legal_under_its_own_stored_mask(collected) -> None:
    _collector, buffer, _metrics = collected
    for sequence in buffer.sequences.values():
        for transition in sequence:
            assert bool(transition.legal_action_mask[transition.action])
            assert not bool(transition.legal_action_mask[NO_TRUMP_ACTION])


def test_only_learner_seats_are_ever_stored(collected) -> None:
    """Nothing an opponent chose can reach a gradient."""
    collector, buffer, _metrics = collected
    for environment_index, seat in buffer.sequences:
        assert collector.is_learner_seat(environment_index, seat)


# --- event consumption (rules R1, R2, invariant I15) ------------------------


def test_every_seat_has_consumed_exactly_the_engine_s_events(collected) -> None:
    """The core consumption invariant, checked against the engine after the run."""
    collector, _buffer, _metrics = collected
    for environment_index, environment in enumerate(collector.environments.environments):
        state = environment.state
        assert state is not None
        collector.state.assert_consumption_matches(environment_index, len(state.events))


def test_all_four_seats_advance_together_not_only_the_acting_one(collected) -> None:
    """R1. A seat must not lose events merely because another player acted."""
    collector, _buffer, _metrics = collected
    for environment_index in range(ENVIRONMENTS):
        cursors = collector.state.last_seq_consumed[environment_index]
        assert len(set(cursors.tolist())) == 1


def test_the_total_event_consumption_is_four_per_emitted_event(collected) -> None:
    _collector, _buffer, metrics = collected
    assert metrics.total_events_consumed % NUM_SEATS == 0
    assert metrics.total_events_consumed > 0


def test_no_stored_segment_exceeds_the_proved_bound(collected) -> None:
    """K_MAX is 16 with the tight bound proved at 14 (blueprint section 4.8)."""
    _collector, _buffer, metrics = collected
    assert 0 < metrics.max_segment_length <= K_MAX
    assert metrics.max_segment_length <= 14


def test_stored_segments_of_one_chunk_are_contiguous_and_never_overlap(collected) -> None:
    """Invariant I15, measured on real data rather than constructed data."""
    _collector, buffer, _metrics = collected
    for key, start, stop in buffer.chunk_slices():
        covered: list[int] = []
        for transition in buffer.sequences[key][start:stop]:
            if transition.previous_event_length:
                covered.extend(
                    range(
                        transition.first_event_seq,
                        transition.first_event_seq + transition.previous_event_length,
                    )
                )
        assert len(covered) == len(set(covered)), f"chunk {key} consumed an event twice"
        if covered:
            assert covered == sorted(covered)


def test_the_relativized_ids_of_one_event_differ_between_seats() -> None:
    """The same event encodes differently per seat; that is what lets one net serve all."""
    environments = VecTarneeb(num_envs=1, root_seed=5)
    step = environments.reset()
    del step
    result = environments.step(np.array([environments.legal_action_masks()[0].argmax()]))
    events = result.events[0]
    assert events

    rows = [relativized_event_ids(events, seat) for seat in range(NUM_SEATS)]
    actor_columns = {tuple(row[:, 1].tolist()) for row in rows}
    assert len(actor_columns) == NUM_SEATS


# --- chunk heads ------------------------------------------------------------


def test_every_chunk_head_stores_an_empty_segment_and_a_hidden_state(collected) -> None:
    _collector, buffer, _metrics = collected
    heads = 0
    for sequence in buffer.sequences.values():
        for transition in sequence:
            if transition.is_chunk_head:
                heads += 1
                assert transition.previous_event_length == 0
                assert transition.actor_hidden_head is not None
                assert transition.critic_hidden_head is not None
            else:
                assert transition.actor_hidden_head is None
    assert heads > 0


def test_a_non_head_transition_always_carries_a_real_segment(collected) -> None:
    """Between two consecutive decisions of a seat, something always happened."""
    _collector, buffer, _metrics = collected
    for sequence in buffer.sequences.values():
        for transition in sequence:
            if not transition.is_chunk_head:
                assert transition.previous_event_length > 0
                assert transition.first_event_seq >= 0


def test_the_rollout_produces_usable_chunks(collected) -> None:
    collector, buffer, _metrics = collected
    batch = buffer.to_chunks(collector.bootstrap_values(), 1.0, 0.95)
    assert batch.chunk_count > 0
    assert batch.valid_transition_count == buffer.stored_decision_count
    assert batch.is_finite()


# --- resets -----------------------------------------------------------------


def test_a_newly_dealt_hand_resets_memory_and_forces_a_chunk_boundary() -> None:
    """Covers both a scored deal and an all-pass redeal: the engine clears events."""
    collector = build_collector(seed=21, rollout_length=200)
    _buffer, metrics = collector.collect()

    assert metrics.completed_deals > 0
    assert metrics.new_hand_boundaries >= metrics.completed_deals
    assert metrics.new_hand_boundaries == metrics.completed_deals + metrics.passed_out_deals


def test_the_state_is_exactly_zero_immediately_after_a_hand_boundary() -> None:
    """Run until a deal completes and check the environment that just reset."""
    collector = build_collector(seed=31, rollout_length=1)
    for _step in range(400):
        collector.collect()
        for environment_index, environment in enumerate(collector.environments.environments):
            state = environment.state
            if state is not None and len(state.events) == 0:
                assert fresh_hand_state_is_zero(collector.state, environment_index)
                assert collector.state.force_new_chunk[environment_index].all()
                return
    pytest.fail("no hand boundary was reached in 400 steps")


def test_a_redeal_is_a_chunk_boundary_but_not_an_episode_boundary() -> None:
    """The two concepts must stay separate (docs/RECURRENT_CONTRACT.md section 5)."""
    collector = build_collector(seed=11, rollout_length=ROLLOUT)
    buffer, metrics = collector.collect()

    assert metrics.passed_out_deals > 0, "this seed should produce at least one redeal"
    settled = sum(
        1 for sequence in buffer.sequences.values() for transition in sequence if transition.done
    )
    # `done` is set only by settlement, which only a scored deal triggers. A
    # redeal contributes a chunk head but never a `done`.
    assert settled <= metrics.completed_deals * 2
    heads = sum(
        1
        for sequence in buffer.sequences.values()
        for transition in sequence
        if transition.is_chunk_head
    )
    assert heads > settled


def test_a_rollout_boundary_truncates_bptt_without_erasing_memory() -> None:
    collector = build_collector(seed=41, rollout_length=40)
    collector.collect()
    remembered = collector.state.actor_hidden[LEARNER_POLICY_ID].clone()
    live = [
        index
        for index, environment in enumerate(collector.environments.environments)
        if environment.state is not None and len(environment.state.events) > 0
    ]
    assert live, "expected at least one deal still in progress"

    buffer, _metrics = collector.collect()

    # The second window starts every sequence at a head, but a mid-deal
    # environment kept the memory it had.
    for sequence in buffer.sequences.values():
        assert sequence[0].is_chunk_head
    assert not torch.equal(
        collector.state.actor_hidden[LEARNER_POLICY_ID], torch.zeros_like(remembered)
    )


# --- reward -----------------------------------------------------------------


def test_reward_attribution_is_conserved(collected) -> None:
    """Nothing emitted for a learner seat is lost: it is paid or reported unattached."""
    _collector, _buffer, metrics = collected
    assert metrics.total_reward_paid + metrics.total_reward_unattached == pytest.approx(
        metrics.total_learner_reward_emitted, abs=1e-5
    )


def test_the_engine_s_reward_is_zero_sum(collected) -> None:
    _collector, _buffer, metrics = collected
    assert metrics.total_reward_emitted == pytest.approx(0.0, abs=1e-5)


def test_each_settled_seat_is_paid_exactly_once_per_deal(collected) -> None:
    _collector, buffer, _metrics = collected
    for sequence in buffer.sequences.values():
        for first, second in itertools.pairwise(sequence):
            # Two consecutive `done` transitions would mean a deal paid twice.
            assert not (first.done and second.done)


# --- determinism ------------------------------------------------------------


def test_two_identically_seeded_rollouts_agree_in_every_recorded_quantity() -> None:
    """Actions, masks, log-probabilities, values, segments and hidden states."""
    first_collector = build_collector(seed=77)
    second_collector = build_collector(seed=77)
    first_buffer, first_metrics = first_collector.collect()
    second_buffer, second_metrics = second_collector.collect()

    assert first_buffer.sequences.keys() == second_buffer.sequences.keys()
    for key, first_sequence in first_buffer.sequences.items():
        second_sequence = second_buffer.sequences[key]
        assert len(first_sequence) == len(second_sequence)
        for left, right in zip(first_sequence, second_sequence, strict=True):
            assert left.action == right.action
            assert np.array_equal(left.legal_action_mask, right.legal_action_mask)
            assert left.old_log_probability == right.old_log_probability
            assert left.value == right.value
            assert left.previous_event_length == right.previous_event_length
            assert np.array_equal(left.previous_event_ids, right.previous_event_ids)
            assert left.is_chunk_head == right.is_chunk_head
            assert left.reward == right.reward
            assert left.done == right.done

    assert torch.equal(
        first_collector.state.actor_hidden[LEARNER_POLICY_ID],
        second_collector.state.actor_hidden[LEARNER_POLICY_ID],
    )
    assert torch.equal(
        first_collector.state.critic_hidden[LEARNER_POLICY_ID],
        second_collector.state.critic_hidden[LEARNER_POLICY_ID],
    )
    assert np.array_equal(
        first_collector.state.last_seq_consumed, second_collector.state.last_seq_consumed
    )
    assert (
        first_metrics.as_dict()["learner_decision_count"]
        == (second_metrics.as_dict()["learner_decision_count"])
    )


def test_a_different_environment_seed_produces_a_different_rollout() -> None:
    """A negative control: the determinism test above must be saying something."""
    first_buffer, _ = build_collector(seed=77).collect()
    second_buffer, _ = build_collector(seed=78).collect()

    first_actions = [
        transition.action
        for key in sorted(first_buffer.sequences)
        for transition in first_buffer.sequences[key]
    ]
    second_actions = [
        transition.action
        for key in sorted(second_buffer.sequences)
        for transition in second_buffer.sequences[key]
    ]
    assert first_actions != second_actions


# --- bootstrap --------------------------------------------------------------


def test_bootstrap_values_are_produced_for_every_environment_and_seat(collected) -> None:
    collector, _buffer, _metrics = collected
    values = collector.bootstrap_values()
    assert values.shape == (ENVIRONMENTS, NUM_SEATS)
    assert np.isfinite(values).all()


def test_computing_bootstrap_values_does_not_disturb_recurrent_state(collected) -> None:
    """Evaluation must never mutate the rollout's state."""
    collector, _buffer, _metrics = collected
    before_actor = collector.state.actor_hidden[LEARNER_POLICY_ID].clone()
    before_critic = collector.state.critic_hidden[LEARNER_POLICY_ID].clone()
    before_cursor = collector.state.last_seq_consumed.copy()

    collector.bootstrap_values()

    assert torch.equal(collector.state.actor_hidden[LEARNER_POLICY_ID], before_actor)
    assert torch.equal(collector.state.critic_hidden[LEARNER_POLICY_ID], before_critic)
    assert np.array_equal(collector.state.last_seq_consumed, before_cursor)


# --- resume round trip ------------------------------------------------------


def test_the_collector_snapshot_round_trips() -> None:
    collector = build_collector(seed=91, rollout_length=30)
    collector.collect()
    snapshot = collector.snapshot()

    restored = build_collector(seed=91, rollout_length=30)
    restored.restore(snapshot)

    assert torch.equal(
        restored.state.actor_hidden[LEARNER_POLICY_ID],
        collector.state.actor_hidden[LEARNER_POLICY_ID],
    )
    assert np.array_equal(restored.state.last_seq_consumed, collector.state.last_seq_consumed)
    assert np.array_equal(restored.redeal_counts, collector.redeal_counts)


def test_a_fresh_collector_has_no_consumption_history() -> None:
    collector = build_collector()
    assert (collector.state.last_seq_consumed == NO_EVENT_CONSUMED).all()


# --- long runs (opt-in) -----------------------------------------------------


@pytest.mark.slow
def test_a_long_rollout_never_violates_the_segment_bound() -> None:
    """Blueprint T-E9(b), measured on the recurrent collector's own segments.

    Reports the observed maximum. The proved tight bound is 14 and ``K_MAX`` is
    16, so an observation above 14 would mean the derivation is wrong, not that
    the constant needs raising.
    """
    observed = 0
    deals = 0
    for seed in (101, 202, 303):
        collector = build_collector(seed=seed, environments=8, rollout_length=600)
        _buffer, metrics = collector.collect()
        observed = max(observed, metrics.max_segment_length)
        deals += metrics.completed_deals
        collector.assert_consumption_invariant()

    assert deals > 50, f"expected a meaningful number of deals, got {deals}"
    assert observed <= 14, f"observed segment length {observed} exceeds the proved bound"


@pytest.mark.slow
def test_consumption_stays_exact_across_many_deals() -> None:
    """Invariant I15 over a long run: no event dropped, duplicated or reordered."""
    collector = build_collector(seed=404, environments=8, rollout_length=800)
    buffer, metrics = collector.collect()
    collector.assert_consumption_invariant()

    assert metrics.completed_deals > 20
    assert metrics.illegal_action_count == 0
    assert metrics.no_trump_action_count == 0
    for key, start, stop in buffer.chunk_slices():
        covered: list[int] = []
        for transition in buffer.sequences[key][start:stop]:
            if transition.previous_event_length:
                covered.extend(
                    range(
                        transition.first_event_seq,
                        transition.first_event_seq + transition.previous_event_length,
                    )
                )
        assert len(covered) == len(set(covered))
        assert covered == sorted(covered)
