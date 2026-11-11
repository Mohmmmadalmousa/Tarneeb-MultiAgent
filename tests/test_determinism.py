"""Determinism: the same seed must reproduce the same deal exactly (invariant I8).

The engine must never touch global randomness, and dealing must be a pure
function of ``(match_seed, deal_index)`` so duplicate-deal evaluation can
replay the same cards for a rotated seating.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tarneeb.core.actions import PASS_ACTION, action_for_bid, action_for_trump_suit
from tarneeb.core.cards import derive_deal_seed
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import NUM_SEATS, PublicEvent
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


def play_scripted_deal(
    engine: TarneebEngine,
    match_seed: int,
    dealer_seat: int = 0,
    deal_index: int = 0,
    policy_seed: int = 5,
) -> tuple[TarneebState, list[int], list[PublicEvent]]:
    """Play one full deal with a seeded card-choice policy.

    Returns the final state, the action sequence and the event stream, so two
    runs can be compared field by field.
    """
    state = engine.deal(engine.new_deal(dealer_seat, deal_index=deal_index, match_seed=match_seed))
    policy_rng = np.random.default_rng(policy_seed)
    applied_actions: list[int] = []

    for action_id in (
        action_for_bid(9),
        PASS_ACTION,
        PASS_ACTION,
        PASS_ACTION,
        action_for_trump_suit(2),
    ):
        engine.step(state, action_id)
        applied_actions.append(action_id)

    deal_result = None
    while deal_result is None:
        legal_action_ids = np.flatnonzero(engine.legal_actions(state))
        chosen_action = int(policy_rng.choice(legal_action_ids))
        _, _events, deal_result = engine.step(state, chosen_action)
        applied_actions.append(chosen_action)

    return state, applied_actions, list(state.events)


# --- same seed, identical everything ----------------------------------------


@pytest.mark.parametrize("match_seed", [0, 1, 42, 2024])
def test_same_seed_gives_identical_deals_actions_events_and_scores(
    engine: TarneebEngine, match_seed: int
) -> None:
    first_state, first_actions, first_events = play_scripted_deal(engine, match_seed)
    second_state, second_actions, second_events = play_scripted_deal(engine, match_seed)

    assert first_state.original_hands == second_state.original_hands
    assert first_actions == second_actions
    assert first_events == second_events  # PublicEvent is a frozen dataclass
    assert first_state.tricks_won == second_state.tricks_won
    assert first_state.deal_result == second_state.deal_result
    assert first_state.match_score == second_state.match_score
    assert first_state.completed_tricks == second_state.completed_tricks
    assert first_state.trick_winners == second_state.trick_winners


def test_two_engine_instances_agree(engine: TarneebEngine) -> None:
    other_engine = TarneebEngine(RulesConfig())
    first = play_scripted_deal(engine, 7)
    second = play_scripted_deal(other_engine, 7)
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_different_seeds_give_different_deals(engine: TarneebEngine) -> None:
    first_state, _, _ = play_scripted_deal(engine, 1)
    second_state, _, _ = play_scripted_deal(engine, 2)
    assert first_state.original_hands != second_state.original_hands


def test_deal_index_changes_the_cards(engine: TarneebEngine) -> None:
    first_state, _, _ = play_scripted_deal(engine, 42, deal_index=0)
    second_state, _, _ = play_scripted_deal(engine, 42, deal_index=1)
    assert first_state.original_hands != second_state.original_hands


# --- independence from global randomness ------------------------------------


def test_global_random_state_does_not_affect_the_engine(engine: TarneebEngine) -> None:
    """The engine must use its own generator, never numpy's or Python's global."""
    np.random.seed(1)
    random.seed(1)
    first_state, first_actions, first_events = play_scripted_deal(engine, 99)

    np.random.seed(999)
    random.seed(999)
    for _ in range(50):  # churn both global streams
        np.random.random()
        random.random()
    second_state, second_actions, second_events = play_scripted_deal(engine, 99)

    assert first_state.original_hands == second_state.original_hands
    assert first_actions == second_actions
    assert first_events == second_events


def test_interleaving_two_deals_does_not_couple_them(engine: TarneebEngine) -> None:
    """A deal's cards depend only on its own seed, not on how many deals ran."""
    isolated_state, _, _ = play_scripted_deal(engine, 314, deal_index=3)

    for other_seed in (11, 12, 13):
        play_scripted_deal(engine, other_seed)
    after_other_deals, _, _ = play_scripted_deal(engine, 314, deal_index=3)

    assert isolated_state.original_hands == after_other_deals.original_hands


# --- duplicate deals (the evaluation protocol depends on this) --------------


def test_the_same_cards_can_be_dealt_to_a_rotated_seating(engine: TarneebEngine) -> None:
    """Duplicate-deal evaluation needs identical cards under a different dealer."""
    first_state = engine.deal(engine.new_deal(0, match_seed=77))
    second_state = engine.deal(engine.new_deal(2, match_seed=77))

    assert first_state.original_hands == second_state.original_hands
    assert first_state.dealer_seat != second_state.dealer_seat
    assert first_state.current_player_seat != second_state.current_player_seat


def test_dealing_is_a_pure_function_of_seed_and_index(engine: TarneebEngine) -> None:
    for match_seed in (0, 5, 500):
        for deal_index in range(3):
            expected_seed = derive_deal_seed(match_seed, deal_index)
            first = engine.deal(engine.new_deal(0, deal_index=deal_index, match_seed=match_seed))
            second = engine.deal(engine.new_deal(0, deal_index=deal_index, match_seed=match_seed))
            assert first.original_hands == second.original_hands
            assert expected_seed == derive_deal_seed(match_seed, deal_index)


def test_a_match_produces_a_distinct_deal_each_hand(engine: TarneebEngine) -> None:
    state = engine.new_match(match_seed=2024)
    dealt_hands_per_deal = []
    for _ in range(NUM_SEATS):
        engine.deal(state)
        dealt_hands_per_deal.append(tuple(state.original_hands))
        state.phase = state.phase.DEAL  # rewind without playing; only dealing is under test
        state.deal_index += 1
    assert len(set(dealt_hands_per_deal)) == NUM_SEATS
