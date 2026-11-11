"""Random-legal stress testing (invariant I9).

Plays many complete deals by sampling uniformly from the legal-action mask and
asserts that nothing raises, that no illegal action is ever offered, and that
the structural invariants hold at every single step.

The default run is small enough for the normal suite; a larger run is marked
``slow`` and is opt-in with ``pytest -m slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.core.actions import (
    NO_TRUMP_ACTION,
    NUM_ACTIONS,
    PASS_ACTION,
    bid_for_action,
    card_for_action,
    is_bid_action,
    is_card_action,
    is_trump_action,
)
from tarneeb.core.cards import (
    CARDS_PER_HAND,
    FULL_DECK_MASK,
    NUM_CARDS,
    hand_contains,
    hand_size,
    holds_suit,
    suit_of,
)
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import NUM_SEATS, EventType, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TRICKS_PER_DEAL, TarneebState

DEFAULT_DEAL_COUNT = 200
SLOW_DEAL_COUNT = 5000
MAX_EVENTS_PER_DEAL = 77


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


def check_mask_invariants(engine: TarneebEngine, state: TarneebState) -> np.ndarray:
    """Assert the mask itself is well formed, and return it."""
    legal_action_mask = engine.legal_actions(state)

    assert legal_action_mask.shape == (NUM_ACTIONS,)
    assert legal_action_mask.dtype == np.bool_
    assert legal_action_mask.any(), "some action must always be legal"
    assert not legal_action_mask[NO_TRUMP_ACTION], "action 64 is permanently masked"

    acting_seat = state.current_player_seat
    assert acting_seat is not None
    legal_action_ids = np.flatnonzero(legal_action_mask).tolist()

    if state.phase is Phase.BID:
        assert all(is_bid_action(action) or action == PASS_ACTION for action in legal_action_ids)
        for action in legal_action_ids:
            if is_bid_action(action):
                bid_value = bid_for_action(action)
                assert engine.rules.min_bid <= bid_value <= engine.rules.max_bid
                if state.current_high_bid is not None:
                    assert bid_value > state.current_high_bid  # C08
        assert acting_seat in state.active_bidders  # C09

    elif state.phase is Phase.DECLARE:
        assert all(is_trump_action(action) for action in legal_action_ids)
        assert acting_seat == state.declarer_seat

    else:
        assert state.phase is Phase.PLAY
        assert all(is_card_action(action) for action in legal_action_ids)
        for action in legal_action_ids:
            assert hand_contains(state.hands[acting_seat], card_for_action(action))  # I3
        if state.current_trick:
            led_suit = suit_of(state.current_trick[0][1])
            if holds_suit(state.hands[acting_seat], led_suit):
                assert all(
                    suit_of(card_for_action(action)) == led_suit for action in legal_action_ids
                )

    return legal_action_mask


def check_state_invariants(state: TarneebState, cards_played_by_seat: list[int]) -> None:
    """Assert the deal-level invariants that must hold after every action."""
    # I2: what a seat still holds plus what it has played is always 13.
    for seat in range(NUM_SEATS):
        assert hand_size(state.hands[seat]) + cards_played_by_seat[seat] == CARDS_PER_HAND

    # I1: no card is ever duplicated or lost.
    union_of_hands = 0
    for hand in state.hands:
        assert union_of_hands & hand == 0
        union_of_hands |= hand
    played_cards = [card for trick in state.completed_tricks for _seat, card in trick]
    played_cards += [card for _seat, card in state.current_trick]
    assert len(set(played_cards)) == len(played_cards)
    assert union_of_hands | sum(1 << card for card in played_cards) == FULL_DECK_MASK

    # I14: sequence numbers are zero-based, contiguous and bounded.
    assert [event.seq for event in state.events] == list(range(len(state.events)))
    assert len(state.events) <= MAX_EVENTS_PER_DEAL

    assert 0 <= state.trick_number <= TRICKS_PER_DEAL
    assert len(state.current_trick) < NUM_SEATS
    assert sum(state.tricks_won) == len(state.completed_tricks)


def play_random_deals(engine: TarneebEngine, deal_count: int, base_seed: int = 0) -> dict:
    """Play ``deal_count`` random-legal deals, checking invariants throughout."""
    statistics = {
        "deals_played": 0,
        "redeals": 0,
        "contracts": 0,
        "events_max": 0,
        "bids": [],
    }

    for deal_number in range(deal_count):
        rng = np.random.default_rng(base_seed + deal_number)
        state = engine.deal(
            engine.new_deal(deal_number % NUM_SEATS, match_seed=base_seed + deal_number)
        )
        cards_played_by_seat = [0] * NUM_SEATS

        while state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            legal_action_mask = check_mask_invariants(engine, state)
            acting_seat = state.current_player_seat
            assert acting_seat is not None

            chosen_action = int(rng.choice(np.flatnonzero(legal_action_mask)))
            assert legal_action_mask[chosen_action]

            was_playing = state.phase is Phase.PLAY
            engine.step(state, chosen_action)
            if was_playing and is_card_action(chosen_action):
                cards_played_by_seat[acting_seat] += 1

            check_state_invariants(state, cards_played_by_seat)

        statistics["events_max"] = max(statistics["events_max"], len(state.events))

        if state.phase is Phase.DEAL:  # C10 all-pass redeal
            statistics["redeals"] += 1
            assert state.active_bidders == set()
            assert state.deal_result is None
            continue

        assert state.phase is Phase.SCORE
        statistics["deals_played"] += 1
        statistics["contracts"] += 1
        assert state.current_high_bid is not None
        statistics["bids"].append(state.current_high_bid)

        assert sum(state.tricks_won) == TRICKS_PER_DEAL
        assert all(hand == 0 for hand in state.hands)
        assert state.deal_result is not None
        assert sum(1 for e in state.events if e.event_type is EventType.CARD_PLAYED) == NUM_CARDS

    return statistics


# --- the tests --------------------------------------------------------------


@pytest.fixture(scope="module")
def stress_statistics() -> dict:
    """One shared random-legal run; every assertion below reads from it.

    Kept as a single run because the per-step invariant checks are the expensive
    part, and repeating them in several tests buys no extra coverage.
    """
    return play_random_deals(TarneebEngine(RulesConfig()), DEFAULT_DEAL_COUNT, base_seed=1000)


def test_random_legal_deals_raise_nothing(stress_statistics: dict) -> None:
    """Reaching this point means no step raised and every invariant held."""
    assert stress_statistics["deals_played"] + stress_statistics["redeals"] == DEFAULT_DEAL_COUNT
    assert stress_statistics["deals_played"] > 0


def test_no_deal_exceeds_the_event_bound(stress_statistics: dict) -> None:
    assert stress_statistics["events_max"] <= MAX_EVENTS_PER_DEAL


def test_every_contract_is_a_legal_bid_value(stress_statistics: dict) -> None:
    rules = RulesConfig()
    observed_bids = set(stress_statistics["bids"])
    assert observed_bids
    assert observed_bids <= set(range(rules.min_bid, rules.max_bid + 1))


def test_all_pass_redeals_are_counted_not_scored(stress_statistics: dict) -> None:
    assert stress_statistics["redeals"] >= 0
    assert stress_statistics["contracts"] == stress_statistics["deals_played"]


@pytest.mark.slow
def test_large_random_stress_run(engine: TarneebEngine) -> None:
    statistics = play_random_deals(engine, SLOW_DEAL_COUNT, base_seed=10_000)
    assert statistics["deals_played"] + statistics["redeals"] == SLOW_DEAL_COUNT
    assert statistics["events_max"] <= MAX_EVENTS_PER_DEAL
