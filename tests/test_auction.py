"""Behavioural tests for the auction and declaration phases.

Expected behaviour comes from docs/RULES.md (C04-C14), not from the engine.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tarneeb.core.actions import (
    NO_TRUMP_ACTION,
    NUM_ACTIONS,
    PASS_ACTION,
    action_for_bid,
    action_for_card,
    action_for_trump_suit,
)
from tarneeb.core.cards import CARDS_PER_HAND, cards_in_hand
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import NUM_SEATS, EventType, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState, first_bidder_seat, next_seat

BID_ACTION_IDS = frozenset(range(52, 59))
TRUMP_ACTION_IDS = frozenset(range(60, 64))


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


def opened_auction(engine: TarneebEngine, dealer_seat: int = 0) -> TarneebState:
    return engine.deal(engine.new_deal(dealer_seat))


def legal_action_ids(engine: TarneebEngine, state: TarneebState) -> set[int]:
    return set(np.flatnonzero(engine.legal_actions(state)).tolist())


# --- seating and turn order -------------------------------------------------


@pytest.mark.parametrize("dealer_seat", range(NUM_SEATS))
def test_first_bidder_is_to_the_dealers_right(engine: TarneebEngine, dealer_seat: int) -> None:
    state = opened_auction(engine, dealer_seat)
    assert state.current_player_seat == (dealer_seat + 1) % NUM_SEATS
    assert state.current_player_seat == first_bidder_seat(dealer_seat)


@pytest.mark.parametrize("dealer_seat", range(NUM_SEATS))
def test_bidding_proceeds_counter_clockwise(engine: TarneebEngine, dealer_seat: int) -> None:
    state = opened_auction(engine, dealer_seat)
    observed_seats = []
    for _ in range(NUM_SEATS - 1):
        observed_seats.append(state.current_player_seat)
        engine.step(state, action_for_bid(7 + len(observed_seats) - 1))
    observed_seats.append(state.current_player_seat)

    expected_seats = [(dealer_seat + 1 + offset) % NUM_SEATS for offset in range(NUM_SEATS)]
    assert observed_seats == expected_seats


def test_next_seat_helper_wraps() -> None:
    assert [next_seat(seat) for seat in range(NUM_SEATS)] == [1, 2, 3, 0]


def test_new_deal_rejects_an_invalid_dealer_seat(engine: TarneebEngine) -> None:
    with pytest.raises(ValueError, match="dealer_seat must be in"):
        engine.new_deal(4)


def test_deal_requires_the_deal_phase(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    with pytest.raises(ValueError, match="requires phase DEAL"):
        engine.deal(state)


# --- legal bids -------------------------------------------------------------


def test_opening_bidder_may_bid_any_value_or_pass(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    assert legal_action_ids(engine, state) == BID_ACTION_IDS | {PASS_ACTION}


@pytest.mark.parametrize("current_high_bid", range(7, 14))
def test_only_strictly_higher_bids_are_legal(engine: TarneebEngine, current_high_bid: int) -> None:
    state = opened_auction(engine)
    engine.step(state, action_for_bid(current_high_bid))

    expected_bid_ids = {action_for_bid(value) for value in range(current_high_bid + 1, 14)}
    assert legal_action_ids(engine, state) == expected_bid_ids | {PASS_ACTION}


def test_equalling_the_high_bid_is_illegal(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    engine.step(state, action_for_bid(9))
    with pytest.raises(ValueError, match="is illegal"):
        engine.step(state, action_for_bid(9))


def test_lower_bids_are_illegal(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    engine.step(state, action_for_bid(9))
    with pytest.raises(ValueError, match="is illegal"):
        engine.step(state, action_for_bid(8))


def test_after_a_bid_of_13_only_pass_remains(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    engine.step(state, action_for_bid(13))
    assert legal_action_ids(engine, state) == {PASS_ACTION}


def test_pass_is_always_legal_during_bidding(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    for bid_value in range(7, 13):
        assert PASS_ACTION in legal_action_ids(engine, state)
        engine.step(state, action_for_bid(bid_value))
    assert PASS_ACTION in legal_action_ids(engine, state)


def test_trump_and_card_actions_are_illegal_during_bidding(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    legal_ids = legal_action_ids(engine, state)
    assert not legal_ids & TRUMP_ACTION_IDS
    assert not legal_ids & set(range(0, 52))
    assert NO_TRUMP_ACTION not in legal_ids


def test_step_rejects_out_of_range_action_ids(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    for action_id in (-1, NUM_ACTIONS, 100):
        with pytest.raises(ValueError, match="action_id must be in"):
            engine.step(state, action_id)


# --- permanent passing ------------------------------------------------------


def test_passing_removes_the_seat_from_the_auction(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    passing_seat = state.current_player_seat
    engine.step(state, PASS_ACTION)
    assert passing_seat not in state.active_bidders
    assert len(state.active_bidders) == NUM_SEATS - 1


def test_a_seat_that_passed_never_acts_again(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    passing_seat = state.current_player_seat
    engine.step(state, PASS_ACTION)

    seats_that_acted = []
    while state.phase is Phase.BID:
        seats_that_acted.append(state.current_player_seat)
        engine.step(state, action_for_bid(7) if state.current_high_bid is None else PASS_ACTION)

    assert passing_seat not in seats_that_acted


def test_turn_skips_seats_that_have_passed(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)  # first bidder is seat 1
    engine.step(state, action_for_bid(7))  # seat 1 bids
    assert state.current_player_seat == 2
    engine.step(state, PASS_ACTION)  # seat 2 passes
    assert state.current_player_seat == 3
    engine.step(state, action_for_bid(8))  # seat 3 bids
    assert state.current_player_seat == 0
    engine.step(state, action_for_bid(9))  # seat 0 bids
    assert state.current_player_seat == 1  # seat 2 is skipped


# --- auction end and declarer ----------------------------------------------


def test_auction_ends_when_one_bidder_remains(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    engine.step(state, action_for_bid(10))  # seat 1
    engine.step(state, PASS_ACTION)  # seat 2
    engine.step(state, PASS_ACTION)  # seat 3
    assert state.phase is Phase.BID
    engine.step(state, PASS_ACTION)  # seat 0

    assert state.phase is Phase.DECLARE
    assert state.declarer_seat == 1
    assert state.current_high_bid == 10
    assert state.current_high_bidder == 1
    assert state.active_bidders == {1}
    assert engine.is_auction_complete(state)


def test_the_declarer_does_not_act_again_in_the_auction(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    engine.step(state, action_for_bid(7))
    engine.step(state, PASS_ACTION)
    engine.step(state, PASS_ACTION)
    engine.step(state, PASS_ACTION)
    assert state.phase is Phase.DECLARE
    assert legal_action_ids(engine, state) == TRUMP_ACTION_IDS


def test_the_last_bidder_wins_the_contract(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    engine.step(state, action_for_bid(7))  # seat 1
    engine.step(state, action_for_bid(8))  # seat 2
    engine.step(state, PASS_ACTION)  # seat 3
    engine.step(state, PASS_ACTION)  # seat 0
    engine.step(state, PASS_ACTION)  # seat 1
    assert state.declarer_seat == 2
    assert state.current_high_bid == 8


# --- all-pass redeal --------------------------------------------------------


@pytest.mark.parametrize("dealer_seat", range(NUM_SEATS))
def test_all_pass_triggers_a_redeal_with_the_same_dealer(
    engine: TarneebEngine, dealer_seat: int
) -> None:
    state = opened_auction(engine, dealer_seat)
    for _ in range(NUM_SEATS):
        engine.step(state, PASS_ACTION)

    assert state.phase is Phase.DEAL
    assert state.dealer_seat == dealer_seat  # the dealer does NOT rotate (C10)
    assert state.active_bidders == set()
    assert state.declarer_seat is None
    assert state.current_high_bid is None
    assert state.redeal_count == 1
    assert engine.is_redeal_required(state)


def test_redeal_restarts_the_auction_with_the_same_dealer(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=2)
    for _ in range(NUM_SEATS):
        engine.step(state, PASS_ACTION)

    engine.deal(state)
    assert state.phase is Phase.BID
    assert state.dealer_seat == 2
    assert state.current_player_seat == first_bidder_seat(2)
    assert state.active_bidders == set(range(NUM_SEATS))
    assert state.events == []  # seq restarts at 0 for the new deal


def test_three_passes_then_a_bid_is_not_a_redeal(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    for _ in range(NUM_SEATS - 1):
        engine.step(state, PASS_ACTION)
    assert state.phase is Phase.BID
    engine.step(state, action_for_bid(7))

    assert state.phase is Phase.DECLARE
    assert state.declarer_seat == 0
    assert state.redeal_count == 0


def test_a_fresh_deal_is_not_reported_as_a_redeal(engine: TarneebEngine) -> None:
    state = engine.new_deal(0)
    assert not engine.is_redeal_required(state)


# --- trump selection --------------------------------------------------------


def declared_state(engine: TarneebEngine, dealer_seat: int = 0) -> TarneebState:
    state = opened_auction(engine, dealer_seat)
    engine.step(state, action_for_bid(9))
    for _ in range(NUM_SEATS - 1):
        engine.step(state, PASS_ACTION)
    assert state.phase is Phase.DECLARE
    return state


def test_declaration_offers_exactly_the_four_suits(engine: TarneebEngine) -> None:
    state = declared_state(engine)
    assert legal_action_ids(engine, state) == {60, 61, 62, 63}


@pytest.mark.parametrize("suit", range(4))
def test_declarer_may_choose_any_suit(engine: TarneebEngine, suit: int) -> None:
    state = declared_state(engine)
    engine.step(state, action_for_trump_suit(suit))
    assert state.selected_trump_suit == suit


def test_no_trump_is_illegal_during_declaration(engine: TarneebEngine) -> None:
    state = declared_state(engine)
    assert NO_TRUMP_ACTION not in legal_action_ids(engine, state)
    with pytest.raises(ValueError, match="is illegal"):
        engine.step(state, NO_TRUMP_ACTION)


def test_bid_and_pass_are_illegal_during_declaration(engine: TarneebEngine) -> None:
    state = declared_state(engine)
    legal_ids = legal_action_ids(engine, state)
    assert not legal_ids & BID_ACTION_IDS
    assert PASS_ACTION not in legal_ids


def test_the_declarer_leads_after_choosing_trump(engine: TarneebEngine) -> None:
    state = declared_state(engine, dealer_seat=0)
    declarer_seat = state.declarer_seat
    engine.step(state, action_for_trump_suit(2))

    assert state.phase is Phase.PLAY
    assert state.current_player_seat == declarer_seat  # C14


def test_the_opening_lead_offers_exactly_the_declarers_hand(engine: TarneebEngine) -> None:
    """C15: the declarer leads first and may lead any card it holds."""
    state = declared_state(engine)
    declarer_seat = state.declarer_seat
    assert declarer_seat is not None
    engine.step(state, action_for_trump_suit(0))

    declarer_cards = set(cards_in_hand(state.hands[declarer_seat]))
    assert len(declarer_cards) == CARDS_PER_HAND
    assert legal_action_ids(engine, state) == {action_for_card(card) for card in declarer_cards}


def test_legal_actions_rejects_phases_with_no_seat_on_turn(engine: TarneebEngine) -> None:
    state = engine.new_deal(0)
    with pytest.raises(ValueError, match="no seat is on turn"):
        engine.legal_actions(state)


# --- the mask itself --------------------------------------------------------


def test_mask_shape_and_dtype(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    legal_action_mask = engine.legal_actions(state)
    assert legal_action_mask.shape == (NUM_ACTIONS,)
    assert legal_action_mask.dtype == np.bool_


def test_no_trump_is_never_legal_anywhere_in_the_auction(engine: TarneebEngine) -> None:
    """Walks a full auction and a declaration, checking id 64 at every decision."""
    for dealer_seat in range(NUM_SEATS):
        state = opened_auction(engine, dealer_seat)
        while state.phase in (Phase.BID, Phase.DECLARE):
            legal_action_mask = engine.legal_actions(state)
            assert not legal_action_mask[NO_TRUMP_ACTION]
            if state.phase is Phase.BID:
                engine.step(state, PASS_ACTION if state.current_high_bid else action_for_bid(7))
            else:
                engine.step(state, action_for_trump_suit(0))


def test_masks_are_independent_arrays(engine: TarneebEngine) -> None:
    state = opened_auction(engine)
    first_mask = engine.legal_actions(state)
    second_mask = engine.legal_actions(state)
    first_mask[PASS_ACTION] = False
    assert second_mask[PASS_ACTION]


# --- events -----------------------------------------------------------------


def test_bidding_emits_bid_and_pass_events_in_order(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    _, bid_events, deal_result = engine.step(state, action_for_bid(11))
    assert deal_result is None
    assert len(bid_events) == 1
    assert bid_events[0].event_type is EventType.BID
    assert bid_events[0].actor_seat == 1
    assert bid_events[0].bid_value == 11
    assert bid_events[0].phase is Phase.BID

    _, pass_events, _ = engine.step(state, PASS_ACTION)
    assert len(pass_events) == 1
    assert pass_events[0].event_type is EventType.PASS
    assert pass_events[0].actor_seat == 2
    assert pass_events[0].bid_value is None


def test_declaration_emits_a_trump_selected_event(engine: TarneebEngine) -> None:
    state = declared_state(engine)
    declarer_seat = state.declarer_seat
    _, trump_events, _ = engine.step(state, action_for_trump_suit(3))

    assert len(trump_events) == 1
    assert trump_events[0].event_type is EventType.TRUMP_SELECTED
    assert trump_events[0].actor_seat == declarer_seat
    assert trump_events[0].trump == 3
    assert trump_events[0].phase is Phase.DECLARE


def test_event_sequence_numbers_are_zero_based_and_contiguous(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    engine.step(state, action_for_bid(7))
    engine.step(state, action_for_bid(8))
    engine.step(state, PASS_ACTION)
    engine.step(state, PASS_ACTION)
    engine.step(state, PASS_ACTION)
    engine.step(state, action_for_trump_suit(1))

    assert [event.seq for event in state.events] == list(range(len(state.events)))
    assert state.next_event_sequence_number == len(state.events)


def test_the_event_stream_matches_the_executed_actions(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=1)
    executed_actions = [
        action_for_bid(7),
        PASS_ACTION,
        action_for_bid(12),
        PASS_ACTION,
        PASS_ACTION,
        action_for_trump_suit(2),
    ]
    for action_id in executed_actions:
        engine.step(state, action_id)

    observed_event_types = [event.event_type for event in state.events]
    assert observed_event_types == [
        EventType.BID,
        EventType.PASS,
        EventType.BID,
        EventType.PASS,
        EventType.PASS,
        EventType.TRUMP_SELECTED,
    ]


def test_events_never_carry_a_card_or_hidden_information(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    engine.step(state, action_for_bid(13))
    engine.step(state, PASS_ACTION)
    engine.step(state, PASS_ACTION)
    engine.step(state, PASS_ACTION)
    engine.step(state, action_for_trump_suit(0))

    for event in state.events:
        assert event.card is None  # no card is public before trick play
        assert event.deal_result is None
        assert event.trick_winner_seat is None
        assert event.trick_no == 0


def test_all_pass_events_are_kept_until_the_redeal(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    for _ in range(NUM_SEATS):
        engine.step(state, PASS_ACTION)

    assert [event.event_type for event in state.events] == [EventType.PASS] * NUM_SEATS
    assert [event.seq for event in state.events] == [0, 1, 2, 3]
    engine.deal(state)
    assert state.events == []


def test_returned_events_are_the_same_objects_appended_to_the_state(
    engine: TarneebEngine,
) -> None:
    state = opened_auction(engine)
    _, emitted_events, _ = engine.step(state, action_for_bid(7))
    assert state.events[-1] is emitted_events[0]


# --- state hygiene ----------------------------------------------------------


def test_clone_is_independent(engine: TarneebEngine) -> None:
    state = opened_auction(engine, dealer_seat=0)
    engine.step(state, action_for_bid(7))
    cloned_state = state.clone()

    engine.step(cloned_state, PASS_ACTION)
    assert len(cloned_state.events) == 2
    assert len(state.events) == 1
    assert cloned_state.active_bidders != state.active_bidders


def test_hidden_hands_exist_in_state_but_never_reach_an_event(engine: TarneebEngine) -> None:
    """The state holds private hands; no auction event may expose them."""
    field_names = {field.name for field in dataclasses.fields(TarneebState)}
    assert {"hands", "original_hands"} <= field_names

    state = declared_state(engine)
    engine.step(state, action_for_trump_suit(1))
    for event in state.events:
        assert event.card is None  # no card is public before the opening lead
    assert all(hand != 0 for hand in state.hands)


def test_engine_rejects_unimplemented_auction_variants() -> None:
    with pytest.raises(NotImplementedError, match="egyptian_bidding"):
        TarneebEngine(RulesConfig(egyptian_bidding=True))
    with pytest.raises(NotImplementedError, match="auction variants"):
        TarneebEngine(RulesConfig(single_bid_round=True))
