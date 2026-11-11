"""Trick-play behaviour: legality, trick resolution, events and scoring.

Covers invariants I2 (cards held plus cards played is always 13), I3 (following
suit is mandatory), I14 (event sequence numbers and the 77-event bound), and the
event ordering rule that a trick-completing CARD_PLAYED is emitted before its
TRICK_COMPLETED, with DEAL_COMPLETED last.

Expected behaviour comes from docs/RULES.md C14-C21, not from the engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.core.actions import PASS_ACTION, action_for_bid, action_for_card, action_for_trump_suit
from tarneeb.core.cards import (
    CARDS_PER_HAND,
    NUM_CARDS,
    NUM_SUITS,
    card_id,
    cards_in_hand,
    hand_from_cards,
    hand_size,
    holds_suit,
    suit_of,
)
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import NUM_SEATS, EventType, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.scoring import score_deal
from tarneeb.core.state import TRICKS_PER_DEAL, TarneebState, team_of_seat

MAX_EVENTS_PER_DEAL = 77  # 10 auction + 1 trump + 52 cards + 13 tricks + 1 deal


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


def state_in_play(
    engine: TarneebEngine,
    bid_value: int = 9,
    trump_suit: int = 2,
    dealer_seat: int = 0,
    match_seed: int = 42,
) -> TarneebState:
    """Return a state at the opening lead, with a known contract."""
    state = engine.deal(engine.new_deal(dealer_seat, match_seed=match_seed))
    engine.step(state, action_for_bid(bid_value))
    for _ in range(NUM_SEATS - 1):
        engine.step(state, PASS_ACTION)
    engine.step(state, action_for_trump_suit(trump_suit))
    assert state.phase is Phase.PLAY
    return state


def legal_cards(engine: TarneebEngine, state: TarneebState) -> set[int]:
    return set(engine.legal_cards(state))


def play_out_deal(engine: TarneebEngine, state: TarneebState, rng: np.random.Generator) -> object:
    """Play random legal cards until the deal is scored; return the result."""
    deal_result = None
    while deal_result is None:
        legal_action_ids = np.flatnonzero(engine.legal_actions(state))
        _, _events, deal_result = engine.step(state, int(rng.choice(legal_action_ids)))
    return deal_result


# --- the opening lead -------------------------------------------------------


def test_declarer_leads_the_first_trick(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    assert state.current_player_seat == state.declarer_seat  # C14
    assert state.leader_seat == state.declarer_seat
    assert state.trick_number == 0
    assert state.current_trick == []


def test_the_leader_may_play_any_card(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    declarer_seat = state.declarer_seat
    assert declarer_seat is not None
    assert legal_cards(engine, state) == set(cards_in_hand(state.hands[declarer_seat]))  # C15


# --- following suit (I3) ----------------------------------------------------


def test_a_seat_holding_the_led_suit_must_follow_it(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    led_card = engine.legal_cards(state)[0]
    engine.step(state, action_for_card(led_card))
    led_suit = suit_of(led_card)

    follower_seat = state.current_player_seat
    assert follower_seat is not None
    follower_hand = state.hands[follower_seat]
    if not holds_suit(follower_hand, led_suit):
        pytest.skip("this seeded seat is void in the led suit")

    playable = legal_cards(engine, state)
    assert playable  # C16
    assert all(suit_of(card) == led_suit for card in playable)
    assert playable == {card for card in cards_in_hand(follower_hand) if suit_of(card) == led_suit}


def test_a_void_seat_may_play_anything(engine: TarneebEngine) -> None:
    """Construct a void by hand rather than hunting for one in a random deal."""
    state = state_in_play(engine)
    leader_seat = state.current_player_seat
    assert leader_seat is not None

    led_card = card_id(0, 12)  # ace of clubs
    state.hands[leader_seat] = hand_from_cards([led_card])
    engine.step(state, action_for_card(led_card))

    void_seat = state.current_player_seat
    assert void_seat is not None
    hand_without_clubs = hand_from_cards([card_id(1, 3), card_id(2, 7), card_id(3, 11)])
    state.hands[void_seat] = hand_without_clubs

    assert legal_cards(engine, state) == set(cards_in_hand(hand_without_clubs))  # C17


def test_playing_off_suit_sets_the_public_void_flag(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    leader_seat = state.current_player_seat
    assert leader_seat is not None

    led_card = card_id(0, 12)
    state.hands[leader_seat] = hand_from_cards([led_card])
    engine.step(state, action_for_card(led_card))

    void_seat = state.current_player_seat
    assert void_seat is not None
    discard = card_id(2, 7)
    state.hands[void_seat] = hand_from_cards([discard])
    assert not state.void_flags[void_seat][0]

    engine.step(state, action_for_card(discard))
    assert state.void_flags[void_seat][0]  # publicly void in clubs
    assert not state.void_flags[void_seat][2]


def test_following_suit_never_sets_a_void_flag(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    rng = np.random.default_rng(0)
    play_out_deal(engine, state, rng)

    for seat in range(NUM_SEATS):
        for suit in range(NUM_SUITS):
            if state.void_flags[seat][suit]:
                # a flag may only be set if the seat played off-suit at some point
                assert any(
                    played_seat == seat and suit_of(card) != suit
                    for trick in state.completed_tricks
                    for played_seat, card in trick
                )


def test_an_illegal_card_is_rejected(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    leader_seat = state.current_player_seat
    assert leader_seat is not None
    not_in_hand = next(
        card for card in range(NUM_CARDS) if card not in cards_in_hand(state.hands[leader_seat])
    )
    with pytest.raises(ValueError, match="is illegal"):
        engine.step(state, action_for_card(not_in_hand))


# --- trick resolution -------------------------------------------------------


def test_a_completed_trick_is_recorded_and_the_winner_leads(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    for _ in range(NUM_SEATS):
        engine.step(state, action_for_card(engine.legal_cards(state)[0]))

    assert state.trick_number == 1
    assert state.current_trick == []
    assert len(state.completed_tricks) == 1
    assert len(state.completed_tricks[0]) == NUM_SEATS

    winning_seat = state.trick_winners[0]
    assert state.current_player_seat == winning_seat  # C19
    assert state.leader_seat == winning_seat
    assert sum(state.tricks_won) == 1
    assert state.tricks_won[team_of_seat(winning_seat)] == 1


def test_the_recorded_winner_matches_the_trick_winner_rule(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    rng = np.random.default_rng(4)
    play_out_deal(engine, state, rng)

    trump_suit = state.selected_trump_suit
    assert trump_suit is not None
    for trick, recorded_winner in zip(state.completed_tricks, state.trick_winners, strict=True):
        assert engine.trick_winner(trick, trump_suit) == recorded_winner


def test_thirteen_tricks_are_played_and_all_hands_empty(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    play_out_deal(engine, state, np.random.default_rng(1))

    assert len(state.completed_tricks) == TRICKS_PER_DEAL
    assert sum(state.tricks_won) == TRICKS_PER_DEAL
    assert all(hand == 0 for hand in state.hands)
    assert state.phase is Phase.SCORE


def test_cards_held_plus_cards_played_is_always_thirteen(engine: TarneebEngine) -> None:
    """Invariant I2, checked after every single card."""
    state = state_in_play(engine)
    rng = np.random.default_rng(2)
    cards_played_by_seat = [0] * NUM_SEATS

    while state.phase is Phase.PLAY:
        acting_seat = state.current_player_seat
        assert acting_seat is not None
        legal_action_ids = np.flatnonzero(engine.legal_actions(state))
        engine.step(state, int(rng.choice(legal_action_ids)))
        cards_played_by_seat[acting_seat] += 1
        for seat in range(NUM_SEATS):
            assert hand_size(state.hands[seat]) + cards_played_by_seat[seat] == CARDS_PER_HAND


def test_every_dealt_card_is_played_exactly_once(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    dealt_cards = [set(cards_in_hand(hand)) for hand in state.original_hands]
    play_out_deal(engine, state, np.random.default_rng(6))

    played_cards = [card for trick in state.completed_tricks for _seat, card in trick]
    assert len(played_cards) == NUM_CARDS
    assert len(set(played_cards)) == NUM_CARDS

    for seat in range(NUM_SEATS):
        seat_cards = {
            card
            for trick in state.completed_tricks
            for played_seat, card in trick
            if played_seat == seat
        }
        assert seat_cards == dealt_cards[seat]


# --- events (I14 and ordering) ----------------------------------------------


def test_card_played_precedes_trick_completed(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    for _ in range(NUM_SEATS - 1):
        _, events, _ = engine.step(state, action_for_card(engine.legal_cards(state)[0]))
        assert [event.event_type for event in events] == [EventType.CARD_PLAYED]

    _, closing_events, _ = engine.step(state, action_for_card(engine.legal_cards(state)[0]))
    assert [event.event_type for event in closing_events] == [
        EventType.CARD_PLAYED,
        EventType.TRICK_COMPLETED,
    ]
    assert closing_events[0].seq + 1 == closing_events[1].seq


def test_the_final_action_emits_card_trick_then_deal_completed(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    rng = np.random.default_rng(3)
    final_events: list = []
    while state.phase is Phase.PLAY:
        legal_action_ids = np.flatnonzero(engine.legal_actions(state))
        _, final_events, _ = engine.step(state, int(rng.choice(legal_action_ids)))

    assert [event.event_type for event in final_events] == [
        EventType.CARD_PLAYED,
        EventType.TRICK_COMPLETED,
        EventType.DEAL_COMPLETED,
    ]
    assert [event.seq for event in final_events] == [
        final_events[0].seq,
        final_events[0].seq + 1,
        final_events[0].seq + 2,
    ]


def test_event_sequence_numbers_stay_contiguous_through_a_whole_deal(
    engine: TarneebEngine,
) -> None:
    state = state_in_play(engine)
    play_out_deal(engine, state, np.random.default_rng(8))
    assert [event.seq for event in state.events] == list(range(len(state.events)))


def test_event_counts_and_the_77_event_bound(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    play_out_deal(engine, state, np.random.default_rng(9))

    counts = {
        event_type: sum(1 for event in state.events if event.event_type is event_type)
        for event_type in EventType
    }
    assert counts[EventType.CARD_PLAYED] == NUM_CARDS
    assert counts[EventType.TRICK_COMPLETED] == TRICKS_PER_DEAL
    assert counts[EventType.DEAL_COMPLETED] == 1
    assert counts[EventType.TRUMP_SELECTED] == 1
    assert counts[EventType.PAD] == 0
    assert len(state.events) <= MAX_EVENTS_PER_DEAL  # I14


def test_card_events_carry_the_played_card_and_nothing_hidden(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    play_out_deal(engine, state, np.random.default_rng(10))

    for event in state.events:
        if event.event_type is EventType.CARD_PLAYED:
            assert event.card is not None
            assert event.actor_seat is not None
            assert 0 <= event.trick_no < TRICKS_PER_DEAL
        else:
            assert event.card is None  # EV-1: only a played card is ever public


def test_trick_completed_events_carry_the_winner_and_no_actor(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    play_out_deal(engine, state, np.random.default_rng(11))

    trick_events = [e for e in state.events if e.event_type is EventType.TRICK_COMPLETED]
    assert [event.trick_no for event in trick_events] == list(range(TRICKS_PER_DEAL))
    assert [event.trick_winner_seat for event in trick_events] == state.trick_winners
    assert all(event.actor_seat is None for event in trick_events)


def test_a_card_event_never_reveals_a_card_still_in_a_hand(engine: TarneebEngine) -> None:
    """Invariant I10, resolved as: measured immediately after event k."""
    state = state_in_play(engine)
    rng = np.random.default_rng(12)
    revealed_cards: set[int] = set()

    while state.phase is Phase.PLAY:
        legal_action_ids = np.flatnonzero(engine.legal_actions(state))
        _, events, _ = engine.step(state, int(rng.choice(legal_action_ids)))
        for event in events:
            if event.card is not None:
                revealed_cards.add(event.card)
        still_held = {card for hand in state.hands for card in cards_in_hand(hand)}
        assert not (revealed_cards & still_held)


# --- scoring integration ----------------------------------------------------


def test_the_deal_result_matches_the_frozen_scoring_function(engine: TarneebEngine) -> None:
    state = state_in_play(engine, bid_value=9)
    deal_result = play_out_deal(engine, state, np.random.default_rng(13))

    declaring_team = state.declaring_team
    assert declaring_team is not None
    expected = score_deal(9, state.tricks_won[declaring_team], engine.rules)
    assert deal_result.score_delta_bidding_team == expected[0]
    assert deal_result.score_delta_defending_team == expected[1]
    assert deal_result.bid == 9
    assert deal_result.tricks_bidding_team == state.tricks_won[declaring_team]


def test_the_deal_result_is_returned_only_on_the_final_action(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    rng = np.random.default_rng(14)
    results_seen = []
    while state.phase is Phase.PLAY:
        legal_action_ids = np.flatnonzero(engine.legal_actions(state))
        _, _events, deal_result = engine.step(state, int(rng.choice(legal_action_ids)))
        results_seen.append(deal_result)

    assert all(result is None for result in results_seen[:-1])
    assert results_seen[-1] is not None


def test_match_score_accumulates_the_deal_result(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    deal_result = play_out_deal(engine, state, np.random.default_rng(15))

    declaring_team = state.declaring_team
    assert declaring_team is not None
    defending_team = 1 - declaring_team
    assert state.match_score[declaring_team] == deal_result.score_delta_bidding_team
    assert state.match_score[defending_team] == deal_result.score_delta_defending_team


def test_deal_completed_event_carries_the_result(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    deal_result = play_out_deal(engine, state, np.random.default_rng(16))

    final_event = state.events[-1]
    assert final_event.event_type is EventType.DEAL_COMPLETED
    assert final_event.deal_result == deal_result
    assert final_event.trick_no == TRICKS_PER_DEAL
    assert final_event.phase is Phase.SCORE


def test_scoring_a_deal_without_a_contract_is_rejected(engine: TarneebEngine) -> None:
    state = engine.deal(engine.new_deal(0))
    with pytest.raises(ValueError, match="no contract"):
        engine.score_deal(state)


# --- match lifecycle --------------------------------------------------------


def test_next_deal_rotates_the_dealer(engine: TarneebEngine) -> None:
    state = state_in_play(engine, dealer_seat=1)
    play_out_deal(engine, state, np.random.default_rng(17))

    assert engine.is_deal_complete(state)
    engine.next_deal(state)
    assert state.dealer_seat == 2  # C05: the deal passes to the right
    assert state.phase is Phase.DEAL
    assert state.deal_index == 1
    assert state.redeal_count == 0


def test_next_deal_requires_a_scored_deal(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    with pytest.raises(ValueError, match="requires phase SCORE"):
        engine.next_deal(state)


def test_a_second_deal_deals_different_cards(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    first_hands = list(state.original_hands)
    play_out_deal(engine, state, np.random.default_rng(18))
    engine.next_deal(state)
    engine.deal(state)
    assert state.original_hands != first_hands
    assert all(hand_size(hand) == CARDS_PER_HAND for hand in state.hands)


def test_match_completion_uses_the_frozen_target(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    assert not engine.is_match_complete(state)
    state.match_score = [engine.rules.target_score, 0]
    assert engine.is_match_complete(state)  # C25
    state.match_score = [engine.rules.target_score - 1, 0]
    assert not engine.is_match_complete(state)


def test_legal_actions_rejects_the_score_phase(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    play_out_deal(engine, state, np.random.default_rng(19))
    with pytest.raises(ValueError, match="no seat is on turn"):
        engine.legal_actions(state)


# --- clone independence during play -----------------------------------------


def test_clone_isolates_hands_and_tricks(engine: TarneebEngine) -> None:
    state = state_in_play(engine)
    cloned_state = state.clone()

    engine.step(cloned_state, action_for_card(engine.legal_cards(cloned_state)[0]))
    assert cloned_state.hands != state.hands
    assert state.current_trick == []
    assert len(cloned_state.current_trick) == 1

    cloned_state.void_flags[0][0] = True
    assert not state.void_flags[0][0]
    cloned_state.tricks_won[0] = 5
    assert state.tricks_won[0] == 0
