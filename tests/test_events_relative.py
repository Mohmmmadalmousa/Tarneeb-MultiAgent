"""Relative-seat conversion tests for event_v1.

Convention under test: 0 = self, 1 = right-hand opponent, 2 = partner,
3 = left-hand opponent, with play proceeding counter-clockwise so the seat
after ``p`` is ``(p + 1) % 4``. Partnerships are (0, 2) and (1, 3).
"""

from __future__ import annotations

import itertools

import pytest

from tarneeb.core.events import (
    NUM_SEATS,
    DealResult,
    EventType,
    Phase,
    PublicEvent,
    relativize,
)

SEATS = range(NUM_SEATS)
TEAM_OF = {0: 0, 1: 1, 2: 0, 3: 1}


def _event(**overrides: object) -> PublicEvent:
    base: dict[str, object] = {
        "seq": 0,
        "phase": Phase.BID,
        "actor_seat": None,
        "event_type": EventType.PAD,
        "bid_value": None,
        "trump": None,
        "card": None,
        "trick_no": 0,
        "trick_winner_seat": None,
        "deal_result": None,
    }
    base.update(overrides)
    return PublicEvent(**base)  # type: ignore[arg-type]


# --- actor seat -------------------------------------------------------------


@pytest.mark.parametrize(("actor", "observer"), list(itertools.product(SEATS, SEATS)))
def test_actor_rel_for_every_combination(actor: int, observer: int) -> None:
    event = _event(actor_seat=actor, event_type=EventType.PASS)
    assert relativize(event, observer).actor_rel == (actor - observer) % NUM_SEATS


@pytest.mark.parametrize("observer", SEATS)
def test_a_seat_sees_its_own_action_as_zero(observer: int) -> None:
    event = _event(actor_seat=observer, event_type=EventType.BID, bid_value=9)
    assert relativize(event, observer).actor_rel == 0


@pytest.mark.parametrize("observer", SEATS)
def test_right_hand_opponent_is_one(observer: int) -> None:
    actor = (observer + 1) % NUM_SEATS
    event = _event(actor_seat=actor, event_type=EventType.PASS)
    assert relativize(event, observer).actor_rel == 1
    assert TEAM_OF[actor] != TEAM_OF[observer]


@pytest.mark.parametrize("observer", SEATS)
def test_partner_is_two_and_is_the_same_team(observer: int) -> None:
    actor = (observer + 2) % NUM_SEATS
    event = _event(actor_seat=actor, event_type=EventType.PASS)
    assert relativize(event, observer).actor_rel == 2
    assert TEAM_OF[actor] == TEAM_OF[observer]


@pytest.mark.parametrize("observer", SEATS)
def test_left_hand_opponent_is_three(observer: int) -> None:
    actor = (observer + 3) % NUM_SEATS
    event = _event(actor_seat=actor, event_type=EventType.PASS)
    assert relativize(event, observer).actor_rel == 3
    assert TEAM_OF[actor] != TEAM_OF[observer]


def test_the_documented_worked_example() -> None:
    """Absolute actor seat 3 observed from seat 1 is the observer's partner."""
    event = _event(actor_seat=3, event_type=EventType.PASS)
    assert relativize(event, 1).actor_rel == 2


# --- trick winner -----------------------------------------------------------


@pytest.mark.parametrize(("winner", "observer"), list(itertools.product(SEATS, SEATS)))
def test_winner_rel_for_every_combination(winner: int, observer: int) -> None:
    event = _event(
        phase=Phase.PLAY,
        event_type=EventType.TRICK_COMPLETED,
        trick_no=4,
        trick_winner_seat=winner,
    )
    assert relativize(event, observer).winner_rel == (winner - observer) % NUM_SEATS


@pytest.mark.parametrize(
    ("actor", "winner", "observer"), list(itertools.product(SEATS, SEATS, SEATS))
)
def test_actor_and_winner_convert_independently(actor: int, winner: int, observer: int) -> None:
    event = _event(
        phase=Phase.PLAY,
        actor_seat=actor,
        event_type=EventType.CARD_PLAYED,
        card=17,
        trick_no=3,
        trick_winner_seat=winner,
    )
    rel = relativize(event, observer)
    assert rel.actor_rel == (actor - observer) % NUM_SEATS
    assert rel.winner_rel == (winner - observer) % NUM_SEATS


# --- None handling ----------------------------------------------------------


@pytest.mark.parametrize("observer", SEATS)
def test_none_actor_and_none_winner_stay_none(observer: int) -> None:
    rel = relativize(_event(event_type=EventType.PAD), observer)
    assert rel.actor_rel is None
    assert rel.winner_rel is None


@pytest.mark.parametrize("observer", SEATS)
def test_none_winner_stays_none_when_the_actor_is_known(observer: int) -> None:
    event = _event(actor_seat=0, event_type=EventType.BID, bid_value=7)
    rel = relativize(event, observer)
    assert rel.actor_rel is not None
    assert rel.winner_rel is None


# --- everything else is carried through unchanged ---------------------------


def test_non_seat_fields_are_preserved() -> None:
    event = _event(
        seq=37,
        phase=Phase.PLAY,
        actor_seat=1,
        event_type=EventType.CARD_PLAYED,
        card=44,
        trick_no=8,
    )
    rel = relativize(event, 3)
    assert rel.seq == 37
    assert rel.phase is Phase.PLAY
    assert rel.event_type is EventType.CARD_PLAYED
    assert rel.card == 44
    assert rel.trick_no == 8
    assert rel.bid_value is None
    assert rel.trump is None


def test_bid_and_trump_values_are_preserved() -> None:
    bid_event = _event(actor_seat=0, event_type=EventType.BID, bid_value=11)
    trump_event = _event(
        phase=Phase.DECLARE, actor_seat=0, event_type=EventType.TRUMP_SELECTED, trump=3
    )
    assert relativize(bid_event, 2).bid_value == 11
    assert relativize(trump_event, 2).trump == 3


def test_deal_result_is_not_carried_into_the_relative_form() -> None:
    """The relative event has no deal_result field; the absolute event keeps it."""
    result = DealResult(
        bid=13,
        tricks_bidding_team=0,
        score_delta_bidding_team=-16,
        score_delta_defending_team=26,
    )
    event = _event(
        phase=Phase.SCORE,
        event_type=EventType.DEAL_COMPLETED,
        trick_no=13,
        deal_result=result,
    )
    rel = relativize(event, 0)
    assert not hasattr(rel, "deal_result")
    assert event.deal_result is result


def test_relativize_does_not_mutate_the_source_event() -> None:
    event = _event(actor_seat=2, event_type=EventType.PASS, trick_winner_seat=1)
    before = (event.actor_seat, event.trick_winner_seat)
    for observer in SEATS:
        relativize(event, observer)
    assert (event.actor_seat, event.trick_winner_seat) == before


def test_relativize_is_deterministic() -> None:
    event = _event(actor_seat=2, event_type=EventType.PASS)
    assert relativize(event, 1) == relativize(event, 1)


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize("observer", [-1, 4, 7, 52])
def test_invalid_observer_seat_raises(observer: int) -> None:
    event = _event(actor_seat=0, event_type=EventType.PASS)
    with pytest.raises(ValueError, match="observer_seat must be in"):
        relativize(event, observer)
