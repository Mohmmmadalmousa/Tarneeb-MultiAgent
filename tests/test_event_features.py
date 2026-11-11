"""Factored event encoding tests.

Every expected id below is written by hand from the field layout, not read back
from the implementation:

| field | index | cardinality | encoding |
|---|---|---|---|
| event_type | 0 | 7 | the EventType value itself, PAD = 0 |
| actor_rel | 1 | 5 | none = 0, else 1 + relative seat |
| bid_value | 2 | 8 | none = 0, else bid - 6 |
| trump | 3 | 6 | none = 0, else 1 + suit; 5 reserved for no-trump |
| card | 4 | 53 | none = 0, else 1 + card id |
| trick_no | 5 | 15 | none = 0, else 1 + trick number |
| winner_rel | 6 | 5 | none = 0, else 1 + relative seat |
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.core.cards import NUM_SEATS, card_id
from tarneeb.core.events import DealResult, EventType, Phase, PublicEvent
from tarneeb.env.event_features import (
    EVENT_FIELD_CARDINALITIES,
    EVENT_FIELD_NAMES,
    EVENT_FIELD_OFFSETS,
    EVENT_ID_DTYPE,
    PAD_EVENT_IDS,
    event_ids,
    event_onehot,
    expand_event_id_rows,
    expand_event_ids,
)
from tarneeb.env.spaces import D_EVENT_ONEHOT, N_EVENT_FIELDS


def make_event(**overrides: object) -> PublicEvent:
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


# --- the frozen contract ----------------------------------------------------


def test_field_count_and_names() -> None:
    assert N_EVENT_FIELDS == 7
    assert EVENT_FIELD_NAMES == (
        "event_type",
        "actor_rel",
        "bid_value",
        "trump",
        "card",
        "trick_no",
        "winner_rel",
    )


def test_field_cardinalities_sum_to_99() -> None:
    assert EVENT_FIELD_CARDINALITIES == (7, 5, 8, 6, 53, 15, 5)
    assert sum(EVENT_FIELD_CARDINALITIES) == D_EVENT_ONEHOT == 99


def test_field_offsets() -> None:
    assert EVENT_FIELD_OFFSETS == (0, 7, 12, 20, 26, 79, 94)


def test_pad_row_is_all_zeros() -> None:
    assert PAD_EVENT_IDS.tolist() == [0] * N_EVENT_FIELDS
    assert PAD_EVENT_IDS.dtype == EVENT_ID_DTYPE


# --- hand-written examples --------------------------------------------------


def test_bid_event_ids() -> None:
    """Seat 2 bids 9, observed by seat 0: actor is the observer's partner."""
    event = make_event(actor_seat=2, event_type=EventType.BID, bid_value=9)
    assert event_ids(event, 0).tolist() == [1, 3, 3, 0, 0, 1, 0]
    #                                       ^  ^  ^           ^
    #                       BID=1  partner->1+2  9-6=3   trick 0 -> 1


def test_bid_event_from_each_observer() -> None:
    event = make_event(actor_seat=2, event_type=EventType.BID, bid_value=13)
    assert event_ids(event, 0).tolist()[1] == 3  # partner
    assert event_ids(event, 1).tolist()[1] == 2  # right-hand opponent
    assert event_ids(event, 2).tolist()[1] == 1  # self
    assert event_ids(event, 3).tolist()[1] == 4  # left-hand opponent
    assert event_ids(event, 0).tolist()[2] == 7  # bid 13 -> 13 - 6


@pytest.mark.parametrize(("bid_value", "expected_id"), [(7, 1), (8, 2), (10, 4), (13, 7)])
def test_every_bid_value_id(bid_value: int, expected_id: int) -> None:
    event = make_event(actor_seat=0, event_type=EventType.BID, bid_value=bid_value)
    assert event_ids(event, 0)[2] == expected_id


def test_pass_event_ids() -> None:
    event = make_event(actor_seat=1, event_type=EventType.PASS)
    assert event_ids(event, 0).tolist() == [2, 2, 0, 0, 0, 1, 0]


def test_trump_selected_event_ids() -> None:
    """Seat 3 declares hearts (suit 2), observed by seat 1: seat 3 is the partner."""
    event = make_event(
        phase=Phase.DECLARE, actor_seat=3, event_type=EventType.TRUMP_SELECTED, trump=2
    )
    assert event_ids(event, 1).tolist() == [3, 3, 0, 3, 0, 1, 0]


@pytest.mark.parametrize(("suit", "expected_id"), [(0, 1), (1, 2), (2, 3), (3, 4)])
def test_every_trump_suit_id(suit: int, expected_id: int) -> None:
    event = make_event(actor_seat=0, event_type=EventType.TRUMP_SELECTED, trump=suit)
    assert event_ids(event, 0)[3] == expected_id


def test_no_trump_id_is_reserved_and_never_produced() -> None:
    """Index 5 of the trump field exists for a future rules version only."""
    assert EVENT_FIELD_CARDINALITIES[3] == 6
    for suit in range(4):
        event = make_event(actor_seat=0, event_type=EventType.TRUMP_SELECTED, trump=suit)
        assert event_ids(event, 0)[3] < 5


def test_card_played_event_ids() -> None:
    """Seat 0 plays the ace of spades on trick 4, observed by seat 3."""
    ace_of_spades = card_id(3, 12)  # 51
    event = make_event(
        phase=Phase.PLAY,
        actor_seat=0,
        event_type=EventType.CARD_PLAYED,
        card=ace_of_spades,
        trick_no=4,
    )
    assert event_ids(event, 3).tolist() == [4, 2, 0, 0, 52, 5, 0]


@pytest.mark.parametrize("card", [0, 1, 25, 50, 51])
def test_card_ids_are_offset_by_one(card: int) -> None:
    event = make_event(actor_seat=0, event_type=EventType.CARD_PLAYED, card=card)
    assert event_ids(event, 0)[4] == card + 1


def test_trick_completed_event_ids() -> None:
    """No actor; the winner is seat 1, observed by seat 0 as the right-hand opponent."""
    event = make_event(
        phase=Phase.PLAY,
        event_type=EventType.TRICK_COMPLETED,
        trick_no=7,
        trick_winner_seat=1,
    )
    assert event_ids(event, 0).tolist() == [5, 0, 0, 0, 0, 8, 2]


def test_deal_completed_event_ids() -> None:
    result = DealResult(
        bid=9,
        tricks_bidding_team=9,
        score_delta_bidding_team=9,
        score_delta_defending_team=0,
    )
    event = make_event(
        phase=Phase.SCORE, event_type=EventType.DEAL_COMPLETED, trick_no=13, deal_result=result
    )
    assert event_ids(event, 0).tolist() == [6, 0, 0, 0, 0, 14, 0]


def test_deal_completed_carries_no_score_information() -> None:
    """The seven fields cannot encode a score, so the result stays out of the ids."""
    high_result = DealResult(13, 13, 26, 0)
    low_result = DealResult(13, 0, -16, 26)
    high_event = make_event(
        event_type=EventType.DEAL_COMPLETED, trick_no=13, deal_result=high_result
    )
    low_event = make_event(event_type=EventType.DEAL_COMPLETED, trick_no=13, deal_result=low_result)
    assert event_ids(high_event, 0).tolist() == event_ids(low_event, 0).tolist()


# --- shapes, dtypes and ranges ----------------------------------------------


def test_event_ids_shape_and_dtype() -> None:
    event = make_event(actor_seat=0, event_type=EventType.PASS)
    ids = event_ids(event, 0)
    assert ids.shape == (7,)
    assert ids.dtype == np.int16


def test_event_onehot_shape_and_dtype() -> None:
    event = make_event(actor_seat=0, event_type=EventType.PASS)
    onehot = event_onehot(event, 0)
    assert onehot.shape == (99,)
    assert onehot.dtype == np.float32


def test_onehot_has_exactly_one_bit_per_field() -> None:
    event = make_event(
        phase=Phase.PLAY, actor_seat=2, event_type=EventType.CARD_PLAYED, card=17, trick_no=3
    )
    onehot = event_onehot(event, 1)
    assert onehot.sum() == N_EVENT_FIELDS
    for field_index, offset in enumerate(EVENT_FIELD_OFFSETS):
        width = EVENT_FIELD_CARDINALITIES[field_index]
        assert onehot[offset : offset + width].sum() == 1.0


def test_onehot_positions_match_the_ids() -> None:
    event = make_event(
        phase=Phase.PLAY, actor_seat=3, event_type=EventType.CARD_PLAYED, card=40, trick_no=6
    )
    ids = event_ids(event, 2)
    onehot = event_onehot(event, 2)
    for field_index, field_id in enumerate(ids):
        assert onehot[EVENT_FIELD_OFFSETS[field_index] + int(field_id)] == 1.0


def test_pad_row_expands_to_the_zero_categories() -> None:
    onehot = expand_event_ids(PAD_EVENT_IDS)
    assert onehot.sum() == N_EVENT_FIELDS
    for offset in EVENT_FIELD_OFFSETS:
        assert onehot[offset] == 1.0  # index 0 of every field


def test_expand_rejects_a_bad_shape() -> None:
    with pytest.raises(ValueError, match="expected 7 ids"):
        expand_event_ids(np.zeros(6, dtype=np.int16))


def test_expand_rejects_an_out_of_range_id() -> None:
    bad_ids = np.zeros(7, dtype=np.int16)
    bad_ids[3] = 6  # trump field has cardinality 6, so 6 is out of range
    with pytest.raises(ValueError, match="out of range"):
        expand_event_ids(bad_ids)


def test_expand_rows() -> None:
    rows = np.zeros((5, 7), dtype=np.int16)
    expanded = expand_event_id_rows(rows)
    assert expanded.shape == (5, 99)
    assert expanded.dtype == np.float32


def test_expand_rows_rejects_a_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"expected an \(L, 7\) array"):
        expand_event_id_rows(np.zeros((5, 6), dtype=np.int16))


# --- relative-seat consistency ----------------------------------------------


@pytest.mark.parametrize("actor_seat", range(NUM_SEATS))
@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
def test_actor_field_follows_the_relative_convention(actor_seat: int, observer_seat: int) -> None:
    event = make_event(actor_seat=actor_seat, event_type=EventType.PASS)
    expected_relative = (actor_seat - observer_seat) % NUM_SEATS
    assert event_ids(event, observer_seat)[1] == 1 + expected_relative


@pytest.mark.parametrize("winner_seat", range(NUM_SEATS))
@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
def test_winner_field_follows_the_relative_convention(winner_seat: int, observer_seat: int) -> None:
    event = make_event(
        event_type=EventType.TRICK_COMPLETED, trick_no=0, trick_winner_seat=winner_seat
    )
    expected_relative = (winner_seat - observer_seat) % NUM_SEATS
    assert event_ids(event, observer_seat)[6] == 1 + expected_relative


def test_all_ids_stay_within_their_cardinality_for_every_event_type() -> None:
    events = [
        make_event(actor_seat=0, event_type=EventType.BID, bid_value=13),
        make_event(actor_seat=1, event_type=EventType.PASS),
        make_event(actor_seat=2, event_type=EventType.TRUMP_SELECTED, trump=3),
        make_event(actor_seat=3, event_type=EventType.CARD_PLAYED, card=51, trick_no=12),
        make_event(event_type=EventType.TRICK_COMPLETED, trick_no=12, trick_winner_seat=0),
        make_event(event_type=EventType.DEAL_COMPLETED, trick_no=13),
        make_event(event_type=EventType.PAD),
    ]
    for event in events:
        for observer_seat in range(NUM_SEATS):
            ids = event_ids(event, observer_seat)
            for field_index, field_id in enumerate(ids):
                assert 0 <= field_id < EVENT_FIELD_CARDINALITIES[field_index]
