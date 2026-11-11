"""Tests for the frozen action encoding act_v1.

Expected ids are written from the blueprint's action table, not read back from
the implementation.
"""

from __future__ import annotations

import pytest

from tarneeb.core.actions import (
    ACTION_SPEC_VERSION,
    BID_ACTION_START,
    BID_ACTION_STOP,
    CARD_ACTION_START,
    CARD_ACTION_STOP,
    HIGHEST_BID,
    LOWEST_BID,
    NO_TRUMP_ACTION,
    NUM_ACTIONS,
    NUM_CARDS,
    NUM_SUITS,
    PASS_ACTION,
    SUIT_NAMES,
    TRUMP_ACTION_START,
    TRUMP_ACTION_STOP,
    action_for_bid,
    action_for_card,
    action_for_trump_suit,
    bid_for_action,
    card_for_action,
    is_bid_action,
    is_card_action,
    is_trump_action,
    trump_suit_for_action,
)
from tarneeb.env.actions import ActionEncoder

# The frozen layout, transcribed by hand.
EXPECTED_BID_ACTION_IDS = {
    7: 52,
    8: 53,
    9: 54,
    10: 55,
    11: 56,
    12: 57,
    13: 58,
}
EXPECTED_TRUMP_ACTION_IDS = {0: 60, 1: 61, 2: 62, 3: 63}


def test_action_spec_version() -> None:
    assert ACTION_SPEC_VERSION == "act_v1"


def test_action_space_size() -> None:
    assert NUM_ACTIONS == 65


def test_block_boundaries() -> None:
    assert (CARD_ACTION_START, CARD_ACTION_STOP) == (0, 52)
    assert (BID_ACTION_START, BID_ACTION_STOP) == (52, 59)
    assert PASS_ACTION == 59
    assert (TRUMP_ACTION_START, TRUMP_ACTION_STOP) == (60, 64)
    assert NO_TRUMP_ACTION == 64


def test_blocks_tile_the_action_space_without_gaps_or_overlap() -> None:
    card_ids = set(range(CARD_ACTION_START, CARD_ACTION_STOP))
    bid_ids = set(range(BID_ACTION_START, BID_ACTION_STOP))
    trump_ids = set(range(TRUMP_ACTION_START, TRUMP_ACTION_STOP))
    blocks = [card_ids, bid_ids, {PASS_ACTION}, trump_ids, {NO_TRUMP_ACTION}]

    union_of_blocks: set[int] = set()
    for block in blocks:
        assert not union_of_blocks & block
        union_of_blocks |= block
    assert union_of_blocks == set(range(NUM_ACTIONS))


def test_bid_range_matches_the_frozen_rules() -> None:
    assert (LOWEST_BID, HIGHEST_BID) == (7, 13)
    assert BID_ACTION_STOP - BID_ACTION_START == HIGHEST_BID - LOWEST_BID + 1


def test_suit_and_card_counts() -> None:
    assert NUM_SUITS == 4
    assert NUM_CARDS == 52
    assert SUIT_NAMES == ("clubs", "diamonds", "hearts", "spades")


# --- bid conversions --------------------------------------------------------


@pytest.mark.parametrize(
    ("bid_value", "expected_action_id"), sorted(EXPECTED_BID_ACTION_IDS.items())
)
def test_action_for_bid(bid_value: int, expected_action_id: int) -> None:
    assert action_for_bid(bid_value) == expected_action_id
    assert bid_for_action(expected_action_id) == bid_value
    assert is_bid_action(expected_action_id)


@pytest.mark.parametrize("bid_value", [-1, 0, 6, 14, 26])
def test_action_for_bid_rejects_illegal_bids(bid_value: int) -> None:
    with pytest.raises(ValueError, match="bid must be in"):
        action_for_bid(bid_value)


@pytest.mark.parametrize("action_id", [51, 59, 60, 64])
def test_bid_for_action_rejects_non_bid_ids(action_id: int) -> None:
    with pytest.raises(ValueError, match="action_id must be in"):
        bid_for_action(action_id)


# --- trump conversions ------------------------------------------------------


@pytest.mark.parametrize(("suit", "expected_action_id"), sorted(EXPECTED_TRUMP_ACTION_IDS.items()))
def test_action_for_trump_suit(suit: int, expected_action_id: int) -> None:
    assert action_for_trump_suit(suit) == expected_action_id
    assert trump_suit_for_action(expected_action_id) == suit
    assert is_trump_action(expected_action_id)


@pytest.mark.parametrize("suit", [-1, 4, 5])
def test_action_for_trump_suit_rejects_non_suits(suit: int) -> None:
    with pytest.raises(ValueError, match="suit must be in"):
        action_for_trump_suit(suit)


def test_no_trump_action_is_not_a_suit_action() -> None:
    assert not is_trump_action(NO_TRUMP_ACTION)
    with pytest.raises(ValueError, match="action_id must be in"):
        trump_suit_for_action(NO_TRUMP_ACTION)


# --- card conversions -------------------------------------------------------


@pytest.mark.parametrize("card_id", [0, 1, 13, 26, 51])
def test_card_actions_are_the_identity_mapping(card_id: int) -> None:
    assert action_for_card(card_id) == card_id
    assert card_for_action(card_id) == card_id
    assert is_card_action(card_id)


@pytest.mark.parametrize("card_id", [-1, 52, 65])
def test_action_for_card_rejects_non_cards(card_id: int) -> None:
    with pytest.raises(ValueError, match="card_id must be in"):
        action_for_card(card_id)


# --- predicates -------------------------------------------------------------


def test_predicates_partition_the_action_space() -> None:
    for action_id in range(NUM_ACTIONS):
        predicate_hits = [
            is_card_action(action_id),
            is_bid_action(action_id),
            action_id == PASS_ACTION,
            is_trump_action(action_id),
            action_id == NO_TRUMP_ACTION,
        ]
        assert sum(predicate_hits) == 1, action_id


# --- the env-layer wrapper --------------------------------------------------


def test_action_encoder_exposes_the_same_layout() -> None:
    assert ActionEncoder.N_ACTIONS == NUM_ACTIONS
    assert ActionEncoder.ACTION_SPEC_VERSION == ACTION_SPEC_VERSION
    assert ActionEncoder.PASS_ACTION == PASS_ACTION
    assert ActionEncoder.NO_TRUMP_ACTION == NO_TRUMP_ACTION
    assert ActionEncoder.BID_ACTION_START == BID_ACTION_START
    assert ActionEncoder.TRUMP_ACTION_START == TRUMP_ACTION_START


def test_action_encoder_conversions_delegate_to_core() -> None:
    assert ActionEncoder.action_for_bid(13) == 58
    assert ActionEncoder.bid_for_action(58) == 13
    assert ActionEncoder.action_for_trump_suit(3) == 63
    assert ActionEncoder.trump_suit_for_action(63) == 3


@pytest.mark.parametrize(
    ("action_id", "expected_description"),
    [
        (0, "CARD(0)"),
        (51, "CARD(51)"),
        (52, "BID(7)"),
        (58, "BID(13)"),
        (59, "PASS"),
        (60, "TRUMP(0)"),
        (63, "TRUMP(3)"),
        (64, "TRUMP(NT, reserved)"),
    ],
)
def test_action_encoder_describe(action_id: int, expected_description: str) -> None:
    assert ActionEncoder.describe(action_id) == expected_description


@pytest.mark.parametrize("action_id", [-1, 65, 100])
def test_action_encoder_describe_rejects_out_of_range(action_id: int) -> None:
    with pytest.raises(ValueError, match="action_id must be in"):
        ActionEncoder.describe(action_id)
