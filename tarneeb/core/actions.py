"""Frozen action encoding ``act_v1``.

One flat discrete action space of 65 ids covers every phase of the game. Which
ids are legal at a given moment is decided by a mask, never by penalising an
illegal choice.

::

    0..51  CARD(card_id)
    52..58 BID(7)..BID(13)
    59     PASS
    60..63 TRUMP(clubs, diamonds, hearts, spades)
    64     TRUMP(no-trump) -- reserved, permanently illegal under
           tarneeb-classic-v1

Id 64 exists only so a checkpoint trained with no-trump disabled stays
loadable if the rule is ever enabled in a future rules version. It is masked
in every reachable state.

This module lives in ``tarneeb.core`` because the engine needs the layout to
build its legal-action mask, and ``tarneeb.core`` may not import from
``tarneeb.env``. ``tarneeb/env/actions.py`` wraps it for the env layer.
"""

from __future__ import annotations

ACTION_SPEC_VERSION = "act_v1"

NUM_ACTIONS = 65

CARD_ACTION_START = 0
CARD_ACTION_STOP = 52
"""Exclusive upper bound of the card block, i.e. ids 0..51."""

BID_ACTION_START = 52
BID_ACTION_STOP = 59
"""Exclusive upper bound of the bid block, i.e. ids 52..58 for bids 7..13."""

PASS_ACTION = 59

TRUMP_ACTION_START = 60
TRUMP_ACTION_STOP = 64
"""Exclusive upper bound of the suit-trump block, i.e. ids 60..63."""

NO_TRUMP_ACTION = 64
"""Reserved; masked in every state under tarneeb-classic-v1."""

LOWEST_BID = 7
HIGHEST_BID = 13

NUM_SUITS = 4
NUM_CARDS = 52

SUIT_NAMES = ("clubs", "diamonds", "hearts", "spades")


def action_for_bid(bid: int) -> int:
    """Return the action id that bids ``bid``."""
    if not LOWEST_BID <= bid <= HIGHEST_BID:
        raise ValueError(f"bid must be in {LOWEST_BID}..{HIGHEST_BID}, got {bid}")
    return BID_ACTION_START + (bid - LOWEST_BID)


def bid_for_action(action_id: int) -> int:
    """Return the bid value encoded by a bid action id."""
    if not BID_ACTION_START <= action_id < BID_ACTION_STOP:
        raise ValueError(
            f"action_id must be in {BID_ACTION_START}..{BID_ACTION_STOP - 1}, got {action_id}"
        )
    return LOWEST_BID + (action_id - BID_ACTION_START)


def action_for_trump_suit(suit: int) -> int:
    """Return the action id that declares ``suit`` as trump."""
    if not 0 <= suit < NUM_SUITS:
        raise ValueError(f"suit must be in 0..{NUM_SUITS - 1}, got {suit}")
    return TRUMP_ACTION_START + suit


def trump_suit_for_action(action_id: int) -> int:
    """Return the suit declared by a trump action id.

    ``NO_TRUMP_ACTION`` is rejected: it never denotes a playable suit.
    """
    if not TRUMP_ACTION_START <= action_id < TRUMP_ACTION_STOP:
        raise ValueError(
            f"action_id must be in {TRUMP_ACTION_START}..{TRUMP_ACTION_STOP - 1}, got {action_id}"
        )
    return action_id - TRUMP_ACTION_START


def action_for_card(card_id: int) -> int:
    """Return the action id that plays ``card_id`` (the identity mapping)."""
    if not 0 <= card_id < NUM_CARDS:
        raise ValueError(f"card_id must be in 0..{NUM_CARDS - 1}, got {card_id}")
    return CARD_ACTION_START + card_id


def card_for_action(action_id: int) -> int:
    """Return the card played by a card action id."""
    if not CARD_ACTION_START <= action_id < CARD_ACTION_STOP:
        raise ValueError(
            f"action_id must be in {CARD_ACTION_START}..{CARD_ACTION_STOP - 1}, got {action_id}"
        )
    return action_id - CARD_ACTION_START


def is_bid_action(action_id: int) -> bool:
    return BID_ACTION_START <= action_id < BID_ACTION_STOP


def is_trump_action(action_id: int) -> bool:
    """True for the four suit trump ids; False for the reserved no-trump id."""
    return TRUMP_ACTION_START <= action_id < TRUMP_ACTION_STOP


def is_card_action(action_id: int) -> bool:
    return CARD_ACTION_START <= action_id < CARD_ACTION_STOP
