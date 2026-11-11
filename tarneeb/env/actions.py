"""Environment-facing view of the frozen action space ``act_v1``.

The layout itself lives in :mod:`tarneeb.core.actions`, because the engine
needs it to build legal-action masks and ``tarneeb.core`` may never import from
``tarneeb.env``. This module is the thin env-layer wrapper the blueprint names.
"""

from __future__ import annotations

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
    NUM_SUITS,
    PASS_ACTION,
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

__all__ = [
    "ACTION_SPEC_VERSION",
    "NUM_ACTIONS",
    "ActionEncoder",
]


class ActionEncoder:
    """Names and conversions for the 65 action ids.

    ``0..51`` play a card, ``52..58`` bid 7..13, ``59`` passes, ``60..63``
    declare a trump suit, and ``64`` is the reserved no-trump id that is masked
    in every state under ``tarneeb-classic-v1``.
    """

    ACTION_SPEC_VERSION: str = ACTION_SPEC_VERSION
    N_ACTIONS: int = NUM_ACTIONS

    CARD_ACTION_START: int = CARD_ACTION_START
    CARD_ACTION_STOP: int = CARD_ACTION_STOP
    BID_ACTION_START: int = BID_ACTION_START
    BID_ACTION_STOP: int = BID_ACTION_STOP
    PASS_ACTION: int = PASS_ACTION
    TRUMP_ACTION_START: int = TRUMP_ACTION_START
    TRUMP_ACTION_STOP: int = TRUMP_ACTION_STOP
    NO_TRUMP_ACTION: int = NO_TRUMP_ACTION

    LOWEST_BID: int = LOWEST_BID
    HIGHEST_BID: int = HIGHEST_BID
    NUM_SUITS: int = NUM_SUITS

    action_for_bid = staticmethod(action_for_bid)
    bid_for_action = staticmethod(bid_for_action)
    action_for_trump_suit = staticmethod(action_for_trump_suit)
    trump_suit_for_action = staticmethod(trump_suit_for_action)
    action_for_card = staticmethod(action_for_card)
    card_for_action = staticmethod(card_for_action)
    is_bid_action = staticmethod(is_bid_action)
    is_trump_action = staticmethod(is_trump_action)
    is_card_action = staticmethod(is_card_action)

    @staticmethod
    def describe(action_id: int) -> str:
        """Return a short human-readable name for an action id."""
        if is_card_action(action_id):
            return f"CARD({card_for_action(action_id)})"
        if is_bid_action(action_id):
            return f"BID({bid_for_action(action_id)})"
        if action_id == PASS_ACTION:
            return "PASS"
        if is_trump_action(action_id):
            return f"TRUMP({trump_suit_for_action(action_id)})"
        if action_id == NO_TRUMP_ACTION:
            return "TRUMP(NT, reserved)"
        raise ValueError(f"action_id must be in 0..{NUM_ACTIONS - 1}, got {action_id}")
