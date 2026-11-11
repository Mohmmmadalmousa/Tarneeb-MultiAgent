"""Basic heuristic agent -- deliberately simple and readable.

Implements blueprint section 11.2 exactly:

*Bidding*: ``HCP = 4A + 3K + 2Q + 1J``;
``Strength = HCP + sum_s max(0, length_s - 4) + 2 * voids + 1 * singletons``.
Bid 7 if ``Strength >= 13``, else pass. Never raises its own bid.

*Trump*: longest suit, ties broken by high-card points in that suit, then by
suit id so the choice is always deterministic.

*Play*: lead the highest card of the longest non-trump suit; when following and
able to win, play the lowest winning card; otherwise the lowest card of the led
suit; when void, discard the lowest card of the shortest non-trump suit.

Every choice is intersected with the legal mask, so the heuristic can never
bypass engine legality.
"""

from __future__ import annotations

from tarneeb.agents.base import Agent, checked_action, first_legal_action
from tarneeb.core.actions import (
    PASS_ACTION,
    action_for_bid,
    action_for_card,
    action_for_trump_suit,
    card_for_action,
    is_card_action,
)
from tarneeb.core.cards import NUM_SUITS, rank_of, suit_of
from tarneeb.core.events import Phase
from tarneeb.env.public_view import PublicView

HIGH_CARD_POINTS_BY_RANK = {12: 4, 11: 3, 10: 2, 9: 1}
"""Ace 4, king 3, queen 2, jack 1; everything else zero."""

LONG_SUIT_THRESHOLD = 4
VOID_BONUS = 2
SINGLETON_BONUS = 1
DEFAULT_BID_THRESHOLD = 13
OPENING_BID = 7


def high_card_points(cards: tuple[int, ...]) -> int:
    """Return the Milton-Work high-card points of ``cards``."""
    return sum(HIGH_CARD_POINTS_BY_RANK.get(rank_of(card), 0) for card in cards)


def hand_strength(view: PublicView) -> int:
    """Return the basic strength score of the observer's hand."""
    suit_lengths = view.own_suit_lengths()
    length_bonus = sum(max(0, length - LONG_SUIT_THRESHOLD) for length in suit_lengths)
    void_count = sum(1 for length in suit_lengths if length == 0)
    singleton_count = sum(1 for length in suit_lengths if length == 1)
    return (
        high_card_points(view.own_hand)
        + length_bonus
        + VOID_BONUS * void_count
        + SINGLETON_BONUS * singleton_count
    )


class BasicHeuristicAgent(Agent):
    """A simple, fully deterministic baseline."""

    def __init__(self, bid_threshold: int = DEFAULT_BID_THRESHOLD, name: str = "basic") -> None:
        self.bid_threshold = bid_threshold
        self.name = name

    def act(self, view: PublicView) -> int:
        if view.phase is Phase.BID:
            selected_action = self._choose_bid(view)
        elif view.phase is Phase.DECLARE:
            selected_action = self._choose_trump(view)
        else:
            selected_action = self._choose_card(view)
        return checked_action(selected_action, view.legal_action_mask, self.name)

    # --- bidding ------------------------------------------------------

    def _choose_bid(self, view: PublicView) -> int:
        """Open 7 with a strong enough hand; otherwise pass. Never raises."""
        opening_action = action_for_bid(OPENING_BID)
        if (
            view.current_high_bid is None
            and hand_strength(view) >= self.bid_threshold
            and view.legal_action_mask[opening_action]
        ):
            return opening_action
        return PASS_ACTION

    # --- trump --------------------------------------------------------

    def _choose_trump(self, view: PublicView) -> int:
        """Longest suit, ties by high-card points in it, then by suit id."""
        candidate_suits = [
            suit for suit in range(NUM_SUITS) if view.legal_action_mask[action_for_trump_suit(suit)]
        ]
        if not candidate_suits:  # pragma: no cover - DECLARE always offers four suits
            return first_legal_action(view.legal_action_mask)

        def suit_ranking_key(suit: int) -> tuple[int, int, int]:
            suit_cards = view.cards_of_suit_in_own_hand(suit)
            return (len(suit_cards), high_card_points(suit_cards), -suit)

        best_suit = max(candidate_suits, key=suit_ranking_key)
        return action_for_trump_suit(best_suit)

    # --- card play ----------------------------------------------------

    def _choose_card(self, view: PublicView) -> int:
        playable_cards = [
            card_for_action(action_id)
            for action_id in range(len(view.legal_action_mask))
            if view.legal_action_mask[action_id] and is_card_action(action_id)
        ]
        if not playable_cards:  # pragma: no cover - PLAY always offers a card
            return first_legal_action(view.legal_action_mask)

        if not view.current_trick:
            return action_for_card(self._choose_lead(view, playable_cards))

        led_suit = view.led_suit
        assert led_suit is not None
        following_cards = [card for card in playable_cards if suit_of(card) == led_suit]

        if following_cards:
            return action_for_card(self._choose_follow(view, following_cards))
        return action_for_card(self._choose_discard(view, playable_cards))

    def _choose_lead(self, view: PublicView, playable_cards: list[int]) -> int:
        """Lead the highest card of the longest non-trump suit."""
        non_trump_cards = [card for card in playable_cards if suit_of(card) != view.trump_suit]
        candidates = non_trump_cards or playable_cards

        suit_lengths = view.own_suit_lengths()
        return max(candidates, key=lambda card: (suit_lengths[suit_of(card)], rank_of(card), card))

    def _choose_follow(self, view: PublicView, following_cards: list[int]) -> int:
        """Play the lowest winning card if one exists, else the lowest card."""
        winning_cards = [card for card in following_cards if self._would_currently_win(view, card)]
        if winning_cards:
            return min(winning_cards, key=lambda card: (rank_of(card), card))
        return min(following_cards, key=lambda card: (rank_of(card), card))

    def _choose_discard(self, view: PublicView, playable_cards: list[int]) -> int:
        """Discard the lowest card of the shortest non-trump suit."""
        non_trump_cards = [card for card in playable_cards if suit_of(card) != view.trump_suit]
        candidates = non_trump_cards or playable_cards

        suit_lengths = view.own_suit_lengths()
        return min(candidates, key=lambda card: (suit_lengths[suit_of(card)], rank_of(card), card))

    def _would_currently_win(self, view: PublicView, card: int) -> bool:
        """Whether ``card`` beats every card already played to this trick.

        Uses the same comparison rule as the engine (C18) but only over cards
        that are already public, so it reads no hidden information.
        """
        led_suit = view.led_suit
        if led_suit is None:
            return True
        best_card = view.current_trick[0][1]
        for _seat, played_card in view.current_trick[1:]:
            if _beats(played_card, best_card, led_suit, view.trump_suit):
                best_card = played_card
        return _beats(card, best_card, led_suit, view.trump_suit)


def _beats(card: int, current_best: int, led_suit: int, trump_suit: int | None) -> bool:
    """Return whether ``card`` beats ``current_best`` (C18)."""
    card_is_trump = trump_suit is not None and suit_of(card) == trump_suit
    best_is_trump = trump_suit is not None and suit_of(current_best) == trump_suit
    if card_is_trump != best_is_trump:
        return card_is_trump
    if card_is_trump:
        return rank_of(card) > rank_of(current_best)
    if suit_of(card) != led_suit:
        return False
    return suit_of(current_best) == led_suit and rank_of(card) > rank_of(current_best)
