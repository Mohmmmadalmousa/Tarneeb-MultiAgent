"""Card encoding and hand bitmasks.

Cards are plain integers, not objects: ``card_id = suit * 13 + rank`` with
``suit`` in 0..3 (clubs, diamonds, hearts, spades) and ``rank`` in 0..12
(2 low, ace high). Hands are ``int`` bitmasks, one bit per card id.

This is a deliberate performance decision. The simulator runs millions of
deals during training, so the hot path allocates nothing per card and tests
suit membership with a single bitwise ``and``. Human-readable views
(:func:`card_name`, :func:`cards_in_hand`) exist for the CLI and for tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

Suit = int
"""0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades."""

Rank = int
"""0 = two ... 8 = ten, 9 = jack, 10 = queen, 11 = king, 12 = ace."""

Card = int
"""0..51, equal to ``suit * 13 + rank``."""

NUM_SUITS = 4
NUM_RANKS = 13
NUM_CARDS = NUM_SUITS * NUM_RANKS
CARDS_PER_HAND = 13
NUM_SEATS = 4

SUIT_NAMES = ("clubs", "diamonds", "hearts", "spades")
SUIT_SYMBOLS = ("C", "D", "H", "S")
RANK_SYMBOLS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

FULL_DECK_MASK = (1 << NUM_CARDS) - 1

SUIT_MASKS: tuple[int, ...] = tuple(
    sum(1 << (suit * NUM_RANKS + rank) for rank in range(NUM_RANKS)) for suit in range(NUM_SUITS)
)
"""``SUIT_MASKS[suit]`` has a bit set for every card of that suit."""


def suit_of(card: Card) -> Suit:
    """Return the suit of ``card``."""
    return card // NUM_RANKS


def rank_of(card: Card) -> Rank:
    """Return the rank of ``card``; higher beats lower within a suit."""
    return card % NUM_RANKS


def card_id(suit: Suit, rank: Rank) -> Card:
    """Return the card id for ``suit`` and ``rank``."""
    if not 0 <= suit < NUM_SUITS:
        raise ValueError(f"suit must be in 0..{NUM_SUITS - 1}, got {suit}")
    if not 0 <= rank < NUM_RANKS:
        raise ValueError(f"rank must be in 0..{NUM_RANKS - 1}, got {rank}")
    return suit * NUM_RANKS + rank


def card_name(card: Card) -> str:
    """Return a short human-readable name such as ``"AH"`` or ``"10S"``."""
    if not 0 <= card < NUM_CARDS:
        raise ValueError(f"card must be in 0..{NUM_CARDS - 1}, got {card}")
    return f"{RANK_SYMBOLS[rank_of(card)]}{SUIT_SYMBOLS[suit_of(card)]}"


# --- hand bitmasks ----------------------------------------------------------


def hand_from_cards(cards: Iterable[int]) -> int:
    """Return the bitmask holding exactly ``cards``."""
    hand_mask = 0
    for card in cards:
        card_value = int(card)
        if not 0 <= card_value < NUM_CARDS:
            raise ValueError(f"card must be in 0..{NUM_CARDS - 1}, got {card_value}")
        hand_mask |= 1 << card_value
    return hand_mask


def cards_in_hand(hand_mask: int) -> list[Card]:
    """Return the cards in ``hand_mask``, ascending by card id."""
    return [card for card in range(NUM_CARDS) if hand_mask >> card & 1]


def hand_contains(hand_mask: int, card: Card) -> bool:
    """Return whether ``hand_mask`` holds ``card``."""
    return bool(hand_mask >> card & 1)


def remove_card(hand_mask: int, card: Card) -> int:
    """Return ``hand_mask`` without ``card``.

    Raises:
        ValueError: If the hand does not hold the card.
    """
    if not hand_contains(hand_mask, card):
        raise ValueError(f"hand does not contain card {card}")
    return hand_mask & ~(1 << card)


def hand_size(hand_mask: int) -> int:
    """Return how many cards ``hand_mask`` holds."""
    return int(hand_mask.bit_count())


def cards_of_suit_in_hand(hand_mask: int, suit: Suit) -> int:
    """Return the sub-mask of ``hand_mask`` belonging to ``suit``."""
    return hand_mask & SUIT_MASKS[suit]


def holds_suit(hand_mask: int, suit: Suit) -> bool:
    """Return whether ``hand_mask`` holds at least one card of ``suit``."""
    return bool(hand_mask & SUIT_MASKS[suit])


# --- dealing ----------------------------------------------------------------


def derive_deal_seed(match_seed: int, deal_index: int) -> int:
    """Return the deterministic seed for one deal of a match.

    Deriving the seed from ``(match_seed, deal_index)`` rather than advancing a
    single stream makes dealing a pure function of those two numbers, which is
    what lets duplicate-deal evaluation replay the same deal at will.
    """
    digest = hashlib.sha256(f"{match_seed}:{deal_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def deal_hands(rng: np.random.Generator) -> list[int]:
    """Shuffle a full deck and return four 13-card hand bitmasks.

    The four hands are disjoint and together cover all 52 cards.
    """
    shuffled_deck = rng.permutation(NUM_CARDS)
    hands = []
    for seat in range(NUM_SEATS):
        seat_cards = shuffled_deck[seat * CARDS_PER_HAND : (seat + 1) * CARDS_PER_HAND]
        hands.append(hand_from_cards(seat_cards))
    return hands
