"""Tests for card encoding, hand bitmasks and dealing.

Covers invariants I1 (52 unique cards, four disjoint hands covering the deck)
and the encoding rules from docs/RULES.md C02 and C03.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tarneeb.core.cards import (
    CARDS_PER_HAND,
    FULL_DECK_MASK,
    NUM_CARDS,
    NUM_RANKS,
    NUM_SEATS,
    NUM_SUITS,
    RANK_SYMBOLS,
    SUIT_MASKS,
    SUIT_NAMES,
    SUIT_SYMBOLS,
    card_id,
    card_name,
    cards_in_hand,
    cards_of_suit_in_hand,
    deal_hands,
    derive_deal_seed,
    hand_contains,
    hand_from_cards,
    hand_size,
    holds_suit,
    rank_of,
    remove_card,
    suit_of,
)

ALL_CARDS = range(NUM_CARDS)


# --- constants --------------------------------------------------------------


def test_deck_shape_matches_the_frozen_rules() -> None:
    assert NUM_SUITS == 4
    assert NUM_RANKS == 13
    assert NUM_CARDS == 52  # C02
    assert CARDS_PER_HAND == 13  # C03
    assert NUM_SEATS == 4  # C01


def test_symbol_tables_are_complete() -> None:
    assert len(SUIT_NAMES) == NUM_SUITS
    assert len(SUIT_SYMBOLS) == NUM_SUITS
    assert len(RANK_SYMBOLS) == NUM_RANKS
    assert RANK_SYMBOLS[-1] == "A"  # ace high (C02)
    assert RANK_SYMBOLS[0] == "2"


def test_full_deck_mask_has_52_bits() -> None:
    assert hand_size(FULL_DECK_MASK) == NUM_CARDS


def test_suit_masks_partition_the_deck() -> None:
    union_of_suits = 0
    for suit in range(NUM_SUITS):
        assert hand_size(SUIT_MASKS[suit]) == NUM_RANKS
        assert union_of_suits & SUIT_MASKS[suit] == 0
        union_of_suits |= SUIT_MASKS[suit]
    assert union_of_suits == FULL_DECK_MASK


# --- encoding ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("suit", "rank", "expected_card"),
    [
        (0, 0, 0),  # two of clubs
        (0, 12, 12),  # ace of clubs
        (1, 0, 13),  # two of diamonds
        (2, 0, 26),  # two of hearts
        (3, 0, 39),  # two of spades
        (3, 12, 51),  # ace of spades
    ],
)
def test_card_id_layout(suit: int, rank: int, expected_card: int) -> None:
    assert card_id(suit, rank) == expected_card
    assert suit_of(expected_card) == suit
    assert rank_of(expected_card) == rank


def test_every_card_round_trips_through_suit_and_rank() -> None:
    assert [card_id(suit_of(card), rank_of(card)) for card in ALL_CARDS] == list(ALL_CARDS)


def test_card_ids_are_unique_across_suit_rank_pairs() -> None:
    all_pairs = list(itertools.product(range(NUM_SUITS), range(NUM_RANKS)))
    assert len({card_id(suit, rank) for suit, rank in all_pairs}) == NUM_CARDS


@pytest.mark.parametrize("suit", [-1, 4])
def test_card_id_rejects_bad_suits(suit: int) -> None:
    with pytest.raises(ValueError, match="suit must be in"):
        card_id(suit, 0)


@pytest.mark.parametrize("rank", [-1, 13])
def test_card_id_rejects_bad_ranks(rank: int) -> None:
    with pytest.raises(ValueError, match="rank must be in"):
        card_id(0, rank)


@pytest.mark.parametrize(
    ("card", "expected_name"),
    [(0, "2C"), (12, "AC"), (13, "2D"), (34, "10H"), (48, "JS"), (49, "QS"), (51, "AS")],
)
def test_card_name(card: int, expected_name: str) -> None:
    assert card_name(card) == expected_name


def test_card_names_are_unique() -> None:
    assert len({card_name(card) for card in ALL_CARDS}) == NUM_CARDS


@pytest.mark.parametrize("card", [-1, 52])
def test_card_name_rejects_non_cards(card: int) -> None:
    with pytest.raises(ValueError, match="card must be in"):
        card_name(card)


def test_higher_rank_means_higher_card_within_a_suit() -> None:
    for suit in range(NUM_SUITS):
        suit_cards = [card_id(suit, rank) for rank in range(NUM_RANKS)]
        assert suit_cards == sorted(suit_cards)
        assert [rank_of(card) for card in suit_cards] == list(range(NUM_RANKS))


# --- hand bitmasks ----------------------------------------------------------


def test_hand_from_cards_and_back() -> None:
    chosen_cards = [0, 5, 13, 26, 51]
    hand_mask = hand_from_cards(chosen_cards)
    assert cards_in_hand(hand_mask) == chosen_cards
    assert hand_size(hand_mask) == len(chosen_cards)


def test_empty_hand() -> None:
    assert hand_from_cards([]) == 0
    assert cards_in_hand(0) == []
    assert hand_size(0) == 0


def test_hand_from_cards_rejects_non_cards() -> None:
    with pytest.raises(ValueError, match="card must be in"):
        hand_from_cards([0, 52])


def test_hand_contains() -> None:
    hand_mask = hand_from_cards([3, 40])
    assert hand_contains(hand_mask, 3)
    assert hand_contains(hand_mask, 40)
    assert not hand_contains(hand_mask, 4)


def test_remove_card() -> None:
    hand_mask = hand_from_cards([3, 40])
    assert cards_in_hand(remove_card(hand_mask, 3)) == [40]
    assert hand_size(remove_card(hand_mask, 3)) == 1


def test_remove_card_rejects_a_card_not_held() -> None:
    with pytest.raises(ValueError, match="does not contain card"):
        remove_card(hand_from_cards([3]), 4)


def test_remove_card_does_not_mutate_the_original_mask() -> None:
    hand_mask = hand_from_cards([3, 40])
    remove_card(hand_mask, 3)
    assert hand_contains(hand_mask, 3)  # ints are immutable; the caller reassigns


def test_suit_queries() -> None:
    hand_mask = hand_from_cards([card_id(0, 5), card_id(0, 9), card_id(2, 1)])
    assert holds_suit(hand_mask, 0)
    assert holds_suit(hand_mask, 2)
    assert not holds_suit(hand_mask, 1)
    assert not holds_suit(hand_mask, 3)
    assert cards_in_hand(cards_of_suit_in_hand(hand_mask, 0)) == [card_id(0, 5), card_id(0, 9)]
    assert cards_of_suit_in_hand(hand_mask, 3) == 0


# --- seeding ----------------------------------------------------------------


def test_deal_seed_is_deterministic() -> None:
    assert derive_deal_seed(42, 0) == derive_deal_seed(42, 0)


def test_deal_seed_depends_on_both_inputs() -> None:
    assert derive_deal_seed(42, 0) != derive_deal_seed(42, 1)
    assert derive_deal_seed(42, 0) != derive_deal_seed(43, 0)


def test_deal_seeds_are_spread_out() -> None:
    """A weak derivation would collide; 500 distinct pairs must give 500 seeds."""
    seeds = {
        derive_deal_seed(match_seed, deal_index)
        for match_seed in range(50)
        for deal_index in range(10)
    }
    assert len(seeds) == 500
    assert all(0 <= seed < 2**63 for seed in seeds)


# --- dealing (I1) -----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 2024])
def test_deal_gives_four_disjoint_thirteen_card_hands(seed: int) -> None:
    hands = deal_hands(np.random.default_rng(seed))

    assert len(hands) == NUM_SEATS
    assert all(hand_size(hand) == CARDS_PER_HAND for hand in hands)

    union_of_hands = 0
    for hand in hands:
        assert union_of_hands & hand == 0  # disjoint
        union_of_hands |= hand
    assert union_of_hands == FULL_DECK_MASK  # and together the whole deck


def test_dealing_is_reproducible_from_the_same_seed() -> None:
    assert deal_hands(np.random.default_rng(7)) == deal_hands(np.random.default_rng(7))


def test_different_seeds_give_different_deals() -> None:
    assert deal_hands(np.random.default_rng(7)) != deal_hands(np.random.default_rng(8))


def test_dealing_uses_only_the_supplied_generator() -> None:
    """Two generators seeded alike must deal alike regardless of global state."""
    np.random.seed(123)  # global state that must be ignored
    first_deal = deal_hands(np.random.default_rng(99))
    np.random.seed(456)
    second_deal = deal_hands(np.random.default_rng(99))
    assert first_deal == second_deal


def test_repeated_deals_from_one_generator_differ() -> None:
    rng = np.random.default_rng(5)
    assert deal_hands(rng) != deal_hands(rng)


def test_every_card_is_dealt_to_someone_across_many_deals() -> None:
    rng = np.random.default_rng(3)
    seen_cards: set[int] = set()
    for _ in range(20):
        for hand in deal_hands(rng):
            seen_cards.update(cards_in_hand(hand))
    assert seen_cards == set(ALL_CARDS)
