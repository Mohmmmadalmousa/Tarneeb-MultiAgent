"""Hand-written trick-winner table (invariant I4).

Every expected winner below is derived from docs/RULES.md C18:

- the highest trump played wins;
- if no trump was played, the highest card of the led suit wins;
- a card of any other suit can never win.

Suits are 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades. Ranks are 0 = two
... 12 = ace. Tricks are listed as ``(seat, card)`` in play order.
"""

from __future__ import annotations

import itertools

import pytest

from tarneeb.core.cards import NUM_RANKS, NUM_SUITS, card_id
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.rules import RulesConfig

CLUBS, DIAMONDS, HEARTS, SPADES = 0, 1, 2, 3
TWO, FIVE, NINE, JACK, QUEEN, KING, ACE = 0, 3, 7, 9, 10, 11, 12


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


# (description, trick, trump_suit, expected_winning_seat)
TRICK_WINNER_CASES = [
    (
        "all follow the led suit; highest rank wins",
        [
            (0, card_id(CLUBS, TWO)),
            (1, card_id(CLUBS, NINE)),
            (2, card_id(CLUBS, FIVE)),
            (3, card_id(CLUBS, JACK)),
        ],
        SPADES,
        3,
    ),
    (
        "leader holds the highest card of the led suit",
        [
            (2, card_id(HEARTS, ACE)),
            (3, card_id(HEARTS, KING)),
            (0, card_id(HEARTS, QUEEN)),
            (1, card_id(HEARTS, JACK)),
        ],
        SPADES,
        2,
    ),
    (
        "one trump beats every card of the led suit",
        [
            (0, card_id(CLUBS, ACE)),
            (1, card_id(CLUBS, KING)),
            (2, card_id(SPADES, TWO)),
            (3, card_id(CLUBS, QUEEN)),
        ],
        SPADES,
        2,
    ),
    (
        "highest trump wins when several are played",
        [
            (0, card_id(CLUBS, ACE)),
            (1, card_id(SPADES, FIVE)),
            (2, card_id(SPADES, NINE)),
            (3, card_id(SPADES, TWO)),
        ],
        SPADES,
        2,
    ),
    (
        "a discard of a third suit never wins",
        [
            (1, card_id(DIAMONDS, NINE)),
            (2, card_id(HEARTS, ACE)),
            (3, card_id(CLUBS, ACE)),
            (0, card_id(DIAMONDS, TWO)),
        ],
        SPADES,
        1,
    ),
    (
        "the led suit is trump; highest trump still wins",
        [
            (0, card_id(SPADES, TWO)),
            (1, card_id(SPADES, ACE)),
            (2, card_id(SPADES, KING)),
            (3, card_id(SPADES, FIVE)),
        ],
        SPADES,
        1,
    ),
    (
        "a later trump overtrumps an earlier one",
        [
            (3, card_id(HEARTS, KING)),
            (0, card_id(CLUBS, TWO)),
            (1, card_id(CLUBS, ACE)),
            (2, card_id(HEARTS, ACE)),
        ],
        CLUBS,
        1,
    ),
    (
        "a low trump beats an ace of a plain suit",
        [
            (0, card_id(HEARTS, ACE)),
            (1, card_id(DIAMONDS, TWO)),
            (2, card_id(HEARTS, KING)),
            (3, card_id(HEARTS, QUEEN)),
        ],
        DIAMONDS,
        1,
    ),
    (
        "the last player wins with the top trump",
        [
            (0, card_id(CLUBS, KING)),
            (1, card_id(HEARTS, TWO)),
            (2, card_id(HEARTS, JACK)),
            (3, card_id(HEARTS, ACE)),
        ],
        HEARTS,
        3,
    ),
    (
        "no trump played and two discards",
        [
            (2, card_id(DIAMONDS, FIVE)),
            (3, card_id(CLUBS, ACE)),
            (0, card_id(DIAMONDS, QUEEN)),
            (1, card_id(HEARTS, ACE)),
        ],
        SPADES,
        0,
    ),
]


@pytest.mark.parametrize(
    ("description", "trick", "trump_suit", "expected_winning_seat"),
    TRICK_WINNER_CASES,
    ids=[case[0] for case in TRICK_WINNER_CASES],
)
def test_trick_winner_table(
    engine: TarneebEngine,
    description: str,
    trick: list[tuple[int, int]],
    trump_suit: int,
    expected_winning_seat: int,
) -> None:
    assert engine.trick_winner(trick, trump_suit) == expected_winning_seat, description


# --- properties that must hold for every trump suit -------------------------


@pytest.mark.parametrize("trump_suit", range(NUM_SUITS))
def test_the_only_trump_always_wins(engine: TarneebEngine, trump_suit: int) -> None:
    plain_suit = (trump_suit + 1) % NUM_SUITS
    other_plain_suit = (trump_suit + 2) % NUM_SUITS
    for trump_position in range(4):
        trick = [
            (seat, card_id(plain_suit if seat % 2 == 0 else other_plain_suit, ACE - seat))
            for seat in range(4)
        ]
        trick[trump_position] = (trump_position, card_id(trump_suit, TWO))
        assert engine.trick_winner(trick, trump_suit) == trump_position


@pytest.mark.parametrize("trump_suit", range(NUM_SUITS))
def test_highest_led_suit_card_wins_when_no_trump_appears(
    engine: TarneebEngine, trump_suit: int
) -> None:
    led_suit = (trump_suit + 1) % NUM_SUITS
    ranks = [FIVE, ACE, TWO, NINE]
    trick = [(seat, card_id(led_suit, ranks[seat])) for seat in range(4)]
    assert engine.trick_winner(trick, trump_suit) == 1  # the ace


@pytest.mark.parametrize("trump_suit", range(NUM_SUITS))
def test_winner_is_independent_of_seat_rotation(engine: TarneebEngine, trump_suit: int) -> None:
    """Rotating who sits where must not change which card wins."""
    led_suit = (trump_suit + 1) % NUM_SUITS
    cards = [
        card_id(led_suit, FIVE),
        card_id(led_suit, ACE),
        card_id(trump_suit, TWO),
        card_id(led_suit, KING),
    ]
    for rotation in range(4):
        trick = [((seat + rotation) % 4, cards[seat]) for seat in range(4)]
        winning_seat = engine.trick_winner(trick, trump_suit)
        assert winning_seat == (2 + rotation) % 4  # the trump, wherever it sits


def test_exhaustive_single_suit_tricks_are_won_by_the_highest_rank(engine: TarneebEngine) -> None:
    """Every four-card same-suit combination, with trump elsewhere."""
    led_suit, trump_suit = CLUBS, SPADES
    for ranks in itertools.combinations(range(NUM_RANKS), 4):
        trick = [(seat, card_id(led_suit, ranks[seat])) for seat in range(4)]
        assert engine.trick_winner(trick, trump_suit) == 3  # combinations are ascending


def test_exhaustive_trump_versus_plain_for_every_pair_of_suits(engine: TarneebEngine) -> None:
    for trump_suit, led_suit in itertools.permutations(range(NUM_SUITS), 2):
        trick = [
            (0, card_id(led_suit, ACE)),
            (1, card_id(led_suit, KING)),
            (2, card_id(trump_suit, TWO)),
            (3, card_id(led_suit, QUEEN)),
        ]
        assert engine.trick_winner(trick, trump_suit) == 2


# --- input validation -------------------------------------------------------


@pytest.mark.parametrize("card_count", [0, 1, 3, 5])
def test_trick_winner_requires_exactly_four_cards(engine: TarneebEngine, card_count: int) -> None:
    trick = [(seat, card_id(CLUBS, seat)) for seat in range(card_count)]
    with pytest.raises(ValueError, match="a trick has 4 cards"):
        engine.trick_winner(trick, SPADES)
