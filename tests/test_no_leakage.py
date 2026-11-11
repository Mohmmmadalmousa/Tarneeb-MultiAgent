"""Hidden-information leakage gate (invariant I7).

The actor observation must be a function of what the observer may legally see.
The test therefore takes a real reachable state, redistributes the other three
players' still-hidden cards among them in every way it can while keeping every
public fact identical, and requires the actor vector to be **byte-identical**.

Every such test also asserts that the critic vector *did* change. Without that,
a builder that ignored the mutation entirely would pass vacuously.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tarneeb.core.actions import PASS_ACTION, action_for_bid, action_for_trump_suit
from tarneeb.core.cards import (
    NUM_CARDS,
    NUM_SEATS,
    card_name,
    cards_in_hand,
    hand_from_cards,
    hand_size,
)
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.spaces import CRITIC_FEATURE_SLICES, D_STATIC


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


@pytest.fixture
def builder(engine: TarneebEngine) -> ObservationBuilder:
    return ObservationBuilder(engine)


def state_after_auction(engine: TarneebEngine, match_seed: int = 42) -> TarneebState:
    state = engine.deal(engine.new_deal(0, match_seed=match_seed))
    engine.step(state, action_for_bid(9))
    for _ in range(NUM_SEATS - 1):
        engine.step(state, PASS_ACTION)
    engine.step(state, action_for_trump_suit(2))
    return state


def advance_plays(engine: TarneebEngine, state: TarneebState, count: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(count):
        if state.phase is not Phase.PLAY:
            return
        engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))


def redistribute_hidden_cards(
    state: TarneebState, observer_seat: int, rng: np.random.Generator
) -> TarneebState:
    """Return a copy whose other three hands are a reshuffle of the same cards.

    Hand sizes are preserved, the observer's hand is untouched, and every public
    field (events, tricks, voids, scores) is left exactly as it was. Only the
    private assignment of unseen cards changes.
    """
    mutated_state = state.clone()
    other_seats = [seat for seat in range(NUM_SEATS) if seat != observer_seat]

    hidden_cards: list[int] = []
    for seat in other_seats:
        hidden_cards.extend(cards_in_hand(state.hands[seat]))

    shuffled = list(rng.permutation(hidden_cards))
    cursor = 0
    for seat in other_seats:
        seat_size = hand_size(state.hands[seat])
        mutated_state.hands[seat] = hand_from_cards(shuffled[cursor : cursor + seat_size])
        cursor += seat_size

    assert cursor == len(hidden_cards)
    return mutated_state


def assert_actor_invariant_and_critic_sensitive(
    builder: ObservationBuilder,
    original_state: TarneebState,
    mutated_state: TarneebState,
    observer_seat: int,
) -> None:
    original_actor = builder.static_actor(original_state, observer_seat)
    mutated_actor = builder.static_actor(mutated_state, observer_seat)
    assert original_actor.tobytes() == mutated_actor.tobytes(), (
        f"actor observation for seat {observer_seat} changed when only hidden cards moved"
    )

    original_critic = builder.static_critic(original_state, observer_seat)
    mutated_critic = builder.static_critic(mutated_state, observer_seat)
    if original_state.hands != mutated_state.hands:
        assert original_critic.tobytes() != mutated_critic.tobytes(), (
            "the critic ignored a hidden-hand change, so the actor test proves nothing"
        )
        assert np.array_equal(original_critic[:D_STATIC], mutated_critic[:D_STATIC])


# --- the core invariance gate -----------------------------------------------


@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
@pytest.mark.parametrize("cards_played", [0, 1, 3, 7, 20, 40])
def test_actor_is_invariant_to_hidden_hand_permutation(
    engine: TarneebEngine,
    builder: ObservationBuilder,
    observer_seat: int,
    cards_played: int,
) -> None:
    """Every seat, at six points in the deal, under a random redistribution."""
    state = state_after_auction(engine)
    advance_plays(engine, state, cards_played, seed=cards_played)

    rng = np.random.default_rng(1000 + cards_played)
    for _ in range(5):
        mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
        assert_actor_invariant_and_critic_sensitive(builder, state, mutated_state, observer_seat)


@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
def test_actor_is_invariant_during_the_auction(
    engine: TarneebEngine, builder: ObservationBuilder, observer_seat: int
) -> None:
    """The auction phase has the most hidden information of all."""
    state = engine.deal(engine.new_deal(0, match_seed=11))
    rng = np.random.default_rng(7)
    for _ in range(4):
        mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
        assert_actor_invariant_and_critic_sensitive(builder, state, mutated_state, observer_seat)

    engine.step(state, action_for_bid(11))
    for _ in range(3):
        mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
        assert_actor_invariant_and_critic_sensitive(builder, state, mutated_state, observer_seat)


@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
def test_actor_is_invariant_under_every_pairwise_hand_swap(
    engine: TarneebEngine, builder: ObservationBuilder, observer_seat: int
) -> None:
    """Swap two opponents' entire hands: maximally different, publicly identical."""
    state = state_after_auction(engine, match_seed=5)
    advance_plays(engine, state, 4, seed=2)

    other_seats = [seat for seat in range(NUM_SEATS) if seat != observer_seat]
    for first_seat, second_seat in itertools.combinations(other_seats, 2):
        if hand_size(state.hands[first_seat]) != hand_size(state.hands[second_seat]):
            continue
        mutated_state = state.clone()
        mutated_state.hands[first_seat] = state.hands[second_seat]
        mutated_state.hands[second_seat] = state.hands[first_seat]
        assert_actor_invariant_and_critic_sensitive(builder, state, mutated_state, observer_seat)


def test_actor_is_invariant_across_many_random_states(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    """A broader sweep: many deals, many decision points, all four observers."""
    checked_comparisons = 0
    for match_seed in range(12):
        state = state_after_auction(engine, match_seed=match_seed)
        rng = np.random.default_rng(match_seed)
        while state.phase is Phase.PLAY:
            for observer_seat in range(NUM_SEATS):
                mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
                assert_actor_invariant_and_critic_sensitive(
                    builder, state, mutated_state, observer_seat
                )
                checked_comparisons += 1
            engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))
    assert checked_comparisons > 2000


# --- unseen-card semantics --------------------------------------------------


def test_unseen_mask_means_not_mine_and_not_yet_public(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_after_auction(engine)
    advance_plays(engine, state, 9, seed=3)

    publicly_played = {card for trick in state.completed_tricks for _seat, card in trick}
    publicly_played |= {card for _seat, card in state.current_trick}

    for observer_seat in range(NUM_SEATS):
        observation = builder.static_actor(state, observer_seat)
        unseen_block = observation[CRITIC_FEATURE_SLICES["unseen_card_mask"]]
        own_cards = set(cards_in_hand(state.hands[observer_seat]))
        expected_unseen = {
            card
            for card in range(NUM_CARDS)
            if card not in own_cards and card not in publicly_played
        }
        assert set(np.flatnonzero(unseen_block).tolist()) == expected_unseen


def test_unseen_mask_does_not_identify_the_holder(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    """The mask must say 'somewhere among the others', never 'seat X holds it'."""
    state = state_after_auction(engine, match_seed=21)
    advance_plays(engine, state, 6, seed=5)
    unseen_slice = CRITIC_FEATURE_SLICES["unseen_card_mask"]

    rng = np.random.default_rng(9)
    for observer_seat in range(NUM_SEATS):
        original_unseen = builder.static_actor(state, observer_seat)[unseen_slice]
        for _ in range(6):
            mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
            mutated_unseen = builder.static_actor(mutated_state, observer_seat)[unseen_slice]
            assert np.array_equal(original_unseen, mutated_unseen)


def test_unseen_mask_excludes_the_observers_own_cards(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_after_auction(engine)
    unseen_slice = CRITIC_FEATURE_SLICES["unseen_card_mask"]
    for observer_seat in range(NUM_SEATS):
        observation = builder.static_actor(state, observer_seat)
        unseen_cards = set(np.flatnonzero(observation[unseen_slice]).tolist())
        own_cards = set(cards_in_hand(state.hands[observer_seat]))
        assert not (unseen_cards & own_cards)
        assert len(unseen_cards) == NUM_CARDS - len(own_cards)  # nothing public yet


# --- the critic is allowed to see, and does ---------------------------------


def test_critic_block_contains_the_other_three_hands(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_after_auction(engine)
    hidden_slice = CRITIC_FEATURE_SLICES["hidden_hands_by_relative_seat"]

    for observer_seat in range(NUM_SEATS):
        critic_observation = builder.static_critic(state, observer_seat)
        hidden_block = critic_observation[hidden_slice]
        for relative in (1, 2, 3):
            seat = (relative + observer_seat) % NUM_SEATS
            block = hidden_block[(relative - 1) * NUM_CARDS : relative * NUM_CARDS]
            assert set(np.flatnonzero(block).tolist()) == set(cards_in_hand(state.hands[seat]))


def test_critic_never_duplicates_the_observers_own_hand_in_the_hidden_block(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    state = state_after_auction(engine)
    hidden_slice = CRITIC_FEATURE_SLICES["hidden_hands_by_relative_seat"]
    critic_observation = builder.static_critic(state, 0)
    assert critic_observation[hidden_slice].sum() == 3 * 13


# --- the UI view -------------------------------------------------------------


def test_ui_view_never_exposes_another_hand(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    """Only the observer's own cards may appear as card names in the view."""
    import json

    state = state_after_auction(engine)
    advance_plays(engine, state, 5, seed=6)

    for observer_seat in range(NUM_SEATS):
        view = builder.ui_view(state, observer_seat)
        own_card_names = {card_name(card) for card in cards_in_hand(state.hands[observer_seat])}
        assert set(view["own_hand"]) == own_card_names

        publicly_played_names = {
            card_name(card) for trick in state.completed_tricks for _seat, card in trick
        } | {card_name(card) for _seat, card in state.current_trick}

        # Every card name anywhere in the serialised view must be either the
        # observer's own or already public.
        serialized = json.dumps(view)
        for other_seat in range(NUM_SEATS):
            if other_seat == observer_seat:
                continue
            for card in cards_in_hand(state.hands[other_seat]):
                label = card_name(card)
                if label in own_card_names or label in publicly_played_names:
                    continue  # cannot occur: hands are disjoint and these are unplayed
                assert f'"{label}"' not in serialized, (
                    f"ui_view for seat {observer_seat} leaked {label} held by seat {other_seat}"
                )


def test_ui_view_is_json_safe(engine: TarneebEngine, builder: ObservationBuilder) -> None:
    import json

    state = state_after_auction(engine)
    advance_plays(engine, state, 8, seed=7)
    for observer_seat in range(NUM_SEATS):
        encoded = json.dumps(builder.ui_view(state, observer_seat))
        assert json.loads(encoded)["observer_seat"] == observer_seat


def test_ui_view_hidden_hands_cannot_be_recovered(
    engine: TarneebEngine, builder: ObservationBuilder
) -> None:
    """Permuting hidden hands must not change the observer's UI view at all."""
    import json

    state = state_after_auction(engine, match_seed=33)
    advance_plays(engine, state, 10, seed=8)
    rng = np.random.default_rng(12)

    for observer_seat in range(NUM_SEATS):
        original_view = json.dumps(builder.ui_view(state, observer_seat), sort_keys=True)
        for _ in range(5):
            mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
            mutated_view = json.dumps(builder.ui_view(mutated_state, observer_seat), sort_keys=True)
            assert original_view == mutated_view
