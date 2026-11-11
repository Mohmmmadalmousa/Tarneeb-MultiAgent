"""Hidden-information hard gate for the heuristic agents.

The heuristics may infer anything they like from public history -- that is skill,
not leakage -- but they must never read private state. The test constructs paired
states in which the observer's own hand and every public fact are identical while
the other three hidden hands are consistently redistributed, and requires the
deterministic heuristics to choose the same action.

It also checks the structural side: the actor-safe view itself is invariant, no
heuristic accepts a state or a critic vector, and no agent module imports the
critic path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.core.actions import PASS_ACTION, action_for_bid, action_for_trump_suit
from tarneeb.core.cards import NUM_SEATS, cards_in_hand, hand_from_cards, hand_size
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicView, PublicViewBuilder

AGENT_MODULE_PATHS = [
    Path("tarneeb/agents/base.py"),
    Path("tarneeb/agents/random_agent.py"),
    Path("tarneeb/agents/heuristic_basic.py"),
    Path("tarneeb/agents/heuristic_strong.py"),
]


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


@pytest.fixture
def views(engine: TarneebEngine) -> PublicViewBuilder:
    return PublicViewBuilder(engine)


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
    """Reshuffle the other three hands, preserving sizes and every public fact."""
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
    return mutated_state


def view_fingerprint(view: PublicView) -> tuple:
    """Everything the view exposes, in a comparable form."""
    return (
        view.observer_seat,
        view.phase,
        view.legal_action_mask.tobytes(),
        view.own_hand,
        view.dealer_seat,
        view.current_high_bid,
        view.current_high_bidder,
        view.seats_that_passed,
        view.declarer_seat,
        view.trump_suit,
        view.leader_seat,
        view.trick_number,
        view.current_trick,
        view.played_cards_by_seat,
        view.void_flags,
        view.tricks_won_by_team,
        view.match_score_by_team,
        view.events,
        tuple(sorted(view.unseen_cards)),
    )


# --- the view itself is actor-safe ------------------------------------------


@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
@pytest.mark.parametrize("cards_played", [0, 1, 5, 17, 35])
def test_public_view_is_invariant_to_hidden_redistribution(
    engine: TarneebEngine, views: PublicViewBuilder, observer_seat: int, cards_played: int
) -> None:
    state = state_after_auction(engine)
    advance_plays(engine, state, cards_played, seed=cards_played)

    original_fingerprint = view_fingerprint(views.build(state, observer_seat))
    rng = np.random.default_rng(1000 + cards_played)
    for _ in range(6):
        mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
        assert view_fingerprint(views.build(mutated_state, observer_seat)) == original_fingerprint


def test_public_view_never_contains_another_hand(
    engine: TarneebEngine, views: PublicViewBuilder
) -> None:
    state = state_after_auction(engine)
    for observer_seat in range(NUM_SEATS):
        view = views.build(state, observer_seat)
        assert set(view.own_hand) == set(cards_in_hand(state.hands[observer_seat]))

        others_cards: set[int] = set()
        for seat in range(NUM_SEATS):
            if seat != observer_seat:
                others_cards |= set(cards_in_hand(state.hands[seat]))
        assert not set(view.own_hand) & others_cards
        # The unseen set contains those cards but attributes them to nobody.
        assert others_cards <= view.unseen_cards


def test_unseen_set_never_identifies_a_holder(
    engine: TarneebEngine, views: PublicViewBuilder
) -> None:
    state = state_after_auction(engine, match_seed=21)
    advance_plays(engine, state, 9, seed=5)
    rng = np.random.default_rng(3)

    for observer_seat in range(NUM_SEATS):
        original_unseen = views.build(state, observer_seat).unseen_cards
        for _ in range(5):
            mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
            assert views.build(mutated_state, observer_seat).unseen_cards == original_unseen


# --- the heuristics act on public information only --------------------------


@pytest.mark.parametrize(
    "agent_factory",
    [lambda: BasicHeuristicAgent(), lambda: StrongHeuristicAgent()],
    ids=["basic", "strong"],
)
@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
@pytest.mark.parametrize(
    ("phase_label", "cards_played"),
    [("early_play", 2), ("mid_play", 20), ("late_play", 44)],
)
def test_heuristic_action_is_invariant_during_play(
    engine: TarneebEngine,
    views: PublicViewBuilder,
    agent_factory,
    observer_seat: int,
    phase_label: str,
    cards_played: int,
) -> None:
    state = state_after_auction(engine)
    advance_plays(engine, state, cards_played, seed=cards_played)
    if state.phase is not Phase.PLAY:
        pytest.skip("deal ended before this point")

    # Move the turn to the observer so it actually has a legal mask.
    while state.current_player_seat != observer_seat and state.phase is Phase.PLAY:
        engine.step(state, int(np.flatnonzero(engine.legal_actions(state))[0]))
    if state.phase is not Phase.PLAY:
        pytest.skip("deal ended while rotating to the observer")

    agent = agent_factory()
    baseline_action = agent.act(views.build(state, observer_seat))

    rng = np.random.default_rng(500 + cards_played)
    for _ in range(8):
        mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
        mutated_action = agent_factory().act(views.build(mutated_state, observer_seat))
        assert mutated_action == baseline_action, (
            f"{phase_label}: {agent.name} changed its action when only hidden cards moved"
        )


@pytest.mark.parametrize(
    "agent_factory",
    [lambda: BasicHeuristicAgent(), lambda: StrongHeuristicAgent()],
    ids=["basic", "strong"],
)
@pytest.mark.parametrize("observer_seat", range(NUM_SEATS))
def test_heuristic_bidding_is_invariant(
    engine: TarneebEngine, views: PublicViewBuilder, agent_factory, observer_seat: int
) -> None:
    state = engine.deal(engine.new_deal(0, match_seed=11))
    while state.current_player_seat != observer_seat:
        engine.step(state, PASS_ACTION)

    agent = agent_factory()
    baseline_action = agent.act(views.build(state, observer_seat))

    rng = np.random.default_rng(7)
    for _ in range(8):
        mutated_state = redistribute_hidden_cards(state, observer_seat, rng)
        assert agent_factory().act(views.build(mutated_state, observer_seat)) == baseline_action


@pytest.mark.parametrize(
    "agent_factory",
    [lambda: BasicHeuristicAgent(), lambda: StrongHeuristicAgent()],
    ids=["basic", "strong"],
)
def test_heuristic_trump_choice_is_invariant(
    engine: TarneebEngine, views: PublicViewBuilder, agent_factory
) -> None:
    state = engine.deal(engine.new_deal(0, match_seed=5))
    engine.step(state, action_for_bid(8))
    for _ in range(NUM_SEATS - 1):
        engine.step(state, PASS_ACTION)
    assert state.phase is Phase.DECLARE

    declarer_seat = state.declarer_seat
    assert declarer_seat is not None
    baseline_action = agent_factory().act(views.build(state, declarer_seat))

    rng = np.random.default_rng(9)
    for _ in range(8):
        mutated_state = redistribute_hidden_cards(state, declarer_seat, rng)
        assert agent_factory().act(views.build(mutated_state, declarer_seat)) == baseline_action


def test_heuristics_are_invariant_across_a_whole_deal(
    engine: TarneebEngine, views: PublicViewBuilder
) -> None:
    """A sweep across every decision of several deals, for both heuristics."""
    checked_decisions = 0
    for match_seed in range(6):
        state = state_after_auction(engine, match_seed=match_seed)
        rng = np.random.default_rng(match_seed)
        while state.phase is Phase.PLAY:
            acting_seat = state.current_player_seat
            assert acting_seat is not None
            for agent_factory in (BasicHeuristicAgent, StrongHeuristicAgent):
                baseline = agent_factory().act(views.build(state, acting_seat))
                for _ in range(3):
                    mutated_state = redistribute_hidden_cards(state, acting_seat, rng)
                    assert agent_factory().act(views.build(mutated_state, acting_seat)) == baseline
                checked_decisions += 1
            engine.step(state, int(rng.choice(np.flatnonzero(engine.legal_actions(state)))))
    assert checked_decisions > 300


# --- structural checks ------------------------------------------------------


def test_no_agent_module_imports_the_critic_path() -> None:
    for module_path in AGENT_MODULE_PATHS:
        source = module_path.read_text(encoding="utf-8")
        assert "static_critic" not in source, f"{module_path} references the critic"
        assert "ObservationBuilder" not in source, f"{module_path} builds observations"
        assert "original_hands" not in source, f"{module_path} references original hands"
        assert "state.hands" not in source, f"{module_path} reads hands directly"


def test_public_view_exposes_no_hands_field() -> None:
    import dataclasses

    field_names = {field.name for field in dataclasses.fields(PublicView)}
    assert "hands" not in field_names
    assert "original_hands" not in field_names
    assert "own_hand" in field_names


def test_agents_accept_only_a_public_view() -> None:
    """An agent's act() signature must not accept a state or a critic vector."""
    import inspect

    for agent in (BasicHeuristicAgent(), StrongHeuristicAgent()):
        signature = inspect.signature(agent.act)
        parameter_names = list(signature.parameters)
        assert parameter_names == ["view"], f"{agent.name} takes {parameter_names}"


def test_critic_observation_is_never_reachable_from_a_view(
    engine: TarneebEngine, views: PublicViewBuilder
) -> None:
    """The critic vector exists but nothing hands it to an agent."""
    state = state_after_auction(engine)
    critic_observation = ObservationBuilder(engine).static_critic(state, 0)
    assert critic_observation.shape == (856,)

    view = views.build(state, 0)
    assert not hasattr(view, "critic_observation")
    assert not hasattr(view, "static_critic")
