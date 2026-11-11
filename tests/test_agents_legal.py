"""Agent legality and reproducibility stress.

Every agent must return a legal action at every reachable decision, never
action 64, and never repair an illegal choice silently. Deterministic agents
must reproduce their actions exactly; the random agent must reproduce its
stochastic sequence from its seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.agents.base import Agent, checked_action, first_legal_action, legal_action_ids
from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.core.actions import NO_TRUMP_ACTION, NUM_ACTIONS, is_card_action
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.public_view import PublicViewBuilder

DEFAULT_DEAL_COUNT = 120
LARGE_DEAL_COUNT = 3000


def make_agents() -> list[Agent]:
    return [RandomAgent(seed=7), BasicHeuristicAgent(), StrongHeuristicAgent()]


def play_deals(agent: Agent, deal_count: int, base_seed: int = 0) -> dict:
    """Drive complete deals with one agent in all four seats, checking legality."""
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    statistics = {
        "decisions": 0,
        "deals": 0,
        "illegal_actions": 0,
        "no_trump_selections": 0,
        "phases": {Phase.BID: 0, Phase.DECLARE: 0, Phase.PLAY: 0},
        "actions": [],
    }

    for deal_number in range(deal_count):
        agent.reset()
        state = engine.deal(
            engine.new_deal(deal_number % NUM_SEATS, match_seed=base_seed + deal_number)
        )
        guard = 0
        while state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            guard += 1
            assert guard < 300, "deal did not terminate"

            acting_seat = state.current_player_seat
            assert acting_seat is not None
            view = views.build(state, acting_seat)

            assert view.legal_action_mask.shape == (NUM_ACTIONS,)
            assert view.legal_action_mask.dtype == np.bool_
            assert view.legal_action_mask.any()

            statistics["phases"][state.phase] += 1
            selected_action = agent.act(view)
            statistics["actions"].append(selected_action)

            if not view.legal_action_mask[selected_action]:
                statistics["illegal_actions"] += 1
            if selected_action == NO_TRUMP_ACTION:
                statistics["no_trump_selections"] += 1

            engine.step(state, selected_action)
            statistics["decisions"] += 1

            if state.phase is Phase.DEAL:  # all-pass redeal
                engine.deal(state)

        statistics["deals"] += 1

    return statistics


# --- legality ---------------------------------------------------------------


@pytest.mark.parametrize("agent", make_agents(), ids=lambda agent: agent.name)
def test_agent_never_returns_an_illegal_action(agent: Agent) -> None:
    statistics = play_deals(agent, DEFAULT_DEAL_COUNT, base_seed=1000)

    assert statistics["illegal_actions"] == 0
    assert statistics["no_trump_selections"] == 0
    assert statistics["deals"] == DEFAULT_DEAL_COUNT
    assert statistics["decisions"] > 5000
    for phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
        assert statistics["phases"][phase] > 0


@pytest.mark.slow
@pytest.mark.parametrize("agent", make_agents(), ids=lambda agent: agent.name)
def test_agent_legality_over_many_deals(agent: Agent) -> None:
    statistics = play_deals(agent, LARGE_DEAL_COUNT, base_seed=50_000)
    assert statistics["illegal_actions"] == 0
    assert statistics["no_trump_selections"] == 0
    assert statistics["decisions"] > 100_000


@pytest.mark.parametrize("agent", make_agents(), ids=lambda agent: agent.name)
def test_agents_return_ids_inside_the_action_space(agent: Agent) -> None:
    statistics = play_deals(agent, 20, base_seed=2000)
    assert all(0 <= action < NUM_ACTIONS for action in statistics["actions"])


# --- determinism ------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_factory",
    [lambda: BasicHeuristicAgent(), lambda: StrongHeuristicAgent()],
    ids=["basic", "strong"],
)
def test_deterministic_agents_reproduce_their_actions(agent_factory) -> None:
    first_run = play_deals(agent_factory(), 25, base_seed=3000)
    second_run = play_deals(agent_factory(), 25, base_seed=3000)
    assert first_run["actions"] == second_run["actions"]


def test_random_agent_reproduces_its_sequence_from_its_seed() -> None:
    first_run = play_deals(RandomAgent(seed=99), 25, base_seed=4000)
    second_run = play_deals(RandomAgent(seed=99), 25, base_seed=4000)
    assert first_run["actions"] == second_run["actions"]


def test_random_agents_with_different_seeds_diverge() -> None:
    first_run = play_deals(RandomAgent(seed=1), 25, base_seed=4000)
    second_run = play_deals(RandomAgent(seed=2), 25, base_seed=4000)
    assert first_run["actions"] != second_run["actions"]


def test_random_agent_restart_rewinds_the_stream() -> None:
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    agent = RandomAgent(seed=5)

    state = engine.deal(engine.new_deal(0, match_seed=1))
    view = views.build(state, state.current_player_seat)
    first_choices = [agent.act(view) for _ in range(10)]

    agent.restart()
    assert [agent.act(view) for _ in range(10)] == first_choices


def test_random_agent_reset_continues_the_stream_by_default() -> None:
    """Independent behaviour across deals; see the docstring in random_agent.py."""
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    agent = RandomAgent(seed=5)

    state = engine.deal(engine.new_deal(0, match_seed=1))
    view = views.build(state, state.current_player_seat)
    first_choices = [agent.act(view) for _ in range(10)]

    agent.reset()
    assert [agent.act(view) for _ in range(10)] != first_choices


def test_random_agent_can_opt_into_reseeding_on_reset() -> None:
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    agent = RandomAgent(seed=5, reseed_on_reset=True)

    state = engine.deal(engine.new_deal(0, match_seed=1))
    view = views.build(state, state.current_player_seat)
    first_choices = [agent.act(view) for _ in range(10)]

    agent.reset()
    assert [agent.act(view) for _ in range(10)] == first_choices


def test_random_agent_does_not_use_global_numpy_state() -> None:
    np.random.seed(1)
    first_run = play_deals(RandomAgent(seed=11), 10, base_seed=5000)
    np.random.seed(2)
    for _ in range(100):
        np.random.random()
    second_run = play_deals(RandomAgent(seed=11), 10, base_seed=5000)
    assert first_run["actions"] == second_run["actions"]


# --- the shared helpers -----------------------------------------------------


def test_legal_action_ids_rejects_a_bad_mask() -> None:
    with pytest.raises(ValueError, match="must be"):
        legal_action_ids(np.zeros(10, dtype=bool))
    with pytest.raises(ValueError, match="must be bool"):
        legal_action_ids(np.zeros(NUM_ACTIONS, dtype=np.float32))


def test_legal_action_ids_rejects_an_empty_mask() -> None:
    with pytest.raises(ValueError, match="no legal action"):
        legal_action_ids(np.zeros(NUM_ACTIONS, dtype=bool))


def test_legal_action_ids_rejects_a_mask_offering_no_trump() -> None:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[NO_TRUMP_ACTION] = True
    with pytest.raises(ValueError, match="action 64 is reserved"):
        legal_action_ids(mask)


def test_checked_action_raises_rather_than_repairing() -> None:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[5] = True
    assert checked_action(5, mask, "test") == 5
    with pytest.raises(ValueError, match="illegal action"):
        checked_action(6, mask, "test")
    with pytest.raises(ValueError, match=r"outside 0\.\.64"):
        checked_action(99, mask, "test")


def test_checked_action_rejects_no_trump() -> None:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[NO_TRUMP_ACTION] = True
    with pytest.raises(ValueError, match="reserved no-trump"):
        checked_action(NO_TRUMP_ACTION, mask, "test")


def test_first_legal_action_is_the_lowest_legal_id() -> None:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[[9, 3, 40]] = True
    assert first_legal_action(mask) == 3


def test_random_agent_fails_clearly_on_an_all_false_mask() -> None:
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    state = engine.deal(engine.new_deal(0, match_seed=1))
    view = views.build(state, (state.current_player_seat + 1) % NUM_SEATS)  # not on turn
    assert not view.legal_action_mask.any()
    with pytest.raises(ValueError, match="no legal action"):
        RandomAgent(seed=1).act(view)


# --- heuristic sanity -------------------------------------------------------


def test_heuristics_play_only_cards_during_play() -> None:
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    for agent in (BasicHeuristicAgent(), StrongHeuristicAgent()):
        state = engine.deal(engine.new_deal(0, match_seed=42))
        while state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            view = views.build(state, state.current_player_seat)
            selected_action = agent.act(view)
            if state.phase is Phase.PLAY:
                assert is_card_action(selected_action)
            engine.step(state, selected_action)
            if state.phase is Phase.DEAL:
                engine.deal(state)


def test_basic_agent_only_ever_opens_seven() -> None:
    """The basic heuristic bids 7 or passes; it never raises."""
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    agent = BasicHeuristicAgent()
    observed_bids = set()

    for deal_number in range(80):
        state = engine.deal(engine.new_deal(0, match_seed=deal_number))
        while state.phase is Phase.BID:
            view = views.build(state, state.current_player_seat)
            selected_action = agent.act(view)
            if 52 <= selected_action <= 58:
                observed_bids.add(selected_action - 52 + 7)
            engine.step(state, selected_action)
            if state.phase is Phase.DEAL:
                break

    assert observed_bids <= {7}
