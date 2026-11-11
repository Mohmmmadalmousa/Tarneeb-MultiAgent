"""The trained policy playing inside the Phase 5 arena.

Legality, determinism, reproducibility and the information boundary. The agent
must be exactly as constrained as the heuristics it is compared against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.agents.neural_agent import NeuralAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.core.actions import NO_TRUMP_ACTION
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.env.spaces import D_CRITIC, D_STATIC
from tarneeb.evaluation.arena import Arena, build_seat_assignment
from tarneeb.models.checkpoint import save_checkpoint
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.training.ppo import PPOConfig, build_optimizer

LIVE_PHASES = (Phase.BID, Phase.DECLARE, Phase.PLAY)


def build_agent(deterministic: bool = True, seed: int = 0) -> NeuralAgent:
    actor, _critic = build_actor_and_critic(seed=1)
    return NeuralAgent(actor=actor, name="neural", deterministic=deterministic, seed=seed)


def test_every_action_is_legal_over_many_deals() -> None:
    agent = build_agent()
    arena = Arena()
    opponent = RandomAgent(seed=3, name="random")
    for deal_index in range(6):
        outcome = arena.play_deal(
            build_seat_assignment(agent, opponent, policy_a_team=deal_index % 2),
            match_seed=10**9 + deal_index,
            dealer_seat=deal_index % 4,
            deal_index=deal_index,
            verify_replay=True,
        )
        assert outcome.replay_verified is not False
    assert arena.illegal_action_count == 0


def test_the_reserved_no_trump_action_is_never_chosen() -> None:
    agent = build_agent()
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    rng = np.random.default_rng(0)

    for deal_index in range(4):
        state = engine.new_deal(deal_index % 4, deal_index=deal_index, match_seed=555)
        engine.deal(state)
        while state.phase in LIVE_PHASES:
            acting_seat = state.current_player_seat
            assert acting_seat is not None
            view = views.build(state, acting_seat)
            chosen = agent.act(view)
            assert chosen != NO_TRUMP_ACTION
            assert view.legal_action_mask[chosen]
            legal = np.flatnonzero(engine.legal_actions(state))
            engine.step(state, int(rng.choice(legal)))
            if state.phase is Phase.DEAL:
                engine.deal(state)


def test_deterministic_mode_repeats_exactly() -> None:
    first = build_agent(deterministic=True)
    second = build_agent(deterministic=True)
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)

    state = engine.new_deal(0, deal_index=0, match_seed=2024)
    engine.deal(state)
    acting_seat = state.current_player_seat
    assert acting_seat is not None
    view = views.build(state, acting_seat)

    choices = [first.act(view) for _ in range(5)]
    assert len(set(choices)) == 1
    assert second.act(view) == choices[0]


def test_stochastic_mode_is_reproducible_from_its_seed() -> None:
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=2025)
    engine.deal(state)
    acting_seat = state.current_player_seat
    assert acting_seat is not None
    view = views.build(state, acting_seat)

    first = build_agent(deterministic=False, seed=99)
    second = build_agent(deterministic=False, seed=99)
    first_draws = [first.act(view) for _ in range(12)]
    second_draws = [second.act(view) for _ in range(12)]
    assert first_draws == second_draws

    first.restart()
    assert [first.act(view) for _ in range(12)] == first_draws


def test_stochastic_mode_actually_varies() -> None:
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=4242)
    engine.deal(state)
    acting_seat = state.current_player_seat
    assert acting_seat is not None
    view = views.build(state, acting_seat)

    agent = build_agent(deterministic=False, seed=5)
    draws = {agent.act(view) for _ in range(40)}
    assert len(draws) > 1


def test_what_reaches_the_network_is_the_actor_observation_and_nothing_wider() -> None:
    """The information boundary, checked on the tensors themselves."""
    agent = build_agent()
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=7)
    engine.deal(state)
    acting_seat = state.current_player_seat
    assert acting_seat is not None

    observation, history_ids, history_masks = agent.encode(views.build(state, acting_seat))
    assert observation.shape == (1, D_STATIC)
    assert observation.shape[-1] != D_CRITIC
    assert history_ids.shape == (1, 80, 7)
    assert history_masks.shape == (1, 80)
    assert agent.actor.spec.actor_observation_dim == D_STATIC


def test_the_agent_holds_no_critic() -> None:
    agent = build_agent()
    assert not any("critic" in name.lower() for name in vars(agent) if not name.startswith("__"))
    assert not hasattr(agent, "critic")


def test_loading_from_a_checkpoint_never_builds_a_critic(tmp_path: Path) -> None:
    actor, critic = build_actor_and_critic(seed=1)
    path = save_checkpoint(
        path=tmp_path / "actor.pt",
        actor=actor,
        critic=critic,
        optimizer=build_optimizer(actor, critic, PPOConfig()),
        rules=RulesConfig(),
        training_config={},
        global_environment_steps=1,
        completed_updates=1,
        wall_clock_seconds=0.0,
        seeds={},
    )
    agent = NeuralAgent.from_checkpoint(path, name="loaded")
    assert not hasattr(agent, "critic")
    for original, restored in zip(actor.parameters(), agent.actor.parameters(), strict=True):
        assert torch.equal(original, restored)


def test_the_agent_is_in_evaluation_mode() -> None:
    """Dropout and normalization behaviour must not depend on training state."""
    agent = build_agent()
    assert not agent.actor.training


def test_two_agents_sharing_an_actor_do_not_share_a_sampling_stream() -> None:
    actor, _critic = build_actor_and_critic(seed=1)
    first = NeuralAgent(actor=actor, name="a", deterministic=False, seed=1)
    second = NeuralAgent(actor=actor, name="b", deterministic=False, seed=2)
    assert first._generator is not second._generator


def test_an_arena_evaluation_runs_end_to_end() -> None:
    agent = build_agent(deterministic=False, seed=1)
    opponent = RandomAgent(seed=7, name="random")
    arena = Arena()
    result = arena.evaluate(
        policy_a=agent,
        policy_b=opponent,
        paired_deal_count=4,
        evaluation_seed=20250910,
        bootstrap_resamples=200,
    )
    assert result.metrics["illegal_action_rate"] == 0.0
    assert result.metrics["no_trump_action_rate"] == 0.0
    assert result.metrics["paired_deal_count"] == 4
    assert np.isfinite(result.metrics["mean_paired_score_difference"])


def test_an_unknown_action_from_a_broken_policy_would_be_caught() -> None:
    """``checked_action`` raises rather than repairing, so a bug cannot hide."""
    agent = build_agent()
    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=1)
    engine.deal(state)
    acting_seat = state.current_player_seat
    assert acting_seat is not None
    view = views.build(state, acting_seat)

    from tarneeb.agents.base import checked_action

    with pytest.raises(ValueError, match="illegal action"):
        illegal = int(np.flatnonzero(~view.legal_action_mask)[0])
        checked_action(illegal, view.legal_action_mask, agent.name)
