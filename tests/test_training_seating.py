"""Who sits where during training, and what may reach a gradient.

The learner holds one partnership and a frozen baseline holds the other. Two
properties must hold and are easy to break silently:

1. no decision the opponent made may ever enter the training batch;
2. across environments the shared policy must occupy all four absolute seats,
   or it would learn a seat-specific convention that evaluation would then
   reward for the wrong reason.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from tarneeb.agents.base import Agent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.buffers.rollout import OPPONENT_DECISION_SEAT
from tarneeb.core.state import team_of_seat
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.training.config import SELECTABLE_TRAINING_OPPONENTS, TrainingConfig
from tarneeb.training.rollout import RolloutCollector, learner_team_by_environment
from tarneeb.training.train import build_baseline_agent, build_training_opponents


def build_collector(
    opponent_name: str = "basic",
    num_envs: int = 4,
    rollout_length: int = 60,
    seed: int = 0,
) -> RolloutCollector:
    environments = VecTarneeb(num_envs=num_envs, root_seed=seed)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    opponents: list[Agent] = [
        build_baseline_agent(opponent_name, seed=index) for index in range(num_envs)
    ]
    return RolloutCollector(
        environments, actor, critic, rollout_length, generator, opponents=opponents
    )


# --- seating ----------------------------------------------------------------


def test_the_learner_partnership_alternates_across_environments() -> None:
    teams = learner_team_by_environment(8)
    assert teams.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]


def test_the_learner_occupies_all_four_absolute_seats_across_environments() -> None:
    collector = build_collector(num_envs=4)
    occupied = {
        seat
        for environment_index in range(4)
        for seat in range(4)
        if collector.is_learner_seat(environment_index, seat)
    }
    assert occupied == {0, 1, 2, 3}


def test_a_learner_seat_is_exactly_one_partnership_per_environment() -> None:
    collector = build_collector(num_envs=4)
    for environment_index in range(4):
        learner_seats = {
            seat for seat in range(4) if collector.is_learner_seat(environment_index, seat)
        }
        assert len(learner_seats) == 2
        assert len({team_of_seat(seat) for seat in learner_seats}) == 1


def test_current_policy_self_play_is_not_available() -> None:
    """Four-copy self-play is a later curriculum stage and must not be shippable.

    It is not merely unused: an early measurement showed it converges to an
    all-pass equilibrium, so offering it as a setting would offer a mode known to
    destroy the training signal.
    """
    assert "self" not in SELECTABLE_TRAINING_OPPONENTS
    assert set(SELECTABLE_TRAINING_OPPONENTS) == {"random", "basic", "strong"}

    # The collector cannot be built without an opponent partnership either.
    parameter = inspect.signature(RolloutCollector.__init__).parameters["opponents"]
    assert parameter.default is inspect.Parameter.empty, "opponents must be required"
    assert parameter.annotation == "list[Agent]"


def test_the_opponent_count_must_match_the_environment_count() -> None:
    environments = VecTarneeb(num_envs=3, root_seed=0)
    actor, critic = build_actor_and_critic(seed=1)
    with pytest.raises(ValueError, match="one opponent per environment"):
        RolloutCollector(
            environments,
            actor,
            critic,
            4,
            torch.Generator(),
            opponents=[RandomAgent(seed=0)],
        )


# --- what reaches a gradient ------------------------------------------------


def test_opponent_decisions_are_excluded_from_the_training_batch() -> None:
    collector = build_collector(num_envs=4, rollout_length=60)
    buffer, metrics = collector.collect()

    stored = buffer.acting_seats[: buffer.step_cursor]
    opponent_rows = int((stored == OPPONENT_DECISION_SEAT).sum())
    assert opponent_rows > 0, "the opponent must actually have played"
    assert buffer.learner_decision_count == metrics.learner_decision_count
    assert buffer.learner_decision_count + opponent_rows == buffer.stored_decision_count

    batch = buffer.to_batch(collector.bootstrap_values(), 1.0, 0.95)
    assert batch.decision_count == metrics.learner_decision_count
    assert batch.decision_count < buffer.stored_decision_count
    assert batch.is_finite()


def test_every_stored_learner_seat_really_belongs_to_the_learner() -> None:
    collector = build_collector(num_envs=4, rollout_length=60)
    buffer, _metrics = collector.collect()

    stored = buffer.acting_seats[: buffer.step_cursor]
    for environment_index in range(buffer.environment_count):
        for seat in stored[:, environment_index]:
            if int(seat) == OPPONENT_DECISION_SEAT:
                continue
            assert collector.is_learner_seat(environment_index, int(seat))


def test_only_learner_seats_are_paid_the_deal_reward() -> None:
    collector = build_collector(num_envs=4, rollout_length=200)
    buffer, metrics = collector.collect()
    assert metrics.completed_deals > 0

    paid_rows = np.argwhere(buffer.rewards[: buffer.step_cursor] != 0.0)
    assert paid_rows.size > 0
    for row, environment_index in paid_rows:
        seat = int(buffer.acting_seats[row, environment_index])
        assert seat != OPPONENT_DECISION_SEAT
        assert collector.is_learner_seat(environment_index, seat)


def test_the_opponent_actually_bids_so_deals_are_played_out() -> None:
    """The reason for a fixed opponent: it keeps deals flowing from step one."""
    collector = build_collector(opponent_name="basic", num_envs=4, rollout_length=200)
    _buffer, metrics = collector.collect()
    assert metrics.completed_deals > 0
    assert metrics.mean_bid is not None


def test_reward_conservation_holds_with_an_opponent_at_the_table() -> None:
    collector = build_collector(num_envs=4, rollout_length=200)
    buffer, metrics = collector.collect()
    # Only the learner's two seats are paid, so the buffer holds half of what the
    # engine emitted; the engine's four seat rewards sum to zero by construction.
    assert metrics.total_reward_emitted == pytest.approx(0.0, abs=1e-5)
    assert float(buffer.rewards.sum()) == pytest.approx(metrics.total_reward_paid, abs=1e-5)


def test_no_illegal_or_reserved_action_is_taken_with_an_opponent_present() -> None:
    for opponent_name in ("random", "basic", "strong"):
        collector = build_collector(opponent_name=opponent_name, num_envs=2, rollout_length=80)
        _buffer, metrics = collector.collect()
        assert metrics.illegal_action_count == 0
        assert metrics.no_trump_action_count == 0


# --- configuration ----------------------------------------------------------


def test_every_selectable_opponent_can_be_built() -> None:
    for name in sorted(SELECTABLE_TRAINING_OPPONENTS):
        assert build_baseline_agent(name, seed=0).name == name


def test_an_unknown_training_opponent_is_refused() -> None:
    with pytest.raises(ValueError, match="training_opponent"):
        TrainingConfig(training_opponent="alphazero")


def test_a_full_opponent_partnership_is_always_built() -> None:
    opponents = build_training_opponents(
        TrainingConfig(training_opponent="basic", environment_count=3)
    )
    assert len(opponents) == 3
    assert all(opponent.name == "basic" for opponent in opponents)


def test_each_environment_gets_its_own_opponent_instance() -> None:
    """A shared seeded opponent would interleave one stream across environments."""
    opponents = build_training_opponents(
        TrainingConfig(training_opponent="random", environment_count=4)
    )
    assert opponents is not None
    assert len({id(opponent) for opponent in opponents}) == 4
    assert len({opponent.seed for opponent in opponents}) == 4  # type: ignore[attr-defined]


def test_the_shipped_baseline_configuration_does_not_use_self_play() -> None:
    from pathlib import Path

    from tarneeb.training.config import load_training_config

    config = load_training_config(Path("configs/train/ppo_ff_baseline.yaml"))
    assert config.training_opponent != "self"
    assert config.training_opponent in SELECTABLE_TRAINING_OPPONENTS
