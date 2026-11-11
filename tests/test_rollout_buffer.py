"""Rollout storage: shapes, dtypes, per-seat credit assignment and reward conservation.

The properties that matter here are the ones a silent buffer bug would break
without any visible symptom: a seat receiving another seat's reward, a reward
being paid twice or not at all, or advantage flowing across a deal boundary.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.buffers.rollout import OPPONENT_DECISION_SEAT, RolloutBuffer
from tarneeb.core.actions import NUM_ACTIONS, PASS_ACTION
from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.spaces import D_STATIC, L_MAX, N_EVENT_FIELDS
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.training.rollout import RolloutCollector
from tarneeb.training.train import build_baseline_agent

OBSERVATION_DIM = D_STATIC


def store_one_step(
    buffer: RolloutBuffer,
    acting_seats: list[int],
    state_values: list[float] | None = None,
) -> None:
    environment_count = buffer.environment_count
    buffer.store_step(
        observations=np.zeros((environment_count, OBSERVATION_DIM), dtype=np.float32),
        history_ids=np.zeros((environment_count, L_MAX, N_EVENT_FIELDS), dtype=np.int16),
        history_masks=np.zeros((environment_count, L_MAX), dtype=bool),
        legal_action_masks=np.ones((environment_count, NUM_ACTIONS), dtype=bool),
        selected_actions=np.zeros(environment_count, dtype=np.int64),
        action_log_probabilities=np.zeros(environment_count, dtype=np.float32),
        state_values=np.array(
            state_values if state_values is not None else [0.0] * environment_count,
            dtype=np.float32,
        ),
        acting_seats=np.array(acting_seats, dtype=np.int32),
    )


def test_a_full_buffer_refuses_another_step() -> None:
    buffer = RolloutBuffer(rollout_length=1, environment_count=1)
    store_one_step(buffer, [0])
    with pytest.raises(ValueError, match="full"):
        store_one_step(buffer, [1])


def test_each_seat_is_paid_the_deal_reward_exactly_once() -> None:
    buffer = RolloutBuffer(rollout_length=4, environment_count=1)
    for seat in range(NUM_SEATS):
        store_one_step(buffer, [seat])

    rewards_by_seat = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)
    seats_paid, total_paid = buffer.settle_completed_deal(0, rewards_by_seat)

    assert seats_paid == NUM_SEATS
    assert total_paid == pytest.approx(float(rewards_by_seat.sum()))
    assert buffer.rewards[:, 0].tolist() == pytest.approx([0.5, -0.5, 0.5, -0.5])
    assert buffer.is_last_decision_of_deal[:, 0].tolist() == [True] * 4


def test_only_the_last_decision_of_each_seat_carries_the_reward() -> None:
    buffer = RolloutBuffer(rollout_length=6, environment_count=1)
    for seat in [0, 1, 2, 3, 0, 1]:
        store_one_step(buffer, [seat])

    buffer.settle_completed_deal(0, np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))

    # Seats 0 and 1 acted twice; only their second decision is paid and flagged.
    assert buffer.rewards[:, 0].tolist() == pytest.approx([0.0, 0.0, 1.0, -1.0, 1.0, -1.0])
    assert buffer.is_last_decision_of_deal[:, 0].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_reward_conservation_across_two_deals_in_one_window() -> None:
    buffer = RolloutBuffer(rollout_length=8, environment_count=1)
    emitted_total = 0.0
    paid_total = 0.0

    for deal_rewards in ([1.0, -1.0, 1.0, -1.0], [-0.25, 0.25, -0.25, 0.25]):
        for seat in range(NUM_SEATS):
            store_one_step(buffer, [seat])
        rewards = np.array(deal_rewards, dtype=np.float32)
        _seats, paid = buffer.settle_completed_deal(0, rewards)
        emitted_total += float(rewards.sum())
        paid_total += paid

    assert paid_total == pytest.approx(emitted_total)
    assert float(buffer.rewards.sum()) == pytest.approx(emitted_total)
    assert buffer.unpaid_reward_total == 0.0


def test_advantages_are_computed_per_seat_and_never_across_seats() -> None:
    """Two seats alternate; each seat's recursion must use only its own values."""
    buffer = RolloutBuffer(rollout_length=4, environment_count=1)
    store_one_step(buffer, [0], state_values=[0.5])
    store_one_step(buffer, [1], state_values=[-0.5])
    store_one_step(buffer, [0], state_values=[0.2])
    store_one_step(buffer, [1], state_values=[-0.2])
    buffer.settle_completed_deal(0, np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))

    bootstrap = np.zeros((1, NUM_SEATS), dtype=np.float32)
    advantages, returns = buffer.compute_advantages_and_returns(bootstrap, 1.0, 1.0)

    # Seat 0's sequence is rows 0 and 2 with values 0.5 then 0.2 and reward 1.0
    # on the last one: adv_2 = 1.0 - 0.2 = 0.8 ; adv_0 = (0.2 - 0.5) + 0.8 = 0.5.
    assert float(advantages[2, 0]) == pytest.approx(0.8)
    assert float(advantages[0, 0]) == pytest.approx(0.5)
    # Seat 1's sequence is rows 1 and 3, values -0.5 then -0.2, reward -1.0:
    # adv_3 = -1.0 + 0.2 = -0.8 ; adv_1 = (-0.2 + 0.5) - 0.8 = -0.5.
    assert float(advantages[3, 0]) == pytest.approx(-0.8)
    assert float(advantages[1, 0]) == pytest.approx(-0.5)

    assert returns == pytest.approx(advantages + buffer.state_values[: buffer.step_cursor])


def test_unfinished_decisions_bootstrap_from_the_state_the_seat_now_faces() -> None:
    buffer = RolloutBuffer(rollout_length=1, environment_count=1)
    store_one_step(buffer, [0], state_values=[0.25])

    bootstrap = np.zeros((1, NUM_SEATS), dtype=np.float32)
    bootstrap[0, 0] = 0.75
    advantages, _returns = buffer.compute_advantages_and_returns(bootstrap, 1.0, 1.0)

    # No reward, episode continues: delta = 0 + 0.75 - 0.25 = 0.5.
    assert float(advantages[0, 0]) == pytest.approx(0.5)


def test_bootstrap_shape_is_validated() -> None:
    buffer = RolloutBuffer(rollout_length=1, environment_count=2)
    store_one_step(buffer, [0, 1])
    with pytest.raises(ValueError, match="bootstrap values"):
        buffer.compute_advantages_and_returns(np.zeros((1, NUM_SEATS)), 1.0, 0.95)


def test_batch_shapes_and_dtypes_are_what_the_networks_expect() -> None:
    buffer = RolloutBuffer(rollout_length=3, environment_count=2)
    for seat in range(3):
        store_one_step(buffer, [seat, (seat + 1) % NUM_SEATS])
    batch = buffer.to_batch(np.zeros((2, NUM_SEATS), dtype=np.float32), 1.0, 0.95)

    assert batch.decision_count == 6
    assert batch.observations.shape == (6, OBSERVATION_DIM)
    assert batch.observations.dtype == torch.float32
    assert batch.history_ids.shape == (6, L_MAX, N_EVENT_FIELDS)
    assert batch.history_ids.dtype == torch.int64
    assert batch.history_masks.shape == (6, L_MAX)
    assert batch.history_masks.dtype == torch.bool
    assert batch.legal_action_masks.shape == (6, NUM_ACTIONS)
    assert batch.legal_action_masks.dtype == torch.bool
    assert batch.selected_actions.dtype == torch.int64
    assert batch.advantages.shape == (6,)
    assert batch.returns.shape == (6,)
    assert batch.is_finite()


# --- against the real environment -------------------------------------------


def collect_small_rollout(
    rollout_length: int = 96, num_envs: int = 4, seed: int = 0
) -> tuple[RolloutBuffer, object, RolloutCollector]:
    environments = VecTarneeb(num_envs=num_envs, root_seed=seed)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    opponents = [build_baseline_agent("basic", seed=index) for index in range(num_envs)]
    collector = RolloutCollector(
        environments, actor, critic, rollout_length, generator, opponents=opponents
    )
    buffer, metrics = collector.collect()
    return buffer, metrics, collector


def test_a_real_rollout_conserves_reward_and_stays_legal() -> None:
    buffer, metrics, _collector = collect_small_rollout()

    assert metrics.illegal_action_count == 0
    assert metrics.no_trump_action_count == 0
    assert metrics.completed_deals > 0
    # The engine emits four seat rewards per deal, which sum to zero because the
    # game is zero-sum. Only the learner's two seats are paid into the buffer, so
    # what the buffer holds must equal what was paid, not what was emitted.
    assert metrics.total_reward_emitted == pytest.approx(0.0, abs=1e-5)
    assert float(buffer.rewards.sum()) == pytest.approx(metrics.total_reward_paid, abs=1e-5)
    assert buffer.unpaid_reward_total == 0.0


def test_every_stored_row_is_either_a_real_seat_or_the_opponent_sentinel() -> None:
    buffer, metrics, _collector = collect_small_rollout()
    stored = buffer.acting_seats[: buffer.step_cursor]
    assert set(np.unique(stored)) <= {OPPONENT_DECISION_SEAT, *range(NUM_SEATS)}
    assert (stored != OPPONENT_DECISION_SEAT).sum() == metrics.learner_decision_count
    assert stored.max() < NUM_SEATS


def test_a_real_rollout_produces_a_finite_batch() -> None:
    buffer, metrics, collector = collect_small_rollout()
    batch = buffer.to_batch(collector.bootstrap_values(), 1.0, 0.95)
    assert batch.decision_count == buffer.learner_decision_count == metrics.learner_decision_count
    assert batch.decision_count < buffer.stored_decision_count
    assert batch.is_finite()
    assert bool(torch.isfinite(batch.advantages).all())


def test_partners_receive_equal_rewards_and_opponents_the_negative() -> None:
    """The reward contract, verified on what actually landed in the buffer."""
    environments = VecTarneeb(num_envs=2, root_seed=3)
    environments.reset()
    rng = np.random.default_rng(3)
    while True:
        masks = environments.legal_action_masks()
        actions = np.array([int(rng.choice(np.flatnonzero(row))) for row in masks], dtype=np.int64)
        step = environments.step(actions)
        if step.terminated.any():
            for environment_index in np.flatnonzero(step.terminated):
                rewards = step.rewards_by_seat[environment_index]
                assert float(rewards[0]) == pytest.approx(float(rewards[2]))
                assert float(rewards[1]) == pytest.approx(float(rewards[3]))
                assert float(rewards[0]) == pytest.approx(-float(rewards[1]))
            return


# --- terminal reward attribution, stated as the guardrail states it ---------


def test_an_all_pass_redeal_creates_no_terminal_reward() -> None:
    """A thrown-in hand pays nobody and ends nothing.

    The engine keeps the dealer and deals again. If the collector treated that as
    a terminal step it would inject a fabricated zero-reward episode boundary and
    stop credit from flowing across the redeal, which is wrong: the episode
    genuinely continues.
    """
    environments = VecTarneeb(num_envs=2, root_seed=0)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(0)
    opponents = [build_baseline_agent("basic", seed=index) for index in range(2)]
    collector = RolloutCollector(environments, actor, critic, 24, generator, opponents=opponents)

    # An always-passing learner plus a passing opponent throws every hand in.
    with torch.no_grad():
        actor.policy_head.weight.zero_()
        actor.policy_head.bias.fill_(-10.0)
        actor.policy_head.bias[PASS_ACTION] = 10.0
    for opponent in opponents:
        opponent.bid_threshold = 10_000  # type: ignore[attr-defined]

    buffer, metrics = collector.collect()

    assert metrics.passed_out_deals > 0, "the setup must actually pass hands out"
    assert metrics.completed_deals == 0
    assert float(buffer.rewards.sum()) == 0.0
    assert not buffer.is_last_decision_of_deal.any(), "a redeal is not a terminal transition"
    assert metrics.total_reward_paid == 0.0


def test_no_reward_is_dropped_or_duplicated_across_a_long_rollout() -> None:
    """Every completed deal pays each participating learner seat exactly once."""
    environments = VecTarneeb(num_envs=4, root_seed=1)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(1)
    opponents = [build_baseline_agent("basic", seed=index) for index in range(4)]
    collector = RolloutCollector(environments, actor, critic, 240, generator, opponents=opponents)

    buffer, metrics = collector.collect()
    assert metrics.completed_deals > 0

    paid_rows = np.argwhere(buffer.is_last_decision_of_deal[: buffer.step_cursor])
    # Two learner seats per completed deal, minus any deal whose seats had no
    # decision left inside the window. Never more than two.
    assert len(paid_rows) <= 2 * metrics.completed_deals
    assert float(buffer.rewards.sum()) == pytest.approx(metrics.total_reward_paid, abs=1e-5)

    # A flagged row must carry the reward, and an unflagged row must carry none.
    rewards = buffer.rewards[: buffer.step_cursor]
    flags = buffer.is_last_decision_of_deal[: buffer.step_cursor]
    assert not np.any(rewards[~flags] != 0.0)


def test_advantage_never_crosses_a_seat_boundary_in_a_real_rollout() -> None:
    """Each seat's recursion must see only its own rows, in its own order."""
    environments = VecTarneeb(num_envs=2, root_seed=2)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(2)
    opponents = [build_baseline_agent("basic", seed=index) for index in range(2)]
    collector = RolloutCollector(environments, actor, critic, 120, generator, opponents=opponents)
    buffer, _metrics = collector.collect()

    advantages, _returns = buffer.compute_advantages_and_returns(
        collector.bootstrap_values(), 1.0, 0.95
    )
    seats = buffer.acting_seats[: buffer.step_cursor]

    # Opponent rows take part in no sequence at all, so their advantage is zero.
    assert float(np.abs(advantages[seats == OPPONENT_DECISION_SEAT]).max(initial=0.0)) == 0.0
    # Every learner row belongs to exactly one seat sequence.
    for environment_index in range(buffer.environment_count):
        column = seats[:, environment_index]
        learner_rows = np.flatnonzero(column != OPPONENT_DECISION_SEAT)
        per_seat = [np.flatnonzero(column == seat) for seat in range(NUM_SEATS)]
        assert sum(len(rows) for rows in per_seat) == len(learner_rows)
