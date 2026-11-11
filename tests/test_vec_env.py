"""Vectorized environment: batching, determinism and state independence.

The vector wrapper is the training hot path, so the properties that matter are
that environments never alias each other's state, that a whole run is
reproducible from one root seed, and that one environment finishing cannot
disturb another.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from tarneeb.core.actions import NUM_ACTIONS
from tarneeb.core.cards import CARDS_PER_HAND, NUM_SEATS, hand_size
from tarneeb.core.events import Phase
from tarneeb.env.spaces import D_CRITIC, D_STATIC
from tarneeb.env.vec import VecStep, VecTarneeb, derive_environment_seed

NUM_ENVS = 8


def masked_actions(vector_step: VecStep, rng: np.random.Generator) -> list[int]:
    """A deterministic masked sampler: only ever pick a legal action."""
    return [
        int(rng.choice(np.flatnonzero(mask))) if mask.any() else 0
        for mask in vector_step.legal_action_masks
    ]


def run_vector_episode(
    vector_env: VecTarneeb, policy_seed: int, steps: int = 200
) -> tuple[list, list]:
    """Drive the vector env deterministically; return the action and reward traces."""
    rng = np.random.default_rng(policy_seed)
    vector_step = vector_env.reset()
    action_trace: list[list[int]] = []
    reward_trace: list[np.ndarray] = []

    for _ in range(steps):
        actions = masked_actions(vector_step, rng)
        action_trace.append(actions)
        vector_step = vector_env.step(actions)
        reward_trace.append(vector_step.rewards_by_seat.copy())
        if vector_step.terminated.any():
            vector_step = vector_env.reset_terminated(vector_step.terminated)

    return action_trace, reward_trace


# --- construction and batch shapes -------------------------------------------


def test_batch_shapes_and_dtypes() -> None:
    vector_env = VecTarneeb(num_envs=NUM_ENVS, root_seed=42)
    vector_step = vector_env.reset()

    assert vector_step.acting_seats.shape == (NUM_ENVS,)
    assert vector_step.acting_seats.dtype == np.int32
    assert vector_step.legal_action_masks.shape == (NUM_ENVS, NUM_ACTIONS)
    assert vector_step.legal_action_masks.dtype == np.bool_
    assert vector_step.rewards_by_seat.shape == (NUM_ENVS, NUM_SEATS)
    assert vector_step.rewards_by_seat.dtype == np.float32
    assert vector_step.terminated.shape == (NUM_ENVS,)
    assert vector_step.terminated.dtype == np.bool_
    assert vector_step.truncated.shape == (NUM_ENVS,)
    assert len(vector_step.events) == NUM_ENVS


def test_observation_batch_shapes() -> None:
    vector_env = VecTarneeb(num_envs=NUM_ENVS, root_seed=42)
    vector_env.reset()
    assert vector_env.actor_observations().shape == (NUM_ENVS, D_STATIC)
    assert vector_env.actor_observations().dtype == np.float32
    assert vector_env.critic_observations().shape == (NUM_ENVS, D_CRITIC)
    assert vector_env.critic_observations().dtype == np.float32
    assert vector_env.legal_action_masks().shape == (NUM_ENVS, NUM_ACTIONS)
    assert vector_env.acting_seats().shape == (NUM_ENVS,)


def test_one_acting_seat_per_environment() -> None:
    vector_env = VecTarneeb(num_envs=NUM_ENVS, root_seed=42)
    vector_step = vector_env.reset()
    assert (vector_step.acting_seats >= 0).all()
    assert (vector_step.acting_seats < NUM_SEATS).all()


def test_num_envs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_envs must be at least 1"):
        VecTarneeb(num_envs=0)


def test_action_batch_shape_is_validated() -> None:
    vector_env = VecTarneeb(num_envs=4, root_seed=1)
    vector_env.reset()
    with pytest.raises(ValueError, match="expected 4 actions"):
        vector_env.step([0, 0])


def test_terminated_flag_batch_shape_is_validated() -> None:
    vector_env = VecTarneeb(num_envs=4, root_seed=1)
    vector_env.reset()
    with pytest.raises(ValueError, match="expected 4 flags"):
        vector_env.reset_terminated(np.zeros(2, dtype=bool))


# --- seeding and determinism -------------------------------------------------


def test_per_environment_seeds_are_distinct_and_deterministic() -> None:
    seeds = [derive_environment_seed(42, index) for index in range(32)]
    assert len(set(seeds)) == 32
    assert seeds == [derive_environment_seed(42, index) for index in range(32)]
    assert seeds != [derive_environment_seed(43, index) for index in range(32)]


def test_environments_are_dealt_different_hands() -> None:
    vector_env = VecTarneeb(num_envs=NUM_ENVS, root_seed=42)
    vector_env.reset()
    dealt_hands = [
        tuple(environment.state.original_hands) for environment in vector_env.environments
    ]
    assert len(set(dealt_hands)) == NUM_ENVS


def test_same_root_seed_reproduces_the_whole_run() -> None:
    first_env = VecTarneeb(num_envs=NUM_ENVS, root_seed=2024)
    second_env = VecTarneeb(num_envs=NUM_ENVS, root_seed=2024)

    first_actions, first_rewards = run_vector_episode(first_env, policy_seed=7)
    second_actions, second_rewards = run_vector_episode(second_env, policy_seed=7)

    assert first_actions == second_actions
    for first_reward, second_reward in zip(first_rewards, second_rewards, strict=True):
        assert np.array_equal(first_reward, second_reward)


def test_different_root_seeds_diverge() -> None:
    first_actions, _ = run_vector_episode(VecTarneeb(NUM_ENVS, root_seed=1), policy_seed=7)
    second_actions, _ = run_vector_episode(VecTarneeb(NUM_ENVS, root_seed=2), policy_seed=7)
    assert first_actions != second_actions


def test_reset_with_a_root_seed_reproduces_the_deals() -> None:
    vector_env = VecTarneeb(num_envs=4, root_seed=1)
    vector_env.reset(root_seed=99)
    first_hands = [tuple(env.state.original_hands) for env in vector_env.environments]

    vector_env.reset(root_seed=99)
    assert [tuple(env.state.original_hands) for env in vector_env.environments] == first_hands


def test_global_rng_churn_does_not_affect_the_vector_env() -> None:
    np.random.seed(1)
    random.seed(1)
    first_actions, _ = run_vector_episode(VecTarneeb(4, root_seed=5), policy_seed=3)

    np.random.seed(12345)
    random.seed(12345)
    for _ in range(500):
        np.random.random()
        random.random()
    second_actions, _ = run_vector_episode(VecTarneeb(4, root_seed=5), policy_seed=3)

    assert first_actions == second_actions


def test_observations_are_reproducible_across_instances() -> None:
    first_env = VecTarneeb(num_envs=4, root_seed=77)
    second_env = VecTarneeb(num_envs=4, root_seed=77)
    first_env.reset()
    second_env.reset()
    assert np.array_equal(first_env.actor_observations(), second_env.actor_observations())
    assert np.array_equal(first_env.critic_observations(), second_env.critic_observations())


# --- state independence ------------------------------------------------------


def test_environments_do_not_share_state_objects() -> None:
    vector_env = VecTarneeb(num_envs=NUM_ENVS, root_seed=42)
    vector_env.reset()
    states = [environment.state for environment in vector_env.environments]

    assert len({id(state) for state in states}) == NUM_ENVS
    assert len({id(state.hands) for state in states}) == NUM_ENVS
    assert len({id(state.events) for state in states}) == NUM_ENVS
    assert len({id(state.tricks_won) for state in states}) == NUM_ENVS
    assert len({id(state.void_flags) for state in states}) == NUM_ENVS


def test_stepping_one_environment_does_not_disturb_another() -> None:
    vector_env = VecTarneeb(num_envs=3, root_seed=13)
    vector_step = vector_env.reset()

    untouched_hands = list(vector_env.environments[2].state.hands)
    untouched_events = len(vector_env.environments[2].state.events)

    rng = np.random.default_rng(0)
    for _ in range(5):
        actions = masked_actions(vector_step, rng)
        actions[2] = int(np.flatnonzero(vector_step.legal_action_masks[2])[0])
        vector_step = vector_env.step(actions)

    assert vector_env.environments[0].state.hands != untouched_hands
    assert len(vector_env.environments[2].state.events) == untouched_events + 5


def test_one_environment_terminating_does_not_corrupt_the_others() -> None:
    vector_env = VecTarneeb(num_envs=6, root_seed=21)
    vector_step = vector_env.reset()
    rng = np.random.default_rng(1)

    saw_termination = False
    for _ in range(400):
        actions = masked_actions(vector_step, rng)
        vector_step = vector_env.step(actions)

        if vector_step.terminated.any():
            saw_termination = True
            terminated_indices = set(np.flatnonzero(vector_step.terminated).tolist())
            for environment_index, environment in enumerate(vector_env.environments):
                state = environment.state
                assert state is not None
                if environment_index in terminated_indices:
                    assert state.phase is Phase.SCORE
                else:
                    assert state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY)
                    assert sum(state.tricks_won) == len(state.completed_tricks)

            vector_step = vector_env.reset_terminated(vector_step.terminated)
            for environment_index in terminated_indices:
                restarted_state = vector_env.environments[environment_index].state
                assert restarted_state.phase is Phase.BID
                assert all(hand_size(hand) == CARDS_PER_HAND for hand in restarted_state.hands)
    assert saw_termination


def test_reset_terminated_leaves_running_environments_alone() -> None:
    vector_env = VecTarneeb(num_envs=4, root_seed=31)
    vector_step = vector_env.reset()
    rng = np.random.default_rng(2)
    for _ in range(6):
        vector_step = vector_env.step(masked_actions(vector_step, rng))

    events_before = [len(env.state.events) for env in vector_env.environments]
    vector_env.reset_terminated(np.zeros(4, dtype=bool))
    assert [len(env.state.events) for env in vector_env.environments] == events_before


# --- rewards and termination -------------------------------------------------


def test_rewards_are_zero_until_an_environment_terminates() -> None:
    vector_env = VecTarneeb(num_envs=4, root_seed=41)
    vector_step = vector_env.reset()
    rng = np.random.default_rng(3)

    for _ in range(300):
        vector_step = vector_env.step(masked_actions(vector_step, rng))
        for environment_index in range(4):
            if not vector_step.terminated[environment_index]:
                assert not vector_step.rewards_by_seat[environment_index].any()
            else:
                rewards = vector_step.rewards_by_seat[environment_index]
                assert rewards[0] == rewards[2]
                assert rewards[1] == rewards[3]
                assert rewards.sum() == pytest.approx(0.0)
        if vector_step.terminated.any():
            vector_step = vector_env.reset_terminated(vector_step.terminated)


def test_truncated_is_always_false_in_phase_four() -> None:
    vector_env = VecTarneeb(num_envs=4, root_seed=51)
    vector_step = vector_env.reset()
    rng = np.random.default_rng(4)
    for _ in range(150):
        vector_step = vector_env.step(masked_actions(vector_step, rng))
        assert not vector_step.truncated.any()
        if vector_step.terminated.any():
            vector_step = vector_env.reset_terminated(vector_step.terminated)


def test_terminated_environments_report_no_acting_seat() -> None:
    vector_env = VecTarneeb(num_envs=4, root_seed=61)
    vector_step = vector_env.reset()
    rng = np.random.default_rng(5)
    for _ in range(300):
        vector_step = vector_env.step(masked_actions(vector_step, rng))
        for environment_index in np.flatnonzero(vector_step.terminated).tolist():
            assert vector_step.acting_seats[environment_index] == -1
            assert not vector_step.legal_action_masks[environment_index].any()
        if vector_step.terminated.any():
            vector_step = vector_env.reset_terminated(vector_step.terminated)


def test_dealer_rotation_across_environment_indices() -> None:
    vector_env = VecTarneeb(num_envs=8, root_seed=71)
    vector_env.reset()
    dealers = [environment.state.dealer_seat for environment in vector_env.environments]
    assert dealers == [0, 1, 2, 3, 0, 1, 2, 3]


def test_dealer_rotation_can_be_disabled() -> None:
    vector_env = VecTarneeb(num_envs=4, root_seed=71, rotate_dealer_by_index=False)
    vector_env.reset()
    assert [env.state.dealer_seat for env in vector_env.environments] == [0, 0, 0, 0]


def test_a_single_environment_vector_works() -> None:
    vector_env = VecTarneeb(num_envs=1, root_seed=81)
    vector_step = vector_env.reset()
    assert vector_step.acting_seats.shape == (1,)
    rng = np.random.default_rng(6)
    vector_step = vector_env.step(masked_actions(vector_step, rng))
    assert vector_step.legal_action_masks.shape == (1, NUM_ACTIONS)
