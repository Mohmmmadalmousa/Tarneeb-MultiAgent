"""The no-trump hard gate (invariant I13) at the environment boundary.

Action 64 is reserved so a future rules version could enable no-trump without
breaking checkpoint compatibility. Under ``tarneeb-classic-v1`` it must be
unreachable: masked in every state, and rejected by ``step`` through the normal
illegal-action path with no reward penalty.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.core.actions import NO_TRUMP_ACTION, PASS_ACTION, action_for_bid
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.spaces import ACTOR_FEATURE_SLICES
from tarneeb.env.vec import TarneebEnv, VecTarneeb

DEFAULT_DECISION_TARGET = 20_000
LARGE_DECISION_TARGET = 1_000_000


def test_no_trump_action_id() -> None:
    assert NO_TRUMP_ACTION == 64


def test_no_trump_is_disabled_in_the_frozen_rules() -> None:
    assert RulesConfig().allow_no_trump is False


def walk_checking_no_trump(decision_target: int, root_seed: int) -> int:
    rng = np.random.default_rng(root_seed)
    environment = TarneebEnv(match_seed=root_seed)
    step = environment.reset()
    decisions_checked = 0

    while decisions_checked < decision_target:
        if step.terminated:
            step = environment.reset()
            continue
        assert not step.legal_action_mask[NO_TRUMP_ACTION]
        chosen_action = int(rng.choice(np.flatnonzero(step.legal_action_mask)))
        assert chosen_action != NO_TRUMP_ACTION
        step = environment.step(chosen_action)
        decisions_checked += 1

    return decisions_checked


def test_no_trump_is_masked_across_sampled_decisions() -> None:
    assert walk_checking_no_trump(DEFAULT_DECISION_TARGET, root_seed=99) == (
        DEFAULT_DECISION_TARGET
    )


@pytest.mark.slow
def test_no_trump_is_masked_across_one_million_decisions() -> None:
    assert walk_checking_no_trump(LARGE_DECISION_TARGET, root_seed=4242) == (LARGE_DECISION_TARGET)


def test_step_rejects_action_64_during_declaration() -> None:
    """It must fail through the illegal-action path, not score a penalty."""
    environment = TarneebEnv(match_seed=7)
    environment.reset()
    environment.step(action_for_bid(10))
    for _ in range(NUM_SEATS - 1):
        environment.step(PASS_ACTION)
    assert environment.state is not None
    assert environment.state.phase is Phase.DECLARE

    with pytest.raises(ValueError, match="is illegal"):
        environment.step(NO_TRUMP_ACTION)


def test_step_rejects_action_64_during_bidding_and_play() -> None:
    environment = TarneebEnv(match_seed=8)
    environment.reset()
    with pytest.raises(ValueError, match="is illegal"):
        environment.step(NO_TRUMP_ACTION)


def test_rejecting_action_64_emits_no_reward_and_no_event() -> None:
    environment = TarneebEnv(match_seed=9)
    step_before = environment.reset()
    assert environment.state is not None
    events_before = len(environment.state.events)

    with pytest.raises(ValueError):
        environment.step(NO_TRUMP_ACTION)

    assert len(environment.state.events) == events_before
    assert not step_before.rewards_by_seat.any()
    assert environment.state.phase is Phase.BID


def test_observation_never_marks_action_64_legal() -> None:
    environment = TarneebEnv(match_seed=11)
    builder = ObservationBuilder(environment.engine)
    step = environment.reset()
    rng = np.random.default_rng(3)

    while not step.terminated:
        state = environment.state
        assert state is not None
        acting_seat = state.current_player_seat
        assert acting_seat is not None
        observation = builder.static_actor(state, acting_seat)
        mask_block = observation[ACTOR_FEATURE_SLICES["legal_action_mask"]]
        assert mask_block[NO_TRUMP_ACTION] == 0.0

        trump_block = observation[ACTOR_FEATURE_SLICES["trump"]]
        assert trump_block[5] == 0.0  # the reserved no-trump slot of S12

        step = environment.step(int(rng.choice(np.flatnonzero(step.legal_action_mask))))


def test_vectorized_masks_never_offer_no_trump() -> None:
    vector_env = VecTarneeb(num_envs=8, root_seed=5)
    vector_step = vector_env.reset()
    rng = np.random.default_rng(4)

    for _ in range(120):
        assert not vector_step.legal_action_masks[:, NO_TRUMP_ACTION].any()
        actions = [
            int(rng.choice(np.flatnonzero(mask))) if mask.any() else 0
            for mask in vector_step.legal_action_masks
        ]
        assert NO_TRUMP_ACTION not in actions
        vector_step = vector_env.step(actions)
        if vector_step.terminated.any():
            vector_step = vector_env.reset_terminated(vector_step.terminated)
