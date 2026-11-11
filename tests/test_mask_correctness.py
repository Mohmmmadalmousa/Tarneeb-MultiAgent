"""Mask correctness gate: the environment mask must equal the engine mask.

The environment never computes legality itself, so this checks the contract
holds at the boundary, and that the mask is phase-appropriate at every decision.

The large audit (at least 1,000,000 reachable decisions) is marked ``slow`` and
runs with ``pytest -m slow``; the default suite runs a smaller version of the
same walk so a regression still fails fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.core.actions import (
    NO_TRUMP_ACTION,
    NUM_ACTIONS,
    PASS_ACTION,
    action_for_bid,
    bid_for_action,
    card_for_action,
    is_bid_action,
    is_card_action,
    is_trump_action,
)
from tarneeb.core.cards import NUM_SEATS, hand_contains, holds_suit, suit_of
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import TarneebEnv, VecTarneeb

DEFAULT_DECISION_TARGET = 20_000
LARGE_DECISION_TARGET = 1_000_000

TRUMP_ACTION_IDS = frozenset(range(60, 64))
BID_ACTION_IDS = frozenset(range(52, 59))
CARD_ACTION_IDS = frozenset(range(0, 52))


@pytest.fixture
def engine() -> TarneebEngine:
    return TarneebEngine(RulesConfig())


def check_mask_against_engine(environment: TarneebEnv) -> None:
    """The hard gate: environment mask is exactly the engine mask."""
    state = environment.state
    assert state is not None
    environment_mask = environment.legal_action_mask()
    engine_mask = environment.engine.legal_actions(state)
    assert np.array_equal(environment_mask, engine_mask)


def check_mask_shape_and_phase(environment: TarneebEnv) -> np.ndarray:
    """Check the mask is well formed and only enables the right action block."""
    state = environment.state
    assert state is not None
    mask = environment.legal_action_mask()

    assert mask.shape == (NUM_ACTIONS,)
    assert mask.dtype == np.bool_
    assert mask.any(), "every live decision must offer at least one legal action"
    assert not mask[NO_TRUMP_ACTION]

    legal_ids = set(np.flatnonzero(mask).tolist())
    acting_seat = state.current_player_seat
    assert acting_seat is not None

    if state.phase is Phase.BID:
        assert legal_ids <= BID_ACTION_IDS | {PASS_ACTION}
        assert PASS_ACTION in legal_ids
        for action_id in legal_ids:
            if is_bid_action(action_id):
                bid_value = bid_for_action(action_id)
                assert 7 <= bid_value <= 13
                if state.current_high_bid is not None:
                    assert bid_value > state.current_high_bid
    elif state.phase is Phase.DECLARE:
        assert legal_ids == TRUMP_ACTION_IDS
        assert acting_seat == state.declarer_seat
    else:
        assert state.phase is Phase.PLAY
        assert legal_ids <= CARD_ACTION_IDS
        for action_id in legal_ids:
            assert is_card_action(action_id)
            assert hand_contains(state.hands[acting_seat], card_for_action(action_id))
        if state.current_trick:
            led_suit = suit_of(state.current_trick[0][1])
            if holds_suit(state.hands[acting_seat], led_suit):
                assert all(suit_of(card_for_action(a)) == led_suit for a in legal_ids)
    return mask


def walk_decisions(decision_target: int, root_seed: int) -> dict:
    """Sample reachable decisions, checking the mask contract at each one."""
    statistics = {
        "decisions_checked": 0,
        "deals_completed": 0,
        "illegal_actions_selected": 0,
        "no_trump_offered": 0,
        "empty_masks": 0,
        "phase_counts": {Phase.BID: 0, Phase.DECLARE: 0, Phase.PLAY: 0},
    }
    rng = np.random.default_rng(root_seed)
    environment = TarneebEnv(match_seed=root_seed)
    step = environment.reset()

    while statistics["decisions_checked"] < decision_target:
        if step.terminated:
            statistics["deals_completed"] += 1
            step = environment.reset()
            continue

        state = environment.state
        assert state is not None
        statistics["phase_counts"][state.phase] += 1

        check_mask_against_engine(environment)
        mask = check_mask_shape_and_phase(environment)
        if not mask.any():
            statistics["empty_masks"] += 1
        if mask[NO_TRUMP_ACTION]:
            statistics["no_trump_offered"] += 1

        # A masked sampler: only ever choose from the legal set.
        chosen_action = int(rng.choice(np.flatnonzero(mask)))
        if not mask[chosen_action]:
            statistics["illegal_actions_selected"] += 1

        step = environment.step(chosen_action)
        statistics["decisions_checked"] += 1

    return statistics


# --- targeted checks --------------------------------------------------------


def test_mask_matches_engine_in_every_phase(engine: TarneebEngine) -> None:
    environment = TarneebEnv(match_seed=42)
    step = environment.reset()
    phases_seen = set()

    rng = np.random.default_rng(0)
    while not step.terminated:
        state = environment.state
        assert state is not None
        phases_seen.add(state.phase)
        check_mask_against_engine(environment)
        check_mask_shape_and_phase(environment)
        step = environment.step(int(rng.choice(np.flatnonzero(step.legal_action_mask))))

    assert phases_seen == {Phase.BID, Phase.DECLARE, Phase.PLAY}


def test_mask_shape_and_dtype() -> None:
    environment = TarneebEnv(match_seed=1)
    step = environment.reset()
    assert step.legal_action_mask.shape == (65,)
    assert step.legal_action_mask.dtype == np.bool_


def test_terminated_step_has_an_all_false_mask() -> None:
    environment = TarneebEnv(match_seed=3)
    step = environment.reset()
    rng = np.random.default_rng(1)
    while not step.terminated:
        step = environment.step(int(rng.choice(np.flatnonzero(step.legal_action_mask))))
    assert not step.legal_action_mask.any()
    assert step.acting_seat is None


def test_bid_masks_are_strictly_increasing() -> None:
    environment = TarneebEnv(match_seed=5)
    environment.reset()
    environment.step(action_for_bid(9))
    legal_ids = set(np.flatnonzero(environment.legal_action_mask()).tolist())
    assert legal_ids == {action_for_bid(v) for v in range(10, 14)} | {PASS_ACTION}


def test_declaration_mask_is_only_the_four_suits() -> None:
    environment = TarneebEnv(match_seed=7)
    environment.reset()
    environment.step(action_for_bid(8))
    for _ in range(NUM_SEATS - 1):
        environment.step(PASS_ACTION)
    legal_ids = set(np.flatnonzero(environment.legal_action_mask()).tolist())
    assert legal_ids == TRUMP_ACTION_IDS
    assert all(is_trump_action(action_id) for action_id in legal_ids)


def test_vector_masks_match_each_environments_engine_mask() -> None:
    vector_env = VecTarneeb(num_envs=6, root_seed=11)
    vector_step = vector_env.reset()
    rng = np.random.default_rng(2)

    for _ in range(40):
        batched_masks = vector_env.legal_action_masks()
        assert np.array_equal(batched_masks, vector_step.legal_action_masks)
        for environment_index, environment in enumerate(vector_env.environments):
            state = environment.state
            assert state is not None
            if state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
                assert np.array_equal(
                    batched_masks[environment_index], vector_env.engine.legal_actions(state)
                )

        actions = [
            int(rng.choice(np.flatnonzero(mask))) if mask.any() else 0
            for mask in vector_step.legal_action_masks
        ]
        vector_step = vector_env.step(actions)
        if vector_step.terminated.any():
            vector_step = vector_env.reset_terminated(vector_step.terminated)


# --- the sampled audit ------------------------------------------------------


def test_default_mask_audit() -> None:
    statistics = walk_decisions(DEFAULT_DECISION_TARGET, root_seed=1000)

    assert statistics["decisions_checked"] == DEFAULT_DECISION_TARGET
    assert statistics["illegal_actions_selected"] == 0
    assert statistics["no_trump_offered"] == 0
    assert statistics["empty_masks"] == 0
    assert statistics["deals_completed"] > 0
    for phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
        assert statistics["phase_counts"][phase] > 0


@pytest.mark.slow
def test_large_mask_audit_over_one_million_decisions() -> None:
    """The Phase 4 gate: at least 1,000,000 reachable decisions."""
    statistics = walk_decisions(LARGE_DECISION_TARGET, root_seed=2024)

    assert statistics["decisions_checked"] == LARGE_DECISION_TARGET
    assert statistics["illegal_actions_selected"] == 0
    assert statistics["no_trump_offered"] == 0
    assert statistics["empty_masks"] == 0
    assert statistics["deals_completed"] > 10_000
