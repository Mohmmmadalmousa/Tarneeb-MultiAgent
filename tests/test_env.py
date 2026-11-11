"""Single-environment adapter semantics.

The adapter must implement no game rule of its own: it delegates legality to
``TarneebEngine.legal_actions``, transitions to ``TarneebEngine.step`` and
scoring to the engine's ``DealResult``. One episode is one deal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tarneeb.core.actions import PASS_ACTION, action_for_bid
from tarneeb.core.cards import CARDS_PER_HAND, NUM_SEATS, hand_size
from tarneeb.core.events import EventType, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TRICKS_PER_DEAL
from tarneeb.env.spaces import D_CRITIC, D_STATIC
from tarneeb.env.vec import EnvStep, TarneebEnv


def play_to_termination(environment: TarneebEnv, step: EnvStep, seed: int = 0) -> EnvStep:
    rng = np.random.default_rng(seed)
    while not step.terminated:
        step = environment.step(int(rng.choice(np.flatnonzero(step.legal_action_mask))))
    return step


# --- reset ------------------------------------------------------------------


def test_reset_deals_a_full_hand_and_opens_the_auction() -> None:
    environment = TarneebEnv(match_seed=42)
    step = environment.reset()

    assert environment.state is not None
    assert environment.state.phase is Phase.BID
    assert all(hand_size(hand) == CARDS_PER_HAND for hand in environment.state.hands)
    assert step.acting_seat == 1  # dealer 0, so the first bidder is seat 1
    assert not step.terminated and not step.truncated
    assert not step.rewards_by_seat.any()
    assert step.events == ()


def test_reset_is_deterministic_for_a_seed() -> None:
    first = TarneebEnv(match_seed=7)
    second = TarneebEnv(match_seed=7)
    first.reset()
    second.reset()
    assert first.state is not None and second.state is not None
    assert first.state.original_hands == second.state.original_hands


def test_reset_with_a_new_seed_changes_the_deal() -> None:
    environment = TarneebEnv(match_seed=1)
    environment.reset()
    assert environment.state is not None
    first_hands = list(environment.state.original_hands)

    environment.reset(match_seed=2)
    assert environment.state.original_hands != first_hands


def test_stepping_before_reset_is_rejected() -> None:
    with pytest.raises(ValueError, match="has not been reset"):
        TarneebEnv().step(PASS_ACTION)


def test_observations_before_reset_are_rejected() -> None:
    with pytest.raises(ValueError, match="has not been reset"):
        TarneebEnv().actor_observation()


# --- stepping ---------------------------------------------------------------


def test_step_returns_the_emitted_events() -> None:
    environment = TarneebEnv(match_seed=42)
    environment.reset()
    step = environment.step(action_for_bid(9))
    assert len(step.events) == 1
    assert step.events[0].event_type is EventType.BID
    assert step.events[0].bid_value == 9


def test_intermediate_rewards_are_zero() -> None:
    environment = TarneebEnv(match_seed=42)
    step = environment.reset()
    rng = np.random.default_rng(0)
    intermediate_steps = 0

    while True:
        step = environment.step(int(rng.choice(np.flatnonzero(step.legal_action_mask))))
        if step.terminated:
            break
        assert not step.rewards_by_seat.any()
        intermediate_steps += 1

    assert intermediate_steps > 50
    assert step.rewards_by_seat.any()


def test_terminal_step_carries_the_deal_result_and_rewards() -> None:
    environment = TarneebEnv(match_seed=42)
    step = play_to_termination(environment, environment.reset())

    assert step.terminated
    assert not step.truncated
    assert step.acting_seat is None
    assert not step.legal_action_mask.any()
    assert step.deal_result is not None
    assert step.rewards_by_seat.shape == (4,)
    assert step.rewards_by_seat[0] == step.rewards_by_seat[2]
    assert step.rewards_by_seat[1] == step.rewards_by_seat[3]
    assert step.rewards_by_seat.sum() == pytest.approx(0.0)


def test_stepping_after_termination_is_rejected() -> None:
    environment = TarneebEnv(match_seed=42)
    play_to_termination(environment, environment.reset())
    with pytest.raises(ValueError, match="episode has ended"):
        environment.step(0)


def test_the_environment_does_not_auto_reset() -> None:
    """The caller must see the terminal result before a new deal begins."""
    environment = TarneebEnv(match_seed=42)
    step = play_to_termination(environment, environment.reset())
    assert environment.state is not None
    assert environment.state.phase is Phase.SCORE
    assert environment.is_terminated()

    reset_step = environment.reset()
    assert not reset_step.terminated
    assert environment.state.phase is Phase.BID
    assert step.deal_result is not None  # the old result is still in the caller's hand


def test_illegal_actions_raise_and_are_not_penalised() -> None:
    environment = TarneebEnv(match_seed=42)
    environment.reset()
    environment.step(action_for_bid(9))
    with pytest.raises(ValueError, match="is illegal"):
        environment.step(action_for_bid(8))


def test_a_full_deal_takes_13_tricks_and_ends_scored() -> None:
    environment = TarneebEnv(match_seed=42)
    play_to_termination(environment, environment.reset())
    assert environment.state is not None
    assert len(environment.state.completed_tricks) == TRICKS_PER_DEAL
    assert sum(environment.state.tricks_won) == TRICKS_PER_DEAL


# --- the all-pass redeal is not terminal ------------------------------------


def test_all_pass_redeals_without_terminating_the_episode() -> None:
    environment = TarneebEnv(match_seed=42)
    step = environment.reset()
    assert environment.state is not None
    first_hands = list(environment.state.original_hands)
    dealer_before = environment.state.dealer_seat

    for _ in range(NUM_SEATS):
        step = environment.step(PASS_ACTION)

    assert not step.terminated  # a thrown-in hand is not an episode end
    assert not step.rewards_by_seat.any()  # and it earns no reward
    assert step.deal_result is None
    assert environment.state.phase is Phase.BID  # a fresh auction is already open
    assert environment.state.dealer_seat == dealer_before  # C10: the same dealer
    assert environment.state.original_hands != first_hands
    assert step.acting_seat == (dealer_before + 1) % NUM_SEATS


def test_an_episode_survives_several_redeals() -> None:
    environment = TarneebEnv(match_seed=42)
    step = environment.reset()
    for _ in range(3):
        for _ in range(NUM_SEATS):
            step = environment.step(PASS_ACTION)
        assert not step.terminated
    assert environment.state is not None
    assert environment.state.redeal_count == 3

    step = play_to_termination(environment, step, seed=1)
    assert step.terminated


# --- observation access -----------------------------------------------------


def test_actor_and_critic_observation_shapes() -> None:
    environment = TarneebEnv(match_seed=42)
    environment.reset()
    assert environment.actor_observation().shape == (D_STATIC,)
    assert environment.critic_observation().shape == (D_CRITIC,)


def test_observations_default_to_the_acting_seat() -> None:
    environment = TarneebEnv(match_seed=42)
    step = environment.reset()
    assert step.acting_seat is not None
    assert np.array_equal(
        environment.actor_observation(), environment.actor_observation(step.acting_seat)
    )


def test_observation_for_a_named_seat() -> None:
    environment = TarneebEnv(match_seed=42)
    environment.reset()
    for observer_seat in range(NUM_SEATS):
        assert environment.actor_observation(observer_seat).shape == (D_STATIC,)


def test_padded_history_and_ui_view_are_available() -> None:
    environment = TarneebEnv(match_seed=42)
    environment.reset()
    environment.step(action_for_bid(9))
    padded_event_history, valid_mask = environment.padded_history(0)
    assert padded_event_history.shape == (80, 7)
    assert int(valid_mask.sum()) == 1

    view = environment.ui_view(0)
    assert view["obs_spec_version"] == "obs_v2"
    assert len(view["own_hand"]) == CARDS_PER_HAND


def test_observations_after_termination_need_an_explicit_seat() -> None:
    environment = TarneebEnv(match_seed=42)
    play_to_termination(environment, environment.reset())
    with pytest.raises(ValueError, match="no seat is on turn"):
        environment.actor_observation()
    assert environment.actor_observation(0).shape == (D_STATIC,)


# --- the environment implements no rules ------------------------------------


def test_the_environment_delegates_legality_to_the_engine() -> None:
    environment = TarneebEnv(match_seed=42)
    environment.reset()
    assert environment.state is not None
    assert np.array_equal(
        environment.legal_action_mask(), environment.engine.legal_actions(environment.state)
    )


def test_the_environment_module_defines_no_game_rule() -> None:
    """A structural check: no rule constant or scoring value lives in vec.py."""
    source = Path("tarneeb/env/vec.py").read_text(encoding="utf-8")
    for forbidden in ("bid13", "kaboot", "follow", "trick_winner", "min_bid", "max_bid"):
        assert forbidden not in source, f"vec.py must not implement {forbidden}"


def test_reward_comes_from_the_engine_result() -> None:
    environment = TarneebEnv(match_seed=42)
    step = play_to_termination(environment, environment.reset())
    assert environment.state is not None
    deal_result = environment.state.deal_result
    assert deal_result is not None
    declaring_team = environment.state.declaring_team
    expected = (deal_result.score_delta_bidding_team - deal_result.score_delta_defending_team) / 42
    assert step.rewards_by_seat[declaring_team] == pytest.approx(expected)


def test_custom_rules_and_dealer_are_respected() -> None:
    environment = TarneebEnv(rules=RulesConfig(), match_seed=3, dealer_seat=2)
    step = environment.reset()
    assert environment.state is not None
    assert environment.state.dealer_seat == 2
    assert step.acting_seat == 3
