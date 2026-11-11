"""The curriculum: what each stage configures, and what it must never configure.

The scripted auction is the load-bearing idea here. A stage "without bidding" is
not a modified engine -- it is a real auction whose outcome was decided in advance
and then played through the frozen engine. These tests prove that: the deal is
legal, the contract and trump are the ones the stage asked for, the event stream
is a normal one, and the learner is never asked to bid.

They also pin the two things that would quietly corrupt the curriculum: stage 2
shaping leaking into a later stage, and action 64 becoming reachable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tarneeb.core.actions import (
    BID_ACTION_START,
    BID_ACTION_STOP,
    NO_TRUMP_ACTION,
    PASS_ACTION,
)
from tarneeb.core.cards import NUM_SEATS, NUM_SUITS
from tarneeb.core.events import EventType, Phase
from tarneeb.core.state import team_of_seat
from tarneeb.curriculum.auction_script import (
    AuctionScript,
    build_auction_script,
    scripted_auction_action,
)
from tarneeb.curriculum.scripted_env import CurriculumVecTarneeb
from tarneeb.curriculum.stages import (
    AUCTION_REMOVED,
    FULL_HISTORY,
    LEARNED_AUCTION,
    SAMPLED,
    SCRIPTED_AUCTION,
    SHIPPED_STAGES,
    StageSpec,
    stage_five_strong_play,
    stage_four_sampled_contract,
    stage_six_full_bidding,
    stage_three_deal_reward,
    stage_two_trick_play,
)
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.training.rollout import RolloutCollector
from tarneeb.training.train import build_baseline_agent


def build_environments(stage: StageSpec, num_envs: int = 4, seed: int = 0) -> CurriculumVecTarneeb:
    environments = CurriculumVecTarneeb(num_envs=num_envs, stage=stage, root_seed=seed)
    environments.reset()
    return environments


# --- stage specifications ---------------------------------------------------


def test_stage_two_is_fixed_trump_fixed_contract_against_random() -> None:
    stage = stage_two_trick_play()
    assert stage.curriculum_stage == 2
    assert stage.auction_mode == SCRIPTED_AUCTION
    assert stage.trump_mode == "fixed"
    assert stage.contract_mode == "fixed"
    assert stage.opponent == "random"
    assert stage.trick_shaping_weight == 0.0
    assert stage.history_condition == FULL_HISTORY


def test_stage_three_uses_the_frozen_deal_reward_only() -> None:
    for opponent in ("random", "basic"):
        stage = stage_three_deal_reward(opponent=opponent)
        assert stage.curriculum_stage == 3
        assert stage.opponent == opponent
        assert stage.trick_shaping_weight == 0.0
        assert stage.reward_description == "frozen normalized deal reward"


def test_stage_four_samples_trump_and_contract_against_basic() -> None:
    stage = stage_four_sampled_contract()
    assert stage.trump_mode == SAMPLED
    assert stage.contract_mode == SAMPLED
    assert stage.opponent == "basic"
    assert stage.uses_scripted_auction


def test_stage_five_is_the_strong_opponent_without_bidding() -> None:
    stage = stage_five_strong_play()
    assert stage.opponent == "strong"
    assert stage.uses_scripted_auction
    assert stage.trick_shaping_weight == 0.0


def test_stage_six_enables_the_learned_auction() -> None:
    stage = stage_six_full_bidding()
    assert stage.auction_mode == LEARNED_AUCTION
    assert not stage.uses_scripted_auction
    assert stage.opponent == "strong"


def test_trick_shaping_is_refused_outside_stage_two() -> None:
    with pytest.raises(ValueError, match="stage 2 device only"):
        StageSpec(name="bad", curriculum_stage=4, trick_shaping_weight=0.05)


def test_trick_shaping_is_refused_with_a_learned_auction() -> None:
    with pytest.raises(ValueError, match="frozen deal reward alone"):
        StageSpec(
            name="bad",
            curriculum_stage=2,
            auction_mode=LEARNED_AUCTION,
            trick_shaping_weight=0.05,
        )


def test_no_shipped_stage_after_two_carries_shaping() -> None:
    """Stage 2 shaping must not be able to leak forward through the shipped set."""
    for key, stage in SHIPPED_STAGES.items():
        if stage.curriculum_stage == 2:
            continue
        assert stage.trick_shaping_weight == 0.0, key
        assert stage.reward_description == "frozen normalized deal reward", key


def test_self_play_is_not_a_selectable_stage_opponent() -> None:
    with pytest.raises(ValueError, match="self-play is not available"):
        StageSpec(name="bad", curriculum_stage=2, opponent="self")


def test_an_illegal_fixed_contract_is_refused() -> None:
    with pytest.raises(ValueError, match="fixed_contract"):
        StageSpec(name="bad", curriculum_stage=2, fixed_contract=14)


# --- the auction script -----------------------------------------------------


def test_the_script_is_a_pure_function_of_the_deal() -> None:
    stage = stage_four_sampled_contract()
    first = build_auction_script(stage, match_seed=12345, deal_index=3, dealer_seat=1)
    second = build_auction_script(stage, match_seed=12345, deal_index=3, dealer_seat=1)
    assert first == second

    # A different deal index must at least move the declarer, which rotates with it.
    different = build_auction_script(stage, match_seed=12345, deal_index=4, dealer_seat=1)
    assert different.declarer_seat != first.declarer_seat


def test_a_fixed_stage_always_scripts_the_same_trump_and_contract() -> None:
    stage = stage_two_trick_play()
    for deal_index in range(12):
        script = build_auction_script(stage, match_seed=7, deal_index=deal_index, dealer_seat=0)
        assert script.trump_suit == stage.fixed_trump_suit
        assert script.contract == stage.fixed_contract


def test_a_sampled_stage_reaches_every_trump_suit_and_contract() -> None:
    stage = stage_four_sampled_contract()
    trumps = {
        build_auction_script(stage, match_seed=11, deal_index=index, dealer_seat=0).trump_suit
        for index in range(400)
    }
    contracts = {
        build_auction_script(stage, match_seed=11, deal_index=index, dealer_seat=0).contract
        for index in range(400)
    }
    assert trumps == set(range(NUM_SUITS))
    assert contracts == set(stage.sampled_contracts)


def test_the_declarer_rotates_through_every_seat() -> None:
    stage = stage_two_trick_play()
    declarers = {
        build_auction_script(stage, match_seed=5, deal_index=index, dealer_seat=2).declarer_seat
        for index in range(8)
    }
    assert declarers == set(range(NUM_SEATS))


def test_the_script_bids_once_and_otherwise_passes() -> None:
    script = AuctionScript(declarer_seat=2, contract=8, trump_suit=1)
    assert scripted_auction_action(script, Phase.BID, acting_seat=2, current_high_bid=None) == (
        BID_ACTION_START + 1
    )
    assert scripted_auction_action(script, Phase.BID, acting_seat=0, current_high_bid=None) == (
        PASS_ACTION
    )
    # Once the bid exists even the declarer passes, so the auction terminates.
    assert scripted_auction_action(script, Phase.BID, acting_seat=2, current_high_bid=8) == (
        PASS_ACTION
    )
    assert scripted_auction_action(script, Phase.DECLARE, acting_seat=2, current_high_bid=8) == 61


def test_the_script_refuses_to_act_during_play() -> None:
    script = AuctionScript(declarer_seat=0, contract=7, trump_suit=0)
    with pytest.raises(ValueError, match="does not act in phase"):
        scripted_auction_action(script, Phase.PLAY, acting_seat=0, current_high_bid=7)


# --- the scripted environment ----------------------------------------------


def test_a_scripted_stage_starts_every_deal_already_in_play() -> None:
    environments = build_environments(stage_two_trick_play(), num_envs=6)
    for environment in environments.environments:
        assert environment.state is not None
        assert environment.state.phase is Phase.PLAY


def test_the_scripted_contract_and_trump_are_the_ones_the_stage_asked_for() -> None:
    stage = stage_two_trick_play()
    environments = build_environments(stage, num_envs=6)
    for environment in environments.environments:
        state = environment.state
        assert state is not None
        assert state.selected_trump_suit == stage.fixed_trump_suit
        assert state.current_high_bid == stage.fixed_contract
        assert state.declarer_seat is not None


def test_a_sampled_stage_produces_several_trumps_across_environments() -> None:
    environments = build_environments(stage_four_sampled_contract(), num_envs=32, seed=3)
    trumps = {environment.state.selected_trump_suit for environment in environments.environments}
    assert len(trumps) > 1


def test_the_learner_is_never_asked_to_bid_in_a_scripted_stage() -> None:
    """The whole point of stages 2-5: no auction decision ever reaches the policy."""
    stage = stage_two_trick_play()
    environments = build_environments(stage, num_envs=4)
    rng = np.random.default_rng(0)

    for _step in range(400):
        masks = environments.legal_action_masks()
        for mask in masks:
            assert not mask[PASS_ACTION], "a pass was offered during a scripted stage"
            assert not mask[BID_ACTION_START:BID_ACTION_STOP].any()
            assert not mask[NO_TRUMP_ACTION]
        actions = np.array([int(rng.choice(np.flatnonzero(row))) for row in masks], dtype=np.int64)
        vector_step = environments.step(actions)
        if vector_step.terminated.any():
            environments.reset_terminated(vector_step.terminated)


def test_the_scripted_auction_leaves_a_normal_event_stream() -> None:
    """A scripted auction is a real auction: the events are ordinary event_v1."""
    environments = build_environments(stage_two_trick_play(), num_envs=2)
    for environment in environments.environments:
        state = environment.state
        assert state is not None
        auction_events = [
            event
            for event in state.events
            if event.event_type in (EventType.BID, EventType.PASS, EventType.TRUMP_SELECTED)
        ]
        assert len(auction_events) == 5, "four auction actions plus the trump declaration"
        assert sum(1 for event in auction_events if event.event_type is EventType.BID) == 1
        assert sum(1 for event in auction_events if event.event_type is EventType.PASS) == 3
        assert [event.seq for event in state.events] == list(range(len(state.events)))


def test_a_full_bidding_stage_offers_the_auction_to_the_learner() -> None:
    environments = build_environments(stage_six_full_bidding(), num_envs=4)
    masks = environments.legal_action_masks()
    assert masks[:, PASS_ACTION].all(), "stage 6 must let the policy pass"
    assert masks[:, BID_ACTION_START:BID_ACTION_STOP].any()


def test_action_64_is_never_legal_in_any_stage() -> None:
    for stage in SHIPPED_STAGES.values():
        environments = build_environments(stage, num_envs=4, seed=1)
        rng = np.random.default_rng(2)
        for _step in range(120):
            masks = environments.legal_action_masks()
            assert not masks[:, NO_TRUMP_ACTION].any(), stage.name
            actions = np.array(
                [int(rng.choice(np.flatnonzero(row))) for row in masks], dtype=np.int64
            )
            vector_step = environments.step(actions)
            if vector_step.terminated.any():
                environments.reset_terminated(vector_step.terminated)


# --- stage 2 shaping --------------------------------------------------------


def test_shaping_changes_only_the_terminal_reward_and_only_when_enabled() -> None:
    shaped_stage = stage_two_trick_play(trick_shaping_weight=0.5)
    plain_stage = stage_two_trick_play()

    shaped_rewards, plain_rewards = [], []
    for stage, sink in ((shaped_stage, shaped_rewards), (plain_stage, plain_rewards)):
        environments = build_environments(stage, num_envs=2, seed=4)
        rng = np.random.default_rng(4)
        for _step in range(300):
            masks = environments.legal_action_masks()
            actions = np.array(
                [int(rng.choice(np.flatnonzero(row))) for row in masks], dtype=np.int64
            )
            vector_step = environments.step(actions)
            for environment_index in np.flatnonzero(vector_step.terminated):
                state = environments.environments[int(environment_index)].state
                assert state is not None
                from tarneeb.env.reward import deal_rewards_for_state

                sink.append(
                    (
                        tuple(vector_step.rewards_by_seat[environment_index]),
                        (state.tricks_won[0], state.tricks_won[1]),
                        tuple(deal_rewards_for_state(state)),
                    )
                )
            if vector_step.terminated.any():
                environments.reset_terminated(vector_step.terminated)
            if len(sink) >= 3:
                break

    from tarneeb.env.reward import deal_rewards_by_seat

    assert shaped_rewards and plain_rewards
    for rewards, tricks, frozen_rewards in shaped_rewards:
        for seat in range(NUM_SEATS):
            own = team_of_seat(seat)
            expected_shaping = 0.5 * (tricks[own] - tricks[1 - own]) / 13
            assert rewards[seat] == pytest.approx(frozen_rewards[seat] + expected_shaping, abs=1e-6)
    del deal_rewards_by_seat


def test_an_unshaped_stage_emits_exactly_the_frozen_reward() -> None:
    """With shaping off the terminal reward must be the frozen value, untouched."""
    from tarneeb.env.reward import deal_rewards_for_state

    environments = build_environments(stage_two_trick_play(), num_envs=2, seed=6)
    rng = np.random.default_rng(6)
    checked = 0
    for _step in range(400):
        masks = environments.legal_action_masks()
        actions = np.array([int(rng.choice(np.flatnonzero(row))) for row in masks], dtype=np.int64)
        vector_step = environments.step(actions)
        for environment_index in np.flatnonzero(vector_step.terminated):
            state = environments.environments[int(environment_index)].state
            assert state is not None
            assert np.allclose(
                vector_step.rewards_by_seat[environment_index], deal_rewards_for_state(state)
            )
            checked += 1
        if vector_step.terminated.any():
            environments.reset_terminated(vector_step.terminated)
        if checked >= 3:
            break
    assert checked >= 3


# --- the frozen collector drives it unchanged -------------------------------


def test_the_frozen_rollout_collector_runs_on_a_curriculum_environment() -> None:
    stage = stage_two_trick_play()
    environments = CurriculumVecTarneeb(num_envs=4, stage=stage, root_seed=0)
    actor, critic = build_actor_and_critic(seed=1)
    generator = torch.Generator()
    generator.manual_seed(0)
    opponents = [build_baseline_agent(stage.opponent, seed=index) for index in range(4)]
    collector = RolloutCollector(environments, actor, critic, 120, generator, opponents=opponents)

    buffer, metrics = collector.collect()

    assert metrics.illegal_action_count == 0
    assert metrics.no_trump_action_count == 0
    assert metrics.completed_deals > 0
    assert metrics.learner_decision_count > 0
    assert metrics.pass_rate is None, "a scripted stage offers the learner no auction decision"
    batch = buffer.to_batch(collector.bootstrap_values(), 1.0, 0.95)
    assert batch.is_finite()


def test_every_scripted_deal_reaches_the_configured_contract() -> None:
    """The declarer always bids exactly the scripted contract, never more."""
    stage = stage_four_sampled_contract()
    environments = build_environments(stage, num_envs=8, seed=9)
    for environment in environments.environments:
        state = environment.state
        assert state is not None
        assert state.current_high_bid in stage.sampled_contracts


def test_history_conditioning_is_off_by_default() -> None:
    from tarneeb.env.observation import ObservationBuilder

    environments = build_environments(stage_two_trick_play(), num_envs=2)
    assert type(environments.observations) is ObservationBuilder


def test_an_ablated_stage_installs_the_conditioned_builder() -> None:
    from tarneeb.curriculum.history import HistoryConditionedObservationBuilder

    stage = stage_six_full_bidding(history_condition=AUCTION_REMOVED)
    environments = CurriculumVecTarneeb(num_envs=2, stage=stage, root_seed=0)
    assert isinstance(environments.observations, HistoryConditionedObservationBuilder)
    for environment in environments.environments:
        assert isinstance(environment.observations, HistoryConditionedObservationBuilder)
