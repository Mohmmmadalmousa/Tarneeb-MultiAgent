"""Stage evaluation: held-out seeds, duplicate pairing, and honest bookkeeping.

Phase 7 adds one thing to the frozen Phase 5 arena: the ability to install a
scripted auction before each deal, which ``Arena.evaluate`` has no hook for. That
addition must not weaken the protocol, so these tests check the protocol survives
intact -- the same cards played twice with the partnerships swapped, seeds drawn
only from the held-out range, and the same auction in both orientations so the
pairing still cancels card luck.
"""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.core.cards import NUM_SUITS, SUIT_NAMES
from tarneeb.curriculum.agents import ScriptedAuctionAgent
from tarneeb.curriculum.auction_script import build_auction_script
from tarneeb.curriculum.config import (
    load_stage_configuration,
    stage_configuration_from_dict,
)
from tarneeb.curriculum.evaluation import (
    evaluate_stage,
    evaluation_seeds_are_isolated,
    summarise_by_trump_suit,
)
from tarneeb.curriculum.stages import (
    stage_four_sampled_contract,
    stage_six_full_bidding,
    stage_two_trick_play,
)
from tarneeb.curriculum.trainer import planned_evaluation_updates
from tarneeb.evaluation.arena import (
    HELD_OUT_EVALUATION_SEEDS,
    TRAINING_SEEDS,
    evaluation_match_seed,
)

DEVELOPMENT_EVALUATION_SEED = 70000007
FINAL_EVALUATION_SEED = 20250910


# --- seed isolation ---------------------------------------------------------


def test_evaluation_seeds_are_disjoint_from_training_seeds() -> None:
    assert evaluation_seeds_are_isolated()
    assert HELD_OUT_EVALUATION_SEEDS.start >= TRAINING_SEEDS.stop


def test_every_evaluation_deal_comes_from_the_held_out_range() -> None:
    for seed in (DEVELOPMENT_EVALUATION_SEED, FINAL_EVALUATION_SEED):
        for pair_index in range(2000):
            assert evaluation_match_seed(seed, pair_index) in HELD_OUT_EVALUATION_SEEDS


def test_the_development_and_final_seeds_select_different_deals() -> None:
    """A checkpoint must not be chosen by repeatedly probing the final deal set."""
    development = {evaluation_match_seed(DEVELOPMENT_EVALUATION_SEED, i) for i in range(2000)}
    final = {evaluation_match_seed(FINAL_EVALUATION_SEED, i) for i in range(2000)}
    assert development.isdisjoint(final)


def test_every_shipped_training_seed_is_a_training_seed() -> None:
    from pathlib import Path

    for path in sorted(Path("configs/train").glob("stage*.yaml")) + sorted(
        Path("configs/ablation").glob("a7_*.yaml")
    ):
        config, _stage = load_stage_configuration(path)
        assert config.environment_root_seed in TRAINING_SEEDS, path
        assert config.environment_root_seed not in HELD_OUT_EVALUATION_SEEDS, path


# --- the duplicate-pair protocol -------------------------------------------


def test_a_stage_evaluation_plays_each_deal_twice_with_swapped_partnerships() -> None:
    stage = stage_two_trick_play()
    result = evaluate_stage(
        stage=stage,
        policy_a=BasicHeuristicAgent(name="basic"),
        policy_b=RandomAgent(name="random", seed=1),
        paired_deal_count=6,
        evaluation_seed=DEVELOPMENT_EVALUATION_SEED,
        bootstrap_resamples=200,
    )
    assert result.metrics["paired_deal_count"] == 6
    assert result.metrics["played_deal_count"] == 12
    assert result.metrics["illegal_action_rate"] == 0.0
    assert result.metrics["no_trump_action_rate"] == 0.0
    assert "paired_score_difference_ci" in result.metrics
    assert "win_rate_wilson_interval" in result.metrics


def test_both_orientations_of_a_pair_get_the_same_scripted_auction() -> None:
    """If the auction differed by orientation, the pairing would not cancel."""
    stage = stage_four_sampled_contract()
    for pair_index in range(20):
        match_seed = evaluation_match_seed(DEVELOPMENT_EVALUATION_SEED, pair_index)
        dealer_seat = pair_index % 4
        first = build_auction_script(stage, match_seed, deal_index=0, dealer_seat=dealer_seat)
        second = build_auction_script(stage, match_seed, deal_index=0, dealer_seat=dealer_seat)
        assert first == second


def test_a_scripted_evaluation_never_asks_an_agent_to_bid() -> None:
    class RefusesToBid(BasicHeuristicAgent):
        def act(self, view):  # type: ignore[no-untyped-def]
            from tarneeb.core.events import Phase

            assert view.phase is Phase.PLAY, "the inner agent must only ever play cards"
            return super().act(view)

    stage = stage_two_trick_play()
    result = evaluate_stage(
        stage=stage,
        policy_a=RefusesToBid(name="basic"),
        policy_b=RefusesToBid(name="basic2"),
        paired_deal_count=4,
        evaluation_seed=DEVELOPMENT_EVALUATION_SEED,
        bootstrap_resamples=100,
    )
    assert result.metrics["passed_out_rate"] == 0.0


def test_the_scripted_wrapper_refuses_to_act_without_a_script() -> None:
    from tarneeb.core.engine import TarneebEngine
    from tarneeb.core.rules import RulesConfig
    from tarneeb.env.public_view import PublicViewBuilder

    engine = TarneebEngine(RulesConfig())
    views = PublicViewBuilder(engine)
    state = engine.new_deal(0, deal_index=0, match_seed=1)
    engine.deal(state)
    view = views.build(state, state.current_player_seat)  # type: ignore[arg-type]

    agent = ScriptedAuctionAgent(BasicHeuristicAgent())
    with pytest.raises(ValueError, match="no auction script"):
        agent.act(view)


def test_a_full_game_stage_uses_the_frozen_arena_without_scripting() -> None:
    stage = stage_six_full_bidding()
    assert not stage.uses_scripted_auction
    result = evaluate_stage(
        stage=stage,
        policy_a=BasicHeuristicAgent(name="basic"),
        policy_b=RandomAgent(name="random", seed=2),
        paired_deal_count=4,
        evaluation_seed=DEVELOPMENT_EVALUATION_SEED,
        bootstrap_resamples=100,
    )
    assert result.trump_suit_breakdown == {}, "no script means no scripted trump to split by"
    assert result.metrics["mean_bid"] is not None


def test_an_empty_evaluation_is_refused() -> None:
    with pytest.raises(ValueError, match="paired_deal_count"):
        evaluate_stage(
            stage=stage_two_trick_play(),
            policy_a=BasicHeuristicAgent(),
            policy_b=RandomAgent(seed=0),
            paired_deal_count=0,
            evaluation_seed=DEVELOPMENT_EVALUATION_SEED,
        )


# --- the trump-suit breakdown ----------------------------------------------


def test_the_trump_breakdown_splits_the_pairs_it_was_given() -> None:
    class FakePair:
        def __init__(self, difference: float) -> None:
            self.paired_score_difference = difference

    pairs = [FakePair(value) for value in (2.0, -1.0, 4.0, 0.0)]
    trumps = [0, 0, 1, 1]
    breakdown = summarise_by_trump_suit(pairs, trumps)  # type: ignore[arg-type]

    assert set(breakdown) == {SUIT_NAMES[0], SUIT_NAMES[1]}
    assert breakdown[SUIT_NAMES[0]]["paired_deal_count"] == 2.0
    assert breakdown[SUIT_NAMES[0]]["mean_paired_score_difference"] == pytest.approx(0.5)
    assert breakdown[SUIT_NAMES[0]]["win_rate"] == pytest.approx(0.5)
    assert breakdown[SUIT_NAMES[1]]["mean_paired_score_difference"] == pytest.approx(2.0)


def test_a_sampled_stage_evaluation_reports_every_trump_suit() -> None:
    result = evaluate_stage(
        stage=stage_four_sampled_contract(),
        policy_a=BasicHeuristicAgent(name="basic"),
        policy_b=RandomAgent(name="random", seed=3),
        paired_deal_count=40,
        evaluation_seed=DEVELOPMENT_EVALUATION_SEED,
        bootstrap_resamples=200,
    )
    assert set(result.trump_suit_breakdown) == {SUIT_NAMES[suit] for suit in range(NUM_SUITS)}
    total = sum(entry["paired_deal_count"] for entry in result.trump_suit_breakdown.values())
    assert total == 40


# --- the reported checkpoints are chosen in advance -------------------------


def test_the_evaluation_schedule_is_fixed_before_training_starts() -> None:
    assert planned_evaluation_updates(total_updates=144, every_updates=36) == {36, 72, 108, 144}
    assert planned_evaluation_updates(total_updates=100, every_updates=36) == {36, 72, 100}
    assert planned_evaluation_updates(total_updates=5, every_updates=10) == {5}


def test_the_schedule_always_includes_the_final_update() -> None:
    for total in (1, 7, 48, 144):
        assert total in planned_evaluation_updates(total, every_updates=36)


def test_an_invalid_schedule_is_refused() -> None:
    with pytest.raises(ValueError, match="every_updates"):
        planned_evaluation_updates(total_updates=10, every_updates=0)


# --- configuration bookkeeping ---------------------------------------------


def test_a_stage_config_must_agree_with_its_training_block() -> None:
    with pytest.raises(ValueError, match="requires 'strong'"):
        stage_configuration_from_dict(
            {
                "stage": {"name": "x", "curriculum_stage": 5, "opponent": "strong"},
                "training": {"training_opponent": "basic", "environment_root_seed": 1},
            }
        )


def test_an_unknown_config_section_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown stage config sections"):
        stage_configuration_from_dict({"stage": {"name": "x", "curriculum_stage": 2}, "extra": {}})


def test_a_config_without_a_stage_section_is_refused() -> None:
    with pytest.raises(ValueError, match="must contain a 'stage' section"):
        stage_configuration_from_dict({"training": {}})


def test_every_shipped_stage_config_loads_and_describes_itself() -> None:
    from pathlib import Path

    from tarneeb.curriculum.config import describe_stage_configuration

    for path in sorted(Path("configs/train").glob("stage*.yaml")):
        config, stage = load_stage_configuration(path)
        description = describe_stage_configuration(config, stage)
        assert description["stage_name"] == stage.name
        assert description["reward_mode"].startswith("frozen normalized deal reward")
        assert description["opponent"] == config.training_opponent
        assert description["history_condition"] == "full"


def test_the_a7_configs_differ_only_in_the_history_condition() -> None:
    """The controlled-experiment requirement, checked on the shipped files."""
    from pathlib import Path

    loaded = {
        path.stem: load_stage_configuration(path)
        for path in sorted(Path("configs/ablation").glob("a7_*.yaml"))
    }
    assert len(loaded) == 3

    conditions = set()
    reference_training: dict | None = None
    reference_stage: dict | None = None
    for _name, (config, stage) in loaded.items():
        conditions.add(stage.history_condition)

        training = config.as_dict()
        training.pop("run_name")
        if reference_training is None:
            reference_training = training
        else:
            assert training == reference_training, "training configuration must be identical"

        stage_fields = stage.as_dict()
        stage_fields.pop("name")
        stage_fields.pop("history_condition")
        if reference_stage is None:
            reference_stage = stage_fields
        else:
            assert stage_fields == reference_stage, "only history_condition may differ"

    assert conditions == {"full", "auction_removed", "current_bid_only"}


def test_the_a7_conditions_share_one_training_seed_and_budget() -> None:
    from pathlib import Path

    seeds, budgets = set(), set()
    for path in sorted(Path("configs/ablation").glob("a7_*.yaml")):
        config, _stage = load_stage_configuration(path)
        seeds.add(
            (
                config.environment_root_seed,
                config.initialization_seed,
                config.action_sampling_seed,
                config.minibatch_shuffle_seed,
                config.opponent_seed,
            )
        )
        budgets.add(
            (config.total_environment_steps, config.environment_count, config.rollout_length)
        )
    assert len(seeds) == 1
    assert len(budgets) == 1


def test_the_a7_conditions_share_one_evaluation_deal_set() -> None:
    from pathlib import Path

    evaluation_seeds = {
        load_stage_configuration(path)[0].evaluation.evaluation_seed
        for path in sorted(Path("configs/ablation").glob("a7_*.yaml"))
    }
    assert len(evaluation_seeds) == 1


def test_paired_differences_are_symmetric_under_relabelling() -> None:
    """Swapping which policy is A must negate the mean exactly.

    Both policies must be deterministic for this to be a statement about the
    arena. A seeded ``RandomAgent`` continues one stream across deals, so
    relabelling changes the order in which it consumes draws and therefore which
    cards it plays -- a property of that agent, not an asymmetry in the pairing.
    """
    from tarneeb.agents.heuristic_strong import StrongHeuristicAgent

    arguments = {
        "stage": stage_two_trick_play(),
        "paired_deal_count": 8,
        "evaluation_seed": DEVELOPMENT_EVALUATION_SEED,
        "bootstrap_resamples": 100,
    }
    forward = evaluate_stage(
        policy_a=StrongHeuristicAgent(name="strong"),
        policy_b=BasicHeuristicAgent(name="basic"),
        **arguments,  # type: ignore[arg-type]
    )
    reverse = evaluate_stage(
        policy_a=BasicHeuristicAgent(name="basic"),
        policy_b=StrongHeuristicAgent(name="strong"),
        **arguments,  # type: ignore[arg-type]
    )
    assert np.isfinite(forward.mean_paired_score_difference)
    assert forward.mean_paired_score_difference == pytest.approx(
        -reverse.mean_paired_score_difference, abs=1e-9
    )
