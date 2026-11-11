"""The auction probe, and the stage 5B configuration it was built to watch.

Stage 6 demonstrated that whole-policy entropy cannot detect an auction collapse:
it read 0.125 while the auction policy sat at exactly 0.0 with pass probability
1.0. These tests pin the probe that closes that blind spot, and check that it is a
*measurement* only -- it must never touch the policy or enter the objective.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.core.actions import (
    BID_ACTION_START,
    BID_ACTION_STOP,
    NO_TRUMP_ACTION,
    PASS_ACTION,
)
from tarneeb.core.events import Phase
from tarneeb.core.state import team_of_seat
from tarneeb.curriculum.auction_diagnostics import (
    PROBE_SEED_BASE,
    SITUATION_ALL,
    SITUATION_DECLARE,
    SITUATION_OPEN,
    SITUATION_OPPONENT_HAS_BID,
    SITUATION_RAISE_AVAILABLE,
    auction_entropy,
    collect_auction_probe,
    flatten_probe_report,
    format_probe_report,
    measure_auction_policy,
)
from tarneeb.curriculum.config import load_stage_configuration
from tarneeb.curriculum.stages import (
    LEARNED_AUCTION,
    SHIPPED_STAGES,
    stage_five_b_auction_warmup,
)
from tarneeb.evaluation.arena import HELD_OUT_EVALUATION_SEEDS, TRAINING_SEEDS
from tarneeb.models.nets import build_actor_and_critic

UNIFORM_EIGHT_ACTION_ENTROPY = float(np.log(8))
UNIFORM_FOUR_ACTION_ENTROPY = float(np.log(4))


@pytest.fixture(scope="module")
def probe():
    return collect_auction_probe(deal_count=25)


# --- the probe collects real auction decisions ------------------------------


def test_the_probe_collects_learner_seat_auction_decisions(probe) -> None:
    assert len(probe) > 50
    for probe_state in probe.states:
        assert probe_state.phase in (Phase.BID, Phase.DECLARE)
        assert team_of_seat(probe_state.seat) == 0
        assert probe_state.legal_action_mask.shape == (65,)
        assert probe_state.legal_action_mask.any()


def test_the_probe_uses_development_seeds_never_held_out_ones(probe) -> None:
    assert probe.seed_base == PROBE_SEED_BASE
    for deal_index in range(probe.deal_count):
        match_seed = probe.seed_base + deal_index
        assert match_seed in TRAINING_SEEDS
        assert match_seed not in HELD_OUT_EVALUATION_SEEDS


def test_the_probe_never_offers_the_reserved_action(probe) -> None:
    for probe_state in probe.states:
        assert not probe_state.legal_action_mask[NO_TRUMP_ACTION]


def test_the_probe_covers_every_situation_the_report_names(probe) -> None:
    covered: set[str] = set()
    for probe_state in probe.states:
        covered.update(probe_state.situations)
    assert {
        SITUATION_ALL,
        SITUATION_OPEN,
        SITUATION_OPPONENT_HAS_BID,
        SITUATION_RAISE_AVAILABLE,
        SITUATION_DECLARE,
    } <= covered


def test_situation_membership_matches_the_recorded_auction_state(probe) -> None:
    for probe_state in probe.states:
        situations = probe_state.situations
        if probe_state.phase is Phase.DECLARE:
            assert situations == (SITUATION_DECLARE,)
            continue
        assert (SITUATION_OPEN in situations) == (probe_state.current_high_bid is None)
        assert (SITUATION_OPPONENT_HAS_BID in situations) == probe_state.high_bidder_is_opponent


def test_the_probe_is_deterministic() -> None:
    first = collect_auction_probe(deal_count=6)
    second = collect_auction_probe(deal_count=6)
    assert len(first) == len(second)
    for left, right in zip(first.states, second.states, strict=True):
        assert left.seat == right.seat
        assert left.phase is right.phase
        assert left.current_high_bid == right.current_high_bid
        assert np.array_equal(left.legal_action_mask, right.legal_action_mask)


# --- measuring an untrained policy -----------------------------------------


def test_a_freshly_initialized_policy_measures_as_nearly_uniform(probe) -> None:
    """The signature the diagnosis found at stage 6 entry, reproduced here."""
    actor, _critic = build_actor_and_critic(seed=1)
    report = measure_auction_policy(actor, probe)

    open_auction = report[SITUATION_OPEN]
    assert open_auction["count"] > 0
    assert open_auction["entropy"] == pytest.approx(UNIFORM_EIGHT_ACTION_ENTROPY, abs=0.02)
    assert open_auction["pass"] == pytest.approx(1 / 8, abs=0.02)
    for bid in range(7, 14):
        assert open_auction[f"bid_{bid}"] == pytest.approx(1 / 8, abs=0.02)

    declare = report[SITUATION_DECLARE]
    assert declare["entropy"] == pytest.approx(UNIFORM_FOUR_ACTION_ENTROPY, abs=0.02)
    assert declare["trump_total"] == pytest.approx(1.0, abs=1e-5)


def test_probabilities_sum_to_one_and_never_touch_the_reserved_action(probe) -> None:
    actor, _critic = build_actor_and_critic(seed=1)
    report = measure_auction_policy(actor, probe)
    for situation, entry in report.items():
        if entry.get("count", 0) == 0:
            continue
        total = entry["pass"] + entry["bid_total"] + entry["trump_total"] + entry["no_trump"]
        assert total == pytest.approx(1.0, abs=1e-4), situation
        assert entry["no_trump"] == 0.0, situation


def test_a_collapsed_auction_policy_is_detected(probe) -> None:
    """Force PASS and check the probe reports entropy 0 where aggregate would not."""
    actor, _critic = build_actor_and_critic(seed=1)
    with torch.no_grad():
        actor.policy_head.weight[PASS_ACTION].zero_()
        actor.policy_head.bias[BID_ACTION_START:BID_ACTION_STOP] = -50.0
        actor.policy_head.bias[PASS_ACTION] = 50.0

    report = measure_auction_policy(actor, probe)
    assert report[SITUATION_OPEN]["pass"] == pytest.approx(1.0, abs=1e-6)
    assert report[SITUATION_OPEN]["entropy"] == pytest.approx(0.0, abs=1e-6)
    assert report[SITUATION_ALL]["bid_total"] == pytest.approx(0.0, abs=1e-6)


# --- the probe is a measurement, not an intervention ------------------------


def test_measuring_changes_no_parameter_and_restores_training_mode(probe) -> None:
    actor, _critic = build_actor_and_critic(seed=1)
    actor.train()
    before = [parameter.detach().clone() for parameter in actor.parameters()]

    measure_auction_policy(actor, probe)

    for original, after in zip(before, actor.parameters(), strict=True):
        assert torch.equal(original, after)
    assert actor.training, "training mode must be restored"


def test_measuring_consumes_no_global_random_state(probe) -> None:
    actor, _critic = build_actor_and_critic(seed=1)
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()

    measure_auction_policy(actor, probe)

    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


def test_the_probe_is_not_referenced_by_the_ppo_objective() -> None:
    """The measured auction entropy must never enter the loss."""
    import ast
    from pathlib import Path

    from tarneeb.training import ppo as ppo_module

    source = Path(ppo_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any("auction_diagnostics" in name for name in imported)
    assert not any("curriculum" in name for name in imported)


# --- reporting helpers ------------------------------------------------------


def test_the_flattened_report_carries_the_scalars_a_reporter_can_plot(probe) -> None:
    actor, _critic = build_actor_and_critic(seed=1)
    report = measure_auction_policy(actor, probe)
    flattened = flatten_probe_report(report)

    assert f"auction_entropy_{SITUATION_OPEN}" in flattened
    assert f"auction_pass_probability_{SITUATION_OPEN}" in flattened
    assert f"auction_bid_probability_{SITUATION_OPEN}" in flattened
    assert all(isinstance(value, float) for value in flattened.values())


def test_auction_entropy_reads_the_requested_situation(probe) -> None:
    actor, _critic = build_actor_and_critic(seed=1)
    report = measure_auction_policy(actor, probe)
    assert auction_entropy(report) == pytest.approx(report[SITUATION_ALL]["entropy"])
    assert auction_entropy(report, SITUATION_OPEN) == pytest.approx(
        report[SITUATION_OPEN]["entropy"]
    )
    assert auction_entropy({"all_auction_decisions": {"count": 0}}) is None


def test_the_report_formats_without_raising(probe) -> None:
    actor, _critic = build_actor_and_critic(seed=1)
    lines = format_probe_report(measure_auction_policy(actor, probe))
    assert lines
    assert any("PASS=" in line for line in lines)


# --- the stage 5B specification --------------------------------------------


def test_stage_five_b_is_a_learned_auction_against_basic() -> None:
    stage = stage_five_b_auction_warmup()
    assert stage.name == "stage5b_auction_warmup"
    assert stage.auction_mode == LEARNED_AUCTION
    assert not stage.uses_scripted_auction
    assert stage.opponent == "basic"
    assert stage.history_condition == "full"
    assert stage.trick_shaping_weight == 0.0
    assert stage.reward_description == "frozen normalized deal reward"


def test_stage_five_b_is_registered() -> None:
    assert SHIPPED_STAGES["stage5b"] == stage_five_b_auction_warmup()


def test_the_stage_five_b_config_changes_only_opponent_and_entropy() -> None:
    """The controlled-experiment claim, checked against the stage 6 config."""
    warmup_config, warmup_stage = load_stage_configuration(Path("configs/train/stage5b.yaml"))
    stage6_config, stage6_stage = load_stage_configuration(Path("configs/train/stage6.yaml"))

    assert warmup_stage.auction_mode == stage6_stage.auction_mode == LEARNED_AUCTION
    assert warmup_stage.history_condition == stage6_stage.history_condition
    assert warmup_stage.trick_shaping_weight == stage6_stage.trick_shaping_weight == 0.0
    assert warmup_stage.opponent == "basic" and stage6_stage.opponent == "strong"

    warmup_ppo = warmup_config.ppo.as_dict()
    stage6_ppo = stage6_config.ppo.as_dict()
    assert warmup_ppo.pop("entropy_coefficient") == 0.03
    assert stage6_ppo.pop("entropy_coefficient") == 0.01
    assert warmup_ppo == stage6_ppo, "only the entropy coefficient may differ"

    assert warmup_config.architecture == stage6_config.architecture
    assert warmup_config.environment_count == stage6_config.environment_count
    assert warmup_config.rollout_length == stage6_config.rollout_length


def test_the_stage_five_b_budget_covers_the_original_collapse_window() -> None:
    """The original collapse became permanent at update 8; this must exceed it."""
    config, _stage = load_stage_configuration(Path("configs/train/stage5b.yaml"))
    assert config.total_updates == 96
    assert config.total_environment_steps == 786_432
    assert config.total_updates >= 12 * 8
    assert config.evaluation.every_updates == 8, "first evaluation must land on update 8"


def test_the_stage_five_b_training_seed_is_a_training_seed() -> None:
    config, _stage = load_stage_configuration(Path("configs/train/stage5b.yaml"))
    assert config.environment_root_seed in TRAINING_SEEDS
    assert config.environment_root_seed not in HELD_OUT_EVALUATION_SEEDS
    assert config.evaluation.evaluation_seed == 70000007, "development seed, not the final one"
