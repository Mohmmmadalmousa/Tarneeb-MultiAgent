"""League state, archive bounds, snapshot cadence and promotion criteria."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.league.population import (
    LeagueConfig,
    LeaguePopulation,
    PromotionEvaluation,
)
from tarneeb.league.registry import PolicyRecord, PolicyRegistry
from tarneeb.league.sampling import PFSPConfig
from tarneeb.models.recurrent import build_recurrent_actor_and_critic
from tarneeb.models.recurrent_checkpoint import save_recurrent_checkpoint


def make_record(policy_id: str, order: int) -> PolicyRecord:
    return PolicyRecord(
        policy_id=policy_id,
        checkpoint_path=f"{policy_id}.pt",
        rules_fingerprint=RulesConfig().fingerprint(),
        obs_spec_version="obs_v2",
        action_spec_version="act_v1",
        event_schema_version="event_v1",
        architecture_spec_version="event_gru_actor_critic_v1",
        checkpoint_format_version="ppo_gru_v1",
        buffer_contract_version="chunk_v2",
        source_phase="phase9",
        seed=order,
        update=order,
        source_commit=None,
        actor_fingerprint=f"sha256:{order:064x}",
        registration_order=order,
        category="snapshot",
        behavioral_fingerprint=(float(order), float(order % 3)),
    )


def test_population_save_resume_restores_sampling_rng(tmp_path: Path) -> None:
    population = LeaguePopulation(
        rng=np.random.default_rng(920),
        registry=PolicyRegistry(),
        config=LeagueConfig(pfsp=PFSPConfig(ema_beta=0.2)),
    )
    for order in range(1, 4):
        record = make_record(f"p{order}", order)
        population.registry.records[record.policy_id] = record
        population.add(record.policy_id, make_latest=True)
    population.update_performance("p3", learner_win_rate_against_opponent=0.8, ema_beta=0.2)
    population.rng.random()
    population.rng.random()
    expected_rng = np.random.default_rng(920)
    expected_rng.random(2)
    expected_next = expected_rng.random()
    state_file = tmp_path / "league-state.json"
    population.save(state_file)
    resumed = LeaguePopulation.load(state_file)

    assert resumed.latest_policy_id == "p3"
    assert resumed.eligible_policy_ids == ["p1", "p2", "p3"]
    assert resumed.config.pfsp.ema_beta == 0.2
    assert resumed.performance["p3"].learner_win_rate_against_opponent_ema == 0.8
    # The state file was captured after two draws; resume matches the next draw.
    assert resumed.rng.random() == expected_next


def test_snapshot_registration_rejects_unvalidated_policy_record() -> None:
    population = LeaguePopulation()
    with pytest.raises(TypeError):
        population.register_snapshot(make_record("forged", 1))  # type: ignore[call-arg]


def test_snapshot_registration_validates_checkpoint_compatibility(tmp_path: Path) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=81)
    checkpoint = tmp_path / "incompatible.pt"
    save_recurrent_checkpoint(
        checkpoint,
        actor,
        critic,
        None,
        RulesConfig(),
        {},
        0,
        0,
        0.0,
        {},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["buffer_contract_version"] = "not-chunk-v2"
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="buffer_contract_version"):
        LeaguePopulation().register_snapshot(
            policy_id="invalid",
            checkpoint_path=checkpoint,
            source_phase="phase11",
        )


def test_archive_uses_last_five_recent_and_at_most_ten_older() -> None:
    population = LeaguePopulation(rng=np.random.default_rng(4))
    for order in range(18):
        record = make_record(f"p{order}", order)
        population.registry.records[record.policy_id] = record
        population.add(record.policy_id)

    categories = population.sample_categories()
    assert categories["recent"] == tuple(f"p{order}" for order in range(13, 18))
    assert len(categories["older"]) == 10
    assert len(set(categories["older"])) == 10


def test_snapshot_cadence_fires_on_either_threshold() -> None:
    population = LeaguePopulation()
    assert not population.should_snapshot(1_999_999, 49)
    assert population.should_snapshot(2_000_000, 1)
    assert population.should_snapshot(1, 50)
    population.note_snapshot(2_000_000, 50)
    assert not population.should_snapshot(2_000_001, 51)


def test_promotion_gate_requires_all_hand_derived_conditions() -> None:
    population = LeaguePopulation()
    valid = PromotionEvaluation(
        paired_deal_count_vs_champion=2_000,
        mean_paired_score_difference_vs_champion=0.5,
        champion_paired_score_difference_ci_low=0.01,
        champion_paired_score_difference_ci_high=0.99,
        required_recent_snapshot_policy_ids=("snap-a", "snap-b", "snap-c"),
        mean_paired_score_difference_by_recent_snapshot={
            "snap-a": 0.0,
            "snap-b": 0.2,
            "snap-c": 0.1,
        },
        mean_paired_score_difference_vs_strong=0.1,
    )
    assert population.promotion_passes(valid)
    assert not population.promotion_passes(replace(valid, paired_deal_count_vs_champion=1_999))
    assert not population.promotion_passes(
        replace(valid, champion_paired_score_difference_ci_low=0.0)
    )
    assert not population.promotion_passes(
        replace(
            valid,
            mean_paired_score_difference_by_recent_snapshot={
                **valid.mean_paired_score_difference_by_recent_snapshot,
                "snap-b": -0.01,
            },
        )
    )
    assert not population.promotion_passes(
        replace(valid, mean_paired_score_difference_vs_strong=0.0)
    )


def test_promotion_gate_accepts_explicitly_zero_required_snapshots() -> None:
    population = LeaguePopulation()
    evaluation = PromotionEvaluation(2_000, 0.5, 0.01, 0.99, (), {}, 0.1)
    assert population.promotion_passes(evaluation)


def test_promotion_gate_supports_one_required_snapshot() -> None:
    population = LeaguePopulation()
    evaluation = PromotionEvaluation(2_000, 0.5, 0.01, 0.99, ("snap-1",), {"snap-1": 0.1}, 0.1)
    assert population.promotion_passes(evaluation)


def test_promotion_gate_supports_five_required_snapshots() -> None:
    population = LeaguePopulation()
    policy_ids = tuple(f"snap-{index}" for index in range(5))
    evaluation = PromotionEvaluation(
        2_000, 0.5, 0.01, 0.99, policy_ids, dict.fromkeys(policy_ids, 0.1), 0.1
    )
    assert population.promotion_passes(evaluation)


def test_successful_promotion_replaces_champion_and_retains_old_champion() -> None:
    registry = PolicyRegistry()
    registry.records["champion-a"] = replace(make_record("champion-a", 1), category="champion")
    registry.records["candidate-b"] = replace(make_record("candidate-b", 2), category="recent")
    population = LeaguePopulation(registry=registry)
    population.add("champion-a", make_latest=False)
    population.add("candidate-b", make_latest=False)
    evaluation = PromotionEvaluation(2_000, 0.5, 0.01, 0.99, (), {}, 0.1)

    population.promote("candidate-b", evaluation)

    champions = [
        record.policy_id
        for record in population.registry.records.values()
        if record.category == "champion"
    ]
    assert champions == ["candidate-b"]
    assert "champion-a" in population.eligible_policy_ids
    assert population.registry.records["champion-a"].category != "champion"


def test_promotion_gate_rejects_missing_required_snapshot_result() -> None:
    population = LeaguePopulation()
    evaluation = PromotionEvaluation(
        2_000, 0.5, 0.01, 0.99, ("snap-a", "snap-b"), {"snap-a": 0.1}, 0.1
    )
    assert not population.promotion_passes(evaluation)


def test_promotion_gate_rejects_candidate_losing_to_a_required_snapshot() -> None:
    population = LeaguePopulation()
    evaluation = PromotionEvaluation(2_000, 0.5, 0.01, 0.99, ("snap-a",), {"snap-a": -0.01}, 0.1)
    assert not population.promotion_passes(evaluation)


def test_population_performance_update_uses_consistent_win_rate_semantics() -> None:
    population = LeaguePopulation()
    assert (
        population.update_performance(
            "opponent-a", learner_win_rate_against_opponent=0.8, ema_beta=0.25
        )
        == 0.8
    )
    assert population.performance["opponent-a"].learner_win_rate_against_opponent_ema == 0.8
