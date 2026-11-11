"""Phase 11 Part 3: league rollout can survive real recurrent PPO updates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.league.population import LeagueConfig, LeaguePopulation
from tarneeb.league.registry import PolicyRegistry, actor_fingerprint
from tarneeb.league.sampling import PFSPConfig
from tarneeb.models.recurrent import build_recurrent_actor_and_critic
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint, save_recurrent_checkpoint
from tarneeb.training.phase11_training import (
    Phase11A3Config,
    Phase11SmokeConfig,
    Phase11SmokeResult,
    _final_checkpoint_suffix,
    _save_final_checkpoint_if_due,
    _validate_a3_population,
    build_phase11_initial_population,
    build_phase11_networks,
    build_phase11_opponent_provider,
    run_phase11_smoke,
    write_phase11_smoke_report,
)
from tarneeb.training.ppo import PPOConfig


def write_historical_checkpoint(path: Path, seed: int) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=seed)
    save_recurrent_checkpoint(
        path=path,
        actor=actor,
        critic=critic,
        optimizer=None,
        rules=RulesConfig(),
        training_config={"source": "phase11_part3_test"},
        global_environment_steps=0,
        completed_updates=0,
        wall_clock_seconds=0.0,
        seeds={"seed_index": seed},
    )


def build_population(tmp_path: Path, *, snapshot_every_update: bool = True) -> LeaguePopulation:
    checkpoint = tmp_path / "historical.pt"
    write_historical_checkpoint(checkpoint, seed=91)
    registry = PolicyRegistry()
    policy_id = registry.register_checkpoint(
        "historical",
        checkpoint,
        source_phase="phase10",
        category="champion",
        seed=0,
        update=250,
    )
    registry.register_anchor("strong", "strong")
    registry.register_anchor("random", "random")
    pfsp = PFSPConfig(
        ema_beta=0.5,
        unobserved_learner_win_rate=0.5,
        empty_category_strategy="proportional",
    )
    population = LeaguePopulation(
        registry=registry,
        config=LeagueConfig(
            pfsp=pfsp,
            snapshot_updates=1 if snapshot_every_update else 50,
            snapshot_environment_steps=10**9,
        ),
        rng=np.random.default_rng(1234),
    )
    population.add(policy_id, make_latest=True)
    population.add("strong", make_latest=False)
    population.add("random", make_latest=False)
    return population


def smoke_config(tmp_path: Path, condition: str, *, updates: int = 1) -> Phase11SmokeConfig:
    return Phase11SmokeConfig(
        run_name=f"phase11_{condition}_test",
        condition=condition,
        environment_count=2,
        rollout_length=1,
        total_updates=updates,
        initialization_seed=17,
        action_sampling_seed=18,
        minibatch_shuffle_seed=19,
        opponent_seed=20,
        checkpoint_directory=tmp_path / "checkpoints",
        results_directory=tmp_path / "results",
        ppo=PPOConfig(
            anneal_learning_rate=False,
            update_epochs=1,
            minibatch_count=1,
        ),
    )


def test_latest_only_completes_a_real_ppo_update_at_a_g6_boundary(tmp_path: Path) -> None:
    population = build_population(tmp_path)
    config = smoke_config(tmp_path, "latest_only")
    initial_actor, _critic = build_phase11_networks(config)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in initial_actor.state_dict().items()
    }

    result = run_phase11_smoke(config, population, progress=False)

    assert result.completed_updates == 1
    assert result.completed_deals > 0
    assert result.illegal_action_count == 0
    assert result.no_trump_action_count == 0
    assert result.non_finite_update_count == 0
    assert result.worst_epoch_zero_deviation < 1e-4
    assert result.update_records[0]["g6_boundary_valid"] is True
    assert result.snapshot_policy_ids
    assert config.league_state_path.is_file()

    loaded = load_recurrent_checkpoint(Path(result.checkpoint_paths[-1]), rules=RulesConfig())
    assert any(
        not torch.equal(initial_state[name], tensor)
        for name, tensor in loaded.actor.state_dict().items()
    )


def test_pfsp_updates_opponent_statistics_and_registers_a_snapshot(tmp_path: Path) -> None:
    population = build_population(tmp_path)
    config = smoke_config(tmp_path, "pfsp")

    result = run_phase11_smoke(config, population, progress=False)

    assert result.completed_updates == 1
    assert result.snapshot_policy_ids
    assert population.snapshot_count == 1
    assert population.latest_policy_id == result.snapshot_policy_ids[-1]
    observed = [stats.deals_observed for stats in population.performance.values()]
    assert observed
    assert sum(observed) == len(result.update_records[0]["completed_deal_outcomes"])
    assert sum(observed) > 0
    assert all(record["g6_boundary_valid"] for record in result.update_records)
    assert result.illegal_action_count == 0
    assert result.no_trump_action_count == 0


def test_new_snapshot_can_be_used_by_the_next_latest_only_update(tmp_path: Path) -> None:
    population = build_population(tmp_path)
    config = smoke_config(tmp_path, "latest_only", updates=2)

    result = run_phase11_smoke(config, population, progress=False)

    first_snapshot = result.snapshot_policy_ids[0]
    assert first_snapshot in result.update_records[1]["opponent_policy_ids"]
    assert result.completed_updates == 2
    assert len(result.snapshot_policy_ids) == 2


def test_smoke_report_contains_separate_league_state(tmp_path: Path) -> None:
    population = build_population(tmp_path)
    config = smoke_config(tmp_path, "latest_only")
    result = run_phase11_smoke(config, population, progress=False)

    report = write_phase11_smoke_report(config, result, population)

    assert report.is_file()
    restored = LeaguePopulation.load(config.league_state_path)
    assert restored.latest_policy_id == population.latest_policy_id
    assert restored.snapshot_count == population.snapshot_count


def test_principal_a3_settings_are_frozen() -> None:
    for condition in ("latest_only", "pfsp"):
        config = Phase11A3Config(condition=condition, seed_index=2)
        assert config.environment_root_seed == 2_200_000
        assert (config.environment_count, config.rollout_length, config.total_updates) == (
            16,
            128,
            250,
        )
        assert config.chunk_length == 16
        assert config.ppo == PPOConfig()
        assert config.league.pfsp.ema_beta == 2 / 501
        assert config.league.diversity_minimum_distance is None
        assert config.final_checkpoint_update == 250
        assert config.write_final_checkpoint is True


def test_principal_a3_rejects_disabling_final_checkpoint() -> None:
    with pytest.raises(ValueError, match="final checkpoint is mandatory"):
        Phase11A3Config(
            condition="latest_only",
            seed_index=0,
            write_final_checkpoint=False,
        )


@pytest.mark.parametrize(
    "league",
    [
        LeagueConfig(promotion_min_duplicated_deals=100),
        LeagueConfig(promotion_confidence_level=0.90),
    ],
)
def test_principal_a3_rejects_nonfrozen_promotion_thresholds(league: LeagueConfig) -> None:
    with pytest.raises(ValueError, match=r"promotion.*settings are frozen"):
        Phase11A3Config(condition="pfsp", seed_index=0, league=league)


def test_update_250_is_the_only_principal_final_checkpoint_selection(
    tmp_path: Path,
) -> None:
    config = Phase11A3Config(
        condition="latest_only",
        seed_index=0,
        checkpoint_directory=tmp_path / "checkpoints",
        results_directory=tmp_path / "results",
    )
    assert _final_checkpoint_suffix(config, 249) is None
    assert _final_checkpoint_suffix(config, 250) == "final_u250"

    actor, critic = build_phase11_networks(config)
    from tarneeb.training.recurrent_ppo import build_recurrent_optimizer

    optimizer = build_recurrent_optimizer(actor, critic, config.ppo)
    result = Phase11SmokeResult(run_name=config.run_name, condition=config.condition)
    result.completed_updates = 249
    assert (
        _save_final_checkpoint_if_due(config, result, actor, critic, optimizer, RulesConfig())
        is None
    )
    result.completed_updates = 250
    final_checkpoint = _save_final_checkpoint_if_due(
        config, result, actor, critic, optimizer, RulesConfig()
    )

    assert final_checkpoint is not None
    assert final_checkpoint.name == "phase11_a3_final_u250.pt"
    payload = load_recurrent_checkpoint(final_checkpoint).payload
    assert payload["training"]["completed_updates"] == 250


def test_principal_training_entry_rejects_population_with_different_promotion_config(
    tmp_path: Path, monkeypatch
) -> None:
    import tarneeb.training.phase11_training as phase11_module

    config = Phase11A3Config(condition="pfsp", seed_index=0)
    population = LeaguePopulation(config=LeagueConfig(promotion_min_duplicated_deals=100))
    with pytest.raises(ValueError, match="population LeagueConfig must match"):
        _validate_a3_population(config, population)

    def should_not_start_training(_config: object) -> None:
        pytest.fail("principal training setup proceeded with an unvalidated population")

    monkeypatch.setattr(phase11_module, "build_phase11_networks", should_not_start_training)
    with pytest.raises(ValueError, match="population LeagueConfig must match"):
        run_phase11_smoke(config, population, progress=False)


def test_latest_only_starts_from_exact_learner_u0_actor(tmp_path: Path) -> None:
    config = Phase11A3Config(
        condition="latest_only",
        seed_index=0,
        checkpoint_directory=tmp_path / "checkpoints",
        results_directory=tmp_path / "results",
    )
    actor, critic = build_phase11_networks(config)
    population = build_phase11_initial_population(config, actor, critic)
    latest = build_phase11_opponent_provider("latest_only", population)
    u0_fingerprint = actor_fingerprint(
        load_recurrent_checkpoint(config.u0_checkpoint_path).actor.state_dict()
    )

    assert latest.sample() == config.u0_policy_id
    assert population.registry.records[latest.sample()].actor_fingerprint == u0_fingerprint
    assert population.snapshot_count == 1


def _resume_test_config(tmp_path: Path, condition: str) -> Phase11SmokeConfig:
    return smoke_config(tmp_path, condition, updates=2)


def _resume_checkpoint(config: Phase11SmokeConfig, update: int = 1) -> Path:
    return config.checkpoint_directory / f"{config.run_name}_resume_u{update}.pt"


def _without_wall_clock_metrics(records: list[dict]) -> list[dict]:
    return [
        {key: value for key, value in record.items() if key != "collection_seconds"}
        for record in records
    ]


def test_uninterrupted_and_g6_stop_resume_runs_match_exactly(tmp_path: Path) -> None:
    reference_config = _resume_test_config(tmp_path / "reference", "latest_only")
    reference = run_phase11_smoke(
        reference_config, build_population(tmp_path / "reference"), progress=False
    )

    resumed_config = _resume_test_config(tmp_path / "resumed", "latest_only")
    first_segment = run_phase11_smoke(
        resumed_config,
        build_population(tmp_path / "resumed"),
        progress=False,
        stop_after_updates=1,
    )
    assert first_segment.completed_updates == 1
    assert _resume_checkpoint(resumed_config).is_file()
    resumed = run_phase11_smoke(
        resumed_config,
        build_population(tmp_path / "resumed"),
        progress=False,
        resume_from=_resume_checkpoint(resumed_config),
    )

    assert resumed.completed_updates == 2
    assert resumed.total_environment_steps == reference.total_environment_steps
    assert _without_wall_clock_metrics(resumed.update_records) == _without_wall_clock_metrics(
        reference.update_records
    )
    assert resumed.evaluations == reference.evaluations


def test_corrupt_action_rng_resume_diverges(tmp_path: Path) -> None:
    reference_config = _resume_test_config(tmp_path / "reference", "latest_only")
    reference = run_phase11_smoke(
        reference_config,
        build_population(tmp_path / "reference"),
        progress=False,
    )
    config = _resume_test_config(tmp_path, "latest_only")
    run_phase11_smoke(
        config,
        build_population(tmp_path),
        progress=False,
        stop_after_updates=1,
    )
    checkpoint = _resume_checkpoint(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["resume"]["action_sampling_generator_state"] = (
        torch.Generator().manual_seed(9_999).get_state()
    )
    corrupt = tmp_path / "corrupt_action_rng.pt"
    torch.save(payload, corrupt)

    resumed = run_phase11_smoke(
        config,
        build_population(tmp_path),
        progress=False,
        resume_from=corrupt,
    )
    assert (
        resumed.update_records[1]["action_trace_fingerprint"]
        != reference.update_records[1]["action_trace_fingerprint"]
    )


def test_corrupt_pfsp_rng_resume_diverges(tmp_path: Path) -> None:
    reference_config = _resume_test_config(tmp_path / "reference", "pfsp")
    reference = run_phase11_smoke(
        reference_config,
        build_population(tmp_path / "reference"),
        progress=False,
    )
    config = _resume_test_config(tmp_path, "pfsp")
    run_phase11_smoke(
        config,
        build_population(tmp_path),
        progress=False,
        stop_after_updates=1,
    )
    payload = torch.load(_resume_checkpoint(config), map_location="cpu", weights_only=False)
    payload["resume"]["population_state"]["rng_state"] = np.random.default_rng(
        9_998
    ).bit_generator.state
    corrupt = tmp_path / "corrupt_pfsp_rng.pt"
    torch.save(payload, corrupt)

    resumed = run_phase11_smoke(
        config,
        build_population(tmp_path),
        progress=False,
        resume_from=corrupt,
    )
    assert (
        resumed.update_records[1]["opponent_policy_ids"]
        != reference.update_records[1]["opponent_policy_ids"]
    )


def test_missing_or_incompatible_resume_payload_is_refused(tmp_path: Path) -> None:
    config = _resume_test_config(tmp_path, "latest_only")
    run_phase11_smoke(
        config,
        build_population(tmp_path),
        progress=False,
        stop_after_updates=1,
    )
    payload = torch.load(_resume_checkpoint(config), map_location="cpu", weights_only=False)
    del payload["resume"]["action_sampling_generator_state"]
    missing = tmp_path / "missing_resume_field.pt"
    torch.save(payload, missing)

    with pytest.raises(ValueError, match="incomplete or incompatible"):
        run_phase11_smoke(
            config,
            build_population(tmp_path),
            progress=False,
            resume_from=missing,
        )

    payload["resume"]["format_version"] = "unknown"
    incompatible = tmp_path / "incompatible_resume.pt"
    torch.save(payload, incompatible)
    with pytest.raises(ValueError, match="incompatible Phase 11 resume payload"):
        run_phase11_smoke(
            config,
            build_population(tmp_path),
            progress=False,
            resume_from=incompatible,
        )
