"""The Phase 9 training driver, and the A1 fairness properties it must guarantee.

The driver runs both A1 conditions. That is only worth doing if the two really do
receive the same treatment, so most of these tests are about *matching*: same
budget, same seeds, same evaluation protocol, same PPO configuration, same
opponents. A difference in any of those would become a confound and A1 would be
measuring the harness instead of the architecture.

The rest check the safety properties that make a training run reportable at all:
epoch-zero holds, no illegal action, action 64 unreachable, nothing non-finite,
and the evaluation path cannot disturb training state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.env.spaces import D_CRITIC, D_STATIC
from tarneeb.evaluation.a1 import (
    ConditionEvaluation,
    compare_paired,
    summarise_seed_stability,
)
from tarneeb.models.bc_checkpoint import save_bc_checkpoint
from tarneeb.models.nets import build_actor_and_critic
from tarneeb.models.recurrent import build_recurrent_actor_and_critic
from tarneeb.training.phase9_seeds import (
    FINAL_HELD_OUT_SEED,
    PPO_DEVELOPMENT_SEEDS,
    training_environment_seed,
)
from tarneeb.training.phase9_training import (
    BASELINE_OPPONENTS,
    Phase9Config,
    build_networks,
    build_opponents,
    evaluate_against_baselines,
    run_phase9_training,
    write_run_report,
)
from tarneeb.training.ppo import PPOConfig

RULES = RulesConfig()


def tiny(architecture: str, directory: Path, **overrides) -> Phase9Config:
    defaults = {
        "run_name": f"t_{architecture}",
        "architecture": architecture,
        "environment_count": 4,
        "rollout_length": 40,
        "total_updates": 2,
        "evaluation_every_updates": 2,
        "evaluation_paired_deals": 6,
        "evaluation_bootstrap_resamples": 50,
        "checkpoint_directory": directory,
        "results_directory": directory,
        "ppo": PPOConfig(update_epochs=1, minibatch_count=2),
    }
    return Phase9Config(**{**defaults, **overrides})


# --- the two conditions are matched -----------------------------------------


def test_both_architectures_receive_an_identical_budget(tmp_path: Path) -> None:
    """The central A1 fairness property."""
    gru = tiny("gru", tmp_path)
    ff = tiny("ff", tmp_path)
    assert gru.steps_per_update == ff.steps_per_update
    assert gru.total_environment_steps == ff.total_environment_steps
    assert gru.total_updates == ff.total_updates
    assert gru.ppo == ff.ppo
    assert gru.evaluation_opponents == ff.evaluation_opponents
    assert gru.evaluation_paired_deals == ff.evaluation_paired_deals


def test_both_architectures_receive_identical_seeds(tmp_path: Path) -> None:
    gru = tiny("gru", tmp_path, seed_index=1)
    ff = tiny("ff", tmp_path, seed_index=1)
    assert gru.seeds() == ff.seeds()
    assert gru.environment_root_seed == training_environment_seed(1)


def test_a_matched_pair_of_runs_shares_its_compute_contract(tmp_path: Path) -> None:
    """What A1 matches is the CONFIGURATION, not the realised counts.

    Realised steps, deals and decisions are policy-dependent: once the two
    architectures diverge they bid for different numbers of actions, throw hands
    in at different rates, and therefore sit at different points in a deal when
    the base budget runs out, which changes the drain. Asserting equality here
    would encode a promise the design deliberately does not make -- and the only
    ways to keep such a promise (truncating a deal, force-resetting, or granting
    extra updates) are exactly what G6 and the pre-registration forbid.

    So this test pins the contract both conditions really do share, and requires
    the realised counts to be *recorded* rather than equal.
    """
    configs = {architecture: tiny(architecture, tmp_path) for architecture in ("gru", "ff")}
    results = {
        architecture: run_phase9_training(config, progress=False)
        for architecture, config in configs.items()
    }

    gru_config, ff_config = configs["gru"], configs["ff"]
    assert gru_config.environment_count == ff_config.environment_count
    assert gru_config.rollout_length == ff_config.rollout_length
    assert gru_config.total_updates == ff_config.total_updates
    assert gru_config.environment_root_seed == ff_config.environment_root_seed
    assert gru_config.training_opponent == ff_config.training_opponent
    assert gru_config.ppo == ff_config.ppo
    assert gru_config.evaluation_every_updates == ff_config.evaluation_every_updates
    assert gru_config.evaluation_opponents == ff_config.evaluation_opponents

    for architecture, result in results.items():
        config = configs[architecture]
        # The same optimizer-update count, which is the budget that is matched.
        assert result.completed_updates == config.total_updates
        # Each condition drained independently, past the shared base budget.
        assert result.total_collection_steps >= config.rollout_length * config.total_updates
        assert result.learner_decisions > 0
        assert result.completed_deals > 0


def test_the_pre_registered_ppo_configuration_is_the_frozen_one() -> None:
    """Phase 9 changes no PPO mathematics."""
    config = PPOConfig()
    assert config.discount_factor == 1.0
    assert config.gae_lambda == 0.95
    assert config.clip_epsilon == 0.2
    assert config.learning_rate == 3e-4
    assert config.update_epochs == 4
    assert config.minibatch_count == 4


def test_the_parameter_counts_show_the_feedforward_path_is_larger(tmp_path: Path) -> None:
    """A GRU win could not be explained by capacity."""
    gru_actor, gru_critic = build_networks(tiny("gru", tmp_path))
    ff_actor, ff_critic = build_networks(tiny("ff", tmp_path))
    from tarneeb.models.nets import count_parameters

    assert count_parameters(gru_actor) == 956_617
    assert count_parameters(ff_actor) == 1_031_041
    assert count_parameters(gru_critic) == 940_169
    assert count_parameters(ff_critic) == 1_014_593
    assert count_parameters(ff_actor) > count_parameters(gru_actor)


# --- configuration ----------------------------------------------------------


def test_an_unknown_architecture_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="architecture must be"):
        tiny("transformer", tmp_path)


def test_self_play_is_not_a_selectable_opponent(tmp_path: Path) -> None:
    """Phase 7 measured self-play collapsing to all-pass; it is out of scope."""
    assert set(BASELINE_OPPONENTS) == {"random", "basic", "strong"}
    with pytest.raises(ValueError, match="training_opponent must be one of"):
        tiny("gru", tmp_path, training_opponent="self_play")


def test_one_opponent_instance_exists_per_environment(tmp_path: Path) -> None:
    config = tiny("gru", tmp_path, environment_count=5)
    opponents = build_opponents(config)
    assert len(opponents) == 5
    assert len({id(opponent) for opponent in opponents}) == 5


def test_training_seeds_never_reach_the_held_out_range(tmp_path: Path) -> None:
    for seed_index in range(3):
        config = tiny("gru", tmp_path, seed_index=seed_index)
        assert config.environment_root_seed < 10**8
        assert config.environment_root_seed != FINAL_HELD_OUT_SEED


def test_scheduled_evaluation_uses_development_seeds_only(tmp_path: Path) -> None:
    """The held-out seed must not be consulted during training."""
    config = tiny("gru", tmp_path)
    actor, _critic = build_networks(config)
    entry = evaluate_against_baselines(actor, config, completed_updates=1)
    assert entry["evaluation_seed"] == PPO_DEVELOPMENT_SEEDS.start
    assert entry["evaluation_seed"] != FINAL_HELD_OUT_SEED


# --- running ----------------------------------------------------------------


@pytest.fixture(scope="module")
def gru_run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gru_run")
    config = tiny("gru", directory)
    return config, run_phase9_training(config, progress=False)


def test_a_run_completes_and_records_its_budget(gru_run) -> None:
    """Actual collection EXCEEDS the nominal budget, by design.

    The G6-safe collector finishes every active deal before updating rather than
    truncating one to hit a round number, so the realised step count is the
    budget plus a drain. The nominal figure is the trigger, not the total.
    """
    config, result = gru_run
    assert result.completed_updates == config.total_updates
    assert result.total_environment_steps > config.total_environment_steps
    assert result.total_collection_steps >= config.rollout_length * config.total_updates
    assert result.learner_decisions > 0
    assert len(result.update_records) == config.total_updates
    for record in result.update_records:
        assert record["drain_steps"] > 0
        assert record["collection_steps"] >= config.rollout_length


def test_a_run_stays_numerically_safe(gru_run) -> None:
    _config, result = gru_run
    assert result.illegal_action_count == 0
    assert result.no_trump_action_count == 0
    assert result.non_finite_update_count == 0
    for record in result.update_records:
        for field in ("policy_loss", "value_loss", "entropy", "gradient_norm"):
            assert np.isfinite(record[field]), field


def test_the_epoch_zero_gate_holds_throughout(gru_run) -> None:
    _config, result = gru_run
    assert result.worst_epoch_zero_deviation < 1e-4
    for record in result.update_records:
        assert record["epoch_zero_deviation"] < 1e-4


def test_recurrent_diagnostics_are_recorded(gru_run) -> None:
    """Events per decision, segment length and reset count are reportable."""
    _config, result = gru_run
    for record in result.update_records:
        assert record["events_per_decision"] > 0
        assert 0 < record["max_segment_length"] <= 16
        assert record["new_hand_boundaries"] >= 0
        assert record["chunk_count"] > 0


def test_bidding_diagnostics_are_recorded(gru_run) -> None:
    """Phase 7's failure showed up first as a pass rate; it must be visible."""
    _config, result = gru_run
    for record in result.update_records:
        assert record["auction_decisions"] >= 0
        if record["auction_decisions"]:
            assert 0.0 <= record["pass_rate"] <= 1.0


def test_checkpoints_and_evaluations_are_written(gru_run) -> None:
    _config, result = gru_run
    assert result.checkpoint_paths
    assert all(Path(path).exists() for path in result.checkpoint_paths)
    assert result.evaluations
    entry = result.evaluations[-1]
    assert set(entry["opponents"]) == {"random", "basic", "strong"}
    for values in entry["opponents"].values():
        assert values["illegal_action_rate"] == 0.0
        assert values["no_trump_action_rate"] == 0.0
        assert values["replay_integrity_rate"] == 1.0


def test_the_run_report_is_written_and_json_safe(gru_run, tmp_path: Path) -> None:
    config, result = gru_run
    path = write_run_report(config, result)
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"]["architecture"] == "gru"
    assert payload["result"]["completed_updates"] == config.total_updates
    assert payload["epoch_zero_tolerance"] == 1e-4


def test_evaluation_does_not_disturb_training_mode(gru_run) -> None:
    """Evaluation must leave the networks exactly as it found them."""
    config, _result = gru_run
    actor, _critic = build_networks(config)
    actor.train()
    evaluate_against_baselines(actor, config, completed_updates=0, paired_deals=3)
    assert actor.training


def test_two_identical_runs_agree(tmp_path: Path) -> None:
    first = run_phase9_training(tiny("gru", tmp_path / "a"), progress=False)
    second = run_phase9_training(tiny("gru", tmp_path / "b"), progress=False)
    for left, right in zip(first.update_records, second.update_records, strict=True):
        assert left["policy_loss"] == pytest.approx(right["policy_loss"], abs=1e-9)
        assert left["entropy"] == pytest.approx(right["entropy"], abs=1e-9)


def test_a_different_training_seed_diverges(tmp_path: Path) -> None:
    """The negative control for the determinism test above."""
    first = run_phase9_training(tiny("gru", tmp_path / "a", seed_index=0), progress=False)
    second = run_phase9_training(tiny("gru", tmp_path / "b", seed_index=1), progress=False)
    assert first.update_records[-1]["policy_loss"] != pytest.approx(
        second.update_records[-1]["policy_loss"], abs=1e-9
    )


# --- BC initialization ------------------------------------------------------


def test_bc_initialization_loads_the_actor_and_leaves_the_critic_fresh(
    tmp_path: Path,
) -> None:
    """The Strong heuristic has no value function, so the critic is not cloned."""
    cloned_actor, _ = build_recurrent_actor_and_critic(seed=99)
    path = save_bc_checkpoint(
        path=tmp_path / "bc.pt",
        actor=cloned_actor,
        architecture="gru",
        rules=RULES,
        bc_config={},
        seed_split={},
        metrics={},
        dataset_metadata={},
    )
    config = tiny("gru", tmp_path, bc_checkpoint=path, initialization_seed=1)

    actor, critic = build_networks(config)

    for cloned, loaded in zip(cloned_actor.parameters(), actor.parameters(), strict=True):
        assert torch.equal(cloned, loaded)
    fresh_actor, fresh_critic = build_recurrent_actor_and_critic(seed=1)
    del fresh_actor
    for fresh, built in zip(fresh_critic.parameters(), critic.parameters(), strict=True):
        assert torch.equal(fresh, built)


def test_a_bc_checkpoint_for_the_wrong_architecture_is_refused(tmp_path: Path) -> None:
    ff_actor, _ = build_actor_and_critic(seed=1)
    path = save_bc_checkpoint(
        path=tmp_path / "bc_ff.pt",
        actor=ff_actor,
        architecture="ff",
        rules=RULES,
        bc_config={},
        seed_split={},
        metrics={},
        dataset_metadata={},
    )
    config = tiny("gru", tmp_path, bc_checkpoint=path)
    with pytest.raises(ValueError, match="but this run is"):
        build_networks(config)


def test_the_critic_reads_the_actor_safe_observation(tmp_path: Path) -> None:
    """692, never 856. The centralized critic is Phase 10."""
    for architecture in ("gru", "ff"):
        _actor, critic = build_networks(tiny(architecture, tmp_path))
        assert critic.spec.critic_observation_dim == D_STATIC == 692
        assert critic.spec.critic_observation_dim != D_CRITIC


# --- A1 statistics ----------------------------------------------------------


def test_pairing_requires_the_same_anchor_and_the_same_deals() -> None:
    gru = ConditionEvaluation("gru", "strong", (1.0, 2.0), {})
    other_anchor = ConditionEvaluation("ff", "basic", (1.0, 2.0), {})
    shorter = ConditionEvaluation("ff", "strong", (1.0,), {})

    with pytest.raises(ValueError, match="different anchors"):
        compare_paired(gru, other_anchor, bootstrap_seed=0)
    with pytest.raises(ValueError, match="pair counts differ"):
        compare_paired(gru, shorter, bootstrap_seed=0)


def test_a_clear_advantage_is_detected() -> None:
    gru = ConditionEvaluation("gru", "strong", tuple(float(x) for x in range(50)), {})
    ff = ConditionEvaluation("ff", "strong", tuple(float(x) - 3.0 for x in range(50)), {})

    comparison = compare_paired(gru, ff, bootstrap_seed=1)

    assert comparison.difference_mean == pytest.approx(3.0)
    assert comparison.interval_excludes_zero
    assert comparison.verdict == "GRU better"
    assert comparison.gru_better_pair_count == 50


def test_no_difference_is_reported_as_no_difference() -> None:
    """A null result must be an admissible conclusion, not a failure."""
    rng = np.random.default_rng(0)
    samples = rng.normal(size=200)
    gru = ConditionEvaluation("gru", "strong", tuple(samples), {})
    ff = ConditionEvaluation("ff", "strong", tuple(samples + rng.normal(scale=1e-9, size=200)), {})

    comparison = compare_paired(gru, ff, bootstrap_seed=2)

    assert not comparison.interval_excludes_zero
    assert comparison.verdict == "no detectable difference"


def test_a_disadvantage_is_reported_honestly() -> None:
    gru = ConditionEvaluation("gru", "strong", tuple(float(x) for x in range(50)), {})
    ff = ConditionEvaluation("ff", "strong", tuple(float(x) + 2.0 for x in range(50)), {})

    comparison = compare_paired(gru, ff, bootstrap_seed=3)

    assert comparison.verdict == "GRU worse"
    assert comparison.difference_mean == pytest.approx(-2.0)


def test_seed_stability_reports_spread_not_a_confidence_interval() -> None:
    summary = summarise_seed_stability([1.0, 2.0, 4.0])
    assert summary["mean"] == pytest.approx(7.0 / 3.0)
    assert summary["spread"] == pytest.approx(3.0)
    assert summarise_seed_stability([])["mean"] is None
