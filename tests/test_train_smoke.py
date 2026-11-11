"""End-to-end training: it runs, it is reproducible, and evaluation is isolated.

These are correctness tests, not learning tests. A tiny run exercises every part
of the pipeline -- rollout, advantages, PPO, evaluation through the Phase 5
arena, checkpointing -- so a wiring break is caught before a long run is started.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.evaluation.arena import HELD_OUT_EVALUATION_SEEDS, TRAINING_SEEDS
from tarneeb.models.checkpoint import load_checkpoint
from tarneeb.training.config import (
    EvaluationSchedule,
    TrainingConfig,
    evaluation_seed_is_held_out,
    load_training_config,
    training_config_from_dict,
)
from tarneeb.training.ppo import PPOConfig
from tarneeb.training.train import evaluate_against_baseline, train


def tiny_config(tmp_path: Path, **overrides: object) -> TrainingConfig:
    config = TrainingConfig(
        run_name="smoke",
        environment_count=2,
        rollout_length=24,
        total_environment_steps=96,
        environment_root_seed=11,
        checkpoint_directory=tmp_path / "checkpoints",
        checkpoint_every_updates=2,
        ppo=PPOConfig(update_epochs=1, minibatch_count=2),
        evaluation=EvaluationSchedule(
            every_updates=2, paired_deal_count=3, bootstrap_resamples=100
        ),
    )
    return config.with_overrides(**overrides) if overrides else config


# --- configuration ----------------------------------------------------------


def test_the_shipped_configurations_load(tmp_path: Path) -> None:
    for name in ("ppo_ff_baseline", "ppo_ff_smoke"):
        config = load_training_config(Path("configs/train") / f"{name}.yaml")
        assert config.ppo.discount_factor == 1.0
        assert config.architecture.history_encoder == "ff"
        assert config.environment_root_seed in TRAINING_SEEDS


def test_an_unknown_configuration_key_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown training config keys"):
        training_config_from_dict({"learning_rate": 0.1})


def test_training_seeds_are_disjoint_from_the_held_out_evaluation_seeds() -> None:
    assert HELD_OUT_EVALUATION_SEEDS.start >= TRAINING_SEEDS.stop
    assert evaluation_seed_is_held_out(20250910)


def test_an_environment_seed_from_the_held_out_range_is_refused() -> None:
    with pytest.raises(ValueError, match="held-out"):
        TrainingConfig(environment_root_seed=HELD_OUT_EVALUATION_SEEDS.start)


def test_derived_sizes_are_what_they_claim() -> None:
    config = TrainingConfig(environment_count=8, rollout_length=16, total_environment_steps=1000)
    assert config.decisions_per_rollout == 128
    assert config.total_updates == 8  # ceil(1000 / 128)
    assert config.minibatch_size == 32


# --- the run itself ---------------------------------------------------------


def test_a_tiny_run_completes_and_records_everything(tmp_path: Path) -> None:
    result = train(tiny_config(tmp_path))

    assert result.total_environment_steps == 96
    assert len(result.update_records) == 2
    assert result.checkpoint_paths
    assert result.evaluation_records

    for record in result.update_records:
        assert record["illegal_action_count"] == 0
        assert record["no_trump_action_count"] == 0
        assert np.isfinite(record["policy_loss"])
        assert np.isfinite(record["value_loss"])
        assert np.isfinite(record["entropy"])
        assert np.isfinite(record["gradient_norm"])
        assert record["epoch_zero_log_probability_max_deviation"] < 1e-4


def test_metrics_leaving_training_are_plain_values(tmp_path: Path) -> None:
    result = train(tiny_config(tmp_path))
    for record in result.update_records:
        for key, value in record.items():
            assert not isinstance(value, torch.Tensor), key


def test_the_same_seed_reproduces_the_same_run(tmp_path: Path) -> None:
    first = train(tiny_config(tmp_path / "a"))
    second = train(tiny_config(tmp_path / "b"))

    for left, right in zip(first.update_records, second.update_records, strict=True):
        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl_divergence",
            "clip_fraction",
            "explained_variance",
            "completed_deals",
        ):
            assert left[key] == pytest.approx(right[key]), key


def test_a_different_seed_produces_a_different_run(tmp_path: Path) -> None:
    first = train(tiny_config(tmp_path / "a"))
    second = train(tiny_config(tmp_path / "b", initialization_seed=1234))
    assert first.update_records[0]["policy_loss"] != pytest.approx(
        second.update_records[0]["policy_loss"]
    )


def test_the_checkpoint_a_run_writes_can_be_loaded_back(tmp_path: Path) -> None:
    result = train(tiny_config(tmp_path))
    loaded = load_checkpoint(Path(result.checkpoint_paths[-1]))

    assert loaded.global_environment_steps == result.total_environment_steps
    assert loaded.payload["rules_fingerprint"] == RulesConfig().fingerprint()
    assert loaded.payload["seeds"]["environment_root_seed"] == 11
    assert loaded.critic is not None


# --- evaluation isolation ---------------------------------------------------


def test_evaluation_does_not_move_a_single_weight(tmp_path: Path) -> None:
    from tarneeb.models.nets import build_actor_and_critic

    actor, _critic = build_actor_and_critic(seed=1)
    actor.train()
    before = [parameter.detach().clone() for parameter in actor.parameters()]

    evaluate_against_baseline(
        actor=actor,
        opponent_name="random",
        paired_deal_count=3,
        evaluation_seed=20250910,
        bootstrap_resamples=100,
        deterministic_actions=True,
    )

    for original, after in zip(before, actor.parameters(), strict=True):
        assert torch.equal(original, after)
    assert actor.training, "training mode must be restored after evaluation"


def test_evaluation_consumes_no_global_random_state() -> None:
    from tarneeb.models.nets import build_actor_and_critic

    actor, _critic = build_actor_and_critic(seed=1)
    torch_state_before = torch.random.get_rng_state()
    numpy_state_before = np.random.get_state()

    evaluate_against_baseline(
        actor=actor,
        opponent_name="random",
        paired_deal_count=3,
        evaluation_seed=20250910,
        bootstrap_resamples=100,
        deterministic_actions=True,
    )

    assert torch.equal(torch_state_before, torch.random.get_rng_state())
    assert np.array_equal(numpy_state_before[1], np.random.get_state()[1])


def test_evaluation_is_reproducible(tmp_path: Path) -> None:
    from tarneeb.models.nets import build_actor_and_critic

    actor, _critic = build_actor_and_critic(seed=1)
    arguments = {
        "actor": actor,
        "opponent_name": "random",
        "paired_deal_count": 5,
        "evaluation_seed": 20250910,
        "bootstrap_resamples": 100,
        "deterministic_actions": True,
    }
    first = evaluate_against_baseline(**arguments)  # type: ignore[arg-type]
    second = evaluate_against_baseline(**arguments)  # type: ignore[arg-type]
    assert first["mean_paired_score_difference"] == pytest.approx(
        second["mean_paired_score_difference"]
    )
    assert first["win_rate"] == pytest.approx(second["win_rate"])


def test_an_unknown_baseline_is_refused() -> None:
    from tarneeb.models.nets import build_actor_and_critic

    actor, _critic = build_actor_and_critic(seed=1)
    with pytest.raises(ValueError, match="unknown baseline"):
        evaluate_against_baseline(
            actor=actor,
            opponent_name="superhuman",
            paired_deal_count=1,
            evaluation_seed=20250910,
            bootstrap_resamples=10,
            deterministic_actions=True,
        )
