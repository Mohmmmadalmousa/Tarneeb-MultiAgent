"""Partial continuation: what it restores, what it refuses to pretend, and the schedule.

The distinction this file defends is the one that makes the stage 4 continuation
experiment interpretable at all:

- ``initialize_from`` copies **weights only**. Optimizer, streams and the
  learning-rate schedule all restart.
- ``resume_training_state_from`` additionally restores the **optimizer** and
  **continues the learning-rate schedule**, but does *not* restore the
  environment or RNG streams, and says so in its provenance record.

Silently restarting the optimizer and calling that a continuation would make a
plateau experiment meaningless, so it is tested rather than trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarneeb.curriculum.config import load_stage_configuration
from tarneeb.curriculum.stages import stage_four_sampled_contract
from tarneeb.curriculum.trainer import train_curriculum_stage
from tarneeb.evaluation.arena import HELD_OUT_EVALUATION_SEEDS, TRAINING_SEEDS
from tarneeb.models.checkpoint import load_checkpoint
from tarneeb.training.config import EvaluationSchedule, TrainingConfig
from tarneeb.training.ppo import PPOConfig, linear_learning_rate

SOURCE_CHECKPOINT = Path("checkpoints/stage4_sampled_contract_update288.pt")


def tiny_config(checkpoint_directory: Path, updates: int = 2) -> TrainingConfig:
    rollout_length, environment_count = 24, 2
    return TrainingConfig(
        run_name="continuation_test",
        environment_count=environment_count,
        rollout_length=rollout_length,
        total_environment_steps=rollout_length * environment_count * updates,
        environment_root_seed=713_000,
        training_opponent="basic",
        checkpoint_directory=checkpoint_directory,
        checkpoint_every_updates=1_000_000,
        ppo=PPOConfig(update_epochs=1, minibatch_count=1),
        evaluation=EvaluationSchedule(
            every_updates=updates, paired_deal_count=2, bootstrap_resamples=50
        ),
    )


def stage() -> object:
    return stage_four_sampled_contract()


# --- the two modes are genuinely different ---------------------------------


def test_the_two_start_modes_cannot_both_be_used(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not both"):
        train_curriculum_stage(
            config=tiny_config(tmp_path),
            stage=stage(),  # type: ignore[arg-type]
            initialize_from=SOURCE_CHECKPOINT,
            resume_training_state_from=SOURCE_CHECKPOINT,
        )


def test_weights_only_initialization_records_no_continuation(tmp_path: Path) -> None:
    result = train_curriculum_stage(
        config=tiny_config(tmp_path),
        stage=stage(),  # type: ignore[arg-type]
        initialize_from=SOURCE_CHECKPOINT,
    )
    assert result.continuation is None
    assert result.schedule_offset_updates == 0
    assert result.initialized_from == str(SOURCE_CHECKPOINT)
    assert result.update_records[0]["effective_update"] == 1


# --- the optimizer really is restored ---------------------------------------


def test_the_partial_continuation_restores_adam_moments(tmp_path: Path) -> None:
    """The load-bearing test: a fresh optimizer would have zero state entries."""
    source = load_checkpoint(SOURCE_CHECKPOINT, rules=None, load_critic=True)
    stored_state = source.payload["optimizer_state_dict"]
    assert stored_state is not None
    assert len(stored_state["state"]) > 0, "the source checkpoint must carry Adam moments"

    result = train_curriculum_stage(
        config=tiny_config(tmp_path),
        stage=stage(),  # type: ignore[arg-type]
        resume_training_state_from=SOURCE_CHECKPOINT,
    )
    provenance = result.continuation
    assert provenance is not None
    assert provenance["optimizer_restored"] is True
    assert provenance["optimizer_state_entries"] == len(stored_state["state"])
    assert provenance["optimizer_state_entries"] > 0
    assert provenance["optimizer_parameter_groups"] == len(stored_state["param_groups"])


def test_the_restored_moments_are_the_source_moments(tmp_path: Path) -> None:
    """Not merely non-empty -- the same tensors."""
    from tarneeb.models.checkpoint import restore_optimizer
    from tarneeb.models.nets import build_actor_and_critic
    from tarneeb.training.ppo import build_optimizer

    loaded = load_checkpoint(SOURCE_CHECKPOINT, rules=None, load_critic=True)
    actor, critic = build_actor_and_critic(seed=1)
    optimizer = build_optimizer(actor, critic, PPOConfig())
    assert len(optimizer.state_dict()["state"]) == 0, "a fresh optimizer has no moments"

    restore_optimizer(optimizer, loaded)
    restored = optimizer.state_dict()["state"]
    stored = loaded.payload["optimizer_state_dict"]["state"]
    assert len(restored) == len(stored) > 0
    for key in stored:
        for moment in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(restored[key][moment], stored[key][moment])
        assert restored[key]["step"] == stored[key]["step"]


def test_a_checkpoint_without_optimizer_state_is_refused(tmp_path: Path) -> None:
    from tarneeb.core.rules import RulesConfig
    from tarneeb.models.checkpoint import save_checkpoint
    from tarneeb.models.nets import build_actor_and_critic

    actor, critic = build_actor_and_critic(seed=1)
    path = save_checkpoint(
        path=tmp_path / "no_optimizer.pt",
        actor=actor,
        critic=critic,
        optimizer=None,
        rules=RulesConfig(),
        training_config={},
        global_environment_steps=0,
        completed_updates=0,
        wall_clock_seconds=0.0,
        seeds={},
    )
    with pytest.raises(Exception, match="optimizer"):
        train_curriculum_stage(
            config=tiny_config(tmp_path / "run"),
            stage=stage(),  # type: ignore[arg-type]
            resume_training_state_from=path,
        )


# --- provenance never overstates what happened ------------------------------


def test_the_provenance_states_plainly_that_this_is_not_a_resume(tmp_path: Path) -> None:
    result = train_curriculum_stage(
        config=tiny_config(tmp_path),
        stage=stage(),  # type: ignore[arg-type]
        resume_training_state_from=SOURCE_CHECKPOINT,
    )
    provenance = result.continuation
    assert provenance is not None
    assert provenance["exact_resume"] is False
    assert provenance["environment_stream_restored"] is False
    assert provenance["rng_streams_restored"] is False
    assert provenance["source_effective_update"] == 288
    assert provenance["learning_rate_schedule_continued"] is True
    assert "not a deterministic resume" in provenance["disclosure"].lower()
    assert provenance["mode"] == (
        "weight + optimizer + LR-schedule continuation, fresh data/RNG stream"
    )
    assert provenance["fresh_continuation_seeds"]["environment_root_seed"] == 713_000


def test_the_continuation_continues_the_global_step_count(tmp_path: Path) -> None:
    result = train_curriculum_stage(
        config=tiny_config(tmp_path),
        stage=stage(),  # type: ignore[arg-type]
        resume_training_state_from=SOURCE_CHECKPOINT,
    )
    assert result.starting_environment_steps == 2_359_296
    assert result.update_records[0]["global_environment_steps"] > 2_359_296


# --- the learning-rate schedule -------------------------------------------


def test_the_continuation_does_not_restart_the_annealed_learning_rate(
    tmp_path: Path,
) -> None:
    result = train_curriculum_stage(
        config=tiny_config(tmp_path),
        stage=stage(),  # type: ignore[arg-type]
        resume_training_state_from=SOURCE_CHECKPOINT,
    )
    first = result.update_records[0]["learning_rate_scheduled"]
    assert first != pytest.approx(3e-4), "the anneal must not restart at the initial value"
    assert first < 3e-4
    assert result.update_records[0]["effective_update"] == 289


def test_the_schedule_is_expressed_in_effective_update_numbers(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, updates=3)
    result = train_curriculum_stage(
        config=config,
        stage=stage(),  # type: ignore[arg-type]
        resume_training_state_from=SOURCE_CHECKPOINT,
    )
    scheduled_total = 288 + config.total_updates
    assert result.scheduled_total_updates == scheduled_total
    for index, record in enumerate(result.update_records):
        expected = linear_learning_rate(3e-4, 288 + index, scheduled_total)
        assert record["learning_rate_scheduled"] == pytest.approx(expected)
        assert record["effective_update"] == 289 + index


def test_the_shipped_continuation_schedule_has_no_jump_within_its_own_anneal() -> None:
    """Across the 576-update schedule the boundary is one ordinary decrement."""
    notional_288 = linear_learning_rate(3e-4, 287, 576)
    continuation_289 = linear_learning_rate(3e-4, 288, 576)
    assert continuation_289 < notional_288
    assert abs(continuation_289 / notional_288 - 1) < 0.01

    # And it is nowhere near a naive restart.
    assert continuation_289 == pytest.approx(1.5e-4)
    assert continuation_289 < 3e-4 / 2 + 1e-12


# --- new checkpoints carry resume state ------------------------------------


def test_new_checkpoints_now_save_resume_state(tmp_path: Path) -> None:
    """The Phase 7 gap that made the original stage 4 checkpoint unresumable."""
    result = train_curriculum_stage(
        config=tiny_config(tmp_path),
        stage=stage(),  # type: ignore[arg-type]
        resume_training_state_from=SOURCE_CHECKPOINT,
    )
    assert result.checkpoint_paths
    payload = load_checkpoint(Path(result.checkpoint_paths[-1])).payload
    resume = payload["resume"]
    assert resume is not None, "new Phase 7 checkpoints must carry resume state"
    assert set(resume) == {
        "completed_updates",
        "total_environment_steps",
        "action_sampling_generator_state",
        "minibatch_shuffle_generator_state",
        "environment_states",
        "environment_deal_indices",
        "environment_match_seeds",
        "opponent_generator_states",
        "collector_redeal_counts",
    }
    assert resume["completed_updates"] == 288 + len(result.update_records)
    assert len(resume["environment_states"]) == 2


def test_the_original_stage_four_checkpoint_still_has_no_resume_state() -> None:
    """Fixing the trainer does not retroactively make the old checkpoint resumable."""
    payload = load_checkpoint(SOURCE_CHECKPOINT).payload
    assert payload["resume"] is None
    assert payload["optimizer_state_dict"] is not None


def test_checkpoint_filenames_use_effective_update_numbers(tmp_path: Path) -> None:
    """So a continuation cannot overwrite the run it continues from."""
    result = train_curriculum_stage(
        config=tiny_config(tmp_path),
        stage=stage(),  # type: ignore[arg-type]
        resume_training_state_from=SOURCE_CHECKPOINT,
    )
    assert Path(result.checkpoint_paths[-1]).stem.endswith("update290")


# --- the shipped continuation config ---------------------------------------


def test_the_continuation_config_keeps_the_stage_four_task_exactly() -> None:
    continuation_config, continuation_stage = load_stage_configuration(
        Path("configs/train/stage4_continuation.yaml")
    )
    original_config, original_stage = load_stage_configuration(Path("configs/train/stage4.yaml"))

    original_task = original_stage.as_dict()
    continuation_task = continuation_stage.as_dict()
    assert original_task.pop("name") != continuation_task.pop("name")
    assert continuation_task == original_task, "only the run label may differ"

    assert continuation_config.ppo.as_dict() == original_config.ppo.as_dict()
    assert continuation_config.architecture == original_config.architecture
    assert continuation_config.environment_count == original_config.environment_count
    assert continuation_config.rollout_length == original_config.rollout_length
    assert continuation_config.total_environment_steps == 2_359_296


def test_the_continuation_uses_fresh_recorded_training_seeds() -> None:
    continuation_config, _stage = load_stage_configuration(
        Path("configs/train/stage4_continuation.yaml")
    )
    original_config, _original = load_stage_configuration(Path("configs/train/stage4.yaml"))

    for field in (
        "environment_root_seed",
        "action_sampling_seed",
        "minibatch_shuffle_seed",
        "opponent_seed",
    ):
        assert getattr(continuation_config, field) != getattr(original_config, field), field
    assert continuation_config.environment_root_seed in TRAINING_SEEDS
    assert continuation_config.environment_root_seed not in HELD_OUT_EVALUATION_SEEDS


def test_the_continuation_evaluation_schedule_is_the_pre_registered_one() -> None:
    from tarneeb.curriculum.trainer import planned_evaluation_updates

    config, _stage = load_stage_configuration(Path("configs/train/stage4_continuation.yaml"))
    assert config.total_updates == 288
    assert config.evaluation.every_updates == 48
    local = sorted(
        planned_evaluation_updates(config.total_updates, config.evaluation.every_updates)
    )
    assert local == [48, 96, 144, 192, 240, 288]
    assert [288 + update for update in local] == [336, 384, 432, 480, 528, 576]
