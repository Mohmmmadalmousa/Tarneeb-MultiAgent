"""Recurrent save/resume determinism, and the smoke PPO path end to end.

A resumed run must continue where it stopped, not merely start again from
similar weights. For a recurrent policy that requires restoring things a
feedforward run does not have: the hidden slots, the pending event queues, the
consumption cursors and the chunk positions. Leaving any of them out makes
either determinism or the epoch-zero identity fail on the first update after the
resume, which is exactly what these tests check.

Every positive claim is paired with a negative control that corrupts one piece
of the restored state and requires divergence. Without them, "the two runs
agree" could just mean the comparison is blind.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint
from tarneeb.training.recurrent_ppo import (
    RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
)
from tarneeb.training.recurrent_state import LEARNER_POLICY_ID
from tarneeb.training.recurrent_train import (
    RecurrentSmokeConfig,
    load_recurrent_smoke_config,
    recurrent_smoke_config_from_dict,
    run_recurrent_smoke,
    smoke_run_signature,
)

SMOKE_CONFIG_PATH = Path("configs/train/ppo_gru_smoke.yaml")


def smoke_config(directory: Path, **overrides) -> RecurrentSmokeConfig:
    """A deliberately tiny correctness harness, never a training run."""
    defaults = {
        "environment_count": 2,
        "rollout_length": 48,
        "total_updates": 2,
        "checkpoint_directory": directory,
        "checkpoint_every_updates": 1,
    }
    return RecurrentSmokeConfig(**{**defaults, **overrides})


# --- the smoke path ---------------------------------------------------------


def test_the_smoke_run_completes_and_stays_numerically_safe(tmp_path: Path) -> None:
    result = run_recurrent_smoke(smoke_config(tmp_path))

    assert result.completed_updates == 2
    assert result.total_environment_steps == 2 * 2 * 48
    assert result.learner_decision_count > 0
    assert result.illegal_action_count == 0
    assert result.no_trump_action_count == 0
    assert result.losses_are_finite()
    assert result.worst_epoch_zero_deviation < RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    assert len(result.checkpoint_paths) == 2
    assert all(path.exists() for path in result.checkpoint_paths)


def test_the_epoch_zero_gate_holds_on_every_update(tmp_path: Path) -> None:
    """Not only the first: each update recollects and must reconstruct exactly."""
    result = run_recurrent_smoke(smoke_config(tmp_path, total_updates=3))

    for metrics in result.update_metrics:
        assert (
            metrics.epoch_zero_log_probability_max_deviation
            < RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
        )
        assert metrics.optimizer_steps >= 1


def test_two_identically_configured_smoke_runs_agree_exactly(tmp_path: Path) -> None:
    first = run_recurrent_smoke(smoke_config(tmp_path / "a"))
    second = run_recurrent_smoke(smoke_config(tmp_path / "b"))

    assert smoke_run_signature(first) == smoke_run_signature(second)
    assert first.total_environment_steps == second.total_environment_steps


def test_a_different_seed_produces_a_different_run(tmp_path: Path) -> None:
    """The negative control for the determinism test above."""
    first = run_recurrent_smoke(smoke_config(tmp_path / "a"))
    second = run_recurrent_smoke(smoke_config(tmp_path / "b", environment_root_seed=99))

    assert smoke_run_signature(first) != smoke_run_signature(second)


# --- resume -----------------------------------------------------------------


def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path: Path) -> None:
    """The central resume claim, over the updates that follow the resume point."""
    uninterrupted = run_recurrent_smoke(smoke_config(tmp_path / "full", total_updates=3))

    partial = run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=1))
    assert partial.completed_updates == 1
    resumed = run_recurrent_smoke(
        smoke_config(tmp_path / "part", total_updates=3),
        resume_from=partial.checkpoint_paths[-1],
    )

    assert resumed.resumed_at_update == 1
    assert resumed.completed_updates == 3
    assert smoke_run_signature(resumed) == smoke_run_signature(uninterrupted)[1:]


def test_the_epoch_zero_gate_passes_on_the_first_update_after_a_resume(
    tmp_path: Path,
) -> None:
    """The sharpest test that the recurrent bookkeeping was restored correctly.

    If the pending queues or the consumption cursors came back wrong, the stored
    segments of the next rollout would be wrong and this would be the first thing
    to fail.
    """
    partial = run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=1))
    resumed = run_recurrent_smoke(
        smoke_config(tmp_path / "part", total_updates=2),
        resume_from=partial.checkpoint_paths[-1],
    )

    assert resumed.update_metrics
    assert (
        resumed.update_metrics[0].epoch_zero_log_probability_max_deviation
        < RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
    )


def test_the_resume_payload_carries_every_recurrent_quantity(tmp_path: Path) -> None:
    """docs/RECURRENT_CONTRACT.md section 9, checked field by field."""
    result = run_recurrent_smoke(smoke_config(tmp_path, total_updates=1))
    payload = load_recurrent_checkpoint(result.checkpoint_paths[-1]).payload["resume"]

    assert payload["completed_updates"] == 1
    assert payload["total_environment_steps"] > 0
    assert "action_sampling_generator_state" in payload
    assert "minibatch_shuffle_generator_state" in payload
    assert len(payload["environment_states"]) == 2
    assert len(payload["environment_deal_indices"]) == 2
    assert len(payload["environment_match_seeds"]) == 2
    assert len(payload["opponent_generator_states"]) == 2

    recurrent_state = payload["collector"]["recurrent_state"]
    assert recurrent_state["policy_ids"] == [LEARNER_POLICY_ID]
    assert recurrent_state["actor_hidden"][LEARNER_POLICY_ID].shape == (2, 4, 1, 256)
    assert recurrent_state["critic_hidden"][LEARNER_POLICY_ID].shape == (2, 4, 1, 256)
    assert len(recurrent_state["pending"]) == 2
    assert len(recurrent_state["pending"][0]) == 4
    assert len(recurrent_state["last_seq_consumed"]) == 2
    assert len(recurrent_state["chunk_position"]) == 2
    assert len(recurrent_state["force_new_chunk"]) == 2
    assert "redeal_counts" in payload["collector"]


def test_the_restored_hidden_state_is_the_saved_one(tmp_path: Path) -> None:
    partial = run_recurrent_smoke(smoke_config(tmp_path, total_updates=1))
    payload = load_recurrent_checkpoint(partial.checkpoint_paths[-1]).payload
    saved = payload["resume"]["collector"]["recurrent_state"]["actor_hidden"][LEARNER_POLICY_ID]

    assert torch.isfinite(saved).all()
    # A mid-rollout state is genuinely non-zero; a saved all-zero state would
    # mean the snapshot was taken from a freshly reset store.
    assert float(saved.abs().sum()) > 0.0


# --- negative controls ------------------------------------------------------


def corrupt_resume(path: Path, mutate) -> Path:
    """Load a checkpoint, mutate its resume payload, and write it back."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload["resume"])
    corrupted = path.with_name(path.stem + "_corrupted.pt")
    torch.save(payload, corrupted)
    return corrupted


def test_corrupting_the_saved_hidden_state_makes_the_resume_diverge(
    tmp_path: Path,
) -> None:
    """Without this, "resume matches" could mean the state is simply unused."""
    uninterrupted = run_recurrent_smoke(smoke_config(tmp_path / "full", total_updates=2))
    partial = run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=1))

    def scramble(resume: dict) -> None:
        hidden = resume["collector"]["recurrent_state"]["actor_hidden"]
        hidden[LEARNER_POLICY_ID] = hidden[LEARNER_POLICY_ID] + 1.0

    corrupted = corrupt_resume(partial.checkpoint_paths[-1], scramble)
    resumed = run_recurrent_smoke(
        smoke_config(tmp_path / "part", total_updates=2), resume_from=corrupted
    )

    assert smoke_run_signature(resumed) != smoke_run_signature(uninterrupted)[1:]


def test_corrupting_the_action_sampling_rng_makes_the_resume_diverge(
    tmp_path: Path,
) -> None:
    uninterrupted = run_recurrent_smoke(smoke_config(tmp_path / "full", total_updates=2))
    partial = run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=1))

    def scramble(resume: dict) -> None:
        generator = torch.Generator()
        generator.manual_seed(123456)
        resume["action_sampling_generator_state"] = generator.get_state()

    corrupted = corrupt_resume(partial.checkpoint_paths[-1], scramble)
    resumed = run_recurrent_smoke(
        smoke_config(tmp_path / "part", total_updates=2), resume_from=corrupted
    )

    assert smoke_run_signature(resumed) != smoke_run_signature(uninterrupted)[1:]


def test_corrupting_the_consumption_cursor_is_caught_by_the_collector(
    tmp_path: Path,
) -> None:
    """A wrong cursor is a lost or duplicated event, and must not pass silently."""
    partial = run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=1))

    def scramble(resume: dict) -> None:
        cursors = resume["collector"]["recurrent_state"]["last_seq_consumed"]
        cursors[0] = [value + 3 for value in cursors[0]]

    corrupted = corrupt_resume(partial.checkpoint_paths[-1], scramble)

    with pytest.raises(AssertionError, match="has consumed up to seq"):
        run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=2), resume_from=corrupted)


def test_resuming_from_a_checkpoint_without_resume_state_is_refused(
    tmp_path: Path,
) -> None:
    partial = run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=1))
    path = partial.checkpoint_paths[-1]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["resume"] = None
    stripped = path.with_name("stripped.pt")
    torch.save(payload, stripped)

    with pytest.raises(ValueError, match="cannot be resumed from"):
        run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=2), resume_from=stripped)


def test_resuming_into_a_different_environment_count_is_refused(
    tmp_path: Path,
) -> None:
    partial = run_recurrent_smoke(smoke_config(tmp_path / "part", total_updates=1))

    with pytest.raises(ValueError, match="environment"):
        run_recurrent_smoke(
            smoke_config(tmp_path / "part", total_updates=2, environment_count=3),
            resume_from=partial.checkpoint_paths[-1],
        )


# --- configuration ----------------------------------------------------------


def test_the_shipped_smoke_configuration_loads_and_matches_the_contract() -> None:
    config = load_recurrent_smoke_config(SMOKE_CONFIG_PATH)

    assert config.run_name == "ppo_gru_smoke"
    assert config.chunk_length == 16
    assert config.check_epoch_zero is True
    assert config.architecture.event_embedding_dims == (16, 8, 8, 8, 32, 8, 8)
    assert config.architecture.hidden_size == 256
    assert config.architecture.gru_layer_count == 1
    assert config.architecture.critic_observation_dim == 692
    assert config.architecture.gru_reset == "deal"
    # PPO mathematics is carried unchanged from the frozen Phase 6 configuration.
    assert config.ppo.discount_factor == 1.0
    assert config.ppo.gae_lambda == 0.95
    assert config.ppo.clip_epsilon == 0.2
    assert config.ppo.learning_rate == 3e-4


def test_an_unknown_configuration_key_is_refused() -> None:
    """A silently dropped hyper-parameter is a reproducibility bug."""
    with pytest.raises(ValueError, match="unknown recurrent smoke config keys"):
        recurrent_smoke_config_from_dict({"not_a_real_key": 1})


def test_self_play_is_not_a_selectable_opponent() -> None:
    """Four copies of the learner converge to an all-pass equilibrium; out of scope."""
    with pytest.raises(ValueError, match="training_opponent must be one of"):
        recurrent_smoke_config_from_dict({"training_opponent": "self_play"})


def test_the_shipped_configuration_runs(tmp_path: Path) -> None:
    config = load_recurrent_smoke_config(SMOKE_CONFIG_PATH)
    result = run_recurrent_smoke(
        RecurrentSmokeConfig(
            **{
                **{field: getattr(config, field) for field in RecurrentSmokeConfig.__slots__},
                "total_updates": 1,
                "checkpoint_directory": tmp_path,
            }
        )
    )

    assert result.completed_updates == 1
    assert result.illegal_action_count == 0
    assert result.no_trump_action_count == 0
    assert np.isfinite(result.update_metrics[0].policy_loss)
