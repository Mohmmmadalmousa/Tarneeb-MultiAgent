"""Checkpoint round-trip and the refusal to load an incompatible one.

A checkpoint carries the contracts it was trained under. Loading one produced
under different rules, a different observation layout or a different network
family would silently evaluate a model on a problem it never saw, so it must
fail with a message naming the mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tests.test_nets import random_history

from tarneeb.core.rules import RULES_VERSION, RulesConfig
from tarneeb.env.spaces import D_STATIC
from tarneeb.models.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    COMPATIBILITY_FIELDS,
    IncompatibleCheckpointError,
    compatibility_metadata,
    load_checkpoint,
    restore_optimizer,
    save_checkpoint,
)
from tarneeb.models.nets import ArchitectureSpec, build_actor_and_critic, count_parameters
from tarneeb.training.ppo import PPOConfig, build_optimizer

FROZEN_RULES_FINGERPRINT = "sha1:c18b43e7e9ebfc6855ff5fb56a968308d222a2f5"


def write_checkpoint(directory: Path, seed: int = 1) -> tuple[Path, object, object]:
    actor, critic = build_actor_and_critic(seed=seed)
    rules = RulesConfig()
    optimizer = build_optimizer(actor, critic, PPOConfig())
    path = save_checkpoint(
        path=directory / "checkpoint.pt",
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        rules=rules,
        training_config={"run_name": "test"},
        global_environment_steps=12_345,
        completed_updates=7,
        wall_clock_seconds=1.5,
        seeds={"initialization_seed": seed},
        evaluation={"mean_paired_score_difference": 1.25},
    )
    return path, actor, critic


def test_metadata_records_every_frozen_contract(tmp_path: Path) -> None:
    metadata = compatibility_metadata(RulesConfig())
    assert set(metadata) == set(COMPATIBILITY_FIELDS)
    assert metadata["rules_version"] == RULES_VERSION == "tarneeb-classic-v1"
    assert metadata["rules_fingerprint"] == FROZEN_RULES_FINGERPRINT
    assert metadata["obs_spec_version"] == "obs_v2"
    assert metadata["action_spec_version"] == "act_v1"
    assert metadata["event_schema_version"] == "event_v1"
    assert metadata["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION


def test_round_trip_restores_identical_weights_and_outputs(tmp_path: Path) -> None:
    path, actor, critic = write_checkpoint(tmp_path)
    loaded = load_checkpoint(path)

    for original, restored in zip(actor.parameters(), loaded.actor.parameters(), strict=True):
        assert torch.equal(original, restored)
    assert loaded.critic is not None
    for original, restored in zip(critic.parameters(), loaded.critic.parameters(), strict=True):
        assert torch.equal(original, restored)

    rng = np.random.default_rng(0)
    observations = torch.from_numpy(rng.random((3, D_STATIC), dtype=np.float32))
    history_ids, history_masks = random_history(3, rng)
    actor.eval()
    loaded.actor.eval()
    with torch.no_grad():
        assert torch.equal(
            actor(observations, history_ids, history_masks),
            loaded.actor(observations, history_ids, history_masks),
        )


def test_metadata_survives_the_round_trip(tmp_path: Path) -> None:
    path, actor, critic = write_checkpoint(tmp_path)
    loaded = load_checkpoint(path)

    assert loaded.global_environment_steps == 12_345
    assert loaded.payload["training"]["completed_updates"] == 7
    assert loaded.payload["parameter_counts"]["actor"] == count_parameters(actor)
    assert loaded.payload["parameter_counts"]["critic"] == count_parameters(critic)
    assert loaded.payload["rules_config"] == RulesConfig().as_dict()
    assert loaded.architecture == ArchitectureSpec()
    assert loaded.payload["seeds"]["initialization_seed"] == 1
    assert loaded.payload["evaluation"]["mean_paired_score_difference"] == 1.25
    assert "created_utc" in loaded.payload
    assert "git_sha" in loaded.payload


def test_the_inference_path_can_refuse_to_load_the_critic(tmp_path: Path) -> None:
    path, _actor, _critic = write_checkpoint(tmp_path)
    loaded = load_checkpoint(path, load_critic=False)
    assert loaded.critic is None


@pytest.mark.parametrize("field", COMPATIBILITY_FIELDS)
def test_a_changed_contract_field_is_refused(tmp_path: Path, field: str) -> None:
    path, _actor, _critic = write_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload[field] = "something-else"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match=field):
        load_checkpoint(path)


def test_an_edited_rule_is_caught_by_the_fingerprint_even_if_the_version_is_not_bumped(
    tmp_path: Path,
) -> None:
    """The version string can be forgotten. The fingerprint cannot."""
    path, _actor, _critic = write_checkpoint(tmp_path)
    edited_rules = RulesConfig(target_score=41)

    assert edited_rules.version == RULES_VERSION
    assert edited_rules.fingerprint() != FROZEN_RULES_FINGERPRINT
    with pytest.raises(IncompatibleCheckpointError, match="rules_fingerprint"):
        load_checkpoint(path, rules=edited_rules)


def test_optimizer_state_can_be_restored_for_a_resume(tmp_path: Path) -> None:
    path, _actor, _critic = write_checkpoint(tmp_path)
    loaded = load_checkpoint(path)
    assert loaded.critic is not None
    optimizer = build_optimizer(loaded.actor, loaded.critic, PPOConfig())
    restore_optimizer(optimizer, loaded)
    assert optimizer.state_dict()["param_groups"]


def test_a_checkpoint_without_optimizer_state_refuses_to_resume(tmp_path: Path) -> None:
    actor, critic = build_actor_and_critic(seed=2)
    path = save_checkpoint(
        path=tmp_path / "deploy_only.pt",
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
    loaded = load_checkpoint(path)
    optimizer = build_optimizer(actor, critic, PPOConfig())
    with pytest.raises(IncompatibleCheckpointError, match="optimizer"):
        restore_optimizer(optimizer, loaded)


def test_mismatched_actor_and_critic_specs_are_refused(tmp_path: Path) -> None:
    actor, _ = build_actor_and_critic(seed=3)
    from tarneeb.models.nets import CriticNetwork

    wider_critic = CriticNetwork(ArchitectureSpec(critic_observation_dim=856))
    with pytest.raises(ValueError, match="architecture specs"):
        save_checkpoint(
            path=tmp_path / "bad.pt",
            actor=actor,
            critic=wider_critic,
            optimizer=None,
            rules=RulesConfig(),
            training_config={},
            global_environment_steps=0,
            completed_updates=0,
            wall_clock_seconds=0.0,
            seeds={},
        )
