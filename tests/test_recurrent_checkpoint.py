"""Recurrent checkpoints, and the guarantee that the two families never mix.

A checkpoint is only meaningful next to the contracts it was trained under. The
recurrent family records different values in ``checkpoint_format_version``,
``architecture_spec_version`` and ``buffer_contract_version``, so each loader
refuses the other's file by construction rather than by convention.

The feedforward path is exercised here too: Phase 8 must not have changed how a
Phase 6 checkpoint loads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.env.spaces import K_MAX
from tarneeb.models.checkpoint import (
    BUFFER_CONTRACT_VERSION,
    CHECKPOINT_FORMAT_VERSION,
    IncompatibleCheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from tarneeb.models.nets import build_actor_and_critic, count_parameters
from tarneeb.models.recurrent import (
    RECURRENT_ARCHITECTURE_SPEC_VERSION,
    RecurrentArchitectureSpec,
    build_recurrent_actor_and_critic,
)
from tarneeb.models.recurrent_checkpoint import (
    RECURRENT_CHECKPOINT_FORMAT_VERSION,
    load_recurrent_checkpoint,
    recurrent_compatibility_metadata,
    restore_recurrent_optimizer,
    save_recurrent_checkpoint,
)
from tarneeb.training.ppo import PPOConfig, build_optimizer
from tarneeb.training.recurrent_ppo import build_recurrent_optimizer

RULES = RulesConfig()


def write_recurrent(path: Path, with_optimizer: bool = True):
    actor, critic = build_recurrent_actor_and_critic(seed=5)
    optimizer = build_recurrent_optimizer(actor, critic, PPOConfig()) if with_optimizer else None
    save_recurrent_checkpoint(
        path=path,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        rules=RULES,
        training_config={"run_name": "test"},
        global_environment_steps=128,
        completed_updates=2,
        wall_clock_seconds=1.0,
        seeds={"environment_root_seed": 0},
        chunk_length=16,
    )
    return actor, critic


def write_feedforward(path: Path):
    actor, critic = build_actor_and_critic(seed=5)
    save_checkpoint(
        path=path,
        actor=actor,
        critic=critic,
        optimizer=None,
        rules=RULES,
        training_config={"run_name": "ff"},
        global_environment_steps=64,
        completed_updates=1,
        wall_clock_seconds=1.0,
        seeds={"environment_root_seed": 0},
    )
    return actor, critic


# --- version identity -------------------------------------------------------


def test_the_two_checkpoint_formats_are_distinct() -> None:
    assert RECURRENT_CHECKPOINT_FORMAT_VERSION == "ppo_gru_v1"
    assert CHECKPOINT_FORMAT_VERSION == "ppo_ff_v1"
    assert RECURRENT_CHECKPOINT_FORMAT_VERSION != CHECKPOINT_FORMAT_VERSION


def test_recurrent_metadata_records_every_frozen_contract() -> None:
    metadata = recurrent_compatibility_metadata(RULES)
    assert metadata["rules_version"] == "tarneeb-classic-v1"
    assert metadata["rules_fingerprint"] == RULES.fingerprint()
    assert metadata["obs_spec_version"] == "obs_v2"
    assert metadata["action_spec_version"] == "act_v1"
    assert metadata["event_schema_version"] == "event_v1"
    assert metadata["architecture_spec_version"] == RECURRENT_ARCHITECTURE_SPEC_VERSION
    assert metadata["buffer_contract_version"] == "chunk_v2"
    assert metadata["buffer_contract_version"] != BUFFER_CONTRACT_VERSION


# --- round trip -------------------------------------------------------------


def test_a_recurrent_checkpoint_round_trips_exactly(tmp_path: Path) -> None:
    path = tmp_path / "recurrent.pt"
    actor, critic = write_recurrent(path)

    loaded = load_recurrent_checkpoint(path, rules=RULES)

    assert loaded.critic is not None
    for saved, restored in zip(actor.parameters(), loaded.actor.parameters(), strict=True):
        assert torch.equal(saved, restored)
    for saved, restored in zip(critic.parameters(), loaded.critic.parameters(), strict=True):
        assert torch.equal(saved, restored)
    assert loaded.architecture == actor.spec
    assert loaded.global_environment_steps == 128
    assert loaded.chunk_length == 16


def test_the_recurrent_checkpoint_records_the_architecture_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recurrent.pt"
    write_recurrent(path)

    payload = load_recurrent_checkpoint(path, rules=RULES).payload["recurrent"]

    assert payload["hidden_size"] == 256
    assert payload["gru_layer_count"] == 1
    assert payload["event_projection_dim"] == 64
    assert payload["event_embedding_dims"] == [16, 8, 8, 8, 32, 8, 8]
    assert payload["actor_observation_dim"] == 692
    assert payload["critic_observation_dim"] == 692
    assert payload["gru_reset"] == "deal"
    assert payload["chunk_length"] == 16
    assert payload["k_max"] == K_MAX


def test_the_recurrent_checkpoint_records_the_phase_eight_parameter_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recurrent.pt"
    write_recurrent(path)

    counts = load_recurrent_checkpoint(path, rules=RULES).payload["parameter_counts"]

    assert counts["actor"] == 956_617
    assert counts["critic"] == 940_169


def test_the_inference_path_can_refuse_to_load_the_critic(tmp_path: Path) -> None:
    """Never loading the critic makes a privileged value net unreachable at deploy."""
    path = tmp_path / "recurrent.pt"
    write_recurrent(path)

    loaded = load_recurrent_checkpoint(path, rules=RULES, load_critic=False)

    assert loaded.critic is None
    assert count_parameters(loaded.actor) == 956_617


def test_optimizer_state_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "recurrent.pt"
    write_recurrent(path)
    loaded = load_recurrent_checkpoint(path, rules=RULES)
    assert loaded.critic is not None

    optimizer = build_recurrent_optimizer(loaded.actor, loaded.critic, PPOConfig())
    restore_recurrent_optimizer(optimizer, loaded)

    assert optimizer.state_dict()["param_groups"]


def test_restoring_an_optimizer_from_a_checkpoint_without_one_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recurrent.pt"
    write_recurrent(path, with_optimizer=False)
    loaded = load_recurrent_checkpoint(path, rules=RULES)
    assert loaded.critic is not None

    with pytest.raises(IncompatibleCheckpointError, match="no optimizer state"):
        restore_recurrent_optimizer(
            build_recurrent_optimizer(loaded.actor, loaded.critic, PPOConfig()), loaded
        )


# --- cross-family refusal ---------------------------------------------------


def test_the_feedforward_loader_refuses_a_recurrent_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "recurrent.pt"
    write_recurrent(path)

    with pytest.raises(IncompatibleCheckpointError) as error:
        load_checkpoint(path, rules=RULES)

    message = str(error.value)
    assert "checkpoint_format_version" in message
    assert "architecture_spec_version" in message
    assert "buffer_contract_version" in message


def test_the_recurrent_loader_refuses_a_feedforward_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "feedforward.pt"
    write_feedforward(path)

    with pytest.raises(IncompatibleCheckpointError) as error:
        load_recurrent_checkpoint(path, rules=RULES)

    message = str(error.value)
    assert "ppo_ff_v1" in message
    assert "chunk_v2" in message


def test_a_feedforward_checkpoint_still_loads_through_its_own_loader(
    tmp_path: Path,
) -> None:
    """Phase 8 is additive: Phase 6 checkpoints must be completely unaffected."""
    path = tmp_path / "feedforward.pt"
    actor, _critic = write_feedforward(path)

    loaded = load_checkpoint(path, rules=RULES)

    for saved, restored in zip(actor.parameters(), loaded.actor.parameters(), strict=True):
        assert torch.equal(saved, restored)
    assert count_parameters(loaded.actor) == 1_031_041


def test_a_feedforward_optimizer_still_builds_over_the_frozen_networks() -> None:
    actor, critic = build_actor_and_critic(seed=1)
    optimizer = build_optimizer(actor, critic, PPOConfig())
    assert optimizer.param_groups


# --- contract drift refusal -------------------------------------------------


def test_a_changed_rules_fingerprint_is_refused(tmp_path: Path) -> None:
    """A version string can be left unchanged by accident; a fingerprint cannot."""
    path = tmp_path / "recurrent.pt"
    write_recurrent(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["rules_fingerprint"] = "sha1:0000000000000000000000000000000000000000"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="rules_fingerprint"):
        load_recurrent_checkpoint(path, rules=RULES)


def test_a_changed_buffer_contract_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "recurrent.pt"
    write_recurrent(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["buffer_contract_version"] = "chunk_v3"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="buffer_contract_version"):
        load_recurrent_checkpoint(path, rules=RULES)


def test_a_changed_observation_spec_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "recurrent.pt"
    write_recurrent(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["obs_spec_version"] = "obs_v3"
    torch.save(payload, path)

    with pytest.raises(IncompatibleCheckpointError, match="obs_spec_version"):
        load_recurrent_checkpoint(path, rules=RULES)


def test_a_checkpoint_from_a_different_hidden_size_rebuilds_at_its_own_size(
    tmp_path: Path,
) -> None:
    """The architecture travels with the weights, so a smaller net still loads."""
    path = tmp_path / "small.pt"
    spec = RecurrentArchitectureSpec(hidden_size=32, static_hidden_dim=64)
    actor, critic = build_recurrent_actor_and_critic(spec, seed=3)
    save_recurrent_checkpoint(
        path=path,
        actor=actor,
        critic=critic,
        optimizer=None,
        rules=RULES,
        training_config={},
        global_environment_steps=0,
        completed_updates=0,
        wall_clock_seconds=0.0,
        seeds={},
    )

    loaded = load_recurrent_checkpoint(path, rules=RULES)

    assert loaded.architecture.hidden_size == 32
    assert count_parameters(loaded.actor) == count_parameters(actor)
