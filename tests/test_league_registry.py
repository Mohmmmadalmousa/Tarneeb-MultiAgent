"""Phase 11 registry compatibility, provenance and actor deduplication."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.league.registry import PolicyRegistry
from tarneeb.models.recurrent import build_recurrent_actor_and_critic
from tarneeb.models.recurrent_checkpoint import save_recurrent_checkpoint


def write_checkpoint(path: Path, actor: torch.nn.Module, critic: torch.nn.Module) -> None:
    save_recurrent_checkpoint(
        path=path,
        actor=actor,  # type: ignore[arg-type]
        critic=critic,  # type: ignore[arg-type]
        optimizer=None,
        rules=RulesConfig(),
        training_config={},
        global_environment_steps=128,
        completed_updates=7,
        wall_clock_seconds=1.0,
        seeds={"seed_index": 3},
    )


def test_registry_round_trips_checkpoint_provenance(tmp_path: Path) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=41)
    checkpoint = tmp_path / "snapshot.pt"
    write_checkpoint(checkpoint, actor, critic)

    registry = PolicyRegistry()
    assert registry.register_checkpoint("policy-a", checkpoint, source_phase="phase9") == "policy-a"
    restored = PolicyRegistry.from_dict(registry.to_dict())

    record = restored.records["policy-a"]
    assert record.checkpoint_path == str(checkpoint.resolve())
    assert record.rules_fingerprint == RulesConfig().fingerprint()
    assert record.source_phase == "phase9"
    assert record.seed == 3
    assert record.update == 7
    assert record.actor_fingerprint.startswith("sha256:")
    assert restored.aliases[str(checkpoint.resolve())] == "policy-a"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("rules_fingerprint", "sha1:" + "0" * 40),
        ("obs_spec_version", "obs_v3"),
        ("action_spec_version", "act_v2"),
        ("event_schema_version", "event_v2"),
        ("architecture_spec_version", "other_model"),
        ("checkpoint_format_version", "other_checkpoint"),
        ("buffer_contract_version", "chunk_v3"),
    ],
)
def test_registry_rejects_incompatible_contracts(
    tmp_path: Path, field: str, bad_value: str
) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=42)
    checkpoint = tmp_path / "bad.pt"
    write_checkpoint(checkpoint, actor, critic)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload[field] = bad_value
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match=field):
        PolicyRegistry().register_checkpoint("bad", checkpoint, source_phase="phase9")


def test_bit_identical_actor_files_become_one_policy_and_one_alias(tmp_path: Path) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=43)
    first = tmp_path / "phase9_actor.pt"
    duplicate = tmp_path / "phase10_decentralized_actor.pt"
    write_checkpoint(first, actor, critic)
    write_checkpoint(duplicate, actor, critic)
    registry = PolicyRegistry()

    assert registry.register_checkpoint("phase9", first, source_phase="phase9") == "phase9"
    assert (
        registry.register_checkpoint("phase10_decentralized", duplicate, source_phase="phase10")
        == "phase9"
    )
    assert tuple(registry.records) == ("phase9",)
    assert registry.aliases[str(duplicate.resolve())] == "phase9"
    assert registry.records["phase9"].aliases == (str(duplicate.resolve()),)


def test_reusing_policy_id_is_rejected_before_identical_actor_deduplication(
    tmp_path: Path,
) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=44)
    first = tmp_path / "first.pt"
    duplicate = tmp_path / "duplicate.pt"
    write_checkpoint(first, actor, critic)
    write_checkpoint(duplicate, actor, critic)
    registry = PolicyRegistry()
    registry.register_checkpoint("same-id", first, source_phase="phase9")

    with pytest.raises(ValueError, match="policy id 'same-id' is already registered"):
        registry.register_checkpoint("same-id", duplicate, source_phase="phase10")
    assert str(duplicate.resolve()) not in registry.aliases


def test_reusing_policy_id_for_a_different_actor_is_rejected(tmp_path: Path) -> None:
    first_actor, first_critic = build_recurrent_actor_and_critic(seed=45)
    other_actor, other_critic = build_recurrent_actor_and_critic(seed=46)
    first = tmp_path / "first.pt"
    other = tmp_path / "other.pt"
    write_checkpoint(first, first_actor, first_critic)
    write_checkpoint(other, other_actor, other_critic)
    registry = PolicyRegistry()
    registry.register_checkpoint("same-id", first, source_phase="phase9")

    with pytest.raises(ValueError, match="policy id 'same-id' is already registered"):
        registry.register_checkpoint("same-id", other, source_phase="phase10")


def test_new_deduplicated_policy_id_is_reserved_for_future_registrations(
    tmp_path: Path,
) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=47)
    other_actor, other_critic = build_recurrent_actor_and_critic(seed=48)
    first = tmp_path / "first.pt"
    duplicate = tmp_path / "duplicate.pt"
    different = tmp_path / "different.pt"
    write_checkpoint(first, actor, critic)
    write_checkpoint(duplicate, actor, critic)
    write_checkpoint(different, other_actor, other_critic)
    registry = PolicyRegistry()
    registry.register_checkpoint("canonical", first, source_phase="phase9")
    assert (
        registry.register_checkpoint("alias-id", duplicate, source_phase="phase10") == "canonical"
    )

    with pytest.raises(ValueError, match="policy id 'alias-id' is already registered"):
        registry.register_checkpoint("alias-id", different, source_phase="phase10")
    assert registry.to_dict()["policy_id_aliases"] == {"alias-id": "canonical"}
