"""Read-mostly registry for compatible historical recurrent actors."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.models.recurrent import RecurrentArchitectureSpec
from tarneeb.models.recurrent_checkpoint import (
    check_recurrent_compatibility,
    expected_recurrent_versions,
)

POLICY_CATEGORIES = ("champion", "snapshot", "strong", "random")


@dataclass(frozen=True, slots=True)
class PolicyRecord:
    """Checkpoint provenance and league membership for one unique actor."""

    policy_id: str
    checkpoint_path: str | None
    rules_fingerprint: str
    obs_spec_version: str
    action_spec_version: str
    event_schema_version: str
    architecture_spec_version: str
    checkpoint_format_version: str
    buffer_contract_version: str
    source_phase: str
    seed: int | None
    update: int | None
    source_commit: str | None
    actor_fingerprint: str | None
    registration_order: int
    category: str
    eligible: bool = True
    aliases: tuple[str, ...] = ()
    behavioral_fingerprint: tuple[float, ...] | None = None


def actor_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash actor tensor names, shapes, dtypes and bytes deterministically."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


@dataclass
class PolicyRegistry:
    """Unique actor records plus checkpoint aliases for identical weights."""

    rules: RulesConfig = field(default_factory=RulesConfig)
    records: dict[str, PolicyRecord] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    policy_id_aliases: dict[str, str] = field(default_factory=dict)

    def register_checkpoint(
        self,
        policy_id: str,
        checkpoint_path: Path,
        source_phase: str,
        category: str = "snapshot",
        seed: int | None = None,
        update: int | None = None,
        behavioral_fingerprint: tuple[float, ...] | None = None,
    ) -> str:
        """Validate a recurrent checkpoint and register its unique actor.

        Duplicate actor weights become aliases of the first registered policy,
        so different checkpoint filenames cannot increase its sampling mass.
        """
        if category not in POLICY_CATEGORIES:
            raise ValueError(f"category must be one of {POLICY_CATEGORIES}, got {category!r}")
        if policy_id in self.records or policy_id in self.policy_id_aliases:
            raise ValueError(f"policy id {policy_id!r} is already registered")
        path = checkpoint_path.resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        check_recurrent_compatibility(payload, self.rules)
        architecture = RecurrentArchitectureSpec.from_dict(payload["architecture"])
        if architecture.actor_observation_dim != 692 or architecture.action_count != 65:
            raise ValueError("league actors must preserve the obs_v2/act_v1 692x65 contract")
        if architecture.gru_reset != "deal":
            raise ValueError("league actors must preserve gru_reset='deal'")
        fingerprint = actor_fingerprint(payload["actor_state_dict"])
        prior_id = next(
            (
                record.policy_id
                for record in self.records.values()
                if record.actor_fingerprint == fingerprint
            ),
            None,
        )
        if prior_id is not None:
            self.aliases[str(path)] = prior_id
            self.policy_id_aliases[policy_id] = prior_id
            existing = self.records[prior_id]
            self.records[prior_id] = PolicyRecord(
                **{
                    **asdict(existing),
                    "aliases": tuple(sorted(set(existing.aliases) | {str(path)})),
                }
            )
            return prior_id
        versions = expected_recurrent_versions()
        training = payload.get("training", {})
        training_seeds = payload.get("seeds", {})
        record = PolicyRecord(
            policy_id=policy_id,
            checkpoint_path=str(path),
            rules_fingerprint=self.rules.fingerprint(),
            obs_spec_version=versions["obs_spec_version"],
            action_spec_version=versions["action_spec_version"],
            event_schema_version=versions["event_schema_version"],
            architecture_spec_version=versions["architecture_spec_version"],
            checkpoint_format_version=versions["checkpoint_format_version"],
            buffer_contract_version=versions["buffer_contract_version"],
            source_phase=source_phase,
            seed=seed if seed is not None else training_seeds.get("seed_index"),
            update=update if update is not None else training.get("completed_updates"),
            source_commit=payload.get("git_sha"),
            actor_fingerprint=fingerprint,
            registration_order=len(self.records),
            category=category,
            behavioral_fingerprint=behavioral_fingerprint,
        )
        self.records[policy_id] = record
        self.aliases[str(path)] = policy_id
        return policy_id

    def register_anchor(self, policy_id: str, category: str) -> str:
        """Register a non-neural Strong or random-legal anchor."""
        if category not in ("strong", "random"):
            raise ValueError("non-neural anchors must use category 'strong' or 'random'")
        if policy_id in self.records or policy_id in self.policy_id_aliases:
            raise ValueError(f"policy id {policy_id!r} is already registered")
        versions = expected_recurrent_versions()
        self.records[policy_id] = PolicyRecord(
            policy_id=policy_id,
            checkpoint_path=None,
            rules_fingerprint=self.rules.fingerprint(),
            obs_spec_version=versions["obs_spec_version"],
            action_spec_version=versions["action_spec_version"],
            event_schema_version=versions["event_schema_version"],
            architecture_spec_version=versions["architecture_spec_version"],
            checkpoint_format_version=versions["checkpoint_format_version"],
            buffer_contract_version=versions["buffer_contract_version"],
            source_phase="frozen_baseline",
            seed=None,
            update=None,
            source_commit=None,
            actor_fingerprint=None,
            registration_order=len(self.records),
            category=category,
        )
        return policy_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules_config": self.rules.as_dict(),
            "records": {key: asdict(record) for key, record in self.records.items()},
            "aliases": dict(self.aliases),
            "policy_id_aliases": dict(self.policy_id_aliases),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyRegistry:
        records: dict[str, PolicyRecord] = {}
        for key, data in value["records"].items():
            record_data = dict(data)
            record_data["aliases"] = tuple(record_data.get("aliases", ()))
            if record_data["behavioral_fingerprint"] is not None:
                record_data["behavioral_fingerprint"] = tuple(record_data["behavioral_fingerprint"])
            records[key] = PolicyRecord(**record_data)
        return cls(
            rules=RulesConfig(**value["rules_config"]),
            records=records,
            aliases=dict(value["aliases"]),
            policy_id_aliases=dict(value.get("policy_id_aliases", {})),
        )
