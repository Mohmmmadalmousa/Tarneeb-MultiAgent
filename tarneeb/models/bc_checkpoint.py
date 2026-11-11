"""Behaviour-cloning checkpoints (``bc_actor_v1``), Phase 9.

Additive alongside the frozen feedforward (``ppo_ff_v1``) and recurrent
(``ppo_gru_v1``) formats. A BC checkpoint is a **different kind of artifact**
from either: it carries an actor only, it was produced by supervised imitation
rather than by PPO, and it is meaningless without knowing *whose* decisions it
imitated and *which* deals those came from.

So beyond the frozen contract fields it records the teacher's identity, the
dataset version, and the seed split the demonstrations were drawn from. A
checkpoint cloned from a different teacher, or fitted to seeds that later
overlap an evaluation range, is a different experiment -- and the file says so
rather than leaving it to a lab notebook.

Both architectures are supported, because A1 clones the feedforward baseline
from the same demonstrations. ``architecture`` records which, and loading
rebuilds the matching network.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tarneeb.buffers.bc import BC_DATASET_VERSION, TEACHER_IDENTITY
from tarneeb.core.actions import ACTION_SPEC_VERSION
from tarneeb.core.events import EVENT_SCHEMA_VERSION
from tarneeb.core.rules import RulesConfig
from tarneeb.env.spaces import OBS_SPEC_VERSION
from tarneeb.models.checkpoint import IncompatibleCheckpointError, current_git_sha
from tarneeb.models.nets import (
    ARCHITECTURE_SPEC_VERSION,
    ActorNetwork,
    ArchitectureSpec,
    count_parameters,
)
from tarneeb.models.recurrent import (
    RECURRENT_ARCHITECTURE_SPEC_VERSION,
    RecurrentActorNetwork,
    RecurrentArchitectureSpec,
)

BC_CHECKPOINT_FORMAT_VERSION = "bc_actor_v1"
"""Never equal to ``ppo_ff_v1`` or ``ppo_gru_v1``: this is an imitation artifact."""

BC_COMPATIBILITY_FIELDS = (
    "checkpoint_format_version",
    "rules_version",
    "rules_fingerprint",
    "obs_spec_version",
    "action_spec_version",
    "event_schema_version",
    "architecture_spec_version",
    "bc_dataset_version",
    "teacher",
)
"""Every field that must match the running code before weights are loaded."""


def bc_compatibility_metadata(rules: RulesConfig, architecture: str) -> dict[str, str]:
    """Return the frozen contract strings a BC checkpoint must carry."""
    if architecture not in ("gru", "ff"):
        raise ValueError(f"architecture must be 'gru' or 'ff', got {architecture!r}")
    return {
        "checkpoint_format_version": BC_CHECKPOINT_FORMAT_VERSION,
        "rules_version": rules.version,
        "rules_fingerprint": rules.fingerprint(),
        "obs_spec_version": OBS_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "architecture_spec_version": (
            RECURRENT_ARCHITECTURE_SPEC_VERSION
            if architecture == "gru"
            else ARCHITECTURE_SPEC_VERSION
        ),
        "bc_dataset_version": BC_DATASET_VERSION,
        "teacher": TEACHER_IDENTITY,
    }


@dataclass(frozen=True, slots=True)
class LoadedBCCheckpoint:
    """What :func:`load_bc_checkpoint` gives back after the checks passed."""

    actor: RecurrentActorNetwork | ActorNetwork
    payload: dict[str, Any]

    @property
    def architecture(self) -> str:
        return str(self.payload["architecture"])

    @property
    def seed_split(self) -> dict[str, Any]:
        return dict(self.payload["seed_split"])

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self.payload["metrics"])


def save_bc_checkpoint(
    path: Path,
    actor: RecurrentActorNetwork | ActorNetwork,
    architecture: str,
    rules: RulesConfig,
    bc_config: dict[str, Any],
    seed_split: dict[str, Any],
    metrics: dict[str, Any],
    dataset_metadata: dict[str, Any],
) -> Path:
    """Write a BC checkpoint and return the path it was written to."""
    payload: dict[str, Any] = {
        **bc_compatibility_metadata(rules, architecture),
        "architecture": architecture,
        "rules_config": rules.as_dict(),
        "architecture_spec": actor.spec.as_dict(),
        "parameter_count": count_parameters(actor),
        "actor_state_dict": actor.state_dict(),
        "bc_config": bc_config,
        "seed_split": seed_split,
        "metrics": metrics,
        "dataset_metadata": dataset_metadata,
        "git_sha": current_git_sha(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def check_bc_compatibility(payload: dict[str, Any], rules: RulesConfig) -> None:
    """Raise :class:`IncompatibleCheckpointError` if any recorded contract differs."""
    architecture = payload.get("architecture")
    if architecture not in ("gru", "ff"):
        raise IncompatibleCheckpointError(
            f"checkpoint records architecture {architecture!r}, which is not a BC architecture"
        )
    expected = bc_compatibility_metadata(rules, str(architecture))
    mismatches = [
        f"{field}: checkpoint has {payload.get(field)!r}, code has {expected[field]!r}"
        for field in BC_COMPATIBILITY_FIELDS
        if payload.get(field) != expected[field]
    ]
    if mismatches:
        raise IncompatibleCheckpointError(
            "refusing to load a BC checkpoint built under different contracts:\n  "
            + "\n  ".join(mismatches)
        )


def load_bc_checkpoint(
    path: Path, rules: RulesConfig | None = None, map_location: str = "cpu"
) -> LoadedBCCheckpoint:
    """Load a BC actor after proving it matches the running contracts."""
    resolved_rules = rules if rules is not None else RulesConfig()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    check_bc_compatibility(payload, resolved_rules)

    architecture = str(payload["architecture"])
    actor: RecurrentActorNetwork | ActorNetwork
    if architecture == "gru":
        actor = RecurrentActorNetwork(
            RecurrentArchitectureSpec.from_dict(payload["architecture_spec"])
        )
    else:
        actor = ActorNetwork(ArchitectureSpec.from_dict(payload["architecture_spec"]))
    actor.load_state_dict(payload["actor_state_dict"])

    stored = int(payload["parameter_count"])
    if count_parameters(actor) != stored:  # pragma: no cover - spec implies count
        raise IncompatibleCheckpointError(
            f"actor has {count_parameters(actor)} parameters, checkpoint recorded {stored}"
        )
    return LoadedBCCheckpoint(actor=actor, payload=payload)
