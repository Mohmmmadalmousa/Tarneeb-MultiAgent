"""Checkpoint writing and loading, with hard compatibility checks.

A checkpoint is only meaningful next to the contracts it was trained under. If
the rules, the observation layout, the action space, the event schema or the
network architecture changed, the stored weights describe a different problem.
Loading such a file would silently produce nonsense, so
:func:`load_checkpoint` refuses: it raises :class:`IncompatibleCheckpointError`
naming every field that disagrees.

The rules fingerprint is checked as well as the rules version, because a version
string can be left unchanged by accident while a field is edited. The fingerprint
cannot.
"""

from __future__ import annotations

import datetime as dt
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tarneeb.core.actions import ACTION_SPEC_VERSION
from tarneeb.core.events import EVENT_SCHEMA_VERSION
from tarneeb.core.rules import RULES_VERSION, RulesConfig
from tarneeb.env.spaces import OBS_SPEC_VERSION
from tarneeb.models.nets import (
    ARCHITECTURE_SPEC_VERSION,
    ActorNetwork,
    ArchitectureSpec,
    CriticNetwork,
    count_parameters,
)

CHECKPOINT_FORMAT_VERSION = "ppo_ff_v1"
"""Phase 6 checkpoint layout. The recurrent format is a later phase."""

BUFFER_CONTRACT_VERSION = "flat_v1"
"""The rollout data schema this checkpoint was trained under.

Phase 6 stores one flat row per decision with no chunking and no recurrent
state. The later recurrent contract (``chunk_v2``) stores chunked sequences plus
per-``(environment, seat, policy_id)`` hidden state, which is a different schema
with different invariants. Recording the contract here and checking it on load
means the two can never be silently mixed. See docs/BUFFER_CONTRACT.md.
"""

COMPATIBILITY_FIELDS = (
    "checkpoint_format_version",
    "rules_version",
    "rules_fingerprint",
    "obs_spec_version",
    "action_spec_version",
    "event_schema_version",
    "architecture_spec_version",
    "buffer_contract_version",
)
"""Every field that must match the running code exactly before weights are loaded."""


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a checkpoint was produced under different frozen contracts."""


def current_git_sha() -> str | None:
    """Return the current commit SHA, or ``None`` outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    return completed.stdout.strip() or None


def compatibility_metadata(rules: RulesConfig) -> dict[str, str]:
    """Return the frozen contract strings the running code implements."""
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "rules_version": rules.version,
        "rules_fingerprint": rules.fingerprint(),
        "obs_spec_version": OBS_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "architecture_spec_version": ARCHITECTURE_SPEC_VERSION,
        "buffer_contract_version": BUFFER_CONTRACT_VERSION,
    }


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """What :func:`load_checkpoint` gives back after the checks passed."""

    actor: ActorNetwork
    critic: CriticNetwork | None
    payload: dict[str, Any]

    @property
    def architecture(self) -> ArchitectureSpec:
        return ArchitectureSpec.from_dict(self.payload["architecture"])

    @property
    def global_environment_steps(self) -> int:
        return int(self.payload["training"]["global_environment_steps"])


def save_checkpoint(
    path: Path,
    actor: ActorNetwork,
    critic: CriticNetwork,
    optimizer: torch.optim.Optimizer | None,
    rules: RulesConfig,
    training_config: dict[str, Any],
    global_environment_steps: int,
    completed_updates: int,
    wall_clock_seconds: float,
    seeds: dict[str, int],
    evaluation: dict[str, Any] | None = None,
    resume_state: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint and return the path it was written to.

    The optimizer state is stored whenever an optimizer is given, and
    ``resume_state`` carries everything else a run needs to continue exactly
    where it stopped rather than merely to be deployed.
    """
    if actor.spec != critic.spec:
        raise ValueError("actor and critic were built from different architecture specs")

    payload: dict[str, Any] = {
        **compatibility_metadata(rules),
        "rules_config": rules.as_dict(),
        "architecture": actor.spec.as_dict(),
        "parameter_counts": {
            "actor": count_parameters(actor),
            "critic": count_parameters(critic),
        },
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "training_config": training_config,
        "training": {
            "global_environment_steps": int(global_environment_steps),
            "completed_updates": int(completed_updates),
            "wall_clock_seconds": float(wall_clock_seconds),
        },
        "seeds": dict(seeds),
        "resume": resume_state,
        "evaluation": evaluation,
        "git_sha": current_git_sha(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def check_compatibility(payload: dict[str, Any], rules: RulesConfig) -> None:
    """Raise :class:`IncompatibleCheckpointError` if any frozen contract differs."""
    expected = compatibility_metadata(rules)
    mismatches = [
        f"{field}: checkpoint has {payload.get(field)!r}, code has {expected[field]!r}"
        for field in COMPATIBILITY_FIELDS
        if payload.get(field) != expected[field]
    ]
    if mismatches:
        raise IncompatibleCheckpointError(
            "refusing to load a checkpoint built under different contracts:\n  "
            + "\n  ".join(mismatches)
        )


def load_checkpoint(
    path: Path,
    rules: RulesConfig | None = None,
    load_critic: bool = True,
    map_location: str = "cpu",
) -> LoadedCheckpoint:
    """Load a checkpoint after proving it matches the running contracts.

    ``load_critic=False`` is the inference path: the actor alone is what plays,
    and never loading the critic makes it structurally impossible for a
    privileged value network to reach a deployed agent.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    check_compatibility(payload, resolved_rules)

    architecture = ArchitectureSpec.from_dict(payload["architecture"])
    actor = ActorNetwork(architecture)
    actor.load_state_dict(payload["actor_state_dict"])

    stored_actor_parameters = int(payload["parameter_counts"]["actor"])
    if count_parameters(actor) != stored_actor_parameters:  # pragma: no cover - spec implies count
        raise IncompatibleCheckpointError(
            f"actor has {count_parameters(actor)} parameters, "
            f"checkpoint recorded {stored_actor_parameters}"
        )

    critic: CriticNetwork | None = None
    if load_critic and payload.get("critic_state_dict") is not None:
        critic = CriticNetwork(architecture)
        critic.load_state_dict(payload["critic_state_dict"])

    return LoadedCheckpoint(actor=actor, critic=critic, payload=payload)


def restore_optimizer(
    optimizer: torch.optim.Optimizer, loaded: LoadedCheckpoint
) -> torch.optim.Optimizer:
    """Restore optimizer state from a loaded checkpoint, for resuming a run."""
    state = loaded.payload.get("optimizer_state_dict")
    if state is None:
        raise IncompatibleCheckpointError("this checkpoint carries no optimizer state to resume")
    optimizer.load_state_dict(state)
    return optimizer


def expected_rules_versions() -> dict[str, str]:
    """Return the module-level frozen strings, for reporting."""
    return {
        "rules_version": RULES_VERSION,
        "obs_spec_version": OBS_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "architecture_spec_version": ARCHITECTURE_SPEC_VERSION,
        "buffer_contract_version": BUFFER_CONTRACT_VERSION,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
    }
