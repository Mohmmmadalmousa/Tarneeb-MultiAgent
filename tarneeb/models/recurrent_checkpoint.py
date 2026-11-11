"""Checkpoint writing and loading for the recurrent family (``ppo_gru_v1``).

Additive alongside :mod:`tarneeb.models.checkpoint`. The two families record
different values in the same three fields -- ``checkpoint_format_version``,
``architecture_spec_version`` and ``buffer_contract_version`` -- so neither
loader can be handed the other's file without failing loudly. That is the point:
a recurrent checkpoint loaded as feedforward would either raise deep inside
``load_state_dict`` with an unhelpful message or, worse, load partially.

The rules fingerprint is checked as well as the rules version, because a version
string can be left unchanged by accident while a field is edited. The
fingerprint cannot.

The recurrent architecture metadata is recorded in full -- hidden size, GRU layer
count, event embedding widths, both observation widths, the chunk length and
``K_MAX`` -- because a recurrent checkpoint is only meaningful next to the
sequence shape it was trained under.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tarneeb.buffers.recurrent import (
    DEFAULT_CHUNK_LENGTH,
    RECURRENT_BUFFER_CONTRACT_VERSION,
)
from tarneeb.core.actions import ACTION_SPEC_VERSION
from tarneeb.core.events import EVENT_SCHEMA_VERSION
from tarneeb.core.rules import RulesConfig
from tarneeb.env.spaces import K_MAX, OBS_SPEC_VERSION
from tarneeb.models.checkpoint import (
    COMPATIBILITY_FIELDS,
    IncompatibleCheckpointError,
    current_git_sha,
)
from tarneeb.models.nets import count_parameters
from tarneeb.models.recurrent import (
    RECURRENT_ARCHITECTURE_SPEC_VERSION,
    RecurrentActorNetwork,
    RecurrentArchitectureSpec,
    RecurrentCriticNetwork,
)

RECURRENT_CHECKPOINT_FORMAT_VERSION = "ppo_gru_v1"
"""The recurrent checkpoint layout. Never equal to the Phase 6 ``ppo_ff_v1``."""


def recurrent_compatibility_metadata(rules: RulesConfig) -> dict[str, str]:
    """Return the frozen contract strings the running recurrent code implements."""
    return {
        "checkpoint_format_version": RECURRENT_CHECKPOINT_FORMAT_VERSION,
        "rules_version": rules.version,
        "rules_fingerprint": rules.fingerprint(),
        "obs_spec_version": OBS_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "architecture_spec_version": RECURRENT_ARCHITECTURE_SPEC_VERSION,
        "buffer_contract_version": RECURRENT_BUFFER_CONTRACT_VERSION,
    }


@dataclass(frozen=True, slots=True)
class LoadedRecurrentCheckpoint:
    """What :func:`load_recurrent_checkpoint` gives back after the checks passed."""

    actor: RecurrentActorNetwork
    critic: RecurrentCriticNetwork | None
    payload: dict[str, Any]

    @property
    def architecture(self) -> RecurrentArchitectureSpec:
        return RecurrentArchitectureSpec.from_dict(self.payload["architecture"])

    @property
    def global_environment_steps(self) -> int:
        return int(self.payload["training"]["global_environment_steps"])

    @property
    def chunk_length(self) -> int:
        return int(self.payload["recurrent"]["chunk_length"])


def save_recurrent_checkpoint(
    path: Path,
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    optimizer: torch.optim.Optimizer | None,
    rules: RulesConfig,
    training_config: dict[str, Any],
    global_environment_steps: int,
    completed_updates: int,
    wall_clock_seconds: float,
    seeds: dict[str, int],
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    evaluation: dict[str, Any] | None = None,
    resume_state: dict[str, Any] | None = None,
) -> Path:
    """Write a recurrent checkpoint and return the path it was written to."""
    if actor.spec != critic.spec:
        raise ValueError("actor and critic were built from different architecture specs")

    payload: dict[str, Any] = {
        **recurrent_compatibility_metadata(rules),
        "rules_config": rules.as_dict(),
        "architecture": actor.spec.as_dict(),
        "recurrent": {
            "hidden_size": actor.spec.hidden_size,
            "gru_layer_count": actor.spec.gru_layer_count,
            "event_projection_dim": actor.spec.event_projection_dim,
            "event_embedding_dims": list(actor.spec.event_embedding_dims),
            "actor_observation_dim": actor.spec.actor_observation_dim,
            "critic_observation_dim": actor.spec.critic_observation_dim,
            "gru_reset": actor.spec.gru_reset,
            "chunk_length": int(chunk_length),
            "k_max": K_MAX,
        },
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


def check_recurrent_compatibility(payload: dict[str, Any], rules: RulesConfig) -> None:
    """Raise :class:`IncompatibleCheckpointError` if any frozen contract differs."""
    expected = recurrent_compatibility_metadata(rules)
    mismatches = [
        f"{field}: checkpoint has {payload.get(field)!r}, code has {expected[field]!r}"
        for field in COMPATIBILITY_FIELDS
        if payload.get(field) != expected[field]
    ]
    if mismatches:
        raise IncompatibleCheckpointError(
            "refusing to load a recurrent checkpoint built under different contracts:\n  "
            + "\n  ".join(mismatches)
        )


def load_recurrent_checkpoint(
    path: Path,
    rules: RulesConfig | None = None,
    load_critic: bool = True,
    map_location: str = "cpu",
) -> LoadedRecurrentCheckpoint:
    """Load a recurrent checkpoint after proving it matches the running contracts.

    ``load_critic=False`` is the inference path: the actor alone is what plays,
    and never loading the critic makes it structurally impossible for a
    privileged value network to reach a deployed agent.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    check_recurrent_compatibility(payload, resolved_rules)

    architecture = RecurrentArchitectureSpec.from_dict(payload["architecture"])
    actor = RecurrentActorNetwork(architecture)
    actor.load_state_dict(payload["actor_state_dict"])

    stored_actor_parameters = int(payload["parameter_counts"]["actor"])
    if count_parameters(actor) != stored_actor_parameters:  # pragma: no cover - spec implies count
        raise IncompatibleCheckpointError(
            f"actor has {count_parameters(actor)} parameters, "
            f"checkpoint recorded {stored_actor_parameters}"
        )

    critic: RecurrentCriticNetwork | None = None
    if load_critic and payload.get("critic_state_dict") is not None:
        critic = RecurrentCriticNetwork(architecture)
        critic.load_state_dict(payload["critic_state_dict"])

    return LoadedRecurrentCheckpoint(actor=actor, critic=critic, payload=payload)


def restore_recurrent_optimizer(
    optimizer: torch.optim.Optimizer, loaded: LoadedRecurrentCheckpoint
) -> torch.optim.Optimizer:
    """Restore optimizer state from a loaded recurrent checkpoint."""
    state = loaded.payload.get("optimizer_state_dict")
    if state is None:
        raise IncompatibleCheckpointError("this checkpoint carries no optimizer state to resume")
    optimizer.load_state_dict(state)
    return optimizer


def expected_recurrent_versions() -> dict[str, str]:
    """Return the module-level frozen strings, for reporting."""
    return {
        "checkpoint_format_version": RECURRENT_CHECKPOINT_FORMAT_VERSION,
        "architecture_spec_version": RECURRENT_ARCHITECTURE_SPEC_VERSION,
        "buffer_contract_version": RECURRENT_BUFFER_CONTRACT_VERSION,
        "obs_spec_version": OBS_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
    }
