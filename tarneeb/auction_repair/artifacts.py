"""Provenance and exclusive-write artifacts for additive Phase 11B work."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb.auction_repair.data import PublicExamples
from tarneeb.auction_repair.learning import assert_preserved
from tarneeb.models.recurrent import RecurrentActorNetwork
from tarneeb.models.recurrent_checkpoint import LoadedRecurrentCheckpoint, load_recurrent_checkpoint


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provenance(path: Path, loaded: LoadedRecurrentCheckpoint) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        **{
            key: loaded.payload[key]
            for key in (
                "git_sha",
                "created_utc",
                "training",
                "seeds",
                "training_config",
                "rules_fingerprint",
                "checkpoint_format_version",
                "architecture",
            )
        },
    }


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_examples_new(path: Path, examples: PublicExamples, **extra: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            statics=examples.statics,
            histories=examples.histories,
            lengths=examples.lengths,
            legal=examples.legal,
            labels=examples.labels,
            metadata_json=json.dumps(examples.metadata),
            **extra,
        )


def save_repair_new(
    path: Path,
    parent_path: Path,
    parent: LoadedRecurrentCheckpoint,
    actor: RecurrentActorNetwork,
    report: dict[str, Any],
) -> None:
    """Validated ppo_gru_v1 inference artifact with separate Phase 11B provenance.

    No critic/optimizer/resume is carried forward as if it had been trained for
    this policy. Normal recurrent inference remains compatible; PPO resume is
    deliberately unavailable.
    """
    if "phase11b" not in path.parts or path.resolve() == parent_path.resolve():
        raise ValueError("Repair checkpoints must use a new phase11b directory")
    assert_preserved(parent.actor, actor)
    payload = deepcopy(parent.payload)
    payload.update(
        actor_state_dict=actor.state_dict(),
        critic_state_dict=None,
        parameter_counts={"actor": sum(p.numel() for p in actor.parameters()), "critic": 0},
        optimizer_state_dict=None,
        resume=None,
        evaluation=None,
        created_utc=dt.datetime.now(dt.UTC).isoformat(),
        git_sha="uncommitted_phase11b",
        training_config={"phase": "11B", "method": "auction_rows_bc"},
        training={
            "global_environment_steps": 0,
            "completed_updates": 0,
            "wall_clock_seconds": report.get("wall_seconds", 0.0),
        },
        seeds={"shuffle": report["config"]["shuffle_seed"]},
        phase11b={
            "format": "auction_rows_bc_v1",
            "inference_only": True,
            "parent": provenance(parent_path, parent),
            "report": report,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    reloaded = load_recurrent_checkpoint(path, load_critic=False)
    for name, tensor in actor.state_dict().items():
        if not torch.equal(tensor, reloaded.actor.state_dict()[name]):
            raise ValueError("Saved repair failed round-trip validation")
