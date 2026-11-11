"""Loading a Phase 7 stage configuration.

A stage file has two blocks. The ``training`` block is handed verbatim to the
frozen :func:`~tarneeb.training.config.training_config_from_dict`, so every PPO
hyper-parameter keeps exactly the meaning and validation it has in Phase 6. The
``stage`` block describes the curriculum task and is Phase 7's own.

Splitting them this way means the frozen loader never sees a key it does not know,
and Phase 7 never has to restate a hyper-parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tarneeb.curriculum.stages import StageSpec
from tarneeb.training.config import TrainingConfig, training_config_from_dict

STAGE_CONFIG_SECTIONS = ("stage", "training")


def load_stage_configuration(path: Path) -> tuple[TrainingConfig, StageSpec]:
    """Return the ``(TrainingConfig, StageSpec)`` a stage file describes."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return stage_configuration_from_dict(raw)


def stage_configuration_from_dict(raw: dict[str, Any]) -> tuple[TrainingConfig, StageSpec]:
    """Build a stage configuration from a plain nested dictionary.

    Unknown top-level sections raise rather than being ignored, matching the frozen
    loader's refusal to drop a hyper-parameter silently.
    """
    unknown_sections = set(raw) - set(STAGE_CONFIG_SECTIONS)
    if unknown_sections:
        raise ValueError(
            f"unknown stage config sections: {sorted(unknown_sections)}; "
            f"expected only {list(STAGE_CONFIG_SECTIONS)}"
        )
    if "stage" not in raw:
        raise ValueError("a stage config must contain a 'stage' section")

    stage = StageSpec.from_dict(raw["stage"])
    training_config = training_config_from_dict(dict(raw.get("training", {})))
    validate_stage_configuration(training_config, stage)
    return training_config, stage


def validate_stage_configuration(config: TrainingConfig, stage: StageSpec) -> None:
    """Check the two halves agree with each other and with the frozen contracts."""
    if config.training_opponent != stage.opponent:
        raise ValueError(
            f"the training block names opponent {config.training_opponent!r} but stage "
            f"{stage.name!r} requires {stage.opponent!r}"
        )
    if stage.uses_trick_shaping and stage.curriculum_stage != 2:
        raise ValueError("trick shaping is a stage 2 device only")


def describe_stage_configuration(config: TrainingConfig, stage: StageSpec) -> dict[str, Any]:
    """Return a flat record of what this configuration will run.

    Everything a later reader needs to identify the experiment without opening the
    YAML again.
    """
    return {
        "stage_name": stage.name,
        "curriculum_stage": stage.curriculum_stage,
        "auction_mode": stage.auction_mode,
        "trump_mode": stage.trump_mode,
        "contract_mode": stage.contract_mode,
        "fixed_trump_suit": stage.fixed_trump_suit,
        "fixed_contract": stage.fixed_contract,
        "sampled_contracts": list(stage.sampled_contracts),
        "opponent": stage.opponent,
        "reward_mode": stage.reward_description,
        "history_condition": stage.history_condition,
        "trick_shaping_weight": stage.trick_shaping_weight,
        "environment_count": config.environment_count,
        "rollout_length": config.rollout_length,
        "total_environment_steps": config.total_environment_steps,
        "total_updates": config.total_updates,
        "decisions_per_rollout": config.decisions_per_rollout,
        "seeds": config.seeds(),
    }
