"""Training configuration and the seed architecture for a Phase 6 run.

Every stochastic component gets its own explicitly derived seed, so no two
streams can silently share state and no global RNG is consulted for anything
that matters:

===========================  ==================================================
stream                        derived from
===========================  ==================================================
environment dealing           ``derive_environment_seed(environment_root_seed, i)``
network initialization        ``initialization_seed``
action sampling               ``action_sampling_seed`` (a torch generator)
minibatch shuffling           ``minibatch_shuffle_seed`` (a numpy generator)
training opponents            ``opponent_seed`` (one generator per environment)
evaluation                    ``evaluation_seed`` (held-out seeds only)
===========================  ==================================================

The environment root seed is drawn from ``TRAINING_SEEDS``, which Phase 5 proved
disjoint from ``HELD_OUT_EVALUATION_SEEDS``. Training therefore cannot touch a
deal that evaluation will later report on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from tarneeb.evaluation.arena import HELD_OUT_EVALUATION_SEEDS, TRAINING_SEEDS
from tarneeb.models.nets import ArchitectureSpec
from tarneeb.training.ppo import PPOConfig

DEFAULT_TRAINING_SEED = 7

SELECTABLE_TRAINING_OPPONENTS = frozenset({"random", "basic", "strong"})
"""Who may hold the other partnership during training.

Current-policy self-play -- four copies of the learner at one table -- is a later
curriculum stage and is deliberately NOT selectable here. It is not merely unused
in Phase 6: an early measurement showed it converges to an all-pass equilibrium,
so shipping it as a configuration option would offer a setting that is known to
destroy the training signal.
"""


@dataclass(frozen=True, slots=True)
class EvaluationSchedule:
    """When and how the training loop evaluates against the frozen baselines.

    Evaluation is isolated by construction: it builds its own arena and its own
    environments, draws only from the held-out seed range, consumes none of the
    training generators, and restores the networks' training mode afterwards.
    """

    every_updates: int = 10
    paired_deal_count: int = 200
    evaluation_seed: int = 20250910
    bootstrap_resamples: int = 2000
    opponent_names: tuple[str, ...] = ("random",)
    deterministic_actions: bool = False

    def __post_init__(self) -> None:
        if self.evaluation_seed < 0:
            raise ValueError("evaluation_seed must be non-negative")
        if self.every_updates < 1:
            raise ValueError("every_updates must be at least 1")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Everything one Phase 6 training run needs."""

    run_name: str = "ppo_ff_baseline"
    total_environment_steps: int = 1_000_000
    environment_count: int = 32
    rollout_length: int = 256

    training_opponent: str = "basic"
    """Which frozen non-neural baseline holds the other partnership.

    One of :data:`SELECTABLE_TRAINING_OPPONENTS`. A fixed opponent is the
    blueprint's curriculum for the feedforward stage; self-play belongs to a
    later stage and cannot be selected here.
    """

    opponent_seed: int = 4

    training_seed: int = DEFAULT_TRAINING_SEED
    environment_root_seed: int = 0
    initialization_seed: int = 1
    action_sampling_seed: int = 2
    minibatch_shuffle_seed: int = 3

    checkpoint_directory: Path = Path("checkpoints")
    checkpoint_every_updates: int = 25

    ppo: PPOConfig = field(default_factory=PPOConfig)
    architecture: ArchitectureSpec = field(default_factory=ArchitectureSpec)
    evaluation: EvaluationSchedule = field(default_factory=EvaluationSchedule)

    def __post_init__(self) -> None:
        if self.environment_count < 1 or self.rollout_length < 1:
            raise ValueError("environment_count and rollout_length must be at least 1")
        if self.total_environment_steps < self.decisions_per_rollout:
            raise ValueError(
                f"total_environment_steps ({self.total_environment_steps}) is smaller than one "
                f"rollout ({self.decisions_per_rollout})"
            )
        if self.training_opponent not in SELECTABLE_TRAINING_OPPONENTS:
            raise ValueError(
                f"training_opponent must be one of {sorted(SELECTABLE_TRAINING_OPPONENTS)}, "
                f"got {self.training_opponent!r}"
            )
        if self.environment_root_seed not in TRAINING_SEEDS:
            raise ValueError(
                f"environment_root_seed {self.environment_root_seed} is outside TRAINING_SEEDS; "
                "training must never draw from the held-out evaluation range"
            )

    # --- derived quantities -------------------------------------------------

    @property
    def decisions_per_rollout(self) -> int:
        return self.environment_count * self.rollout_length

    @property
    def total_updates(self) -> int:
        """How many PPO updates the step budget buys, rounded up."""
        return max(
            1,
            -(-self.total_environment_steps // self.decisions_per_rollout),
        )

    @property
    def minibatch_size(self) -> int:
        return max(1, self.decisions_per_rollout // self.ppo.minibatch_count)

    def seeds(self) -> dict[str, int]:
        return {
            "training_seed": self.training_seed,
            "environment_root_seed": self.environment_root_seed,
            "initialization_seed": self.initialization_seed,
            "action_sampling_seed": self.action_sampling_seed,
            "minibatch_shuffle_seed": self.minibatch_shuffle_seed,
            "opponent_seed": self.opponent_seed,
            "evaluation_seed": self.evaluation.evaluation_seed,
        }

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        return values

    def with_overrides(self, **overrides: Any) -> TrainingConfig:
        """Return a copy with the given top-level fields replaced."""
        return replace(self, **overrides)


def evaluation_seed_is_held_out(evaluation_seed: int) -> bool:
    """Return True when the evaluation seed maps into the held-out range.

    Phase 5's ``evaluation_match_seed`` folds any seed into
    ``HELD_OUT_EVALUATION_SEEDS``, so this is always true; the function exists so
    the property is asserted in a test instead of assumed.
    """
    del evaluation_seed
    return HELD_OUT_EVALUATION_SEEDS.start >= TRAINING_SEEDS.stop


def load_training_config(path: Path) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from a YAML file.

    Unknown keys raise rather than being ignored: a silently dropped
    hyper-parameter is a reproducibility bug that is very hard to notice later.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return training_config_from_dict(raw)


def training_config_from_dict(raw: dict[str, Any]) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from a plain nested dictionary."""
    values = dict(raw)
    ppo = PPOConfig(**values.pop("ppo", {}))
    architecture = ArchitectureSpec(**values.pop("architecture", {}))

    evaluation_values = dict(values.pop("evaluation", {}))
    if "opponent_names" in evaluation_values:
        evaluation_values["opponent_names"] = tuple(evaluation_values["opponent_names"])
    evaluation = EvaluationSchedule(**evaluation_values)

    if "checkpoint_directory" in values:
        values["checkpoint_directory"] = Path(values["checkpoint_directory"])

    unknown_keys = set(values) - set(TrainingConfig.__slots__)
    if unknown_keys:
        raise ValueError(f"unknown training config keys: {sorted(unknown_keys)}")

    return TrainingConfig(ppo=ppo, architecture=architecture, evaluation=evaluation, **values)
