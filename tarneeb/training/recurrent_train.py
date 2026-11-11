"""The minimal recurrent PPO smoke loop, and its resume path.

Phase 8 exists to prove the recurrent infrastructure correct, not to train
anything. This module therefore implements exactly the path that proof requires
-- forward, rollout, chunk construction, loss, backward, optimizer step,
checkpoint, resume -- and nothing more. Real recurrent training is Phase 9, and
nothing here should be read as a performance result.

The configuration deliberately reuses the frozen :class:`PPOConfig` and the
frozen seed architecture rather than inventing parallel ones, so the recurrent
run cannot drift away from the mathematics the feedforward baseline was verified
under. ``gamma``, ``lambda``, ``clip_epsilon`` and the learning rate all carry
their existing values.

Resume captures everything a recurrent run would otherwise lose. The list is not
guesswork: it is docs/RECURRENT_CONTRACT.md section 9, and the reason each item
is on it is that leaving it out makes either determinism or the epoch-zero check
fail on the first update after a resume.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tarneeb.agents.base import Agent
from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.buffers.recurrent import DEFAULT_CHUNK_LENGTH
from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.recurrent import (
    RecurrentArchitectureSpec,
    build_recurrent_actor_and_critic,
)
from tarneeb.models.recurrent_checkpoint import (
    load_recurrent_checkpoint,
    restore_recurrent_optimizer,
    save_recurrent_checkpoint,
)
from tarneeb.training.config import SELECTABLE_TRAINING_OPPONENTS
from tarneeb.training.ppo import PPOConfig, linear_learning_rate
from tarneeb.training.recurrent_ppo import (
    RecurrentUpdateMetrics,
    build_recurrent_optimizer,
    recurrent_ppo_update,
)
from tarneeb.training.recurrent_rollout import (
    RecurrentRolloutCollector,
    RecurrentRolloutMetrics,
)

BASELINE_OPPONENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "basic": BasicHeuristicAgent,
    "strong": StrongHeuristicAgent,
}


@dataclass(frozen=True, slots=True)
class RecurrentSmokeConfig:
    """Everything one Phase 8 recurrent smoke run needs.

    The defaults are deliberately tiny. This is a correctness harness: a few
    environments, a short horizon and one or two updates. Anything larger would
    invite reading a number off it as if it meant something.
    """

    run_name: str = "ppo_gru_smoke"
    environment_count: int = 4
    rollout_length: int = 96
    total_updates: int = 2
    chunk_length: int = DEFAULT_CHUNK_LENGTH

    training_opponent: str = "basic"
    opponent_seed: int = 4
    environment_root_seed: int = 0
    initialization_seed: int = 1
    action_sampling_seed: int = 2
    minibatch_shuffle_seed: int = 3

    checkpoint_directory: Path = Path("checkpoints")
    checkpoint_every_updates: int = 1
    check_epoch_zero: bool = True

    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(update_epochs=1, minibatch_count=2))
    architecture: RecurrentArchitectureSpec = field(default_factory=RecurrentArchitectureSpec)

    def __post_init__(self) -> None:
        if self.environment_count < 1 or self.rollout_length < 1:
            raise ValueError("environment_count and rollout_length must be at least 1")
        if self.total_updates < 1:
            raise ValueError("total_updates must be at least 1")
        if self.training_opponent not in SELECTABLE_TRAINING_OPPONENTS:
            raise ValueError(
                f"training_opponent must be one of {sorted(SELECTABLE_TRAINING_OPPONENTS)}, "
                f"got {self.training_opponent!r}"
            )

    def seeds(self) -> dict[str, int]:
        return {
            "environment_root_seed": self.environment_root_seed,
            "initialization_seed": self.initialization_seed,
            "action_sampling_seed": self.action_sampling_seed,
            "minibatch_shuffle_seed": self.minibatch_shuffle_seed,
            "opponent_seed": self.opponent_seed,
        }

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        return values


@dataclass
class RecurrentSmokeResult:
    """What one smoke run produced. Diagnostics only, never a training result."""

    run_name: str
    completed_updates: int = 0
    total_environment_steps: int = 0
    wall_clock_seconds: float = 0.0
    rollout_metrics: list[RecurrentRolloutMetrics] = field(default_factory=list)
    update_metrics: list[RecurrentUpdateMetrics] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)
    resumed_from: str | None = None
    resumed_at_update: int = 0

    @property
    def worst_epoch_zero_deviation(self) -> float:
        """The largest epoch-zero log-probability deviation seen across updates."""
        if not self.update_metrics:
            return 0.0
        return max(
            metrics.epoch_zero_log_probability_max_deviation for metrics in self.update_metrics
        )

    @property
    def illegal_action_count(self) -> int:
        return sum(metrics.illegal_action_count for metrics in self.rollout_metrics)

    @property
    def no_trump_action_count(self) -> int:
        return sum(metrics.no_trump_action_count for metrics in self.rollout_metrics)

    @property
    def learner_decision_count(self) -> int:
        return sum(metrics.learner_decision_count for metrics in self.rollout_metrics)

    def losses_are_finite(self) -> bool:
        return all(
            np.isfinite(
                [
                    metrics.policy_loss,
                    metrics.value_loss,
                    metrics.entropy,
                    metrics.gradient_norm,
                ]
            ).all()
            for metrics in self.update_metrics
        )


def load_recurrent_smoke_config(path: Path) -> RecurrentSmokeConfig:
    """Build a :class:`RecurrentSmokeConfig` from a YAML file.

    Unknown keys raise rather than being ignored: a silently dropped
    hyper-parameter is a reproducibility bug that is very hard to notice later.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return recurrent_smoke_config_from_dict(raw)


def recurrent_smoke_config_from_dict(raw: dict[str, Any]) -> RecurrentSmokeConfig:
    """Build a :class:`RecurrentSmokeConfig` from a plain nested dictionary."""
    values = dict(raw)
    ppo = PPOConfig(**values.pop("ppo", {}))

    architecture_values = dict(values.pop("architecture", {}))
    if "event_embedding_dims" in architecture_values:
        architecture_values["event_embedding_dims"] = tuple(
            architecture_values["event_embedding_dims"]
        )
    architecture = RecurrentArchitectureSpec(**architecture_values)

    if "checkpoint_directory" in values:
        values["checkpoint_directory"] = Path(values["checkpoint_directory"])

    unknown_keys = set(values) - set(RecurrentSmokeConfig.__slots__)
    if unknown_keys:
        raise ValueError(f"unknown recurrent smoke config keys: {sorted(unknown_keys)}")

    return RecurrentSmokeConfig(ppo=ppo, architecture=architecture, **values)


def build_baseline_agent(name: str, seed: int) -> Agent:
    """Return one frozen non-neural opponent, seeded where it is stochastic."""
    if name not in BASELINE_OPPONENTS:
        raise ValueError(f"unknown opponent {name!r}; choose from {sorted(BASELINE_OPPONENTS)}")
    agent_class = BASELINE_OPPONENTS[name]
    if agent_class is RandomAgent:
        return agent_class(seed=seed)
    return agent_class()


def build_smoke_opponents(config: RecurrentSmokeConfig) -> list[Agent]:
    """One opponent instance per environment, each with its own derived seed."""
    return [
        build_baseline_agent(config.training_opponent, config.opponent_seed + index)
        for index in range(config.environment_count)
    ]


def build_smoke_collector(
    config: RecurrentSmokeConfig,
    rules: RulesConfig | None = None,
) -> tuple[RecurrentRolloutCollector, torch.Generator, np.random.Generator]:
    """Build the collector and both explicit RNG streams for a smoke run."""
    resolved_rules = rules if rules is not None else RulesConfig()
    environments = VecTarneeb(
        num_envs=config.environment_count,
        root_seed=config.environment_root_seed,
        rules=resolved_rules,
    )
    actor, critic = build_recurrent_actor_and_critic(
        config.architecture, seed=config.initialization_seed
    )
    action_sampling_generator = torch.Generator()
    action_sampling_generator.manual_seed(config.action_sampling_seed)
    minibatch_shuffle_generator = np.random.default_rng(config.minibatch_shuffle_seed)

    collector = RecurrentRolloutCollector(
        environments=environments,
        actor=actor,
        critic=critic,
        rollout_length=config.rollout_length,
        action_sampling_generator=action_sampling_generator,
        opponents=build_smoke_opponents(config),
        chunk_length=config.chunk_length,
    )
    return collector, action_sampling_generator, minibatch_shuffle_generator


def capture_recurrent_resume_state(
    collector: RecurrentRolloutCollector,
    action_sampling_generator: torch.Generator,
    minibatch_shuffle_generator: np.random.Generator,
    completed_updates: int,
    total_environment_steps: int,
) -> dict[str, Any]:
    """Snapshot every piece of mutable state a recurrent run would otherwise lose.

    Weights and optimizer state live in the checkpoint proper. This is everything
    else, including the recurrent state that is genuinely not derivable: hidden
    slots, pending event queues, consumption cursors and chunk positions. Engine
    states are captured whole because a deal in progress is not a function of its
    seed -- the seed fixes the cards, not the actions already taken.
    """
    return {
        "completed_updates": int(completed_updates),
        "total_environment_steps": int(total_environment_steps),
        "action_sampling_generator_state": action_sampling_generator.get_state(),
        "minibatch_shuffle_generator_state": minibatch_shuffle_generator.bit_generator.state,
        "environment_states": [
            environment.state for environment in collector.environments.environments
        ],
        "environment_deal_indices": [
            environment.deal_index for environment in collector.environments.environments
        ],
        "environment_match_seeds": [
            environment.match_seed for environment in collector.environments.environments
        ],
        "opponent_generator_states": [
            _opponent_generator_state(opponent) for opponent in collector.opponents
        ],
        "collector": collector.snapshot(),
    }


def _opponent_generator_state(opponent: Agent) -> dict[str, Any] | None:
    generator = getattr(opponent, "rng", None)
    if generator is None:
        return None
    state: dict[str, Any] = generator.bit_generator.state
    return state


def restore_recurrent_resume_state(
    resume_state: dict[str, Any],
    collector: RecurrentRolloutCollector,
    action_sampling_generator: torch.Generator,
    minibatch_shuffle_generator: np.random.Generator,
) -> tuple[int, int]:
    """Put a snapshot back. Returns ``(completed_updates, environment_steps)``."""
    action_sampling_generator.set_state(resume_state["action_sampling_generator_state"])
    minibatch_shuffle_generator.bit_generator.state = resume_state[
        "minibatch_shuffle_generator_state"
    ]

    environments = collector.environments.environments
    stored_states = resume_state["environment_states"]
    if len(stored_states) != len(environments):
        raise ValueError(
            f"checkpoint holds {len(stored_states)} environment states but the run has "
            f"{len(environments)}; resume requires the same environment count"
        )
    for environment, state, deal_index, match_seed in zip(
        environments,
        stored_states,
        resume_state["environment_deal_indices"],
        resume_state["environment_match_seeds"],
        strict=True,
    ):
        environment.state = state
        environment.deal_index = deal_index
        environment.match_seed = match_seed

    for opponent, generator_state in zip(
        collector.opponents, resume_state["opponent_generator_states"], strict=True
    ):
        if generator_state is not None and hasattr(opponent, "rng"):
            opponent.rng.bit_generator.state = generator_state

    collector.restore(resume_state["collector"])
    return int(resume_state["completed_updates"]), int(resume_state["total_environment_steps"])


def run_recurrent_smoke(
    config: RecurrentSmokeConfig,
    rules: RulesConfig | None = None,
    resume_from: Path | None = None,
    write_checkpoints: bool = True,
) -> RecurrentSmokeResult:
    """Run a tiny recurrent PPO job and return its records.

    This is a correctness harness. It is not a training run, it is not tuned, and
    no number it produces is a performance claim.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    collector, action_sampling_generator, minibatch_shuffle_generator = build_smoke_collector(
        config, resolved_rules
    )
    optimizer = build_recurrent_optimizer(collector.actor, collector.critic, config.ppo)
    result = RecurrentSmokeResult(run_name=config.run_name)
    already_completed_updates = 0

    if resume_from is not None:
        loaded = load_recurrent_checkpoint(resume_from, rules=resolved_rules, load_critic=True)
        if loaded.critic is None or loaded.payload.get("resume") is None:
            raise ValueError(
                f"{resume_from} cannot be resumed from: it carries no critic or no resume state"
            )
        collector.actor.load_state_dict(loaded.payload["actor_state_dict"])
        collector.critic.load_state_dict(loaded.payload["critic_state_dict"])
        restore_recurrent_optimizer(optimizer, loaded)
        already_completed_updates, result.total_environment_steps = restore_recurrent_resume_state(
            loaded.payload["resume"],
            collector,
            action_sampling_generator,
            minibatch_shuffle_generator,
        )
        result.resumed_from = str(resume_from)
        result.resumed_at_update = already_completed_updates

    start_time = time.perf_counter()
    for update_index in range(already_completed_updates, config.total_updates):
        buffer, rollout_metrics = collector.collect()
        result.rollout_metrics.append(rollout_metrics)
        result.total_environment_steps += rollout_metrics.environment_steps

        batch = buffer.to_chunks(
            collector.bootstrap_values(),
            config.ppo.discount_factor,
            config.ppo.gae_lambda,
        )
        if not batch.is_finite():
            raise AssertionError("the recurrent rollout produced a non-finite tensor")

        learning_rate = (
            linear_learning_rate(config.ppo.learning_rate, update_index, config.total_updates)
            if config.ppo.anneal_learning_rate
            else config.ppo.learning_rate
        )
        update_metrics = recurrent_ppo_update(
            actor=collector.actor,
            critic=collector.critic,
            optimizer=optimizer,
            batch=batch,
            config=config.ppo,
            shuffle_generator=minibatch_shuffle_generator,
            learning_rate=learning_rate,
            check_epoch_zero=config.check_epoch_zero,
        )
        result.update_metrics.append(update_metrics)
        result.completed_updates = update_index + 1

        should_checkpoint = write_checkpoints and (
            result.completed_updates % config.checkpoint_every_updates == 0
            or result.completed_updates == config.total_updates
        )
        if should_checkpoint:
            path = config.checkpoint_directory / (
                f"{config.run_name}_update{result.completed_updates}.pt"
            )
            save_recurrent_checkpoint(
                path=path,
                actor=collector.actor,
                critic=collector.critic,
                optimizer=optimizer,
                rules=resolved_rules,
                training_config=config.as_dict(),
                global_environment_steps=result.total_environment_steps,
                completed_updates=result.completed_updates,
                wall_clock_seconds=time.perf_counter() - start_time,
                seeds=config.seeds(),
                chunk_length=config.chunk_length,
                resume_state=capture_recurrent_resume_state(
                    collector,
                    action_sampling_generator,
                    minibatch_shuffle_generator,
                    result.completed_updates,
                    result.total_environment_steps,
                ),
            )
            result.checkpoint_paths.append(path)

    result.wall_clock_seconds = time.perf_counter() - start_time
    return result


def smoke_run_signature(result: RecurrentSmokeResult) -> list[tuple[float, ...]]:
    """Return a compact numeric fingerprint of a run, for determinism comparisons.

    Two runs agree when their signatures agree. Using the reported metrics rather
    than the weights keeps the comparison readable when it fails, and the
    epoch-zero deviation is included so a divergence in the recurrent bookkeeping
    shows up here rather than only in a loss.
    """
    return [
        (
            float(metrics.policy_loss),
            float(metrics.value_loss),
            float(metrics.entropy),
            float(metrics.approximate_kl_divergence),
            float(metrics.gradient_norm),
            float(metrics.epoch_zero_log_probability_max_deviation),
        )
        for metrics in result.update_metrics
    ]
