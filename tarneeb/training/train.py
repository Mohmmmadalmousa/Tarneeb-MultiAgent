"""The Phase 6 training loop: collect, update, evaluate, checkpoint.

This module orchestrates; it computes no metric of its own beyond joining the
ones the rollout collector and the PPO update already returned. Following the
observability boundary in project requirements, it prints nothing, opens no writer and
takes no ``verbose`` flag. It returns a :class:`TrainingResult` holding plain
dictionaries, and a caller may render, log or discard them without changing a
single gradient.

Resume. :func:`capture_resume_state` snapshots everything a run needs to continue
exactly where it stopped -- both networks, the optimizer, the action-sampling and
minibatch-shuffling generators, every environment's engine state, each frozen
opponent's own generator, and the update counter that drives learning-rate
annealing. :func:`restore_resume_state` puts it all back, so a saved-and-resumed
run reproduces the uninterrupted run update for update.

Evaluation isolation. :func:`evaluate_against_baseline` builds its own
:class:`~tarneeb.evaluation.arena.Arena`, its own opponent and its own
environments. It draws only from the held-out evaluation seed range, consumes
none of the training generators, puts the networks in evaluation mode and
restores the previous mode when it is finished. It never touches the rollout
buffer, the optimizer or the training environments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb.agents.base import Agent
from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.neural_agent import NeuralAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import VecTarneeb
from tarneeb.evaluation.arena import Arena
from tarneeb.models.checkpoint import load_checkpoint, restore_optimizer, save_checkpoint
from tarneeb.models.nets import ActorNetwork, CriticNetwork, build_actor_and_critic
from tarneeb.training.config import TrainingConfig
from tarneeb.training.ppo import (
    build_optimizer,
    linear_learning_rate,
    ppo_update,
)
from tarneeb.training.reporting import MetricsRecord, Reporter
from tarneeb.training.rollout import RolloutCollector

BASELINE_OPPONENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "basic": BasicHeuristicAgent,
    "strong": StrongHeuristicAgent,
}


def build_baseline_agent(name: str, seed: int) -> Agent:
    """Build one frozen baseline opponent by name.

    ``RandomAgent`` is the only baseline that takes a seed; the heuristics are
    deterministic, so seeding them would be meaningless rather than merely
    harmless.
    """
    if name not in BASELINE_OPPONENTS:
        raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(BASELINE_OPPONENTS)}")
    agent_class = BASELINE_OPPONENTS[name]
    if agent_class is RandomAgent:
        return RandomAgent(name=name, seed=seed)
    return agent_class(name=name)  # type: ignore[call-arg]


def build_training_opponents(config: TrainingConfig) -> list[Agent]:
    """Return one frozen opponent per environment.

    Each environment gets its own instance so a seeded opponent's stream is not
    interleaved across environments, which would make a run depend on the order
    the environments happened to be stepped in.
    """
    return [
        build_baseline_agent(
            config.training_opponent, seed=config.opponent_seed + environment_index
        )
        for environment_index in range(config.environment_count)
    ]


@dataclass
class TrainingResult:
    """Everything one run produced, as plain values."""

    config: dict[str, Any]
    update_records: list[dict[str, Any]] = field(default_factory=list)
    evaluation_records: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    total_environment_steps: int = 0
    wall_clock_seconds: float = 0.0
    resumed_from: str | None = None
    resumed_at_update: int = 0

    @property
    def environment_steps_per_second(self) -> float:
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.total_environment_steps / self.wall_clock_seconds

    @property
    def final_evaluation(self) -> dict[str, Any] | None:
        return self.evaluation_records[-1] if self.evaluation_records else None


def evaluate_against_baseline(
    actor: ActorNetwork,
    opponent_name: str,
    paired_deal_count: int,
    evaluation_seed: int,
    bootstrap_resamples: int,
    deterministic_actions: bool,
    rules: RulesConfig | None = None,
) -> dict[str, Any]:
    """Run one isolated paired evaluation of ``actor`` against a frozen baseline.

    Returns the reportable metrics. The sign convention is Phase 5's: a positive
    mean paired score difference means the neural policy outscored the baseline,
    in points per deal.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    was_training = actor.training
    actor.eval()
    try:
        neural_agent = NeuralAgent(
            actor=actor,
            name="neural",
            deterministic=deterministic_actions,
            seed=evaluation_seed,
            rules=resolved_rules,
        )
        opponent = build_baseline_agent(opponent_name, seed=evaluation_seed)

        arena = Arena(rules=resolved_rules)
        with torch.no_grad():
            result = arena.evaluate(
                policy_a=neural_agent,
                policy_b=opponent,
                paired_deal_count=paired_deal_count,
                evaluation_seed=evaluation_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
    finally:
        if was_training:
            actor.train()

    metrics = dict(result.metrics)
    metrics["opponent"] = opponent_name
    metrics["deterministic_actions"] = deterministic_actions
    metrics["paired_deal_count"] = paired_deal_count
    metrics["evaluation_seed"] = evaluation_seed
    return metrics


def _opponent_generator_state(opponent: Agent) -> dict[str, Any] | None:
    """Return a seeded opponent's generator state, or ``None`` if it has none.

    Only ``RandomAgent`` carries a generator; the heuristics are deterministic
    functions of the view, so they have no stream to preserve.
    """
    generator = getattr(opponent, "rng", None)
    if generator is None:
        return None
    state: dict[str, Any] = generator.bit_generator.state
    return state


def capture_resume_state(
    collector: RolloutCollector,
    action_sampling_generator: torch.Generator,
    minibatch_shuffle_generator: np.random.Generator,
    completed_updates: int,
    total_environment_steps: int,
) -> dict[str, Any]:
    """Snapshot every piece of mutable state a run would otherwise lose.

    Network weights and optimizer state live in the checkpoint proper; this is
    everything else. Engine states are captured whole because a deal in progress
    cannot be re-derived from its seed alone -- the seed fixes the cards, not the
    actions already taken.
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
        "collector_redeal_counts": collector.redeal_counts.tolist(),
    }


def restore_resume_state(
    resume_state: dict[str, Any],
    collector: RolloutCollector,
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

    collector.redeal_counts = np.array(resume_state["collector_redeal_counts"], dtype=np.int64)
    return int(resume_state["completed_updates"]), int(resume_state["total_environment_steps"])


def train(
    config: TrainingConfig,
    report: Reporter | None = None,
    report_evaluation: Reporter | None = None,
    resume_from: Path | None = None,
) -> TrainingResult:
    """Run one Phase 6 training job and return its records.

    ``report`` and ``report_evaluation`` are optional sinks that receive one
    already-detached metrics dictionary per update and per evaluation. Passing
    neither disables logging entirely and changes nothing about learning.
    ``resume_from`` continues a run from a checkpoint written by this function.
    """
    rules = RulesConfig()

    environments = VecTarneeb(
        num_envs=config.environment_count,
        root_seed=config.environment_root_seed,
        rules=rules,
    )
    actor, critic = build_actor_and_critic(config.architecture, seed=config.initialization_seed)
    optimizer = build_optimizer(actor, critic, config.ppo)

    action_sampling_generator = torch.Generator()
    action_sampling_generator.manual_seed(config.action_sampling_seed)
    minibatch_shuffle_generator = np.random.default_rng(config.minibatch_shuffle_seed)

    collector = RolloutCollector(
        environments=environments,
        actor=actor,
        critic=critic,
        rollout_length=config.rollout_length,
        action_sampling_generator=action_sampling_generator,
        opponents=build_training_opponents(config),
    )

    result = TrainingResult(config=config.as_dict())
    total_updates = config.total_updates
    already_completed_updates = 0

    if resume_from is not None:
        loaded = load_checkpoint(resume_from, rules=rules, load_critic=True)
        if loaded.critic is None or loaded.payload.get("resume") is None:
            raise ValueError(
                f"{resume_from} cannot be resumed from: it carries no critic or no resume state"
            )
        actor.load_state_dict(loaded.payload["actor_state_dict"])
        critic.load_state_dict(loaded.payload["critic_state_dict"])
        restore_optimizer(optimizer, loaded)
        already_completed_updates, result.total_environment_steps = restore_resume_state(
            loaded.payload["resume"],
            collector,
            action_sampling_generator,
            minibatch_shuffle_generator,
        )
        result.resumed_from = str(resume_from)
        result.resumed_at_update = already_completed_updates

    start_time = time.perf_counter()

    for update_index in range(already_completed_updates, total_updates):
        buffer, rollout_metrics = collector.collect()
        bootstrap_values = collector.bootstrap_values()
        batch = buffer.to_batch(bootstrap_values, config.ppo.discount_factor, config.ppo.gae_lambda)

        learning_rate = (
            linear_learning_rate(config.ppo.learning_rate, update_index, total_updates)
            if config.ppo.anneal_learning_rate
            else config.ppo.learning_rate
        )
        update_start = time.perf_counter()
        update_metrics = ppo_update(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            batch=batch,
            config=config.ppo,
            shuffle_generator=minibatch_shuffle_generator,
            learning_rate=learning_rate,
        )
        update_seconds = time.perf_counter() - update_start

        result.total_environment_steps += rollout_metrics.environment_steps
        result.wall_clock_seconds = time.perf_counter() - start_time
        update_record: MetricsRecord = {
            "update": update_index + 1,
            "global_environment_steps": result.total_environment_steps,
            "update_seconds": update_seconds,
            "environment_steps_per_second": result.environment_steps_per_second,
            **rollout_metrics.as_dict(),
            **update_metrics.as_dict(),
        }
        result.update_records.append(update_record)
        if report is not None:
            report(update_record)

        is_final_update = update_index + 1 == total_updates
        if is_final_update or (update_index + 1) % config.evaluation.every_updates == 0:
            for opponent_name in config.evaluation.opponent_names:
                evaluation = evaluate_against_baseline(
                    actor=actor,
                    opponent_name=opponent_name,
                    paired_deal_count=config.evaluation.paired_deal_count,
                    evaluation_seed=config.evaluation.evaluation_seed,
                    bootstrap_resamples=config.evaluation.bootstrap_resamples,
                    deterministic_actions=config.evaluation.deterministic_actions,
                    rules=rules,
                )
                evaluation["update"] = update_index + 1
                evaluation["global_environment_steps"] = result.total_environment_steps
                result.evaluation_records.append(evaluation)
                if report_evaluation is not None:
                    report_evaluation(evaluation)

        if is_final_update or (update_index + 1) % config.checkpoint_every_updates == 0:
            path = save_checkpoint(
                path=config.checkpoint_directory / f"{config.run_name}_update{update_index + 1}.pt",
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                rules=rules,
                training_config=config.as_dict(),
                global_environment_steps=result.total_environment_steps,
                completed_updates=update_index + 1,
                wall_clock_seconds=result.wall_clock_seconds,
                seeds=config.seeds(),
                evaluation=result.final_evaluation,
                resume_state=capture_resume_state(
                    collector=collector,
                    action_sampling_generator=action_sampling_generator,
                    minibatch_shuffle_generator=minibatch_shuffle_generator,
                    completed_updates=update_index + 1,
                    total_environment_steps=result.total_environment_steps,
                ),
            )
            result.checkpoint_paths.append(str(path))

    result.wall_clock_seconds = time.perf_counter() - start_time
    return result


def final_checkpoint_path(result: TrainingResult) -> Path | None:
    """Return the last checkpoint a run wrote, if it wrote one."""
    if not result.checkpoint_paths:
        return None
    return Path(result.checkpoint_paths[-1])


def build_actor_for_inference(
    config: TrainingConfig,
) -> tuple[ActorNetwork, CriticNetwork]:
    """Build freshly initialized networks matching ``config``. Used by tests."""
    return build_actor_and_critic(config.architecture, seed=config.initialization_seed)
