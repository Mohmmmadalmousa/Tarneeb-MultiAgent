"""The Phase 9 training driver: recurrent PPO, and its matched feedforward twin.

One driver runs **both** A1 conditions. That is the point: if the recurrent and
feedforward runs went through separate code, any difference in budget,
evaluation cadence, seeding, opponent handling or checkpoint selection would
become a confound, and the comparison would measure the harness rather than the
architecture. Here the loop, the budget, the seeds, the opponents, the evaluation
protocol and the stopping rule are shared, and only the model-specific pieces --
collector, buffer, update and checkpoint format -- branch.

**What is deliberately not here.**

*Self-play.* Phase 7 measured four-copy self-play collapsing to an all-pass
equilibrium: pass rate 1.000, every deal thrown in, entropy zero, no signal at
all. Phase 9's exit gate is stable recurrent training plus fair A1 evidence, not
a win-rate threshold, so self-play is not needed to satisfy it and introducing it
casually would risk repeating a known failure. Opponents are fixed frozen
baselines.

*A centralized critic.* The critic reads the 692-dim actor-safe observation, as
in Phases 6-8. The 856-dim vector is Phase 10.

*League machinery.* No PFSP, no historical opponent pool, no Elo. Phase 11.

**Update boundaries and guardrail G6.** An earlier version of this driver used a
fixed rollout window that could end mid-deal, relying on the chunk boundary there
to keep gradients from crossing the optimizer step. That is not sufficient.
Detaching stops gradient flow; it does **not** make a carried hidden state
semantically current under the new weights. Measured here, one ordinary update
moved the GRU by 2.7e-4 and left the carried actor state 1.1e-2 away from the
same history recomputed with the updated network -- against a control of 1.5e-7.
``project requirements`` G6 forbids exactly that continuation, and epoch-zero cannot detect
it, because epoch-zero is a statement about fixed weights.

This driver therefore collects through
:class:`~tarneeb.training.complete_deal_rollout.CompleteDealRecurrentCollector`,
which reaches the step budget and then finishes every active deal before the
update, leaving every recurrent slot zero. No hidden state survives an optimizer
step at all. The matched feedforward condition runs the same schedule so the
comparison stays about the history encoder.

The batch size therefore varies between updates. That is accepted deliberately:
truncating an active deal to hit a round number is the behaviour G6 forbids.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb.agents.base import Agent
from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.neural_agent import NeuralAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.buffers.recurrent import DEFAULT_CHUNK_LENGTH
from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import VecTarneeb
from tarneeb.evaluation.arena import Arena
from tarneeb.models.bc_checkpoint import load_bc_checkpoint
from tarneeb.models.checkpoint import save_checkpoint
from tarneeb.models.nets import ArchitectureSpec, build_actor_and_critic, count_parameters
from tarneeb.models.recurrent import (
    RecurrentArchitectureSpec,
    build_recurrent_actor_and_critic,
)
from tarneeb.models.recurrent_checkpoint import save_recurrent_checkpoint
from tarneeb.training.complete_deal_rollout import (
    CompleteDealFeedforwardCollector,
    CompleteDealRecurrentCollector,
)
from tarneeb.training.phase9_seeds import PPO_DEVELOPMENT_SEEDS, training_environment_seed
from tarneeb.training.ppo import (
    EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
    PPOConfig,
    build_optimizer,
    linear_learning_rate,
    ppo_update,
)
from tarneeb.training.recurrent_ppo import (
    RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE,
    build_recurrent_optimizer,
    recurrent_ppo_update,
)

BASELINE_OPPONENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "basic": BasicHeuristicAgent,
    "strong": StrongHeuristicAgent,
}

EVALUATION_OPPONENTS = ("random", "basic", "strong")


@dataclass(frozen=True, slots=True)
class Phase9Config:
    """One pre-registered Phase 9 training run.

    Everything that defines the compute contract is here and is written into the
    run's report, so "what was this run" is answerable from the artifact rather
    than from memory.
    """

    run_name: str
    architecture: str = "gru"
    environment_count: int = 16
    rollout_length: int = 128
    total_updates: int = 250
    chunk_length: int = DEFAULT_CHUNK_LENGTH

    training_opponent: str = "basic"
    bc_checkpoint: Path | None = None

    seed_index: int = 0
    initialization_seed: int = 1
    action_sampling_seed: int = 2
    minibatch_shuffle_seed: int = 3
    opponent_seed: int = 4

    evaluation_every_updates: int = 50
    evaluation_paired_deals: int = 150
    evaluation_bootstrap_resamples: int = 2000
    evaluation_opponents: tuple[str, ...] = EVALUATION_OPPONENTS

    checkpoint_directory: Path = Path("checkpoints")
    results_directory: Path = Path("results/phase9")
    check_epoch_zero: bool = True

    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        if self.architecture not in ("gru", "ff"):
            raise ValueError(f"architecture must be 'gru' or 'ff', got {self.architecture!r}")
        if self.training_opponent not in BASELINE_OPPONENTS:
            raise ValueError(
                f"training_opponent must be one of {sorted(BASELINE_OPPONENTS)}, "
                f"got {self.training_opponent!r}"
            )
        if self.environment_count < 1 or self.rollout_length < 1 or self.total_updates < 1:
            raise ValueError("environment_count, rollout_length and total_updates must be >= 1")

    @property
    def environment_root_seed(self) -> int:
        return training_environment_seed(self.seed_index)

    @property
    def steps_per_update(self) -> int:
        return self.environment_count * self.rollout_length

    @property
    def total_environment_steps(self) -> int:
        """The **nominal** budget.

        Actual collection exceeds it, because the G6-safe collector finishes
        every active deal before updating rather than truncating one to hit a
        round number. The realised counts are recorded per run and reported.
        """
        return self.steps_per_update * self.total_updates

    def seeds(self) -> dict[str, int]:
        return {
            "environment_root_seed": self.environment_root_seed,
            "seed_index": self.seed_index,
            "initialization_seed": self.initialization_seed,
            "action_sampling_seed": self.action_sampling_seed,
            "minibatch_shuffle_seed": self.minibatch_shuffle_seed,
            "opponent_seed": self.opponent_seed,
        }

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        values["results_directory"] = str(self.results_directory)
        values["bc_checkpoint"] = None if self.bc_checkpoint is None else str(self.bc_checkpoint)
        values["environment_root_seed"] = self.environment_root_seed
        values["total_environment_steps"] = self.total_environment_steps
        return values


@dataclass
class Phase9Result:
    """Everything one run produced, including its safety counters."""

    run_name: str
    architecture: str
    completed_updates: int = 0
    total_environment_steps: int = 0
    total_collection_steps: int = 0
    learner_decisions: int = 0
    completed_deals: int = 0
    passed_out_deals: int = 0
    wall_clock_seconds: float = 0.0
    update_records: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    worst_epoch_zero_deviation: float = 0.0
    illegal_action_count: int = 0
    no_trump_action_count: int = 0
    non_finite_update_count: int = 0
    parameter_counts: dict[str, int] = field(default_factory=dict)

    @property
    def decisions_per_second(self) -> float:
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.total_environment_steps / self.wall_clock_seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "architecture": self.architecture,
            "completed_updates": self.completed_updates,
            "total_environment_steps": self.total_environment_steps,
            "total_collection_steps": self.total_collection_steps,
            "learner_decisions": self.learner_decisions,
            "completed_deals": self.completed_deals,
            "passed_out_deals": self.passed_out_deals,
            "wall_clock_seconds": self.wall_clock_seconds,
            "environment_steps_per_second": self.decisions_per_second,
            "worst_epoch_zero_deviation": self.worst_epoch_zero_deviation,
            "illegal_action_count": self.illegal_action_count,
            "no_trump_action_count": self.no_trump_action_count,
            "non_finite_update_count": self.non_finite_update_count,
            "parameter_counts": self.parameter_counts,
            "update_records": self.update_records,
            "evaluations": self.evaluations,
            "checkpoint_paths": self.checkpoint_paths,
        }


def build_opponents(config: Phase9Config) -> list[Agent]:
    """One frozen opponent instance per environment, seeded where stochastic."""
    agent_class = BASELINE_OPPONENTS[config.training_opponent]
    return [
        agent_class(seed=config.opponent_seed + index)
        if agent_class is RandomAgent
        else agent_class()
        for index in range(config.environment_count)
    ]


def _load_bc_actor_weights(actor: torch.nn.Module, path: Path, architecture: str) -> None:
    """Copy BC-cloned weights into a freshly built actor.

    Weights only. The critic is **not** initialized from BC: the Strong heuristic
    has no value function, and its internal evaluations are not an estimate of
    the frozen reward. Starting the critic from a fresh initialization is the
    honest option, and it is what the recurrent contract's PPO-after-BC path
    specifies.
    """
    loaded = load_bc_checkpoint(path)
    if loaded.architecture != architecture:
        raise ValueError(
            f"BC checkpoint is {loaded.architecture!r} but this run is {architecture!r}"
        )
    actor.load_state_dict(loaded.payload["actor_state_dict"])


def build_networks(config: Phase9Config) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build the actor and critic, optionally initializing the actor from BC."""
    actor: torch.nn.Module
    critic: torch.nn.Module
    if config.architecture == "gru":
        actor, critic = build_recurrent_actor_and_critic(
            RecurrentArchitectureSpec(), seed=config.initialization_seed
        )
    else:
        actor, critic = build_actor_and_critic(ArchitectureSpec(), seed=config.initialization_seed)
    if config.bc_checkpoint is not None:
        _load_bc_actor_weights(actor, config.bc_checkpoint, config.architecture)
    return actor, critic


def build_agent_for_evaluation(actor: torch.nn.Module, architecture: str, name: str) -> Agent:
    """Wrap the live actor as an arena agent, for a scheduled evaluation.

    Deterministic: an evaluation should report the policy's considered opinion,
    not a draw from it.
    """
    if architecture == "gru":
        return RecurrentNeuralAgent(actor, name=name, deterministic=True)  # type: ignore[arg-type]
    return NeuralAgent(actor, name=name, deterministic=True)  # type: ignore[arg-type]


def evaluate_against_baselines(
    actor: torch.nn.Module,
    config: Phase9Config,
    completed_updates: int,
    paired_deals: int | None = None,
    evaluation_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate in the frozen arena on development seeds.

    Isolated by construction: its own arena, its own agent instance, its own
    seeds. It consumes none of the training generators and mutates no training
    state.
    """
    was_training = actor.training
    actor.eval()
    pairs = config.evaluation_paired_deals if paired_deals is None else paired_deals
    seed = PPO_DEVELOPMENT_SEEDS.start if evaluation_seed is None else evaluation_seed

    entry: dict[str, Any] = {
        "completed_updates": completed_updates,
        "paired_deal_count": pairs,
        "evaluation_seed": seed,
        "opponents": {},
    }
    for opponent_name in config.evaluation_opponents:
        agent = build_agent_for_evaluation(
            actor, config.architecture, f"{config.run_name}_u{completed_updates}"
        )
        opponent_class = BASELINE_OPPONENTS[opponent_name]
        opponent = opponent_class(seed=0) if opponent_class is RandomAgent else opponent_class()
        arena = Arena()
        evaluation = arena.evaluate(
            agent,
            opponent,
            paired_deal_count=pairs,
            evaluation_seed=seed,
            bootstrap_resamples=config.evaluation_bootstrap_resamples,
        )
        metrics = evaluation.metrics
        entry["opponents"][opponent_name] = {
            "mean_paired_score_difference": metrics["mean_paired_score_difference"],
            "paired_score_difference_ci": metrics["paired_score_difference_ci"],
            "win_rate": metrics["win_rate"],
            "win_rate_wilson_interval": metrics["win_rate_wilson_interval"],
            "contract_success_rate": metrics["contract_success_rate"],
            "mean_bid": metrics["mean_bid"],
            "passed_out_rate": metrics["passed_out_rate"],
            "pass_action_rate": metrics["pass_action_rate"],
            "role_split_mean_score": metrics["role_split_mean_score"],
            "illegal_action_rate": metrics["illegal_action_rate"],
            "no_trump_action_rate": metrics["no_trump_action_rate"],
            "replay_integrity_rate": metrics["replay_integrity_rate"],
        }
    if was_training:
        actor.train()
    return entry


def _write_checkpoint(
    config: Phase9Config,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rules: RulesConfig,
    result: Phase9Result,
    elapsed: float,
) -> Path:
    path = config.checkpoint_directory / f"{config.run_name}_update{result.completed_updates}.pt"
    if config.architecture == "gru":
        save_recurrent_checkpoint(
            path=path,
            actor=actor,  # type: ignore[arg-type]
            critic=critic,  # type: ignore[arg-type]
            optimizer=optimizer,
            rules=rules,
            training_config=config.as_dict(),
            global_environment_steps=result.total_environment_steps,
            completed_updates=result.completed_updates,
            wall_clock_seconds=elapsed,
            seeds=config.seeds(),
            chunk_length=config.chunk_length,
        )
    else:
        save_checkpoint(
            path=path,
            actor=actor,  # type: ignore[arg-type]
            critic=critic,  # type: ignore[arg-type]
            optimizer=optimizer,
            rules=rules,
            training_config=config.as_dict(),
            global_environment_steps=result.total_environment_steps,
            completed_updates=result.completed_updates,
            wall_clock_seconds=elapsed,
            seeds=config.seeds(),
        )
    result.checkpoint_paths.append(str(path))
    return path


def run_phase9_training(
    config: Phase9Config,
    rules: RulesConfig | None = None,
    progress: bool = True,
) -> Phase9Result:
    """Run one pre-registered Phase 9 training job and return its records."""
    resolved_rules = rules if rules is not None else RulesConfig()
    actor, critic = build_networks(config)
    result = Phase9Result(run_name=config.run_name, architecture=config.architecture)
    result.parameter_counts = {
        "actor": count_parameters(actor),
        "critic": count_parameters(critic),
    }

    environments = VecTarneeb(
        num_envs=config.environment_count,
        root_seed=config.environment_root_seed,
        rules=resolved_rules,
    )
    action_generator = torch.Generator()
    action_generator.manual_seed(config.action_sampling_seed)
    shuffle_generator = np.random.default_rng(config.minibatch_shuffle_seed)
    opponents = build_opponents(config)

    recurrent = config.architecture == "gru"
    if recurrent:
        collector: Any = CompleteDealRecurrentCollector(
            environments=environments,
            actor=actor,  # type: ignore[arg-type]
            critic=critic,  # type: ignore[arg-type]
            rollout_length=config.rollout_length,
            action_sampling_generator=action_generator,
            opponents=opponents,
            chunk_length=config.chunk_length,
        )
        optimizer = build_recurrent_optimizer(actor, critic, config.ppo)  # type: ignore[arg-type]
    else:
        collector = CompleteDealFeedforwardCollector(
            environments=environments,
            actor=actor,  # type: ignore[arg-type]
            critic=critic,  # type: ignore[arg-type]
            rollout_length=config.rollout_length,
            action_sampling_generator=action_generator,
            opponents=opponents,
        )
        optimizer = build_optimizer(actor, critic, config.ppo)  # type: ignore[arg-type]

    start_time = time.perf_counter()
    for update_index in range(config.total_updates):
        buffer, rollout_metrics = collector.collect()
        result.total_environment_steps += rollout_metrics.environment_steps
        result.total_collection_steps += collector.last_collection_steps
        result.learner_decisions += rollout_metrics.learner_decision_count
        result.completed_deals += rollout_metrics.completed_deals
        result.passed_out_deals += rollout_metrics.passed_out_deals
        result.illegal_action_count += rollout_metrics.illegal_action_count
        result.no_trump_action_count += rollout_metrics.no_trump_action_count

        learning_rate = (
            linear_learning_rate(config.ppo.learning_rate, update_index, config.total_updates)
            if config.ppo.anneal_learning_rate
            else config.ppo.learning_rate
        )

        if recurrent:
            batch = buffer.to_chunks(
                collector.bootstrap_values(), config.ppo.discount_factor, config.ppo.gae_lambda
            )
            if not batch.is_finite():
                raise AssertionError("the recurrent rollout produced a non-finite tensor")
            update_metrics: Any = recurrent_ppo_update(
                actor=actor,  # type: ignore[arg-type]
                critic=critic,  # type: ignore[arg-type]
                optimizer=optimizer,
                batch=batch,
                config=config.ppo,
                shuffle_generator=shuffle_generator,
                learning_rate=learning_rate,
                check_epoch_zero=config.check_epoch_zero,
            )
            extra = {
                "collection_steps": collector.last_collection_steps,
                "drain_steps": collector.last_collection_steps - config.rollout_length,
                "chunk_count": update_metrics.chunk_count,
                "valid_transition_count": update_metrics.valid_transition_count,
                "max_segment_length": rollout_metrics.max_segment_length,
                "new_hand_boundaries": rollout_metrics.new_hand_boundaries,
                "events_per_decision": (
                    rollout_metrics.total_events_consumed / max(1, rollout_metrics.decision_count)
                ),
            }
        else:
            batch = buffer.to_batch(
                collector.bootstrap_values(), config.ppo.discount_factor, config.ppo.gae_lambda
            )
            if not batch.is_finite():
                raise AssertionError("the rollout produced a non-finite tensor")
            update_metrics = ppo_update(
                actor=actor,  # type: ignore[arg-type]
                critic=critic,  # type: ignore[arg-type]
                optimizer=optimizer,
                batch=batch,
                config=config.ppo,
                shuffle_generator=shuffle_generator,
                learning_rate=learning_rate,
            )
            deviation = update_metrics.epoch_zero_log_probability_max_deviation
            if config.check_epoch_zero and deviation > EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE:
                raise AssertionError(
                    f"epoch-zero log-probability deviation {deviation:.3e} exceeds "
                    f"{EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE:.0e}"
                )
            extra = {
                "collection_steps": collector.last_collection_steps,
                "drain_steps": collector.last_collection_steps - config.rollout_length,
                "decision_count": batch.decision_count,
            }

        result.completed_updates = update_index + 1
        deviation = update_metrics.epoch_zero_log_probability_max_deviation
        result.worst_epoch_zero_deviation = max(result.worst_epoch_zero_deviation, deviation)
        if not np.isfinite(
            [
                update_metrics.policy_loss,
                update_metrics.value_loss,
                update_metrics.entropy,
                update_metrics.gradient_norm,
            ]
        ).all():
            result.non_finite_update_count += 1

        record = {
            "update": result.completed_updates,
            "environment_steps": result.total_environment_steps,
            "learning_rate": learning_rate,
            "policy_loss": update_metrics.policy_loss,
            "value_loss": update_metrics.value_loss,
            "entropy": update_metrics.entropy,
            "approximate_kl_divergence": update_metrics.approximate_kl_divergence,
            "clip_fraction": update_metrics.clip_fraction,
            "explained_variance": update_metrics.explained_variance,
            "gradient_norm": update_metrics.gradient_norm,
            "epoch_zero_deviation": deviation,
            "mean_learner_reward": rollout_metrics.mean_learner_reward,
            "completed_deals": rollout_metrics.completed_deals,
            "passed_out_deals": rollout_metrics.passed_out_deals,
            "pass_rate": (
                rollout_metrics.pass_action_count / rollout_metrics.auction_decision_count
                if rollout_metrics.auction_decision_count
                else None
            ),
            "auction_decisions": rollout_metrics.auction_decision_count,
            "collection_seconds": rollout_metrics.collection_seconds,
            **extra,
        }
        result.update_records.append(record)

        if progress and (result.completed_updates % 10 == 0 or result.completed_updates == 1):
            print(
                f"[{config.run_name}] update {result.completed_updates:>4}/{config.total_updates} "
                f"steps {result.total_environment_steps:>8} "
                f"pl {record['policy_loss']:+.4f} ent {record['entropy']:.3f} "
                f"ev {record['explained_variance']:+.3f} "
                f"rew {record['mean_learner_reward'] if record['mean_learner_reward'] is not None else float('nan'):+.4f} "
                f"e0 {deviation:.1e}",
                flush=True,
            )

        scheduled = result.completed_updates % config.evaluation_every_updates == 0
        if scheduled or result.completed_updates == config.total_updates:
            _write_checkpoint(
                config,
                actor,
                critic,
                optimizer,
                resolved_rules,
                result,
                time.perf_counter() - start_time,
            )
            entry = evaluate_against_baselines(actor, config, result.completed_updates)
            result.evaluations.append(entry)
            if progress:
                summary = " ".join(
                    f"{name}:{values['win_rate']:.3f}"
                    for name, values in entry["opponents"].items()
                )
                print(
                    f"[{config.run_name}] eval @{result.completed_updates}: {summary}", flush=True
                )

    result.wall_clock_seconds = time.perf_counter() - start_time
    return result


def write_run_report(config: Phase9Config, result: Phase9Result) -> Path:
    """Write the run's full record as JSON and return the path."""
    config.results_directory.mkdir(parents=True, exist_ok=True)
    path = config.results_directory / f"{config.run_name}.json"
    payload = {
        "config": config.as_dict(),
        "result": result.as_dict(),
        "epoch_zero_tolerance": (
            RECURRENT_EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
            if config.architecture == "gru"
            else EPOCH_ZERO_LOG_PROBABILITY_TOLERANCE
        ),
    }
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path
