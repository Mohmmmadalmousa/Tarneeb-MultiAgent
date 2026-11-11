"""Driving one curriculum stage with the frozen Phase 6 training stack.

This is glue, not an algorithm. Every piece of machinery it uses is frozen:

- :class:`~tarneeb.training.rollout.RolloutCollector` collects the rollout;
- :class:`~tarneeb.buffers.rollout.RolloutBuffer` stores it, attributes the deal
  reward and computes GAE;
- :func:`~tarneeb.training.ppo.ppo_update` performs the update;
- :func:`~tarneeb.models.checkpoint.save_checkpoint` writes the checkpoint;
- the Phase 6 reporters own every file that gets written.

The frozen :func:`~tarneeb.training.train.train` builds its own ``VecTarneeb``
internally and offers no hook to replace it, so a curriculum stage -- which needs a
scripted auction or an ablated history -- cannot run through it. Rather than modify
a frozen file, this module supplies the same loop around a
:class:`~tarneeb.curriculum.scripted_env.CurriculumVecTarneeb`.

Evaluation happens on a schedule fixed **before** training starts, so the reported
checkpoints are not chosen after the fact for looking good.

At every scheduled evaluation the auction policy is also measured separately, on a
fixed probe of real bidding states. Whole-policy entropy proved unable to detect
the stage 6 auction collapse -- it stayed at 0.125 while the auction policy sat at
exactly 0.0 -- so auction-specific entropy is recorded alongside it. That
measurement is a diagnostic only: it never enters the PPO objective, which keeps
using the frozen whole-policy entropy term with its configured coefficient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb.agents.base import Agent
from tarneeb.core.rules import RulesConfig
from tarneeb.curriculum.agents import HistoryConditionedNeuralAgent
from tarneeb.curriculum.auction_diagnostics import (
    AuctionProbe,
    auction_entropy,
    collect_auction_probe,
    flatten_probe_report,
    measure_auction_policy,
)
from tarneeb.curriculum.evaluation import (
    StageEvaluation,
    evaluate_full_game,
    evaluate_stage,
)
from tarneeb.curriculum.scripted_env import build_curriculum_environments
from tarneeb.curriculum.stages import LEARNED_AUCTION, StageSpec
from tarneeb.models.checkpoint import load_checkpoint, restore_optimizer, save_checkpoint
from tarneeb.models.nets import (
    ActorNetwork,
    CriticNetwork,
    build_actor_and_critic,
    count_parameters,
)
from tarneeb.training.config import TrainingConfig
from tarneeb.training.ppo import build_optimizer, linear_learning_rate, ppo_update
from tarneeb.training.reporting import MetricsRecord, Reporter
from tarneeb.training.rollout import RolloutCollector
from tarneeb.training.train import (
    build_baseline_agent,
    build_training_opponents,
    capture_resume_state,
)


@dataclass(frozen=True, slots=True)
class ContinuationProvenance:
    """Exactly what a partial continuation restored, and what it did not.

    This record exists so a continuation can never be mistaken for a
    deterministic resume. The distinction is real: the original stage 4
    checkpoints were written before the trainer saved resume state, so their
    environment and RNG streams were never recorded and cannot be recovered.
    """

    source_checkpoint: str
    source_completed_updates: int
    source_environment_steps: int
    optimizer_restored: bool
    optimizer_state_entries: int
    optimizer_parameter_groups: int
    learning_rate_schedule_continued: bool
    scheduled_total_updates: int
    continuation_effective_updates: str
    exact_resume: bool
    environment_stream_restored: bool
    rng_streams_restored: bool
    fresh_continuation_seeds: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "weight + optimizer + LR-schedule continuation, fresh data/RNG stream",
            "source_checkpoint": self.source_checkpoint,
            "source_effective_update": self.source_completed_updates,
            "source_environment_steps": self.source_environment_steps,
            "optimizer_restored": self.optimizer_restored,
            "optimizer_state_entries": self.optimizer_state_entries,
            "optimizer_parameter_groups": self.optimizer_parameter_groups,
            "learning_rate_schedule_continued": self.learning_rate_schedule_continued,
            "scheduled_total_updates": self.scheduled_total_updates,
            "continuation_effective_updates": self.continuation_effective_updates,
            "exact_resume": self.exact_resume,
            "environment_stream_restored": self.environment_stream_restored,
            "rng_streams_restored": self.rng_streams_restored,
            "fresh_continuation_seeds": dict(self.fresh_continuation_seeds),
            "disclosure": (
                "The data/RNG stream was restarted because the original checkpoint "
                "did not contain resume state. This is NOT a deterministic resume."
            ),
        }


@dataclass
class StageResult:
    """Everything one curriculum stage produced, as plain values."""

    stage: dict[str, Any]
    config: dict[str, Any]
    update_records: list[dict[str, Any]] = field(default_factory=list)
    evaluation_records: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_paths: list[str] = field(default_factory=list)
    total_environment_steps: int = 0
    total_learner_decisions: int = 0
    wall_clock_seconds: float = 0.0
    initialized_from: str | None = None
    continuation: dict[str, Any] | None = None
    starting_environment_steps: int = 0
    schedule_offset_updates: int = 0
    scheduled_total_updates: int = 0

    @property
    def environment_steps_per_second(self) -> float:
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.total_environment_steps / self.wall_clock_seconds

    @property
    def final_checkpoint(self) -> Path | None:
        return Path(self.checkpoint_paths[-1]) if self.checkpoint_paths else None

    @property
    def final_evaluation(self) -> dict[str, Any] | None:
        return self.evaluation_records[-1] if self.evaluation_records else None


def build_stage_evaluation_agent(
    actor: torch.nn.Module,
    stage: StageSpec,
    evaluation_seed: int,
    rules: RulesConfig,
    deterministic: bool = False,
) -> Agent:
    """Wrap the live actor as an evaluation agent under this stage's conditions."""
    return HistoryConditionedNeuralAgent(
        actor=actor,  # type: ignore[arg-type]
        history_condition=stage.history_condition,
        name="neural",
        deterministic=deterministic,
        seed=evaluation_seed,
        rules=rules,
    )


def run_stage_evaluation(
    actor: torch.nn.Module,
    stage: StageSpec,
    opponent_name: str,
    paired_deal_count: int,
    evaluation_seed: int,
    bootstrap_resamples: int,
    deterministic_actions: bool,
    rules: RulesConfig,
) -> StageEvaluation:
    """Evaluate the current policy on held-out deals, in isolation.

    The actor is put in evaluation mode and its previous mode restored, a fresh
    arena and a fresh opponent are built, and no training generator is consumed.
    """
    was_training = actor.training
    actor.eval()
    try:
        neural_agent = build_stage_evaluation_agent(
            actor=actor,
            stage=stage,
            evaluation_seed=evaluation_seed,
            rules=rules,
            deterministic=deterministic_actions,
        )
        opponent = build_baseline_agent(opponent_name, seed=evaluation_seed)
        with torch.no_grad():
            if stage.uses_scripted_auction:
                return evaluate_stage(
                    stage=stage,
                    policy_a=neural_agent,
                    policy_b=opponent,
                    paired_deal_count=paired_deal_count,
                    evaluation_seed=evaluation_seed,
                    bootstrap_resamples=bootstrap_resamples,
                    rules=rules,
                )
            full_game = evaluate_full_game(
                policy_a=neural_agent,
                policy_b=opponent,
                paired_deal_count=paired_deal_count,
                evaluation_seed=evaluation_seed,
                bootstrap_resamples=bootstrap_resamples,
                rules=rules,
            )
            return StageEvaluation(
                stage_name=stage.name,
                policy_a_name=full_game.policy_a_name,
                policy_b_name=full_game.policy_b_name,
                evaluation_seed=evaluation_seed,
                paired_deal_count=paired_deal_count,
                metrics=full_game.metrics,
                trump_suit_breakdown={},
            )
    finally:
        if was_training:
            actor.train()


def _restore_partial_continuation(
    path: Path,
    actor: ActorNetwork,
    critic: CriticNetwork,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    rules: RulesConfig,
) -> tuple[ActorNetwork, CriticNetwork, torch.optim.Optimizer, ContinuationProvenance]:
    """Restore actor, critic and optimizer, and describe exactly what was not restored.

    The optimizer matters: Adam keeps running gradient moments, and throwing them
    away turns a continuation into a restart with a cold optimizer. The frozen
    ``restore_optimizer`` raises if the checkpoint carries no optimizer state, so
    a silent reset is impossible here.
    """
    loaded = load_checkpoint(path, rules=rules, load_critic=True)
    if loaded.critic is None:
        raise ValueError(f"{path} carries no critic; a continuation needs both networks")

    actor.load_state_dict(loaded.payload["actor_state_dict"])
    critic.load_state_dict(loaded.payload["critic_state_dict"])
    restore_optimizer(optimizer, loaded)

    optimizer_state = optimizer.state_dict()
    source_completed_updates = int(loaded.payload["training"]["completed_updates"])
    scheduled_total_updates = source_completed_updates + config.total_updates

    return (
        actor,
        critic,
        optimizer,
        ContinuationProvenance(
            source_checkpoint=str(path),
            source_completed_updates=source_completed_updates,
            source_environment_steps=int(loaded.payload["training"]["global_environment_steps"]),
            optimizer_restored=True,
            optimizer_state_entries=len(optimizer_state["state"]),
            optimizer_parameter_groups=len(optimizer_state["param_groups"]),
            learning_rate_schedule_continued=True,
            scheduled_total_updates=scheduled_total_updates,
            continuation_effective_updates=(
                f"{source_completed_updates + 1}-{scheduled_total_updates}"
            ),
            exact_resume=False,
            environment_stream_restored=loaded.payload.get("resume") is not None,
            rng_streams_restored=loaded.payload.get("resume") is not None,
            fresh_continuation_seeds=config.seeds(),
        ),
    )


def train_curriculum_stage(
    config: TrainingConfig,
    stage: StageSpec,
    report: Reporter | None = None,
    report_evaluation: Reporter | None = None,
    initialize_from: Path | None = None,
    resume_training_state_from: Path | None = None,
    auction_probe: AuctionProbe | None = None,
) -> StageResult:
    """Train one curriculum stage and return its records.

    Two mutually exclusive ways to start from an existing checkpoint:

    ``initialize_from`` -- **weights only**, unchanged. The optimizer, the
    environments, every random stream and the learning-rate schedule all start
    fresh. This is what moving between *different* curriculum tasks wants, since
    stale optimizer moments from the previous task would fight the new gradient.

    ``resume_training_state_from`` -- a **partial continuation** of the *same*
    task: actor, critic and optimizer are restored, and the learning-rate
    schedule continues rather than restarting. It is deliberately not called a
    resume, because the environment and RNG streams are **not** restored -- older
    checkpoints were written before this trainer saved resume state, and that
    state cannot be reconstructed. Every continuation records a
    :class:`ContinuationProvenance` saying exactly that.

    ``auction_probe`` is measured at every scheduled evaluation for a stage whose
    auction is learned. Passing ``None`` collects one automatically; a scripted
    stage skips it, having no auction decisions of its own.
    """
    rules = RulesConfig()

    environments = build_curriculum_environments(
        num_envs=config.environment_count,
        stage=stage,
        root_seed=config.environment_root_seed,
        rules=rules,
    )
    actor, critic = build_actor_and_critic(config.architecture, seed=config.initialization_seed)

    if initialize_from is not None and resume_training_state_from is not None:
        raise ValueError(
            "pass either initialize_from (weights only) or resume_training_state_from "
            "(partial continuation), not both"
        )

    result = StageResult(stage=stage.as_dict(), config=config.as_dict())
    if initialize_from is not None:
        loaded = load_checkpoint(initialize_from, rules=rules, load_critic=True)
        actor.load_state_dict(loaded.payload["actor_state_dict"])
        if loaded.payload.get("critic_state_dict") is not None:
            critic.load_state_dict(loaded.payload["critic_state_dict"])
        result.initialized_from = str(initialize_from)

    optimizer = build_optimizer(actor, critic, config.ppo)

    schedule_offset = 0
    scheduled_total_updates = config.total_updates
    if resume_training_state_from is not None:
        actor, critic, optimizer, provenance = _restore_partial_continuation(
            path=resume_training_state_from,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            config=config,
            rules=rules,
        )
        schedule_offset = provenance.source_completed_updates
        scheduled_total_updates = provenance.scheduled_total_updates
        result.continuation = provenance.as_dict()
        result.starting_environment_steps = provenance.source_environment_steps
        result.total_environment_steps = provenance.source_environment_steps

    result.schedule_offset_updates = schedule_offset
    result.scheduled_total_updates = scheduled_total_updates
    action_sampling_generator = torch.Generator()
    action_sampling_generator.manual_seed(config.action_sampling_seed)
    minibatch_shuffle_generator = np.random.default_rng(config.minibatch_shuffle_seed)

    stage_opponent_config = config.with_overrides(training_opponent=stage.opponent)
    collector = RolloutCollector(
        environments=environments,
        actor=actor,
        critic=critic,
        rollout_length=config.rollout_length,
        action_sampling_generator=action_sampling_generator,
        opponents=build_training_opponents(stage_opponent_config),
    )

    total_updates = config.total_updates
    evaluation_updates = planned_evaluation_updates(total_updates, config.evaluation.every_updates)

    probe: AuctionProbe | None = None
    if stage.auction_mode == LEARNED_AUCTION:
        probe = auction_probe if auction_probe is not None else collect_auction_probe(rules=rules)

    start_time = time.perf_counter()

    for update_index in range(total_updates):
        buffer, rollout_metrics = collector.collect()
        batch = buffer.to_batch(
            collector.bootstrap_values(), config.ppo.discount_factor, config.ppo.gae_lambda
        )

        # The schedule is expressed in EFFECTIVE update numbers, so a partial
        # continuation picks up where the source run's schedule would have been
        # rather than restarting the anneal at the initial learning rate.
        learning_rate = (
            linear_learning_rate(
                config.ppo.learning_rate,
                schedule_offset + update_index,
                scheduled_total_updates,
            )
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
        result.total_learner_decisions += rollout_metrics.learner_decision_count
        result.wall_clock_seconds = time.perf_counter() - start_time

        effective_update = schedule_offset + update_index + 1
        update_record: MetricsRecord = {
            "stage": stage.curriculum_stage,
            "stage_name": stage.name,
            "update": update_index + 1,
            "effective_update": effective_update,
            "learning_rate_scheduled": learning_rate,
            "global_environment_steps": result.total_environment_steps,
            "global_learner_decisions": result.total_learner_decisions,
            "update_seconds": update_seconds,
            "environment_steps_per_second": result.environment_steps_per_second,
            **rollout_metrics.as_dict(),
            **update_metrics.as_dict(),
        }
        result.update_records.append(update_record)
        if report is not None:
            report(update_record)

        completed_updates = update_index + 1
        if completed_updates in evaluation_updates:
            evaluation = run_stage_evaluation(
                actor=actor,
                stage=stage,
                opponent_name=stage.opponent,
                paired_deal_count=config.evaluation.paired_deal_count,
                evaluation_seed=config.evaluation.evaluation_seed,
                bootstrap_resamples=config.evaluation.bootstrap_resamples,
                deterministic_actions=config.evaluation.deterministic_actions,
                rules=rules,
            )
            evaluation_record = evaluation.as_dict()
            evaluation_record["update"] = completed_updates
            evaluation_record["effective_update"] = schedule_offset + completed_updates
            evaluation_record["learning_rate_scheduled"] = learning_rate
            evaluation_record["global_environment_steps"] = result.total_environment_steps
            if probe is not None:
                probe_report = measure_auction_policy(
                    actor, probe, observations=environments.observations, rules=rules
                )
                evaluation_record["auction_probe"] = probe_report
                evaluation_record["auction_entropy"] = auction_entropy(probe_report)
                evaluation_record.update(flatten_probe_report(probe_report))
            result.evaluation_records.append(evaluation_record)
            if report_evaluation is not None:
                report_evaluation(evaluation_record)

        if completed_updates in evaluation_updates or completed_updates == total_updates:
            path = save_checkpoint(
                path=config.checkpoint_directory
                / f"{stage.name}_update{schedule_offset + completed_updates}.pt",
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                rules=rules,
                training_config={
                    "training": config.as_dict(),
                    "stage": stage.as_dict(),
                },
                global_environment_steps=result.total_environment_steps,
                completed_updates=completed_updates,
                wall_clock_seconds=result.wall_clock_seconds,
                seeds=config.seeds(),
                evaluation=result.final_evaluation,
                # Phase 7 previously wrote no resume state, which is why the
                # original stage 4 checkpoints cannot be resumed exactly. New
                # checkpoints carry it, using the frozen Phase 6 capture.
                resume_state=capture_resume_state(
                    collector=collector,
                    action_sampling_generator=action_sampling_generator,
                    minibatch_shuffle_generator=minibatch_shuffle_generator,
                    completed_updates=schedule_offset + completed_updates,
                    total_environment_steps=result.total_environment_steps,
                ),
            )
            result.checkpoint_paths.append(str(path))

    result.wall_clock_seconds = time.perf_counter() - start_time
    return result


def planned_evaluation_updates(total_updates: int, every_updates: int) -> set[int]:
    """Return the update numbers that will be evaluated, decided up front.

    Always includes the final update, so a stage always reports where it ended.
    """
    if every_updates < 1:
        raise ValueError(f"every_updates must be at least 1, got {every_updates}")
    scheduled = set(range(every_updates, total_updates + 1, every_updates))
    scheduled.add(total_updates)
    return scheduled


def parameter_summary(config: TrainingConfig) -> dict[str, int]:
    """Return the parameter counts a stage will train. Used by the scripts."""
    actor, critic = build_actor_and_critic(config.architecture, seed=config.initialization_seed)
    return {"actor": count_parameters(actor), "critic": count_parameters(critic)}
