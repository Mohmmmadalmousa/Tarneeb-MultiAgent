"""The Phase 10 driver: ablation A2, centralized critic versus decentralized.

One driver runs both conditions, for the same reason Phase 9's did: if control
and treatment went through separate code, any difference in budget, seeding,
evaluation or checkpoint selection would become a confound and the experiment
would measure the harness instead of the critic.

What is held constant
---------------------
Everything except the critic's input width. Same Event-GRU actor, same actor
initialization seed, same BC initialization, same training opponent, same base
collection budget, same complete-deal G6 drain, same PPO hyper-parameters, same
environment count, same optimizer-update count, same environment seeds, same
evaluation schedule, same checkpoint-selection rule, same evaluation deals.

What differs
------------
The critic reads 692 actor-safe features (control) or 856 environment features
(treatment). That is the independent variable, and it is the only one.

What is inherited from Phase 9, not reinvented
----------------------------------------------
- the G6-safe complete-deal update boundary, through
  :class:`~tarneeb.training.centralized_rollout.CentralizedCriticCollector`,
  which subclasses the Phase 9 collector;
- ``evaluate_against_baselines``, used unmodified, so both phases are scored by
  identical machinery;
- the seed splits, so Phase 10 cannot reach the held-out range either.

The actor's information boundary is unchanged and unchangeable here. The actor
is handed the 692 observation and owns no parameter shaped to accept 856, so a
wiring mistake raises rather than leaking.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from tarneeb.buffers.recurrent import DEFAULT_CHUNK_LENGTH
from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.bc_checkpoint import load_bc_checkpoint
from tarneeb.models.centralized_critic import (
    CRITIC_MODE_CENTRALIZED,
    CRITIC_MODE_DECENTRALIZED,
    CRITIC_MODES,
    build_actor_and_critic_for,
    critic_observation_dim_for,
    describe_condition,
    verify_parameter_counts,
)
from tarneeb.models.recurrent import RecurrentActorNetwork, RecurrentCriticNetwork
from tarneeb.models.recurrent_checkpoint import save_recurrent_checkpoint
from tarneeb.training.centralized_rollout import (
    CentralizedCriticCollector,
    DecentralizedCriticCollector,
)
from tarneeb.training.phase9_seeds import training_environment_seed
from tarneeb.training.phase9_training import (
    BASELINE_OPPONENTS,
    EVALUATION_OPPONENTS,
    Phase9Config,
    build_opponents,
    evaluate_against_baselines,
)
from tarneeb.training.ppo import PPOConfig, linear_learning_rate
from tarneeb.training.recurrent_ppo import build_recurrent_optimizer, recurrent_ppo_update


@dataclass(frozen=True, slots=True)
class Phase10Config:
    """One pre-registered A2 run.

    Field names deliberately mirror ``Phase9Config`` so the frozen Phase 9
    evaluation helper can score a Phase 10 run without modification.
    """

    run_name: str
    critic_mode: str = CRITIC_MODE_DECENTRALIZED
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
    results_directory: Path = Path("results/phase10")
    check_epoch_zero: bool = True

    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        if self.critic_mode not in CRITIC_MODES:
            raise ValueError(f"critic_mode must be one of {CRITIC_MODES}, got {self.critic_mode!r}")
        if self.architecture != "gru":
            raise ValueError("Phase 10 varies the critic, not the actor; architecture must be gru")
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
    def nominal_environment_steps(self) -> int:
        """The budget trigger. Realised collection exceeds it by the G6 drain."""
        return self.steps_per_update * self.total_updates

    @property
    def critic_observation_dim(self) -> int:
        return critic_observation_dim_for(self.critic_mode)

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
        values["nominal_environment_steps"] = self.nominal_environment_steps
        values["condition"] = describe_condition(self.critic_mode)
        return values


@dataclass
class Phase10Result:
    """Everything one A2 run produced, including its safety counters."""

    run_name: str
    critic_mode: str
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
    def environment_steps_per_second(self) -> float:
        if self.wall_clock_seconds <= 0:
            return 0.0
        return self.total_environment_steps / self.wall_clock_seconds

    @property
    def mean_explained_variance(self) -> float:
        """The headline A2 diagnostic: does privileged information help the critic?"""
        if not self.update_records:
            return 0.0
        return float(np.mean([r["explained_variance"] for r in self.update_records]))

    @property
    def final_explained_variance(self) -> float:
        """Mean explained variance over the last fifth of training."""
        if not self.update_records:
            return 0.0
        tail = self.update_records[max(0, len(self.update_records) * 4 // 5) :]
        return float(np.mean([r["explained_variance"] for r in tail]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "critic_mode": self.critic_mode,
            "completed_updates": self.completed_updates,
            "total_environment_steps": self.total_environment_steps,
            "total_collection_steps": self.total_collection_steps,
            "learner_decisions": self.learner_decisions,
            "completed_deals": self.completed_deals,
            "passed_out_deals": self.passed_out_deals,
            "wall_clock_seconds": self.wall_clock_seconds,
            "environment_steps_per_second": self.environment_steps_per_second,
            "mean_explained_variance": self.mean_explained_variance,
            "final_explained_variance": self.final_explained_variance,
            "worst_epoch_zero_deviation": self.worst_epoch_zero_deviation,
            "illegal_action_count": self.illegal_action_count,
            "no_trump_action_count": self.no_trump_action_count,
            "non_finite_update_count": self.non_finite_update_count,
            "parameter_counts": self.parameter_counts,
            "update_records": self.update_records,
            "evaluations": self.evaluations,
            "checkpoint_paths": self.checkpoint_paths,
        }


def build_phase10_networks(
    config: Phase10Config,
) -> tuple[RecurrentActorNetwork, RecurrentCriticNetwork]:
    """Build the A2 networks and verify the contract before training starts.

    The actor is built from ``initialization_seed`` in both conditions, so
    control and treatment begin from an identical policy. The critic is fresh in
    both: the Strong heuristic has no value function, so nothing privileged is
    used to initialize it. Only the critic INPUT is privileged, and only during
    training.
    """
    actor, critic = build_actor_and_critic_for(config.critic_mode, seed=config.initialization_seed)
    if config.bc_checkpoint is not None:
        loaded = load_bc_checkpoint(config.bc_checkpoint)
        if loaded.architecture != "gru":
            raise ValueError(
                f"Phase 10 needs a recurrent BC checkpoint, got {loaded.architecture!r}"
            )
        actor.load_state_dict(loaded.payload["actor_state_dict"])
    verify_parameter_counts(actor, critic, config.critic_mode)
    return actor, critic


def build_phase10_collector(
    config: Phase10Config,
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    rules: RulesConfig,
) -> tuple[Any, torch.Generator]:
    """Return the collector for one condition.

    Both classes are the same code except for ``critic_observation``; the G6
    complete-deal boundary comes from their shared Phase 9 ancestor.
    """
    environments = VecTarneeb(
        num_envs=config.environment_count,
        root_seed=config.environment_root_seed,
        rules=rules,
    )
    generator = torch.Generator()
    generator.manual_seed(config.action_sampling_seed)
    collector_class = (
        CentralizedCriticCollector
        if config.critic_mode == CRITIC_MODE_CENTRALIZED
        else DecentralizedCriticCollector
    )
    collector = collector_class(
        environments=environments,
        actor=actor,
        critic=critic,
        rollout_length=config.rollout_length,
        action_sampling_generator=generator,
        # Phase10Config mirrors Phase9Config field for field precisely so the
        # frozen Phase 9 helpers can be reused without editing them. The cast
        # records that this is structural reuse, not an accident.
        opponents=build_opponents(cast("Phase9Config", config)),
        chunk_length=config.chunk_length,
    )
    return collector, generator


def run_phase10_training(
    config: Phase10Config,
    rules: RulesConfig | None = None,
    progress: bool = True,
) -> Phase10Result:
    """Run one pre-registered A2 job and return its records."""
    from tarneeb.models.nets import count_parameters

    resolved_rules = rules if rules is not None else RulesConfig()
    actor, critic = build_phase10_networks(config)
    collector, _generator = build_phase10_collector(config, actor, critic, resolved_rules)
    optimizer = build_recurrent_optimizer(actor, critic, config.ppo)
    shuffle_generator = np.random.default_rng(config.minibatch_shuffle_seed)

    result = Phase10Result(run_name=config.run_name, critic_mode=config.critic_mode)
    result.parameter_counts = {
        "actor": count_parameters(actor),
        "critic": count_parameters(critic),
        "critic_observation_dim": config.critic_observation_dim,
    }

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

        batch = buffer.to_chunks(
            collector.bootstrap_values(), config.ppo.discount_factor, config.ppo.gae_lambda
        )
        if not batch.is_finite():
            raise AssertionError("the recurrent rollout produced a non-finite tensor")

        learning_rate = (
            linear_learning_rate(config.ppo.learning_rate, update_index, config.total_updates)
            if config.ppo.anneal_learning_rate
            else config.ppo.learning_rate
        )
        update_metrics = recurrent_ppo_update(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            batch=batch,
            config=config.ppo,
            shuffle_generator=shuffle_generator,
            learning_rate=learning_rate,
            check_epoch_zero=config.check_epoch_zero,
        )

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

        result.update_records.append(
            {
                "update": result.completed_updates,
                "environment_steps": result.total_environment_steps,
                "collection_steps": collector.last_collection_steps,
                "drain_steps": collector.last_collection_steps - config.rollout_length,
                "learning_rate": learning_rate,
                "policy_loss": update_metrics.policy_loss,
                "value_loss": update_metrics.value_loss,
                "entropy": update_metrics.entropy,
                "approximate_kl_divergence": update_metrics.approximate_kl_divergence,
                "clip_fraction": update_metrics.clip_fraction,
                "explained_variance": update_metrics.explained_variance,
                "gradient_norm": update_metrics.gradient_norm,
                "epoch_zero_deviation": deviation,
                "chunk_count": update_metrics.chunk_count,
                "valid_transition_count": update_metrics.valid_transition_count,
                "max_segment_length": rollout_metrics.max_segment_length,
                "new_hand_boundaries": rollout_metrics.new_hand_boundaries,
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
            }
        )

        if progress and (result.completed_updates % 10 == 0 or result.completed_updates == 1):
            record = result.update_records[-1]
            reward = record["mean_learner_reward"]
            reward_text = "  n/a" if reward is None else f"{reward:+.4f}"
            print(
                f"[{config.run_name}] update {result.completed_updates:>4}/{config.total_updates} "
                f"steps {result.total_environment_steps:>8} "
                f"pl {record['policy_loss']:+.4f} ent {record['entropy']:.3f} "
                f"ev {record['explained_variance']:+.3f} rew {reward_text} "
                f"e0 {deviation:.1e}",
                flush=True,
            )

        scheduled = result.completed_updates % config.evaluation_every_updates == 0
        if scheduled or result.completed_updates == config.total_updates:
            path = config.checkpoint_directory / (
                f"{config.run_name}_update{result.completed_updates}.pt"
            )
            save_recurrent_checkpoint(
                path=path,
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                rules=resolved_rules,
                training_config=config.as_dict(),
                global_environment_steps=result.total_environment_steps,
                completed_updates=result.completed_updates,
                wall_clock_seconds=time.perf_counter() - start_time,
                seeds=config.seeds(),
                chunk_length=config.chunk_length,
            )
            result.checkpoint_paths.append(str(path))
            entry = evaluate_against_baselines(
                actor, cast("Phase9Config", config), result.completed_updates
            )
            result.evaluations.append(entry)
            if progress:
                summary = " ".join(
                    f"{name}:{values['win_rate']:.3f}"
                    for name, values in entry["opponents"].items()
                )
                print(
                    f"[{config.run_name}] eval @{result.completed_updates}: {summary}",
                    flush=True,
                )

    result.wall_clock_seconds = time.perf_counter() - start_time
    return result


def write_phase10_report(config: Phase10Config, result: Phase10Result) -> Path:
    """Write the run full record as JSON and return the path."""
    config.results_directory.mkdir(parents=True, exist_ok=True)
    path = config.results_directory / f"{config.run_name}.json"
    payload = {"config": config.as_dict(), "result": result.as_dict()}
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path
