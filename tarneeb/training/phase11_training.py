"""Phase 11 league-training smoke path: PPO updates, snapshots, and league state.

This module is intentionally a correctness harness for Phase 11 Part 3. It wires
``LeagueRecurrentCollector`` into the frozen recurrent PPO update without
starting the principal A3 experiment or consulting held-out evaluation seeds.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb.buffers.recurrent import DEFAULT_CHUNK_LENGTH
from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import VecTarneeb
from tarneeb.league.cache import HistoricalActorCache
from tarneeb.league.population import LeagueConfig, LeaguePopulation
from tarneeb.league.registry import PolicyRegistry
from tarneeb.league.sampling import LatestOnlyProvider, OpponentProvider, PFSPConfig, PFSPProvider
from tarneeb.models.bc_checkpoint import load_bc_checkpoint
from tarneeb.models.centralized_critic import (
    CRITIC_MODE_CENTRALIZED,
    build_actor_and_critic_for,
    verify_parameter_counts,
)
from tarneeb.models.nets import count_parameters
from tarneeb.models.recurrent import RecurrentActorNetwork, RecurrentCriticNetwork
from tarneeb.models.recurrent_checkpoint import (
    load_recurrent_checkpoint,
    restore_recurrent_optimizer,
    save_recurrent_checkpoint,
)
from tarneeb.training.league_rollout import LeagueRecurrentCollector
from tarneeb.training.phase9_seeds import PPO_DEVELOPMENT_SEEDS, training_environment_seed
from tarneeb.training.phase9_training import Phase9Config, evaluate_against_baselines
from tarneeb.training.ppo import PPOConfig, linear_learning_rate
from tarneeb.training.recurrent_ppo import build_recurrent_optimizer, recurrent_ppo_update

PHASE11_CONDITIONS = ("latest_only", "pfsp")
PHASE11_RESUME_FORMAT = "phase11_g6_resume_v1"
PHASE11_INITIAL_CHAMPION = (
    "phase9_gru_bc_s0",
    Path("checkpoints/phase9_gru_bc_s0_update250.pt"),
    "phase9",
    0,
    250,
)
PHASE11_INITIAL_SNAPSHOTS = (
    ("phase9_gru_bc_s1", Path("checkpoints/phase9_gru_bc_s1_update250.pt"), "phase9", 1, 250),
    ("phase9_gru_bc_s2", Path("checkpoints/phase9_gru_bc_s2_update250.pt"), "phase9", 2, 250),
    (
        "phase10_centralized_s0",
        Path("checkpoints/phase10_centralized_s0_update250.pt"),
        "phase10",
        0,
        250,
    ),
    (
        "phase10_centralized_s1",
        Path("checkpoints/phase10_centralized_s1_update250.pt"),
        "phase10",
        1,
        250,
    ),
    (
        "phase10_centralized_s2",
        Path("checkpoints/phase10_centralized_s2_update250.pt"),
        "phase10",
        2,
        250,
    ),
)


@dataclass(frozen=True, slots=True)
class Phase11SmokeConfig:
    """A tiny Phase 11 training configuration; not a principal-run contract."""

    run_name: str
    condition: str
    environment_count: int = 4
    rollout_length: int = 8
    total_updates: int = 2
    chunk_length: int = DEFAULT_CHUNK_LENGTH
    bc_checkpoint: Path | None = None

    seed_index: int = 0
    initialization_seed: int = 1
    action_sampling_seed: int = 2
    minibatch_shuffle_seed: int = 3
    opponent_seed: int = 4

    checkpoint_directory: Path = Path("checkpoints/phase11_smoke")
    results_directory: Path = Path("results/phase11_smoke")
    check_epoch_zero: bool = True
    write_final_checkpoint: bool = True
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        if self.condition not in PHASE11_CONDITIONS:
            raise ValueError(
                f"condition must be one of {PHASE11_CONDITIONS}, got {self.condition!r}"
            )
        if self.environment_count < 1 or self.rollout_length < 1 or self.total_updates < 1:
            raise ValueError("environment_count, rollout_length and total_updates must be >= 1")

    @property
    def environment_root_seed(self) -> int:
        return training_environment_seed(self.seed_index)

    @property
    def league_state_path(self) -> Path:
        return self.results_directory / f"{self.run_name}_league_state.json"

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
        values["bc_checkpoint"] = None if self.bc_checkpoint is None else str(self.bc_checkpoint)
        values["checkpoint_directory"] = str(self.checkpoint_directory)
        values["results_directory"] = str(self.results_directory)
        values["environment_root_seed"] = self.environment_root_seed
        values["league_state_path"] = str(self.league_state_path)
        return values


@dataclass(frozen=True, slots=True)
class Phase11A3Config:
    """Frozen principal A3 settings; construction does not launch training."""

    condition: str
    seed_index: int
    run_name: str = "phase11_a3"
    environment_count: int = 16
    rollout_length: int = 128
    total_updates: int = 250
    chunk_length: int = DEFAULT_CHUNK_LENGTH
    bc_checkpoint: Path = Path("checkpoints/phase9_bc_gru.pt")
    initialization_seed: int = 1
    action_sampling_seed: int = 2
    minibatch_shuffle_seed: int = 3
    opponent_seed: int = 4
    check_epoch_zero: bool = True
    development_every_updates: int = 50
    development_paired_deals_per_anchor: int = 150
    development_anchors: tuple[str, ...] = ("random", "basic", "strong")
    final_checkpoint_update: int = 250
    checkpoint_selection: str = "final_update_250"
    checkpoint_directory: Path = Path("checkpoints/phase11")
    results_directory: Path = Path("results/phase11")
    write_final_checkpoint: bool = True
    ppo: PPOConfig = field(default_factory=PPOConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)

    def __post_init__(self) -> None:
        if self.condition not in PHASE11_CONDITIONS:
            raise ValueError(f"condition must be one of {PHASE11_CONDITIONS}")
        if self.seed_index not in (0, 1, 2):
            raise ValueError("principal A3 seed_index must be one of 0, 1, 2")
        if (
            self.environment_count != 16
            or self.rollout_length != 128
            or self.total_updates != 250
            or self.chunk_length != DEFAULT_CHUNK_LENGTH
            or self.initialization_seed != 1
            or self.action_sampling_seed != 2
            or self.minibatch_shuffle_seed != 3
            or self.opponent_seed != 4
            or not self.check_epoch_zero
            or self.development_every_updates != 50
            or self.development_paired_deals_per_anchor != 150
            or self.development_anchors != ("random", "basic", "strong")
            or self.final_checkpoint_update != 250
            or self.checkpoint_selection != "final_update_250"
            or self.bc_checkpoint != Path("checkpoints/phase9_bc_gru.pt")
            or not self.write_final_checkpoint
        ):
            raise ValueError(
                "Phase 11 principal settings are frozen; update-250 final checkpoint is mandatory"
            )
        if self.ppo != PPOConfig():
            raise ValueError("Phase 11 must preserve the frozen recurrent PPO configuration")
        if asdict(self.league.pfsp) != asdict(PFSPConfig()):
            raise ValueError("Phase 11 A3 must use the frozen PFSP configuration")
        if (
            self.league.snapshot_environment_steps != 2_000_000
            or self.league.snapshot_updates != 50
            or self.league.older_reservoir_capacity != 10
            or self.league.diversity_minimum_distance is not None
            or self.league.promotion_min_duplicated_deals != 2_000
            or self.league.promotion_confidence_level != 0.95
        ):
            raise ValueError("Phase 11 A3 league, promotion, and diversity settings are frozen")

    @property
    def environment_root_seed(self) -> int:
        return training_environment_seed(self.seed_index)

    @property
    def u0_policy_id(self) -> str:
        return f"phase11_a3_s{self.seed_index}_u0"

    @property
    def u0_checkpoint_path(self) -> Path:
        return self.checkpoint_directory / f"phase11_a3_s{self.seed_index}_u0.pt"

    @property
    def league_state_path(self) -> Path:
        return self.results_directory / f"{self.run_name}_s{self.seed_index}_league_state.json"

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
        for name in ("bc_checkpoint", "checkpoint_directory", "results_directory"):
            values[name] = str(values[name])
        values["environment_root_seed"] = self.environment_root_seed
        values["u0_policy_id"] = self.u0_policy_id
        return values


@dataclass
class Phase11SmokeResult:
    """Diagnostics from a tiny Phase 11 PPO run."""

    run_name: str
    condition: str
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
    snapshot_policy_ids: list[str] = field(default_factory=list)
    worst_epoch_zero_deviation: float = 0.0
    illegal_action_count: int = 0
    no_trump_action_count: int = 0
    non_finite_update_count: int = 0
    parameter_counts: dict[str, int] = field(default_factory=dict)

    @property
    def environment_steps_per_second(self) -> float:
        if self.wall_clock_seconds <= 0.0:
            return 0.0
        return self.total_environment_steps / self.wall_clock_seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "condition": self.condition,
            "completed_updates": self.completed_updates,
            "total_environment_steps": self.total_environment_steps,
            "total_collection_steps": self.total_collection_steps,
            "learner_decisions": self.learner_decisions,
            "completed_deals": self.completed_deals,
            "passed_out_deals": self.passed_out_deals,
            "wall_clock_seconds": self.wall_clock_seconds,
            "environment_steps_per_second": self.environment_steps_per_second,
            "worst_epoch_zero_deviation": self.worst_epoch_zero_deviation,
            "illegal_action_count": self.illegal_action_count,
            "no_trump_action_count": self.no_trump_action_count,
            "non_finite_update_count": self.non_finite_update_count,
            "parameter_counts": self.parameter_counts,
            "checkpoint_paths": self.checkpoint_paths,
            "snapshot_policy_ids": self.snapshot_policy_ids,
            "update_records": self.update_records,
            "evaluations": self.evaluations,
        }


def build_phase11_networks(
    config: Phase11SmokeConfig | Phase11A3Config,
) -> tuple[RecurrentActorNetwork, RecurrentCriticNetwork]:
    """Build the unchanged recurrent actor with the Phase 10 centralized critic."""
    actor, critic = build_actor_and_critic_for(
        CRITIC_MODE_CENTRALIZED, seed=config.initialization_seed
    )
    if config.bc_checkpoint is not None:
        loaded = load_bc_checkpoint(config.bc_checkpoint)
        if loaded.architecture != "gru":
            raise ValueError(
                f"Phase 11 needs a recurrent BC checkpoint, got {loaded.architecture!r}"
            )
        actor.load_state_dict(loaded.payload["actor_state_dict"])
    verify_parameter_counts(actor, critic, CRITIC_MODE_CENTRALIZED)
    return actor, critic


def build_phase11_initial_population(
    config: Phase11A3Config,
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    rules: RulesConfig | None = None,
) -> LeaguePopulation:
    """Build the pre-registered A3 manifest and freeze the exact learner u0 actor."""
    resolved_rules = rules if rules is not None else RulesConfig()
    missing = [
        str(path)
        for _policy_id, path, _phase, _seed, _update in (
            PHASE11_INITIAL_CHAMPION,
            *PHASE11_INITIAL_SNAPSHOTS,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "A3 initial population checkpoint(s) missing: " + ", ".join(missing)
        )

    save_recurrent_checkpoint(
        path=config.u0_checkpoint_path,
        actor=actor,
        critic=critic,
        optimizer=None,
        rules=resolved_rules,
        training_config=config.as_dict(),
        global_environment_steps=0,
        completed_updates=0,
        wall_clock_seconds=0.0,
        seeds={
            "seed_index": config.seed_index,
            "environment_root_seed": config.environment_root_seed,
            "initialization_seed": config.initialization_seed,
            "action_sampling_seed": config.action_sampling_seed,
            "minibatch_shuffle_seed": config.minibatch_shuffle_seed,
            "opponent_seed": config.opponent_seed,
        },
        chunk_length=config.chunk_length,
    )

    registry = PolicyRegistry(rules=resolved_rules)
    population = LeaguePopulation(
        registry=registry,
        config=config.league,
        rng=np.random.default_rng(config.opponent_seed),
    )
    manifest = [
        (PHASE11_INITIAL_CHAMPION, "champion"),
        *((entry, "snapshot") for entry in PHASE11_INITIAL_SNAPSHOTS),
    ]
    for (policy_id, checkpoint_path, source_phase, seed, update), category in manifest:
        registered_id = registry.register_checkpoint(
            policy_id=policy_id,
            checkpoint_path=checkpoint_path,
            source_phase=source_phase,
            category=category,
            seed=seed,
            update=update,
        )
        if registered_id not in population.eligible_policy_ids:
            population.add(registered_id, make_latest=False)

    population.register_snapshot(
        policy_id=config.u0_policy_id,
        checkpoint_path=config.u0_checkpoint_path,
        source_phase="phase11_a3_initialization",
        seed=config.seed_index,
        update=0,
        make_latest=True,
    )
    registry.register_anchor("strong", "strong")
    registry.register_anchor("random", "random")
    population.add("strong", make_latest=False)
    population.add("random", make_latest=False)
    population.note_snapshot(environment_steps=0, updates=0)
    return population


def build_phase11_opponent_provider(
    condition: str,
    population: LeaguePopulation,
) -> OpponentProvider:
    """Build one A3 opponent provider from the current league state."""
    if condition == "latest_only":
        if population.latest_policy_id is None:
            raise ValueError("latest-only training requires population.latest_policy_id")
        return LatestOnlyProvider(population.latest_policy_id)
    if condition != "pfsp":
        raise ValueError(f"unknown Phase 11 condition {condition!r}")
    if population.config.pfsp.ema_beta is None:
        raise ValueError("PFSP smoke requires an explicit ema_beta")
    return PFSPProvider(
        policies_by_category=population.sample_categories(),
        performance=population.performance,
        rng=population.rng,
        config=population.config.pfsp,
    )


def build_phase11_collector(
    config: Phase11SmokeConfig | Phase11A3Config,
    population: LeaguePopulation,
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    rules: RulesConfig,
) -> tuple[LeagueRecurrentCollector, torch.Generator, HistoricalActorCache]:
    """Build the single rollout path shared by latest-only and PFSP."""
    environments = VecTarneeb(
        num_envs=config.environment_count,
        root_seed=config.environment_root_seed,
        rules=rules,
    )
    action_sampling_generator = torch.Generator()
    action_sampling_generator.manual_seed(config.action_sampling_seed)
    actor_cache = HistoricalActorCache(population.registry, rules=rules)
    collector = LeagueRecurrentCollector(
        environments=environments,
        actor=actor,
        critic=critic,
        rollout_length=config.rollout_length,
        action_sampling_generator=action_sampling_generator,
        opponent_provider=build_phase11_opponent_provider(config.condition, population),
        actor_cache=actor_cache,
        chunk_length=config.chunk_length,
        opponent_seed=config.opponent_seed,
    )
    return collector, action_sampling_generator, actor_cache


def update_pfsp_performance_from_collection(
    population: LeaguePopulation,
    completed_deal_outcomes: tuple[tuple[str, float], ...] | list[tuple[str, float]],
) -> None:
    """Update PFSP EMAs from completed deals, one observed result at a time."""
    ema_beta = population.config.pfsp.ema_beta
    if ema_beta is None:
        raise ValueError("PFSP performance updates require an explicit ema_beta")
    for policy_id, learner_win in completed_deal_outcomes:
        if policy_id not in population.registry.records:
            raise KeyError(f"completed deal referenced unregistered opponent {policy_id!r}")
        population.update_performance(
            policy_id,
            learner_win_rate_against_opponent=float(learner_win),
            ema_beta=ema_beta,
        )


def _evaluate_a3_development(
    config: Phase11A3Config,
    actor: RecurrentActorNetwork,
    completed_updates: int,
) -> dict[str, Any]:
    evaluation_seed = PPO_DEVELOPMENT_SEEDS.start + config.seed_index * 150
    if evaluation_seed not in PPO_DEVELOPMENT_SEEDS or evaluation_seed == 20250910:
        raise AssertionError("A3 development evaluation seed escaped its frozen development split")
    phase9_config = Phase9Config(
        run_name=f"{config.run_name}_s{config.seed_index}",
        architecture="gru",
        environment_count=config.environment_count,
        rollout_length=config.rollout_length,
        total_updates=config.total_updates,
        chunk_length=config.chunk_length,
        training_opponent="basic",
        bc_checkpoint=config.bc_checkpoint,
        seed_index=config.seed_index,
        initialization_seed=config.initialization_seed,
        action_sampling_seed=config.action_sampling_seed,
        minibatch_shuffle_seed=config.minibatch_shuffle_seed,
        opponent_seed=config.opponent_seed,
        evaluation_every_updates=config.development_every_updates,
        evaluation_paired_deals=config.development_paired_deals_per_anchor,
        evaluation_opponents=config.development_anchors,
        checkpoint_directory=config.checkpoint_directory,
        results_directory=config.results_directory,
        check_epoch_zero=config.check_epoch_zero,
        ppo=config.ppo,
    )
    return evaluate_against_baselines(
        actor,
        phase9_config,
        completed_updates,
        paired_deals=config.development_paired_deals_per_anchor,
        evaluation_seed=evaluation_seed,
    )


def _save_learner_checkpoint(
    config: Phase11SmokeConfig | Phase11A3Config,
    result: Phase11SmokeResult,
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    optimizer: torch.optim.Optimizer,
    rules: RulesConfig,
    elapsed: float,
    *,
    suffix: str,
    resume_state: dict[str, Any] | None = None,
) -> Path:
    config.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_directory / f"{config.run_name}_{suffix}.pt"
    save_recurrent_checkpoint(
        path=path,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        rules=rules,
        training_config=config.as_dict(),
        global_environment_steps=result.total_environment_steps,
        completed_updates=result.completed_updates,
        wall_clock_seconds=elapsed,
        seeds=config.seeds(),
        chunk_length=config.chunk_length,
        resume_state=resume_state,
    )
    result.checkpoint_paths.append(str(path))
    return path


def _register_due_snapshot(
    config: Phase11SmokeConfig | Phase11A3Config,
    result: Phase11SmokeResult,
    population: LeaguePopulation,
    collector: LeagueRecurrentCollector,
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    optimizer: torch.optim.Optimizer,
    rules: RulesConfig,
    elapsed: float,
) -> str | None:
    if not population.should_snapshot(result.total_environment_steps, result.completed_updates):
        return None
    path = _save_learner_checkpoint(
        config,
        result,
        actor,
        critic,
        optimizer,
        rules,
        elapsed,
        suffix=f"snapshot_u{result.completed_updates}",
    )
    requested_policy_id = f"{config.run_name}_u{result.completed_updates}"
    registered_policy_id = population.register_snapshot(
        policy_id=requested_policy_id,
        checkpoint_path=path,
        source_phase="phase11_smoke",
        seed=config.seed_index,
        update=result.completed_updates,
    )
    population.note_snapshot(result.total_environment_steps, result.completed_updates)
    population.save(config.league_state_path)
    collector.refresh_registered_policies()
    result.snapshot_policy_ids.append(registered_policy_id)
    return registered_policy_id


def _final_checkpoint_suffix(
    config: Phase11SmokeConfig | Phase11A3Config, completed_updates: int
) -> str | None:
    if not config.write_final_checkpoint or completed_updates != config.total_updates:
        return None
    if isinstance(config, Phase11A3Config) and completed_updates != config.final_checkpoint_update:
        raise AssertionError("principal A3 final checkpoint must be selected at update 250")
    return f"final_u{completed_updates}"


def _validate_a3_population(config: Phase11A3Config, population: LeaguePopulation) -> None:
    if population.config != config.league:
        raise ValueError("principal A3 population LeagueConfig must match the validated A3 config")


def _save_final_checkpoint_if_due(
    config: Phase11SmokeConfig | Phase11A3Config,
    result: Phase11SmokeResult,
    actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork,
    optimizer: torch.optim.Optimizer,
    rules: RulesConfig,
) -> Path | None:
    suffix = _final_checkpoint_suffix(config, result.completed_updates)
    if suffix is None:
        return None
    return _save_learner_checkpoint(
        config,
        result,
        actor,
        critic,
        optimizer,
        rules,
        result.wall_clock_seconds,
        suffix=suffix,
    )


def _action_history_fingerprint(collector: LeagueRecurrentCollector) -> str:
    digest = hashlib.sha256()
    for environment_index, seat, policy_id, action, legal_mask in collector.action_history:
        digest.update(f"{environment_index}:{seat}:{policy_id}:{action}:".encode())
        digest.update(np.asarray(legal_mask, dtype=np.uint8).tobytes())
    return f"sha256:{digest.hexdigest()}"


def _capture_phase11_resume_state(
    config: Phase11SmokeConfig | Phase11A3Config,
    result: Phase11SmokeResult,
    population: LeaguePopulation,
    collector: LeagueRecurrentCollector,
    action_generator: torch.Generator,
    shuffle_generator: np.random.Generator,
) -> dict[str, Any]:
    if not collector.recurrent_state_is_zero():
        raise AssertionError("Phase 11 resume checkpoints require a G6 zero-state boundary")
    environments = collector.environments.environments
    if any(environment.state is None or environment.state.events for environment in environments):
        raise AssertionError("Phase 11 resume checkpoints require fresh post-deal environments")

    opponent_generators: dict[str, list[dict[str, Any] | None]] = {}
    for policy_id, agents in collector.opponent_agents_by_policy.items():
        opponent_generators[policy_id] = [
            copy.deepcopy(agent.rng.bit_generator.state) if hasattr(agent, "rng") else None
            for agent in agents
        ]

    return {
        "format_version": PHASE11_RESUME_FORMAT,
        "config": config.as_dict(),
        "completed_updates": result.completed_updates,
        "total_environment_steps": result.total_environment_steps,
        "action_sampling_generator_state": action_generator.get_state().clone(),
        "minibatch_shuffle_generator_state": copy.deepcopy(shuffle_generator.bit_generator.state),
        "environment_states": [copy.deepcopy(environment.state) for environment in environments],
        "environment_deal_indices": [environment.deal_index for environment in environments],
        "environment_match_seeds": [environment.match_seed for environment in environments],
        "environment_episode_counts": [environment.episode_count for environment in environments],
        "collector_state": collector.snapshot(),
        "opponent_generator_states": opponent_generators,
        "population_state": population.state_dict(),
        "snapshot_cadence": {
            "last_snapshot_environment_steps": population.last_snapshot_environment_steps,
            "last_snapshot_update": population.last_snapshot_update,
            "snapshot_count": population.snapshot_count,
        },
        "result_state": {
            "total_collection_steps": result.total_collection_steps,
            "learner_decisions": result.learner_decisions,
            "completed_deals": result.completed_deals,
            "passed_out_deals": result.passed_out_deals,
            "update_records": copy.deepcopy(result.update_records),
            "evaluations": copy.deepcopy(result.evaluations),
            "snapshot_policy_ids": list(result.snapshot_policy_ids),
            "worst_epoch_zero_deviation": result.worst_epoch_zero_deviation,
            "illegal_action_count": result.illegal_action_count,
            "no_trump_action_count": result.no_trump_action_count,
            "non_finite_update_count": result.non_finite_update_count,
            "parameter_counts": dict(result.parameter_counts),
            "checkpoint_paths": list(result.checkpoint_paths),
        },
    }


def _restore_phase11_resume_state(
    resume_state: dict[str, Any],
    config: Phase11SmokeConfig | Phase11A3Config,
    population: LeaguePopulation,
    collector: LeagueRecurrentCollector,
    action_generator: torch.Generator,
    shuffle_generator: np.random.Generator,
    result: Phase11SmokeResult,
) -> None:
    required = {
        "format_version",
        "config",
        "completed_updates",
        "total_environment_steps",
        "action_sampling_generator_state",
        "minibatch_shuffle_generator_state",
        "environment_states",
        "environment_deal_indices",
        "environment_match_seeds",
        "environment_episode_counts",
        "collector_state",
        "opponent_generator_states",
        "population_state",
        "snapshot_cadence",
        "result_state",
    }
    if resume_state.get("format_version") != PHASE11_RESUME_FORMAT or not required.issubset(
        resume_state
    ):
        raise ValueError("incomplete or incompatible Phase 11 resume payload")
    if resume_state["config"] != config.as_dict():
        raise ValueError("Phase 11 resume config does not match the requested run")

    environments = collector.environments.environments
    environment_fields = (
        "environment_states",
        "environment_deal_indices",
        "environment_match_seeds",
        "environment_episode_counts",
    )
    if any(len(resume_state[field]) != len(environments) for field in environment_fields):
        raise ValueError("Phase 11 resume environment count does not match the configured run")

    action_generator.set_state(resume_state["action_sampling_generator_state"])
    shuffle_generator.bit_generator.state = copy.deepcopy(
        resume_state["minibatch_shuffle_generator_state"]
    )
    for environment, state, deal_index, match_seed, episode_count in zip(
        environments,
        resume_state["environment_states"],
        resume_state["environment_deal_indices"],
        resume_state["environment_match_seeds"],
        resume_state["environment_episode_counts"],
        strict=True,
    ):
        environment.state = copy.deepcopy(state)
        environment.deal_index = int(deal_index)
        environment.match_seed = int(match_seed)
        environment.episode_count = int(episode_count)
    collector.restore(resume_state["collector_state"])

    stored_opponents = resume_state["opponent_generator_states"]
    if set(stored_opponents) != set(collector.opponent_agents_by_policy):
        raise ValueError("Phase 11 resume opponent-agent set is incompatible")
    for policy_id, agents in collector.opponent_agents_by_policy.items():
        states = stored_opponents[policy_id]
        if len(states) != len(agents):
            raise ValueError(f"Phase 11 resume random-agent count differs for {policy_id!r}")
        for agent, state in zip(agents, states, strict=True):
            has_rng = hasattr(agent, "rng")
            if has_rng != (state is not None):
                raise ValueError(f"Phase 11 resume RNG shape differs for opponent {policy_id!r}")
            if state is not None:
                generator = getattr(agent, "rng", None)
                if generator is None:
                    raise ValueError(f"opponent {policy_id!r} lost its saved RNG")
                generator.bit_generator.state = copy.deepcopy(state)

    if not collector.recurrent_state_is_zero():
        raise ValueError("Phase 11 resume recurrent state is not zero at the G6 boundary")
    if any(environment.state is None or environment.state.events for environment in environments):
        raise ValueError("Phase 11 resume environment is not at a fresh G6 deal boundary")

    result.completed_updates = int(resume_state["completed_updates"])
    result.total_environment_steps = int(resume_state["total_environment_steps"])
    for name, value in resume_state["result_state"].items():
        setattr(result, name, copy.deepcopy(value))


def run_phase11_smoke(
    config: Phase11SmokeConfig | Phase11A3Config,
    population: LeaguePopulation,
    rules: RulesConfig | None = None,
    progress: bool = True,
    resume_from: Path | None = None,
    stop_after_updates: int | None = None,
) -> Phase11SmokeResult:
    """Run Phase 11 updates; principal use is explicitly opt-in via a frozen config."""
    if isinstance(config, Phase11A3Config):
        _validate_a3_population(config, population)
    resolved_rules = rules if rules is not None else RulesConfig()
    loaded_resume = None
    resume_state = None
    if resume_from is not None:
        loaded_resume = load_recurrent_checkpoint(
            resume_from, rules=resolved_rules, load_critic=True
        )
        if loaded_resume.critic is None:
            raise ValueError("Phase 11 checkpoint cannot be resumed without the centralized critic")
        if loaded_resume.payload.get("optimizer_state_dict") is None:
            raise ValueError("Phase 11 checkpoint cannot be resumed without optimizer state")
        resume_state = loaded_resume.payload.get("resume")
        if not isinstance(resume_state, dict):
            raise ValueError("Phase 11 checkpoint cannot be resumed without a resume payload")
        if resume_state.get("format_version") != PHASE11_RESUME_FORMAT:
            raise ValueError("incompatible Phase 11 resume payload format")
        if resume_state.get("config") != config.as_dict():
            raise ValueError("Phase 11 resume config does not match the requested run")
        population.restore_state_dict(resume_state["population_state"])

    actor, critic = build_phase11_networks(config)
    collector, action_generator, _actor_cache = build_phase11_collector(
        config, population, actor, critic, resolved_rules
    )
    optimizer = build_recurrent_optimizer(actor, critic, config.ppo)
    shuffle_generator = np.random.default_rng(config.minibatch_shuffle_seed)

    result = Phase11SmokeResult(run_name=config.run_name, condition=config.condition)
    result.completed_updates = 0
    result.parameter_counts = {
        "actor": count_parameters(actor),
        "critic": count_parameters(critic),
        "critic_observation_dim": critic.spec.critic_observation_dim,
    }

    start_time = time.perf_counter()
    if resume_from is not None:
        assert loaded_resume is not None and isinstance(resume_state, dict)
        collector.actor.load_state_dict(loaded_resume.payload["actor_state_dict"])
        collector.critic.load_state_dict(loaded_resume.payload["critic_state_dict"])
        restore_recurrent_optimizer(optimizer, loaded_resume)
        _restore_phase11_resume_state(
            resume_state,
            config,
            population,
            collector,
            action_generator,
            shuffle_generator,
            result,
        )
    if result.completed_updates > config.total_updates:
        raise ValueError("resume checkpoint completed_updates exceeds configured total_updates")
    if stop_after_updates is not None and stop_after_updates < 1:
        raise ValueError("stop_after_updates must be positive")
    invocation_stop = min(
        config.total_updates,
        result.completed_updates + stop_after_updates
        if stop_after_updates is not None
        else config.total_updates,
    )

    for update_index in range(result.completed_updates, invocation_stop):
        collector.opponent_provider = build_phase11_opponent_provider(config.condition, population)
        buffer, rollout_metrics = collector.collect()
        if not collector.no_unfinished_learner_sequence(buffer):
            raise AssertionError("G6 violation: learner sequence survived the collection boundary")
        if not collector.recurrent_state_is_zero():
            raise AssertionError("G6 violation: recurrent state survived the collection boundary")

        result.total_environment_steps += rollout_metrics.environment_steps
        result.total_collection_steps += collector.last_collection_steps
        result.learner_decisions += rollout_metrics.learner_decision_count
        result.completed_deals += rollout_metrics.completed_deals
        result.passed_out_deals += rollout_metrics.passed_out_deals
        result.illegal_action_count += rollout_metrics.illegal_action_count
        result.no_trump_action_count += rollout_metrics.no_trump_action_count

        if config.condition == "pfsp":
            update_pfsp_performance_from_collection(population, collector.completed_deal_outcomes)

        batch = buffer.to_chunks(
            collector.bootstrap_values(), config.ppo.discount_factor, config.ppo.gae_lambda
        )
        if not batch.is_finite():
            raise AssertionError("the Phase 11 recurrent rollout produced a non-finite tensor")

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
        update_values = np.asarray(
            [
                update_metrics.policy_loss,
                update_metrics.value_loss,
                update_metrics.entropy,
                update_metrics.gradient_norm,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(update_values).all():
            result.non_finite_update_count += 1
            raise AssertionError("Phase 11 PPO update produced a non-finite diagnostic")

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
                "completed_deals": rollout_metrics.completed_deals,
                "passed_out_deals": rollout_metrics.passed_out_deals,
                "mean_learner_reward": rollout_metrics.mean_learner_reward,
                "auction_decisions": rollout_metrics.auction_decision_count,
                "pass_rate": (
                    rollout_metrics.pass_action_count / rollout_metrics.auction_decision_count
                    if rollout_metrics.auction_decision_count
                    else None
                ),
                "collection_seconds": rollout_metrics.collection_seconds,
                "opponent_policy_ids": tuple(
                    collector.last_collection_statistics["unique_opponent_policy_ids"]
                ),
                "action_trace_fingerprint": _action_history_fingerprint(collector),
                "completed_deal_outcomes": tuple(collector.completed_deal_outcomes),
                "cache_load_count": collector.cache_load_count,
                "cache_hit_count": collector.cache_hit_count,
                "g6_boundary_valid": True,
            }
        )

        snapshot_policy_id = _register_due_snapshot(
            config,
            result,
            population,
            collector,
            actor,
            critic,
            optimizer,
            resolved_rules,
            time.perf_counter() - start_time,
        )
        if snapshot_policy_id is None:
            population.save(config.league_state_path)

        if isinstance(config, Phase11A3Config) and (
            result.completed_updates % config.development_every_updates == 0
        ):
            result.evaluations.append(
                _evaluate_a3_development(config, actor, result.completed_updates)
            )

        if not collector.recurrent_state_is_zero():
            raise AssertionError("G6 violation: recurrent state survived the optimizer boundary")
        resume_state = _capture_phase11_resume_state(
            config,
            result,
            population,
            collector,
            action_generator,
            shuffle_generator,
        )
        _save_learner_checkpoint(
            config,
            result,
            actor,
            critic,
            optimizer,
            resolved_rules,
            time.perf_counter() - start_time,
            suffix=f"resume_u{result.completed_updates}",
            resume_state=resume_state,
        )

        if progress:
            print(
                f"[{config.run_name}] update {result.completed_updates}/{config.total_updates} "
                f"deals={rollout_metrics.completed_deals} "
                f"e0={deviation:.2e} opponents="
                f"{collector.last_collection_statistics['unique_opponent_policy_ids']}",
                flush=True,
            )

    result.wall_clock_seconds = time.perf_counter() - start_time
    _save_final_checkpoint_if_due(config, result, actor, critic, optimizer, resolved_rules)
    population.save(config.league_state_path)
    return result


def write_phase11_smoke_report(
    config: Phase11SmokeConfig,
    result: Phase11SmokeResult,
    population: LeaguePopulation,
) -> Path:
    """Write the smoke record; no held-out evaluation is included."""
    config.results_directory.mkdir(parents=True, exist_ok=True)
    path = config.results_directory / f"{config.run_name}.json"
    payload = {
        "config": config.as_dict(),
        "result": result.as_dict(),
        "league_state": population.state_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path
