"""Benchmark the frozen recurrent rollout and full PPO update paths.

The default full-update mode does not write checkpoints or run evaluation. The
rollout-only mode preserves the earlier PFSP scaling diagnosis. Neither mode
uses held-out evaluation seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.buffers.recurrent import DEFAULT_CHUNK_LENGTH
from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import VecTarneeb
from tarneeb.league.cache import HistoricalActorCache
from tarneeb.league.registry import PolicyRegistry
from tarneeb.league.sampling import (
    LatestOnlyProvider,
    PFSPConfig,
    PFSPPerformance,
    PFSPProvider,
)
from tarneeb.models.bc_checkpoint import load_bc_checkpoint
from tarneeb.models.centralized_critic import (
    CRITIC_MODE_CENTRALIZED,
    build_actor_and_critic_for,
    verify_parameter_counts,
)
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.recurrent import RecurrentActorNetwork
from tarneeb.training.centralized_rollout import CentralizedCriticCollector
from tarneeb.training.league_rollout import LeagueRecurrentCollector
from tarneeb.training.phase10_training import (
    Phase10Config,
    build_phase10_collector,
    build_phase10_networks,
)
from tarneeb.training.ppo import linear_learning_rate
from tarneeb.training.recurrent_ppo import build_recurrent_optimizer, recurrent_ppo_update

ENVIRONMENT_COUNT = 8
BASE_VECTOR_STEPS = 200
TIMED_REPEATS = 3
PFSP_POPULATION_SIZES = (1, 2, 4)
TORCH_CPU_THREADS = 8
FULL_ENVIRONMENT_COUNT = 16
FULL_BASE_VECTOR_STEPS = 128
HISTORICAL_PHASE10_SECONDS = 16.49
CURRENT_PHASE10_SECONDS = 5.927
CURRENT_PHASE10_GATE_SECONDS = CURRENT_PHASE10_SECONDS * 1.3
PHASE10_EXPECTED_PPO_CONFIG = {
    "learning_rate": 3e-4,
    "anneal_learning_rate": True,
    "discount_factor": 1.0,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_function_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "max_gradient_norm": 0.5,
    "update_epochs": 4,
    "minibatch_count": 4,
    "target_kl_divergence": 0.02,
    "adam_epsilon": 1e-5,
    "normalize_advantages": True,
}
DEVELOPMENT_ROOT_SEED = 6_765_432
ACTION_SAMPLING_SEED = 6_765_433
LEARNER_INITIALIZATION_SEED = 6_765_434
MINIBATCH_SHUFFLE_SEED = 6_765_436
OPPONENT_SEED = 6_765_435
LEARNER_POLICY_ID = "learner"
LEARNER_INITIALIZATION_CHECKPOINT = Path("checkpoints/phase9_bc_gru.pt")
OPPONENT_CHECKPOINTS = (
    ("phase9_gru_bc_s0", Path("checkpoints/phase9_gru_bc_s0_update250.pt"), "phase9", 0, 250),
    ("phase9_gru_bc_s1", Path("checkpoints/phase9_gru_bc_s1_update250.pt"), "phase9", 1, 250),
    ("phase9_gru_bc_s2", Path("checkpoints/phase9_gru_bc_s2_update250.pt"), "phase9", 2, 250),
    (
        "phase10_centralized_s0",
        Path("checkpoints/phase10_centralized_s0_update250.pt"),
        "phase10",
        0,
        250,
    ),
)


@dataclass(frozen=True, slots=True)
class TimedRun:
    """Metrics from one warm-up or timed complete-deal collection."""

    decision_count: int
    elapsed_seconds: float
    decisions_per_second: float
    learner_decisions: int
    opponent_neural_decisions: int
    neural_policy_group_calls: int
    unique_neural_policy_ids: tuple[str, ...]
    unique_opponent_policy_ids: tuple[str, ...]
    actor_cache_loads: int
    actor_cache_hits: int
    illegal_actions: int
    action64_actions: int
    non_finite_count: int
    completed_deals: int
    passed_out_deals: int
    base_vector_steps: int
    actual_vector_steps: int
    g6_boundary_valid: bool
    no_unfinished_learner_sequence: bool
    recurrent_state_safe_zero: bool


@dataclass(frozen=True, slots=True)
class ConditionResult:
    """Warm-up and repeated measurements for one collection condition."""

    condition: str
    warmup: TimedRun
    timed_runs: tuple[TimedRun, ...]
    actor_cache_loads_including_warmup: int
    actor_cache_hits_including_warmup: int

    @property
    def best_decisions_per_second(self) -> float:
        return max(run.decisions_per_second for run in self.timed_runs)

    @property
    def median_decisions_per_second(self) -> float:
        return statistics.median(run.decisions_per_second for run in self.timed_runs)


@dataclass(frozen=True, slots=True)
class TimedPhase11Run(TimedRun):
    """One Phase 11 sample with benchmark-only section timings."""

    timings: dict[str, float]
    average_neural_policies_per_vector_step: float


@dataclass(frozen=True, slots=True)
class FullUpdateRun:
    """Collection, chunk construction, PPO, and safety metrics for one run."""

    collect_seconds: float
    chunk_seconds: float
    ppo_seconds: float
    total_seconds: float
    environment_steps: int
    learner_decisions: int
    completed_deals: int
    passed_out_deals: int
    base_vector_steps: int
    actual_vector_steps: int
    drain_steps: int
    chunk_count: int
    chunk_length: int
    valid_transition_count: int
    minibatch_chunk_counts: tuple[int, ...]
    minibatch_valid_timestep_counts: tuple[int, ...]
    ppo_config: dict[str, Any]
    effective_learning_rate: float
    optimizer_steps: int
    early_stopped: bool
    approximate_kl: float
    actor_parameters: int
    critic_parameters: int
    actor_observation_dim: int
    critic_observation_dim: int
    torch_threads: int
    epoch_zero_deviation: float
    policy_loss: float
    value_loss: float
    entropy: float
    explained_variance: float
    gradient_norm: float
    illegal_actions: int
    action64_actions: int
    finite_checks_passed: bool
    g6_boundary_valid: bool
    unfinished_learner_sequence: bool
    recurrent_state_zero: bool
    historical_opponent_decisions: int
    policy_group_calls: int
    average_neural_policies_per_vector_step: float
    unique_opponents: tuple[str, ...]
    cache_loads: int
    cache_hits: int


class BenchmarkActorCache(HistoricalActorCache):
    """HistoricalActorCache with lookup timing local to this benchmark script."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lookup_seconds = 0.0

    def get(self, policy_id: str) -> RecurrentActorNetwork:
        started = time.perf_counter()
        try:
            return super().get(policy_id)
        finally:
            self.lookup_seconds += time.perf_counter() - started


class TimedOpponentProvider:
    """Time provider selection without changing its sampling/RNG behavior."""

    def __init__(self, provider: LatestOnlyProvider | PFSPProvider) -> None:
        self.provider = provider
        self.sampling_seconds = 0.0

    def sample(self) -> str:
        started = time.perf_counter()
        try:
            return self.provider.sample()
        finally:
            self.sampling_seconds += time.perf_counter() - started

    def probabilities(self) -> dict[str, float]:
        return self.provider.probabilities()


class TimedLeagueCollector(LeagueRecurrentCollector):
    """Time major rollout sections through benchmark-only overrides."""

    def __init__(
        self,
        *args: Any,
        benchmark_base_vector_steps: int = BASE_VECTOR_STEPS,
        **kwargs: Any,
    ) -> None:
        self.benchmark_base_vector_steps = benchmark_base_vector_steps
        super().__init__(*args, **kwargs)
        self.benchmark_timings = {
            "learner_forward": 0.0,
            "historical_opponent_forward": 0.0,
            "event_advancement": 0.0,
            "action_sampling": 0.0,
            "environment_step": 0.0,
            "g6_drain": 0.0,
        }
        self.benchmark_vector_step_index = 0
        self.benchmark_environment_step = self.environments.step

        def timed_environment_step(action_ids: np.ndarray) -> Any:
            started = time.perf_counter()
            try:
                return self.benchmark_environment_step(action_ids)
            finally:
                self.benchmark_timings["environment_step"] += time.perf_counter() - started

        self.environments.step = timed_environment_step

    def _one_step(self, buffer: Any, metrics: Any) -> None:
        started = time.perf_counter()
        is_drain = self.benchmark_vector_step_index >= self.benchmark_base_vector_steps
        try:
            super()._one_step(buffer, metrics)
        finally:
            elapsed = time.perf_counter() - started
            if is_drain:
                self.benchmark_timings["g6_drain"] += elapsed
            self.benchmark_vector_step_index += 1

    def _actor_decision_outputs(self, policy_id: str, *args: Any, **kwargs: Any) -> Any:
        section = (
            "learner_forward" if policy_id == self.policy_id else "historical_opponent_forward"
        )
        started = time.perf_counter()
        try:
            return super()._actor_decision_outputs(policy_id, *args, **kwargs)
        finally:
            self.benchmark_timings[section] += time.perf_counter() - started

    def _learner_critic_values(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return super()._learner_critic_values(*args, **kwargs)
        finally:
            self.benchmark_timings["learner_forward"] += time.perf_counter() - started

    def _advance_policy_states(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return super()._advance_policy_states(*args, **kwargs)
        finally:
            self.benchmark_timings["event_advancement"] += time.perf_counter() - started


def _collect_with_action_sampling_timer(collector: Any) -> tuple[Any, Any, float]:
    """Time the frozen categorical sample operation without editing production code."""
    sampling_seconds = 0.0
    original_sample = MaskedCategorical.sample

    def timed_sample(distribution: MaskedCategorical, generator: torch.Generator | None = None):
        nonlocal sampling_seconds
        started = time.perf_counter()
        try:
            return original_sample(distribution, generator)
        finally:
            sampling_seconds += time.perf_counter() - started

    MaskedCategorical.sample = timed_sample
    try:
        buffer, metrics = collector.collect()
    finally:
        MaskedCategorical.sample = original_sample
    return buffer, metrics, sampling_seconds


class BenchmarkOldCollector(CentralizedCriticCollector):
    """Phase 10 G6-safe collector with legacy per-environment neural agents."""

    def __init__(self, *args: Any, opponent_policy_id: str, **kwargs: Any) -> None:
        self.opponent_policy_id = opponent_policy_id
        self.neural_policy_group_calls = 0
        self.unique_neural_policy_ids: set[str] = {LEARNER_POLICY_ID, opponent_policy_id}
        super().__init__(*args, **kwargs)

    def _one_step(self, buffer, metrics) -> None:
        acting_seats = self.environments.acting_seats()
        active_policy_ids = {
            self.policy_id
            if self.is_learner_seat(environment_index, int(seat))
            else self.opponent_policy_id
            for environment_index, seat in enumerate(acting_seats)
            if int(seat) >= 0
        }
        self.neural_policy_group_calls += len(active_policy_ids)
        super()._one_step(buffer, metrics)


def _register_benchmark_population(
    rules: RulesConfig,
) -> tuple[PolicyRegistry, str, tuple[str, ...]]:
    missing = [
        str(path)
        for _policy_id, path, _phase, _seed, _update in OPPONENT_CHECKPOINTS
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("benchmark checkpoint(s) not found: " + ", ".join(missing))
    if not LEARNER_INITIALIZATION_CHECKPOINT.is_file():
        raise FileNotFoundError(str(LEARNER_INITIALIZATION_CHECKPOINT))

    registry = PolicyRegistry(rules=rules)
    policy_ids: list[str] = []
    latest_policy_id = ""
    for index, (policy_id, checkpoint_path, source_phase, seed, update) in enumerate(
        OPPONENT_CHECKPOINTS
    ):
        registered_policy_id = registry.register_checkpoint(
            policy_id=policy_id,
            checkpoint_path=checkpoint_path,
            source_phase=source_phase,
            category="champion" if index == 0 else "snapshot",
            seed=seed,
            update=update,
        )
        if registered_policy_id not in policy_ids:
            policy_ids.append(registered_policy_id)
        latest_policy_id = registered_policy_id

    if len(policy_ids) < 2:
        raise ValueError("benchmark requires at least two unique historical actor fingerprints")
    return registry, latest_policy_id, tuple(policy_ids)


def _build_learner_networks() -> tuple[torch.nn.Module, torch.nn.Module]:
    actor, critic = build_actor_and_critic_for(
        CRITIC_MODE_CENTRALIZED, seed=LEARNER_INITIALIZATION_SEED
    )
    bc_actor = load_bc_checkpoint(LEARNER_INITIALIZATION_CHECKPOINT)
    if bc_actor.architecture != "gru":
        raise ValueError("benchmark learner initialization must be a GRU BC checkpoint")
    actor.load_state_dict(bc_actor.payload["actor_state_dict"])
    return actor, critic


def _finite_buffer_count(buffer: Any) -> int:
    non_finite = 0
    for sequence in buffer.sequences.values():
        for transition in sequence:
            values = (
                transition.static,
                transition.critic_static,
                transition.previous_event_ids,
                transition.legal_action_mask,
            )
            non_finite += sum(int(not np.isfinite(value).all()) for value in values)
            non_finite += int(not np.isfinite(transition.old_log_probability))
            non_finite += int(not np.isfinite(transition.value))
            non_finite += int(not np.isfinite(transition.reward))
    return non_finite


def _make_old_collector(
    rules: RulesConfig,
    learner_actor: torch.nn.Module,
    learner_critic: torch.nn.Module,
    opponent_actor: RecurrentActorNetwork,
    opponent_policy_id: str,
) -> BenchmarkOldCollector:
    environments = VecTarneeb(
        num_envs=ENVIRONMENT_COUNT,
        root_seed=DEVELOPMENT_ROOT_SEED,
        rules=rules,
    )
    opponents = [
        RecurrentNeuralAgent(
            actor=opponent_actor,
            name="phase9_historical",
            deterministic=False,
            seed=OPPONENT_SEED + environment_index,
            rules=rules,
        )
        for environment_index in range(ENVIRONMENT_COUNT)
    ]
    action_sampling_generator = torch.Generator()
    action_sampling_generator.manual_seed(ACTION_SAMPLING_SEED)
    return BenchmarkOldCollector(
        environments=environments,
        actor=learner_actor,
        critic=learner_critic,
        rollout_length=BASE_VECTOR_STEPS,
        action_sampling_generator=action_sampling_generator,
        opponents=opponents,
        opponent_policy_id=opponent_policy_id,
    )


def _make_phase11_collector(
    rules: RulesConfig,
    learner_actor: torch.nn.Module,
    learner_critic: torch.nn.Module,
    actor_cache: BenchmarkActorCache,
    opponent_provider: TimedOpponentProvider,
    *,
    environment_count: int = ENVIRONMENT_COUNT,
    rollout_length: int = BASE_VECTOR_STEPS,
    root_seed: int | None = None,
    action_seed: int = ACTION_SAMPLING_SEED,
    opponent_seed: int = OPPONENT_SEED,
) -> TimedLeagueCollector:
    environments = VecTarneeb(
        num_envs=environment_count,
        root_seed=DEVELOPMENT_ROOT_SEED if root_seed is None else root_seed,
        rules=rules,
    )
    action_sampling_generator = torch.Generator()
    action_sampling_generator.manual_seed(action_seed)
    return TimedLeagueCollector(
        environments=environments,
        actor=learner_actor,
        critic=learner_critic,
        rollout_length=rollout_length,
        action_sampling_generator=action_sampling_generator,
        opponent_provider=opponent_provider,
        actor_cache=actor_cache,
        opponent_seed=opponent_seed,
        benchmark_base_vector_steps=rollout_length,
    )


def _new_pfsp_provider(policy_ids: tuple[str, ...], seed: int) -> PFSPProvider:
    champion_policy_id, *snapshot_policy_ids = policy_ids
    return PFSPProvider(
        policies_by_category={
            "champion": (champion_policy_id,),
            "recent": tuple(snapshot_policy_ids),
            "older": (),
            "strong": (),
            "random": (),
        },
        performance={
            policy_id: PFSPPerformance(
                learner_win_rate_against_opponent_ema=0.5,
                deals_observed=500,
            )
            for policy_id in policy_ids
        },
        rng=np.random.default_rng(seed),
        config=PFSPConfig(
            ema_beta=0.2,
            unobserved_learner_win_rate=0.5,
            empty_category_strategy="proportional",
        ),
    )


def _record_run(
    collector: Any,
    buffer: Any,
    metrics: Any,
    elapsed_seconds: float,
    condition: str,
) -> TimedRun:
    decision_count = int(metrics.decision_count)
    if condition == "old":
        opponent_neural_decisions = decision_count - int(metrics.learner_decision_count)
        group_calls = int(collector.neural_policy_group_calls)
        unique_neural_policy_ids = tuple(sorted(collector.unique_neural_policy_ids))
        unique_opponent_policy_ids = (collector.opponent_policy_id,)
        cache_loads = 0
        cache_hits = 0
        non_finite_count = _finite_buffer_count(buffer)
        g6_boundary_valid = collector.no_unfinished_learner_sequence(buffer)
        no_unfinished = g6_boundary_valid
        recurrent_safe_zero = collector.recurrent_state_is_zero()
        actual_steps = int(collector.last_collection_steps)
        base_steps = BASE_VECTOR_STEPS
    else:
        stats = collector.last_collection_statistics
        opponent_neural_decisions = int(stats["opponent_neural_decision_count"])
        group_calls = int(stats["neural_policy_group_count"])
        unique_neural_policy_ids = tuple(
            sorted(
                {
                    policy_id
                    for groups in stats["neural_policy_group_sizes"]
                    for policy_id, _request_count in groups
                }
            )
        )
        unique_opponent_policy_ids = tuple(stats["unique_opponent_policy_ids"])
        cache_loads = int(stats["cache_load_count"])
        cache_hits = int(stats["cache_hit_count"])
        non_finite_count = int(stats["non_finite_count"]) + _finite_buffer_count(buffer)
        no_unfinished = bool(stats["no_unfinished_learner_sequence"])
        recurrent_safe_zero = bool(stats["recurrent_state_is_zero"])
        g6_boundary_valid = no_unfinished and recurrent_safe_zero
        actual_steps = int(stats["last_collection_steps"])
        base_steps = BASE_VECTOR_STEPS

    if not np.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise AssertionError("benchmark elapsed time must be finite and positive")
    return TimedRun(
        decision_count=decision_count,
        elapsed_seconds=elapsed_seconds,
        decisions_per_second=decision_count / elapsed_seconds,
        learner_decisions=int(metrics.learner_decision_count),
        opponent_neural_decisions=opponent_neural_decisions,
        neural_policy_group_calls=group_calls,
        unique_neural_policy_ids=unique_neural_policy_ids,
        unique_opponent_policy_ids=unique_opponent_policy_ids,
        actor_cache_loads=cache_loads,
        actor_cache_hits=cache_hits,
        illegal_actions=int(metrics.illegal_action_count),
        action64_actions=int(metrics.no_trump_action_count),
        non_finite_count=non_finite_count,
        completed_deals=int(metrics.completed_deals),
        passed_out_deals=int(metrics.passed_out_deals),
        base_vector_steps=base_steps,
        actual_vector_steps=actual_steps,
        g6_boundary_valid=g6_boundary_valid,
        no_unfinished_learner_sequence=no_unfinished,
        recurrent_state_safe_zero=recurrent_safe_zero,
    )


def _run_collection(collector: Any, condition: str) -> TimedRun:
    start_time = time.perf_counter()
    if isinstance(collector, TimedLeagueCollector):
        cache_lookup_before = collector.actor_cache.lookup_seconds
        buffer, metrics, action_sampling_seconds = _collect_with_action_sampling_timer(collector)
        provider_sampling_seconds = collector.opponent_provider.sampling_seconds
        cache_lookup_seconds = collector.actor_cache.lookup_seconds - cache_lookup_before
        section_timings = dict(collector.benchmark_timings)
        section_timings["provider_sampling"] = provider_sampling_seconds
        section_timings["cache_lookup"] = cache_lookup_seconds
        section_timings["action_sampling"] = action_sampling_seconds
    else:
        buffer, metrics = collector.collect()
        section_timings = {}
    elapsed_seconds = time.perf_counter() - start_time
    run = _record_run(collector, buffer, metrics, elapsed_seconds, condition)
    if isinstance(collector, TimedLeagueCollector):
        stats = collector.last_collection_statistics
        unique_policy_group_calls = int(stats["neural_policy_group_count"])
        average_neural_policies_per_vector_step = unique_policy_group_calls / max(
            int(stats["last_collection_steps"]), 1
        )
        return TimedPhase11Run(
            **asdict(run),
            timings=section_timings,
            average_neural_policies_per_vector_step=average_neural_policies_per_vector_step,
        )
    return run


def _benchmark_old(
    rules: RulesConfig,
    learner_actor: torch.nn.Module,
    learner_critic: torch.nn.Module,
    opponent_actor: RecurrentActorNetwork,
    opponent_policy_id: str,
) -> ConditionResult:
    warmup_collector = _make_old_collector(
        rules,
        copy.deepcopy(learner_actor),
        copy.deepcopy(learner_critic),
        opponent_actor,
        opponent_policy_id,
    )
    warmup = _run_collection(warmup_collector, "old")
    timed_runs = tuple(
        _run_collection(
            _make_old_collector(
                rules,
                copy.deepcopy(learner_actor),
                copy.deepcopy(learner_critic),
                opponent_actor,
                opponent_policy_id,
            ),
            "old",
        )
        for _repeat_index in range(TIMED_REPEATS)
    )
    return ConditionResult("old", warmup, timed_runs, 0, 0)


def _benchmark_phase11(
    condition: str,
    rules: RulesConfig,
    learner_actor: torch.nn.Module,
    learner_critic: torch.nn.Module,
    registry: PolicyRegistry,
    latest_policy_id: str,
    population_policy_ids: tuple[str, ...],
) -> ConditionResult:
    actor_cache = BenchmarkActorCache(registry)
    cache_before = actor_cache.statistics()

    def make_collector(repeat_index: int) -> LeagueRecurrentCollector:
        if condition == "latest_only":
            base_provider: LatestOnlyProvider | PFSPProvider = LatestOnlyProvider(latest_policy_id)
        else:
            base_provider = _new_pfsp_provider(
                population_policy_ids, seed=ACTION_SAMPLING_SEED + repeat_index
            )
        provider = TimedOpponentProvider(base_provider)
        return _make_phase11_collector(
            rules,
            copy.deepcopy(learner_actor),
            copy.deepcopy(learner_critic),
            actor_cache,
            provider,
        )

    warmup = _run_collection(make_collector(-1), condition)
    timed_runs = tuple(
        _run_collection(make_collector(repeat_index), condition)
        for repeat_index in range(TIMED_REPEATS)
    )
    cache_after = actor_cache.statistics()
    return ConditionResult(
        condition,
        warmup,
        timed_runs,
        cache_after["loads"] - cache_before["loads"],
        cache_after["hits"] - cache_before["hits"],
    )


def _median_section_timings(result: ConditionResult) -> dict[str, float]:
    timed_runs = [run for run in result.timed_runs if isinstance(run, TimedPhase11Run)]
    section_names = (
        "learner_forward",
        "historical_opponent_forward",
        "event_advancement",
        "provider_sampling",
        "cache_lookup",
        "action_sampling",
        "environment_step",
        "g6_drain",
    )
    return {
        section: statistics.median(run.timings.get(section, 0.0) for run in timed_runs)
        for section in section_names
    }


def _full_update_config(seed_index: int) -> Phase10Config:
    return replace(
        _phase10_validation_config(seed_index),
        run_name="phase11_full_update_benchmark",
    )


def _phase10_validation_config(seed_index: int) -> Phase10Config:
    config = Phase10Config(
        run_name="phase10_full_update_validation",
        critic_mode=CRITIC_MODE_CENTRALIZED,
        environment_count=FULL_ENVIRONMENT_COUNT,
        rollout_length=FULL_BASE_VECTOR_STEPS,
        total_updates=250,
        chunk_length=DEFAULT_CHUNK_LENGTH,
        training_opponent="basic",
        bc_checkpoint=LEARNER_INITIALIZATION_CHECKPOINT,
        seed_index=seed_index,
    )
    if asdict(config.ppo) != PHASE10_EXPECTED_PPO_CONFIG:
        raise AssertionError("Phase 10 PPOConfig differs from its recorded frozen values")
    return config


def _run_full_update(
    condition: str,
    rules: RulesConfig,
    config: Phase10Config,
    learner_actor: torch.nn.Module,
    learner_critic: torch.nn.Module,
    registry: PolicyRegistry,
    latest_policy_id: str,
    population_policy_ids: tuple[str, ...],
) -> FullUpdateRun:
    actor = copy.deepcopy(learner_actor)
    critic = copy.deepcopy(learner_critic)
    network_contract = verify_parameter_counts(actor, critic, CRITIC_MODE_CENTRALIZED)
    torch_threads = torch.get_num_threads()
    if torch_threads != TORCH_CPU_THREADS:
        raise AssertionError(
            f"benchmark requires {TORCH_CPU_THREADS} Torch CPU threads, found {torch_threads}"
        )

    if condition == "P10 centralized":
        collector, _action_generator = build_phase10_collector(config, actor, critic, rules)
        opponent_decisions = 0
        policy_group_calls = 0
        unique_opponents: tuple[str, ...] = ()
        cache_loads = 0
        cache_hits = 0
    else:
        actor_cache = BenchmarkActorCache(registry)
        if condition == "P11 latest-only":
            base_provider: LatestOnlyProvider | PFSPProvider = LatestOnlyProvider(latest_policy_id)
        else:
            population_size = int(condition.rsplit(" ", maxsplit=1)[1])
            base_provider = _new_pfsp_provider(
                population_policy_ids[:population_size], seed=config.opponent_seed
            )
        collector = _make_phase11_collector(
            rules,
            actor,
            critic,
            actor_cache,
            TimedOpponentProvider(base_provider),
            environment_count=config.environment_count,
            rollout_length=config.rollout_length,
            root_seed=config.environment_root_seed,
            action_seed=config.action_sampling_seed,
            opponent_seed=config.opponent_seed,
        )
        opponent_decisions = -1
        policy_group_calls = -1
        unique_opponents = ()
        cache_loads = -1
        cache_hits = -1

    optimizer = build_recurrent_optimizer(actor, critic, config.ppo)
    shuffle_generator = np.random.default_rng(config.minibatch_shuffle_seed)

    collect_started = time.perf_counter()
    buffer, rollout_metrics, _sampling_seconds = _collect_with_action_sampling_timer(collector)
    collect_seconds = time.perf_counter() - collect_started

    no_unfinished = collector.no_unfinished_learner_sequence(buffer)
    recurrent_state_zero = collector.recurrent_state_is_zero()
    g6_boundary_valid = no_unfinished and recurrent_state_zero

    chunk_started = time.perf_counter()
    batch = buffer.to_chunks(
        collector.bootstrap_values(), config.ppo.discount_factor, config.ppo.gae_lambda
    )
    batch_finite = batch.is_finite()
    chunk_seconds = time.perf_counter() - chunk_started

    minibatch_rng = np.random.default_rng()
    minibatch_rng.bit_generator.state = copy.deepcopy(shuffle_generator.bit_generator.state)
    chunk_permutation = minibatch_rng.permutation(batch.chunk_count)
    chunk_groups = [
        group
        for group in np.array_split(chunk_permutation, config.ppo.minibatch_count)
        if group.size > 0
    ]
    valid_per_chunk = batch.valid.sum(dim=0).cpu().numpy()
    minibatch_chunk_counts = tuple(int(group.size) for group in chunk_groups)
    minibatch_valid_timestep_counts = tuple(
        int(valid_per_chunk[group].sum()) for group in chunk_groups
    )
    effective_learning_rate = (
        linear_learning_rate(config.ppo.learning_rate, 0, config.total_updates)
        if config.ppo.anneal_learning_rate
        else config.ppo.learning_rate
    )

    ppo_started = time.perf_counter()
    update_metrics = recurrent_ppo_update(
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        batch=batch,
        config=config.ppo,
        shuffle_generator=shuffle_generator,
        learning_rate=effective_learning_rate,
        check_epoch_zero=config.check_epoch_zero,
    )
    ppo_seconds = time.perf_counter() - ppo_started
    total_seconds = collect_seconds + chunk_seconds + ppo_seconds
    if update_metrics.learning_rate != effective_learning_rate:
        raise AssertionError("recurrent PPO did not use the effective Phase 10 learning rate")

    metrics_finite = bool(
        np.isfinite(
            [
                update_metrics.policy_loss,
                update_metrics.value_loss,
                update_metrics.entropy,
                update_metrics.explained_variance,
                update_metrics.gradient_norm,
                update_metrics.epoch_zero_log_probability_max_deviation,
            ]
        ).all()
    )
    finite_checks_passed = (
        _finite_buffer_count(buffer) == 0
        and batch_finite
        and metrics_finite
        and np.isfinite(collect_seconds + chunk_seconds + ppo_seconds)
    )

    actual_vector_steps = int(collector.last_collection_steps)
    if condition != "P10 centralized":
        stats = collector.last_collection_statistics
        opponent_decisions = int(stats["opponent_neural_decision_count"])
        policy_group_calls = int(stats["neural_policy_group_count"])
        unique_opponents = tuple(stats["unique_opponent_policy_ids"])
        cache_loads = int(stats["cache_load_count"])
        cache_hits = int(stats["cache_hit_count"])

    result = FullUpdateRun(
        collect_seconds=collect_seconds,
        chunk_seconds=chunk_seconds,
        ppo_seconds=ppo_seconds,
        total_seconds=total_seconds,
        environment_steps=int(rollout_metrics.environment_steps),
        learner_decisions=int(rollout_metrics.learner_decision_count),
        completed_deals=int(rollout_metrics.completed_deals),
        passed_out_deals=int(rollout_metrics.passed_out_deals),
        base_vector_steps=config.rollout_length,
        actual_vector_steps=actual_vector_steps,
        drain_steps=actual_vector_steps - config.rollout_length,
        chunk_count=int(update_metrics.chunk_count),
        chunk_length=int(batch.chunk_length),
        valid_transition_count=int(update_metrics.valid_transition_count),
        minibatch_chunk_counts=minibatch_chunk_counts,
        minibatch_valid_timestep_counts=minibatch_valid_timestep_counts,
        ppo_config=asdict(config.ppo),
        effective_learning_rate=effective_learning_rate,
        optimizer_steps=int(update_metrics.optimizer_steps),
        early_stopped=bool(update_metrics.early_stopped),
        approximate_kl=float(update_metrics.approximate_kl_divergence),
        actor_parameters=network_contract["actor"],
        critic_parameters=network_contract["critic"],
        actor_observation_dim=network_contract["actor_observation_dim"],
        critic_observation_dim=network_contract["critic_observation_dim"],
        torch_threads=torch_threads,
        epoch_zero_deviation=float(update_metrics.epoch_zero_log_probability_max_deviation),
        policy_loss=float(update_metrics.policy_loss),
        value_loss=float(update_metrics.value_loss),
        entropy=float(update_metrics.entropy),
        explained_variance=float(update_metrics.explained_variance),
        gradient_norm=float(update_metrics.gradient_norm),
        illegal_actions=int(rollout_metrics.illegal_action_count),
        action64_actions=int(rollout_metrics.no_trump_action_count),
        finite_checks_passed=finite_checks_passed,
        g6_boundary_valid=g6_boundary_valid,
        unfinished_learner_sequence=not no_unfinished,
        recurrent_state_zero=recurrent_state_zero,
        historical_opponent_decisions=opponent_decisions,
        policy_group_calls=policy_group_calls,
        average_neural_policies_per_vector_step=(
            policy_group_calls / actual_vector_steps if policy_group_calls >= 0 else 0.0
        ),
        unique_opponents=unique_opponents,
        cache_loads=cache_loads,
        cache_hits=cache_hits,
    )

    if (
        result.illegal_actions
        or result.action64_actions
        or not result.finite_checks_passed
        or not result.g6_boundary_valid
        or result.unfinished_learner_sequence
        or not result.recurrent_state_zero
    ):
        raise AssertionError(f"invalid benchmark run for {condition}: {result}")
    return result


def _benchmark_full_update_condition(
    condition: str,
    rules: RulesConfig,
    config: Phase10Config,
    learner_actor: torch.nn.Module,
    learner_critic: torch.nn.Module,
    registry: PolicyRegistry,
    latest_policy_id: str,
    population_policy_ids: tuple[str, ...],
) -> tuple[FullUpdateRun, tuple[FullUpdateRun, ...]]:
    warmup = _run_full_update(
        condition,
        rules,
        config,
        learner_actor,
        learner_critic,
        registry,
        latest_policy_id,
        population_policy_ids,
    )
    timed_runs = tuple(
        _run_full_update(
            condition,
            rules,
            config,
            learner_actor,
            learner_critic,
            registry,
            latest_policy_id,
            population_policy_ids,
        )
        for _repeat_index in range(TIMED_REPEATS)
    )
    return warmup, timed_runs


def _print_p10_validation(
    config: Phase10Config,
    warmup: FullUpdateRun,
    timed_runs: tuple[FullUpdateRun, ...],
) -> None:
    print("Phase 10 validation against run_phase10_training")
    print(
        "Config contract: "
        f"environment_count={config.environment_count}, rollout_length={config.rollout_length}, "
        f"total_updates={config.total_updates}, training_opponent={config.training_opponent}, "
        f"seed_index={config.seed_index}, initialization/action/minibatch/opponent seeds="
        f"{config.initialization_seed}/{config.action_sampling_seed}/"
        f"{config.minibatch_shuffle_seed}/{config.opponent_seed}, "
        f"chunk_length={config.chunk_length}, "
        f"PPOConfig_matches_recorded={asdict(config.ppo) == PHASE10_EXPECTED_PPO_CONFIG}"
    )
    print(
        "Network contract: "
        f"actor_parameters={warmup.actor_parameters} (expected 956617), "
        f"critic_parameters={warmup.critic_parameters} (expected 1024137), "
        f"actor_observation={warmup.actor_observation_dim} (expected 692), "
        f"critic_observation={warmup.critic_observation_dim} (expected 856), "
        f"chunk_length={warmup.chunk_length} (expected {DEFAULT_CHUNK_LENGTH}), "
        f"torch.get_num_threads()={warmup.torch_threads} (expected {TORCH_CPU_THREADS})"
    )
    print(
        f"Runtime: OS={platform.platform()}, Python={platform.python_version()}, "
        f"PyTorch={torch.__version__}"
    )
    print(
        "Path: build_phase10_networks -> build_phase10_collector -> "
        "RecurrentRolloutBuffer.to_chunks -> recurrent_ppo_update"
    )

    runs_to_print = [
        ("warm-up", warmup),
        *((f"timed repeat {index}", run) for index, run in enumerate(timed_runs, start=1)),
    ]
    for label, run in runs_to_print:
        padded_timesteps = run.chunk_length * run.chunk_count
        print(
            f"\n{label}: collect={run.collect_seconds:.6f}s, "
            f"chunks={run.chunk_seconds:.6f}s, PPO-call={run.ppo_seconds:.6f}s, "
            f"total={run.total_seconds:.6f}s"
        )
        print(f"  PPOConfig actually used: {json.dumps(run.ppo_config, sort_keys=True)}")
        print(
            f"  effective_learning_rate={run.effective_learning_rate:.9g}, "
            f"optimizer_steps={run.optimizer_steps}, early_stopped={run.early_stopped}, "
            f"configured_max_optimizer_steps="
            f"{run.ppo_config['update_epochs'] * run.ppo_config['minibatch_count']}, "
            f"approximate_kl={run.approximate_kl:.9g}"
        )
        print(
            f"  chunk_count={run.chunk_count}, chunk_length={run.chunk_length}, "
            f"padded_recurrent_timesteps={padded_timesteps}, "
            f"valid_recurrent_timesteps={run.valid_transition_count}, "
            f"learner_decisions={run.learner_decisions}, "
            f"minibatch_chunks={run.minibatch_chunk_counts}, "
            f"minibatch_valid_timesteps={run.minibatch_valid_timestep_counts}"
        )
        print(
            f"  actor={run.actor_parameters} params/{run.actor_observation_dim} features, "
            f"critic={run.critic_parameters} params/{run.critic_observation_dim} features, "
            f"torch_threads={run.torch_threads}, env_steps={run.environment_steps}, "
            f"vector_steps={run.actual_vector_steps} (drain={run.drain_steps})"
        )
        print(
            f"  G6={run.g6_boundary_valid}, unfinished={run.unfinished_learner_sequence}, "
            f"state_zero={run.recurrent_state_zero}, illegal={run.illegal_actions}, "
            f"action64={run.action64_actions}, finite={run.finite_checks_passed}"
        )

    timed_ppo_median = statistics.median(run.ppo_seconds for run in timed_runs)
    print(
        f"\nP10 timed-repeat medians: collect="
        f"{statistics.median(run.collect_seconds for run in timed_runs):.6f}s, "
        f"PPO={timed_ppo_median:.6f}s, "
        f"total={statistics.median(run.total_seconds for run in timed_runs):.6f}s"
    )
    print(
        "Historical comparison: the P10 experiment record documents 8 Torch threads; "
        "its pre-launch table reports 176 vector steps and 1426 learner decisions. "
        "The current seed-0 P10 run matches the saved first-update record at 172 vector "
        "steps and 1375 learner decisions. The historical JSON has no per-update "
        "optimizer_steps, early_stopped flag, or PPO-only timer, so those timing-work "
        "details cannot be compared directly."
    )


def _median_full_update(runs: tuple[FullUpdateRun, ...]) -> FullUpdateRun:
    median_total = statistics.median(run.total_seconds for run in runs)
    return min(runs, key=lambda run: abs(run.total_seconds - median_total))


def _print_full_update_report(
    results: dict[str, tuple[FullUpdateRun, ...]],
) -> None:
    print("\nTable 1: median timing and PPO work")
    print(
        f"{'Condition':<20} {'Collect':>9} {'Chunks':>9} {'PPO':>9} {'Total':>9} "
        f"{'Steps':>7} {'vs P10':>9}"
    )
    for condition, runs in results.items():
        total_median = statistics.median(run.total_seconds for run in runs)
        print(
            f"{condition:<20} {statistics.median(run.collect_seconds for run in runs):>8.3f}s "
            f"{statistics.median(run.chunk_seconds for run in runs):>8.3f}s "
            f"{statistics.median(run.ppo_seconds for run in runs):>8.3f}s "
            f"{total_median:>8.3f}s "
            f"{statistics.median(run.optimizer_steps for run in runs):>7.0f} "
            f"{total_median / CURRENT_PHASE10_SECONDS:>8.3f}x"
        )

    print("\nTable 2: median realized work")
    print(
        f"{'Condition':<20} {'Env steps':>10} {'Learner decisions':>19} "
        f"{'Valid timesteps':>17} {'Drain':>8}"
    )
    for condition, runs in results.items():
        median_run = _median_full_update(runs)
        print(
            f"{condition:<20} {median_run.environment_steps:>10} "
            f"{median_run.learner_decisions:>19} {median_run.valid_transition_count:>17} "
            f"{median_run.drain_steps:>8}"
        )

    print("\nTable 3: total seconds by timed repeat")
    print(
        f"{'Condition':<20} {'Repeat 1':>10} {'Repeat 2':>10} {'Repeat 3':>10} {'Median':>10} {'Min':>10} {'Max':>10}"
    )
    for condition, runs in results.items():
        times = [run.total_seconds for run in runs]
        print(
            f"{condition:<20} {times[0]:>10.3f} {times[1]:>10.3f} {times[2]:>10.3f} {statistics.median(times):>10.3f} {min(times):>10.3f} {max(times):>10.3f}"
        )

    print("\nTable 4: Phase 11 league work")
    print(
        f"{'Condition':<20} {'Hist opp decisions':>19} {'Group calls':>13} {'Avg policies/step':>19} {'Cache loads/hits':>19}"
    )
    for condition, runs in results.items():
        if condition == "P10 centralized":
            print(f"{condition:<20} {'n/a':>19} {'n/a':>13} {'n/a':>19} {'n/a':>19}")
            continue
        print(
            f"{condition:<20} "
            f"{statistics.median(run.historical_opponent_decisions for run in runs):>19.0f} "
            f"{statistics.median(run.policy_group_calls for run in runs):>13.0f} "
            f"{statistics.median(run.average_neural_policies_per_vector_step for run in runs):>19.3f} "
            f"{statistics.median(run.cache_loads for run in runs):.0f}/"
            f"{statistics.median(run.cache_hits for run in runs):.0f}"
        )

    print("\nPer-repeat PPO work, timing, and safety")
    p10_steps = statistics.median(run.optimizer_steps for run in results["P10 centralized"])
    for condition, runs in results.items():
        print(f"{condition}:")
        for index, run in enumerate(runs, start=1):
            max_steps = run.ppo_config["update_epochs"] * run.ppo_config["minibatch_count"]
            work_relation = (
                "equal"
                if run.optimizer_steps == p10_steps
                else ("greater" if run.optimizer_steps > p10_steps else "less")
            )
            print(
                f"  repeat {index}: collect/chunks/PPO/total="
                f"{run.collect_seconds:.3f}/{run.chunk_seconds:.3f}/"
                f"{run.ppo_seconds:.3f}/{run.total_seconds:.3f}s; "
                f"env_steps={run.environment_steps}, vector_steps={run.actual_vector_steps}, "
                f"drain={run.drain_steps}, deals={run.completed_deals}, "
                f"passed_out={run.passed_out_deals}, learner_decisions={run.learner_decisions}, "
                f"chunks={run.chunk_count}, valid_timesteps={run.valid_transition_count}; "
                f"LR={run.effective_learning_rate:.7g}, optimizer_steps="
                f"{run.optimizer_steps}/{max_steps} ({work_relation} vs P10), "
                f"early_stopped={run.early_stopped}, KL={run.approximate_kl:.7g}; "
                f"policy_loss={run.policy_loss:.6g}, value_loss={run.value_loss:.6g}, "
                f"entropy={run.entropy:.6g}, EV={run.explained_variance:.6g}, "
                f"grad_norm={run.gradient_norm:.6g}; "
                f"league_decisions={run.historical_opponent_decisions}, "
                f"group_calls={run.policy_group_calls}, "
                f"policies/vector_step={run.average_neural_policies_per_vector_step:.3f}, "
                f"opponents={run.unique_opponents}, cache={run.cache_loads}/{run.cache_hits}; "
                f"illegal={run.illegal_actions}, action64={run.action64_actions}, "
                f"finite={run.finite_checks_passed}, G6={run.g6_boundary_valid}, "
                f"unfinished={run.unfinished_learner_sequence}, "
                f"state_zero={run.recurrent_state_zero}"
            )

    print("\nCurrent-machine performance gate")
    print(
        f"P10 reference={CURRENT_PHASE10_SECONDS:.3f}s, "
        f"1.3x gate={CURRENT_PHASE10_GATE_SECONDS:.3f}s; historical P10="
        f"{HISTORICAL_PHASE10_SECONDS:.2f}s (provenance only)"
    )
    for condition in ("P11 latest-only", "P11 PFSP 1", "P11 PFSP 2", "P11 PFSP 4"):
        median_total = statistics.median(run.total_seconds for run in results[condition])
        print(
            f"{condition}: {median_total:.3f}s -> {'PASS' if median_total <= CURRENT_PHASE10_GATE_SECONDS else 'FAIL'}"
        )

    p10_collect = statistics.median(run.collect_seconds for run in results["P10 centralized"])
    p10_ppo = statistics.median(run.ppo_seconds for run in results["P10 centralized"])
    p10_steps = statistics.median(run.optimizer_steps for run in results["P10 centralized"])
    p11_conditions = ("P11 latest-only", "P11 PFSP 1", "P11 PFSP 2", "P11 PFSP 4")
    print("\nCollection overhead and PPO workload attribution vs P10")
    for condition in p11_conditions:
        runs = results[condition]
        collect = statistics.median(run.collect_seconds for run in runs)
        ppo = statistics.median(run.ppo_seconds for run in runs)
        steps = statistics.median(run.optimizer_steps for run in runs)
        relation = "equal" if steps == p10_steps else ("greater" if steps > p10_steps else "less")
        print(
            f"{condition}: collection delta={collect - p10_collect:+.3f}s; "
            f"PPO delta={ppo - p10_ppo:+.3f}s with optimizer work {relation} "
            f"({steps:.0f} vs {p10_steps:.0f}); total delta="
            f"{statistics.median(run.total_seconds for run in runs) - CURRENT_PHASE10_SECONDS:+.3f}s"
        )
    largest_collection = max(
        p11_conditions,
        key=lambda name: statistics.median(run.collect_seconds for run in results[name]),
    )
    pfsp_collect_medians = [
        statistics.median(run.collect_seconds for run in results[name])
        for name in ("P11 PFSP 1", "P11 PFSP 2", "P11 PFSP 4")
    ]
    print(f"Largest collection median: {largest_collection}")
    print(
        f"PFSP collection medians 1/2/4: {pfsp_collect_medians}; monotonic={pfsp_collect_medians[0] <= pfsp_collect_medians[1] <= pfsp_collect_medians[2]}"
    )
    print(
        "Historical 16.49s remains provenance only; the performance gate is based on "
        "the current-machine P10 reference. No optimization was performed."
    )


def _print_pfsp_diagnosis(results: tuple[ConditionResult, ...]) -> None:
    print("\nPFSP population comparison")
    print(
        f"{'PFSP population':<18} {'Median dec/s':>14} {'Group calls':>13} "
        f"{'Policies/vector step':>22} {'Median time':>13}"
    )
    for result in results:
        phase11_runs = [run for run in result.timed_runs if isinstance(run, TimedPhase11Run)]
        median_group_calls = statistics.median(
            run.neural_policy_group_calls for run in phase11_runs
        )
        median_policies_per_step = statistics.median(
            run.average_neural_policies_per_vector_step for run in phase11_runs
        )
        median_elapsed = statistics.median(run.elapsed_seconds for run in result.timed_runs)
        print(
            f"{result.condition:<18} {result.median_decisions_per_second:>14.1f} "
            f"{median_group_calls:>13.0f} {median_policies_per_step:>22.3f} "
            f"{median_elapsed:>12.3f}s"
        )

    print("\nTiming breakdown (median seconds; section percentages of median elapsed)")
    timing_sections = (
        ("learner forward", "learner_forward"),
        ("historical opponent forward", "historical_opponent_forward"),
        ("event advancement", "event_advancement"),
        ("provider sampling", "provider_sampling"),
        ("cache lookup", "cache_lookup"),
        ("action sampling", "action_sampling"),
        ("environment", "environment_step"),
        ("G6 drain wall time (overlapping)", "g6_drain"),
    )
    print(f"{'Section':<38}" + "".join(f"{size:>23}" for size in PFSP_POPULATION_SIZES))
    for label, section in timing_sections:
        cells = []
        for result in results:
            phase11_runs = [run for run in result.timed_runs if isinstance(run, TimedPhase11Run)]
            seconds = statistics.median(run.timings.get(section, 0.0) for run in phase11_runs)
            elapsed = statistics.median(run.elapsed_seconds for run in result.timed_runs)
            cells.append(f"{seconds:.3f}s ({100.0 * seconds / elapsed:.1f}%)")
        print(f"{label:<38}" + "".join(f"{cell:>23}" for cell in cells))
    for result in results:
        timing = _median_section_timings(result)
        elapsed = statistics.median(run.elapsed_seconds for run in result.timed_runs)
        exclusive_sections = sum(
            timing[section]
            for section in (
                "learner_forward",
                "historical_opponent_forward",
                "event_advancement",
                "provider_sampling",
                "cache_lookup",
                "action_sampling",
                "environment_step",
            )
        )
        other_seconds = max(0.0, elapsed - exclusive_sections)
        print(
            f"{result.condition}: other/uninstrumented={other_seconds:.3f}s "
            f"({100.0 * other_seconds / elapsed:.1f}%)"
        )

    print("\nRun diagnostics")
    for result in results:
        median_elapsed = statistics.median(run.elapsed_seconds for run in result.timed_runs)
        median_run = min(
            result.timed_runs,
            key=lambda run: abs(run.elapsed_seconds - median_elapsed),
        )
        phase11_runs = [run for run in result.timed_runs if isinstance(run, TimedPhase11Run)]
        print(
            f"{result.condition}: decisions={median_run.decision_count}, "
            f"learner={median_run.learner_decisions}, "
            f"opponent_neural={median_run.opponent_neural_decisions}, "
            f"opponent_ids={median_run.unique_opponent_policy_ids}, "
            f"cache_loads/hits={result.actor_cache_loads_including_warmup}/"
            f"{result.actor_cache_hits_including_warmup}, "
            f"illegal={median_run.illegal_actions}, action64={median_run.action64_actions}, "
            f"non_finite={median_run.non_finite_count}, "
            f"G6={median_run.g6_boundary_valid}, "
            f"unfinished={not median_run.no_unfinished_learner_sequence}, "
            f"state_zero={median_run.recurrent_state_safe_zero}, "
            f"repeat_decisions_per_second="
            f"{[round(run.decisions_per_second, 1) for run in result.timed_runs]}, "
            f"sampled_policies_by_repeat="
            f"{[run.unique_opponent_policy_ids for run in phase11_runs]}"
        )


def _print_report(results: tuple[ConditionResult, ...]) -> None:
    old_median = next(
        result.median_decisions_per_second for result in results if result.condition == "old"
    )
    print("\nComparison")
    print(f"{'Condition':<14} {'Best dec/s':>12} {'Median dec/s':>14} {'Slowdown vs old':>17}")
    for result in results:
        slowdown = old_median / result.median_decisions_per_second
        print(
            f"{result.condition:<14} {result.best_decisions_per_second:>12.1f} "
            f"{result.median_decisions_per_second:>14.1f} {slowdown:>16.3f}x"
        )

    print("\nConfiguration")
    print(
        json.dumps(
            {
                "environment_count": ENVIRONMENT_COUNT,
                "base_vector_steps": BASE_VECTOR_STEPS,
                "timed_repeats": TIMED_REPEATS,
                "warmup_runs_per_condition": 1,
                "torch_cpu_threads": TORCH_CPU_THREADS,
                "development_root_seed": DEVELOPMENT_ROOT_SEED,
                "action_sampling_seed": ACTION_SAMPLING_SEED,
                "learner_initialization_seed": LEARNER_INITIALIZATION_SEED,
                "opponent_seed": OPPONENT_SEED,
                "learner_initialization": str(LEARNER_INITIALIZATION_CHECKPOINT),
                "old_opponent_actor": str(OPPONENT_CHECKPOINTS[-1][1]),
                "pfsp_population_sizes": list(PFSP_POPULATION_SIZES),
                "pfsp_candidate_policy_ids": [policy_id for policy_id, *_ in OPPONENT_CHECKPOINTS],
                "benchmark_only_pfsp_values": {
                    "ema_beta": 0.2,
                    "unobserved_learner_win_rate": 0.5,
                    "empty_category_strategy": "proportional",
                    "observed_learner_win_rate_ema": 0.5,
                },
                "g6": "200 base vector steps, then drain active deals to scored boundaries",
                "old_condition": "CentralizedCriticCollector + RecurrentNeuralAgent opponent",
            },
            indent=2,
        )
    )

    print("\nTimed repeat details")
    for result in results:
        print(f"\n{result.condition}:")
        for repeat_index, run in enumerate(result.timed_runs, start=1):
            print(
                f"  repeat {repeat_index}: decisions={run.decision_count}, "
                f"elapsed={run.elapsed_seconds:.3f}s, decisions/s={run.decisions_per_second:.1f}, "
                f"learner={run.learner_decisions}, opponent_neural={run.opponent_neural_decisions}, "
                f"neural_group_calls={run.neural_policy_group_calls}, "
                f"neural_policy_ids={run.unique_neural_policy_ids}, "
                f"opponent_policy_ids={run.unique_opponent_policy_ids}, "
                f"cache_loads={run.actor_cache_loads}, cache_hits={run.actor_cache_hits}, "
                f"illegal={run.illegal_actions}, action64={run.action64_actions}, "
                f"non_finite={run.non_finite_count}, deals={run.completed_deals}, "
                f"passed_out={run.passed_out_deals}, actual_vector_steps={run.actual_vector_steps}, "
                f"G6={run.g6_boundary_valid}, unfinished={not run.no_unfinished_learner_sequence}, "
                f"state_zero={run.recurrent_state_safe_zero}"
            )
        print(
            f"  actor cache totals including warm-up: loads={result.actor_cache_loads_including_warmup}, "
            f"hits={result.actor_cache_hits_including_warmup}"
        )


def main() -> None:
    global DEVELOPMENT_ROOT_SEED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="development/training seed index; held-out evaluation seed is prohibited",
    )
    parser.add_argument(
        "--mode",
        choices=("p10-validation", "full-update", "rollout-only"),
        default="full-update",
        help="validate only P10, run the five-condition benchmark, or use the preserved rollout-only diagnosis",
    )
    arguments = parser.parse_args()
    if arguments.seed == 20250910:
        parser.error("held-out seed 20250910 is prohibited for this benchmark")
    if not 0 <= arguments.seed < 10:
        parser.error("seed index must be in [0, 9], the available PPO training-seed indices")

    config = (
        _phase10_validation_config(arguments.seed)
        if arguments.mode == "p10-validation"
        else _full_update_config(arguments.seed)
    )
    DEVELOPMENT_ROOT_SEED = config.environment_root_seed
    torch.set_num_threads(TORCH_CPU_THREADS)
    rules = RulesConfig()

    if arguments.mode == "p10-validation":
        learner_actor, learner_critic = build_phase10_networks(config)
        warmup, timed_runs = _benchmark_full_update_condition(
            "P10 centralized",
            rules,
            config,
            learner_actor,
            learner_critic,
            PolicyRegistry(rules=rules),
            "unused",
            (),
        )
        _print_p10_validation(config, warmup, timed_runs)
        return

    registry, latest_policy_id, population_policy_ids = _register_benchmark_population(rules)
    if arguments.mode == "rollout-only":
        learner_actor, learner_critic = _build_learner_networks()
        results = tuple(
            _benchmark_phase11(
                f"{population_size} actor{'s' if population_size > 1 else ''}",
                rules,
                learner_actor,
                learner_critic,
                registry,
                latest_policy_id,
                population_policy_ids[:population_size],
            )
            for population_size in PFSP_POPULATION_SIZES
        )
        _print_pfsp_diagnosis(results)
        return

    learner_actor, learner_critic = build_phase10_networks(config)
    conditions = (
        "P10 centralized",
        "P11 latest-only",
        "P11 PFSP 1",
        "P11 PFSP 2",
        "P11 PFSP 4",
    )
    full_update_results = {
        condition: _benchmark_full_update_condition(
            condition,
            rules,
            config,
            learner_actor,
            learner_critic,
            registry,
            latest_policy_id,
            population_policy_ids,
        )[1]
        for condition in conditions
    }
    _print_full_update_report(full_update_results)


if __name__ == "__main__":
    main()
