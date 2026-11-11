"""Opt-in seating experiment; no optimizer, training loop, or historical edits.

Resume callers reconstruct the population from the saved population_state before
building the collector, exactly as the frozen Phase 11 driver does. Actor/critic
and any future optimizer restoration remain the checkpoint loader's job.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.env.vec import VecTarneeb
from tarneeb.league.cache import HistoricalActorCache
from tarneeb.league.population import LeaguePopulation
from tarneeb.models.recurrent import RecurrentActorNetwork, RecurrentCriticNetwork
from tarneeb.training.league_rollout import LeagueRecurrentCollector
from tarneeb.training.phase11_training import (
    Phase11A3Config,
    Phase11SmokeConfig,
    Phase11SmokeResult,
    _capture_phase11_resume_state,
    _restore_phase11_resume_state,
    build_phase11_collector,
    build_phase11_opponent_provider,
)

Mode = Literal["legacy", "balanced"]
Config = Phase11SmokeConfig | Phase11A3Config
SEATING_KEY = "seating_experiment_v1"


def dealers(count: int, mode: Mode = "legacy") -> tuple[int, ...]:
    if count < 1 or mode not in ("legacy", "balanced"):
        raise ValueError("Positive environment count and legacy/balanced mode required")
    return tuple((i if mode == "legacy" else i // 2) % 4 for i in range(count))


class SeatingEnvironmentsV1(VecTarneeb):
    """Set fixed dealers before any reset, deal, or recurrent initialization."""

    def __init__(
        self, num_envs: int, root_seed: int = 0,
        rules: RulesConfig | None = None, *, mode: Mode = "legacy",
    ) -> None:
        assignment = dealers(num_envs, mode)
        super().__init__(num_envs, root_seed, rules)
        self.seating_mode = mode
        for environment, dealer in zip(self.environments, assignment, strict=True):
            assert environment.state is None
            environment.dealer_seat = dealer


def build_collector(
    config: Config, population: LeaguePopulation, actor: RecurrentActorNetwork,
    critic: RecurrentCriticNetwork, rules: RulesConfig, *, mode: Mode = "legacy",
) -> tuple[LeagueRecurrentCollector, torch.Generator, HistoricalActorCache]:
    dealers(config.environment_count, mode)
    if mode == "legacy":
        return build_phase11_collector(config, population, actor, critic, rules)
    environments = SeatingEnvironmentsV1(
        config.environment_count, config.environment_root_seed, rules, mode=mode,
    )
    generator = torch.Generator().manual_seed(config.action_sampling_seed)
    cache = HistoricalActorCache(population.registry, rules=rules)
    collector = LeagueRecurrentCollector(
        environments=environments, actor=actor, critic=critic,
        rollout_length=config.rollout_length, action_sampling_generator=generator,
        opponent_provider=build_phase11_opponent_provider(config.condition, population),
        actor_cache=cache, chunk_length=config.chunk_length, opponent_seed=config.opponent_seed,
    )
    return collector, generator, cache


def seating_identity(collector: LeagueRecurrentCollector) -> dict[str, Any]:
    environments = collector.environments
    mode: Mode = getattr(environments, "seating_mode", "legacy")
    assignment = dealers(environments.num_envs, mode)
    for i, (environment, dealer) in enumerate(
        zip(environments.environments, assignment, strict=True)
    ):
        if environment.dealer_seat != dealer or int(collector.learner_teams[i]) != i % 2:
            raise ValueError("Collector seating changed")
        if environment.state is not None and environment.state.dealer_seat != dealer:
            raise ValueError("Live dealer differs from configured seating")
    return {"version": 1, "mode": mode, "dealers": list(assignment),
            "learner_teams": [i % 2 for i in range(environments.num_envs)]}


def capture_resume(
    config: Config, result: Phase11SmokeResult, population: LeaguePopulation,
    collector: LeagueRecurrentCollector, action_generator: torch.Generator,
    shuffle_generator: np.random.Generator,
) -> dict[str, Any]:
    identity = seating_identity(collector)
    payload = _capture_phase11_resume_state(
        config, result, population, collector, action_generator, shuffle_generator,
    )
    payload[SEATING_KEY] = identity
    return payload


def restore_resume(
    payload: dict[str, Any], config: Config, population: LeaguePopulation,
    collector: LeagueRecurrentCollector, action_generator: torch.Generator,
    shuffle_generator: np.random.Generator, result: Phase11SmokeResult,
) -> None:
    """Reject cross-mode/malformed seating before mutating collector or RNGs.

    Untagged historical resumes are accepted ONLY for legacy seating. Starting a
    balanced fine-tune from historical weights is initialization, never resume.
    """
    identity = seating_identity(collector)
    saved = payload.get(SEATING_KEY)
    if saved != identity and not (saved is None and identity["mode"] == "legacy"):
        raise ValueError("Resume seating mode/identity mismatch")
    states = payload.get("environment_states", [])
    if len(states) != len(identity["dealers"]) or any(
        state is None or state.dealer_seat != dealer
        for state, dealer in zip(states, identity["dealers"], strict=True)
    ):
        raise ValueError("Resume state dealer mismatch")
    if population.state_dict() != payload.get("population_state"):
        raise ValueError("Reconstruct saved population before building resume collector")
    _restore_phase11_resume_state(
        payload, config, population, collector, action_generator, shuffle_generator, result,
    )
