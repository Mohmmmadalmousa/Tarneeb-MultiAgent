"""Serializable Phase 11 population, archive bookkeeping and promotion gates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from tarneeb.league.registry import PolicyRegistry
from tarneeb.league.sampling import PFSPConfig, PFSPPerformance


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """Phase 11 snapshot and PFSP configuration; no learner/PPO fields."""

    pfsp: PFSPConfig = field(default_factory=PFSPConfig)
    snapshot_environment_steps: int = 2_000_000
    snapshot_updates: int = 50
    promotion_min_duplicated_deals: int = 2_000
    promotion_confidence_level: float = 0.95
    older_reservoir_capacity: int = 10
    diversity_minimum_distance: float | None = None

    def __post_init__(self) -> None:
        if self.snapshot_environment_steps < 1 or self.snapshot_updates < 1:
            raise ValueError("snapshot cadence thresholds must be positive")
        if self.promotion_min_duplicated_deals < 1 or self.older_reservoir_capacity < 0:
            raise ValueError("promotion and reservoir thresholds are invalid")
        if not 0.0 < self.promotion_confidence_level < 1.0:
            raise ValueError("promotion_confidence_level must be between 0 and 1")
        if self.older_reservoir_capacity != self.pfsp.older_limit:
            raise ValueError("older reservoir capacity must match the PFSP older-pool limit")


@dataclass(frozen=True, slots=True)
class PromotionEvaluation:
    """Synthetic or measured paired results used by the promotion rule."""

    paired_deal_count_vs_champion: int
    mean_paired_score_difference_vs_champion: float
    champion_paired_score_difference_ci_low: float
    champion_paired_score_difference_ci_high: float
    required_recent_snapshot_policy_ids: tuple[str, ...]
    mean_paired_score_difference_by_recent_snapshot: dict[str, float]
    mean_paired_score_difference_vs_strong: float


@dataclass
class LeaguePopulation:
    """Policy archive and independent PFSP RNG, persisted outside checkpoints."""

    registry: PolicyRegistry = field(default_factory=PolicyRegistry)
    config: LeagueConfig = field(default_factory=LeagueConfig)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    eligible_policy_ids: list[str] = field(default_factory=list)
    latest_policy_id: str | None = None
    champion_policy_id: str | None = None
    performance: dict[str, PFSPPerformance] = field(default_factory=dict)
    last_snapshot_environment_steps: int = 0
    last_snapshot_update: int = 0
    snapshot_count: int = 0
    older_reservoir_seen: int = 0
    older_policy_ids: list[str] = field(default_factory=list)
    older_considered_policy_ids: list[str] = field(default_factory=list)

    def add(self, policy_id: str, make_latest: bool = True) -> None:
        if policy_id not in self.registry.records:
            raise KeyError(f"policy {policy_id!r} is not registered")
        if policy_id not in self.eligible_policy_ids:
            self.eligible_policy_ids.append(policy_id)
        if make_latest:
            self.latest_policy_id = policy_id
        record = self.registry.records[policy_id]
        if record.category == "champion":
            self.champion_policy_id = policy_id
        self._refresh_archive_categories()

    def register_snapshot(
        self,
        *,
        policy_id: str,
        checkpoint_path: Path,
        source_phase: str,
        seed: int | None = None,
        update: int | None = None,
        make_latest: bool = True,
    ) -> str:
        """Validate checkpoint compatibility and actor identity before archiving."""
        registered_id = self.registry.register_checkpoint(
            policy_id=policy_id,
            checkpoint_path=checkpoint_path,
            source_phase=source_phase,
            category="snapshot",
            seed=seed,
            update=update,
        )
        self.add(registered_id, make_latest=make_latest)
        self.snapshot_count += 1
        return registered_id

    def sample_categories(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {key: [] for key in self.config.pfsp.category_shares}
        for policy_id in self.eligible_policy_ids:
            record = self.registry.records[policy_id]
            if record.eligible:
                category = self._sampling_category(record.category)
                if category is not None:
                    groups[category].append(policy_id)
        for name, values in groups.items():
            if name in ("recent", "older"):
                values.sort(key=lambda policy: self.registry.records[policy].registration_order)
            else:
                values.sort()
        return {key: tuple(values) for key, values in groups.items()}

    def should_snapshot(self, environment_steps: int, updates: int) -> bool:
        return (
            environment_steps - self.last_snapshot_environment_steps
            >= self.config.snapshot_environment_steps
            or updates - self.last_snapshot_update >= self.config.snapshot_updates
        )

    def note_snapshot(self, environment_steps: int, updates: int) -> None:
        self.last_snapshot_environment_steps = environment_steps
        self.last_snapshot_update = updates

    def update_performance(
        self,
        policy_id: str,
        learner_win_rate_against_opponent: float,
        ema_beta: float,
    ) -> float:
        statistic = self.performance.setdefault(policy_id, PFSPPerformance())
        return statistic.update(learner_win_rate_against_opponent, ema_beta)

    def promotion_passes(self, evaluation: PromotionEvaluation) -> bool:
        required_policy_ids = evaluation.required_recent_snapshot_policy_ids
        recent_score_differences = evaluation.mean_paired_score_difference_by_recent_snapshot
        if len(required_policy_ids) != len(set(required_policy_ids)):
            return False
        if set(required_policy_ids) != set(recent_score_differences):
            return False
        return (
            evaluation.paired_deal_count_vs_champion >= self.config.promotion_min_duplicated_deals
            and evaluation.champion_paired_score_difference_ci_low > 0.0
            and all(value >= 0.0 for value in recent_score_differences.values())
            and evaluation.mean_paired_score_difference_vs_strong > 0.0
        )

    def promote(self, policy_id: str, evaluation: PromotionEvaluation) -> None:
        """Promote a registered candidate after the frozen gate succeeds."""
        if policy_id not in self.eligible_policy_ids:
            raise ValueError(f"promotion candidate {policy_id!r} is not eligible history")
        if not self.promotion_passes(evaluation):
            raise ValueError("candidate does not satisfy the promotion contract")
        candidate = self.registry.records[policy_id]
        if candidate.checkpoint_path is None or candidate.actor_fingerprint is None:
            raise ValueError("only a registered neural snapshot can become champion")

        previous_champion = self.champion_policy_id
        if previous_champion is not None and previous_champion != policy_id:
            previous = self.registry.records[previous_champion]
            self.registry.records[previous_champion] = replace(previous, category="snapshot")
        self.registry.records[policy_id] = replace(candidate, category="champion")
        self.champion_policy_id = policy_id
        self._refresh_archive_categories()
        champions = [
            record.policy_id
            for record in self.registry.records.values()
            if record.category == "champion"
        ]
        if champions != [policy_id]:
            raise AssertionError(
                f"successful promotion must leave exactly one champion, got {champions}"
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": "phase11_league_state_v1",
            "registry": self.registry.to_dict(),
            "config": _config_to_dict(self.config),
            "rng_state": self.rng.bit_generator.state,
            "eligible_policy_ids": list(self.eligible_policy_ids),
            "latest_policy_id": self.latest_policy_id,
            "champion_policy_id": self.champion_policy_id,
            "performance": {
                policy_id: asdict(stats) for policy_id, stats in self.performance.items()
            },
            "last_snapshot_environment_steps": self.last_snapshot_environment_steps,
            "last_snapshot_update": self.last_snapshot_update,
            "snapshot_count": self.snapshot_count,
            "older_reservoir_seen": self.older_reservoir_seen,
            "older_policy_ids": list(self.older_policy_ids),
            "older_considered_policy_ids": list(self.older_considered_policy_ids),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), sort_keys=True, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LeaguePopulation:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_state_dict(payload)

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> LeaguePopulation:
        """Reconstruct the registry, archive, performance, cadence and PFSP RNG."""
        if payload.get("format_version") != "phase11_league_state_v1":
            raise ValueError("unsupported Phase 11 league-state format")
        config_data = dict(payload["config"])
        pfsp = PFSPConfig(**config_data.pop("pfsp"))
        population = cls(
            registry=PolicyRegistry.from_dict(payload["registry"]),
            config=LeagueConfig(pfsp=pfsp, **config_data),
            rng=np.random.default_rng(),
            eligible_policy_ids=list(payload["eligible_policy_ids"]),
            latest_policy_id=payload["latest_policy_id"],
            champion_policy_id=payload["champion_policy_id"],
            performance={
                key: PFSPPerformance(**value) for key, value in payload["performance"].items()
            },
            last_snapshot_environment_steps=payload["last_snapshot_environment_steps"],
            last_snapshot_update=payload["last_snapshot_update"],
            snapshot_count=payload["snapshot_count"],
            older_reservoir_seen=payload["older_reservoir_seen"],
            older_policy_ids=list(payload["older_policy_ids"]),
            older_considered_policy_ids=list(payload["older_considered_policy_ids"]),
        )
        population.rng.bit_generator.state = payload["rng_state"]
        return population

    def restore_state_dict(self, payload: dict[str, Any]) -> None:
        """Restore this population in place from a validated serialized state."""
        restored = type(self).from_state_dict(payload)
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)

    def _refresh_archive_categories(self) -> None:
        snapshots = [
            policy_id
            for policy_id in self.eligible_policy_ids
            if self.registry.records[policy_id].category in ("snapshot", "recent", "older")
        ]
        snapshots.sort(key=lambda policy: self.registry.records[policy].registration_order)
        recent = snapshots[-self.config.pfsp.recent_limit :]
        older_candidates = snapshots[: max(0, len(snapshots) - len(recent))]
        newly_older = [
            policy_id
            for policy_id in older_candidates
            if policy_id not in self.older_considered_policy_ids
        ]
        for policy_id in newly_older:
            self.older_considered_policy_ids.append(policy_id)
            self._consider_older_snapshot(policy_id)
        for policy_id in snapshots:
            record = self.registry.records[policy_id]
            new_category = (
                "recent"
                if policy_id in recent
                else "older"
                if policy_id in self.older_policy_ids
                else "snapshot"
            )
            self.registry.records[policy_id] = replace(record, category=new_category)

    def _consider_older_snapshot(self, policy_id: str) -> None:
        """Apply behavioral filtering and classic reservoir replacement once."""
        record = self.registry.records[policy_id]
        if self.config.diversity_minimum_distance is not None:
            candidate = record.behavioral_fingerprint
            archived = [
                self.registry.records[existing].behavioral_fingerprint
                for existing in self.older_policy_ids
            ]
            if candidate is None or any(value is None for value in archived):
                return
            distances = [
                float(np.linalg.norm(np.asarray(candidate) - np.asarray(value)))
                for value in archived
            ]
            if distances and min(distances) < self.config.diversity_minimum_distance:
                return
        self.older_reservoir_seen += 1
        capacity = self.config.older_reservoir_capacity
        if len(self.older_policy_ids) < capacity:
            self.older_policy_ids.append(policy_id)
        elif capacity:
            slot = int(self.rng.integers(self.older_reservoir_seen))
            if slot < capacity:
                self.older_policy_ids[slot] = policy_id

    @staticmethod
    def _sampling_category(category: str) -> str | None:
        if category in ("champion", "recent", "older", "strong", "random"):
            return category
        return None


def _config_to_dict(config: LeagueConfig) -> dict[str, Any]:
    values = asdict(config)
    return values
