"""Common latest-only and PFSP opponent providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class PFSPConfig:
    """Frozen Phase 11 A3 PFSP settings."""

    alpha: float = 2.0
    ema_window_deals: int = 500
    ema_beta: float | None = 2.0 / 501.0
    champion_share: float = 0.30
    recent_share: float = 0.35
    older_share: float = 0.22
    strong_share: float = 0.09
    random_share: float = 0.04
    recent_limit: int = 5
    older_limit: int = 10
    unobserved_learner_win_rate: float | None = 0.5
    empty_category_strategy: str | None = "proportional"

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.ema_window_deals < 1:
            raise ValueError("alpha and ema_window_deals must be positive")
        if self.ema_beta is not None and not 0.0 <= self.ema_beta <= 1.0:
            raise ValueError("ema_beta must be in [0, 1] when supplied")
        if self.unobserved_learner_win_rate is not None and (
            not np.isfinite(self.unobserved_learner_win_rate)
            or not 0.0 <= self.unobserved_learner_win_rate <= 1.0
        ):
            raise ValueError("unobserved_learner_win_rate must be finite and in [0, 1]")
        if self.empty_category_strategy not in (None, "proportional"):
            raise ValueError("empty_category_strategy must be None or 'proportional'")
        if (self.recent_limit != 5 or self.older_limit != 10) and (
            self.recent_limit < 1 or self.older_limit < 0
        ):
            raise ValueError("recent_limit must be positive and older_limit non-negative")
        shares = self.category_shares
        if any(not np.isfinite(share) or share < 0 for share in shares.values()):
            raise ValueError("category shares must be finite and non-negative")
        if not np.isclose(sum(shares.values()), 1.0):
            raise ValueError("PFSP category shares must sum to 1")

    @property
    def category_shares(self) -> dict[str, float]:
        return {
            "champion": self.champion_share,
            "recent": self.recent_share,
            "older": self.older_share,
            "strong": self.strong_share,
            "random": self.random_share,
        }


@dataclass
class PFSPPerformance:
    """EMA probability that the learner beats this opponent."""

    learner_win_rate_against_opponent_ema: float | None = None
    deals_observed: int = 0

    def update(self, learner_win_rate_against_opponent: float, ema_beta: float) -> float:
        """Update with ``new = beta * NEW + (1 - beta) * OLD``.

        For A3, ``beta=2/501`` is intentionally the small weight on the newest
        deal observation; it is not the coefficient on the retained history.
        """
        if (
            not np.isfinite(learner_win_rate_against_opponent)
            or not 0.0 <= learner_win_rate_against_opponent <= 1.0
        ):
            raise ValueError("learner_win_rate_against_opponent must be in [0, 1]")
        if not 0.0 <= ema_beta <= 1.0:
            raise ValueError("ema_beta must be in [0, 1]")
        self.learner_win_rate_against_opponent_ema = (
            learner_win_rate_against_opponent
            if self.learner_win_rate_against_opponent_ema is None
            else ema_beta * learner_win_rate_against_opponent
            + (1.0 - ema_beta) * self.learner_win_rate_against_opponent_ema
        )
        self.deals_observed += 1
        return self.learner_win_rate_against_opponent_ema


class OpponentProvider(Protocol):
    """Selection interface shared by the A3 conditions."""

    def sample(self) -> str: ...

    def probabilities(self) -> dict[str, float]: ...


@dataclass
class LatestOnlyProvider:
    """Always select the configured latest frozen/current learner snapshot."""

    latest_policy_id: str

    def sample(self) -> str:
        return self.latest_policy_id

    def probabilities(self) -> dict[str, float]:
        return {self.latest_policy_id: 1.0}


@dataclass
class PFSPProvider:
    """Sample category shares, then apply PFSP weighting inside archive pools."""

    policies_by_category: dict[str, tuple[str, ...]]
    performance: dict[str, PFSPPerformance]
    rng: np.random.Generator
    config: PFSPConfig = field(default_factory=PFSPConfig)
    last_probabilities: dict[str, float] = field(default_factory=dict)

    def probabilities(self) -> dict[str, float]:
        category_by_policy: dict[str, str] = {}
        for category, policy_ids in self.policies_by_category.items():
            for policy_id in policy_ids:
                previous_category = category_by_policy.get(policy_id)
                if previous_category is not None and previous_category != category:
                    raise ValueError(
                        f"policy {policy_id!r} appears in multiple PFSP categories: "
                        f"{previous_category!r} and {category!r}"
                    )
                category_by_policy[policy_id] = category
        available = {
            category: tuple(policy_ids)
            for category, policy_ids in self.policies_by_category.items()
            if policy_ids and self.config.category_shares.get(category, 0.0) > 0.0
        }
        if not available:
            raise ValueError("PFSP requires at least one eligible opponent")
        missing = {
            category
            for category, share in self.config.category_shares.items()
            if share > 0.0 and category not in available
        }
        if missing and self.config.empty_category_strategy is None:
            raise ValueError(
                "PFSP categories are empty: "
                f"{sorted(missing)}; empty_category_strategy is unresolved and must be set"
            )
        if missing and self.config.empty_category_strategy != "proportional":
            raise ValueError(f"unsupported empty-category handling for {sorted(missing)}")
        available_share_sum = sum(self.config.category_shares[category] for category in available)
        result: dict[str, float] = {}
        for category, policy_ids in available.items():
            category_share = self.config.category_shares[category] / available_share_sum
            if category in ("recent", "older"):
                weights = np.array(
                    [
                        (1.0 - self._learner_win_rate_against_opponent(policy_id))
                        ** self.config.alpha
                        for policy_id in policy_ids
                    ],
                    dtype=np.float64,
                )
                if not np.isfinite(weights).all() or weights.sum() <= 0.0:
                    weights = np.ones(len(policy_ids), dtype=np.float64)
                within_category = weights / weights.sum()
            else:
                within_category = np.full(len(policy_ids), 1.0 / len(policy_ids))
            for policy_id, within_probability in zip(policy_ids, within_category, strict=True):
                result[policy_id] = category_share * float(within_probability)
        total = sum(result.values())
        if not np.isfinite(list(result.values())).all() or total <= 0:
            raise AssertionError("PFSP produced invalid probability mass")
        self.last_probabilities = {
            policy: probability / total for policy, probability in result.items()
        }
        return dict(self.last_probabilities)

    def sample(self) -> str:
        probabilities = self.probabilities()
        policy_ids = tuple(probabilities)
        selected = int(self.rng.choice(len(policy_ids), p=np.array(list(probabilities.values()))))
        return policy_ids[selected]

    def _learner_win_rate_against_opponent(self, policy_id: str) -> float:
        value = self.performance.get(policy_id)
        if value is not None and value.learner_win_rate_against_opponent_ema is not None:
            return value.learner_win_rate_against_opponent_ema
        unobserved_rate = self.config.unobserved_learner_win_rate
        if unobserved_rate is None:
            raise ValueError(
                f"no learner win-rate observations for opponent {policy_id!r}; "
                "unobserved_learner_win_rate is unresolved and must be configured"
            )
        return unobserved_rate


def restore_rng(state: dict[str, object]) -> np.random.Generator:
    """Build a generator at an exact serialized bit-generator state."""
    generator = np.random.default_rng()
    generator.bit_generator.state = state
    return generator
