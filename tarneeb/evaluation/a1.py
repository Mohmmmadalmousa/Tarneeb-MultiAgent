"""Ablation A1: Event-GRU history versus padded feedforward history.

The blueprint's question is whether learned recurrence over the public event
stream beats a fixed-length ordered encoding of it. Answering that fairly is
harder than running two trainings and comparing two numbers, because almost
every difference between two runs is a confound. A1 controls what it can and
states what it cannot.

**How the comparison is made paired.** Each condition is evaluated against the
*same* frozen anchor on the *same* duplicate deals: the frozen arena derives its
match seeds from ``(evaluation_seed, pair_index)``, so two evaluations with the
same seed and pair count play an identical sequence of hands. Pair *i* therefore
means the same cards for both conditions, and the per-pair scores can be
subtracted directly. That removes deal luck from the comparison instead of
merely averaging over it.

A difference of averages would also have removed deal luck *in expectation*, but
with a few hundred pairs the residual variance is large enough to hide the effect
sizes the blueprint anticipates (0.3-0.8 points per deal). Pairing removes it
exactly.

**Two views, because either alone would mislead.**

- *Against a common anchor.* Both conditions play the frozen Strong, Basic and
  Random heuristics. This is what the blueprint's A1 describes and what makes the
  result comparable to every other number in this project.
- *Head to head.* The two conditions play each other directly on duplicate
  deals. This answers "which would you rather hold" without routing through a
  third policy, and it can disagree with the anchor view -- non-transitivity is
  ordinary in card games, and seeing it is informative rather than embarrassing.

**Capacity.** The frozen parameter counts are reported because the feedforward
history path is *larger* (329,536 against 255,112). The baseline is therefore not
capacity-handicapped, so a GRU win could not be dismissed as extra parameters.

Nothing here assumes a direction. ``GRU better``, ``no detectable difference``
and ``GRU worse`` are all admissible, and the verdict is derived from the
interval rather than chosen.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from tarneeb.agents.base import Agent
from tarneeb.evaluation.arena import Arena
from tarneeb.evaluation.metrics import PairedOutcome
from tarneeb.evaluation.stats import (
    bootstrap_mean_confidence_interval,
    sample_standard_deviation,
    wilson_score_interval,
)

A1_BOOTSTRAP_RESAMPLES = 10_000
"""Resamples for every A1 interval. Fixed so results are reproducible."""


@dataclass(frozen=True, slots=True)
class ConditionEvaluation:
    """One condition measured against one anchor, with its per-pair scores kept."""

    condition: str
    opponent: str
    paired_scores: tuple[float, ...]
    metrics: dict[str, Any]

    @property
    def mean_paired_score(self) -> float:
        return float(np.mean(self.paired_scores))


def evaluate_condition(
    agent: Agent,
    opponent: Agent,
    opponent_name: str,
    condition: str,
    paired_deal_count: int,
    evaluation_seed: int,
) -> ConditionEvaluation:
    """Evaluate one condition and keep the per-pair scores for pairing later."""
    arena = Arena()
    if hasattr(agent, "restart"):
        agent.restart()
    result = arena.evaluate(
        agent,
        opponent,
        paired_deal_count=paired_deal_count,
        evaluation_seed=evaluation_seed,
        bootstrap_resamples=A1_BOOTSTRAP_RESAMPLES,
    )
    scores = tuple(float(outcome.paired_score_difference) for outcome in result.paired_outcomes)
    return ConditionEvaluation(
        condition=condition,
        opponent=opponent_name,
        paired_scores=scores,
        metrics=result.metrics,
    )


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """``GRU - FF`` on identical deals against one anchor."""

    opponent: str
    gru_mean: float
    ff_mean: float
    difference_mean: float
    difference_ci: dict[str, Any]
    difference_sigma: float
    gru_better_pair_count: int
    ff_better_pair_count: int
    tie_count: int
    pair_count: int
    gru_better_rate: float
    gru_better_rate_wilson: dict[str, Any]

    @property
    def interval_excludes_zero(self) -> bool:
        """True when the 95% interval lies entirely on one side of zero."""
        lower = float(self.difference_ci["lower_bound"])
        upper = float(self.difference_ci["upper_bound"])
        return lower > 0.0 or upper < 0.0

    @property
    def verdict(self) -> str:
        """The admissible conclusion this interval supports."""
        if not self.interval_excludes_zero:
            return "no detectable difference"
        return "GRU better" if self.difference_mean > 0 else "GRU worse"

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["verdict"] = self.verdict
        values["interval_excludes_zero"] = self.interval_excludes_zero
        return values


def compare_paired(
    gru: ConditionEvaluation,
    feedforward: ConditionEvaluation,
    bootstrap_seed: int,
) -> PairedComparison:
    """Difference two conditions pair by pair against the same anchor.

    Raises if the two evaluations do not describe the same deals, because a
    mismatched pairing would silently produce a meaningless interval.
    """
    if gru.opponent != feedforward.opponent:
        raise ValueError(
            f"cannot pair across different anchors: {gru.opponent!r} vs {feedforward.opponent!r}"
        )
    if len(gru.paired_scores) != len(feedforward.paired_scores):
        raise ValueError(
            f"pair counts differ ({len(gru.paired_scores)} vs "
            f"{len(feedforward.paired_scores)}); the deals are not the same"
        )

    differences = np.asarray(gru.paired_scores, dtype=np.float64) - np.asarray(
        feedforward.paired_scores, dtype=np.float64
    )
    interval = bootstrap_mean_confidence_interval(
        differences, seed=bootstrap_seed, resamples=A1_BOOTSTRAP_RESAMPLES
    )
    gru_better = int((differences > 0).sum())
    ff_better = int((differences < 0).sum())
    ties = int((differences == 0).sum())
    decisive = gru_better + ff_better

    return PairedComparison(
        opponent=gru.opponent,
        gru_mean=gru.mean_paired_score,
        ff_mean=feedforward.mean_paired_score,
        difference_mean=float(differences.mean()),
        difference_ci=interval.as_dict(),
        difference_sigma=sample_standard_deviation(differences),
        gru_better_pair_count=gru_better,
        ff_better_pair_count=ff_better,
        tie_count=ties,
        pair_count=int(differences.size),
        gru_better_rate=gru_better / decisive if decisive else 0.0,
        gru_better_rate_wilson=(
            wilson_score_interval(gru_better, decisive).as_dict() if decisive else {"mean": None}
        ),
    )


def head_to_head(
    gru_agent: Agent,
    ff_agent: Agent,
    paired_deal_count: int,
    evaluation_seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Play the two conditions against each other on duplicate deals."""
    arena = Arena()
    for agent in (gru_agent, ff_agent):
        if hasattr(agent, "restart"):
            agent.restart()
    result = arena.evaluate(
        gru_agent,
        ff_agent,
        paired_deal_count=paired_deal_count,
        evaluation_seed=evaluation_seed,
        bootstrap_resamples=A1_BOOTSTRAP_RESAMPLES,
    )
    scores = np.asarray(
        [outcome.paired_score_difference for outcome in result.paired_outcomes],
        dtype=np.float64,
    )
    interval = bootstrap_mean_confidence_interval(
        scores, seed=bootstrap_seed, resamples=A1_BOOTSTRAP_RESAMPLES
    )
    return {
        "mean_paired_score_difference": float(scores.mean()),
        "paired_score_difference_ci": interval.as_dict(),
        "win_rate": result.metrics["win_rate"],
        "win_rate_wilson_interval": result.metrics["win_rate_wilson_interval"],
        "pair_count": int(scores.size),
        "illegal_action_rate": result.metrics["illegal_action_rate"],
        "no_trump_action_rate": result.metrics["no_trump_action_rate"],
        "replay_integrity_rate": result.metrics["replay_integrity_rate"],
        "role_split_mean_score": result.metrics["role_split_mean_score"],
    }


def measure_inference_latency(
    agent: Agent, paired_deal_count: int, evaluation_seed: int, opponent: Agent
) -> dict[str, float]:
    """Return decisions per second and per-decision latency for one condition.

    Measured through the frozen arena rather than on synthetic batches, so the
    number reflects the cost a deployed seat actually pays -- one decision at a
    time, including observation construction.
    """
    arena = Arena()
    if hasattr(agent, "restart"):
        agent.restart()
    start = time.perf_counter()
    result = arena.evaluate(
        agent,
        opponent,
        paired_deal_count=paired_deal_count,
        evaluation_seed=evaluation_seed,
        bootstrap_resamples=100,
    )
    elapsed = time.perf_counter() - start
    decisions = int(result.metrics["decision_count"])
    return {
        "decisions": decisions,
        "seconds": elapsed,
        "decisions_per_second": decisions / elapsed if elapsed else 0.0,
        "milliseconds_per_decision": 1000.0 * elapsed / decisions if decisions else 0.0,
    }


@dataclass
class A1Report:
    """The complete A1 record: both views, every seed, and the capacity note."""

    evaluation_seed: int
    paired_deal_count: int
    seeds: list[int] = field(default_factory=list)
    parameter_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    per_seed: list[dict[str, Any]] = field(default_factory=list)
    pooled: dict[str, Any] = field(default_factory=dict)
    head_to_head_result: dict[str, Any] = field(default_factory=dict)
    throughput: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    seed_stability: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_seed": self.evaluation_seed,
            "paired_deal_count": self.paired_deal_count,
            "seeds": self.seeds,
            "parameter_counts": self.parameter_counts,
            "per_seed": self.per_seed,
            "pooled": self.pooled,
            "head_to_head": self.head_to_head_result,
            "throughput": self.throughput,
            "latency": self.latency,
            "seed_stability": self.seed_stability,
            "notes": self.notes,
        }


def summarise_seed_stability(values: list[float]) -> dict[str, float | None]:
    """Return spread across independent training seeds.

    Reported because a single lucky run is not evidence. With three seeds the
    spread is a weak estimate and is labelled as such rather than dressed up as
    a confidence interval.
    """
    if not values:
        return {"mean": None, "min": None, "max": None, "spread": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "spread": float(array.max() - array.min()),
        "standard_deviation": (sample_standard_deviation(array) if array.size > 1 else 0.0),
    }


def paired_outcome_scores(outcomes: list[PairedOutcome]) -> np.ndarray:
    """Return the per-pair score differences of an evaluation, in order."""
    return np.asarray([outcome.paired_score_difference for outcome in outcomes], dtype=np.float64)
