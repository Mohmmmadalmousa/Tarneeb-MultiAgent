"""Per-deal records and the aggregated metrics of an evaluation run.

A :class:`DealOutcome` is what the arena records for one played deal; a
:class:`PairedOutcome` joins the two orientations of one duplicate pair; and
:func:`aggregate_metrics` turns a list of pairs into the numbers reported.

Sign convention, used everywhere: a positive paired score difference means
**policy A outscored policy B**, in points per deal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tarneeb.core.state import team_of_seat
from tarneeb.evaluation.stats import (
    ConfidenceInterval,
    bootstrap_mean_confidence_interval,
    sample_standard_deviation,
    wilson_score_interval,
)

ROLE_DECLARER = "declarer"
ROLE_PARTNER = "partner"
ROLE_DEFENDER = "defender"


@dataclass(frozen=True, slots=True)
class DealOutcome:
    """What one played deal produced, from the arena's point of view."""

    match_seed: int
    deal_index: int
    dealer_seat: int
    policy_a_team: int
    """Which team (0 or 1) policy A controlled in this run."""

    score_change_by_team: tuple[int, int]
    declarer_seat: int | None
    winning_bid: int | None
    tricks_won_by_team: tuple[int, int]
    contract_made: bool | None
    """``None`` when the deal was passed out with no contract."""

    passed_out: bool
    decision_count: int
    bid_action_count: int
    pass_action_count: int
    replay_verified: bool | None = None
    """``None`` when this deal was not sampled for replay verification."""

    @property
    def policy_a_score(self) -> int:
        return self.score_change_by_team[self.policy_a_team]

    @property
    def policy_b_score(self) -> int:
        return self.score_change_by_team[1 - self.policy_a_team]

    def role_of_seat(self, seat: int) -> str:
        """Return declarer / partner / defender for ``seat``."""
        if self.declarer_seat is None:
            return ROLE_DEFENDER
        if seat == self.declarer_seat:
            return ROLE_DECLARER
        if team_of_seat(seat) == team_of_seat(self.declarer_seat):
            return ROLE_PARTNER
        return ROLE_DEFENDER

    def policy_a_role(self) -> str:
        """The role policy A held: declarer if it declared, else partner/defender."""
        if self.declarer_seat is None:
            return ROLE_DEFENDER
        if team_of_seat(self.declarer_seat) == self.policy_a_team:
            return ROLE_DECLARER
        return ROLE_DEFENDER


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    """One duplicate pair: the same dealt cards played in both orientations."""

    match_seed: int
    deal_index: int
    first_run: DealOutcome
    """Policy A on one team."""

    second_run: DealOutcome
    """Policy A on the other team, same cards."""

    @property
    def paired_score_difference(self) -> float:
        """Mean of policy A's margin over the two orientations.

        Averaging both seatings cancels the advantage of holding the better
        cards, which is the whole point of duplicate evaluation.
        """
        first_margin = self.first_run.policy_a_score - self.first_run.policy_b_score
        second_margin = self.second_run.policy_a_score - self.second_run.policy_b_score
        return (first_margin + second_margin) / 2.0

    @property
    def policy_a_wins(self) -> bool:
        return self.paired_score_difference > 0.0

    @property
    def is_tie(self) -> bool:
        return self.paired_score_difference == 0.0


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_metrics(
    paired_outcomes: list[PairedOutcome],
    bootstrap_seed: int,
    bootstrap_resamples: int = 10_000,
    illegal_action_count: int = 0,
    decision_count: int = 0,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Return the full metric dictionary for an evaluation run."""
    if not paired_outcomes:
        raise ValueError("cannot aggregate an empty evaluation")

    differences = np.array(
        [pair.paired_score_difference for pair in paired_outcomes], dtype=np.float64
    )
    paired_deal_count = len(paired_outcomes)

    confidence_interval = bootstrap_mean_confidence_interval(
        differences, seed=bootstrap_seed, resamples=bootstrap_resamples
    )
    sigma_d = sample_standard_deviation(differences) if paired_deal_count >= 2 else None

    decisive_pairs = [pair for pair in paired_outcomes if not pair.is_tie]
    policy_a_wins = sum(1 for pair in decisive_pairs if pair.policy_a_wins)
    win_rate_interval: ConfidenceInterval | None = (
        wilson_score_interval(policy_a_wins, len(decisive_pairs)) if decisive_pairs else None
    )

    all_runs = [run for pair in paired_outcomes for run in (pair.first_run, pair.second_run)]

    contracts = [run for run in all_runs if not run.passed_out and run.winning_bid is not None]
    contract_success_rate = (
        sum(1 for run in contracts if run.contract_made) / len(contracts) if contracts else None
    )
    mean_bid = _mean([float(run.winning_bid) for run in contracts])  # type: ignore[arg-type]
    passed_out_rate = sum(1 for run in all_runs if run.passed_out) / len(all_runs)

    total_bid_actions = sum(run.bid_action_count for run in all_runs)
    total_pass_actions = sum(run.pass_action_count for run in all_runs)
    total_auction_actions = total_bid_actions + total_pass_actions
    pass_action_rate = total_pass_actions / total_auction_actions if total_auction_actions else None

    score_by_bid_level: dict[int, dict[str, float]] = {}
    grouped_by_bid: dict[int, list[DealOutcome]] = defaultdict(list)
    for run in contracts:
        grouped_by_bid[int(run.winning_bid)].append(run)  # type: ignore[arg-type]
    for bid_level, runs in sorted(grouped_by_bid.items()):
        declaring_scores = [
            float(run.score_change_by_team[team_of_seat(run.declarer_seat)])  # type: ignore[arg-type]
            for run in runs
        ]
        score_by_bid_level[bid_level] = {
            "count": len(runs),
            "contract_success_rate": sum(1 for run in runs if run.contract_made) / len(runs),
            "mean_declaring_score_change": float(np.mean(declaring_scores)),
        }

    role_scores: dict[str, list[float]] = {
        ROLE_DECLARER: [],
        ROLE_PARTNER: [],
        ROLE_DEFENDER: [],
    }
    for run in all_runs:
        for seat in range(4):
            if team_of_seat(seat) != run.policy_a_team:
                continue
            role_scores[run.role_of_seat(seat)].append(
                float(run.score_change_by_team[run.policy_a_team])
            )
    role_split = {role: _mean(values) for role, values in role_scores.items()}

    mean_tricks_declaring = _mean(
        [
            float(run.tricks_won_by_team[team_of_seat(run.declarer_seat)])  # type: ignore[arg-type]
            for run in contracts
        ]
    )

    verified_runs = [run for run in all_runs if run.replay_verified is not None]
    replay_integrity_rate = (
        sum(1 for run in verified_runs if run.replay_verified) / len(verified_runs)
        if verified_runs
        else None
    )

    total_decisions = decision_count or sum(run.decision_count for run in all_runs)

    metrics: dict[str, Any] = {
        "paired_deal_count": paired_deal_count,
        "played_deal_count": len(all_runs),
        "mean_paired_score_difference": confidence_interval.mean,
        "paired_score_difference_ci": confidence_interval.as_dict(),
        "sigma_d": sigma_d,
        "win_rate": win_rate_interval.mean if win_rate_interval else None,
        "win_rate_wilson_interval": (win_rate_interval.as_dict() if win_rate_interval else None),
        "decisive_pair_count": len(decisive_pairs),
        "tie_count": paired_deal_count - len(decisive_pairs),
        "contract_success_rate": contract_success_rate,
        "mean_bid": mean_bid,
        "passed_out_rate": passed_out_rate,
        "pass_action_rate": pass_action_rate,
        "score_by_bid_level": score_by_bid_level,
        "role_split_mean_score": role_split,
        "mean_tricks_declaring_team": mean_tricks_declaring,
        "illegal_action_rate": illegal_action_count / total_decisions if total_decisions else 0.0,
        "no_trump_action_rate": 0.0,
        "replay_integrity_rate": replay_integrity_rate,
        "replay_verified_count": len(verified_runs),
        "decision_count": total_decisions,
    }

    if runtime_seconds is not None and runtime_seconds > 0:
        metrics["runtime_seconds"] = runtime_seconds
        metrics["decisions_per_second"] = total_decisions / runtime_seconds
        metrics["deals_per_second"] = len(all_runs) / runtime_seconds
        metrics["paired_deals_per_second"] = paired_deal_count / runtime_seconds

    return metrics


@dataclass
class EvaluationResult:
    """A complete evaluation: what was compared, and what came out."""

    policy_a_name: str
    policy_b_name: str
    evaluation_seed: int
    metrics: dict[str, Any]
    paired_outcomes: list[PairedOutcome] = field(default_factory=list, repr=False)

    @property
    def strong_evidence_a_beats_b(self) -> bool:
        """True when the 95% paired CI lies entirely above zero."""
        interval = self.metrics["paired_score_difference_ci"]
        return bool(interval["lower_bound"] > 0.0)
