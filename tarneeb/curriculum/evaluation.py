"""Held-out evaluation for curriculum stages, on the frozen Phase 5 arena.

No new evaluation method is invented here. Every deal is played by
``Arena.play_deal``, every seed comes from ``evaluation_match_seed`` and therefore
from ``HELD_OUT_EVALUATION_SEEDS``, and every number is produced by the frozen
``aggregate_metrics`` -- bootstrap confidence interval, Wilson win-rate interval,
contract and bidding metrics, replay verification.

The one thing this module adds is the ability to install a **scripted auction**
before each deal, which the frozen ``Arena.evaluate`` loop has no hook for. The
loop below is the same duplicate-pair protocol: the same dealt cards played twice
with the partnerships swapped, the dealer rotating with the pair index. Because
the script is a pure function of the deal, both orientations get an identical
auction, so pairing still cancels card luck and the comparison isolates play.

Stage 6 and the A7 conditions use a learned auction and therefore call the frozen
``Arena.evaluate`` directly, with nothing added.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tarneeb.agents.base import Agent
from tarneeb.core.cards import NUM_SEATS, NUM_SUITS, SUIT_NAMES
from tarneeb.core.rules import RulesConfig
from tarneeb.curriculum.agents import ScriptedAuctionAgent
from tarneeb.curriculum.auction_script import build_auction_script
from tarneeb.curriculum.stages import StageSpec
from tarneeb.evaluation.arena import (
    HELD_OUT_EVALUATION_SEEDS,
    TRAINING_SEEDS,
    Arena,
    build_seat_assignment,
    evaluation_match_seed,
)
from tarneeb.evaluation.metrics import EvaluationResult, PairedOutcome, aggregate_metrics
from tarneeb.evaluation.stats import wilson_score_interval

DEFAULT_REPLAY_SAMPLE_RATE = 0.05


@dataclass(frozen=True, slots=True)
class StageEvaluation:
    """One held-out evaluation of a stage checkpoint."""

    stage_name: str
    policy_a_name: str
    policy_b_name: str
    evaluation_seed: int
    paired_deal_count: int
    metrics: dict[str, Any]
    trump_suit_breakdown: dict[str, dict[str, float]]

    @property
    def mean_paired_score_difference(self) -> float:
        return float(self.metrics["mean_paired_score_difference"])

    @property
    def win_rate(self) -> float | None:
        value = self.metrics["win_rate"]
        return None if value is None else float(value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "policy_a_name": self.policy_a_name,
            "policy_b_name": self.policy_b_name,
            "evaluation_seed": self.evaluation_seed,
            "paired_deal_count": self.paired_deal_count,
            "trump_suit_breakdown": self.trump_suit_breakdown,
            **self.metrics,
        }


def evaluation_seeds_are_isolated() -> bool:
    """True when no held-out evaluation seed can ever be a training seed."""
    return HELD_OUT_EVALUATION_SEEDS.start >= TRAINING_SEEDS.stop


def evaluate_stage(
    stage: StageSpec,
    policy_a: Agent,
    policy_b: Agent,
    paired_deal_count: int,
    evaluation_seed: int,
    bootstrap_resamples: int = 10_000,
    replay_sample_rate: float = DEFAULT_REPLAY_SAMPLE_RATE,
    rules: RulesConfig | None = None,
) -> StageEvaluation:
    """Run a held-out duplicate-deal evaluation under one stage's conditions.

    A positive mean paired score difference means ``policy_a`` outscored
    ``policy_b``, in points per deal -- the frozen Phase 5 sign convention.
    """
    if paired_deal_count < 1:
        raise ValueError(f"paired_deal_count must be at least 1, got {paired_deal_count}")

    resolved_rules = rules if rules is not None else RulesConfig()
    arena = Arena(rules=resolved_rules, replay_sample_rate=replay_sample_rate)
    replay_every = round(1 / replay_sample_rate) if replay_sample_rate > 0 else 0

    seated_a: Agent = policy_a
    seated_b: Agent = policy_b
    if stage.uses_scripted_auction:
        seated_a = ScriptedAuctionAgent(policy_a, name=policy_a.name)
        seated_b = ScriptedAuctionAgent(policy_b, name=policy_b.name)

    paired_outcomes: list[PairedOutcome] = []
    trump_by_pair: list[int | None] = []
    start_time = time.perf_counter()

    for pair_index in range(paired_deal_count):
        match_seed = evaluation_match_seed(evaluation_seed, pair_index)
        dealer_seat = pair_index % NUM_SEATS
        verify_replay = replay_every > 0 and pair_index % replay_every == 0

        scripted_trump: int | None = None
        if stage.uses_scripted_auction:
            script = build_auction_script(
                stage=stage, match_seed=match_seed, deal_index=0, dealer_seat=dealer_seat
            )
            scripted_trump = script.trump_suit
            assert isinstance(seated_a, ScriptedAuctionAgent)
            assert isinstance(seated_b, ScriptedAuctionAgent)
            seated_a.set_script(script)
            seated_b.set_script(script)

        first_run = arena.play_deal(
            build_seat_assignment(seated_a, seated_b, policy_a_team=0),
            match_seed=match_seed,
            dealer_seat=dealer_seat,
            deal_index=0,
            verify_replay=verify_replay,
        )
        second_run = arena.play_deal(
            build_seat_assignment(seated_a, seated_b, policy_a_team=1),
            match_seed=match_seed,
            dealer_seat=dealer_seat,
            deal_index=0,
            verify_replay=verify_replay,
        )
        paired_outcomes.append(
            PairedOutcome(
                match_seed=match_seed,
                deal_index=0,
                first_run=first_run,
                second_run=second_run,
            )
        )
        trump_by_pair.append(scripted_trump)

    runtime_seconds = time.perf_counter() - start_time
    metrics = aggregate_metrics(
        paired_outcomes,
        bootstrap_seed=evaluation_seed,
        bootstrap_resamples=bootstrap_resamples,
        illegal_action_count=arena.illegal_action_count,
        runtime_seconds=runtime_seconds,
    )

    return StageEvaluation(
        stage_name=stage.name,
        policy_a_name=policy_a.name,
        policy_b_name=policy_b.name,
        evaluation_seed=evaluation_seed,
        paired_deal_count=paired_deal_count,
        metrics=metrics,
        trump_suit_breakdown=summarise_by_trump_suit(paired_outcomes, trump_by_pair),
    )


def summarise_by_trump_suit(
    paired_outcomes: list[PairedOutcome], trump_by_pair: list[int | None]
) -> dict[str, dict[str, float]]:
    """Split the paired result by the deal's trump suit.

    Stage 4 asks whether any one suit is systematically worse, which a single
    aggregate number would hide.
    """
    breakdown: dict[str, dict[str, float]] = {}
    for suit in range(NUM_SUITS):
        differences = [
            pair.paired_score_difference
            for pair, trump_suit in zip(paired_outcomes, trump_by_pair, strict=True)
            if trump_suit == suit
        ]
        if not differences:
            continue
        wins = sum(1 for difference in differences if difference > 0)
        decisive = sum(1 for difference in differences if difference != 0)
        interval = wilson_score_interval(wins, decisive) if decisive else None
        breakdown[SUIT_NAMES[suit]] = {
            "paired_deal_count": float(len(differences)),
            "mean_paired_score_difference": float(sum(differences) / len(differences)),
            "win_rate": float(wins / decisive) if decisive else 0.0,
            "win_rate_lower_bound": float(interval.lower_bound) if interval else 0.0,
            "win_rate_upper_bound": float(interval.upper_bound) if interval else 0.0,
        }
    return breakdown


def evaluate_full_game(
    policy_a: Agent,
    policy_b: Agent,
    paired_deal_count: int,
    evaluation_seed: int,
    bootstrap_resamples: int = 10_000,
    rules: RulesConfig | None = None,
) -> EvaluationResult:
    """Evaluate the complete game through the frozen arena, unchanged.

    Used by stage 6 and by every A7 condition, where the auction is the policy's
    own and nothing may be scripted.
    """
    arena = Arena(rules=rules if rules is not None else RulesConfig())
    return arena.evaluate(
        policy_a=policy_a,
        policy_b=policy_b,
        paired_deal_count=paired_deal_count,
        evaluation_seed=evaluation_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
