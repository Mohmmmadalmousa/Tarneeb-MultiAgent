"""Independent development duplicate deals; never maps into held-out seed space."""

from __future__ import annotations

from typing import Any

from tarneeb.agents.base import Agent
from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.auction_repair.data import validate_seeds
from tarneeb.auction_repair.diagnostics import summarize_learner_auction_metrics
from tarneeb.core.events import Phase
from tarneeb.env.public_view import PublicView
from tarneeb.evaluation.arena import Arena
from tarneeb.evaluation.metrics import PairedOutcome, aggregate_metrics
from tarneeb.evaluation.stats import bootstrap_mean_confidence_interval
from tarneeb.models.recurrent import RecurrentActorNetwork


class ObservedLearner(Agent):
    def __init__(self, actor: RecurrentActorNetwork) -> None:
        self.name = "phase11b_learner"
        self.agent = RecurrentNeuralAgent(actor, deterministic=True)
        self.records: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.agent.reset()

    def act(self, view: PublicView) -> int:
        action = self.agent.act(view)
        if view.phase is Phase.BID:
            self.records.append(
                {
                    "action": action,
                    "current_high_bid": view.current_high_bid,
                    "is_learner": True,
                    "was_auction_decision": True,
                }
            )
        return action


def evaluate_development(
    actor: RecurrentActorNetwork,
    seeds: range,
    anchor: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    validate_seeds(seeds)
    if bootstrap_seed == 20250910:
        raise ValueError("Forbidden development seed")
    if anchor not in ("random", "basic", "strong"):
        raise ValueError("Unknown anchor")
    learner = ObservedLearner(actor)
    arena = Arena(replay_sample_rate=0.0)
    pairs: list[PairedOutcome] = []
    for index, seed in enumerate(seeds):
        # Independent anchor RNG for each duplicate pair; before/after share it.
        opponent: Agent
        if anchor == "random":
            opponent = RandomAgent(seed=seed + 100_000)
        elif anchor == "basic":
            opponent = BasicHeuristicAgent()
        else:
            opponent = StrongHeuristicAgent()
        pairs.append(
            arena.play_duplicate_pair(
                learner,
                opponent,
                match_seed=seed,
                dealer_seat=index % 4,
                verify_replay=(index % 20 == 0),
            )
        )
    contracts = [
        run
        for pair in pairs
        for run in (pair.first_run, pair.second_run)
        if run.declarer_seat is not None and run.declarer_seat % 2 == run.policy_a_team
    ]
    by_bid = {}
    for bid in range(7, 14):
        rows = [r for r in contracts if r.winning_bid == bid]
        by_bid[str(bid)] = {
            "contracts": len(rows),
            "made": sum(bool(r.contract_made) for r in rows),
            "success_rate": sum(bool(r.contract_made) for r in rows) / len(rows) if rows else None,
        }
    return {
        "anchor": anchor,
        "seed_start": seeds.start,
        "seed_stop": seeds.stop,
        "metrics": aggregate_metrics(
            pairs,
            bootstrap_seed,
            bootstrap_resamples=2000,
            illegal_action_count=arena.illegal_action_count,
        ),
        "learner_auction": summarize_learner_auction_metrics(learner.records),
        "learner_action_bid_counts": {
            str(b): sum(r["action"] == b + 45 for r in learner.records) for b in range(7, 14)
        },
        "learner_contracts": len(contracts),
        "learner_declaration_opportunities": 2 * len(pairs),
        "learner_declaration_rate": len(contracts) / (2 * len(pairs)),
        "learner_contract_success_rate": sum(bool(r.contract_made) for r in contracts)
        / len(contracts)
        if contracts
        else None,
        "learner_contract_by_bid": by_bid,
        "paired_scores": [pair.paired_score_difference for pair in pairs],
    }


def compare_development(
    original: RecurrentActorNetwork,
    repaired: RecurrentActorNetwork,
    seeds: range,
) -> dict[str, Any]:
    result = {}
    for index, anchor in enumerate(("random", "basic", "strong")):
        bootstrap_seed = 8_300_010 + index
        before = evaluate_development(original, seeds, anchor, bootstrap_seed)
        after = evaluate_development(repaired, seeds, anchor, bootstrap_seed)
        changes = [
            a - b for a, b in zip(after["paired_scores"], before["paired_scores"], strict=True)
        ]
        result[anchor] = {
            "before": before,
            "after": after,
            "paired_change_95ci": bootstrap_mean_confidence_interval(
                changes, bootstrap_seed, resamples=2000
            ).as_dict(),
        }
    return result
