"""Bounded recurrent auction diagnostics; no optimizer is constructed."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from tarneeb.auction_repair.data import PublicExamples, frozen_features
from tarneeb.auction_repair.diagnostics import summarize_learner_auction_metrics
from tarneeb.env.vec import VecTarneeb
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.recurrent import RecurrentActorNetwork
from tarneeb.training.recurrent_rollout import learner_team_by_environment


def training_seating_diagnostic() -> dict[str, Any]:
    """Read-only exposure check of the frozen principal collector's defaults."""
    environments = VecTarneeb(16, root_seed=8_240_000)
    learner_teams = learner_team_by_environment(16)
    rows = []
    for reset in range(2):
        environments.reset()
        for index, environment in enumerate(environments.environments):
            seat = environment.acting_seat
            assert seat is not None
            rows.append(
                {
                    "reset": reset,
                    "environment": index,
                    "dealer": environment.dealer_seat,
                    "first_bidder": seat,
                    "learner_team": int(learner_teams[index]),
                    "learner_opens_first": seat % 2 == int(learner_teams[index]),
                }
            )
    return {
        "environment_count": 16,
        "reset_count": 2,
        "learner_first_bidder_count": sum(r["learner_opens_first"] for r in rows),
        "first_bidder_opportunities": len(rows),
        "rows": rows,
        "interpretation": "Dealer i%4 and learner team i%2 make the first bidder an opponent in every environment. Learner open-auction decisions require an opponent PASS. This is an exposure confound, not proof of reward-optimal bidding or a complete causal explanation.",
    }


@torch.no_grad()
def measure_actor(actor: RecurrentActorNetwork, examples: PublicExamples) -> dict[str, Any]:
    features, _ = frozen_features(actor, examples)
    raw = actor.policy_head(features)
    legal = torch.from_numpy(examples.legal)
    distribution = MaskedCategorical.from_logits(raw, legal)
    probabilities = distribution.probabilities.numpy()
    actions = distribution.most_probable_action().numpy()
    rows = []
    for i, meta in enumerate(examples.metadata):
        legal_bids = examples.legal[i, 52:59]
        margin = (
            float(raw[i, 59] - raw[i, 52:59][torch.from_numpy(legal_bids)].max())
            if legal_bids.any()
            else None
        )
        rows.append(
            {
                **meta,
                "is_learner": True,
                "was_auction_decision": True,
                "action": int(actions[i]),
                "teacher_action": int(examples.labels[i]),
                "pass_probability": float(probabilities[i, 59]),
                "legal_bid_probabilities": {
                    str(7 + j): float(probabilities[i, 52 + j]) for j in range(7) if legal_bids[j]
                },
                "auction_logits": raw[i, 52:60].tolist(),
                "pass_minus_best_legal_bid_logit": margin,
            }
        )

    def summarize(indices: list[int]) -> dict[str, Any]:
        records = [rows[i] for i in indices]
        if not records:
            return {"count": 0}
        margins = [
            r["pass_minus_best_legal_bid_logit"]
            for r in records
            if r["pass_minus_best_legal_bid_logit"] is not None
        ]
        return {
            "count": len(indices),
            **summarize_learner_auction_metrics(records),
            "mean_pass_probability": float(probabilities[indices, 59].mean()),
            "mean_bid_probabilities": {
                str(b): float(probabilities[indices, b + 45].mean()) for b in range(7, 14)
            },
            "mean_pass_minus_best_bid_logit": float(np.mean(margins)) if margins else None,
            "teacher_bid_rate": float(np.mean(examples.labels[indices] != 59)),
        }

    buckets: dict[str, Any] = {}
    for field in (
        "learner_team",
        "current_high_bid",
        "strength",
        "best_suit",
        "high_bidder_is_partner",
        "case_family",
    ):
        values = sorted({str(row[field]) for row in rows})
        buckets[field] = {
            value: summarize([i for i, row in enumerate(rows) if str(row[field]) == value])
            for value in values
        }
    estimates = np.array([r["estimate"] for r in rows])
    bid_probability = 1 - probabilities[:, 59]
    correlation = (
        float(np.corrcoef(estimates, bid_probability)[0, 1])
        if bid_probability.std() > 0 and estimates.std() > 0
        else None
    )
    head = actor.policy_head
    return {
        "overall": summarize(list(range(len(rows)))),
        "buckets": buckets,
        "strength_bid_probability_correlation": correlation,
        "auction_head_bias": head.bias[52:60].tolist(),
        "auction_head_row_norms": head.weight[52:60].norm(dim=1).tolist(),
        "max_illegal_probability": float(distribution.probabilities[~legal].max()),
        "rows": rows,
    }
