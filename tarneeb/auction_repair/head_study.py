"""Additive fixed-budget head study. No environment or dataset generation."""

from __future__ import annotations

import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from tarneeb.auction_repair.learning import AuctionRows, assert_preserved
from tarneeb.models.masking import MaskedCategorical


def role(meta):
    if meta["current_high_bid"] is None:
        return "opening"
    return "partner" if meta["high_bidder_is_partner"] else "opponent"


def weights_for(examples):
    """Same retained indices; six equal strata, then equal level/cell masses."""
    keep = np.flatnonzero(examples.legal.sum(axis=1) > 1)
    groups = defaultdict(lambda: defaultdict(list))
    for j, i in enumerate(keep):
        label = int(examples.labels[i])
        meta = examples.metadata[i]
        kind = "PASS" if label == 59 else "BID"
        level = str(meta["current_high_bid"]) if kind == "PASS" else str(label - 45)
        groups[f"{role(meta)}:{kind}"][level].append(j)
    if set(groups) != {
        f"{r}:{k}" for r in ("opening", "partner", "opponent") for k in ("PASS", "BID")
    }:
        raise ValueError("All six predeclared strata must be nonempty")
    balanced = np.zeros(len(keep), dtype=np.float64)
    cells = []
    for group, levels in sorted(groups.items()):
        for level, indices in sorted(levels.items()):
            weight = 1 / (6 * len(levels) * len(indices))
            balanced[indices] = weight
            cells.append(
                {
                    "stratum": group,
                    "level": level,
                    "count": len(indices),
                    "weight_each": weight,
                    "mass": weight * len(indices),
                    "original_indices": keep[indices].tolist(),
                }
            )
    assert np.isclose(balanced.sum(), 1) and np.all(balanced > 0)
    return keep, {"uniform": np.full(len(keep), 1 / len(keep)), "balanced": balanced}, cells


def losses(raw, legal, labels):
    return -MaskedCategorical.from_logits(raw, legal).log_probability_of(labels)


def export_rows(original, rows):
    candidate = deepcopy(original).cpu().eval().requires_grad_(False)
    with torch.no_grad():
        candidate.policy_head.weight[52:60].copy_(rows.auction_weight)
        candidate.policy_head.bias[52:60].copy_(rows.auction_bias)
    assert_preserved(original, candidate)
    return candidate


@dataclass(frozen=True)
class Settings:
    steps: int = 512
    lr: float = 3e-4
    clip: float = 0.5
    drift: float = 0.05
    cpu_seconds: float = 240


DEFAULT_SETTINGS = Settings()


def optimize_head(
    original,
    features,
    legal,
    labels,
    weights,
    check,
    record,
    overall_deadline,
    settings=DEFAULT_SETTINGS,
    cpu_clock=time.process_time,
    wall_clock=time.monotonic,
):
    """Return last within-bound rows and explicit completion status.

    Drift is checked after each Adam proposal before accepting it. No clipping
    or projection of weights. A rejected proposal ends this arm permanently.
    Last-safe rows are diagnostic progress, never a selected final checkpoint.
    """
    if features.requires_grad or weights.requires_grad or original.training:
        raise ValueError("Use detached frozen features and an eval-mode actor")
    rows = AuctionRows(original)
    initial = {k: v.detach().clone() for k, v in rows.state_dict().items()}
    safe = deepcopy(rows).requires_grad_(False)
    optimizer = torch.optim.Adam(
        rows.parameters(), lr=settings.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    start_cpu, start_wall = cpu_clock(), wall_clock()
    status, accepted, attempted, rejected_drift = "complete", 0, 0, None
    with torch.no_grad():
        initial_loss = float((losses(rows(features), legal, labels) * weights).sum())
    check(safe)
    for step in range(1, settings.steps + 1):
        if wall_clock() >= overall_deadline:
            status = "stopped_overall_wall_limit"
            break
        if cpu_clock() - start_cpu >= settings.cpu_seconds:
            status = "stopped_arm_cpu_limit"
            break
        optimizer.zero_grad(set_to_none=True)
        loss = (losses(rows(features), legal, labels) * weights).sum()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite training loss; stop entire study")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            rows.parameters(), settings.clip, error_if_nonfinite=True
        )
        optimizer.step()
        attempted = step
        drift = max(float((v - initial[k]).abs().max()) for k, v in rows.state_dict().items())
        if not all(bool(torch.isfinite(v).all()) for v in rows.state_dict().values()):
            raise ValueError("Nonfinite parameter; stop entire study")
        entry = {
            "step": step,
            "loss_before_step": float(loss.detach()),
            "unclipped_gradient_norm": float(norm),
            "proposal_drift": drift,
            "cpu_seconds": cpu_clock() - start_cpu,
            "wall_seconds": wall_clock() - start_wall,
        }
        if drift > settings.drift:
            entry["accepted"] = False
            record(entry)
            rejected_drift = drift
            status = "stopped_drift_limit"
            break
        safe.load_state_dict(rows.state_dict())
        accepted = step
        entry["accepted"] = True
        record(entry)
        if step % 32 == 0:
            check(safe)
        if cpu_clock() - start_cpu > settings.cpu_seconds:
            status = "stopped_arm_cpu_limit"
            break
        if wall_clock() > overall_deadline:
            status = "stopped_overall_wall_limit"
            break
    check(safe)
    final_drift = max(float((v - initial[k]).abs().max()) for k, v in safe.state_dict().items())
    with torch.no_grad():
        final_loss = float((losses(safe(features), legal, labels) * weights).sum())
    return safe, {
        "status": status,
        "accepted_updates": accepted,
        "attempted_updates": attempted,
        "initial_objective": initial_loss,
        "final_or_partial_objective": final_loss,
        "maximum_accepted_drift": final_drift,
        "rejected_proposal_drift": rejected_drift,
        "cpu_seconds": cpu_clock() - start_cpu,
        "wall_seconds": wall_clock() - start_wall,
        "eligible_final_checkpoint": status == "complete" and accepted == settings.steps,
    }


@torch.no_grad()
def describe(actor, features, examples, keys=None):
    raw = actor.policy_head(features)
    legal = torch.from_numpy(examples.legal)
    labels = torch.from_numpy(examples.labels)
    dist = MaskedCategorical.from_logits(raw, legal)
    probs = dist.probabilities.numpy()
    actions = dist.most_probable_action().numpy()
    nll = losses(raw, legal, labels).numpy()
    assert float(dist.probabilities[~legal].max()) == 0
    rows = []
    for i, meta in enumerate(examples.metadata):
        bids = np.flatnonzero(examples.legal[i, 52:59]) + 52
        row = {
            **meta,
            "role": role(meta),
            "label": int(examples.labels[i]),
            "action": int(actions[i]),
            "nll": float(nll[i]),
            "pass_probability": float(probs[i, 59]),
            "legal_logits": {str(a): float(raw[i, a]) for a in [59, *bids]},
            "legal_probabilities": {str(a): float(probs[i, a]) for a in [59, *bids]},
            "pass_minus_best_bid": float(raw[i, 59] - raw[i, bids].max()) if len(bids) else None,
            "nontrivial": bool(len(bids)),
        }
        if keys is not None:
            row["id"] = keys[i]
        rows.append(row)

    def summary(indices):
        if not indices:
            return {"count": 0}
        a, y = actions[indices], examples.labels[indices]
        passed, raised = y == 59, y != 59
        return {
            "count": len(indices),
            "nll": float(nll[indices].mean()),
            "teacher_accuracy": float((a == y).mean()),
            "bid_count": int((a != 59).sum()),
            "teacher_pass_count": int(passed.sum()),
            "teacher_bid_count": int(raised.sum()),
            "bids_on_teacher_pass": int(((a != 59) & passed).sum()),
            "passes_on_teacher_bid": int(((a == 59) & raised).sum()),
            "exact_teacher_bids": int(((a == y) & raised).sum()),
            "overbid_teacher_count": int(((a > y) & raised & (a != 59)).sum()),
            "bid_11_to_13_count": int(((a >= 56) & (a <= 58)).sum()),
            "action_counts": {str(k): int((a == k).sum()) for k in range(52, 60)},
        }

    nt = [i for i, r in enumerate(rows) if r["nontrivial"]]
    groups = defaultdict(list)
    for i in nt:
        r = rows[i]
        groups[f"{r['role']}|bid={r['current_high_bid']}|strength={r['strength']}"].append(i)
    return {
        "all": summary(list(range(len(rows)))),
        "nontrivial": summary(nt),
        "roles": {
            role: summary([i for i in nt if rows[i]["role"] == role])
            for role in ("opening", "partner", "opponent")
        },
        "strata": {k: summary(v) for k, v in sorted(groups.items())},
        "illegal_probability": 0.0,
        "rows": rows,
    }
