"""One read-only gradient check; no optimizer construction, step, or output writes.

Checks a fresh pilot's FIRST Adam direction only (zero initial moments), using
the existing learning rate, epsilon and clipping. This is not a training run or
a guarantee about later Adam moments or gameplay. Historical plans stay frozen.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_phase11b_retention_pilot as pilot

from tarneeb.auction_repair.retention_objective import ObjectiveWeights, loss_terms
from tarneeb.auction_value_recovery import checkpoint_read_locks
from tarneeb.core.engine import TarneebEngine


def first_adam_displacement(
    gradients: tuple[torch.Tensor, ...],
    *,
    lr: float = 3e-5,
    clip: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, ...]:
    """Pure zero-moment Adam formula, with PyTorch's global norm clipping.

    No parameter or optimizer state is accepted or mutated. No weight decay.
    Bias correction cancels beta factors on step one, including (.9, .999).
    """
    if not gradients or min(lr, clip, eps) <= 0:
        raise ValueError("Invalid preview arguments")
    if not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ValueError("Nonfinite gradient")
    norm = torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in gradients]))
    scale = (clip / (norm + 1e-6)).clamp(max=1.0)
    clipped = tuple(g.detach() * scale for g in gradients)
    return tuple(-lr * g / (g.abs() + eps) for g in clipped)


def main() -> None:
    torch.set_num_threads(4)
    config = ObjectiveWeights()
    paths = [pilot.inputs.SOURCE, pilot.inputs.TRAIN, pilot.inputs.DEV, pilot.FIXTURE]
    hashes = {str(p): pilot.sha256(p) for p in paths}
    with (
        checkpoint_read_locks(),
        patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer updates")),
        patch.object(TarneebEngine, "deal", side_effect=AssertionError("No new deals")),
    ):
        c = pilot.load_context()  # Authenticates saved evidence; public-prefix replay only.
        with patch.object(TarneebEngine, "step", side_effect=AssertionError("No gameplay")):
            rows = c["rows"]
            params = tuple(rows.parameters())
            initial = tuple(p.detach().clone() for p in params)
            x = rows(c["features"]["train"])
            terms = loss_terms(
                x,
                c["baseline"]["train"],
                c["masks"]["train"],
                c["labels"]["train"],
                c["evidence"],
                row_ids=torch.from_numpy(c["ids"]["train"]),
                row_fingerprints=c["train_fingerprints"],
                config=config,
            )
            assert [
                int(terms[k])
                for k in (
                    "bid_pass_pair_count",
                    "bid_pass_case_count",
                    "bid_bid_pair_count",
                    "bid_bid_case_count",
                    "supported_pair_count",
                    "heuristic_excluded_count",
                )
            ] == [28, 8, 2, 2, 30, 8]
            gradients = torch.autograd.grad(terms["total"], params, retain_graph=True)
            component_norms = {}
            for key in ("heuristic_ce", "bid_pass", "bid_bid", "bid_retention", "pass_retention"):
                gs = torch.autograd.grad(
                    getattr(config, key) * terms[key], params, retain_graph=True
                )
                component_norms[key] = float(torch.cat([g.flatten() for g in gs]).norm())
            delta = first_adam_displacement(gradients)
            # Account for float32 parameter storage without assigning any parameters.
            rounded = tuple(
                (p.detach() + d).double() - p.detach().double()
                for p, d in zip(params, delta, strict=True)
            )
            positions = {int(row): local for local, row in enumerate(c["ids"]["train"])}
            focal, blocked, improved = {}, [], 0
            positive = 0
            for pair in c["evidence"].pairs:
                local = positions[pair.row]
                f = c["features"]["train"][local].double()
                a, b = pair.preferred - 52, pair.other - 52
                margin = float((x[local, pair.preferred] - x[local, pair.other]).detach())
                gain = float((rounded[0][a] - rounded[0][b]) @ f + rounded[1][a] - rounded[1][b])
                need = max(0.0, -margin) / float(2 * (f.abs().sum() + 1))
                positive += margin > 0
                improved += gain > 0
                if need > 0.02:
                    blocked.append([pair.row, pair.preferred, pair.other])
                if pair.row in (2652, 2682) and (pair.preferred, pair.other) == (58, 57):
                    slope = -float(
                        (gradients[0][a] - gradients[0][b]).double() @ f
                        + gradients[1][a]
                        - gradients[1][b]
                    )
                    focal[pair.row] = {
                        "initial_margin": margin,
                        "negative_gradient_slope": slope,
                        "first_adam_margin_gain_float32_storage": gain,
                        "individual_drift_needed_for_margin_0_1": max(0.0, 0.1 - margin)
                        / float(2 * (f.abs().sum() + 1)),
                    }
            assert all(torch.equal(p, old) for p, old in zip(params, initial, strict=True))
            assert all(p.grad is None for p in (*params, *c["actor"].parameters()))
            assert hashes == {str(p): pilot.sha256(p) for p in paths}
            # No inferred real-game frequency: these are enumerated prefix examples.
            counts = {}
            for name in ("train", "dev"):
                data = c["data"][name]
                counts[name] = dict(sorted(Counter(map(int, data.labels)).items()))
            report = {
                "coefficients": asdict(config),
                "weighted_gradient_norms": component_norms,
                "focal": focal,
                "baseline_positive_pairs": positive,
                "positive_first_adam_directions": improved,
                "pairs": 30,
                "individually_impossible_to_cross_zero_at_drift_0_02": blocked,
                "teacher_target_counts_not_on_policy_frequencies": counts,
                "protected_train_anchors": [
                    int(terms[k]) for k in ("bid_anchor_count", "pass_anchor_count")
                ],
                "protected_dev_anchors": [int(a.sum()) for a in c["anchors"]["dev"]],
                "parameters_and_input_hashes_unchanged": True,
                "actual_updates": 0,
                "first_step_only_no_generalization_claim": True,
            }
            print(json.dumps(report, indent=2))
            assert len(focal) == 2 and all(
                d["negative_gradient_slope"] > 0
                and d["first_adam_margin_gain_float32_storage"] > 1e-6
                for d in focal.values()
            ), "Combined Adam direction fails the proposed objective check"
            print("PASS: read-only split-objective gradient check; no training")


if __name__ == "__main__":
    main()
