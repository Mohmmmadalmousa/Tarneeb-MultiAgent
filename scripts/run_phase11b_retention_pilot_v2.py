"""V2 split-objective pilot; preparation never constructs an optimizer.

Versioned from the tested V1 runner without changing historical files or plans.
Only --execute-training can optimize, after separate explicit authorization.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import phase11b_retention_real_data_preflight as inputs

from tarneeb.auction_repair.artifacts import sha256, write_json_new
from tarneeb.auction_repair.data import frozen_features
from tarneeb.auction_repair.head_study import export_rows, role
from tarneeb.auction_repair.learning import AuctionRows, assert_preserved
from tarneeb.auction_repair.retention_objective import ObjectiveWeights, loss_terms, retention_masks
from tarneeb.auction_repair.train_evidence import fingerprints_for, load_train_evidence
from tarneeb.auction_value_recovery import checkpoint_read_locks, digest, read
from tarneeb.core.engine import TarneebEngine
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

ROOT = inputs.ROOT
OUT = ROOT / "results/phase11b/retention_pilot_v2"
FIXTURE = ROOT / "results/phase11b/diagnosis_20261014_v4/play_fixture.npz"
FIXTURE_SHA = "faedcf39bfdd1e2d929a8e7ab11d5bd7e96b5411550f4645f4cb777b7611a151"
WEIGHTS = ObjectiveWeights(
    heuristic_ce=0.1,
    supported_pair=None,
    bid_pass=0.02,
    bid_bid=0.02,
    bid_retention=1.0,
    pass_retention=1.0,
    pair_margin=0.1,
    retention_margin=0.1,
)
CONFIG = {
    "max_updates": 20,
    "validation_every": 5,
    "learning_rate": 3e-5,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "weight_decay": 0,
    "gradient_clip": 0.5,
    "max_drift": 0.02,
    "seed": 8930001,
    "threads": 4,
    "cpu_seconds": 120,
    "wall_seconds": 300,
    "batch": "all 2496 nontrivial TRAIN rows; no shuffling/resampling",
    "max_lost_dev_bids": 0,
    "max_lost_dev_pass": 0,
    "max_lost_train_anchors": 0,
}
CODE = [
    "scripts/run_phase11b_retention_pilot_v2.py",
    "tests/test_phase11b_retention_pilot_v2.py",
    "scripts/phase11b_retention_real_data_preflight.py",
    "tarneeb/auction_repair/train_evidence.py",
    "tarneeb/auction_repair/retention_objective.py",
    "tarneeb/auction_repair/learning.py",
    "tarneeb/auction_repair/head_study.py",
    "tarneeb/auction_repair/data.py",
    "tarneeb/auction_repair/artifacts.py",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config():
    require(
        ObjectiveWeights(
            heuristic_ce=0.1,
            supported_pair=None,
            bid_pass=0.02,
            bid_bid=0.02,
            bid_retention=1.0,
            pass_retention=1.0,
            pair_margin=0.1,
            retention_margin=0.1,
        )
        == WEIGHTS,
        "V2 requires the fixed split objective, never legacy aggregate evidence",
    )
    require(
        CONFIG["learning_rate"] == 3e-5
        and CONFIG["gradient_clip"] == 0.5
        and CONFIG["adam_betas"] == [0.9, 0.999]
        and CONFIG["adam_eps"] == 1e-8
        and CONFIG["weight_decay"] == 0,
        "Optimizer comparison budget changed",
    )
    require(
        0 < CONFIG["max_updates"] <= 20
        and CONFIG["validation_every"] == 5
        and CONFIG["max_updates"] % 5 == 0
        and 0 < CONFIG["max_drift"] <= 0.02,
        "Pilot bounds changed",
    )
    require(CONFIG["seed"] != 20250910 and CONFIG["seed"] >= 0, "Forbidden seed")


def load_context():
    validate_config()
    evidence = load_train_evidence()
    require(sha256(FIXTURE) == FIXTURE_SHA, "Frozen fixture hash mismatch")
    actor = load_recurrent_checkpoint(inputs.SOURCE, load_critic=False).actor.cpu().eval()
    actor.requires_grad_(False)
    data = {
        "train": inputs.examples(inputs.TRAIN),
        "dev": inputs.examples(inputs.DEV),
        "fixture": inputs.examples(FIXTURE),
    }
    require(len(data["fixture"]) == 424, "Preservation fixture size")
    seeds = [{m["seed"] for m in data[k].metadata} for k in ("train", "dev", "fixture")]
    require(
        all(20250910 not in s and CONFIG["seed"] not in s for s in seeds)
        and seeds[0].isdisjoint(seeds[1]),
        "Seed separation failed",
    )
    fingerprints = {k: fingerprints_for(data[k]) for k in ("train", "dev")}
    require(set(fingerprints["train"]).isdisjoint(fingerprints["dev"]), "TRAIN/DEV overlap")
    features, baseline, masks, labels, ids = {}, {}, {}, {}, {}
    with torch.no_grad():
        for key, examples in data.items():
            ids[key] = (
                np.flatnonzero(examples.legal.sum(1) > 1)
                if key != "fixture"
                else np.arange(len(examples))
            )
            all_features, _ = frozen_features(actor, examples)
            features[key] = all_features[ids[key]]
            baseline[key] = actor.policy_head(features[key]).detach()
            masks[key] = torch.from_numpy(examples.legal[ids[key]])
            labels[key] = torch.from_numpy(examples.labels[ids[key]])
    require(len(ids["train"]) == 2496, "TRAIN subset changed")
    rows = AuctionRows(actor)
    require(
        set(dict(rows.named_parameters())) == {"auction_weight", "auction_bias"},
        "Optimizer scope must be isolated auction tensors",
    )
    for key in ("train", "dev"):
        with torch.no_grad():
            require(
                torch.equal(
                    greedy(rows(features[key]), masks[key]), greedy(baseline[key], masks[key])
                ),
                "Initial auction decisions differ",
            )
    anchors = {}
    for key in ("train", "dev"):
        bid, passing = retention_masks(baseline[key], masks[key], labels[key])
        if key == "train":
            audited = {e.row for e in evidence.pairs}
            unaffected = torch.tensor([int(i) not in audited for i in ids[key]])
            bid &= unaffected
            passing &= unaffected
        anchors[key] = (bid, passing)
    return {
        "actor": actor,
        "rows": rows,
        "data": data,
        "features": features,
        "baseline": baseline,
        "masks": masks,
        "labels": labels,
        "ids": ids,
        "anchors": anchors,
        "evidence": evidence,
        "train_fingerprints": [fingerprints["train"][i] for i in ids["train"]],
    }


def greedy(logits, mask):
    return logits.masked_fill(~mask, -torch.inf).argmax(-1)


def objective(c):
    return loss_terms(
        c["rows"](c["features"]["train"]),
        c["baseline"]["train"],
        c["masks"]["train"],
        c["labels"]["train"],
        c["evidence"],
        row_ids=torch.from_numpy(c["ids"]["train"]),
        row_fingerprints=c["train_fingerprints"],
        config=WEIGHTS,
    )


def pair_metrics(pairs):
    """Report the two authenticated evidence components separately; no new labels."""
    result = {}
    for name, includes_pass in (("bid_pass", True), ("bid_bid", False)):
        selected = [p for p in pairs if (59 in (p["preferred"], p["other"])) == includes_pass]
        cases = {p["row"] for p in selected}
        result[name] = {
            "comparisons": len(selected),
            "cases": len(cases),
            "positive_pairs": sum(p["margin"] > 0 for p in selected),
            "pairs_at_target_margin": sum(p["margin"] >= WEIGHTS.pair_margin for p in selected),
            "case_mean_margin": (
                sum(
                    sum(p["margin"] for p in selected if p["row"] == row)
                    / sum(p["row"] == row for p in selected)
                    for row in sorted(cases)
                )
                / len(cases)
                if cases
                else None
            ),
        }
    return result


@torch.no_grad()
def metrics(c):
    report = {}
    raw = {k: c["rows"](c["features"][k]) for k in ("train", "dev")}
    for key in ("train", "dev"):
        require(bool(torch.isfinite(raw[key]).all()), "Nonfinite validation logits")
        actions, labels = greedy(raw[key], c["masks"][key]), c["labels"][key]
        bid, passing = c["anchors"][key]
        report[key] = {
            "rows": len(actions),
            "teacher_bids": int((labels != 59).sum()),
            "exact_teacher_bids": int(((labels != 59) & (labels == actions)).sum()),
            "teacher_pass": int((labels == 59).sum()),
            "bids_on_teacher_pass": int(((labels == 59) & (actions != 59)).sum()),
            "bid_anchors": int(bid.sum()),
            "pass_anchors": int(passing.sum()),
            "lost_bid_anchors": int((actions[bid] != labels[bid]).sum()),
            "lost_pass_anchors": int((actions[passing] != 59).sum()),
        }
    by_row = {int(row): i for i, row in enumerate(c["ids"]["train"])}
    pairs = [
        {
            "row": e.row,
            "preferred": e.preferred,
            "other": e.other,
            "margin": float(
                raw["train"][by_row[e.row], e.preferred] - raw["train"][by_row[e.row], e.other]
            ),
        }
        for e in c["evidence"].pairs
    ]
    report["supported_pairs"] = pairs
    report["positive_pairs"] = sum(p["margin"] > 0 for p in pairs)
    report["pairs_at_target_margin"] = sum(p["margin"] >= WEIGHTS.pair_margin for p in pairs)
    report["minimum_pair_margin"] = min(p["margin"] for p in pairs)
    report["maximum_pair_margin"] = max(p["margin"] for p in pairs)
    report["evidence_components"] = pair_metrics(pairs)
    report["loss_components"] = {
        key: float(value)
        for key, value in objective(c).items()
        if key
        in ("total", "heuristic_ce", "bid_pass", "bid_bid", "bid_retention", "pass_retention")
    }
    report["drift"] = max(
        float((c["rows"].auction_weight - c["actor"].policy_head.weight[52:60]).abs().max()),
        float((c["rows"].auction_bias - c["actor"].policy_head.bias[52:60]).abs().max()),
    )
    dev_actions = greedy(raw["dev"], c["masks"]["dev"])
    report["dev_roles"] = {}
    for name in ("opening", "partner", "opponent"):
        index = torch.tensor([role(c["data"]["dev"].metadata[i]) == name for i in c["ids"]["dev"]])
        teacher_pass = index & (c["labels"]["dev"] == 59)
        report["dev_roles"][name] = {
            "teacher_pass": int(teacher_pass.sum()),
            "bids_on_teacher_pass": int((teacher_pass & (dev_actions != 59)).sum()),
        }
    return report


def stop_reason(report, baseline):
    if report["dev"]["lost_bid_anchors"] > CONFIG["max_lost_dev_bids"]:
        return "forgotten_development_bid"
    if report["dev"]["lost_pass_anchors"] > CONFIG["max_lost_dev_pass"]:
        return "lost_protected_development_pass"
    if any(
        report["train"][k] > CONFIG["max_lost_train_anchors"]
        for k in ("lost_bid_anchors", "lost_pass_anchors")
    ):
        return "lost_training_retention_anchor"
    if any(
        report["dev_roles"][role]["bids_on_teacher_pass"]
        > baseline["dev_roles"][role]["bids_on_teacher_pass"]
        for role in baseline["dev_roles"]
    ):
        return "development_false_raise_increase"
    return None


def bounded_loop(propose, validate, record, *, start_cpu, start_wall):
    """Control logic testable without executing optimizer updates."""
    validated, attempted, status = 0, 0, "complete"
    for step in range(1, CONFIG["max_updates"] + 1):
        if (
            time.process_time() - start_cpu >= CONFIG["cpu_seconds"]
            or time.monotonic() - start_wall >= CONFIG["wall_seconds"]
        ):
            status = "resource_limit"
            break
        proposal = propose(step)
        attempted = step
        if not all(math.isfinite(proposal[k]) for k in ("loss", "gradient_norm", "drift")):
            status = "numerical_instability"
        elif proposal["drift"] > CONFIG["max_drift"]:
            status = "drift_limit"
        if status == "complete" and (step % CONFIG["validation_every"] == 0):
            status = validate(step) or "complete"
            if status == "complete":
                validated = step
        if (
            time.process_time() - start_cpu >= CONFIG["cpu_seconds"]
            or time.monotonic() - start_wall >= CONFIG["wall_seconds"]
        ):
            status = "resource_limit"
        record({"step": step, **proposal, "status": status})
        if status != "complete":
            break
    return {
        "status": status,
        "attempted_updates": attempted,
        "validated_updates": validated,
        "promotion": False,
    }


def preservation(c):
    candidate = export_rows(c["actor"], c["rows"])
    assert_preserved(c["actor"], candidate)
    with torch.no_grad():
        now = candidate.policy_head(c["features"]["fixture"])
    legal = c["masks"]["fixture"]
    require(torch.equal(now[legal], c["baseline"]["fixture"][legal]), "PLAY/DECLARE logits changed")
    require(
        torch.equal(greedy(now, legal), greedy(c["baseline"]["fixture"], legal)),
        "PLAY/DECLARE decisions changed",
    )
    require(
        all(not p.requires_grad and p.grad is None for p in c["actor"].parameters()),
        "Original actor received gradients",
    )
    return {
        "decisions": 424,
        "agreement": 1.0,
        "legal_logit_drift": 0.0,
        "frozen_parameters_equal": True,
    }


def inspect_baseline(c):
    start = time.perf_counter()
    before = {k: v.clone() for k, v in c["rows"].state_dict().items()}
    terms = objective(c)
    require(
        [
            int(terms[k])
            for k in (
                "bid_pass_pair_count",
                "bid_pass_case_count",
                "bid_bid_pair_count",
                "bid_bid_case_count",
                "supported_pair_count",
                "heuristic_excluded_count",
            )
        ]
        == [28, 8, 2, 2, 30, 8],
        "Authenticated split evidence panel changed",
    )
    components = {}
    for name in ("heuristic_ce", "bid_pass", "bid_bid", "bid_retention", "pass_retention"):
        gradient = torch.autograd.grad(
            terms[name], tuple(c["rows"].parameters()), retain_graph=True
        )
        norm = math.sqrt(sum(float(g.square().sum()) for g in gradient))
        require(math.isfinite(norm), "Nonfinite component gradient")
        components[name] = {
            "loss": float(terms[name].detach()),
            "gradient_l2": norm,
            "coefficient": getattr(WEIGHTS, name),
            "weighted_gradient_l2": norm * getattr(WEIGHTS, name),
            "prototype_coefficient": getattr(ObjectiveWeights(), name),
            "prototype_weighted_gradient_l2": norm * getattr(ObjectiveWeights(), name),
        }
    require(
        all(torch.equal(v, before[k]) for k, v in c["rows"].state_dict().items()),
        "Dry-run mutation",
    )
    result = metrics(c)
    require(stop_reason(result, result) is None, "Baseline stopping gate failed")
    result.update(
        components=components,
        preservation=preservation(c),
        optimizer_updates=0,
        trainable_parameters=sum(p.numel() for p in c["rows"].parameters()),
    )
    result["protected_row_ids"] = {
        key: {
            label: c["ids"][key][mask.numpy()].tolist()
            for label, mask in zip(("bid", "pass"), c["anchors"][key], strict=True)
        }
        for key in ("train", "dev")
    }
    return result, time.perf_counter() - start


def provenance():
    # Includes completed audit/source pins transitively checked by its authenticated loader.
    files = [
        *CODE,
        str(inputs.SOURCE.relative_to(ROOT)),
        str(inputs.TRAIN.relative_to(ROOT)),
        str(inputs.DEV.relative_to(ROOT)),
        str(FIXTURE.relative_to(ROOT)),
        "results/phase11b/retention_train_evidence_v1/selection.json",
        "results/phase11b/retention_train_value_audit_v2/preregistration.json",
        "results/phase11b/retention_train_value_audit_v2/completion.json",
    ]
    return {p: sha256(ROOT / p) for p in files}


def numerical_runtime():
    return {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version,
        "threads": CONFIG["threads"],
    }


def registered_plan():
    plan = read(OUT / "preregistration.json")
    require(
        plan["identity"] == digest({k: v for k, v in plan.items() if k != "identity"})
        and plan["config"] == CONFIG
        and plan["weights"] == asdict(WEIGHTS)
        and plan["input_source_hashes"] == provenance()
        and plan["numerical_runtime"] == numerical_runtime(),
        "Preregistered identity changed",
    )
    baseline = read(OUT / "baseline.json")
    require(digest(baseline) == plan["baseline_digest"], "Baseline identity changed")
    return plan, baseline


def prepare_or_dry_run(prepare):
    validate_config()
    if prepare and OUT.exists():
        raise FileExistsError("Pilot directory already exists; never overwrite")
    pinned = provenance()
    with (
        checkpoint_read_locks(),
        patch.object(
            torch.optim.Adam,
            "__init__",
            side_effect=RuntimeError("No optimizer during preparation"),
        ),
    ):
        c = load_context()
        # Public-prefix reconstruction above is allowed; no engine steps from here onward.
        with patch.object(TarneebEngine, "step", side_effect=RuntimeError("No simulations")):
            baseline, diagnostic_seconds = inspect_baseline(c)
    require(provenance() == pinned, "Input/source changed during preparation")
    if prepare:
        plan = {
            "version": "retention_pilot_v2",
            "config": CONFIG,
            "weights": asdict(WEIGHTS),
            "input_source_hashes": pinned,
            "baseline_digest": digest(baseline),
            "numerical_runtime": numerical_runtime(),
            "validation_updates": [0, 5, 10, 15, 20],
            "stop_rules": "Any previously correct DEV bid forgotten, any protected DEV PASS lost, any unaffected TRAIN anchor lost, or any role's teacher-PASS false raises increase: stop at validation. Nonfinite loss/gradient/parameters or >0.02 drift: stop each proposal. No retry, tuning, projection or promotion.",
            "coefficient_rationale": "Fixed split-objective coefficients CE=.10, BID/PASS=.02, BID/BID=.02, BID retention=1, PASS retention=1. Each evidence component averages within case, then across its participating cases (28 comparisons/8 cases and 2 comparisons/2 cases). This removes ranking dilution; no legacy aggregate pair loss is added. Previously validated initial weighted gradient norms: .607249/.604289/.457962; retention initially zero. Prioritize retention and generalization; neither perfect ranking nor all 30 preferences is a success gate. TRAIN preferences are diagnostics, not independent gameplay confirmation; DEV positive teacher labels cover BID7-8 only. Enumerated prefix frequencies do not establish on-policy prevalence or gameplay impact. No tuning or automatic continuation.",
            "v3": "Not loaded or used for training/tuning in this pilot; remains previously examined development evidence, never a sealed confirmation test.",
            "output_policy": "New run subdirectory, exclusive started marker, diagnostic auction rows only. Full original checkpoint never written; stopped runs cannot resume.",
            "evidence": dict(c["evidence"].provenance),
            "objective_components": {
                "bid_pass": {"pairs": 28, "cases": 8},
                "bid_bid": {"pairs": 2, "cases": 2},
            },
            "historical_comparator": "retention_pilot_v1: 20 updates, 5/30 before and after; preserved unchanged. Start V2 from original PFSP, not V1 candidate.",
        }
        plan["identity"] = digest(plan)
        write_json_new(OUT / "baseline.json", baseline)
        write_json_new(OUT / "preregistration.json", plan)
        write_json_new(
            OUT / "preparation.json",
            {
                "optimizer_updates": 0,
                "diagnostic_wall_seconds": diagnostic_seconds,
                "plan_identity": plan["identity"],
            },
        )
    elif (OUT / "preregistration.json").exists():
        _, saved_baseline = registered_plan()
        require(baseline == saved_baseline, "Read-only baseline reproduction failed")
    print(
        json.dumps(
            {
                k: v
                for k, v in baseline.items()
                if k not in ("protected_row_ids", "supported_pairs")
            },
            indent=2,
        )
    )


def execute():
    plan, baseline = registered_plan()
    run = OUT / "run"
    run.mkdir(exist_ok=False)  # Single attempt; even interrupted initialization cannot rerun.
    write_json_new(run / "started.json", {"plan_identity": plan["identity"]})
    start_cpu, start_wall = time.process_time(), time.monotonic()
    try:
        with checkpoint_read_locks():
            c = load_context()
            original_report, _ = inspect_baseline(c)
            require(digest(original_report) == plan["baseline_digest"], "Baseline not reproducible")
            rows = c["rows"]
            initial = {k: v.clone() for k, v in rows.state_dict().items()}
            optimizer = torch.optim.Adam(
                rows.parameters(),
                lr=CONFIG["learning_rate"],
                betas=tuple(CONFIG["adam_betas"]),
                eps=CONFIG["adam_eps"],
                weight_decay=0,
            )
            safe = deepcopy(initial)

            def propose(step):
                optimizer.zero_grad(set_to_none=True)
                terms = objective(c)
                loss = terms["total"]
                require(bool(torch.isfinite(loss)), "Nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    rows.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                require(
                    all(bool(torch.isfinite(v).all()) for v in rows.parameters()),
                    "Nonfinite parameters",
                )
                return {
                    "loss": float(loss.detach()),
                    "pre_update_loss_components": {
                        key: float(terms[key].detach())
                        for key in (
                            "heuristic_ce",
                            "bid_pass",
                            "bid_bid",
                            "bid_retention",
                            "pass_retention",
                        )
                    },
                    "gradient_norm": float(norm),
                    "drift": max(
                        float((v - initial[k]).abs().max()) for k, v in rows.state_dict().items()
                    ),
                }

            def validate(step):
                nonlocal safe
                report = metrics(c)
                report["preservation"] = preservation(c)
                reason = stop_reason(report, baseline)
                write_json_new(run / f"validation_{step:02d}.json", report)
                if reason is None:
                    safe = {k: v.clone() for k, v in rows.state_dict().items()}
                return reason

            with patch.object(TarneebEngine, "step", side_effect=RuntimeError("No simulations")):
                result = bounded_loop(
                    propose,
                    validate,
                    lambda entry: write_json_new(run / f"update_{entry['step']:02d}.json", entry),
                    start_cpu=start_cpu,
                    start_wall=start_wall,
                )
            rows.load_state_dict(safe)
            result["preservation"] = preservation(c)
            result.update(
                cpu_seconds=time.process_time() - start_cpu,
                wall_seconds=time.monotonic() - start_wall,
                plan_identity=plan["identity"],
                artifact="last_validated_auction_rows.npz",
            )
            with (run / "last_validated_auction_rows.npz").open("xb") as stream:
                np.savez_compressed(stream, **{k: v.detach().numpy() for k, v in safe.items()})
            write_json_new(run / "result.json", result)
            print(json.dumps(result))
    except BaseException as exc:
        write_json_new(
            run / "error.json", {"error": repr(exc), "no_restart": True, "promotion": False}
        )
        raise
    finally:
        require(plan["input_source_hashes"] == provenance(), "Protected input/source changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute-training", action="store_true")
    args = parser.parse_args(argv)
    require(Path.cwd().resolve() == ROOT, "Run from repository root")
    validate_config()
    torch.set_num_threads(CONFIG["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(CONFIG["seed"])
    if args.execute_training:
        execute()
    else:
        prepare_or_dry_run(args.prepare)


if __name__ == "__main__":
    main()
