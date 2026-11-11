"""Prepare and execute the owner-approved paired offline head study exactly once."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_phase11b_audit_policies import checkpoint_locks, sha
from scripts.diagnose_phase11b_specific_decisions import examples, read

from tarneeb.auction_repair.artifacts import save_repair_new, write_json_new
from tarneeb.auction_repair.data import PublicExamples, frozen_features
from tarneeb.auction_repair.head_study import describe, export_rows, optimize_head, weights_for
from tarneeb.auction_repair.learning import preservation_report
from tarneeb.core.engine import TarneebEngine
from tarneeb.env.spaces import ACTOR_FEATURE_SLICES
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

PROPOSAL = Path("results/phase11b/specific_decision_diagnosis_v2/PROPOSED_EXPERIMENT.json")
OUT = Path("results/phase11b/head_capacity_control_v1")
CODE = [
    Path(__file__),
    Path("tarneeb/auction_repair/head_study.py"),
    Path("tests/test_phase11b_head_study.py"),
]


def write(name, value):
    write_json_new(OUT / name, value)


def fingerprint(data, i):
    import hashlib

    h = hashlib.sha256()
    for a in (data.statics[i], data.histories[i, : data.lengths[i]], data.legal[i]):
        h.update(a.tobytes())
    return h.hexdigest()


def prerequisite(config, train, dev):
    for name in ("training", "development"):
        assert sha(config[f"{name}_data"]) == config[f"{name}_sha256"]
    assert sha(config["source_checkpoint"]) == config["source_sha256"]
    assert (
        sha("checkpoints/phase11b/phase11b_auction_bc_smoke_s8300001_v1.pt")
        == "407fb22dc2c90c848952abbb5fdc201661de7459c2296b72c1083b4732fa0b0e"
    )
    seeds = [{m["seed"] for m in d.metadata} for d in (train, dev)]
    assert not seeds[0] & seeds[1] and all(20250910 not in s for s in seeds)
    fingerprints = [{fingerprint(d, i) for i in range(len(d))} for d in (train, dev)]
    assert not fingerprints[0] & fingerprints[1]
    for data in (train, dev):
        assert data.statics.shape == (len(data), 692)
        assert np.isfinite(data.statics).all()
        assert np.array_equal(
            data.statics[:, ACTOR_FEATURE_SLICES["legal_action_mask"]], data.legal
        )
        assert (data.statics[:, :52].sum(1) == 13).all()
        assert not data.legal[:, :52].any() and not data.legal[:, 60:].any()
        assert data.legal[:, 59].all() and data.legal[np.arange(len(data)), data.labels].all()
        for i, length in enumerate(data.lengths):
            events = data.histories[i, :length]
            assert np.isin(events[:, 0], [1, 2]).all() and not events[:, 4].any()
    assert len(train) == 2880 and len(dev) == 3840
    assert all(m["case_family"] == "natural" for m in dev.metadata)
    return {
        "train_rows": len(train),
        "development_rows": len(dev),
        "train_seed_count": len(seeds[0]),
        "development_seed_count": len(seeds[1]),
        "seed_overlap": 0,
        "exact_input_overlap": 0,
        "actor_boundary": "Pinned obs_v2 public datasets; no critic, hidden cards, or generated states",
        "labels": "Unchanged original teacher labels; uncertain audit cases remain unverified",
        "focal_states": "Already present in training; diagnostic fit only, not independent generalization",
    }


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    config = read(PROPOSAL)
    assert all(not Path(p).exists() for p in config["checkpoint_paths"])
    train, dev = examples(config["training_data"]), examples(config["development_data"])
    prerequisites = prerequisite(config, train, dev)
    keep, weights, cells = weights_for(train)
    assert len(keep) == 2496
    protected = {str(p): sha(p) for p in Path("results/phase11b").rglob("*") if p.is_file()}
    registered = read("results/phase11b/counterfactual_value_audit_v2/preregistration.json")
    protected.update({n: sha(n) for n in registered["source_hashes"]})
    protected.update({str(p): sha(p) for p in Path("checkpoints").rglob("*") if p.is_file()})
    OUT.mkdir()
    write("preservation_before.json", protected)
    write("prerequisites.json", prerequisites)
    with (OUT / "fixed_weights.npz").open("xb") as f:
        np.savez_compressed(f, indices=keep, labels=train.labels[keep], **weights)
    write("weight_cells.json", cells)
    write(
        "preregistration.json",
        {
            "approved_proposal": config,
            "proposal_sha256": sha(PROPOSAL),
            "code_hashes": {str(p): sha(p) for p in CODE},
            "weights_sha256": sha(OUT / "fixed_weights.npz"),
            "execution_order": ["uniform", "balanced"],
            "stop_rules": "Drift/arm CPU limit stops that arm, retaining last-safe diagnostic rows only; the independently planned other arm still runs once. Overall wall, nonfinite values, integrity failures, or unexpected errors stop the study. No restart, projection, budget increase or final checkpoint from a partial arm.",
            "evaluation_timing": "Final step 512 only for completed arms. A limited arm receives explicitly partial diagnostic evaluation, never a final checkpoint or success label.",
        },
    )
    print(json.dumps(prerequisites), flush=True)
    print("Prepared and pinned weights/code; no training performed.", flush=True)


def audit_examples():
    base = Path("results/phase11b/neural_policy_comparison_v2")
    inputs = read(base / "inputs.json")
    rows = {r["case"]["id"]: r for r in read(base / "comparison.json")}
    ids = list(inputs)
    histories = np.zeros((32, 80, 7), dtype=np.int64)
    for i, key in enumerate(ids):
        if inputs[key]["event_count"]:
            histories[i, : inputs[key]["event_count"]] = inputs[key]["event_ids"]
    data = PublicExamples(
        np.array([inputs[k]["observation"] for k in ids], dtype=np.float32),
        histories,
        np.array([inputs[k]["event_count"] for k in ids]),
        np.array([inputs[k]["legal_action_mask"] for k in ids], dtype=bool),
        np.array([rows[k]["case"]["teacher"] for k in ids]),
        tuple(
            {
                "current_high_bid": rows[k]["case"]["high_bid"],
                "high_bidder_is_partner": rows[k]["case"]["owner"] == "partner",
                "strength": rows[k]["case"]["heuristic_strength"],
                "audit_classification": rows[k]["classification"],
            }
            for k in ids
        ),
    )
    return data, ids


def gates(baseline, development, audit, preservation, eligible):
    expected = {"C9PS": 55, "C7OS": 53, "O8": 58, "C9OW": 59}
    actual = {r["id"]: r["action"] for r in audit["rows"]}
    roles = {}
    for role in baseline["roles"]:
        before, after = baseline["roles"][role], development["roles"][role]
        n = before["teacher_pass_count"]
        delta = (after["bids_on_teacher_pass"] - before["bids_on_teacher_pass"]) / n if n else 0
        roles[role] = {"denominator": n, "false_bid_rate_increase": delta, "passes": delta <= 0.01}
    result = {
        "eligible_completed_arm": eligible,
        "focal": {k: actual[k] == a for k, a in expected.items()},
        "pass_controls": roles,
        "development_nll_nonincreasing": development["all"]["nll"] <= baseline["all"]["nll"],
        "preservation": preservation,
    }
    result["feasibility_pass"] = bool(
        eligible
        and all(result["focal"].values())
        and all(v["passes"] for v in roles.values())
        and result["development_nll_nonincreasing"]
    )
    return result


def execute():
    plan = read(OUT / "preregistration.json")
    config = plan["approved_proposal"]
    assert sha(PROPOSAL) == plan["proposal_sha256"]
    assert all(sha(p) == h for p, h in plan["code_hashes"].items())
    assert sha(OUT / "fixed_weights.npz") == plan["weights_sha256"]
    assert all(not Path(p).exists() for p in config["checkpoint_paths"])
    protected = read(OUT / "preservation_before.json")
    assert all(sha(p) == h for p, h in protected.items())
    train, dev = examples(config["training_data"]), examples(config["development_data"])
    prerequisite(config, train, dev)
    fixture = examples(config["fixed_play_fixture"])
    assert len(fixture) == 424
    with np.load(OUT / "fixed_weights.npz", allow_pickle=False) as d:
        keep = d["indices"].copy()
        weights = {
            k: torch.tensor(d[k].copy(), dtype=torch.float32) for k in ("uniform", "balanced")
        }
        assert np.array_equal(d["labels"], train.labels[keep])
    audit, audit_ids = audit_examples()
    # Exclusive marker prevents rerunning either arm or the paired study.
    write(
        "started.json",
        {"execution": "single authorized paired study", "threads": 4, "seed": 8600001},
    )
    start = time.monotonic()
    deadline = start + 600
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(8600001)
    results = {}
    try:
        with (
            checkpoint_locks(),
            patch.object(TarneebEngine, "deal", side_effect=AssertionError("No new deals")),
            patch.object(TarneebEngine, "step", side_effect=AssertionError("No simulation")),
        ):
            parent = load_recurrent_checkpoint(config["source_checkpoint"], load_critic=False)
            original = parent.actor.cpu().eval().requires_grad_(False)
            tensors = {k: v.clone() for k, v in original.state_dict().items()}
            features = {}
            for name, data in (
                ("train", train),
                ("dev", dev),
                ("fixture", fixture),
                ("audit", audit),
            ):
                features[name], _ = frozen_features(original, data)
            baseline = describe(original, features["dev"], dev)
            write("original_development.json", baseline)
            write("original_training.json", describe(original, features["train"], train))
            write("original_audit.json", describe(original, features["audit"], audit, audit_ids))
            baseline_preservation = preservation_report(original, original, fixture)
            write("baseline_preservation.json", baseline_preservation)
            legal_fixture = torch.from_numpy(fixture.legal)
            with torch.no_grad():
                fixture_raw = original.policy_head(features["fixture"])

            def check(rows):
                assert all(torch.equal(v, original.state_dict()[k]) for k, v in tensors.items())
                candidate = export_rows(original, rows)
                with torch.no_grad():
                    current = candidate.policy_head(features["fixture"])
                assert torch.equal(current[legal_fixture], fixture_raw[legal_fixture])

            for arm, path in zip(("uniform", "balanced"), config["checkpoint_paths"], strict=True):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Overall wall limit; no more arms")
                print(f"Starting {arm} once", flush=True)
                write(
                    f"{arm}/started.json",
                    {"same_source_sha256": config["source_sha256"], "same_indices": True},
                )
                with (OUT / arm / "steps.jsonl").open("x", encoding="utf-8") as stream:

                    def record(entry, stream=stream):
                        stream.write(json.dumps(entry) + "\n")
                        stream.flush()

                    rows, metrics = optimize_head(
                        original,
                        features["train"][keep],
                        torch.from_numpy(train.legal[keep]),
                        torch.from_numpy(train.labels[keep]),
                        weights[arm],
                        check,
                        record,
                        deadline,
                    )
                candidate = export_rows(original, rows)
                preservation = preservation_report(original, candidate, fixture)
                assert (
                    preservation["deterministic_action_agreement"] == 1
                    and preservation["max_legal_logit_drift"] == 0
                )
                assert (
                    preservation["hidden_state_bit_equal"]
                    and preservation["max_masked_log_probability_drift"] == 0
                )
                with (OUT / arm / "diagnostic_head_rows.npz").open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        weight=rows.auction_weight.detach().numpy(),
                        bias=rows.auction_bias.detach().numpy(),
                    )
                with torch.no_grad():
                    after_fixture = candidate.policy_head(features["fixture"])
                    before_actions = fixture_raw.masked_fill(~legal_fixture, -1e9).argmax(-1)
                    after_actions = after_fixture.masked_fill(~legal_fixture, -1e9).argmax(-1)
                write(
                    f"{arm}/preservation_decisions.json",
                    [
                        {
                            **meta,
                            "index": i,
                            "before_action": int(before_actions[i]),
                            "after_action": int(after_actions[i]),
                            "max_legal_logit_drift": float(
                                (after_fixture[i] - fixture_raw[i])[legal_fixture[i]].abs().max()
                            ),
                        }
                        for i, meta in enumerate(fixture.metadata)
                    ],
                )
                train_result = describe(candidate, features["train"], train)
                dev_result = describe(candidate, features["dev"], dev)
                audit_result = describe(candidate, features["audit"], audit, audit_ids)
                for name, data in (
                    ("training", train_result),
                    ("development", dev_result),
                    ("audit", audit_result),
                ):
                    write(f"{arm}/{name}.json", data)
                result = {
                    **metrics,
                    "arm": arm,
                    "preservation": preservation,
                    "gates": gates(
                        baseline,
                        dev_result,
                        audit_result,
                        preservation,
                        metrics["eligible_final_checkpoint"],
                    ),
                    "checkpoint": None,
                    "config": {
                        **config["optimizer"],
                        "shuffle_seed": 8600001,
                        "shuffle_used": False,
                    },
                    "preregistration_sha256": sha(OUT / "preregistration.json"),
                }
                if metrics["eligible_final_checkpoint"]:
                    save_repair_new(
                        Path(path), Path(config["source_checkpoint"]), parent, candidate, result
                    )
                    result["checkpoint"] = {"path": path, "sha256": sha(path)}
                write(f"{arm}/result.json", result)
                results[arm] = result
                print(json.dumps({"arm": arm, **metrics}), flush=True)
                if metrics["status"] == "stopped_overall_wall_limit":
                    break
            assert all(torch.equal(v, original.state_dict()[k]) for k, v in tensors.items())
        write(
            "study_result.json",
            {
                "arms": results,
                "wall_seconds": time.monotonic() - start,
                "all_arms_completed": len(results) == 2
                and all(v["eligible_final_checkpoint"] for v in results.values()),
                "promotion": False,
                "new_gameplay_simulations": 0,
            },
        )
    except BaseException as exc:
        write(
            "error.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "completed_arm_reports": list(results),
                "no_restart": True,
            },
        )
        raise
    finally:
        changed = [p for p, h in protected.items() if sha(p) != h]
        write(
            "preservation_after.json",
            {
                "protected_files": len(protected),
                "changed": changed,
                "wall_seconds_including_final_hashing": time.monotonic() - start,
            },
        )
        assert not changed, changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else execute()
