"""Read-only inference for Phase 11B V3's 16 fixed auction cases.

Run from the tarneeb-rl repository root. No sampling, engine stepping,
training, checkpoint export, or changes to previous experiments.

Reuses the already-tested feature construction from the V2 diagnosis.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import compare_phase11b_audit_policies as prior  # noqa: E402
from scripts import diagnose_phase11b_specific_decisions as diagnosis  # noqa: E402

from tarneeb.core.engine import TarneebEngine  # noqa: E402
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint  # noqa: E402

V3_DATA = ROOT / "results/phase11b/independent_controls_v3"
V3_AUDIT = ROOT / "results/phase11b/counterfactual_value_audit_v3"
OUT = ROOT / "results/phase11b/v3_frozen_actor_comparison_v1"
EXPECTED_PLAN_ID = "56dcf7413d8dae98d897084503d2cad3da343f220560e81fb8da9c85ca8a3fa8"
EXPECTED_INPUT_SHA = "9c452730a66cdd47be4b11e6a78a57282e83dcc40a8945371dea774c796eb697"
EXPECTED_CASES_SHA = "9edcdf1533825b7879c13c5ebfa674c54f983cc8072c92c4c522f46cca1737d0"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def action_name(a: int) -> str:
    if a == 59:
        return "PASS"
    if 52 <= a <= 58:
        return f"BID {a - 45}"
    return f"INVALID/OTHER {a}"


def main():
    if OUT.exists():
        raise FileExistsError(f"Results already exist; refusing overwrite: {OUT}")

    plan_path = V3_AUDIT / "preregistration.json"
    input_path = V3_DATA / "inputs.json"
    cases_path = V3_DATA / "cases.json"
    plan = load_json(plan_path)
    if plan.get("identity") != EXPECTED_PLAN_ID:
        raise ValueError("V3 plan identity does not match the completed preregistration")
    if digest(input_path) != EXPECTED_INPUT_SHA or digest(cases_path) != EXPECTED_CASES_SHA:
        raise ValueError("V3 saved inputs/cases differ from the accepted V3 dataset")
    completion = load_json(V3_AUDIT / "completion.json")
    if completion.get("plan_identity") != EXPECTED_PLAN_ID:
        raise ValueError("V3 completion is not for the pinned preregistration")

    cases = load_json(cases_path)
    raw = load_json(input_path)
    if isinstance(raw, dict) and "inputs" in raw and isinstance(raw["inputs"], dict):
        raw = raw["inputs"]
    if isinstance(cases, dict) and "cases" in cases:
        cases = cases["cases"]
    if not isinstance(raw, dict) or not isinstance(cases, list):
        raise TypeError("Unexpected V3 input/case format; stop without writing results")
    ids = [c["id"] for c in cases]
    if len(ids) != 16 or len(set(ids)) != 16 or set(raw) != set(ids):
        raise ValueError("Expected exactly 16 matching V3 actor-input IDs")
    inputs = {case_id: raw[case_id] for case_id in ids}
    cases_by_id = {c["id"]: c for c in cases}
    results_by_id = {}
    for case_id in ids:
        result = load_json(V3_AUDIT / "states" / case_id / "result.json")
        if result.get("status") != "complete_state_result" or result.get("completed_worlds") != 64:
            raise ValueError(f"Incomplete V3 audit result: {case_id}")
        results_by_id[case_id] = result

    paths = {name: Path(path) for name, path in prior.MODELS.items()}
    if not paths or not all(p.is_file() for p in paths.values()):
        raise FileNotFoundError(f"Missing original model checkpoints: {paths}")
    checkpoint_hashes_before = {name: digest(path) for name, path in paths.items()}

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = {
        case_id: {
            "case": cases_by_id[case_id],
            "robust_classification": results_by_id[case_id]["robust_classification"],
            "teacher_evidence": results_by_id[case_id]["robust_teacher_status"],
            "actors": {},
        }
        for case_id in ids
    }

    # Prevent accidental card dealing or game simulation during this diagnostic.
    with (
        prior.checkpoint_locks(),
        patch.object(TarneebEngine, "deal", side_effect=AssertionError("No dealing allowed")),
        patch.object(TarneebEngine, "step", side_effect=AssertionError("No simulation allowed")),
        torch.no_grad(),
    ):
        for name, path in paths.items():
            actor = (
                load_recurrent_checkpoint(path, load_critic=False)
                .actor.cpu()
                .eval()
                .requires_grad_(False)
            )
            features = diagnosis.features(actor, inputs)
            if tuple(features.shape) != (16, 256):
                raise AssertionError(f"Unexpected {name} feature shape: {features.shape}")
            logits = actor.policy_head(features)
            for i, case_id in enumerate(ids):
                x = inputs[case_id]
                legal = torch.as_tensor(x["legal_action_mask"], dtype=torch.bool)
                if legal.numel() != 65 or not legal[59]:
                    raise ValueError(f"Invalid legal mask at {case_id}")
                masked = logits[i].masked_fill(~legal, -torch.inf)
                probabilities = torch.softmax(masked, dim=-1)
                chosen = int(masked.argmax().item())
                teacher = int(cases_by_id[case_id]["teacher"])
                legal_bids = [a for a in range(52, 59) if bool(legal[a])]
                best_bid = max(legal_bids, key=lambda a: float(masked[a])) if legal_bids else None
                item = {
                    "action": chosen,
                    "action_name": action_name(chosen),
                    "teacher_action": teacher,
                    "matches_teacher": chosen == teacher,
                    "pass_probability": float(probabilities[59]),
                    "legal_probabilities": {
                        action_name(a): float(probabilities[a])
                        for a in range(52, 60)
                        if bool(legal[a])
                    },
                    "legal_logits": {
                        action_name(a): float(masked[a]) for a in range(52, 60) if bool(legal[a])
                    },
                    "pass_minus_best_bid_logit": (
                        float(masked[59] - masked[best_bid]) if best_bid is not None else None
                    ),
                }
                report[case_id]["actors"][name] = item
            print(f"{name}: 16 frozen policy decisions evaluated", flush=True)

    checkpoint_hashes_after = {name: digest(path) for name, path in paths.items()}
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise AssertionError("A model checkpoint changed during read-only inference")
    if digest(input_path) != EXPECTED_INPUT_SHA or digest(cases_path) != EXPECTED_CASES_SHA:
        raise AssertionError("Accepted dataset changed during inference")

    OUT.mkdir(parents=True, exist_ok=False)
    payload = {
        "purpose": "read_only_v3_frozen_actor_inference",
        "source_plan_identity": EXPECTED_PLAN_ID,
        "checkpoints_sha256_unchanged": checkpoint_hashes_after,
        "training_updates": 0,
        "new_simulation_worlds": 0,
        "cases": report,
    }
    with (OUT / "comparison.json").open("x", encoding="utf-8") as writer:
        json.dump(payload, writer, indent=2, allow_nan=False)

    names = list(paths)
    headers = ["State", "Teacher", "V3 result", *names]
    rows = []
    for case_id in ids:
        r = report[case_id]
        rows.append(
            [
                case_id,
                action_name(int(r["case"]["teacher"])),
                r["robust_classification"],
                *(r["actors"][name]["action_name"] for name in names),
            ]
        )
    lines = [
        "# Phase 11B V3 — frozen actor decisions",
        "",
        "No training or simulations were run.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines += [
        "",
        "Each decision is greedy over the saved legal-action mask.",
        "The V3 result is based on simulated counterfactual evidence, not proven optimal play.",
        "Actor vs teacher disagreement alone is not an error on uncertain states.",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT / 'comparison.json'} and {OUT / 'SUMMARY.md'}", flush=True)
    for key in ("V3_04", "V3_09", "V3_05"):
        print(
            key,
            "V3:",
            report[key]["robust_classification"],
            ", ".join(f"{name}={report[key]['actors'][name]['action_name']}" for name in names),
            flush=True,
        )


if __name__ == "__main__":
    main()
