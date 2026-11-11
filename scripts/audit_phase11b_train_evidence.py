"""Prepare/explicitly execute the frozen TRAIN panel through the V3 recovery adapter."""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_phase11b_values_v3 as v3
from scripts import prepare_phase11b_train_evidence as selection

BASE = Path(__file__).resolve().parents[1]
MANIFEST = selection.OUTPUT
MANIFEST_SHA = "8d9785ef159296fc3e70d739a2484840c39e2cd5152a50c090386e2c9fe9d4af"
OUTPUT = BASE / "results/phase11b/retention_train_value_audit_v2"
V3_PLAN = v3.OUTPUT / "preregistration.json"
PURPOSE = "retention_train_evidence_v2"
LIMITS = {"cpu_seconds_per_state": 180, "wall_seconds": 1800, "workers": 4}
CODE = [
    "scripts/audit_phase11b_train_evidence.py",
    "scripts/prepare_phase11b_train_evidence.py",
    "tests/test_phase11b_train_evidence_audit.py",
]
r = v3.recovery
require = selection.require


def validated_cases():
    require(selection.sha256(MANIFEST) == MANIFEST_SHA, "Frozen selection hash mismatch")
    manifest = r.read(MANIFEST)
    with patch.object(selection.TarneebEngine, "deal", side_effect=RuntimeError("No dealing")):
        require(manifest == selection.build_selection(), "Frozen selection reconstruction mismatch")
    # Pin and check the accepted evaluator, not freshly trust changed scientific sources.
    prior = r.read(V3_PLAN)
    require(
        prior["identity"] == r.digest({k: v for k, v in prior.items() if k != "identity"}),
        "V3 plan checksum mismatch",
    )
    for path, expected in prior["source_hashes"].items():
        # README is a living project report, not executable scientific code.
        # Record both versions below; all other prior sources must still match.
        if path != "README.md":
            require(
                selection.sha256(BASE / path) == expected, f"Accepted V3 source changed: {path}"
            )
    require(prior["scientific_config"] == v3.v2.v1.CONFIG, "Scientific method changed")
    prior_seeds = {
        c[key]
        for plan in (prior, r.read(r.V1_PLAN))
        for c in plan["cases"]
        for key in ("world_seed", "bootstrap_seed")
    }
    forbidden = prior_seeds | set(manifest["train_seeds"]) | set(manifest["dev_seeds"]) | {20250910}
    seeds, cases = set(), []
    for index, original in enumerate(manifest["cases"]):
        for key in ("world_seed", "bootstrap_seed"):
            require(original[key] not in forbidden | seeds, "Audit seed collision")
            seeds.add(original[key])
        # V2/V3 use actor=0. This bijective seat relabeling preserves the exact
        # relative observation, partnership, bids and hand; it is not a new case.
        actor = original["actor"]
        case = {
            "index": index,
            "id": original["id"],
            "own_hand": original["own_hand"],
            "dealer": (original["dealer"] - actor) % 4,
            "prefix": original["public_auction_prefix"],
            "actions": [59] + [a for a in original["legal_actions"] if a != 59],
            "world_seed": original["world_seed"],
            "bootstrap_seed": original["bootstrap_seed"],
            "high_bid": 9,
            "owner": "partner",
            "teacher": original["teacher_action"],
            "strength": original["metadata"]["strength"],
            "classification": None,
            "training_selection": copy.deepcopy(original),
            "seat_mapping": {str(s): (s - actor) % 4 for s in range(4)},
        }
        for reverse in (False, True):
            unseen = sorted(set(range(52)) - set(case["own_hand"]), reverse=reverse)
            hands = [selection.hand_from_cards(case["own_hand"])] + [
                selection.hand_from_cards(unseen[i : i + 13]) for i in range(0, 39, 13)
            ]
            view = v3.v2.v1.decision_view(case, hands)
            static, history, length = selection.encode_public_view(view)
            encoded = {
                "statics": [static],
                "histories": [history],
                "lengths": [length],
                "legal": [view.legal_action_mask],
            }
            require(
                selection.fingerprint(encoded, 0) == original["observation_history_mask_sha256"],
                "Seat relabeling changed the frozen observation/history/mask fingerprint",
            )
            require(
                np.flatnonzero(view.legal_action_mask).tolist() == original["legal_actions"],
                "Action mask changed",
            )
            require(
                view.current_high_bidder == 2 and view.current_high_bid == 9,
                "Partner contract reconstruction mismatch",
            )
            require(
                selection.public_teacher_label(view) == original["teacher_action"],
                "Teacher mismatch",
            )
        cases.append(case)
    require(
        len(cases) == 12 and all(c["actions"] == [59, 55, 56, 57, 58] for c in cases),
        "Unexpected cases/actions",
    )
    return cases


def make_plan():
    plan = r.make_plan(validated_cases(), PURPOSE, copy.deepcopy(LIMITS))
    plan.update(
        output_directory=str(OUTPUT.absolute()),
        selection_sha256=MANIFEST_SHA,
        accepted_v3_plan_sha256=selection.sha256(V3_PLAN),
        scope="TRAIN evidence acquisition only; not independent generalization validation. No evidence labels before complete robust results.",
        seat_convention="audit_seat=(original_seat-original_actor)%4; actor partnership is audit seats 0/2. Original case stored intact; actor-visible fingerprints verified exactly. Result declarer seats use audit coordinates.",
        workload={
            "states": 12,
            "worlds_per_state": 64,
            "paired_worlds": 768,
            "actions_per_state": 5,
            "continuation_policies": 2,
            "candidate_continuations": 7680,
            "beliefs": 2,
            "maximum_state_cpu_seconds_total": 2160,
        },
    )
    prior = r.read(V3_PLAN)
    plan["source_hashes"].update(
        {p: h for p, h in prior["source_hashes"].items() if p != "README.md"}
    )
    plan["readme_provenance"] = {
        "v3_sha256": prior["source_hashes"]["README.md"],
        "current_sha256": selection.sha256(BASE / "README.md"),
        "note": "Living documentation changed since V3; scientific sources match. No prior plan rewritten.",
    }
    plan["source_hashes"].update({p: selection.sha256(BASE / p) for p in CODE})
    plan["identity"] = r.digest({k: v for k, v in plan.items() if k != "identity"})
    return plan


def verify_plan(plan, *, verify_files=True):
    # Regeneration binds every field, source, seed and original case before any restart.
    # File verification is intentionally mandatory for this production-only adapter.
    require(plan == make_plan(), "TRAIN audit plan/source/dataset identity mismatch")
    checked_root(OUTPUT)
    r.verify_plan(plan, verify_files=True)


def checked_root(root):
    with patch.object(v3, "OUTPUT", OUTPUT):
        return v3.checked_root(root)


def atomic(path, value, *, replace=False):
    with patch.object(v3, "OUTPUT", OUTPUT):
        return v3.atomic(path, value, replace=replace)


@contextlib.contextmanager
def bound_components():
    # Reuse V3's guarded writes, completed-state publication and V2 worker/watchdogs.
    # Each spawned worker installs the same process-local bindings. No source edits.
    with (
        patch.object(v3, "OUTPUT", OUTPUT),
        patch.object(v3.Components, "verify_plan", staticmethod(verify_plan)),
        patch.object(v3, "worker", worker),
        v3.bound_components(),
    ):
        yield


def worker(root, plan, case, remaining_cpu, stop, pipe, max_new, lease, wall_deadline):
    checked_root(root)
    with bound_components():
        return v3._WORKER(
            root, plan, case, remaining_cpu, stop, pipe, max_new, lease, wall_deadline
        )


def prepare():
    checked_root(OUTPUT)
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("Existing audit artifacts: refuse overwrite; use --preflight")
    plan = make_plan()
    verify_plan(plan)
    atomic(OUTPUT / "preregistration.json", plan)
    atomic(
        OUTPUT / "preflight.json",
        {
            "status": "prepared_not_executed",
            "plan_identity": plan["identity"],
            "workload": plan["workload"],
            "worlds_executed": 0,
            "frozen_selection_sha256": MANIFEST_SHA,
            "validation": "Exact source/checkpoint/manifest identity, TRAIN/DEV/prior-audit seed separation, original public reconstruction and actor=0 fingerprint equivalence passed.",
        },
    )
    print(f"PASS prepared only: {plan['identity']}")


def preflight():
    checked_root(OUTPUT)
    plan = r.read(OUTPUT / "preregistration.json")
    verify_plan(plan)
    print(f"PASS read-only preflight: {plan['identity']}")
    return plan


def execute():
    plan = preflight()
    with bound_components():
        result = v3.v2.run_session(OUTPUT, plan, workers=LIMITS["workers"])
        if result["status"] != "finished":
            raise SystemExit(2)  # Preserve partial worlds; never restart automatically.
        finals = [r.validate_result(OUTPUT, plan, c) for c in plan["cases"]]
        completion = {
            "status": "complete_audit",
            "states": 12,
            "worlds": 768,
            "plan_identity": plan["identity"],
            "result_sha256": [r.digest(x) for x in finals],
        }
        path = OUTPUT / "completion.json"
        if path.exists():
            require(r.read(path) == completion, "Completion identity mismatch")
        else:
            atomic(path, completion)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute-full-audit", action="store_true")
    args = parser.parse_args(argv)
    require(Path.cwd().resolve() == BASE, "Run from the repository root")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    else:
        execute()


if __name__ == "__main__":
    main()
