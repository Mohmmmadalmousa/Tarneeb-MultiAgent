"""Isolated V3 identity/entry point over the unchanged V2 execution machinery."""

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

from scripts import audit_phase11b_values_v2 as v2
from scripts import prepare_phase11b_controls_v3 as data

from tarneeb import auction_value_recovery as recovery

BASE = Path(__file__).resolve().parents[1]
DATASET = BASE / "results/phase11b/independent_controls_v3"
CASES = DATASET / "cases.json"
OUTPUT = BASE / "results/phase11b/counterfactual_value_audit_v3"
CASE_SHA = "9edcdf1533825b7879c13c5ebfa674c54f983cc8072c92c4c522f46cca1737d0"
INPUT_SHA = "9c452730a66cdd47be4b11e6a78a57282e83dcc40a8945371dea774c796eb697"
OVERLAP_SHA = "9e77c9d005aa0f2e1592c728dbf0e048914109155acbce9f0bc0669fd3bdc3be"
MANIFEST_SHA = "00e9e60c57b7048d6a7a4db4a9b8980e52a42569e652d67fcd5326e0f2ae4311"
LIMITS = {"cpu_seconds_per_state": 180, "wall_seconds": 1800, "workers": 4}
PURPOSE = "independent_audit_v3"
CODE = [
    "scripts/audit_phase11b_values_v3.py",
    "tests/test_phase11b_adapter_v3.py",
    "scripts/prepare_phase11b_controls_v3.py",
]
_ATOMIC = recovery.atomic
_WORKER = v2.worker


def checked_root(root):
    root = Path(root)
    if root.resolve() != OUTPUT.absolute() or OUTPUT.resolve() != OUTPUT.absolute():
        raise ValueError("Only the isolated V3 output directory is permitted")
    return root.resolve()


def checked_path(path):
    checked_root(OUTPUT)
    path = Path(path)
    if not path.resolve().is_relative_to(OUTPUT.absolute()):
        raise ValueError("Write would escape V3 output directory")
    return path


def atomic(path, value, *, replace=False):
    return _ATOMIC(checked_path(path), value, replace=replace)


def dataset_hashes():
    pins = {
        "cases.json": CASE_SHA,
        "inputs.json": INPUT_SHA,
        "overlap_verified.json": OVERLAP_SHA,
        "verified_completion.json": MANIFEST_SHA,
    }
    for name, expected in pins.items():
        if data.sha(DATASET / name) != expected:
            raise ValueError(f"Accepted V3 dataset identity changed: {name}")
    manifest = data.read(DATASET / "verified_completion.json")["files"]
    for name, expected in manifest.items():
        if Path(name).name != name or data.sha(DATASET / name) != expected:
            raise ValueError(f"Accepted dataset artifact changed: {name}")
    return {**manifest, "verified_completion.json": MANIFEST_SHA}


def validate_dataset():
    """Only public prefix reconstruction and existing arrays; never sample worlds."""
    dataset_hashes()
    cases, inputs = data.read(CASES), data.read(DATASET / "inputs.json")
    groups, prior_inputs, prior_seeds, _ = data.load_references()
    declaration = data.read(DATASET / "selection_preregistration.json")
    for name, expected in declaration["input_hashes"].items():
        if data.sha(name) != expected:
            raise ValueError("Prior overlap-reference dataset changed")
    if len(cases) != 16 or list(inputs) != [c["id"] for c in cases]:
        raise ValueError("Incorrect V3 case count/order")
    seen = []
    for i, case in enumerate(cases):
        if (
            case["id"] != f"V3_{i + 1:02d}"
            or case["index"] != i
            or case["world_seed"] != 8710000 + i
            or case["bootstrap_seed"] != 8720000 + i
            or case["selection_seed"] != 8700000 + i
            or case["classification"] is not None
            or case["actor_seat"] != 0
            or case["dealer"] != (-1 - len(case["prefix"])) % 4
        ):
            raise ValueError("Case/seed/dealer identity mismatch")
        if {case[k] for k in ("selection_seed", "world_seed", "bootstrap_seed")} & (
            prior_seeds | {20250910}
        ):
            raise ValueError("Seed overlap")
        expected_actions = [59, 55, 56, 57, 58] if i < 8 else [59, 52, 53, 54, 55, 56, 57, 58]
        if case["actions"] != expected_actions or case["teacher"] not in expected_actions:
            raise ValueError("Incorrect legal candidate enumeration")
        x = inputs[case["id"]]
        for reverse in (False, True):
            view, obs, history, legal = data.encode(case, reverse=reverse)
            if (
                obs.tolist() != x["observation"]
                or history.tolist() != x["event_ids"]
                or len(history) != x["event_count"]
                or legal.tolist() != x["legal_action_mask"]
                or view.observer_seat != 0
                or view.current_high_bid != case["high_bid"]
                or (i < 8 and view.current_high_bidder != 2)
                or data.public_teacher_label(view) != case["teacher"]
                or np.flatnonzero(legal).tolist() != sorted(expected_actions)
            ):
                raise ValueError("Public observation/history/mask reconstruction mismatch")
            if data.fingerprint(obs, history, legal) in prior_inputs:
                raise ValueError("Exact prior information-set overlap")
        for refs in [*groups.values(), seen]:
            if data.nearest(case["own_hand"], refs)["minimum_replacements"] < 3:
                raise ValueError("Prior/internal hand-family overlap")
        seen.append((case["id"], data.suit_masks(case["own_hand"])))
    return cases


def make_plan():
    cases = validate_dataset()
    plan = recovery.make_plan(cases, PURPOSE, copy.deepcopy(LIMITS))
    plan.update(
        dataset_hashes=dataset_hashes(),
        output_directory=str(OUTPUT.absolute()),
        case_seed_source="Exact accepted V3 per-case world/bootstrap seeds",
        evaluation_only=True,
    )
    plan["source_hashes"].update({name: data.sha(BASE / name) for name in CODE})
    plan["identity"] = recovery.digest({k: v for k, v in plan.items() if k != "identity"})
    return plan


def verify_plan(plan, *, verify_files=True):
    if (
        plan.get("purpose") != PURPOSE
        or plan.get("limits") != LIMITS
        or plan.get("output_directory") != str(OUTPUT.absolute())
        or plan.get("evaluation_only") is not True
        or plan.get("source_checkpoint") != str(v2.v1.SOURCE)
        or plan.get("source_sha256") != v2.v1.SOURCE_SHA
    ):
        raise ValueError("V3 purpose/output/limits/checkpoint mismatch")
    checked_root(OUTPUT)
    if plan.get("dataset_hashes") != dataset_hashes() or plan.get("cases") != data.read(CASES):
        raise ValueError("V3 dataset binding mismatch")
    required = set(recovery.source_hashes()) | set(CODE)
    if set(plan.get("source_hashes", {})) != required:
        raise ValueError("Incomplete source identity")
    # Fail closed directly; V3 has no source-repair amendment exception.
    if verify_files and any(
        data.sha(BASE / n) != value for n, value in plan["source_hashes"].items()
    ):
        raise ValueError("Pinned V3 source changed")
    recovery.verify_plan(plan, verify_files=verify_files)


class Components:
    """V3 identity and complete-state publication; all algorithms delegate to V2."""

    def __getattr__(self, name):
        return getattr(recovery, name)

    verify_plan = staticmethod(verify_plan)
    atomic = staticmethod(atomic)

    @staticmethod
    def exclusive_run(root):
        return recovery.exclusive_run(checked_root(root))

    @staticmethod
    def process_worlds(root, plan, case, actor, **kwargs):
        checked_root(root)
        result = recovery.process_worlds(root, plan, case, actor, **kwargs)
        if result["complete"]:
            recovery.publish_result(root, plan, case)
        return result


@contextlib.contextmanager
def bound_components():
    # Per-process scope, restored on exit. No source files or V2 plan are changed.
    # Intercept recovery's internal writes as well as coordinator/watchdog writes.
    with (
        patch.object(recovery, "atomic", atomic),
        patch.object(v2, "recovery", Components()),
        patch.object(v2, "worker", worker),
    ):
        yield


def worker(root, plan, case, remaining_cpu, stop, pipe, max_new, lease, wall_deadline):
    checked_root(root)
    with bound_components():
        return _WORKER(root, plan, case, remaining_cpu, stop, pipe, max_new, lease, wall_deadline)


def prepare():
    checked_root(OUTPUT)
    if (OUTPUT / "preregistration.json").exists():
        raise FileExistsError("Already prepared; use --preflight, never overwrite")
    if OUTPUT.exists() and any(p.name != "_tests" for p in OUTPUT.iterdir()):
        raise FileExistsError("Unexpected existing output; do not reset or overwrite")
    plan = make_plan()
    verify_plan(plan)
    atomic(OUTPUT / "preregistration.json", plan)
    atomic(
        OUTPUT / "preflight.json",
        {
            "status": "prepared_not_executed",
            "plan_identity": plan["identity"],
            "cases": 16,
            "paired_worlds_planned": 1024,
            "worlds_executed": 0,
            "candidate_continuations_planned": 13312,
            "overlap_checks": "passed",
            "public_input_reconstruction": "passed for two deterministic placeholder allocations",
            "labels": "No predetermined value labels; retain uncertain cases unchanged",
        },
    )
    print(f"Prepared V3 only: {plan['identity']}", flush=True)


def preflight():
    checked_root(OUTPUT)
    plan = data.read(OUTPUT / "preregistration.json")
    verify_plan(plan)
    validate_dataset()
    print(f"Read-only preflight passed: {plan['identity']}; no hidden worlds evaluated", flush=True)
    return plan


def execute():
    plan = preflight()
    with bound_components():
        result = v2.run_session(OUTPUT, plan, workers=4)
        if result["status"] != "finished":
            raise SystemExit(2)
        finals = [recovery.validate_result(OUTPUT, plan, c) for c in plan["cases"]]
        completion = {
            "status": "complete_audit",
            "states": 16,
            "worlds": 1024,
            "plan_identity": plan["identity"],
            "result_sha256": [recovery.digest(x) for x in finals],
        }
        path = OUTPUT / "completion.json"
        if path.exists():
            if data.read(path) != completion:
                raise ValueError("Completion identity mismatch")
        else:
            atomic(path, completion)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute-full-audit", action="store_true")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.cases.resolve() != CASES.absolute():
        raise ValueError("Only the accepted V3 case file is permitted")
    checked_root(args.output)
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
