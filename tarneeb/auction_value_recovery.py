"""World-atomic recovery for the unchanged Phase 11B v1 scientific calculation.

Only fully paired worlds are committed. Timing/progress is never a statistical
result. This module contains no optimizer, training entry point or teacher edit.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb import auction_value_audit as v1
from tarneeb.core.recorder import DealRecord, ReplayEngine
from tarneeb.core.rules import RulesConfig
from tarneeb.env.observation import ObservationBuilder

ROOT = Path("results/phase11b/counterfactual_value_audit_v2")
V1_PLAN = v1.OUTPUT / "preregistration.json"
FORMAT = "auction_value_world_recovery_v2"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic(path: Path, value: Any, *, replace: bool = False) -> None:
    """Publish complete JSON; serialize mutable writers and retry transient locks.

    Eight attempts, at most 1.90 seconds backoff per lock/replacement operation.
    Persistent errors preserve the old destination and flushed temporary file.
    Immutable world/result publication retains its original no-overwrite behavior.
    """
    import errno
    import msvcrt

    path.parent.mkdir(parents=True, exist_ok=True)
    delays = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.64)

    def retry(operation, *, locking=False):
        for attempt in range(8):
            try:
                return operation()
            except OSError as exc:
                transient = getattr(exc, "winerror", None) in (5, 32, 33)
                if locking:
                    transient |= exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK)
                if not transient or attempt == 7:
                    raise
                time.sleep(delays[attempt])

    @contextlib.contextmanager
    def writer_lock():
        if not replace:
            yield
            return
        # Stable sidecar: never lock/hold open the destination being replaced.
        with path.with_name(path.name + ".write.lock").open("a+b") as lock:
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            retry(lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1), locking=True)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)

    with writer_lock():
        temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temp.open("xb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise FileExistsError(path)
        retry(lambda: (os.replace if replace else os.rename)(temp, path))


@contextlib.contextmanager
def exclusive_run(root: Path):
    """OS lock releases on crash; persistent lock file itself is not a stale lock."""
    import msvcrt

    path = root / "run.lock"
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "threads": 1,
        "interop_threads": 1,
    }


def source_hashes() -> dict[str, str]:
    prior = read(V1_PLAN)
    paths = set(prior["preserved_sources"]) | set(prior["code_sha256"])
    paths.update(["tarneeb/auction_value_recovery.py", "scripts/audit_phase11b_values_v2.py"])
    return {name: v1.sha(Path(name)) for name in sorted(paths)}


def make_plan(cases: list[dict[str, Any]], purpose: str, limits: dict[str, Any]) -> dict[str, Any]:
    if purpose == "audit" and cases != read(V1_PLAN)["cases"]:
        raise ValueError("The 32 original information sets must remain exact")
    for case in cases:
        if case["world_seed"] == 20250910 or case["bootstrap_seed"] == 20250910:
            raise ValueError("Forbidden held-out seed")
        if purpose != "audit" and 8_410_000 <= case["world_seed"] < 8_410_032:
            raise ValueError("Tests/benchmark must not evaluate final-audit seeds")
    body = {
        "format": FORMAT,
        "purpose": purpose,
        "v1_plan_sha256": v1.sha(V1_PLAN),
        "scientific_config": copy.deepcopy(v1.CONFIG),
        "cases": copy.deepcopy(cases),
        "source_checkpoint": str(v1.SOURCE),
        "source_sha256": v1.SOURCE_SHA,
        "source_hashes": source_hashes(),
        "runtime": runtime_identity(),
        "limits": limits,
    }
    return {**body, "identity": digest(body)}


def verify_plan(plan: dict[str, Any], *, verify_files: bool = True) -> None:
    if digest({k: v for k, v in plan.items() if k != "identity"}) != plan["identity"]:
        raise ValueError("Plan identity/checksum mismatch")
    if plan["format"] != FORMAT or plan["scientific_config"] != v1.CONFIG:
        raise ValueError("Scientific configuration mismatch")
    if plan["runtime"] != runtime_identity():
        raise ValueError("Numerical runtime mismatch")
    if plan["purpose"] == "audit" and plan["cases"] != read(V1_PLAN)["cases"]:
        raise ValueError("Original state/seed identities changed")
    if verify_files:
        if v1.sha(V1_PLAN) != plan["v1_plan_sha256"]:
            raise ValueError("v1 preregistration changed")
        for name, expected in plan["source_hashes"].items():
            if v1.sha(Path(name)) != expected:
                # Explicit owner-authorized persistence amendment; the original
                # scientific plan and world identities are never rewritten.
                import ast

                fix = ROOT / "persistence_fix"
                amendment = read(fix / "amendment.json")
                original = fix / "original_auction_value_recovery.py"
                if (
                    name != "tarneeb/auction_value_recovery.py"
                    or amendment["plan_identity"] != plan["identity"]
                    or amendment["original_sha256"] != expected
                    or v1.sha(original) != expected
                    or amendment["patched_sha256"] != v1.sha(Path(name))
                ):
                    raise ValueError(f"Source changed outside approved amendment: {name}")

                def scientific_ast(path):
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                    tree.body = [
                        node
                        for node in tree.body
                        if not isinstance(node, ast.FunctionDef)
                        or node.name not in ("atomic", "verify_plan")
                    ]
                    return ast.dump(tree, include_attributes=False)

                if scientific_ast(original) != scientific_ast(Path(name)):
                    raise ValueError("Persistence amendment altered scientific code")
        if v1.sha(v1.SOURCE) != plan["source_sha256"]:
            raise ValueError("Source checkpoint changed")


def worlds_for(case: dict[str, Any]) -> list[list[int]]:
    rng = np.random.default_rng(case["world_seed"])
    return [v1.sample_hands(case["own_hand"], rng) for _ in range(64)]


def world_path(root: Path, case: dict[str, Any], index: int) -> Path:
    return root / "states" / case["id"] / f"world_{index:03d}.json"


def compute_world(
    plan,
    case,
    index,
    hands,
    actor,
    *,
    stop=lambda: False,
    candidate=v1.play_candidate,
    timings=None,
):
    engine, state = v1.make_state(case, hands)
    for action in case["prefix"]:
        engine.step(state, action)
    observation = ObservationBuilder(engine).static_actor(state, 0)
    weight, lapse = v1.history_weight(case, hands)
    outcomes, replays = {}, {}
    for policy in v1.CONFIG["continuations"]:
        policy_cpu = time.process_time()
        outcomes[policy], replays[policy] = [], []
        for action in case["actions"]:
            if stop():
                raise InterruptedError("Incomplete world discarded; committed worlds preserved")
            result, record = candidate(case, hands.copy(), action, policy, actor, verify=index == 0)
            outcomes[policy].append(result)
            replays[policy].append(record)
        if timings is not None:
            timings[policy + "_cpu_seconds"] = time.process_time() - policy_cpu
    return {
        "status": "complete_world_not_state_result",
        "plan_identity": plan["identity"],
        "state_identity": digest(case),
        "world_index": index,
        "hands": hands,
        "actions": case["actions"],
        "weight": weight,
        "lapse": lapse,
        "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
        "outcomes": outcomes,
        "replays": replays,
    }


def validate_world(payload, plan, case, index, hands, *, verify_replay=True):
    expected = {
        "status": "complete_world_not_state_result",
        "plan_identity": plan["identity"],
        "state_identity": digest(case),
        "world_index": index,
        "hands": hands,
        "actions": case["actions"],
    }
    if any(payload.get(k) != value for k, value in expected.items()):
        raise ValueError("World/state/seed/configuration identity mismatch")
    weight, lapse = v1.history_weight(case, hands)
    if payload.get("weight") != weight or payload.get("lapse") != lapse:
        raise ValueError("Hidden-world likelihood mismatch")
    engine, state = v1.make_state(case, hands)
    for action in case["prefix"]:
        engine.step(state, action)
    obs = ObservationBuilder(engine).static_actor(state, 0)
    if payload.get("observation_sha256") != hashlib.sha256(obs.tobytes()).hexdigest():
        raise ValueError("Actor information-set mismatch")
    if set(payload.get("outcomes", {})) != set(v1.CONFIG["continuations"]):
        raise ValueError("Missing continuation")
    if set(payload.get("replays", {})) != set(v1.CONFIG["continuations"]):
        raise ValueError("Missing replay metadata")
    for policy in v1.CONFIG["continuations"]:
        rows = payload["outcomes"][policy]
        replay_rows = payload["replays"][policy]
        if len(rows) != len(case["actions"]) or len(replay_rows) != len(rows):
            raise ValueError("Incomplete candidate-action pairing")
        for action, row, record in zip(case["actions"], rows, replay_rows, strict=True):
            if len(row) != 7 or any(type(x) is not int for x in row):
                raise ValueError("Invalid outcome schema")
            margin, success, own, bid, declarer, trump, passed = row
            if passed:
                if row != [0, -1, -1, -1, -1, -1, 1] or record is not None:
                    raise ValueError("Invalid passed-out outcome")
                if action != 59 or case["high_bid"] is not None:
                    raise ValueError("Passed-out outcome after a bid is impossible")
            elif not (
                -42 <= margin <= 42
                and success in (0, 1)
                and own in (0, 1)
                and 7 <= bid <= 13
                and declarer in range(4)
                and trump in range(4)
                and own == int(declarer % 2 == 0)
            ):
                raise ValueError("Invalid scored outcome")
            if index == 0 and not passed:
                if record is None:
                    raise ValueError("Required replay missing")
                if verify_replay and not ReplayEngine(RulesConfig()).verify(
                    DealRecord.from_dict(record)
                ):
                    raise ValueError("Replay verification failed")
            elif record is not None:
                raise ValueError("Unexpected replay metadata")


def load_world(path, plan, case, index, hands, *, verify_replay=True):
    envelope = read(path)
    if set(envelope) != {"payload", "sha256", "timing"}:
        raise ValueError("Invalid record envelope")
    if envelope["sha256"] != digest(envelope["payload"]):
        raise ValueError("Corrupt world checksum")
    if any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in envelope["timing"].values()
    ):
        raise ValueError("Invalid timing record")
    validate_world(envelope["payload"], plan, case, index, hands, verify_replay=verify_replay)
    return envelope


def process_worlds(
    root, plan, case, actor, *, stop=lambda: False, max_new=None, candidate=v1.play_candidate
):
    """Must be called by the single coordinator owning root's OS lock."""
    worlds = worlds_for(case)
    existing = list((root / "states" / case["id"]).glob("world_*.json"))
    allowed = {world_path(root, case, i).name for i in range(64)}
    if any(p.name not in allowed for p in existing):
        raise ValueError("Out-of-range/duplicate world name")
    committed, new = [], 0
    for i, hands in enumerate(worlds):
        path = world_path(root, case, i)
        if path.exists():
            committed.append(load_world(path, plan, case, i, hands))
            continue
        if stop() or (max_new is not None and new >= max_new):
            break
        cpu, wall = time.process_time(), time.monotonic()
        policy_timings = {}
        payload = compute_world(
            plan, case, i, hands, actor, stop=stop, candidate=candidate, timings=policy_timings
        )
        validate_world(payload, plan, case, i, hands)
        envelope = {
            "payload": payload,
            "sha256": digest(payload),
            "timing": {
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.monotonic() - wall,
                **policy_timings,
            },
        }
        atomic(path, envelope)
        committed.append(envelope)
        new += 1
    # Every observer has the same known hand/public prefix despite hypothetical cards.
    if len({r["payload"]["observation_sha256"] for r in committed}) > 1:
        raise ValueError("Hidden cards changed the acting player's observation")
    return {"committed_worlds": len(committed), "new_worlds": new, "complete": len(committed) == 64}


def state_result(root, plan, case):
    """Refuse even to calculate classifications until all 64 paired worlds validate."""
    if plan["purpose"] == "benchmark":
        raise ValueError("Benchmark outcomes must not become statistical labels")
    worlds = worlds_for(case)
    records = [
        load_world(world_path(root, case, i), plan, case, i, h) for i, h in enumerate(worlds)
    ]
    weights = np.array([r["payload"]["weight"] for r in records])
    w = weights / weights.sum()
    ess = float(1 / (w @ w))
    lapse = float(w @ np.array([r["payload"]["lapse"] for r in records]))
    support = (
        ess >= v1.CONFIG["minimum_ess"]
        and w.max() <= v1.CONFIG["maximum_normalized_weight"]
        and lapse <= v1.CONFIG["maximum_lapse_responsibility"]
    )
    summaries = {}
    for policy in v1.CONFIG["continuations"]:
        values = np.array([r["payload"]["outcomes"][policy] for r in records], dtype=float)
        summaries[policy] = {
            belief: v1.summarize(
                values,
                weights if belief == "bid_conditioned" else np.ones(64),
                case["actions"],
                case["teacher"],
                case["bootstrap_seed"],
            )
            for belief in v1.CONFIG["beliefs"]
        }
    all_summaries = [s for p in summaries.values() for s in p.values()]
    classes = {s["classification"] for s in all_summaries}
    teachers = {s["teacher_status"] for s in all_summaries}
    result = {
        "status": "complete_state_result",
        "plan_identity": plan["identity"],
        "state_identity": digest(case),
        "world_sha256": [r["sha256"] for r in records],
        "completed_worlds": 64,
        "ess": ess,
        "max_weight": float(w.max()),
        "weighted_lapse_responsibility": lapse,
        "posterior_support_gate": bool(support),
        "robust_classification": next(iter(classes))
        if support and len(classes) == 1
        else "uncertain",
        "robust_teacher_status": next(iter(teachers))
        if support and len(teachers) == 1
        else "uncertain",
        "summaries": summaries,
    }
    return {**result, "result_sha256": digest(result)}


def publish_result(root, plan, case):
    result = state_result(root, plan, case)
    path = root / "states" / case["id"] / "result.json"
    if path.exists():
        if read(path) != result:
            raise ValueError("Existing final result does not match all validated worlds")
    else:
        atomic(path, result)
    return result


def validate_result(root, plan, case):
    """A final file alone is insufficient: require all immutable input worlds."""
    path = root / "states" / case["id"] / "result.json"
    result = read(path)
    if result.get("result_sha256") != digest(
        {k: v for k, v in result.items() if k != "result_sha256"}
    ):
        raise ValueError("Final result checksum mismatch")
    if (
        result.get("status") != "complete_state_result"
        or result.get("completed_worlds") != 64
        or result.get("plan_identity") != plan["identity"]
        or result.get("state_identity") != digest(case)
    ):
        raise ValueError("Invalid final state identity/completeness")
    records = [
        load_world(world_path(root, case, i), plan, case, i, h)
        for i, h in enumerate(worlds_for(case))
    ]
    if result.get("world_sha256") != [r["sha256"] for r in records]:
        raise ValueError("Final result inputs differ")
    return result


def crash_cpu_charge(root, plan, state):
    """Use a durable watchdog allowance, or fail closed to the full reservation."""
    reserved = state.get("reserved_cpu", 0)
    if not reserved or not state.get("cpu_lease"):
        return reserved
    try:
        lease = read(root / state["cpu_lease"])
        upper = lease["cpu_upper_bound"]
        if (
            lease["plan_identity"] == plan["identity"]
            and lease["attempt"] == state["cpu_lease"]
            and isinstance(upper, (int, float))
            and math.isfinite(upper)
            and 0 <= upper <= reserved
        ):
            return upper
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return reserved


def load_ledger(root, plan):
    path = root / "budget.json"
    ledger = (
        read(path)
        if path.exists()
        else {
            "plan_identity": plan["identity"],
            "wall_spent": 0.0,
            "active_session": None,
            "states": {},
        }
    )
    if ledger["plan_identity"] != plan["identity"]:
        raise ValueError("Budget identity mismatch")
    # Durable watchdog allowances include unreported work. Missing/invalid lease
    # telemetry conservatively charges the full reservation, never a fresh budget.
    if ledger["active_session"] is not None:
        ledger["wall_spent"] += ledger["active_session"]["reserved_wall"]
        ledger["active_session"] = None
        for state in ledger["states"].values():
            if state.get("reserved_cpu", 0):
                state["spent"] += crash_cpu_charge(root, plan, state)
                state["reserved_cpu"] = 0
                state["status"] = "unclean_stop_budget_charged"
        atomic(path, ledger, replace=True)
    return ledger


@contextlib.contextmanager
def checkpoint_read_locks():
    """Windows read/share-read handles deny writes/deletion during execution."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = []
    try:
        for path in [
            v1.SOURCE,
            Path("checkpoints/phase11b/phase11b_auction_bc_smoke_s8300001_v1.pt"),
        ]:
            handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 1, None, 3, 0, None)
            if handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            kernel.CloseHandle(handle)
