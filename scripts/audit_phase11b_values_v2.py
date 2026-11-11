"""Prepare recovery, run a diagnostic-only benchmark, or explicitly execute v2.

No automatic path from preparation or benchmark to the 32-state evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import multiprocessing as mp
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarneeb import auction_value_audit as v1
from tarneeb import auction_value_recovery as recovery


def worker(root, plan, case, remaining_cpu, stop, pipe, max_new, lease, wall_deadline):
    """An isolated process per state, with a CPU watchdog independent of rollouts."""
    done = threading.Event()

    def reserve_cpu():
        upper = min(remaining_cpu, time.process_time() + 2.0)
        recovery.atomic(
            Path(root) / lease,
            {"plan_identity": plan["identity"], "attempt": lease, "cpu_upper_bound": upper},
            replace=True,
        )
        return upper

    initial_upper = reserve_cpu()

    def watchdog():
        upper = initial_upper
        while not done.wait(0.02):
            if time.process_time() >= remaining_cpu:
                os._exit(124)
            if time.monotonic() >= wall_deadline.value:
                os._exit(126)
            if not mp.parent_process().is_alive():
                os._exit(125)
            if time.process_time() >= upper - 1.0:
                try:
                    upper = reserve_cpu()
                except BaseException:
                    os._exit(127)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        v1.initialize_worker()
        recovery.verify_plan(plan)
        progress = recovery.process_worlds(
            Path(root), plan, case, v1._ACTOR, stop=stop.is_set, max_new=max_new
        )
        if progress["complete"] and plan["purpose"] == "audit":
            recovery.publish_result(Path(root), plan, case)
        pipe.send(
            {
                "status": "complete" if progress["complete"] else "partial",
                **progress,
                "cpu_seconds": time.process_time(),
            }
        )
    except BaseException as exc:
        pipe.send(
            {
                "status": "interrupted" if isinstance(exc, InterruptedError) else "error",
                "error": repr(exc),
                "cpu_seconds": time.process_time(),
            }
        )
    finally:
        done.set()
        pipe.close()


def run_session(root, plan, *, workers, max_new=None):
    """Cumulative budgets survive ordinary stops; unknowable crash costs fail closed."""
    recovery.verify_plan(plan)
    ctx = mp.get_context("spawn")
    limits = plan["limits"]
    started = time.monotonic()
    session_id = str(time.time_ns())
    active = {}
    messages = []
    reason = "finished"
    with recovery.exclusive_run(root), recovery.checkpoint_read_locks():
        ledger = recovery.load_ledger(root, plan)
        remaining_wall = limits["wall_seconds"] - ledger["wall_spent"]
        if remaining_wall <= 0:
            raise TimeoutError("Cumulative wall budget exhausted; no automatic extension")
        # Windows spawn plus concurrent Torch imports can occupy the coordinator
        # for several seconds. This crash lease is not the overall wall cap.
        allowance = min(remaining_wall, 30.0)
        ledger["active_session"] = {"id": session_id, "reserved_wall": allowance}
        recovery.atomic(root / "budget.json", ledger, replace=True)
        wall_deadline = ctx.Value("d", started + allowance)
        last_heartbeat = started
        stop = ctx.Event()
        # Largest cases first reduces the four-worker tail; case/world identities
        # and RNG streams do not depend on scheduling or process identity.
        pending = []
        for case in plan["cases"]:
            if (root / "states" / case["id"] / "result.json").exists():
                recovery.validate_result(root, plan, case)
            else:
                pending.append(case)
        pending.sort(key=lambda c: (-len(c["actions"]), c["index"]))

        def collect(identifier, force=False):
            process, receive, _reserved = active.pop(identifier)
            if force and process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            message = None
            # Windows reports a broken pipe when a watchdog kills its writer.
            with contextlib.suppress(EOFError, OSError):
                if receive.poll():
                    message = receive.recv()
            receive.close()
            if message is None:
                message = {
                    "status": "cpu_limit" if process.exitcode == 124 else "unclean_stop",
                    "cpu_seconds": recovery.crash_cpu_charge(
                        root, plan, ledger["states"][identifier]
                    ),
                    "exitcode": process.exitcode,
                }
            state = ledger["states"][identifier]
            # Missing telemetry charges the durable allowance, or full reservation.
            state["spent"] += float(message["cpu_seconds"])
            state["reserved_cpu"] = 0
            state["status"] = message["status"]
            messages.append({"case": identifier, **message})
            recovery.atomic(root / "budget.json", ledger, replace=True)
            print(
                json.dumps(
                    {
                        "case": identifier,
                        "status": message["status"],
                        "cpu_seconds": round(message["cpu_seconds"], 3),
                        "new_worlds": message.get("new_worlds"),
                    }
                ),
                flush=True,
            )
            return message["status"]

        try:
            while pending or active:
                if time.monotonic() - started >= remaining_wall:
                    reason = "wall_limit"
                    break
                if time.monotonic() - last_heartbeat >= 1.0:
                    allowance = min(remaining_wall, time.monotonic() - started + 30.0)
                    ledger["active_session"]["reserved_wall"] = allowance
                    recovery.atomic(root / "budget.json", ledger, replace=True)
                    wall_deadline.value = started + allowance
                    last_heartbeat = time.monotonic()
                while pending and len(active) < workers:
                    case = pending.pop(0)
                    state = ledger["states"].setdefault(
                        case["id"], {"spent": 0.0, "reserved_cpu": 0.0, "status": "pending"}
                    )
                    remaining = limits["cpu_seconds_per_state"] - state["spent"]
                    if remaining <= 0:
                        reason = "state_cpu_budget_exhausted"
                        break
                    lease = f"states/{case['id']}/cpu_lease_{session_id}.json"
                    state.update(reserved_cpu=remaining, status="active", cpu_lease=lease)
                    recovery.atomic(root / "budget.json", ledger, replace=True)
                    receive, send = ctx.Pipe(duplex=False)
                    process = ctx.Process(
                        target=worker,
                        args=(
                            str(root),
                            plan,
                            case,
                            remaining,
                            stop,
                            send,
                            max_new,
                            lease,
                            wall_deadline,
                        ),
                        daemon=True,
                    )
                    process.start()
                    send.close()
                    active[case["id"]] = (process, receive, remaining)
                if reason != "finished":
                    break
                for identifier in list(active):
                    if not active[identifier][0].is_alive():
                        status = collect(identifier)
                        if status not in ("complete", "partial"):
                            reason = status
                            break
                if reason != "finished":
                    break
                time.sleep(0.02)
        except KeyboardInterrupt:
            reason = "user_interrupt"
        except BaseException:
            reason = "coordinator_error"
            raise
        finally:
            stop.set()
            # On wall limit terminate immediately; otherwise allow a bounded
            # cooperative exit at a candidate boundary to retain actual CPU usage.
            grace = 0 if reason == "wall_limit" else 5
            until = time.monotonic() + grace
            while active and time.monotonic() < until:
                for identifier in list(active):
                    if not active[identifier][0].is_alive():
                        collect(identifier)
                if active:
                    time.sleep(0.02)
            for identifier in list(active):
                collect(identifier, force=True)
            elapsed = time.monotonic() - started
            ledger["wall_spent"] += elapsed
            ledger["active_session"] = None
            recovery.atomic(root / "budget.json", ledger, replace=True)
            report = {
                "status": reason,
                "wall_seconds": elapsed,
                "messages": messages,
                "cpu_seconds_this_session": sum(m["cpu_seconds"] for m in messages),
                "plan_identity": plan["identity"],
                "budget": ledger,
            }
            recovery.atomic(root / f"session_{session_id}.json", report)
    return report


def benchmark():
    root = recovery.ROOT / "benchmark"
    if root.exists():
        raise FileExistsError("Benchmark already exists; no automatic repeat")
    cases = []
    original = recovery.read(recovery.V1_PLAN)["cases"]
    for i, index in enumerate([0, 15, 21, 24]):
        case = copy.deepcopy(original[index])
        case.update(
            id="BENCH_" + case["id"], world_seed=8_440_000 + i, bootstrap_seed=8_440_100 + i
        )
        cases.append(case)
    limits = {"cpu_seconds_per_state": 120.0, "wall_seconds": 300.0, "workers": 4}
    plan = recovery.make_plan(cases, "benchmark", limits)
    root.mkdir()
    recovery.atomic(
        root / "preregistration.json",
        {
            "plan": plan,
            "worlds_per_case": 2,
            "serial_then_parallel": True,
            "candidate_continuations_total": 168,
            "maximum_combined_wall_seconds": 600,
            "no_statistical_classification": True,
            "no_outcome_based_changes": True,
        },
    )
    reports = {}
    for name, workers in [("serial", 1), ("parallel", 4)]:
        path = root / name
        path.mkdir()
        reports[name] = run_session(path, plan, workers=workers, max_new=2)
        if reports[name]["status"] != "finished":
            raise RuntimeError("Benchmark stopped; preserve partial records, do not rerun")
    comparisons = []
    timing = []
    for case in cases:
        for i, hands in enumerate(recovery.worlds_for(case)[:2]):
            a = recovery.load_world(
                recovery.world_path(root / "serial", case, i), plan, case, i, hands
            )
            b = recovery.load_world(
                recovery.world_path(root / "parallel", case, i), plan, case, i, hands
            )
            if a["payload"] != b["payload"]:
                raise AssertionError("Serial/parallel numerical mismatch")
            comparisons.append(a["sha256"])
            timing.append(
                {
                    "case": case["id"],
                    "world": i,
                    "actions": len(case["actions"]),
                    "serial": a["timing"],
                    "parallel": b["timing"],
                }
            )
    # Outcomes deliberately stay opaque: report timing and exact equality only.
    recovery.atomic(
        root / "timing_report.json",
        {
            "serial_parallel_worlds_identical": len(comparisons),
            "timings": timing,
            "sessions": reports,
            "classification_computed": False,
            "outcome_values_inspected_for_tuning": False,
        },
    )
    print(
        json.dumps(
            {
                "benchmark_complete": True,
                "serial_wall": reports["serial"]["wall_seconds"],
                "parallel_wall": reports["parallel"]["wall_seconds"],
                "identical_worlds": len(comparisons),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--benchmark", action="store_true")
    modes.add_argument("--execute-full-audit", action="store_true")
    parser.add_argument("--state-cpu-seconds", type=float, default=180.0)
    parser.add_argument("--wall-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[key] = "1"
    if args.benchmark:
        benchmark()
        return
    if args.prepare:
        cases = recovery.read(recovery.V1_PLAN)["cases"]
        plan = recovery.make_plan(
            cases,
            "audit",
            {
                "cpu_seconds_per_state": args.state_cpu_seconds,
                "wall_seconds": args.wall_seconds,
                "workers": 4,
            },
        )
        recovery.atomic(recovery.ROOT / "preregistration.json", plan)
        print("Prepared only. Full evaluation still requires explicit approval.")
        return
    plan = recovery.read(recovery.ROOT / "preregistration.json")
    if plan["purpose"] != "audit":
        raise ValueError("Not a full-audit plan")
    result = run_session(recovery.ROOT, plan, workers=plan["limits"]["workers"])
    if result["status"] != "finished":
        raise SystemExit(2)
    # Workers validated all inputs before publishing; no second bootstrap run.
    finals = [
        recovery.read(recovery.ROOT / "states" / case["id"] / "result.json")
        for case in plan["cases"]
    ]
    completion = {
        "status": "complete_audit",
        "states": len(finals),
        "plan_identity": plan["identity"],
        "result_sha256": [recovery.digest(x) for x in finals],
    }
    path = recovery.ROOT / "completion.json"
    if path.exists():
        if recovery.read(path) != completion:
            raise ValueError("Completion mismatch")
    else:
        recovery.atomic(path, completion)


if __name__ == "__main__":
    main()
