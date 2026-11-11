"""One bounded four-worker startup regression check; reuse saved serial controls."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_phase11b_values_v2 import run_session

from tarneeb import auction_value_recovery as r


def main():
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[key] = "1"
    original = r.ROOT / "benchmark"
    root = r.ROOT / "benchmark_startup_validation"
    if root.exists():
        raise FileExistsError(root)
    old = r.read(original / "preregistration.json")["plan"]
    plan = r.make_plan(old["cases"], "benchmark", old["limits"])
    changed = [k for k, v in old["source_hashes"].items() if plan["source_hashes"][k] != v]
    assert changed == ["scripts/audit_phase11b_values_v2.py"]
    root.mkdir()
    r.atomic(
        root / "preregistration.json",
        {
            "plan": plan,
            "worlds_per_case": 1,
            "workers": 4,
            "candidate_continuations": 42,
            "wall_limit_seconds": 300,
            "purpose": "Validate Windows startup lease correction against existing serial worlds",
            "changes": "Wall crash lease 5 to 30 seconds; unused local variable rename only",
            "saved_serial_plan_identity": old["identity"],
            "validation_script_sha256": r.v1.sha(Path(__file__)),
            "no_outcome_based_changes": True,
        },
    )
    report = run_session(root, plan, workers=4, max_new=1)
    if report["status"] != "finished":
        raise RuntimeError("Validation incomplete; preserve and stop")
    timings = []
    for case in plan["cases"]:
        hands = r.worlds_for(case)[0]
        previous = r.load_world(r.world_path(original / "serial", case, 0), old, case, 0, hands)
        current = r.load_world(r.world_path(root, case, 0), plan, case, 0, hands)
        # The two preregistrations intentionally bind distinct supervisor sources.
        # Every scientific field, sampled hand, action, outcome and replay must match.
        assert {k: v for k, v in previous["payload"].items() if k != "plan_identity"} == {
            k: v for k, v in current["payload"].items() if k != "plan_identity"
        }
        timings.append(
            {
                "case": case["id"],
                "actions": len(case["actions"]),
                "serial": previous["timing"],
                "parallel": current["timing"],
            }
        )
    r.atomic(
        root / "timing_report.json",
        {
            "identical_scientific_worlds": len(timings),
            "session": report,
            "timings": timings,
            "classifications": None,
            "outcomes_inspected_for_tuning": False,
        },
    )
    print(f"Four saved serial worlds matched exactly. Wall seconds: {report['wall_seconds']:.3f}")


if __name__ == "__main__":
    main()
