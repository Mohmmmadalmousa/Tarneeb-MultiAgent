"""Prepare or run exactly the preregistered, additive counterfactual audit."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarneeb.auction_value_audit import (
    CONFIG,
    OUTPUT,
    SOURCE,
    SOURCE_SHA,
    initialize_worker,
    run_case,
    select_cases,
    sha,
    write_new,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("Choose exactly --prepare or --run")
    if args.prepare:
        assert not OUTPUT.exists(), "Audit output path already exists; refuse overwrite"
        assert sha(SOURCE) == SOURCE_SHA
        prior = json.loads(
            Path("results/phase11b/auction_bc_smoke_s8300001_v1_preflight.json").read_text()
        )
        frozen = {p: sha(Path(p)) for p in prior["file_sha256"]}
        assert frozen == prior["file_sha256"]
        paths = [
            *Path("results/phase11b").rglob("*"),
            *Path("checkpoints/phase11b").rglob("*.pt"),
            SOURCE,
        ]
        preserved = {str(p): sha(p) for p in paths if p.is_file()}
        code = {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("tarneeb/auction_value_audit.py"),
                Path("tests/test_phase11b_value_audit.py"),
                Path("docs/PHASE11B_VALUE_AUDIT_PREREGISTRATION.md"),
            ]
        }
        cases = select_cases()
        assert len(cases) == 32 and all(c["world_seed"] != 20250910 for c in cases)
        plan = {
            "config": CONFIG,
            "cases": cases,
            "source_sha256": SOURCE_SHA,
            "preserved_sources": frozen,
            "preserved_artifacts": preserved,
            "code_sha256": code,
            "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "maximum_candidate_rollouts": sum(len(c["actions"]) for c in cases) * 64 * 2,
            "git_status_before": subprocess.check_output(
                ["git", "-c", "safe.directory=C:/dev/tarneeb-rl", "status", "--short"], text=True
            ),
        }
        OUTPUT.mkdir()
        write_new(OUTPUT / "preregistration.json", plan)
        print(
            json.dumps(
                {
                    "prepared": str(OUTPUT),
                    "states": len(cases),
                    "rollouts": plan["maximum_candidate_rollouts"],
                    "config": CONFIG,
                },
                indent=2,
            )
        )
        return
    plan = json.loads((OUTPUT / "preregistration.json").read_text())
    assert plan["config"] == CONFIG
    for p, h in plan["code_sha256"].items():
        assert sha(Path(p)) == h, p
    for group in ["preserved_sources", "preserved_artifacts"]:
        for p, h in plan[group].items():
            assert sha(Path(p)) == h, p
    write_new(
        OUTPUT / "execution_started.json",
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preregistration_sha256": sha(OUTPUT / "preregistration.json"),
        },
    )
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[key] = "1"
    start = time.monotonic()
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=CONFIG["workers"], initializer=initialize_worker
    ) as pool:
        futures = {pool.submit(run_case, c): c["id"] for c in plan["cases"]}
        try:
            for future in concurrent.futures.as_completed(futures, timeout=CONFIG["wall_seconds"]):
                result = future.result()
                results.append(result)
                write_new(OUTPUT / (result["case"]["id"] + ".json"), result)
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "case": result["case"]["id"],
                            "cpu_seconds": round(result["cpu_seconds"], 2),
                            "classification": result["robust_classification"],
                            "ess": round(result["ess"], 1),
                        }
                    ),
                    flush=True,
                )
        except BaseException as exc:
            write_new(
                OUTPUT / "execution_failure.json",
                {"error": repr(exc), "completed_cases": len(results)},
            )
            # Python 3.12 has no public terminate_workers method. These are only
            # the diagnostic's own child processes, never external training jobs.
            for process in list(pool._processes.values()):
                process.terminate()
            for process in list(pool._processes.values()):
                process.join(timeout=5)
            raise
    changed = []
    for group in ["preserved_sources", "preserved_artifacts"]:
        changed += [p for p, h in plan[group].items() if sha(Path(p)) != h]
    write_new(
        OUTPUT / "completion.json",
        {
            "cases": len(results),
            "wall_seconds": time.monotonic() - start,
            "simulation_cpu_seconds": sum(r["cpu_seconds"] for r in results),
            "preserved_files_changed": changed,
            "source_sha256": sha(SOURCE),
            "git_status_after": subprocess.check_output(
                ["git", "-c", "safe.directory=C:/dev/tarneeb-rl", "status", "--short"], text=True
            ),
        },
    )
    assert not changed


if __name__ == "__main__":
    main()
