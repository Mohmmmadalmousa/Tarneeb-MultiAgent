"""Additive A3 adapter. Preparation is read-only except new pinned manifests.

Selection and confirmation are separate, explicitly invoked commands. No training,
promotion, or changes to the frozen arena. Incomplete evidence fails closed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_a2 import seed_level

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.auction_repair.artifacts import sha256, write_json_new
from tarneeb.evaluation.a1 import ConditionEvaluation, compare_paired, evaluate_condition
from tarneeb.evaluation.arena import evaluation_match_seed
from tarneeb.league.population import LeagueConfig, LeaguePopulation, PromotionEvaluation
from tarneeb.league.registry import PolicyRegistry, actor_fingerprint
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint
from tarneeb.training.phase11_training import Phase11A3Config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase11_a3_evaluation_v1"
ADDENDUM = ROOT / "docs/PHASE11_A3_EVALUATION_ADDENDUM_V1.md"
CANDIDATES = tuple(f"{mode}_s{s}" for mode in ("latest_only", "pfsp") for s in range(3))
ANCHORS = {"random": 250, "basic": 1000, "strong": 2000}
SELECTION_SEED = 1_000_060_000
CONFIRMATION_SEED = 1_000_070_000
TEST_SEED = 1_000_090_000
CHAMPION = "phase9_gru_bc_s0"
HISTORY = (
    "phase9_gru_bc_s1",
    "phase9_gru_bc_s2",
    "phase10_centralized_s0",
    "phase10_centralized_s1",
    "phase10_centralized_s2",
)
CODE = (
    "scripts/evaluate_phase11_a3.py",
    "tests/test_phase11_a3_evaluation.py",
    "scripts/evaluate_a2.py",
    "docs/PHASE11_DECISIONS.md",
    "docs/PHASE11_A3_EVALUATION_ADDENDUM_V1.md",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_paths():
    paths = {CHAMPION: Path("checkpoints/phase9_gru_bc_s0_update250.pt")}
    paths.update({name: Path("checkpoints") / f"{name}_update250.pt" for name in HISTORY})
    for candidate in CANDIDATES:
        mode, seed = candidate.rsplit("_s", 1)
        folder = Path("checkpoints/phase11")
        if mode == "pfsp" and seed != "0":
            folder /= f"pfsp_s{seed}"
        stem = f"phase11_a3_{candidate}"
        paths[candidate] = folder / f"{stem}_final_u250.pt"
        # Shared u0 files predate PFSP directory isolation. Verified at preparation.
        paths[f"{candidate}_u0"] = folder / f"phase11_a3_s{seed}_u0.pt"
        for update in (50, 100, 150, 200):
            paths[f"{candidate}_u{update}"] = folder / f"{stem}_snapshot_u{update}.pt"
    return paths


def predecessors(candidate):
    require(candidate in CANDIDATES, "Unknown candidate")
    return tuple(f"{candidate}_u{u}" for u in (0, 50, 100, 150, 200))


def identities(seed, count):
    """Card identities; ignore dealer to conservatively exclude reused hands."""
    require(
        isinstance(seed, int) and isinstance(count, int) and 0 < count <= 100_000,
        "Invalid/unknown deal allocation",
    )
    return {(evaluation_match_seed(seed, i), 0) for i in range(count)}


def exposure_from(value, source, entries, inherited_count=None):
    """Only inspect seed metadata; outcomes are never used for allocation."""
    if isinstance(value, list):
        for item in value:
            exposure_from(item, source, entries, inherited_count)
    elif isinstance(value, dict):
        count = next(
            (
                value[k]
                for k in ("paired_deal_count", "paired_deals", "pair_count")
                if type(value.get(k)) is int
            ),
            inherited_count,
        )
        if type(value.get("evaluation_seed")) is int:
            require(count is not None, f"Unresolved evaluation count: {source}")
            identities(value["evaluation_seed"], count)
            entries.add((source, "arena", value["evaluation_seed"], count))
        if type(value.get("match_seed")) is int:
            require(
                type(value.get("deal_index")) is int, f"Unresolved direct deal identity: {source}"
            )
            entries.add((source, "direct", value["match_seed"], value["deal_index"]))
        if type(value.get("seed_start")) is int and type(value.get("seed_stop")) is int:
            start, stop = value["seed_start"], value["seed_stop"]
            require(0 <= start < stop and stop - start <= 100_000, "Invalid seed range")
            # Include both interpretations conservatively, even for direct-deal BC probes.
            entries.add((source, "arena", start, stop - start))
            entries.add((source, "direct_range", start, stop))
        for key, item in value.items():
            if key not in ("input_source_hashes", "source_hashes", "rng_state"):
                exposure_from(item, source, entries, count)


def audit_exposure():
    """Inventory persisted experiment metadata, plus conservative test reservations.

    World/state records have parent preregistrations and are direct synthetic
    conditional samples, not arena seed allocations. _tests is synthetic scratch.
    Unknown readable experiment JSON or malformed evaluation declarations abort.
    This cannot attest to deleted logs or unrecorded external experiments.
    """
    files, entries = {}, set()
    for base, dirs, names in os.walk(ROOT / "results", onerror=lambda e: (_ for _ in ()).throw(e)):
        dirs[:] = sorted(
            d for d in dirs if d not in ("_tests", "states", "worlds") and Path(base) / d != OUT
        )
        for name in sorted(names):
            path = Path(base) / name
            if path.suffix != ".json" and not (path.suffix == ".log" and "phase11_a3_" in name):
                continue
            relative = path.relative_to(ROOT).as_posix()
            raw = path.read_bytes()
            text = raw.decode(
                "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            )
            value = (
                json.loads(text)
                if path.suffix == ".json"
                else ast.literal_eval(text[text.index("{") :].strip())
            )
            files[relative] = sha256(path)
            exposure_from(value, relative, entries)
    # Persisted earlier A1/A2/MVP records are compulsory, not optional evidence.
    for name in (
        "results/phase9/a1_report.json",
        "results/phase10/a2_report.json",
        "results/mvp/stage2_final_evaluation.json",
    ):
        require(name in files, f"Missing prior exposure record: {name}")
    used = set()
    for _, kind, seed, count in entries:
        if kind == "arena":
            used.update(identities(seed, count))
        elif kind == "direct":
            used.add((seed, count))
        else:
            used.update((s, 0) for s in range(seed, count))
    # Historical automated-test roots are not all recorded: reserve the entire
    # low-offset block, and explicitly reserve the new adapter's test block.
    used.update(identities(0, 10_000))
    used.update(identities(TEST_SEED, 100))
    selection = identities(SELECTION_SEED, 2000)
    confirmation = identities(CONFIRMATION_SEED, 2000)
    require(not selection & confirmation, "Selection/confirmation deal overlap")
    require(not (selection | confirmation) & used, "Previously examined deal overlap")
    return {
        "files": files,
        "entries": sorted(entries),
        "used_identity_count": len(used),
        "used_identity_digest": digest(sorted(used)),
        "selection_identity_digest": digest(sorted(selection)),
        "confirmation_identity_digest": digest(sorted(confirmation)),
        "scope": "Persisted repository experiment metadata; deleted/unrecorded external runs unprovable",
        "excluded": ["_tests synthetic scratch", "states/worlds conditional simulation records"],
        "conservative_test_offset_reservation": [0, 9999, 90000, 90099],
    }


def jobs_for(candidate, stage):
    require(stage in ("selection", "confirmation"), "Unknown stage")
    seed = SELECTION_SEED if stage == "selection" else CONFIRMATION_SEED
    anchors = ANCHORS if stage == "selection" else {"strong": 2000}
    history = [(p, 300) for p in HISTORY] if stage == "selection" else []
    return [
        {"a": candidate, "b": opponent, "count": count, "seed": seed}
        for opponent, count in [
            *anchors.items(),
            (CHAMPION, 2000),
            *((p, 300) for p in predecessors(candidate)),
            *history,
        ]
    ]


def selection_jobs():
    return [j for c in CANDIDATES for j in jobs_for(c, "selection")] + [
        {"a": a, "b": b, "count": 300, "seed": SELECTION_SEED}
        for a, b in combinations(CANDIDATES, 2)
    ]


def source_hashes():
    paths = [ROOT / name for name in CODE]
    # Pin all shared scientific dependencies without editing any of them.
    paths += sorted((ROOT / "tarneeb").rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def prepare():
    require(not OUT.exists(), "Preregistration already exists; never overwrite")
    exposure = audit_exposure()
    models = {}
    for name, relative in model_paths().items():
        loaded = load_recurrent_checkpoint(ROOT / relative, load_critic=False)
        expected = (
            250 if name in (*CANDIDATES, CHAMPION, *HISTORY) else int(name.rsplit("_u", 1)[1])
        )
        require(
            loaded.payload["training"]["completed_updates"] == expected,
            f"Wrong checkpoint update: {name}",
        )
        require(loaded.payload["recurrent"]["gru_reset"] == "deal", "Wrong GRU reset")
        if name in CANDIDATES:
            mode, seed = name.rsplit("_s", 1)
            expected_config = Phase11A3Config(
                condition=mode,
                seed_index=int(seed),
                run_name=f"phase11_a3_{name}",
                checkpoint_directory=relative.parent,
            ).as_dict()
            require(
                json.loads(json.dumps(loaded.payload["training_config"]))
                == json.loads(json.dumps(expected_config)),
                "Frozen A3 training config mismatch",
            )
        require(
            all(bool(torch.isfinite(v).all()) for v in loaded.actor.state_dict().values()),
            "Nonfinite checkpoint",
        )
        candidate = name if name in CANDIDATES else name.rsplit("_u", 1)[0]
        if name in (CHAMPION, *HISTORY):
            owner, seed, policy_id = "pfsp_s0", "0", name
        else:
            owner, seed = candidate, candidate.rsplit("_s", 1)[1]
            policy_id = (
                f"phase11_a3_s{seed}_u0" if expected == 0 else f"phase11_a3_{candidate}_u{expected}"
            )
        league = read(ROOT / f"results/phase11/phase11_a3_{owner}_s{seed}_league_state.json")
        require(
            actor_fingerprint(loaded.actor.state_dict())
            == league["registry"]["records"][policy_id]["actor_fingerprint"],
            f"Original A3 registry fingerprint mismatch: {name}",
        )
        models[name] = {"path": relative.as_posix(), "sha256": sha256(ROOT / relative)}
    plan = {
        "version": "a3_evaluation_v1",
        "models": models,
        "source_hashes": source_hashes(),
        "exposure_digest": digest(exposure),
        "selection_jobs": selection_jobs(),
        "confirmation_jobs": {c: jobs_for(c, "confirmation") for c in CANDIDATES},
        "selection_pairs": sum(j["count"] for j in selection_jobs()),
        "confirmation_pairs_max": 5500,
        "bootstrap_resamples": 10_000,
        "threads": 4,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version,
        "deterministic_actions": True,
        "promotion": False,
        "selection_order": [
            "eligible gate",
            "Strong mean descending",
            "worst required snapshot mean descending",
            "champion mean descending",
            "candidate ID lexicographically ascending",
        ],
    }
    plan["identity"] = digest(plan)
    write_json_new(OUT / "exposure.json", exposure)
    write_json_new(OUT / "preregistration.json", plan)
    print(
        json.dumps(
            {
                "prepared": plan["identity"],
                "selection_pairs": plan["selection_pairs"],
                "confirmation_pairs_max": 5500,
                "models": len(models),
                "evaluated_pairs": 0,
            }
        )
    )


def preflight():
    plan = read(OUT / "preregistration.json")
    require(
        plan["identity"] == digest({k: v for k, v in plan.items() if k != "identity"}),
        "Plan identity changed",
    )
    require(plan["source_hashes"] == source_hashes(), "Source changed")
    require(plan["exposure_digest"] == digest(audit_exposure()), "Exposure inventory changed")
    require(plan["selection_jobs"] == selection_jobs(), "Selection jobs changed")
    require(
        plan["confirmation_jobs"] == {c: jobs_for(c, "confirmation") for c in CANDIDATES},
        "Confirmation jobs changed",
    )
    require(
        (plan["torch"], plan["numpy"], plan["python"])
        == (torch.__version__, np.__version__, sys.version),
        "Numerical runtime changed",
    )
    for model in plan["models"].values():
        require(sha256(ROOT / model["path"]) == model["sha256"], "Checkpoint changed")
    return plan


def load_agent(name, models):
    if name == "random":
        return RandomAgent(seed=0)
    if name == "basic":
        return BasicHeuristicAgent()
    if name == "strong":
        return StrongHeuristicAgent()
    loaded = load_recurrent_checkpoint(ROOT / models[name]["path"], load_critic=False)
    loaded.actor.eval().requires_grad_(False)
    return RecurrentNeuralAgent(loaded.actor, name=name, deterministic=True)


def validate_result(record, job, plan_id):
    require(
        record["digest"] == digest({k: v for k, v in record.items() if k != "digest"}),
        "Result digest mismatch",
    )
    require(record["job"] == job and record["plan_identity"] == plan_id, "Wrong result identity")
    scores, metrics = record["scores"], record["metrics"]
    require(
        len(scores) == job["count"] and all(math.isfinite(x) for x in scores),
        "Incomplete/nonfinite paired results",
    )
    ci = metrics["paired_score_difference_ci"]
    require(
        ci["sample_count"] == job["count"] and ci["confidence"] == 0.95,
        "Wrong uncertainty/sample count",
    )
    require(all(math.isfinite(ci[k]) for k in ("mean", "lower_bound", "upper_bound")), "Invalid CI")
    require(ci["lower_bound"] <= ci["mean"] <= ci["upper_bound"], "Malformed CI")
    require(
        abs(float(np.mean(scores)) - metrics["mean_paired_score_difference"]) < 1e-10,
        "Mean does not match paired scores",
    )
    require(
        abs(ci["mean"] - metrics["mean_paired_score_difference"]) < 1e-10,
        "CI mean does not match paired scores",
    )
    require(
        metrics["illegal_action_rate"] == metrics["no_trump_action_rate"] == 0
        and metrics["replay_integrity_rate"] == 1,
        "Evaluation correctness failure",
    )


def evaluate_job(job, plan, stage):
    path = OUT / stage / f"{job['a']}__{job['b']}.json"
    if path.exists():
        record = read(path)
    else:
        with torch.inference_mode():
            ev = evaluate_condition(
                load_agent(job["a"], plan["models"]),
                load_agent(job["b"], plan["models"]),
                job["b"],
                job["a"],
                job["count"],
                job["seed"],
            )
        record = {
            "job": job,
            "plan_identity": plan["identity"],
            "scores": list(ev.paired_scores),
            "metrics": ev.metrics,
        }
        # Canonical JSON representation also makes fresh/cached returns identical
        # when shared metric functions contain integer keys or tuples.
        record = json.loads(json.dumps(record, allow_nan=False))
        record["digest"] = digest(record)
        validate_result(record, job, plan["identity"])
        write_json_new(path, record)
    validate_result(record, job, plan["identity"])
    print(f"{stage}: {job['a']} vs {job['b']} complete ({job['count']} pairs)", flush=True)
    return record


def gate(candidate, records, plan_id, stage):
    """Require all five pinned predecessors, then delegate to the frozen gate."""
    required = jobs_for(candidate, stage)
    try:
        for j in required:
            validate_result(records[(j["a"], j["b"])], j, plan_id)
        champ = records[(candidate, CHAMPION)]["metrics"]
        recent = {
            p: records[(candidate, p)]["metrics"]["mean_paired_score_difference"]
            for p in predecessors(candidate)
        }
        ev = PromotionEvaluation(
            2000,
            champ["mean_paired_score_difference"],
            champ["paired_score_difference_ci"]["lower_bound"],
            champ["paired_score_difference_ci"]["upper_bound"],
            predecessors(candidate),
            recent,
            records[(candidate, "strong")]["metrics"]["mean_paired_score_difference"],
        )
        population = LeaguePopulation(PolicyRegistry(), LeagueConfig(), np.random.default_rng(4))
        return population.promotion_passes(ev)
    except (KeyError, TypeError, ValueError):
        return False


def choose(records, plan_id):
    eligible = [c for c in CANDIDATES if gate(c, records, plan_id, "selection")]

    def order(c):
        def mean(opponent):
            return records[(c, opponent)]["metrics"]["mean_paired_score_difference"]

        return (-mean("strong"), -min(mean(p) for p in predecessors(c)), -mean(CHAMPION), c)

    return min(eligible, key=order) if eligible else None


def comparisons(records):
    report = {}
    for anchor in ANCHORS:
        per_seed = []
        for s in range(3):
            inputs = []
            for mode in ("pfsp", "latest_only"):
                c = f"{mode}_s{s}"
                r = records[(c, anchor)]
                inputs.append(ConditionEvaluation(c, anchor, tuple(r["scores"]), r["metrics"]))
            comp = compare_paired(*inputs, bootstrap_seed=SELECTION_SEED + s)
            per_seed.append(
                {
                    "seed": s,
                    "pfsp_minus_latest_only": comp.difference_mean,
                    "ci": comp.difference_ci,
                }
            )
        summary = seed_level([r["pfsp_minus_latest_only"] for r in per_seed])
        summary["verdict"] = summary["verdict"].replace("centralized", "PFSP")
        if not math.isfinite(summary.get("t", 0)):
            summary["t"] = None
        report[anchor] = {"per_seed": per_seed, "seed_level": summary}
    return report


def cross_play(records):
    matrix = {a: {a: 0.0} for a in CANDIDATES}
    for a, b in combinations(CANDIDATES, 2):
        value = records[(a, b)]["metrics"]["mean_paired_score_difference"]
        matrix[a][b], matrix[b][a] = value, -value
    cycles = []
    for a, b, c in combinations(CANDIDATES, 3):
        if (matrix[a][b] > 0 and matrix[b][c] > 0 and matrix[c][a] > 0) or (
            matrix[b][a] > 0 and matrix[c][b] > 0 and matrix[a][c] > 0
        ):
            cycles.append([a, b, c])
    return {
        "matrix": matrix,
        "point_estimate_three_cycles": cycles,
        "worst_finalist_opponent_mean": {
            a: min(v for b, v in row.items() if b != a) for a, row in matrix.items()
        },
        "note": "Descriptive cycles, not significance claims; matchup CIs in job files",
    }


def execute_selection(plan):
    require(not (OUT / "selection.json").exists(), "Selection already sealed")
    records = {(j["a"], j["b"]): evaluate_job(j, plan, "selection") for j in plan["selection_jobs"]}
    preflight()  # Source, checkpoint and exposure identities must still match.
    result = {
        "plan_identity": plan["identity"],
        "selected": choose(records, plan["identity"]),
        "evidence": {f"{a}__{b}": r["digest"] for (a, b), r in records.items()},
        "comparisons": comparisons(records),
        "cross_play": cross_play(records),
        "promotion": False,
    }
    result["digest"] = digest(result)
    write_json_new(OUT / "selection.json", result)


def execute_confirmation(plan):
    require(not (OUT / "confirmation.json").exists(), "Confirmation already sealed")
    selected = read(OUT / "selection.json")
    require(
        selected["digest"] == digest({k: v for k, v in selected.items() if k != "digest"})
        and selected["plan_identity"] == plan["identity"],
        "Invalid sealed selection",
    )
    records = {}
    for j in plan["selection_jobs"]:
        r = read(OUT / "selection" / f"{j['a']}__{j['b']}.json")
        validate_result(r, j, plan["identity"])
        require(
            selected["evidence"][f"{j['a']}__{j['b']}"] == r["digest"], "Selection evidence changed"
        )
        records[(j["a"], j["b"])] = r
    candidate = choose(records, plan["identity"])
    require(
        candidate is not None and candidate == selected["selected"], "No eligible sealed candidate"
    )
    # No confirmation outcomes are accessed until the fixed winner is authenticated.
    confirm = {
        (j["a"], j["b"]): evaluate_job(j, plan, "confirmation")
        for j in plan["confirmation_jobs"][candidate]
    }
    preflight()
    result = {
        "plan_identity": plan["identity"],
        "selection_digest": selected["digest"],
        "candidate": candidate,
        "gate_passed": gate(candidate, confirm, plan["identity"], "confirmation"),
        "promotion": False,
        "fallback": "retain existing champion if gate fails",
        "evidence": {f"{a}__{b}": r["digest"] for (a, b), r in confirm.items()},
    }
    result["digest"] = digest(result)
    write_json_new(OUT / "confirmation.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "preflight", "select", "confirm"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "Run from repository root")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    if args.prepare:
        prepare()
    else:
        plan = preflight()
        if args.select:
            execute_selection(plan)
        elif args.confirm:
            execute_confirmation(plan)
        else:
            print(f"PASS read-only preflight: {plan['identity']}")


if __name__ == "__main__":
    main()
