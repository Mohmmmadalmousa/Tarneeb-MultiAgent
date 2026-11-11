"""Run ablation A2: centralized versus decentralized critic (Phase 10).

Evaluates the pre-registered final checkpoint of every Phase 10 run on the final
held-out seed, pairs the two conditions deal by deal, and writes the record.

Reuses the Phase 9 A1 statistics module unchanged. That is deliberate: if A2 had
its own comparison code, a difference between the two phases could come from the
analysis rather than from the experiment. The only thing that changes is which
two conditions are being differenced.

The conclusion is derived from the interval, not chosen. Centralized better,
no detectable difference, and centralized worse are all admissible.

Usage::

    python scripts/evaluate_a2.py --seeds 0 1 2 --paired-deals 400
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.evaluation.a1 import (
    compare_paired,
    evaluate_condition,
    head_to_head,
    measure_inference_latency,
    summarise_seed_stability,
)
from tarneeb.evaluation.stats import bootstrap_mean_confidence_interval
from tarneeb.models.centralized_critic import describe_condition
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint
from tarneeb.training.phase9_seeds import FINAL_HELD_OUT_SEED

ANCHORS = ("random", "basic", "strong")
SEED_LEVEL_T_CRITICAL = {2: 4.302652729911275, 3: 3.182446305284263}
"""Two-sided 95% t critical values by degrees of freedom, for the n=3 analysis."""


def build_anchor(name: str):
    if name == "random":
        return RandomAgent(seed=0)
    if name == "basic":
        return BasicHeuristicAgent()
    return StrongHeuristicAgent()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 10 ablation A2")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--paired-deals", type=int, default=400)
    parser.add_argument("--head-to-head-deals", type=int, default=400)
    parser.add_argument("--latency-deals", type=int, default=40)
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-directory", type=Path, default=Path("results/phase10"))
    return parser.parse_args()


def load_agent(directory: Path, critic_mode: str, seed_index: int, updates: int, name: str):
    path = directory / f"phase10_{critic_mode}_s{seed_index}_update{updates}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/train_phase10.py first")
    loaded = load_recurrent_checkpoint(path, load_critic=False)
    return RecurrentNeuralAgent(loaded.actor, name=name, deterministic=True)


def seed_level(values: list[float]) -> dict:
    """Return the n=3 t-interval, the analysis that governs an ablation claim."""
    array = np.asarray(values, dtype=np.float64)
    count = array.size
    mean = float(array.mean())
    if count < 2:
        return {"mean": mean, "n": count, "verdict": "insufficient seeds"}
    sd = float(array.std(ddof=1))
    se = sd / np.sqrt(count)
    tcrit = SEED_LEVEL_T_CRITICAL.get(count - 1, 1.96)
    lower, upper = mean - tcrit * se, mean + tcrit * se
    verdict = (
        "no detectable difference"
        if lower <= 0.0 <= upper
        else ("centralized better" if mean > 0 else "centralized worse")
    )
    return {
        "mean": mean,
        "sd": sd,
        "standard_error": se,
        "n": count,
        "t": mean / se if se else float("nan"),
        "lower_bound": lower,
        "upper_bound": upper,
        "verdict": verdict,
    }


def main() -> None:
    arguments = parse_arguments()
    started = time.perf_counter()
    report: dict = {
        "evaluation_seed": FINAL_HELD_OUT_SEED,
        "paired_deal_count": arguments.paired_deals,
        "seeds": list(arguments.seeds),
        "conditions": {mode: describe_condition(mode) for mode in ("decentralized", "centralized")},
        "per_seed": [],
        "pooled": {},
        "seed_level": {},
        "seed_stability": {},
        "notes": [],
    }

    pooled: dict[str, list[float]] = {a: [] for a in ANCHORS}
    per_seed_diff: dict[str, list[float]] = {a: [] for a in ANCHORS}
    per_anchor: dict[str, dict[str, list[float]]] = {
        a: {"centralized": [], "decentralized": []} for a in ANCHORS
    }

    def summarise(metrics: dict) -> dict:
        keys = (
            "mean_paired_score_difference",
            "paired_score_difference_ci",
            "win_rate",
            "win_rate_wilson_interval",
            "role_split_mean_score",
            "contract_success_rate",
            "mean_bid",
            "pass_action_rate",
            "illegal_action_rate",
            "no_trump_action_rate",
        )
        return {k: metrics[k] for k in keys}

    for seed_index in arguments.seeds:
        central = load_agent(
            arguments.checkpoint_directory,
            "centralized",
            seed_index,
            arguments.updates,
            f"central_s{seed_index}",
        )
        plain = load_agent(
            arguments.checkpoint_directory,
            "decentralized",
            seed_index,
            arguments.updates,
            f"plain_s{seed_index}",
        )
        entry: dict = {"seed_index": seed_index, "anchors": {}}

        for anchor in ANCHORS:
            central_eval = evaluate_condition(
                central,
                build_anchor(anchor),
                anchor,
                "centralized",
                arguments.paired_deals,
                FINAL_HELD_OUT_SEED,
            )
            plain_eval = evaluate_condition(
                plain,
                build_anchor(anchor),
                anchor,
                "decentralized",
                arguments.paired_deals,
                FINAL_HELD_OUT_SEED,
            )
            # compare_paired computes first minus second, so centralized is first.
            comparison = compare_paired(
                central_eval, plain_eval, bootstrap_seed=FINAL_HELD_OUT_SEED + seed_index
            )
            entry["anchors"][anchor] = {
                "centralized": summarise(central_eval.metrics),
                "decentralized": summarise(plain_eval.metrics),
                "paired_difference": comparison.as_dict(),
            }
            pooled[anchor].extend(
                float(c - p)
                for c, p in zip(central_eval.paired_scores, plain_eval.paired_scores, strict=True)
            )
            per_seed_diff[anchor].append(comparison.difference_mean)
            per_anchor[anchor]["centralized"].append(central_eval.mean_paired_score)
            per_anchor[anchor]["decentralized"].append(plain_eval.mean_paired_score)

            lo = comparison.difference_ci["lower_bound"]
            hi = comparison.difference_ci["upper_bound"]
            print(
                f"[a2 s{seed_index}] vs {anchor:<7} "
                f"central {central_eval.mean_paired_score:+8.4f} "
                f"plain {plain_eval.mean_paired_score:+8.4f} "
                f"diff {comparison.difference_mean:+8.4f} [{lo:+.4f}, {hi:+.4f}]",
                flush=True,
            )
        report["per_seed"].append(entry)

    for anchor in ANCHORS:
        interval = bootstrap_mean_confidence_interval(
            pooled[anchor], seed=FINAL_HELD_OUT_SEED, resamples=10_000
        )
        lower, upper = interval.lower_bound, interval.upper_bound
        report["pooled"][anchor] = {
            "centralized_mean": float(np.mean(per_anchor[anchor]["centralized"])),
            "decentralized_mean": float(np.mean(per_anchor[anchor]["decentralized"])),
            "difference_mean": interval.mean,
            "difference_ci": interval.as_dict(),
            "pair_count": len(pooled[anchor]),
            "verdict": (
                "no detectable difference"
                if lower <= 0.0 <= upper
                else ("centralized better" if interval.mean > 0 else "centralized worse")
            ),
        }
        report["seed_level"][anchor] = seed_level(per_seed_diff[anchor])
        report["seed_stability"][anchor] = {
            mode: summarise_seed_stability(per_anchor[anchor][mode])
            for mode in ("centralized", "decentralized")
        }
        s = report["seed_level"][anchor]
        p = report["pooled"][anchor]
        print(
            f"[a2 {anchor:<7}] pooled {interval.mean:+.4f} [{lower:+.4f}, {upper:+.4f}] "
            f"-> {p['verdict']}  ||  seed-level {s['mean']:+.4f} "
            f"[{s['lower_bound']:+.4f}, {s['upper_bound']:+.4f}] -> {s['verdict']}",
            flush=True,
        )

    central0 = load_agent(
        arguments.checkpoint_directory,
        "centralized",
        arguments.seeds[0],
        arguments.updates,
        "central",
    )
    plain0 = load_agent(
        arguments.checkpoint_directory,
        "decentralized",
        arguments.seeds[0],
        arguments.updates,
        "plain",
    )
    report["head_to_head"] = head_to_head(
        central0,
        plain0,
        arguments.head_to_head_deals,
        FINAL_HELD_OUT_SEED,
        bootstrap_seed=FINAL_HELD_OUT_SEED,
    )
    h2h_mean = report["head_to_head"]["mean_paired_score_difference"]
    h2h_win = report["head_to_head"]["win_rate"]
    print(f"[a2 h2h] centralized vs decentralized {h2h_mean:+.4f} win {h2h_win:.4f}", flush=True)

    report["latency"] = {
        "centralized_actor": measure_inference_latency(
            central0, arguments.latency_deals, FINAL_HELD_OUT_SEED, build_anchor("strong")
        ),
        "decentralized_actor": measure_inference_latency(
            plain0, arguments.latency_deals, FINAL_HELD_OUT_SEED, build_anchor("strong")
        ),
    }
    central_ms = report["latency"]["centralized_actor"]["milliseconds_per_decision"]
    plain_ms = report["latency"]["decentralized_actor"]["milliseconds_per_decision"]
    print(
        f"[a2 latency] central actor {central_ms:.3f} ms | plain actor {plain_ms:.3f} ms",
        flush=True,
    )

    summary_path = arguments.results_directory / "training_summary.json"
    if summary_path.exists():
        runs = json.loads(summary_path.read_text(encoding="utf-8"))
        report["training"] = {
            mode: {
                "throughput": summarise_seed_stability(
                    [r["environment_steps_per_second"] for r in runs if r["critic_mode"] == mode]
                ),
                "final_explained_variance": summarise_seed_stability(
                    [r["final_explained_variance"] for r in runs if r["critic_mode"] == mode]
                ),
            }
            for mode in ("centralized", "decentralized")
        }

    report["notes"] = [
        "Checkpoints are the pre-registered final update, selected by design.",
        "The final held-out seed was consulted once, after selection was frozen.",
        "Both conditions were initialized from the same Phase 9 recurrent BC actor.",
        "The actor is identical in both conditions; only the critic input width differs.",
        "The seed-level analysis governs: the unit of replication for an architecture "
        "claim is the training run, not the evaluation deal. The pooled analysis is "
        "reported as the anti-conservative reading.",
        "Three seeds bound what is detectable; a small true effect would not be "
        "resolvable at this budget and is not claimed either way.",
    ]

    arguments.results_directory.mkdir(parents=True, exist_ok=True)
    path = arguments.results_directory / "a2_report.json"
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nA2 finished in {time.perf_counter() - started:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
