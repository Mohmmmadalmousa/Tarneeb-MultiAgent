"""Run ablation A1: Event-GRU versus padded feedforward history (Phase 9).

Evaluates the pre-registered final checkpoint of every training run on the
**final held-out seed**, pairs the two architectures deal by deal, and writes the
complete record.

Two properties make the comparison fair rather than merely side by side:

- every condition plays the identical sequence of duplicate deals, because the
  frozen arena derives match seeds from ``(evaluation_seed, pair_index)``;
- the checkpoint is chosen by design -- the pre-registered final update -- not by
  looking at held-out scores.

The final held-out seed is consulted **once**, here, after selection is frozen.

Usage::

    python scripts/evaluate_a1.py --seeds 0 1 2 --paired-deals 400
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.neural_agent import NeuralAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.evaluation.a1 import (
    A1Report,
    compare_paired,
    evaluate_condition,
    head_to_head,
    measure_inference_latency,
    summarise_seed_stability,
)
from tarneeb.models.checkpoint import load_checkpoint
from tarneeb.models.nets import count_parameters
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint
from tarneeb.training.phase9_seeds import FINAL_HELD_OUT_SEED

ANCHORS = ("random", "basic", "strong")


def build_anchor(name: str):
    if name == "random":
        return RandomAgent(seed=0)
    if name == "basic":
        return BasicHeuristicAgent()
    return StrongHeuristicAgent()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 ablation A1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--updates", type=int, default=250, help="pre-registered final update")
    parser.add_argument("--paired-deals", type=int, default=400)
    parser.add_argument("--head-to-head-deals", type=int, default=400)
    parser.add_argument("--latency-deals", type=int, default=40)
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-directory", type=Path, default=Path("results/phase9"))
    return parser.parse_args()


def load_agent(directory: Path, architecture: str, seed_index: int, updates: int, name: str):
    """Load one condition's pre-registered final checkpoint as an arena agent."""
    path = directory / f"phase9_{architecture}_bc_s{seed_index}_update{updates}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/train_phase9.py first")
    if architecture == "gru":
        loaded = load_recurrent_checkpoint(path, load_critic=False)
        return RecurrentNeuralAgent(loaded.actor, name=name, deterministic=True), loaded.actor
    loaded = load_checkpoint(path, load_critic=False)
    return NeuralAgent(loaded.actor, name=name, deterministic=True), loaded.actor


def main() -> None:
    arguments = parse_arguments()
    started = time.perf_counter()
    report = A1Report(
        evaluation_seed=FINAL_HELD_OUT_SEED,
        paired_deal_count=arguments.paired_deals,
        seeds=list(arguments.seeds),
    )

    pooled_differences: dict[str, list[float]] = {name: [] for name in ANCHORS}
    per_anchor_gru: dict[str, list[float]] = {name: [] for name in ANCHORS}
    per_anchor_ff: dict[str, list[float]] = {name: [] for name in ANCHORS}

    for seed_index in arguments.seeds:
        gru_agent, gru_actor = load_agent(
            arguments.checkpoint_directory,
            "gru",
            seed_index,
            arguments.updates,
            f"gru_s{seed_index}",
        )
        ff_agent, ff_actor = load_agent(
            arguments.checkpoint_directory,
            "ff",
            seed_index,
            arguments.updates,
            f"ff_s{seed_index}",
        )
        if not report.parameter_counts:
            report.parameter_counts = {
                "gru": {"actor": count_parameters(gru_actor), "history_path": 255_112},
                "ff": {"actor": count_parameters(ff_actor), "history_path": 329_536},
            }

        seed_entry: dict = {"seed_index": seed_index, "anchors": {}}
        for anchor_name in ANCHORS:
            gru_evaluation = evaluate_condition(
                gru_agent,
                build_anchor(anchor_name),
                anchor_name,
                "gru",
                arguments.paired_deals,
                FINAL_HELD_OUT_SEED,
            )
            ff_evaluation = evaluate_condition(
                ff_agent,
                build_anchor(anchor_name),
                anchor_name,
                "ff",
                arguments.paired_deals,
                FINAL_HELD_OUT_SEED,
            )
            comparison = compare_paired(
                gru_evaluation, ff_evaluation, bootstrap_seed=FINAL_HELD_OUT_SEED + seed_index
            )
            seed_entry["anchors"][anchor_name] = {
                "gru": {
                    "mean_paired_score_difference": gru_evaluation.metrics[
                        "mean_paired_score_difference"
                    ],
                    "paired_score_difference_ci": gru_evaluation.metrics[
                        "paired_score_difference_ci"
                    ],
                    "win_rate": gru_evaluation.metrics["win_rate"],
                    "win_rate_wilson_interval": gru_evaluation.metrics["win_rate_wilson_interval"],
                    "role_split_mean_score": gru_evaluation.metrics["role_split_mean_score"],
                    "contract_success_rate": gru_evaluation.metrics["contract_success_rate"],
                    "mean_bid": gru_evaluation.metrics["mean_bid"],
                    "passed_out_rate": gru_evaluation.metrics["passed_out_rate"],
                    "pass_action_rate": gru_evaluation.metrics["pass_action_rate"],
                    "illegal_action_rate": gru_evaluation.metrics["illegal_action_rate"],
                    "no_trump_action_rate": gru_evaluation.metrics["no_trump_action_rate"],
                },
                "ff": {
                    "mean_paired_score_difference": ff_evaluation.metrics[
                        "mean_paired_score_difference"
                    ],
                    "paired_score_difference_ci": ff_evaluation.metrics[
                        "paired_score_difference_ci"
                    ],
                    "win_rate": ff_evaluation.metrics["win_rate"],
                    "win_rate_wilson_interval": ff_evaluation.metrics["win_rate_wilson_interval"],
                    "role_split_mean_score": ff_evaluation.metrics["role_split_mean_score"],
                    "contract_success_rate": ff_evaluation.metrics["contract_success_rate"],
                    "mean_bid": ff_evaluation.metrics["mean_bid"],
                    "passed_out_rate": ff_evaluation.metrics["passed_out_rate"],
                    "pass_action_rate": ff_evaluation.metrics["pass_action_rate"],
                    "illegal_action_rate": ff_evaluation.metrics["illegal_action_rate"],
                    "no_trump_action_rate": ff_evaluation.metrics["no_trump_action_rate"],
                },
                "paired_difference": comparison.as_dict(),
            }
            pooled_differences[anchor_name].extend(
                (np_gru - np_ff)
                for np_gru, np_ff in zip(
                    gru_evaluation.paired_scores, ff_evaluation.paired_scores, strict=True
                )
            )
            per_anchor_gru[anchor_name].append(gru_evaluation.mean_paired_score)
            per_anchor_ff[anchor_name].append(ff_evaluation.mean_paired_score)

            print(
                f"[a1 s{seed_index}] vs {anchor_name:<7} "
                f"GRU {gru_evaluation.mean_paired_score:+8.4f} "
                f"FF {ff_evaluation.mean_paired_score:+8.4f} "
                f"diff {comparison.difference_mean:+8.4f} "
                f"[{comparison.difference_ci['lower_bound']:+.4f}, "
                f"{comparison.difference_ci['upper_bound']:+.4f}] "
                f"-> {comparison.verdict}",
                flush=True,
            )

        report.per_seed.append(seed_entry)

    # Pooled across seeds: every per-pair difference from every seed.
    from tarneeb.evaluation.stats import bootstrap_mean_confidence_interval

    for anchor_name in ANCHORS:
        differences = pooled_differences[anchor_name]
        interval = bootstrap_mean_confidence_interval(
            differences, seed=FINAL_HELD_OUT_SEED, resamples=10_000
        )
        lower, upper = interval.lower_bound, interval.upper_bound
        verdict = (
            "no detectable difference"
            if lower <= 0.0 <= upper
            else ("GRU better" if interval.mean > 0 else "GRU worse")
        )
        report.pooled[anchor_name] = {
            "gru_mean": float(sum(per_anchor_gru[anchor_name]) / len(per_anchor_gru[anchor_name])),
            "ff_mean": float(sum(per_anchor_ff[anchor_name]) / len(per_anchor_ff[anchor_name])),
            "difference_mean": interval.mean,
            "difference_ci": interval.as_dict(),
            "pair_count": len(differences),
            "verdict": verdict,
        }
        report.seed_stability[anchor_name] = {
            "gru": summarise_seed_stability(per_anchor_gru[anchor_name]),
            "ff": summarise_seed_stability(per_anchor_ff[anchor_name]),
        }
        print(
            f"[a1 pooled] vs {anchor_name:<7} diff {interval.mean:+8.4f} "
            f"[{lower:+.4f}, {upper:+.4f}] over {len(differences)} pairs -> {verdict}",
            flush=True,
        )

    # Head to head, seed 0 against seed 0.
    gru_agent, _ = load_agent(
        arguments.checkpoint_directory, "gru", arguments.seeds[0], arguments.updates, "gru"
    )
    ff_agent, _ = load_agent(
        arguments.checkpoint_directory, "ff", arguments.seeds[0], arguments.updates, "ff"
    )
    report.head_to_head_result = head_to_head(
        gru_agent,
        ff_agent,
        arguments.head_to_head_deals,
        FINAL_HELD_OUT_SEED,
        bootstrap_seed=FINAL_HELD_OUT_SEED,
    )
    print(
        f"[a1 h2h] GRU vs FF {report.head_to_head_result['mean_paired_score_difference']:+.4f} "
        f"win {report.head_to_head_result['win_rate']:.4f}",
        flush=True,
    )

    anchor = build_anchor("strong")
    report.latency = {
        "gru": measure_inference_latency(
            gru_agent, arguments.latency_deals, FINAL_HELD_OUT_SEED, anchor
        ),
        "ff": measure_inference_latency(
            ff_agent, arguments.latency_deals, FINAL_HELD_OUT_SEED, build_anchor("strong")
        ),
    }
    print(
        f"[a1 latency] GRU {report.latency['gru']['milliseconds_per_decision']:.3f} ms/decision "
        f"| FF {report.latency['ff']['milliseconds_per_decision']:.3f} ms/decision",
        flush=True,
    )

    summary_path = arguments.results_directory / "training_summary.json"
    if summary_path.exists():
        runs = json.loads(summary_path.read_text(encoding="utf-8"))
        report.throughput = {
            architecture: summarise_seed_stability(
                [
                    run["environment_steps_per_second"]
                    for run in runs
                    if run["architecture"] == architecture
                ]
            )
            for architecture in ("gru", "ff")
        }

    report.notes = [
        "Checkpoints are the pre-registered final update, selected by design.",
        "The final held-out seed was consulted once, after selection was frozen.",
        "Both conditions were BC-initialized from identical demonstrations.",
        "The feedforward history path has MORE parameters (329,536 vs 255,112), "
        "so a GRU win could not be explained by capacity.",
        "Three seeds bound what is detectable; a small true difference would not "
        "be resolvable at this budget and is not claimed either way.",
    ]

    arguments.results_directory.mkdir(parents=True, exist_ok=True)
    path = arguments.results_directory / "a1_report.json"
    path.write_text(json.dumps(report.as_dict(), indent=2, default=float), encoding="utf-8")
    print(f"\nA1 finished in {time.perf_counter() - started:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
