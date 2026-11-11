"""Run the pre-registered Phase 9 training runs (Part C and A1).

One invocation runs one or more `(architecture, seed_index)` conditions
sequentially. Sequential rather than parallel on purpose: PyTorch already uses
several threads per process, so running the conditions side by side on one CPU
would make the throughput numbers meaningless and could perturb the wall-clock
comparison that A1 reports.

The budget, seeds, evaluation cadence and stopping rule are fixed in
``docs/PHASE9_DECISIONS.md`` before any of this runs. The defaults here match
that document; overriding them makes a run something other than the
pre-registered one, which is why they are printed into every report.

Usage::

    python scripts/train_phase9.py --architectures gru ff --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tarneeb.training.phase9_training import (
    Phase9Config,
    run_phase9_training,
    write_run_report,
)
from tarneeb.training.ppo import PPOConfig

PRE_REGISTERED = {
    "environment_count": 16,
    "rollout_length": 128,
    "total_updates": 250,
    "evaluation_every_updates": 50,
    "evaluation_paired_deals": 150,
    "training_opponent": "basic",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 principal training")
    parser.add_argument("--architectures", nargs="+", choices=("gru", "ff"), default=["gru"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--environment-count", type=int, default=PRE_REGISTERED["environment_count"]
    )
    parser.add_argument("--rollout-length", type=int, default=PRE_REGISTERED["rollout_length"])
    parser.add_argument("--total-updates", type=int, default=PRE_REGISTERED["total_updates"])
    parser.add_argument(
        "--evaluation-every-updates", type=int, default=PRE_REGISTERED["evaluation_every_updates"]
    )
    parser.add_argument(
        "--evaluation-paired-deals", type=int, default=PRE_REGISTERED["evaluation_paired_deals"]
    )
    parser.add_argument("--training-opponent", default=PRE_REGISTERED["training_opponent"])
    parser.add_argument("--bc-checkpoint-directory", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-directory", type=Path, default=Path("results/phase9"))
    parser.add_argument(
        "--no-bc",
        action="store_true",
        help="random initialization instead of BC; the documented control condition",
    )
    return parser.parse_args()


def build_config(arguments: argparse.Namespace, architecture: str, seed_index: int) -> Phase9Config:
    bc_checkpoint = (
        None
        if arguments.no_bc
        else arguments.bc_checkpoint_directory / f"phase9_bc_{architecture}.pt"
    )
    suffix = "nobc" if arguments.no_bc else "bc"
    return Phase9Config(
        run_name=f"phase9_{architecture}_{suffix}_s{seed_index}",
        architecture=architecture,
        environment_count=arguments.environment_count,
        rollout_length=arguments.rollout_length,
        total_updates=arguments.total_updates,
        training_opponent=arguments.training_opponent,
        bc_checkpoint=bc_checkpoint,
        seed_index=seed_index,
        evaluation_every_updates=arguments.evaluation_every_updates,
        evaluation_paired_deals=arguments.evaluation_paired_deals,
        checkpoint_directory=arguments.checkpoint_directory,
        results_directory=arguments.results_directory,
        ppo=PPOConfig(),
    )


def main() -> None:
    arguments = parse_arguments()
    started = time.perf_counter()
    summary: list[dict] = []

    for architecture in arguments.architectures:
        for seed_index in arguments.seeds:
            config = build_config(arguments, architecture, seed_index)
            print(
                f"\n=== {config.run_name} | {config.total_updates} updates "
                f"x {config.steps_per_update} steps = {config.total_environment_steps:,} "
                f"| root seed {config.environment_root_seed} ===",
                flush=True,
            )
            if config.bc_checkpoint is not None and not config.bc_checkpoint.exists():
                raise FileNotFoundError(
                    f"{config.bc_checkpoint} is missing; run scripts/train_bc.py first"
                )

            result = run_phase9_training(config)
            path = write_run_report(config, result)

            final = result.evaluations[-1]["opponents"] if result.evaluations else {}
            summary.append(
                {
                    "run_name": result.run_name,
                    "architecture": architecture,
                    "seed_index": seed_index,
                    "environment_steps": result.total_environment_steps,
                    "learner_decisions": result.learner_decisions,
                    "wall_clock_seconds": result.wall_clock_seconds,
                    "environment_steps_per_second": result.decisions_per_second,
                    "worst_epoch_zero_deviation": result.worst_epoch_zero_deviation,
                    "illegal_action_count": result.illegal_action_count,
                    "no_trump_action_count": result.no_trump_action_count,
                    "non_finite_update_count": result.non_finite_update_count,
                    "final_evaluation": {
                        name: {
                            "mean_paired_score_difference": values["mean_paired_score_difference"],
                            "win_rate": values["win_rate"],
                        }
                        for name, values in final.items()
                    },
                    "report": str(path),
                }
            )
            print(
                f"[{result.run_name}] done in {result.wall_clock_seconds:.0f}s "
                f"({result.decisions_per_second:.0f} env steps/s) -> {path}",
                flush=True,
            )

    arguments.results_directory.mkdir(parents=True, exist_ok=True)
    summary_path = arguments.results_directory / "training_summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    summary_path.write_text(
        json.dumps(existing + summary, indent=2, default=float), encoding="utf-8"
    )
    print(
        f"\nAll runs finished in {time.perf_counter() - started:.0f}s -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
