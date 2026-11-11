"""Run the pre-registered Phase 10 runs: ablation A2, centralized vs decentralized critic.

One invocation runs one or more (critic_mode, seed_index) conditions
sequentially. Sequential rather than parallel for the same reason as Phase 9:
PyTorch already uses several threads per process, so running conditions side by
side would make the throughput numbers meaningless.

The budget, seeds, evaluation cadence and stopping rule are fixed in
docs/PHASE10_DECISIONS.md before any of this runs. The defaults here match that
document, and they are printed into every report.

Usage::

    python scripts/train_phase10.py --critic-modes decentralized centralized --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tarneeb.models.centralized_critic import CRITIC_MODES, describe_condition
from tarneeb.training.phase10_training import (
    Phase10Config,
    run_phase10_training,
    write_phase10_report,
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
    parser = argparse.ArgumentParser(description="Phase 10 ablation A2")
    parser.add_argument(
        "--critic-modes", nargs="+", choices=CRITIC_MODES, default=list(CRITIC_MODES)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
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
    parser.add_argument("--bc-checkpoint", type=Path, default=Path("checkpoints/phase9_bc_gru.pt"))
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-directory", type=Path, default=Path("results/phase10"))
    return parser.parse_args()


def build_config(arguments: argparse.Namespace, critic_mode: str, seed_index: int) -> Phase10Config:
    return Phase10Config(
        run_name=f"phase10_{critic_mode}_s{seed_index}",
        critic_mode=critic_mode,
        environment_count=arguments.environment_count,
        rollout_length=arguments.rollout_length,
        total_updates=arguments.total_updates,
        training_opponent=arguments.training_opponent,
        bc_checkpoint=arguments.bc_checkpoint,
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

    if not arguments.bc_checkpoint.exists():
        raise FileNotFoundError(
            f"{arguments.bc_checkpoint} is missing; Phase 10 initializes both A2 "
            "conditions from the same Phase 9 recurrent BC checkpoint"
        )

    for critic_mode in arguments.critic_modes:
        for seed_index in arguments.seeds:
            config = build_config(arguments, critic_mode, seed_index)
            condition = describe_condition(critic_mode)
            print(
                f"\n=== {config.run_name} | critic {condition['critic_observation_dim']}d "
                f"({condition['critic_parameters']:,} params) | actor "
                f"{condition['actor_observation_dim']}d | {config.total_updates} updates "
                f"| root seed {config.environment_root_seed} ===",
                flush=True,
            )

            result = run_phase10_training(config)
            path = write_phase10_report(config, result)

            final = result.evaluations[-1]["opponents"] if result.evaluations else {}
            summary.append(
                {
                    "run_name": result.run_name,
                    "critic_mode": critic_mode,
                    "seed_index": seed_index,
                    "environment_steps": result.total_environment_steps,
                    "collection_steps": result.total_collection_steps,
                    "learner_decisions": result.learner_decisions,
                    "completed_deals": result.completed_deals,
                    "wall_clock_seconds": result.wall_clock_seconds,
                    "environment_steps_per_second": result.environment_steps_per_second,
                    "mean_explained_variance": result.mean_explained_variance,
                    "final_explained_variance": result.final_explained_variance,
                    "worst_epoch_zero_deviation": result.worst_epoch_zero_deviation,
                    "illegal_action_count": result.illegal_action_count,
                    "no_trump_action_count": result.no_trump_action_count,
                    "non_finite_update_count": result.non_finite_update_count,
                    "parameter_counts": result.parameter_counts,
                    "final_evaluation": {
                        name: {
                            "mean_paired_score_difference": v["mean_paired_score_difference"],
                            "win_rate": v["win_rate"],
                        }
                        for name, v in final.items()
                    },
                    "report": str(path),
                }
            )
            print(
                f"[{result.run_name}] done in {result.wall_clock_seconds:.0f}s "
                f"({result.environment_steps_per_second:.0f} env steps/s) "
                f"final_ev {result.final_explained_variance:+.3f} -> {path}",
                flush=True,
            )

    arguments.results_directory.mkdir(parents=True, exist_ok=True)
    summary_path = arguments.results_directory / "training_summary.json"
    existing = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
    summary_path.write_text(
        json.dumps(existing + summary, indent=2, default=float), encoding="utf-8"
    )
    print(
        f"\nAll runs finished in {time.perf_counter() - started:.0f}s -> {summary_path}", flush=True
    )


if __name__ == "__main__":
    main()
