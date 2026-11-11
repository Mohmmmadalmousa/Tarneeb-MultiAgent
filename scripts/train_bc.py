"""Generate Strong-heuristic demonstrations and clone them (Phase 9, Part B).

Runs one architecture end to end: generate (or reuse) the three dataset splits,
train the actor by masked cross-entropy, report accuracy and likelihood on the
untouched test split, evaluate the cloned policy in the frozen arena, and write a
BC checkpoint plus a JSON report.

Teacher-label accuracy is deliberately **not** treated as the result. A policy
can imitate a heuristic closely and still play worse than it, because imitation
errors compound over a deal in a way per-decision accuracy does not show. So the
arena evaluation is part of this script rather than an optional follow-up.

Usage::

    python scripts/train_bc.py --architecture gru --train-deals 2500
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
from tarneeb.core.rules import RulesConfig
from tarneeb.evaluation.arena import Arena
from tarneeb.models.bc_checkpoint import save_bc_checkpoint
from tarneeb.models.nets import build_actor_and_critic, count_parameters
from tarneeb.models.recurrent import build_recurrent_actor_and_critic
from tarneeb.training.bc import (
    BCConfig,
    evaluate_bc,
    generate_bc_dataset,
    iterate_batches,
    train_behaviour_cloning,
)
from tarneeb.training.phase9_seeds import (
    BC_TEST_SEEDS,
    BC_TRAIN_SEEDS,
    BC_VALIDATION_SEEDS,
    PPO_DEVELOPMENT_SEEDS,
    describe_splits,
)

RULES = RulesConfig()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 behaviour cloning")
    parser.add_argument("--architecture", choices=("gru", "ff"), default="gru")
    parser.add_argument("--train-deals", type=int, default=2500)
    parser.add_argument("--validation-deals", type=int, default=500)
    parser.add_argument("--test-deals", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-sequences", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initialization-seed", type=int, default=1)
    parser.add_argument("--arena-pairs", type=int, default=200)
    parser.add_argument("--checkpoint-directory", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-directory", type=Path, default=Path("results/phase9"))
    parser.add_argument("--dataset-directory", type=Path, default=Path("data/bc"))
    return parser.parse_args()


def arena_evaluation(actor, architecture: str, paired_deals: int) -> dict:
    """Measure the cloned policy against all three frozen anchors.

    Uses development seeds only. The final held-out seed is never consulted
    during Phase 9 development.
    """
    agent = (
        RecurrentNeuralAgent(actor, name=f"bc_{architecture}", deterministic=True)
        if architecture == "gru"
        else NeuralAgent(actor, name=f"bc_{architecture}", deterministic=True)
    )
    opponents = {
        "random": RandomAgent(seed=0),
        "basic": BasicHeuristicAgent(),
        "strong": StrongHeuristicAgent(),
    }
    results: dict[str, dict] = {}
    for name, opponent in opponents.items():
        arena = Arena()
        agent.restart()
        evaluation = arena.evaluate(
            agent,
            opponent,
            paired_deal_count=paired_deals,
            evaluation_seed=PPO_DEVELOPMENT_SEEDS.start,
            bootstrap_resamples=2000,
        )
        metrics = evaluation.metrics
        results[name] = {
            "mean_paired_score_difference": metrics["mean_paired_score_difference"],
            "paired_score_difference_ci": metrics["paired_score_difference_ci"],
            "win_rate": metrics["win_rate"],
            "win_rate_wilson_interval": metrics["win_rate_wilson_interval"],
            "contract_success_rate": metrics["contract_success_rate"],
            "mean_bid": metrics["mean_bid"],
            "passed_out_rate": metrics["passed_out_rate"],
            "pass_action_rate": metrics["pass_action_rate"],
            "role_split_mean_score": metrics["role_split_mean_score"],
            "illegal_action_rate": metrics["illegal_action_rate"],
            "no_trump_action_rate": metrics["no_trump_action_rate"],
            "replay_integrity_rate": metrics["replay_integrity_rate"],
        }
    return results


def main() -> None:
    arguments = parse_arguments()
    architecture = arguments.architecture
    start = time.perf_counter()

    train_seeds = range(BC_TRAIN_SEEDS.start, BC_TRAIN_SEEDS.start + arguments.train_deals)
    validation_seeds = range(
        BC_VALIDATION_SEEDS.start, BC_VALIDATION_SEEDS.start + arguments.validation_deals
    )
    test_seeds = range(BC_TEST_SEEDS.start, BC_TEST_SEEDS.start + arguments.test_deals)

    print(f"[bc:{architecture}] generating demonstrations from the Strong heuristic", flush=True)
    datasets = {
        "train": generate_bc_dataset(train_seeds, "bc_train", RULES),
        "validation": generate_bc_dataset(validation_seeds, "bc_validation", RULES),
        "test": generate_bc_dataset(test_seeds, "bc_test", RULES),
    }
    arguments.dataset_directory.mkdir(parents=True, exist_ok=True)
    for name, dataset in datasets.items():
        dataset.save(arguments.dataset_directory / f"bc_{name}.npz")
        print(
            f"  {name:<10} {len(dataset):>5} deals  {dataset.metadata.decision_count:>7} decisions",
            flush=True,
        )

    print(f"[bc:{architecture}] building batches", flush=True)
    batches = {
        name: iterate_batches(dataset, arguments.batch_sequences, architecture=architecture)
        for name, dataset in datasets.items()
    }

    actor = (
        build_recurrent_actor_and_critic(seed=arguments.initialization_seed)[0]
        if architecture == "gru"
        else build_actor_and_critic(seed=arguments.initialization_seed)[0]
    )
    config = BCConfig(
        learning_rate=arguments.learning_rate,
        epochs=arguments.epochs,
        batch_sequences=arguments.batch_sequences,
        architecture=architecture,
    )

    print(f"[bc:{architecture}] cloning ({count_parameters(actor):,} parameters)", flush=True)
    result = train_behaviour_cloning(actor, batches["train"], batches["validation"], config)
    for index, (train_metrics, validation_metrics) in enumerate(
        zip(result.train_metrics, result.validation_metrics, strict=True), start=1
    ):
        print(
            f"  epoch {index}: train acc {train_metrics.accuracy:.4f} "
            f"| val acc {validation_metrics.accuracy:.4f} "
            f"nll {validation_metrics.negative_log_likelihood:.4f}",
            flush=True,
        )

    test_metrics = evaluate_bc(actor, batches["test"])
    print(
        f"[bc:{architecture}] TEST acc {test_metrics.accuracy:.4f} "
        f"nll {test_metrics.negative_log_likelihood:.4f} "
        f"phase {test_metrics.phase_accuracy()}",
        flush=True,
    )

    print(f"[bc:{architecture}] arena evaluation on development seeds", flush=True)
    arena_results = arena_evaluation(actor, architecture, arguments.arena_pairs)
    for name, entry in arena_results.items():
        print(
            f"  vs {name:<7} paired {entry['mean_paired_score_difference']:+8.4f} "
            f"win {entry['win_rate']:.4f}",
            flush=True,
        )

    checkpoint_path = arguments.checkpoint_directory / f"phase9_bc_{architecture}.pt"
    save_bc_checkpoint(
        path=checkpoint_path,
        actor=actor,
        architecture=architecture,
        rules=RULES,
        bc_config=config.as_dict(),
        seed_split=describe_splits(),
        metrics={
            "train": [m.as_dict() for m in result.train_metrics],
            "validation": [m.as_dict() for m in result.validation_metrics],
            "test": test_metrics.as_dict(),
            "arena_development": arena_results,
        },
        dataset_metadata=datasets["train"].metadata.as_dict(),
    )

    report = {
        "architecture": architecture,
        "parameter_count": count_parameters(actor),
        "config": config.as_dict(),
        "dataset": {name: dataset.metadata.as_dict() for name, dataset in datasets.items()},
        "seed_split": describe_splits(),
        "train_metrics": [m.as_dict() for m in result.train_metrics],
        "validation_metrics": [m.as_dict() for m in result.validation_metrics],
        "test_metrics": test_metrics.as_dict(),
        "arena_development": arena_results,
        "optimizer_steps": result.optimizer_steps,
        "training_seconds": result.wall_clock_seconds,
        "total_seconds": time.perf_counter() - start,
        "checkpoint": str(checkpoint_path),
    }
    arguments.results_directory.mkdir(parents=True, exist_ok=True)
    report_path = arguments.results_directory / f"bc_{architecture}.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"[bc:{architecture}] wrote {checkpoint_path} and {report_path}", flush=True)
    print(f"[bc:{architecture}] done in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
