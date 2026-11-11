"""Run one Phase 6 feedforward masked-PPO training job.

Usage::

    python scripts/train_ppo.py --config configs/train/ppo_ff_baseline.yaml
    python scripts/train_ppo.py --config configs/train/ppo_ff_smoke.yaml

Writes, per run, into ``<output>/<run name>/``:

- ``updates.jsonl``  -- one JSON record per PPO update, flushed as it happens;
- ``tensorboard/``   -- the same scalars as TensorBoard curves;
- ``summary.json``   -- the run configuration, contracts and evaluations.

``--no-logging`` disables both sinks. Learning is bit-identical either way: the
sinks are passed into :func:`~tarneeb.training.train.train` as optional
callables, and nothing inside PPO or the rollout collector knows they exist.
That is the observability boundary project requirements requires.

``--resume-from <checkpoint>`` continues an interrupted run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tarneeb.core.rules import RulesConfig
from tarneeb.models.checkpoint import expected_rules_versions
from tarneeb.models.nets import build_actor_and_critic, count_parameters
from tarneeb.training.config import TrainingConfig, load_training_config
from tarneeb.training.reporting import (
    JsonlReporter,
    TensorBoardReporter,
    derived_safety_rates,
)
from tarneeb.training.train import train


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/ppo_ff_baseline.yaml"))
    parser.add_argument("--total-environment-steps", type=int, default=None)
    parser.add_argument("--environment-count", type=int, default=None)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--environment-root-seed", type=int, default=None)
    parser.add_argument("--initialization-seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-directory", type=Path, default=Path("runs"))
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="run with every metric sink disabled; learning is unaffected",
    )
    return parser.parse_args(argv)


def apply_overrides(config: TrainingConfig, arguments: argparse.Namespace) -> TrainingConfig:
    """Return ``config`` with any command-line overrides applied."""
    overrides: dict[str, Any] = {}
    for attribute in (
        "total_environment_steps",
        "environment_count",
        "rollout_length",
        "environment_root_seed",
        "initialization_seed",
        "run_name",
    ):
        value = getattr(arguments, attribute)
        if value is not None:
            overrides[attribute] = value
    return config.with_overrides(**overrides) if overrides else config


def format_update_line(record: dict[str, Any]) -> str:
    def number(key: str, digits: int = 4) -> str:
        value = record.get(key)
        return "n/a" if value is None else f"{value:.{digits}f}"

    return (
        f"update {record['update']:>4}  "
        f"steps {record['global_environment_steps']:>9,}  "
        f"deals {record['completed_deals']:>4}  "
        f"pi {number('policy_loss')}  "
        f"v {number('value_loss')}  "
        f"H {number('entropy', 3)}  "
        f"kl {number('approximate_kl_divergence', 5)}  "
        f"clip {number('clip_fraction', 3)}  "
        f"ev {number('explained_variance', 3)}  "
        f"gn {number('gradient_norm', 3)}  "
        f"bid {number('mean_bid', 2)}  "
        f"made {number('contract_success_rate', 3)}  "
        f"sps {record['environment_steps_per_second']:.0f}"
    )


def format_evaluation_line(record: dict[str, Any]) -> str:
    interval = record["paired_score_difference_ci"]
    win_interval = record["win_rate_wilson_interval"]
    return (
        f"  eval vs {record['opponent']}: "
        f"paired diff {record['mean_paired_score_difference']:+.4f} "
        f"95% CI [{interval['lower_bound']:+.4f}, {interval['upper_bound']:+.4f}]  "
        f"win rate {record['win_rate']:.4f} "
        f"Wilson [{win_interval['lower_bound']:.4f}, {win_interval['upper_bound']:.4f}]  "
        f"illegal {record['illegal_action_rate']:.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config = apply_overrides(load_training_config(arguments.config), arguments)

    output_directory = arguments.output_directory / config.run_name
    output_directory.mkdir(parents=True, exist_ok=True)

    actor, critic = build_actor_and_critic(config.architecture, seed=config.initialization_seed)
    print(f"config              : {arguments.config}")
    print(f"run name            : {config.run_name}")
    print(f"contracts           : {expected_rules_versions()}")
    print(f"rules fingerprint   : {RulesConfig().fingerprint()}")
    print(f"actor parameters    : {count_parameters(actor):,}")
    print(f"critic parameters   : {count_parameters(critic):,}")
    print(f"environments        : {config.environment_count}")
    print(f"rollout length      : {config.rollout_length}")
    print(f"decisions / rollout : {config.decisions_per_rollout:,}")
    print(f"planned updates     : {config.total_updates}")
    print(f"seeds               : {config.seeds()}")
    print(f"training opponent   : {config.training_opponent}")
    print(f"logging             : {'disabled' if arguments.no_logging else 'jsonl + tensorboard'}")
    if arguments.resume_from is not None:
        print(f"resuming from       : {arguments.resume_from}")
    print()

    jsonl_reporter: JsonlReporter | None = None
    tensorboard_reporter: TensorBoardReporter | None = None
    if not arguments.no_logging:
        jsonl_reporter = JsonlReporter(output_directory / "updates.jsonl")
        tensorboard_reporter = TensorBoardReporter(output_directory / "tensorboard")

    def report_update(record: dict[str, Any]) -> None:
        print(format_update_line(record))
        if jsonl_reporter is not None:
            jsonl_reporter(record)
        if tensorboard_reporter is not None:
            tensorboard_reporter(record)

    def report_evaluation(record: dict[str, Any]) -> None:
        print(format_evaluation_line(record))
        if tensorboard_reporter is not None:
            tensorboard_reporter.add_evaluation(derived_safety_rates(record))

    try:
        result = train(
            config,
            report=report_update,
            report_evaluation=report_evaluation,
            resume_from=arguments.resume_from,
        )
    finally:
        if jsonl_reporter is not None:
            jsonl_reporter.close()
        if tensorboard_reporter is not None:
            tensorboard_reporter.close()

    summary = {
        "run_name": config.run_name,
        "config": result.config,
        "total_environment_steps": result.total_environment_steps,
        "wall_clock_seconds": result.wall_clock_seconds,
        "environment_steps_per_second": result.environment_steps_per_second,
        "checkpoint_paths": result.checkpoint_paths,
        "evaluations": result.evaluation_records,
        "contracts": expected_rules_versions(),
        "rules_fingerprint": RulesConfig().fingerprint(),
        "resumed_from": result.resumed_from,
        "resumed_at_update": result.resumed_at_update,
        "logging_enabled": not arguments.no_logging,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print(f"total steps    : {result.total_environment_steps:,}")
    print(f"wall clock     : {result.wall_clock_seconds:.1f}s")
    print(f"throughput     : {result.environment_steps_per_second:.0f} env steps/s")
    print(f"checkpoints    : {len(result.checkpoint_paths)} -> {result.checkpoint_paths[-1:]}")
    if jsonl_reporter is not None:
        print(f"jsonl          : {jsonl_reporter.path}")
    if tensorboard_reporter is not None:
        print(f"tensorboard    : {tensorboard_reporter.log_directory}")
    print(f"summary        : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
