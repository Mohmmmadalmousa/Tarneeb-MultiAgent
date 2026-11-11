"""Run one Phase 7 curriculum stage.

Usage::

    python scripts/train_stage.py --config configs/train/stage2.yaml
    python scripts/train_stage.py --config configs/train/stage3.yaml \\
        --initialize-from checkpoints/stage2_trick_play_update12.pt
    python scripts/train_stage.py --config configs/ablation/a7_no_auction.yaml

Like the Phase 6 training script, this is the only place that prints or writes
files. :func:`~tarneeb.curriculum.trainer.train_curriculum_stage` returns plain
dictionaries, and ``--no-logging`` removes every sink without changing learning.

Writes ``updates.jsonl``, a TensorBoard directory and an ``experiment.json``
record carrying the full provenance of the run.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tarneeb.core.rules import RulesConfig
from tarneeb.curriculum.config import describe_stage_configuration, load_stage_configuration
from tarneeb.curriculum.trainer import parameter_summary, train_curriculum_stage
from tarneeb.models.checkpoint import expected_rules_versions
from tarneeb.training.reporting import (
    JsonlReporter,
    TensorBoardReporter,
    derived_safety_rates,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help="start from a checkpoint's WEIGHTS only; optimizer, streams and LR restart",
    )
    parser.add_argument(
        "--resume-training-state-from",
        type=Path,
        default=None,
        help=(
            "partial continuation of the SAME task: restores actor, critic and "
            "optimizer and continues the learning-rate schedule. The environment "
            "and RNG streams are NOT restored, so this is not a deterministic resume."
        ),
    )
    parser.add_argument("--total-environment-steps", type=int, default=None)
    parser.add_argument("--environment-count", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-directory", type=Path, default=Path("runs"))
    parser.add_argument("--no-logging", action="store_true")
    return parser.parse_args(argv)


def working_tree_state() -> str:
    """Return whether the checkout is clean, so provenance is never overstated."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return "unknown"
    return "clean" if not completed.stdout.strip() else "dirty (uncommitted Phase 7 source)"


def current_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return None
    return completed.stdout.strip() or None


def format_update_line(record: dict[str, Any]) -> str:
    def number(key: str, digits: int = 4) -> str:
        value = record.get(key)
        return "n/a" if value is None else f"{value:.{digits}f}"

    return (
        f"update {record.get('effective_update', record['update']):>4}  "
        f"steps {record['global_environment_steps']:>9,}  "
        f"deals {record['completed_deals']:>4}  "
        f"pi {number('policy_loss')}  "
        f"v {number('value_loss')}  "
        f"H {number('entropy', 3)}  "
        f"kl {number('approximate_kl_divergence', 5)}  "
        f"ev {number('explained_variance', 3)}  "
        f"bid {number('mean_bid', 2)}  "
        f"pass {number('pass_rate', 3)}  "
        f"made {number('contract_success_rate', 3)}  "
        f"R {number('mean_learner_reward', 4)}  "
        f"sps {record['environment_steps_per_second']:.0f}"
    )


def format_evaluation_line(record: dict[str, Any]) -> str:
    interval = record["paired_score_difference_ci"]
    wilson = record.get("win_rate_wilson_interval")
    win_rate = record.get("win_rate")
    win_text = "n/a" if win_rate is None else f"{win_rate:.4f}"
    wilson_text = (
        ""
        if wilson is None
        else f" Wilson [{wilson['lower_bound']:.4f}, {wilson['upper_bound']:.4f}]"
    )
    return (
        f"  eval@{record['update']} vs {record['policy_b_name']}: "
        f"paired {record['mean_paired_score_difference']:+.4f} "
        f"CI [{interval['lower_bound']:+.4f}, {interval['upper_bound']:+.4f}]  "
        f"win {win_text}{wilson_text}  illegal {record['illegal_action_rate']:.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config, stage = load_stage_configuration(arguments.config)

    overrides: dict[str, Any] = {}
    if arguments.total_environment_steps is not None:
        overrides["total_environment_steps"] = arguments.total_environment_steps
    if arguments.environment_count is not None:
        overrides["environment_count"] = arguments.environment_count
    if arguments.run_name is not None:
        overrides["run_name"] = arguments.run_name
    if overrides:
        config = config.with_overrides(**overrides)

    output_directory = arguments.output_directory / config.run_name
    output_directory.mkdir(parents=True, exist_ok=True)

    description = describe_stage_configuration(config, stage)
    parameters = parameter_summary(config)
    print(f"config              : {arguments.config}")
    print(f"stage               : {stage.name} (curriculum stage {stage.curriculum_stage})")
    print(f"auction             : {stage.auction_mode}")
    print(f"trump / contract    : {stage.trump_mode} / {stage.contract_mode}")
    print(f"opponent            : {stage.opponent}")
    print(f"reward              : {stage.reward_description}")
    print(f"history condition   : {stage.history_condition}")
    print(f"contracts           : {expected_rules_versions()}")
    print(f"rules fingerprint   : {RulesConfig().fingerprint()}")
    print(f"actor / critic      : {parameters['actor']:,} / {parameters['critic']:,}")
    print(f"environments        : {config.environment_count} x {config.rollout_length}")
    print(f"planned updates     : {config.total_updates}")
    print(f"seeds               : {config.seeds()}")
    print(f"git sha             : {current_git_sha()}  working tree: {working_tree_state()}")
    if arguments.initialize_from is not None:
        print(f"initialize from     : {arguments.initialize_from} (weights only)")
    if arguments.resume_training_state_from is not None:
        print(f"continue from       : {arguments.resume_training_state_from}")
        print("continuation mode   : weight + optimizer + LR-schedule, FRESH data/RNG stream")
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
        result = train_curriculum_stage(
            config=config,
            stage=stage,
            report=report_update,
            report_evaluation=report_evaluation,
            initialize_from=arguments.initialize_from,
            resume_training_state_from=arguments.resume_training_state_from,
        )
    finally:
        if jsonl_reporter is not None:
            jsonl_reporter.close()
        if tensorboard_reporter is not None:
            tensorboard_reporter.close()

    experiment_record = {
        "experiment_name": config.run_name,
        "stage": result.stage,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_sha": current_git_sha(),
        "working_tree": working_tree_state(),
        "contracts": expected_rules_versions(),
        "rules_fingerprint": RulesConfig().fingerprint(),
        "configuration": description,
        "parameter_counts": parameters,
        "initialized_from": result.initialized_from,
        "continuation": result.continuation,
        "starting_environment_steps": result.starting_environment_steps,
        "schedule_offset_updates": result.schedule_offset_updates,
        "scheduled_total_updates": result.scheduled_total_updates,
        "total_environment_steps": result.total_environment_steps,
        "total_learner_decisions": result.total_learner_decisions,
        "updates": len(result.update_records),
        "wall_clock_seconds": result.wall_clock_seconds,
        "environment_steps_per_second": result.environment_steps_per_second,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "checkpoint_paths": result.checkpoint_paths,
        "evaluations": result.evaluation_records,
    }
    record_path = output_directory / "experiment.json"
    record_path.write_text(json.dumps(experiment_record, indent=2, default=str), encoding="utf-8")

    print()
    print(f"steps          : {result.total_environment_steps:,}")
    print(f"learner moves  : {result.total_learner_decisions:,}")
    print(f"wall clock     : {result.wall_clock_seconds:.1f}s")
    print(f"throughput     : {result.environment_steps_per_second:.0f} env steps/s")
    print(f"checkpoints    : {len(result.checkpoint_paths)} -> {result.checkpoint_paths[-1:]}")
    print(f"experiment     : {record_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
