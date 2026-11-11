"""Run the Phase 5 baseline evaluations described by a config file.

Usage::

    python scripts/run_baseline_evaluation.py --config configs/eval/baselines.yaml
    python scripts/run_baseline_evaluation.py --paired-deal-count 200   # quick check

Writes one JSON report and one per-pair CSV per experiment, plus a summary CSV,
into the configured output directory. Prints the headline numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from tarneeb.agents.base import Agent
from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.random_agent import RandomAgent
from tarneeb.core.rules import RulesConfig
from tarneeb.evaluation.arena import Arena
from tarneeb.evaluation.report import EvaluationReport, write_summary_csv

AGENT_CLASSES: dict[str, type[Agent]] = {
    "RandomAgent": RandomAgent,
    "BasicHeuristicAgent": BasicHeuristicAgent,
    "StrongHeuristicAgent": StrongHeuristicAgent,
}


def build_agent(agent_name: str, agent_config: dict[str, Any]) -> Agent:
    """Instantiate one configured agent."""
    settings = dict(agent_config)
    class_name = settings.pop("class")
    agent_class = AGENT_CLASSES[class_name]
    return agent_class(name=agent_name, **settings)  # type: ignore[call-arg]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval/baselines.yaml"))
    parser.add_argument("--paired-deal-count", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=None)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--experiment", type=str, default=None, help="run only this experiment")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))

    evaluation_config = config["evaluation"]
    paired_deal_count = arguments.paired_deal_count or evaluation_config["paired_deal_count"]
    evaluation_seed = (
        arguments.evaluation_seed
        if arguments.evaluation_seed is not None
        else evaluation_config["evaluation_seed"]
    )
    bootstrap_resamples = arguments.bootstrap_resamples or evaluation_config["bootstrap_resamples"]
    replay_sample_rate = evaluation_config["replay_verification_sample_rate"]

    output_directory = arguments.output_directory or Path(config["output"]["directory"])
    rules = RulesConfig()
    reports: list[EvaluationReport] = []

    experiments = config["experiments"]
    if arguments.experiment:
        experiments = [item for item in experiments if item["name"] == arguments.experiment]
        if not experiments:
            print(f"no experiment named {arguments.experiment!r}", file=sys.stderr)
            return 1

    print(
        f"paired_deal_count={paired_deal_count}  evaluation_seed={evaluation_seed}  "
        f"bootstrap_resamples={bootstrap_resamples}  replay_rate={replay_sample_rate}"
    )
    print()

    for experiment in experiments:
        policy_a_key = experiment["policy_a"]
        policy_b_key = experiment["policy_b"]
        policy_a_config = config["agents"][policy_a_key]
        policy_b_config = config["agents"][policy_b_key]

        # Distinct instances even when the two policies are the same class, so
        # the two sides never share RNG state.
        policy_a = build_agent(f"{policy_a_key}_a", policy_a_config)
        policy_b = build_agent(f"{policy_b_key}_b", policy_b_config)

        arena = Arena(rules=rules, replay_sample_rate=replay_sample_rate)
        result = arena.evaluate(
            policy_a,
            policy_b,
            paired_deal_count=paired_deal_count,
            evaluation_seed=evaluation_seed,
            bootstrap_resamples=bootstrap_resamples,
        )

        report = EvaluationReport(
            result=result,
            rules=rules,
            policy_a_config=policy_a_config,
            policy_b_config=policy_b_config,
        )
        reports.append(report)

        if config["output"].get("write_json", True):
            report.write_json(output_directory / f"{experiment['name']}.json")
        if config["output"].get("write_pairs_csv", True):
            report.write_pairs_csv(output_directory / f"{experiment['name']}_pairs.csv")

        metrics = result.metrics
        interval = metrics["paired_score_difference_ci"]
        win_interval = metrics["win_rate_wilson_interval"] or {}
        print(f"--- {experiment['name']} ---")
        print(
            f"  mean paired score difference : {metrics['mean_paired_score_difference']:+.4f}"
            f"  95% CI [{interval['lower_bound']:+.4f}, {interval['upper_bound']:+.4f}]"
        )
        print(f"  sigma_d                      : {metrics['sigma_d']:.4f}")
        if metrics["win_rate"] is None:
            print("  win rate                     : undefined (every pair tied)")
        else:
            print(
                f"  win rate                     : {metrics['win_rate']:.4f}"
                f"  Wilson [{win_interval['lower_bound']:.4f}, "
                f"{win_interval['upper_bound']:.4f}]"
                f"  (ties {metrics['tie_count']})"
            )
        print(
            f"  contract success             : {metrics['contract_success_rate']}"
            f"  mean bid {metrics['mean_bid']}"
            f"  passed-out rate {metrics['passed_out_rate']:.4f}"
        )
        print(
            f"  illegal rate {metrics['illegal_action_rate']}"
            f"  replay integrity {metrics['replay_integrity_rate']}"
            f"  ({metrics['replay_verified_count']} deals verified)"
        )
        print(
            f"  runtime {metrics['runtime_seconds']:.2f}s"
            f"  {metrics['deals_per_second']:.1f} deals/s"
            f"  {metrics['decisions_per_second']:.0f} decisions/s"
            f"  {metrics['paired_deals_per_second']:.1f} pairs/s"
        )
        print()

    if config["output"].get("write_summary_csv", True) and reports:
        summary_path = write_summary_csv(reports, output_directory / "summary.csv")
        print(f"wrote {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
