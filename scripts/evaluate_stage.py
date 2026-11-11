"""Evaluate a curriculum checkpoint on held-out deals.

Usage::

    python scripts/evaluate_stage.py --checkpoint checkpoints/stage2_trick_play_update144.pt \\
        --stage-config configs/train/stage2.yaml --opponents random \\
        --paired-deal-count 2500 --evaluation-seed 20250910

    python scripts/evaluate_stage.py --checkpoint checkpoints/stage6_full_bidding_update288.pt \\
        --stage-config configs/train/stage6.yaml --opponents random basic strong

Phase 6 already owns ``scripts/evaluate_checkpoint.py``, which evaluates the full
game. This script adds the one thing that file cannot do: evaluate under a
*curriculum stage's* conditions, where the auction is scripted and the result must
be split by trump suit. For a full-bidding stage it falls through to the frozen
arena unchanged.

A positive mean paired score difference means the checkpoint outscored the
baseline, in points per deal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tarneeb.core.rules import RulesConfig
from tarneeb.curriculum.agents import HistoryConditionedNeuralAgent
from tarneeb.curriculum.config import load_stage_configuration
from tarneeb.curriculum.evaluation import StageEvaluation, evaluate_stage
from tarneeb.models.checkpoint import load_checkpoint
from tarneeb.training.train import build_baseline_agent

FINAL_EVALUATION_SEED = 20250910
"""The seed reserved for reported results. Development tuning uses a different one."""


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", default=["random"])
    parser.add_argument("--paired-deal-count", type=int, default=1000)
    parser.add_argument("--evaluation-seed", type=int, default=FINAL_EVALUATION_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--deterministic-actions", action="store_true")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args(argv)


def print_evaluation(evaluation: StageEvaluation) -> None:
    metrics = evaluation.metrics
    interval = metrics["paired_score_difference_ci"]
    wilson = metrics.get("win_rate_wilson_interval")
    win_rate = metrics.get("win_rate")

    print(f"--- {evaluation.policy_a_name} vs {evaluation.policy_b_name} ---")
    print(
        f"  mean paired score difference : {metrics['mean_paired_score_difference']:+.4f}  "
        f"95% CI [{interval['lower_bound']:+.4f}, {interval['upper_bound']:+.4f}]"
    )
    if win_rate is not None and wilson is not None:
        print(
            f"  win rate                     : {win_rate:.4f}  "
            f"Wilson [{wilson['lower_bound']:.4f}, {wilson['upper_bound']:.4f}]"
        )
    print(f"  sigma_d                      : {metrics['sigma_d']}")
    print(
        f"  paired deals / played deals  : {metrics['paired_deal_count']} / {metrics['played_deal_count']}"
    )
    print(
        f"  contract success             : {metrics['contract_success_rate']}  "
        f"mean bid {metrics['mean_bid']}  passed out {metrics['passed_out_rate']:.4f}"
    )
    print(f"  pass action rate             : {metrics['pass_action_rate']}")
    print(f"  score by bid level           : {metrics['score_by_bid_level']}")
    print(f"  mean tricks declaring team   : {metrics['mean_tricks_declaring_team']}")
    print(f"  role split                   : {metrics['role_split_mean_score']}")
    print(
        f"  illegal rate {metrics['illegal_action_rate']:.1f}  "
        f"no-trump rate {metrics['no_trump_action_rate']:.1f}  "
        f"replay integrity {metrics['replay_integrity_rate']}"
    )
    if evaluation.trump_suit_breakdown:
        print("  by trump suit:")
        for suit, entry in evaluation.trump_suit_breakdown.items():
            print(
                f"    {suit:<9} n={int(entry['paired_deal_count']):>5}  "
                f"paired {entry['mean_paired_score_difference']:+.4f}  "
                f"win {entry['win_rate']:.4f} "
                f"[{entry['win_rate_lower_bound']:.4f}, {entry['win_rate_upper_bound']:.4f}]"
            )
    print()


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    rules = RulesConfig()
    _config, stage = load_stage_configuration(arguments.stage_config)

    loaded = load_checkpoint(arguments.checkpoint, rules=rules, load_critic=False)
    print(f"checkpoint        : {arguments.checkpoint}")
    print(f"stage             : {stage.name} (curriculum stage {stage.curriculum_stage})")
    print(f"auction           : {stage.auction_mode}   history: {stage.history_condition}")
    print(f"trained steps     : {loaded.global_environment_steps:,}")
    print(f"rules fingerprint : {loaded.payload['rules_fingerprint']}")
    print(f"buffer contract   : {loaded.payload['buffer_contract_version']}")
    print(f"checkpoint git sha: {loaded.payload.get('git_sha')}")
    print(f"paired deals      : {arguments.paired_deal_count}")
    print(f"evaluation seed   : {arguments.evaluation_seed}")
    print(
        f"action mode       : {'deterministic' if arguments.deterministic_actions else 'sampled'}"
    )
    print()

    results: dict[str, Any] = {
        "label": arguments.label or arguments.checkpoint.stem,
        "checkpoint": str(arguments.checkpoint),
        "stage": stage.as_dict(),
        "evaluation_seed": arguments.evaluation_seed,
        "paired_deal_count": arguments.paired_deal_count,
        "deterministic_actions": arguments.deterministic_actions,
        "trained_environment_steps": loaded.global_environment_steps,
        "checkpoint_git_sha": loaded.payload.get("git_sha"),
        "evaluations": {},
    }

    for opponent_name in arguments.opponents:
        neural_agent = HistoryConditionedNeuralAgent(
            actor=loaded.actor,
            history_condition=stage.history_condition,
            name="neural",
            deterministic=arguments.deterministic_actions,
            seed=arguments.evaluation_seed,
            rules=rules,
        )
        evaluation = evaluate_stage(
            stage=stage,
            policy_a=neural_agent,
            policy_b=build_baseline_agent(opponent_name, seed=arguments.evaluation_seed),
            paired_deal_count=arguments.paired_deal_count,
            evaluation_seed=arguments.evaluation_seed,
            bootstrap_resamples=arguments.bootstrap_resamples,
            rules=rules,
        )
        print_evaluation(evaluation)
        results["evaluations"][opponent_name] = evaluation.as_dict()

    if arguments.output_path is not None:
        arguments.output_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {arguments.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
