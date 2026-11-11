"""Evaluate a trained checkpoint against the frozen Phase 5 baselines.

Usage::

    python scripts/evaluate_checkpoint.py --checkpoint checkpoints/ppo_ff_baseline_update245.pt
    python scripts/evaluate_checkpoint.py --checkpoint ... --opponents random basic strong

Uses the Phase 5 arena unchanged: duplicate deals with the partnerships swapped,
held-out evaluation seeds, mean paired score difference with a bootstrap
confidence interval, and a Wilson interval on the win rate. A positive mean means
the neural policy outscored the baseline, in points per deal.

Like the training script, this is a reporting program: everything it prints comes
from values the evaluation already returned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tarneeb.core.rules import RulesConfig
from tarneeb.models.checkpoint import load_checkpoint
from tarneeb.training.train import evaluate_against_baseline

DEFAULT_OPPONENTS = ("random", "basic", "strong")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--paired-deal-count", type=int, default=1000)
    parser.add_argument("--evaluation-seed", type=int, default=20250910)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--deterministic-actions", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    rules = RulesConfig()

    loaded = load_checkpoint(arguments.checkpoint, rules=rules, load_critic=False)
    print(f"checkpoint        : {arguments.checkpoint}")
    print(f"trained steps     : {loaded.global_environment_steps:,}")
    print(f"rules fingerprint : {loaded.payload['rules_fingerprint']}")
    print(f"architecture      : {loaded.payload['architecture_spec_version']}")
    print(f"git sha           : {loaded.payload.get('git_sha')}")
    print(f"paired deals      : {arguments.paired_deal_count}")
    print(f"evaluation seed   : {arguments.evaluation_seed}")
    print(
        f"action mode       : {'deterministic' if arguments.deterministic_actions else 'sampled'}"
    )
    print()

    results = []
    for opponent_name in arguments.opponents:
        metrics = evaluate_against_baseline(
            actor=loaded.actor,
            opponent_name=opponent_name,
            paired_deal_count=arguments.paired_deal_count,
            evaluation_seed=arguments.evaluation_seed,
            bootstrap_resamples=arguments.bootstrap_resamples,
            deterministic_actions=arguments.deterministic_actions,
            rules=rules,
        )
        interval = metrics["paired_score_difference_ci"]
        wilson = metrics["win_rate_wilson_interval"]
        print(f"--- neural vs {opponent_name} ---")
        print(
            f"  mean paired score difference : {metrics['mean_paired_score_difference']:+.4f}  "
            f"95% CI [{interval['lower_bound']:+.4f}, {interval['upper_bound']:+.4f}]"
        )
        print(
            f"  win rate                     : {metrics['win_rate']:.4f}  "
            f"Wilson [{wilson['lower_bound']:.4f}, {wilson['upper_bound']:.4f}]"
        )
        print(f"  sigma_d                      : {metrics['sigma_d']:.4f}")
        print(
            f"  contract success             : {metrics['contract_success_rate']}  "
            f"mean bid {metrics['mean_bid']}  passed out {metrics['passed_out_rate']:.4f}"
        )
        print(
            f"  illegal rate {metrics['illegal_action_rate']:.1f}  "
            f"no-trump rate {metrics['no_trump_action_rate']:.1f}  "
            f"replay integrity {metrics['replay_integrity_rate']}"
        )
        print(f"  role split                   : {metrics['role_split_mean_score']}")
        print()
        results.append((opponent_name, metrics))

    if arguments.output_directory is not None:
        arguments.output_directory.mkdir(parents=True, exist_ok=True)
        for opponent_name, metrics in results:
            path = arguments.output_directory / f"neural_vs_{opponent_name}.json"
            path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
            print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
