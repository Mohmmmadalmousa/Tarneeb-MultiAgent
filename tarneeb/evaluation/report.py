"""Reproducible evaluation output: JSON and CSV.

Every report carries enough metadata to reproduce it -- policy names and
configuration, the evaluation seed, the deal count, all four version strings,
the rules fingerprint, the git SHA when available, and the runtime.

No dashboards, no TensorBoard, no UI: structured numbers only.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tarneeb.core.actions import ACTION_SPEC_VERSION
from tarneeb.core.events import EVENT_SCHEMA_VERSION
from tarneeb.core.recorder import DEAL_RECORD_VERSION
from tarneeb.core.rules import RULES_VERSION, RulesConfig
from tarneeb.env.spaces import OBS_SPEC_VERSION
from tarneeb.evaluation.metrics import EvaluationResult

REPORT_FORMAT_VERSION = "eval_report_v1"


def current_git_sha() -> str | None:
    """Return the short git SHA, or ``None`` outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - environment dependent
        return None
    return completed.stdout.strip() or None


@dataclass
class EvaluationReport:
    """One evaluation, with the metadata needed to reproduce it."""

    result: EvaluationResult
    rules: RulesConfig
    policy_a_config: dict[str, Any] = field(default_factory=dict)
    policy_b_config: dict[str, Any] = field(default_factory=dict)
    git_sha: str | None = None
    created_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable report."""
        return {
            "report_format_version": REPORT_FORMAT_VERSION,
            "created_utc": self.created_utc or datetime.now(UTC).isoformat(),
            "git_sha": self.git_sha if self.git_sha is not None else current_git_sha(),
            "versions": {
                "rules_version": RULES_VERSION,
                "rules_fingerprint": self.rules.fingerprint(),
                "obs_spec_version": OBS_SPEC_VERSION,
                "action_spec_version": ACTION_SPEC_VERSION,
                "event_schema_version": EVENT_SCHEMA_VERSION,
                "deal_record_version": DEAL_RECORD_VERSION,
            },
            "comparison": {
                "policy_a_name": self.result.policy_a_name,
                "policy_b_name": self.result.policy_b_name,
                "policy_a_config": self.policy_a_config,
                "policy_b_config": self.policy_b_config,
                "evaluation_seed": self.result.evaluation_seed,
                "paired_deal_count": self.result.metrics["paired_deal_count"],
                "sign_convention": (
                    "positive mean_paired_score_difference means policy_a outscored policy_b, "
                    "in points per deal, averaged over both seatings of each duplicate pair"
                ),
            },
            "metrics": self.result.metrics,
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_pairs_csv(self, path: Path) -> Path:
        """Write one row per duplicate pair, for independent re-analysis."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "pair_index",
                    "match_seed",
                    "deal_index",
                    "dealer_seat",
                    "paired_score_difference",
                    "first_run_policy_a_score",
                    "first_run_policy_b_score",
                    "second_run_policy_a_score",
                    "second_run_policy_b_score",
                    "first_run_bid",
                    "second_run_bid",
                    "first_run_contract_made",
                    "second_run_contract_made",
                ]
            )
            for pair_index, pair in enumerate(self.result.paired_outcomes):
                writer.writerow(
                    [
                        pair_index,
                        pair.match_seed,
                        pair.deal_index,
                        pair.first_run.dealer_seat,
                        pair.paired_score_difference,
                        pair.first_run.policy_a_score,
                        pair.first_run.policy_b_score,
                        pair.second_run.policy_a_score,
                        pair.second_run.policy_b_score,
                        pair.first_run.winning_bid,
                        pair.second_run.winning_bid,
                        pair.first_run.contract_made,
                        pair.second_run.contract_made,
                    ]
                )
        return path


def write_summary_csv(reports: list[EvaluationReport], path: Path) -> Path:
    """Write one row per evaluation, for comparing several experiments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "policy_a",
                "policy_b",
                "paired_deal_count",
                "mean_paired_score_difference",
                "ci_lower_bound",
                "ci_upper_bound",
                "sigma_d",
                "win_rate",
                "win_rate_lower",
                "win_rate_upper",
                "contract_success_rate",
                "mean_bid",
                "illegal_action_rate",
                "replay_integrity_rate",
                "deals_per_second",
            ]
        )
        for report in reports:
            metrics = report.result.metrics
            interval = metrics["paired_score_difference_ci"]
            win_interval = metrics["win_rate_wilson_interval"] or {}
            writer.writerow(
                [
                    report.result.policy_a_name,
                    report.result.policy_b_name,
                    metrics["paired_deal_count"],
                    metrics["mean_paired_score_difference"],
                    interval["lower_bound"],
                    interval["upper_bound"],
                    metrics["sigma_d"],
                    metrics["win_rate"],
                    win_interval.get("lower_bound"),
                    win_interval.get("upper_bound"),
                    metrics["contract_success_rate"],
                    metrics["mean_bid"],
                    metrics["illegal_action_rate"],
                    metrics["replay_integrity_rate"],
                    metrics.get("deals_per_second"),
                ]
            )
    return path
