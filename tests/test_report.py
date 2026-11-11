"""Evaluation report output: reproducible JSON and CSV with full metadata."""

from __future__ import annotations

import csv
import json

import pytest

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.core.actions import ACTION_SPEC_VERSION
from tarneeb.core.events import EVENT_SCHEMA_VERSION
from tarneeb.core.recorder import DEAL_RECORD_VERSION
from tarneeb.core.rules import RULES_VERSION, RulesConfig
from tarneeb.env.spaces import OBS_SPEC_VERSION
from tarneeb.evaluation.arena import Arena
from tarneeb.evaluation.metrics import EvaluationResult
from tarneeb.evaluation.report import (
    REPORT_FORMAT_VERSION,
    EvaluationReport,
    write_summary_csv,
)

EXPECTED_FINGERPRINT = "sha1:c18b43e7e9ebfc6855ff5fb56a968308d222a2f5"


@pytest.fixture
def evaluation_result() -> EvaluationResult:
    arena = Arena(replay_sample_rate=0.5)
    return arena.evaluate(
        StrongHeuristicAgent(),
        BasicHeuristicAgent(),
        paired_deal_count=8,
        evaluation_seed=123,
        bootstrap_resamples=500,
    )


@pytest.fixture
def report(evaluation_result: EvaluationResult) -> EvaluationReport:
    return EvaluationReport(
        result=evaluation_result,
        rules=RulesConfig(),
        policy_a_config={"class": "StrongHeuristicAgent", "bid_margin": 1.5},
        policy_b_config={"class": "BasicHeuristicAgent", "bid_threshold": 13},
        git_sha="abc1234",
        created_utc="2026-01-01T00:00:00+00:00",
    )


def test_report_carries_every_version_string(report: EvaluationReport) -> None:
    versions = report.to_dict()["versions"]
    assert versions["rules_version"] == RULES_VERSION == "tarneeb-classic-v1"
    assert versions["rules_fingerprint"] == EXPECTED_FINGERPRINT
    assert versions["obs_spec_version"] == OBS_SPEC_VERSION == "obs_v2"
    assert versions["action_spec_version"] == ACTION_SPEC_VERSION == "act_v1"
    assert versions["event_schema_version"] == EVENT_SCHEMA_VERSION == "event_v1"
    assert versions["deal_record_version"] == DEAL_RECORD_VERSION == "deal_v2"


def test_report_carries_reproduction_metadata(report: EvaluationReport) -> None:
    payload = report.to_dict()
    assert payload["report_format_version"] == REPORT_FORMAT_VERSION
    assert payload["git_sha"] == "abc1234"
    assert payload["created_utc"] == "2026-01-01T00:00:00+00:00"

    comparison = payload["comparison"]
    assert comparison["policy_a_name"] == "strong"
    assert comparison["policy_b_name"] == "basic"
    assert comparison["evaluation_seed"] == 123
    assert comparison["paired_deal_count"] == 8
    assert comparison["policy_a_config"]["bid_margin"] == 1.5
    assert "positive" in comparison["sign_convention"]


def test_report_is_json_serialisable_and_reproducible(report: EvaluationReport, tmp_path) -> None:
    path = report.write_json(tmp_path / "report.json")
    first_text = path.read_text(encoding="utf-8")
    payload = json.loads(first_text)

    assert payload["metrics"]["paired_deal_count"] == 8
    assert payload["metrics"]["illegal_action_rate"] == 0.0
    assert payload["metrics"]["replay_integrity_rate"] == 1.0

    # Writing the same report again produces byte-identical output.
    assert report.write_json(tmp_path / "again.json").read_text(encoding="utf-8") == first_text


def test_two_runs_of_the_same_evaluation_report_the_same_numbers(tmp_path) -> None:
    def build_report() -> dict:
        arena = Arena(replay_sample_rate=0.0)
        result = arena.evaluate(
            StrongHeuristicAgent(),
            BasicHeuristicAgent(),
            paired_deal_count=10,
            evaluation_seed=555,
            bootstrap_resamples=500,
        )
        payload = EvaluationReport(
            result=result, rules=RulesConfig(), git_sha="x", created_utc="t"
        ).to_dict()
        payload["metrics"].pop("runtime_seconds", None)
        payload["metrics"].pop("decisions_per_second", None)
        payload["metrics"].pop("deals_per_second", None)
        payload["metrics"].pop("paired_deals_per_second", None)
        return payload

    assert build_report() == build_report()


def test_pairs_csv_has_one_row_per_pair(report: EvaluationReport, tmp_path) -> None:
    path = report.write_pairs_csv(tmp_path / "pairs.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert rows[0]["pair_index"] == "0"
    for row, pair in zip(rows, report.result.paired_outcomes, strict=True):
        assert int(row["match_seed"]) == pair.match_seed
        assert float(row["paired_score_difference"]) == pytest.approx(pair.paired_score_difference)
        assert int(row["first_run_policy_a_score"]) == pair.first_run.policy_a_score


def test_summary_csv_has_one_row_per_report(report: EvaluationReport, tmp_path) -> None:
    path = write_summary_csv([report, report], tmp_path / "summary.csv")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["policy_a"] == "strong"
    assert rows[0]["policy_b"] == "basic"
    assert float(rows[0]["mean_paired_score_difference"]) == pytest.approx(
        report.result.metrics["mean_paired_score_difference"]
    )
    assert float(rows[0]["ci_lower_bound"]) <= float(rows[0]["ci_upper_bound"])


def test_report_metrics_include_the_required_fields(report: EvaluationReport) -> None:
    metrics = report.to_dict()["metrics"]
    for field_name in (
        "mean_paired_score_difference",
        "paired_score_difference_ci",
        "sigma_d",
        "win_rate",
        "win_rate_wilson_interval",
        "contract_success_rate",
        "mean_bid",
        "pass_action_rate",
        "score_by_bid_level",
        "role_split_mean_score",
        "mean_tricks_declaring_team",
        "illegal_action_rate",
        "no_trump_action_rate",
        "replay_integrity_rate",
        "decision_count",
        "runtime_seconds",
        "decisions_per_second",
        "deals_per_second",
    ):
        assert field_name in metrics, f"missing metric {field_name}"


def test_strong_evidence_flag_follows_the_lower_bound(
    evaluation_result: EvaluationResult,
) -> None:
    lower_bound = evaluation_result.metrics["paired_score_difference_ci"]["lower_bound"]
    assert evaluation_result.strong_evidence_a_beats_b == (lower_bound > 0.0)
