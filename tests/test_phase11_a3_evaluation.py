"""Adapter tests: synthetic evidence plus four TEST-only duplicate pairs.

Never evaluate selection/confirmation deals or construct an optimizer.
"""

from copy import deepcopy

import pytest
import torch
from scripts import evaluate_phase11_a3 as a3

from tarneeb.evaluation import arena


@pytest.fixture(autouse=True)
def boundaries(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Training/promotion/checkpoint writing is forbidden")

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    monkeypatch.setattr(a3.LeaguePopulation, "promote", forbidden)
    original = arena.evaluation_match_seed

    def test_only(seed, index):
        assert seed == a3.TEST_SEED, "Reserved outcomes must never be accessed by tests"
        return original(seed, index)

    monkeypatch.setattr(arena, "evaluation_match_seed", test_only)


def record(job, mean=1.0, low=0.5):
    value = {
        "job": job,
        "plan_identity": "synthetic",
        "scores": [mean] * job["count"],
        "metrics": {
            "mean_paired_score_difference": mean,
            "paired_score_difference_ci": {
                "mean": mean,
                "lower_bound": low,
                "upper_bound": mean + 1,
                "confidence": 0.95,
                "sample_count": job["count"],
            },
            "illegal_action_rate": 0,
            "no_trump_action_rate": 0,
            "replay_integrity_rate": 1,
        },
    }
    value["digest"] = a3.digest(value)
    return value


def reseal(value):
    value["digest"] = a3.digest({k: v for k, v in value.items() if k != "digest"})


def evidence(candidate, stage="selection"):
    return {(j["a"], j["b"]): record(j) for j in a3.jobs_for(candidate, stage)}


def test_fixed_budget_exact_predecessors_and_nonoverlap():
    jobs = a3.selection_jobs()
    assert len(jobs) == len({(j["a"], j["b"]) for j in jobs}) == 99
    assert sum(j["count"] for j in jobs) == 54000
    for c in a3.CANDIDATES:
        assert sum(j["count"] for j in a3.jobs_for(c, "confirmation")) == 5500
        assert a3.predecessors(c) == tuple(f"{c}_u{u}" for u in (0, 50, 100, 150, 200))
        assert c not in a3.predecessors(c)
    assert not a3.identities(a3.SELECTION_SEED, 2000) & a3.identities(a3.CONFIRMATION_SEED, 2000)
    assert a3.identities(60000, 2000) == a3.identities(a3.SELECTION_SEED, 2000)
    assert a3.identities(99999, 2) == {(1000099999, 0), (1000000000, 0)}


def test_exposure_parser_fails_closed_and_preserves_effective_deal_identity():
    entries = set()
    with pytest.raises(ValueError, match="Unresolved"):
        a3.exposure_from({"evaluation_seed": 42}, "unknown", entries)
    with pytest.raises(ValueError, match="identity"):
        a3.exposure_from({"match_seed": 1000070000}, "unknown", entries)
    a3.exposure_from({"paired_deal_count": 2, "nested": {"evaluation_seed": 60000}}, "ok", entries)
    assert ("ok", "arena", 60000, 2) in entries
    a3.exposure_from({"match_seed": 1000070000, "deal_index": 4}, "direct", entries)
    assert ("direct", "direct", 1000070000, 4) in entries


@pytest.mark.parametrize("candidate", a3.CANDIDATES)
def test_actual_final_checkpoint_loads_without_critic_or_parameter_changes(candidate):
    path = a3.ROOT / a3.model_paths()[candidate]
    before = a3.sha256(path)
    agent = a3.load_agent(candidate, {candidate: {"path": str(path)}})
    assert agent.deterministic and not agent.actor.training
    assert all(not p.requires_grad and p.grad is None for p in agent.actor.parameters())
    assert a3.sha256(path) == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "partial",
        "wrong_stage",
        "wrong_digest",
        "confidence",
        "illegal",
        "replay",
        "nonfinite",
    ],
)
def test_incomplete_or_invalid_evidence_fails_promotion(fault):
    c = a3.CANDIDATES[0]
    records = evidence(c)
    key = (c, a3.predecessors(c)[0])
    r = records[key]
    if fault == "missing":
        del records[key]
    elif fault == "partial":
        r["scores"].pop()
        reseal(r)
    elif fault == "wrong_stage":
        r["job"]["seed"] = a3.CONFIRMATION_SEED
        reseal(r)
    elif fault == "wrong_digest":
        r["digest"] = "bad"
    elif fault == "confidence":
        r["metrics"]["paired_score_difference_ci"]["confidence"] = 0.9
        reseal(r)
    elif fault == "illegal":
        r["metrics"]["illegal_action_rate"] = 0.01
        reseal(r)
    elif fault == "replay":
        r["metrics"]["replay_integrity_rate"] = 0
        reseal(r)
    else:
        r["scores"][0] = float("inf")  # Invalid JSON/digest itself also fails closed.
    assert not a3.gate(c, records, "synthetic", "selection")


@pytest.mark.parametrize(
    "target,mean,low,expected",
    [
        (a3.CHAMPION, 1.0, 0.0, False),
        (a3.CHAMPION, 1.0, 0.001, True),
        ("strong", 0.0, -0.5, False),
        ("strong", -0.1, -0.5, False),
        ("recent", 0.0, -0.5, True),
        ("recent", -0.001, -0.5, False),
    ],
)
def test_original_strict_thresholds(target, mean, low, expected):
    c = a3.CANDIDATES[0]
    rows = evidence(c)
    target = a3.predecessors(c)[0] if target == "recent" else target
    job = rows[(c, target)]["job"]
    rows[(c, target)] = record(job, mean, low)
    assert a3.gate(c, rows, "synthetic", "selection") is expected


def test_champion_requires_2000_pairs_and_all_five_recent_ids():
    c = a3.CANDIDATES[0]
    rows = evidence(c)
    r = rows[(c, a3.CHAMPION)]
    r["scores"].pop()
    r["job"]["count"] = r["metrics"]["paired_score_difference_ci"]["sample_count"] = 1999
    reseal(r)
    assert not a3.gate(c, rows, "synthetic", "selection")
    assert not a3.gate(c, {}, "synthetic", "selection")


def test_six_way_selection_and_deterministic_tie_break():
    rows = {}
    for c in reversed(a3.CANDIDATES):
        rows.update(evidence(c))
    assert a3.choose(rows, "synthetic") == min(a3.CANDIDATES)
    c = "pfsp_s2"
    rows[(c, "strong")] = record(rows[(c, "strong")]["job"], 2)
    assert a3.choose(rows, "synthetic") == c
    assert a3.choose({}, "synthetic") is None


def test_missing_selection_cannot_touch_confirmation(monkeypatch, tmp_path):
    monkeypatch.setattr(a3, "OUT", tmp_path)
    monkeypatch.setattr(a3, "evaluate_job", lambda *args: pytest.fail("Confirmation executed"))
    with pytest.raises(FileNotFoundError):
        a3.execute_confirmation({"identity": "synthetic"})


@pytest.mark.parametrize("tampered_winner", [False, True])
def test_confirmation_uses_only_sealed_winner_and_never_falls_back(
    monkeypatch, tmp_path, tampered_winner
):
    jobs = a3.selection_jobs()
    records = {f"{j['a']}__{j['b']}": record(j) for j in jobs}
    winner = min(a3.CANDIDATES)
    selected = {
        "plan_identity": "synthetic",
        "selected": "pfsp_s2" if tampered_winner else winner,
        "evidence": {k: v["digest"] for k, v in records.items()},
    }
    selected["digest"] = a3.digest(selected)
    plan = {
        "identity": "synthetic",
        "selection_jobs": jobs,
        "confirmation_jobs": {c: a3.jobs_for(c, "confirmation") for c in a3.CANDIDATES},
    }
    monkeypatch.setattr(a3, "OUT", tmp_path)
    monkeypatch.setattr(
        a3, "read", lambda p: selected if p.name == "selection.json" else records[p.stem]
    )
    monkeypatch.setattr(a3, "preflight", lambda: plan)
    calls, saved = [], []

    def evaluate(job, plan, stage):
        calls.append(job)
        # Synthetic confirmation fails strictly-positive champion lower bound.
        return record(job, low=0.0 if job["b"] == a3.CHAMPION else 0.5)

    monkeypatch.setattr(a3, "evaluate_job", evaluate)
    monkeypatch.setattr(a3, "write_json_new", lambda path, value: saved.append(value))
    if tampered_winner:
        with pytest.raises(ValueError, match="eligible sealed"):
            a3.execute_confirmation(plan)
        assert not calls and not saved
    else:
        a3.execute_confirmation(plan)
        assert len(calls) == 7 and {j["a"] for j in calls} == {winner}
        assert len(saved) == 1 and not saved[0]["gate_passed"] and not saved[0]["promotion"]


def test_deterministic_paired_neural_evaluation_and_saved_completeness(monkeypatch, tmp_path):
    torch.set_num_threads(2)
    c = "pfsp_s0"
    job = {"a": c, "b": "basic", "count": 2, "seed": a3.TEST_SEED}
    plan = {"identity": "synthetic", "models": {c: {"path": str(a3.model_paths()[c])}}}
    monkeypatch.setattr(a3, "OUT", tmp_path / "first")
    first = a3.evaluate_job(job, plan, "test")
    monkeypatch.setattr(a3, "OUT", tmp_path / "second")
    second = a3.evaluate_job(job, plan, "test")
    assert first["scores"] == second["scores"]
    assert (
        first["metrics"]["paired_score_difference_ci"]
        == second["metrics"]["paired_score_difference_ci"]
    )
    # Valid saved results are reused without playing any games; corrupted ones abort.
    monkeypatch.setattr(a3, "evaluate_condition", lambda *args: pytest.fail("Repeated evaluation"))
    assert a3.evaluate_job(job, plan, "test") == second
    broken = deepcopy(second)
    broken["scores"].pop()
    reseal(broken)
    monkeypatch.setattr(a3, "read", lambda p: broken)
    with pytest.raises(ValueError, match="Incomplete"):
        a3.evaluate_job(job, plan, "test")
