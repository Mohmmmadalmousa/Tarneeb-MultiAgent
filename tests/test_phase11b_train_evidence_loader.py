"""Read-only checks against the frozen completed audit; no worlds regenerated."""

import copy

import pytest
from scripts import audit_phase11b_train_evidence as audit

from tarneeb.auction_repair import train_evidence as evidence


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Evidence authentication must not execute or sample")

    for name in ("sample_hands", "play_candidate", "initialize_worker"):
        monkeypatch.setattr(audit.v3.v2.v1, name, forbidden)
    monkeypatch.setattr(audit.r, "worlds_for", forbidden)
    monkeypatch.setattr(audit.selection.TarneebEngine, "deal", forbidden)


def saved(index):
    plan = audit.r.read(evidence.ROOT / evidence.AUDIT / "preregistration.json")
    case = plan["cases"][index]
    result = audit.r.read(evidence.ROOT / evidence.AUDIT / "states" / case["id"] / "result.json")
    return case, result, plan["scientific_config"]


def test_real_audit_authenticates_exact_mapping():
    bundle = evidence.load_train_evidence()
    bundle.validate()
    triples = {(p.row, p.preferred, p.other) for p in bundle.pairs}
    assert triples == evidence.EXPECTED
    assert len(triples) == 30 and len({p.row for p in bundle.pairs}) == 8
    assert not {p.row for p in bundle.pairs} & evidence.EXCLUDED
    assert all(p.source == evidence.SOURCE for p in bundle.pairs)
    assert len(bundle.rows) == 2880


@pytest.mark.parametrize("index", [3, 8, 10, 11])
def test_uncertain_cases_never_supply_pairs(index):
    case, result, config = saved(index)
    assert result["robust_classification"] == "uncertain"
    assert evidence._supported_comparisons(case, result, config) == set()


@pytest.mark.parametrize("fault", ["gate", "ci", "classification", "actions", "policy"])
def test_invalid_support_or_bounds_rejected(fault):
    case, result, config = saved(4)
    result = copy.deepcopy(result)
    summary = result["summaries"]["all_strong"]["uniform"]
    if fault == "gate":
        result["posterior_support_gate"] = False
    elif fault == "ci":
        summary["actions"][-1]["teacher_advantage_simultaneous_95ci"][0] += 1
    elif fault == "classification":
        result["robust_classification"] = "pass_favored"
    elif fault == "actions":
        summary["actions"].pop()
    else:
        del result["summaries"]["all_strong"]
    with pytest.raises(ValueError):
        evidence._supported_comparisons(case, result, config)


def test_directions_include_bid13_over_bid12_and_pass_controls():
    for index in (4, 5):
        case, result, config = saved(index)
        pairs = evidence._supported_comparisons(case, result, config)
        row = case["training_selection"]["training_row_id"]
        assert (row, 58, 57) in pairs and (row, 57, 58) not in pairs
    case, result, config = saved(9)
    assert evidence._supported_comparisons(case, result, config) == {
        (942, 59, a) for a in (55, 56, 57, 58)
    }


def test_pinned_completion_change_rejected(monkeypatch):
    sha = audit.selection.sha256
    monkeypatch.setattr(
        audit.selection, "sha256", lambda p: "bad" if p.name == "completion.json" else sha(p)
    )
    with pytest.raises(ValueError, match=r"completion\.json hash"):
        evidence.load_train_evidence()


@pytest.mark.parametrize("target", ["result.json", "world_000.json"])
def test_saved_record_corruption_rejected_without_writes(monkeypatch, target):
    read = audit.r.read

    def corrupted(path):
        value = read(path)
        if path.name == target:
            value = copy.deepcopy(value)
            if target == "result.json":
                value["completed_worlds"] = 63
            else:
                value["payload"]["world_index"] = 12
        return value

    monkeypatch.setattr(audit.r, "read", corrupted)
    with pytest.raises(ValueError, match=r"hash mismatch|checksum mismatch"):
        evidence.load_train_evidence()
