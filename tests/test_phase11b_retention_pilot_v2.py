"""Runner gates with synthetic telemetry only; never execute optimizer updates."""

import copy
import time

import pytest
import torch
from scripts import run_phase11b_retention_pilot_v2 as pilot


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Optimizer construction/updates are forbidden during preparation tests")

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


def loop(propose=None, validate=None):
    events, checks = [], []

    def check(step):
        checks.append(step)
        return validate(step) if validate else None

    result = pilot.bounded_loop(
        propose or (lambda step: {"loss": 1.0, "gradient_norm": 0.2, "drift": step * 0.0001}),
        check,
        events.append,
        start_cpu=time.process_time(),
        start_wall=time.monotonic(),
    )
    return result, events, checks


def test_exact_twenty_proposals_and_five_update_validation_cadence():
    result, events, checks = loop()
    assert result["status"] == "complete" and result["attempted_updates"] == 20
    assert result["validated_updates"] == 20
    assert len(events) == 20 and checks == [5, 10, 15, 20]
    assert result["promotion"] is False


@pytest.mark.parametrize("bad", ["drift", "loss", "gradient_norm"])
def test_drift_and_nonfinite_proposal_stop_immediately(bad):
    def proposal(step):
        entry = {"loss": 1.0, "gradient_norm": 0.1, "drift": 0.001}
        if step == 2:
            entry[bad] = 0.020001 if bad == "drift" else float("nan")
        return entry

    result, events, checks = loop(proposal)
    assert result["attempted_updates"] == 2 and result["validated_updates"] == 0
    assert result["status"] == ("drift_limit" if bad == "drift" else "numerical_instability")
    assert len(events) == 2 and checks == []


def test_validation_stop_does_not_continue_or_retry():
    result, events, checks = loop(
        validate=lambda step: "forgotten_development_bid" if step == 10 else None
    )
    assert result["attempted_updates"] == 10 and result["validated_updates"] == 5
    assert result["status"] == "forgotten_development_bid"
    assert len(events) == 10 and checks == [5, 10]


def report():
    return {
        "train": {"lost_bid_anchors": 0, "lost_pass_anchors": 0},
        "dev": {"lost_bid_anchors": 0, "lost_pass_anchors": 0},
        "dev_roles": {"opening": {"bids_on_teacher_pass": 3}},
    }


@pytest.mark.parametrize(
    "split,field,reason",
    [
        ("dev", "lost_bid_anchors", "forgotten_development_bid"),
        ("dev", "lost_pass_anchors", "lost_protected_development_pass"),
        ("train", "lost_bid_anchors", "lost_training_retention_anchor"),
        ("train", "lost_pass_anchors", "lost_training_retention_anchor"),
    ],
)
def test_any_single_forgotten_protected_decision_stops(split, field, reason):
    original = report()
    changed = copy.deepcopy(original)
    changed[split][field] = 1
    assert pilot.stop_reason(original, original) is None
    assert pilot.stop_reason(changed, original) == reason


def test_false_raise_increase_stops_even_without_anchor_loss():
    original = report()
    changed = copy.deepcopy(original)
    changed["dev_roles"]["opening"]["bids_on_teacher_pass"] += 1
    assert pilot.stop_reason(changed, original) == "development_false_raise_increase"


def test_expired_budget_never_calls_proposal():
    result = pilot.bounded_loop(
        lambda step: pytest.fail("proposal executed"),
        lambda step: None,
        lambda entry: None,
        start_cpu=time.process_time(),
        start_wall=time.monotonic() - 301,
    )
    assert result["status"] == "resource_limit" and result["attempted_updates"] == 0


def test_training_scope_is_two_separate_auction_tensors_only():
    actor = torch.nn.Module()
    actor.policy_head = torch.nn.Linear(3, 65)
    actor.trunk = torch.nn.Linear(3, 3)
    actor.spec = "synthetic"
    actor.eval().requires_grad_(False)
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    rows = pilot.AuctionRows(actor)
    assert rows.auction_weight.shape == (8, 3) and rows.auction_bias.shape == (8,)
    assert rows.auction_weight.data_ptr() != actor.policy_head.weight.data_ptr()
    rows(torch.ones(2, 3)).sum().backward()
    assert all(p.grad is None for p in actor.parameters())
    candidate = pilot.export_rows(actor, rows)
    pilot.assert_preserved(actor, candidate)
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in before.items())


def test_dry_run_writes_nothing_and_never_constructs_optimizer(monkeypatch, tmp_path):
    from contextlib import nullcontext

    monkeypatch.setattr(pilot, "OUT", tmp_path / "not_created")
    monkeypatch.setattr(pilot, "provenance", lambda: {"pin": "value"})
    monkeypatch.setattr(pilot, "load_context", lambda: {})
    monkeypatch.setattr(pilot, "inspect_baseline", lambda c: ({"optimizer_updates": 0}, 0.01))
    monkeypatch.setattr(pilot, "checkpoint_read_locks", nullcontext)
    monkeypatch.setattr(pilot, "write_json_new", lambda *a: pytest.fail("Dry-run wrote a file"))
    pilot.prepare_or_dry_run(False)


def test_prepare_refuses_existing_output(monkeypatch):
    monkeypatch.setattr(pilot.OUT.__class__, "exists", lambda p: True)
    with pytest.raises(FileExistsError):
        pilot.prepare_or_dry_run(True)


def test_execute_refuses_existing_run_before_loading_or_optimizer(monkeypatch):
    monkeypatch.setattr(pilot, "registered_plan", lambda: ({}, {}))
    monkeypatch.setattr(pilot, "load_context", lambda: pytest.fail("Attempted restart"))

    def existing(*args, **kwargs):
        raise FileExistsError("Prior run")

    monkeypatch.setattr(pilot.OUT.__class__, "mkdir", existing)
    with pytest.raises(FileExistsError):
        pilot.execute()


@pytest.mark.parametrize(
    "key,value", [("max_updates", 21), ("max_drift", 0.03), ("seed", 20250910)]
)
def test_configuration_caps_fail_closed(monkeypatch, key, value):
    monkeypatch.setitem(pilot.CONFIG, key, value)
    with pytest.raises(ValueError):
        pilot.validate_config()


def test_v2_isolated_and_fixed_split_weights():
    from scripts import run_phase11b_retention_pilot as historical

    pilot.validate_config()
    assert pilot.OUT == pilot.ROOT / "results/phase11b/retention_pilot_v2"
    assert pilot.OUT != historical.OUT
    assert pilot.WEIGHTS.supported_pair is None
    assert historical.WEIGHTS.supported_pair == 0.02
    assert pilot.WEIGHTS.bid_pass == pilot.WEIGHTS.bid_bid == 0.02
    assert pilot.CONFIG == historical.CONFIG
    assert "scripts/run_phase11b_retention_pilot_v2.py" in pilot.CODE
    assert "tests/test_phase11b_retention_pilot_v2.py" in pilot.CODE


def test_legacy_objective_rejected_before_context_loading(monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(pilot, "WEIGHTS", replace(pilot.WEIGHTS, supported_pair=0.02))
    monkeypatch.setattr(pilot, "load_context", lambda: pytest.fail("Loaded invalid pilot"))
    with pytest.raises(ValueError, match="split objective"):
        pilot.prepare_or_dry_run(False)


def test_separate_pair_reporting_counts_both_pass_directions_and_case_means():
    pairs = [
        {"row": 1, "preferred": 55, "other": 59, "margin": -2.0},
        {"row": 1, "preferred": 56, "other": 59, "margin": 0.0},
        {"row": 2, "preferred": 59, "other": 55, "margin": 3.0},
        {"row": 1, "preferred": 58, "other": 57, "margin": 0.05},
    ]
    report = pilot.pair_metrics(pairs)
    assert report["bid_pass"] == {
        "comparisons": 3,
        "cases": 2,
        "positive_pairs": 1,
        "pairs_at_target_margin": 1,
        "case_mean_margin": 1.0,
    }
    assert report["bid_bid"] == {
        "comparisons": 1,
        "cases": 1,
        "positive_pairs": 1,
        "pairs_at_target_margin": 0,
        "case_mean_margin": 0.05,
    }
    assert pilot.pair_metrics([])["bid_bid"]["case_mean_margin"] is None


def test_unsatisfied_ranking_is_not_a_stopping_gate():
    original = report()
    original["evidence_components"] = {"bid_bid": {"positive_pairs": 0}}
    assert pilot.stop_reason(original, original) is None


@pytest.mark.parametrize("key,value", [("learning_rate", 0.0003), ("gradient_clip", 1.0)])
def test_optimizer_budget_changes_fail_closed(monkeypatch, key, value):
    monkeypatch.setitem(pilot.CONFIG, key, value)
    with pytest.raises(ValueError, match="budget"):
        pilot.validate_config()
