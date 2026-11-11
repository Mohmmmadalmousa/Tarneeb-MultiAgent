"""Synthetic-only checks: no research optimization, dealing, or game replay."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from tarneeb.auction_repair.data import PublicExamples
from tarneeb.auction_repair.head_study import (
    Settings,
    export_rows,
    losses,
    optimize_head,
    weights_for,
)
from tarneeb.auction_repair.learning import assert_preserved
from tarneeb.core.engine import TarneebEngine
from tarneeb.models.recurrent import RecurrentArchitectureSpec, build_recurrent_actor_and_critic


@pytest.fixture(autouse=True)
def no_games(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No simulation in focused tests")

    monkeypatch.setattr(TarneebEngine, "deal", forbidden)
    monkeypatch.setattr(TarneebEngine, "step", forbidden)
    torch.set_num_threads(1)


def dataset():
    metadata, labels = [], []
    for owner in ("opening", "partner", "opponent"):
        for target in (59, 53, 54):
            for _ in range(1 if target != 53 else 3):
                metadata.append(
                    {
                        "current_high_bid": None if owner == "opening" else 7,
                        "high_bidder_is_partner": owner == "partner",
                    }
                )
                labels.append(target)
    legal = np.zeros((len(labels) + 1, 65), dtype=bool)
    legal[:-1, 52:60] = True
    legal[-1, 59] = True
    labels.append(59)
    metadata.append({"current_high_bid": 13, "high_bidder_is_partner": False})
    n = len(labels)
    return PublicExamples(
        np.zeros((n, 692), dtype=np.float32),
        np.zeros((n, 80, 7), dtype=np.int64),
        np.zeros(n, dtype=np.int64),
        legal,
        np.array(labels),
        tuple(metadata),
    )


def test_fixed_weights_equal_strata_and_levels_without_extra_examples():
    data = dataset()
    initial = deepcopy(data)
    keep, weights, cells = weights_for(data)
    assert keep.tolist() == list(range(15))
    assert np.allclose(weights["uniform"], 1 / 15)
    for owner in ("opening", "partner", "opponent"):
        for kind in ("PASS", "BID"):
            selected = [c for c in cells if c["stratum"] == f"{owner}:{kind}"]
            assert sum(c["mass"] for c in selected) == pytest.approx(1 / 6)
            if kind == "BID":
                assert [c["mass"] for c in selected] == pytest.approx([1 / 12, 1 / 12])
    assert data.metadata == initial.metadata
    assert np.array_equal(data.labels, initial.labels)
    assert np.array_equal(data.statics, initial.statics)
    assert np.all(weights["balanced"] > 0)


def test_missing_stratum_rejected():
    d = dataset()
    with pytest.raises(ValueError):
        weights_for(replace(d, metadata=tuple({**m, "current_high_bid": None} for m in d.metadata)))


def test_loss_has_exact_masking_and_known_weighted_value():
    raw = torch.zeros(2, 65, requires_grad=True)
    legal = torch.zeros(2, 65, dtype=torch.bool)
    legal[0, [53, 59]] = True
    legal[1, 59] = True
    per_row = losses(raw, legal, torch.tensor([53, 59]))
    assert per_row.detach().tolist() == pytest.approx([np.log(2), 0])
    objective = (per_row * torch.tensor([0.25, 0.75])).sum()
    objective.backward()
    assert objective.item() == pytest.approx(np.log(2) / 4)
    assert torch.count_nonzero(raw.grad[~legal]) == 0


def problem():
    spec = RecurrentArchitectureSpec(hidden_size=8, static_hidden_dim=16)
    actor = build_recurrent_actor_and_critic(spec, seed=8600991)[0].eval().requires_grad_(False)
    features = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32
    legal = torch.zeros(4, 65, dtype=torch.bool)
    legal[:, [53, 59]] = True
    return actor, features, legal, torch.tensor([53, 59, 53, 59]), torch.full((4,), 0.25)


def run(settings, **kwargs):
    p = problem()
    log = []
    return (
        p,
        optimize_head(
            *p,
            lambda r: assert_preserved(p[0], export_rows(p[0], r)),
            log.append,
            float("inf"),
            settings=settings,
            **kwargs,
        ),
        log,
    )


def test_identical_initialization_reproducibility_and_frozen_rows():
    config = Settings(steps=3)
    p, (first, metrics), _ = run(config)
    _, (second, other), _ = run(config)
    assert metrics["accepted_updates"] == 3 and metrics["eligible_final_checkpoint"]
    assert other["accepted_updates"] == 3
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    candidate = export_rows(p[0], first)
    assert_preserved(p[0], candidate)
    assert not torch.equal(candidate.policy_head.weight[52:60], p[0].policy_head.weight[52:60])
    raw_before = p[0].policy_head(p[1])
    raw_after = candidate.policy_head(p[1])
    assert torch.equal(raw_before[:, :52], raw_after[:, :52])
    assert torch.equal(raw_before[:, 60:], raw_after[:, 60:])
    assert all(t.grad is None for t in p[0].parameters())


def test_drift_rejects_proposal_without_projection_or_final_checkpoint():
    p, (rows, metrics), log = run(Settings(steps=3, drift=1e-10))
    assert metrics["status"] == "stopped_drift_limit"
    assert metrics["accepted_updates"] == 0 and metrics["attempted_updates"] == 1
    assert not metrics["eligible_final_checkpoint"] and log[0]["accepted"] is False
    assert torch.equal(rows.auction_weight, p[0].policy_head.weight[52:60])


def test_cpu_budget_stops_before_optimization():
    ticks = iter([0.0, 2.0, 3.0, 4.0])
    _, (_, metrics), log = run(Settings(cpu_seconds=1), cpu_clock=lambda: next(ticks))
    assert metrics["status"] == "stopped_arm_cpu_limit"
    assert metrics["attempted_updates"] == 0 and log == []


def test_wall_budget_stops_before_optimization():
    p = problem()
    _, metrics = optimize_head(*p, lambda r: None, lambda e: None, 0.0)
    assert metrics["status"] == "stopped_overall_wall_limit"
    assert metrics["attempted_updates"] == 0


def test_saved_audit_inputs_include_empty_history_without_invention():
    from scripts.run_phase11b_head_study import audit_examples

    data, ids = audit_examples()
    assert len(data) == 32 and data.statics.shape == (32, 692)
    assert data.lengths[ids.index("O1")] == 0
    assert not data.histories[ids.index("O1")].any()
    assert data.labels[ids.index("C9PS")] == 55
    assert data.labels[ids.index("O8")] == 58
