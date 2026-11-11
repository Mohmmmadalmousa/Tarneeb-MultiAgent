"""Read-only adapter preparation checks; sampling and execution are forbidden."""

import copy

import pytest
from scripts import audit_phase11b_train_evidence as audit


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation attempted simulation, actor loading, or disk writes")

    for name in ("sample_hands", "play_candidate", "initialize_worker"):
        monkeypatch.setattr(audit.v3.v2.v1, name, forbidden)
    monkeypatch.setattr(audit.r, "worlds_for", forbidden)
    monkeypatch.setattr(audit.v3.v2, "run_session", forbidden)
    monkeypatch.setattr(audit.selection.TarneebEngine, "deal", forbidden)
    monkeypatch.setattr(audit.v3, "_ATOMIC", forbidden)


@pytest.fixture
def plan():
    return audit.make_plan()


def test_exact_cases_seat_mapping_fingerprints_and_determinism(plan):
    assert audit.make_plan() == plan
    original = audit.r.read(audit.MANIFEST)["cases"]
    assert [c["training_selection"] for c in plan["cases"]] == original
    assert [c["index"] for c in plan["cases"]] == list(range(12))
    assert all(type(c["index"]) is int for c in plan["cases"])
    # Exercise the frozen runner's scheduling key without launching its session.
    pending = list(reversed(plan["cases"]))
    pending.sort(key=lambda c: (-len(c["actions"]), c["index"]))
    assert pending == plan["cases"]
    required = {
        "index",
        "id",
        "actions",
        "own_hand",
        "dealer",
        "prefix",
        "world_seed",
        "bootstrap_seed",
        "high_bid",
        "teacher",
    }
    assert all(required <= c.keys() for c in plan["cases"])
    assert all(c["actions"] == [59, 55, 56, 57, 58] for c in plan["cases"])
    assert all(c["dealer"] == 1 for c in plan["cases"])
    assert plan["scientific_config"] == audit.v3.v2.v1.CONFIG
    assert plan["workload"]["paired_worlds"] == 768
    assert plan["workload"]["candidate_continuations"] == 7680
    assert plan["limits"] == {"cpu_seconds_per_state": 180, "wall_seconds": 1800, "workers": 4}
    audit.verify_plan(plan)


@pytest.mark.parametrize(
    "fault", ["index", "seed", "action", "fingerprint", "source", "limit", "output"]
)
def test_identity_tampering_rejected_even_if_resigned(plan, fault):
    modified = copy.deepcopy(plan)
    if fault == "index":
        modified["cases"][0].pop("index")
    elif fault == "seed":
        modified["cases"][0]["world_seed"] = 20250910
    elif fault == "action":
        modified["cases"][0]["actions"].pop()
    elif fault == "fingerprint":
        modified["cases"][0]["training_selection"]["observation_history_mask_sha256"] = "0" * 64
    elif fault == "source":
        modified["source_hashes"].pop(next(iter(modified["source_hashes"])))
    elif fault == "limit":
        modified["limits"]["cpu_seconds_per_state"] += 1
    else:
        modified["output_directory"] = str(audit.v3.OUTPUT)
    modified["identity"] = audit.r.digest({k: v for k, v in modified.items() if k != "identity"})
    with pytest.raises(ValueError, match="identity mismatch"):
        audit.verify_plan(modified)


def test_prepare_and_restart_preflight_only(monkeypatch, plan):
    saved = {}
    monkeypatch.setattr(audit, "make_plan", lambda: copy.deepcopy(plan))
    monkeypatch.setattr(audit.OUTPUT.__class__, "exists", lambda p: p != audit.OUTPUT)
    monkeypatch.setattr(audit, "atomic", lambda path, value: saved.setdefault(path, value))
    audit.prepare()
    assert set(saved) == {audit.OUTPUT / "preregistration.json", audit.OUTPUT / "preflight.json"}
    assert saved[audit.OUTPUT / "preflight.json"]["worlds_executed"] == 0
    read = audit.r.read
    monkeypatch.setattr(audit.r, "read", lambda path: saved[path] if path in saved else read(path))
    assert audit.preflight() == plan
    assert audit.preflight() == plan


def test_existing_preparation_refused(monkeypatch):
    monkeypatch.setattr(audit.OUTPUT.__class__, "exists", lambda p: True)
    monkeypatch.setattr(audit.OUTPUT.__class__, "iterdir", lambda p: iter([p / "budget.json"]))
    with pytest.raises(FileExistsError):
        audit.prepare()


def test_v3_bindings_write_isolation_and_worker_routing(monkeypatch, plan):
    old_output, old_worker = audit.v3.OUTPUT, audit.v3.v2.worker
    old_verify, old_atomic = audit.v3.Components.verify_plan, audit.r.atomic
    seen = []

    def body(*args):
        assert audit.v3.v2.worker is audit.worker
        assert audit.v3.v2.recovery.verify_plan is audit.verify_plan
        assert audit.v3.OUTPUT == audit.OUTPUT
        # Guard rejects before the forbidden physical atomic writer is called.
        for path in (old_output / "never.json", audit.MANIFEST, audit.r.ROOT / "never.json"):
            with pytest.raises(ValueError, match="escape"):
                audit.r.atomic(path, {})
        seen.append(args)

    monkeypatch.setattr(audit.v3, "_WORKER", body)
    audit.worker(audit.OUTPUT, plan, plan["cases"][0], 180, None, None, None, None, None)
    assert len(seen) == 1
    assert old_output == audit.v3.OUTPUT and audit.v3.v2.worker is old_worker
    assert audit.v3.Components.verify_plan is old_verify and audit.r.atomic is old_atomic


def test_frozen_manifest_hash_fails_closed(monkeypatch):
    sha = audit.selection.sha256
    monkeypatch.setattr(
        audit.selection, "sha256", lambda p: "bad" if p == audit.MANIFEST else sha(p)
    )
    with pytest.raises(ValueError, match="Frozen selection hash"):
        audit.validated_cases()
