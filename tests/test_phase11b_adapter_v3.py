"""Adapter tests: deterministic placeholders/synthetic outcomes, never simulation."""

import copy
import multiprocessing as mp
from pathlib import Path

import pytest
import torch
from scripts import audit_phase11b_values_v3 as v3

r = v3.recovery


def resign(plan):
    plan["identity"] = r.digest({k: v for k, v in plan.items() if k != "identity"})


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No hidden-world sampling, continuation, neural execution or training")

    for name in ("sample_hands", "play_candidate", "initialize_worker"):
        monkeypatch.setattr(v3.v2.v1, name, forbidden)
    monkeypatch.setattr(r, "worlds_for", forbidden)
    monkeypatch.setattr(v3.data.TarneebEngine, "deal", forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    step = v3.data.TarneebEngine.step

    def prefix_only(self, state, action):
        assert state.phase is v3.data.Phase.BID and 52 <= action <= 59
        return step(self, state, action)

    monkeypatch.setattr(v3.data.TarneebEngine, "step", prefix_only)


@pytest.fixture(scope="module")
def original_plan():
    return v3.make_plan()


@pytest.fixture
def plan(monkeypatch, tmp_path, original_plan):
    monkeypatch.setattr(v3, "OUTPUT", tmp_path)
    plan = copy.deepcopy(original_plan)
    plan["output_directory"] = str(tmp_path.absolute())
    resign(plan)
    return plan


def test_dataset_identity_legal_masks_histories_and_overlap(original_plan):
    assert original_plan["cases"] == v3.validate_dataset()
    assert len(original_plan["cases"]) == 16
    assert sum(len(c["actions"]) for c in original_plan["cases"]) == 104
    assert original_plan["limits"] == {
        "cpu_seconds_per_state": 180,
        "wall_seconds": 1800,
        "workers": 4,
    }
    assert original_plan["scientific_config"] == v3.v2.v1.CONFIG
    assert v3.make_plan() == original_plan
    v3.verify_plan(original_plan)


@pytest.mark.parametrize(
    "fault", ["case", "seed", "limits", "purpose", "output", "source", "dataset", "manifest"]
)
def test_changed_identity_rejected_even_with_recomputed_digest(original_plan, fault):
    plan = copy.deepcopy(original_plan)
    if fault == "case":
        plan["cases"][0]["prefix"] = []
    elif fault == "seed":
        plan["cases"][0]["world_seed"] += 100
    elif fault == "limits":
        plan["limits"]["wall_seconds"] += 1
    elif fault == "purpose":
        plan["purpose"] = "audit"
    elif fault == "output":
        plan["output_directory"] = str(r.ROOT)
    elif fault == "source":
        plan["source_sha256"] = "0" * 64
    elif fault == "dataset":
        plan["dataset_hashes"]["cases.json"] = "0" * 64
    else:
        plan["source_hashes"].pop(next(iter(plan["source_hashes"])))
    resign(plan)
    with pytest.raises(ValueError):
        v3.verify_plan(plan)


def test_preflight_cannot_execute_and_does_not_reset_budget(plan, monkeypatch, tmp_path):
    v3.atomic(tmp_path / "preregistration.json", plan)
    v3.atomic(tmp_path / "budget.json", {"sentinel": "must remain unchanged"})
    before = (tmp_path / "budget.json").read_bytes()
    monkeypatch.setattr(
        v3.v2, "run_session", lambda *a, **kw: pytest.fail("execution during preflight")
    )
    assert v3.preflight() == plan
    assert (tmp_path / "budget.json").read_bytes() == before
    assert not list(tmp_path.glob("states/*/world_*.json"))


def test_prepare_refuses_existing_artifacts(plan, tmp_path):
    v3.atomic(tmp_path / "preregistration.json", plan)
    with pytest.raises(FileExistsError):
        v3.prepare()


def test_write_guard_and_context_restoration(plan, tmp_path):
    old_atomic, old_worker, old_components = r.atomic, v3.v2.worker, v3.v2.recovery
    with v3.bound_components():
        r.atomic(tmp_path / "safe.json", {"ok": True})
        for path in [
            r.ROOT / "NEVER_WRITE.json",
            v3.DATASET / "NEVER_WRITE.json",
            tmp_path / "../escape.json",
        ]:
            with pytest.raises(ValueError, match="escape"):
                r.atomic(path, {})
        with pytest.raises(ValueError):
            v3.v2.recovery.exclusive_run(r.ROOT)
    assert (
        r.atomic is old_atomic and v3.v2.worker is old_worker and v3.v2.recovery is old_components
    )
    assert r.read(tmp_path / "safe.json") == {"ok": True}


def test_cli_rejects_old_output_and_other_case_file():
    with pytest.raises(ValueError):
        v3.main(["--prepare", "--output", str(r.ROOT)])
    with pytest.raises(ValueError):
        v3.main(["--prepare", "--cases", str(r.V1_PLAN)])


def placeholders(case):
    unseen = sorted(set(range(52)) - set(case["own_hand"]))
    return [v3.data.hand_from_cards(case["own_hand"])] + [
        v3.data.hand_from_cards(unseen[i : i + 13]) for i in range(0, 39, 13)
    ]


def fake_candidate(case, hands, action, policy, actor, verify):
    # Synthetic outcome schema, not a simulated game. World-zero fake replay
    # metadata is checked only for presence in this storage test (see fixture).
    return [case["actions"].index(action), 1, 1, 10, 0, 0, 0], (
        {"synthetic": True} if verify else None
    )


@pytest.fixture
def synthetic_storage(monkeypatch, plan):
    case = plan["cases"][0]
    fixed = placeholders(case)
    monkeypatch.setattr(r, "worlds_for", lambda c: [fixed.copy() for _ in range(64)])
    real_validate = r.validate_world

    def validate(payload, plan, case, index, hands, **kwargs):
        return real_validate(payload, plan, case, index, hands, verify_replay=False)

    monkeypatch.setattr(r, "validate_world", validate)
    return case, fixed


def test_complete_action_pairing_uses_identical_supplied_world(plan):
    for case in (plan["cases"][0], plan["cases"][8]):
        fixed, seen = placeholders(case), []

        def candidate(case, hands, action, policy, actor, verify, seen=seen):
            seen.append((tuple(hands), action, policy))
            hands[1] = -1  # Per-candidate list copy prevents contamination.
            return fake_candidate(case, hands, action, policy, actor, verify)

        payload = r.compute_world(plan, case, 1, fixed, None, candidate=candidate)
        assert {x[0] for x in seen} == {tuple(fixed)}
        assert len(seen) == 2 * len(case["actions"])
        assert {(a, p) for _, a, p in seen} == {
            (a, p) for a in case["actions"] for p in v3.v2.v1.CONFIG["continuations"]
        }
        r.validate_world(payload, plan, case, 1, fixed)
        payload["outcomes"]["all_strong"].pop()
        with pytest.raises(ValueError, match="pairing"):
            r.validate_world(payload, plan, case, 1, fixed)


def test_resume_identical_no_duplicate_worlds_and_partial_not_final(
    plan, synthetic_storage, tmp_path, monkeypatch
):
    case, _ = synthetic_storage
    calls = []
    original_compute = r.compute_world

    def compute(plan, case, index, *args, **kwargs):
        calls.append(index)
        return original_compute(plan, case, index, *args, **kwargs)

    monkeypatch.setattr(r, "compute_world", compute)
    with v3.bound_components():
        baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
        r.process_worlds(baseline, plan, case, None, candidate=fake_candidate)
        r.process_worlds(resumed, plan, case, None, max_new=3, candidate=fake_candidate)
        with pytest.raises(FileNotFoundError):
            r.publish_result(resumed, plan, case)
        mid_calls = []

        def interrupted(*args, **kwargs):
            mid_calls.append(1)
            return fake_candidate(*args, **kwargs)

        with pytest.raises(InterruptedError):
            r.process_worlds(
                resumed, plan, case, None, candidate=interrupted, stop=lambda: bool(mid_calls)
            )
        assert not r.world_path(resumed, case, 3).exists()
        calls.clear()
        r.process_worlds(resumed, plan, case, None, candidate=fake_candidate)
        assert calls == list(range(3, 64))
        assert r.publish_result(baseline, plan, case) == r.publish_result(resumed, plan, case)
        calls.clear()
        r.process_worlds(resumed, plan, case, None, candidate=fake_candidate)
        assert calls == []
        r.world_path(resumed, case, 63).unlink()
        with pytest.raises(FileNotFoundError):
            r.validate_result(resumed, plan, case)


def test_v3_publishes_only_after_complete_state(plan, synthetic_storage, tmp_path):
    case, _ = synthetic_storage
    with v3.bound_components():
        api = v3.Components()
        api.process_worlds(tmp_path, plan, case, None, max_new=1, candidate=fake_candidate)
        assert not (tmp_path / "states" / case["id"] / "result.json").exists()
        api.process_worlds(tmp_path, plan, case, None, candidate=fake_candidate)
        result = r.validate_result(tmp_path, plan, case)
        assert result["completed_worlds"] == 64


@pytest.mark.parametrize(
    "fault", ["checksum", "policy", "actions", "plan", "state", "hands", "likelihood"]
)
def test_corrupt_or_incomplete_saved_world_rejected(plan, synthetic_storage, tmp_path, fault):
    case, _ = synthetic_storage
    with v3.bound_components():
        r.process_worlds(tmp_path, plan, case, None, max_new=1, candidate=fake_candidate)
        path = r.world_path(tmp_path, case, 0)
        envelope = r.read(path)
        payload = envelope["payload"]
        if fault == "checksum":
            envelope["sha256"] = "invalid"
        elif fault == "policy":
            del payload["outcomes"]["all_strong"]
        elif fault == "actions":
            payload["outcomes"]["all_strong"].pop()
        elif fault == "plan":
            payload["plan_identity"] = "invalid"
        elif fault == "state":
            payload["state_identity"] = "invalid"
        elif fault == "hands":
            payload["hands"][1], payload["hands"][2] = payload["hands"][2], payload["hands"][1]
        else:
            payload["weight"] *= 2
        if fault != "checksum":
            envelope["sha256"] = r.digest(payload)
        # Deliberately corrupt only the isolated synthetic test record.
        v3.atomic(path, envelope, replace=True)
        with pytest.raises(ValueError):
            r.process_worlds(tmp_path, plan, case, None, candidate=fake_candidate)
        assert not (path.parent / "result.json").exists()


def spawn_probe(root, plan, send):
    # Explicit child-side stand-in for V2 worker body: test routing, never initialize actor.
    from scripts import audit_phase11b_values_v3 as child

    child.OUTPUT = Path(root)

    def body(*args):
        child.v2.recovery.verify_plan(plan)
        child.recovery.atomic(Path(root) / "spawn_probe.json", {"verified": True})
        try:
            child.recovery.atomic(child.DATASET / "NEVER_WRITE.json", {})
        except ValueError:
            send.send("isolated")

    child._WORKER = body
    child.worker(root, plan, plan["cases"][0], 180, None, None, None, None, None)
    send.close()


def test_windows_spawn_installs_v3_identity_and_write_guard(plan, tmp_path):
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=spawn_probe, args=(str(tmp_path), plan, send))
    process.start()
    send.close()
    try:
        assert receive.poll(30) and receive.recv() == "isolated"
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        receive.close()


def test_cpu_and_wall_exhaustion_never_launch_worker(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(v3, "worker", lambda *a, **kw: pytest.fail("worker started"))
    ledger = {
        "plan_identity": plan["identity"],
        "wall_spent": 0,
        "active_session": None,
        "states": {c["id"]: {"spent": 180, "reserved_cpu": 0} for c in plan["cases"]},
    }
    v3.atomic(tmp_path / "budget.json", ledger)
    with v3.bound_components():
        report = v3.v2.run_session(tmp_path, plan, workers=4)
    assert report["status"] == "state_cpu_budget_exhausted"
    ledger = r.read(tmp_path / "budget.json")
    ledger["wall_spent"] = 1800
    v3.atomic(tmp_path / "budget.json", ledger, replace=True)
    with v3.bound_components(), pytest.raises(TimeoutError):
        v3.v2.run_session(tmp_path, plan, workers=4)
