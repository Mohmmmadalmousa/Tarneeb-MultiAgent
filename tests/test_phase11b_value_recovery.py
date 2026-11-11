"""Recovery tests use separate diagnostic seeds and isolated temporary output."""

import copy
import json

import numpy as np
import pytest
import torch

from tarneeb import auction_value_audit as v1
from tarneeb import auction_value_recovery as r


def fixture_plan():
    case = copy.deepcopy(r.read(r.V1_PLAN)["cases"][21])
    case.update(id="TEST_C12OS", world_seed=8_450_000, bootstrap_seed=8_450_100)
    plan = r.make_plan(
        [case], "test", {"cpu_seconds_per_state": 180, "wall_seconds": 1800, "workers": 1}
    )
    return plan, case


def synthetic_candidate(case, hands, action, policy, actor, verify):
    """Only storage/statistics tests: real replay for index 0, synthetic later rows."""
    if verify:
        return v1.play_candidate(case, hands, action, "all_strong", verify=True)
    return [26, 1, 1, 13, 2, 0, 0], None


def test_mid_world_interrupt_resume_exact_and_no_duplicate_saved_worlds(tmp_path):
    plan, case = fixture_plan()
    hands = r.worlds_for(case)
    baseline = tmp_path / "baseline"
    resumed = tmp_path / "resumed"
    r.process_worlds(baseline, plan, case, None, candidate=synthetic_candidate)
    r.process_worlds(resumed, plan, case, None, max_new=1, candidate=synthetic_candidate)
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(tuple(args[1]))
        return synthetic_candidate(*args, **kwargs)

    with pytest.raises(InterruptedError):
        r.process_worlds(
            resumed, plan, case, None, stop=lambda: len(calls) >= 1, candidate=interrupted
        )
    assert not r.world_path(resumed, case, 1).exists()
    assert set(calls) == {tuple(hands[1])}
    resumed_calls = []

    def counted(*args, **kwargs):
        resumed_calls.append(tuple(args[1]))
        return synthetic_candidate(*args, **kwargs)

    progress = r.process_worlds(resumed, plan, case, None, candidate=counted)
    assert progress == {"committed_worlds": 64, "new_worlds": 63, "complete": True}
    assert tuple(hands[0]) not in resumed_calls
    for i in range(1, 64):
        assert resumed_calls.count(tuple(hands[i])) == 2 * len(case["actions"])
    assert r.publish_result(baseline, plan, case) == r.publish_result(resumed, plan, case)
    again = r.process_worlds(
        resumed, plan, case, None, candidate=lambda *a, **kw: pytest.fail("duplicate world")
    )
    assert again["new_worlds"] == 0


def test_atomic_record_excludes_partial_temp_and_refuses_overwrite(tmp_path):
    plan, case = fixture_plan()
    path = r.world_path(tmp_path, case, 0)
    path.parent.mkdir(parents=True)
    path.with_name(path.name + ".interrupted.tmp").write_text('{"payload":')
    assert not path.exists()
    r.process_worlds(tmp_path, plan, case, None, max_new=1, candidate=synthetic_candidate)
    assert path.exists()
    with pytest.raises(FileExistsError):
        r.atomic(path, {"wrong": True})
    r.load_world(path, plan, case, 0, r.worlds_for(case)[0])


@pytest.mark.parametrize(
    "fault", ["checksum", "policy", "actions", "seed", "state", "hands", "likelihood"]
)
def test_invalid_records_fail_closed(tmp_path, fault):
    plan, case = fixture_plan()
    r.process_worlds(tmp_path, plan, case, None, max_new=1, candidate=synthetic_candidate)
    path = r.world_path(tmp_path, case, 0)
    envelope = r.read(path)
    p = envelope["payload"]
    if fault == "checksum":
        envelope["sha256"] = "bad"
    elif fault == "policy":
        del p["outcomes"]["all_strong"]
    elif fault == "actions":
        p["outcomes"]["all_strong"].pop()
    elif fault == "seed":
        p["plan_identity"] = "other_plan"
    elif fault == "state":
        p["state_identity"] = "other_state"
    elif fault == "hands":
        p["hands"][1], p["hands"][2] = p["hands"][2], p["hands"][1]
    elif fault == "likelihood":
        p["weight"] *= 2
    if fault != "checksum":
        envelope["sha256"] = r.digest(p)
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError):
        r.process_worlds(tmp_path, plan, case, None, max_new=1)
    assert not (path.parent / "result.json").exists()


def test_no_final_statistics_until_64_complete_worlds(tmp_path):
    plan, case = fixture_plan()
    r.process_worlds(tmp_path, plan, case, None, max_new=2, candidate=synthetic_candidate)
    with pytest.raises(FileNotFoundError):
        r.publish_result(tmp_path, plan, case)
    assert not (r.world_path(tmp_path, case, 0).parent / "result.json").exists()
    r.process_worlds(tmp_path, plan, case, None, candidate=synthetic_candidate)
    result = r.publish_result(tmp_path, plan, case)
    assert result["completed_worlds"] == 64
    r.world_path(tmp_path, case, 63).unlink()
    with pytest.raises(FileNotFoundError):
        r.validate_result(tmp_path, plan, case)


def test_state_statistics_exactly_reuse_v1_rules(tmp_path):
    plan, case = fixture_plan()
    r.process_worlds(tmp_path, plan, case, None, candidate=synthetic_candidate)
    records = [r.read(r.world_path(tmp_path, case, i))["payload"] for i in range(64)]
    result = r.publish_result(tmp_path, plan, case)
    weights = np.array([x["weight"] for x in records])
    for policy in v1.CONFIG["continuations"]:
        values = np.array([x["outcomes"][policy] for x in records], dtype=float)
        for belief in v1.CONFIG["beliefs"]:
            expected = v1.summarize(
                values,
                weights if belief == "bid_conditioned" else np.ones(64),
                case["actions"],
                case["teacher"],
                case["bootstrap_seed"],
            )
            assert result["summaries"][policy][belief] == expected


def test_configuration_source_and_seed_binding():
    plan, case = fixture_plan()
    r.verify_plan(plan)
    changed = copy.deepcopy(plan)
    changed["cases"][0]["world_seed"] += 1
    with pytest.raises(ValueError):
        r.verify_plan(changed)
    changed = copy.deepcopy(plan)
    changed["source_sha256"] = "0" * 64
    changed["identity"] = r.digest({k: v for k, v in changed.items() if k != "identity"})
    with pytest.raises(ValueError):
        r.verify_plan(changed)
    case["world_seed"] = 8_410_000
    with pytest.raises(ValueError):
        r.make_plan([case], "benchmark", {})


def test_unclean_restart_does_not_reset_budgets(tmp_path):
    plan, case = fixture_plan()
    r.atomic(
        tmp_path / "budget.json",
        {
            "plan_identity": plan["identity"],
            "wall_spent": 3,
            "active_session": {"reserved_wall": 17},
            "states": {case["id"]: {"spent": 4, "reserved_cpu": 6}},
        },
    )
    ledger = r.load_ledger(tmp_path, plan)
    assert ledger["wall_spent"] == 20
    assert ledger["states"][case["id"]]["spent"] == 10
    assert r.load_ledger(tmp_path, plan) == ledger


def test_concurrent_writer_is_rejected(tmp_path):
    with r.exclusive_run(tmp_path), pytest.raises(OSError), r.exclusive_run(tmp_path):
        pytest.fail("second coordinator acquired the same lock")


def test_crash_lease_retains_remaining_budget_and_invalid_lease_fails_closed(tmp_path):
    plan, case = fixture_plan()
    state = {"spent": 4, "reserved_cpu": 100, "cpu_lease": "lease.json"}
    r.atomic(
        tmp_path / "lease.json",
        {"plan_identity": plan["identity"], "attempt": "lease.json", "cpu_upper_bound": 7.5},
    )
    assert r.crash_cpu_charge(tmp_path, plan, state) == 7.5
    r.atomic(
        tmp_path / "budget.json",
        {
            "plan_identity": plan["identity"],
            "wall_spent": 3,
            "active_session": {"reserved_wall": 12},
            "states": {case["id"]: state},
        },
    )
    recovered = r.load_ledger(tmp_path, plan)
    assert recovered["states"][case["id"]]["spent"] == 11.5
    assert recovered["wall_spent"] == 15
    r.atomic(
        tmp_path / "lease.json",
        {"plan_identity": "wrong", "attempt": "lease.json", "cpu_upper_bound": 0},
        replace=True,
    )
    assert r.crash_cpu_charge(tmp_path, plan, state) == 100


def test_real_neural_resume_matches_uninterrupted_and_preserves_weights(tmp_path):
    from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

    torch.set_num_threads(1)
    plan, case = fixture_plan()
    actor = (
        load_recurrent_checkpoint(v1.SOURCE, load_critic=False).actor.eval().requires_grad_(False)
    )
    tensors = {k: v.clone() for k, v in actor.state_dict().items()}
    before = v1.sha(v1.SOURCE)
    one = tmp_path / "one"
    two = tmp_path / "two"
    seen = []

    def observed(*args, **kwargs):
        seen.append(tuple(args[1]))
        return v1.play_candidate(*args, **kwargs)

    r.process_worlds(one, plan, case, actor, max_new=2, candidate=observed)
    hands = r.worlds_for(case)
    assert seen.count(tuple(hands[0])) == 4 and seen.count(tuple(hands[1])) == 4
    r.process_worlds(two, plan, case, actor, max_new=1)
    r.process_worlds(two, plan, case, actor, max_new=1)
    for i in range(2):
        assert (
            r.read(r.world_path(one, case, i))["payload"]
            == r.read(r.world_path(two, case, i))["payload"]
        )
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in tensors.items())
    assert v1.sha(v1.SOURCE) == before


def test_original_checkpoints_are_write_locked():
    with r.checkpoint_read_locks(), pytest.raises(PermissionError), v1.SOURCE.open("r+b"):
        pytest.fail("Original checkpoint was writable during diagnostic execution")


def test_cpu_watchdog_stops_worker_and_debits_budget(tmp_path):
    from scripts.audit_phase11b_values_v2 import run_session

    plan, case = fixture_plan()
    plan["limits"]["cpu_seconds_per_state"] = 0.001
    plan["identity"] = r.digest({k: v for k, v in plan.items() if k != "identity"})
    report = run_session(tmp_path, plan, workers=1, max_new=1)
    assert report["status"] == "cpu_limit"
    assert report["budget"]["states"][case["id"]]["spent"] >= 0.001
    assert not r.world_path(tmp_path, case, 0).exists()
    assert report["budget"]["active_session"] is None


def test_wall_watchdog_stops_before_any_final_result(tmp_path):
    from scripts.audit_phase11b_values_v2 import run_session

    plan, _ = fixture_plan()
    plan["limits"]["wall_seconds"] = 0.001
    plan["identity"] = r.digest({k: v for k, v in plan.items() if k != "identity"})
    report = run_session(tmp_path, plan, workers=1, max_new=1)
    assert report["status"] == "wall_limit"
    assert not list(tmp_path.rglob("result.json"))
    assert report["budget"]["active_session"] is None
