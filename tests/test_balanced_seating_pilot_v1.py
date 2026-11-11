"""Preparation only: never execute the training loop or evaluate games."""

import json
from types import SimpleNamespace

import pytest
import torch
from scripts import run_balanced_seating_pilot_v1 as pilot

from tarneeb.core.rules import RulesConfig
from tarneeb.models.recurrent import RecurrentArchitectureSpec, build_recurrent_actor_and_critic
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint, save_recurrent_checkpoint


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    actor, critic = build_recurrent_actor_and_critic(
        RecurrentArchitectureSpec(critic_observation_dim=856, hidden_size=16, static_hidden_dim=24),
        seed=41)
    parent = tmp_path / "parent.pt"
    save_recurrent_checkpoint(parent, actor, critic, None, RulesConfig(), {}, 0, 0, 0, {})
    monkeypatch.setattr(pilot, "PARENT", parent)
    monkeypatch.setattr(pilot, "PARENT_SHA", pilot.sha(parent))
    monkeypatch.setattr(pilot, "CHAMPION", parent)
    monkeypatch.setattr(pilot, "OUT", tmp_path / "results")
    monkeypatch.setattr(pilot, "CHECKPOINTS", tmp_path / "checkpoints")

    def forbidden(*args, **kwargs):
        pytest.fail("Training/evaluation prohibited in preparation tests")

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    monkeypatch.setattr(pilot, "evaluate_condition", forbidden)
    return pilot.preflight()


def test_matched_configuration_and_read_only_preflight(prepared):
    a, b = (pilot.configuration(mode).as_dict() for mode in pilot.MODES)
    for key in ("run_name", "checkpoint_directory", "results_directory", "league_state_path"):
        assert a.pop(key) != b.pop(key)
    assert a == b
    assert a["total_updates"] == 20 and a["environment_count"] == 16
    assert a["bc_checkpoint"] is None and a["ppo"]["learning_rate"] == 3e-5
    assert not pilot.OUT.exists() and not pilot.CHECKPOINTS.exists()


def test_isolated_driver_initialization_and_collector_binding(prepared, monkeypatch):
    originals = (pilot.frozen.build_phase11_networks, pilot.frozen.build_phase11_collector,
                 pilot.frozen._capture_phase11_resume_state)
    pilot.OUT.mkdir()
    pilot.CHECKPOINTS.mkdir()
    initialized = []
    modes = []

    def collector(*args, mode):
        modes.append(mode)
        return SimpleNamespace(collect=lambda: None), None, None

    monkeypatch.setattr(pilot.seating, "build_collector", collector)

    def fake_bind(networks, collector_factory):
        bound = real_bind(networks, collector_factory)
        assert bound.__code__ is pilot.frozen.run_phase11_smoke.__code__
        assert bound.__globals__ is not pilot.frozen.run_phase11_smoke.__globals__
        assert bound.__globals__["_capture_phase11_resume_state"] is pilot.seating.capture_resume

        def stop_before_training(cfg, population, resume_from):
            assert resume_from is None
            actor, critic = networks(cfg)
            initialized.append((actor.state_dict(), critic.state_dict()))
            assert population.latest_policy_id == "parent"
            collector_factory(cfg, population, actor, critic, RulesConfig())
            raise RuntimeError("intentional stop before optimizer construction")

        return stop_before_training

    real_bind = pilot.bind_driver
    monkeypatch.setattr(pilot, "bind_driver", fake_bind)
    for mode in pilot.MODES:
        with pytest.raises(RuntimeError, match="intentional stop"):
            pilot.run_arm(mode, prepared)
    original = load_recurrent_checkpoint(pilot.PARENT)
    for actor_state, critic_state in initialized:
        for expected, actual in ((original.actor.state_dict(), actor_state),
                                 (original.critic.state_dict(), critic_state)):
            assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert modes == list(pilot.MODES)
    assert originals == (pilot.frozen.build_phase11_networks, pilot.frozen.build_phase11_collector,
                         pilot.frozen._capture_phase11_resume_state)


def test_sequential_dispatch_and_no_overwrite(prepared, monkeypatch):
    calls = []
    monkeypatch.setattr(pilot, "run_arm", lambda mode, plan: calls.append(mode))
    pilot.train_both()
    assert calls == ["legacy", "balanced"]
    assert json.loads((pilot.OUT / "plan.json").read_text()) == prepared
    with pytest.raises(FileExistsError):
        pilot.train_both()
    assert calls == ["legacy", "balanced"]


def test_failure_stops_before_second_arm(prepared, monkeypatch):
    calls = []

    def fail(mode, plan):
        calls.append(mode)
        raise RuntimeError("stop")

    monkeypatch.setattr(pilot, "run_arm", fail)
    with pytest.raises(RuntimeError, match="stop"):
        pilot.train_both()
    assert calls == ["legacy"]


def test_hash_mismatch_fails_closed(prepared, monkeypatch):
    monkeypatch.setattr(pilot, "PARENT_SHA", "bad")
    with pytest.raises(ValueError, match="SHA256"):
        pilot.preflight()
    assert not pilot.OUT.exists()


def test_evaluation_requires_complete_arms(prepared):
    pilot.OUT.mkdir()
    pilot.write_new(pilot.OUT / "plan.json", prepared)
    with pytest.raises(FileNotFoundError):
        pilot.evaluate()
    assert not (pilot.OUT / "evaluation").exists()
