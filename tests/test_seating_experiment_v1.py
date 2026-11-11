"""Synthetic correctness fixtures only: no optimizer or research checkpoints."""

from collections import Counter
from copy import deepcopy
from dataclasses import fields

import numpy as np
import pytest
import torch

from tarneeb.core.actions import PASS_ACTION
from tarneeb.core.rules import RulesConfig
from tarneeb.league.population import LeagueConfig, LeaguePopulation
from tarneeb.league.registry import PolicyRegistry
from tarneeb.league.sampling import PFSPConfig
from tarneeb.models.recurrent import (
    RecurrentArchitectureSpec,
    build_recurrent_actor_and_critic,
    zero_hidden_state,
)
from tarneeb.models.recurrent_checkpoint import (
    load_recurrent_checkpoint,
    save_recurrent_checkpoint,
)
from tarneeb.training.phase11_training import (
    Phase11SmokeConfig,
    Phase11SmokeResult,
    build_phase11_collector,
)
from tarneeb.training.recurrent_rollout import RecurrentRolloutMetrics, relativized_event_ids
from tarneeb.training.seating_experiment_v1 import (
    SEATING_KEY,
    SeatingEnvironmentsV1,
    build_collector,
    capture_resume,
    dealers,
    restore_resume,
    seating_identity,
)

SPEC = RecurrentArchitectureSpec(critic_observation_dim=856, hidden_size=16, static_hidden_dim=24)


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Optimizer use is prohibited in seating correctness tests")

    monkeypatch.setattr(torch.optim.Adam, "__init__", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def population(tmp_path):
    registry = PolicyRegistry()
    for policy, opening in (("champion", 52), ("passer", PASS_ACTION)):
        actor, critic = models(opening)
        path = tmp_path / f"{policy}.pt"
        save_recurrent_checkpoint(path, actor, critic, None, RulesConfig(), {}, 0, 0, 0, {})
        registry.register_checkpoint(policy, path, source_phase="seating_fixture",
                                     category="champion" if policy == "champion" else "snapshot")
    registry.register_anchor("strong", "strong")
    registry.register_anchor("random", "random")
    pop = LeaguePopulation(registry, LeagueConfig(pfsp=PFSPConfig(ema_beta=0.2)),
                           np.random.default_rng(177))
    for policy in ("strong", "random", "passer", "champion"):
        pop.add(policy)
    return pop


def models(opening=52):
    actor, critic = build_recurrent_actor_and_critic(SPEC, seed=171)
    with torch.no_grad():
        actor.policy_head.bias[52:60] = -40
        actor.policy_head.bias[PASS_ACTION] = 20
        actor.policy_head.bias[opening] = 40
    return actor, critic


def config(condition="latest_only", count=16):
    return Phase11SmokeConfig(run_name="seating_correctness", condition=condition,
                             environment_count=count, rollout_length=1)


def make(population, *, mode="balanced", condition="latest_only", count=16, opening=52):
    actor, critic = models(opening)
    return build_collector(config(condition, count), population, actor, critic,
                           RulesConfig(), mode=mode)


def assert_buffers_equal(left, right):
    assert left.sequences.keys() == right.sequences.keys()
    for key in left.sequences:
        assert len(left.sequences[key]) == len(right.sequences[key])
        for a, b in zip(left.sequences[key], right.sequences[key], strict=True):
            for field in fields(a):
                name = field.name
                x, y = getattr(a, name), getattr(b, name)
                if isinstance(x, np.ndarray):
                    np.testing.assert_array_equal(x, y)
                else:
                    assert x == y, name


def test_assignment_before_reset_and_fixed_across_resets():
    envs = SeatingEnvironmentsV1(16, 173, mode="balanced")
    assert all(env.state is None for env in envs.environments)
    assert Counter((i % 2, env.dealer_seat) for i, env in enumerate(envs.environments)) == {
        (team, dealer): 2 for team in range(2) for dealer in range(4)
    }
    for reset in (lambda: envs.reset(),
                  lambda: envs.reset_terminated(np.arange(16) % 2 == 0),
                  lambda: envs.reset(root_seed=174)):
        reset()
        assert [env.state.dealer_seat for env in envs.environments] == list(dealers(16, "balanced"))
        assert sum(env.acting_seat % 2 == i % 2 for i, env in enumerate(envs.environments)) == 8
    assert sum(((d + 1) % 4) % 2 == i % 2 for i, d in enumerate(dealers(16))) == 0
    with pytest.raises(ValueError):
        dealers(16, "typo")


@pytest.mark.parametrize("opening", [52, PASS_ACTION])
@pytest.mark.parametrize("condition", ["latest_only", "pfsp"])
def test_actual_opening_decisions_rewards_and_g6(population, monkeypatch, opening, condition):
    collector, _, _ = make(population, opening=opening, condition=condition)
    before = deepcopy(collector.actor.state_dict())
    critic_before = deepcopy(collector.critic.state_dict())
    rewards = np.zeros((16, 4))
    for i, env in enumerate(collector.environments.environments):
        original = env.step

        def record(action, original=original, i=i):
            step = original(action)
            rewards[i] += step.rewards_by_seat
            return step

        monkeypatch.setattr(env, "step", record)
    buffer, metrics = collector.collect()
    openers = []
    for i, dealer in enumerate(dealers(16, "balanced")):
        seat = (dealer + 1) % 4
        if seat % 2 == i % 2:
            transition = buffer.sequences[i, seat][0]
            assert transition.previous_event_length == 0
            assert transition.action == opening
            assert transition.legal_action_mask[52:60].all()
            openers.append(seat)
    assert Counter(openers) == dict.fromkeys(range(4), 2)
    for (i, seat), sequence in buffer.sequences.items():
        assert seat % 2 == i % 2
        assert all(t.legal_action_mask[t.action] for t in sequence)
        assert sum(t.reward for t in sequence) == pytest.approx(rewards[i, seat], abs=1e-6)
    np.testing.assert_array_equal(rewards[:, 0], rewards[:, 2])
    np.testing.assert_array_equal(rewards[:, 1], rewards[:, 3])
    np.testing.assert_allclose(rewards.sum(axis=1), 0, atol=1e-6)
    assert collector.no_unfinished_learner_sequence(buffer)
    assert collector.recurrent_state_is_zero()
    assert metrics.illegal_action_count == metrics.no_trump_action_count == 0
    assert collector.last_collection_statistics["non_finite_count"] == 0
    assert all(torch.equal(before[k], v) for k, v in collector.actor.state_dict().items())
    assert all(torch.equal(critic_before[k], v) for k, v in collector.critic.state_dict().items())
    seating_identity(collector)


@pytest.mark.parametrize("condition", ["latest_only", "pfsp"])
def test_default_is_exact_legacy_path(population, condition):
    clone = LeaguePopulation.from_state_dict(population.state_dict())
    actor, critic = models()
    actual, _, _ = build_collector(config(condition, 4), population, actor, critic, RulesConfig())
    actor, critic = models()
    expected, _, _ = build_phase11_collector(config(condition, 4), clone, actor, critic, RulesConfig())
    assert_buffers_equal(actual.collect()[0], expected.collect()[0])


def test_g5_each_policy_encoder_and_hidden_card_boundary(population):
    collector, _, _ = make(population, count=8)
    for i, env in enumerate(collector.environments.environments):
        collector._assign_opponent_policy(i)
        observer = env.acting_seat
        original = env.actor_observation()
        critic_original = env.critic_observation()
        other = [seat for seat in range(4) if seat != observer]
        a, b = other[:2]
        env.state.hands[a], env.state.hands[b] = env.state.hands[b], env.state.hands[a]
        np.testing.assert_array_equal(original, env.actor_observation())
        assert not np.array_equal(critic_original, env.critic_observation())
        env.state.hands[a], env.state.hands[b] = env.state.hands[b], env.state.hands[a]
    step = collector.environments.step(np.full(8, 52))
    collector._advance_policy_states(step.events, RecurrentRolloutMetrics())
    for i, events in enumerate(step.events):
        for seat in range(4):
            policy = collector.policy_id_for_seat(i, seat)
            actor = collector.actor if collector.is_learner_seat(i, seat) else collector.actor_cache.get(policy)
            ids = relativized_event_ids(events, seat)
            padded = torch.zeros((1, actor.spec.max_segment_length, ids.shape[-1]))
            padded[0, :len(events)] = torch.from_numpy(ids)
            expected = actor.trunk.history_encoder.advance(
                padded, torch.tensor([len(events)]), zero_hidden_state(1, actor.spec))
            actual = collector.state.actor_hidden[policy][i, seat].unsqueeze(1)
            assert torch.allclose(actual, expected, atol=1e-5)
    with pytest.raises(AssertionError, match="G6"):
        capture_resume(config(count=8), Phase11SmokeResult("seating_correctness", "latest_only"),
                       population, collector, torch.Generator(), np.random.default_rng(175))


def test_all_pass_redeal_keeps_dealers_and_zero_rewards():
    envs = SeatingEnvironmentsV1(16, 176, mode="balanced")
    envs.reset()
    hands = [env.state.hands.copy() for env in envs.environments]
    for _ in range(4):
        step = envs.step(np.full(16, PASS_ACTION))
        assert not step.terminated.any()
        assert not step.rewards_by_seat.any()
    for i, env in enumerate(envs.environments):
        assert env.dealer_seat == env.state.dealer_seat == (i // 2) % 4
        assert env.acting_seat == (env.dealer_seat + 1) % 4
        assert env.state.hands != hands[i]


def test_g8_redeal_resets_recurrent_state(population):
    collector, _, _ = make(population, count=2, opening=PASS_ACTION)

    class RedealThenBid:
        count = 0

        def sample(self):
            self.count += 1
            return "passer" if self.count <= 2 else "champion"

        def probabilities(self):
            return {"passer": 0.5, "champion": 0.5}

    collector.opponent_provider = RedealThenBid()
    buffer, metrics = collector.collect()
    assert metrics.passed_out_deals >= 2
    assert collector.recurrent_state_is_zero()
    assert collector.no_unfinished_learner_sequence(buffer)
    assert sum(t.done for seq in buffer.sequences.values() for t in seq) == 2 * metrics.completed_deals
    assert collector.last_collection_statistics["non_finite_count"] == 0
    seating_identity(collector)


@pytest.mark.parametrize("condition", ["latest_only", "pfsp"])
def test_deterministic_checkpoint_resume(population, tmp_path, condition):
    cfg = config(condition, 4)
    collector, generator, _ = make(population, condition=condition, count=4)
    clone = LeaguePopulation.from_state_dict(population.state_dict())
    repeat, _, _ = make(clone, condition=condition, count=4)
    assert_buffers_equal(collector.collect()[0], repeat.collect()[0])
    shuffle = np.random.default_rng(178)
    result = Phase11SmokeResult(cfg.run_name, cfg.condition)
    payload = capture_resume(cfg, result, population, collector, generator, shuffle)
    path = tmp_path / "resume.pt"
    save_recurrent_checkpoint(path, collector.actor, collector.critic, None, RulesConfig(),
                              cfg.as_dict(), 0, 0, 0, {}, resume_state=payload)
    loaded = load_recurrent_checkpoint(path, RulesConfig())
    restored_population = LeaguePopulation.from_state_dict(payload["population_state"])
    restored, rng, _ = build_collector(cfg, restored_population, loaded.actor, loaded.critic,
                                       RulesConfig(), mode="balanced")
    restored_shuffle = np.random.default_rng(999)
    restore_resume(loaded.payload["resume"], cfg, restored_population, restored, rng,
                   restored_shuffle, Phase11SmokeResult(cfg.run_name, cfg.condition))
    assert_buffers_equal(collector.collect()[0], restored.collect()[0])
    assert shuffle.integers(2**30) == restored_shuffle.integers(2**30)
    assert torch.equal(generator.get_state(), rng.get_state())


@pytest.mark.parametrize("mode", ["legacy", "balanced"])
def test_resume_identity_fail_closed_and_legacy_compatibility(population, mode):
    collector, rng, _ = make(population, mode=mode, count=4)
    cfg = config(count=4)
    result = Phase11SmokeResult(cfg.run_name, cfg.condition)
    shuffle = np.random.default_rng(179)
    payload = capture_resume(cfg, result, population, collector, rng, shuffle)
    original_rng = rng.get_state().clone()
    tampered = deepcopy(payload)
    tampered[SEATING_KEY]["mode"] = "balanced" if mode == "legacy" else "legacy"
    with pytest.raises(ValueError, match="seating"):
        restore_resume(tampered, cfg, population, collector, rng, shuffle, result)
    tampered = deepcopy(payload)
    tampered["environment_states"][0].dealer_seat = 3
    with pytest.raises(ValueError, match="dealer"):
        restore_resume(tampered, cfg, population, collector, rng, shuffle, result)
    del payload[SEATING_KEY]
    if mode == "legacy":
        restore_resume(payload, cfg, population, collector, rng, shuffle, result)
    else:
        with pytest.raises(ValueError, match="seating"):
            restore_resume(payload, cfg, population, collector, rng, shuffle, result)
    assert torch.equal(original_rng, rng.get_state())
