"""Historical actor cache loads once and exposes inference-only networks."""

from __future__ import annotations

from pathlib import Path

import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.league.cache import HistoricalActorCache, actor_parameters_are_frozen
from tarneeb.league.registry import PolicyRegistry
from tarneeb.models.recurrent import RecurrentCriticNetwork, build_recurrent_actor_and_critic
from tarneeb.models.recurrent_checkpoint import save_recurrent_checkpoint


def test_cache_loads_actor_only_once_and_freezes_parameters(tmp_path: Path, monkeypatch) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=51)
    checkpoint = tmp_path / "actor.pt"
    save_recurrent_checkpoint(
        checkpoint,
        actor,
        critic,
        None,
        RulesConfig(),
        {},
        0,
        0,
        0.0,
        {},
    )
    registry = PolicyRegistry()
    registry.register_checkpoint("historical", checkpoint, source_phase="phase9")
    cache = HistoricalActorCache(registry)

    import tarneeb.league.cache as cache_module

    original_loader = cache_module.load_recurrent_checkpoint
    calls = 0

    def counted_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(cache_module, "load_recurrent_checkpoint", counted_loader)
    first = cache.get("historical")
    second = cache.get("historical")

    assert first is second
    assert calls == 1
    assert first.training is False
    assert actor_parameters_are_frozen(first)
    assert all(not parameter.requires_grad for parameter in first.parameters())
    assert cache.statistics() == {"loads": 1, "hits": 1, "cached_actors": 1}


def test_actor_only_inference_returns_no_grad_outputs(tmp_path: Path) -> None:
    actor, critic = build_recurrent_actor_and_critic(seed=52)
    checkpoint = tmp_path / "actor.pt"
    save_recurrent_checkpoint(
        checkpoint,
        actor,
        critic,
        None,
        RulesConfig(),
        {},
        0,
        0,
        0.0,
        {},
    )
    registry = PolicyRegistry()
    registry.register_checkpoint("historical", checkpoint, source_phase="phase10")
    cache = HistoricalActorCache(registry)
    inference_actor = cache.get("historical")

    assert not isinstance(inference_actor, RecurrentCriticNetwork)
    observation = torch.zeros((1, inference_actor.spec.actor_observation_dim))
    events = torch.zeros((1, inference_actor.spec.max_segment_length, 7), dtype=torch.int64)
    lengths = torch.zeros((1,), dtype=torch.int64)
    hidden = torch.zeros((1, 1, inference_actor.spec.hidden_size))
    logits, _hidden = cache.infer("historical", observation, events, lengths, hidden)
    assert logits.shape == (1, 65)
    assert not logits.requires_grad
