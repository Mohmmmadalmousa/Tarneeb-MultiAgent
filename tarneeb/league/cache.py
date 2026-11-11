"""Actor-only inference cache for immutable historical policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tarneeb.core.rules import RulesConfig
from tarneeb.league.registry import PolicyRegistry, actor_fingerprint
from tarneeb.models.recurrent import RecurrentActorNetwork
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint


class HistoricalActorCache:
    """Load each registered historical actor once, never its critic/optimizer."""

    def __init__(
        self,
        registry: PolicyRegistry,
        rules: RulesConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.registry = registry
        self.rules = rules if rules is not None else registry.rules
        self.device = device
        self._actors: dict[str, RecurrentActorNetwork] = {}
        self.loads = 0
        self.hits = 0

    def get(self, policy_id: str) -> RecurrentActorNetwork:
        """Return the frozen actor, loading and verifying it on the first request."""
        if policy_id in self._actors:
            self.hits += 1
            return self._actors[policy_id]
        record = self.registry.records[policy_id]
        if record.checkpoint_path is None or record.actor_fingerprint is None:
            raise ValueError(f"policy {policy_id!r} has no neural checkpoint")
        loaded = load_recurrent_checkpoint(
            Path(record.checkpoint_path), rules=self.rules, load_critic=False, map_location="cpu"
        )
        if loaded.critic is not None:
            raise AssertionError("actor-only checkpoint loading unexpectedly constructed a critic")
        actual_fingerprint = actor_fingerprint(loaded.actor.state_dict())
        if actual_fingerprint != record.actor_fingerprint:
            raise ValueError(f"actor fingerprint changed for policy {policy_id!r}")
        actor = loaded.actor.to(self.device)
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        self._actors[policy_id] = actor
        self.loads += 1
        return actor

    def infer(self, policy_id: str, *args: Any, **kwargs: Any) -> Any:
        """Call a cached actor with autograd disabled."""
        actor = self.get(policy_id)
        with torch.inference_mode():
            return actor(*args, **kwargs)

    def clear(self, policy_id: str | None = None) -> None:
        """Explicitly evict one cached actor or the entire cache."""
        if policy_id is None:
            self._actors.clear()
        else:
            self._actors.pop(policy_id, None)

    def statistics(self) -> dict[str, int]:
        return {"loads": self.loads, "hits": self.hits, "cached_actors": len(self._actors)}


def actor_parameters_are_frozen(actor: RecurrentActorNetwork) -> bool:
    """Return whether every cached actor parameter is excluded from gradients."""
    return all(not parameter.requires_grad for parameter in actor.parameters())
