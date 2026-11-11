"""Feedforward actor and critic networks for the Phase 6 masked-PPO baseline.

Two entirely separate networks, as required by the actor/critic separation contract: separate
static encoders, separate history encoders, separate fusion stacks, separate
heads. Nothing is shared, so no gradient path exists from the critic's value
objective into the policy, and the disjointness is asserted by a test rather
than trusted.

Both networks read the same three inputs, all produced by the frozen Phase 4
environment and never recomputed here:

- a static observation vector (692 floats for the actor under ``obs_v2``);
- the padded ordered event history, ``(80, 7)`` int16 ids plus an ``(80,)``
  bool validity mask;
- for the actor only, the ``(65,)`` legal-action mask.

The history path is the blueprint's padded feedforward encoder: expand the
seven factored ids to the 99-dim one-hot, project each event to a small vector,
**zero the padded rows after the projection** so the projection bias cannot leak
a constant, flatten, and compress. Padding is on the right, so index 0 is always
the first event of the deal.

This module is deliberately free of any recurrence. The GRU history encoder is a
later phase and must not appear here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from tarneeb.core.actions import ACTION_SPEC_VERSION, NUM_ACTIONS
from tarneeb.env.event_features import EVENT_FIELD_CARDINALITIES, EVENT_FIELD_OFFSETS
from tarneeb.env.spaces import (
    D_EVENT_ONEHOT,
    D_STATIC,
    L_MAX,
    N_EVENT_FIELDS,
    OBS_SPEC_VERSION,
)

ARCHITECTURE_SPEC_VERSION = "ff_actor_critic_v1"
"""Identifies this network family in a checkpoint. A change here invalidates checkpoints."""

HISTORY_ENCODER_FEEDFORWARD = "ff"
"""The only history encoder implemented in Phase 6. A GRU encoder is a later phase."""

RELU_GAIN = 2.0**0.5
"""Orthogonal-initialization gain for a layer followed by ReLU."""

POLICY_HEAD_GAIN = 0.01
"""Small gain so the initial policy is close to uniform over the legal actions."""


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    """Every shape the networks are built from.

    Defaults are the frozen Phase 4 contract. ``critic_observation_dim`` is a
    separate field because the Phase 6 critic is decentralized: it reads the
    same actor-safe 692-dim observation. A later centralized-critic phase can
    widen it without touching this module.
    """

    actor_observation_dim: int = D_STATIC
    critic_observation_dim: int = D_STATIC
    action_count: int = NUM_ACTIONS
    history_length: int = L_MAX
    event_field_count: int = N_EVENT_FIELDS
    event_onehot_dim: int = D_EVENT_ONEHOT
    event_projection_dim: int = 16
    static_hidden_dim: int = 512
    hidden_dim: int = 256
    history_encoder: str = HISTORY_ENCODER_FEEDFORWARD

    def __post_init__(self) -> None:
        if self.history_encoder != HISTORY_ENCODER_FEEDFORWARD:
            raise ValueError(
                f"Phase 6 implements only the {HISTORY_ENCODER_FEEDFORWARD!r} history "
                f"encoder, got {self.history_encoder!r}"
            )
        if self.action_count != NUM_ACTIONS:
            raise ValueError(f"action_count is frozen at {NUM_ACTIONS}, got {self.action_count}")
        if self.event_field_count != N_EVENT_FIELDS or self.event_onehot_dim != D_EVENT_ONEHOT:
            raise ValueError("the event encoding is frozen by event_v1 and obs_v2")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ArchitectureSpec:
        return cls(**values)


def _orthogonal_linear(in_features: int, out_features: int, gain: float) -> nn.Linear:
    """Return a Linear layer with orthogonal weights and zero bias.

    Orthogonal initialization with ``gain=sqrt(2)`` on hidden layers, a small
    gain on the policy head and unit gain on the value head is the standard PPO
    initialization. The small policy gain matters here because the very first
    rollouts must explore a 65-way masked action space.
    """
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class PaddedFeedforwardHistoryEncoder(nn.Module):
    """Encode the padded ordered event history into one fixed-size vector."""

    field_offsets: Tensor
    field_cardinalities: Tensor

    def __init__(self, spec: ArchitectureSpec) -> None:
        super().__init__()
        self.spec = spec
        self.event_projection = _orthogonal_linear(
            spec.event_onehot_dim, spec.event_projection_dim, gain=RELU_GAIN
        )
        self.compression = _orthogonal_linear(
            spec.history_length * spec.event_projection_dim, spec.hidden_dim, gain=RELU_GAIN
        )
        self.activation = nn.ReLU()

        # Where each factored field's one-hot block starts inside the 99 dims,
        # and how wide it is. Registered as buffers so they follow the module
        # across devices; the cardinalities also validate every forward pass.
        self.register_buffer(
            "field_offsets",
            torch.tensor(EVENT_FIELD_OFFSETS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "field_cardinalities",
            torch.tensor(EVENT_FIELD_CARDINALITIES, dtype=torch.long),
            persistent=False,
        )

    def expand_history_ids(self, history_ids: Tensor) -> Tensor:
        """Expand ``(B, L, 7)`` factored ids into the ``(B, L, 99)`` one-hot.

        This is the on-device version of
        :func:`tarneeb.env.event_features.expand_event_ids`, and the two are
        asserted equal by a test so the buffer format and the network can never
        drift apart.
        """
        if history_ids.ndim != 3 or history_ids.shape[2] != self.spec.event_field_count:
            raise ValueError(
                f"history_ids must be (B, L, {self.spec.event_field_count}), "
                f"got {tuple(history_ids.shape)}"
            )
        ids = history_ids.long()
        if bool(((ids < 0) | (ids >= self.field_cardinalities)).any()):
            raise ValueError("a factored event id is out of range for its field")

        onehot = torch.zeros(
            (*ids.shape[:2], self.spec.event_onehot_dim),
            dtype=torch.float32,
            device=ids.device,
        )
        expanded: Tensor = onehot.scatter_(2, ids + self.field_offsets, 1.0)
        return expanded

    def forward(self, history_ids: Tensor, history_masks: Tensor) -> Tensor:
        """Return ``(B, hidden_dim)`` from ``(B, L, 7)`` ids and a ``(B, L)`` mask."""
        if history_masks.shape != history_ids.shape[:2]:
            raise ValueError(
                f"history_masks must be {tuple(history_ids.shape[:2])}, "
                f"got {tuple(history_masks.shape)}"
            )
        one_hot_events = self.expand_history_ids(history_ids)
        projected_events = self.activation(self.event_projection(one_hot_events))
        # Zero AFTER the projection: a padded row must contribute exactly zero,
        # and the projection bias would otherwise make it contribute a constant.
        valid_rows = history_masks.to(projected_events.dtype).unsqueeze(-1)
        flattened_history = (projected_events * valid_rows).flatten(start_dim=1)
        encoded_history: Tensor = self.activation(self.compression(flattened_history))
        return encoded_history


class StaticEncoder(nn.Module):
    """Two normalized fully connected layers over the static observation."""

    def __init__(self, observation_dim: int, spec: ArchitectureSpec) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _orthogonal_linear(observation_dim, spec.static_hidden_dim, gain=RELU_GAIN),
            nn.LayerNorm(spec.static_hidden_dim),
            nn.ReLU(),
            _orthogonal_linear(spec.static_hidden_dim, spec.hidden_dim, gain=RELU_GAIN),
            nn.LayerNorm(spec.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        encoded: Tensor = self.layers(observations)
        return encoded


class FusedTrunk(nn.Module):
    """Shared body *shape* of both networks: static + history to fused features.

    Shared shape, never shared parameters: the actor and the critic each own
    their own instance, which is what keeps their parameter sets disjoint.
    """

    def __init__(self, observation_dim: int, spec: ArchitectureSpec) -> None:
        super().__init__()
        self.static_encoder = StaticEncoder(observation_dim, spec)
        self.history_encoder = PaddedFeedforwardHistoryEncoder(spec)
        self.fusion = nn.Sequential(
            _orthogonal_linear(2 * spec.hidden_dim, spec.hidden_dim, gain=RELU_GAIN),
            nn.ReLU(),
            _orthogonal_linear(spec.hidden_dim, spec.hidden_dim, gain=RELU_GAIN),
            nn.ReLU(),
        )

    def forward(self, observations: Tensor, history_ids: Tensor, history_masks: Tensor) -> Tensor:
        static_features = self.static_encoder(observations)
        history_features = self.history_encoder(history_ids, history_masks)
        fused: Tensor = self.fusion(torch.cat((static_features, history_features), dim=-1))
        return fused


class ActorNetwork(nn.Module):
    """Maps an actor-safe observation to logits over the 65 actions.

    The network only ever receives ``actor_observation_dim`` floats. It owns no
    parameter that could accept the critic's wider input, so a wiring mistake
    that fed it hidden information raises a shape error instead of training
    silently on leaked cards.
    """

    def __init__(self, spec: ArchitectureSpec | None = None) -> None:
        super().__init__()
        self.spec = spec if spec is not None else ArchitectureSpec()
        self.trunk = FusedTrunk(self.spec.actor_observation_dim, self.spec)
        self.policy_head = _orthogonal_linear(
            self.spec.hidden_dim, self.spec.action_count, gain=POLICY_HEAD_GAIN
        )

    def forward(self, observations: Tensor, history_ids: Tensor, history_masks: Tensor) -> Tensor:
        """Return raw ``(B, 65)`` logits. Legal-action masking is applied by the caller."""
        if observations.shape[-1] != self.spec.actor_observation_dim:
            raise ValueError(
                f"actor observations must have {self.spec.actor_observation_dim} features, "
                f"got {observations.shape[-1]}"
            )
        logits: Tensor = self.policy_head(self.trunk(observations, history_ids, history_masks))
        return logits


class CriticNetwork(nn.Module):
    """Maps a value observation to a scalar state value.

    In Phase 6 the value input is the same actor-safe observation, which makes
    this an independent (IPPO-style) critic. The observation width is a
    constructor parameter so a later centralized-critic phase is a configuration
    change rather than a rewrite.
    """

    def __init__(self, spec: ArchitectureSpec | None = None) -> None:
        super().__init__()
        self.spec = spec if spec is not None else ArchitectureSpec()
        self.trunk = FusedTrunk(self.spec.critic_observation_dim, self.spec)
        self.value_head = _orthogonal_linear(self.spec.hidden_dim, 1, gain=1.0)

    def forward(self, observations: Tensor, history_ids: Tensor, history_masks: Tensor) -> Tensor:
        """Return ``(B,)`` state values."""
        if observations.shape[-1] != self.spec.critic_observation_dim:
            raise ValueError(
                f"critic observations must have {self.spec.critic_observation_dim} features, "
                f"got {observations.shape[-1]}"
            )
        trunk_features = self.trunk(observations, history_ids, history_masks)
        values: Tensor = self.value_head(trunk_features).squeeze(-1)
        return values


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in ``module``."""
    return sum(parameter.numel() for parameter in module.parameters())


def parameters_are_disjoint(first: nn.Module, second: nn.Module) -> bool:
    """Return True when the two modules share no parameter tensor."""
    return {id(p) for p in first.parameters()}.isdisjoint({id(p) for p in second.parameters()})


def build_actor_and_critic(
    spec: ArchitectureSpec | None = None, seed: int | None = None
) -> tuple[ActorNetwork, CriticNetwork]:
    """Build both networks, optionally from a fixed initialization seed.

    Seeding is bracketed: the global torch RNG state is saved and restored, so
    building networks reproducibly never disturbs any other stream.
    """
    resolved_spec = spec if spec is not None else ArchitectureSpec()
    if seed is None:
        return ActorNetwork(resolved_spec), CriticNetwork(resolved_spec)

    previous_rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        actor = ActorNetwork(resolved_spec)
        critic = CriticNetwork(resolved_spec)
    finally:
        torch.random.set_rng_state(previous_rng_state)
    return actor, critic


def specification_versions() -> dict[str, str]:
    """Return the frozen contract strings this architecture is bound to."""
    return {
        "architecture_spec_version": ARCHITECTURE_SPEC_VERSION,
        "obs_spec_version": OBS_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
    }
