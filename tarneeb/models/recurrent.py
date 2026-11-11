"""Event-GRU actor and critic networks (``event_gru_actor_critic_v1``).

The recurrent counterpart of :mod:`tarneeb.models.nets`. Both families exist
side by side: nothing here imports a frozen module's private name, changes a
frozen contract, or alters how a feedforward checkpoint loads.

The difference from the feedforward baseline is only in the history path. A
feedforward policy re-reads the whole padded ``(80, 7)`` event block at every
decision. A recurrent policy instead carries a hidden state that already
summarizes everything before the current decision, and consumes only the events
that happened since it last acted. That is cheaper per decision and, unlike a
fixed-length window, cannot silently truncate.

Both networks read:

- a static observation (692 floats under ``obs_v2`` for the actor; the critic
  reads the same actor-safe vector in Phase 8 -- see docs/RECURRENT_CONTRACT.md
  section 2.3, the 856-dim centralized critic is Phase 10);
- ``prev_ev_ids`` ``(B, K, 7)`` and ``prev_ev_len`` ``(B,)``, the factored
  ``event_v1`` ids of the events since this seat's previous decision;
- ``h_in`` ``(layers, B, hidden)``, that seat's own recurrent state.

and return logits (or a value) together with the advanced hidden state.

**Why a row with ``prev_ev_len == 0`` must be a strict no-op.** During rollout
the acting seat consumes nothing -- its state was already advanced when the
events happened (rule R2). During replay the first transition of every chunk
likewise consumes nothing (rule R4). If an empty segment perturbed the state by
even a rounding error, the epoch-zero PPO ratio would stop being 1 and every
update would be quietly wrong. :meth:`EventGRUHistoryEncoder.advance` therefore
returns the input tensor itself for empty rows rather than a recomputed copy.

Actor and critic share no parameter and no hidden state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence

from tarneeb.core.actions import ACTION_SPEC_VERSION, NUM_ACTIONS
from tarneeb.env.event_features import EVENT_FIELD_CARDINALITIES, EVENT_FIELD_NAMES
from tarneeb.env.spaces import (
    D_STATIC,
    K_MAX,
    N_EVENT_FIELDS,
    OBS_SPEC_VERSION,
)

RECURRENT_ARCHITECTURE_SPEC_VERSION = "event_gru_actor_critic_v1"
"""Identifies this network family in a checkpoint. Never equal to ``ff_actor_critic_v1``."""

HISTORY_ENCODER_EVENT_GRU = "event_gru"
"""The only history encoder implemented here. The feedforward one lives in nets.py."""

GRU_RESET_DEAL = "deal"
"""The only implemented reset mode: zero every slot at each newly dealt hand."""

EVENT_EMBEDDING_DIMS: tuple[int, ...] = (16, 8, 8, 8, 32, 8, 8)
"""Per-field embedding widths, taken verbatim from blueprint section 5.4.

Ordered like :data:`~tarneeb.env.event_features.EVENT_FIELD_NAMES`:
``(event_type, actor_rel, bid_value, trump, card, trick_no, winner_rel)``. They
sum to the blueprint's concatenated width of 88, and together with the
``Linear(88 -> 64)`` projection they reproduce its stated 7,816-parameter event
encoder exactly -- which is what pins this particular split rather than any
other one summing to 88.
"""

EVENT_CONCAT_DIM = sum(EVENT_EMBEDDING_DIMS)

RELU_GAIN = 2.0**0.5
"""Orthogonal-initialization gain for a layer followed by ReLU."""

POLICY_HEAD_GAIN = 0.01
"""Small gain so the initial policy is close to uniform over the legal actions."""

assert len(EVENT_EMBEDDING_DIMS) == N_EVENT_FIELDS
assert EVENT_CONCAT_DIM == 88, (
    f"blueprint section 5.4 fixes the concat at 88, got {EVENT_CONCAT_DIM}"
)


@dataclass(frozen=True, slots=True)
class RecurrentArchitectureSpec:
    """Every shape the recurrent networks are built from.

    ``critic_observation_dim`` is a separate field for the same reason it is one
    in the feedforward spec: Phase 10's centralized critic should be a
    configuration change rather than a rewrite. In Phase 8 it is the actor-safe
    width, which makes this an independent (IPPO-style) critic.
    """

    actor_observation_dim: int = D_STATIC
    critic_observation_dim: int = D_STATIC
    action_count: int = NUM_ACTIONS
    event_field_count: int = N_EVENT_FIELDS
    event_embedding_dims: tuple[int, ...] = EVENT_EMBEDDING_DIMS
    event_projection_dim: int = 64
    hidden_size: int = 256
    gru_layer_count: int = 1
    static_hidden_dim: int = 512
    max_segment_length: int = K_MAX
    history_encoder: str = HISTORY_ENCODER_EVENT_GRU
    gru_reset: str = GRU_RESET_DEAL

    def __post_init__(self) -> None:
        if self.history_encoder != HISTORY_ENCODER_EVENT_GRU:
            raise ValueError(
                f"this module implements only the {HISTORY_ENCODER_EVENT_GRU!r} history "
                f"encoder, got {self.history_encoder!r}"
            )
        if self.gru_reset != GRU_RESET_DEAL:
            raise ValueError(
                f"only gru_reset={GRU_RESET_DEAL!r} is implemented; cross-deal recurrence "
                f"is a later optional experiment, got {self.gru_reset!r}"
            )
        if self.action_count != NUM_ACTIONS:
            raise ValueError(f"action_count is frozen at {NUM_ACTIONS}, got {self.action_count}")
        if self.event_field_count != N_EVENT_FIELDS:
            raise ValueError("the event encoding is frozen by event_v1")
        if len(self.event_embedding_dims) != N_EVENT_FIELDS:
            raise ValueError(
                f"need one embedding width per event field ({N_EVENT_FIELDS}), "
                f"got {len(self.event_embedding_dims)}"
            )
        if self.gru_layer_count < 1 or self.hidden_size < 1:
            raise ValueError("gru_layer_count and hidden_size must be at least 1")

    @property
    def event_concat_dim(self) -> int:
        return sum(self.event_embedding_dims)

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["event_embedding_dims"] = list(self.event_embedding_dims)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RecurrentArchitectureSpec:
        resolved = dict(values)
        if "event_embedding_dims" in resolved:
            resolved["event_embedding_dims"] = tuple(resolved["event_embedding_dims"])
        return cls(**resolved)


def _orthogonal_linear(in_features: int, out_features: int, gain: float) -> nn.Linear:
    """Return a Linear layer with orthogonal weights and zero bias."""
    layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def zero_hidden_state(
    batch_size: int, spec: RecurrentArchitectureSpec, device: torch.device | None = None
) -> Tensor:
    """Return a ``(layers, batch, hidden)`` zero state, the defined reset value."""
    return torch.zeros(
        (spec.gru_layer_count, batch_size, spec.hidden_size),
        dtype=torch.float32,
        device=device,
    )


class FactoredEventEmbedding(nn.Module):
    """Embed the seven factored ``event_v1`` ids and project them to the GRU input.

    One table per field, concatenated to 88 dims, then ``Linear(88 -> 64) + ReLU``.
    Ids are validated against the frozen cardinalities rather than trusted: an
    out-of-range id would otherwise either index silently into the wrong row or
    raise deep inside an embedding kernel with no useful message.
    """

    def __init__(self, spec: RecurrentArchitectureSpec) -> None:
        super().__init__()
        self.spec = spec
        self.tables = nn.ModuleList(
            nn.Embedding(cardinality, width)
            for cardinality, width in zip(
                EVENT_FIELD_CARDINALITIES, spec.event_embedding_dims, strict=True
            )
        )
        self.projection = _orthogonal_linear(
            spec.event_concat_dim, spec.event_projection_dim, gain=RELU_GAIN
        )
        self.activation = nn.ReLU()

    def forward(self, event_ids: Tensor) -> Tensor:
        """Return ``(B, K, event_projection_dim)`` from ``(B, K, 7)`` factored ids."""
        if event_ids.ndim != 3 or event_ids.shape[2] != self.spec.event_field_count:
            raise ValueError(
                f"event_ids must be (B, K, {self.spec.event_field_count}), "
                f"got {tuple(event_ids.shape)}"
            )
        ids = event_ids.long()
        for field_index, cardinality in enumerate(EVENT_FIELD_CARDINALITIES):
            column = ids[..., field_index]
            if bool(((column < 0) | (column >= cardinality)).any()):
                raise ValueError(
                    f"an id is out of range for event field "
                    f"{EVENT_FIELD_NAMES[field_index]!r} (cardinality {cardinality})"
                )
        embedded = torch.cat(
            [table(ids[..., field_index]) for field_index, table in enumerate(self.tables)],
            dim=-1,
        )
        projected: Tensor = self.activation(self.projection(embedded))
        return projected


class EventGRUHistoryEncoder(nn.Module):
    """Advance a per-seat hidden state over a batch of event segments."""

    def __init__(self, spec: RecurrentArchitectureSpec) -> None:
        super().__init__()
        self.spec = spec
        self.embedding = FactoredEventEmbedding(spec)
        self.gru = nn.GRU(
            input_size=spec.event_projection_dim,
            hidden_size=spec.hidden_size,
            num_layers=spec.gru_layer_count,
            batch_first=True,
        )

    def advance(self, event_ids: Tensor, event_lengths: Tensor, hidden: Tensor) -> Tensor:
        """Return the state after consuming ``event_lengths`` events per row.

        Rows whose length is zero are returned **byte-identical** to their input,
        which is what makes a chunk-head transition a strict no-op (R2 and R4).

        Args:
            event_ids: ``(B, K, 7)`` factored ids, right-padded.
            event_lengths: ``(B,)`` how many leading rows are real, ``0..K``.
            hidden: ``(layers, B, hidden_size)``.
        """
        batch_size = hidden.shape[1]
        expected_hidden = (self.spec.gru_layer_count, batch_size, self.spec.hidden_size)
        if tuple(hidden.shape) != expected_hidden:
            raise ValueError(f"hidden must be {expected_hidden}, got {tuple(hidden.shape)}")
        if event_ids.shape[0] != batch_size or event_lengths.shape != (batch_size,):
            raise ValueError(
                f"event_ids {tuple(event_ids.shape)} and event_lengths "
                f"{tuple(event_lengths.shape)} must both have batch {batch_size}"
            )

        lengths = event_lengths.long()
        if bool((lengths < 0).any()):
            raise ValueError("an event segment length is negative")
        longest = int(lengths.max()) if batch_size else 0
        if longest > self.spec.max_segment_length:
            raise ValueError(
                f"an event segment holds {longest} events, which exceeds "
                f"K_MAX={self.spec.max_segment_length}"
            )
        if longest > event_ids.shape[1]:
            raise ValueError(
                f"event_lengths claims {longest} events but only "
                f"{event_ids.shape[1]} id rows were given"
            )
        if longest == 0:
            # Nothing to consume anywhere. Returning the input object rather than
            # a clone keeps this exactly free of numerical drift.
            return hidden

        projected = self.embedding(event_ids[:, :longest])
        # Padded positions inside a row are excluded by packing, not by zeroing,
        # so the GRU never takes a step on a pad row at all.
        packed = pack_padded_sequence(
            projected,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _output, advanced = self.gru(packed, hidden)

        # clamp(min=1) above made empty rows consume their first pad row, so put
        # the untouched state back for exactly those rows.
        keep_unchanged = (lengths == 0).view(1, batch_size, 1)
        result: Tensor = torch.where(keep_unchanged, hidden, advanced)
        return result

    def forward(self, event_ids: Tensor, event_lengths: Tensor, hidden: Tensor) -> Tensor:
        return self.advance(event_ids, event_lengths, hidden)


class RecurrentStaticEncoder(nn.Module):
    """Two normalized fully connected layers over the static observation."""

    def __init__(self, observation_dim: int, spec: RecurrentArchitectureSpec) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _orthogonal_linear(observation_dim, spec.static_hidden_dim, gain=RELU_GAIN),
            nn.LayerNorm(spec.static_hidden_dim),
            nn.ReLU(),
            _orthogonal_linear(spec.static_hidden_dim, spec.hidden_size, gain=RELU_GAIN),
            nn.LayerNorm(spec.hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        encoded: Tensor = self.layers(observations)
        return encoded


class RecurrentTrunk(nn.Module):
    """Shared body *shape* of both recurrent networks.

    Shared shape, never shared parameters: the actor and the critic each own an
    instance, which is what keeps their parameter sets and their hidden states
    separate objects.
    """

    def __init__(self, observation_dim: int, spec: RecurrentArchitectureSpec) -> None:
        super().__init__()
        self.static_encoder = RecurrentStaticEncoder(observation_dim, spec)
        self.history_encoder = EventGRUHistoryEncoder(spec)
        self.fusion = nn.Sequential(
            _orthogonal_linear(2 * spec.hidden_size, spec.hidden_size, gain=RELU_GAIN),
            nn.ReLU(),
            _orthogonal_linear(spec.hidden_size, spec.hidden_size, gain=RELU_GAIN),
            nn.ReLU(),
        )

    def forward(
        self,
        observations: Tensor,
        event_ids: Tensor,
        event_lengths: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(fused_features, advanced_hidden)``."""
        advanced = self.history_encoder.advance(event_ids, event_lengths, hidden)
        static_features = self.static_encoder(observations)
        history_features = advanced[-1]
        fused: Tensor = self.fusion(torch.cat((static_features, history_features), dim=-1))
        return fused, advanced


class RecurrentActorNetwork(nn.Module):
    """Maps an actor-safe observation plus recurrent history to 65 logits.

    The network only ever receives ``actor_observation_dim`` floats, so a wiring
    mistake that fed it the centralized critic vector raises a shape error
    instead of training silently on leaked cards.
    """

    def __init__(self, spec: RecurrentArchitectureSpec | None = None) -> None:
        super().__init__()
        self.spec = spec if spec is not None else RecurrentArchitectureSpec()
        self.trunk = RecurrentTrunk(self.spec.actor_observation_dim, self.spec)
        self.policy_head = _orthogonal_linear(
            self.spec.hidden_size, self.spec.action_count, gain=POLICY_HEAD_GAIN
        )

    def forward(
        self,
        observations: Tensor,
        event_ids: Tensor,
        event_lengths: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return raw ``(B, 65)`` logits and the advanced hidden state.

        Legal-action masking is applied by the caller through the frozen
        :class:`~tarneeb.models.masking.MaskedCategorical`.
        """
        if observations.shape[-1] != self.spec.actor_observation_dim:
            raise ValueError(
                f"actor observations must have {self.spec.actor_observation_dim} features, "
                f"got {observations.shape[-1]}"
            )
        fused, advanced = self.trunk(observations, event_ids, event_lengths, hidden)
        logits: Tensor = self.policy_head(fused)
        return logits, advanced

    def initial_hidden_state(self, batch_size: int) -> Tensor:
        return zero_hidden_state(batch_size, self.spec)


class RecurrentCriticNetwork(nn.Module):
    """Maps a value observation plus recurrent history to a scalar state value.

    In Phase 8 the value input is the same actor-safe observation, which makes
    this an independent (IPPO-style) critic. The width is a spec field so that
    Phase 10's centralized critic is a configuration change.
    """

    def __init__(self, spec: RecurrentArchitectureSpec | None = None) -> None:
        super().__init__()
        self.spec = spec if spec is not None else RecurrentArchitectureSpec()
        self.trunk = RecurrentTrunk(self.spec.critic_observation_dim, self.spec)
        self.value_head = _orthogonal_linear(self.spec.hidden_size, 1, gain=1.0)

    def forward(
        self,
        observations: Tensor,
        event_ids: Tensor,
        event_lengths: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(B,)`` state values and the advanced hidden state."""
        if observations.shape[-1] != self.spec.critic_observation_dim:
            raise ValueError(
                f"critic observations must have {self.spec.critic_observation_dim} features, "
                f"got {observations.shape[-1]}"
            )
        fused, advanced = self.trunk(observations, event_ids, event_lengths, hidden)
        values: Tensor = self.value_head(fused).squeeze(-1)
        return values, advanced

    def initial_hidden_state(self, batch_size: int) -> Tensor:
        return zero_hidden_state(batch_size, self.spec)


def build_recurrent_actor_and_critic(
    spec: RecurrentArchitectureSpec | None = None, seed: int | None = None
) -> tuple[RecurrentActorNetwork, RecurrentCriticNetwork]:
    """Build both recurrent networks, optionally from a fixed initialization seed.

    Seeding is bracketed: the global torch RNG state is saved and restored, so
    building networks reproducibly never disturbs any other stream.
    """
    resolved_spec = spec if spec is not None else RecurrentArchitectureSpec()
    if seed is None:
        return RecurrentActorNetwork(resolved_spec), RecurrentCriticNetwork(resolved_spec)

    previous_rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        actor = RecurrentActorNetwork(resolved_spec)
        critic = RecurrentCriticNetwork(resolved_spec)
    finally:
        torch.random.set_rng_state(previous_rng_state)
    return actor, critic


def recurrent_specification_versions() -> dict[str, str]:
    """Return the frozen contract strings this architecture is bound to."""
    return {
        "architecture_spec_version": RECURRENT_ARCHITECTURE_SPEC_VERSION,
        "obs_spec_version": OBS_SPEC_VERSION,
        "action_spec_version": ACTION_SPEC_VERSION,
    }
