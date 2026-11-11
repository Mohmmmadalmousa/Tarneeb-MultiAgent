"""Per-``(environment, seat, policy)`` recurrent state and event bookkeeping.

This module owns the answer to "what does each seat remember, and which events
has it already seen?". It holds no network and computes no loss; it is a
container with strict invariants, because almost every recurrent bug is a
bookkeeping bug rather than a mathematics bug.

The three quantities it tracks, per policy:

``actor_hidden`` / ``critic_hidden``
    ``(n_envs, 4, layers, hidden)``. One slot per ``(environment, seat)``, for
    the actor and the critic **independently**. Never one per environment, never
    one per team, never shared between actor and critic. The critic needs its
    own per-seat state for the same reason the actor does: its observation and
    its event stream are both expressed relative to the seat, so folding two
    seats into one vector would mix coordinate frames in which "partner" means
    different players.

``pending`` event queues
    Per ``(environment, seat)``, the factored ids of the events consumed since
    that seat's last **stored** decision. This is the segment rule R4 writes into
    the buffer, and it is cleared at every store.

``last_seq_consumed``
    Per ``(environment, seat)``, the ``seq`` of the last event folded into that
    seat's state, or ``-1`` at a freshly dealt hand. Because
    ``TarneebEngine.deal`` restarts ``seq`` at 0 and ``state.events`` holds
    exactly the current hand's events, this must always equal
    ``len(state.events) - 1``. The collector asserts that on every step, which
    catches a dropped, duplicated or out-of-order consumption at the moment it
    happens rather than three hours into a run.

Keying by ``policy_id`` costs nothing now and prevents a whole bug class later:
when a frozen checkpoint or a league opponent holds a seat, that seat's memory
belongs to *that policy's* recurrent weights and must not be advanced by the
current learner's encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.event_features import EVENT_ID_DTYPE
from tarneeb.env.spaces import K_MAX, N_EVENT_FIELDS
from tarneeb.models.recurrent import RecurrentArchitectureSpec

LEARNER_POLICY_ID = "learner"
"""The policy id of the network being trained. Others appear in later phases."""

NO_EVENT_CONSUMED = -1
"""``last_seq_consumed`` at a freshly dealt hand, before any event exists."""


def empty_segment_ids(batch_size: int = 1, length: int = K_MAX) -> np.ndarray:
    """Return an all-zero ``(batch, length, 7)`` id block: the empty segment."""
    return np.zeros((batch_size, length, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE)


@dataclass
class PendingEventQueue:
    """The events one seat has consumed since its last stored decision.

    Kept as a plain list because a seat the learner does not control is never
    stored, so its backlog legitimately grows to the length of a whole hand.
    The ``K_MAX`` bound applies to a *stored* segment and is checked at the point
    of storage, where a violation actually means something.
    """

    rows: list[np.ndarray]

    def __init__(self) -> None:
        self.rows = []

    def extend(self, id_rows: np.ndarray) -> None:
        """Append relativized event id rows, in chronological order."""
        for row in id_rows:
            self.rows.append(np.asarray(row, dtype=EVENT_ID_DTYPE).copy())

    def clear(self) -> None:
        self.rows.clear()

    def __len__(self) -> int:
        return len(self.rows)

    def as_padded_block(self, length: int = K_MAX) -> tuple[np.ndarray, int]:
        """Return ``(ids (length, 7) right-padded, count)``.

        Raises:
            ValueError: If the queue is longer than ``length``. A stored segment
                longer than ``K_MAX`` means the proved bound is wrong, so this
                fails loudly rather than truncating history.
        """
        count = len(self.rows)
        if count > length:
            raise ValueError(
                f"a between-decision segment holds {count} events, which exceeds K_MAX={length}; "
                "the proved bound is 14, so this means the consumption rule is broken"
            )
        block = np.zeros((length, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE)
        for position, row in enumerate(self.rows):
            block[position] = row
        return block, count

    def snapshot(self) -> list[list[int]]:
        """Return a plain-Python copy, for a resume payload."""
        return [[int(value) for value in row] for row in self.rows]

    @classmethod
    def from_snapshot(cls, rows: list[list[int]]) -> PendingEventQueue:
        queue = cls()
        queue.rows = [np.asarray(row, dtype=EVENT_ID_DTYPE) for row in rows]
        return queue


class RecurrentStateStore:
    """All recurrent bookkeeping for one vectorized run.

    Every mutating method is deliberately explicit about which slots it touches,
    so that "did this reset leak into another environment?" is answerable by
    reading the code rather than by running an experiment.
    """

    def __init__(
        self,
        environment_count: int,
        spec: RecurrentArchitectureSpec,
        policy_ids: tuple[str, ...] = (LEARNER_POLICY_ID,),
    ) -> None:
        if environment_count < 1:
            raise ValueError(f"environment_count must be at least 1, got {environment_count}")
        if not policy_ids:
            raise ValueError("at least one policy id is required")

        self.environment_count = environment_count
        self.spec = spec
        self.policy_ids = tuple(policy_ids)

        shape = (environment_count, NUM_SEATS, spec.gru_layer_count, spec.hidden_size)
        # Separate tensors per policy AND per network. Four distinct storage
        # objects even in the single-policy case, so no aliasing is possible.
        self.actor_hidden: dict[str, Tensor] = {
            policy_id: torch.zeros(shape, dtype=torch.float32) for policy_id in self.policy_ids
        }
        self.critic_hidden: dict[str, Tensor] = {
            policy_id: torch.zeros(shape, dtype=torch.float32) for policy_id in self.policy_ids
        }

        self.pending: list[list[PendingEventQueue]] = [
            [PendingEventQueue() for _ in range(NUM_SEATS)] for _ in range(environment_count)
        ]
        self.last_seq_consumed = np.full(
            (environment_count, NUM_SEATS), NO_EVENT_CONSUMED, dtype=np.int64
        )
        self.chunk_position = np.zeros((environment_count, NUM_SEATS), dtype=np.int64)
        self.force_new_chunk = np.ones((environment_count, NUM_SEATS), dtype=bool)

    # --- reads --------------------------------------------------------------

    def actor_slot(
        self, environment_index: int, seat: int, policy_id: str = LEARNER_POLICY_ID
    ) -> Tensor:
        """Return this seat's actor state as ``(layers, 1, hidden)``, a view."""
        return self.actor_hidden[policy_id][environment_index, seat].unsqueeze(1)

    def critic_slot(
        self, environment_index: int, seat: int, policy_id: str = LEARNER_POLICY_ID
    ) -> Tensor:
        """Return this seat's critic state as ``(layers, 1, hidden)``, a view."""
        return self.critic_hidden[policy_id][environment_index, seat].unsqueeze(1)

    def gather_actor(
        self,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        policy_id: str = LEARNER_POLICY_ID,
    ) -> Tensor:
        """Return ``(layers, B, hidden)`` for the given ``(env, seat)`` pairs."""
        rows = self.actor_hidden[policy_id][environment_indices, seats]
        return rows.permute(1, 0, 2).contiguous()

    def gather_critic(
        self,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        policy_id: str = LEARNER_POLICY_ID,
    ) -> Tensor:
        """Return ``(layers, B, hidden)`` for the given ``(env, seat)`` pairs."""
        rows = self.critic_hidden[policy_id][environment_indices, seats]
        return rows.permute(1, 0, 2).contiguous()

    # --- writes -------------------------------------------------------------

    def scatter_actor(
        self,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        hidden: Tensor,
        policy_id: str = LEARNER_POLICY_ID,
    ) -> None:
        """Write ``(layers, B, hidden)`` back into the given slots."""
        self.actor_hidden[policy_id][environment_indices, seats] = (
            hidden.permute(1, 0, 2).detach().contiguous()
        )

    def scatter_critic(
        self,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        hidden: Tensor,
        policy_id: str = LEARNER_POLICY_ID,
    ) -> None:
        """Write ``(layers, B, hidden)`` back into the given slots."""
        self.critic_hidden[policy_id][environment_indices, seats] = (
            hidden.permute(1, 0, 2).detach().contiguous()
        )

    def append_events(self, environment_index: int, seat: int, id_rows: np.ndarray) -> None:
        """Append consumed events to one seat's pending queue."""
        self.pending[environment_index][seat].extend(id_rows)

    def stored_segment(
        self, environment_index: int, seat: int, is_chunk_head: bool
    ) -> tuple[np.ndarray, int]:
        """Return the ``prev_events`` block for a transition about to be stored.

        A chunk head stores the empty segment (R4): the events before it are
        already summarized by the stored ``h0``, so replaying them would consume
        them twice.
        """
        if is_chunk_head:
            return np.zeros((K_MAX, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE), 0
        return self.pending[environment_index][seat].as_padded_block()

    def note_decision_stored(self, environment_index: int, seat: int, chunk_length: int) -> None:
        """Clear this seat's backlog and advance its chunk position."""
        self.pending[environment_index][seat].clear()
        self.force_new_chunk[environment_index, seat] = False
        self.chunk_position[environment_index, seat] = (
            self.chunk_position[environment_index, seat] + 1
        ) % chunk_length

    def is_chunk_head(self, environment_index: int, seat: int) -> bool:
        """True when the next stored decision starts a new chunk."""
        return bool(
            self.force_new_chunk[environment_index, seat]
            or self.chunk_position[environment_index, seat] == 0
        )

    def begin_rollout(self) -> None:
        """Force a chunk boundary everywhere, **without** touching hidden state.

        A rollout window ends wherever it happens to end, usually mid-deal, and
        the next window is stored in a fresh buffer. A chunk cannot span the two,
        so every ``(environment, seat)`` starts the new window at a head.

        Crucially this does *not* reset the recurrent state: the deal is still in
        progress and the seat's memory of it is still valid. The head simply
        stores that continuing state as its ``h0``, and the backlog it discards
        is already summarized there -- which is exactly the truncated-BPTT
        boundary, not a loss of information.
        """
        self.force_new_chunk[:, :] = True
        self.chunk_position[:, :] = 0

    def reset_environment(self, environment_index: int) -> None:
        """Apply the newly-dealt-hand reset to exactly one environment.

        Zeroes all four actor slots and all four critic slots of **every**
        policy, clears all four queues, forgets the consumption cursor and forces
        a chunk boundary. Other environments are untouched, which is asserted by
        a test rather than trusted.
        """
        for policy_id in self.policy_ids:
            self.actor_hidden[policy_id][environment_index].zero_()
            self.critic_hidden[policy_id][environment_index].zero_()
        for seat in range(NUM_SEATS):
            self.pending[environment_index][seat].clear()
        self.last_seq_consumed[environment_index, :] = NO_EVENT_CONSUMED
        self.chunk_position[environment_index, :] = 0
        self.force_new_chunk[environment_index, :] = True

    def reset_all(self) -> None:
        for environment_index in range(self.environment_count):
            self.reset_environment(environment_index)

    # --- invariants ---------------------------------------------------------

    def assert_consumption_matches(
        self, environment_index: int, event_count: int, seat: int | None = None
    ) -> None:
        """Assert a seat has consumed exactly the events the engine has emitted.

        ``event_count`` is ``len(state.events)`` for the environment's current
        hand. Because ``seq`` is 0-based, contiguous and restarts at each deal,
        the last consumed ``seq`` must be ``event_count - 1``.
        """
        seats = range(NUM_SEATS) if seat is None else (seat,)
        for checked_seat in seats:
            observed = int(self.last_seq_consumed[environment_index, checked_seat])
            expected = event_count - 1
            if observed != expected:
                raise AssertionError(
                    f"environment {environment_index} seat {checked_seat} has consumed up to "
                    f"seq {observed} but the engine has emitted {event_count} events "
                    f"(expected last seq {expected})"
                )

    # --- resume -------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return everything a resume needs. Tensors are cloned, not aliased."""
        return {
            "policy_ids": list(self.policy_ids),
            "actor_hidden": {
                policy_id: tensor.clone() for policy_id, tensor in self.actor_hidden.items()
            },
            "critic_hidden": {
                policy_id: tensor.clone() for policy_id, tensor in self.critic_hidden.items()
            },
            "pending": [
                [self.pending[environment_index][seat].snapshot() for seat in range(NUM_SEATS)]
                for environment_index in range(self.environment_count)
            ],
            "last_seq_consumed": self.last_seq_consumed.tolist(),
            "chunk_position": self.chunk_position.tolist(),
            "force_new_chunk": self.force_new_chunk.tolist(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Put a snapshot back, refusing one that does not describe this run."""
        stored_policy_ids = tuple(snapshot["policy_ids"])
        if stored_policy_ids != self.policy_ids:
            raise ValueError(
                f"snapshot holds policies {stored_policy_ids} but this run has {self.policy_ids}"
            )
        for policy_id in self.policy_ids:
            stored_actor = snapshot["actor_hidden"][policy_id]
            if tuple(stored_actor.shape) != tuple(self.actor_hidden[policy_id].shape):
                raise ValueError(
                    f"snapshot actor state is {tuple(stored_actor.shape)} but this run needs "
                    f"{tuple(self.actor_hidden[policy_id].shape)}"
                )
            self.actor_hidden[policy_id] = stored_actor.clone()
            self.critic_hidden[policy_id] = snapshot["critic_hidden"][policy_id].clone()

        self.pending = [
            [PendingEventQueue.from_snapshot(rows) for rows in per_environment]
            for per_environment in snapshot["pending"]
        ]
        self.last_seq_consumed = np.array(snapshot["last_seq_consumed"], dtype=np.int64)
        self.chunk_position = np.array(snapshot["chunk_position"], dtype=np.int64)
        self.force_new_chunk = np.array(snapshot["force_new_chunk"], dtype=bool)


def recurrent_states_are_disjoint(store: RecurrentStateStore) -> bool:
    """Return True when no two hidden tensors in ``store`` share storage."""
    tensors = [*store.actor_hidden.values(), *store.critic_hidden.values()]
    storage_ids = [tensor.untyped_storage().data_ptr() for tensor in tensors]
    return len(set(storage_ids)) == len(storage_ids)
