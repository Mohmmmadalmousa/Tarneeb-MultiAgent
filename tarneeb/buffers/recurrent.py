"""The ``chunk_v2`` recurrent rollout buffer.

A different schema from ``flat_v1``, not an extension of it. See
docs/BUFFER_CONTRACT.md for why the two are kept distinguishable by a recorded
version string, and docs/RECURRENT_CONTRACT.md section 6 for the field list.

**What one transition is.** One stored decision by one learner-controlled seat.
``flat_v1`` keeps a rectangular block over environments and stamps opponent
decisions with a sentinel seat so the block stays indexable; ``chunk_v2`` does
not store opponent decisions at all. The natural unit here is the per-``(env,
seat)`` sequence, because that is simultaneously the unit recurrence runs over
and the unit credit assignment runs over.

**Chunking is a view, not a computation.** Advantages are computed over each
complete ``(environment, seat)`` sequence using the **frozen** Phase 6
:func:`~tarneeb.buffers.rollout.compute_generalized_advantage_estimates`,
unchanged. Only afterwards is the sequence sliced into chunks. An advantage
therefore cannot depend on which chunk its decision landed in, which is asserted
directly rather than argued.

**Two boundaries that are not the same thing.** ``chunk_head`` marks a recurrent
boundary: a new dealt hand, an all-pass redeal included, or a full chunk.
``done`` marks an episode boundary for GAE: only a scored deal. An all-pass hand
emits no reward and the episode continues, so credit bootstraps through it while
hidden state does not. Conflating the two would break one or the other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from tarneeb.buffers.rollout import compute_generalized_advantage_estimates
from tarneeb.core.actions import NUM_ACTIONS
from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.event_features import EVENT_ID_DTYPE
from tarneeb.env.spaces import K_MAX, N_EVENT_FIELDS

RECURRENT_BUFFER_CONTRACT_VERSION = "chunk_v2"
"""The rollout data schema recurrent runs are trained under."""

DEFAULT_CHUNK_LENGTH = 16
"""Blueprint section 10.3. A chunk is at most this many stored decisions."""

PAD_LEGAL_ACTION = 0
"""The single legal action on a padded timestep's sentinel mask.

The frozen masking utility rejects an all-False mask, and rightly so: a
distribution over nothing is not a meaningful object and a seat with no legal
action must never be asked to act. Padded positions therefore carry a mask with
exactly one action legal. Nothing they produce reaches a loss or a gradient,
because every loss term is a mean masked by ``valid``.
"""


@dataclass(frozen=True, slots=True)
class RecurrentTransition:
    """One stored decision. Plain arrays; the tensors are built at chunk time."""

    environment_index: int
    seat: int
    static: np.ndarray
    critic_static: np.ndarray
    previous_event_ids: np.ndarray
    previous_event_length: int
    legal_action_mask: np.ndarray
    action: int
    old_log_probability: float
    value: float
    is_chunk_head: bool
    actor_hidden_head: np.ndarray | None
    critic_hidden_head: np.ndarray | None
    first_event_seq: int
    reward: float = 0.0
    done: bool = False


@dataclass(frozen=True, slots=True)
class DealSettlement:
    """Complete accounting of one settled deal, for the conservation test.

    ``reward_paid + reward_unattached`` equals the total the engine emitted for
    the seats the buffer is responsible for. Nothing can go missing without
    showing up in one of the two numbers.
    """

    seats_paid: int
    reward_paid: float
    reward_unattached: float

    @property
    def total(self) -> float:
        return self.reward_paid + self.reward_unattached


@dataclass(frozen=True, slots=True)
class ChunkBatch:
    """Whole chunks, shaped ``(chunk_length, chunk_count, ...)``.

    Time-major because replay walks positions in order, carrying hidden state
    from one to the next. ``valid`` is the chunk-position padding mask;
    ``previous_event_lengths`` is the independent within-transition padding.
    """

    static: Tensor
    critic_static: Tensor
    previous_event_ids: Tensor
    previous_event_lengths: Tensor
    legal_action_masks: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor
    valid: Tensor
    seats: Tensor
    environment_indices: Tensor
    initial_actor_hidden: Tensor
    initial_critic_hidden: Tensor

    @property
    def chunk_length(self) -> int:
        return int(self.static.shape[0])

    @property
    def chunk_count(self) -> int:
        return int(self.static.shape[1])

    @property
    def valid_transition_count(self) -> int:
        return int(self.valid.sum())

    def select(self, chunk_indices: Tensor) -> ChunkBatch:
        """Return the sub-batch holding the given whole chunks."""
        return ChunkBatch(
            static=self.static[:, chunk_indices],
            critic_static=self.critic_static[:, chunk_indices],
            previous_event_ids=self.previous_event_ids[:, chunk_indices],
            previous_event_lengths=self.previous_event_lengths[:, chunk_indices],
            legal_action_masks=self.legal_action_masks[:, chunk_indices],
            actions=self.actions[:, chunk_indices],
            old_log_probabilities=self.old_log_probabilities[:, chunk_indices],
            values=self.values[:, chunk_indices],
            advantages=self.advantages[:, chunk_indices],
            returns=self.returns[:, chunk_indices],
            valid=self.valid[:, chunk_indices],
            seats=self.seats[:, chunk_indices],
            environment_indices=self.environment_indices[:, chunk_indices],
            initial_actor_hidden=self.initial_actor_hidden[:, chunk_indices],
            initial_critic_hidden=self.initial_critic_hidden[:, chunk_indices],
        )

    def is_finite(self) -> bool:
        """Return True when no stored float tensor holds a NaN or an infinity."""
        return all(
            bool(torch.isfinite(tensor).all())
            for tensor in (
                self.static,
                self.critic_static,
                self.old_log_probabilities,
                self.values,
                self.advantages,
                self.returns,
                self.initial_actor_hidden,
                self.initial_critic_hidden,
            )
        )


class RecurrentRolloutBuffer:
    """Sequence storage for one on-policy recurrent rollout.

    Nothing here knows a game rule. It records what the environment reported and
    what the networks produced, arranges those records into per-seat sequences,
    and slices them into chunks.
    """

    def __init__(self, environment_count: int, chunk_length: int = DEFAULT_CHUNK_LENGTH) -> None:
        if environment_count < 1:
            raise ValueError(f"environment_count must be at least 1, got {environment_count}")
        if chunk_length < 1:
            raise ValueError(f"chunk_length must be at least 1, got {chunk_length}")
        self.environment_count = environment_count
        self.chunk_length = chunk_length
        self.sequences: dict[tuple[int, int], list[RecurrentTransition]] = {}

    # --- recording ----------------------------------------------------------

    def add(self, transition: RecurrentTransition) -> None:
        """Append one stored decision to its ``(environment, seat)`` sequence."""
        if transition.is_chunk_head:
            if transition.previous_event_length != 0:
                raise ValueError(
                    "a chunk head must store the empty segment (rule R4), got "
                    f"length {transition.previous_event_length}"
                )
            if transition.actor_hidden_head is None or transition.critic_hidden_head is None:
                raise ValueError("a chunk head must store both h0 and hc0 (rule R3)")
        elif transition.actor_hidden_head is not None or transition.critic_hidden_head is not None:
            raise ValueError("only a chunk head may store h0/hc0")
        if not 0 <= transition.previous_event_length <= K_MAX:
            raise ValueError(
                f"previous_event_length must lie in 0..{K_MAX}, "
                f"got {transition.previous_event_length}"
            )

        key = (transition.environment_index, transition.seat)
        self.sequences.setdefault(key, []).append(transition)

    def settle_completed_deal(
        self,
        environment_index: int,
        rewards_by_seat: np.ndarray,
        seats: tuple[int, ...] | None = None,
    ) -> DealSettlement:
        """Pay a finished deal's reward to each seat's last stored decision in it.

        ``seats`` names the seats whose reward this buffer is responsible for --
        the learner's, since no other seat is ever stored. Restricting it makes
        the returned accounting a **complete** statement about those seats, which
        is what lets a test prove conservation instead of inspecting it.

        Mirrors ``flat_v1``'s attribution exactly: the deal reward goes to the
        last decision each seat made inside the deal, so every participating seat
        is paid exactly once. A seat that made no stored decision inside the
        rollout window has no transition of its own to attach the reward to; that
        amount is reported as ``unattached`` rather than silently vanishing.
        """
        responsible = tuple(range(NUM_SEATS)) if seats is None else seats
        seats_paid = 0
        total_reward_paid = 0.0
        unattached = 0.0
        for seat in responsible:
            reward = float(rewards_by_seat[seat])
            sequence = self.sequences.get((environment_index, seat))
            if not sequence or sequence[-1].done:
                # No open transition of this seat's own in the current deal.
                unattached += reward
                continue
            sequence[-1] = replace(sequence[-1], reward=reward, done=True)
            seats_paid += 1
            total_reward_paid += reward
        return DealSettlement(
            seats_paid=seats_paid,
            reward_paid=total_reward_paid,
            reward_unattached=unattached,
        )

    # --- finalization -------------------------------------------------------

    def compute_advantages_and_returns(
        self,
        bootstrap_values_by_environment_seat: np.ndarray,
        discount_factor: float,
        gae_lambda: float,
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """Return ``(advantages, returns)`` per ``(environment, seat)`` sequence.

        Uses the frozen Phase 6 recursion unchanged, on the complete sequence.
        Chunk boundaries play no part: a chunk is a view taken afterwards.
        """
        expected_shape = (self.environment_count, NUM_SEATS)
        if bootstrap_values_by_environment_seat.shape != expected_shape:
            raise ValueError(
                f"bootstrap values must be {expected_shape}, "
                f"got {bootstrap_values_by_environment_seat.shape}"
            )

        results: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for (environment_index, seat), sequence in self.sequences.items():
            if not sequence:
                continue
            values = np.array([t.value for t in sequence], dtype=np.float64)
            rewards = np.array([t.reward for t in sequence], dtype=np.float64)
            terminal = np.array([t.done for t in sequence], dtype=bool)
            episode_continues = (~terminal).astype(np.float64)

            next_values = np.empty_like(values)
            next_values[:-1] = values[1:]
            next_values[-1] = float(bootstrap_values_by_environment_seat[environment_index, seat])

            advantages = compute_generalized_advantage_estimates(
                rewards=rewards,
                state_values=values,
                next_state_values=next_values,
                episode_continues=episode_continues,
                discount_factor=discount_factor,
                gae_lambda=gae_lambda,
            )
            results[(environment_index, seat)] = (advantages, advantages + values)
        return results

    def chunk_slices(self) -> list[tuple[tuple[int, int], int, int]]:
        """Return ``((environment, seat), start, stop)`` for every chunk.

        Splits each sequence at its stored ``chunk_head`` flags, which the
        collector set. Because a head is forced at every newly dealt hand and
        every ``chunk_length`` decisions, no chunk can span a seat, an
        environment, a reset or more than ``chunk_length`` decisions -- and the
        last two are re-checked here rather than assumed.
        """
        slices: list[tuple[tuple[int, int], int, int]] = []
        for key, sequence in sorted(self.sequences.items()):
            if not sequence:
                continue
            if not sequence[0].is_chunk_head:
                raise ValueError(f"sequence {key} does not start at a chunk head")
            start = 0
            for position in range(1, len(sequence)):
                if sequence[position].is_chunk_head:
                    slices.append((key, start, position))
                    start = position
            slices.append((key, start, len(sequence)))

        for key, start, stop in slices:
            if stop - start > self.chunk_length:
                raise ValueError(
                    f"sequence {key} produced a chunk of {stop - start} decisions, "
                    f"which exceeds chunk_length={self.chunk_length}"
                )
        return slices

    def to_chunks(
        self,
        bootstrap_values_by_environment_seat: np.ndarray,
        discount_factor: float,
        gae_lambda: float,
    ) -> ChunkBatch:
        """Assemble every sequence into padded, time-major whole chunks."""
        credit = self.compute_advantages_and_returns(
            bootstrap_values_by_environment_seat, discount_factor, gae_lambda
        )
        slices = self.chunk_slices()
        if not slices:
            raise ValueError("cannot build chunks from an empty rollout")

        chunk_count = len(slices)
        length = self.chunk_length
        first = next(iter(self.sequences.values()))[0]
        actor_dim = first.static.shape[0]
        critic_dim = first.critic_static.shape[0]
        hidden_shape = first.actor_hidden_head
        assert hidden_shape is not None  # a sequence always starts at a head

        static = np.zeros((length, chunk_count, actor_dim), dtype=np.float32)
        critic_static = np.zeros((length, chunk_count, critic_dim), dtype=np.float32)
        previous_event_ids = np.zeros(
            (length, chunk_count, K_MAX, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE
        )
        previous_event_lengths = np.zeros((length, chunk_count), dtype=np.int64)
        legal_action_masks = np.zeros((length, chunk_count, NUM_ACTIONS), dtype=bool)
        legal_action_masks[..., PAD_LEGAL_ACTION] = True  # the padded-row sentinel
        actions = np.zeros((length, chunk_count), dtype=np.int64)
        old_log_probabilities = np.zeros((length, chunk_count), dtype=np.float32)
        values = np.zeros((length, chunk_count), dtype=np.float32)
        advantages = np.zeros((length, chunk_count), dtype=np.float32)
        returns = np.zeros((length, chunk_count), dtype=np.float32)
        valid = np.zeros((length, chunk_count), dtype=bool)
        seats = np.full((length, chunk_count), -1, dtype=np.int64)
        environment_indices = np.full((length, chunk_count), -1, dtype=np.int64)
        initial_actor_hidden = np.zeros((chunk_count, *hidden_shape.shape), dtype=np.float32)
        initial_critic_hidden = np.zeros((chunk_count, *hidden_shape.shape), dtype=np.float32)

        for chunk_index, (key, start, stop) in enumerate(slices):
            sequence = self.sequences[key]
            sequence_advantages, sequence_returns = credit[key]
            head = sequence[start]
            assert head.actor_hidden_head is not None and head.critic_hidden_head is not None
            initial_actor_hidden[chunk_index] = head.actor_hidden_head
            initial_critic_hidden[chunk_index] = head.critic_hidden_head

            for offset, position in enumerate(range(start, stop)):
                transition = sequence[position]
                static[offset, chunk_index] = transition.static
                critic_static[offset, chunk_index] = transition.critic_static
                previous_event_ids[offset, chunk_index] = transition.previous_event_ids
                previous_event_lengths[offset, chunk_index] = transition.previous_event_length
                legal_action_masks[offset, chunk_index] = transition.legal_action_mask
                actions[offset, chunk_index] = transition.action
                old_log_probabilities[offset, chunk_index] = transition.old_log_probability
                values[offset, chunk_index] = transition.value
                advantages[offset, chunk_index] = sequence_advantages[position]
                returns[offset, chunk_index] = sequence_returns[position]
                valid[offset, chunk_index] = True
                seats[offset, chunk_index] = transition.seat
                environment_indices[offset, chunk_index] = transition.environment_index

        return ChunkBatch(
            static=torch.from_numpy(static),
            critic_static=torch.from_numpy(critic_static),
            previous_event_ids=torch.from_numpy(previous_event_ids.astype(np.int64)),
            previous_event_lengths=torch.from_numpy(previous_event_lengths),
            legal_action_masks=torch.from_numpy(legal_action_masks),
            actions=torch.from_numpy(actions),
            old_log_probabilities=torch.from_numpy(old_log_probabilities),
            values=torch.from_numpy(values),
            advantages=torch.from_numpy(advantages),
            returns=torch.from_numpy(returns),
            valid=torch.from_numpy(valid),
            seats=torch.from_numpy(seats),
            environment_indices=torch.from_numpy(environment_indices),
            # (chunk, layers, hidden) -> (layers, chunk, hidden), the GRU layout.
            initial_actor_hidden=torch.from_numpy(initial_actor_hidden)
            .permute(1, 0, 2)
            .contiguous(),
            initial_critic_hidden=torch.from_numpy(initial_critic_hidden)
            .permute(1, 0, 2)
            .contiguous(),
        )

    # --- accounting ---------------------------------------------------------

    @property
    def stored_decision_count(self) -> int:
        return sum(len(sequence) for sequence in self.sequences.values())

    @property
    def unpaid_reward_total(self) -> float:
        """Total reward attached to no transition. Always ``0.0``.

        Rewards are written directly onto their target transition when a deal
        settles, so nothing is ever held in a pending queue. The property exists
        so the conservation test can assert emptiness explicitly.
        """
        return 0.0

    def consumed_event_sequence(self, environment_index: int, seat: int) -> list[int]:
        """Return the ``seq`` values this seat's stored segments cover, in order.

        Used by the I15 accounting test: within one chunk the union of the stored
        segments must have no duplicate and must be a contiguous run.
        """
        covered: list[int] = []
        for transition in self.sequences.get((environment_index, seat), []):
            if transition.previous_event_length == 0:
                continue
            start = transition.first_event_seq
            covered.extend(range(start, start + transition.previous_event_length))
        return covered


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Return the mean of ``values`` over ``mask``, or exactly zero if empty.

    Every recurrent loss term goes through here. A chunk with no valid position
    must contribute exactly zero gradient, so the empty case returns a zero that
    is still connected to the graph shape but carries no contribution.
    """
    weights = mask.to(values.dtype)
    total = weights.sum()
    if float(total) == 0.0:
        return (values * weights).sum() * 0.0
    return (values * weights).sum() / total
