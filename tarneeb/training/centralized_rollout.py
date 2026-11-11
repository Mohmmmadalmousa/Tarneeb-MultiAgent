"""Rollout collection for the centralized critic (Phase 10 / ablation A2).

The control condition stores the 692-dim actor-safe observation as the critic's
input. The treatment condition stores the 856-dim environment observation
instead. **Nothing else differs**: the actor still receives 692 features, still
consumes the public event stream, still carries its own recurrent state, and the
G6 complete-deal update boundary is inherited unchanged from Phase 9.

Why this is a subclass and not a flag inside Phase 9
---------------------------------------------------
Phase 9 is frozen. The critic-observation choice therefore enters as an override
on a Phase 10 class rather than as a branch added to frozen code. The method it
overrides is duplicated rather than shared, which is the same trade Phase 9 made
for ``_settle_and_reset``: duplication is acceptable precisely because a test
compares the two paths and would catch any drift.

The one thing that must not go wrong
------------------------------------
It would be very easy to write ``critic_static=observation`` and silently feed
the actor observation to a critic that was built expecting 856 -- or worse, to
feed the 856 vector to the actor. Both are guarded:

- the actor is handed ``observations`` and would raise a shape error on 856,
  because it owns no parameter of that width;
- the critic is handed ``critic_observations`` and would raise on 692 when built
  centralized;
- ``assert_actor_input_is_actor_safe`` re-checks the widths on every stored
  transition in tests.

So a wiring mistake fails loudly rather than producing a policy that has quietly
seen the opponents' cards.
"""

from __future__ import annotations

import numpy as np
import torch

from tarneeb.buffers.recurrent import RecurrentRolloutBuffer
from tarneeb.core.actions import NO_TRUMP_ACTION, PASS_ACTION
from tarneeb.core.cards import NUM_SEATS
from tarneeb.env.spaces import K_MAX, N_EVENT_FIELDS
from tarneeb.models.masking import MaskedCategorical, actions_are_legal
from tarneeb.training.complete_deal_rollout import CompleteDealRecurrentCollector
from tarneeb.training.recurrent_rollout import RecurrentRolloutMetrics


class CentralizedCriticCollector(CompleteDealRecurrentCollector):
    """Complete-deal recurrent collection with a privileged critic input.

    Inherits the Phase 9 G6-safe boundary, the Part A cross-environment batching,
    R1/R2 event consumption and every isolation property. The single change is
    which observation the critic reads and which one is stored for PPO replay.
    """

    def critic_observation(self, environment_index: int, seat: int) -> np.ndarray:
        """Return the critic's input for one decision.

        Overridden by the control condition to return the actor-safe vector, so
        the two A2 conditions differ in exactly this one method.
        """
        environment = self.environments.environments[environment_index]
        return environment.critic_observation(seat)

    def _act_and_store_batched(
        self,
        environment_indices: np.ndarray,
        seats: np.ndarray,
        step_actions: np.ndarray,
        buffer: RecurrentRolloutBuffer,
        metrics: RecurrentRolloutMetrics,
    ) -> None:
        """One batched forward per step, with separate actor and critic inputs.

        A deliberate mirror of the Phase 9 method, with two arrays instead of
        one: ``observations`` (692, actor) and ``critic_observations``
        (692 or 856, critic). The actor never sees the second array.
        """
        row_count = environment_indices.size
        actor_dim = self.actor.spec.actor_observation_dim
        critic_dim = self.critic.spec.critic_observation_dim

        observations = np.zeros((row_count, actor_dim), dtype=np.float32)
        critic_observations = np.zeros((row_count, critic_dim), dtype=np.float32)
        legal_masks = np.zeros((row_count, self.actor.spec.action_count), dtype=bool)
        for row, environment_index in enumerate(environment_indices):
            environment = self.environments.environments[int(environment_index)]
            seat = int(seats[row])
            observations[row] = environment.actor_observation(seat)
            critic_observations[row] = self.critic_observation(int(environment_index), seat)
            legal_masks[row] = environment.legal_action_mask()

        observation_tensor = torch.from_numpy(observations)
        critic_tensor = torch.from_numpy(critic_observations)
        legal_mask_tensor = torch.from_numpy(legal_masks)
        # R2: nothing is consumed at a decision point, so every row is empty.
        empty_ids = torch.zeros((row_count, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
        empty_lengths = torch.zeros(row_count, dtype=torch.int64)

        actor_hidden = self.state.gather_actor(environment_indices, seats, self.policy_id)
        critic_hidden = self.state.gather_critic(environment_indices, seats, self.policy_id)

        with torch.no_grad():
            logits, advanced_actor = self.actor(
                observation_tensor, empty_ids, empty_lengths, actor_hidden
            )
            values, advanced_critic = self.critic(
                critic_tensor, empty_ids, empty_lengths, critic_hidden
            )

        # R2 is a correctness property, so it is verified rather than trusted.
        if not bool(torch.equal(advanced_actor, actor_hidden)):
            raise AssertionError(
                "the actor forward advanced the hidden state at a decision point; "
                "rule R2 requires prev_ev_len == 0 to be a strict no-op"
            )
        if not bool(torch.equal(advanced_critic, critic_hidden)):
            raise AssertionError("the critic forward advanced the hidden state at a decision point")

        # Sampling stays one row at a time, in environment order, so the
        # generator is consumed exactly as every earlier collector consumes it.
        for row in range(row_count):
            environment_index = int(environment_indices[row])
            seat = int(seats[row])
            row_mask = legal_mask_tensor[row : row + 1]
            distribution = MaskedCategorical.from_logits(logits[row : row + 1], row_mask)
            with torch.no_grad():
                selected = distribution.sample(self.action_sampling_generator)
                log_probability = distribution.log_probability_of(selected)

            if not actions_are_legal(selected, row_mask):
                metrics.illegal_action_count += 1
                raise AssertionError("the policy sampled an action its legal mask forbids")
            action = int(selected.item())
            if action == NO_TRUMP_ACTION:
                metrics.no_trump_action_count += 1
                raise AssertionError("the policy selected the reserved no-trump action 64")

            if bool(legal_masks[row, PASS_ACTION]):
                metrics.auction_decision_count += 1
                if action == PASS_ACTION:
                    metrics.pass_action_count += 1

            self._store_decision(
                environment_index=environment_index,
                seat=seat,
                observation=observations[row],
                legal_action_mask=legal_masks[row],
                action=action,
                log_probability=float(log_probability.item()),
                value=float(values[row].item()),
                actor_slot=actor_hidden[:, row : row + 1],
                critic_slot=critic_hidden[:, row : row + 1],
                buffer=buffer,
                metrics=metrics,
                critic_observation=critic_observations[row],
            )
            step_actions[environment_index] = action

    def _store_decision(
        self,
        environment_index: int,
        seat: int,
        observation: np.ndarray,
        legal_action_mask: np.ndarray,
        action: int,
        log_probability: float,
        value: float,
        actor_slot: torch.Tensor,
        critic_slot: torch.Tensor,
        buffer: RecurrentRolloutBuffer,
        metrics: RecurrentRolloutMetrics,
        critic_observation: np.ndarray | None = None,
    ) -> None:
        """Store one decision, keeping the actor and critic inputs separate.

        ``critic_observation`` defaults to the actor observation so that any
        inherited call path -- there are none today, but a future subclass might
        add one -- degrades to the decentralized behaviour rather than silently
        storing an array of the wrong width.
        """
        if not bool(self._active[environment_index]):
            return

        from tarneeb.buffers.recurrent import RecurrentTransition

        critic_input = observation if critic_observation is None else critic_observation
        is_head = self.state.is_chunk_head(environment_index, seat)
        segment_ids, segment_length = self.state.stored_segment(environment_index, seat, is_head)
        metrics.max_segment_length = max(metrics.max_segment_length, segment_length)

        last_consumed = int(self.state.last_seq_consumed[environment_index, seat])
        first_event_seq = last_consumed - segment_length + 1 if segment_length else -1

        buffer.add(
            RecurrentTransition(
                environment_index=environment_index,
                seat=seat,
                static=observation.copy(),
                critic_static=critic_input.copy(),
                previous_event_ids=segment_ids,
                previous_event_length=segment_length,
                legal_action_mask=legal_action_mask.copy(),
                action=action,
                old_log_probability=log_probability,
                value=value,
                is_chunk_head=is_head,
                actor_hidden_head=(
                    actor_slot.squeeze(1).detach().numpy().copy() if is_head else None
                ),
                critic_hidden_head=(
                    critic_slot.squeeze(1).detach().numpy().copy() if is_head else None
                ),
                first_event_seq=first_event_seq,
            )
        )
        self.state.note_decision_stored(environment_index, seat, self.chunk_length)
        metrics.learner_decision_count += 1

    def bootstrap_values(self) -> np.ndarray:
        """Return per-seat bootstrap values using the critic's own observation.

        Never actually consumed under complete-deal collection, because every
        stored sequence ends at a scored deal. It is kept correct anyway: a
        value that is silently wrong but unused is a trap for the next phase.
        """
        environment_count = self.environments.num_envs
        critic_dim = self.critic.spec.critic_observation_dim
        rows = environment_count * NUM_SEATS

        observations = np.zeros((rows, critic_dim), dtype=np.float32)
        environment_indices = np.zeros(rows, dtype=np.int64)
        seats = np.zeros(rows, dtype=np.int64)
        for environment_index in range(environment_count):
            for seat in range(NUM_SEATS):
                row = environment_index * NUM_SEATS + seat
                observations[row] = self.critic_observation(environment_index, seat)
                environment_indices[row] = environment_index
                seats[row] = seat

        empty_ids = torch.zeros((rows, K_MAX, N_EVENT_FIELDS), dtype=torch.int64)
        empty_lengths = torch.zeros(rows, dtype=torch.int64)
        with torch.no_grad():
            hidden = self.state.gather_critic(environment_indices, seats, self.policy_id)
            predicted, _advanced = self.critic(
                torch.from_numpy(observations), empty_ids, empty_lengths, hidden
            )
        values: np.ndarray = predicted.numpy().reshape(environment_count, NUM_SEATS)
        return values


class DecentralizedCriticCollector(CentralizedCriticCollector):
    """The A2 control: identical in every respect except the critic's input.

    Expressing the control as a subclass that overrides one method is what makes
    the ablation defensible. Both conditions run the same collection code, the
    same sampling order, the same storage and the same G6 boundary, so the only
    thing that can differ in the measurement is what the critic was allowed to
    see.
    """

    def critic_observation(self, environment_index: int, seat: int) -> np.ndarray:
        """Return the 692-dim actor-safe observation, as in Phases 6 to 9."""
        environment = self.environments.environments[environment_index]
        return environment.actor_observation(seat)


def assert_actor_input_is_actor_safe(
    buffer: RecurrentRolloutBuffer, actor_dim: int, critic_dim: int
) -> bool:
    """Return True when every stored transition kept the two inputs separate.

    Checks the widths rather than the values: a 692 actor row and an 856 critic
    row cannot be confused by shape, and shape is what a wiring mistake would
    get wrong.
    """
    for sequence in buffer.sequences.values():
        for transition in sequence:
            if transition.static.shape != (actor_dim,):
                return False
            if transition.critic_static.shape != (critic_dim,):
                return False
    return True
