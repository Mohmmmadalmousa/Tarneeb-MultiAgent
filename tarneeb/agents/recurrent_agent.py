"""A trained Event-GRU policy, wrapped as a Phase 5 arena agent (Phase 9).

The recurrent counterpart of the frozen :class:`~tarneeb.agents.neural_agent.
NeuralAgent`. This is what lets a recurrent policy be measured by exactly the
same fair machinery as every other policy in this project -- duplicate deals,
held-out seeds, paired score differences, bootstrap intervals. No new arena, no
special-casing, no separate protocol that might quietly flatter it.

**Where the memory lives.** The arena hands an agent a
:class:`~tarneeb.env.public_view.PublicView` only when it is that seat's turn,
and a single agent instance holds **two** seats of a partnership. So the agent
keeps a hidden state and an event cursor **per seat**, exactly as the recurrent
contract requires, and never one shared between them.

**Catching up, and why it is equivalent.** During training, rule R1 advances
every seat's memory the moment events are emitted, and rule R2 then has the
acting seat consume nothing. Here the agent only wakes when it is on turn, so it
consumes the events that arrived since its last decision and advances through
them in one call.

That is not a different convention -- it is the same one, evaluated lazily.
``GRU(h, [e1..ek])`` is identical whether the events are folded in one at a time
as they happen or all at once when the seat next acts; the frozen
incremental-versus-batch equivalence test (blueprint T-E5) is precisely the proof
of that. So this agent reaches the same state a training rollout would give it,
which is what makes an evaluation of a trained checkpoint meaningful.

**Information boundary.** The agent reconstructs an engine state from the public
view, builds **only** the 692-dim actor observation and the relativized event
ids, and holds no reference to a critic. It never calls ``static_critic`` and
never touches another seat's hand. Handing it privileged information would raise
a shape error rather than play on silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tarneeb.agents.base import Agent, checked_action
from tarneeb.agents.state_adapter import reconstruct_state_from_public_view
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.rules import RulesConfig
from tarneeb.env.event_features import event_ids
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicView
from tarneeb.env.spaces import K_MAX, N_EVENT_FIELDS
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.recurrent import RecurrentActorNetwork, zero_hidden_state
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint


class RecurrentNeuralAgent(Agent):
    """Plays with a trained :class:`RecurrentActorNetwork`, one memory per seat."""

    def __init__(
        self,
        actor: RecurrentActorNetwork,
        name: str = "recurrent",
        deterministic: bool = True,
        seed: int = 0,
        rules: RulesConfig | None = None,
    ) -> None:
        self.actor = actor
        self.name = name
        self.deterministic = deterministic
        self.seed = seed
        self.rules = rules if rules is not None else RulesConfig()

        # Its own engine and builder, so the agent never shares mutable state
        # with the arena that is scoring it.
        self._engine = TarneebEngine(self.rules)
        self._observations = ObservationBuilder(self._engine)
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

        self._hidden: dict[int, torch.Tensor] = {}
        self._cursor: dict[int, int] = {}
        self.reset()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        name: str = "recurrent",
        deterministic: bool = True,
        seed: int = 0,
        rules: RulesConfig | None = None,
    ) -> RecurrentNeuralAgent:
        """Load an actor for inference. The critic is never loaded."""
        loaded = load_recurrent_checkpoint(path, rules=rules, load_critic=False)
        return cls(
            actor=loaded.actor,
            name=name,
            deterministic=deterministic,
            seed=seed,
            rules=rules,
        )

    # --- per-deal lifecycle -------------------------------------------------

    def reset(self) -> None:
        """Zero every seat's memory and cursor. The arena calls this per deal.

        This is the deployment form of the recurrent contract's deal boundary:
        the engine restarts its event stream at each newly dealt hand, so memory
        of the previous one describes a history that no longer exists.
        """
        self._hidden = {seat: zero_hidden_state(1, self.actor.spec) for seat in range(NUM_SEATS)}
        self._cursor = dict.fromkeys(range(NUM_SEATS), 0)

    def restart(self) -> None:
        """Rewind the sampling stream, so a repeated evaluation is reproducible."""
        self._generator.manual_seed(self.seed)
        self.reset()

    # --- the agent contract -------------------------------------------------

    def act(self, view: PublicView) -> int:
        """Return one legal action id chosen by the recurrent policy."""
        seat = view.observer_seat
        observation, segment_ids, segment_length = self.encode(view)
        legal_action_mask = torch.from_numpy(view.legal_action_mask).unsqueeze(0)

        with torch.no_grad():
            logits, advanced = self.actor(
                observation, segment_ids, segment_length, self._hidden[seat]
            )
            distribution = MaskedCategorical.from_logits(logits, legal_action_mask)
            selected_action = (
                distribution.most_probable_action()
                if self.deterministic
                else distribution.sample(self._generator)
            )

        self._hidden[seat] = advanced.detach()
        self._cursor[seat] = len(view.events)
        return checked_action(int(selected_action.item()), view.legal_action_mask, self.name)

    def encode(self, view: PublicView) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(observation, segment_ids, segment_length)`` for this decision.

        Exposed separately so a test can assert that what reaches the network is
        exactly the frozen actor observation and the public event ids, and
        nothing wider.
        """
        seat = view.observer_seat
        state = reconstruct_state_from_public_view(view, self.rules)
        observation = self._observations.static_actor(state, seat)

        start = self._cursor[seat]
        pending = view.events[start:]
        if len(pending) > K_MAX:
            raise AssertionError(
                f"seat {seat} faced a {len(pending)}-event segment, which exceeds "
                f"K_MAX={K_MAX}; the proved bound is 14"
            )
        segment = np.zeros((1, K_MAX, N_EVENT_FIELDS), dtype=np.int64)
        for position, event in enumerate(pending):
            segment[0, position] = event_ids(event, seat)

        return (
            torch.from_numpy(observation).unsqueeze(0),
            torch.from_numpy(segment),
            torch.tensor([len(pending)], dtype=torch.int64),
        )

    # --- introspection, for tests and diagnostics ---------------------------

    def hidden_state(self, seat: int) -> torch.Tensor:
        """Return one seat's current memory. Read-only by convention."""
        return self._hidden[seat]

    def consumed_event_count(self, seat: int) -> int:
        """Return how many of the deal's events this seat has folded in."""
        return self._cursor[seat]

    def states_are_separate(self) -> bool:
        """Return True when no two seats share hidden-state storage."""
        pointers = [tensor.data_ptr() for tensor in self._hidden.values()]
        return len(set(pointers)) == len(pointers)
