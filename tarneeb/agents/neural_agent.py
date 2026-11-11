"""A trained feedforward policy, wrapped as a Phase 5 arena agent.

This is what lets the neural baseline be measured with exactly the same fair
machinery as the heuristics: duplicate deals, held-out seeds, paired score
differences and bootstrap intervals. No new arena is introduced.

Information boundary. The agent receives a
:class:`~tarneeb.env.public_view.PublicView`, reconstructs an engine state from
it (see :mod:`tarneeb.agents.state_adapter`), and builds **only** the 692-dim
actor observation and the padded public event history. It never calls
``static_critic``, never touches ``original_hands``, and holds no reference to a
critic network. Feeding it privileged information would raise a shape error
rather than train or play silently on leaked cards.

Two action modes:

- stochastic -- sample from the masked distribution, using this agent's own
  seeded torch generator;
- deterministic -- take the most probable legal action, which is what an
  evaluation should use when it wants the policy's considered opinion rather
  than a draw from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tarneeb.agents.base import Agent, checked_action
from tarneeb.agents.state_adapter import reconstruct_state_from_public_view
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.rules import RulesConfig
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicView
from tarneeb.models.checkpoint import load_checkpoint
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.nets import ActorNetwork


class NeuralAgent(Agent):
    """Plays with a trained :class:`~tarneeb.models.nets.ActorNetwork`."""

    def __init__(
        self,
        actor: ActorNetwork,
        name: str = "neural",
        deterministic: bool = True,
        seed: int = 0,
        rules: RulesConfig | None = None,
    ) -> None:
        self.actor = actor
        self.name = name
        self.deterministic = deterministic
        self.seed = seed
        self.rules = rules if rules is not None else RulesConfig()

        # Its own engine and observation builder, so the agent never shares
        # mutable state with the arena that is scoring it.
        self._engine = TarneebEngine(self.rules)
        self._observations = ObservationBuilder(self._engine)
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

        self.actor.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        name: str = "neural",
        deterministic: bool = True,
        seed: int = 0,
        rules: RulesConfig | None = None,
    ) -> NeuralAgent:
        """Load an actor from a checkpoint, refusing incompatible contracts.

        The critic is deliberately not loaded: a playing agent has no use for a
        value network, and never constructing one removes the possibility of
        wiring privileged information into play.
        """
        resolved_rules = rules if rules is not None else RulesConfig()
        loaded = load_checkpoint(path, rules=resolved_rules, load_critic=False)
        return cls(
            actor=loaded.actor,
            name=name,
            deterministic=deterministic,
            seed=seed,
            rules=resolved_rules,
        )

    def restart(self) -> None:
        """Rewind the sampling stream, so a repeated evaluation is reproducible."""
        self._generator.manual_seed(self.seed)

    # --- the agent contract -------------------------------------------------

    def act(self, view: PublicView) -> int:
        """Return one legal action id chosen by the policy."""
        observation, history_ids, history_masks = self.encode(view)
        legal_action_mask = torch.from_numpy(view.legal_action_mask).unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(observation, history_ids, history_masks)
            distribution = MaskedCategorical.from_logits(logits, legal_action_mask)
            if self.deterministic:
                selected_action = distribution.most_probable_action()
            else:
                selected_action = distribution.sample(self._generator)

        return checked_action(int(selected_action.item()), view.legal_action_mask, self.name)

    def encode(self, view: PublicView) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the batched ``(observation, history_ids, history_masks)`` tensors.

        Exposed separately so a test can assert that what reaches the network is
        exactly the frozen actor observation and nothing wider.
        """
        state = reconstruct_state_from_public_view(view, self.rules)
        observation = self._observations.static_actor(state, view.observer_seat)
        history_ids, history_masks = self._observations.padded_history(state, view.observer_seat)
        return (
            torch.from_numpy(observation).unsqueeze(0),
            torch.from_numpy(history_ids.astype(np.int64)).unsqueeze(0),
            torch.from_numpy(history_masks).unsqueeze(0),
        )
