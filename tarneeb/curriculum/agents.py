"""Phase 7 agent adapters, so the frozen arena can evaluate curriculum stages.

Two thin wrappers, neither of which decides what is legal and neither of which
duplicates a rule:

- :class:`ScriptedAuctionAgent` plays the predetermined auction during ``BID`` and
  ``DECLARE`` and hands every ``PLAY`` decision to the agent it wraps. All four
  seats are wrapped identically, so the auction is the same in both orientations
  of a duplicate-deal pair and the paired comparison measures card play alone.

- :class:`HistoryConditionedNeuralAgent` is the frozen
  :class:`~tarneeb.agents.neural_agent.NeuralAgent` with the A7 history transform
  attached, so a checkpoint is evaluated under exactly the condition it was
  trained under.

Both keep the frozen information boundary: a ``PublicView`` goes in, one legal
action id comes out, and legality is still verified by the frozen
``checked_action``.
"""

from __future__ import annotations

from pathlib import Path

from tarneeb.agents.base import Agent, checked_action
from tarneeb.agents.neural_agent import NeuralAgent
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.curriculum.auction_script import AuctionScript, scripted_auction_action
from tarneeb.curriculum.history import HistoryConditionedObservationBuilder
from tarneeb.curriculum.stages import FULL_HISTORY
from tarneeb.env.public_view import PublicView
from tarneeb.models.nets import ActorNetwork


class ScriptedAuctionAgent(Agent):
    """Plays a predetermined auction, then defers to the wrapped policy.

    The script is set per deal by the evaluation driver. Because every seat at the
    table is wrapped with the same script, the auction is a fixed property of the
    deal rather than of whoever is sitting there.
    """

    def __init__(self, inner_agent: Agent, name: str | None = None) -> None:
        self.inner_agent = inner_agent
        self.name = name if name is not None else inner_agent.name
        self.script: AuctionScript | None = None

    def set_script(self, script: AuctionScript) -> None:
        """Install the auction this deal must follow."""
        self.script = script

    def reset(self) -> None:
        self.inner_agent.reset()

    def act(self, view: PublicView) -> int:
        if view.phase in (Phase.BID, Phase.DECLARE):
            if self.script is None:
                raise ValueError(f"{self.name} was asked to bid with no auction script installed")
            selected_action = scripted_auction_action(
                script=self.script,
                phase=view.phase,
                acting_seat=view.observer_seat,
                current_high_bid=view.current_high_bid,
            )
            return checked_action(selected_action, view.legal_action_mask, self.name)
        return self.inner_agent.act(view)

    def __repr__(self) -> str:
        return f"ScriptedAuctionAgent(inner={self.inner_agent!r})"


class HistoryConditionedNeuralAgent(NeuralAgent):
    """A neural agent that sees the history under one A7 condition."""

    def __init__(
        self,
        actor: ActorNetwork,
        history_condition: str = FULL_HISTORY,
        name: str = "neural",
        deterministic: bool = True,
        seed: int = 0,
        rules: RulesConfig | None = None,
    ) -> None:
        super().__init__(
            actor=actor, name=name, deterministic=deterministic, seed=seed, rules=rules
        )
        self.history_condition = history_condition
        if history_condition != FULL_HISTORY:
            self._observations = HistoryConditionedObservationBuilder(
                self._engine, history_condition
            )

    @classmethod
    def from_checkpoint(  # type: ignore[override]
        cls,
        path: Path,
        history_condition: str = FULL_HISTORY,
        name: str = "neural",
        deterministic: bool = True,
        seed: int = 0,
        rules: RulesConfig | None = None,
    ) -> HistoryConditionedNeuralAgent:
        """Load an actor, refusing any incompatible contract, and condition it.

        The critic is deliberately not loaded, exactly as in the frozen loader.
        """
        from tarneeb.models.checkpoint import load_checkpoint

        resolved_rules = rules if rules is not None else RulesConfig()
        loaded = load_checkpoint(path, rules=resolved_rules, load_critic=False)
        return cls(
            actor=loaded.actor,
            history_condition=history_condition,
            name=name,
            deterministic=deterministic,
            seed=seed,
            rules=resolved_rules,
        )


def wrap_with_scripted_auction(agents: list[Agent]) -> list[ScriptedAuctionAgent]:
    """Wrap every agent so none of them has to decide the auction."""
    return [ScriptedAuctionAgent(agent) for agent in agents]
