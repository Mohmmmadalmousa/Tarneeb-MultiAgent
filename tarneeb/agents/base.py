"""Common interface for non-neural agents.

An agent chooses one action id. It never decides what is legal, never touches
the engine, and never sees a hand but its own: everything it may use arrives in
a :class:`~tarneeb.env.public_view.PublicView`, which is actor-safe by
construction and proved so by the leakage tests.

Illegal actions are prevented by masking. An agent that proposes an illegal
action is a bug, so :func:`checked_action` raises rather than silently
substituting a legal one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tarneeb.core.actions import NO_TRUMP_ACTION, NUM_ACTIONS
from tarneeb.env.public_view import PublicView


class Agent(ABC):
    """A policy that maps an actor-safe view to one legal action id."""

    name: str = "agent"

    @abstractmethod
    def act(self, view: PublicView) -> int:
        """Return one action id in ``0..64`` that is legal in ``view``."""

    def reset(self) -> None:
        """Clear any per-deal state. Stateless agents need not override this."""
        return

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def legal_action_ids(legal_action_mask: np.ndarray) -> np.ndarray:
    """Return the ids that ``legal_action_mask`` allows, validating it first."""
    if legal_action_mask.shape != (NUM_ACTIONS,):
        raise ValueError(f"legal mask must be ({NUM_ACTIONS},), got {legal_action_mask.shape}")
    if legal_action_mask.dtype != np.bool_:
        raise ValueError(f"legal mask must be bool, got {legal_action_mask.dtype}")
    if not legal_action_mask.any():
        raise ValueError("no legal action is available; the agent must not be asked to act")
    if legal_action_mask[NO_TRUMP_ACTION]:
        raise ValueError("action 64 is reserved and must never be legal")
    return np.flatnonzero(legal_action_mask)


def checked_action(selected_action: int, legal_action_mask: np.ndarray, agent_name: str) -> int:
    """Return ``selected_action`` after proving it is legal.

    Raises rather than repairing: a silent substitution would hide the bug and
    quietly corrupt an evaluation.
    """
    if not 0 <= selected_action < NUM_ACTIONS:
        raise ValueError(f"{agent_name} returned action {selected_action}, outside 0..64")
    if selected_action == NO_TRUMP_ACTION:
        raise ValueError(f"{agent_name} returned the reserved no-trump action 64")
    if not legal_action_mask[selected_action]:
        raise ValueError(f"{agent_name} returned illegal action {selected_action}")
    return int(selected_action)


def first_legal_action(legal_action_mask: np.ndarray) -> int:
    """Return the lowest legal action id.

    A deterministic last-resort fallback for heuristics whose preference set is
    empty. It is never a way to bypass legality, only a stable tie-break.
    """
    return int(legal_action_ids(legal_action_mask)[0])
