"""Uniform random legal agent -- the evaluation floor and the fuzz driver."""

from __future__ import annotations

import numpy as np

from tarneeb.agents.base import Agent, checked_action, legal_action_ids
from tarneeb.env.public_view import PublicView


class RandomAgent(Agent):
    """Samples uniformly from the legal-action mask.

    Draws come from an explicit seeded ``numpy.random.Generator``, never the
    global NumPy or Python RNG, so a run is reproducible from its seed and from
    the sequence of masks it is shown.

    By default the stream CONTINUES across deals, which makes the agent's
    behaviour independent from one deal to the next. Re-seeding at every deal
    instead would make it replay the same draws in every deal, correlating its
    choices and injecting a small systematic effect into paired evaluation --
    measured at roughly 0.4 points per deal before this default was changed.
    Pass ``reseed_on_reset=True`` when you deliberately want the repeating form.
    """

    def __init__(self, seed: int = 0, name: str = "random", reseed_on_reset: bool = False) -> None:
        self.seed = seed
        self.name = name
        self.reseed_on_reset = reseed_on_reset
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Start a new deal.

        Continues the existing stream unless ``reseed_on_reset`` was requested.
        """
        if self.reseed_on_reset:
            self.restart()

    def restart(self) -> None:
        """Rewind to the beginning of the seed's stream."""
        self.rng = np.random.default_rng(self.seed)

    def act(self, view: PublicView) -> int:
        candidate_actions = legal_action_ids(view.legal_action_mask)
        selected_action = int(self.rng.choice(candidate_actions))
        return checked_action(selected_action, view.legal_action_mask, self.name)
