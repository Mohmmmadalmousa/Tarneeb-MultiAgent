"""Deal state for one hand of Tarneeb.

Covers the full deal cycle ``DEAL -> BID -> DECLARE -> PLAY -> SCORE`` and the
public event stream recorded along the way.

``Phase`` itself is defined in :mod:`tarneeb.core.events` and re-exported here,
because ``PublicEvent`` carries a phase and this module holds a list of
``PublicEvent``.

**Hidden information.** ``hands`` and ``original_hands`` are private: they are
the only fields in the project that a seat may not see. They must never reach a
``PublicEvent`` or an actor observation. A training-only centralized critic may
read them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tarneeb.core.cards import CARDS_PER_HAND, NUM_SUITS
from tarneeb.core.events import NUM_SEATS, DealResult, Phase, PublicEvent
from tarneeb.core.rules import RulesConfig

__all__ = [
    "NUM_SEATS",
    "TRICKS_PER_DEAL",
    "Phase",
    "TarneebState",
    "first_bidder_seat",
    "next_seat",
    "team_of_seat",
]

TRICKS_PER_DEAL = CARDS_PER_HAND

BIDDING_TEAM = 0
"""Index into ``tricks_won`` / ``match_score`` for the team that won the auction."""

DEFENDING_TEAM = 1


def team_of_seat(seat: int) -> int:
    """Return the team of ``seat``: seats (0, 2) are team 0, seats (1, 3) are team 1 (C01)."""
    return seat % 2


def next_seat(seat: int) -> int:
    """Return the seat that acts after ``seat``.

    Bidding and play run counter-clockwise, which this project encodes as
    ascending seat index (C04).
    """
    return (seat + 1) % NUM_SEATS


def first_bidder_seat(dealer_seat: int) -> int:
    """Return the seat to the dealer's right, who opens the auction (C06)."""
    return next_seat(dealer_seat)


@dataclass
class TarneebState:
    """Mutable state of one deal.

    Everything except ``hands`` and ``original_hands`` is public. Those two are
    hidden and must never leak into an event or an actor observation.
    """

    rules: RulesConfig
    dealer_seat: int
    phase: Phase
    current_player_seat: int | None
    """The seat to act, or ``None`` when no seat is on turn (DEAL, SCORE, DONE)."""

    active_bidders: set[int] = field(default_factory=set)
    """Seats that have not yet passed. Passing is permanent (C09)."""

    current_high_bid: int | None = None
    current_high_bidder: int | None = None
    declarer_seat: int | None = None
    selected_trump_suit: int | None = None

    # --- hidden state -------------------------------------------------
    hands: list[int] = field(default_factory=lambda: [0] * NUM_SEATS)
    """HIDDEN. Card bitmask still held by each seat."""

    original_hands: list[int] = field(default_factory=lambda: [0] * NUM_SEATS)
    """HIDDEN. The 13 cards each seat was dealt; needed for replay and the critic."""

    # --- trick play ---------------------------------------------------
    leader_seat: int | None = None
    """The seat that led the current trick."""

    trick_number: int = 0
    """0..12 while playing; 13 once the deal is complete."""

    current_trick: list[tuple[int, int]] = field(default_factory=list)
    """``(seat, card)`` in play order for the trick in progress."""

    completed_tricks: list[list[tuple[int, int]]] = field(default_factory=list)
    """Every finished trick, in order, as ``(seat, card)`` lists."""

    trick_winners: list[int] = field(default_factory=list)
    """The winning seat of each completed trick, in order."""

    tricks_won: list[int] = field(default_factory=lambda: [0, 0])
    """Tricks taken per team, indexed by :func:`team_of_seat`."""

    void_flags: list[list[bool]] = field(
        default_factory=lambda: [[False] * NUM_SUITS for _ in range(NUM_SEATS)]
    )
    """PUBLIC. ``void_flags[seat][suit]`` once that seat has failed to follow it."""

    match_score: list[int] = field(default_factory=lambda: [0, 0])
    """Cumulative match points per team."""

    deal_result: DealResult | None = None
    """The scored outcome of the deal, set when DEAL_COMPLETED is emitted."""

    rng: np.random.Generator | None = None
    """The seeded generator this deal was dealt from. Never the global RNG."""

    match_seed: int | None = None

    events: list[PublicEvent] = field(default_factory=list)
    """The canonical ordered public history of the current deal."""

    deal_index: int = 0
    """Counts dealt hands within a match, including all-pass redeals."""

    redeal_count: int = 0
    """How many all-pass redeals this dealer has already caused (C10)."""

    @property
    def next_event_sequence_number(self) -> int:
        """The ``seq`` the next emitted event must carry.

        Sequence numbers are zero-based and contiguous within a deal, so the
        next one is simply the number of events already recorded.
        """
        return len(self.events)

    @property
    def declaring_team(self) -> int | None:
        """The team that won the auction, or ``None`` before a declarer exists."""
        return None if self.declarer_seat is None else team_of_seat(self.declarer_seat)

    def clone(self) -> TarneebState:
        """Return a deep-enough copy for exhaustive tree search.

        ``rules`` and ``PublicEvent`` are frozen, and the generator is shared by
        reference on purpose: a clone used for search must not consume draws
        that the original would later make.
        """
        return TarneebState(
            rules=self.rules,
            dealer_seat=self.dealer_seat,
            phase=self.phase,
            current_player_seat=self.current_player_seat,
            active_bidders=set(self.active_bidders),
            current_high_bid=self.current_high_bid,
            current_high_bidder=self.current_high_bidder,
            declarer_seat=self.declarer_seat,
            selected_trump_suit=self.selected_trump_suit,
            hands=list(self.hands),
            original_hands=list(self.original_hands),
            leader_seat=self.leader_seat,
            trick_number=self.trick_number,
            current_trick=list(self.current_trick),
            completed_tricks=[list(trick) for trick in self.completed_tricks],
            trick_winners=list(self.trick_winners),
            tricks_won=list(self.tricks_won),
            void_flags=[list(row) for row in self.void_flags],
            match_score=list(self.match_score),
            deal_result=self.deal_result,
            rng=self.rng,
            match_seed=self.match_seed,
            events=list(self.events),
            deal_index=self.deal_index,
            redeal_count=self.redeal_count,
        )
