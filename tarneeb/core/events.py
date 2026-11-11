"""Canonical public-event schema ``event_v1``.

This module defines the data structures only. Nothing here emits events, and
nothing here knows about the auction, the engine or any hidden state.

Invariant EV-1: a ``PublicEvent`` never carries information that is not already
public. ``card`` is populated only for ``CARD_PLAYED`` -- a card that has just
been placed face up. No field can hold a hand, a bitmask or any private engine
state. This is enforced by tests, not by convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

EVENT_SCHEMA_VERSION = "event_v1"

#: Number of seats at the table; the modulus for seat-relative conversion.
NUM_SEATS = 4


class Phase(IntEnum):
    """Deal phases.

    Defined here rather than in ``state.py`` because ``PublicEvent`` needs it
    and the state module will in turn hold a list of ``PublicEvent``; putting
    it any higher would make the two modules import each other.
    """

    DEAL = 0
    BID = 1
    DECLARE = 2
    PLAY = 3
    SCORE = 4
    DONE = 5


class EventType(IntEnum):
    """The seven public event types of ``event_v1``.

    The numeric values are part of the frozen schema: they index the ``type``
    categorical of the event encoding, so reordering them would silently
    invalidate every stored buffer and checkpoint.
    """

    PAD = 0
    BID = 1
    PASS = 2
    TRUMP_SELECTED = 3
    CARD_PLAYED = 4
    TRICK_COMPLETED = 5
    DEAL_COMPLETED = 6


@dataclass(frozen=True, slots=True)
class DealResult:
    """The public outcome of one deal, carried by ``DEAL_COMPLETED``.

    Every field is already public once the last trick is played: the winning
    bid, how many tricks the bidding team took, and both score changes.
    """

    bid: int
    tricks_bidding_team: int
    score_delta_bidding_team: int
    score_delta_defending_team: int


@dataclass(frozen=True, slots=True)
class PublicEvent:
    """One entry in the canonical ordered public history.

    Seat numbers are ABSOLUTE here. Use :func:`relativize` to obtain the form a
    particular observer sees.
    """

    seq: int
    """0-based, monotonic and contiguous within a deal; resets at each deal."""

    phase: Phase
    actor_seat: int | None
    """Absolute seat 0..3; ``None`` for TRICK_COMPLETED and DEAL_COMPLETED."""

    event_type: EventType
    bid_value: int | None
    """7..13, only for BID."""

    trump: int | None
    """0..3, only for TRUMP_SELECTED."""

    card: int | None
    """Card id 0..51, only for CARD_PLAYED (EV-1)."""

    trick_no: int
    """0..12 during PLAY; 13 on DEAL_COMPLETED; 0 during BID and DECLARE."""

    trick_winner_seat: int | None
    """Absolute seat 0..3, only on TRICK_COMPLETED."""

    deal_result: DealResult | None
    """Only on DEAL_COMPLETED."""


@dataclass(frozen=True, slots=True)
class RelPublicEvent:
    """A ``PublicEvent`` expressed from one observer's point of view.

    Seat fields become relative offsets: ``0`` self, ``1`` right-hand opponent,
    ``2`` partner, ``3`` left-hand opponent. This is what a policy consumes, so
    the same network can be shared across all four seats.
    """

    seq: int
    phase: Phase
    event_type: EventType
    actor_rel: int | None
    bid_value: int | None
    trump: int | None
    card: int | None
    trick_no: int
    winner_rel: int | None


def _relative_seat(seat: int | None, observer_seat: int) -> int | None:
    """Return ``seat`` as an offset from ``observer_seat``, preserving ``None``."""
    if seat is None:
        return None
    return (seat - observer_seat) % NUM_SEATS


def relativize(event: PublicEvent, observer_seat: int) -> RelPublicEvent:
    """Return ``event`` as seen from ``observer_seat``.

    ``actor_rel`` and ``winner_rel`` use 0 = self, 1 = right-hand opponent,
    2 = partner, 3 = left-hand opponent. A ``None`` seat stays ``None``.

    Args:
        event: The absolute-seat event to convert.
        observer_seat: The seat doing the observing, 0..3.

    Returns:
        The observer-relative form of the event.

    Raises:
        ValueError: If ``observer_seat`` is not a seat.
    """
    if not 0 <= observer_seat < NUM_SEATS:
        raise ValueError(f"observer_seat must be in 0..{NUM_SEATS - 1}, got {observer_seat}")

    return RelPublicEvent(
        seq=event.seq,
        phase=event.phase,
        event_type=event.event_type,
        actor_rel=_relative_seat(event.actor_seat, observer_seat),
        bid_value=event.bid_value,
        trump=event.trump,
        card=event.card,
        trick_no=event.trick_no,
        winner_rel=_relative_seat(event.trick_winner_seat, observer_seat),
    )
