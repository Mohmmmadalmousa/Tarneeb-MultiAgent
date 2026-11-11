"""Factored categorical encoding of ``event_v1`` public events.

Every public event becomes exactly seven small categorical ids, and those ids
expand to a 99-dimensional one-hot. The ids are what buffers store (about seven
times smaller than one-hot), and the expansion happens on device.

Field 0 is the only one whose zero is meaningful data (``PAD``); the other six
reserve index 0 for "none", so a padded row is all zeros.
"""

from __future__ import annotations

import numpy as np

from tarneeb.core.cards import NUM_CARDS, NUM_SEATS, NUM_SUITS
from tarneeb.core.events import EventType, PublicEvent, relativize
from tarneeb.core.state import TRICKS_PER_DEAL
from tarneeb.env.spaces import D_EVENT_ONEHOT, N_EVENT_FIELDS

LOWEST_BID = 7
HIGHEST_BID = 13

# Field order is frozen: it defines what column k of an id row means.
EVENT_FIELD_NAMES: tuple[str, ...] = (
    "event_type",
    "actor_rel",
    "bid_value",
    "trump",
    "card",
    "trick_no",
    "winner_rel",
)

EVENT_FIELD_CARDINALITIES: tuple[int, ...] = (
    len(EventType),  # 7 event types, PAD included
    NUM_SEATS + 1,  # none + four relative seats
    (HIGHEST_BID - LOWEST_BID + 1) + 1,  # none + bids 7..13
    NUM_SUITS + 2,  # none + four suits + reserved no-trump
    NUM_CARDS + 1,  # none + 52 cards
    (TRICKS_PER_DEAL + 1) + 1,  # none + trick numbers 0..13
    NUM_SEATS + 1,  # none + four relative seats
)

EVENT_FIELD_OFFSETS: tuple[int, ...] = tuple(
    int(offset) for offset in np.cumsum((0, *EVENT_FIELD_CARDINALITIES[:-1]))
)

EVENT_ID_DTYPE = np.int16

assert len(EVENT_FIELD_CARDINALITIES) == N_EVENT_FIELDS
assert len(EVENT_FIELD_NAMES) == N_EVENT_FIELDS
assert sum(EVENT_FIELD_CARDINALITIES) == D_EVENT_ONEHOT, (
    f"factored one-hot must total {D_EVENT_ONEHOT}, got {sum(EVENT_FIELD_CARDINALITIES)}"
)

PAD_EVENT_IDS: np.ndarray = np.zeros(N_EVENT_FIELDS, dtype=EVENT_ID_DTYPE)
"""An all-zero row: event type PAD and "none" in every other field."""


def _optional_id(value: int | None, offset: int = 0) -> int:
    """Return 0 for ``None``, else ``value - offset + 1`` so 0 stays reserved."""
    return 0 if value is None else value - offset + 1


def event_ids(event: PublicEvent, observer_seat: int) -> np.ndarray:
    """Return the seven factored categorical ids of ``event`` for ``observer_seat``.

    Seats are converted to the observer-relative convention, so the same event
    encodes differently for each of the four seats -- which is what lets one
    shared network serve every seat.
    """
    relative_event = relativize(event, observer_seat)

    ids = np.empty(N_EVENT_FIELDS, dtype=EVENT_ID_DTYPE)
    ids[0] = int(relative_event.event_type)
    ids[1] = _optional_id(relative_event.actor_rel)
    ids[2] = _optional_id(relative_event.bid_value, offset=LOWEST_BID)
    ids[3] = _optional_id(relative_event.trump)
    ids[4] = _optional_id(relative_event.card)
    ids[5] = _optional_id(relative_event.trick_no)
    ids[6] = _optional_id(relative_event.winner_rel)
    return ids


def event_onehot(event: PublicEvent, observer_seat: int) -> np.ndarray:
    """Return the 99-dimensional one-hot expansion of :func:`event_ids`."""
    return expand_event_ids(event_ids(event, observer_seat))


def expand_event_ids(ids: np.ndarray) -> np.ndarray:
    """Expand one row of seven ids into the 99-dim concatenated one-hot."""
    if ids.shape != (N_EVENT_FIELDS,):
        raise ValueError(f"expected {N_EVENT_FIELDS} ids, got shape {ids.shape}")

    onehot = np.zeros(D_EVENT_ONEHOT, dtype=np.float32)
    for field_index, field_id in enumerate(ids):
        cardinality = EVENT_FIELD_CARDINALITIES[field_index]
        if not 0 <= field_id < cardinality:
            raise ValueError(
                f"id {field_id} out of range for field "
                f"{EVENT_FIELD_NAMES[field_index]!r} (cardinality {cardinality})"
            )
        onehot[EVENT_FIELD_OFFSETS[field_index] + int(field_id)] = 1.0
    return onehot


def expand_event_id_rows(id_rows: np.ndarray) -> np.ndarray:
    """Expand an ``(L, 7)`` id array into ``(L, 99)`` one-hot rows."""
    if id_rows.ndim != 2 or id_rows.shape[1] != N_EVENT_FIELDS:
        raise ValueError(f"expected an (L, {N_EVENT_FIELDS}) array, got {id_rows.shape}")
    return np.stack([expand_event_ids(row) for row in id_rows])
