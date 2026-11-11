"""The A7 history conditions, applied at the experiment boundary.

A7 asks whether the *ordered auction history* actually helps a feedforward policy.
To answer it honestly, three conditions must differ in exactly one respect: what
the history encoder is shown.

- **full** -- the ordered public event stream, unchanged.
- **auction_removed** -- every auction event (bid, pass, trump declaration) is
  deleted from the history. Play events keep their order.
- **current_bid_only** -- the ordered auction is replaced by a single synthetic
  summary row carrying the winning bid and who made it. This reproduces the older,
  weaker design in which only aggregate bid state survived.

**Nothing frozen is touched.** ``obs_v2`` is unchanged, event generation is
unchanged, and the actor architecture is unchanged. The transform is a subclass of
the frozen :class:`~tarneeb.env.observation.ObservationBuilder` that overrides
``padded_history`` only, so the 692-float static vector and the legal-action mask
are produced by exactly the same frozen code in all three conditions.

The transform is also information-safe by construction: it only ever *deletes* or
*summarises* rows the frozen builder already produced, and never adds a field the
observer could not legally see.
"""

from __future__ import annotations

import numpy as np

from tarneeb.core.events import EventType
from tarneeb.core.state import TarneebState
from tarneeb.curriculum.stages import (
    AUCTION_REMOVED,
    CURRENT_BID_ONLY,
    FULL_HISTORY,
    HISTORY_CONDITIONS,
)
from tarneeb.env.event_features import EVENT_ID_DTYPE, LOWEST_BID
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.spaces import L_MAX, N_EVENT_FIELDS, relative_seat

AUCTION_EVENT_TYPES: frozenset[int] = frozenset(
    {int(EventType.BID), int(EventType.PASS), int(EventType.TRUMP_SELECTED)}
)
"""The event types that carry auction information, as stored in field 0."""

EVENT_TYPE_FIELD = 0
ACTOR_RELATIVE_FIELD = 1
BID_VALUE_FIELD = 2


def strip_auction_rows(
    history_ids: np.ndarray, history_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Delete auction rows and close the gap, keeping play events in order.

    Padding stays on the right and the mask stays consistent with it, which is what
    the frozen encoder's "zero after the projection" rule relies on.
    """
    kept_rows = [
        history_ids[position]
        for position in range(L_MAX)
        if history_masks[position]
        and int(history_ids[position, EVENT_TYPE_FIELD]) not in AUCTION_EVENT_TYPES
    ]
    return _pack_rows(kept_rows)


def summarise_auction_rows(
    history_ids: np.ndarray, history_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the ordered auction with one row holding the winning bid.

    The summary is built from rows the observer already had: the highest bid seen
    and the relative seat that made it. Who passed, in what order, and every
    outbid intermediate bid are all discarded -- which is precisely the information
    this condition is designed to withhold.
    """
    summary_row: np.ndarray | None = None
    highest_bid_value = 0
    play_rows: list[np.ndarray] = []

    for position in range(L_MAX):
        if not history_masks[position]:
            continue
        row = history_ids[position]
        if int(row[EVENT_TYPE_FIELD]) not in AUCTION_EVENT_TYPES:
            play_rows.append(row)
            continue
        bid_value = int(row[BID_VALUE_FIELD])
        if int(row[EVENT_TYPE_FIELD]) == int(EventType.BID) and bid_value > highest_bid_value:
            highest_bid_value = bid_value
            summary_row = np.zeros(N_EVENT_FIELDS, dtype=EVENT_ID_DTYPE)
            summary_row[EVENT_TYPE_FIELD] = int(EventType.BID)
            summary_row[ACTOR_RELATIVE_FIELD] = row[ACTOR_RELATIVE_FIELD]
            summary_row[BID_VALUE_FIELD] = bid_value

    kept_rows = ([summary_row] if summary_row is not None else []) + play_rows
    return _pack_rows(kept_rows)


def _pack_rows(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad ``rows`` back into the frozen ``(80, 7)`` / ``(80,)`` shape."""
    packed_ids = np.zeros((L_MAX, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE)
    packed_mask = np.zeros(L_MAX, dtype=bool)
    for position, row in enumerate(rows[:L_MAX]):
        packed_ids[position] = row
        packed_mask[position] = True
    return packed_ids, packed_mask


def apply_history_condition(
    history_condition: str, history_ids: np.ndarray, history_masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the history the encoder should see under ``history_condition``."""
    if history_condition == FULL_HISTORY:
        return history_ids, history_masks
    if history_condition == AUCTION_REMOVED:
        return strip_auction_rows(history_ids, history_masks)
    if history_condition == CURRENT_BID_ONLY:
        return summarise_auction_rows(history_ids, history_masks)
    raise ValueError(
        f"unknown history_condition {history_condition!r}; expected one of {HISTORY_CONDITIONS}"
    )


class HistoryConditionedObservationBuilder(ObservationBuilder):
    """The frozen observation builder with one method specialised for A7.

    Only ``padded_history`` is overridden. ``static_actor``, ``static_critic``,
    ``ui_view`` and the legal-action mask all remain exactly the frozen
    implementations, which is what makes the ablation a controlled experiment
    rather than a different observation.
    """

    def __init__(self, engine: object, history_condition: str = FULL_HISTORY) -> None:
        super().__init__(engine)  # type: ignore[arg-type]
        if history_condition not in HISTORY_CONDITIONS:
            raise ValueError(f"unknown history_condition {history_condition!r}")
        self.history_condition = history_condition

    def padded_history(
        self, state: TarneebState, observer_seat: int
    ) -> tuple[np.ndarray, np.ndarray]:
        history_ids, history_masks = super().padded_history(state, observer_seat)
        return apply_history_condition(self.history_condition, history_ids, history_masks)


def auction_event_count(history_ids: np.ndarray, history_masks: np.ndarray) -> int:
    """Count the auction rows a history still contains. Used by the A7 tests."""
    return sum(
        1
        for position in range(L_MAX)
        if history_masks[position]
        and int(history_ids[position, EVENT_TYPE_FIELD]) in AUCTION_EVENT_TYPES
    )


def winning_bid_in_history(history_ids: np.ndarray, history_masks: np.ndarray) -> int | None:
    """Return the highest bid value encoded in a history, or ``None``.

    Field 2 stores ``bid - 7 + 1``, so this converts back to a real bid value.
    """
    highest = 0
    for position in range(L_MAX):
        if not history_masks[position]:
            continue
        row = history_ids[position]
        if int(row[EVENT_TYPE_FIELD]) == int(EventType.BID):
            highest = max(highest, int(row[BID_VALUE_FIELD]))
    return None if highest == 0 else highest + LOWEST_BID - 1


def relative_declarer_in_history(history_ids: np.ndarray, history_masks: np.ndarray) -> int | None:
    """Return the relative seat of the highest bidder recorded in a history."""
    highest, actor = 0, None
    for position in range(L_MAX):
        if not history_masks[position]:
            continue
        row = history_ids[position]
        if int(row[EVENT_TYPE_FIELD]) == int(EventType.BID) and int(row[BID_VALUE_FIELD]) > highest:
            highest = int(row[BID_VALUE_FIELD])
            actor = int(row[ACTOR_RELATIVE_FIELD]) - 1
    return actor


__all__ = [
    "AUCTION_EVENT_TYPES",
    "HistoryConditionedObservationBuilder",
    "apply_history_condition",
    "auction_event_count",
    "relative_declarer_in_history",
    "relative_seat",
    "strip_auction_rows",
    "summarise_auction_rows",
    "winning_bid_in_history",
]
