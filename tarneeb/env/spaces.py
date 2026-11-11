"""Frozen observation contract ``obs_v2``: sizes, layout and seat helpers.

This module is the single source of truth for how many floats each observation
has and where every feature block starts. Nothing here reads game state; it only
describes the shape of the data, so the layout can be asserted mechanically and
reused by the observation builder, the tests and later neural code.

Seat indices in every observation are RELATIVE to the observer:
``0`` self, ``1`` right-hand opponent, ``2`` partner, ``3`` left-hand opponent.
"""

from __future__ import annotations

from tarneeb.core.actions import NUM_ACTIONS
from tarneeb.core.cards import NUM_CARDS, NUM_SEATS, NUM_SUITS
from tarneeb.core.state import TRICKS_PER_DEAL

OBS_SPEC_VERSION = "obs_v2"

# --- relative seats ---------------------------------------------------------

RELATIVE_SELF = 0
RELATIVE_RIGHT_HAND_OPPONENT = 1
RELATIVE_PARTNER = 2
RELATIVE_LEFT_HAND_OPPONENT = 3

RELATIVE_SEAT_NAMES = ("self", "right_hand_opponent", "partner", "left_hand_opponent")


def relative_seat(absolute_seat: int, observer_seat: int) -> int:
    """Return ``absolute_seat`` as an offset from ``observer_seat``.

    Matches :func:`tarneeb.core.events.relativize` exactly, so the static
    features and the event stream always agree on what "partner" means.
    """
    return (absolute_seat - observer_seat) % NUM_SEATS


def absolute_seat(relative: int, observer_seat: int) -> int:
    """Inverse of :func:`relative_seat`."""
    return (relative + observer_seat) % NUM_SEATS


def is_partner_relative(relative: int) -> bool:
    return relative == RELATIVE_PARTNER


def is_opponent_relative(relative: int) -> bool:
    return relative in (RELATIVE_RIGHT_HAND_OPPONENT, RELATIVE_LEFT_HAND_OPPONENT)


# --- static feature layout (blueprint section 5.3) --------------------------

# (block name, width). The order is frozen: changing it changes obs_v2.
ACTOR_FEATURE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("own_hand", NUM_CARDS),  # S01
    ("played_cards_by_relative_seat", NUM_SEATS * NUM_CARDS),  # S02
    ("current_trick_by_relative_seat", NUM_SEATS * NUM_CARDS),  # S03
    ("unseen_card_mask", NUM_CARDS),  # S04
    ("legal_action_mask", NUM_ACTIONS),  # S05
    ("void_matrix", NUM_SEATS * NUM_SUITS),  # S06
    ("play_order_in_trick", NUM_SEATS + 1),  # S07: position one-hot + leading bit
    ("led_suit", NUM_SUITS + 1),  # S08: none + four suits
    ("current_high_bid", 8),  # S09: none + 7..13
    ("high_bidder_relative", NUM_SEATS + 1),  # S10: none + four seats
    ("pass_status_relative", NUM_SEATS),  # S11
    ("trump", NUM_SUITS + 2),  # S12: none + four suits + reserved no-trump
    ("contract", 8),  # S13: none + 7..13
    ("declarer_role", 4),  # S14: self / partner / opponent / none
    ("tricks_won_one_hot", 2 * (TRICKS_PER_DEAL + 1)),  # S15
    ("tricks_remaining", 1),  # S16
    ("tricks_still_needed", 1),  # S17
    ("match_scores", 2),  # S18
    ("score_gap", 1),  # S19
    ("dealer_relative", NUM_SEATS),  # S20
    ("leader_relative", NUM_SEATS),  # S21
    ("phase", 3),  # S22: BID / DECLARE / PLAY
    ("trumps_unaccounted_for", 1),  # S23
    ("trick_number", 1),  # S24
)

CRITIC_ONLY_FEATURE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("hidden_hands_by_relative_seat", 3 * NUM_CARDS),  # X01: the other three hands
    ("trump_count_by_relative_seat", NUM_SEATS),  # X02
    ("high_card_count_by_relative_seat", NUM_SEATS),  # X03
)


def _build_offsets(blocks: tuple[tuple[str, int], ...], start: int = 0) -> dict[str, slice]:
    offsets: dict[str, slice] = {}
    cursor = start
    for block_name, width in blocks:
        offsets[block_name] = slice(cursor, cursor + width)
        cursor += width
    return offsets


ACTOR_FEATURE_SLICES: dict[str, slice] = _build_offsets(ACTOR_FEATURE_BLOCKS)

D_STATIC = sum(width for _name, width in ACTOR_FEATURE_BLOCKS)
D_CRITIC_ONLY = sum(width for _name, width in CRITIC_ONLY_FEATURE_BLOCKS)
D_CRITIC = D_STATIC + D_CRITIC_ONLY

CRITIC_FEATURE_SLICES: dict[str, slice] = {
    **ACTOR_FEATURE_SLICES,
    **_build_offsets(CRITIC_ONLY_FEATURE_BLOCKS, start=D_STATIC),
}

# The frozen sizes. Asserted here so an accidental layout edit fails on import.
assert D_STATIC == 692, f"static actor observation must be 692 dims, got {D_STATIC}"
assert D_CRITIC_ONLY == 164, f"critic-only block must be 164 dims, got {D_CRITIC_ONLY}"
assert D_CRITIC == 856, f"critic observation must be 856 dims, got {D_CRITIC}"

# --- event history ----------------------------------------------------------

N_EVENT_FIELDS = 7
D_EVENT_ONEHOT = 99
L_MAX = 80
"""Padded history length for the feedforward baseline; 77 max events plus headroom."""

K_MAX = 16
"""Max events between two consecutive decisions of one seat; the proved bound is 14."""

# --- scalar normalisation ---------------------------------------------------

MATCH_SCORE_CLIP = 1.5
"""Match-score scalars are divided by the target and clipped to +/- this value."""

MAX_ABSOLUTE_SCORE_DIFFERENCE = 42
"""The frozen |score difference| bound from docs/RULES.md section 5.3."""
