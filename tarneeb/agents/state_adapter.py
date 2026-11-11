"""Rebuild an engine state from an actor-safe view, so one seat can be observed.

The Phase 5 arena hands an agent a :class:`~tarneeb.env.public_view.PublicView`.
The Phase 4 observation builder takes a
:class:`~tarneeb.core.state.TarneebState`. A neural agent needs both: the arena's
fair pairing and the frozen 692-dim ``obs_v2`` vector.

The tempting shortcut -- writing a second builder that turns a ``PublicView``
straight into 692 floats -- would create a second implementation of ``obs_v2``
that can drift from the real one. This module takes the other route: it
reconstructs a ``TarneebState`` from the view and lets the **one** existing
observation builder do the work.

**Why the reconstruction is faithful.** The actor observation is proved invariant
to how the unseen cards are distributed among the other three seats (invariant
I7, the Phase 4 leakage gate). So the reconstruction does not need to know who
holds what -- it only needs *a* distribution with the right hand sizes. It deals
the unseen cards deterministically in ascending order, and the resulting
observation is byte-identical to the one built from the true state. That
equality is asserted over thousands of real states by
``tests/test_state_adapter.py``.

Everything else the observation reads -- the ordered trick record, voids, the
auction, trump, scores, the phase -- is public and is reconstructed exactly, the
trick record by replaying the view's own canonical event stream.
"""

from __future__ import annotations

from tarneeb.core.cards import (
    CARDS_PER_HAND,
    NUM_CARDS,
    NUM_SEATS,
    NUM_SUITS,
    hand_from_cards,
    hand_size,
)
from tarneeb.core.events import EventType
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState, team_of_seat
from tarneeb.env.public_view import PublicView


def _replay_completed_tricks(view: PublicView) -> tuple[list[list[tuple[int, int]]], list[int]]:
    """Rebuild the ordered completed tricks and their winners from the events.

    The event stream is the canonical public history, so this is a reconstruction
    of a public record, not an inference about hidden cards.
    """
    completed_tricks: list[list[tuple[int, int]]] = []
    trick_winners: list[int] = []
    trick_in_progress: list[tuple[int, int]] = []

    for event in view.events:
        if event.event_type is EventType.CARD_PLAYED:
            assert event.actor_seat is not None and event.card is not None
            trick_in_progress.append((event.actor_seat, event.card))
        elif event.event_type is EventType.TRICK_COMPLETED:
            assert event.trick_winner_seat is not None
            completed_tricks.append(trick_in_progress)
            trick_winners.append(event.trick_winner_seat)
            trick_in_progress = []

    return completed_tricks, trick_winners


def _distribute_unseen_cards(view: PublicView, hand_sizes: list[int]) -> list[int]:
    """Deal the unseen cards to the other three seats, ascending, deterministically.

    Any distribution with the correct hand sizes yields the same actor
    observation, so this arbitrary one is sufficient. It is never shown to the
    policy: it exists only so the engine can compute the observer's legal moves
    and so the observation builder finds a well-formed state.
    """
    unseen = sorted(view.unseen_cards)
    hands = [0] * NUM_SEATS
    hands[view.observer_seat] = hand_from_cards(view.own_hand)

    cursor = 0
    for seat in range(NUM_SEATS):
        if seat == view.observer_seat:
            continue
        take = hand_sizes[seat]
        hands[seat] = hand_from_cards(unseen[cursor : cursor + take])
        cursor += take
    return hands


def reconstruct_state_from_public_view(
    view: PublicView, rules: RulesConfig | None = None
) -> TarneebState:
    """Return a ``TarneebState`` that reproduces ``view`` for the observer.

    The returned state is valid for building the observer's actor observation,
    its padded event history and its legal-action mask. It is not a replayable
    game state: the three hidden hands are placeholders with the correct sizes.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    completed_tricks, trick_winners = _replay_completed_tricks(view)

    cards_played_by_seat = [len(cards) for cards in view.played_cards_by_seat]
    hand_sizes = [CARDS_PER_HAND - played for played in cards_played_by_seat]
    hand_sizes[view.observer_seat] = len(view.own_hand)
    hands = _distribute_unseen_cards(view, hand_sizes)

    tricks_won = [0, 0]
    for winner_seat in trick_winners:
        tricks_won[team_of_seat(winner_seat)] += 1

    state = TarneebState(
        rules=resolved_rules,
        dealer_seat=view.dealer_seat,
        phase=view.phase,
        current_player_seat=view.observer_seat,
        active_bidders={seat for seat in range(NUM_SEATS) if seat not in view.seats_that_passed},
        current_high_bid=view.current_high_bid,
        current_high_bidder=view.current_high_bidder,
        declarer_seat=view.declarer_seat,
        selected_trump_suit=view.trump_suit,
        hands=hands,
        original_hands=[0] * NUM_SEATS,
        leader_seat=view.leader_seat,
        trick_number=view.trick_number,
        current_trick=list(view.current_trick),
        completed_tricks=completed_tricks,
        trick_winners=trick_winners,
        tricks_won=tricks_won,
        void_flags=[list(row) for row in view.void_flags],
        match_score=list(view.match_score_by_team),
        events=list(view.events),
    )
    _validate_reconstruction(state, view)
    return state


def _validate_reconstruction(state: TarneebState, view: PublicView) -> None:
    """Fail loudly if the reconstruction disagrees with the view it came from."""
    total_cards = sum(hand_size(hand) for hand in state.hands)
    publicly_played = len(view.publicly_played_cards)
    if total_cards + publicly_played != NUM_CARDS:
        raise ValueError(
            f"reconstruction accounts for {total_cards + publicly_played} cards, expected "
            f"{NUM_CARDS}; the public view is inconsistent"
        )
    if len(state.void_flags) != NUM_SEATS or any(
        len(row) != NUM_SUITS for row in state.void_flags
    ):  # pragma: no cover - shape comes straight from the view
        raise ValueError("void flags have the wrong shape")
