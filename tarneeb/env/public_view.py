"""Actor-safe structured view of a deal, for non-neural agents.

A heuristic needs to reason about cards and suits, not about 692 floats, but it
must be held to exactly the same information boundary as a policy network. This
module builds a read-only snapshot containing precisely what the observer may
legally see:

- the observer's own hand;
- the complete public event stream;
- everything already public in the state -- bidding, trump, contract, declarer,
  the current trick, completed tricks, public void flags, trick counts, scores,
  dealer, leader, phase;
- the observer's own legal-action mask.

It must never contain another seat's hand, ``original_hands``, or any hint of
which opponent holds an unseen card. That is enforced by tests, not convention:
redistributing the hidden cards must leave this view byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tarneeb.core.cards import (
    NUM_CARDS,
    NUM_SEATS,
    NUM_SUITS,
    cards_in_hand,
    suit_of,
)
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase, PublicEvent
from tarneeb.core.state import TarneebState, team_of_seat
from tarneeb.env.spaces import relative_seat

LIVE_PHASES = (Phase.BID, Phase.DECLARE, Phase.PLAY)


@dataclass(frozen=True)
class PublicView:
    """What one seat may legally see, in structured form."""

    observer_seat: int
    phase: Phase
    legal_action_mask: np.ndarray
    """``(65,)`` bool, straight from the engine. All False when not on turn."""

    own_hand: tuple[int, ...]
    """The observer's remaining cards, ascending."""

    # --- auction ------------------------------------------------------
    dealer_seat: int
    current_high_bid: int | None
    current_high_bidder: int | None
    seats_that_passed: frozenset[int]
    declarer_seat: int | None
    trump_suit: int | None

    # --- play ---------------------------------------------------------
    leader_seat: int | None
    trick_number: int
    current_trick: tuple[tuple[int, int], ...]
    """``(seat, card)`` in play order for the trick in progress."""

    played_cards_by_seat: tuple[frozenset[int], ...]
    """Public: every card each seat has already played."""

    void_flags: tuple[tuple[bool, ...], ...]
    """Public: ``void_flags[seat][suit]`` once that seat failed to follow."""

    tricks_won_by_team: tuple[int, int]
    match_score_by_team: tuple[int, int]

    events: tuple[PublicEvent, ...]
    """The canonical ordered public history of this deal."""

    target_score: int
    min_bid: int
    max_bid: int

    # --- derived conveniences, all from public information -------------

    @property
    def observer_team(self) -> int:
        return team_of_seat(self.observer_seat)

    @property
    def partner_seat(self) -> int:
        return (self.observer_seat + 2) % NUM_SEATS

    @property
    def opponent_seats(self) -> tuple[int, int]:
        return (
            (self.observer_seat + 1) % NUM_SEATS,
            (self.observer_seat + 3) % NUM_SEATS,
        )

    @property
    def observer_is_declarer(self) -> bool:
        return self.declarer_seat == self.observer_seat

    @property
    def observer_is_declaring_team(self) -> bool:
        return (
            self.declarer_seat is not None
            and team_of_seat(self.declarer_seat) == self.observer_team
        )

    @property
    def led_suit(self) -> int | None:
        """The suit led to the current trick, or ``None`` if it has not started."""
        if not self.current_trick:
            return None
        return suit_of(self.current_trick[0][1])

    @property
    def publicly_played_cards(self) -> frozenset[int]:
        """Every card played so far by anyone, including the current trick."""
        played: set[int] = set()
        for seat_cards in self.played_cards_by_seat:
            played |= seat_cards
        return frozenset(played)

    @property
    def unseen_cards(self) -> frozenset[int]:
        """Cards not in my hand and not yet public.

        Says a card is somewhere among the other three seats. Never who holds it.
        """
        own = set(self.own_hand)
        public = self.publicly_played_cards
        return frozenset(
            card for card in range(NUM_CARDS) if card not in own and card not in public
        )

    @property
    def tricks_remaining(self) -> int:
        return len(self.own_hand)

    def relative_of(self, seat: int) -> int:
        return relative_seat(seat, self.observer_seat)

    def cards_of_suit_in_own_hand(self, suit: int) -> tuple[int, ...]:
        return tuple(card for card in self.own_hand if suit_of(card) == suit)

    def own_suit_lengths(self) -> tuple[int, ...]:
        return tuple(len(self.cards_of_suit_in_own_hand(suit)) for suit in range(NUM_SUITS))

    def seat_is_void_in(self, seat: int, suit: int) -> bool:
        return self.void_flags[seat][suit]

    def unseen_cards_of_suit(self, suit: int) -> frozenset[int]:
        return frozenset(card for card in self.unseen_cards if suit_of(card) == suit)

    def contract_tricks_needed(self) -> int | None:
        """Tricks the declaring side still needs, or ``None`` before a contract."""
        if self.declarer_seat is None or self.current_high_bid is None:
            return None
        declaring_team = team_of_seat(self.declarer_seat)
        return max(0, self.current_high_bid - self.tricks_won_by_team[declaring_team])


@dataclass
class PublicViewBuilder:
    """Builds :class:`PublicView` snapshots, taking legality from the engine."""

    engine: TarneebEngine
    _empty_mask: np.ndarray = field(default_factory=lambda: np.zeros(65, dtype=bool), repr=False)

    def build(self, state: TarneebState, observer_seat: int) -> PublicView:
        """Return the actor-safe view of ``state`` for ``observer_seat``."""
        if not 0 <= observer_seat < NUM_SEATS:
            raise ValueError(f"observer_seat must be in 0..{NUM_SEATS - 1}, got {observer_seat}")

        # Only the observer's own mask: during PLAY the engine's mask is derived
        # from the acting seat's hand, so handing it to another seat would leak.
        if state.phase in LIVE_PHASES and state.current_player_seat == observer_seat:
            legal_action_mask = self.engine.legal_actions(state)
        else:
            legal_action_mask = self._empty_mask.copy()

        played_cards_by_seat: list[set[int]] = [set() for _ in range(NUM_SEATS)]
        for trick in state.completed_tricks:
            for seat, card in trick:
                played_cards_by_seat[seat].add(card)
        for seat, card in state.current_trick:
            played_cards_by_seat[seat].add(card)

        return PublicView(
            observer_seat=observer_seat,
            phase=state.phase,
            legal_action_mask=legal_action_mask,
            own_hand=tuple(cards_in_hand(state.hands[observer_seat])),
            dealer_seat=state.dealer_seat,
            current_high_bid=state.current_high_bid,
            current_high_bidder=state.current_high_bidder,
            seats_that_passed=frozenset(
                seat for seat in range(NUM_SEATS) if seat not in state.active_bidders
            ),
            declarer_seat=state.declarer_seat,
            trump_suit=state.selected_trump_suit,
            leader_seat=state.leader_seat,
            trick_number=state.trick_number,
            current_trick=tuple(state.current_trick),
            played_cards_by_seat=tuple(frozenset(cards) for cards in played_cards_by_seat),
            void_flags=tuple(tuple(row) for row in state.void_flags),
            tricks_won_by_team=(state.tricks_won[0], state.tricks_won[1]),
            match_score_by_team=(state.match_score[0], state.match_score[1]),
            events=tuple(state.events),
            target_score=state.rules.target_score,
            min_bid=state.rules.min_bid,
            max_bid=state.rules.max_bid,
        )
