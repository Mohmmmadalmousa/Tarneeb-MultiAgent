"""Deterministic Tarneeb engine.

Runs the complete deal cycle ``DEAL -> BID -> DECLARE -> PLAY -> SCORE``. Every
legality decision is exposed as a 65-entry boolean mask, and every publicly
observable state change emits a :class:`~tarneeb.core.events.PublicEvent`.

This module is pure: no torch, no gymnasium, no I/O, no global randomness. All
shuffling comes from a per-deal seeded ``numpy.random.Generator`` derived from
``(match_seed, deal_index)``, which makes dealing a pure function of those two
numbers and duplicate-deal evaluation possible.
"""

from __future__ import annotations

import numpy as np

from tarneeb.core.actions import (
    NO_TRUMP_ACTION,
    NUM_ACTIONS,
    NUM_SUITS,
    PASS_ACTION,
    action_for_bid,
    action_for_card,
    action_for_trump_suit,
    bid_for_action,
    card_for_action,
    is_bid_action,
    is_card_action,
    is_trump_action,
    trump_suit_for_action,
)
from tarneeb.core.cards import (
    cards_in_hand,
    cards_of_suit_in_hand,
    deal_hands,
    derive_deal_seed,
    hand_contains,
    holds_suit,
    rank_of,
    remove_card,
    suit_of,
)
from tarneeb.core.events import (
    NUM_SEATS,
    DealResult,
    EventType,
    Phase,
    PublicEvent,
)
from tarneeb.core.rules import RulesConfig
from tarneeb.core.scoring import score_deal
from tarneeb.core.state import (
    TRICKS_PER_DEAL,
    TarneebState,
    first_bidder_seat,
    next_seat,
    team_of_seat,
)

ALL_SEATS = frozenset(range(NUM_SEATS))
CARDS_PER_TRICK = NUM_SEATS


class TarneebEngine:
    """Applies the frozen rules of ``tarneeb-classic-v1`` to a deal state."""

    def __init__(self, rules: RulesConfig) -> None:
        if rules.egyptian_bidding:
            raise NotImplementedError(
                "egyptian_bidding is out of scope; it is not part of tarneeb-classic-v1"
            )
        if rules.single_bid_round or rules.dealer_may_equal or rules.dealer_forced_seven:
            raise NotImplementedError(
                "auction variants (single_bid_round, dealer_may_equal, dealer_forced_seven) "
                "are not implemented under tarneeb-classic-v1"
            )
        self.rules = rules

    # --- deal lifecycle ----------------------------------------------------

    def new_match(self, match_seed: int, dealer_seat: int = 0) -> TarneebState:
        """Return the opening state of a match, in the DEAL phase."""
        return self.new_deal(dealer_seat, deal_index=0, match_seed=match_seed)

    def new_deal(self, dealer_seat: int, deal_index: int = 0, match_seed: int = 0) -> TarneebState:
        """Return a fresh state in the DEAL phase for ``dealer_seat``.

        The cards are not dealt yet; :meth:`deal` does that, using a generator
        derived from ``(match_seed, deal_index)``.
        """
        if not 0 <= dealer_seat < NUM_SEATS:
            raise ValueError(f"dealer_seat must be in 0..{NUM_SEATS - 1}, got {dealer_seat}")
        return TarneebState(
            rules=self.rules,
            dealer_seat=dealer_seat,
            phase=Phase.DEAL,
            current_player_seat=None,
            deal_index=deal_index,
            match_seed=match_seed,
        )

    def deal(self, state: TarneebState) -> TarneebState:
        """Shuffle, deal 13 cards to each seat, and open the auction.

        Clears the event stream so ``seq`` restarts at 0 for the new deal,
        reactivates all four bidders and puts the seat to the dealer's right on
        turn (C06). The shuffle uses a generator derived from
        ``(match_seed, deal_index)``, so the same pair always deals the same
        four hands.
        """
        if state.phase is not Phase.DEAL:
            raise ValueError(f"deal() requires phase DEAL, state is in {state.phase.name}")

        deal_seed = derive_deal_seed(state.match_seed or 0, state.deal_index)
        state.rng = np.random.default_rng(deal_seed)
        state.hands = deal_hands(state.rng)
        state.original_hands = list(state.hands)

        state.events = []
        state.active_bidders = set(ALL_SEATS)
        state.current_high_bid = None
        state.current_high_bidder = None
        state.declarer_seat = None
        state.selected_trump_suit = None

        state.leader_seat = None
        state.trick_number = 0
        state.current_trick = []
        state.completed_tricks = []
        state.trick_winners = []
        state.tricks_won = [0, 0]
        state.void_flags = [[False] * NUM_SUITS for _ in range(NUM_SEATS)]
        state.deal_result = None

        state.phase = Phase.BID
        state.current_player_seat = first_bidder_seat(state.dealer_seat)
        return state

    def next_deal(self, state: TarneebState) -> TarneebState:
        """Advance to the next hand after a scored deal, rotating the dealer (C05).

        The all-pass redeal is NOT this method: that keeps the same dealer and
        is handled inside the auction.
        """
        if state.phase is not Phase.SCORE:
            raise ValueError(f"next_deal() requires phase SCORE, state is in {state.phase.name}")
        state.dealer_seat = next_seat(state.dealer_seat)
        state.deal_index += 1
        state.redeal_count = 0
        state.phase = Phase.DEAL
        state.current_player_seat = None
        return state

    # --- legality ----------------------------------------------------------

    def legal_actions(self, state: TarneebState) -> np.ndarray:
        """Return the 65-entry boolean legal-action mask for the seat on turn.

        Entry ``NO_TRUMP_ACTION`` is False in every reachable state (C13).
        """
        legal_action_mask = np.zeros(NUM_ACTIONS, dtype=bool)

        if state.phase is Phase.BID:
            lowest_legal_bid = (
                self.rules.min_bid if state.current_high_bid is None else state.current_high_bid + 1
            )
            for bid_value in range(lowest_legal_bid, self.rules.max_bid + 1):
                legal_action_mask[action_for_bid(bid_value)] = True
            legal_action_mask[PASS_ACTION] = True
            return legal_action_mask

        if state.phase is Phase.DECLARE:
            for suit in range(NUM_SUITS):
                legal_action_mask[action_for_trump_suit(suit)] = True
            return legal_action_mask

        if state.phase is Phase.PLAY:
            for card in self.legal_cards(state):
                legal_action_mask[action_for_card(card)] = True
            return legal_action_mask

        raise ValueError(f"no seat is on turn in phase {state.phase.name}")

    def legal_cards(self, state: TarneebState) -> list[int]:
        """Return the cards the seat on turn may play (C16, C17).

        A seat holding the led suit must follow it; a seat void in the led suit
        may play anything, with no obligation to trump.
        """
        acting_seat = state.current_player_seat
        assert acting_seat is not None
        hand_mask = state.hands[acting_seat]

        if not state.current_trick:
            return cards_in_hand(hand_mask)  # C15: the leader may play any card

        led_suit = suit_of(state.current_trick[0][1])
        if holds_suit(hand_mask, led_suit):
            return cards_in_hand(cards_of_suit_in_hand(hand_mask, led_suit))
        return cards_in_hand(hand_mask)

    # --- transition --------------------------------------------------------

    def step(
        self, state: TarneebState, action_id: int
    ) -> tuple[TarneebState, list[PublicEvent], DealResult | None]:
        """Apply one action for the seat on turn.

        Returns the state, the events emitted by this action in chronological
        order, and the deal result -- non-``None`` only on the action that
        completes the thirteenth trick.
        """
        if not 0 <= action_id < NUM_ACTIONS:
            raise ValueError(f"action_id must be in 0..{NUM_ACTIONS - 1}, got {action_id}")

        legal_action_mask = self.legal_actions(state)
        if not legal_action_mask[action_id]:
            raise ValueError(
                f"action {action_id} is illegal in phase {state.phase.name} "
                f"for seat {state.current_player_seat}"
            )

        if state.phase is Phase.BID:
            emitted_events = self._step_auction(state, action_id)
        elif state.phase is Phase.DECLARE:
            emitted_events = self._step_declaration(state, action_id)
        else:
            emitted_events = self._step_card_play(state, action_id)

        deal_result = state.deal_result if state.phase is Phase.SCORE else None
        return state, emitted_events, deal_result

    def _step_auction(self, state: TarneebState, action_id: int) -> list[PublicEvent]:
        acting_seat = state.current_player_seat
        assert acting_seat is not None  # guaranteed by phase BID

        emitted_events: list[PublicEvent] = []

        if is_bid_action(action_id):
            bid_value = bid_for_action(action_id)
            state.current_high_bid = bid_value
            state.current_high_bidder = acting_seat
            emitted_events.append(
                self._emit(
                    state,
                    phase=Phase.BID,
                    actor_seat=acting_seat,
                    event_type=EventType.BID,
                    bid_value=bid_value,
                )
            )
        else:
            state.active_bidders.discard(acting_seat)  # passing is permanent (C09)
            emitted_events.append(
                self._emit(
                    state,
                    phase=Phase.BID,
                    actor_seat=acting_seat,
                    event_type=EventType.PASS,
                )
            )

        self._advance_auction(state, acting_seat)
        return emitted_events

    def _advance_auction(self, state: TarneebState, acting_seat: int) -> None:
        """Move to the next bidder, or end the auction (C10, C11)."""
        if not state.active_bidders:
            # All four passed with no bid: throw the hand in and redeal with the
            # SAME dealer (C10).
            state.phase = Phase.DEAL
            state.current_player_seat = None
            state.redeal_count += 1
            state.deal_index += 1
            return

        if len(state.active_bidders) == 1 and state.current_high_bid is not None:
            # Exactly one active bidder and a bid exists: that seat is the
            # declarer and does not act again in the auction (C11).
            (declarer_seat,) = state.active_bidders
            state.declarer_seat = declarer_seat
            state.phase = Phase.DECLARE
            state.current_player_seat = declarer_seat
            return

        state.current_player_seat = self._next_active_bidder(state, acting_seat)

    def _next_active_bidder(self, state: TarneebState, acting_seat: int) -> int:
        """Return the next seat counter-clockwise that has not passed."""
        candidate_seat = next_seat(acting_seat)
        for _ in range(NUM_SEATS):
            if candidate_seat in state.active_bidders:
                return candidate_seat
            candidate_seat = next_seat(candidate_seat)
        raise AssertionError("no active bidder remains; the auction should have ended")

    def _step_declaration(self, state: TarneebState, action_id: int) -> list[PublicEvent]:
        if action_id == NO_TRUMP_ACTION:  # pragma: no cover - masked, never reachable
            raise AssertionError("no-trump is permanently masked under tarneeb-classic-v1")
        assert is_trump_action(action_id)

        declarer_seat = state.declarer_seat
        assert declarer_seat is not None  # guaranteed by phase DECLARE

        selected_trump_suit = trump_suit_for_action(action_id)
        state.selected_trump_suit = selected_trump_suit

        trump_event = self._emit(
            state,
            phase=Phase.DECLARE,
            actor_seat=declarer_seat,
            event_type=EventType.TRUMP_SELECTED,
            trump=selected_trump_suit,
        )

        # The declarer leads to the first trick (C14).
        state.phase = Phase.PLAY
        state.leader_seat = declarer_seat
        state.current_player_seat = declarer_seat
        state.trick_number = 0
        state.current_trick = []
        return [trump_event]

    # --- trick play --------------------------------------------------------

    def _step_card_play(self, state: TarneebState, action_id: int) -> list[PublicEvent]:
        assert is_card_action(action_id)
        acting_seat = state.current_player_seat
        assert acting_seat is not None

        played_card = card_for_action(action_id)
        if not hand_contains(state.hands[acting_seat], played_card):
            raise AssertionError(f"seat {acting_seat} does not hold card {played_card}")

        # A seat that fails to follow the led suit is publicly void in it (C16).
        if state.current_trick:
            led_suit = suit_of(state.current_trick[0][1])
            if suit_of(played_card) != led_suit:
                state.void_flags[acting_seat][led_suit] = True

        state.hands[acting_seat] = remove_card(state.hands[acting_seat], played_card)
        state.current_trick.append((acting_seat, played_card))

        emitted_events = [
            self._emit(
                state,
                phase=Phase.PLAY,
                actor_seat=acting_seat,
                event_type=EventType.CARD_PLAYED,
                card=played_card,
                trick_no=state.trick_number,
            )
        ]

        if len(state.current_trick) < CARDS_PER_TRICK:
            state.current_player_seat = next_seat(acting_seat)
            return emitted_events

        emitted_events.extend(self._complete_trick(state))
        return emitted_events

    def _complete_trick(self, state: TarneebState) -> list[PublicEvent]:
        """Resolve the finished trick, and the deal if it was the thirteenth."""
        assert state.selected_trump_suit is not None
        winning_seat = self.trick_winner(state.current_trick, state.selected_trump_suit)

        state.trick_winners.append(winning_seat)
        state.completed_tricks.append(list(state.current_trick))
        state.tricks_won[team_of_seat(winning_seat)] += 1

        # The completing CARD_PLAYED is already emitted; TRICK_COMPLETED follows it.
        emitted_events = [
            self._emit(
                state,
                phase=Phase.PLAY,
                actor_seat=None,
                event_type=EventType.TRICK_COMPLETED,
                trick_no=state.trick_number,
                trick_winner_seat=winning_seat,
            )
        ]

        state.current_trick = []
        state.trick_number += 1

        if state.trick_number < TRICKS_PER_DEAL:
            state.leader_seat = winning_seat  # the trick winner leads next (C19)
            state.current_player_seat = winning_seat
            return emitted_events

        emitted_events.append(self._complete_deal(state))
        return emitted_events

    def _complete_deal(self, state: TarneebState) -> PublicEvent:
        """Score the finished deal and emit DEAL_COMPLETED."""
        deal_result = self.score_deal(state)
        state.deal_result = deal_result

        declaring_team = state.declaring_team
        assert declaring_team is not None
        defending_team = 1 - declaring_team
        state.match_score[declaring_team] += deal_result.score_delta_bidding_team
        state.match_score[defending_team] += deal_result.score_delta_defending_team

        state.phase = Phase.SCORE
        state.current_player_seat = None
        state.leader_seat = None

        return self._emit(
            state,
            phase=Phase.SCORE,
            actor_seat=None,
            event_type=EventType.DEAL_COMPLETED,
            trick_no=TRICKS_PER_DEAL,
            deal_result=deal_result,
        )

    def trick_winner(self, trick: list[tuple[int, int]], trump_suit: int) -> int:
        """Return the seat winning ``trick``.

        Highest trump wins; with no trump played, the highest card of the led
        suit wins; a discard of a third suit can never win (C18).
        """
        if len(trick) != CARDS_PER_TRICK:
            raise ValueError(f"a trick has {CARDS_PER_TRICK} cards, got {len(trick)}")

        led_suit = suit_of(trick[0][1])
        winning_seat, winning_card = trick[0]
        for seat, card in trick[1:]:
            if self._beats(card, winning_card, led_suit, trump_suit):
                winning_seat, winning_card = seat, card
        return winning_seat

    @staticmethod
    def _beats(card: int, current_best: int, led_suit: int, trump_suit: int) -> bool:
        """Return whether ``card`` beats ``current_best`` under ``trump_suit``."""
        card_is_trump = suit_of(card) == trump_suit
        best_is_trump = suit_of(current_best) == trump_suit
        if card_is_trump != best_is_trump:
            return card_is_trump  # any trump beats any non-trump
        if card_is_trump:
            return rank_of(card) > rank_of(current_best)
        if suit_of(card) != led_suit:
            return False  # a discard of a third suit can never win
        return suit_of(current_best) == led_suit and rank_of(card) > rank_of(current_best)

    def score_deal(self, state: TarneebState) -> DealResult:
        """Score a played-out deal with the frozen Phase 1 scoring function."""
        if state.current_high_bid is None or state.declaring_team is None:
            raise ValueError("cannot score a deal that has no contract")

        bidding_team_tricks = state.tricks_won[state.declaring_team]
        bidding_team_change, defending_team_change = score_deal(
            state.current_high_bid, bidding_team_tricks, self.rules
        )
        return DealResult(
            bid=state.current_high_bid,
            tricks_bidding_team=bidding_team_tricks,
            score_delta_bidding_team=bidding_team_change,
            score_delta_defending_team=defending_team_change,
        )

    # --- events ------------------------------------------------------------

    def _emit(
        self,
        state: TarneebState,
        *,
        phase: Phase,
        actor_seat: int | None,
        event_type: EventType,
        bid_value: int | None = None,
        trump: int | None = None,
        card: int | None = None,
        trick_no: int = 0,
        trick_winner_seat: int | None = None,
        deal_result: DealResult | None = None,
    ) -> PublicEvent:
        """Append one public event, taking the next contiguous sequence number."""
        event = PublicEvent(
            seq=state.next_event_sequence_number,
            phase=phase,
            actor_seat=actor_seat,
            event_type=event_type,
            bid_value=bid_value,
            trump=trump,
            card=card,
            trick_no=trick_no,
            trick_winner_seat=trick_winner_seat,
            deal_result=deal_result,
        )
        state.events.append(event)
        return event

    # --- predicates --------------------------------------------------------

    def is_auction_complete(self, state: TarneebState) -> bool:
        """True once the auction has produced a declarer."""
        return state.declarer_seat is not None

    def is_redeal_required(self, state: TarneebState) -> bool:
        """True when all four seats passed and the hand must be thrown in.

        A hand thrown in this way is back in phase DEAL with no active bidders
        but still carrying the four PASS events of the auction that failed; a
        never-started deal has an empty event stream.
        """
        return state.phase is Phase.DEAL and not state.active_bidders and bool(state.events)

    def is_auction_phase(self, state: TarneebState) -> bool:
        return state.phase in (Phase.BID, Phase.DECLARE)

    def is_deal_complete(self, state: TarneebState) -> bool:
        """True once all 13 tricks are played and the deal is scored."""
        return state.phase is Phase.SCORE

    def is_match_complete(self, state: TarneebState) -> bool:
        """True once either team has reached the target score (C25)."""
        return max(state.match_score) >= self.rules.target_score
