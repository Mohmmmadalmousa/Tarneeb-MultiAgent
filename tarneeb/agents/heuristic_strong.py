"""Strong counting, contract-aware heuristic -- the main non-neural benchmark.

Implements blueprint section 11.3. It uses only legal player information: its own
hand, the public event stream, public void flags, the bidding record, trump,
contract, declarer, public trick counts and scores, and its own legal mask. It
never sees another seat's cards, and inference from public history is not
leakage.

Trick estimate for a candidate trump suit ``t``:

    T_hat = sum_s [ w_A*[has ace] + w_K*[has king]*g(len_s) + w_Q*[has queen]*g(len_s)
                    + max(0, len_s - 3) * lambda_s ]
            + r_void * n_voids + r_singleton * n_singletons

with ``w_A=1.0``, ``w_K=0.75``, ``w_Q=0.4``, ``g(l)=min(1, (l-1)/2)``,
``lambda_s = 0.6`` when ``s`` is the candidate trump else ``0.25``,
``r_void=1.0``, ``r_singleton=0.5``.

Nothing here is claimed to be optimal Tarneeb. It is a hand-coded baseline
strong enough to make later evaluation informative.
"""

from __future__ import annotations

from tarneeb.agents.base import Agent, checked_action, first_legal_action
from tarneeb.agents.heuristic_basic import _beats
from tarneeb.core.actions import (
    NO_TRUMP_ACTION,
    PASS_ACTION,
    action_for_bid,
    action_for_card,
    action_for_trump_suit,
    card_for_action,
    is_card_action,
)
from tarneeb.core.cards import NUM_RANKS, NUM_SUITS, rank_of, suit_of
from tarneeb.core.events import Phase
from tarneeb.core.state import team_of_seat
from tarneeb.env.public_view import PublicView

ACE_RANK = 12
KING_RANK = 11
QUEEN_RANK = 10

ACE_WEIGHT = 1.0
KING_WEIGHT = 0.75
QUEEN_WEIGHT = 0.4
TRUMP_LENGTH_WEIGHT = 0.6
SIDE_LENGTH_WEIGHT = 0.25
VOID_WEIGHT = 1.0
SINGLETON_WEIGHT = 0.5
TRUMP_LENGTH_PREFERENCE = 0.3

DEFAULT_BID_MARGIN = 1.5
BEHIND_AGGRESSION_BONUS = 1.0
AHEAD_CAUTION_PENALTY = 0.5
NEAR_TARGET_FRACTION = 0.75

DRAW_TRUMP_THRESHOLD = 2
TRICKS_PER_DEAL = 13


def _protection_factor(suit_length: int) -> float:
    """``g(l) = min(1, (l - 1) / 2)``: a king is worth more with length behind it."""
    return min(1.0, max(0.0, (suit_length - 1) / 2.0))


def estimate_tricks(view: PublicView, candidate_trump_suit: int) -> float:
    """Return the expected trick count if ``candidate_trump_suit`` were trump."""
    estimated_tricks = 0.0
    void_count = 0
    singleton_count = 0

    for suit in range(NUM_SUITS):
        suit_cards = view.cards_of_suit_in_own_hand(suit)
        suit_length = len(suit_cards)
        ranks_held = {rank_of(card) for card in suit_cards}

        if suit_length == 0:
            void_count += 1
        elif suit_length == 1:
            singleton_count += 1

        protection = _protection_factor(suit_length)
        if ACE_RANK in ranks_held:
            estimated_tricks += ACE_WEIGHT
        if KING_RANK in ranks_held:
            estimated_tricks += KING_WEIGHT * protection
        if QUEEN_RANK in ranks_held:
            estimated_tricks += QUEEN_WEIGHT * protection

        length_weight = TRUMP_LENGTH_WEIGHT if suit == candidate_trump_suit else SIDE_LENGTH_WEIGHT
        estimated_tricks += max(0, suit_length - 3) * length_weight

    # A void or singleton is only worth ruffing value if we hold trumps.
    if len(view.cards_of_suit_in_own_hand(candidate_trump_suit)) >= 2:
        estimated_tricks += VOID_WEIGHT * void_count + SINGLETON_WEIGHT * singleton_count

    return estimated_tricks


def best_trump_candidate(view: PublicView) -> tuple[int, float]:
    """Return ``(suit, estimate)`` maximising the trick estimate plus a length bonus."""

    def candidate_score(suit: int) -> tuple[float, int, int]:
        suit_length = len(view.cards_of_suit_in_own_hand(suit))
        score = estimate_tricks(view, suit) + TRUMP_LENGTH_PREFERENCE * max(0, suit_length - 5)
        return (score, suit_length, -suit)  # deterministic tie-break

    best_suit = max(range(NUM_SUITS), key=candidate_score)
    return best_suit, estimate_tricks(view, best_suit)


class StrongHeuristicAgent(Agent):
    """Counting, contract-aware baseline. Deterministic given identical input."""

    def __init__(self, bid_margin: float = DEFAULT_BID_MARGIN, name: str = "strong") -> None:
        self.bid_margin = bid_margin
        self.name = name

    def act(self, view: PublicView) -> int:
        if view.phase is Phase.BID:
            selected_action = self._choose_bid(view)
        elif view.phase is Phase.DECLARE:
            selected_action = self._choose_trump(view)
        else:
            selected_action = self._choose_card(view)
        return checked_action(selected_action, view.legal_action_mask, self.name)

    # --- bidding ------------------------------------------------------

    def _bid_margin_for(self, view: PublicView) -> float:
        """Adjust aggression using the public match score."""
        margin = self.bid_margin
        own_score = view.match_score_by_team[view.observer_team]
        opponent_score = view.match_score_by_team[1 - view.observer_team]
        near_target = NEAR_TARGET_FRACTION * view.target_score

        if opponent_score >= near_target and own_score < opponent_score:
            margin += BEHIND_AGGRESSION_BONUS  # must gamble to catch up
        elif own_score >= near_target and own_score > opponent_score:
            margin -= AHEAD_CAUTION_PENALTY  # protect a winning position
        return margin

    def _choose_bid(self, view: PublicView) -> int:
        _best_suit, best_estimate = best_trump_candidate(view)
        target_bid = round(best_estimate + self._bid_margin_for(view))
        target_bid = max(view.min_bid, min(view.max_bid, int(target_bid)))

        if view.current_high_bid is not None and target_bid <= view.current_high_bid:
            return PASS_ACTION

        bid_action = action_for_bid(target_bid)
        if not view.legal_action_mask[bid_action]:
            return PASS_ACTION
        return bid_action

    # --- trump --------------------------------------------------------

    def _choose_trump(self, view: PublicView) -> int:
        best_suit, _estimate = best_trump_candidate(view)
        trump_action = action_for_trump_suit(best_suit)
        # No-trump does not exist under tarneeb-classic-v1; assert rather than assume.
        if view.legal_action_mask[NO_TRUMP_ACTION]:  # pragma: no cover - masked by the engine
            raise AssertionError("no-trump must never be legal under tarneeb-classic-v1")
        if not view.legal_action_mask[trump_action]:  # pragma: no cover
            return first_legal_action(view.legal_action_mask)
        return trump_action

    # --- card play ----------------------------------------------------

    def _choose_card(self, view: PublicView) -> int:
        playable_cards = [
            card_for_action(action_id)
            for action_id in range(len(view.legal_action_mask))
            if view.legal_action_mask[action_id] and is_card_action(action_id)
        ]
        if not playable_cards:  # pragma: no cover - PLAY always offers a card
            return first_legal_action(view.legal_action_mask)

        if not view.current_trick:
            return action_for_card(self._choose_lead(view, playable_cards))

        led_suit = view.led_suit
        assert led_suit is not None
        following_cards = [card for card in playable_cards if suit_of(card) == led_suit]
        if following_cards:
            return action_for_card(self._choose_follow(view, following_cards, led_suit))
        return action_for_card(self._choose_void_play(view, playable_cards, led_suit))

    # --- leading ------------------------------------------------------

    def _choose_lead(self, view: PublicView, playable_cards: list[int]) -> int:
        trump_suit = view.trump_suit

        if trump_suit is not None and self._should_draw_trumps(view):
            own_trumps = [card for card in playable_cards if suit_of(card) == trump_suit]
            if own_trumps:
                return max(own_trumps, key=lambda card: (rank_of(card), card))

        # Cash established side-suit winners: a card no unseen card can beat.
        established_winners = [
            card
            for card in playable_cards
            if suit_of(card) != trump_suit and self._is_established_winner(view, card)
        ]
        if established_winners:
            return max(established_winners, key=lambda card: (rank_of(card), card))

        # Otherwise lead a long side suit the opponents are least able to ruff.
        side_cards = [card for card in playable_cards if suit_of(card) != trump_suit]
        candidates = side_cards or playable_cards
        suit_lengths = view.own_suit_lengths()

        def lead_preference(card: int) -> tuple[int, int, int, int]:
            suit = suit_of(card)
            ruff_risk = sum(
                1 for opponent in view.opponent_seats if view.seat_is_void_in(opponent, suit)
            )
            return (-ruff_risk, suit_lengths[suit], rank_of(card), card)

        return max(candidates, key=lead_preference)

    def _should_draw_trumps(self, view: PublicView) -> bool:
        """As declarer, draw trumps while enough are out and we hold the best one."""
        if not view.observer_is_declarer or view.trump_suit is None:
            return False
        trumps_out = self._unseen_trump_count(view)
        return trumps_out > DRAW_TRUMP_THRESHOLD and self._holds_top_remaining_trump(view)

    def _unseen_trump_count(self, view: PublicView) -> int:
        """Trumps neither in my hand nor already played. Public information."""
        if view.trump_suit is None:
            return 0
        return len(view.unseen_cards_of_suit(view.trump_suit))

    def _holds_top_remaining_trump(self, view: PublicView) -> bool:
        if view.trump_suit is None:
            return False
        own_trumps = view.cards_of_suit_in_own_hand(view.trump_suit)
        if not own_trumps:
            return False
        highest_own = max(rank_of(card) for card in own_trumps)
        unseen_trumps = view.unseen_cards_of_suit(view.trump_suit)
        highest_unseen = max((rank_of(card) for card in unseen_trumps), default=-1)
        return highest_own > highest_unseen

    def _is_established_winner(self, view: PublicView, card: int) -> bool:
        """True when no unseen card of that suit can beat ``card``."""
        suit = suit_of(card)
        unseen_higher = [
            unseen for unseen in view.unseen_cards_of_suit(suit) if rank_of(unseen) > rank_of(card)
        ]
        return not unseen_higher

    # --- following ----------------------------------------------------

    def _choose_follow(self, view: PublicView, following_cards: list[int], led_suit: int) -> int:
        winning_cards = [card for card in following_cards if self._would_currently_win(view, card)]
        lowest_card = min(following_cards, key=lambda card: (rank_of(card), card))

        if not winning_cards:
            return lowest_card

        if self._partner_is_currently_winning(view) and self._partner_win_is_safe(view, led_suit):
            return lowest_card  # do not overtake a partner who is already safe

        if self._should_take_the_trick(view):
            return min(winning_cards, key=lambda card: (rank_of(card), card))
        return lowest_card

    def _choose_void_play(self, view: PublicView, playable_cards: list[int], led_suit: int) -> int:
        """Void in the led suit: ruff when it is worth it, else discard."""
        trump_suit = view.trump_suit
        trump_cards = (
            [card for card in playable_cards if suit_of(card) == trump_suit]
            if trump_suit is not None
            else []
        )

        partner_safe = self._partner_is_currently_winning(view) and self._partner_win_is_safe(
            view, led_suit
        )
        if trump_cards and not partner_safe and self._should_take_the_trick(view):
            ruffing_winners = [
                card for card in trump_cards if self._would_currently_win(view, card)
            ]
            if ruffing_winners:
                return min(ruffing_winners, key=lambda card: (rank_of(card), card))

        return self._choose_discard(view, playable_cards)

    def _choose_discard(self, view: PublicView, playable_cards: list[int]) -> int:
        """Discard from the suit with the fewest likely winners; never bare a king."""
        trump_suit = view.trump_suit
        side_cards = [card for card in playable_cards if suit_of(card) != trump_suit]
        candidates = side_cards or playable_cards
        suit_lengths = view.own_suit_lengths()

        def discard_preference(card: int) -> tuple[int, int, int, int]:
            suit = suit_of(card)
            suit_cards = view.cards_of_suit_in_own_hand(suit)
            likely_winners = sum(
                1 for held in suit_cards if self._is_established_winner(view, held)
            )
            bares_a_king = (
                1
                if suit_lengths[suit] == 2
                and rank_of(card) != KING_RANK
                and any(rank_of(held) == KING_RANK for held in suit_cards)
                else 0
            )
            return (bares_a_king, likely_winners, rank_of(card), card)

        return min(candidates, key=discard_preference)

    # --- contract awareness -------------------------------------------

    def _should_take_the_trick(self, view: PublicView) -> bool:
        """Contract-aware: take tricks the contract still needs, or that beat it."""
        if view.declarer_seat is None or view.current_high_bid is None:
            return True

        declaring_team = team_of_seat(view.declarer_seat)
        tricks_needed = max(0, view.current_high_bid - view.tricks_won_by_team[declaring_team])
        tricks_remaining = view.tricks_remaining

        if view.observer_is_declaring_team:
            # A contract already in hand needs no more honours spent on it.
            return tricks_needed != 0

        # Defending: every trick we take is one the declarer cannot have.
        defenders_needed = tricks_remaining - tricks_needed + 1
        return defenders_needed > 0

    def _partner_is_currently_winning(self, view: PublicView) -> bool:
        if not view.current_trick:
            return False
        led_suit = view.led_suit
        assert led_suit is not None
        best_seat, best_card = view.current_trick[0]
        for seat, card in view.current_trick[1:]:
            if _beats(card, best_card, led_suit, view.trump_suit):
                best_seat, best_card = seat, card
        return best_seat == view.partner_seat

    def _partner_win_is_safe(self, view: PublicView, led_suit: int) -> bool:
        """True when no seat still to play can plausibly beat the partner's card.

        Judged only from public information: the unseen set and public voids.
        """
        led_suit_position = len(view.current_trick)
        seats_still_to_play = [
            (view.observer_seat + offset) % 4
            for offset in range(1, 4 - led_suit_position + 1)
            if offset <= 3 - led_suit_position
        ]
        opponents_to_play = [seat for seat in seats_still_to_play if seat in view.opponent_seats]
        if not opponents_to_play:
            return True

        best_card = view.current_trick[0][1]
        for _seat, card in view.current_trick[1:]:
            if _beats(card, best_card, led_suit, view.trump_suit):
                best_card = card

        for unseen_card in view.unseen_cards:
            if _beats(unseen_card, best_card, led_suit, view.trump_suit):
                # Could any opponent still to play actually hold it?
                unseen_suit = suit_of(unseen_card)
                return not any(
                    not view.seat_is_void_in(opponent, unseen_suit)
                    for opponent in opponents_to_play
                )
        return True

    def _would_currently_win(self, view: PublicView, card: int) -> bool:
        led_suit = view.led_suit
        if led_suit is None:
            return True
        best_card = view.current_trick[0][1]
        for _seat, played_card in view.current_trick[1:]:
            if _beats(played_card, best_card, led_suit, view.trump_suit):
                best_card = played_card
        return _beats(card, best_card, led_suit, view.trump_suit)


assert NUM_RANKS == TRICKS_PER_DEAL  # the rank count and hand size coincide at 13
