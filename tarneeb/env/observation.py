"""Observation construction for ``obs_v2``.

Three views of the same state, with strictly different visibility:

- :meth:`ObservationBuilder.static_actor` -- 692 floats, what a policy sees. It is
  invariant to the other three hands: two states that differ only in how the
  unseen cards are distributed among opponents must produce byte-identical
  vectors (invariant I7).
- :meth:`ObservationBuilder.static_critic` -- 856 floats, the actor block plus the
  three hidden hands and derived counts. Training only, never loaded at inference.
- :meth:`ObservationBuilder.ui_view` -- a JSON-safe dictionary for a CLI or viewer,
  carrying no hidden hand.

Every seat-indexed feature is relative to the observer, matching
:func:`tarneeb.core.events.relativize`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tarneeb.core.actions import NUM_ACTIONS
from tarneeb.core.cards import (
    NUM_CARDS,
    NUM_SEATS,
    NUM_SUITS,
    card_name,
    cards_in_hand,
    cards_of_suit_in_hand,
    hand_size,
    rank_of,
    suit_of,
)
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.state import TRICKS_PER_DEAL, TarneebState, team_of_seat
from tarneeb.env.event_features import (
    EVENT_ID_DTYPE,
    event_ids,
)
from tarneeb.env.spaces import (
    ACTOR_FEATURE_SLICES,
    CRITIC_FEATURE_SLICES,
    D_CRITIC,
    D_STATIC,
    L_MAX,
    MATCH_SCORE_CLIP,
    N_EVENT_FIELDS,
    OBS_SPEC_VERSION,
    RELATIVE_PARTNER,
    RELATIVE_SELF,
    absolute_seat,
    relative_seat,
)

LOWEST_BID = 7
HIGH_CARD_LOWEST_RANK = 9
"""Jack and above count as high cards for the critic's X03 summary."""

OBSERVED_PHASES = (Phase.BID, Phase.DECLARE, Phase.PLAY)


class ObservationBuilder:
    """Builds actor, critic and UI views of a deal state.

    The builder holds no per-deal state of its own, so one instance can serve
    many environments.
    """

    OBS_SPEC_VERSION: str = OBS_SPEC_VERSION
    D_STATIC: int = D_STATIC
    D_CRITIC: int = D_CRITIC
    L_MAX: int = L_MAX
    N_EVENT_FIELDS: int = N_EVENT_FIELDS

    def __init__(self, engine: TarneebEngine) -> None:
        self.engine = engine

    # --- public entry points ------------------------------------------------

    def static_actor(self, state: TarneebState, observer_seat: int) -> np.ndarray:
        """Return the 692-dim actor-safe observation for ``observer_seat``."""
        observation = np.zeros(D_STATIC, dtype=np.float32)
        self._fill_actor_features(observation, ACTOR_FEATURE_SLICES, state, observer_seat)
        return observation

    def static_critic(self, state: TarneebState, observer_seat: int) -> np.ndarray:
        """Return the 856-dim critic observation for ``observer_seat``.

        Training only. This vector contains the other three hands and must never
        reach an actor network.
        """
        observation = np.zeros(D_CRITIC, dtype=np.float32)
        self._fill_actor_features(observation, CRITIC_FEATURE_SLICES, state, observer_seat)
        self._fill_critic_only_features(observation, state, observer_seat)
        return observation

    def event_ids(self, event: Any, observer_seat: int) -> np.ndarray:
        """Return the seven factored ids of one public event."""
        return event_ids(event, observer_seat)

    def padded_history(
        self, state: TarneebState, observer_seat: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(event_ids (80, 7) int16, valid_mask (80,) bool)``.

        Events keep their chronological order and are right-padded, so index 0 is
        always the first event of the deal. Padded rows are all zeros, which is
        event type PAD with "none" in every other field.

        Raises:
            ValueError: If the deal somehow produced more than ``L_MAX`` events.
                Silently dropping history would corrupt the auction record, so
                this fails loudly instead.
        """
        history_length = len(state.events)
        if history_length > L_MAX:
            raise ValueError(f"deal produced {history_length} events, which exceeds L_MAX={L_MAX}")

        padded_event_history = np.zeros((L_MAX, N_EVENT_FIELDS), dtype=EVENT_ID_DTYPE)
        valid_mask = np.zeros(L_MAX, dtype=bool)
        for position, event in enumerate(state.events):
            padded_event_history[position] = event_ids(event, observer_seat)
            valid_mask[position] = True
        return padded_event_history, valid_mask

    def ui_view(self, state: TarneebState, observer_seat: int) -> dict[str, Any]:
        """Return a JSON-safe human-readable view for ``observer_seat``.

        Carries the observer's own hand and public state only. No other seat's
        hand, and no unseen-card ownership, is representable here.
        """
        # As in S05, only the observer's own legal actions may be shown: during
        # PLAY the mask depends on the acting seat's hand.
        legal_action_mask = (
            self._legal_action_mask(state) if state.current_player_seat == observer_seat else None
        )
        return {
            "obs_spec_version": OBS_SPEC_VERSION,
            "rules_version": state.rules.version,
            "observer_seat": observer_seat,
            "phase": state.phase.name,
            "acting_seat": state.current_player_seat,
            "acting_relative_seat": (
                None
                if state.current_player_seat is None
                else relative_seat(state.current_player_seat, observer_seat)
            ),
            "dealer_seat": state.dealer_seat,
            "own_hand": [card_name(card) for card in cards_in_hand(state.hands[observer_seat])],
            "current_high_bid": state.current_high_bid,
            "current_high_bidder": state.current_high_bidder,
            "declarer_seat": state.declarer_seat,
            "trump_suit": state.selected_trump_suit,
            "leader_seat": state.leader_seat,
            "trick_number": state.trick_number,
            "current_trick": [
                {"seat": seat, "card": card_name(card)} for seat, card in state.current_trick
            ],
            "completed_tricks": [
                {
                    "trick_no": trick_index,
                    "cards": [{"seat": seat, "card": card_name(card)} for seat, card in trick],
                    "winner_seat": state.trick_winners[trick_index],
                }
                for trick_index, trick in enumerate(state.completed_tricks)
            ],
            "tricks_won_by_team": list(state.tricks_won),
            "match_score_by_team": list(state.match_score),
            "void_flags_by_seat": [list(row) for row in state.void_flags],
            "legal_actions": (
                [] if legal_action_mask is None else np.flatnonzero(legal_action_mask).tolist()
            ),
            "events": [
                {
                    "seq": event.seq,
                    "type": event.event_type.name,
                    "actor_seat": event.actor_seat,
                    "bid_value": event.bid_value,
                    "trump": event.trump,
                    "card": None if event.card is None else card_name(event.card),
                    "trick_no": event.trick_no,
                    "trick_winner_seat": event.trick_winner_seat,
                }
                for event in state.events
            ],
            "deal_result": (
                None
                if state.deal_result is None
                else {
                    "bid": state.deal_result.bid,
                    "tricks_bidding_team": state.deal_result.tricks_bidding_team,
                    "score_delta_bidding_team": state.deal_result.score_delta_bidding_team,
                    "score_delta_defending_team": state.deal_result.score_delta_defending_team,
                }
            ),
        }

    # --- internals ----------------------------------------------------------

    def _legal_action_mask(self, state: TarneebState) -> np.ndarray | None:
        """Return the engine's mask, or ``None`` when no seat is on turn.

        Legality is never recomputed here: it always comes from the engine.
        """
        if state.phase not in OBSERVED_PHASES:
            return None
        return self.engine.legal_actions(state)

    def _fill_actor_features(
        self,
        observation: np.ndarray,
        slices: dict[str, slice],
        state: TarneebState,
        observer_seat: int,
    ) -> None:
        if not 0 <= observer_seat < NUM_SEATS:
            raise ValueError(f"observer_seat must be in 0..{NUM_SEATS - 1}, got {observer_seat}")

        own_hand = state.hands[observer_seat]
        observer_team = team_of_seat(observer_seat)

        # S01 own hand
        own_hand_block = observation[slices["own_hand"]]
        for card in cards_in_hand(own_hand):
            own_hand_block[card] = 1.0

        # S02 cards played, by relative seat
        played_block = observation[slices["played_cards_by_relative_seat"]]
        publicly_played_cards: set[int] = set()
        for trick in state.completed_tricks:
            for seat, card in trick:
                played_block[relative_seat(seat, observer_seat) * NUM_CARDS + card] = 1.0
                publicly_played_cards.add(card)
        for seat, card in state.current_trick:
            played_block[relative_seat(seat, observer_seat) * NUM_CARDS + card] = 1.0
            publicly_played_cards.add(card)

        # S03 current trick, by relative seat
        current_trick_block = observation[slices["current_trick_by_relative_seat"]]
        for seat, card in state.current_trick:
            current_trick_block[relative_seat(seat, observer_seat) * NUM_CARDS + card] = 1.0

        # S04 unseen-card mask: not mine and not yet public. Never says who holds it.
        unseen_block = observation[slices["unseen_card_mask"]]
        own_cards = set(cards_in_hand(own_hand))
        for card in range(NUM_CARDS):
            if card not in own_cards and card not in publicly_played_cards:
                unseen_block[card] = 1.0

        # S05 legal-action mask, taken straight from the engine.
        #
        # Only ever the OBSERVER's own mask. During PLAY the engine's mask is a
        # function of the acting seat's hand, so writing it for an observer who
        # is not on turn would leak that hand (invariant I7). A seat that is not
        # on turn has no legal actions, so the block stays all zeros.
        if state.current_player_seat == observer_seat:
            legal_action_mask = self._legal_action_mask(state)
            if legal_action_mask is not None:
                observation[slices["legal_action_mask"]] = legal_action_mask.astype(np.float32)

        # S06 void matrix, relative seat x suit
        void_block = observation[slices["void_matrix"]]
        for seat in range(NUM_SEATS):
            relative = relative_seat(seat, observer_seat)
            for suit in range(NUM_SUITS):
                if state.void_flags[seat][suit]:
                    void_block[relative * NUM_SUITS + suit] = 1.0

        # S07 play order in the current trick, plus an "am I leading" bit
        play_order_block = observation[slices["play_order_in_trick"]]
        if state.phase is Phase.PLAY:
            position_in_trick = len(state.current_trick)
            play_order_block[position_in_trick] = 1.0
            if position_in_trick == 0:
                play_order_block[NUM_SEATS] = 1.0

        # S08 led suit (or none)
        led_suit_block = observation[slices["led_suit"]]
        if state.current_trick:
            led_suit_block[1 + suit_of(state.current_trick[0][1])] = 1.0
        else:
            led_suit_block[0] = 1.0

        # S09 current high bid (or none)
        high_bid_block = observation[slices["current_high_bid"]]
        high_bid_block[
            0 if state.current_high_bid is None else state.current_high_bid - LOWEST_BID + 1
        ] = 1.0

        # S10 high bidder, relative (or none)
        high_bidder_block = observation[slices["high_bidder_relative"]]
        high_bidder_block[
            0
            if state.current_high_bidder is None
            else 1 + relative_seat(state.current_high_bidder, observer_seat)
        ] = 1.0

        # S11 pass status, relative
        pass_status_block = observation[slices["pass_status_relative"]]
        if state.phase is Phase.BID or state.active_bidders:
            for seat in range(NUM_SEATS):
                if seat not in state.active_bidders:
                    pass_status_block[relative_seat(seat, observer_seat)] = 1.0

        # S12 trump: none / four suits / reserved no-trump (always zero here)
        trump_block = observation[slices["trump"]]
        trump_block[0 if state.selected_trump_suit is None else 1 + state.selected_trump_suit] = 1.0

        # S13 contract (none, 7..13)
        contract_block = observation[slices["contract"]]
        contract_value = state.current_high_bid if state.declarer_seat is not None else None
        contract_block[0 if contract_value is None else contract_value - LOWEST_BID + 1] = 1.0

        # S14 declarer role: self / partner / opponent / none
        declarer_role_block = observation[slices["declarer_role"]]
        if state.declarer_seat is None:
            declarer_role_block[3] = 1.0
        else:
            declarer_relative = relative_seat(state.declarer_seat, observer_seat)
            if declarer_relative == RELATIVE_SELF:
                declarer_role_block[0] = 1.0
            elif declarer_relative == RELATIVE_PARTNER:
                declarer_role_block[1] = 1.0
            else:
                declarer_role_block[2] = 1.0

        # S15 tricks won, one-hot for my team then their team
        tricks_block = observation[slices["tricks_won_one_hot"]]
        my_team_tricks = state.tricks_won[observer_team]
        their_team_tricks = state.tricks_won[1 - observer_team]
        tricks_block[my_team_tricks] = 1.0
        tricks_block[(TRICKS_PER_DEAL + 1) + their_team_tricks] = 1.0

        # S16 tricks remaining
        tricks_played = len(state.completed_tricks)
        observation[slices["tricks_remaining"]][0] = (
            TRICKS_PER_DEAL - tricks_played
        ) / TRICKS_PER_DEAL

        # S17 tricks still needed by the declaring side
        still_needed_block = observation[slices["tricks_still_needed"]]
        if state.declarer_seat is not None and state.current_high_bid is not None:
            declaring_team = team_of_seat(state.declarer_seat)
            still_needed = max(0, state.current_high_bid - state.tricks_won[declaring_team])
            still_needed_block[0] = still_needed / TRICKS_PER_DEAL

        # S18 match scores (mine, theirs), scaled by the target and clipped
        target_score = state.rules.target_score
        match_score_block = observation[slices["match_scores"]]
        match_score_block[0] = np.clip(
            state.match_score[observer_team] / target_score, -MATCH_SCORE_CLIP, MATCH_SCORE_CLIP
        )
        match_score_block[1] = np.clip(
            state.match_score[1 - observer_team] / target_score,
            -MATCH_SCORE_CLIP,
            MATCH_SCORE_CLIP,
        )

        # S19 score gap
        observation[slices["score_gap"]][0] = np.clip(
            (state.match_score[observer_team] - state.match_score[1 - observer_team])
            / target_score,
            -MATCH_SCORE_CLIP,
            MATCH_SCORE_CLIP,
        )

        # S20 dealer, relative
        observation[slices["dealer_relative"]][relative_seat(state.dealer_seat, observer_seat)] = (
            1.0
        )

        # S21 leader, relative
        if state.leader_seat is not None:
            observation[slices["leader_relative"]][
                relative_seat(state.leader_seat, observer_seat)
            ] = 1.0

        # S22 phase
        phase_block = observation[slices["phase"]]
        if state.phase in OBSERVED_PHASES:
            phase_block[OBSERVED_PHASES.index(state.phase)] = 1.0

        # S23 trumps unaccounted for: trump cards neither mine nor already public
        trumps_unaccounted = 0
        if state.selected_trump_suit is not None:
            own_trumps = hand_size(cards_of_suit_in_hand(own_hand, state.selected_trump_suit))
            played_trumps = sum(
                1 for card in publicly_played_cards if suit_of(card) == state.selected_trump_suit
            )
            trumps_unaccounted = TRICKS_PER_DEAL - own_trumps - played_trumps
        observation[slices["trumps_unaccounted_for"]][0] = trumps_unaccounted / TRICKS_PER_DEAL

        # S24 trick number
        observation[slices["trick_number"]][0] = state.trick_number / TRICKS_PER_DEAL

    def _fill_critic_only_features(
        self, observation: np.ndarray, state: TarneebState, observer_seat: int
    ) -> None:
        """Fill X01-X03. Training only -- these expose the other three hands."""
        hidden_hands_block = observation[CRITIC_FEATURE_SLICES["hidden_hands_by_relative_seat"]]
        trump_count_block = observation[CRITIC_FEATURE_SLICES["trump_count_by_relative_seat"]]
        high_card_count_block = observation[
            CRITIC_FEATURE_SLICES["high_card_count_by_relative_seat"]
        ]

        for relative in range(NUM_SEATS):
            seat = absolute_seat(relative, observer_seat)
            hand = state.hands[seat]

            # X01 covers the three OTHER hands; relative 0 is the observer's own
            # hand, already present in S01, so it is not repeated here.
            if relative != RELATIVE_SELF:
                block_start = (relative - 1) * NUM_CARDS
                for card in cards_in_hand(hand):
                    hidden_hands_block[block_start + card] = 1.0

            # X02 trump count per relative seat
            if state.selected_trump_suit is not None:
                trump_count_block[relative] = (
                    hand_size(cards_of_suit_in_hand(hand, state.selected_trump_suit))
                    / TRICKS_PER_DEAL
                )

            # X03 high-card count per relative seat
            high_card_count_block[relative] = (
                sum(1 for card in cards_in_hand(hand) if rank_of(card) >= HIGH_CARD_LOWEST_RANK)
                / TRICKS_PER_DEAL
            )


def empty_legal_action_mask() -> np.ndarray:
    """Return an all-False 65-entry mask, for states with no seat on turn."""
    return np.zeros(NUM_ACTIONS, dtype=bool)
