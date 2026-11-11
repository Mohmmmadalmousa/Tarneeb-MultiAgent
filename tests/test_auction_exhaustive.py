"""Guardrail G1 -- the disputed auction enumeration, settled by two implementations.

The blueprint claims 1,823 terminal auction sequences and 2,528 decision nodes.
project requirements records an earlier independent count of 1,703 and 2,401.

This file settles it. It contains a reference enumerator written directly from
``docs/RULES.md`` (C06-C11) that does NOT import or call ``TarneebEngine``, and
an exhaustive traversal of the production engine. The two must agree before any
number is frozen into a test constant.

Result: the reference enumerator and the production engine agree on
1,703 / 2,401 / maximum length 10, for every dealer seat. The blueprint's
1,823 / 2,528 could not be reproduced.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tarneeb.core.actions import (
    NO_TRUMP_ACTION,
    PASS_ACTION,
    action_for_bid,
    bid_for_action,
    is_bid_action,
)
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import NUM_SEATS, EventType, Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState

# Proven by the two independent implementations below. Do not edit these
# without rerunning both and explaining the change.
PROVEN_TERMINAL_AUCTION_COUNT = 1703
PROVEN_DECISION_NODE_COUNT = 2401
PROVEN_MAXIMUM_AUCTION_LENGTH = 10
PROVEN_MAXIMUM_PER_SEAT_AUCTION_SPAN = 10

# The two disputed claims, kept here so the test names them explicitly.
BLUEPRINT_CLAIMED_TERMINAL_COUNT = 1823
BLUEPRINT_CLAIMED_DECISION_NODE_COUNT = 2528

LOWEST_BID = 7
HIGHEST_BID = 13

# One auction action: (acting_seat, "PASS", None) or (acting_seat, "BID", value).
# A uniform, sortable shape so the two enumerations can be compared directly.
AuctionAction = tuple[int, str, int | None]
AuctionSequence = tuple[AuctionAction, ...]

PASS_LABEL = "PASS"
BID_LABEL = "BID"


# ---------------------------------------------------------------------------
# Reference enumerator -- written from docs/RULES.md, independent of the engine
# ---------------------------------------------------------------------------


class ReferenceAuctionEnumerator:
    """An exhaustive enumerator of every legal auction under C06-C11.

    Deliberately written in a different style from ``TarneebEngine`` and using
    none of its code, so that agreement between the two is real evidence rather
    than a shared bug.
    """

    def __init__(self, dealer_seat: int) -> None:
        self.dealer_seat = dealer_seat
        self.first_bidder_seat = (dealer_seat + 1) % NUM_SEATS
        self.terminal_sequences: list[AuctionSequence] = []
        self.decision_node_count = 0

    def enumerate(self) -> None:
        self._walk(
            current_player_seat=self.first_bidder_seat,
            active_bidders=frozenset(range(NUM_SEATS)),
            current_high_bid=None,
            actions_so_far=(),
        )

    def _next_active_bidder(self, current_player_seat: int, active_bidders: frozenset[int]) -> int:
        candidate_seat = (current_player_seat + 1) % NUM_SEATS
        while candidate_seat not in active_bidders:
            candidate_seat = (candidate_seat + 1) % NUM_SEATS
        return candidate_seat

    def _walk(
        self,
        current_player_seat: int,
        active_bidders: frozenset[int],
        current_high_bid: int | None,
        actions_so_far: AuctionSequence,
    ) -> None:
        # C10: everybody passed and nobody bid -> the auction is over (redeal).
        if not active_bidders:
            self.terminal_sequences.append(actions_so_far)
            return
        # C11: one active bidder left and a bid exists -> that seat is declarer
        # and does not act again.
        if len(active_bidders) == 1 and current_high_bid is not None:
            self.terminal_sequences.append(actions_so_far)
            return

        self.decision_node_count += 1

        # PASS is always available to the seat on turn, and is permanent (C09).
        remaining_bidders = active_bidders - {current_player_seat}
        next_seat_after_pass = (
            self._next_active_bidder(current_player_seat, remaining_bidders)
            if remaining_bidders
            else current_player_seat
        )
        self._walk(
            current_player_seat=next_seat_after_pass,
            active_bidders=remaining_bidders,
            current_high_bid=current_high_bid,
            actions_so_far=(*actions_so_far, (current_player_seat, PASS_LABEL, None)),
        )

        # C07/C08: any bid strictly above the current high bid, up to 13.
        lowest_legal_bid = LOWEST_BID if current_high_bid is None else current_high_bid + 1
        for bid_value in range(lowest_legal_bid, HIGHEST_BID + 1):
            self._walk(
                current_player_seat=self._next_active_bidder(current_player_seat, active_bidders),
                active_bidders=active_bidders,
                current_high_bid=bid_value,
                actions_so_far=(*actions_so_far, (current_player_seat, BID_LABEL, bid_value)),
            )


def reference_enumeration(dealer_seat: int) -> ReferenceAuctionEnumerator:
    enumerator = ReferenceAuctionEnumerator(dealer_seat)
    enumerator.enumerate()
    return enumerator


# ---------------------------------------------------------------------------
# Exhaustive traversal of the production engine
# ---------------------------------------------------------------------------


class ProductionAuctionTraversal:
    """Walks every legal auction by driving ``TarneebEngine`` itself."""

    def __init__(self, dealer_seat: int) -> None:
        self.dealer_seat = dealer_seat
        self.engine = TarneebEngine(RulesConfig())
        self.terminal_sequences: list[AuctionSequence] = []
        self.decision_node_count = 0
        self.terminal_states: list[TarneebState] = []

    def traverse(self) -> None:
        root_state = self.engine.deal(self.engine.new_deal(self.dealer_seat))
        self._walk(root_state, ())

    def _walk(self, state: TarneebState, actions_so_far: AuctionSequence) -> None:
        if state.phase is not Phase.BID:
            self.terminal_sequences.append(actions_so_far)
            self.terminal_states.append(state)
            return

        self.decision_node_count += 1
        acting_seat = state.current_player_seat
        assert acting_seat is not None

        legal_action_mask = self.engine.legal_actions(state)
        for action_id in np.flatnonzero(legal_action_mask).tolist():
            child_state = state.clone()
            self.engine.step(child_state, action_id)
            recorded_action: AuctionAction = (
                (acting_seat, BID_LABEL, bid_for_action(action_id))
                if is_bid_action(action_id)
                else (acting_seat, PASS_LABEL, None)
            )
            self._walk(child_state, (*actions_so_far, recorded_action))


def production_traversal(dealer_seat: int) -> ProductionAuctionTraversal:
    traversal = ProductionAuctionTraversal(dealer_seat)
    traversal.traverse()
    return traversal


# ---------------------------------------------------------------------------
# G1: the two implementations must agree before anything is frozen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dealer_seat", range(NUM_SEATS))
def test_reference_and_production_enumerations_agree(dealer_seat: int) -> None:
    reference = reference_enumeration(dealer_seat)
    production = production_traversal(dealer_seat)

    assert sorted(reference.terminal_sequences) == sorted(production.terminal_sequences)
    assert len(reference.terminal_sequences) == len(production.terminal_sequences)
    assert reference.decision_node_count == production.decision_node_count


@pytest.mark.parametrize("dealer_seat", range(NUM_SEATS))
def test_proven_terminal_auction_sequence_count(dealer_seat: int) -> None:
    reference = reference_enumeration(dealer_seat)
    production = production_traversal(dealer_seat)
    assert len(reference.terminal_sequences) == PROVEN_TERMINAL_AUCTION_COUNT
    assert len(production.terminal_sequences) == PROVEN_TERMINAL_AUCTION_COUNT


@pytest.mark.parametrize("dealer_seat", range(NUM_SEATS))
def test_proven_decision_node_count(dealer_seat: int) -> None:
    reference = reference_enumeration(dealer_seat)
    production = production_traversal(dealer_seat)
    assert reference.decision_node_count == PROVEN_DECISION_NODE_COUNT
    assert production.decision_node_count == PROVEN_DECISION_NODE_COUNT


@pytest.mark.parametrize("dealer_seat", range(NUM_SEATS))
def test_proven_maximum_auction_length(dealer_seat: int) -> None:
    reference = reference_enumeration(dealer_seat)
    production = production_traversal(dealer_seat)
    reference_maximum = max(len(sequence) for sequence in reference.terminal_sequences)
    production_maximum = max(len(sequence) for sequence in production.terminal_sequences)
    assert reference_maximum == production_maximum == PROVEN_MAXIMUM_AUCTION_LENGTH


def test_the_blueprint_counts_are_not_reproducible() -> None:
    """Documents that G1 resolved against the blueprint, not silently for it."""
    reference = reference_enumeration(0)
    assert len(reference.terminal_sequences) != BLUEPRINT_CLAIMED_TERMINAL_COUNT
    assert reference.decision_node_count != BLUEPRINT_CLAIMED_DECISION_NODE_COUNT
    assert len(reference.terminal_sequences) == PROVEN_TERMINAL_AUCTION_COUNT
    assert reference.decision_node_count == PROVEN_DECISION_NODE_COUNT


def test_maximum_length_auction_is_seven_bids_and_three_passes() -> None:
    """The longest auction is the full bid ladder plus the three closing passes."""
    reference = reference_enumeration(0)
    longest_sequences = [
        sequence
        for sequence in reference.terminal_sequences
        if len(sequence) == PROVEN_MAXIMUM_AUCTION_LENGTH
    ]
    assert longest_sequences
    for sequence in longest_sequences:
        bid_values = [value for _, label, value in sequence if label == BID_LABEL]
        assert bid_values == list(range(LOWEST_BID, HIGHEST_BID + 1))
        pass_count = sum(1 for _, label, _ in sequence if label == PASS_LABEL)
        assert pass_count == 3


# --- per-seat auction span (the auction half of T-E9) -----------------------


def _maximum_per_seat_auction_span(sequences: list[AuctionSequence]) -> int:
    """Largest count of (a seat's last auction action + every action after it)."""
    maximum_span = 0
    for sequence in sequences:
        for seat in range(NUM_SEATS):
            action_indices = [
                index for index, (actor, _, _) in enumerate(sequence) if actor == seat
            ]
            if action_indices:
                span = len(sequence) - action_indices[-1]
                maximum_span = max(maximum_span, span)
    return maximum_span


@pytest.mark.parametrize("dealer_seat", range(NUM_SEATS))
def test_proven_maximum_per_seat_auction_span(dealer_seat: int) -> None:
    reference = reference_enumeration(dealer_seat)
    production = production_traversal(dealer_seat)
    assert (
        _maximum_per_seat_auction_span(reference.terminal_sequences)
        == _maximum_per_seat_auction_span(production.terminal_sequences)
        == PROVEN_MAXIMUM_PER_SEAT_AUCTION_SPAN
    )


# --- properties that must hold across every enumerated auction --------------


def test_every_terminal_auction_ends_in_declare_or_redeal() -> None:
    production = production_traversal(0)
    for state in production.terminal_states:
        if state.phase is Phase.DECLARE:
            assert state.declarer_seat is not None
            assert state.current_high_bid is not None
            assert state.active_bidders == {state.declarer_seat}
        else:
            assert state.phase is Phase.DEAL
            assert state.active_bidders == set()
            assert state.current_high_bid is None


def test_exactly_one_all_pass_auction_exists() -> None:
    reference = reference_enumeration(0)
    all_pass_sequences = [
        sequence
        for sequence in reference.terminal_sequences
        if all(label == PASS_LABEL for _, label, _ in sequence)
    ]
    assert len(all_pass_sequences) == 1
    assert len(all_pass_sequences[0]) == NUM_SEATS


def test_no_auction_contains_a_non_increasing_bid() -> None:
    reference = reference_enumeration(0)
    for sequence in reference.terminal_sequences:
        bid_values = [value for _, label, value in sequence if label == BID_LABEL]
        assert bid_values == sorted(set(bid_values))
        assert all(value is not None and LOWEST_BID <= value <= HIGHEST_BID for value in bid_values)


def test_no_seat_acts_after_passing() -> None:
    reference = reference_enumeration(0)
    for sequence in reference.terminal_sequences:
        seats_that_passed: set[int] = set()
        for acting_seat, label, _ in sequence:
            assert acting_seat not in seats_that_passed
            if label == PASS_LABEL:
                seats_that_passed.add(acting_seat)


def test_production_never_offers_no_trump_or_card_actions_during_the_auction() -> None:
    production = production_traversal(0)
    engine = TarneebEngine(RulesConfig())
    root_state = engine.deal(engine.new_deal(0))
    stack: list[TarneebState] = [root_state]
    while stack:
        state = stack.pop()
        if state.phase is not Phase.BID:
            continue
        legal_action_mask = engine.legal_actions(state)
        legal_action_ids = set(np.flatnonzero(legal_action_mask).tolist())
        assert NO_TRUMP_ACTION not in legal_action_ids
        assert legal_action_ids <= set(range(52, 60))
        assert PASS_ACTION in legal_action_ids
        for action_id in legal_action_ids:
            child_state = state.clone()
            engine.step(child_state, action_id)
            stack.append(child_state)
    assert production.decision_node_count == PROVEN_DECISION_NODE_COUNT


def test_event_stream_matches_the_executed_auction_actions() -> None:
    """Every enumerated auction's BID/PASS events reproduce its action list."""
    engine = TarneebEngine(RulesConfig())
    checked_auction_count = 0
    for dealer_seat in range(NUM_SEATS):
        production = production_traversal(dealer_seat)
        for auction_sequence, terminal_state in zip(
            production.terminal_sequences, production.terminal_states, strict=True
        ):
            auction_events = [
                event
                for event in terminal_state.events
                if event.event_type in (EventType.BID, EventType.PASS)
            ]
            assert len(auction_events) == len(auction_sequence)
            for event, (acting_seat, label, bid_value) in zip(
                auction_events, auction_sequence, strict=True
            ):
                assert event.actor_seat == acting_seat
                if label == PASS_LABEL:
                    assert event.event_type is EventType.PASS
                    assert event.bid_value is None
                else:
                    assert event.event_type is EventType.BID
                    assert event.bid_value == bid_value
            assert [event.seq for event in terminal_state.events] == list(
                range(len(terminal_state.events))
            )
            checked_auction_count += 1
    assert checked_auction_count == PROVEN_TERMINAL_AUCTION_COUNT * NUM_SEATS
    assert engine.rules.version == "tarneeb-classic-v1"


def test_every_bid_action_id_appears_somewhere_in_the_enumeration() -> None:
    reference = reference_enumeration(0)
    observed_bid_action_ids = {
        action_for_bid(value)
        for sequence in reference.terminal_sequences
        for _, label, value in sequence
        if label == BID_LABEL and value is not None
    }
    assert observed_bid_action_ids == set(range(52, 59))


def test_all_dealer_seats_produce_the_same_counts() -> None:
    counts_per_dealer = {
        dealer_seat: (
            len(reference_enumeration(dealer_seat).terminal_sequences),
            reference_enumeration(dealer_seat).decision_node_count,
        )
        for dealer_seat in range(NUM_SEATS)
    }
    assert set(counts_per_dealer.values()) == {
        (PROVEN_TERMINAL_AUCTION_COUNT, PROVEN_DECISION_NODE_COUNT)
    }


def test_sequences_are_all_distinct() -> None:
    reference = reference_enumeration(0)
    assert len(set(reference.terminal_sequences)) == PROVEN_TERMINAL_AUCTION_COUNT


def test_auction_length_distribution_is_consistent_between_implementations() -> None:
    reference = reference_enumeration(0)
    production = production_traversal(0)
    reference_lengths = sorted(len(sequence) for sequence in reference.terminal_sequences)
    production_lengths = sorted(len(sequence) for sequence in production.terminal_sequences)
    assert reference_lengths == production_lengths
    assert min(reference_lengths) == NUM_SEATS  # the all-pass auction
    assert max(reference_lengths) == PROVEN_MAXIMUM_AUCTION_LENGTH


def test_the_documented_ten_action_witness_is_a_real_auction() -> None:
    """The blueprint's worked example is reachable and has span 10 for seat 0."""
    engine = TarneebEngine(RulesConfig())
    state = engine.deal(engine.new_deal(3))  # dealer 3 -> first bidder 0
    assert state.current_player_seat == 0

    engine.step(state, PASS_ACTION)  # seat 0 passes
    for bid_value in range(7, 14):
        engine.step(state, action_for_bid(bid_value))  # seats 1,2,3,1,2,3,1
    engine.step(state, PASS_ACTION)  # seat 2
    engine.step(state, PASS_ACTION)  # seat 3

    assert state.phase is Phase.DECLARE
    assert state.declarer_seat == 1
    auction_events = [
        event for event in state.events if event.event_type in (EventType.BID, EventType.PASS)
    ]
    assert len(auction_events) == PROVEN_MAXIMUM_AUCTION_LENGTH
    seat_zero_indices = [
        index for index, event in enumerate(auction_events) if event.actor_seat == 0
    ]
    assert len(auction_events) - seat_zero_indices[-1] == PROVEN_MAXIMUM_PER_SEAT_AUCTION_SPAN


def test_enumeration_is_deterministic() -> None:
    first_run = reference_enumeration(0)
    second_run = reference_enumeration(0)
    assert first_run.terminal_sequences == second_run.terminal_sequences
    assert first_run.decision_node_count == second_run.decision_node_count


def test_no_sequence_repeats_a_bid_value() -> None:
    reference = reference_enumeration(0)
    for sequence in reference.terminal_sequences:
        bid_values = [value for _, label, value in sequence if label == BID_LABEL]
        assert len(bid_values) == len(set(bid_values))


def test_pairs_of_distinct_auctions_have_distinct_event_streams() -> None:
    """Ordered history must distinguish auctions that differ in any way."""
    production = production_traversal(0)
    event_signatures = {
        tuple(
            (event.seq, event.event_type, event.actor_seat, event.bid_value)
            for event in state.events
        )
        for state in production.terminal_states
    }
    assert len(event_signatures) == PROVEN_TERMINAL_AUCTION_COUNT


def test_no_two_seats_share_a_decision_at_the_same_moment() -> None:
    production = production_traversal(0)
    for state in production.terminal_states:
        auction_events = [
            event for event in state.events if event.event_type in (EventType.BID, EventType.PASS)
        ]
        for earlier_event, later_event in itertools.pairwise(auction_events):
            assert earlier_event.seq + 1 == later_event.seq
