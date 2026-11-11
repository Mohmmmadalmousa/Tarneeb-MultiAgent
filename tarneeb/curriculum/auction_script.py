"""A deterministic auction, driven through the frozen engine.

Stages 2 to 5 want the policy to learn card play without also having to learn to
bid. The wrong way to do that would be to invent a "no auction" mode inside the
engine. The right way -- and the one taken here -- is to keep the real auction and
simply decide its outcome in advance: one seat bids the chosen contract, the other
three pass, the winner names the chosen trump.

Every one of those actions goes through ``TarneebEngine.step``, so the deal is a
completely ordinary legal deal. It scores normally, emits a normal ``event_v1``
stream, and replays under ``deal_v2``. The only thing that changed is who decided
the bids.

The script is a **pure function** of ``(match_seed, deal_index, dealer_seat)``, so
training and evaluation derive the same auction for the same deal, and both
orientations of a duplicate-deal pair get an identical auction -- which is what
lets a paired comparison isolate card play.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tarneeb.core.actions import (
    NUM_SUITS,
    PASS_ACTION,
    TRUMP_ACTION_START,
    action_for_bid,
)
from tarneeb.core.cards import NUM_SEATS, derive_deal_seed
from tarneeb.core.events import Phase
from tarneeb.curriculum.stages import SAMPLED, StageSpec

AUCTION_SCRIPT_SALT = 0x7A17
"""Keeps the script's stream distinct from the dealing stream of the same deal."""

MAX_SCRIPTED_AUCTION_ACTIONS = 8
"""Four auction actions plus the trump declaration, with headroom. A safety stop."""


@dataclass(frozen=True, slots=True)
class AuctionScript:
    """The predetermined outcome of one deal's auction."""

    declarer_seat: int
    contract: int
    trump_suit: int

    def __post_init__(self) -> None:
        if not 0 <= self.declarer_seat < NUM_SEATS:
            raise ValueError(f"declarer_seat must be 0..3, got {self.declarer_seat}")
        if not 0 <= self.trump_suit < NUM_SUITS:
            raise ValueError(f"trump_suit must be 0..3, got {self.trump_suit}")

    @property
    def trump_action(self) -> int:
        return TRUMP_ACTION_START + self.trump_suit

    @property
    def bid_action(self) -> int:
        return action_for_bid(self.contract)


def build_auction_script(
    stage: StageSpec, match_seed: int, deal_index: int, dealer_seat: int
) -> AuctionScript:
    """Return the scripted auction for one deal.

    The declarer rotates with the deal index so that, over a rollout, the learning
    policy declares about as often as it defends and sees every seat.
    """
    generator = np.random.default_rng(
        derive_deal_seed(match_seed, deal_index) ^ (AUCTION_SCRIPT_SALT + stage.script_seed)
    )
    declarer_seat = (dealer_seat + 1 + deal_index) % NUM_SEATS

    if stage.trump_mode == SAMPLED:
        trump_suit = int(generator.integers(0, NUM_SUITS))
    else:
        trump_suit = stage.fixed_trump_suit

    if stage.contract_mode == SAMPLED:
        contract = int(generator.choice(np.asarray(stage.sampled_contracts, dtype=np.int64)))
    else:
        contract = stage.fixed_contract

    return AuctionScript(declarer_seat=declarer_seat, contract=contract, trump_suit=trump_suit)


def scripted_auction_action(
    script: AuctionScript, phase: Phase, acting_seat: int, current_high_bid: int | None
) -> int:
    """Return the action the script dictates for whoever is on turn.

    During ``BID`` the designated declarer bids once and everyone else passes;
    during ``DECLARE`` the declarer names the trump. The rule is stated in terms of
    the *public* auction state, so the same function serves the training
    environment and the evaluation agent.
    """
    if phase is Phase.BID:
        if acting_seat == script.declarer_seat and current_high_bid is None:
            return script.bid_action
        return PASS_ACTION
    if phase is Phase.DECLARE:
        return script.trump_action
    raise ValueError(f"the auction script does not act in phase {phase!r}")


def is_auction_phase(phase: Phase) -> bool:
    return phase in (Phase.BID, Phase.DECLARE)
