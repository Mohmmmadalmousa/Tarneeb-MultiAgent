from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.core.actions import PASS_ACTION, is_bid_action
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState
from tarneeb.curriculum.auction_diagnostics import AuctionProbe, collect_auction_probe
from tarneeb.env.public_view import PublicView, PublicViewBuilder


@dataclass
class Phase11BProbe(AuctionProbe):
    learner_team: int = 0


@dataclass(frozen=True, slots=True)
class AuctionProbeSummary:
    """Deterministic summary of a public-only diagnostic probe."""

    probe_size: int
    learner_team: int
    legal_actions: int
    pass_count: int
    openings: int
    raises: int
    bid_distribution: dict[int, int] = field(default_factory=dict)


@dataclass(slots=True)
class AuctionTeacher:
    """Public-only stimulus teacher for Phase 11B.

    This teacher never reads hidden cards and never reasons from the actual
    opponent hands. It only consumes the public view and legal-action mask, which
    keeps it within the actor-observable information boundary.
    """

    strategy: str = "strong"
    last_input_is_public: bool = False
    last_hidden_card_leak: bool = False
    _agent: StrongHeuristicAgent | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.strategy != "strong":
            raise ValueError("Phase 11B supports only the public strong heuristic teacher")
        self._agent = StrongHeuristicAgent(name="phase11b_teacher")

    def choose_action(
        self,
        state: TarneebState,
        seat: int,
        *,
        view: PublicView | None = None,
        engine: TarneebEngine | None = None,
    ) -> int:
        """Select a legal action using only public actor-visible state."""
        if state.phase not in (Phase.BID, Phase.DECLARE):
            raise ValueError("Auction teacher only applies during bidding and declaration")

        if engine is None:
            engine = TarneebEngine(state.rules)
        builder = PublicViewBuilder(engine)
        resolved_view = view if view is not None else builder.build(state, seat)

        self.last_input_is_public = True
        self.last_hidden_card_leak = False
        legal_mask = resolved_view.legal_action_mask.copy()
        if not legal_mask.any():
            raise ValueError("Auction teacher received a state with no legal actions")

        if self._agent is None:
            raise RuntimeError("AuctionTeacher was not initialized")

        candidate = int(self._agent.act(resolved_view))
        if not 0 <= candidate < len(legal_mask) or not bool(legal_mask[candidate]):
            raise ValueError(f"Auction teacher produced illegal label {candidate}")
        return candidate


def summarize_learner_auction_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return learner-owned auction metrics with explicit ownership semantics.

    The denominator and ownership are explicitly learner-only and BID-phase-only.
    Opponent actions and trump declarations never enter these statistics.
    """
    learner_records = [
        record
        for record in records
        if bool(record.get("is_learner", False))
        and bool(record.get("was_auction_decision", False))
        and 52 <= int(record.get("action", -1)) <= PASS_ACTION
    ]
    learner_passes = sum(
        1 for record in learner_records if int(record.get("action", -1)) == PASS_ACTION
    )
    learner_decisions = len(learner_records)
    pass_rate = learner_passes / learner_decisions if learner_decisions else None

    def bid_value(action: Any) -> int | None:
        action_int = int(action)
        if not is_bid_action(action_int):
            return None
        return 7 + (action_int - 52)

    opening_opportunities = sum(r.get("current_high_bid") is None for r in learner_records)
    raise_opportunities = sum(
        r.get("current_high_bid") is not None and int(r["current_high_bid"]) < 13
        for r in learner_records
    )
    opening_bids = sum(
        r.get("current_high_bid") is None and is_bid_action(int(r["action"]))
        for r in learner_records
    )
    learner_raise_count = 0
    for record in learner_records:
        current_high_bid = record.get("current_high_bid")
        action = record.get("action")
        bid = bid_value(action)
        if current_high_bid is not None and bid is not None and bid > int(current_high_bid):
            learner_raise_count += 1

    return {
        "owner_scope": "learner_only",
        "denominator_owner": "learner",
        "learner_auction_decisions": learner_decisions,
        "learner_pass_actions": learner_passes,
        "learner_pass_rate": pass_rate,
        "learner_bid_count": sum(
            1 for record in learner_records if is_bid_action(int(record.get("action", -1)))
        ),
        "learner_opening_bid_count": sum(
            1
            for record in learner_records
            if record.get("current_high_bid") is None
            and is_bid_action(int(record.get("action", -1)))
        ),
        "learner_raise_count": learner_raise_count,
        "learner_opening_opportunities": opening_opportunities,
        "learner_raise_opportunities": raise_opportunities,
        "learner_opening_rate": opening_bids / opening_opportunities
        if opening_opportunities
        else None,
        "learner_raise_rate": learner_raise_count / raise_opportunities
        if raise_opportunities
        else None,
    }


def collect_phase11b_probe(
    rules: RulesConfig | None = None,
    *,
    deal_count: int = 32,
    seed_base: int = 800_000,
    learner_team: int = 0,
) -> Phase11BProbe:
    """Collect a fixed public-only auction probe for Phase 11B diagnostics."""
    if learner_team not in (0, 1):
        raise ValueError("learner_team must be 0 or 1")
    if deal_count < 0 or seed_base < 0 or seed_base + deal_count > 100_000_000:
        raise ValueError("Invalid development seed range")
    if 20250910 in range(seed_base, seed_base + deal_count):
        raise ValueError("20250910 is forbidden for Phase 11B development")
    probe = collect_auction_probe(
        rules=rules,
        deal_count=deal_count,
        seed_base=seed_base,
        learner_team=learner_team,
    )
    return Phase11BProbe(probe.states, probe.deal_count, probe.seed_base, learner_team)


def summarize_probe_auction_states(probe: AuctionProbe) -> AuctionProbeSummary:
    """Count legal opportunities, NOT chosen actions (pass_count is PASS-legal states)."""
    teams = {state.seat % 2 for state in probe.states}
    learner_team = probe.learner_team if isinstance(probe, Phase11BProbe) else next(iter(teams), 0)
    if teams - {learner_team}:
        raise ValueError("Probe contains opponent-owned states")
    legal_actions = 0
    pass_count = 0
    openings = 0
    raises = 0
    bid_distribution: dict[int, int] = {}

    for state in probe.states:
        legal_actions += int(np.count_nonzero(state.legal_action_mask))
        if state.phase is Phase.BID and state.current_high_bid is None:
            openings += 1
        if state.current_high_bid is not None:
            legal_raise_mask = state.legal_action_mask[52:59]
            if np.any(legal_raise_mask):
                raises += 1
        if state.phase is Phase.BID:
            pass_count += int(state.legal_action_mask[PASS_ACTION])
            for action in range(52, 59):
                if state.legal_action_mask[action]:
                    bid_amount = 7 + (action - 52)
                    bid_distribution[bid_amount] = bid_distribution.get(bid_amount, 0) + 1
    return AuctionProbeSummary(
        probe_size=len(probe.states),
        learner_team=learner_team,
        legal_actions=legal_actions,
        pass_count=pass_count,
        openings=openings,
        raises=raises,
        bid_distribution=bid_distribution,
    )
