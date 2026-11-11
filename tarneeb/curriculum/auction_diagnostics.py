"""Measuring what the auction policy actually does, separately from card play.

Stage 6 showed why this module is needed. Its whole-policy entropy stayed at a
respectable 0.125 while the auction policy had collapsed to entropy 0.0000 and a
pass probability of exactly 1.0. An aggregate number averaged over ~13 card plays
and ~1 auction decision per deal simply cannot see that.

So this measures the auction separately, on a **fixed probe**: a set of real
bidding states collected once from development seeds by letting the frozen Strong
heuristic drive every seat. Holding the probe fixed makes it a measuring
instrument rather than a moving target -- two checkpoints from different stages
are asked about literally the same decisions.

Nothing here trains, and nothing here enters the PPO objective. The entropy
reported is a *diagnostic*; the entropy that PPO optimises remains the frozen
whole-policy term with its configured coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.core.actions import (
    BID_ACTION_START,
    BID_ACTION_STOP,
    LOWEST_BID,
    NO_TRUMP_ACTION,
    PASS_ACTION,
    TRUMP_ACTION_START,
)
from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState, team_of_seat
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.nets import ActorNetwork

PROBE_SEED_BASE = 800_000
"""Development match seeds. Inside TRAINING_SEEDS, never the held-out range."""

PROBE_DEAL_COUNT = 200
"""Enough deals to yield several hundred auction decisions without being slow."""

PROBE_LEARNER_TEAM = 0
"""Which partnership the probe records decisions for. Arbitrary but fixed."""

BID_ACTIONS: tuple[int, ...] = tuple(range(BID_ACTION_START, BID_ACTION_STOP))

SITUATION_ALL = "all_auction_decisions"
SITUATION_OPEN = "open_auction"
SITUATION_OPPONENT_HAS_BID = "opponent_has_bid"
SITUATION_RAISE_AVAILABLE = "raise_available"
SITUATION_DECLARE = "declare_phase"

PROBE_SITUATIONS = (
    SITUATION_ALL,
    SITUATION_OPEN,
    SITUATION_OPPONENT_HAS_BID,
    SITUATION_RAISE_AVAILABLE,
    SITUATION_DECLARE,
)


@dataclass(frozen=True, slots=True)
class AuctionProbeState:
    """One recorded auction decision the probe will ask every checkpoint about."""

    state: TarneebState
    seat: int
    phase: Phase
    current_high_bid: int | None
    high_bidder_is_opponent: bool
    legal_action_mask: np.ndarray

    @property
    def situations(self) -> tuple[str, ...]:
        """Which measurement buckets this decision belongs to."""
        if self.phase is Phase.DECLARE:
            return (SITUATION_DECLARE,)
        buckets = [SITUATION_ALL]
        if self.current_high_bid is None:
            buckets.append(SITUATION_OPEN)
        if self.high_bidder_is_opponent:
            buckets.append(SITUATION_OPPONENT_HAS_BID)
        if (
            self.current_high_bid is not None
            and self.legal_action_mask[BID_ACTION_START:BID_ACTION_STOP].any()
        ):
            buckets.append(SITUATION_RAISE_AVAILABLE)
        return tuple(buckets)


@dataclass
class AuctionProbe:
    """A fixed set of auction decisions, collected once and reused."""

    states: list[AuctionProbeState] = field(default_factory=list)
    deal_count: int = 0
    seed_base: int = PROBE_SEED_BASE

    def __len__(self) -> int:
        return len(self.states)


def collect_auction_probe(
    rules: RulesConfig | None = None,
    deal_count: int = PROBE_DEAL_COUNT,
    seed_base: int = PROBE_SEED_BASE,
    learner_team: int = PROBE_LEARNER_TEAM,
) -> AuctionProbe:
    """Record real auction decision points from development deals.

    The frozen Strong heuristic drives every seat, so the distribution of auction
    situations does not depend on the policy being measured.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    engine = TarneebEngine(resolved_rules)
    views = PublicViewBuilder(engine)
    driver = StrongHeuristicAgent(name="strong")
    probe = AuctionProbe(deal_count=deal_count, seed_base=seed_base)

    for deal_index in range(deal_count):
        state = engine.new_deal(
            deal_index % NUM_SEATS, deal_index=0, match_seed=seed_base + deal_index
        )
        engine.deal(state)
        while state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            seat = state.current_player_seat
            assert seat is not None
            if state.phase in (Phase.BID, Phase.DECLARE) and team_of_seat(seat) == learner_team:
                probe.states.append(
                    AuctionProbeState(
                        state=state.clone(),
                        seat=seat,
                        phase=state.phase,
                        current_high_bid=state.current_high_bid,
                        high_bidder_is_opponent=(
                            state.current_high_bidder is not None
                            and team_of_seat(state.current_high_bidder) != learner_team
                        ),
                        legal_action_mask=engine.legal_actions(state).copy(),
                    )
                )
            engine.step(state, driver.act(views.build(state, seat)))
            # Re-read the phase after the step, which mutated it. An all-pass
            # auction throws the hand in and keeps the same dealer.
            phase_after_step: Phase = state.phase
            if phase_after_step is Phase.DEAL:
                engine.deal(state)
    return probe


def measure_auction_policy(
    actor: ActorNetwork,
    probe: AuctionProbe,
    observations: ObservationBuilder | None = None,
    rules: RulesConfig | None = None,
) -> dict[str, Any]:
    """Return auction entropy and mean action probabilities, by situation.

    The actor is put in evaluation mode and its previous mode restored, so calling
    this during training changes nothing.
    """
    resolved_rules = rules if rules is not None else RulesConfig()
    builder = (
        observations
        if observations is not None
        else ObservationBuilder(TarneebEngine(resolved_rules))
    )

    was_training = actor.training
    actor.eval()
    try:
        probabilities_by_situation: dict[str, list[np.ndarray]] = {
            situation: [] for situation in PROBE_SITUATIONS
        }
        entropies_by_situation: dict[str, list[float]] = {
            situation: [] for situation in PROBE_SITUATIONS
        }

        for probe_state in probe.states:
            static = torch.from_numpy(
                builder.static_actor(probe_state.state, probe_state.seat)
            ).unsqueeze(0)
            history_ids, history_mask = builder.padded_history(probe_state.state, probe_state.seat)
            legal = torch.from_numpy(probe_state.legal_action_mask).unsqueeze(0)

            with torch.no_grad():
                logits = actor(
                    static,
                    torch.from_numpy(history_ids.astype(np.int64)).unsqueeze(0),
                    torch.from_numpy(history_mask).unsqueeze(0),
                )
                distribution = MaskedCategorical.from_logits(logits, legal)
                probabilities = distribution.probabilities[0].numpy()
                entropy = float(distribution.entropy()[0])

            for situation in probe_state.situations:
                probabilities_by_situation[situation].append(probabilities)
                entropies_by_situation[situation].append(entropy)
    finally:
        if was_training:
            actor.train()

    report: dict[str, Any] = {}
    for situation in PROBE_SITUATIONS:
        rows = probabilities_by_situation[situation]
        if not rows:
            report[situation] = {"count": 0}
            continue
        mean_probabilities = np.mean(np.stack(rows), axis=0)
        entry: dict[str, Any] = {
            "count": len(rows),
            "entropy": float(np.mean(entropies_by_situation[situation])),
            "pass": float(mean_probabilities[PASS_ACTION]),
            "bid_total": float(mean_probabilities[BID_ACTION_START:BID_ACTION_STOP].sum()),
            "trump_total": float(mean_probabilities[TRUMP_ACTION_START:NO_TRUMP_ACTION].sum()),
            "no_trump": float(mean_probabilities[NO_TRUMP_ACTION]),
        }
        for offset, action in enumerate(BID_ACTIONS):
            entry[f"bid_{LOWEST_BID + offset}"] = float(mean_probabilities[action])
        report[situation] = entry
    return report


def auction_entropy(report: dict[str, Any], situation: str = SITUATION_ALL) -> float | None:
    """Pull one entropy number out of a probe report, for the metrics record."""
    entry = report.get(situation)
    if not entry or entry.get("count", 0) == 0:
        return None
    return float(entry["entropy"])


def flatten_probe_report(report: dict[str, Any]) -> dict[str, float]:
    """Flatten the probe into scalar metrics a reporter can plot."""
    flattened: dict[str, float] = {}
    for situation, entry in report.items():
        if not entry or entry.get("count", 0) == 0:
            continue
        flattened[f"auction_entropy_{situation}"] = float(entry["entropy"])
        flattened[f"auction_pass_probability_{situation}"] = float(entry["pass"])
        flattened[f"auction_bid_probability_{situation}"] = float(entry["bid_total"])
    return flattened


def format_probe_report(report: dict[str, Any]) -> list[str]:
    """Render the probe as human-readable lines."""
    lines = []
    for situation in PROBE_SITUATIONS:
        entry = report.get(situation)
        if not entry or entry.get("count", 0) == 0:
            continue
        bids = "  ".join(
            f"b{LOWEST_BID + offset}={entry[f'bid_{LOWEST_BID + offset}']:.4f}"
            for offset in range(len(BID_ACTIONS))
        )
        lines.append(
            f"    {situation:<22} n={entry['count']:>4}  H={entry['entropy']:.4f}  "
            f"PASS={entry['pass']:.4f}  bidtot={entry['bid_total']:.4f}  {bids}"
        )
    return lines
