"""Fair evaluation arena: duplicate deals, seat rotation, held-out seeds.

The arena assigns one policy to each partnership, drives the frozen engine to
the end of a deal, and records the outcome. It implements no game rule: legality
comes from ``TarneebEngine.legal_actions`` and scoring from the engine's
``DealResult``.

**Duplicate-deal protocol.** For each underlying deal -- fixed by
``(match_seed, deal_index)``, which determines the four hands -- the same cards
are played twice:

- run 1: policy A holds team 0 (seats 0 and 2), policy B holds team 1;
- run 2: policy B holds team 0, policy A holds team 1.

The paired statistic averages policy A's margin over the two orientations, so
the advantage of holding the better cards cancels.

**Held-out seeds.** All evaluation deals are drawn from
``HELD_OUT_EVALUATION_SEEDS``, a range reserved for evaluation and provably
disjoint from the training seed range.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from tarneeb.agents.base import Agent
from tarneeb.core.actions import PASS_ACTION, is_bid_action
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.recorder import GameRecorder, ReplayEngine
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import NUM_SEATS, team_of_seat
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.evaluation.metrics import (
    DealOutcome,
    EvaluationResult,
    PairedOutcome,
    aggregate_metrics,
)

HELD_OUT_EVALUATION_SEEDS = range(10**9, 10**9 + 100_000)
"""Evaluation-only match seeds. Training must never draw from this range."""

TRAINING_SEEDS = range(0, 10**8)
"""The range reserved for training. Disjoint from the evaluation range."""

DEFAULT_REPLAY_SAMPLE_RATE = 0.05
MAX_DECISIONS_PER_DEAL = 200
"""A deal needs at most ~10 auction + 1 trump + 52 cards; this is a safety stop."""


def evaluation_match_seed(evaluation_seed: int, pair_index: int) -> int:
    """Return the held-out match seed for one duplicate pair.

    Deterministic in ``(evaluation_seed, pair_index)`` and always inside
    ``HELD_OUT_EVALUATION_SEEDS``.
    """
    if pair_index < 0:
        raise ValueError(f"pair_index must be non-negative, got {pair_index}")
    offset = (evaluation_seed + pair_index) % len(HELD_OUT_EVALUATION_SEEDS)
    return HELD_OUT_EVALUATION_SEEDS.start + offset


@dataclass
class SeatAssignment:
    """Which policy controls which seat for one run."""

    agent_by_seat: tuple[Agent, Agent, Agent, Agent]
    policy_a_team: int

    def agent_for(self, seat: int) -> Agent:
        return self.agent_by_seat[seat]


def build_seat_assignment(policy_a: Agent, policy_b: Agent, policy_a_team: int) -> SeatAssignment:
    """Seat two partnership policies; team 0 is seats (0, 2), team 1 is (1, 3)."""
    if policy_a_team not in (0, 1):
        raise ValueError(f"policy_a_team must be 0 or 1, got {policy_a_team}")
    agents = tuple(
        policy_a if team_of_seat(seat) == policy_a_team else policy_b for seat in range(NUM_SEATS)
    )
    return SeatAssignment(agent_by_seat=agents, policy_a_team=policy_a_team)  # type: ignore[arg-type]


class Arena:
    """Plays deals between two partnership policies and records the outcomes."""

    def __init__(
        self,
        rules: RulesConfig | None = None,
        replay_sample_rate: float = DEFAULT_REPLAY_SAMPLE_RATE,
    ) -> None:
        self.rules = rules if rules is not None else RulesConfig()
        self.engine = TarneebEngine(self.rules)
        self.views = PublicViewBuilder(self.engine)
        self.replay_engine = ReplayEngine(self.rules)
        self.replay_sample_rate = replay_sample_rate
        self.illegal_action_count = 0

    # --- one deal -----------------------------------------------------

    def play_deal(
        self,
        assignment: SeatAssignment,
        match_seed: int,
        dealer_seat: int,
        deal_index: int = 0,
        verify_replay: bool = False,
    ) -> DealOutcome:
        """Play one deal to completion and return its outcome.

        A deal in which all four seats pass is recorded as passed out with a
        zero score for both teams, which is exactly what the rules award.
        """
        state = self.engine.new_deal(dealer_seat, deal_index=deal_index, match_seed=match_seed)
        self.engine.deal(state)

        recorder = GameRecorder.start(state) if verify_replay else None
        for agent in set(assignment.agent_by_seat):
            agent.reset()

        decision_count = 0
        bid_action_count = 0
        pass_action_count = 0
        passed_out = False

        while state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
            if decision_count > MAX_DECISIONS_PER_DEAL:  # pragma: no cover - safety stop
                raise AssertionError("deal exceeded the maximum decision count")

            acting_seat = state.current_player_seat
            assert acting_seat is not None
            view = self.views.build(state, acting_seat)
            selected_action = assignment.agent_for(acting_seat).act(view)

            if not view.legal_action_mask[selected_action]:  # pragma: no cover - agents check
                self.illegal_action_count += 1
                raise AssertionError(
                    f"agent returned illegal action {selected_action} for seat {acting_seat}"
                )

            if state.phase is Phase.BID:
                if is_bid_action(selected_action):
                    bid_action_count += 1
                elif selected_action == PASS_ACTION:
                    pass_action_count += 1

            _state, emitted_events, _result = self.engine.step(state, selected_action)
            if recorder is not None:
                recorder.record_step(selected_action, emitted_events)
            decision_count += 1

            # All four passed: the hand is thrown in and nobody scores. The
            # phase is re-read after the step, which mutated it.
            phase_after_step: Phase = state.phase
            if phase_after_step is Phase.DEAL:
                passed_out = True
                break

        replay_verified: bool | None = None
        if recorder is not None and not passed_out:
            replay_verified = self.replay_engine.verify(recorder.finish(state))

        if passed_out:
            return DealOutcome(
                match_seed=match_seed,
                deal_index=deal_index,
                dealer_seat=dealer_seat,
                policy_a_team=assignment.policy_a_team,
                score_change_by_team=(0, 0),
                declarer_seat=None,
                winning_bid=None,
                tricks_won_by_team=(0, 0),
                contract_made=None,
                passed_out=True,
                decision_count=decision_count,
                bid_action_count=bid_action_count,
                pass_action_count=pass_action_count,
                replay_verified=replay_verified,
            )

        deal_result = state.deal_result
        assert deal_result is not None
        declaring_team = team_of_seat(state.declarer_seat)  # type: ignore[arg-type]
        score_change_by_team = [0, 0]
        score_change_by_team[declaring_team] = deal_result.score_delta_bidding_team
        score_change_by_team[1 - declaring_team] = deal_result.score_delta_defending_team

        return DealOutcome(
            match_seed=match_seed,
            deal_index=deal_index,
            dealer_seat=dealer_seat,
            policy_a_team=assignment.policy_a_team,
            score_change_by_team=(score_change_by_team[0], score_change_by_team[1]),
            declarer_seat=state.declarer_seat,
            winning_bid=state.current_high_bid,
            tricks_won_by_team=(state.tricks_won[0], state.tricks_won[1]),
            contract_made=deal_result.tricks_bidding_team >= deal_result.bid,
            passed_out=False,
            decision_count=decision_count,
            bid_action_count=bid_action_count,
            pass_action_count=pass_action_count,
            replay_verified=replay_verified,
        )

    # --- one duplicate pair -------------------------------------------

    def play_duplicate_pair(
        self,
        policy_a: Agent,
        policy_b: Agent,
        match_seed: int,
        dealer_seat: int,
        deal_index: int = 0,
        verify_replay: bool = False,
    ) -> PairedOutcome:
        """Play the same dealt cards twice with the partnerships swapped."""
        first_run = self.play_deal(
            build_seat_assignment(policy_a, policy_b, policy_a_team=0),
            match_seed=match_seed,
            dealer_seat=dealer_seat,
            deal_index=deal_index,
            verify_replay=verify_replay,
        )
        second_run = self.play_deal(
            build_seat_assignment(policy_a, policy_b, policy_a_team=1),
            match_seed=match_seed,
            dealer_seat=dealer_seat,
            deal_index=deal_index,
            verify_replay=verify_replay,
        )
        return PairedOutcome(
            match_seed=match_seed,
            deal_index=deal_index,
            first_run=first_run,
            second_run=second_run,
        )

    # --- a full evaluation --------------------------------------------

    def evaluate(
        self,
        policy_a: Agent,
        policy_b: Agent,
        paired_deal_count: int,
        evaluation_seed: int,
        bootstrap_resamples: int = 10_000,
        replay_sample_rate: float | None = None,
    ) -> EvaluationResult:
        """Run ``paired_deal_count`` duplicate pairs and aggregate the metrics.

        The dealer rotates with the pair index so no policy sits systematically
        in a favourable seat or dealer position.
        """
        if paired_deal_count < 1:
            raise ValueError(f"paired_deal_count must be at least 1, got {paired_deal_count}")

        sample_rate = self.replay_sample_rate if replay_sample_rate is None else replay_sample_rate
        replay_every = round(1 / sample_rate) if sample_rate > 0 else 0

        self.illegal_action_count = 0
        paired_outcomes: list[PairedOutcome] = []
        start_time = time.perf_counter()

        for pair_index in range(paired_deal_count):
            verify_replay = replay_every > 0 and pair_index % replay_every == 0
            paired_outcomes.append(
                self.play_duplicate_pair(
                    policy_a,
                    policy_b,
                    match_seed=evaluation_match_seed(evaluation_seed, pair_index),
                    dealer_seat=pair_index % NUM_SEATS,
                    verify_replay=verify_replay,
                )
            )

        runtime_seconds = time.perf_counter() - start_time

        metrics = aggregate_metrics(
            paired_outcomes,
            bootstrap_seed=evaluation_seed,
            bootstrap_resamples=bootstrap_resamples,
            illegal_action_count=self.illegal_action_count,
            runtime_seconds=runtime_seconds,
        )

        return EvaluationResult(
            policy_a_name=policy_a.name,
            policy_b_name=policy_b.name,
            evaluation_seed=evaluation_seed,
            metrics=metrics,
            paired_outcomes=paired_outcomes,
        )


def seed_ranges_are_disjoint() -> bool:
    """True when the evaluation and training seed ranges do not overlap."""
    evaluation_seeds = set(
        range(HELD_OUT_EVALUATION_SEEDS.start, HELD_OUT_EVALUATION_SEEDS.stop, 1)
    )
    return not (
        HELD_OUT_EVALUATION_SEEDS.start < TRAINING_SEEDS.stop
        and TRAINING_SEEDS.start < HELD_OUT_EVALUATION_SEEDS.stop
    ) and bool(evaluation_seeds)


def sample_random_legal_actions(
    legal_action_masks: np.ndarray, rng: np.random.Generator
) -> np.ndarray:  # pragma: no cover - convenience for ad-hoc scripts
    """Sample one legal action per row of ``legal_action_masks``."""
    return np.array(
        [int(rng.choice(np.flatnonzero(mask))) for mask in legal_action_masks], dtype=np.int64
    )
