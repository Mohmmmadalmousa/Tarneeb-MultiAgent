"""A vectorized environment whose auction is decided in advance.

This is the training-side half of the curriculum. It subclasses the frozen
:class:`~tarneeb.env.vec.VecTarneeb` and overrides only the two reset paths: after
the frozen code has dealt a fresh hand, the scripted auction is driven to
completion through ``TarneebEnv.step``, so by the time the learner is asked for an
action the deal is already in ``PLAY``.

Because the frozen :class:`~tarneeb.training.rollout.RolloutCollector` accepts any
``VecTarneeb``, the whole Phase 6 training stack -- rollout storage, reward
attribution, GAE, the PPO update, checkpointing -- runs unmodified on top of this.

Two other Phase 7 hooks live here:

- an optional history-conditioned observation builder, for the A7 ablations;
- an optional stage 2 trick-shaping term, added to the **terminal** reward only.

Neither touches frozen code. The shaping term in particular is added on top of the
value the frozen reward implementation produced, never in place of it, and
:class:`~tarneeb.curriculum.stages.StageSpec` refuses a non-zero weight outside
stage 2.
"""

from __future__ import annotations

import numpy as np

from tarneeb.core.cards import NUM_SEATS
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TRICKS_PER_DEAL, team_of_seat
from tarneeb.curriculum.auction_script import (
    MAX_SCRIPTED_AUCTION_ACTIONS,
    AuctionScript,
    build_auction_script,
    is_auction_phase,
    scripted_auction_action,
)
from tarneeb.curriculum.history import HistoryConditionedObservationBuilder
from tarneeb.curriculum.stages import FULL_HISTORY, StageSpec
from tarneeb.env.vec import TarneebEnv, VecStep, VecTarneeb

AUCTION_PHASES = (Phase.BID, Phase.DECLARE)


class CurriculumVecTarneeb(VecTarneeb):
    """The frozen vectorized environment, shaped by one curriculum stage."""

    def __init__(
        self,
        num_envs: int,
        stage: StageSpec,
        root_seed: int = 0,
        rules: RulesConfig | None = None,
    ) -> None:
        super().__init__(num_envs=num_envs, root_seed=root_seed, rules=rules)
        self.stage = stage
        self.scripts_by_environment: list[AuctionScript | None] = [None] * num_envs

        if stage.history_condition != FULL_HISTORY:
            conditioned_builder = HistoryConditionedObservationBuilder(
                self.engine, stage.history_condition
            )
            self.observations = conditioned_builder
            for environment in self.environments:
                environment.observations = conditioned_builder

    # --- lifecycle ----------------------------------------------------------

    def reset(self, root_seed: int | None = None) -> VecStep:
        super().reset(root_seed)
        self._run_scripted_auctions()
        return self.current_vector_step()

    def reset_terminated(self, terminated: np.ndarray) -> VecStep:
        super().reset_terminated(terminated)
        self._run_scripted_auctions()
        return self.current_vector_step()

    def step(self, action_ids: np.ndarray | list[int]) -> VecStep:
        """Step the frozen environment, then add stage 2 shaping if configured."""
        vector_step = super().step(action_ids)
        if self.stage.uses_trick_shaping:
            self._add_trick_shaping(vector_step)
        return vector_step

    # --- the scripted auction ----------------------------------------------

    def _run_scripted_auctions(self) -> None:
        """Drive any freshly dealt environment through its scripted auction.

        Environments already in ``PLAY`` are untouched, so this is safe to call
        after a partial reset.
        """
        if not self.stage.uses_scripted_auction:
            return
        for environment_index, environment in enumerate(self.environments):
            state = environment.state
            if state is None or state.phase not in AUCTION_PHASES:
                continue
            script = build_auction_script(
                stage=self.stage,
                match_seed=environment.match_seed,
                deal_index=environment.deal_index,
                dealer_seat=environment.dealer_seat,
            )
            self.scripts_by_environment[environment_index] = script
            self._play_scripted_auction(environment, script)

    @staticmethod
    def _play_scripted_auction(environment: TarneebEnv, script: AuctionScript) -> None:
        """Step the frozen engine until the deal reaches ``PLAY``."""
        for _action_index in range(MAX_SCRIPTED_AUCTION_ACTIONS):
            state = environment.state
            assert state is not None
            if not is_auction_phase(state.phase):
                return
            acting_seat = state.current_player_seat
            assert acting_seat is not None
            environment.step(
                scripted_auction_action(
                    script=script,
                    phase=state.phase,
                    acting_seat=acting_seat,
                    current_high_bid=state.current_high_bid,
                )
            )
        raise AssertionError(  # pragma: no cover - the script always terminates
            "the scripted auction did not reach the play phase"
        )

    # --- stage 2 shaping ----------------------------------------------------

    def _add_trick_shaping(self, vector_step: VecStep) -> None:
        """Add a trick-count term to the terminal reward of finished deals.

        Phase 7 curriculum logic, stage 2 only. With ``gamma = 1.0`` the return of
        every decision in a deal is the sum of that deal's rewards, so adding the
        aggregated trick term once, at the terminal transition, gives each decision
        exactly the return that per-trick rewards would have produced.
        """
        for environment_index in np.flatnonzero(vector_step.terminated):
            state = self.environments[int(environment_index)].state
            if state is None:  # pragma: no cover - terminated implies a scored state
                continue
            tricks_by_team = (state.tricks_won[0], state.tricks_won[1])
            for seat in range(NUM_SEATS):
                own_team = team_of_seat(seat)
                trick_difference = tricks_by_team[own_team] - tricks_by_team[1 - own_team]
                vector_step.rewards_by_seat[environment_index, seat] += (
                    self.stage.trick_shaping_weight * trick_difference / TRICKS_PER_DEAL
                )

    # --- views --------------------------------------------------------------

    def current_vector_step(self) -> VecStep:
        """Return the batch describing the state every environment now faces.

        Built from the frozen public accessors only. No reward is emitted and no
        environment is terminated, because this is always called immediately after
        a reset.
        """
        return VecStep(
            acting_seats=self.acting_seats(),
            legal_action_masks=self.legal_action_masks(),
            rewards_by_seat=np.zeros((self.num_envs, NUM_SEATS), dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
            events=[() for _ in range(self.num_envs)],
        )


def build_curriculum_environments(
    num_envs: int, stage: StageSpec, root_seed: int, rules: RulesConfig | None = None
) -> CurriculumVecTarneeb:
    """Build the vectorized environment one curriculum stage needs."""
    return CurriculumVecTarneeb(num_envs=num_envs, stage=stage, root_seed=root_seed, rules=rules)
