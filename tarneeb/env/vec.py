"""Headless RL environment adapters around the frozen engine.

:class:`TarneebEnv` wraps one :class:`~tarneeb.core.engine.TarneebEngine` state as
a single-decision-at-a-time environment. :class:`VecTarneeb` runs many of them
side by side and returns batched NumPy arrays; it is the training hot path.

Neither class implements a game rule. Legality always comes from
``TarneebEngine.legal_actions``, transitions always go through
``TarneebEngine.step``, and rewards always come from the engine's scored
``DealResult``. An episode is one deal.

An all-pass redeal is not a terminal state: the engine throws the hand in and
keeps the same dealer, so the environment simply deals again and the episode
continues with no reward emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tarneeb.core.actions import NUM_ACTIONS
from tarneeb.core.cards import derive_deal_seed
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import NUM_SEATS, Phase, PublicEvent
from tarneeb.core.rules import RulesConfig
from tarneeb.core.state import TarneebState
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.reward import deal_rewards_for_state, zero_rewards_by_seat
from tarneeb.env.spaces import D_CRITIC, D_STATIC

LIVE_PHASES = (Phase.BID, Phase.DECLARE, Phase.PLAY)


def derive_environment_seed(root_seed: int, environment_index: int) -> int:
    """Return the match seed for one environment of a vectorized run.

    Reuses the proven Phase 3 seed derivation, which is a pure SHA-256 function
    of two integers, so per-environment streams are independent and reproducible
    without touching any global RNG.
    """
    return derive_deal_seed(root_seed, environment_index)


@dataclass(frozen=True, slots=True)
class EnvStep:
    """What one environment returns after a single action."""

    acting_seat: int | None
    """The seat that must act next, or ``None`` when the episode has ended."""

    legal_action_mask: np.ndarray
    """``(65,)`` bool, taken directly from the engine. All False when terminated."""

    rewards_by_seat: np.ndarray
    """``(4,)`` float32. Non-zero only on the step that completes the deal."""

    terminated: bool
    """True when the deal was played out and scored."""

    truncated: bool
    """True when the episode ended for a non-game reason. Always False in Phase 4."""

    events: tuple[PublicEvent, ...]
    """Public events emitted by this action, in chronological order."""

    deal_result: object = None
    """The scored :class:`~tarneeb.core.events.DealResult` on the terminal step."""


class TarneebEnv:
    """One headless deal, stepped one legal action at a time."""

    def __init__(
        self,
        rules: RulesConfig | None = None,
        match_seed: int = 0,
        dealer_seat: int = 0,
        engine: TarneebEngine | None = None,
        observation_builder: ObservationBuilder | None = None,
    ) -> None:
        self.rules = rules if rules is not None else RulesConfig()
        self.engine = engine if engine is not None else TarneebEngine(self.rules)
        self.observations = (
            observation_builder
            if observation_builder is not None
            else ObservationBuilder(self.engine)
        )
        self.match_seed = match_seed
        self.dealer_seat = dealer_seat
        self.deal_index = 0
        self.episode_count = 0
        self.state: TarneebState | None = None

    # --- lifecycle ----------------------------------------------------------

    def reset(self, match_seed: int | None = None, dealer_seat: int | None = None) -> EnvStep:
        """Start a new episode (one deal) and return its opening step.

        Passing ``match_seed`` restarts the deal counter, so the same seed always
        reproduces the same first deal.
        """
        if match_seed is not None:
            self.match_seed = match_seed
            self.deal_index = 0
        if dealer_seat is not None:
            self.dealer_seat = dealer_seat

        self.state = self.engine.new_deal(
            self.dealer_seat, deal_index=self.deal_index, match_seed=self.match_seed
        )
        self.engine.deal(self.state)
        self.episode_count += 1
        return self._current_step(events=(), rewards=zero_rewards_by_seat())

    def step(self, action_id: int) -> EnvStep:
        """Apply one action for the acting seat.

        Illegal actions raise through the engine's normal path; they are never
        punished with a reward.
        """
        state = self._require_state()
        if state.phase not in LIVE_PHASES:
            raise ValueError("the episode has ended; call reset() before stepping again")

        _state, emitted_events, deal_result = self.engine.step(state, action_id)

        # An all-pass auction throws the hand in and keeps the same dealer. The
        # episode continues with a fresh deal; no reward is emitted.
        if state.phase is Phase.DEAL:
            self.deal_index = state.deal_index
            self.engine.deal(state)
            return self._current_step(events=tuple(emitted_events), rewards=zero_rewards_by_seat())

        if deal_result is not None:
            self.deal_index = state.deal_index + 1
            return EnvStep(
                acting_seat=None,
                legal_action_mask=np.zeros(NUM_ACTIONS, dtype=bool),
                rewards_by_seat=deal_rewards_for_state(state),
                terminated=True,
                truncated=False,
                events=tuple(emitted_events),
                deal_result=deal_result,
            )

        return self._current_step(events=tuple(emitted_events), rewards=zero_rewards_by_seat())

    # --- views --------------------------------------------------------------

    @property
    def acting_seat(self) -> int | None:
        state = self._require_state()
        return state.current_player_seat

    def legal_action_mask(self) -> np.ndarray:
        """Return the engine's ``(65,)`` bool mask for the acting seat."""
        state = self._require_state()
        if state.phase not in LIVE_PHASES:
            return np.zeros(NUM_ACTIONS, dtype=bool)
        return self.engine.legal_actions(state)

    def actor_observation(self, observer_seat: int | None = None) -> np.ndarray:
        """Return the ``(692,)`` actor-safe observation, by default for the acting seat."""
        state = self._require_state()
        seat = self._resolve_observer_seat(state, observer_seat)
        return self.observations.static_actor(state, seat)

    def critic_observation(self, observer_seat: int | None = None) -> np.ndarray:
        """Return the ``(856,)`` critic observation. Training only.

        This is a separate method on purpose: an actor path that never calls it
        cannot accidentally receive hidden information.
        """
        state = self._require_state()
        seat = self._resolve_observer_seat(state, observer_seat)
        return self.observations.static_critic(state, seat)

    def padded_history(self, observer_seat: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        state = self._require_state()
        seat = self._resolve_observer_seat(state, observer_seat)
        return self.observations.padded_history(state, seat)

    def ui_view(self, observer_seat: int | None = None) -> dict[str, object]:
        state = self._require_state()
        seat = self._resolve_observer_seat(state, observer_seat)
        return self.observations.ui_view(state, seat)

    def is_terminated(self) -> bool:
        state = self._require_state()
        return state.phase is Phase.SCORE

    # --- internals ----------------------------------------------------------

    def _require_state(self) -> TarneebState:
        if self.state is None:
            raise ValueError("environment has not been reset")
        return self.state

    @staticmethod
    def _resolve_observer_seat(state: TarneebState, observer_seat: int | None) -> int:
        if observer_seat is not None:
            return observer_seat
        if state.current_player_seat is None:
            raise ValueError("no seat is on turn; pass observer_seat explicitly")
        return state.current_player_seat

    def _current_step(self, events: tuple[PublicEvent, ...], rewards: np.ndarray) -> EnvStep:
        state = self._require_state()
        return EnvStep(
            acting_seat=state.current_player_seat,
            legal_action_mask=self.legal_action_mask(),
            rewards_by_seat=rewards,
            terminated=False,
            truncated=False,
            events=events,
        )


@dataclass
class VecStep:
    """Batched result of one vectorized step."""

    acting_seats: np.ndarray
    """``(n_envs,)`` int32. The seat to act in each environment."""

    legal_action_masks: np.ndarray
    """``(n_envs, 65)`` bool."""

    rewards_by_seat: np.ndarray
    """``(n_envs, 4)`` float32."""

    terminated: np.ndarray
    """``(n_envs,)`` bool."""

    truncated: np.ndarray
    """``(n_envs,)`` bool. Always False in Phase 4."""

    events: list[tuple[PublicEvent, ...]] = field(default_factory=list)
    """Per-environment events emitted by this step, in order."""


class VecTarneeb:
    """Many independent environments stepped together.

    Each environment has its own engine state and its own seed stream derived
    from one root seed, so two runs with the same configuration produce identical
    trajectories under a deterministic driver. No global RNG is touched.

    At each vector step exactly one seat is on turn per environment, so a batch of
    actions has one entry per environment.
    """

    def __init__(
        self,
        num_envs: int,
        root_seed: int = 0,
        rules: RulesConfig | None = None,
        rotate_dealer_by_index: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be at least 1, got {num_envs}")

        self.num_envs = num_envs
        self.root_seed = root_seed
        self.rules = rules if rules is not None else RulesConfig()

        # One engine and one builder are shared: both are stateless with respect
        # to a deal, so sharing them cannot alias environment state.
        self.engine = TarneebEngine(self.rules)
        self.observations = ObservationBuilder(self.engine)

        self.environments: list[TarneebEnv] = [
            TarneebEnv(
                rules=self.rules,
                match_seed=derive_environment_seed(root_seed, environment_index),
                dealer_seat=environment_index % NUM_SEATS if rotate_dealer_by_index else 0,
                engine=self.engine,
                observation_builder=self.observations,
            )
            for environment_index in range(num_envs)
        ]

    # --- lifecycle ----------------------------------------------------------

    def reset(self, root_seed: int | None = None) -> VecStep:
        """Reset every environment and return the opening batch."""
        if root_seed is not None:
            self.root_seed = root_seed
            for environment_index, environment in enumerate(self.environments):
                environment.match_seed = derive_environment_seed(root_seed, environment_index)
                environment.deal_index = 0

        steps = [environment.reset() for environment in self.environments]
        return self._batch(steps)

    def step(self, action_ids: np.ndarray | list[int]) -> VecStep:
        """Apply one action per environment.

        Environments that terminated on this step are NOT auto-reset: the caller
        must observe the terminal reward first and then call
        :meth:`reset_terminated`.
        """
        actions = np.asarray(action_ids, dtype=np.int64)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} actions, got shape {actions.shape}")

        steps = [
            environment.step(int(action))
            for environment, action in zip(self.environments, actions, strict=True)
        ]
        return self._batch(steps)

    def reset_terminated(self, terminated: np.ndarray) -> VecStep:
        """Reset exactly the environments flagged in ``terminated``.

        Environments that are still running keep their current state, so a
        finished deal in one environment never disturbs another.
        """
        terminated_flags = np.asarray(terminated, dtype=bool)
        if terminated_flags.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} flags, got shape {terminated_flags.shape}")

        steps = [
            environment.reset()
            if is_terminated
            else environment._current_step(events=(), rewards=zero_rewards_by_seat())
            for environment, is_terminated in zip(self.environments, terminated_flags, strict=True)
        ]
        return self._batch(steps)

    # --- batched views ------------------------------------------------------

    def actor_observations(self) -> np.ndarray:
        """Return ``(n_envs, 692)`` float32 observations for the acting seats."""
        batch = np.zeros((self.num_envs, D_STATIC), dtype=np.float32)
        for environment_index, environment in enumerate(self.environments):
            if environment.acting_seat is not None:
                batch[environment_index] = environment.actor_observation()
        return batch

    def critic_observations(self) -> np.ndarray:
        """Return ``(n_envs, 856)`` float32 critic observations. Training only."""
        batch = np.zeros((self.num_envs, D_CRITIC), dtype=np.float32)
        for environment_index, environment in enumerate(self.environments):
            if environment.acting_seat is not None:
                batch[environment_index] = environment.critic_observation()
        return batch

    def legal_action_masks(self) -> np.ndarray:
        """Return ``(n_envs, 65)`` bool masks straight from the engine."""
        batch = np.zeros((self.num_envs, NUM_ACTIONS), dtype=bool)
        for environment_index, environment in enumerate(self.environments):
            batch[environment_index] = environment.legal_action_mask()
        return batch

    def acting_seats(self) -> np.ndarray:
        """Return ``(n_envs,)`` int32 acting seats; ``-1`` where the episode ended."""
        return np.array(
            [
                -1 if environment.acting_seat is None else environment.acting_seat
                for environment in self.environments
            ],
            dtype=np.int32,
        )

    # --- internals ----------------------------------------------------------

    def _batch(self, steps: list[EnvStep]) -> VecStep:
        return VecStep(
            acting_seats=np.array(
                [-1 if step.acting_seat is None else step.acting_seat for step in steps],
                dtype=np.int32,
            ),
            legal_action_masks=np.stack([step.legal_action_mask for step in steps]),
            rewards_by_seat=np.stack([step.rewards_by_seat for step in steps]),
            terminated=np.array([step.terminated for step in steps], dtype=bool),
            truncated=np.array([step.truncated for step in steps], dtype=bool),
            events=[step.events for step in steps],
        )
