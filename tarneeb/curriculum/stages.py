"""What each curriculum stage asks the policy to do.

The curriculum runs from easy to hard. Early stages remove the auction entirely so
the policy can learn card play in isolation; later stages restore it. Nothing here
changes a rule: an auction is "removed" by *scripting* it with the frozen engine,
so every deal is still a legal ``tarneeb-classic-v1`` deal with a legal event
stream that replays.

===== ===================================== ================ ==================
stage task                                  opponent         auction
===== ===================================== ================ ==================
2     fixed trump, fixed contract           random           scripted
3     fixed trump, fixed contract           random -> basic  scripted
4     sampled trump, sampled contract       basic            scripted
5     sampled trump, sampled contract       strong           scripted
5B    full game, gentle auction debut       basic            learned
6     full game                             strong           learned
===== ===================================== ================ ==================

Stage 2 may optionally add a trick-count shaping term. It is **off by default**,
is owned entirely by Phase 7, and later stages assert it is zero, so it cannot
leak forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tarneeb.core.actions import HIGHEST_BID, LOWEST_BID, NUM_SUITS, SUIT_NAMES

LEARNED_AUCTION = "learned"
"""The policy bids for itself; the engine runs the real auction."""

SCRIPTED_AUCTION = "scripted"
"""A deterministic auction is driven through the frozen engine before play starts."""

FIXED = "fixed"
SAMPLED = "sampled"

FULL_HISTORY = "full"
AUCTION_REMOVED = "auction_removed"
CURRENT_BID_ONLY = "current_bid_only"

HISTORY_CONDITIONS = (FULL_HISTORY, AUCTION_REMOVED, CURRENT_BID_ONLY)
"""The three A7 conditions. Only the history the encoder sees differs."""

SELECTABLE_STAGE_OPPONENTS = frozenset({"random", "basic", "strong"})
"""Frozen non-neural opponents. Self-play stays forbidden, as in Phase 6."""

DEFAULT_FIXED_TRUMP_SUIT = 3
"""Spades. Any suit would do; it is fixed so stage 2 varies one thing at a time."""

DEFAULT_FIXED_CONTRACT = 8
"""A contract slightly above the 6.5 tricks an average team takes by chance."""

DEFAULT_SAMPLED_CONTRACTS = (7, 8, 9)
"""Contracts sampled in stages 4 and 5. Higher contracts are almost never made by
a mid-curriculum policy, so they would supply gradient without information."""


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One curriculum stage, fully described.

    ``trick_shaping_weight`` is the only reward modification available anywhere in
    Phase 7, it applies to the terminal transition only, and it is validated to be
    zero for every stage except stage 2.
    """

    name: str
    curriculum_stage: int
    auction_mode: str = SCRIPTED_AUCTION
    trump_mode: str = FIXED
    contract_mode: str = FIXED
    fixed_trump_suit: int = DEFAULT_FIXED_TRUMP_SUIT
    fixed_contract: int = DEFAULT_FIXED_CONTRACT
    sampled_contracts: tuple[int, ...] = DEFAULT_SAMPLED_CONTRACTS
    opponent: str = "random"
    trick_shaping_weight: float = 0.0
    history_condition: str = FULL_HISTORY
    script_seed: int = 90_000

    def __post_init__(self) -> None:
        if self.auction_mode not in (LEARNED_AUCTION, SCRIPTED_AUCTION):
            raise ValueError(f"unknown auction_mode {self.auction_mode!r}")
        if self.trump_mode not in (FIXED, SAMPLED):
            raise ValueError(f"unknown trump_mode {self.trump_mode!r}")
        if self.contract_mode not in (FIXED, SAMPLED):
            raise ValueError(f"unknown contract_mode {self.contract_mode!r}")
        if self.history_condition not in HISTORY_CONDITIONS:
            raise ValueError(f"unknown history_condition {self.history_condition!r}")
        if self.opponent not in SELECTABLE_STAGE_OPPONENTS:
            raise ValueError(
                f"opponent must be one of {sorted(SELECTABLE_STAGE_OPPONENTS)}, "
                f"got {self.opponent!r}; current-policy self-play is not available"
            )
        if not 0 <= self.fixed_trump_suit < NUM_SUITS:
            raise ValueError(f"fixed_trump_suit must be a suit 0..3, got {self.fixed_trump_suit}")
        if not LOWEST_BID <= self.fixed_contract <= HIGHEST_BID:
            raise ValueError(
                f"fixed_contract must lie in {LOWEST_BID}..{HIGHEST_BID}, got {self.fixed_contract}"
            )
        if not self.sampled_contracts:
            raise ValueError("sampled_contracts must not be empty")
        for contract in self.sampled_contracts:
            if not LOWEST_BID <= contract <= HIGHEST_BID:
                raise ValueError(f"sampled contract {contract} is outside the legal bid range")
        if self.trick_shaping_weight < 0.0:
            raise ValueError("trick_shaping_weight must not be negative")
        if self.trick_shaping_weight and self.curriculum_stage != 2:
            raise ValueError(
                f"trick shaping is a stage 2 device only, but stage "
                f"{self.curriculum_stage} requested weight {self.trick_shaping_weight}"
            )
        if self.auction_mode == LEARNED_AUCTION and self.trick_shaping_weight:
            raise ValueError("a learned auction must use the frozen deal reward alone")

    # --- derived -----------------------------------------------------------

    @property
    def uses_scripted_auction(self) -> bool:
        return self.auction_mode == SCRIPTED_AUCTION

    @property
    def uses_trick_shaping(self) -> bool:
        return self.trick_shaping_weight > 0.0

    @property
    def reward_description(self) -> str:
        """How the reward for this stage is described in the experiment record."""
        if self.uses_trick_shaping:
            return (
                "frozen normalized deal reward "
                f"+ phase7 trick shaping (weight {self.trick_shaping_weight})"
            )
        return "frozen normalized deal reward"

    @property
    def fixed_trump_name(self) -> str:
        return SUIT_NAMES[self.fixed_trump_suit]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StageSpec:
        settings = dict(values)
        if "sampled_contracts" in settings:
            settings["sampled_contracts"] = tuple(settings["sampled_contracts"])
        return cls(**settings)


# --- the shipped stages ----------------------------------------------------


def stage_two_trick_play(opponent: str = "random", trick_shaping_weight: float = 0.0) -> StageSpec:
    """Trick play only: fixed trump, fixed contract, no auction to learn."""
    return StageSpec(
        name="stage2_trick_play",
        curriculum_stage=2,
        opponent=opponent,
        trick_shaping_weight=trick_shaping_weight,
    )


def stage_three_deal_reward(opponent: str = "basic") -> StageSpec:
    """Same task as stage 2 with the frozen deal reward alone and a real opponent."""
    return StageSpec(
        name=f"stage3_deal_reward_{opponent}",
        curriculum_stage=3,
        opponent=opponent,
    )


def stage_four_sampled_contract(opponent: str = "basic") -> StageSpec:
    """Generalize card play across all four trumps and several contracts."""
    return StageSpec(
        name="stage4_sampled_contract",
        curriculum_stage=4,
        trump_mode=SAMPLED,
        contract_mode=SAMPLED,
        opponent=opponent,
    )


def stage_five_strong_play(opponent: str = "strong") -> StageSpec:
    """Sampled contracts against the strongest frozen heuristic, still no bidding."""
    return StageSpec(
        name="stage5_strong_play",
        curriculum_stage=5,
        trump_mode=SAMPLED,
        contract_mode=SAMPLED,
        opponent=opponent,
    )


def stage_five_b_auction_warmup(opponent: str = "basic") -> StageSpec:
    """Stage 5B: the first learned auction, introduced gently.

    The diagnosis of the original stage 6 found that the BID, PASS and TRUMP
    output-head rows were still bit-identical to their initialization after
    stages 2-5, because those actions were masked throughout. Stage 6 therefore
    exposed a *uniform* auction policy (entropy 2.0793 = ln 8) directly to the
    strongest opponent in the project. Uniform bidding escalates the auction, so
    the first contracts averaged bid 12.07 and succeeded 0.8% of the time, and
    PPO removed bidding entirely within 8 updates.

    This stage changes exactly two things about that first exposure, both of them
    ordinary Phase 7 configuration:

    - the opponent is the Basic heuristic rather than Strong, so early
      declarations are not punished by the best card player available;
    - the stage is configured with a higher entropy coefficient (in its YAML, via
      the frozen ``PPOConfig``), so the auction policy is not free to become
      deterministic before it has sampled a selective low bid.

    Nothing else moves. Same reward, same masks, same observation, same network,
    same PPO. This stage exists to test whether the collapse was caused by the
    curriculum transition; it is not a redesign of the learner.
    """
    return StageSpec(
        name="stage5b_auction_warmup",
        curriculum_stage=5,
        auction_mode=LEARNED_AUCTION,
        trump_mode=SAMPLED,
        contract_mode=SAMPLED,
        opponent=opponent,
        history_condition=FULL_HISTORY,
    )


def stage_six_full_bidding(
    opponent: str = "strong", history_condition: str = FULL_HISTORY
) -> StageSpec:
    """The complete game, auction included. The A7 conditions are run here."""
    suffix = "" if history_condition == FULL_HISTORY else f"_{history_condition}"
    return StageSpec(
        name=f"stage6_full_bidding{suffix}",
        curriculum_stage=6,
        auction_mode=LEARNED_AUCTION,
        trump_mode=SAMPLED,
        contract_mode=SAMPLED,
        opponent=opponent,
        history_condition=history_condition,
    )


SHIPPED_STAGES: dict[str, StageSpec] = {
    "stage2": stage_two_trick_play(),
    "stage3_random": stage_three_deal_reward(opponent="random"),
    "stage3": stage_three_deal_reward(opponent="basic"),
    "stage4": stage_four_sampled_contract(),
    "stage5": stage_five_strong_play(),
    "stage5b": stage_five_b_auction_warmup(),
    "stage6": stage_six_full_bidding(),
}
"""Convenience lookup used by the tests and by the stage scripts."""
