"""Frozen rule configuration for Tarneeb.

This module is the single source of truth for *which* variant of Tarneeb the
rest of the project implements. It is pure configuration: no game logic, no
scoring, no engine. See ``docs/`` and blueprint v1.2 section 2 for the rule
identifiers (C01-C26) referenced in the comments below.

Changing any value in the "frozen core" block is a NEW rules version. It
requires a new ``RULES_VERSION`` string, updated tests, and invalidation of
every checkpoint and result produced under the old version.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

RULES_VERSION = "tarneeb-classic-v1"


@dataclass(frozen=True, slots=True)
class RulesConfig:
    """The frozen ``tarneeb-classic-v1`` rule set.

    Instances are immutable. The defaults *are* the frozen specification;
    constructing a ``RulesConfig`` with non-default values describes a
    different (currently unreleased) rules version.
    """

    version: str = RULES_VERSION

    # --- frozen core (C01-C26) -----------------------------------------
    # Changing any of these is a NEW rules version.
    target_score: int = 31  # C25: first team to reach 31 wins the match
    counter_clockwise: bool = True  # C04: bidding and play direction
    min_bid: int = 7  # C07: lowest legal bid
    max_bid: int = 13  # C07: highest legal bid
    pass_is_permanent: bool = True  # C09: a pass leaves the auction for good
    allow_no_trump: bool = False  # C13: no-trump is not part of v1
    kaboot_total: int = 16  # C22: all 13 tricks after a bid < 13
    defender_kaboot_bonus: bool = True  # C26 FROZEN
    defender_kaboot_score: int = 16  # C26 FROZEN
    bid13_made: int = 26  # C23: a made bid of 13
    bid13_failed: int = -16  # C24: a failed bid of 13
    bid13_failed_defender_multiplier: int = 2  # C24: defenders score double

    # --- optional variants (section 2.4) -------------------------------
    # Off in v1. Each is a documented alternative played at other tables;
    # none is implemented yet.
    dealer_forced_seven: bool = False
    dealer_may_equal: bool = False
    single_bid_round: bool = False
    bidder_must_lead_trump: bool = False
    weak_hand_throw_in: bool = False
    long_suit_rule: bool = False
    egyptian_bidding: bool = False  # out of scope for v1

    def as_dict(self) -> dict[str, object]:
        """Return every field as a plain dictionary, for serialization."""
        return asdict(self)

    def fingerprint(self) -> str:
        """Return ``"sha1:<hex>"`` identifying this exact rule configuration.

        The digest is taken over the full field dictionary with keys sorted, so
        it is stable across processes and across field reordering, and changes
        whenever any rule value changes. Stamp it into every checkpoint and
        result file; refuse to load an artifact whose fingerprint differs.
        """
        canonical_json = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        sha1_hex_digest = hashlib.sha1(canonical_json.encode("utf-8")).hexdigest()
        return f"sha1:{sha1_hex_digest}"
