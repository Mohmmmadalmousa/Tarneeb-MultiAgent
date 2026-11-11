"""Deal scoring for ``tarneeb-classic-v1``.

This module contains the scoring rule and nothing else: no state, no engine,
no cards. It implements C20-C24 and C26 of the frozen specification.
"""

from __future__ import annotations

from tarneeb.core.rules import RulesConfig

TRICKS_PER_DEAL = 13


def score_deal(
    bid: int,
    tricks_bidding_team: int,
    rules: RulesConfig,
) -> tuple[int, int]:
    """Return the score change for the bidding and defending teams.

    Args:
        bid: The winning bid, in ``rules.min_bid .. rules.max_bid``.
        tricks_bidding_team: Tricks taken by the bidding team, ``0..13``.
        rules: The rule configuration in force.

    Returns:
        ``(delta_bidding_team, delta_defending_team)``.

    Raises:
        ValueError: If ``bid`` or ``tricks_bidding_team`` is out of range.
    """
    if not rules.min_bid <= bid <= rules.max_bid:
        raise ValueError(
            f"bid must be in {rules.min_bid}..{rules.max_bid} under {rules.version}, got {bid}"
        )
    if not 0 <= tricks_bidding_team <= TRICKS_PER_DEAL:
        raise ValueError(
            f"tricks_bidding_team must be in 0..{TRICKS_PER_DEAL}, got {tricks_bidding_team}"
        )

    bidding_team_tricks = tricks_bidding_team
    defending_team_tricks = TRICKS_PER_DEAL - bidding_team_tricks

    # The bid-13 rules are checked first: they take precedence over both kaboot
    # rules. See docs/RULES.md section 5.1.
    if bid == rules.max_bid:
        if bidding_team_tricks == TRICKS_PER_DEAL:
            return (rules.bid13_made, 0)  # C23
        defending_team_score_change = rules.bid13_failed_defender_multiplier * defending_team_tricks
        return (rules.bid13_failed, defending_team_score_change)  # C24

    # bid < 13
    if bidding_team_tricks == TRICKS_PER_DEAL:
        return (rules.kaboot_total, 0)  # C22 bidder kaboot
    if bidding_team_tricks >= bid:
        return (bidding_team_tricks, 0)  # C20 contract made
    if rules.defender_kaboot_bonus and defending_team_tricks == TRICKS_PER_DEAL:
        return (-bid, rules.defender_kaboot_score)  # C26 defender kaboot
    return (-bid, defending_team_tricks)  # C21 contract failed
