"""Tests that the default RulesConfig is exactly the frozen tarneeb-classic-v1.

Expected values are transcribed from the frozen rule table (C01-C26), not from
the implementation.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
import sys

import pytest

from tarneeb.core.rules import RULES_VERSION, RulesConfig

# Every field of the frozen specification, written out by hand.
FROZEN_DEFAULTS: dict[str, object] = {
    "version": "tarneeb-classic-v1",
    # frozen core (C01-C26)
    "target_score": 31,  # C25
    "counter_clockwise": True,  # C04
    "min_bid": 7,  # C07
    "max_bid": 13,  # C07
    "pass_is_permanent": True,  # C09
    "allow_no_trump": False,  # C13
    "kaboot_total": 16,  # C22
    "defender_kaboot_bonus": True,  # C26
    "defender_kaboot_score": 16,  # C26
    "bid13_made": 26,  # C23
    "bid13_failed": -16,  # C24
    "bid13_failed_defender_multiplier": 2,  # C24
    # optional variants (section 2.4), all off in v1
    "dealer_forced_seven": False,
    "dealer_may_equal": False,
    "single_bid_round": False,
    "bidder_must_lead_trump": False,
    "weak_hand_throw_in": False,
    "long_suit_rule": False,
    "egyptian_bidding": False,
}


def test_rules_version_constant() -> None:
    assert RULES_VERSION == "tarneeb-classic-v1"


@pytest.mark.parametrize(("field_name", "expected_value"), sorted(FROZEN_DEFAULTS.items()))
def test_default_field_value(field_name: str, expected_value: object) -> None:
    actual_value = getattr(RulesConfig(), field_name)
    assert actual_value == expected_value
    assert type(actual_value) is type(expected_value)


def test_no_unlisted_fields() -> None:
    """A new field must be added to the truth table above, deliberately."""
    field_names = {field.name for field in dataclasses.fields(RulesConfig)}
    assert field_names == set(FROZEN_DEFAULTS)


def test_config_is_immutable() -> None:
    rules_config = RulesConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rules_config.target_score = 41  # type: ignore[misc]


def test_configs_compare_by_value() -> None:
    assert RulesConfig() == RulesConfig()
    assert RulesConfig() != RulesConfig(target_score=41)


# --- fingerprint ------------------------------------------------------------


def test_fingerprint_format() -> None:
    assert re.fullmatch(r"sha1:[0-9a-f]{40}", RulesConfig().fingerprint())


def test_fingerprint_is_stable_within_a_process() -> None:
    assert RulesConfig().fingerprint() == RulesConfig().fingerprint()


def test_fingerprint_is_stable_across_processes() -> None:
    """Guards against hash randomization or dict-ordering dependence."""
    print_fingerprint_script = (
        "from tarneeb.core.rules import RulesConfig; print(RulesConfig().fingerprint())"
    )
    fingerprints_from_fresh_processes = [
        subprocess.run(
            [sys.executable, "-c", print_fingerprint_script],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    first_fingerprint, second_fingerprint = fingerprints_from_fresh_processes
    assert first_fingerprint == second_fingerprint == RulesConfig().fingerprint()


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("version", "tarneeb-classic-v1-alt13"),
        ("target_score", 41),
        ("counter_clockwise", False),
        ("min_bid", 8),
        ("max_bid", 12),
        ("pass_is_permanent", False),
        ("allow_no_trump", True),
        ("kaboot_total", 13),
        ("defender_kaboot_bonus", False),
        ("defender_kaboot_score", 13),
        ("bid13_made", 25),
        ("bid13_failed", -13),
        ("bid13_failed_defender_multiplier", 1),
        ("dealer_forced_seven", True),
        ("dealer_may_equal", True),
        ("single_bid_round", True),
        ("bidder_must_lead_trump", True),
        ("weak_hand_throw_in", True),
        ("long_suit_rule", True),
        ("egyptian_bidding", True),
    ],
)
def test_fingerprint_changes_when_any_field_changes(field_name: str, changed_value: object) -> None:
    variant_config = dataclasses.replace(RulesConfig(), **{field_name: changed_value})
    assert variant_config.fingerprint() != RulesConfig().fingerprint()


def test_as_dict_round_trips() -> None:
    field_values = RulesConfig().as_dict()
    assert set(field_values) == set(FROZEN_DEFAULTS)
    assert RulesConfig(**field_values) == RulesConfig()  # type: ignore[arg-type]
