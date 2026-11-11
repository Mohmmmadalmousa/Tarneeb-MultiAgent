"""Hand-derived PFSP probabilities and common opponent-provider behavior."""

from __future__ import annotations

import numpy as np
import pytest

from tarneeb.league.sampling import (
    LatestOnlyProvider,
    PFSPConfig,
    PFSPPerformance,
    PFSPProvider,
)


def test_latest_only_always_selects_its_configured_policy() -> None:
    provider = LatestOnlyProvider("latest-u250")
    assert [provider.sample() for _ in range(5)] == ["latest-u250"] * 5
    assert provider.probabilities() == {"latest-u250": 1.0}


def test_pfsp_probabilities_match_hand_calculated_category_and_weight_shares() -> None:
    # Recent weights: (1-.5)^2=.25 and (1-.75)^2=.0625, normalized 4/5,1/5.
    # Recent mass: .35 => .28,.07. Other categories have one member each.
    provider = PFSPProvider(
        policies_by_category={
            "champion": ("champ",),
            "recent": ("r1", "r2"),
            "older": ("old",),
            "strong": ("strong",),
            "random": ("random",),
        },
        performance={
            "r1": PFSPPerformance(0.5, 500),
            "r2": PFSPPerformance(0.75, 500),
            "old": PFSPPerformance(0.0, 500),
        },
        rng=np.random.default_rng(101),
    )

    probabilities = provider.probabilities()
    assert probabilities == pytest.approx(
        {
            "champ": 0.30,
            "r1": 0.28,
            "r2": 0.07,
            "old": 0.22,
            "strong": 0.09,
            "random": 0.04,
        }
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(np.isfinite(value) for value in probabilities.values())


def test_pfsp_rejects_one_policy_in_multiple_categories() -> None:
    provider = PFSPProvider(
        policies_by_category={
            "champion": ("duplicate",),
            "recent": ("duplicate",),
        },
        performance={},
        rng=np.random.default_rng(9),
        config=PFSPConfig(),
    )
    with pytest.raises(ValueError, match="appears in multiple PFSP categories"):
        provider.probabilities()


def test_pfsp_category_mass_redistributes_only_when_categories_are_empty() -> None:
    provider = PFSPProvider(
        policies_by_category={"recent": ("a", "b"), "strong": ("s",)},
        performance={"a": PFSPPerformance(0.0, 2), "b": PFSPPerformance(0.0, 2)},
        rng=np.random.default_rng(11),
        config=PFSPConfig(empty_category_strategy="proportional"),
    )
    # Available category masses .35 and .09 renormalize to 35/44 and 9/44.
    assert provider.probabilities() == pytest.approx({"a": 35 / 88, "b": 35 / 88, "s": 9 / 44})


def test_pfsp_seeded_sampling_reproduces_identical_sequences() -> None:
    arguments = {
        "policies_by_category": {"recent": ("a", "b", "c")},
        "performance": {
            "a": PFSPPerformance(0.1, 50),
            "b": PFSPPerformance(0.5, 50),
            "c": PFSPPerformance(0.9, 50),
        },
    }
    config = PFSPConfig(empty_category_strategy="proportional")
    first = PFSPProvider(**arguments, rng=np.random.default_rng(874), config=config)
    second = PFSPProvider(**arguments, rng=np.random.default_rng(874), config=config)
    assert [first.sample() for _ in range(100)] == [second.sample() for _ in range(100)]


def test_ema_beta_weights_the_new_observation() -> None:
    stats = PFSPPerformance(learner_win_rate_against_opponent_ema=0.4, deals_observed=3)
    beta = 2 / 501
    expected = beta * 0.8 + (1 - beta) * 0.4
    assert stats.update(learner_win_rate_against_opponent=0.8, ema_beta=beta) == pytest.approx(
        expected
    )
    assert beta * 0.8 > 0
    assert stats.deals_observed == 4


def test_frozen_ema_beta_convention_is_new_observation_weight() -> None:
    config = PFSPConfig()
    assert config.ema_beta == pytest.approx(2 / 501)
    assert 0.0 < config.ema_beta < 0.01
    # new_ema = beta * NEW + (1 - beta) * OLD
    old = 0.4
    new = 0.8
    expected = (2 / 501) * new + (1 - 2 / 501) * old
    assert PFSPPerformance(old).update(new, config.ema_beta) == pytest.approx(expected)


def test_frozen_empty_category_setting_proportionally_redistributes_mass() -> None:
    provider = PFSPProvider(
        policies_by_category={"recent": ("a",), "strong": ("s",)},
        performance={},
        rng=np.random.default_rng(1),
        config=PFSPConfig(),
    )
    # Available mass is .35 recent + .09 strong; its normalized split is 35/44:9/44.
    assert provider.probabilities() == pytest.approx({"a": 35 / 44, "s": 9 / 44})


def test_frozen_unobserved_learner_win_rate_is_half() -> None:
    provider = PFSPProvider(
        policies_by_category={
            "champion": ("champion",),
            "recent": ("unobserved",),
            "older": ("older",),
            "strong": ("strong",),
            "random": ("random",),
        },
        performance={},
        rng=np.random.default_rng(2),
    )
    assert provider.probabilities()["unobserved"] > 0
    assert provider.config.unobserved_learner_win_rate == 0.5


def test_a3_pfsp_settings_are_frozen() -> None:
    config = PFSPConfig()
    assert config.alpha == 2
    assert config.champion_share == 0.30
    assert config.recent_share == 0.35
    assert config.older_share == 0.22
    assert config.strong_share == 0.09
    assert config.random_share == 0.04
    assert config.recent_limit == 5
    assert config.older_limit == 10
    assert config.ema_window_deals == 500
    assert config.empty_category_strategy == "proportional"
