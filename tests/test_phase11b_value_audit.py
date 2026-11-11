"""Correctness gates for the diagnostic; never performs neural optimization."""

import numpy as np

from tarneeb.auction_value_audit import (
    CONFIG,
    bid_likelihood,
    decision_view,
    history_weight,
    make_state,
    play_candidate,
    sample_hands,
    select_cases,
    summarize,
)
from tarneeb.core.cards import FULL_DECK_MASK
from tarneeb.core.rules import RulesConfig
from tarneeb.core.scoring import score_deal
from tarneeb.env.observation import ObservationBuilder


def test_design_and_public_information_boundary():
    cases = select_cases()
    assert len(cases) == 32
    assert len({(c["high_bid"], c["owner"], c["selected_strength"]) for c in cases[:24]}) == 24
    for case in cases:
        rng = np.random.default_rng(case["world_seed"])
        hands1 = sample_hands(case["own_hand"], rng)
        hands2 = sample_hands(case["own_hand"], rng)
        assert hands1[0] == hands2[0]
        assert sum(h.bit_count() for h in hands1) == 52
        assert sum(hands1) == FULL_DECK_MASK
        view1 = decision_view(case, hands1)
        view2 = decision_view(case, hands2)
        assert view1.own_hand == view2.own_hand and view1.events == view2.events
        assert set(case["actions"]) == set(np.flatnonzero(view1.legal_action_mask))
        assert case["teacher"] in case["actions"]
        arrays = []
        for hands in [hands1, hands2]:
            engine, state = make_state(case, hands)
            for action in case["prefix"]:
                engine.step(state, action)
            arrays.append(ObservationBuilder(engine).static_actor(state, 0))
        np.testing.assert_array_equal(*arrays)
        assert history_weight(case, hands1)[0] > 0
        ps = [bid_likelihood(view1, a)[0] for a in case["actions"]]
        assert abs(sum(ps) - 1) < 1e-12


def test_every_legal_action_replays_and_scoring_is_frozen():
    # Hand-written scoring facts: making a higher sub-13 bid earns no extra points.
    rules = RulesConfig()
    assert score_deal(7, 10, rules) == (10, 0)
    assert score_deal(10, 10, rules) == (10, 0)
    assert score_deal(11, 10, rules) == (-11, 3)
    assert score_deal(12, 13, rules) == (16, 0)
    assert score_deal(13, 13, rules) == (26, 0)
    case = select_cases()[-1]
    hands = sample_hands(case["own_hand"], np.random.default_rng(8_430_001))
    for action in case["actions"]:
        result, record = play_candidate(case, hands, action, "all_strong", verify=True)
        assert record is not None
        if action != 59:
            assert result[0] in (16, 26)
            assert result[1:3] == [1, 1]


def test_paired_contrasts_and_support_classification():
    values = np.zeros((64, 3, 7))
    values[:, :, 1:3] = 1
    values[:, :, 0] = np.arange(64)[:, None] + np.array([0, 5, -2])
    report = summarize(values, np.ones(64), [59, 52, 53], 53, 8_430_002)
    assert report["classification"] == "raise_favored"
    assert report["teacher_status"] == "contradicted"
    assert report["actions"][1]["advantage_vs_pass"] == 5
    assert report["actions"][1]["advantage_simultaneous_95ci"][0] > 4.999
    # A constant worldwise difference must survive paired resampling.
    assert CONFIG["worlds_per_state"] == 64


def test_frozen_neural_continuation_and_recurrent_reset():
    import torch

    from tarneeb.auction_value_audit import SOURCE
    from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

    torch.set_num_threads(1)
    actor = load_recurrent_checkpoint(SOURCE, load_critic=False).actor.eval().requires_grad_(False)
    tensors = {k: v.clone() for k, v in actor.state_dict().items()}
    case = select_cases()[23]
    hands = sample_hands(case["own_hand"], np.random.default_rng(8_430_003))
    first, _ = play_candidate(case, hands, 59, "pfsp_vs_strong", actor, verify=True)
    play_candidate(case, hands, 58, "pfsp_vs_strong", actor, verify=True)
    repeated, _ = play_candidate(case, hands, 59, "pfsp_vs_strong", actor, verify=True)
    assert first == repeated
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in tensors.items())
