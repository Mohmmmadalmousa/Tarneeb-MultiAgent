"""Isolated Phase 11B counterfactual diagnostics. No optimizer or weight updates."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tarneeb.agents.heuristic_strong import StrongHeuristicAgent, best_trump_candidate
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.auction_repair.data import public_teacher_label, strength_bucket
from tarneeb.core.cards import card_name, hand_from_cards
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.recorder import GameRecorder, ReplayEngine
from tarneeb.core.rules import RulesConfig
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.evaluation.arena import MAX_DECISIONS_PER_DEAL
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

SOURCE = Path("checkpoints/phase11/phase11_a3_pfsp_s0_final_u250.pt")
SOURCE_SHA = "317fd320fa33e98f435f1eb4b790b3ad5804635f59a6e86c73d8c35fda23ba7a"
OUTPUT = Path("results/phase11b/counterfactual_value_audit_v1")
CONFIG = {
    "worlds_per_state": 64,
    "selection_seed": 8_400_000,
    "world_seed_start": 8_410_000,
    "bootstrap_seed_start": 8_420_000,
    "bootstrap_resamples": 4000,
    "workers": 4,
    "torch_threads_per_worker": 1,
    "cpu_seconds_per_state": 60,
    "maximum_simulation_cpu_seconds": 1920,
    "wall_seconds": 900,
    "minimum_ess": 32,
    "maximum_normalized_weight": 0.1,
    "bid_temperature": 1.0,
    "bid_uniform_mixture": 0.1,
    "maximum_lapse_responsibility": 0.5,
    "continuations": ["pfsp_vs_strong", "all_strong"],
    "beliefs": ["bid_conditioned", "uniform"],
    "score": "seat-0 partnership score minus opposing partnership score, current deal",
    "classification": "Simultaneous within-state 95% paired bootstrap bands; strict sign. Robust only if all four policy/belief combinations agree and posterior support gates pass.",
    "teacher_classification": "Supported if every alternative's advantage upper bound <=0; contradicted if any lower bound >0; otherwise uncertain. Robust across all four combinations and support gates.",
    "all_pass": "Zero current-deal score, matching frozen Arena; no future redeal value included.",
    "scope": "Purposive mechanism cases, not a strength benchmark or independent smoke confirmation. No training.",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def make_state(case: dict[str, Any], hands: list[int]):
    """Only supplied hypothetical hands enter the simulation; no original deal exists."""
    engine = TarneebEngine(RulesConfig())
    state = engine.new_deal(case["dealer"], match_seed=case["world_seed"])
    # This is precisely the frozen deal initialization, with injected sampled cards.
    # No seeded engine deal is generated or inspected to choose an information set.
    state.phase = Phase.BID
    state.current_player_seat = (state.dealer_seat + 1) % 4
    state.active_bidders = set(range(4))
    state.hands = hands.copy()
    state.original_hands = hands.copy()
    return engine, state


def sample_hands(own_hand: list[int], rng: np.random.Generator) -> list[int]:
    assert len(own_hand) == len(set(own_hand)) == 13
    unseen = np.array(sorted(set(range(52)) - set(own_hand)))
    rng.shuffle(unseen)
    return [hand_from_cards(own_hand)] + [
        hand_from_cards(unseen[i : i + 13]) for i in range(0, 39, 13)
    ]


def decision_view(case: dict[str, Any], hands: list[int]):
    engine, state = make_state(case, hands)
    for action in case["prefix"]:
        engine.step(state, action)
    assert state.phase is Phase.BID and state.current_player_seat == 0
    return PublicViewBuilder(engine).build(state, 0)


def select_cases() -> list[dict[str, Any]]:
    """Selection uses own cards and heuristic strength only, before simulation."""
    rng = np.random.default_rng(CONFIG["selection_seed"])
    strong = [
        [6, 7, 8, 9, 10, 11, 12, 25, 38, 50, 13, 26, 39],
        [6, 7, 8, 9, 10, 11, 12, 25, 38, 51, 13, 26, 39],
        [*range(4, 13), 25, 38, 39, 40],
        [*range(4, 13), 25, 38, 39, 51],
        [*range(3, 13), 25, 38, 51],
        list(range(13)),
    ]
    weak = []
    while len(weak) < 8:
        hand = sorted(rng.choice(52, 13, replace=False).tolist())
        case = {"dealer": 3, "world_seed": 8_400_100, "prefix": []}
        # Deterministic placeholders, never used as observed hidden cards.
        unseen = sorted(set(range(52)) - set(hand))
        hands = [hand_from_cards(hand)] + [
            hand_from_cards(unseen[i : i + 13]) for i in range(0, 39, 13)
        ]
        if best_trump_candidate(decision_view(case, hands))[1] < 5:
            weak.append(hand)
    cases = []
    for high in range(7, 13):
        for owner in ("opponent", "partner"):
            if owner == "opponent":
                prefix = [high + 45] if high % 2 else [59, high + 45]
            else:
                prefix = [52, high + 45, 59] if high in (10, 12) else [high + 45, 59]
            for strength, hand in (("weak", weak[high - 7]), ("strong", strong[high - 7])):
                rotation = (high - 7) % 4
                hand = sorted((c + rotation * 13) % 52 for c in hand)
                cases.append(
                    {
                        "id": f"C{high}{owner[0].upper()}{strength[0].upper()}",
                        "high_bid": high,
                        "owner": owner,
                        "selected_strength": strength,
                        "own_hand": hand,
                        "prefix": prefix,
                        "hand_origin": "uniform draw filtered weak"
                        if strength == "weak"
                        else "constructed coverage template",
                    }
                )
    medium = [8, 9, 10, 11, 12, 13, 17, 25, 26, 38, 39, 41, 43]
    opening = [
        weak[6],
        weak[7],
        medium,
        sorted((c + 13) % 52 for c in medium),
        strong[0],
        strong[1],
        strong[3],
        strong[5],
    ]
    for i, hand in enumerate(opening):
        cases.append(
            {
                "id": f"O{i + 1}",
                "high_bid": None,
                "owner": "none",
                "selected_strength": [
                    "weak",
                    "weak",
                    "medium",
                    "medium",
                    "strong",
                    "strong",
                    "exceptional",
                    "exceptional",
                ][i],
                "own_hand": hand,
                "prefix": [59] * (i % 3),
                "hand_origin": "uniform draw filtered weak"
                if i < 2
                else "constructed coverage template",
            }
        )
    for i, case in enumerate(cases):
        case.update(
            index=i,
            dealer=(-1 - len(case["prefix"])) % 4,
            world_seed=CONFIG["world_seed_start"] + i,
            bootstrap_seed=CONFIG["bootstrap_seed_start"] + i,
        )
        unseen = sorted(set(range(52)) - set(case["own_hand"]))
        hands = [hand_from_cards(case["own_hand"])] + [
            hand_from_cards(unseen[j : j + 13]) for j in range(0, 39, 13)
        ]
        view = decision_view(case, hands)
        suit, estimate = best_trump_candidate(view)
        case.update(
            teacher=public_teacher_label(view),
            estimate=estimate,
            heuristic_strength=strength_bucket(estimate),
            teacher_trump=suit,
            cards=[card_name(c) for c in case["own_hand"]],
            actions=[59] + [int(a) for a in np.flatnonzero(view.legal_action_mask) if a != 59],
        )
        assert view.current_high_bid == case["high_bid"]
        if case["owner"] != "none":
            assert (view.current_high_bidder % 2 == 0) == (case["owner"] == "partner")
    return cases


def bid_likelihood(view, action: int) -> tuple[float, float]:
    """Explicit approximate behavior model, NOT the teacher under evaluation.

    Utility(bid b)=-abs(b-(estimate+1.5)); utility(PASS)=-max(0,target-threshold).
    Threshold is current bid or 6 when unopened. Temperature 1; 10% legal-uniform
    lapse mixture. No partner convention is presumed. The second return value is
    the posterior responsibility of the lapse component for the observed action.
    """
    target = best_trump_candidate(view)[1] + 1.5
    threshold = view.current_high_bid if view.current_high_bid is not None else 6
    actions = np.flatnonzero(view.legal_action_mask)
    utility = np.array(
        [-max(0, target - threshold) if a == 59 else -abs(a - 45 - target) for a in actions]
    )
    z = utility / CONFIG["bid_temperature"]
    p = np.exp(z - z.max())
    p /= p.sum()
    floor = CONFIG["bid_uniform_mixture"] / len(actions)
    probability = floor + (1 - CONFIG["bid_uniform_mixture"]) * p[list(actions).index(action)]
    return float(probability), float(floor / probability)


def history_weight(case: dict[str, Any], hands: list[int]) -> tuple[float, float]:
    engine, state = make_state(case, hands)
    views = PublicViewBuilder(engine)
    weight, responsibilities = 1.0, []
    for action in case["prefix"]:
        assert state.current_player_seat != 0
        p, lapse = bid_likelihood(views.build(state, state.current_player_seat), action)
        weight *= p
        if action != 59:
            responsibilities.append(lapse)
        engine.step(state, action)
    return weight, max(responsibilities, default=0.0)


def play_candidate(case, hands, action, policy, actor=None, verify=False):
    engine, state = make_state(case, hands)
    views = PublicViewBuilder(engine)
    strong = StrongHeuristicAgent()
    neural = None
    if policy == "pfsp_vs_strong":
        if actor is None:
            raise ValueError("PFSP continuation needs its frozen actor")
        neural = RecurrentNeuralAgent(actor, deterministic=True, seed=case["world_seed"])
        assert neural.states_are_separate()
    recorder = GameRecorder.start(state) if verify else None
    forced = [*case["prefix"], action]
    decisions = 0
    while state.phase in (Phase.BID, Phase.DECLARE, Phase.PLAY):
        seat = state.current_player_seat
        view = views.build(state, seat)
        agent = neural if neural is not None and seat % 2 == 0 else strong
        # Also consume the actor's own history before externally forced actions.
        proposed = agent.act(view)
        selected = forced[decisions] if decisions < len(forced) else proposed
        assert selected != 64 and view.legal_action_mask[selected]
        _, events, _ = engine.step(state, selected)
        if recorder is not None:
            recorder.record_step(selected, events)
        decisions += 1
        assert decisions <= MAX_DECISIONS_PER_DEAL
    passed = state.phase is Phase.DEAL
    if recorder is not None and not passed:
        assert ReplayEngine(RulesConfig()).verify(recorder.finish(state))
    if passed:
        return [0, -1, -1, -1, -1, -1, 1], None
    result = state.deal_result
    assert result is not None and state.declarer_seat is not None
    sign = 1 if state.declarer_seat % 2 == 0 else -1
    margin = sign * (result.score_delta_bidding_team - result.score_delta_defending_team)
    return [
        margin,
        int(result.tricks_bidding_team >= result.bid),
        int(sign == 1),
        result.bid,
        state.declarer_seat,
        state.selected_trump_suit,
        0,
    ], recorder.finish(state).to_dict() if recorder else None


def summarize(values, weights, actions, teacher, seed):
    """Paired importance-weighted bootstrap. Bands cover all action contrasts per state.

    Uniformly resample original proposal-world indices, then renormalize their
    likelihood weights. Never treat posterior resampling duplicates as new worlds.
    """
    w = weights / weights.sum()
    scores = values[:, :, 0]
    means = w @ scores
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(w), size=(CONFIG["bootstrap_resamples"], len(w)))
    bw = weights[ix]
    bw /= bw.sum(axis=1, keepdims=True)
    bm = np.einsum("bn,bna->ba", bw, scores[ix])
    error = bm - means
    radius = float(np.quantile(error.max(1) - error.min(1), 0.95))
    delta = means - means[0]
    lower, upper = delta - radius, delta + radius
    lower[0] = upper[0] = 0
    classification = (
        "raise_favored"
        if np.any(lower[1:] > 0)
        else "pass_favored"
        if np.all(upper[1:] < 0)
        else "uncertain"
    )
    ti = actions.index(teacher)
    other = np.arange(len(actions)) != ti
    teacher_status = (
        "contradicted"
        if np.any((means - means[ti] - radius)[other] > 0)
        else "supported"
        if np.all((means - means[ti] + radius)[other] <= 0)
        else "uncertain"
    )
    rows = []
    for j, action in enumerate(actions):
        valid = values[:, j, 1] >= 0
        own = values[:, j, 2] == 1

        def rate(mask, column=j):
            den = float(w[mask].sum())
            return float(w[mask] @ values[mask, column, 1] / den) if den else None

        rates = []
        for mask in (valid, own):
            denominator = np.sum(bw * mask[ix], axis=1)
            numerator = np.sum(bw * mask[ix] * values[ix, j, 1], axis=1)
            good = denominator > 0
            rates.append(
                np.quantile(numerator[good] / denominator[good], [0.025, 0.975]).tolist()
                if good.any()
                else None
            )
        rows.append(
            {
                "action": action,
                "mean_margin": float(means[j]),
                "margin_95ci": np.quantile(bm[:, j], [0.025, 0.975]).tolist(),
                "advantage_vs_pass": float(delta[j]),
                "advantage_simultaneous_95ci": [float(lower[j]), float(upper[j])],
                "contract_success": rate(valid),
                "contract_success_95ci": rates[0],
                "own_contract_probability": float(w[own].sum()),
                "own_contract_success": rate(own),
                "own_contract_success_95ci": rates[1],
                "passed_out_probability": float(w @ values[:, j, 6]),
                "final_bid_distribution": {
                    str(b): float(w[values[:, j, 3] == b].sum()) for b in range(7, 14)
                },
                "teacher_advantage": float(means[j] - means[ti]),
                "teacher_advantage_simultaneous_95ci": [
                    float(means[j] - means[ti] - radius),
                    float(means[j] - means[ti] + radius),
                ]
                if j != ti
                else [0.0, 0.0],
            }
        )
    best = int(np.argmax(means))
    return {
        "classification": classification,
        "teacher_status": teacher_status,
        "teacher_agrees_empirical_best": bool(means[ti] >= means.max() - 1e-12),
        "best_action": actions[best],
        "band_radius": radius,
        "actions": rows,
    }


_ACTOR = None


def initialize_worker():
    global _ACTOR
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert sha(SOURCE) == SOURCE_SHA
    _ACTOR = (
        load_recurrent_checkpoint(SOURCE, load_critic=False)
        .actor.cpu()
        .eval()
        .requires_grad_(False)
    )


def run_case(case):
    started = time.process_time()
    rng = np.random.default_rng(case["world_seed"])
    worlds = [sample_hands(case["own_hand"], rng) for _ in range(CONFIG["worlds_per_state"])]
    # Information-boundary check across all sampled worlds, before any rollout.
    encodings = []
    for hands in worlds:
        engine, state = make_state(case, hands)
        for a in case["prefix"]:
            engine.step(state, a)
        encodings.append(ObservationBuilder(engine).static_actor(state, 0))
    assert all(np.array_equal(encodings[0], x) for x in encodings)
    evidence = np.array([history_weight(case, h) for h in worlds])
    weights = evidence[:, 0]
    w = weights / weights.sum()
    ess = float(1 / (w @ w))
    lapse = float(w @ evidence[:, 1])
    supported = (
        ess >= CONFIG["minimum_ess"]
        and w.max() <= CONFIG["maximum_normalized_weight"]
        and lapse <= CONFIG["maximum_lapse_responsibility"]
    )
    raw, summaries, records = {}, {}, []
    for policy in CONFIG["continuations"]:
        values = []
        for wi, hands in enumerate(worlds):
            row = []
            for action in case["actions"]:
                if time.process_time() - started > CONFIG["cpu_seconds_per_state"]:
                    raise TimeoutError(f"{case['id']} reached its preregistered CPU cap; no rerun")
                result, record = play_candidate(case, hands, action, policy, _ACTOR, verify=wi == 0)
                row.append(result)
                if record is not None:
                    records.append(record)
            values.append(row)
        values = np.array(values, dtype=float)
        raw[policy] = values.tolist()
        summaries[policy] = {
            belief: summarize(
                values,
                weights if belief == "bid_conditioned" else np.ones(len(weights)),
                case["actions"],
                case["teacher"],
                case["bootstrap_seed"],
            )
            for belief in CONFIG["beliefs"]
        }
    all_summaries = [s for p in summaries.values() for s in p.values()]
    classes = {s["classification"] for s in all_summaries}
    teacher_classes = {s["teacher_status"] for s in all_summaries}
    return {
        "case": case,
        "worlds": worlds,
        "weights": weights.tolist(),
        "lapse_responsibilities": evidence[:, 1].tolist(),
        "ess": ess,
        "max_weight": float(w.max()),
        "weighted_lapse_responsibility": lapse,
        "posterior_support_gate": bool(supported),
        "robust_classification": next(iter(classes))
        if supported and len(classes) == 1
        else "uncertain",
        "robust_teacher_status": next(iter(teacher_classes))
        if supported and len(teacher_classes) == 1
        else "uncertain",
        "summaries": summaries,
        "raw_outcomes": raw,
        "replay_records": records,
        "cpu_seconds": time.process_time() - started,
    }
