"""Read saved supervision, outcomes and frozen features; never optimize or simulate."""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import compare_phase11b_audit_policies as prior

from tarneeb.auction_repair.data import PublicExamples, frozen_features
from tarneeb.core.engine import TarneebEngine
from tarneeb.env.spaces import ACTOR_FEATURE_SLICES
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

OUT = Path("results/phase11b/specific_decision_diagnosis_v2")
OLD = Path("results/phase11b/neural_policy_comparison_v2")
SMOKE = Path("results/phase11b/auction_bc_smoke_s8300001_v1")
FOCUS = {"C7OS": 53, "C9PS": 55, "O8": 58}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(name, obj):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def examples(path):
    with np.load(path, allow_pickle=False) as d:
        return PublicExamples(
            *(d[k].copy() for k in ("statics", "histories", "lengths", "legal", "labels")),
            tuple(json.loads(str(d["metadata_json"]))),
        )


def hist(labels):
    return dict(sorted(Counter(map(int, labels)).items()))


def score_arithmetic(b, t):
    if b == 13:
        return (26, 0) if t == 13 else (-16, 2 * (13 - t))
    if t >= b:
        return (16 if t == 13 else t, 0)
    return (-b, 16 if t == 0 else 13 - t)


def coverage(train, rows, inputs):
    result = {}
    groups = defaultdict(list)
    for i, m in enumerate(train.metadata):
        owner = (
            "opening"
            if m["current_high_bid"] is None
            else ("partner" if m["high_bidder_is_partner"] else "opponent")
        )
        groups[f"{owner}:{m['current_high_bid']}:{m['case_family']}"].append(i)
    result["smoke_groups"] = {
        k: {"n": len(ix), "labels": hist(train.labels[ix])} for k, ix in groups.items()
    }
    result["smoke_labels"] = hist(train.labels)
    result["focus"] = {}
    for key, target in FOCUS.items():
        case = rows[key]["case"]
        obs = np.array(inputs[key]["observation"], dtype=np.float32)
        same_hand = (train.statics[:, :52] == obs[:52]).all(axis=1)
        same_context = np.array(
            [
                m["current_high_bid"] == case["high_bid"]
                and (
                    case["owner"] == "none"
                    or m["high_bidder_is_partner"] == (case["owner"] == "partner")
                )
                for m in train.metadata
            ]
        )
        same_prefix = np.array([m["prefix"] == case["prefix"] for m in train.metadata])
        near = np.array([abs(m["estimate"] - case["estimate"]) <= 0.25 for m in train.metadata])
        exact_static = (train.statics == obs).all(axis=1)
        exact = exact_static & np.array(
            [
                int(train.lengths[i]) == inputs[key]["event_count"]
                and train.histories[i, : train.lengths[i]].tolist() == inputs[key]["event_ids"]
                for i in range(len(train))
            ]
        )
        masks = {
            "exact_input": exact,
            "same_hand": same_hand,
            "same_hand_context": same_hand & same_context,
            "same_hand_prefix": same_hand & same_prefix,
            "context_estimate_within_0.25": same_context & near,
            "context": same_context,
        }
        result["focus"][key] = {
            k: {
                "n": int(mask.sum()),
                "target_labels": int((train.labels[mask] == target).sum()),
                "labels": hist(train.labels[mask]),
                "indices": np.flatnonzero(mask).tolist(),
            }
            for k, mask in masks.items()
        }
    # Public action sequences suffice for auction ownership and level counts.
    # Do NOT reconstruct the original hidden hands from seeds.
    d = np.load("data/bc/bc_train.npz", allow_pickle=False)
    assert 20250910 not in d["match_seeds"]
    original = defaultdict(list)
    offset = 0
    for dealer, length in zip(d["dealer_seats"], d["action_lengths"], strict=True):
        active = set(range(4))
        seat = (int(dealer) + 1) % 4
        high = None
        owner = None
        for a in d["actions"][offset : offset + int(length)]:
            a = int(a)
            if not 52 <= a <= 59:
                break
            role = (
                "opening"
                if high is None
                else ("partner" if owner == (seat + 2) % 4 else "opponent")
            )
            original[f"{role}:{high}"].append(a)
            if a == 59:
                active.remove(seat)
            else:
                assert high is None or a - 45 > high
                high, owner = a - 45, seat
            if len(active) <= 1:
                break
            seat = next((seat + j) % 4 for j in range(1, 5) if (seat + j) % 4 in active)
        offset += int(length)
    assert offset == len(d["actions"])
    result["original_bc"] = {
        "metadata": json.loads(str(d["metadata"])),
        "groups": {k: {"n": len(v), "labels": hist(v)} for k, v in original.items()},
        "hand_similarity": "unavailable: hands not stored; no seeded deal replay performed",
    }
    logs = {}
    for condition in ("pfsp", "latest_only"):
        p = Path(f"results/phase11/phase11_a3_{condition}_s0.log")
        s = p.read_bytes().decode("utf-8", errors="replace").replace("\x00", "")
        obj = ast.literal_eval(s[s.index("{") :].strip())
        logs[condition] = {
            k: v
            for k, v in obj.items()
            if k not in ("updates", "evaluations", "checkpoint_paths", "development_evaluations")
        }
        # Retain exact available keys; the log does not contain observations or trajectories.
        logs[condition]["top_level_keys"] = list(obj)
    result["phase11_logs"] = logs
    return result


def mechanism():
    answer = {}
    for key in FOCUS:
        records = [
            read(prior.recovery.ROOT / "states" / key / f"world_{i:03d}.json") for i in range(64)
        ]
        for r in records:
            assert prior.recovery.digest(r["payload"]) == r["sha256"]
        payloads = [r["payload"] for r in records]
        actions = payloads[0]["actions"]
        weights = np.array([p["weight"] for p in payloads])
        weights /= weights.sum()
        answer[key] = {}
        for policy in payloads[0]["outcomes"]:
            answer[key][policy] = {}
            for action in [59, FOCUS[key]] if key != "O8" else [59, 52, 53, 58]:
                ai = actions.index(action)
                vals = np.array([p["outcomes"][policy][ai] for p in payloads])
                replays = [p["replays"][policy][ai] for p in payloads]
                tricks = []
                leads = []
                for v, r in zip(vals, replays, strict=True):
                    sign = 1 if int(v[4]) % 2 == 0 else -1
                    candidates = [
                        t
                        for t in range(14)
                        if (t >= v[3]) == bool(v[1])
                        and sign * (score_arithmetic(v[3], t)[0] - score_arithmetic(v[3], t)[1])
                        == v[0]
                    ]
                    assert len(candidates) == 1
                    tricks.append(candidates[0])
                    if r is None:
                        continue
                    dr = r["deal_result"]
                    assert v[0] == sign * (
                        dr["score_delta_bidding_team"] - dr["score_delta_defending_team"]
                    )
                    t, b = dr["tricks_bidding_team"], dr["bid"]
                    # Independent arithmetic under frozen rules, not an engine replay.
                    expected = (
                        (26, 0)
                        if b == 13 and t == 13
                        else (
                            (-16, 2 * (13 - t))
                            if b == 13
                            else (
                                (16 if t == 13 else t, 0)
                                if t >= b
                                else (-b, 16 if t == 0 else 13 - t)
                            )
                        )
                    )
                    assert expected == (
                        dr["score_delta_bidding_team"],
                        dr["score_delta_defending_team"],
                    )
                    plays = [e for e in r["events"] if e["event_type"] == 4]
                    assert len(plays) == 52
                    assert plays[0]["actor_seat"] == v[4]
                    assert sum(e["event_type"] == 5 for e in r["events"]) == 13
                    wins = sum(
                        e["event_type"] == 5 and e["trick_winner_seat"] % 2 == int(v[4]) % 2
                        for e in r["events"]
                    )
                    assert wins == t
                    assert t == candidates[0]
                    leads.append(plays[0]["card"])

                def weighted_dist(values, weights=weights):
                    return {
                        str(int(v)): float(weights[np.array(values) == v].sum())
                        for v in sorted(set(values))
                    }

                answer[key][policy][str(action)] = {
                    "mean_margin": float(weights @ vals[:, 0]),
                    "uniform_mean_margin": float(vals[:, 0].mean()),
                    "success": float(weights @ vals[:, 1]),
                    "mean_tricks": float(weights @ np.array(tricks)),
                    "declarer_counts": hist(vals[:, 4]),
                    "trump_counts": hist(vals[:, 5]),
                    "trump_weighted": weighted_dist(vals[:, 5]),
                    "bid_counts": hist(vals[:, 3]),
                    "tricks_counts": hist(tricks),
                    "lead_counts": hist(leads),
                    "world0": {
                        "outcome": vals[0].tolist(),
                        "deal_result": replays[0]["deal_result"],
                        "first_card": leads[0],
                    },
                    "score_arithmetic_checks": 64,
                    "available_event_traces_checked": len(leads),
                    "tricks_source": "unique inverse of stored margin, bid, success and frozen scoring; directly checked against world-0 events",
                }
    return answer


def features(actor, inputs):
    out = []
    with torch.no_grad():
        for x in inputs.values():
            ids = torch.tensor(x["event_ids"], dtype=torch.long).reshape(1, -1, 7)
            f, _ = actor.trunk(
                torch.tensor([x["observation"]]),
                ids,
                torch.tensor([x["event_count"]]),
                actor.initial_hidden_state(1),
            )
            out.append(f[0])
    return torch.stack(out)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if OUT.exists():
        raise FileExistsError(OUT)
    protected = {str(p): prior.sha(p) for p in Path("results/phase11b").rglob("*") if p.is_file()}
    protected.update({str(p): prior.sha(p) for p in prior.MODELS.values()})
    protected.update(
        {
            n: prior.sha(Path(n))
            for n in read(prior.recovery.ROOT / "preregistration.json")["source_hashes"]
        }
    )
    protected["data/bc/bc_train.npz"] = prior.sha("data/bc/bc_train.npz")
    OUT.mkdir()
    save("preservation_before.json", protected)
    rows = {r["case"]["id"]: r for r in read(OLD / "comparison.json")}
    inputs = read(OLD / "inputs.json")
    ids = list(inputs)
    train = examples(SMOKE / "train_examples.npz")
    save("coverage.json", coverage(train, rows, inputs))
    print("coverage complete", flush=True)
    save("mechanism.json", mechanism())
    print("saved-record mechanism complete", flush=True)
    results = {}
    all_features = {}
    with (
        prior.checkpoint_locks(),
        patch.object(TarneebEngine, "deal", side_effect=AssertionError("No dealing")),
        patch.object(TarneebEngine, "step", side_effect=AssertionError("No simulation")),
    ):
        for name, path in prior.MODELS.items():
            actor = (
                load_recurrent_checkpoint(path, load_critic=False)
                .actor.cpu()
                .eval()
                .requires_grad_(False)
            )
            tensors = {k: v.clone() for k, v in actor.state_dict().items()}
            f = features(actor, inputs)
            all_features[name] = f.numpy()
            aug = torch.cat([f, torch.ones(len(f), 1)], dim=1)
            logits = actor.policy_head(f).detach()
            result = {
                "states": {},
                "feature_rank": int(torch.linalg.matrix_rank(aug.double())),
                "feature_dimension": 256,
            }
            train_f = None
            if name == "pfsp":
                train_f, _ = frozen_features(actor, train)
                train_aug = torch.cat([train_f, torch.ones(len(train_f), 1)], dim=1)
                probs = (
                    actor.policy_head(train_f)
                    .masked_fill(~torch.from_numpy(train.legal), -torch.inf)
                    .softmax(-1)
                )
                residual = probs - torch.nn.functional.one_hot(torch.from_numpy(train.labels), 65)
            for key, target in FOCUS.items():
                i = ids.index(key)
                x = inputs[key]
                legal = torch.tensor(x["legal_action_mask"])
                winner = int(logits[i].masked_fill(~legal, -torch.inf).argmax())
                competitor = 59 if key != "O8" else int(logits[i, 52:58].argmax()) + 52
                gap = float(logits[i, competitor] - logits[i, target])
                obs = torch.tensor([x["observation"]], requires_grad=True)
                event = torch.tensor(x["event_ids"], dtype=torch.long).reshape(1, -1, 7)
                hidden = (
                    actor.trunk.history_encoder.advance(
                        event, torch.tensor([x["event_count"]]), actor.initial_hidden_state(1)
                    )
                    .detach()
                    .requires_grad_(True)
                )
                sf = actor.trunk.static_encoder(obs)
                fused = actor.trunk.fusion(torch.cat([sf, hidden[-1]], dim=1))
                z = actor.policy_head(fused)
                gobs, gh = torch.autograd.grad(z[0, target] - z[0, competitor], (obs, hidden))
                saved_logits = rows[key]["models"][name]["legal_logits"]
                assert all(
                    abs(float(logits[i, a]) - saved_logits[prior.action(a)]) < 1e-5
                    for a in np.flatnonzero(legal)
                )
                cosine = (aug @ aug[i]) / (aug.norm(dim=1) * aug[i].norm())
                correction = max(gap + 0.1, 0)
                transfer = correction * (aug @ aug[i]) / float(aug[i].square().sum())
                details = {
                    "target": target,
                    "winner": winner,
                    "competitor": competitor,
                    "gap": gap,
                    "legal_logits": saved_logits,
                    "target_probability": float(
                        logits[i].masked_fill(~legal, -torch.inf).softmax(-1)[target]
                    ),
                    "feature_norm": float(f[i].norm()),
                    "active_features": int((f[i] > 0).sum()),
                    "readout_bias_gap": float(
                        actor.policy_head.bias[competitor] - actor.policy_head.bias[target]
                    ),
                    "feature_gap": float(
                        (actor.policy_head.weight[competitor] - actor.policy_head.weight[target])
                        @ f[i]
                    ),
                    "minimum_pair_row_l2_to_tie": max(gap, 0) / (2**0.5 * float(aug[i].norm())),
                    "minimum_pair_row_linf_to_tie": max(gap, 0) / (2 * float(aug[i].abs().sum())),
                    "local_input_gradient_block_norms": {
                        k: float(gobs[0, s].norm()) for k, s in ACTOR_FEATURE_SLICES.items()
                    },
                    "local_hidden_gradient_norm": float(gh.norm()),
                    "cosine_with_controls": {
                        c: float(cosine[ids.index(c)]) for c in ("C9OW", "C9PW", "C7OS", "O8")
                    },
                    "hypothetical_minimum_norm_pair_correction": {
                        "definition": "No parameter changes: exact linear-head margin gain from direction proportional to target feature, scaled to source target margin +0.1",
                        "margin_gain": {
                            c: float(transfer[ids.index(c)]) for c in ("C9OW", "C9PW", "C7OS", "O8")
                        },
                        "control_target_minus_competitor_after": {
                            c: float(
                                logits[ids.index(c), target]
                                - logits[ids.index(c), competitor]
                                + transfer[ids.index(c)]
                            )
                            for c in ("C9OW", "C9PW", "C7OS", "O8")
                        },
                    },
                }
                if train_f is not None:
                    # Negative CE-gradient direction, expressed as change of target-minus-competitor.
                    contribution = (residual[:, competitor] - residual[:, target]) * (
                        train_aug @ aug[i]
                    )
                    details["mean_sgd_margin_derivative"] = float(contribution.mean())
                    details["sgd_derivative_by_label"] = {
                        str(a): float(
                            contribution[torch.from_numpy(train.labels == a)].sum() / len(train)
                        )
                        for a in sorted(set(train.labels))
                    }
                    dist = (train_f - f[i]).norm(dim=1)
                    nearest = dist.argsort()[:8].tolist()
                    details["nearest_smoke_features"] = [
                        {
                            "index": j,
                            "distance": float(dist[j]),
                            "label": int(train.labels[j]),
                            "metadata": train.metadata[j],
                        }
                        for j in nearest
                    ]
                result["states"][key] = details
            assert all(torch.equal(v, actor.state_dict()[k]) for k, v in tensors.items())
            results[name] = result
            print(name, "features/sensitivities complete", flush=True)
    save("features.json", results)
    with (OUT / "fused_features.npz").open("xb") as stream:
        np.savez_compressed(stream, state_ids=np.array(ids), **all_features)
    assert np.array_equal(all_features["pfsp"], all_features["smoke"])
    changed = [n for n, h in protected.items() if prior.sha(n) != h]
    assert not changed, changed
    save(
        "verification.json",
        {
            "protected_files": len(protected),
            "changed": changed,
            "actor_tensors_unchanged": True,
            "pfsp_smoke_features_identical": True,
            "new_samples": 0,
            "simulated_actions": 0,
            "optimizer_steps": 0,
            "source_sha256": prior.sha(__file__),
        },
    )
    print("all checks complete", flush=True)


if __name__ == "__main__":
    main()
