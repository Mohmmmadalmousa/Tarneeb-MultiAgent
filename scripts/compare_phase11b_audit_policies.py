"""Inference on saved audit information sets only; no dealing or continuations."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import sys
import time
from ctypes import wintypes
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tarneeb import auction_value_audit as audit
from tarneeb import auction_value_recovery as recovery
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.core.cards import cards_in_hand
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicViewBuilder
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint

OUTPUT = Path("results/phase11b/neural_policy_comparison_v2")
MODELS = {
    "pfsp": Path("checkpoints/phase11/phase11_a3_pfsp_s0_final_u250.pt"),
    "smoke": Path("checkpoints/phase11b/phase11b_auction_bc_smoke_s8300001_v1.pt"),
    "latest_only": Path("checkpoints/phase11/phase11_a3_latest_only_s0_final_u250.pt"),
    "phase9": Path("checkpoints/phase9_gru_bc_s0_update250.pt"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    with (OUTPUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def action(value):
    return "PASS" if value == 59 else f"BID {value - 45}"


@contextlib.contextmanager
def checkpoint_locks():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = []
    try:
        for path in MODELS.values():
            handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 1, None, 3, 0, None)
            if handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            kernel.CloseHandle(handle)


def saved_view(plan, case, result, index):
    """Replay only the existing public BID/PASS prefix using a saved world."""
    record = recovery.read(recovery.world_path(recovery.ROOT, case, index))
    payload = record["payload"]
    assert recovery.digest(payload) == record["sha256"] == result["world_sha256"][index]
    assert payload["plan_identity"] == plan["identity"]
    assert payload["state_identity"] == recovery.digest(case)
    assert payload["world_index"] == index and payload["actions"] == case["actions"]
    assert list(cards_in_hand(payload["hands"][0])) == sorted(case["own_hand"])
    engine, state = audit.make_state(case, payload["hands"])
    acting_seats = []
    for selected in case["prefix"]:
        assert state.phase is Phase.BID and state.current_player_seat != 0
        acting_seats.append(state.current_player_seat)
        engine.step(state, selected)
    assert state.phase is Phase.BID and state.current_player_seat == 0
    view = PublicViewBuilder(engine).build(state, 0)
    obs = ObservationBuilder(engine).static_actor(state, 0)
    assert hashlib.sha256(obs.tobytes()).hexdigest() == payload["observation_sha256"]
    assert view.current_high_bid == case["high_bid"]
    assert sorted(np.flatnonzero(view.legal_action_mask).tolist()) == sorted(case["actions"])
    owner = (
        "none"
        if view.current_high_bidder is None
        else ("partner" if view.current_high_bidder == 2 else "opponent")
    )
    assert owner == case["owner"]
    return view, obs, acting_seats


@torch.inference_mode()
def infer(actor, view, comparison_view, expected_obs):
    agent = RecurrentNeuralAgent(actor, deterministic=True, seed=0)
    assert agent.states_are_separate()
    assert agent.consumed_event_count(0) == 0 and torch.count_nonzero(agent.hidden_state(0)) == 0
    obs, ids, length = agent.encode(view)
    alternate = agent.encode(comparison_view)
    assert all(torch.equal(x, y) for x, y in zip((obs, ids, length), alternate, strict=True))
    assert np.array_equal(obs.numpy()[0], expected_obs)
    assert int(length[0]) == len(view.events)
    hidden = agent.hidden_state(0)
    logits, advanced = actor(obs, ids, length, hidden)
    mask = torch.from_numpy(view.legal_action_mask).unsqueeze(0)
    distribution = MaskedCategorical.from_logits(logits, mask)
    probs = distribution.probabilities[0]
    selected = int(distribution.most_probable_action()[0])
    assert agent.act(view) == selected
    assert torch.equal(agent.hidden_state(0), advanced)
    assert agent.consumed_event_count(0) == len(view.events)
    streamed = hidden.clone()
    for i in range(int(length[0])):
        streamed = actor.trunk.history_encoder.advance(
            ids[:, i : i + 1], torch.ones_like(length), streamed
        )
    drift = float((streamed - advanced).abs().max())
    assert torch.allclose(streamed, advanced, atol=1e-6, rtol=1e-6)
    assert float(probs[~mask[0]].max()) == 0 and bool(mask[0, selected])
    bids = [a for a in range(52, 59) if view.legal_action_mask[a]]
    return {
        "action": selected,
        "action_label": action(selected),
        "pass_probability": float(probs[59]),
        "legal_bid_probabilities": {str(a - 45): float(probs[a]) for a in bids},
        "legal_logits": {action(a): float(logits[0, a]) for a in [59, *bids]},
        "best_legal_bid": action(max(bids, key=lambda a: float(logits[0, a]))),
        "pass_minus_best_legal_bid_logit": float(logits[0, 59] - logits[0, bids].max()),
        "advanced_hidden": advanced.tolist(),
        "streamed_hidden_max_abs_difference": drift,
        "production_agent_agrees": True,
        "illegal_probability": 0.0,
    }, {
        "observation": obs[0].tolist(),
        "event_ids": ids[0, : int(length[0])].tolist(),
        "event_count": int(length[0]),
        "legal_action_mask": view.legal_action_mask.tolist(),
    }


def main():
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan = recovery.read(recovery.ROOT / "preregistration.json")
    recovery.verify_plan(plan)
    completion = recovery.read(recovery.ROOT / "completion.json")
    assert completion["states"] == 32 and completion["plan_identity"] == plan["identity"]
    historical = recovery.read(Path("results/phase11b/diagnosis_20261014_v4/diagnosis.json"))
    expected = {
        "pfsp": audit.SOURCE_SHA,
        "smoke": "407fb22dc2c90c848952abbb5fdc201661de7459c2296b72c1083b4732fa0b0e",
        "latest_only": historical["provenance"]["latest_s0"]["sha256"],
        "phase9": historical["provenance"]["phase9_s0"]["sha256"],
    }
    protected = {str(p): sha(p) for p in Path("results/phase11b").rglob("*") if p.is_file()}
    protected.update({n: sha(Path(n)) for n in plan["source_hashes"]})
    protected.update({str(p): sha(p) for p in MODELS.values()})
    assert all(protected[str(p)] == expected[k] for k, p in MODELS.items())
    OUTPUT.mkdir()
    write(
        "plan.json",
        {
            "audit_plan_identity": plan["identity"],
            "states": 32,
            "models": {k: {"path": str(p), "sha256": expected[k]} for k, p in MODELS.items()},
            "hidden_samples_generated": 0,
            "continuations_run": 0,
            "source_sha256": sha(Path(__file__)),
            "saved_world_indices_for_reconstruction": [0, 1],
            "threads": 1,
            "supersedes": "neural_policy_comparison_v1: O5/O6 had unsorted hand lists; compare identical card sets in canonical order",
            "method": "Each model uses its own encoder from zero seat-0 memory; replay saved public prefix only",
        },
    )
    write("preservation_before.json", protected)
    rows, inputs = [], {}
    started = time.monotonic()
    with checkpoint_locks(), contextlib.ExitStack() as stack:
        for obj, name in [
            (audit, "sample_hands"),
            (audit, "play_candidate"),
            (TarneebEngine, "deal"),
        ]:
            stack.enter_context(
                patch.object(
                    obj, name, side_effect=AssertionError("Forbidden new simulation/sample")
                )
            )
        actors = {
            k: load_recurrent_checkpoint(p, load_critic=False)
            .actor.cpu()
            .eval()
            .requires_grad_(False)
            for k, p in MODELS.items()
        }
        tensors = {k: {n: v.clone() for n, v in a.state_dict().items()} for k, a in actors.items()}
        for case, result_hash in zip(plan["cases"], completion["result_sha256"], strict=True):
            result = recovery.read(recovery.ROOT / "states" / case["id"] / "result.json")
            assert recovery.digest(result) == result_hash
            assert result["completed_worlds"] == 64 and result["state_identity"] == recovery.digest(
                case
            )
            row = {
                "case": case,
                "classification": result["robust_classification"],
                "teacher_status": result["robust_teacher_status"],
                "support_gate": result["posterior_support_gate"],
                "audit_summaries": result["summaries"],
                "models": {},
            }
            try:
                view, obs, seats = saved_view(plan, case, result, 0)
                other, other_obs, _ = saved_view(plan, case, result, 1)
                assert np.array_equal(obs, other_obs) and view.events == other.events
                row.update(
                    available=True,
                    public_events=[asdict(e) for e in view.events],
                    prefix_seats=seats,
                )
                for name, actor in actors.items():
                    prediction, encoded = infer(actor, view, other, obs)
                    prediction["agrees_teacher"] = prediction["action"] == case["teacher"]
                    selected_rows = [
                        next(a for a in s["actions"] if a["action"] == prediction["action"])
                        for p in result["summaries"].values()
                        for s in p.values()
                    ]
                    prediction["selected_bid_supported_vs_pass"] = bool(
                        row["support_gate"]
                        and prediction["action"] != 59
                        and all(a["advantage_simultaneous_95ci"][0] > 0 for a in selected_rows)
                    )
                    row["models"][name] = prediction
                    if case["id"] not in inputs:
                        inputs[case["id"]] = encoded
                    assert inputs[case["id"]] == encoded
            except (AssertionError, ValueError, KeyError) as exc:
                row.update(available=False, unavailable_reason=repr(exc), models={})
            rows.append(row)
        assert all(
            torch.equal(v, actors[k].state_dict()[n])
            for k, t in tensors.items()
            for n, v in t.items()
        )
    write("inputs.json", inputs)
    write("comparison.json", rows)
    summaries = {}
    for name in MODELS:
        valid = [r for r in rows if r["available"]]
        summaries[name] = {
            "available": len(valid),
            "teacher_agreement": sum(r["models"][name]["agrees_teacher"] for r in valid),
            "missed_supported_competitive_raises": [
                r["case"]["id"]
                for r in valid
                if r["case"]["high_bid"] is not None
                and r["classification"] == "raise_favored"
                and r["models"][name]["action"] == 59
            ],
            "wrong_supported_pass": [
                r["case"]["id"]
                for r in valid
                if r["classification"] == "pass_favored" and r["models"][name]["action"] != 59
            ],
            "missed_supported_opening_bids": [
                r["case"]["id"]
                for r in valid
                if r["case"]["high_bid"] is None
                and r["classification"] == "raise_favored"
                and r["models"][name]["action"] == 59
            ],
            "competitive_bid_count": sum(
                r["case"]["high_bid"] is not None and r["models"][name]["action"] != 59
                for r in valid
            ),
        }
    changed = [n for n, h in protected.items() if sha(Path(n)) != h]
    assert not changed, changed
    write(
        "verification.json",
        {
            "existing_artifacts_and_source_unchanged": True,
            "files_checked": len(protected),
            "all_model_tensors_bit_identical": True,
            "new_hidden_samples": 0,
            "continuations": 0,
            "wall_seconds": time.monotonic() - started,
            "available_states": sum(r["available"] for r in rows),
            "production_agent_comparisons": sum(len(r["models"]) for r in rows),
        },
    )
    write("summary.json", summaries)
    print(json.dumps(summaries))
    for row in rows:
        print(
            json.dumps(
                {
                    "state": row["case"]["id"],
                    "available": row["available"],
                    "models": {
                        k: {
                            f: v[f]
                            for f in (
                                "action_label",
                                "pass_probability",
                                "pass_minus_best_legal_bid_logit",
                            )
                        }
                        for k, v in row["models"].items()
                    },
                }
            )
        )


if __name__ == "__main__":
    main()
