"""Opt-in matched pilot; default is a read-only preflight. Never promotes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tarneeb.agents.heuristic_basic import BasicHeuristicAgent
from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.evaluation.a1 import compare_paired, evaluate_condition
from tarneeb.league.population import LeaguePopulation
from tarneeb.league.registry import PolicyRegistry
from tarneeb.models.recurrent_checkpoint import load_recurrent_checkpoint
from tarneeb.training import phase11_training as frozen
from tarneeb.training import seating_experiment_v1 as seating
from tarneeb.training.ppo import PPOConfig

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "checkpoints/phase11/phase11_a3_latest_only_s1_final_u250.pt"
PARENT_SHA = "afea7a4f90a7b3feb1ed7c72611b0e0aa6dcd6737b883a95d7aaf4fe4c5dd1c7"
CHAMPION = ROOT / frozen.PHASE11_INITIAL_CHAMPION[1]
OUT = ROOT / "results/balanced_seating_pilot_v1"
CHECKPOINTS = ROOT / "checkpoints/balanced_seating_pilot_v1"
MODES = ("legacy", "balanced")
PAIRS, EVALUATION_SEED, BOOTSTRAP_SEED = 150, 4_000_000, 18031


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def configuration(mode):
    if mode not in MODES:
        raise ValueError("Unknown seating arm")
    return frozen.Phase11SmokeConfig(
        run_name=f"seating_v1_{mode}", condition="latest_only",
        environment_count=16, rollout_length=128, total_updates=20,
        seed_index=3, action_sampling_seed=18032, minibatch_shuffle_seed=18033,
        opponent_seed=18034, checkpoint_directory=CHECKPOINTS / mode,
        results_directory=OUT / mode,
        ppo=PPOConfig(learning_rate=3e-5),
    )


def preflight():
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("Parent checkpoint SHA256 mismatch")
    loaded = load_recurrent_checkpoint(PARENT)
    if loaded.critic is None or loaded.critic.spec.critic_observation_dim != 856:
        raise ValueError("Centralized parent critic required")
    load_recurrent_checkpoint(CHAMPION, load_critic=False)
    sources = (Path(__file__), Path(frozen.__file__), Path(seating.__file__),
               ROOT / "docs/BALANCED_SEATING_PILOT_V1.md")
    return {"parent_sha256": PARENT_SHA, "champion_sha256": sha(CHAMPION),
            "sources": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "arms": {mode: configuration(mode).as_dict() for mode in MODES},
            "evaluation": {"pairs_per_opponent": PAIRS, "seed": EVALUATION_SEED,
                           "bootstrap_seed": BOOTSTRAP_SEED,
                           "opponents": ["parent", "basic", "strong", "champion"],
                           "status": "reused development evidence; not sealed confirmation"}}


def bind_driver(network_factory, collector_factory):
    """Reuse the exact frozen loop with isolated factory bindings, never patch it.

    Function globals are copied, so historical callers keep their original
    factories. All PPO, collection, checkpoint and report algorithms are reused.
    """
    original = frozen.run_phase11_smoke
    namespace = dict(original.__globals__)
    namespace.update(build_phase11_networks=network_factory,
                     build_phase11_collector=collector_factory,
                     _capture_phase11_resume_state=seating.capture_resume)
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)


def run_arm(mode, plan):
    cfg = configuration(mode)
    cfg.results_directory.mkdir()  # exclusive: parent reserved by train_both
    cfg.checkpoint_directory.mkdir()
    registry = PolicyRegistry()
    registry.register_checkpoint("parent", PARENT, source_phase="phase11", category="snapshot")
    population = LeaguePopulation(registry=registry, rng=np.random.default_rng(18035))
    population.add("parent")
    exposures = []

    def networks(_config):
        if sha(PARENT) != plan["parent_sha256"]:
            raise ValueError("Parent changed")
        loaded = load_recurrent_checkpoint(PARENT)
        return loaded.actor, loaded.critic  # weights only; no historical optimizer/resume

    def collector_factory(*args):
        collector, generator, cache = seating.build_collector(*args, mode=mode)
        collect = collector.collect

        def recorded_collect():
            buffer, metrics = collect()
            if metrics.illegal_action_count or metrics.no_trump_action_count:
                raise AssertionError("Illegal rollout action")
            auction = [t for seq in buffer.sequences.values() for t in seq
                       if t.legal_action_mask[59]]
            exposures.append({"update": len(exposures) + 1, "auction_decisions": len(auction),
                              "opening_decisions": sum(t.previous_event_length == 0 for t in auction),
                              "bid_pass_counts": {str(a): sum(t.action == a for t in auction)
                                                  for a in range(52, 60)}})
            return buffer, metrics

        collector.collect = recorded_collect
        return collector, generator, cache

    driver = bind_driver(networks, collector_factory)
    result = driver(cfg, population, resume_from=None)
    if (result.completed_updates != 20 or result.illegal_action_count
            or result.non_finite_update_count or result.worst_epoch_zero_deviation > 1e-4):
        raise AssertionError("Pilot correctness gate failed")
    frozen.write_phase11_smoke_report(cfg, result, population)
    final = cfg.checkpoint_directory / f"{cfg.run_name}_final_u20.pt"
    write_new(cfg.results_directory / "complete.json",
              {"plan": plan, "final_sha256": sha(final), "exposures": exposures})


def train_both():
    plan = preflight()
    if OUT.exists() or CHECKPOINTS.exists():
        raise FileExistsError("Pilot outputs already exist; no overwrite or automatic resume")
    OUT.mkdir(parents=True, exist_ok=False)
    CHECKPOINTS.mkdir(parents=True, exist_ok=False)
    write_new(OUT / "plan.json", plan)
    torch.set_num_threads(1)
    for mode in MODES:  # exception stops sequence; never retries
        run_arm(mode, plan)
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("Parent changed during pilot")


def evaluate():
    plan = preflight()
    if json.loads((OUT / "plan.json").read_text()) != plan:
        raise ValueError("Plan/source identity changed")
    paths = {}
    for mode in MODES:
        cfg = configuration(mode)
        receipt = json.loads((cfg.results_directory / "complete.json").read_text())
        paths[mode] = cfg.checkpoint_directory / f"{cfg.run_name}_final_u20.pt"
        if receipt["plan"] != plan or sha(paths[mode]) != receipt["final_sha256"]:
            raise ValueError("Incomplete or modified arm")
    destination = OUT / "evaluation"
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(1)

    def neural(path, name):
        actor = load_recurrent_checkpoint(path, load_critic=False).actor
        return RecurrentNeuralAgent(actor.eval().requires_grad_(False), name=name, deterministic=True)

    opponents = {"parent": lambda: neural(PARENT, "parent"), "basic": BasicHeuristicAgent,
                 "strong": StrongHeuristicAgent, "champion": lambda: neural(CHAMPION, "champion")}
    with torch.inference_mode():
        for name, factory in opponents.items():
            records = {}
            for mode in MODES:
                ev = evaluate_condition(neural(paths[mode], mode), factory(), name, mode,
                                        PAIRS, EVALUATION_SEED)
                if (len(ev.paired_scores) != PAIRS or ev.metrics["illegal_action_rate"] != 0
                        or ev.metrics["no_trump_action_rate"] != 0
                        or ev.metrics["replay_integrity_rate"] != 1):
                    raise AssertionError("Evaluation correctness gate failed")
                write_new(destination / f"{mode}_{name}.json", asdict(ev))
                records[mode] = ev
            comparison = compare_paired(records["balanced"], records["legacy"], BOOTSTRAP_SEED)
            write_new(destination / f"difference_{name}.json", asdict(comparison))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument("--train-both", action="store_true")
    flags.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.train_both:
        train_both()
    elif args.evaluate:
        evaluate()
    else:
        print(json.dumps(preflight(), indent=2))


if __name__ == "__main__":
    main()
