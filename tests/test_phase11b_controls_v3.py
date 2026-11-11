"""Own-hand preparation checks only: no deals, continuations or neural execution."""

import json
from itertools import permutations

import numpy as np
import pytest
import torch
from scripts import prepare_phase11b_controls_v3 as v3

from tarneeb.core.events import Phase


@pytest.fixture(autouse=True)
def forbid_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dataset preparation may not simulate or execute neural models")

    monkeypatch.setattr(v3.TarneebEngine, "deal", forbidden)
    monkeypatch.setattr(v3.audit, "sample_hands", forbidden)
    monkeypatch.setattr(v3.audit, "play_candidate", forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    step = v3.TarneebEngine.step

    def auction_only(self, state, action):
        assert state.phase is Phase.BID and 52 <= action <= 59
        return step(self, state, action)

    monkeypatch.setattr(v3.TarneebEngine, "step", auction_only)


@pytest.fixture(scope="module")
def references():
    return v3.load_references()


@pytest.fixture
def dataset(references):
    return v3.select(references)


def test_suit_permutation_distance_matches_independent_sets():
    first = [0, 1, 2, 10, 12, 14, 20, 25, 30, 37, 39, 45, 51]
    second = [1, 3, 4, 9, 11, 15, 21, 24, 31, 36, 40, 46, 50]
    brute = min(
        13 - len(set(first) & {p[c // 13] * 13 + c % 13 for c in second})
        for p in permutations(range(4))
    )
    assert v3.nearest(first, [("second", v3.suit_masks(second))])["minimum_replacements"] == brute
    for p in permutations(range(4)):
        rotated = [p[c // 13] * 13 + c % 13 for c in first]
        assert (
            v3.nearest(first, [("rotation", v3.suit_masks(rotated))])["minimum_replacements"] == 0
        )


def test_exact_input_identity_includes_history_and_mask():
    static = np.zeros(692, dtype=np.float32)
    history = np.zeros((1, 7), dtype=np.int64)
    legal = np.ones(65, dtype=bool)
    baseline = v3.fingerprint(static, history, legal)
    history[0, 2] = 1
    assert v3.fingerprint(static, history, legal) != baseline
    history[0, 2] = 0
    legal[0] = False
    assert v3.fingerprint(static, history, legal) != baseline


def test_semantic_identity_is_independent_of_integer_storage_width():
    static = np.zeros(692, dtype=np.float32)
    history = np.array([[1, 3, 3, 0, 0, 1, 0], [2, 4, 0, 0, 0, 1, 0]], dtype=np.int32)
    legal = np.ones(65, dtype=bool)
    assert v3.history_hash(history) == v3.history_hash(history.astype(np.int64))
    assert v3.fingerprint(static, history, legal) == v3.fingerprint(
        static, history.astype(np.int64), legal
    )


def test_saved_cases_and_inputs_survive_verification_fix(references, dataset):
    if not (v3.ROOT / "cases.json").exists():
        pytest.skip("Saved dataset verification applies after preparation")
    assert dataset["cases"] == v3.read(v3.ROOT / "cases.json")
    assert dataset["inputs"] == v3.read(v3.ROOT / "inputs.json")
    assert dataset["overlap"]["V3_01"]["public_history_only_matches"] > 0
    assert all(x["exact_actor_observation_matches"] == 0 for x in dataset["overlap"].values())


def test_reproducible_first_qualifying_cases(references, dataset):
    assert json.dumps(dataset, sort_keys=True) == json.dumps(v3.select(references), sort_keys=True)


def test_legality_categories_public_history_and_no_value_labels(dataset):
    assert len(dataset["cases"]) == 16
    assert [c["category"] for c in dataset["cases"]] == [k for k in v3.CATEGORIES for _ in range(4)]
    for c in dataset["cases"]:
        assert len(c["own_hand"]) == len(set(c["own_hand"])) == 13
        assert all(0 <= card < 52 for card in c["own_hand"])
        assert v3.category_accepts(c["own_hand"], c["category"])
        assert c["classification"] is None and c["teacher_evidence"] == "heuristic_only"
        assert c["actions"] == (
            [59, 55, 56, 57, 58] if c["index"] < 8 else [59, 52, 53, 54, 55, 56, 57, 58]
        )
        assert c["teacher"] in c["actions"]
        view, static, history, legal = v3.encode(c)
        assert view.observer_seat == 0
        assert c["index"] >= 8 or view.current_high_bidder == 2
        assert static.shape == (692,) and static[:52].sum() == 13
        assert len(history) == len(c["prefix"])
        assert np.isin(history[:, 0], [1, 2]).all() and not history[:, 4].any()
        assert np.flatnonzero(legal).tolist() == sorted(c["actions"])
    assert dataset["statistical_results"] is None


def test_old_and_internal_family_separation(references, dataset):
    for c in dataset["cases"]:
        for refs in references[0].values():
            assert v3.nearest(c["own_hand"], refs)["minimum_replacements"] >= 3
        for other in dataset["cases"]:
            if other["id"] != c["id"]:
                assert (
                    v3.nearest(c["own_hand"], [(other["id"], v3.suit_masks(other["own_hand"]))])[
                        "minimum_replacements"
                    ]
                    >= 3
                )
        assert dataset["overlap"][c["id"]]["input_sha256"] not in references[1]


def test_hidden_placeholder_invariance(dataset):
    for c in dataset["cases"]:
        first, *a = v3.encode(c)
        second, *b = v3.encode(c, reverse=True)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)
        assert first.events == second.events
        assert v3.public_teacher_label(first) == v3.public_teacher_label(second)


def test_exhaustion_and_output_overwrite_fail_closed(monkeypatch, tmp_path, references):
    monkeypatch.setitem(v3.DESIGN, "max_attempts_per_case", 0)
    with pytest.raises(RuntimeError, match="do not relax"):
        v3.select(references)
    monkeypatch.setattr(v3, "ROOT", tmp_path)
    with pytest.raises(FileExistsError, match="never overwrite"):
        v3.prepare()


def test_reserved_seeds_disjoint_and_never_heldout(references):
    seeds = [
        seed
        for key in ("selection_seeds", "reserved_world_seeds", "reserved_bootstrap_seeds")
        for seed in v3.DESIGN[key]
    ]
    assert len(seeds) == len(set(seeds)) == 48
    assert 20250910 not in seeds and not set(seeds) & references[2]
