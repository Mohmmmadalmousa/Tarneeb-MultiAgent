"""Synthetic-only tests. NEVER load training checkpoints or execute a simulation."""

import pytest
import torch

from tarneeb.auction_repair.retention_objective import (
    ObjectiveWeights,
    PairwiseEvidence,
    loss_terms,
    retention_masks,
)
from tarneeb.auction_repair.train_evidence import SOURCE, RowBinding, _seal


def authenticated(base, legal, labels, comparisons):
    # Synthetic issuer fixture only; production uses load_train_evidence().
    return _seal(
        [PairwiseEvidence(row, preferred, other, SOURCE) for row, preferred, other in comparisons],
        [
            RowBinding(f"synthetic:{i}", tuple(mask.nonzero().flatten().tolist()), int(label))
            for i, (mask, label) in enumerate(zip(legal, labels, strict=True))
        ],
        (("fixture", "synthetic"),),
    )


def bound_loss(x, base, legal, labels, evidence, ids=None, config=None):
    if ids is None:
        ids = list(range(len(x)))
    return loss_terms(
        x,
        base,
        legal,
        labels,
        evidence,
        row_ids=torch.tensor(ids),
        row_fingerprints=[f"synthetic:{i}" for i in ids],
        config=config,
    )


def batch():
    base = torch.zeros((4, 65), dtype=torch.float32)
    legal = torch.zeros_like(base, dtype=torch.bool)
    legal[:, [53, 55, 59]] = True
    # Old teacher-agreeing bid, old teacher-agreeing PASS, missed supported BID10,
    # and a teacher-agreeing BID10 that is already successful.
    base[0, 53], base[0, 59] = 3.0, 0.0
    base[1, 59], base[1, 53] = 3.0, 0.0
    base[2, 59], base[2, 55] = 8.0, -2.0
    base[3, 55], base[3, 59] = 3.0, 0.0
    labels = torch.tensor([53, 59, 59, 55], dtype=torch.long)
    return base, legal, labels


def test_masks_protect_only_old_teacher_agreements():
    base, legal, labels = batch()
    bid, passing = retention_masks(base, legal, labels)
    assert bid.tolist() == [True, False, False, True]
    assert passing.tolist() == [False, True, True, False]


def test_new_supported_bid_can_override_old_heuristic_pass():
    base, legal, labels = batch()
    changed = base.clone().requires_grad_(True)
    e = authenticated(base, legal, labels, [(2, 55, 59)])
    terms = bound_loss(changed, base, legal, labels, e)
    assert terms["supported_pair_count"].item() == 1
    assert terms["pass_anchor_count"].item() == 1  # row 2 is excluded
    terms["total"].backward()
    assert changed.grad[2, 55].item() < 0  # preferred BID10 gets pushed up
    assert changed.grad[2, 59].item() > 0  # PASS is pushed down
    assert torch.all(changed.grad[~legal] == 0)


def test_bid_retention_reacts_to_loss_of_previously_correct_bid():
    base, legal, labels = batch()
    changed = base.clone()
    changed[0, 53] = 0.0
    changed[0, 59] = 2.0
    terms = loss_terms(changed, base, legal, labels)
    assert terms["bid_retention"].item() > 0
    assert terms["supported_pair_count"].item() == 0


def test_pass_retention_reacts_to_unwanted_bidding():
    base, legal, labels = batch()
    changed = base.clone()
    changed[1, 53] = 4.0
    terms = loss_terms(changed, base, legal, labels)
    assert terms["pass_retention"].item() > 0


def test_refuses_v3_leakage_and_invalid_comparisons():
    base, legal, labels = batch()
    for e in (
        PairwiseEvidence(2, 55, 59, "v3_evaluation"),
        PairwiseEvidence(2, 56, 59, "v2_counterfactual_supported"),
        PairwiseEvidence(2, 55, 55, "v2_counterfactual_supported"),
    ):
        with pytest.raises(ValueError):
            loss_terms(base, base, legal, labels, [e])


def test_nontrivial_auction_actions_only():
    base, legal, labels = batch()
    legal[0, 52] = True
    legal[0, 53] = False
    with pytest.raises(ValueError):
        loss_terms(base, base, legal, labels)


def test_no_mutation_of_source_data():
    base, legal, labels = batch()
    raw, mask, target = base.clone(), legal.clone(), labels.clone()
    x = base.clone().requires_grad_(True)
    terms = loss_terms(x, base, legal, labels, config=ObjectiveWeights())
    terms["total"].backward()
    assert torch.equal(base, raw)
    assert torch.equal(legal, mask)
    assert torch.equal(labels, target)


def test_minibatch_preserves_original_rows_and_rejects_wrong_identity():
    base, legal, labels = batch()
    e = authenticated(base, legal, labels, [(2, 55, 59)])
    ids = [2, 0]
    x = base[ids].clone().requires_grad_(True)
    terms = bound_loss(x, base[ids], legal[ids], labels[ids], e, ids)
    assert terms["supported_pair_count"] == 1
    terms["supported_pair"].backward()
    assert x.grad[0, 55] < 0 and x.grad[0, 59] > 0
    assert not x.grad[1].any()
    with pytest.raises(ValueError, match="fingerprint"):
        loss_terms(
            x,
            base[ids],
            legal[ids],
            labels[ids],
            e,
            row_ids=torch.tensor(ids),
            row_fingerprints=["synthetic:0", "synthetic:2"],
        )
    with pytest.raises(ValueError, match="requires original"):
        loss_terms(x, base[ids], legal[ids], labels[ids], e)


def test_bid13_over_bid12_has_no_conflicting_teacher_or_retention_loss():
    base = torch.zeros((1, 65), dtype=torch.double)
    base[0, 57] = 3
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:, [55, 56, 57, 58, 59]] = True
    labels = torch.tensor([57])
    e = authenticated(base, mask, labels, [(0, 58, 57)])
    x = base.clone().requires_grad_(True)
    terms = bound_loss(x, base, mask, labels, e)
    for key in ("heuristic_ce", "bid_retention", "pass_retention", "bid_anchor_count"):
        assert terms[key] == 0
    terms["total"].backward()
    assert x.grad[0, 58] < 0 and x.grad[0, 57] > 0
    assert not x.grad[~mask].any()


def test_pair_normalization_per_case_and_gradient_correctness():
    base, legal, labels = batch()
    base = base.double()
    e = authenticated(base, legal, labels, [(0, 53, 59), (0, 55, 59), (1, 59, 55)])
    ids = [0, 1]
    x = base[ids].clone().requires_grad_(True)
    terms = bound_loss(x, base[ids], legal[ids], labels[ids], e, ids)
    softplus = torch.nn.functional.softplus
    expected = (
        (softplus(0.1 - (x[0, 53] - x[0, 59])) + softplus(0.1 - (x[0, 55] - x[0, 59]))) / 2
        + softplus(0.1 - (x[1, 59] - x[1, 55]))
    ) / 2
    assert torch.allclose(terms["supported_pair"], expected)
    assert terms["supported_case_count"] == 2 and terms["supported_pair_count"] == 3
    assert torch.autograd.gradcheck(
        lambda value: bound_loss(value, base[ids], legal[ids], labels[ids], e, ids)["total"], (x,)
    )


def test_unaffected_retention_and_ce_are_preserved():
    base, legal, labels = batch()
    e = authenticated(base, legal, labels, [(2, 55, 59)])
    x = base.clone()
    x[0, 59] = 4  # Damage a protected bid on an unaffected row.
    x[1, 55] = 4  # Damage a protected PASS on an unaffected row.
    terms = bound_loss(x, base, legal, labels, e)
    reference = loss_terms(x[[0, 1, 3]], base[[0, 1, 3]], legal[[0, 1, 3]], labels[[0, 1, 3]])
    for key in (
        "bid_retention",
        "pass_retention",
        "heuristic_ce",
        "bid_anchor_count",
        "pass_anchor_count",
    ):
        assert torch.allclose(terms[key], reference[key])
    assert terms["bid_retention"] > 0 and terms["pass_retention"] > 0


def test_authenticated_bundle_cannot_be_changed_or_replaced_by_a_source_string():
    from dataclasses import replace

    base, legal, labels = batch()
    e = authenticated(base, legal, labels, [(2, 55, 59)])
    with pytest.raises(ValueError, match="modified"):
        bound_loss(base, base, legal, labels, replace(e, pairs=()))
    with pytest.raises(ValueError, match="not authenticated"):
        loss_terms(base, base, legal, labels, [PairwiseEvidence(2, 55, 59, SOURCE)])


def test_pass_preference_gradient_and_illegal_pair_rejection():
    base, legal, labels = batch()
    e = authenticated(base, legal, labels, [(1, 59, 55)])
    x = base.clone().requires_grad_(True)
    terms = bound_loss(x, base, legal, labels, e)
    terms["supported_pair"].backward()
    assert x.grad[1, 59] < 0 and x.grad[1, 55] > 0
    for pairs in ([(2, 56, 59)], [(2, 55, 55)], [(2, 55, 59), (2, 55, 59)]):
        with pytest.raises(ValueError):
            bound_loss(base, base, legal, labels, authenticated(base, legal, labels, pairs))


def test_separate_components_normalize_per_case_and_have_independent_weights():
    base, legal, labels = batch()
    pairs = [(0, 55, 59), (0, 53, 59), (0, 55, 53), (1, 59, 55)]
    e = authenticated(base, legal, labels, pairs)
    x = base.double().requires_grad_(True)
    config = ObjectiveWeights(
        heuristic_ce=0, bid_retention=0, pass_retention=0, bid_pass=0.2, bid_bid=0.3
    )
    terms = bound_loss(x, base.double(), legal, labels, e, config=config)
    sp = torch.nn.functional.softplus
    expected_pass = (
        (sp(0.1 - (x[0, 55] - x[0, 59])) + sp(0.1 - (x[0, 53] - x[0, 59]))) / 2
        + sp(0.1 - (x[1, 59] - x[1, 55]))
    ) / 2
    expected_bid = sp(0.1 - (x[0, 55] - x[0, 53]))
    assert torch.allclose(terms["bid_pass"], expected_pass)
    assert torch.allclose(terms["bid_bid"], expected_bid)
    assert torch.allclose(terms["total"], 0.2 * expected_pass + 0.3 * expected_bid)
    assert terms["bid_pass_pair_count"] == 3 and terms["bid_pass_case_count"] == 2
    assert terms["bid_bid_pair_count"] == 1 and terms["bid_bid_case_count"] == 1
    assert torch.autograd.gradcheck(
        lambda v: bound_loss(v, base.double(), legal, labels, e, config=config)["total"], (x,)
    )


def test_split_changes_conflicting_aggregate_gradient_after_adam():
    from scripts.check_phase11b_split_objective import first_adam_displacement

    base = torch.zeros((1, 65), dtype=torch.double)
    base[0, 59] = 10
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask[:, 55:60] = True
    labels = torch.tensor([57])
    e = authenticated(base, mask, labels, [(0, a, 59) for a in range(55, 59)] + [(0, 58, 57)])
    x = base.clone().requires_grad_(True)
    new = bound_loss(x, base, mask, labels, e)
    old = bound_loss(x, base, mask, labels, e, config=ObjectiveWeights(supported_pair=0.02))
    (new_grad,) = torch.autograd.grad(new["total"], x, retain_graph=True)
    (old_grad,) = torch.autograd.grad(old["total"], x)
    # Legacy raises push both bids up; split ranking can push BID12 down.
    assert old_grad[0, 57] < 0 and old_grad[0, 58] < 0
    assert new_grad[0, 57] > 0 and new_grad[0, 58] < 0
    (delta,) = first_adam_displacement((new_grad,))
    assert delta[0, 58] - delta[0, 57] > 5e-5
    assert not delta[~mask].any()
    assert new["heuristic_ce"] == new["bid_retention"] == new["pass_retention"] == 0


@pytest.mark.parametrize(
    "pairs",
    [
        [(2, 55, 59), (2, 59, 55)],
        [(2, 55, 59), (2, 59, 53), (2, 53, 55)],
    ],
)
def test_conflicting_preferences_rejected_even_outside_minibatch(pairs):
    base, legal, labels = batch()
    e = authenticated(base, legal, labels, pairs)
    with pytest.raises(ValueError, match="Conflicting"):
        bound_loss(base[:1], base[:1], legal[:1], labels[:1], e, ids=[0])


def test_empty_bid_rank_component_does_not_change_raise_or_anchor_gradients():
    from dataclasses import replace

    base, legal, labels = batch()
    e = authenticated(base, legal, labels, [(2, 55, 59)])
    x = base.clone().requires_grad_(True)
    cfg = ObjectiveWeights()
    a = bound_loss(x, base, legal, labels, e, config=cfg)
    b = bound_loss(x, base, legal, labels, e, config=replace(cfg, bid_bid=10))
    assert a["bid_bid"] == b["bid_bid"] == 0
    assert torch.equal(a["total"], b["total"])
    assert torch.equal(torch.autograd.grad(a["total"], x)[0], torch.autograd.grad(b["total"], x)[0])


def test_adam_preview_is_pure_clipped_and_includes_epsilon():
    from scripts.check_phase11b_split_objective import first_adam_displacement

    g = torch.tensor([3.0, 4.0, 0.0], dtype=torch.double)
    original = g.clone()
    (delta,) = first_adam_displacement((g,), lr=0.1, clip=0.5, eps=0.01)
    clipped = g * (0.5 / (5 + 1e-6))
    # Explicit first-step Adam moment/bias correction; no optimizer step.
    m, v = 0.1 * clipped, 0.001 * clipped.square()
    expected = -0.1 * (m / 0.1) / ((v / 0.001).sqrt() + 0.01)
    assert torch.allclose(delta, expected)
    assert torch.equal(g, original) and delta[2] == 0
    with pytest.raises(ValueError, match="Nonfinite"):
        first_adam_displacement((torch.tensor([float("nan")]),))


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_invalid_split_weights_fail_closed(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ObjectiveWeights(bid_bid=value)
