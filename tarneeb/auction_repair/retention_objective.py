"""Phase 11B experimental auction-head objective; no optimizer or simulation.

Review-only prototype. This module does NOT train, select, or export a model.
Information-set evidence and retention masks must be pinned outside this module
from training-only examples. V3 development cases must never be passed as labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import torch
import torch.nn.functional as F
from torch import Tensor

from tarneeb.auction_repair.train_evidence import (
    SOURCE,
    AuthenticatedTrainEvidence,
    PairwiseEvidence,
)


@dataclass(frozen=True)
class ObjectiveWeights:
    # These are PROTOTYPE coefficients, not an approved experiment configuration.
    heuristic_ce: float = 0.1
    # Explicit legacy mode for reproduction of the completed pilot only.
    # None selects the corrected, independently normalized components.
    supported_pair: float | None = None
    bid_retention: float = 1.0
    pass_retention: float = 1.0
    pair_margin: float = 0.1
    retention_margin: float = 0.1
    bid_pass: float = 0.02
    bid_bid: float = 0.02

    def __post_init__(self) -> None:
        if any(v is not None and (not isfinite(v) or v < 0) for v in vars(self).values()):
            raise ValueError("Objective coefficients must be finite and nonnegative")


def _check_preferences(evidence: AuthenticatedTrainEvidence) -> None:
    """Reject illegal, repeated or cyclic preferences, including outside a minibatch."""
    graphs: dict[int, dict[int, set[int]]] = {}
    for pair in evidence.pairs:
        if pair.source != SOURCE or not 0 <= pair.row < len(evidence.rows):
            raise ValueError("Invalid TRAIN evidence source or row")
        actions = evidence.rows[pair.row].actions
        if (
            pair.preferred not in actions
            or pair.other not in actions
            or not 52 <= pair.preferred <= 59
            or not 52 <= pair.other <= 59
        ):
            raise ValueError("Unsupported or illegal comparison")
        graph = graphs.setdefault(pair.row, {})
        children = graph.setdefault(pair.preferred, set())
        if pair.other in children:
            raise ValueError("Repeated evidence comparison")
        children.add(pair.other)

    def visit(graph: dict[int, set[int]], node: int, path: set[int]) -> None:
        if node in path:
            raise ValueError("Conflicting cyclic evidence preferences")
        for child in graph.get(node, ()):
            visit(graph, child, path | {node})

    for graph in graphs.values():
        for node in graph:
            visit(graph, node, set())


def _check_inputs(logits: Tensor, baseline: Tensor, legal: Tensor, labels: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] != 65:
        raise ValueError("Auction logits must have shape (N, 65)")
    n = logits.shape[0]
    if baseline.shape != logits.shape or legal.shape != logits.shape or labels.shape != (n,):
        raise ValueError("Inconsistent input shapes")
    if n == 0 or legal.dtype != torch.bool or labels.dtype != torch.long:
        raise ValueError("Empty batch, invalid legal-mask dtype, or label dtype")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(baseline).all()):
        raise ValueError("Nonfinite logits")
    if not bool(legal[:, 59].all() and legal.sum(1).ge(2).all()):
        raise ValueError("Expected nontrivial auction masks with legal PASS")
    if bool(legal[:, :52].any() or legal[:, 60:].any()):
        raise ValueError("Only BID/PASS actions allowed in this objective")
    if not bool(((labels >= 52) & (labels <= 59)).all()):
        raise ValueError("Invalid auction label")
    if not bool(legal.gather(1, labels[:, None]).all()):
        raise ValueError("Illegal target label")


def retention_masks(baseline: Tensor, legal: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Conservative *behavioral* anchors, not evidence of optimality.

    Protect baseline bids that agree with teacher; protect PASS when baseline
    and teacher both PASS. An audited positive row may be excluded from the
    PASS anchors only by a separately verified comparison in loss_terms().
    """
    greedy = baseline.masked_fill(~legal, -torch.inf).argmax(-1)
    correct = greedy == labels
    return correct & (labels != 59), correct & (labels == 59)


def _winner_margin(logits: Tensor, legal: Tensor, labels: Tensor, rows: Tensor) -> Tensor:
    if not bool(rows.any()):
        return logits.sum() * 0.0
    selected_logits = logits[rows]
    selected_legal = legal[rows]
    selected_labels = labels[rows]
    loser_legal = selected_legal.scatter(1, selected_labels[:, None], False)
    winner = selected_logits.gather(1, selected_labels[:, None]).squeeze(1)
    strongest_other = selected_logits.masked_fill(~loser_legal, -torch.inf).max(-1).values
    return winner - strongest_other


def loss_terms(
    logits: Tensor,
    baseline_logits: Tensor,
    legal_mask: Tensor,
    heuristic_labels: Tensor,
    evidence: AuthenticatedTrainEvidence | Sequence[PairwiseEvidence] = (),
    *,
    row_ids: Tensor | None = None,
    row_fingerprints: Sequence[str] | None = None,
    heuristic_weights: Tensor | None = None,
    config: ObjectiveWeights | None = None,
) -> dict[str, Tensor]:
    """Construct a transparent objective using TRAIN-only rows.

    Retention constraints monitor two separate classes instead of letting the
    majority PASS class dominate. Nonempty evidence must come from the verified
    TRAIN audit loader, with original archive row IDs and public fingerprints.
    Audited rows use only supported comparisons; their heuristic CE and exclusive
    teacher-retention targets are suppressed. Each case has equal loss mass within
    each component. BID/PASS includes both directions (including supported PASS).
    BID/BID is normalized separately so extra raise comparisons cannot dilute it.
    This deliberately changes the gradient, not just the names of loss terms:
    on the frozen 30-pair panel, each ranking pair has 20 times its legacy mass
    at coefficient .02. It remains a diagnostic, not a Phase 11 completion gate.
    """
    if config is None:
        config = ObjectiveWeights()
    _check_inputs(logits, baseline_logits, legal_mask, heuristic_labels)
    n = logits.shape[0]
    base = baseline_logits.detach()
    bid_anchor, pass_anchor = retention_masks(base, legal_mask, heuristic_labels)
    supported_terms: dict[int, list[Tensor]] = {}
    bid_pass_terms: dict[int, list[Tensor]] = {}
    bid_bid_terms: dict[int, list[Tensor]] = {}
    supported_rows = set()
    audited = torch.zeros(n, dtype=torch.bool, device=logits.device)
    if isinstance(evidence, AuthenticatedTrainEvidence):
        evidence.validate()
        _check_preferences(evidence)
        if row_ids is None or row_fingerprints is None:
            raise ValueError(
                "Authenticated TRAIN evidence requires original row IDs and fingerprints"
            )
        if row_ids.shape != (n,) or row_ids.dtype != torch.long or len(row_fingerprints) != n:
            raise ValueError("Invalid original TRAIN row identities")
        ids = row_ids.tolist()
        if len(set(ids)) != n or any(not 0 <= i < len(evidence.rows) for i in ids):
            raise ValueError("Duplicate or invalid original TRAIN rows")
        for local, original in enumerate(ids):
            binding = evidence.rows[original]
            if (
                row_fingerprints[local] != binding.fingerprint
                or int(heuristic_labels[local]) != binding.teacher
                or tuple(legal_mask[local].nonzero().flatten().tolist()) != binding.actions
            ):
                raise ValueError("TRAIN row fingerprint, teacher or legal mask mismatch")
        positions = {original: local for local, original in enumerate(ids)}
        comparisons = evidence.pairs
    else:
        if len(evidence):
            raise ValueError("Raw pair lists/source strings are not authenticated TRAIN evidence")
        positions, comparisons = {}, ()
    for e in comparisons:
        if e.source != SOURCE or not (0 <= e.row < len(evidence.rows)):
            raise ValueError("Invalid TRAIN evidence source or row")
        if not (52 <= e.preferred <= 59 and 52 <= e.other <= 59):
            raise ValueError("Invalid evidence row or action")
        if e.row not in positions:
            continue
        row = positions[e.row]
        if e.preferred == e.other or not bool(
            legal_mask[row, e.preferred] and legal_mask[row, e.other]
        ):
            raise ValueError("Unsupported or illegal comparison")
        if (e.row, e.preferred, e.other) in supported_rows:
            raise ValueError("Repeated evidence comparison")
        supported_rows.add((e.row, e.preferred, e.other))
        # Even a compatible teacher winner can penalize another supported bid.
        # Use directional evidence alone on this row, not an invented optimal label.
        audited[row] = True
        bid_anchor[row] = pass_anchor[row] = False
        term = F.softplus(config.pair_margin - (logits[row, e.preferred] - logits[row, e.other]))
        supported_terms.setdefault(row, []).append(term)
        component = bid_pass_terms if 59 in (e.preferred, e.other) else bid_bid_terms
        component.setdefault(row, []).append(term)

    norm = F.log_softmax(logits.masked_fill(~legal_mask, -torch.inf), dim=-1)
    per_row_ce = -norm.gather(1, heuristic_labels[:, None]).squeeze(1)
    if heuristic_weights is None:
        heuristic_weights = torch.full_like(per_row_ce, 1 / n)
    if heuristic_weights.shape != (n,) or not bool(torch.isfinite(heuristic_weights).all()):
        raise ValueError("Invalid heuristic weights")
    if not bool((heuristic_weights >= 0).all()) or float(heuristic_weights.sum()) <= 0:
        raise ValueError("Nonpositive heuristic mass")
    effective_weights = heuristic_weights.masked_fill(audited, 0)
    ce = (
        (per_row_ce * effective_weights).sum() / effective_weights.sum()
        if bool(effective_weights.sum() > 0)
        else logits.sum() * 0.0
    )

    def retention_term(anchors: Tensor) -> Tensor:
        if not bool(anchors.any()):
            return logits.sum() * 0.0
        # Bound the desired historical decision margin, so a once-overconfident
        # policy is not forced to retain its original excessive confidence.
        old_margin = _winner_margin(base, legal_mask, heuristic_labels, anchors).clamp_min(0)
        requested = old_margin.clamp(max=config.retention_margin)
        new_margin = _winner_margin(logits, legal_mask, heuristic_labels, anchors)
        return F.relu(requested - new_margin).mean()

    def case_mean(terms: dict[int, list[Tensor]]) -> Tensor:
        return (
            torch.stack([torch.stack(values).mean() for values in terms.values()]).mean()
            if terms
            else logits.sum() * 0.0
        )

    pair = case_mean(supported_terms)  # Historical diagnostic, not added twice.
    bid_pass = case_mean(bid_pass_terms)
    bid_bid = case_mean(bid_bid_terms)
    evidence_loss = (
        config.bid_pass * bid_pass + config.bid_bid * bid_bid
        if config.supported_pair is None
        else config.supported_pair * pair
    )
    bid_ret = retention_term(bid_anchor)
    pass_ret = retention_term(pass_anchor)
    total = (
        config.heuristic_ce * ce
        + evidence_loss
        + config.bid_retention * bid_ret
        + config.pass_retention * pass_ret
    )
    return {
        "total": total,
        "heuristic_ce": ce,
        "supported_pair": pair,
        "bid_pass": bid_pass,
        "bid_bid": bid_bid,
        "bid_pass_pair_count": torch.tensor(
            sum(map(len, bid_pass_terms.values())), device=logits.device
        ),
        "bid_bid_pair_count": torch.tensor(
            sum(map(len, bid_bid_terms.values())), device=logits.device
        ),
        "bid_pass_case_count": torch.tensor(len(bid_pass_terms), device=logits.device),
        "bid_bid_case_count": torch.tensor(len(bid_bid_terms), device=logits.device),
        "bid_retention": bid_ret,
        "pass_retention": pass_ret,
        "bid_anchor_count": torch.tensor(int(bid_anchor.sum()), device=logits.device),
        "pass_anchor_count": torch.tensor(int(pass_anchor.sum()), device=logits.device),
        "supported_pair_count": torch.tensor(len(supported_rows), device=logits.device),
        "supported_case_count": audited.sum(),
        "heuristic_excluded_count": audited.sum(),
    }
