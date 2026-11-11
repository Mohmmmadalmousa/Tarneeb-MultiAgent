"""Auction-row-only behavior cloning. Calling fit_auction_rows updates a COPY.

The optimizer owns two separate eight-row tensors, never an actor parameter.
This is stronger than zeroing gradients: Adam momentum or weight decay cannot
touch the other 57 rows. Export copies only rows 52:60 back to a cloned actor.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tarneeb.auction_repair.data import PublicExamples, frozen_features
from tarneeb.models.masking import MaskedCategorical
from tarneeb.models.recurrent import RecurrentActorNetwork
from tarneeb.training.bc import masked_cross_entropy

AUCTION_SLICE = slice(52, 60)


@dataclass(frozen=True)
class AuctionBCConfig:
    learning_rate: float = 3e-5
    epochs: int = 3
    batch_size: int = 128
    shuffle_seed: int = 8_300_001
    max_gradient_norm: float = 0.5
    max_parameter_drift: float = 0.02

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if any(
            not np.isfinite(v) or v <= 0
            for v in (
                self.learning_rate,
                self.max_gradient_norm,
                self.max_parameter_drift,
            )
        ):
            raise ValueError("Learning rate and conservative bounds must be finite and positive")
        if self.shuffle_seed == 20250910 or self.shuffle_seed < 0:
            raise ValueError("Invalid or forbidden optimization seed")


class AuctionRows(nn.Module):
    def __init__(self, actor: RecurrentActorNetwork) -> None:
        super().__init__()
        self.auction_weight = nn.Parameter(actor.policy_head.weight[AUCTION_SLICE].detach().clone())
        self.auction_bias = nn.Parameter(actor.policy_head.bias[AUCTION_SLICE].detach().clone())

    def forward(self, features: Tensor) -> Tensor:
        return F.pad(F.linear(features, self.auction_weight, self.auction_bias), (52, 5))


def assert_preserved(original: RecurrentActorNetwork, candidate: RecurrentActorNetwork) -> None:
    """Bit equality of ALL frozen tensors, not merely low measured output drift."""
    old = original.state_dict()
    new = candidate.state_dict()
    if old.keys() != new.keys() or original.spec != candidate.spec:
        raise ValueError("Actor architecture changed")
    for name, tensor in old.items():
        if name in ("policy_head.weight", "policy_head.bias"):
            unchanged = torch.equal(tensor[:52], new[name][:52]) and torch.equal(
                tensor[60:], new[name][60:]
            )
        else:
            unchanged = torch.equal(tensor, new[name])
        if not unchanged:
            raise ValueError(f"Frozen parameter changed: {name}")
        if not torch.isfinite(new[name]).all():
            raise ValueError(f"Nonfinite parameter: {name}")


def fit_auction_rows(
    original: RecurrentActorNetwork,
    examples: PublicExamples,
    config: AuctionBCConfig,
) -> tuple[RecurrentActorNetwork, dict[str, Any]]:
    """CPU-only offline BC; no environment stepping, critic, reward, or PPO state.

    Caller must obtain experiment approval before using a research checkpoint.
    Tests use small synthetic models only.
    """
    if original.spec.actor_observation_dim != 692:
        raise ValueError("Phase 11B requires obs_v2 actor observations")
    if len(examples) == 0 or examples.legal.shape != (len(examples), 65):
        raise ValueError("Empty or malformed auction data")
    if examples.legal[:, :52].any() or examples.legal[:, 60:].any():
        raise ValueError("Only BID decisions may enter auction BC")
    if not examples.legal[:, 59].all():
        raise ValueError("Every auction decision must permit PASS")
    if not np.all((examples.labels >= 52) & (examples.labels < 60)):
        raise ValueError("Teacher labels must be auction actions")
    if not examples.legal[np.arange(len(examples)), examples.labels].all():
        raise ValueError("Illegal teacher label; refusing to optimize")
    actor = deepcopy(original).cpu().eval()
    actor.requires_grad_(False)
    features, _ = frozen_features(actor, examples)
    rows = AuctionRows(actor)
    # No weight decay, no inherited PPO optimizer state, only eight rows owned.
    optimizer = torch.optim.Adam(rows.parameters(), lr=config.learning_rate, weight_decay=0.0)
    legal = torch.from_numpy(examples.legal)
    labels = torch.from_numpy(examples.labels)
    generator = np.random.default_rng(config.shuffle_seed)
    initial_weight = rows.auction_weight.detach().clone()
    initial_bias = rows.auction_bias.detach().clone()

    def nll() -> float:
        with torch.no_grad():
            logp = MaskedCategorical.from_logits(rows(features), legal).log_probabilities
            return float(
                masked_cross_entropy(logp, labels, torch.ones_like(labels, dtype=torch.bool))
            )

    metrics: dict[str, Any] = {
        "config": asdict(config),
        "initial_nll": nll(),
        "epoch_nll": [],
        "optimizer_steps": 0,
        "trainable_names": [n for n, _ in rows.named_parameters()],
        "trainable_count": sum(p.numel() for p in rows.parameters()),
    }
    for _ in range(config.epochs):
        order = generator.permutation(len(examples))
        for start in range(0, len(examples), config.batch_size):
            indices = torch.from_numpy(order[start : start + config.batch_size])
            logp = MaskedCategorical.from_logits(
                rows(features[indices]), legal[indices]
            ).log_probabilities
            loss = masked_cross_entropy(
                logp, labels[indices], torch.ones_like(labels[indices], dtype=torch.bool)
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite BC loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(
                rows.parameters(), config.max_gradient_norm, error_if_nonfinite=True
            )
            optimizer.step()
            drift = max(
                float((rows.auction_weight.detach() - initial_weight).abs().max()),
                float((rows.auction_bias.detach() - initial_bias).abs().max()),
            )
            if drift > config.max_parameter_drift:
                raise ValueError("Auction parameter drift exceeded bound; no actor exported")
            metrics["optimizer_steps"] += 1
        metrics["epoch_nll"].append(nll())
    with torch.no_grad():
        actor.policy_head.weight[AUCTION_SLICE].copy_(rows.auction_weight)
        actor.policy_head.bias[AUCTION_SLICE].copy_(rows.auction_bias)
    assert_preserved(original, actor)
    metrics["max_parameter_drift"] = drift
    return actor, metrics


@torch.no_grad()
def preservation_report(
    original: RecurrentActorNetwork,
    candidate: RecurrentActorNetwork,
    fixture: PublicExamples,
) -> dict[str, Any]:
    assert_preserved(original, candidate)
    if fixture.legal[:, AUCTION_SLICE].any():
        raise ValueError("Preservation fixture must contain PLAY/DECLARE only")
    before_features, before_hidden = frozen_features(original, fixture)
    after_features, after_hidden = frozen_features(candidate, fixture)
    before = original.policy_head(before_features)
    after = candidate.policy_head(after_features)
    legal = torch.from_numpy(fixture.legal)
    first = MaskedCategorical.from_logits(before, legal)
    second = MaskedCategorical.from_logits(after, legal)
    return {
        "decision_count": len(fixture),
        "deterministic_action_agreement": float(
            (first.most_probable_action() == second.most_probable_action()).float().mean()
        ),
        "max_legal_logit_drift": float((before - after)[legal].abs().max()),
        "max_masked_log_probability_drift": float(
            (first.log_probabilities - second.log_probabilities)[legal].abs().max()
        ),
        "hidden_state_bit_equal": torch.equal(before_hidden, after_hidden),
        "frozen_parameters_bit_equal": True,
        "max_illegal_probability": float(second.probabilities[~legal].max()),
    }
