"""Reproducible, public-only auction examples; no policy optimization here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tarneeb.agents.heuristic_strong import DEFAULT_BID_MARGIN, best_trump_candidate
from tarneeb.agents.state_adapter import reconstruct_state_from_public_view
from tarneeb.core.actions import PASS_ACTION, action_for_bid
from tarneeb.core.cards import hand_from_cards
from tarneeb.core.engine import TarneebEngine
from tarneeb.core.events import Phase
from tarneeb.core.rules import RulesConfig
from tarneeb.env.observation import ObservationBuilder
from tarneeb.env.public_view import PublicView, PublicViewBuilder
from tarneeb.env.spaces import K_MAX
from tarneeb.models.recurrent import RecurrentActorNetwork
from tarneeb.training.bc import generate_bc_dataset, regenerate_sequences

TEACHER_VERSION = "strong_unclamped_opening_v1"
DATA_VERSION = "phase11b_public_auction_v1"
FORBIDDEN_SEED = 20250910


def validate_seeds(seeds: range) -> None:
    if not seeds or seeds.step != 1 or seeds.start < 0 or seeds.stop > 100_000_000:
        raise ValueError("Use a nonempty contiguous development range below 100 million")
    if FORBIDDEN_SEED in seeds:
        raise ValueError("20250910 is forbidden for Phase 11B development")


def validate_label(label: int, legal: np.ndarray) -> int:
    if not 0 <= label < 65 or not bool(legal[label]):
        raise ValueError(f"Teacher produced illegal label {label}; no fallback is permitted")
    return label


def public_teacher_label(view: PublicView) -> int:
    """Strong's hand estimate + frozen margin, WITHOUT clamping weak hands to 7.

    This deliberately limited imitation target is not a value estimate or an
    optimal bidding oracle. No simulations, hidden hands, or bidding rewards.
    The rounded estimate must actually reach a legal bid; otherwise PASS.
    Match-score adjustments are omitted: examples use fresh deal scores (0, 0).
    """
    if view.phase is not Phase.BID:
        raise ValueError("Auction BC labels require BID phase")
    _, estimate = best_trump_candidate(view)
    target = min(13, round(estimate + DEFAULT_BID_MARGIN))
    action = PASS_ACTION
    if target >= 7 and (view.current_high_bid is None or target > view.current_high_bid):
        action = action_for_bid(target)
    return validate_label(action, view.legal_action_mask)


def strength_bucket(estimate: float) -> str:
    """Fixed diagnostic bins of the heuristic estimate, not calibrated trick counts."""
    if estimate < 5:
        return "weak"
    if estimate < 6.5:
        return "medium"
    if estimate < 8:
        return "strong"
    return "exceptional"


@dataclass(frozen=True)
class PublicExamples:
    statics: np.ndarray
    histories: np.ndarray
    lengths: np.ndarray
    legal: np.ndarray
    labels: np.ndarray
    metadata: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return len(self.labels)


def encode_public_view(view: PublicView) -> tuple[np.ndarray, np.ndarray, int]:
    """The encoder receives only PublicView; placeholder hidden hands never become inputs."""
    state = reconstruct_state_from_public_view(view)
    builder = ObservationBuilder(TarneebEngine(RulesConfig()))
    history, history_mask = builder.padded_history(state, view.observer_seat)
    return builder.static_actor(state, view.observer_seat), history, int(history_mask.sum())


def auction_prefixes() -> tuple[tuple[int, ...], ...]:
    # Legal intervention histories, independent of anyone's hidden hand.
    # BID 13 leaves PASS-only decisions until only one active bidder remains.
    prefixes: list[tuple[int, ...]] = [(), (PASS_ACTION,), (PASS_ACTION, PASS_ACTION)]
    for bid in range(7, 14):
        action = action_for_bid(bid)
        prefixes.extend([(action,), (PASS_ACTION, action), (action, PASS_ACTION)])
        if bid > 7:
            prefixes.append((action_for_bid(7), action))
    return tuple(prefixes)


def generate_auction_examples(
    seeds: range,
    learner_team: int | None = None,
    *,
    include_stress: bool = False,
) -> PublicExamples:
    """One record per live prefix/seed. None includes both partnerships, explicitly tagged.

    Prefixes expose competitive raises even if a collapsed actor never visits
    them. Their equal weighting is an intervention dataset, NOT an on-policy
    estimate of how frequently those situations arise.
    """
    validate_seeds(seeds)
    if learner_team not in (None, 0, 1):
        raise ValueError("learner_team must be 0, 1, or None")
    engine = TarneebEngine(RulesConfig())
    builder = PublicViewBuilder(engine)
    statics, histories, lengths, masks, labels, metadata = [], [], [], [], [], []
    cases: list[tuple[int, tuple[int, ...] | None]] = [(seed, None) for seed in seeds]
    if include_stress:
        # Eight valid own hands, each in four suit rotations. These are explicitly
        # synthetic coverage cases, not draws from the natural deal distribution.
        templates = [
            (0, 2, 4, 6, 13, 15, 17, 26, 28, 30, 39, 41, 43),
            (8, 9, 10, 11, 12, 13, 17, 25, 26, 38, 39, 41, 43),
            (6, 7, 8, 9, 10, 11, 12, 25, 38, 50, 13, 26, 39),
            (6, 7, 8, 9, 10, 11, 12, 25, 38, 51, 13, 26, 39),
            (*range(4, 13), 25, 38, 39, 40),
            (*range(4, 13), 25, 38, 39, 51),
            (*range(3, 13), 25, 38, 51),
            tuple(range(13)),
        ]
        for template in templates:
            for rotation in range(4):
                cases.append(
                    (seeds.start, tuple(sorted((card + rotation * 13) % 52 for card in template)))
                )
    for seed, own_hand in cases:
        for prefix in auction_prefixes():
            state = engine.new_deal(seed % 4, match_seed=seed)
            engine.deal(state)
            for action in prefix:
                engine.step(state, action)
            seat = state.current_player_seat
            if state.phase is not Phase.BID or seat is None:
                raise AssertionError("An auction prefix unexpectedly ended bidding")
            if learner_team is not None and seat % 2 != learner_team:
                continue
            if own_hand is not None:
                # Prefix legality is independent of hidden cards. Assign a full
                # valid deck conditioned on THIS observer's hand, before building
                # the view. Targets never see any of these complementary hands.
                unseen = np.array(sorted(set(range(52)) - set(own_hand)))
                np.random.default_rng(seed).shuffle(unseen)
                state.hands[seat] = hand_from_cards(own_hand)
                for index, other in enumerate(s for s in range(4) if s != seat):
                    state.hands[other] = hand_from_cards(
                        unseen[index * 13 : (index + 1) * 13].tolist()
                    )
                state.original_hands = list(state.hands)
            view = builder.build(state, seat)
            static, history, length = encode_public_view(view)
            suit, estimate = best_trump_candidate(view)
            statics.append(static)
            histories.append(history)
            lengths.append(length)
            masks.append(view.legal_action_mask.copy())
            labels.append(public_teacher_label(view))
            metadata.append(
                {
                    "seed": seed,
                    "seat": seat,
                    "learner_team": seat % 2,
                    "current_high_bid": view.current_high_bid,
                    "prefix": list(prefix),
                    "estimate": estimate,
                    "strength": strength_bucket(estimate),
                    "suit_lengths": list(view.own_suit_lengths()),
                    "best_suit": suit,
                    "high_bidder_is_partner": view.current_high_bidder == view.partner_seat,
                    "case_family": "natural" if own_hand is None else "constructed",
                }
            )
    if not labels:
        raise ValueError("No learner decisions in requested probe")
    return PublicExamples(
        np.stack(statics),
        np.stack(histories).astype(np.int64),
        np.array(lengths),
        np.stack(masks),
        np.array(labels, dtype=np.int64),
        tuple(metadata),
    )


def generate_play_examples(seeds: range) -> PublicExamples:
    """Frozen Phase 9 BC replay supplies fixed PLAY and DECLARE observations.

    Only actor observations, public history, masks and actions leave this function.
    The fixed teacher trajectory makes the before/after regression independent
    of any changed auction actions by the candidate.
    """
    validate_seeds(seeds)
    dataset = generate_bc_dataset(seeds, "phase11b_preservation")
    statics, histories, lengths, masks, labels, metadata = [], [], [], [], [], []
    for deal in dataset.deals:
        for sequence in regenerate_sequences(deal):
            for i, phase in enumerate(sequence.phases):
                if phase not in (int(Phase.PLAY), int(Phase.DECLARE)):
                    continue
                statics.append(sequence.statics[i])
                histories.append(sequence.history_ids[i])
                lengths.append(int(sequence.history_masks[i].sum()))
                masks.append(sequence.legal_action_masks[i])
                labels.append(sequence.teacher_actions[i])
                metadata.append(
                    {"seed": deal.match_seed, "seat": sequence.seat, "phase": int(phase)}
                )
    if not labels:
        raise ValueError("No PLAY/DECLARE fixture observations")
    return PublicExamples(
        np.stack(statics),
        np.stack(histories).astype(np.int64),
        np.array(lengths),
        np.stack(masks),
        np.array(labels),
        tuple(metadata),
    )


@torch.no_grad()
def frozen_features(
    actor: RecurrentActorNetwork,
    examples: PublicExamples,
    batch_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Replay each decision's full public history from zero exactly once.

    Chunks respect K_MAX; no cursor or live rollout survives this function.
    With a frozen trunk these features remain valid across output-row updates.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    features, states = [], []
    for start in range(0, len(examples), batch_size):
        stop = min(start + batch_size, len(examples))
        hidden = actor.initial_hidden_state(stop - start)
        history = torch.from_numpy(examples.histories[start:stop])
        lengths = torch.from_numpy(examples.lengths[start:stop])
        for offset in range(0, int(lengths.max()), K_MAX):
            hidden = actor.trunk.history_encoder.advance(
                history[:, offset : offset + K_MAX],
                (lengths - offset).clamp(0, K_MAX),
                hidden,
            )
        static = actor.trunk.static_encoder(torch.from_numpy(examples.statics[start:stop]))
        features.append(actor.trunk.fusion(torch.cat((static, hidden[-1]), dim=-1)))
        states.append(hidden)
    return torch.cat(features), torch.cat(states, dim=1)
