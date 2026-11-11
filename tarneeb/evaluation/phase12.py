"""Additive Phase 12 adapters; all game transitions remain in the frozen arena."""
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from tarneeb.agents.base import Agent
from tarneeb.evaluation.arena import Arena, evaluation_match_seed
from tarneeb.evaluation.stats import bootstrap_mean_confidence_interval


class Partnership(Agent):
    """Separate agent objects, each retaining its own per-seat recurrent state."""

    def __init__(self, first, second):
        if first is second:
            raise ValueError("Use separate agent instances even for identical checkpoints")
        self.members = (first, second)
        self.name = f"{first.name}+{second.name}"

    def member(self, seat):
        return self.members[seat // 2]

    def reset(self):
        for member in self.members:
            member.reset()

    def act(self, view):
        return self.member(view.observer_seat).act(view)


def deal_rows(outcome, policies):
    """One seat-role row; only the actual declarer's row enters calibration."""
    base = asdict(outcome)
    return [dict(base, seat=seat, policy=policy,
                 role=None if outcome.passed_out else outcome.role_of_seat(seat),
                 team_tricks=outcome.tricks_won_by_team[seat % 2],
                 team_score=outcome.score_change_by_team[seat % 2])
            for seat, policy in enumerate(policies)]


class RecordingArena(Arena):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def play_deal(self, assignment, **kwargs):
        result = super().play_deal(assignment, **kwargs)
        names = []
        for seat, agent in enumerate(assignment.agent_by_seat):
            names.append(agent.member(seat).name if isinstance(agent, Partnership) else agent.name)
        self.sink(deal_rows(result, names))
        return result


def calibration(rows):
    """Descriptive OLS, not causal or seed-level inference; no opponent attribution."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['policy']]  # retain policies with no declarations
        if row['role'] == 'declarer':
            groups[row['policy']].append(row)
    result = {}
    for policy, declarations in groups.items():
        x = np.asarray([r['winning_bid'] for r in declarations], dtype=float)
        y = np.asarray([r['team_tricks'] for r in declarations], dtype=float)
        identifiable = len(x) >= 3 and len(set(x)) >= 2
        slope = float(np.sum((x-x.mean())*(y-y.mean())) / np.sum((x-x.mean())**2)) if identifiable else None
        levels = {}
        for bid in range(7, 14):
            selected = [r for r in declarations if r['winning_bid'] == bid]
            n = len(selected)
            levels[bid] = dict(n=n, sparse=n < 20,
                               success=sum(r['contract_made'] for r in selected)/n if n else None,
                               mean_tricks=sum(r['team_tricks'] for r in selected)/n if n else None)
        result[policy] = dict(n=len(x), identifiable=identifiable, slope=slope,
                              intercept=float(y.mean()-slope*x.mean()) if identifiable else None,
                              levels=levels, note='Descriptive fit; no independent-deal OLS CI')
    return result


def sp_xp_gap(first, second, cross, seed, resamples=2000):
    """Resample matched duplicate-pair clusters, never individual seat rows."""
    if not (len(first) == len(second) == len(cross)):
        raise ValueError('Unmatched sample counts')
    difference = (np.asarray(first)+np.asarray(second))/2-np.asarray(cross)
    return bootstrap_mean_confidence_interval(difference, seed, resamples).as_dict()


def check_seed_blocks(entries, blocks):
    """Reuse frozen exposure metadata; reserve whole seeds conservatively."""
    used = set()
    for _, kind, seed, count in entries:
        if kind == 'arena':
            used.update(evaluation_match_seed(seed, i) for i in range(count))
        elif kind == 'direct':
            used.add(seed)
        elif kind == 'direct_range':
            used.update(range(seed, count))
        else:
            raise ValueError(f'Unknown exposure kind {kind}')
    for start, count in [(0, 10000), (60000, 2000), (70000, 2000), (90000, 100)]:
        used.update(evaluation_match_seed(start, i) for i in range(count))
    allocated = set()
    for start, count in blocks:
        values = {evaluation_match_seed(start, i) for i in range(count)}
        if len(values) != count or values & (used | allocated):
            raise ValueError('Seed allocation overlaps prior/reserved/other block')
        allocated.update(values)
    return sorted(allocated)
