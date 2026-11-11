"""Synthetic only: no checkpoints, gameplay, human simulation or reserved deals."""
from types import SimpleNamespace

import pytest

from scripts.phase12_evaluate import write_new
from tarneeb.agents.base import Agent
from tarneeb.evaluation.arena import Arena, build_seat_assignment
from tarneeb.evaluation.metrics import DealOutcome
from tarneeb.evaluation.phase12 import (
    Partnership, RecordingArena, calibration, check_seed_blocks, deal_rows, sp_xp_gap,
)


class Spy(Agent):
    def __init__(self, name):
        self.name = name
        self.views = []

    def reset(self):
        self.views.clear()

    def act(self, view):
        self.views.append(view)
        return 59


def outcome():
    return DealOutcome(17, 0, 0, 0, (8, 0), 2, 7, (8, 5), True,
                       False, 60, 1, 3, True)


def test_routing_passes_exact_private_view_and_resets_independently():
    a, b, other = Spy('a'), Spy('b'), Spy('other')
    router = Partnership(a, b)
    for team in (0, 1):
        assignment = build_seat_assignment(router, other, team)
        for seat, expected in [(team, a), (team+2, b)]:
            view = SimpleNamespace(observer_seat=seat, own_hand=(seat,))
            assignment.agent_for(seat).act(view)
            assert expected.views[-1] is view
    assert a.views is not b.views
    router.reset()
    assert not a.views and not b.views
    with pytest.raises(ValueError):
        Partnership(a, a)


def test_declarer_only_calibration_and_sparse_levels():
    rows = deal_rows(outcome(), ['us', 'them', 'us', 'them'])
    assert [r['role'] for r in rows] == ['partner', 'defender', 'declarer', 'defender']
    result = calibration(rows)
    assert result['us']['n'] == 1  # partner reward must not duplicate a declaration
    assert result['them']['n'] == 0
    assert result['us']['slope'] is None
    assert result['us']['levels'][13]['success'] is None
    sample = [dict(rows[2], winning_bid=x, team_tricks=2*x-7) for x in [7, 8, 9]]
    fit = calibration(sample)['us']
    assert fit['slope'] == pytest.approx(2)
    assert fit['intercept'] == pytest.approx(-7)
    assert fit['levels'][7]['sparse']


def test_recording_delegates_without_new_game_loop(monkeypatch):
    observed = []
    def fake_play(self, assignment, **kwargs):
        observed.append(kwargs)
        return outcome()
    monkeypatch.setattr(Arena, 'play_deal', fake_play)
    rows = []
    arena = RecordingArena(rows.extend)
    assignment = build_seat_assignment(Partnership(Spy('a'), Spy('b')), Spy('strong'), 0)
    arena.play_deal(assignment, match_seed=17, dealer_seat=0, verify_replay=True)
    assert observed == [dict(match_seed=17, dealer_seat=0, verify_replay=True)]
    assert [r['policy'] for r in rows] == ['a', 'strong', 'b', 'strong']


def test_duplicate_fairness_delegates_both_orientations(monkeypatch):
    seen = []
    def fake_play(self, assignment, **kwargs):
        seen.append((assignment.policy_a_team, kwargs))
        return outcome()
    monkeypatch.setattr(Arena, 'play_deal', fake_play)
    Arena().play_duplicate_pair(Spy('a'), Spy('b'), 17, 3, verify_replay=True)
    assert [team for team, _ in seen] == [0, 1]
    assert seen[0][1] == seen[1][1]


def test_gap_sign_and_cluster_unit():
    gap = sp_xp_gap([4, 4, 4], [2, 2, 2], [1, 1, 1], 11, 20)
    assert gap['mean'] == gap['lower_bound'] == gap['upper_bound'] == 2
    assert gap['sample_count'] == 3
    with pytest.raises(ValueError):
        sp_xp_gap([1], [1, 2], [1], 11, 20)


def test_seed_modulo_and_reserved_confirmation():
    with pytest.raises(ValueError):
        check_seed_blocks([('old', 'arena', 1000080000, 4)], [(80000, 2)])
    with pytest.raises(ValueError):
        check_seed_blocks([], [(70000, 1)])
    with pytest.raises(ValueError):
        check_seed_blocks([], [(80000, 2), (80001, 2)])
    assert check_seed_blocks([], [(80000, 2)]) == [1000080000, 1000080001]


def test_no_overwrite(tmp_path):
    path = tmp_path/'result.json'
    write_new(path, {'original': True})
    with pytest.raises(FileExistsError):
        write_new(path, {'original': False})


def test_real_recurrent_memory_storage_without_network_or_game():
    from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
    actor = SimpleNamespace(spec=SimpleNamespace(gru_layer_count=1, hidden_size=2))
    first = RecurrentNeuralAgent(actor, name='first')
    second = RecurrentNeuralAgent(actor, name='second')
    states = [agent.hidden_state(seat) for agent in (first, second) for seat in range(4)]
    assert len({state.data_ptr() for state in states}) == 8
    first.hidden_state(0).fill_(1)
    assert all(state.sum() == 0 for state in states[1:])
    Partnership(first, second).reset()
    assert first.hidden_state(0).sum() == second.hidden_state(0).sum() == 0


def test_human_redeal_resets_without_simulating_a_deal(monkeypatch):
    from scripts.phase12_human import RecordedHumanSession
    from tarneeb.app.session import MatchSession
    monkeypatch.setattr(MatchSession, '_finish_passed_out_deal', lambda self: None)
    session = object.__new__(RecordedHumanSession)
    policy = Spy('partner')
    policy.views.append('old history')
    session.agents_by_seat = {2: policy}
    session._finish_passed_out_deal()
    assert not policy.views


def test_human_collection_refuses_redirected_input(monkeypatch):
    from scripts import phase12_human
    monkeypatch.setattr(phase12_human.sys, 'argv', ['phase12_human', '--participant', 'test'])
    monkeypatch.setattr(phase12_human.sys, 'stdin', SimpleNamespace(isatty=lambda: False))
    with pytest.raises(ValueError, match='interactive human terminal'):
        phase12_human.main()
