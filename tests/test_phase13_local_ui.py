"""Focused Phase 13 integration checks; no training or evaluation runs."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.app.session import MatchSession
from tarneeb.core.cards import card_name, cards_in_hand
from tarneeb.core.events import Phase
from tarneeb.local_ui.server import LocalGame, champion_agents, make_server


def game():
    result = LocalGame({s: StrongHeuristicAgent() for s in (1, 2, 3)})
    result.session = MatchSession(result.agents, match_seed=42, record_replay=False)
    result.session.start_deal()
    return result


def step(g):
    s = g.session
    if s.waiting_for_human:
        action = s.legal_human_actions()[0].action_id
        return g.command("action", g.revision, action)
    return g.command("advance", g.revision)


def test_deal_scoring_next_dealer_and_match_end():
    g = game()
    first_dealer = g.session.dealer_seat
    for _ in range(90):
        payload = step(g)
        if payload.get("deal_complete"):
            break
    else:
        pytest.fail("deal failed to complete")
    assert sum(payload["table"]["tricks_won_by_team"]) == 13
    summary = g.session.completed_deals[-1]
    assert list(summary.score_change_by_team) == payload["table"]["match_score_by_team"]
    assert len(payload["last_trick"]["cards"]) == 4
    assert g.session.dealer_seat == (first_dealer + 1) % 4
    before = g.snapshot()
    with pytest.raises(ValueError):
        g.command("action", g.revision, 0)
    assert g.snapshot() == before
    g.command("next", g.revision)
    assert len(g.snapshot()["table"]["own_hand"]) == 13
    # Exercise the actual match-boundary path without a long match simulation.
    g.session.match_score_by_team = (100, 100)
    while not g.finished_view:
        step(g)
    assert g.snapshot()["match_complete"]
    with pytest.raises(ValueError):
        g.command("next", g.revision)


def test_illegal_stale_actions_and_private_view():
    g = game()
    while not g.session.waiting_for_human:
        step(g)
    before = g.snapshot()
    with pytest.raises(ValueError):
        g.command("action", g.revision, 64)
    assert before == g.snapshot()
    revision = g.revision
    step(g)
    with pytest.raises(ValueError):
        g.command("action", revision, 59)
    payload = g.snapshot()
    assert set(payload["table"]) == set(g.session.public_table_view().as_dict())
    assert not any(
        key in json.dumps(payload) for key in ("original_hands", "checkpoint", "_hidden", "logits")
    )
    assert len(before["table"]["own_hand"]) == 13
    visible = set(payload["table"]["own_hand"])
    visible.update(card for _, card in payload["table"]["current_trick"])
    if payload["last_trick"]:
        visible.update(card for _, card in payload["last_trick"]["cards"])
    for seat in (1, 2, 3):
        hidden = {card_name(c) for c in cards_in_hand(g.session.state.hands[seat])}
        assert visible.isdisjoint(hidden)


class PassAgent(StrongHeuristicAgent):
    resets = 0

    def reset(self):
        self.resets += 1

    def act(self, view):
        return 59


def test_all_pass_redeal_resets_memory_keeps_dealer():
    g = LocalGame({s: PassAgent() for s in (1, 2, 3)})
    g.session = MatchSession(g.agents, match_seed=4, record_replay=False)
    g.session.start_deal()
    for _ in range(3):
        g.command("advance", g.revision)
    g.command("action", g.revision, 59)
    assert g.session.require_state().phase == Phase.BID
    assert g.session.dealer_seat == 0
    assert g.session.deal_index == 1
    assert g.snapshot()["history"][-1]["passed_out"]
    assert all(a.resets == 2 for a in g.agents.values())
    assert g.snapshot()["hand_counts"] == [13] * 4


def test_champion_inference_and_independent_memory():
    agents = champion_agents()
    assert len({id(a) for a in agents.values()}) == 3
    assert len({a.hidden_state(s).data_ptr() for s, a in agents.items()}) == 3
    g = LocalGame(agents)
    g.session = MatchSession(agents, match_seed=42, record_replay=False)
    g.session.start_deal()
    # One real deal is a gameplay smoke test, not a research evaluation.
    for _ in range(100):
        if g.finished_view:
            break
        step(g)
    assert g.finished_view is not None
    assert all(a.states_are_separate() for a in agents.values())


def test_follow_suit_is_engine_authoritative():
    g = game()
    checked = False
    for _ in range(90):
        if g.finished_view:
            break
        s = g.session
        view = s.public_table_view()
        if s.waiting_for_human and view.phase == "PLAY" and view.current_trick:
            led = view.current_trick[0][1][-1]
            if any(c[-1] == led for c in view.own_hand):
                assert all(a.label[-1] == led for a in s.legal_human_actions())
                checked = True
        step(g)
    assert checked


def test_human_bid_pass_and_trump_use_legal_actions():
    g = game()
    g.session = MatchSession(g.agents, match_seed=42, dealer_seat=3, record_replay=False)
    g.session.start_deal()
    assert {a.action_id for a in g.session.legal_human_actions()} == set(range(52, 60))
    g.command("action", g.revision, 58)
    while not g.session.waiting_for_human:
        g.command("advance", g.revision)
    assert g.snapshot()["table"]["phase"] == "DECLARE"
    assert {a.action_id for a in g.session.legal_human_actions()} == {60, 61, 62, 63}
    g.command("action", g.revision, 62)
    assert g.snapshot()["table"]["trump"] == "hearts"
    assert g.snapshot()["table"]["seat_on_turn"] == 0


def test_http_get_is_read_only_and_duplicate_write_rejected():
    g = game()
    server = make_server(g, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(url + "/api/state") as response:
            assert json.load(response)["revision"] == 0
        body = json.dumps({"command": "advance", "revision": 0}).encode()
        req = urllib.request.Request(
            url + "/api/command", body, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as response:
            assert json.load(response)["revision"] == 1
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req)
        assert error.value.code == 409
        bad = urllib.request.Request(
            url + "/api/command",
            body,
            {"Content-Type": "application/json", "Origin": "https://example.com"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_inference_failure_rolls_back_memory_and_engine():
    class BrokenAgent(StrongHeuristicAgent):
        calls = 0

        def act(self, view):
            self.calls += 1
            raise RuntimeError("inference unavailable")

    g = game()
    broken = BrokenAgent()
    g.agents[1] = broken
    g.session.agents_by_seat[1] = broken
    before = g.snapshot()
    with pytest.raises(RuntimeError):
        g.command("advance", g.revision)
    assert g.snapshot() == before
    assert g.agents[1].calls == 0
