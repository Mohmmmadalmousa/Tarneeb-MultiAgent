"""Small, single-match localhost JSON API (Python standard library)."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import secrets
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tarneeb.agents.base import checked_action
from tarneeb.agents.recurrent_agent import RecurrentNeuralAgent
from tarneeb.app.session import MatchSession
from tarneeb.core.cards import card_name
from tarneeb.core.events import EventType

CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints/phase9_gru_bc_s0_update250.pt"
CHAMPION_SHA256 = "73ef1ec852a060f788e94208637e528e4a2d9cc55361da8202cc31de2f27a546"


def champion_agents():
    import torch

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "The trained champion is missing. See README.md: trained champion availability."
        )
    # The frozen research loader uses pickle. Verify the exact approved bytes first.
    with CHECKPOINT.open("rb") as checkpoint_file:
        digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    if digest != CHAMPION_SHA256:
        raise ValueError("Champion checksum mismatch; refusing to deserialize this checkpoint.")
    torch.set_num_threads(1)
    first = RecurrentNeuralAgent.from_checkpoint(CHECKPOINT)
    # Weights are immutable during inference; memory and cursors belong to each agent.
    return {seat: RecurrentNeuralAgent(first.actor, seed=seat) for seat in (1, 2, 3)}


class LocalGame:
    def __init__(self, agents=None):
        self.agents = champion_agents() if agents is None else agents
        self.session = None
        self.revision = 0
        self.finished_view = None
        self.last_trick = None
        self.lock = threading.RLock()

    def snapshot(self):
        if self.session is None:
            return {
                "revision": self.revision,
                "table": None,
                "legal_actions": [],
                "history": [],
                "match_complete": False,
                "last_trick": None,
            }
        session = self.session
        table = self.finished_view or session.public_table_view().as_dict()
        counts = [13] * 4
        if session.state is not None:
            # Counts are derived from public plays, never from hidden hand masks.
            for event in session.views.build(session.state, 0).events:
                if event.event_type == EventType.CARD_PLAYED:
                    counts[event.actor_seat] -= 1
        else:
            counts = [0] * 4
        return {
            "revision": self.revision,
            "table": table,
            "legal_actions": []
            if self.finished_view
            else [asdict(a) for a in session.legal_human_actions()],
            "hand_counts": counts,
            "history": [asdict(d) for d in session.completed_deals],
            "deal_complete": self.finished_view is not None,
            "match_complete": session.match_is_complete,
            "last_trick": self.last_trick,
        }

    def command(self, command, revision, action=None):
        with self.lock:
            if type(revision) is not int or revision != self.revision:
                raise ValueError("Stale action. Refresh the table before trying again.")
            # Roll back both the engine and recurrent memory if inference fails.
            memo = {id(a.actor): a.actor for a in self.agents.values() if hasattr(a, "actor")}
            backup = copy.deepcopy(
                (self.session, self.agents, self.finished_view, self.last_trick), memo
            )
            try:
                self._command(command, action)
            except Exception:
                self.session, self.agents, self.finished_view, self.last_trick = backup
                raise
            self.revision += 1
            return self.snapshot()

    def _command(self, command, action):
        if command == "start":
            if self.session is not None and not self.session.match_is_complete:
                raise ValueError("A match is already in progress.")
            self.session = MatchSession(
                self.agents,
                human_seat=0,
                match_seed=secrets.randbits(32),
                dealer_seat=secrets.randbelow(4),
                record_replay=False,
            )
            self.session.start_deal()
            self.finished_view = self.last_trick = None
            return
        s = self.session
        if s is None:
            raise ValueError("Start a match first.")
        if command == "next":
            if not self.finished_view or s.match_is_complete:
                raise ValueError("The next deal is not available.")
            s.start_deal()
            self.finished_view = self.last_trick = None
            return
        if self.finished_view:
            raise ValueError("This deal has finished.")
        before = s.require_state()
        deal_index = before.deal_index
        old_events = len(before.events)
        if command == "action":
            if type(action) is not int or action not in [
                a.action_id for a in s.legal_human_actions()
            ]:
                raise ValueError("That action is not legal now.")
            s.apply_human_action(action)
        elif command == "advance":
            if not s.deal_is_live or s.waiting_for_human:
                raise ValueError("No AI turn is pending.")
            seat = s.seat_on_turn
            view = s.views.build(before, seat)
            agent = s.agents_by_seat[seat]
            selected = checked_action(agent.act(view), view.legal_action_mask, agent.name)
            s._step(selected)  # Reuse the session's recorder/redeal lifecycle.
        else:
            raise ValueError("Unknown command.")
        state = s.require_state()
        if state.deal_index != deal_index:
            # MatchSession redeals in-place after all-pass; reset recurrent memory here.
            for agent in s.agents_by_seat.values():
                agent.reset()
            self.last_trick = None
        else:
            for event in state.events[old_events:]:
                if event.event_type == EventType.TRICK_COMPLETED:
                    cards = [
                        [e.actor_seat, card_name(e.card)]
                        for e in state.events
                        if e.event_type == EventType.CARD_PLAYED and e.trick_no == event.trick_no
                    ]
                    self.last_trick = {
                        "number": event.trick_no,
                        "winner": event.trick_winner_seat,
                        "cards": cards,
                    }
        if not s.deal_is_live:
            self.finished_view = s.public_table_view().as_dict()
            s.finish_deal()
            self.finished_view["match_score_by_team"] = list(s.match_score_by_team)


def make_server(game, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def valid_host(self):
            # Reject DNS-rebinding names, including on read-only state requests.
            if self.headers.get("Host") not in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            ):
                self.reply(403, {"error": "Host not allowed"})
                return False
            return True

        def reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.valid_host():
                return
            if self.path != "/api/state":
                self.reply(404, {"error": "Not found"})
                return
            with game.lock:
                self.reply(200, game.snapshot())

        def do_POST(self):
            if not self.valid_host():
                return
            # No CORS. Browser writes must come from the local Vite origin.
            if self.headers.get("Origin") not in (
                None,
                "http://127.0.0.1:5173",
                "http://localhost:5173",
            ):
                self.reply(403, {"error": "Origin not allowed"})
                return
            if (
                self.path != "/api/command"
                or self.headers.get("Content-Type") != "application/json"
            ):
                self.reply(400, {"error": "Expected a JSON command"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                result = game.command(body.get("command"), body.get("revision"), body.get("action"))
                self.reply(200, result)
            except (ValueError, TypeError) as exc:
                self.reply(409, {"error": str(exc)})
            except Exception:
                import traceback

                traceback.print_exc()
                self.reply(
                    500,
                    {"error": "Game action failed; no state was advanced. See the Python console."},
                )

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(LocalGame(), args.port)
    print(f"Tarneeb API ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
