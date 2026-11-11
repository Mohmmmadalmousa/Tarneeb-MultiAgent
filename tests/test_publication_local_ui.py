"""Publication safety checks; no training or research evaluation."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from tarneeb.agents.heuristic_strong import StrongHeuristicAgent
from tarneeb.local_ui import server as ui


def test_champion_checked_before_deserialization(tmp_path, monkeypatch):
    checkpoint = tmp_path / "untrusted.pt"
    monkeypatch.setattr(ui, "CHECKPOINT", checkpoint)
    with pytest.raises(FileNotFoundError, match="champion is missing"):
        ui.champion_agents()
    checkpoint.write_bytes(b"not the champion")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ui.champion_agents()


def test_http_rejects_rebinding_and_malformed_requests():
    game = ui.LocalGame({s: StrongHeuristicAgent() for s in (1, 2, 3)})
    server = ui.make_server(game, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    before = game.snapshot()
    cases = [
        ("/api/state", None, {"Host": "attacker.example"}, 403),
        ("/api/command", b"{}", {"Host": "attacker.example"}, 403),
        ("/api/command", b"{", {}, 409),
        ("/api/command", b"[]", {}, 409),
        ("/api/command", b"\xff", {}, 409),
        ("/api/command", b" " * 1025, {}, 409),
        ("/api/command", b'{"command":"start","revision":true}', {}, 409),
        ("/api/command", b'{"command":"unknown","revision":0}', {}, 409),
        ("/api/command", b"{}", {"Origin": "https://attacker.example"}, 403),
        ("/api/command", b"{}", {"Content-Type": "text/plain"}, 400),
    ]
    try:
        for path, body, headers, status in cases:
            req = urllib.request.Request(
                url + path, body, {"Content-Type": "application/json", **headers}
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req, timeout=5)
            assert error.value.code == status
            assert "error" in json.load(error.value)
            assert game.snapshot() == before
        with urllib.request.urlopen(url + "/api/state", timeout=5) as response:
            assert json.load(response) == before
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
