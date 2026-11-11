# Phase 13 local play

Uses the original table UI, the frozen `MatchSession`, and the Phase 9 champion
for seats 1, 2, and 3. Seat 0 is human. State is in memory; restarting Python
starts fresh. Refreshing the browser reconnects to the current match. Only one
local match is active; stale actions from other tabs are rejected.

Install Python dependencies and verify the included champion as described in the
[root README](../README.md#play-locally). The game uses the single checkpoint
`checkpoints/phase9_gru_bc_s0_update250.pt` and validates its checksum before loading.
From PowerShell in your checkout root:

```powershell
# Install frontend dependencies once (Node 22.12+ with npm).
.\scripts\phase13-ui.ps1 install

# Terminal 1
.\.venv\Scripts\python.exe -m tarneeb.local_ui.server

# Terminal 2
.\scripts\phase13-ui.ps1 dev
```

Open http://127.0.0.1:5173. Start from the menu. Press a legal card once to
select it and again to play. Changing cards changes selection. After a scored
deal, use Next deal. Scores and the score sheet come from the engine. AI moves
are paced for the table animations. The completed trick remains visible until
the next card is played; its winner is labeled.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase13_local_ui.py -q
.\scripts\phase13-ui.ps1 typecheck
.\scripts\phase13-ui.ps1 test
.\scripts\phase13-ui.ps1 build
```

No Python dependencies were added. The local Vite config proxies `/api` to
127.0.0.1:8765; it does not use the Replit preview plugins or API/database.
The complete original frontend workspace, including Replit scaffolding, is preserved
in the project. The standalone local app does not run that scaffolding. Original styling,
assets, and font links remain.
Fonts may fall back when offline.
