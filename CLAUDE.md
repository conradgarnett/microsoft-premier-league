# 3D Tactical Replay — project notes for Claude

Hackathon entry for Microsoft x Premier League "Inside the Game" (submissions due 2026-10-27 11:59pm PT).
Synthetic football match engine -> analytics -> multi-agent explanations (Microsoft Agent Framework + MCP)
-> FastAPI -> React/three.js broadcast-style 3D replay viewer. Target category: Best Multi-Agent System.

Legal: fictional clubs and players only. No EA/FIFA/FC24/HyperMotion names, no real club names, badges or players.

## Layout
- `sim/` synthetic match engine, pydantic schema (source of truth), procedural skeletons
- `analytics/` metrics (xG, pass difficulty, PPDA, field tilt, control-vs-chaos, joint angles, key moments)
- `agents/` multi-agent orchestrator, offline fallback provider, Azure OpenAI / Foundry provider
- `mcp_server/` MCP server exposing match tools
- `api/` FastAPI backend (REST + WebSocket live stream), storage interface (local JSON, Cosmos DB)
- `web/` Vite + React + TypeScript + react-three-fiber frontend
- `infra/` Azure Container Apps (Bicep + azd), `docs/` schema and screenshots

## Commands
- Python env: `uv venv && uv pip install -e ".[dev]"`
- Generate a match: `python -m sim --seed 7 --minutes 90`
- API: `uvicorn api.main:app --port 8000` (auto-generates seed 7 if no match exists)
- Web dev: `cd web && npm install && npm run dev` (proxies /api to :8000); prod build `npm run build` is served by the API
- Screenshot: `node scripts/screenshot.mjs http://localhost:8000/ docs/img/replay.png 14`
- Tests: `pytest -q`; lint: `ruff check .`

## Log
- 2026-10-16: Overnight build started on branch `overnight-build`. Added pydantic schema (`sim/schema.py`, JSON schemas in `docs/schema/`), fictional rosters, 25 fps agent-based match engine with seeded events, replay windows around every shot, 2 fps whole-match tracking, procedural 17-joint skeletons, sim CLI and tests.
- 2026-10-16: Analytics package (`analytics/core.py`, `body.py`, `moments.py`), storage interface (`api/store.py`: LocalStore JSON + CosmosStore), FastAPI app (`api/main.py`) with replay, metrics, key moments and WebSocket live stream; serves `web/dist` when built.
- 2026-10-16: Web viewer (`web/`): Vite + React + react-three-fiber. Instanced skeleton players, ball trail, director/broadcast/behind-ball/player/orbit cameras, slow-mo + scrub, overlays (pass lines w/ difficulty, pressure rings, defensive line, sprint trails, speeds, strike + scanning callouts), scorebug, ticker, key-moment timeline, fan/analyst modes. `scripts/screenshot.mjs` captures headless screenshots (docs/img/).
