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
- Tests: `pytest -q`; lint: `ruff check .`

## Log
- 2026-10-16: Overnight build started on branch `overnight-build`. Added pydantic schema (`sim/schema.py`, JSON schemas in `docs/schema/`), fictional rosters, 25 fps agent-based match engine with seeded events, replay windows around every shot, 2 fps whole-match tracking, procedural 17-joint skeletons, sim CLI and tests.
