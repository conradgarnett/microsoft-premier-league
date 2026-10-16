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
- 2026-10-16: Replaced procedural skeleton animation with CMU mocap (`sim/mocap/`: ASF/AMC reader, `build_library.py` -> `library.npz`, 226 KiB). Speed-blended phase-aligned walk/jog/run cycles with cadence matched to ground speed, side-shuffle, real kick clips aligned to ball contact, dive base clip + roll, slide-tackle key pose, turn lean from path curvature, turn-rate limit, rigid-bone projection. Subjects 141+ look like 60 fps data despite the header, so they are excluded. Removed keeper teleports on saves in the engine. Added `tests/test_skeleton.py` (planted feet, rigid bones, joint limits, no teleports).
- 2026-10-16: Multi-agent layer: `agents/orchestrator.py` (Replay Director, Tactical Analyst, Commentator, Recap Writer, Translator as Agent Framework `Agent`s over a shared `MatchState`; JSON contract + evidence-id validation, corrective retry, offline fallback, trace log), `agents/offline.py` (deterministic templates in en/es/fr/pt with re-renderable `_render` slots), `agents/providers.py` (Azure OpenAI via `OpenAIChatCompletionClient`, else `OfflineChatClient`). MCP server `mcp_server/` (mcp 2.x `MCPServer`, stdio or streamable HTTP); model-backed agents get it via `MCPStdioTool`. API `/api/matches/{id}/insights`, `/api/agents`. Tests for agents, fallback, API, WebSocket, MCP.
- 2026-10-16: Infra + docs: multi-stage `Dockerfile` (web build + API), `docker-compose.yml` (app + MCP over HTTP), `.env.example`, CI (`.github/workflows/ci.yml`: ruff, pytest, web build, docker build), `azure.yaml` + `infra/main.bicep` (ACR, Container Apps, managed identity, Azure OpenAI deployment, serverless Cosmos DB with RBAC; validated with `bicep build`). CosmosStore routes >1.5 MB docs to a blob fallback with pointer items. README rewritten (pitch, mermaid architecture, quickstart, Microsoft tech, mocap attribution, synthetic vs real separation). Old ideas README dropped because it named a game franchise.
- 2026-10-16: Vision module started (`vision/`): `calibrate.py` (panorama camera model fit to manual landmarks; equirectangular beats fisheye, ~14 px RMS on the first match), `video.py` (PyAV range-request frame access), `detect.py` (RT-DETR), `pose.py` (ViTPose + single-view lift), `track.py` (ground-plane Hungarian tracker), `extract.py`, `assemble.py` (writes a sim-layout match folder). Real-footage inputs/outputs and calibration landmarks live only in the private repo / local cache. Viewer falls back to raw windows when a match has no key moments.
- 2026-10-16: Calibrated the synthetic engine against aggregate real-footage movement stats (`vision/stats.py`; real clip: median 1.95 m/s, p99 7.45, 1.4% sprinting, team width ~26 m). Engine changes: arrival-speed and recovery-run intensity model, narrower team shape, slower decisions, shot utility retuned (~30 shots, 2-5 goals per match). Synthetic now: median ~2.1 m/s, p99 ~7.4, ~1.8% sprinting, width ~28 m. Added `test_movement_calibrated_to_real_footage`. Ball stays in the net until the restart (no mid-replay teleports). Fixed goalless-recap evidence, concurrent temp-file writes in LocalStore, viewer crash when no keeper is known.
- 2026-10-16: Live mode in the viewer (`web/src/replay/LiveScene.tsx`): WebSocket feed at 5/20/60x, interpolated 2 fps positions, live scorebug/ticker/timeline, toast for goals and big chances that jumps into the 3D replay. Name tags are now constant-size (hidden within 6 m of the camera). `scripts/screenshot.mjs` takes a post-JS wait argument.
- 2026-10-16: Orchestration now runs as an Agent Framework workflow graph (`agents/workflow.py`: DirectorExecutor -> MomentExecutor (fan-out per key moment) -> RecapExecutor (fan-in) -> TranslatorExecutor -> yield_output); `Orchestrator.call` keeps validation/retry/fallback. Recap tab gained a game-rhythm chart (xG per 5 min + chaos index, `web/src/RhythmChart.tsx`).
- 2026-10-16: More tests: analytics (xG geometry, report consistency, pass difficulty, physical), store (path traversal, Cosmos size routing with a fake container), calibration projection round-trip. CosmosStore keeps its not-found exception on the instance.

## HANDOFF (2026-10-16, moved off the laptop mid-session)

### Done (all on `overnight-build`, draft PR #1)
- Synthetic engine (`sim/`), schema + JSON schemas, CMU-mocap skeletons, movement calibrated to aggregate real-footage stats.
- Analytics, FastAPI (REST + WebSocket live), local/Cosmos storage, MCP server, 5-agent Agent Framework workflow with offline fallback.
- Viewer: 3D replay (cameras, slow-mo, overlays, angle callouts), live mode, insight panel (moment / recap / agents trace), rhythm chart, fan/analyst, 4 languages, focus player.
- Docker/compose, CI, azd + Bicep (validated with `bicep build`), README.
- Tests: `pytest -q` (sim, skeleton validation, analytics, agents + fallback, API + WebSocket, MCP, store, calibration). Last full run green.

### In progress (vision/) - exact next steps
1. `vision/ball.py` `main()` was just written and NOT yet run end to end. Run:
   `python -m vision.ball <video> <camera.json> <followcam.json> <directions.det> <extract.npz> <out.npz>`
   then wire the output into `vision/assemble.py` (add a `--ball out.npz` option replacing the detector-based ball; mark z=0.11 where `conf>0.5`, keep the prior with low confidence otherwise).
2. Event heuristics from tracks + ball: possession = nearest player within 1.5 m; pass = possession moves between same-team players; possession_change across teams; shot = fast ball towards a goal mouth. Emit `Event`s with ids so analytics/agents work on video clips.
3. A second, 60 s clip extraction was killed at ~60%; rerun `python -m vision.extract ... <start> 60 <out.npz>` (about 1.5 s per frame at 12.5 fps on an M2; much faster on a GPU box).
4. Mine "real high-school motion" clips from pose tracks (private repo only), compare with the CMU library.
- Inputs live outside this repo: calibration landmarks, fitted camera and follow-cam mapping, notes -> private companion repo (`vision/`, `notes/`); videos/frames -> local cache only. Expected local layout: `data/real/` or `../mpl-cache/` (both outside git). Replay a real clip: `DATA_DIR=<assembled dir> AUTOGENERATE=0 uvicorn api.main:app --port 8010`.
- Model choices: RT-DETR (`PekingU/rtdetr_r50vd_coco_o365`) + ViTPose (`usyd-community/vitpose-base-simple`), both Apache-2.0. Ultralytics avoided (AGPL). Generic "sports ball" detection fires on white boots; ball comes from motion candidates + follow-cam prior (RANSAC fit: bearing ~= 0.74*yaw, ~81/566 inliers).

### Known bugs / caveats
- Scorebug reported as showing 0-2 at the 72' KBY goal (expected 1-2). The scorebug counts `goal` events with `t <= current time`, so it flips only ~0.2 s after the shot; verify whether the report was the pre-goal frame or a real bug (old seed-7 data was regenerated since).
- Headless Chrome (SwiftShader) renders the viewer at ~1 fps, so replay clocks barely advance in screenshots; real GPUs are fine.
- Synthetic striker pass counts and distances are still high-ish (~13-14 km/90); pass totals ~1,550 per match.
- `/insights` for video-derived matches (no shots) is untested; the viewer falls back to raw windows.
- Real-provider (Azure OpenAI) path constructs correctly but was never called (no credentials here).
- Slide tackle and keeper dive roll are procedural (no CMU equivalent).
