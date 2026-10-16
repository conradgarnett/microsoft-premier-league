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
- 2026-10-16 (cloud session 2): Verified the scorebug (1-1 after the 55' IRV goal; counts goal events up to the replay clock, so it was the pre-goal frame). Fixed the web typecheck (nullable metrics), headless Chrome as root (`--no-sandbox`), ad-board text overlap, and CI (`vision` not importable: pytest `pythonpath`, scipy/OpenCV in dev extras, lazy torch import in `vision/pose.py`). Vision: `vision/fetch.py` (segment cache by stream copy), `vision/events.py` (control runs -> pass / possession change / shot / goal; on synthetic ground truth pass P 0.93 R 0.82, shots 26/30), streamed `vision/ball.py` with a full-resolution crop detector around the follow-cam prior, `assemble --ball/--motion lift|mocap` (mocap = CMU motion on real tracks, skeleton source `video-mocap`), `vision.pose.rigidify` (Taylor depth recovery, rigid bones; synthetic one-camera test 12.5 -> 5.7 cm mean joint error). Agents: workflow reaches the recap when a match has no key moments; recap describes real clips by what was tracked. Viewer header names the data source for video clips.
- 2026-10-16 (cloud session 2, later): Ran the full vision chain on a 60 s real clip (CPU, ~8 s/frame). Fixed three bugs it exposed: lazy `NpzFile` access (assemble OOM-killed; now 6 s), ball median smoothing across prior-only frames, detector choice anchored on continuity (locked onto a spare ball; now anchored on the follow-cam prior). Ball vs hand labels: median 0.5 m, 4/4 within 1 m after adding a 2x zoomed crop pass (prior 7.9 m). Tracklet stitching was tried and reverted (no coverage gain: the tracker already bridges 3 s gaps; coverage is limited by far-side detection misses). Added the Data Auditor agent (`get_data_quality` tool, recap "where this data comes from"), localized viewer chrome, readable kit colours, one-at-a-time generation (429), `vision/evaluate.py`, `docs/demo.md`. Real-clip details only in the private repo `notes/`.

## HANDOFF (2026-10-16, end of cloud session 2)

### Done (all on `overnight-build`, draft PR #1, CI green)
- Synthetic engine, schema, CMU-mocap skeletons, analytics, FastAPI (REST + WebSocket live), local/Cosmos storage, MCP server, Docker/compose, CI, azd + Bicep, README, `docs/demo.md` (2-minute shot list).
- Six-agent Agent Framework workflow: Director -> Analyst + Commentator per moment -> Recap Writer + Data Auditor (parallel) -> Translator; evidence validation, retry, offline fallback. Works for matches without key moments (video clips).
- Viewer: 3D replay, live mode, insight panel (moment / recap + data provenance / agents), rhythm chart, fan/analyst, en/es/fr/pt for agent text AND viewer chrome, focus player, video-source header.
- Vision: fetch -> extract -> ball (crop detector + motion + follow-cam prior) -> assemble (`--ball`, `--motion lift|mocap`, events, final score) -> evaluate. Event heuristics scored on synthetic GT; pose depth recovery tested on synthetic mocap.
- Tests: `pytest -q` 36 passed (also passes without torch, as in CI).

### Real-footage run (local cache `/home/user/mpl-cache/m0`, never committed)
- 60 s panorama clip of manifest[0] at t=2400 s; `post.sh` there chains ball -> evaluate -> assemble (lift + mocap) -> stats. Results go to the private repo `notes/`.
- CPU only here: extraction ~10 s/frame at 12.5 fps (RT-DETR + ViTPose on 4 cores). Use a GPU box for longer clips.
- Hand ball labels: private repo `vision/ball_labels_m0_2400.json` (7 frames, 5 usable ground labels). Follow-cam prior error 3-16 m even where the ball is visible.

### Next steps
1. Real clip assembled and replayable (`DATA_DIR=/home/user/mpl-cache/m0/matches AUTOGENERATE=0 uvicorn api.main:app --port 8010`; the cache is lost with the container, re-run `vision.fetch` + `post.sh`-style chain). Detected passes on video were spot-checked: 2 of 6 match an on-ball action (3 false). Improve the ball track first (detector at a higher rate on a GPU, use only measured frames for events); team split (18/10/2) is fragment-inflated.
2. More ball labels (label around the ball output to measure precision, around the prior for recall).
3. Recalibrate synthetic movement if the new clip's aggregate stats differ from the first clip's (`python -m vision.stats`).
4. Record the demo on a GPU machine; try the Azure OpenAI provider once credentials exist.

### Known caveats
- Headless Chrome (SwiftShader) renders at ~1 fps; screenshots need `CHROME=/opt/pw-browsers/chromium-1194/chrome-linux/chrome` in the cloud container.
- Synthetic pass totals ~1,550 per match (high); distance ~140 m/min matches the (jitter-inflated) real measurement it was calibrated to.
- Azure OpenAI path constructed but never called. Slide tackle and keeper dive roll are procedural.
- Cached insights in `data/matches/*/insights/` predate the Data Auditor; delete them to regenerate.
