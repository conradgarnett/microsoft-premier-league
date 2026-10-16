# Demo video plan (under 2 minutes)

One take per shot, 1600x900, real GPU browser (headless SwiftShader renders at ~1 fps).
Start the app with `uvicorn api.main:app --port 8000` after `python -m sim --seed 7 --minutes 90`.

| Time | Shot | What to show | Voice-over (one line) |
|------|------|--------------|-----------------------|
| 0:00-0:10 | Live view, 20x | Synthetic match streaming over WebSocket: scorebug, ticker, timeline filling | "Every event here is synthetic: a seeded match engine, calibrated against real youth-football movement." |
| 0:10-0:25 | Goal toast -> 3D Replay | Toast pops on a goal; click jumps into the replay; Director camera, auto slow-mo at the strike | "Ingest and render: the engine's 25 fps tracking and mocap skeletons replay any moment from any angle." |
| 0:25-0:45 | Analyst mode overlays | Pass lines coloured by difficulty, pressure rings, defensive line, strike angles and scanning callouts; switch to Behind-ball camera | "Interpret: xG, pass difficulty, pressure and body angles, computed from the skeletons." |
| 0:45-1:05 | Insight panel: Moment, then Agents tab | Analyst claims with clickable evidence ids (seek the replay), then the agent trace: Director -> Analyst + Commentator in parallel -> Recap Writer + Data Auditor -> Translator | "Explain: six Agent Framework agents in a workflow graph, including a Data Auditor that states where the data comes from; every claim must cite real event ids or it is rejected and retried." |
| 1:05-1:20 | Personalize | Fan mode + Spanish, then Focus player: recap and commentary change | "Personalize: analyst or casual fan, any of four languages, your favourite player." |
| 1:20-1:40 | Real-footage clip (local only) | Viewer header "real footage · tracked from video": anonymised players (A7, B3), ball, events in the ticker. Record locally; never publish the footage itself | "The same schema from real video: detection, pose, ball and event heuristics turn a sideline recording into a replay." |
| 1:40-1:55 | Architecture slide | README mermaid diagram: sim / vision -> store -> analytics -> agents + MCP -> API -> viewer; Azure Container Apps, Cosmos DB, Azure OpenAI | "Production path: Container Apps, Cosmos DB and Azure OpenAI through azd, with an offline fallback." |

Privacy: the real-footage shot shows only the 3D replay (stick figures, anonymous ids), never video frames,
names or venue. If in doubt, drop that shot and say the module exists.
