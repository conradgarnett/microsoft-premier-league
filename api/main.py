"""FastAPI backend: matches, events, key moments, replay windows, metrics and live streaming."""

from __future__ import annotations

import asyncio
import os
import pathlib
import threading
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.orchestrator import Orchestrator
from agents.providers import provider_name
from agents.tools import MatchTools
from analytics import annotate_window

from .store import MatchRepo, make_store

repo = MatchRepo(make_store())
tools = MatchTools(repo)
orchestrator = Orchestrator(tools)

app = FastAPI(title="3D Tactical Replay API", version="0.1.0")
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])


def _get(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except (KeyError, ValueError, StopIteration, FileNotFoundError) as e:
        raise HTTPException(404, f"not found: {e}") from e


class GenerateRequest(BaseModel):
    seed: int = Field(7, ge=0, le=2**31)
    minutes: float = Field(90, ge=2, le=120)


@app.on_event("startup")
def ensure_default_match() -> None:
    if not repo.store.list_matches() and os.getenv("AUTOGENERATE", "1") == "1":
        repo.generate(int(os.getenv("DEFAULT_SEED", "7")), float(os.getenv("DEFAULT_MINUTES", "90")))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "matches": len(repo.store.list_matches())}


@app.get("/api/matches")
def list_matches() -> list[dict[str, Any]]:
    out = []
    for mid in repo.store.list_matches():
        m = repo.match(mid)
        out.append({"match_id": mid, "home": m.teams[0].name, "away": m.teams[1].name, "score": m.meta.final_score.model_dump(), "source": m.meta.source})
    return out


_generating = threading.Lock()


@app.post("/api/matches/generate")
def generate(req: GenerateRequest) -> dict[str, str]:
    # a 90-minute simulation is ~1 CPU-minute: run one at a time instead of queueing threads
    if not _generating.acquire(blocking=False):
        raise HTTPException(429, "a match is already being generated; retry shortly")
    try:
        mid = repo.generate(req.seed, req.minutes)
    finally:
        _generating.release()
    tools._reports.pop(mid, None)
    return {"match_id": mid}


@app.get("/api/agents")
def agents_info() -> dict[str, Any]:
    from agents.orchestrator import AGENTS

    return {"provider": provider_name(), "agents": [{"id": k, "name": v[0]} for k, v in AGENTS.items()]}


@app.get("/api/matches/{match_id}/insights")
async def insights(match_id: str, mode: Literal["fan", "analyst"] = "analyst", lang: Literal["en", "es", "fr", "pt"] = "en", focus: str | None = None) -> dict[str, Any]:
    """Multi-agent explanations (cached per mode/language/focus player)."""
    m = _get(repo.match, match_id)
    if focus and focus not in {p.id for p in m.players()}:
        raise HTTPException(404, f"unknown player {focus}")
    key = f"insights/{mode}_{lang}_{focus or 'none'}"
    doc = repo.store.get_doc(match_id, key)
    if doc is None:
        doc = await orchestrator.run(match_id, mode, lang, focus)
        repo.store.put_doc(match_id, key, doc)
    return doc


@app.get("/api/matches/{match_id}")
def get_match(match_id: str) -> dict[str, Any]:
    m = _get(repo.match, match_id)
    return {"meta": m.meta.model_dump(), "teams": [t.model_dump() for t in m.teams], "windows": [w.model_dump() for w in m.windows]}


@app.get("/api/matches/{match_id}/events")
def events(match_id: str, start_t: float = 0, end_t: float = 1e9, types: str | None = None, limit: int = Query(5000, le=20000)) -> list[dict[str, Any]]:
    m = _get(repo.match, match_id)
    want = set(types.split(",")) if types else None
    return [e.model_dump(mode="json") for e in m.events if start_t <= e.t <= end_t and (not want or e.type.value in want)][:limit]


@app.get("/api/matches/{match_id}/key-moments")
def key_moments(match_id: str) -> list[dict[str, Any]]:
    return _get(tools.report, match_id)["key_moments"]


@app.get("/api/matches/{match_id}/metrics")
def metrics(match_id: str) -> dict[str, Any]:
    r = _get(tools.report, match_id)
    return {k: v for k, v in r.items() if k != "passes"}


@app.get("/api/matches/{match_id}/players/{player_id}")
def player(match_id: str, player_id: str) -> dict[str, Any]:
    return _get(tools.get_player_stats, match_id, player_id)


@app.get("/api/matches/{match_id}/replay/{window_id}")
def replay(match_id: str, window_id: str) -> dict[str, Any]:
    """Tracking + skeleton + overlay annotations for one replay window."""
    m = _get(repo.match, match_id)
    win = _get(repo.window, match_id, window_id)
    sk = _get(repo.skeleton, match_id, window_id)
    return {
        "tracking": win.model_dump(),
        "skeleton": {"source": sk.source, "t_start": sk.t_start, "fps": sk.fps, "n_frames": sk.n_frames, "joint_names": sk.joint_names, "bones": sk.bones, "player_ids": sk.player_ids, "frames": sk.frames},
        "annotations": annotate_window(m, win, sk),
    }


@app.websocket("/api/matches/{match_id}/live")
async def live(ws: WebSocket, match_id: str, speed: float = 10.0, start_t: float = 0.0) -> None:
    """Replays the match in simulated real time: event messages plus 2 fps position frames."""
    await ws.accept()
    try:
        m = repo.match(match_id)
        lr = repo.lowres(match_id)
    except KeyError:
        await ws.close(code=4404)
        return
    speed = max(0.5, min(speed, 120.0))
    events = [e for e in m.events if e.t >= start_t]
    ei, f = 0, int(start_t * lr.fps)
    try:
        while f < lr.n_frames:
            t = f / lr.fps
            while ei < len(events) and events[ei].t <= t:
                await ws.send_json({"type": "event", "event": events[ei].model_dump(mode="json")})
                ei += 1
            await ws.send_json({"type": "frame", "t": t, "ball": lr.ball[f], "players": lr.players[f]})
            f += 1
            await asyncio.sleep(1 / lr.fps / speed)
        await ws.send_json({"type": "end"})
    except WebSocketDisconnect:
        return


# Serve the built frontend (web/dist) when present, so one container runs everything.
_dist = pathlib.Path(os.getenv("WEB_DIST", pathlib.Path(__file__).resolve().parent.parent / "web" / "dist"))
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
