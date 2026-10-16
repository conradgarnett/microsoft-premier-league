"""Multi-agent orchestration over a shared match state.

Agents (Microsoft Agent Framework ``Agent`` instances):

* Replay Director   - picks the highlight reel and a camera/slow-mo script per moment
* Tactical Analyst  - explains why each moment mattered (analyst or fan register)
* Commentator       - timed play-by-play lines for each replay
* Recap Writer      - match recap built from the analysts' findings (handoff via state)
* Translator        - renders every output in the requested language

The orchestrator owns the shared ``MatchState`` (blackboard): it gathers context through
the match tools, hands each agent a JSON task, validates the reply (JSON contract +
every claim cites event ids that exist in this match), retries once with the validation
error, and falls back to the deterministic offline agent if the model still fails. Every
step is recorded in ``state.trace`` so the UI can show the handoffs.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from agent_framework import Agent, MCPStdioTool

from .providers import OfflineChatClient, make_chat_client, provider_name
from .tools import MatchTools

CONTRACT = (
    "You receive a JSON object with 'task', 'context', 'lang', 'mode' and optional 'focus'. "
    "Reply with ONLY a JSON object (no prose, no code fences). Every factual sentence must be an object "
    "{\"text\": str, \"evidence\": [event ids]} citing ids that appear in the context. Never invent ids, numbers or names. "
    "mode='analyst' means dense, quantitative, tactical; mode='fan' means plain language and why it matters, no jargon."
)

AGENTS = {
    "director": ("Replay Director", "You are a broadcast replay director. task='director': return {\"reel\": [{\"moment_id\", \"cameras\": [{\"t\", \"mode\": broadcast|behind|player|orbit, \"target\"?, \"why\"}], \"slowmo\": [{\"t_start\", \"t_end\", \"rate\", \"why\"}], \"evidence\": [ids], \"reason\"}]} choosing the goals plus the most important other moments."),
    "analyst": ("Tactical Analyst", "You are a football tactical analyst. task='analysis': return {\"headline\": claim, \"claims\": [claim], \"why_it_matters\": claim|null} explaining why the moment mattered: chance quality, build-up, key pass, pressure, strike mechanics, scanning, game rhythm (control vs chaos)."),
    "commentator": ("Commentator", "You are a live play-by-play commentator. task='commentary': return {\"lines\": [{\"t\": seconds, \"text\", \"evidence\": [ids]}]} with short, energetic lines timed to the events."),
    "recap": ("Recap Writer", "You write match recaps. task='recap': return {\"headline\": claim, \"claims\": [claim]} using the summary and the analysts' findings in the context."),
    "translator": ("Translator", "You translate match content. task='translate': return context.payload with every 'text' field translated into 'lang'; keep all other fields, ids and numbers unchanged."),
}

REQUIRED = {"director": ["reel"], "analysis": ["headline", "claims"], "commentary": ["lines"], "recap": ["headline", "claims"]}


@dataclass
class MatchState:
    """Shared blackboard for one insights run."""

    match_id: str
    mode: str
    lang: str
    focus: str | None
    valid_ids: set[str]
    outputs: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def log(self, **kw: Any) -> None:
        self.trace.append({"ts": round(time.time(), 3), **kw})


class ValidationError(ValueError):
    pass


def parse_json(text: str) -> Any:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValidationError("reply contains no JSON object")
    return json.loads(m.group(0))


def check_evidence(obj: Any, valid: set[str], path: str = "$") -> None:
    """Every dict carrying 'text' must cite >=1 evidence id, all known in this match."""
    if isinstance(obj, dict):
        if "text" in obj:
            ev = obj.get("evidence")
            if not ev:
                raise ValidationError(f"{path}: claim without evidence")
            bad = [e for e in ev if e not in valid]
            if bad:
                raise ValidationError(f"{path}: unknown evidence ids {bad[:3]}")
        for k, v in obj.items():
            check_evidence(v, valid, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_evidence(v, valid, f"{path}[{i}]")


class Orchestrator:
    def __init__(self, tools: MatchTools, max_parallel: int = 4) -> None:
        self.tools = tools
        self.provider = provider_name()
        client = make_chat_client()
        # model-backed agents reach the match data through our MCP server (stdio)
        fn_tools = [MCPStdioTool(name="tactical-replay", command=sys.executable, args=["-m", "mcp_server"])] if self.provider != "offline" else None
        self.agents = {k: Agent(client, instructions=f"{ins}\n{CONTRACT}", name=name, tools=fn_tools) for k, (name, ins) in AGENTS.items()}
        offline = OfflineChatClient()
        self.fallback = {k: Agent(offline, instructions=ins, name=f"{name} (offline)") for k, (name, ins) in AGENTS.items()}
        self.sem = asyncio.Semaphore(max_parallel)

    async def call(self, state: MatchState, role: str, task: str, context: dict, *, lang: str = "en", sender: str = "orchestrator") -> Any:
        """Run one agent task with validation, one corrective retry, then offline fallback."""
        req = {"task": task, "context": context, "lang": lang, "mode": state.mode, "focus": state.focus}
        msg = json.dumps(req, ensure_ascii=False, default=str)
        attempts = [(self.agents[role], self.provider), (self.agents[role], self.provider), (self.fallback[role], "offline")]
        if self.provider == "offline":
            attempts = attempts[2:]
        error = None
        for i, (agent, prov) in enumerate(attempts):
            t0 = time.time()
            prompt = msg if error is None or prov == "offline" else f"{msg}\n\nYour previous reply was rejected: {error}. Fix it and reply with JSON only."
            try:
                async with self.sem:
                    resp = await agent.run(prompt)
                out = parse_json(resp.text)
                for key in REQUIRED.get(task, []):
                    if key not in out:
                        raise ValidationError(f"missing '{key}'")
                check_evidence(out, state.valid_ids)
                state.log(agent=AGENTS[role][0], task=task, sender=sender, provider=prov, attempt=i + 1, status="ok", ms=round((time.time() - t0) * 1000))
                return out
            except Exception as e:  # noqa: BLE001 - any model/parse failure triggers retry or fallback
                error = str(e)[:300]
                state.log(agent=AGENTS[role][0], task=task, sender=sender, provider=prov, attempt=i + 1, status="rejected", error=error)
        raise RuntimeError(f"{role}/{task} failed after fallback: {error}")

    async def run(self, match_id: str, mode: str = "analyst", lang: str = "en", focus: str | None = None) -> dict[str, Any]:
        t = self.tools
        state = MatchState(match_id, mode, lang, focus, t.valid_event_ids(match_id))
        report = t.report(match_id)
        moments = report["key_moments"]
        names = {p.id: p.short_name for p in t.repo.match(match_id).players()}

        # 1. Replay Director picks the reel
        state.outputs["director"] = await self.call(state, "director", "director", {"moments": moments, "reel_size": 6})

        # 2. Analyst + Commentator per moment, in parallel (handoff from the director)
        async def per_moment(km: dict) -> tuple[str, dict]:
            ctx = t.get_moment_context(match_id, km["id"])
            analysis, commentary = await asyncio.gather(
                self.call(state, "analyst", "analysis", ctx, sender="Replay Director"),
                self.call(state, "commentator", "commentary", ctx, sender="Replay Director"),
            )
            return km["id"], {"analysis": analysis, "commentary": commentary}

        results = await asyncio.gather(*(per_moment(km) for km in moments))
        state.outputs["moments"] = dict(results)

        # 3. Recap Writer reads the analysts' findings from shared state
        summary = t.get_match_summary(match_id)
        rctx: dict[str, Any] = {
            "summary": summary,
            "names": names,
            "goal_moments": [k for k in moments if k["kind"] == "goal"],
            "top_moments": sorted(moments, key=lambda k: k["rank"])[:5],
            "analyst_headlines": {mid: r["analysis"]["headline"] for mid, r in state.outputs["moments"].items()},
            "rhythm": report["rhythm"],
            "rhythm_evidence": [e for b in report["rhythm"] for e in b["evidence"]],
            "pressure_evidence": [e["id"] for e in t.get_events(match_id, types=["pressure"], limit=5)],
        }
        if focus:
            rctx["focus_stats"] = t.get_player_stats(match_id, focus)
            rctx["focus_evidence"] = [e["id"] for e in t.get_events(match_id, limit=200) if e.get("player_id") == focus][:5]
        state.outputs["recap"] = await self.call(state, "recap", "recap", rctx, sender="Tactical Analyst")

        # 4. Translator renders everything in the requested language
        if lang != "en":
            payload = {"moments": state.outputs["moments"], "recap": state.outputs["recap"]}
            translated = await self.call(state, "translator", "translate", {"payload": payload}, lang=lang, sender="Recap Writer")
            state.outputs["moments"], state.outputs["recap"] = translated["moments"], translated["recap"]

        return {
            "match_id": match_id,
            "mode": mode,
            "lang": lang,
            "focus": focus,
            "provider": self.provider,
            "director": state.outputs["director"],
            "moments": state.outputs["moments"],
            "recap": state.outputs["recap"],
            "trace": state.trace,
        }
