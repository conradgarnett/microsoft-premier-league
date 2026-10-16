"""The insights pipeline as a Microsoft Agent Framework workflow graph.

    Director --(one MomentJob per key moment)--> Moment --(MomentDone)--> Recap --> Translator --> output

* Director: the Replay Director agent picks the reel, then fans out one job per key moment.
* Moment: Tactical Analyst and Commentator run in parallel on the moment's tool context.
* Recap: fan-in; once every moment has reported, the Recap Writer reads the analysts'
  findings from the shared ``MatchState`` and writes the match recap.
* Translator: renders everything in the requested language and yields the workflow output.

Agent calls go through ``Orchestrator.call`` (JSON contract, evidence validation, retry,
offline fallback), so every step is traced in ``MatchState.trace``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from .orchestrator import MatchState, Orchestrator


@dataclass
class InsightsRequest:
    state: MatchState


@dataclass
class MomentJob:
    state: MatchState
    moment: dict[str, Any]
    total: int


@dataclass
class MomentDone:
    state: MatchState
    moment_id: str
    result: dict[str, Any]
    total: int


@dataclass
class RecapReady:
    state: MatchState


class DirectorExecutor(Executor):
    def __init__(self, orc: Orchestrator) -> None:
        super().__init__(id="replay_director")
        self.orc = orc

    @handler
    async def plan(self, req: InsightsRequest, ctx: WorkflowContext[MomentJob]) -> None:
        st = req.state
        moments = self.orc.tools.report(st.match_id)["key_moments"]
        st.outputs["director"] = await self.orc.call(st, "director", "director", {"moments": moments, "reel_size": 6})
        st.outputs["moments"] = {}
        for km in moments:
            await ctx.send_message(MomentJob(st, km, len(moments)))


class MomentExecutor(Executor):
    def __init__(self, orc: Orchestrator) -> None:
        super().__init__(id="moment_analysis")
        self.orc = orc

    @handler
    async def analyse(self, job: MomentJob, ctx: WorkflowContext[MomentDone]) -> None:
        st = job.state
        mctx = self.orc.tools.get_moment_context(st.match_id, job.moment["id"])
        analysis, commentary = await asyncio.gather(
            self.orc.call(st, "analyst", "analysis", mctx, sender="Replay Director"),
            self.orc.call(st, "commentator", "commentary", mctx, sender="Replay Director"),
        )
        await ctx.send_message(MomentDone(st, job.moment["id"], {"analysis": analysis, "commentary": commentary}, job.total))


class RecapExecutor(Executor):
    def __init__(self, orc: Orchestrator) -> None:
        super().__init__(id="recap_writer")
        self.orc = orc

    @handler
    async def collect(self, done: MomentDone, ctx: WorkflowContext[RecapReady]) -> None:
        st = done.state
        st.outputs["moments"][done.moment_id] = done.result
        if len(st.outputs["moments"]) < done.total:
            return  # fan-in: wait for every moment
        t = self.orc.tools
        report = t.report(st.match_id)
        moments = report["key_moments"]
        rctx: dict[str, Any] = {
            "summary": t.get_match_summary(st.match_id),
            "names": {p.id: p.short_name for p in t.repo.match(st.match_id).players()},
            "goal_moments": [k for k in moments if k["kind"] == "goal"],
            "top_moments": sorted(moments, key=lambda k: k["rank"])[:5],
            "analyst_headlines": {mid: r["analysis"]["headline"] for mid, r in st.outputs["moments"].items()},
            "rhythm": report["rhythm"],
            "rhythm_evidence": [e for b in report["rhythm"] for e in b["evidence"]],
            "pressure_evidence": [e["id"] for e in t.get_events(st.match_id, types=["pressure"], limit=5)],
        }
        if st.focus:
            rctx["focus_stats"] = t.get_player_stats(st.match_id, st.focus)
            rctx["focus_evidence"] = [e["id"] for e in t.get_events(st.match_id, limit=200) if e.get("player_id") == st.focus][:5]
        st.outputs["recap"] = await self.orc.call(st, "recap", "recap", rctx, sender="Tactical Analyst")
        await ctx.send_message(RecapReady(st))


class TranslatorExecutor(Executor):
    def __init__(self, orc: Orchestrator) -> None:
        super().__init__(id="translator")
        self.orc = orc

    @handler
    async def finish(self, msg: RecapReady, ctx: WorkflowContext[None, dict]) -> None:
        st = msg.state
        if st.lang != "en":
            payload = {"moments": st.outputs["moments"], "recap": st.outputs["recap"]}
            tr = await self.orc.call(st, "translator", "translate", {"payload": payload}, lang=st.lang, sender="Recap Writer")
            st.outputs["moments"], st.outputs["recap"] = tr["moments"], tr["recap"]
        await ctx.yield_output(
            {
                "match_id": st.match_id,
                "mode": st.mode,
                "lang": st.lang,
                "focus": st.focus,
                "provider": self.orc.provider,
                "director": st.outputs["director"],
                "moments": st.outputs["moments"],
                "recap": st.outputs["recap"],
                "trace": st.trace,
            }
        )


def build_workflow(orc: Orchestrator):
    director, moment, recap, translator = DirectorExecutor(orc), MomentExecutor(orc), RecapExecutor(orc), TranslatorExecutor(orc)
    return (
        WorkflowBuilder(name="match-insights", description="Replay Director -> Analyst/Commentator per moment -> Recap Writer -> Translator", start_executor=director)
        .add_edge(director, moment)
        .add_edge(moment, recap)
        .add_edge(recap, translator)
        .build()
    )
