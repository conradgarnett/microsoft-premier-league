"""Match tools shared by the agents (as Agent Framework function tools) and the MCP server.

Every result carries event ids so agents can cite evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analytics import annotate_window, full_report
from analytics.body import joint_angles
from api.store import MatchRepo

MAX_EVENTS = 200


class MatchTools:
    def __init__(self, repo: MatchRepo) -> None:
        self.repo = repo
        self._reports: dict[str, dict[str, Any]] = {}

    def report(self, match_id: str) -> dict[str, Any]:
        if match_id not in self._reports:
            doc = self.repo.store.get_doc(match_id, "analytics")
            if doc is None:
                doc = full_report(self.repo.match(match_id), self.repo.lowres(match_id))
                self.repo.store.put_doc(match_id, "analytics", doc)
            self._reports[match_id] = doc
        return self._reports[match_id]

    # ---- tools -------------------------------------------------------------
    def get_match_summary(self, match_id: str) -> dict[str, Any]:
        """Teams, final score and team-level metrics for a match."""
        m = self.repo.match(match_id)
        r = self.report(match_id)
        return {
            "match_id": match_id,
            "teams": [{"id": t.id, "name": t.name, "short_name": t.short_name, "formation": t.formation} for t in m.teams],
            "home_team_id": m.meta.home_team_id,
            "source": m.meta.source,
            "duration_s": m.meta.duration_s,
            "score": m.meta.final_score.model_dump(),
            "team_stats": r["teams"],
            "goals": [{"event_id": e.id, "minute": e.minute, "player_id": e.player_id, "team_id": e.team_id} for e in m.events if e.type.value == "goal"],
        }

    def get_events(self, match_id: str, start_t: float = 0, end_t: float = 1e9, types: list[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Events between start_t and end_t seconds, optionally filtered by type."""
        m = self.repo.match(match_id)
        limit = int(np.clip(limit, 1, MAX_EVENTS))
        out = [e.model_dump(mode="json", exclude_defaults=True) for e in m.events if start_t <= e.t <= end_t and (not types or e.type.value in types)]
        return out[:limit]

    def get_key_moments(self, match_id: str, top: int = 10) -> list[dict[str, Any]]:
        """Key moments ranked by importance (goals, big chances, saves)."""
        km = sorted(self.report(match_id)["key_moments"], key=lambda k: k["rank"])
        return km[: max(1, int(top))]

    def get_player_stats(self, match_id: str, player_id: str) -> dict[str, Any]:
        """Season-style stat line for one player in this match (passing, shooting, physical)."""
        players = self.report(match_id)["players"]
        if player_id not in players:
            raise KeyError(f"unknown player {player_id}")
        return {"player_id": player_id, **players[player_id]}

    def get_skeleton_window(self, match_id: str, window_id: str, player_id: str | None = None, stride: int = 5) -> dict[str, Any]:
        """Joint-angle summary (and down-sampled keypoints for one player) for a replay window."""
        m = self.repo.match(match_id)
        win = self.repo.window(match_id, window_id)
        sk = self.repo.skeleton(match_id, window_id)
        ann = annotate_window(m, win, sk)
        out: dict[str, Any] = {"window_id": window_id, "t_start": sk.t_start, "fps": sk.fps, "strikes": ann["strikes"], "scans": ann["scans"]}
        if player_id:
            p = sk.player_ids.index(player_id)
            arr = np.asarray(sk.frames, float)[:: max(1, stride), p]
            ang = joint_angles(arr)
            out["player_id"] = player_id
            out["joint_names"] = sk.joint_names
            out["keypoints"] = np.round(arr, 2).tolist()
            out["angles"] = {k: np.round(v, 1).tolist() for k, v in ang.items()}
        return out

    def get_pressure_map(self, match_id: str, team_id: str | None = None) -> dict[str, Any]:
        """Grid of pressure events per team, normalised so each team attacks to the right."""
        pm = self.report(match_id)["pressure_map"]
        if team_id:
            return {**pm, "grids": {team_id: pm["grids"][team_id]}}
        return pm

    def get_moment_context(self, match_id: str, moment_id: str) -> dict[str, Any]:
        """Everything known about one key moment: build-up events, pass metrics, biomechanics, rhythm."""
        r = self.report(match_id)
        km = next(k for k in r["key_moments"] if k["id"] == moment_id)
        m = self.repo.match(match_id)
        evs = {e.id: e for e in m.events}
        win = self.repo.window(match_id, km["window_id"])
        sk = self.repo.skeleton(match_id, km["window_id"])
        ann = annotate_window(m, win, sk)
        rhythm = next((b for b in r["rhythm"] if b["t_start"] <= km["t"] < b["t_end"]), r["rhythm"][-1])
        names = {p.id: p.short_name for p in m.players()}
        return {
            "moment": km,
            "build_up": [evs[i].model_dump(mode="json", exclude_defaults=True) for i in km["build_up_event_ids"]],
            "pass_metrics": [r["passes"][i] for i in km["build_up_event_ids"] if i in r["passes"]],
            "shot": next((s for s in ann["shots"] if s["event_id"] == km["shot_event_id"]), None),
            "shot_event": evs[km["shot_event_id"]].model_dump(mode="json", exclude_defaults=True),
            "strike": next((s for s in ann["strikes"] if s.get("event_id") == km["shot_event_id"]), None),
            "scans": [s for s in ann["scans"] if s.get("pass_event_id") in km["build_up_event_ids"]],
            "pressure_events": [e.model_dump(mode="json", exclude_defaults=True) for e in m.events if e.type.value == "pressure" and km["t"] - 6 <= e.t <= km["t"]],
            "rhythm": rhythm,
            "names": names,
            "teams": {t.id: t.name for t in m.teams},
            "home_team_id": m.meta.home_team_id,
        }

    def valid_event_ids(self, match_id: str) -> set[str]:
        return {e.id for e in self.repo.match(match_id).events}

    def public_tools(self) -> list:
        """Callables exposed to LLM agents and the MCP server."""
        return [self.get_match_summary, self.get_events, self.get_key_moments, self.get_player_stats, self.get_skeleton_window, self.get_pressure_map]
