"""Match analytics computed from the source-agnostic schema (events + tracking).

Nothing here reads engine internals: every metric derives from ``Match`` events and
``LowResTracking`` / ``TrackingWindow`` / ``SkeletonWindow`` data, so the same code
works on data from a future video-to-skeleton pipeline.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from sim.schema import (
    GOAL_Y_HIGH,
    GOAL_Y_LOW,
    PITCH_LENGTH,
    PITCH_WIDTH,
    EventType,
    LowResTracking,
    Match,
)

L, W = PITCH_LENGTH, PITCH_WIDTH
SPRINT_MPS = 7.0  # ~25 km/h, common broadcast sprint threshold
DEF_ACTIONS = {EventType.TACKLE, EventType.INTERCEPTION, EventType.FOUL}


class Tracking:
    """numpy view over LowResTracking with team-frame helpers."""

    def __init__(self, match: Match, lowres: LowResTracking) -> None:
        self.match = match
        self.fps = lowres.fps
        self.pos = np.asarray(lowres.players, float)  # (F,22,2)
        self.ball = np.asarray(lowres.ball, float)
        self.ids = lowres.player_ids
        self.index = {pid: i for i, pid in enumerate(self.ids)}
        self.team_idx = {t.id: [self.index[p.id] for p in t.players] for t in match.teams}

    def frame(self, t: float) -> int:
        return int(np.clip(round(t * self.fps), 0, len(self.pos) - 1))

    def opponents_at(self, t: float, team_id: str) -> np.ndarray:
        other = next(tid for tid in self.team_idx if tid != team_id)
        return self.pos[self.frame(t), self.team_idx[other]]


def to_attack_frame(match: Match, team_id: str, period: int, x: float, y: float) -> tuple[float, float]:
    """Rotate pitch coordinates so ``team_id`` attacks towards x=105."""
    if match.attack_direction(team_id, period) == 1:
        return x, y
    return L - x, W - y


def seg_dist(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    s = np.clip(((points - a) @ ab) / (float(ab @ ab) or 1e-9), 0, 1)
    return np.linalg.norm(points - (a + s[:, None] * ab), axis=1)


# ----------------------------------------------------------------------------- xG
def xg(x: float, y: float, under_pressure: bool = False, header: bool = False) -> float:
    """Geometric expected-goals model in the attack frame (goal at x=105).

    Logistic on distance and the angle subtended by the posts; pressure and headers
    reduce the chance. Coefficients are hand-calibrated so a central penalty-spot shot
    is ~0.2 and a 25 m central shot is ~0.04.
    """
    dx = max(L - x, 0.5)
    dist = math.hypot(dx, y - W / 2)
    angle = abs(math.atan2(GOAL_Y_HIGH - y, dx) - math.atan2(GOAL_Y_LOW - y, dx))
    z = -1.1 - 0.11 * dist + 1.4 * angle - (0.25 if under_pressure else 0) - (0.6 if header else 0)
    return round(1 / (1 + math.exp(-z)), 3)


def shot_xg(match: Match, ev) -> float:
    x, y = to_attack_frame(match, ev.team_id, ev.period, ev.x, ev.y)
    return xg(x, y, ev.under_pressure, "head" in str(ev.attributes.get("body_part", "")))


# ----------------------------------------------------------------------------- passes
def pass_metrics(match: Match, ev, trk: Tracking | None) -> dict[str, Any]:
    """Distance, progression and a 0..1 difficulty score for one pass."""
    a = np.array([ev.x, ev.y])
    b = np.array([ev.end_x, ev.end_y])
    length = float(np.linalg.norm(b - a))
    sx, _ = to_attack_frame(match, ev.team_id, ev.period, ev.x, ev.y)
    ex, ey = to_attack_frame(match, ev.team_id, ev.period, ev.end_x, ev.end_y)
    if trk is not None:
        opp = trk.opponents_at(ev.t, ev.team_id)
        pressure = float(np.min(np.linalg.norm(opp - a, axis=1)))
        lane = float(np.min(seg_dist(opp, a, b)))
        space = float(np.min(np.linalg.norm(trk.opponents_at(ev.t + length / 15, ev.team_id) - b, axis=1)))
    else:  # shortcut: fall back to producer-supplied context when no tracking exists
        pressure = float(ev.attributes.get("passer_pressure_m", 5))
        lane = float(ev.attributes.get("lane_clearance_m", 5))
        space = float(ev.attributes.get("receiver_space_m", 5))
    lofted = ev.attributes.get("height") == "lofted"
    z = (
        -2.2
        + 0.045 * length
        + 1.1 * math.exp(-pressure / 1.5)
        + 1.4 * math.exp(-lane / 1.2)
        + 1.0 * math.exp(-space / 2.0)
        + (0.4 if lofted else 0)
        + (0.5 if ev.attributes.get("through_ball") else 0)
    )
    return {
        "event_id": ev.id,
        "length_m": round(length, 1),
        "progression_m": round(ex - sx, 1),
        "progressive": (ex - sx) >= 10 or (ex > 88.5 and abs(ey - 34) < 20.16 and sx <= 88.5),
        "into_final_third": ex > 70 and sx <= 70,
        "difficulty": round(1 / (1 + math.exp(-z)), 3),
        "passer_pressure_m": round(pressure, 2),
        "lane_clearance_m": round(lane, 2),
        "receiver_space_m": round(space, 2),
        "complete": ev.outcome == "complete",
    }


# ----------------------------------------------------------------------------- physical
def physical(trk: Tracking) -> dict[str, dict[str, float]]:
    """Distance, top speed and sprints per player from whole-match tracking."""
    dt = 1 / trk.fps
    step = np.linalg.norm(np.diff(trk.pos, axis=0), axis=-1)  # (F-1,22)
    step = np.where(step > 12 * dt, 0, step)  # drop teleports (half-time resets)
    speed = step / dt
    sm = np.apply_along_axis(lambda s: np.convolve(s, np.ones(3) / 3, mode="same"), 0, speed)
    out = {}
    for i, pid in enumerate(trk.ids):
        fast = sm[:, i] > SPRINT_MPS
        # sprints: runs of >= 1 s above threshold
        runs, n = [], 0
        for f in fast:
            n = n + 1 if f else (runs.append(n) or 0) if n else 0
        runs.append(n)
        sprints = [r for r in runs if r * dt >= 1.0]
        out[pid] = {
            "distance_km": round(float(step[:, i].sum()) / 1000, 2),
            "top_speed_kmh": round(float(sm[:, i].max()) * 3.6, 1),
            "sprints": len(sprints),
            "sprint_distance_m": round(float(step[:, i][fast].sum()), 0),
        }
    return out


# ----------------------------------------------------------------------------- team
def possession_spans(match: Match) -> list[tuple[float, float, str]]:
    """(start, end, team_id) spans of team possession from possession_change events."""
    spans, cur, start = [], None, 0.0
    for e in match.events:
        if e.type in (EventType.HALF_START, EventType.POSSESSION_CHANGE) and e.team_id:
            if cur and e.team_id != cur:
                spans.append((start, e.t, cur))
                start = e.t
            elif cur is None:
                start = e.t
            cur = e.team_id
    if cur:
        spans.append((start, match.meta.duration_s, cur))
    return spans


def team_stats(match: Match, trk: Tracking | None, passes: dict[str, dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    spans = possession_spans(match)
    total = sum(b - a for a, b, _ in spans) or 1
    for team in match.teams:
        tid = team.id
        opp = next(t.id for t in match.teams if t.id != tid)
        evs = [e for e in match.events if e.team_id == tid]
        tp = [e for e in evs if e.type == EventType.PASS]
        shots = [e for e in evs if e.type == EventType.SHOT]
        # PPDA: opponent passes in their own 60% / our defensive actions in that zone
        opp_passes = 0
        for e in match.events:
            if e.type == EventType.PASS and e.team_id == opp:
                x, _ = to_attack_frame(match, opp, e.period, e.x, e.y)
                opp_passes += x < 0.6 * L
        def_actions = 0
        for e in evs:
            if e.type in DEF_ACTIONS:
                x, _ = to_attack_frame(match, tid, e.period, e.x, e.y)
                def_actions += x > 0.4 * L
        final_third = sum(1 for e in tp if to_attack_frame(match, tid, e.period, e.x, e.y)[0] > 70)
        out[tid] = {
            "possession_pct": round(100 * sum(b - a for a, b, t in spans if t == tid) / total, 1),
            "passes": len(tp),
            "pass_completion_pct": round(100 * sum(e.outcome == "complete" for e in tp) / max(len(tp), 1), 1),
            "progressive_passes": sum(passes[e.id]["progressive"] for e in tp),
            "shots": len(shots),
            "shots_on_target": sum(bool(e.attributes.get("on_target")) for e in shots),
            "goals": sum(e.type == EventType.GOAL for e in evs),
            "xg": round(sum(shot_xg(match, e) for e in shots), 2),
            "tackles_won": sum(e.type == EventType.TACKLE and e.outcome == "won" for e in evs),
            "interceptions": sum(e.type == EventType.INTERCEPTION for e in evs),
            "pressures": sum(e.type == EventType.PRESSURE for e in evs),
            "fouls": sum(e.type == EventType.FOUL for e in evs),
            "ppda": round(opp_passes / max(def_actions, 1), 2),
            "_final_third_passes": final_third,
        }
        if trk is not None:
            out[tid]["defensive_line_m"] = defensive_line(match, trk, tid)
    tilt_total = sum(v["_final_third_passes"] for v in out.values()) or 1
    for v in out.values():
        v["field_tilt_pct"] = round(100 * v.pop("_final_third_passes") / tilt_total, 1)
    return out


def defensive_line(match: Match, trk: Tracking, team_id: str) -> float:
    """Mean distance (m) from own goal of the four deepest outfield players, out of possession."""
    spans = [(a, b) for a, b, t in possession_spans(match) if t != team_id]
    idx = trk.team_idx[team_id][1:]  # skip goalkeeper (first in roster)
    vals = []
    for a, b in spans:
        for f in range(trk.frame(a), trk.frame(b), 2):
            period = 1 if f / trk.fps < match.meta.half_length_s else 2
            x = trk.pos[f, idx, 0]
            if match.attack_direction(team_id, period) == -1:
                x = L - x
            vals.append(np.sort(x)[:4].mean())
    return round(float(np.mean(vals)), 1) if vals else 0.0


# ----------------------------------------------------------------------------- rhythm
def control_chaos(match: Match, passes: dict[str, dict], bin_s: float = 300) -> list[dict[str, Any]]:
    """Control-vs-chaos index per time bin.

    0 = calm, controlled possession (long spells, completed passes, little pressure);
    100 = chaos (frequent turnovers, failed passes, heavy pressing). Also reports which
    team controlled the bin and each side's xG for a momentum chart.
    """
    home, away = match.meta.home_team_id, match.meta.away_team_id
    bins = []
    n = int(math.ceil(match.meta.duration_s / bin_s))
    spans = possession_spans(match)
    for k in range(n):
        a, b = k * bin_s, min((k + 1) * bin_s, match.meta.duration_s)
        evs = [e for e in match.events if a <= e.t < b]
        mins = (b - a) / 60
        changes = sum(e.type == EventType.POSSESSION_CHANGE for e in evs) / mins
        ps = [e for e in evs if e.type == EventType.PASS]
        fail = 1 - sum(e.outcome == "complete" for e in ps) / max(len(ps), 1)
        press = sum(e.type == EventType.PRESSURE for e in evs) / mins
        diff = np.mean([passes[e.id]["difficulty"] for e in ps]) if ps else 0.3
        z = 0.35 * (changes - 4.5) + 4.0 * (fail - 0.22) + 0.12 * (press - 7) + 3 * (diff - 0.3)
        chaos = 100 / (1 + math.exp(-z))
        poss_home = sum(max(0, min(e_, b) - max(s_, a)) for s_, e_, t in spans if t == home) / (b - a)
        xg_bin = defaultdict(float)
        for e in evs:
            if e.type == EventType.SHOT:
                xg_bin[e.team_id] += shot_xg(match, e)
        bins.append(
            {
                "t_start": a,
                "t_end": b,
                "chaos_index": round(chaos, 1),
                "turnovers_per_min": round(changes, 2),
                "pass_fail_rate": round(fail, 3),
                "pressures_per_min": round(press, 2),
                "home_possession": round(poss_home, 3),
                "controlling_team": home if poss_home >= 0.5 else away,
                "xg": {home: round(xg_bin[home], 3), away: round(xg_bin[away], 3)},
                "evidence": [e.id for e in evs if e.type in (EventType.SHOT, EventType.POSSESSION_CHANGE)][:8],
            }
        )
    return bins


def pressure_map(match: Match, team_id: str | None = None, nx: int = 12, ny: int = 8) -> dict[str, Any]:
    """Grid counts of pressure events (attack frame of the pressing team)."""
    grids = {}
    for team in match.teams:
        if team_id and team.id != team_id:
            continue
        g = np.zeros((ny, nx), int)
        for e in match.events:
            if e.type == EventType.PRESSURE and e.team_id == team.id:
                x, y = to_attack_frame(match, team.id, e.period, e.x, e.y)
                g[min(int(y / W * ny), ny - 1), min(int(x / L * nx), nx - 1)] += 1
        grids[team.id] = g.tolist()
    return {"nx": nx, "ny": ny, "frame": "attack_right", "grids": grids}


# ----------------------------------------------------------------------------- players
def player_stats(match: Match, passes: dict[str, dict], phys: dict[str, dict]) -> dict[str, dict[str, Any]]:
    out = {}
    for p in match.players():
        evs = [e for e in match.events if e.player_id == p.id]
        ps = [e for e in evs if e.type == EventType.PASS]
        shots = [e for e in evs if e.type == EventType.SHOT]
        out[p.id] = {
            "name": p.name,
            "team_id": p.team_id,
            "position": p.position.value,
            "passes": len(ps),
            "pass_completion_pct": round(100 * sum(e.outcome == "complete" for e in ps) / max(len(ps), 1), 1),
            "avg_pass_difficulty": round(float(np.mean([passes[e.id]["difficulty"] for e in ps])), 3) if ps else None,
            "progressive_passes": sum(passes[e.id]["progressive"] for e in ps),
            "shots": len(shots),
            "goals": sum(e.type == EventType.GOAL for e in evs),
            "xg": round(sum(shot_xg(match, e) for e in shots), 2),
            "tackles_won": sum(e.type == EventType.TACKLE and e.outcome == "won" for e in evs),
            "interceptions": sum(e.type == EventType.INTERCEPTION for e in evs),
            "pressures": sum(e.type == EventType.PRESSURE for e in evs),
            **phys.get(p.id, {}),
        }
    return out
