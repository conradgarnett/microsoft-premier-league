"""Key-moment detection/ranking, replay-window annotations and the full analytics report."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.schema import EventType, LowResTracking, Match, SkeletonWindow, TrackingWindow

from . import body
from .core import (
    Tracking,
    control_chaos,
    pass_metrics,
    physical,
    player_stats,
    pressure_map,
    shot_xg,
    team_stats,
    to_attack_frame,
)


def build_chain(match: Match, shot_idx: int, max_len: int = 8) -> list[str]:
    """Event ids of the on-ball build-up in the same possession leading to the shot."""
    shot = match.events[shot_idx]
    chain = []
    for e in reversed(match.events[:shot_idx]):
        if e.possession_id != shot.possession_id or len(chain) >= max_len:
            break
        if e.type in (EventType.PASS, EventType.CARRY, EventType.TACKLE, EventType.INTERCEPTION, EventType.BALL_RECOVERY):
            chain.append(e.id)
    return list(reversed(chain))


def key_moments(match: Match) -> list[dict[str, Any]]:
    """One moment per replay window, ranked by importance (goals first, then xG and context)."""
    idx = {e.id: i for i, e in enumerate(match.events)}
    score = {match.meta.home_team_id: 0, match.meta.away_team_id: 0}
    goal_of = {g.related_ids[0]: g for g in match.events if g.type == EventType.GOAL and g.related_ids}
    moments = []
    shots_by_window = {sid: w for w in match.windows for sid in w.trigger_event_ids}
    for e in match.events:
        if e.type == EventType.GOAL:
            score[e.team_id] += 1
        if e.type != EventType.SHOT or e.id not in shots_by_window:
            continue
        w = shots_by_window[e.id]
        x = shot_xg(match, e)
        goal = goal_of.get(e.id)
        margin = abs(score[match.meta.home_team_id] - score[match.meta.away_team_id])
        late = e.t > match.meta.duration_s * 0.8
        importance = (100 if goal else 0) + 120 * x + (10 if e.outcome == "saved" else 0) + (8 if late else 0) + (6 if margin <= 1 else 0)
        kind = "goal" if goal else "big_chance" if x >= 0.25 else "save" if e.outcome == "saved" else "shot"
        player = match.player(e.player_id)
        team = match.team(e.team_id)
        title = {
            "goal": f"GOAL - {player.short_name} ({team.short_name})",
            "big_chance": f"Big chance - {player.short_name}",
            "save": f"Save denies {player.short_name}",
            "shot": f"Shot - {player.short_name}",
        }[kind]
        chain = build_chain(match, idx[e.id])
        evidence = chain + [e.id] + ([goal.id] if goal else [])
        moments.append(
            {
                "id": f"km-{e.id}",
                "window_id": w.id,
                "t": e.t,
                "minute": e.minute,
                "kind": kind,
                "title": title,
                "team_id": e.team_id,
                "player_id": e.player_id,
                "shot_event_id": e.id,
                "outcome": e.outcome,
                "xg": x,
                "importance": round(importance, 1),
                "score_after": {k: v + (1 if goal and k == e.team_id else 0) for k, v in score.items()},
                "build_up_event_ids": chain,
                "evidence": evidence,
                "replay": {"t_start": w.t_start, "t_end": w.t_end},
            }
        )
    ranked = sorted(moments, key=lambda m: -m["importance"])
    for r, m in enumerate(ranked):
        m["rank"] = r + 1
    return sorted(moments, key=lambda m: m["t"])


def annotate_window(match: Match, win: TrackingWindow, sk: SkeletonWindow | None) -> dict[str, Any]:
    """Overlay data for the 3D viewer: pass lines, shots, strike angles, scanning, sprints."""
    evs = {e.id: e for e in match.events}
    t0, t1 = win.t_start, win.t_start + win.n_frames / win.fps
    passes, shots = [], []
    for e in match.events:
        if not (t0 <= e.t <= t1):
            continue
        if e.type == EventType.PASS:
            pm = pass_metrics(match, e, None)
            passes.append({**pm, "t": e.t, "player_id": e.player_id, "receiver_id": e.receiver_id, "start": [e.x, e.y], "end": [e.end_x, e.end_y], "outcome": e.outcome, "speed_mps": e.attributes.get("ball_speed_mps")})
        elif e.type == EventType.SHOT:
            shots.append({"event_id": e.id, "t": e.t, "player_id": e.player_id, "xg": shot_xg(match, e), "outcome": e.outcome, "start": [e.x, e.y], "end": [e.end_x, e.end_y, e.attributes.get("end_z", 0)], "speed_mps": measured_ball_speed(win, e.t) or e.attributes.get("ball_speed_mps")})
    strikes, scans = [], []
    if sk is not None:
        for a in win.actions:
            if not (t0 <= a["t"] <= t1):
                continue
            if a["kind"] == "kick":
                s = body.strike_analysis(sk, a["player_id"], a["t"], a.get("foot", "right"))
                s["event_id"] = a.get("event_id")
                s["is_shot"] = bool(a.get("shot"))
                strikes.append(s)
            elif a["kind"] == "receive" and a["t"] - t0 > 1.0:
                sc = body.scanning(sk, a["player_id"], a["t"])
                ev = next((e for e in match.events if e.receiver_id == a["player_id"] and abs(e.t - a["t"]) < 4 and e.type == EventType.PASS), None)
                if ev is not None:
                    gx = 105.0 if match.attack_direction(ev.team_id, ev.period) == 1 else 0.0
                    sc["body_to_goal_deg"] = body.body_orientation(sk, a["player_id"], a["t"], (gx, 34.0))
                    sc["open_body"] = sc["body_to_goal_deg"] < 70
                    sc["pass_event_id"] = ev.id
                scans.append(sc)
    # sprints inside the window (25 fps)
    pos = np.asarray(win.players, float)
    sp = np.linalg.norm(np.gradient(pos, 1 / win.fps, axis=0), axis=-1)
    sprints = []
    for p, pid in enumerate(win.player_ids):
        fast = sp[:, p] > 7.0
        f = 0
        while f < len(fast):
            if fast[f]:
                g = f
                while g < len(fast) and fast[g]:
                    g += 1
                if (g - f) / win.fps >= 0.8:
                    sprints.append({"player_id": pid, "t_start": round(t0 + f / win.fps, 2), "t_end": round(t0 + g / win.fps, 2), "top_speed_kmh": round(float(sp[f:g, p].max()) * 3.6, 1)})
                f = g
            f += 1
    return {
        "window_id": win.id,
        "passes": passes,
        "shots": shots,
        "strikes": strikes,
        "scans": scans,
        "sprints": sprints,
        "events": [evs[i].model_dump(mode="json") for i in win.event_ids if i in evs],
    }


def measured_ball_speed(win: TrackingWindow, t: float) -> float | None:
    """Peak ball speed (m/s) measured from 25 fps tracking in the 0.3 s after ``t``."""
    f = int(round((t - win.t_start) * win.fps))
    ball = np.asarray(win.ball, float)
    if f < 0 or f + 3 >= len(ball):
        return None
    seg = ball[f : f + int(0.3 * win.fps) + 1]
    v = np.linalg.norm(np.diff(seg, axis=0), axis=1) * win.fps
    return round(float(v.max()), 1)


def full_report(match: Match, lowres: LowResTracking | None) -> dict[str, Any]:
    trk = Tracking(match, lowres) if lowres is not None else None
    passes = {e.id: pass_metrics(match, e, trk) for e in match.events if e.type == EventType.PASS}
    phys = physical(trk) if trk is not None else {}
    shots = []
    for e in match.events:
        if e.type == EventType.SHOT:
            x, y = to_attack_frame(match, e.team_id, e.period, e.x, e.y)
            shots.append({"event_id": e.id, "t": e.t, "team_id": e.team_id, "player_id": e.player_id, "xg": shot_xg(match, e), "outcome": e.outcome, "x": round(x, 1), "y": round(y, 1)})
    return {
        "match_id": match.meta.match_id,
        "teams": team_stats(match, trk, passes),
        "players": player_stats(match, passes, phys),
        "rhythm": control_chaos(match, passes),
        "pressure_map": pressure_map(match),
        "shots": shots,
        "key_moments": key_moments(match),
        "passes": passes,
    }
