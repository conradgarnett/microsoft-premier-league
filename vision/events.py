"""Stage 3: on-ball events from tracking (players + ball), so video clips get an event stream.

Heuristics, in order:

* **Control**: the nearest observed player within ``control_m`` of a known, low ball
  controls it; runs shorter than ``min_frames`` are dropped (ball rolling past).
* **Pass**: control moves to a team-mate -> complete pass from the last controlling
  frame of the passer to the first of the receiver.
* **Possession change**: control moves to an opponent. If the ball travelled more than
  ``pass_min_m`` in between, the first player's action is also an incomplete pass.
* **Shot**: after a release, the ball moves faster than ``shot_mps`` heading for a goal
  mouth (projected crossing within ``mouth_margin_m`` of the posts) from within 35 m.
  Keepers never shoot (their long clearances look like shots). Outcome: ``goal`` if it crosses the line between the posts below the bar, ``off_target``
  if it crosses elsewhere, otherwise ``saved`` (keeper next in control) or ``blocked``.

The same code runs on synthetic windows, where the engine's own events are ground truth
(``tests/test_vision_events.py`` measures precision/recall there).
"""

from __future__ import annotations

import numpy as np

from sim.schema import GOAL_HEIGHT, GOAL_Y_HIGH, GOAL_Y_LOW, PITCH_LENGTH, Event, EventType


def control_runs(ball: np.ndarray, players: np.ndarray, known: np.ndarray, visible: np.ndarray, control_m: float, min_frames: int) -> list[tuple[int, int, int]]:
    """-> [(player index, first frame, last frame)] of uninterrupted ball control."""
    d = np.linalg.norm(players - ball[:, None, :2], axis=2)
    d[~visible] = np.inf
    near = np.argmin(d, axis=1)
    owner = np.where(known & (ball[:, 2] < 1.2) & (d[np.arange(len(d)), near] < control_m), near, -1)
    runs: list[tuple[int, int, int]] = []
    f = 0
    while f < len(owner):
        g = f
        while g + 1 < len(owner) and owner[g + 1] == owner[f]:
            g += 1
        if owner[f] >= 0 and g - f + 1 >= min_frames:
            if runs and runs[-1][0] == owner[f]:
                runs[-1] = (runs[-1][0], runs[-1][1], g)  # same player again after a brief loss: merge
            else:
                runs.append((int(owner[f]), f, g))
        f = g + 1
    return runs


def _shot(ball: np.ndarray, known: np.ndarray, f0: int, f1: int, fps: float, shot_mps: float, mouth_margin_m: float) -> dict | None:
    """Shot check for the flight between release frame f0 and the next control (f1)."""
    k = max(2, int(0.2 * fps))
    if f0 + k >= len(ball) or not known[f0 : f0 + k + 1].all():
        return None
    v = (ball[f0 + k, :2] - ball[f0, :2]) / (k / fps)
    speed = float(np.linalg.norm(v))
    if speed < shot_mps or abs(v[0]) < 1e-6:
        return None
    gx = PITCH_LENGTH if v[0] > 0 else 0.0
    if abs(gx - ball[f0, 0]) > 35:
        return None
    y_cross = ball[f0, 1] + v[1] * (gx - ball[f0, 0]) / v[0]
    if not GOAL_Y_LOW - mouth_margin_m <= y_cross <= GOAL_Y_HIGH + mouth_margin_m:
        return None
    out = "blocked"
    for f in range(f0, min(f1, len(ball))):
        if known[f] and (ball[f, 0] < -0.11 or ball[f, 0] > PITCH_LENGTH + 0.11):  # whole ball over the line
            inside = GOAL_Y_LOW <= ball[f, 1] <= GOAL_Y_HIGH and ball[f, 2] <= GOAL_HEIGHT
            out = "goal" if inside else "off_target"
            return {"speed": speed, "y_cross": float(y_cross), "outcome": out, "line_frame": f}
    return {"speed": speed, "y_cross": float(y_cross), "outcome": out, "line_frame": None}


def detect_events(
    t0: float,
    fps: float,
    ball: np.ndarray,
    players: np.ndarray,
    player_ids: list[str],
    team_of: dict[str, str],
    known: np.ndarray | None = None,
    visible: np.ndarray | None = None,
    keepers: set[str] = frozenset(),
    control_m: float = 1.5,
    min_frames: int = 3,
    pass_min_m: float = 5.0,
    shot_mps: float = 12.0,
    mouth_margin_m: float = 5.0,
    id_prefix: str = "vx",
) -> list[Event]:
    """ball (F,3), players (F,P,2) in schema pitch metres -> time-ordered Events with stable ids."""
    F = len(ball)
    known = np.ones(F, bool) if known is None else known
    visible = np.ones(players.shape[:2], bool) if visible is None else visible
    runs = control_runs(ball, players, known, visible, control_m, min_frames)
    raw: list[dict] = []

    def add(f: int, kind: EventType, p: int, **kw) -> None:
        raw.append({"f": f, "type": kind, "player_id": player_ids[p], "team_id": team_of.get(player_ids[p]), "x": float(ball[f, 0]), "y": float(ball[f, 1]), **kw})

    for (pa, _, fa1), (pb, fb0, _) in zip(runs, runs[1:], strict=False):
        end = {"end_x": float(ball[fb0, 0]), "end_y": float(ball[fb0, 1])}
        dist = float(np.linalg.norm(ball[fb0, :2] - ball[fa1, :2]))
        shot = None if player_ids[pa] in keepers else _shot(ball, known, fa1, fb0, fps, shot_mps, mouth_margin_m)
        same = team_of.get(player_ids[pa]) == team_of.get(player_ids[pb])
        if shot:
            if shot["outcome"] == "blocked" and player_ids[pb] in keepers and not same:
                shot["outcome"] = "saved"
            add(fa1, EventType.SHOT, pa, outcome=shot["outcome"], **end, attributes={"ball_speed_mps": round(shot["speed"], 1), "detected": "vision"})
            if shot["outcome"] == "goal":
                add(shot["line_frame"], EventType.GOAL, pa, attributes={"detected": "vision"})
        elif same:
            add(fa1, EventType.PASS, pa, outcome="complete", receiver_id=player_ids[pb], **end, attributes={"length_m": round(dist, 1), "detected": "vision"})
        elif dist >= pass_min_m:
            add(fa1, EventType.PASS, pa, outcome="incomplete", **end, attributes={"length_m": round(dist, 1), "detected": "vision"})
        if not same:
            add(fb0, EventType.POSSESSION_CHANGE, pb, opponent_id=player_ids[pa], attributes={"detected": "vision"})
    # a shot from the last control run of the clip (nothing after it)
    if runs:
        pa, _, fa1 = runs[-1]
        shot = None if player_ids[pa] in keepers else _shot(ball, known, fa1, F, fps, shot_mps, mouth_margin_m)
        if shot:
            add(fa1, EventType.SHOT, pa, outcome=shot["outcome"], attributes={"ball_speed_mps": round(shot["speed"], 1), "detected": "vision"})
            if shot["outcome"] == "goal":
                add(shot["line_frame"], EventType.GOAL, pa, attributes={"detected": "vision"})

    raw.sort(key=lambda r: r["f"])
    events, possession = [], 0
    for i, r in enumerate(raw):
        f = r.pop("f")
        t = t0 + f / fps
        if r["type"] == EventType.POSSESSION_CHANGE:
            possession += 1
        events.append(Event(id=f"{id_prefix}{i:05d}", t=round(t, 2), period=1, minute=int(t // 60), second=int(t % 60), possession_id=possession, **r))
    return events
