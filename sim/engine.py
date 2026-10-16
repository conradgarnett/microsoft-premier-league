"""Synthetic football match engine.

A continuous, seeded, agent-based simulation running at 25 fps. Players hold a
formation that shifts with the ball, the defending side presses the ball carrier,
and the carrier picks between passing, dribbling, shooting and clearing using a
utility model. Discrete events (passes, shots, tackles, pressure, ...) emerge from the
continuous state, so tracking, skeletons and events are always mutually consistent.

The engine intentionally favours plausibility and determinism over realism: same
seed + same teams = byte-identical output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import (
    GOAL_HEIGHT,
    GOAL_Y_HIGH,
    GOAL_Y_LOW,
    PITCH_LENGTH,
    PITCH_WIDTH,
    Event,
    EventType,
    LowResTracking,
    Match,
    MatchBundle,
    MatchMeta,
    Position,
    Score,
    Team,
    TrackingWindow,
    WindowRef,
)
from .teams import FORMATIONS, default_teams

L, W = PITCH_LENGTH, PITCH_WIDTH
DEFENDERS = {Position.RB, Position.CB, Position.LB}
MIDS = {Position.DM, Position.CM, Position.RM, Position.LM}
FORWARDS = {Position.RW, Position.LW, Position.ST}


@dataclass
class Flight:
    kind: str  # pass | shot | clearance
    start: np.ndarray
    end: np.ndarray
    t0: float
    T: float
    height: float
    passer: int
    outcome: str
    target: int = -1  # receiver / interceptor / keeper
    event_id: str = ""
    end_z: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Restart:
    kind: str  # kickoff | free_kick | goal_kick
    team: int
    spot: np.ndarray
    taker: int
    at: float


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance of points p (N,2) to segment ab and projection parameter s in [0,1]."""
    ab = b - a
    denom = float(ab @ ab) or 1e-9
    s = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + s[:, None] * ab
    return np.linalg.norm(p - proj, axis=1), s


def shot_quality(x: float, y: float) -> float:
    """Engine-internal chance quality heuristic (attacking towards x=105)."""
    dx = L - x
    dy = abs(y - W / 2)
    dist = math.hypot(dx, dy)
    a1 = math.atan2(GOAL_Y_HIGH - y, dx)
    a2 = math.atan2(GOAL_Y_LOW - y, dx)
    angle = abs(a1 - a2)
    z = -1.2 - 0.11 * dist + 1.6 * angle
    return 1 / (1 + math.exp(-z))


class MatchSimulator:
    def __init__(
        self,
        home: Team | None = None,
        away: Team | None = None,
        seed: int = 7,
        duration_min: float = 90.0,
        fps: int = 25,
        window_pre_s: float = 12.0,
        window_post_s: float = 4.0,
    ) -> None:
        if home is None or away is None:
            home, away = default_teams()
        self.home, self.away = home, away
        self.teams = (home, away)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fps = fps
        self.dt = 1.0 / fps
        self.duration = duration_min * 60.0
        self.half = self.duration / 2
        self.window_pre_s = window_pre_s
        self.window_post_s = window_post_s

        self.players = [p for t in self.teams for p in t.players]
        self.n = len(self.players)
        assert self.n == 22, "engine expects 11 v 11"
        self.team_of = np.array([0] * 11 + [1] * 11)
        self.pos_role = [p.position for p in self.players]
        a = [p.attributes for p in self.players]
        self.pace = np.array([x.pace for x in a], float)
        self.passing = np.array([x.passing for x in a], float)
        self.shooting = np.array([x.shooting for x in a], float)
        self.dribbling = np.array([x.dribbling for x in a], float)
        self.tackling = np.array([x.tackling for x in a], float)
        self.vision = np.array([x.vision for x in a], float)
        self.gk_skill = np.array([x.goalkeeping for x in a], float)
        self.max_speed = 6.2 + 3.0 * (self.pace / 100.0)  # m/s, ~8.6-9.1 for quick players
        self.accel = 3.6 + 1.6 * (self.pace / 100.0)
        self.base = np.array(
            [[bx, by] for t in self.teams for (_, bx, by) in FORMATIONS[t.formation]], float
        )
        self.gk_idx = (0, 11)

        # dynamic state
        self.pos = np.zeros((22, 2))
        self.vel = np.zeros((22, 2))
        self.ball = np.array([L / 2, W / 2, 0.0])
        self.ball_vel = np.zeros(3)
        self.state = "dead"
        self.holder = -1
        self.flight: Flight | None = None
        self.restart: Restart | None = None
        self.team_in_poss = 0
        self.possession_id = 0
        self.poss_start_t = 0.0
        self.poss_passes = 0
        self.period = 1
        self.score = [0, 0]
        self.next_decision = 0.0
        self.receive_t = 0.0
        self.receive_pos = np.zeros(2)
        self.stunned_until = np.zeros(22)
        self.tackle_check = np.zeros(22)
        self.pressure_logged: set[int] = set()
        self.jitter = self.rng.normal(0, 1.5, size=(22, 2))
        self.targets = np.zeros((22, 2))
        self.intensity = np.full(22, 0.5)
        self.last_touch_team = 0
        self.dribble_dir = np.array([1.0, 0.0])

        self.events: list[Event] = []
        self.actions: list[dict[str, Any]] = []
        n_steps = int(round(self.duration * fps))
        self.n_steps = n_steps
        self.hist_pos = np.zeros((n_steps, 22, 2), np.float32)
        self.hist_ball = np.zeros((n_steps, 3), np.float32)
        self.t = 0.0

    # ------------------------------------------------------------------ frames
    def direction(self, team: int) -> int:
        d = 1 if team == 0 else -1
        return d if self.period == 1 else -d

    def to_team(self, xy: np.ndarray, team: int) -> np.ndarray:
        if self.direction(team) == 1:
            return xy.copy()
        return np.array([L, W]) - xy

    def to_world(self, xy: np.ndarray, team: int) -> np.ndarray:
        return self.to_team(xy, team)  # the transform is an involution

    def team_slice(self, team: int) -> slice:
        return slice(0, 11) if team == 0 else slice(11, 22)

    # ------------------------------------------------------------------ events
    def emit(self, etype: EventType, player: int | None = None, **kw: Any) -> Event:
        t = self.t
        team_id = None
        if player is not None and player >= 0:
            team_id = self.teams[self.team_of[player]].id
        elif "team" in kw:
            team_id = self.teams[kw.pop("team")].id
        kw.pop("team", None)
        minute = int(t // 60)
        ev = Event(
            id=f"ev{len(self.events):05d}",
            t=round(t, 2),
            period=self.period,
            minute=minute,
            second=int(t % 60),
            type=etype,
            team_id=team_id,
            player_id=self.players[player].id if player is not None and player >= 0 else None,
            possession_id=self.possession_id,
            **kw,
        )
        for k in ("x", "y", "end_x", "end_y"):
            v = getattr(ev, k)
            if v is not None:
                setattr(ev, k, round(float(v), 2))
        self.events.append(ev)
        return ev

    def nearest_opp_dist(self, i: int) -> float:
        opp = self.team_slice(1 - self.team_of[i])
        return float(np.min(np.linalg.norm(self.pos[opp] - self.pos[i], axis=1)))

    def set_possession(self, team: int, cause_event: Event | None = None) -> None:
        if team != self.team_in_poss:
            prev = self.team_in_poss
            dur = self.t - self.poss_start_t
            self.possession_id += 1
            self.team_in_poss = team
            self.emit(
                EventType.POSSESSION_CHANGE,
                team=team,
                x=float(self.ball[0]),
                y=float(self.ball[1]),
                related_ids=[cause_event.id] if cause_event else [],
                attributes={
                    "from_team": self.teams[prev].id,
                    "previous_duration_s": round(dur, 1),
                    "previous_passes": self.poss_passes,
                },
            )
            self.poss_start_t = self.t
            self.poss_passes = 0
            self.pressure_logged = set()

    def give_ball(self, i: int, delay: tuple[float, float] = (0.4, 1.3)) -> None:
        self.state = "held"
        self.holder = i
        self.flight = None
        self.last_touch_team = int(self.team_of[i])
        self.receive_t = self.t
        self.receive_pos = self.pos[i].copy()
        press = self.nearest_opp_dist(i)
        lo, hi = delay
        if press < 3:
            hi = lo + 0.3
        self.next_decision = self.t + float(self.rng.uniform(lo, hi))
        self.dribble_dir = self.to_world(np.array([1.0, 0.0]) + 0, int(self.team_of[i])) - self.to_world(
            np.zeros(2), int(self.team_of[i])
        )
        self.actions.append({"t": round(self.t, 3), "player": i, "kind": "receive"})

    # ------------------------------------------------------------------ setup
    def kickoff_positions(self, kicking_team: int) -> None:
        for team in (0, 1):
            s = self.team_slice(team)
            base = self.base[s].copy()
            base[:, 0] = np.minimum(base[:, 0] * 0.85, 50.0)
            if team == kicking_team:
                st = [k for k in range(11) if self.pos_role[s.start + k] == Position.ST][0]
                base[st] = [52.0, 34.0]
            for k in range(11):
                self.pos[s.start + k] = self.to_world(base[k], team)
        self.vel[:] = 0

    def setup_restart(self, kind: str, team: int, spot: np.ndarray, delay: float, taker: int | None = None) -> None:
        self.state = "dead"
        self.holder = -1
        self.flight = None
        self.ball[:2] = spot
        self.ball[2] = 0
        self.ball_vel[:] = 0
        if taker is None:
            s = self.team_slice(team)
            d = np.linalg.norm(self.pos[s] - spot, axis=1)
            d[0] = 1e9 if kind != "goal_kick" else -1
            taker = s.start + int(np.argmin(d))
        self.restart = Restart(kind, team, spot.copy(), taker, self.t + delay)

    # ------------------------------------------------------------------ targets
    def compute_targets(self) -> None:
        ball_xy = self.ball[:2]
        for team in (0, 1):
            s = self.team_slice(team)
            bt = self.to_team(ball_xy, team)
            in_poss = team == self.team_in_poss and self.state in ("held", "flight")
            base = self.base[s]
            shift = (bt[0] - 52.5) * 0.55 + (7.0 if in_poss else -5.0)
            tx = base[:, 0] + shift
            width = 1.05 if in_poss else 0.72
            ty = 34 + (base[:, 1] - 34) * width + (bt[1] - 34) * 0.32
            # offside-ish line: attackers do not go beyond the second-last defender
            opp = self.team_slice(1 - team)
            opp_t = np.array([self.to_team(p, team) for p in self.pos[opp]])
            line = np.sort(opp_t[:, 0])[-2]
            for k in range(11):
                role = self.pos_role[s.start + k]
                if role == Position.GK:
                    tx[k] = np.clip(3.5 + 0.09 * bt[0], 2.0, 16.0)
                    ty[k] = 34 + (bt[1] - 34) * 0.22
                    continue
                if in_poss:
                    tx[k] = min(tx[k], max(line - 0.5, bt[0]))
                else:
                    # stay goal-side of the ball
                    tx[k] = min(tx[k], bt[0] - 1.5) if role in DEFENDERS else tx[k]
                    if role in DEFENDERS:
                        tx[k] = max(tx[k], 8.0)
            t2 = np.stack([tx, ty], axis=1) + self.jitter[s]
            t2[:, 0] = np.clip(t2[:, 0], 1.0, L - 1.0)
            t2[:, 1] = np.clip(t2[:, 1], 1.0, W - 1.0)
            for k in range(11):
                self.targets[s.start + k] = self.to_world(t2[k], team)
        self.intensity[:] = 0.42
        # marking pull for the defending team: drift towards nearest attacker, goal side
        if self.state in ("held", "flight"):
            dteam = 1 - self.team_in_poss
            ds, as_ = self.team_slice(dteam), self.team_slice(self.team_in_poss)
            for i in range(ds.start + 1, ds.stop):
                d = np.linalg.norm(self.pos[as_] - self.targets[i], axis=1)
                j = as_.start + int(np.argmin(d))
                if d.min() < 12:
                    goal = self.to_world(np.array([0.0, 34.0]), dteam)
                    mark = self.pos[j] + (goal - self.pos[j]) / (np.linalg.norm(goal - self.pos[j]) + 1e-9) * 2.0
                    self.targets[i] = 0.55 * self.targets[i] + 0.45 * mark

    def apply_dynamic_roles(self) -> None:
        """Per-step overrides: carrier, pressers, receivers, loose-ball chasers."""
        if self.state == "held":
            h = self.holder
            team = int(self.team_of[h])
            # dribble target
            if self.pos_role[h] == Position.GK:
                self.targets[h] = self.pos[h]
                self.intensity[h] = 0.3
            else:
                opp = self.team_slice(1 - team)
                rel = self.pos[opp] - self.pos[h]
                dist = np.linalg.norm(rel, axis=1)
                fwd = self.to_world(np.array([1.0, 0]), team) - self.to_world(np.zeros(2), team)
                dirv = fwd.copy()
                close = dist < 6
                if close.any():
                    away = -(rel[close] / (dist[close, None] ** 2 + 0.5)).sum(axis=0)
                    dirv = dirv + 2.5 * away
                # drift away from touchlines
                if self.pos[h, 1] < 6:
                    dirv[1] += 0.6
                if self.pos[h, 1] > W - 6:
                    dirv[1] -= 0.6
                dirv /= np.linalg.norm(dirv) + 1e-9
                self.dribble_dir = 0.85 * self.dribble_dir + 0.15 * dirv
                self.dribble_dir /= np.linalg.norm(self.dribble_dir) + 1e-9
                self.targets[h] = self.pos[h] + self.dribble_dir * 6
                self.intensity[h] = 0.62 + 0.15 * (self.dribbling[h] / 100)
            # pressers
            dteam = 1 - team
            ds = self.team_slice(dteam)
            cand = [i for i in range(ds.start + 1, ds.stop) if self.t >= self.stunned_until[i]]
            d = np.array([np.linalg.norm(self.pos[i] - self.pos[h]) for i in cand])
            order = np.argsort(d)
            bt = self.to_team(self.pos[h], dteam)
            n_press = 1
            press = self.teams[dteam].press_intensity
            if bt[0] > 55 and press > 0.6:
                n_press = 2
            if bt[0] < 30:
                n_press = 2
            goal = self.to_world(np.array([0.0, 34.0]), dteam)
            for k in order[:n_press]:
                i = cand[k]
                gs = (goal - self.pos[h]) / (np.linalg.norm(goal - self.pos[h]) + 1e-9)
                self.targets[i] = self.pos[h] + gs * 0.7 + self.vel[h] * 0.3
                self.intensity[i] = 0.75 + 0.25 * press
        elif self.state == "flight" and self.flight is not None:
            f = self.flight
            if f.target >= 0:
                self.targets[f.target] = f.end[:2]
                self.intensity[f.target] = 1.0
            if f.kind == "shot":
                gk = self.gk_idx[1 - int(self.team_of[f.passer])]
                self.targets[gk] = [self.pos[gk, 0], np.clip(f.end[1], GOAL_Y_LOW - 1, GOAL_Y_HIGH + 1)]
                self.intensity[gk] = 1.0
        elif self.state == "loose":
            for team in (0, 1):
                s = self.team_slice(team)
                d = np.linalg.norm(self.pos[s] - self.ball[:2], axis=1)
                d[0] = 1e9 if np.linalg.norm(self.ball[:2] - self.pos[s.start]) > 8 else d[0]
                for k in np.argsort(d)[:2]:
                    self.targets[s.start + k] = self.ball[:2] + self.ball_vel[:2] * 0.4
                    self.intensity[s.start + k] = 1.0
        elif self.state == "dead" and self.restart is not None:
            r = self.restart
            self.targets[r.taker] = r.spot
            self.intensity[r.taker] = 0.7
            if r.kind == "kickoff":
                for team in (0, 1):
                    s = self.team_slice(team)
                    for i in range(s.start, s.stop):
                        tt = self.to_team(self.targets[i], team)
                        tt[0] = min(tt[0], 49.0)
                        self.targets[i] = self.to_world(tt, team)
                self.targets[r.taker] = r.spot

    # ------------------------------------------------------------------ movement
    def move(self) -> None:
        delta = self.targets - self.pos
        dist = np.linalg.norm(delta, axis=1) + 1e-9
        speed = np.minimum(self.max_speed * self.intensity, dist * 1.4)
        desired = delta / dist[:, None] * speed[:, None]
        dv = desired - self.vel
        dvn = np.linalg.norm(dv, axis=1) + 1e-9
        lim = self.accel * self.dt
        dv = np.where((dvn > lim)[:, None], dv / dvn[:, None] * lim[:, None], dv)
        self.vel += dv
        stunned = self.t < self.stunned_until
        self.vel[stunned] *= 0.9
        self.pos += self.vel * self.dt
        # light separation so bodies don't overlap
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        d2 = (diff**2).sum(-1) + np.eye(22) * 1e6
        close = d2 < 0.8**2
        if close.any():
            push = (diff / (d2[..., None] + 1e-6) * close[..., None]).sum(1) * 0.02
            self.pos += push
        self.pos[:, 0] = np.clip(self.pos[:, 0], -1, L + 1)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -1, W + 1)

    # ------------------------------------------------------------------ decisions
    def decide(self) -> None:
        h = self.holder
        team = int(self.team_of[h])
        s = self.team_slice(team)
        opp = self.team_slice(1 - team)
        p = self.pos[h]
        pt = self.to_team(p, team)
        pressure = self.nearest_opp_dist(h)
        opp_pos = self.pos[opp]
        options: list[tuple[float, str, Any]] = []

        # carries: emit when the player travelled with the ball
        carried = float(np.linalg.norm(p - self.receive_pos))
        if carried > 5 and self.pos_role[h] != Position.GK:
            self.emit(
                EventType.CARRY,
                h,
                x=float(self.receive_pos[0]),
                y=float(self.receive_pos[1]),
                end_x=float(p[0]),
                end_y=float(p[1]),
                under_pressure=pressure < 2.5,
                attributes={"distance_m": round(carried, 1), "duration_s": round(self.t - self.receive_t, 2)},
            )
            self.receive_pos = p.copy()
            self.receive_t = self.t

        # shoot
        if pt[0] > 72 and abs(pt[1] - 34) < 22 and self.pos_role[h] != Position.GK:
            q = shot_quality(pt[0], pt[1])
            dist_g = math.hypot(L - pt[0], 34 - pt[1])
            u = -0.9 + 6.0 * q + (1.1 if dist_g < 25 else 0) + (self.shooting[h] - 70) / 40 - (0.5 if pressure < 1.5 else 0)
            options.append((u, "shot", None))

        # passes
        for j in range(s.start, s.stop):
            if j == h:
                continue
            target = self.pos[j] + self.vel[j] * 0.6
            d = float(np.linalg.norm(target - p))
            if d < 4 or d > 55:
                continue
            if self.pos_role[j] == Position.GK and pt[0] > 40:
                continue
            tt = self.to_team(target, team)
            gain = tt[0] - pt[0]
            lane, _ = _seg_dist(opp_pos, p, target)
            lane_min = float(lane.min())
            space = float(np.min(np.linalg.norm(opp_pos - target, axis=1)))
            u = (
                0.035 * gain
                + 0.22 * min(space, 8)
                + 0.35 * min(lane_min, 4)
                - 0.018 * d
                + (0.4 if tt[0] > 80 and pt[0] < 80 else 0)
                - (0.8 if self.pos_role[j] == Position.GK else 0)
            )
            options.append((u, "pass", (j, target, d, lane_min, space)))
            # through ball into space behind for forwards
            if self.pos_role[j] in FORWARDS and tt[0] > 62 and self.vision[h] > 60:
                tb_t = tt + np.array([float(self.rng.uniform(5, 11)), 0.0])
                tb_t[0] = min(tb_t[0], 90)
                tb_t[1] = float(np.clip(tb_t[1] + (34 - tb_t[1]) * 0.25, 3, W - 3))
                tb = self.to_world(tb_t, team)
                dtb = float(np.linalg.norm(tb - p))
                lane2, _ = _seg_dist(opp_pos, p, tb)
                space2 = float(np.min(np.linalg.norm(opp_pos - tb, axis=1)))
                if dtb < 45:
                    u2 = 0.035 * (tb_t[0] - pt[0]) + 0.2 * min(space2, 8) + 0.35 * min(float(lane2.min()), 4) - 0.02 * dtb + 0.3
                    options.append((u2, "through", (j, tb, dtb, float(lane2.min()), space2)))

        # dribble
        fwd_world = self.to_world(pt + np.array([5.0, 0]), team)
        front = float(np.min(np.linalg.norm(opp_pos - fwd_world, axis=1)))
        if self.pos_role[h] != Position.GK:
            options.append((0.25 * min(front, 8) + 0.3 * (self.dribbling[h] - 70) / 20 - 0.3, "dribble", None))
        # clearance
        if pt[0] < 30 and pressure < 2.5:
            options.append((2.6 - pressure, "clear", None))

        us = np.array([o[0] for o in options])
        probs = np.exp((us - us.max()) / 0.55)
        probs /= probs.sum()
        choice = options[int(self.rng.choice(len(options), p=probs))]
        kind = choice[1]
        if kind == "shot":
            self.do_shot(h)
        elif kind in ("pass", "through"):
            j, target, d, lane_min, space = choice[2]
            self.do_pass(h, j, target, d, lane_min, space, pressure, through=kind == "through")
        elif kind == "clear":
            self.do_clear(h)
        else:
            # near goal, decide quickly: real attackers rarely dribble into the six-yard box
            self.next_decision = self.t + float(self.rng.uniform(0.15, 0.4) if pt[0] > 78 else self.rng.uniform(0.5, 1.2))

    def foot_of(self, i: int) -> str:
        return self.players[i].foot

    def do_pass(
        self, h: int, j: int, target: np.ndarray, d: float, lane: float, space: float, pressure: float, through: bool
    ) -> None:
        team = int(self.team_of[h])
        target = np.clip(target, [0.5, 0.5], [L - 0.5, W - 0.5])
        p_succ = (
            0.985
            - 0.0055 * d
            - (0.16 if lane < 1.5 else 0.05 if lane < 3 else 0)
            - (0.12 if space < 2 else 0)
            - (0.10 if pressure < 1.5 else 0)
            - (0.07 if through else 0)
            - (0.18 if self.to_team(target, team)[0] > 88 else 0)  # crowded box
            + 0.12 * (self.passing[h] - 70) / 30
        )
        p_succ = float(np.clip(p_succ, 0.25, 0.97))
        success = self.rng.random() < p_succ
        lofted = d > 28 or (lane < 1.2 and d > 15)
        speed = float(np.clip(7.5 + 0.38 * d, 9.0, 23.0)) * (0.85 if lofted else 1.0)
        start = self.ball.copy()
        start[:2] = self.pos[h] + self.dribble_dir * 0.4
        receiver, interceptor, outcome = j, -1, "complete"
        end = target.copy()
        if not success:
            outcome = "incomplete"
            opp = self.team_slice(1 - team)
            lane_d, sproj = _seg_dist(self.pos[opp], start[:2], target)
            k = int(np.argmin(lane_d))
            if lane_d[k] < 5 and not lofted:
                interceptor = opp.start + k
                end = start[:2] + (target - start[:2]) * max(float(sproj[k]), 0.25)
            else:
                ang = float(self.rng.uniform(0, 2 * math.pi))
                end = target + np.array([math.cos(ang), math.sin(ang)]) * float(self.rng.uniform(3, 7))
                end = np.clip(end, [1, 1], [L - 1, W - 1])
                # nearest opponent to the landing point will take it
                interceptor = opp.start + int(np.argmin(np.linalg.norm(self.pos[opp] - end, axis=1)))
            receiver = interceptor
        dist = float(np.linalg.norm(end - start[:2]))
        T = max(dist / speed, 0.35)
        ev = self.emit(
            EventType.PASS,
            h,
            x=float(start[0]),
            y=float(start[1]),
            end_x=float(end[0]),
            end_y=float(end[1]),
            outcome=outcome,
            receiver_id=self.players[j].id if success else None,
            under_pressure=pressure < 2.5,
            attributes={
                "intended_receiver_id": self.players[j].id,
                "length_m": round(dist, 1),
                "height": "lofted" if lofted else "ground",
                "through_ball": through,
                "ball_speed_mps": round(speed, 1),
                "passer_pressure_m": round(pressure, 2),
                "lane_clearance_m": round(lane, 2),
                "receiver_space_m": round(space, 2),
                "body_part": f"{self.foot_of(h)}_foot",
            },
        )
        self.flight = Flight(
            "pass", start, np.array([end[0], end[1], 0.0]), self.t, T, (min(0.15 * dist, 9) if lofted else 0.0), h, outcome, receiver, ev.id
        )
        self.actions.append({"t": round(self.t, 3), "player": h, "kind": "kick", "power": round(min(speed / 25, 1), 2), "foot": self.foot_of(h), "event_id": ev.id})
        self.state = "flight"
        self.holder = -1
        if success:
            self.poss_passes += 1

    def do_clear(self, h: int) -> None:
        team = int(self.team_of[h])
        tt = self.to_team(self.pos[h], team)
        land_t = np.array([tt[0] + float(self.rng.uniform(35, 55)), float(np.clip(tt[1] + self.rng.normal(0, 14), 3, W - 3))])
        land_t[0] = min(land_t[0], 98)
        land = self.to_world(land_t, team)
        start = self.ball.copy()
        dist = float(np.linalg.norm(land - start[:2]))
        speed = 22.0
        ev = self.emit(
            EventType.CLEARANCE,
            h,
            x=float(start[0]),
            y=float(start[1]),
            end_x=float(land[0]),
            end_y=float(land[1]),
            under_pressure=True,
            attributes={"length_m": round(dist, 1), "ball_speed_mps": speed},
        )
        self.flight = Flight("clearance", start, np.array([land[0], land[1], 0.0]), self.t, dist / speed * 1.4, min(0.25 * dist, 18), h, "loose", -1, ev.id)
        self.actions.append({"t": round(self.t, 3), "player": h, "kind": "kick", "power": 1.0, "foot": self.foot_of(h), "event_id": ev.id})
        self.state = "flight"
        self.holder = -1

    def do_shot(self, h: int) -> None:
        team = int(self.team_of[h])
        opp_team = 1 - team
        p = self.pos[h]
        pt = self.to_team(p, team)
        dist = math.hypot(L - pt[0], 34 - pt[1])
        q = shot_quality(pt[0], pt[1])
        pressure = self.nearest_opp_dist(h)
        on_target_p = float(np.clip(0.62 + (self.shooting[h] - 70) / 90 - dist / 90 - (0.1 if pressure < 1.5 else 0), 0.2, 0.85))
        on_target = self.rng.random() < on_target_p
        if on_target:
            yt = float(self.rng.choice([self.rng.uniform(GOAL_Y_LOW + 0.3, 33.2), self.rng.uniform(34.8, GOAL_Y_HIGH - 0.3), self.rng.uniform(33, 35)]))
            zt = float(self.rng.uniform(0.15, GOAL_HEIGHT - 0.2))
        else:
            side = self.rng.choice([-1, 1])
            yt = float(34 + side * self.rng.uniform(3.9, 7.5))
            zt = float(self.rng.uniform(0.2, 3.8))
            if self.rng.random() < 0.35:
                yt = float(self.rng.uniform(31, 37))
                zt = float(self.rng.uniform(2.6, 4.2))
        speed = float(np.clip(15 + 0.16 * self.shooting[h] - 0.08 * dist + self.rng.normal(0, 2.5), 14, 34))
        goal_world = self.to_world(np.array([L + 0.3, yt]), team)
        start = np.array([p[0], p[1], 0.11]) + np.append(self.dribble_dir * 0.4, 0)
        # block check
        opp = self.team_slice(opp_team)
        lane_d, sproj = _seg_dist(self.pos[opp], start[:2], goal_world)
        blockers = [(opp.start + k, sproj[k]) for k in range(11) if lane_d[k] < 1.1 and sproj[k] < 0.5 and k != 0]
        outcome = "goal"
        gk = self.gk_idx[opp_team]
        target = -1
        end = np.array([goal_world[0], goal_world[1], zt])
        if blockers and self.rng.random() < 0.55:
            b, sp = blockers[0]
            outcome = "blocked"
            target = b
            end = np.append(start[:2] + (goal_world - start[:2]) * sp, 0.6)
        elif not on_target:
            outcome = "off_target"
        else:
            p_goal = float(np.clip(q / on_target_p * (0.9 - (self.gk_skill[gk] - 70) / 100), 0.04, 0.92))
            if self.rng.random() >= p_goal:
                outcome = "saved"
                target = gk
        dist3 = float(np.linalg.norm(end - start))
        T = max(dist3 / speed, 0.25)
        ev = self.emit(
            EventType.SHOT,
            h,
            x=float(start[0]),
            y=float(start[1]),
            end_x=float(end[0]),
            end_y=float(end[1]),
            outcome=outcome,
            under_pressure=pressure < 2.0,
            opponent_id=self.players[gk].id,
            attributes={
                "end_z": round(float(end[2]), 2),
                "distance_m": round(dist, 1),
                "ball_speed_mps": round(speed, 1),
                "on_target": bool(on_target),
                "body_part": f"{self.foot_of(h)}_foot",
                "shooter_pressure_m": round(pressure, 2),
                "engine_quality": round(q, 3),
            },
        )
        self.flight = Flight("shot", start, end, self.t, T, 0.0, h, outcome, target, ev.id, end_z=float(end[2]))
        self.actions.append({"t": round(self.t, 3), "player": h, "kind": "kick", "power": 1.0, "foot": self.foot_of(h), "event_id": ev.id, "shot": True})
        if outcome in ("saved", "goal"):
            side = "left" if (goal_world[1] - self.pos[gk, 1]) * (1 if self.direction(opp_team) == 1 else -1) > 0 else "right"
            self.actions.append({"t": round(self.t + T * 0.7, 3), "player": gk, "kind": "dive", "side": side, "dy": round(float(goal_world[1] - self.pos[gk, 1]), 2), "event_id": ev.id})
        self.state = "flight"
        self.holder = -1

    # ------------------------------------------------------------------ resolution
    def update_flight(self) -> None:
        f = self.flight
        assert f is not None
        s = (self.t - f.t0) / f.T
        if f.kind == "shot":
            frac = min(s, 1.0)
            self.ball = f.start + (f.end - f.start) * frac
        else:
            frac = min(1 - (1 - min(s, 1.0)) ** 1.6, 1.0) if f.height == 0 else min(s, 1.0)
            self.ball[:2] = f.start[:2] + (f.end[:2] - f.start[:2]) * frac
            self.ball[2] = 4 * f.height * frac * (1 - frac)
        if s < 1.0:
            return
        if f.kind == "pass":
            r = f.target
            if f.outcome == "complete":
                self.give_ball(r)
            else:
                team = int(self.team_of[r])
                if np.linalg.norm(self.pos[r] - self.ball[:2]) < 2.5:
                    ev = self.emit(EventType.INTERCEPTION, r, x=float(self.ball[0]), y=float(self.ball[1]), related_ids=[f.event_id])
                    self.set_possession(team, ev)
                    self.give_ball(r)
                else:
                    self.state = "loose"
                    self.ball_vel = np.append((f.end[:2] - f.start[:2]) / (np.linalg.norm(f.end[:2] - f.start[:2]) + 1e-9) * 3.0, 0)
            self.flight = None
        elif f.kind == "clearance":
            self.state = "loose"
            self.ball_vel = np.append((f.end[:2] - f.start[:2]) / (np.linalg.norm(f.end[:2] - f.start[:2]) + 1e-9) * 4.0, 0)
            self.flight = None
        elif f.kind == "shot":
            shooter_team = int(self.team_of[f.passer])
            if f.outcome == "goal":
                self.score[shooter_team] += 1
                ev = self.emit(
                    EventType.GOAL,
                    f.passer,
                    x=float(f.end[0]),
                    y=float(f.end[1]),
                    related_ids=[f.event_id],
                    attributes={"score_home": self.score[0], "score_away": self.score[1]},
                )
                self.ball[0] = L + 1.2 if f.end[0] > L / 2 else -1.2
                self.ball[2] = min(self.ball[2], 1.5)
                self.setup_restart("kickoff", 1 - shooter_team, np.array([L / 2, W / 2]), 22.0)
                self.pending_kickoff_reason = ev.id
            elif f.outcome == "saved":
                gk = f.target
                ev = self.emit(EventType.SAVE, gk, x=float(f.end[0]), y=float(f.end[1]), related_ids=[f.event_id], attributes={"end_z": round(f.end_z, 2)})
                if self.rng.random() < 0.3:
                    # parried: rebound chance
                    self.state = "loose"
                    back = self.to_world(np.array([1.0, 0]), 1 - shooter_team) - self.to_world(np.zeros(2), 1 - shooter_team)
                    self.ball[2] = 0.3
                    self.ball_vel = np.array([back[0] * 6, float(self.rng.normal(0, 5)), 0])
                else:
                    self.set_possession(1 - shooter_team, ev)
                    self.give_ball(gk, delay=(2.0, 3.5))
                self.flight = None
            elif f.outcome == "blocked":
                b = f.target
                self.state = "loose"
                self.ball_vel = np.array([float(self.rng.normal(0, 6)), float(self.rng.normal(0, 6)), 0])
                self.actions.append({"t": round(self.t, 3), "player": b, "kind": "block", "event_id": f.event_id})
                self.flight = None
            else:  # off target
                defending = 1 - shooter_team
                spot = self.to_world(np.array([5.5, 34 + float(self.rng.choice([-9, 9]))]), defending)
                self.setup_restart("goal_kick", defending, spot, 7.0, taker=self.gk_idx[defending])
                self.pending_restart_cause = f.event_id
                self.set_possession(defending)

    def update_loose(self) -> None:
        self.ball[:2] += self.ball_vel[:2] * self.dt
        self.ball_vel *= 0.975
        self.ball[2] = max(0.0, self.ball[2] - 2 * self.dt)
        # out of play -> simplified restart for the other team
        if not (0 <= self.ball[0] <= L and 0 <= self.ball[1] <= W):
            team = 1 - self.last_touch_team
            spot = np.clip(self.ball[:2], [1, 0.5], [L - 1, W - 0.5])
            kind = "free_kick"
            if self.ball[0] < 0 or self.ball[0] > L:
                if self.direction(team) * (1 if self.ball[0] > L else -1) < 0:
                    kind = "goal_kick"
                    spot = self.to_world(np.array([5.5, 30.0]), team)
                    self.setup_restart(kind, team, spot, 6.0, taker=self.gk_idx[team])
                    self.set_possession(team)
                    return
                spot = np.array([np.clip(self.ball[0], 1, L - 1), self.ball[1]])
            self.setup_restart(kind, team, spot, 4.0)
            self.set_possession(team)
            return
        d = np.linalg.norm(self.pos - self.ball[:2], axis=1)
        i = int(np.argmin(d))
        if d[i] < 1.0 and self.ball[2] < 1.2:
            team = int(self.team_of[i])
            ev = self.emit(EventType.BALL_RECOVERY, i, x=float(self.ball[0]), y=float(self.ball[1]))
            self.set_possession(team, ev)
            self.give_ball(i, delay=(0.3, 0.9))

    def update_held(self) -> None:
        h = self.holder
        sp = np.linalg.norm(self.vel[h])
        ahead = self.vel[h] / sp * 0.55 if sp > 0.3 else self.dribble_dir * 0.4
        self.ball[:2] = self.pos[h] + ahead
        self.ball[2] = 0.11
        team = int(self.team_of[h])
        dteam = 1 - team
        ds = self.team_slice(dteam)
        d = np.linalg.norm(self.pos[ds] - self.pos[h], axis=1)
        # pressure events
        for k in np.where(d < 3.0)[0]:
            i = ds.start + int(k)
            if i not in self.pressure_logged and self.pos_role[i] != Position.GK:
                self.pressure_logged.add(i)
                closing = float(-np.dot(self.vel[i] - self.vel[h], (self.pos[i] - self.pos[h]) / (d[k] + 1e-9)))
                self.emit(
                    EventType.PRESSURE,
                    i,
                    x=float(self.pos[i, 0]),
                    y=float(self.pos[i, 1]),
                    opponent_id=self.players[h].id,
                    attributes={"distance_m": round(float(d[k]), 2), "closing_speed_mps": round(closing, 2)},
                )
        # tackles
        for k in np.where(d < 1.4)[0]:
            i = ds.start + int(k)
            if self.pos_role[i] == Position.GK or self.t < self.stunned_until[i] or self.t < self.tackle_check[i]:
                continue
            self.tackle_check[i] = self.t + 0.45
            if self.rng.random() > 0.42:
                continue
            self.actions.append({"t": round(self.t, 3), "player": i, "kind": "tackle", "event_id": f"ev{len(self.events):05d}"})
            p_win = float(np.clip(0.5 + (self.tackling[i] - self.dribbling[h]) / 120, 0.2, 0.78))
            r = self.rng.random()
            if r < 0.11:
                ev = self.emit(EventType.FOUL, i, x=float(self.pos[h, 0]), y=float(self.pos[h, 1]), opponent_id=self.players[h].id)
                self.stunned_until[i] = self.t + 1.0
                self.setup_restart("free_kick", team, self.pos[h].copy(), 5.0, taker=h)
                self.pending_restart_cause = ev.id
                return
            if r < 0.11 + p_win * 0.89:
                ev = self.emit(EventType.TACKLE, i, x=float(self.pos[h, 0]), y=float(self.pos[h, 1]), outcome="won", opponent_id=self.players[h].id)
                self.stunned_until[h] = self.t + 0.9
                if self.rng.random() < 0.6:
                    self.set_possession(dteam, ev)
                    self.give_ball(i, delay=(0.4, 1.0))
                else:
                    self.state = "loose"
                    self.holder = -1
                    ang = float(self.rng.uniform(0, 2 * math.pi))
                    self.ball_vel = np.array([math.cos(ang) * 5, math.sin(ang) * 5, 0])
                    self.last_touch_team = dteam
                return
            self.emit(EventType.TACKLE, i, x=float(self.pos[h, 0]), y=float(self.pos[h, 1]), outcome="lost", opponent_id=self.players[h].id)
            self.stunned_until[i] = self.t + 1.1
        if self.t >= self.next_decision:
            self.decide()

    def update_dead(self) -> None:
        r = self.restart
        assert r is not None
        if self.t < r.at:
            return
        if np.linalg.norm(self.pos[r.taker] - r.spot) > 1.5 and self.t < r.at + 6:
            return
        self.pos[r.taker] = r.spot - (self.to_world(np.array([1.0, 0]), r.team) - self.to_world(np.zeros(2), r.team)) * 0.5
        etype = {"kickoff": EventType.KICKOFF, "free_kick": EventType.FREE_KICK, "goal_kick": EventType.GOAL_KICK}[r.kind]
        cause = getattr(self, "pending_restart_cause", None) or getattr(self, "pending_kickoff_reason", None)
        self.emit(etype, r.taker, x=float(r.spot[0]), y=float(r.spot[1]), related_ids=[cause] if cause else [])
        self.pending_restart_cause = None
        self.pending_kickoff_reason = None
        self.set_possession(r.team)
        self.restart = None
        self.give_ball(r.taker, delay=(0.05, 0.2) if r.kind == "kickoff" else (0.4, 1.0))

    # ------------------------------------------------------------------ main loop
    def start_half(self, period: int) -> None:
        self.period = period
        kicking = 0 if period == 1 else 1
        self.kickoff_positions(kicking)
        self.team_in_poss = kicking
        self.emit(EventType.HALF_START, team=kicking, attributes={"period": period})
        st = [i for i in range(kicking * 11, kicking * 11 + 11) if self.pos_role[i] == Position.ST][0]
        self.setup_restart("kickoff", kicking, np.array([L / 2, W / 2]), 0.5, taker=st)
        self.poss_start_t = self.t

    def run(self) -> MatchBundle:
        self.start_half(1)
        half_steps = int(round(self.half * self.fps))
        for step in range(self.n_steps):
            self.t = step * self.dt
            if step == half_steps:
                self.emit(EventType.HALF_END, team=0, attributes={"period": 1, "score_home": self.score[0], "score_away": self.score[1]})
                self.start_half(2)
            if step % 5 == 0:
                if step % 125 == 0:
                    self.jitter = 0.7 * self.jitter + 0.3 * self.rng.normal(0, 2.0, size=(22, 2))
                self.compute_targets()
            self.apply_dynamic_roles()
            if self.state == "held":
                self.update_held()
            elif self.state == "flight":
                self.update_flight()
            elif self.state == "loose":
                self.update_loose()
            elif self.state == "dead":
                self.update_dead()
            self.move()
            self.hist_pos[step] = self.pos
            self.hist_ball[step] = self.ball
        self.t = self.duration
        self.emit(EventType.HALF_END, team=0, attributes={"period": 2, "score_home": self.score[0], "score_away": self.score[1]})
        return self.build_bundle()

    # ------------------------------------------------------------------ output
    def build_bundle(self) -> MatchBundle:
        match_id = f"syn-{self.seed}-{int(self.duration // 60)}m"
        meta = MatchMeta(
            match_id=match_id,
            seed=self.seed,
            duration_s=self.duration,
            half_length_s=self.half,
            fps=self.fps,
            home_team_id=self.home.id,
            away_team_id=self.away.id,
            final_score=Score(home=self.score[0], away=self.score[1]),
        )
        player_ids = [p.id for p in self.players]
        # key-moment windows: every shot, merged when overlapping
        spans: list[list[Any]] = []
        for ev in self.events:
            if ev.type == EventType.SHOT:
                a = max(0.0, ev.t - self.window_pre_s)
                b = min(self.duration - self.dt, ev.t + self.window_post_s)
                if spans and a <= spans[-1][1]:
                    spans[-1][1] = max(spans[-1][1], b)
                    spans[-1][2].append(ev.id)
                else:
                    spans.append([a, b, [ev.id]])
        windows: dict[str, TrackingWindow] = {}
        refs: list[WindowRef] = []
        for k, (a, b, trig) in enumerate(spans):
            wid = f"w{k:03d}"
            f0, f1 = int(round(a * self.fps)), int(round(b * self.fps))
            pos = np.round(self.hist_pos[f0:f1], 2)
            ball = np.round(self.hist_ball[f0:f1], 2)
            t0, t1 = f0 / self.fps, f1 / self.fps
            acts = []
            for act in self.actions:
                if t0 - 1 <= act["t"] <= t1 + 1:
                    a2 = dict(act)
                    a2["player_id"] = player_ids[a2.pop("player")]
                    acts.append(a2)
            evids = [e.id for e in self.events if t0 <= e.t <= t1]
            windows[wid] = TrackingWindow(
                id=wid,
                t_start=round(t0, 3),
                fps=self.fps,
                n_frames=int(f1 - f0),
                player_ids=player_ids,
                ball=ball.tolist(),
                players=pos.tolist(),
                actions=acts,
                event_ids=evids,
            )
            refs.append(WindowRef(id=wid, t_start=round(t0, 2), t_end=round(t1, 2), trigger_event_ids=trig))
        step = int(self.fps // 2)
        lowres = LowResTracking(
            fps=self.fps / step,
            n_frames=len(range(0, self.n_steps, step)),
            player_ids=player_ids,
            ball=np.round(self.hist_ball[::step], 2).tolist(),
            players=np.round(self.hist_pos[::step], 1).tolist(),
        )
        match = Match(meta=meta, teams=list(self.teams), events=self.events, windows=refs)
        return MatchBundle(match=match, lowres=lowres, windows=windows)


def simulate(seed: int = 7, duration_min: float = 90.0) -> MatchBundle:
    return MatchSimulator(seed=seed, duration_min=duration_min).run()
