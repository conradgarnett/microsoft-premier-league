"""Multi-object tracking in pitch coordinates (constant velocity + Hungarian assignment).

Tracking on the ground plane instead of in pixels lets one track continue across the
seam between the two panorama halves and makes the gate physical (metres).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    misses: int = 0
    history: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> measured position
    boxes: dict[int, tuple[int, np.ndarray]] = field(default_factory=dict)  # frame -> (detection id, xyxy)
    colors: list[np.ndarray] = field(default_factory=list)


class PitchTracker:
    def __init__(self, dt: float, gate_m: float = 4.5, max_misses: int = 37, colour_w: float = 0.08) -> None:
        self.dt = dt
        self.gate = gate_m
        self.max_misses = max_misses
        self.colour_w = colour_w  # metres of cost per unit of torso-colour difference
        self.tracks: list[Track] = []
        self.dead: list[Track] = []
        self._next = 0

    def update(self, frame: int, dets: np.ndarray, meta: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
        """dets (N,2) pitch positions; meta[i] = (detection id, box xyxy, torso colour)."""
        pred = np.array([t.pos + t.vel * self.dt * (t.misses + 1) for t in self.tracks]).reshape(-1, 2)
        matched_d: set[int] = set()
        if len(pred) and len(dets):
            dist = np.linalg.norm(pred[:, None] - dets[None], axis=2)
            gate = self.gate + 0.4 * np.array([t.misses for t in self.tracks])[:, None]
            # far-side detections jitter by metres (one pixel ~0.5 m), so appearance breaks ties
            w = np.array([0.5, 1.0, 1.0])  # Lab with half-weight lightness: separates white/black kits, tolerates shade
            tc = np.array([np.mean(t.colors[-15:], axis=0) * w for t in self.tracks])
            dc = np.array([m[2] * w for m in meta])
            cost = dist + self.colour_w * np.linalg.norm(tc[:, None] - dc[None], axis=2)
            cost = np.where(dist <= gate, cost, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] >= 1e6:
                    continue
                t = self.tracks[r]
                new_vel = (dets[c] - t.pos) / (self.dt * (t.misses + 1))
                t.vel = 0.7 * t.vel + 0.3 * np.clip(new_vel, -11, 11)
                t.pos = dets[c].copy()
                t.hits += 1
                t.misses = 0
                t.history[frame] = dets[c].copy()
                t.boxes[frame] = (meta[c][0], meta[c][1])
                t.colors.append(meta[c][2])
                matched_d.add(c)
        for t in self.tracks:
            if frame not in t.history:
                t.misses += 1
        alive = []
        for t in self.tracks:
            (alive if t.misses <= self.max_misses else self.dead).append(t)
        self.tracks = alive
        for c in range(len(dets)):
            if c not in matched_d:
                t = Track(self._next, dets[c].copy())
                t.history[frame] = dets[c].copy()
                t.boxes[frame] = (meta[c][0], meta[c][1])
                t.colors.append(meta[c][2])
                self.tracks.append(t)
                self._next += 1

    def finished(self, min_hits: int = 25) -> list[Track]:
        return [t for t in self.tracks + self.dead if t.hits >= min_hits]
