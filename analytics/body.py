"""Biomechanics from 3D skeleton keypoints (any source: synthetic or pose model)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sim.schema import J, SkeletonWindow


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle ABC in degrees (180 = straight)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angles(sk: np.ndarray) -> dict[str, np.ndarray]:
    """Angles for skeleton array (..., 17, 3). Hip flexion is trunk-to-thigh (180 = upright)."""
    g = lambda n: sk[..., J[n], :]  # noqa: E731
    up = np.array([0.0, 0.0, 1.0])
    trunk = g("neck") - g("pelvis")
    lean = np.degrees(np.arccos(np.clip((trunk @ up) / (np.linalg.norm(trunk, axis=-1) + 1e-9), -1, 1)))
    out = {"trunk_lean_deg": lean}
    for s in ("l", "r"):
        out[f"{s}_knee_deg"] = angle_at(g(f"{s}_hip"), g(f"{s}_knee"), g(f"{s}_ankle"))
        out[f"{s}_hip_deg"] = angle_at(g("neck"), g(f"{s}_hip"), g(f"{s}_knee"))
    return out


def facing_yaw(sk: np.ndarray) -> np.ndarray:
    """Body yaw (rad) from the hip line: forward = up x (r_hip - l_hip) rotated."""
    lr = sk[..., J["l_hip"], :2] - sk[..., J["r_hip"], :2]
    return np.arctan2(-lr[..., 0], lr[..., 1])  # perpendicular to the left-right axis


def gaze_yaw(sk: np.ndarray) -> np.ndarray:
    d = sk[..., J["head"], :2] - sk[..., J["neck"], :2]
    return np.arctan2(d[..., 1], d[..., 0])


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def strike_analysis(sk: SkeletonWindow, player_id: str, t: float, foot: str = "right") -> dict[str, Any]:
    """Key angles around a kick at time ``t``: backswing, contact, follow-through."""
    arr = np.asarray(sk.frames, float)
    p = sk.player_ids.index(player_id)
    fc = int(np.clip(round((t - sk.t_start) * sk.fps), 0, sk.n_frames - 1))
    k = "r" if foot != "left" else "l"
    s = "l" if k == "r" else "r"
    lo, hi = max(fc - 8, 0), min(fc + 8, sk.n_frames)
    ang = joint_angles(arr[lo:hi, p])
    rel = fc - lo
    back = int(np.argmin(ang[f"{k}_knee_deg"][: rel + 1]))
    return {
        "player_id": player_id,
        "t": t,
        "frame": fc,
        "kicking_foot": foot,
        "backswing_knee_deg": round(float(ang[f"{k}_knee_deg"][back]), 1),
        "contact_knee_deg": round(float(ang[f"{k}_knee_deg"][rel]), 1),
        "contact_hip_deg": round(float(ang[f"{k}_hip_deg"][rel]), 1),
        "follow_through_hip_deg": round(float(ang[f"{k}_hip_deg"][rel:].min()), 1),
        "support_knee_deg": round(float(ang[f"{s}_knee_deg"][rel]), 1),
        "trunk_lean_deg": round(float(ang["trunk_lean_deg"][rel]), 1),
        "body_yaw_deg": round(math.degrees(float(facing_yaw(arr[fc, p]))), 1),
    }


def scanning(sk: SkeletonWindow, player_id: str, t_receive: float, lookback_s: float = 3.0, thresh_deg: float = 40) -> dict[str, Any]:
    """Count shoulder checks (head yaw away from body yaw > thresh) before receiving."""
    arr = np.asarray(sk.frames, float)
    p = sk.player_ids.index(player_id)
    f1 = int(np.clip(round((t_receive - sk.t_start) * sk.fps), 0, sk.n_frames - 1))
    f0 = max(0, f1 - int(lookback_s * sk.fps))
    seg = arr[f0 : f1 + 1, p]
    off = np.degrees(np.abs(wrap(gaze_yaw(seg) - facing_yaw(seg))))
    turned = off > thresh_deg
    scans = int(np.sum(turned[1:] & ~turned[:-1]) + (turned[0] if len(turned) else 0))
    return {"player_id": player_id, "t": t_receive, "scans": scans, "window_s": round((f1 - f0) / sk.fps, 2), "max_head_turn_deg": round(float(off.max()), 1) if len(off) else 0.0}


def body_orientation(sk: SkeletonWindow, player_id: str, t: float, goal_xy: tuple[float, float]) -> float:
    """Degrees between body facing and direction to the attacked goal (0 = square to goal)."""
    arr = np.asarray(sk.frames, float)
    p = sk.player_ids.index(player_id)
    f = int(np.clip(round((t - sk.t_start) * sk.fps), 0, sk.n_frames - 1))
    yaw = float(facing_yaw(arr[f, p]))
    pel = arr[f, p, J["pelvis"], :2]
    to_goal = math.atan2(goal_xy[1] - pel[1], goal_xy[0] - pel[0])
    return round(abs(math.degrees(float(wrap(np.array(yaw - to_goal))))), 1)
