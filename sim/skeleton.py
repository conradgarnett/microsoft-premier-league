"""3D skeletons from tracking + body actions, driven by real motion capture.

Locomotion, kicks, the shuffle and the keeper dive come from CMU Graphics Lab mocap
clips (``sim/mocap/library.npz``, rebuilt by ``python -m sim.mocap.build_library``):

  * Locomotion blends phase-aligned walk/jog/run cycles by ground speed. The cycle phase
    advances at ``speed / stride(speed)``, so foot cadence matches the tracked speed and
    stance feet do not slide. Above the fastest captured run (4.45 m/s) the pose is
    extrapolated along the jog->run trend and stride grows linearly.
  * Players moving across their facing direction (e.g. defenders tracking the ball)
    blend in a side-shuffle cycle; bodies face the velocity, or the ball when slow.
  * Kicks, dives and slide tackles are one-shot clips cross-faded in over ~0.25 s and
    time-aligned to the engine's action timestamps (ball contact).
  * Turning at speed leans the body into the turn; heads make quick shoulder checks.
  * Blended poses are re-projected onto fixed bone lengths, so bones stay rigid.

Output is pitch-space metres in the source-agnostic ``SkeletonWindow`` schema.
"""

from __future__ import annotations

import json
import pathlib
from functools import lru_cache

import numpy as np

from .schema import BONES, JOINT_NAMES, J, SkeletonWindow, TrackingWindow

LIB_PATH = pathlib.Path(__file__).with_name("mocap") / "library.npz"
LIB_FPS = 60.0
G = 9.81

# parent -> child order for bone-length projection (root first)
CHAIN = [
    ("pelvis", "l_hip"), ("pelvis", "r_hip"), ("l_hip", "l_knee"), ("r_hip", "r_knee"),
    ("l_knee", "l_ankle"), ("r_knee", "r_ankle"), ("l_ankle", "l_toe"), ("r_ankle", "r_toe"),
    ("pelvis", "neck"), ("neck", "head"), ("neck", "l_shoulder"), ("neck", "r_shoulder"),
    ("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow"), ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist"),
]  # fmt: skip
MIRROR = [JOINT_NAMES.index(n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")) for n in JOINT_NAMES]

# Slide-tackle key pose (no tackle exists in the CMU set): local frame, facing +x, z up.
TACKLE_POSE = np.array([
    [-0.33, 0.0, 0.92], [-0.30, 0.0, 0.75], [-0.30, 0.19, 0.71], [-0.30, -0.19, 0.71],
    [-0.45, 0.35, 0.45], [-0.20, -0.45, 0.65], [-0.55, 0.40, 0.20], [0.0, -0.60, 0.60],
    [0.0, 0.0, 0.32], [0.0, 0.10, 0.30], [0.0, -0.10, 0.34], [0.25, 0.18, 0.08],
    [0.42, -0.12, 0.22], [-0.15, 0.20, 0.06], [0.85, -0.12, 0.10], [-0.25, 0.20, 0.05], [1.0, -0.12, 0.14],
])  # fmt: skip


@lru_cache(maxsize=1)
def library() -> dict:
    d = np.load(LIB_PATH)
    lib = {k: d[k] for k in d.files if k != "meta"}
    lib["meta"] = json.loads(str(d["meta"]))
    # ordered gentlest -> most powerful (the builder sorts by peak foot speed)
    lib["kicks"] = [(lib[f"kick{i}_pose"], float(lib[f"kick{i}_contact"])) for i in range(len(lib["meta"]["kicks"]))]
    idle = lib["idle_pose"]
    lib["bone_len"] = {(a, b): float(np.linalg.norm(idle[:, J[b]] - idle[:, J[a]], axis=-1).mean()) for a, b in CHAIN}
    return lib


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = x.copy()
    for f in range(1, len(x)):
        out[f] = alpha * x[f] + (1 - alpha) * out[f - 1]
    return out


def _envelope(n: int, f0: float, f1: float, ramp: float) -> np.ndarray:
    """Weight over frames [f0, f1): smooth 0->1 ramp, hold, 1->0 ramp."""
    idx = np.arange(n)
    w = np.clip(np.minimum(idx - f0, f1 - idx) / max(ramp, 1e-6), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * w)


def _sample_cycle(cycle: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """cycle (S,17,3), phase (F,) in cycles -> (F,17,3) with linear interpolation and wrap."""
    S = len(cycle)
    x = (phase % 1.0) * S
    i0 = np.floor(x).astype(int) % S
    a = (x - np.floor(x))[:, None, None]
    return cycle[i0] * (1 - a) + cycle[(i0 + 1) % S] * a


def _sample_clip(clip: np.ndarray, frame: np.ndarray) -> np.ndarray:
    f = np.clip(frame, 0, len(clip) - 1)
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, len(clip) - 1)
    a = (f - i0)[:, None, None]
    return clip[i0] * (1 - a) + clip[i1] * a


def _limit_turn(target: np.ndarray, max_step: float) -> np.ndarray:
    """Unwrapped yaw (F,P) that follows ``target`` but turns at most ``max_step`` rad per frame."""
    out = np.empty_like(target)
    out[0] = target[0]
    for f in range(1, len(target)):
        d = (target[f] - out[f - 1] + np.pi) % (2 * np.pi) - np.pi
        out[f] = out[f - 1] + np.clip(d, -max_step, max_step)
    return out


def _rot_z(p: np.ndarray, ang: np.ndarray) -> np.ndarray:
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    out = p.copy()
    out[..., 0] = c * p[..., 0] - s * p[..., 1]
    out[..., 1] = s * p[..., 0] + c * p[..., 1]
    return out


def _roll(p: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Rotate local poses about the forward (x) axis through the ground point under the pelvis."""
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    out = p.copy()
    out[..., 1] = c * p[..., 1] - s * p[..., 2]
    out[..., 2] = s * p[..., 1] + c * p[..., 2]
    return out


def _project_bones(p: np.ndarray, lengths: dict, scale: float) -> np.ndarray:
    for a, b in CHAIN:
        d = p[:, J[b]] - p[:, J[a]]
        n = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
        p[:, J[b]] = p[:, J[a]] + d / n * lengths[(a, b)] * scale
    return p


def locomotion(speed: np.ndarray, lateral: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray:
    """Local-frame locomotion poses (F,17,3) for one player's speed profile."""
    lib = library()
    sp, stride, cyc = lib["loco_speed"], lib["loco_stride"], lib["loco_pose"]
    F = len(speed)
    v = speed
    top = sp[-1]
    stride_v = np.where(v <= top, np.interp(v, sp, stride), stride[-1] + 0.35 * (v - top))
    phase = np.cumsum(v / np.maximum(stride_v, 0.3) * dt) + rng.uniform()
    pose = np.zeros((F, 17, 3))
    # blend between the two neighbouring cycles by speed
    k = np.clip(np.searchsorted(sp, v) - 1, 0, len(sp) - 2)
    a = np.clip((v - sp[k]) / (sp[k + 1] - sp[k]), 0, 1)
    for i in range(len(sp) - 1):
        m = k == i
        if m.any():
            pose[m] = (1 - a[m])[:, None, None] * _sample_cycle(cyc[i], phase[m]) + a[m][:, None, None] * _sample_cycle(cyc[i + 1], phase[m])
    # faster than the fastest capture: extrapolate along the jog->run trend (more knee drive, lean)
    hi = v > top
    if hi.any():
        ext = np.clip((v[hi] - top) / 3.5, 0, 1)[:, None, None] * 0.9
        base = _sample_cycle(cyc[-1], phase[hi])
        pose[hi] = base + ext * (base - _sample_cycle(cyc[-3], phase[hi]))
    # slow: blend to idle (ping-pong over the 1 s idle capture)
    idle = lib["idle_pose"]
    t = np.arange(F) * dt * LIB_FPS + rng.uniform(0, len(idle))
    pp = np.abs((t % (2 * (len(idle) - 1))) - (len(idle) - 1))
    w_idle = np.clip(1 - v / sp[0], 0, 1)[:, None, None]
    pose = (1 - w_idle) * pose + w_idle * _sample_clip(idle, pp)
    # side-shuffle when travelling across the facing direction
    sh_phase = np.cumsum(np.abs(lateral) / lib["shuffle_stride"] * dt)
    sh = _sample_cycle(lib["shuffle_pose"], sh_phase)
    lat_s = _ema(lateral, 0.25)
    w_left = np.clip(0.5 + lat_s / 1.2, 0, 1)[:, None, None]  # cross-fade the mirrored cycle, never snap
    sh = w_left * sh + (1 - w_left) * (sh[:, MIRROR] * np.array([1, -1, 1]))
    w_sh = np.clip((np.abs(lat_s) / (speed + 1e-6) - 0.55) / 0.3, 0, 1) * np.clip(np.abs(lat_s) / 0.6, 0, 1) * (speed < 4.0)
    w_sh = _ema(w_sh, 0.2)[:, None, None]
    return (1 - w_sh) * pose + w_sh * sh


def build_skeleton(win: TrackingWindow, heights: dict[str, float] | None = None, vision: dict[str, int] | None = None) -> SkeletonWindow:
    lib = library()
    fps = win.fps
    dt = 1.0 / fps
    pos = np.asarray(win.players, float)  # (F,P,2)
    ball = np.asarray(win.ball, float)
    F, P, _ = pos.shape
    rng = np.random.default_rng(sum(map(ord, win.id)) * 7919 + int(win.t_start * 10))
    vel = _ema(_ema(np.gradient(pos, dt, axis=0), 0.5), 0.5)
    speed = np.linalg.norm(vel, axis=-1)
    to_ball = ball[:, None, :2] - pos
    tb = to_ball / (np.linalg.norm(to_ball, axis=-1, keepdims=True) + 1e-9)
    vdir = vel / (speed[..., None] + 1e-9)
    cosang = (vdir * tb).sum(-1)
    # face the ball when slow, or when shuffling across it; otherwise face the run
    across = (np.abs(cosang) < 0.55) & (speed < 3.5) & (np.linalg.norm(to_ball, axis=-1) < 25)
    backpedal = (cosang < -0.6) & (speed < 2.2)
    w_ball = np.clip(1 - (speed - 0.8) / 1.2, 0, 1)
    w_ball = np.maximum(w_ball, (across | backpedal).astype(float))
    face = w_ball[..., None] * tb + (1 - w_ball[..., None]) * vdir
    pid_index = {pid: k for k, pid in enumerate(win.player_ids)}

    actions_by_player: dict[int, list[dict]] = {}
    for act in win.actions:
        k = pid_index.get(act["player_id"])
        if k is not None:
            actions_by_player.setdefault(k, []).append(act)
            if act["kind"] == "kick":
                fc = (act["t"] - win.t_start) * fps
                fi, fj = int(np.clip(fc, 0, F - 1)), int(np.clip(fc + 4, 0, F - 1))
                d = ball[fj, :2] - ball[fi, :2]
                if np.linalg.norm(d) > 0.3:
                    w = _envelope(F, fc - 0.8 * fps, fc + 0.4 * fps, 0.3 * fps)
                    face[:, k] = (1 - w[:, None]) * face[:, k] + w[:, None] * d / np.linalg.norm(d)
    face = _ema(face, 0.25)
    yaw = _limit_turn(np.arctan2(face[..., 1], face[..., 0]), 9.0 * dt)  # bodies turn at most ~9 rad/s
    lateral = (vel * np.stack([-np.sin(yaw), np.cos(yaw)], -1)).sum(-1)
    # path curvature (not body yaw) drives the lean, so turning on the spot stays upright
    path_rate = _ema(np.gradient(np.unwrap(np.arctan2(vel[..., 1], vel[..., 0]), axis=0), dt, axis=0) * np.clip((speed - 1.5) / 2, 0, 1), 0.3)

    out = np.zeros((F, P, 17, 3))
    kicks = lib["kicks"]
    for p in range(P):
        pid = win.player_ids[p]
        scale = (heights or {}).get(pid, 1.8) / 1.8
        local = locomotion(speed[:, p], lateral[:, p], dt, rng)
        # lean into turns: centripetal tilt atan(v * omega / g)
        lean = _ema(np.clip(np.arctan(speed[:, p] * path_rate[:, p] / G) * 0.8, -0.35, 0.35), 0.2)
        local = _roll(local, lean)
        root_off = np.zeros((F, 2))
        for act in actions_by_player.get(p, []):
            fc = (act["t"] - win.t_start) * fps
            kind = act["kind"]
            if kind == "kick":
                power = float(act.get("power", 0.8))
                h = sum(map(ord, act.get("event_id", "x")))
                if act.get("shot"):
                    clip, contact = kicks[-1 - h % 2]
                else:
                    clip, contact = kicks[min(int(power * len(kicks)), len(kicks) - 1) - h % 2]
                if act.get("foot") == "left":
                    clip = clip[:, MIRROR] * np.array([1, -1, 1])
                frames = np.arange(F)
                cf = contact + (frames - fc) * LIB_FPS / fps
                pre = min(0.7, contact / LIB_FPS)
                post = min(0.6, (len(clip) - contact) / LIB_FPS)
                w = _envelope(F, fc - pre * fps, fc + post * fps, 0.25 * fps)[:, None, None]
                kp = _sample_clip(clip, cf)
                kp[..., :2] -= kp[:, J["pelvis"]][:, None, :2]  # tracking owns horizontal root motion
                local = (1 - w) * local + w * kp
            elif kind == "tackle":
                w = _envelope(F, fc - 0.15 * fps, fc + 0.8 * fps, 0.22 * fps)[:, None, None]
                local = (1 - w) * local + w * TACKLE_POSE[None]
            elif kind == "dive":
                clip = lib["dive_pose"]
                peak = float(lib["dive_peak"])
                left = act.get("side") == "left"
                if not left:
                    clip = clip[:, MIRROR] * np.array([1, -1, 1])
                frames = np.arange(F)
                cf = peak + (frames - fc) * LIB_FPS / fps * 0.8
                w = _envelope(F, fc - 0.35 * fps, fc + 1.0 * fps, 0.15 * fps)
                dp = _sample_clip(clip, cf)
                reach = float(np.clip(abs(act.get("dy", 1.5)), 0.5, 2.6))
                off = dp[:, J["pelvis"], :2].copy()
                dp[..., :2] -= off[:, None, :]
                root_off += w[:, None] * off * reach / 0.7
                after = np.clip((frames - fc + 0.2 * fps) / (0.3 * fps), 0, 1)
                dp = _roll(dp, (1 if left else -1) * 1.2 * after)
                local = (1 - w[:, None, None]) * local + w[:, None, None] * dp
        # head scans: quick yaw offsets of the head around the neck
        head_off = np.zeros(F)
        p_scan = (vision or {}).get(pid, 65) / 100
        for act in actions_by_player.get(p, []):
            if act["kind"] == "receive":
                fa = (act["t"] - win.t_start) * fps
                for _ in range(2):
                    if rng.random() < p_scan:
                        ts = fa - rng.uniform(0.6, 2.4) * fps
                        head_off += _envelope(F, ts, ts + 0.4 * fps, 0.12 * fps) * rng.choice([-1, 1]) * rng.uniform(0.9, 1.3)
        t = rng.uniform(0, 3)
        while t < F / fps:
            head_off += _envelope(F, t * fps, (t + 0.35) * fps, 0.1 * fps) * rng.choice([-1, 1]) * rng.uniform(0.7, 1.1)
            t += rng.uniform(2.0, 5.0)
        hv = local[:, J["head"]] - local[:, J["neck"]]
        local[:, J["head"]] = local[:, J["neck"]] + _rot_z(hv[:, None, :], head_off)[:, 0]

        local = _project_bones(local * scale, lib["bone_len"], scale)
        world = _rot_z(local, yaw[:, p])
        c, s_ = np.cos(yaw[:, p]), np.sin(yaw[:, p])
        off_w = np.stack([c * root_off[:, 0] - s_ * root_off[:, 1], s_ * root_off[:, 0] + c * root_off[:, 1]], 1)
        world[..., :2] += (pos[:, p] + off_w)[:, None, :]
        low = world[:, [J["l_ankle"], J["r_ankle"], J["l_toe"], J["r_toe"]], 2].min(axis=1)
        world[..., 2] -= np.minimum(low - 0.03, 0)[:, None]  # never through the grass
        out[:, p] = world

    return SkeletonWindow.model_construct(
        id=f"sk-{win.id}",
        window_id=win.id,
        source=win.source,
        t_start=win.t_start,
        fps=fps,
        n_frames=F,
        joint_names=list(JOINT_NAMES),
        bones=list(BONES),
        player_ids=list(win.player_ids),
        frames=np.round(out, 3).tolist(),
        confidence=None,
    )
