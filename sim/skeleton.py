"""Procedural 3D skeletons from tracking + body actions.

Each player is a 17-joint figure (see ``schema.JOINT_NAMES``) driven by:
  * a run cycle whose stride and amplitude scale with speed,
  * keyframed actions from the engine (kick, tackle, goalkeeper dive, receive),
  * head "scans" (quick shoulder checks), more frequent just before receiving,
  * ground contact (the lowest foot is planted), which gives natural bounce.

Output is pitch-space metres, so analytics can compute joint angles exactly as it
would from a real pose-estimation model.
"""

from __future__ import annotations

import numpy as np

from .schema import BONES, J, SkeletonWindow, TrackingWindow

# Kick keyframes for the striking leg: (time rel. to contact s, hip flexion rad, knee flexion rad)
_KICK_KEYS = np.array(
    [
        [-0.40, 0.10, 0.30],
        [-0.14, -0.70, 1.75],
        [0.00, 0.40, 0.30],
        [0.20, 1.15, 0.30],
        [0.36, 0.25, 0.30],
    ]
)


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = x.copy()
    for f in range(1, len(x)):
        out[f] = alpha * x[f] + (1 - alpha) * out[f - 1]
    return out


def _envelope(n: int, f0: int, f1: int, ramp: int) -> np.ndarray:
    """Weight 0..1 over frames [f0, f1) with smooth ramps."""
    idx = np.arange(n)
    w = np.clip(np.minimum(idx - f0 + 1, f1 - idx) / max(ramp, 1), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * w)


def build_skeleton(win: TrackingWindow, heights: dict[str, float] | None = None, vision: dict[str, int] | None = None) -> SkeletonWindow:
    fps = win.fps
    dt = 1.0 / fps
    pos = np.asarray(win.players, float)  # (F,P,2)
    ball = np.asarray(win.ball, float)
    F, P, _ = pos.shape
    pid_index = {pid: k for k, pid in enumerate(win.player_ids)}
    scale = np.array([(heights or {}).get(pid, 1.8) / 1.8 for pid in win.player_ids])[None, :]
    rng = np.random.default_rng(sum(map(ord, win.id)) * 7919 + int(win.t_start * 10))

    vel = np.gradient(pos, dt, axis=0)
    vel = _ema(vel, 0.35)
    speed = np.linalg.norm(vel, axis=-1)

    # facing: movement direction when running, otherwise towards the ball
    to_ball = ball[:, None, :2] - pos
    face = np.where((speed > 0.8)[..., None], vel, to_ball)
    face = face / (np.linalg.norm(face, axis=-1, keepdims=True) + 1e-9)

    # actions: per frame overrides
    hip = np.zeros((F, P, 2))
    knee = np.zeros((F, P, 2))
    shoulder = np.zeros((F, P, 2))
    elbow = np.zeros((F, P, 2))
    abduct = np.full((F, P, 2), 0.12)
    lean = 0.04 + 0.03 * np.clip(speed, 0, 9)
    roll = np.zeros((F, P))
    lateral = np.zeros((F, P))  # sideways body offset (keeper dives)
    airborne = np.zeros((F, P))
    head_off = np.zeros((F, P))

    # run cycle
    cycle_len = 1.5 + 0.22 * speed
    phase = np.cumsum(speed * dt / cycle_len, axis=0) * 2 * np.pi + rng.uniform(0, 2 * np.pi, P)[None, :]
    m = np.clip(speed / 2.0, 0, 1)
    amp = np.clip(0.12 + 0.075 * speed, 0, 0.85)
    for side, off in ((0, np.pi), (1, 0.0)):
        ph = phase + off
        hip[..., side] = m * amp * np.sin(ph)
        knee[..., side] = 0.1 + m * (0.25 + 0.13 * np.clip(speed, 0, 9)) * (0.5 + 0.5 * np.sin(ph + 1.3))
        shoulder[..., 1 - side] = -0.9 * m * amp * np.sin(ph)
    elbow[:] = (0.35 + 0.9 * m)[..., None]

    # body actions
    for act in win.actions:
        k = pid_index.get(act["player_id"])
        if k is None:
            continue
        fa = (act["t"] - win.t_start) * fps
        kind = act["kind"]
        if kind == "kick":
            leg = 0 if act.get("foot") == "left" else 1
            power = float(act.get("power", 0.8))
            f0, f1 = int(fa - 0.42 * fps), int(fa + 0.4 * fps)
            w = _envelope(F, f0, f1, 3)
            u = (np.arange(F) - fa) / fps
            kh = np.interp(u, _KICK_KEYS[:, 0], _KICK_KEYS[:, 1]) * (0.6 + 0.4 * power)
            kk = np.interp(u, _KICK_KEYS[:, 0], _KICK_KEYS[:, 2])
            hip[:, k, leg] = (1 - w) * hip[:, k, leg] + w * kh
            knee[:, k, leg] = (1 - w) * knee[:, k, leg] + w * kk
            hip[:, k, 1 - leg] = (1 - w) * hip[:, k, 1 - leg] + w * 0.12
            knee[:, k, 1 - leg] = (1 - w) * knee[:, k, 1 - leg] + w * 0.38
            abduct[:, k, 1 - leg] = (1 - w) * abduct[:, k, 1 - leg] + w * 1.15
            lean[:, k] = (1 - w) * lean[:, k] + w * (0.02 if act.get("shot") else 0.08)
            # face where the ball goes
            fi = int(np.clip(fa, 0, F - 1))
            fj = int(np.clip(fa + 4, 0, F - 1))
            d = ball[fj, :2] - ball[fi, :2]
            if np.linalg.norm(d) > 0.3:
                d = d / np.linalg.norm(d)
                face[:, k] = (1 - w[:, None]) * face[:, k] + w[:, None] * d
        elif kind == "tackle":
            f0, f1 = int(fa - 0.1 * fps), int(fa + 0.75 * fps)
            w = _envelope(F, f0, f1, 4)
            hip[:, k, 1] = (1 - w) * hip[:, k, 1] + w * 1.35
            knee[:, k, 1] = (1 - w) * knee[:, k, 1] + w * 0.1
            hip[:, k, 0] = (1 - w) * hip[:, k, 0] + w * 0.35
            knee[:, k, 0] = (1 - w) * knee[:, k, 0] + w * 1.5
            lean[:, k] = (1 - w) * lean[:, k] + w * -0.55
            abduct[:, k, :] = (1 - w[:, None]) * abduct[:, k, :] + w[:, None] * 0.7
        elif kind == "dive":
            f0, f1 = int(fa - 0.2 * fps), int(fa + 0.9 * fps)
            w = _envelope(F, f0, f1, 5)
            sgn = 1.0 if act.get("side") == "left" else -1.0
            reach = float(np.clip(abs(act.get("dy", 1.5)), 0.5, 2.6))
            roll[:, k] = w * sgn * 1.2
            lateral[:, k] = w * sgn * reach * 0.6
            airborne[:, k] = w
            shoulder[:, k, :] = (1 - w[:, None]) * shoulder[:, k, :] + w[:, None] * 2.9
            elbow[:, k, :] = (1 - w[:, None]) * elbow[:, k, :] + w[:, None] * 0.15
        elif kind == "receive":
            leg = 1
            f0, f1 = int(fa - 0.12 * fps), int(fa + 0.18 * fps)
            w = _envelope(F, f0, f1, 2)
            hip[:, k, leg] = (1 - w) * hip[:, k, leg] + w * 0.35
            knee[:, k, leg] = (1 - w) * knee[:, k, leg] + w * 0.55
            # shoulder checks before receiving (more likely for high-vision players)
            p_scan = (vision or {}).get(act["player_id"], 65) / 100
            for _ in range(2):
                if rng.random() < p_scan:
                    ts = fa - rng.uniform(0.6, 2.4) * fps
                    sw = _envelope(F, int(ts), int(ts + 0.38 * fps), 3)
                    head_off[:, k] += sw * rng.choice([-1, 1]) * rng.uniform(0.9, 1.35)

    # ambient scans
    for k in range(P):
        t = rng.uniform(0, 3)
        while t < F / fps:
            ts = t * fps
            sw = _envelope(F, int(ts), int(ts + 0.32 * fps), 3)
            head_off[:, k] += sw * rng.choice([-1, 1]) * rng.uniform(0.7, 1.2)
            t += rng.uniform(2.0, 5.0)

    face = _ema(face, 0.3)
    face = face / (np.linalg.norm(face, axis=-1, keepdims=True) + 1e-9)

    # forward kinematics -------------------------------------------------------
    up = np.array([0.0, 0.0, 1.0])
    f3 = np.concatenate([face, np.zeros((F, P, 1))], axis=-1)
    l3 = np.stack([-face[..., 1], face[..., 0], np.zeros((F, P))], axis=-1)
    cr, sr = np.cos(roll)[..., None], np.sin(roll)[..., None]
    leg_up = up * cr + l3 * sr  # body vertical after roll
    lat = l3 * cr - up * sr  # body lateral after roll
    cl, sl = np.cos(lean)[..., None], np.sin(lean)[..., None]
    torso_up = leg_up * cl + f3 * sl
    s = scale[..., None]

    root = np.concatenate([pos, np.zeros((F, P, 1))], axis=-1) + lat * lateral[..., None]
    pelvis = root + leg_up * 0.95 * s
    out = np.zeros((F, P, 17, 3))
    out[:, :, J["pelvis"]] = pelvis
    neck = pelvis + torso_up * 0.52 * s
    out[:, :, J["neck"]] = neck
    hy = np.arctan2(face[..., 1], face[..., 0]) + head_off
    gaze = np.stack([np.cos(hy), np.sin(hy), np.zeros_like(hy)], axis=-1)
    out[:, :, J["head"]] = neck + torso_up * 0.17 * s + gaze * 0.07 * s

    for side, name in ((0, "l"), (1, "r")):
        sgn = 1.0 if side == 0 else -1.0
        sh = neck + lat * sgn * 0.19 * s - torso_up * 0.04 * s
        out[:, :, J[f"{name}_shoulder"]] = sh
        a = shoulder[..., side][..., None]
        ab = abduct[..., side][..., None]
        d1 = (-torso_up * np.cos(a) + f3 * np.sin(a)) * np.cos(ab) + lat * sgn * np.sin(ab)
        el = sh + d1 * 0.29 * s
        out[:, :, J[f"{name}_elbow"]] = el
        a2 = a + elbow[..., side][..., None]
        d2 = (-torso_up * np.cos(a2) + f3 * np.sin(a2)) * np.cos(ab) + lat * sgn * np.sin(ab)
        out[:, :, J[f"{name}_wrist"]] = el + d2 * 0.26 * s

        hp = pelvis + lat * sgn * 0.1 * s
        out[:, :, J[f"{name}_hip"]] = hp
        h = hip[..., side][..., None]
        kn = knee[..., side][..., None]
        thigh = -leg_up * np.cos(h) + f3 * np.sin(h)
        kp = hp + thigh * 0.45 * s
        out[:, :, J[f"{name}_knee"]] = kp
        shin = -leg_up * np.cos(h - kn) + f3 * np.sin(h - kn)
        an = kp + shin * 0.44 * s
        out[:, :, J[f"{name}_ankle"]] = an
        out[:, :, J[f"{name}_toe"]] = an + f3 * 0.16 * s - leg_up * 0.04 * s

    # ground contact: lowest foot joint rests at ~7 cm; airborne keepers float
    low = np.minimum.reduce([out[:, :, J[j], 2] for j in ("l_ankle", "r_ankle", "l_toe", "r_toe")]) - 0.07
    out[..., 2] -= (low * (1 - airborne) + airborne * (low - 0.25))[..., None]
    out[..., 2] = np.maximum(out[..., 2], 0.0)

    return SkeletonWindow.model_construct(
        id=f"sk-{win.id}",
        window_id=win.id,
        source=win.source,
        t_start=win.t_start,
        fps=fps,
        n_frames=F,
        joint_names=list(J.keys()),
        bones=list(BONES),
        player_ids=list(win.player_ids),
        frames=np.round(out, 2).tolist(),
        confidence=None,
    )
