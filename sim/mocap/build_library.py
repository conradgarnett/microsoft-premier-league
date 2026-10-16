"""Rebuild ``sim/mocap/library.npz`` from CMU Graphics Lab Motion Capture Database clips.

Source: http://mocap.cs.cmu.edu/ (free for all uses; see README "Attribution").

    python -m sim.mocap.build_library            # downloads ASF/AMC into .cache/cmu, writes library.npz

Every clip is converted to the 17-joint schema (``sim.schema.JOINT_NAMES``), put in a
root-relative, heading-aligned frame (pelvis over the origin, facing +x, z up), scaled
to a 0.89 m leg (1.80 m player), and either cut into a single phase-aligned locomotion
cycle (starting at left-foot strike) or kept as a one-shot action clip.
"""

from __future__ import annotations

import json
import pathlib
import urllib.request

import numpy as np

from sim.schema import JOINT_NAMES, J

from .cmu import CMU_TO_SCHEMA, forward_kinematics

SRC_FPS = 120
OUT_FPS = 60
CYCLE_SAMPLES = 64
LEG_LEN = 0.89
CACHE = pathlib.Path(".cache/cmu")
OUT = pathlib.Path(__file__).with_name("library.npz")

# name -> (clip, kind). Speeds are measured from the data, not assumed.
CYCLES = {
    "walk": "02_01",
    "walk_fast": "02_02",
    "jog": "16_35",
    "jog_fast": "35_17",
    "run": "09_01",
    "run_fast": "16_46",
    "shuffle": "78_24",
}
# Subjects 141+ appear to be 60 fps despite the 120 fps header (cadence is implausible otherwise), so they are left out.
KICKS = ["10_01", "10_02", "10_03", "10_05", "10_06", "11_01"]
IDLE = "10_01"  # standing frames before the run-up
DIVE = "141_05"  # sideways jump, used as the goalkeeper dive base
JUMP = "13_11"


def _fetch(clip: str) -> tuple[str, str]:
    subj = clip.split("_")[0]
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in (f"{subj}.asf", f"{clip}.amc"):
        p = CACHE / name
        if not p.exists():
            url = f"http://mocap.cs.cmu.edu/subjects/{subj}/{name}"
            p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return (CACHE / f"{subj}.asf").read_text(), (CACHE / f"{clip}.amc").read_text()


def load(clip: str) -> np.ndarray:
    """(F,17,3) joints in metres, z up, floor at z=0, at SRC_FPS."""
    asf, amc = _fetch(clip)
    fk = forward_kinematics(asf, amc)
    y_up = np.stack([fk[CMU_TO_SCHEMA[n]] for n in JOINT_NAMES], axis=1)
    j = np.stack([y_up[..., 0], -y_up[..., 2], y_up[..., 1]], axis=-1)  # (x, -z, y): z up, right-handed
    leg = np.linalg.norm(j[:, J["l_hip"]] - j[:, J["l_knee"]], axis=-1) + np.linalg.norm(j[:, J["l_knee"]] - j[:, J["l_ankle"]], axis=-1)
    j *= LEG_LEN / np.median(leg)
    feet = j[:, [J["l_ankle"], J["r_ankle"], J["l_toe"], J["r_toe"]], 2].min(axis=1)
    j[..., 2] -= np.percentile(feet, 5) - 0.04
    return j


def heading(j: np.ndarray) -> np.ndarray:
    lr = j[:, J["l_hip"], :2] - j[:, J["r_hip"], :2]
    return np.arctan2(-lr[:, 0], lr[:, 1])


def localize(j: np.ndarray, yaw: np.ndarray | float, keep_root_xy: bool = False) -> np.ndarray:
    """Pelvis over origin (unless keep_root_xy), rotate so heading points to +x."""
    yaw = np.broadcast_to(np.asarray(yaw, float), (len(j),))
    out = j.copy()
    root = j[:, J["pelvis"], :2].copy()
    if keep_root_xy:
        root = np.broadcast_to(root[0], root.shape)
    out[..., :2] -= root[:, None, :]
    c, s = np.cos(-yaw)[:, None], np.sin(-yaw)[:, None]
    x, y = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0], out[..., 1] = c * x - s * y, s * x + c * y
    return out


def mirror(j: np.ndarray) -> np.ndarray:
    """Swap left/right and flip the lateral axis (local frame)."""
    idx = [JOINT_NAMES.index(n.replace("l_", "#").replace("r_", "l_").replace("#", "r_")) for n in JOINT_NAMES]
    out = j[:, idx].copy()
    out[..., 1] *= -1
    return out


def resample(a: np.ndarray, n: int) -> np.ndarray:
    src = np.linspace(0, 1, len(a))
    dst = np.linspace(0, 1, n)
    flat = a.reshape(len(a), -1)
    return np.stack([np.interp(dst, src, flat[:, k]) for k in range(flat.shape[1])], axis=1).reshape(n, *a.shape[1:])


def smooth_yaw(yaw: np.ndarray, k: int = 25) -> np.ndarray:
    u = np.unwrap(yaw)
    return np.convolve(np.pad(u, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")


def extract_cycle(clip: str, lateral: bool = False) -> dict:
    j = load(clip)
    root = j[:, J["pelvis"], :2]
    vel = np.gradient(root, 1 / SRC_FPS, axis=0)
    if lateral:
        # shuffle: body faces across the direction of travel; normalise travel to +y (left)
        yaw = smooth_yaw(heading(j))
        local_v = np.stack([np.cos(-yaw) * vel[:, 0] - np.sin(-yaw) * vel[:, 1], np.sin(-yaw) * vel[:, 0] + np.cos(-yaw) * vel[:, 1]], 1)
        lat = local_v[:, 1]
        seg = np.abs(lat) > 0.6 * np.abs(lat).max()
        loc = localize(j, yaw)
        if np.median(lat[seg]) < 0:
            loc = mirror(loc)
        foot = loc[:, J["l_ankle"], 1]  # lateral foot position drives the cycle
        speed = float(np.abs(lat[seg]).mean())
    else:
        yaw = smooth_yaw(np.arctan2(vel[:, 1], vel[:, 0]))
        loc = localize(j, yaw)
        seg = np.ones(len(j), bool)
        foot = loc[:, J["l_ankle"], 0]  # forward foot position
        speed = float(np.linalg.norm(vel, axis=1)[len(j) // 4 : -len(j) // 4].mean())
    # one full cycle between two same-type extremes of the left foot, then rolled so every
    # cycle starts with the left foot furthest forward (phase-aligned for blending)
    def extremes(sig: np.ndarray) -> list[int]:
        pk = [f for f in range(2, len(sig) - 2) if seg[f] and sig[f] == sig[f - 2 : f + 3].max()]
        return [p for i, p in enumerate(pk) if i == 0 or p - pk[i - 1] > SRC_FPS * 0.35]

    peaks = extremes(foot)
    if len(peaks) < 2:
        peaks = extremes(-foot)
    if len(peaks) < 2:
        raise ValueError(f"{clip}: no full cycle")
    mid = len(peaks) // 2 - 1 if len(peaks) > 2 else 0
    a, b = peaks[mid], peaks[mid + 1]
    cyc = loc[a : b + 1].copy()
    drift = cyc[-1] - cyc[0]
    cyc -= np.linspace(0, 1, len(cyc))[:, None, None] * drift[None]  # make it loop
    cyc = np.roll(cyc[:-1], -int(np.argmax(foot[a:b])), axis=0)
    dur = (b - a) / SRC_FPS
    return {"pose": resample(np.concatenate([cyc, cyc[:1]]), CYCLE_SAMPLES + 1)[:-1], "speed": speed, "stride": speed * dur, "duration": dur, "clip": clip}


def extract_kick(clip: str) -> dict:
    j = load(clip)
    toe_v = {s: np.linalg.norm(np.gradient(j[:, J[f"{s}_toe"]], 1 / SRC_FPS, axis=0), axis=1) for s in "lr"}
    side = "l" if toe_v["l"].max() > toe_v["r"].max() else "r"
    contact = int(np.argmax(toe_v[side]))
    a, b = max(0, contact - int(1.0 * SRC_FPS)), min(len(j), contact + int(0.8 * SRC_FPS))
    yaw = smooth_yaw(heading(j))[contact]
    loc = localize(j[a:b], yaw)
    if side == "l":
        loc = mirror(loc)
    n = int(len(loc) * OUT_FPS / SRC_FPS)
    return {"pose": resample(loc, n), "contact": (contact - a) * OUT_FPS / SRC_FPS, "foot_speed": float(toe_v[side].max()), "clip": clip}


def extract_idle() -> np.ndarray:
    j = load(IDLE)
    v = np.linalg.norm(np.gradient(j[:, J["pelvis"], :2], 1 / SRC_FPS, axis=0), axis=1)
    still = np.where(v < 0.15)[0]
    seg = j[still[0] : still[0] + SRC_FPS]
    loc = localize(seg, smooth_yaw(heading(seg)).mean())
    return resample(loc, OUT_FPS)


def extract_dive() -> dict:
    j = load(DIVE)
    yaw0 = smooth_yaw(heading(j))[0]
    loc = localize(j, yaw0, keep_root_xy=True)
    lat_v = np.gradient(loc[:, J["pelvis"], 1], 1 / SRC_FPS)
    peak = int(np.argmax(np.abs(lat_v)))
    if lat_v[peak] < 0:
        loc = mirror(loc)
    a, b = max(0, peak - int(0.45 * SRC_FPS)), min(len(j), peak + int(0.9 * SRC_FPS))
    seg = loc[a:b].copy()
    seg[..., :2] -= seg[0, J["pelvis"], :2]
    return {"pose": resample(seg, int(len(seg) * OUT_FPS / SRC_FPS)), "peak": (peak - a) * OUT_FPS / SRC_FPS, "clip": DIVE}


def main() -> None:
    cycles = {name: extract_cycle(clip, lateral=name == "shuffle") for name, clip in CYCLES.items()}
    kicks = sorted((extract_kick(c) for c in KICKS), key=lambda k: k["foot_speed"])
    idle = extract_idle()
    dive = extract_dive()
    loco = sorted((n for n in cycles if n != "shuffle"), key=lambda n: cycles[n]["speed"])
    arrays = {
        "loco_pose": np.stack([cycles[n]["pose"] for n in loco]).astype(np.float32),
        "loco_speed": np.array([cycles[n]["speed"] for n in loco], np.float32),
        "loco_stride": np.array([cycles[n]["stride"] for n in loco], np.float32),
        "shuffle_pose": cycles["shuffle"]["pose"].astype(np.float32),
        "shuffle_speed": np.float32(cycles["shuffle"]["speed"]),
        "shuffle_stride": np.float32(cycles["shuffle"]["stride"]),
        "idle_pose": idle.astype(np.float32),
        "dive_pose": dive["pose"].astype(np.float32),
        "dive_peak": np.float32(dive["peak"]),
    }
    for i, k in enumerate(kicks):
        arrays[f"kick{i}_pose"] = k["pose"].astype(np.float32)
        arrays[f"kick{i}_contact"] = np.float32(k["contact"])
        arrays[f"kick{i}_foot_speed"] = np.float32(k["foot_speed"])
    meta = {
        "source": "CMU Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu/ (funded by NSF EIA-0196217)",
        "fps": OUT_FPS,
        "cycle_samples": CYCLE_SAMPLES,
        "loco": [{"name": n, "clip": cycles[n]["clip"], "speed": round(cycles[n]["speed"], 2), "stride": round(cycles[n]["stride"], 2)} for n in loco],
        "shuffle": {"clip": CYCLES["shuffle"], "speed": round(cycles["shuffle"]["speed"], 2)},
        "kicks": [{"clip": k["clip"], "foot_speed": round(k["foot_speed"], 1)} for k in kicks],
        "idle": IDLE,
        "dive": DIVE,
    }
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(OUT, **arrays)
    print(json.dumps(meta, indent=1))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
