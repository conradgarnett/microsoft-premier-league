"""Ball localisation for a fixed panorama camera.

A 4-8 px ball defeats the generic detector (it fires on white boots), so we combine:

1. **Motion candidates**: static-camera background subtraction; small, round, bright
   blobs outside person boxes and inside the pitch.
2. **Follow-cam prior**: the recording's virtual broadcast camera (``camera_directions``:
   yaw, pitch, zoom per frame) follows play. A linear map from (yaw, pitch) to the
   bearing/elevation of ground points, fit by RANSAC against motion candidates sampled
   across the match, turns it into a coarse ball position.
3. **Selection**: per frame, the candidate nearest the prior (within a gate), smoothed;
   frames without a candidate fall back to the prior with low confidence.
"""

from __future__ import annotations

import cv2
import numpy as np

from .calibrate import PanoramaCamera


def candidates(gray: np.ndarray, bg: np.ndarray, boxes: np.ndarray, cam: PanoramaCamera) -> np.ndarray:
    """-> (N, 5) [u, v, area, pitch_x, pitch_y] ball-like blobs outside person boxes."""
    diff = gray.astype(np.int16) - bg.astype(np.int16)
    m = ((diff > 28) & (gray > 120)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (3 <= a <= 80 and 0.5 <= w / max(h, 1) <= 2.0 and max(w, h) <= 14):
            continue
        cx, cy = cen[i]
        if len(boxes) and ((boxes[:, 0] - 6 <= cx) & (cx <= boxes[:, 2] + 6) & (boxes[:, 1] - 6 <= cy) & (cy <= boxes[:, 3] + 6)).any():
            continue
        half = cam.half_of(cy)
        g = cam.to_pitch(half, np.array([[cx, cy]]))[0]
        if np.isfinite(g).all() and -2 < g[0] < cam.p["L"] + 2 and -2 < g[1] < cam.p["W"] + 2:
            out.append((cx, cy, a, g[0], g[1]))
    return np.array(out, float).reshape(-1, 5)


def bearing_elev(cam: PanoramaCamera, xy: np.ndarray) -> np.ndarray:
    """Ground points -> (bearing from the across-pitch axis, elevation) seen from the camera, radians."""
    C = np.asarray(cam.p["C"])
    d = np.atleast_2d(xy) - C[:2]
    return np.stack([np.arctan2(d[:, 0], d[:, 1]), -np.arctan2(C[2], np.linalg.norm(d, axis=1))], 1)


def fit_followcam(samples: list[tuple[np.ndarray, np.ndarray]], cam: PanoramaCamera, iters: int = 4000, tol: float = 0.04, seed: int = 0) -> dict:
    """RANSAC: bearing = a*yaw + b and elevation = c*pitch + d from (direction, candidates) samples.

    samples: list of (direction row [yaw, pitch, zoom, roll], candidates (N,5)).
    """
    rows = []
    for dirn, cand in samples:
        for be in bearing_elev(cam, cand[:, 3:5]) if len(cand) else []:
            rows.append((dirn[0], dirn[1], be[0], be[1]))
    r = np.array(rows)
    rng = np.random.default_rng(seed)
    best, best_in = None, -1
    for _ in range(iters):
        i, j = rng.choice(len(r), 2, replace=False)
        if abs(r[i, 0] - r[j, 0]) < 0.05:
            continue
        a = (r[i, 2] - r[j, 2]) / (r[i, 0] - r[j, 0])
        b = r[i, 2] - a * r[i, 0]
        inl = np.abs(a * r[:, 0] + b - r[:, 2]) < tol
        if inl.sum() > best_in:
            best, best_in = (a, b), inl.sum()
    a, b = best
    inl = np.abs(a * r[:, 0] + b - r[:, 2]) < tol
    a, b = np.polyfit(r[inl, 0], r[inl, 2], 1)  # refit on inliers
    c, d = np.polyfit(r[inl, 1], r[inl, 3], 1)
    return {"a": float(a), "b": float(b), "c": float(c), "d": float(d), "inliers": int(inl.sum()), "n": len(r)}


def prior_xy(fc: dict, cam: PanoramaCamera, dirn: np.ndarray) -> np.ndarray:
    """Follow-cam direction rows (N,4) -> ground points (N,2) the virtual camera is aimed at."""
    C = np.asarray(cam.p["C"])
    bearing = fc["a"] * dirn[:, 0] + fc["b"]
    elev = np.minimum(fc["c"] * dirn[:, 1] + fc["d"], -0.05)
    dist = C[2] / np.tan(-elev)
    return C[:2] + np.stack([np.sin(bearing), np.cos(bearing)], 1) * dist[:, None]


def select(prior: np.ndarray, cands: list[np.ndarray], gate_m: float = 9.0, fps: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """Per frame pick the candidate nearest the prior and the last pick; returns (xy (F,2), confidence (F,))."""
    F = len(prior)
    xy = prior.copy()
    conf = np.full(F, 0.2)
    last = None
    for f in range(F):
        c = cands[f]
        if len(c):
            ref = prior[f] if last is None or f - last[0] > fps else 0.5 * prior[f] + 0.5 * last[1]
            d = np.linalg.norm(c[:, 3:5] - ref, axis=1)
            k = int(np.argmin(d))
            if d[k] < gate_m:
                xy[f] = c[k, 3:5]
                conf[f] = 0.8
                last = (f, xy[f])
    # smooth: short median filter, then hold measured points between gaps
    k = 5
    pad = np.pad(xy, ((k // 2, k // 2), (0, 0)), mode="edge")
    xy = np.stack([np.median(pad[i : i + k], axis=0) for i in range(F)])
    return xy, conf
