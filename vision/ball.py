"""Ball localisation for a fixed panorama camera.

A 4-8 px ball defeats the generic detector on downscaled frames (it fires on white boots), so we combine:

0. **Crop detector**: RT-DETR "sports ball" on a full-resolution crop around the prior,
   a few times per second (the strongest cue when it fires).

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


def detector_candidates(det, img: np.ndarray, cam: PanoramaCamera, prior: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """-> (N, 5) [u, v, score, pitch_x, pitch_y] "sports ball" detections in a native-resolution crop.

    Downscaled whole halves shrink the ball to 2-4 px and the detector misses it; a 1024x512
    crop at full resolution around the follow-cam prior keeps it at 5-8 px. Spare balls by
    the benches are dropped by requiring a position on the pitch.
    """
    out = []
    for half in ("top", "bottom"):
        r0, r1 = cam.p["rows"][half]
        uv = cam.to_pixels(half, np.array([[prior[0], prior[1], 0.0]]))[0]
        if not (np.isfinite(uv).all() and 0 <= uv[0] < img.shape[1] and r0 <= uv[1] < r1):
            continue
        x0 = int(np.clip(uv[0] - 512, 0, img.shape[1] - 1024))
        y0 = int(np.clip(uv[1] - 256, r0, r1 - 512))
        res = det([img[y0 : y0 + 512, x0 : x0 + 1024]], size=(512, 1024), threshold=threshold)[0]
        for b, sc in zip(res["boxes"][res["labels"] == det.ball], res["scores"][res["labels"] == det.ball], strict=True):
            u, v = (b[0] + b[2]) / 2 + x0, b[3] + y0
            g = cam.to_pitch(half, np.array([[u, v]]))[0]
            if np.isfinite(g).all() and -1 < g[0] < cam.p["L"] + 1 and -1 < g[1] < cam.p["W"] + 1:
                out.append((u, v, sc, g[0], g[1]))
        break
    return np.array(out, float).reshape(-1, 5)


def select(prior: np.ndarray, cands: list[np.ndarray], dets: list[np.ndarray] | None = None, gate_m: float = 9.0, det_gate_m: float = 25.0, fps: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """Per frame pick a ball position; returns (xy (F,2), confidence (F,)).

    Detector hits (``dets``, sparse frames) win: highest score, penalised by distance to the
    reference (last pick if recent, else the follow-cam prior), confidence 0.9. Otherwise the
    motion candidate nearest the reference within ``gate_m`` (0.8). Else the prior (0.2).
    """
    F = len(prior)
    xy = prior.copy()
    conf = np.full(F, 0.2)
    last = None
    for f in range(F):
        recent = last is not None and f - last[0] <= fps
        ref = 0.3 * prior[f] + 0.7 * last[1] if recent else prior[f]
        dc = dets[f] if dets is not None else np.zeros((0, 5))
        if len(dc):
            d = np.linalg.norm(dc[:, 3:5] - ref, axis=1)
            ok = d < det_gate_m
            if ok.any():
                k = int(np.argmax(np.where(ok, dc[:, 2] - d / 40, -np.inf)))
                xy[f], conf[f] = dc[k, 3:5], 0.9
                last = (f, xy[f])
                continue
        c = cands[f]
        if len(c):
            d = np.linalg.norm(c[:, 3:5] - ref, axis=1)
            k = int(np.argmin(d))
            if d[k] < (4.0 + 25.0 * (f - last[0]) / fps if recent else gate_m):  # ball moves <= ~25 m/s
                xy[f], conf[f] = c[k, 3:5], 0.8
                last = (f, xy[f])
    # smooth measured points only (median of measured neighbours within +-2 frames); prior frames stay as they are
    meas = conf > 0.5
    out = xy.copy()
    for f in np.where(meas)[0]:
        nb = [g for g in range(max(0, f - 2), min(F, f + 3)) if meas[g]]
        out[f] = np.median(xy[nb], axis=0)
    return out, conf


def main() -> None:
    """python -m vision.ball <video> <camera.json> <followcam.json> <directions.det> <extract.npz> <out.npz>"""
    import argparse
    import json
    from collections import deque

    from .video import iter_frames

    ap = argparse.ArgumentParser()
    for name in ("video", "camera", "followcam", "directions", "extract", "out"):
        ap.add_argument(name)
    ap.add_argument("--src-fps", type=float, default=29.97, help="frame rate the directions file is sampled at")
    ap.add_argument("--det-fps", type=float, default=5.0, help="crop detector rate (0 = motion candidates only)")
    a = ap.parse_args()
    cam = PanoramaCamera(json.load(open(a.camera))["params"])
    fc = json.load(open(a.followcam))
    dirs = np.loadtxt(a.directions, delimiter=",")
    ex = np.load(a.extract)
    meta = json.loads(str(ex["meta"]))
    fps = 25.0
    # stream with a +-2 s ring buffer (a full 25 fps clip at 2048x2048 would need gigabytes)
    half_win, step = 50, 10
    buf: deque[np.ndarray] = deque()
    ts: list[float] = []
    cands: list[np.ndarray] = []

    def process(f: int) -> None:
        lo = max(0, f - half_win)
        bg = np.median(np.stack([buf[i - (len(ts) - len(buf))] for i in range(lo, min(len(ts), f + half_win), step)]), axis=0)
        pf = int(round((ts[f] - meta["start"]) * meta["fps"]))
        boxes = ex["box"][ex["frame"] == pf]
        cands.append(candidates(buf[f - (len(ts) - len(buf))], bg, boxes, cam))

    det, every = None, 0
    if a.det_fps > 0:
        from .detect import Detector

        det, every = Detector(), max(1, int(round(fps / a.det_fps)))
    dets: list[np.ndarray] = []
    for t, img in iter_frames(a.video, meta["start"], meta["duration"], fps):
        buf.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        ts.append(t)
        pr = prior_xy(fc, cam, dirs[[min(int(round(t * a.src_fps)), len(dirs) - 1)]])[0]
        dets.append(detector_candidates(det, img, cam, pr) if det and (len(ts) - 1) % every == 0 else np.zeros((0, 5)))
        while len(cands) < len(ts) - half_win:
            process(len(cands))
        while len(buf) > 2 * half_win + 1 and len(ts) - len(buf) < len(cands) - half_win:
            buf.popleft()
    while len(cands) < len(ts):
        process(len(cands))
    prior = prior_xy(fc, cam, dirs[np.clip(np.round(np.array(ts) * a.src_fps).astype(int), 0, len(dirs) - 1)])
    xy, conf = select(prior, cands, dets)
    stack = lambda L: np.concatenate([np.c_[np.full(len(c), f), c] for f, c in enumerate(L)]) if L else np.zeros((0, 6))  # noqa: E731
    # candidates are saved so selection can be re-run without decoding the clip again
    np.savez_compressed(a.out, t=np.array(ts) - meta["start"], xy=xy, conf=conf, prior=prior, n_cands=np.array([len(c) for c in cands]), cands=stack(cands), dets=stack(dets))
    print(f"ball: {len(ts)} frames, {np.mean(conf > 0.85):.0%} detector, {np.mean(conf > 0.5):.0%} with any matched candidate, median candidates/frame {np.median([len(c) for c in cands]):.0f}")


if __name__ == "__main__":
    main()
