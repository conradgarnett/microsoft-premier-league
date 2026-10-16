"""Stage 1: per-frame detections, pitch positions, torso colours and 2D pose for a clip.

    python -m vision.extract <video path or URL> <camera.json> <start_s> <duration_s> <out.npz> [--fps 12.5]

Outputs stay wherever ``out`` points (keep real-footage outputs out of the public repo).
"""

from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from .calibrate import HALVES, PanoramaCamera
from .detect import Detector
from .pose import PoseEstimator
from .video import iter_frames

MARGIN_M = 3.0  # keep detections up to 3 m outside the lines (throw-ins, keepers behind the line)


def torso_colour(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    crop = img[int(y0 + 0.18 * h) : int(y0 + 0.45 * h), int(x0 + 0.3 * w) : int(x0 + 0.7 * w)]
    if crop.size == 0:
        return np.zeros(3)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    return np.median(lab, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("camera")
    ap.add_argument("start", type=float)
    ap.add_argument("duration", type=float)
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=12.5)
    a = ap.parse_args()
    cam = PanoramaCamera(json.load(open(a.camera))["params"])
    L, W = cam.p["L"], cam.p["W"]
    det, pose = Detector(), PoseEstimator()
    rows = []  # one row per kept person detection
    balls = []
    t0 = time.time()
    n = 0
    for n, (t, img) in enumerate(iter_frames(a.src, a.start, a.duration, a.fps)):
        halves = [img[cam.p["rows"][h][0] : cam.p["rows"][h][1]] for h in HALVES]
        res = det(halves)
        for hi, (h, sub, r) in enumerate(zip(HALVES, halves, res, strict=True)):
            off = cam.p["rows"][h][0]
            person = r["labels"] == det.person
            boxes = r["boxes"][person]
            if len(boxes):
                feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3] + off], 1)
                g = cam.to_pitch(h, feet)
                ok = np.isfinite(g).all(1) & (g[:, 0] > -MARGIN_M) & (g[:, 0] < L + MARGIN_M) & (g[:, 1] > -MARGIN_M) & (g[:, 1] < W + MARGIN_M)
                boxes, g, sc = boxes[ok], g[ok], r["scores"][person][ok]
                kp, ks = pose(sub, boxes)
                for k in range(len(boxes)):
                    rows.append({"frame": n, "t": t, "half": hi, "box": boxes[k] + [0, off, 0, off], "score": sc[k], "xy": g[k], "colour": torso_colour(sub, boxes[k]), "kp": kp[k] + [0, off], "kps": ks[k]})
            for k in np.where(r["labels"] == det.ball)[0]:
                b = r["boxes"][k]
                c = np.array([[(b[0] + b[2]) / 2, b[3] + off]])
                balls.append({"frame": n, "t": t, "half": hi, "px": c[0], "score": r["scores"][k], "xy": cam.to_pitch(h, c)[0]})
        if n % 25 == 0:
            print(f"frame {n} t={t:.1f}s persons={sum(1 for r_ in rows if r_['frame'] == n)} elapsed={time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(
        a.out,
        frame=np.array([r["frame"] for r in rows]),
        t=np.array([r["t"] for r in rows]),
        half=np.array([r["half"] for r in rows]),
        box=np.array([r["box"] for r in rows]).reshape(-1, 4),
        score=np.array([r["score"] for r in rows]),
        xy=np.array([r["xy"] for r in rows]).reshape(-1, 2),
        colour=np.array([r["colour"] for r in rows]).reshape(-1, 3),
        kp=np.array([r["kp"] for r in rows]).reshape(-1, 17, 2),
        kps=np.array([r["kps"] for r in rows]).reshape(-1, 17),
        ball=np.array([[b["frame"], b["t"], b["half"], *b["px"], b["score"], *b["xy"]] for b in balls]).reshape(-1, 8),
        meta=np.array(json.dumps({"start": a.start, "duration": a.duration, "fps": a.fps, "frames": n + 1, "camera": a.camera})),
    )
    print(f"done: {n + 1} frames, {len(rows)} person detections, {len(balls)} ball detections in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
