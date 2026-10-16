"""Ball accuracy against hand labels.

    python -m vision.evaluate <ball.npz> <labels.json> <camera.json> <clip_start_s>

labels.json: {"labels": [{"t": source seconds, "half": "top"|"bottom", "px": [u, v], "kind": "ground"|...}]}
with ``px`` the ground point under the ball in the full panorama. Only ``ground`` labels are
scored. Prints per-label errors and the summary; label files of real footage stay private.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .calibrate import PanoramaCamera


def ball_errors(ball: dict, labels: list[dict], cam: PanoramaCamera, clip_start: float, fps: float = 25.0) -> list[dict]:
    out = []
    for lab in labels:
        if lab.get("kind") != "ground":
            continue
        f = int(round((lab["t"] - clip_start) * fps))
        if not 0 <= f < len(ball["xy"]):
            continue
        gt = cam.to_pitch(lab["half"], np.array([lab["px"]], float))[0]
        out.append({"t": lab["t"], "err_m": float(np.linalg.norm(ball["xy"][f] - gt)), "prior_err_m": float(np.linalg.norm(ball["prior"][f] - gt)), "conf": float(ball["conf"][f])})
    return out


def main() -> None:
    ball_path, labels_path, cam_path, start = sys.argv[1:5]
    ball = dict(np.load(ball_path))
    cam = PanoramaCamera(json.load(open(cam_path))["params"])
    rows = ball_errors(ball, json.load(open(labels_path))["labels"], cam, float(start))
    for r in rows:
        print(f"t={r['t']:.1f} err={r['err_m']:.1f} m prior={r['prior_err_m']:.1f} m conf={r['conf']:.1f}")
    if rows:
        e = np.array([r["err_m"] for r in rows])
        print(f"{len(rows)} labels: median {np.median(e):.1f} m, within 2 m {np.mean(e < 2):.0%}, median prior {np.median([r['prior_err_m'] for r in rows]):.1f} m")
    c = ball["conf"]
    print(f"frames: detector {np.mean(c > 0.85):.0%}, motion {np.mean((c > 0.5) & (c <= 0.85)):.0%}, prior only {np.mean(c <= 0.5):.0%}")


if __name__ == "__main__":
    main()
