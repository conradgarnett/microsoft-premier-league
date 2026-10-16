"""Pitch calibration for a fixed wide-angle (fisheye) panorama render.

The panorama render of a fixed side-line camera is two stacked fisheye views (left and
right halves of the pitch) from one camera position. Each half is modelled as an
equidistant fisheye ``r = f * (theta + k1 theta^3 + k2 theta^5)`` with its own orientation, focal
length and principal point; the camera position and the pitch length/width are shared.
Parameters are fit by least squares to a handful of manual correspondences:

* points with both pitch coordinates known (corners, goal posts, halfway-line ends),
* points on a known line (one coordinate known, e.g. any point on the near touchline).

Pitch coordinates follow ``sim.schema``: x along the length (0 = left goal line),
y across (0 = touchline nearest the camera), metres. Symbolic values ``L``, ``W`` and
expressions like ``W/2-3.66`` are allowed in correspondence files, because amateur
pitch dimensions are unknown and are fit too (goal width 7.32 m fixes the scale).

    python -m vision.calibrate calib.json frame.jpg out_params.json [overlay.jpg]
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HALVES = ("top", "bottom")


def _look_at(c: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = target - c
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.stack([right, down, f], axis=1)  # columns: camera x, y, z in world


class PanoramaCamera:
    """Shared position + per-half fisheye intrinsics/orientation. Pixel coords are full-panorama."""

    def __init__(self, params: dict) -> None:
        self.p = params

    # parameter vector <-> dict -------------------------------------------------
    @staticmethod
    def pack(p: dict) -> np.ndarray:
        v = [*p["C"], p["L"], p["W"]]
        for h in HALVES:
            q = p[h]
            v += [*q["rotvec"], q["f"], q["k1"], q["k2"], q["cx"], q["cy"]]
        return np.array(v, float)

    @staticmethod
    def unpack(v: np.ndarray, rows: dict, model: str = "fisheye") -> dict:
        p = {"model": model, "C": v[0:3].tolist(), "L": float(v[3]), "W": float(v[4]), "rows": rows}
        for i, h in enumerate(HALVES):
            o = 5 + 8 * i
            p[h] = {"rotvec": v[o : o + 3].tolist(), "f": float(v[o + 3]), "k1": float(v[o + 4]), "k2": float(v[o + 5]), "cx": float(v[o + 6]), "cy": float(v[o + 7])}
        return p

    # projection ------------------------------------------------------------------
    def to_pitch(self, half: str, px: np.ndarray) -> np.ndarray:
        """Pixels (N,2) of one half -> ground-plane pitch coordinates (N,2). NaN above the horizon."""
        q = self.p[half]
        px = np.atleast_2d(np.asarray(px, float))
        du = px[:, 0] - q["cx"]
        dv = px[:, 1] - self.p["rows"][half][0] - q["cy"]
        if self.p.get("model") == "equirect":
            lon, lp = du / q["f"], dv / q["f"]
            lat = lp.copy()
            for _ in range(4):  # invert lp = lat + k1 lat^3
                lat -= (lat + q["k1"] * lat**3 - lp) / (1 + 3 * q["k1"] * lat**2)
            d = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], 1)
        else:
            r = np.hypot(du, dv)
            th = r / q["f"]
            for _ in range(3):  # Newton on f*(th + k1 th^3 + k2 th^5) = r
                g = q["f"] * (th + q["k1"] * th**3 + q["k2"] * th**5) - r
                th -= g / (q["f"] * (1 + 3 * q["k1"] * th**2 + 5 * q["k2"] * th**4))
            ph = np.arctan2(dv, du)
            d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
        R = Rotation.from_rotvec(q["rotvec"]).as_matrix()
        w = d @ R.T
        C = np.asarray(self.p["C"])
        t = -C[2] / w[:, 2]
        g = C[:2] + t[:, None] * w[:, :2]
        g[t <= 0] = np.nan
        return g

    def to_pixels(self, half: str, xyz: np.ndarray) -> np.ndarray:
        """World points (N,3) -> pixels (N,2) in the full panorama (NaN when behind the lens)."""
        q = self.p[half]
        R = Rotation.from_rotvec(q["rotvec"]).as_matrix()
        c = (np.atleast_2d(xyz) - np.asarray(self.p["C"])) @ R  # world -> camera
        if self.p.get("model") == "equirect":
            n = np.linalg.norm(c, axis=1)
            lon = np.arctan2(c[:, 0], c[:, 2])
            lat = np.arcsin(np.clip(c[:, 1] / n, -1, 1))
            lat = lat + q["k1"] * lat**3  # small vertical warp
            uv = np.stack([q["cx"] + q["f"] * lon, self.p["rows"][half][0] + q["cy"] + q["f"] * lat], 1)
            uv[np.abs(lon) > np.radians(120)] = np.nan
            return uv
        th = np.arctan2(np.hypot(c[:, 0], c[:, 1]), c[:, 2])
        ph = np.arctan2(c[:, 1], c[:, 0])
        r = q["f"] * (th + q["k1"] * th**3 + q["k2"] * th**5)
        uv = np.stack([q["cx"] + r * np.cos(ph), self.p["rows"][half][0] + q["cy"] + r * np.sin(ph)], 1)
        uv[th > np.radians(100)] = np.nan
        return uv

    def half_of(self, v: float) -> str:
        return "top" if v < self.p["rows"]["top"][1] else "bottom"


def _val(expr: str, L: float, W: float) -> float | None:
    if expr == "*":
        return None
    return float(eval(expr, {"__builtins__": {}}, {"L": L, "W": W}))  # noqa: S307 - tiny trusted calibration expressions


def fit(calib: dict, image_size: tuple[int, int] = (2048, 2048)) -> tuple[PanoramaCamera, dict]:
    rows = {h: calib["halves"][h]["rows"] for h in HALVES}
    w, _ = image_size
    L0, W0 = 100.0, 64.0
    points = [(o["half"], np.array(o["px"], float), o["pitch"]) for o in calib["points"]]
    lines = [(ln["half"], np.array(ln["px"], float), ln["pitch"]) for ln in calib["lines"]]
    t = np.linspace(-0.1, 1.1, 400)

    def residuals(v: np.ndarray, model: str) -> np.ndarray:
        """Pixel-space residuals: projecting the model can't cheat by shrinking the world."""
        p = PanoramaCamera.unpack(v, rows, model)
        cam = PanoramaCamera(p)
        L, W = p["L"], p["W"]
        res = []
        for half, px, pitch in points:
            uv = cam.to_pixels(half, np.array([[_val(pitch[0], L, W), _val(pitch[1], L, W), 0.0]]))[0]
            res += list(uv - px) if np.all(np.isfinite(uv)) else [500.0, 500.0]
        for half, pxs, pitch in lines:
            xs, ys = _val(pitch[0], L, W), _val(pitch[1], L, W)
            line = np.stack([t * L if xs is None else np.full_like(t, xs), t * W if ys is None else np.full_like(t, ys), 0 * t], 1)
            uv = cam.to_pixels(half, line)
            d = np.linalg.norm(uv[None, :, :] - pxs[:, None, :], axis=2)
            res += list(np.where(np.isfinite(d).any(1), np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1), 500.0))
        res += [20 * p[h][k] for h in HALVES for k in ("k1", "k2")]
        return np.array(res)

    best, best_model = None, None
    for model, cy_, cz_ in (("fisheye", -12, 9), ("equirect", -8, 6), ("equirect", -12, 9), ("equirect", -20, 14)):
        init = {"C": [L0 / 2, cy_, cz_], "L": L0, "W": W0}
        for h, tx in (("top", 0.25), ("bottom", 0.75)):
            R = _look_at(np.array(init["C"]), np.array([L0 * tx, W0 * 0.5, 0.0]))
            init[h] = {"rotvec": Rotation.from_matrix(R).as_rotvec().tolist(), "f": 650.0 if model == "fisheye" else 2048 / np.pi, "k1": 0.0, "k2": 0.0, "cx": w / 2, "cy": (rows[h][1] - rows[h][0]) / 2}
        v0 = PanoramaCamera.pack(init)
        lo = np.full_like(v0, -np.inf)
        hi = np.full_like(v0, np.inf)
        lo[2], hi[2] = 2, 30  # camera height
        lo[3], hi[3] = 80, 130  # pitch length
        lo[4], hi[4] = 45, 90  # pitch width
        sol = least_squares(residuals, v0, args=(model,), bounds=(lo, hi), loss="soft_l1", f_scale=5.0, x_scale="jac", max_nfev=1500)
        print(model, cy_, cz_, "rms", float(np.sqrt(np.mean(residuals(sol.x, model)[:-4] ** 2))), file=sys.stderr)
        if best is None or sol.cost < best.cost:
            best, best_model = sol, model
    p = PanoramaCamera.unpack(best.x, rows, best_model)
    r = residuals(best.x, best_model)[:-4]
    report = {"model": best_model, "rms_px": float(np.sqrt(np.mean(r**2))), "max_px": float(np.abs(r).max()), "n_residuals": int(len(r)), "L": p["L"], "W": p["W"], "camera": p["C"]}
    return PanoramaCamera(p), report


def overlay(cam: PanoramaCamera, image, color=(0, 255, 255)):
    """Draw the fitted pitch model on an image (BGR numpy array) for visual checking."""
    import cv2

    L, W = cam.p["L"], cam.p["W"]
    segs = [((0, 0), (L, 0)), ((0, W), (L, W)), ((0, 0), (0, W)), ((L, 0), (L, W)), ((L / 2, 0), (L / 2, W))]
    for s in (0, 1):
        x0 = 0 if s == 0 else L
        d = 1 if s == 0 else -1
        segs += [((x0, W / 2 - 20.16), (x0 + d * 16.5, W / 2 - 20.16)), ((x0 + d * 16.5, W / 2 - 20.16), (x0 + d * 16.5, W / 2 + 20.16)), ((x0 + d * 16.5, W / 2 + 20.16), (x0, W / 2 + 20.16))]
    circle = [(L / 2 + 9.15 * np.cos(a), W / 2 + 9.15 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 60)]
    segs += list(zip(circle[:-1], circle[1:], strict=True))
    out = image.copy()
    for half in HALVES:
        for a, b in segs:
            pts = np.array([[a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0] for t in np.linspace(0, 1, 40)])
            uv = cam.to_pixels(half, pts)
            r0, r1 = cam.p["rows"][half]
            for p0, p1 in zip(uv[:-1], uv[1:], strict=True):
                if np.all(np.isfinite([*p0, *p1])) and r0 <= p0[1] < r1 and r0 <= p1[1] < r1 and 0 <= p0[0] < image.shape[1]:
                    cv2.line(out, tuple(int(x) for x in p0), tuple(int(x) for x in p1), color, 2)
    return out


def main() -> None:
    calib = json.load(open(sys.argv[1]))
    import cv2

    img = cv2.imread(sys.argv[2])
    cam, report = fit(calib, (img.shape[1], img.shape[0]))
    json.dump({"params": cam.p, "report": report}, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(report, indent=1))
    if len(sys.argv) > 4:
        cv2.imwrite(sys.argv[4], overlay(cam, img))


if __name__ == "__main__":
    main()
