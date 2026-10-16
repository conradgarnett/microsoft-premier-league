"""Aggregate movement statistics from a schema match folder (real or synthetic).

    python -m vision.stats <match_dir>

Used to calibrate the synthetic engine against real footage: only aggregate numbers
(speed distribution, distance rate, sprint share, team shape) are printed - nothing per player.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np


def movement_stats(pos: np.ndarray, fps: float, visible: np.ndarray | None = None) -> dict[str, float]:
    """pos (F,P,2) metres. Speeds from 1 s displacements (robust to detector jitter)."""
    k = int(round(fps))
    v = np.linalg.norm(pos[k:] - pos[:-k], axis=-1)  # metres per second
    if visible is not None:
        v = v[visible[k:] & visible[:-k]]
    v = v.ravel()
    return {
        "speed_mean_mps": round(float(v.mean()), 2),
        "speed_p50_mps": round(float(np.percentile(v, 50)), 2),
        "speed_p90_mps": round(float(np.percentile(v, 90)), 2),
        "speed_p99_mps": round(float(np.percentile(v, 99)), 2),
        "share_walk_lt2": round(float((v < 2).mean()), 3),
        "share_run_4to7": round(float(((v >= 4) & (v < 7)).mean()), 3),
        "share_sprint_gt7": round(float((v >= 7).mean()), 3),
        "distance_m_per_min": round(float(v.mean() * 60), 1),
    }


def shape_stats(pos: np.ndarray, team_idx: dict[str, list[int]], visible: np.ndarray | None = None) -> dict[str, float]:
    """Team length/width (10th-90th percentile spread) and nearest-teammate distance."""
    out = {}
    lens, wids, nn = [], [], []
    for idx in team_idx.values():
        if len(idx) < 5:
            continue
        for f in range(0, len(pos), 5):
            pts = pos[f, idx]
            if visible is not None:
                pts = pts[visible[f, idx]]
            if len(pts) < 5:
                continue
            lens.append(np.subtract(*np.percentile(pts[:, 0], [90, 10])))
            wids.append(np.subtract(*np.percentile(pts[:, 1], [90, 10])))
            d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 1e9
            nn.append(d.min(1).mean())
    out["team_length_p10_p90_m"] = round(float(np.mean(lens)), 1)
    out["team_width_p10_p90_m"] = round(float(np.mean(wids)), 1)
    out["nearest_teammate_m"] = round(float(np.mean(nn)), 1)
    return out


def main() -> None:
    d = pathlib.Path(sys.argv[1])
    m = json.loads((d / "match.json").read_text())
    w = json.loads(next((d / "windows").glob("*.json")).read_text())
    sk = json.loads((d / "skeletons" / f"{w['id']}.json").read_text())
    pos = np.asarray(w["players"], float)
    vis = np.asarray(sk["frames"], float)[:, :, 8, 2] > 0 if m["meta"]["source"] == "video" else None
    idx = {t["id"]: [w["player_ids"].index(p["id"]) for p in t["players"] if p["id"] in w["player_ids"]] for t in m["teams"]}
    idx = {k: v for k, v in idx.items() if not k.endswith("-o")}
    print(json.dumps({**movement_stats(pos, w["fps"], vis), **shape_stats(pos, idx, vis)}, indent=1))


if __name__ == "__main__":
    main()
