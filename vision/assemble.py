"""Stage 2: detections -> tracks -> teams -> schema bundle (Match + tracking + skeletons).

    python -m vision.assemble <extract.npz> <camera.json> <out_matches_dir> [--match-id real-clip]

The result is a match folder in the same layout ``sim`` writes, so the API and the 3D
viewer replay real-footage clips exactly like synthetic ones (point ``DATA_DIR`` at the
output directory). Players are anonymised ("A7", "B3"); no names leave the footage.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from sim.schema import (
    BONES,
    JOINT_NAMES,
    Event,
    EventType,
    LowResTracking,
    Match,
    MatchMeta,
    Player,
    Position,
    Score,
    SkeletonWindow,
    Team,
    TrackingWindow,
    WindowRef,
)
from sim.skeleton import build_skeleton

from .calibrate import PanoramaCamera
from .events import detect_events
from .pose import coco_to_schema, lift, rigidify
from .track import PitchTracker

OUT_FPS = 25.0


def kmeans2(x: np.ndarray, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Two-cluster k-means (deterministic init: the two most distant points)."""
    d = np.linalg.norm(x[:, None] - x[None], axis=2)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    c = x[[i, j]].astype(float)
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(x[:, None] - c[None], axis=2), axis=1)
        c = np.stack([x[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(2)])
    return lab, c


def resample(frames: np.ndarray, values: np.ndarray, n_out: int, src_fps: float, max_gap_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate per-frame samples onto the output grid; returns (values, observed mask)."""
    t_src = frames / src_fps
    t_out = np.arange(n_out) / OUT_FPS
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(t_out, t_src, flat[:, k]) for k in range(flat.shape[1])], 1).reshape(n_out, *values.shape[1:])
    idx = np.searchsorted(t_src, t_out)
    prev = np.clip(idx - 1, 0, len(t_src) - 1)
    nxt = np.clip(idx, 0, len(t_src) - 1)
    gap = np.minimum(np.abs(t_out - t_src[prev]), np.abs(t_src[nxt] - t_out))
    return out, gap <= max_gap_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("extract")
    ap.add_argument("camera")
    ap.add_argument("out")
    ap.add_argument("--match-id", default="real-clip")
    ap.add_argument("--min-coverage", type=float, default=0.3)
    ap.add_argument("--motion", choices=["lift", "mocap"], default="lift", help="skeletons: single-view pose lift, or CMU mocap driven by the video tracks and detected kicks")
    ap.add_argument("--ball", help="output of vision.ball (replaces the generic detector's ball)")
    a = ap.parse_args()
    d = dict(np.load(a.extract))  # NpzFile re-reads an array on every access
    meta = json.loads(str(d["meta"]))
    cam = PanoramaCamera(json.load(open(a.camera))["params"])
    fps = meta["fps"]
    n_src = meta["frames"]
    n_out = int(round(meta["duration"] * OUT_FPS))

    # 1) de-duplicate the two halves' overlap (same person seen in both) then track on the pitch
    tracker = PitchTracker(dt=1 / fps, max_misses=int(3 * fps))
    for f in range(n_src):
        idx = np.where(d["frame"] == f)[0]
        keep = []
        for i in idx[np.argsort(-d["score"][idx])]:
            if all(np.linalg.norm(d["xy"][i] - d["xy"][k]) > 0.9 for k in keep):
                keep.append(i)
        keep = np.array(keep, int)
        tracker.update(f, d["xy"][keep].reshape(-1, 2), [(int(i), d["box"][i], d["colour"][i]) for i in keep])
    tracks = [t for t in tracker.finished(min_hits=int(fps * 2)) if len(t.history) / n_src >= a.min_coverage]
    print(f"{len(tracker.finished(1))} raw tracks -> {len(tracks)} with >= {a.min_coverage:.0%} coverage")

    # 2) teams from torso colour (outliers = referees / keepers -> 'other')
    cols = np.array([np.median(np.array(t.colors), 0) for t in tracks])
    feat = cols * [0.5, 1.0, 1.0]  # half-weight lightness: white vs black kits differ mostly in L
    lab, centers = kmeans2(feat)
    dist = np.linalg.norm(feat - centers[lab], axis=1)
    other = dist > max(12.0, 2.5 * np.median(dist))
    team_of = np.where(other, 2, lab)

    teams = []
    names = ["Team A", "Team B", "Officials / keepers"]
    shorts = ["A", "B", "O"]
    colours = ["#e8e8e8", "#3a6df0", "#f5c542"]
    for k in range(3):
        c = centers[k] if k < 2 else None
        if c is not None:
            import cv2

            # display colour: keep the kit's hue, stretch lightness apart (shade turns white kits grey)
            light = 235 if c[0] >= centers[1 - k][0] else 45
            rgb = cv2.cvtColor(np.uint8([[[light, c[1], c[2]]]]), cv2.COLOR_LAB2RGB)[0, 0]  # OpenCV 8-bit Lab
            colours[k] = "#" + "".join(f"{int(v):02x}" for v in rgb)
        teams.append(Team(id=f"real-{shorts[k].lower()}", name=names[k], short_name=shorts[k], color_primary=colours[k], color_secondary="#202020", formation="unknown", players=[]))
    pids = []
    for n, k in enumerate(team_of):
        pid = f"{shorts[k].lower()}{n:02d}"
        pids.append(pid)
        teams[k].players.append(Player(id=pid, name=f"{shorts[k]}{n}", short_name=f"{shorts[k]}{n}", number=n, position=Position.CM, team_id=teams[k].id))

    # 3) resample positions + lifted skeletons to 25 fps
    P = len(tracks)
    pos = np.zeros((n_out, P, 2))
    joints = np.zeros((n_out, P, 17, 3))
    conf = np.zeros((n_out, P, 17))
    cam_xy = np.asarray(cam.p["C"][:2])
    for p, t in enumerate(tracks):
        fr = np.array(sorted(t.history))
        xy = np.array([t.history[f] for f in fr])
        # light smoothing of measured ground positions (detector box jitter)
        k = np.ones(3) / 3
        xy_s = np.stack([np.convolve(np.pad(xy[:, i], 1, mode="edge"), k, mode="valid") for i in range(2)], 1)
        pos[:, p], seen = resample(fr, xy_s, n_out, fps)
        sk, cf = [], []
        for f, g in zip(fr, xy_s, strict=True):
            di = t.boxes[f][0]  # detection row index
            jp, jc = coco_to_schema(d["kp"][di][None], d["kps"][di][None])
            sk.append(lift(jp[0], g, cam_xy))
            cf.append(jc[0])
        joints[:, p], _ = resample(fr, np.array(sk), n_out, fps)
        # facing from the smoothed track (held while slow), then rigid bones with recovered depth
        vel = np.gradient(pos[:, p], 1 / OUT_FPS, axis=0)
        vel = np.stack([np.convolve(np.pad(vel[:, i], 12, mode="edge"), np.ones(25) / 25, mode="valid") for i in range(2)], 1)
        facing = np.zeros_like(vel)
        cur = np.array([0.0, 1.0])
        for f in range(n_out):
            sp = np.linalg.norm(vel[f])
            cur = vel[f] / sp if sp > 1.0 else cur
            facing[f] = cur
        rj = rigidify(joints[:, p], cam_xy, facing, scale=1.75 / 1.8)
        rj[..., 2] -= (rj[:, 13:17, 2].min(axis=1) - 0.03)[:, None]  # lowest foot on the grass
        joints[:, p] = rj
        joints[~seen, p, :, 2] = -5.0  # not observed: park the figure under the grass instead of inventing motion
        cnf, _ = resample(fr, np.array(cf), n_out, fps)
        conf[:, p] = cnf * seen[:, None]

    # ball: unknown frames stay below the grass (z=-5) rather than inventing a position
    ball = np.tile([cam.p["L"] / 2, cam.p["W"] / 2, -5.0], (n_out, 1))
    if a.ball:
        bz = np.load(a.ball)
        good = bz["conf"] > 0.5
        if good.sum() >= 2:
            b, seen = resample(np.round(bz["t"][good] * OUT_FPS), bz["xy"][good], n_out, OUT_FPS, max_gap_s=0.6)
            ball[seen, :2] = b[seen]
            ball[seen, 2] = 0.11
    elif len(d["ball"]):
        bd = d["ball"]  # generic detector: fires on boots, only a fallback
        good = bd[:, 5] > 0.4
        if good.sum() >= 2:
            b, seen = resample(bd[good, 0], bd[good, 6:8], n_out, fps, max_gap_s=0.5)
            ball[seen, :2] = b[seen]
            ball[seen, 2] = 0.11

    L, W = cam.p["L"], cam.p["W"]
    # rescale to the schema's 105 x 68 pitch so the viewer's pitch lines match
    sx, sy = 105.0 / L, 68.0 / W
    pos *= [sx, sy]
    joints[..., 0] *= sx
    joints[..., 1] *= sy
    ball[:, 0] *= sx
    ball[:, 1] *= sy

    wid = "w000"
    duration = n_out / OUT_FPS
    win = TrackingWindow(id=wid, source="video", t_start=0.0, fps=OUT_FPS, n_frames=n_out, player_ids=pids, ball=np.round(ball, 2).tolist(), players=np.round(pos, 2).tolist(), actions=[], event_ids=[])
    sk = SkeletonWindow(id=f"sk-{wid}", window_id=wid, source="video", t_start=0.0, fps=OUT_FPS, n_frames=n_out, joint_names=list(JOINT_NAMES), bones=list(BONES), player_ids=pids, frames=np.round(joints, 3).tolist(), confidence=np.round(conf, 2).tolist())
    team_ids = {pid: teams[k].id for pid, k in zip(pids, team_of, strict=True)}
    in_play = (joints[:, :, 0, 2] > -1) & (np.asarray(team_of) < 2)[None]  # officials never control the ball
    ev = [Event(id="ev00000", t=0.0, period=1, minute=0, second=0, type=EventType.HALF_START, team_id=teams[0].id)]
    ev += detect_events(0.0, OUT_FPS, ball, pos, pids, team_ids, known=ball[:, 2] > -1, visible=in_play, control_m=2.0)
    win.event_ids = [e.id for e in ev]
    win.actions = [{"t": e.t, "player_id": e.player_id, "kind": "kick", "power": 1.0 if e.type == EventType.SHOT else min(e.attributes.get("length_m", 15) / 35, 1), "shot": e.type == EventType.SHOT, "event_id": e.id} for e in ev if e.type in (EventType.PASS, EventType.SHOT)]
    if a.motion == "mocap":
        # human motion from the mocap library, placed on the real tracks; unobserved players stay hidden
        mj = np.asarray(build_skeleton(win).frames)
        mj[joints[:, :, 0, 2] < -1, :, 2] = -5.0
        sk = SkeletonWindow(id=f"sk-{wid}", window_id=wid, source="video-mocap", t_start=0.0, fps=OUT_FPS, n_frames=n_out, player_ids=pids, frames=np.round(mj, 3).tolist(), confidence=None)
    goals = [e.team_id for e in ev if e.type == EventType.GOAL]
    m = Match(
        meta=MatchMeta(match_id=a.match_id, source="video", duration_s=duration, half_length_s=duration, home_team_id=teams[0].id, away_team_id=teams[1].id, final_score=Score(home=goals.count(teams[0].id), away=goals.count(teams[1].id))),
        teams=teams,
        events=ev,
        windows=[WindowRef(id=wid, t_start=0.0, t_end=duration, trigger_event_ids=[])],
    )
    step = int(OUT_FPS // 2)
    low = LowResTracking(source="video", fps=OUT_FPS / step, n_frames=len(range(0, n_out, step)), player_ids=pids, ball=np.round(ball[::step], 2).tolist(), players=np.round(pos[::step], 1).tolist())
    out = pathlib.Path(a.out) / a.match_id
    (out / "windows").mkdir(parents=True, exist_ok=True)
    (out / "skeletons").mkdir(exist_ok=True)
    (out / "match.json").write_text(m.model_dump_json())
    (out / "lowres.json").write_text(low.model_dump_json())
    (out / "windows" / f"{wid}.json").write_text(win.model_dump_json())
    (out / "skeletons" / f"{wid}.json").write_text(sk.model_dump_json())
    print(f"wrote {out}: {P} tracks ({[len(t.players) for t in teams]} per group), {duration:.1f}s, ball frames {(ball[:, 2] > 0).sum()}, events {len(ev) - 1}")


if __name__ == "__main__":
    main()
