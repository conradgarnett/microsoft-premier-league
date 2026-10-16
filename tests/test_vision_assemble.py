"""vision.assemble end to end on a fake extract: output must be a valid sim-layout match."""

import json
import sys

import numpy as np
import pytest

from sim.schema import Match, SkeletonWindow, TrackingWindow

CAM = {
    "model": "equirect", "C": [50.0, -20.0, 12.0], "L": 100.0, "W": 64.0,
    "rows": {"top": [0, 1024], "bottom": [1024, 2048]},
    "top": {"rotvec": [-1.7, -0.6, 0.4], "f": 652.0, "k1": 0.05, "k2": 0.0, "cx": 1024.0, "cy": 512.0},
    "bottom": {"rotvec": [-1.8, 0.6, -0.45], "f": 652.0, "k1": 0.05, "k2": 0.0, "cx": 1024.0, "cy": 512.0},
}  # fmt: skip


@pytest.mark.parametrize("motion", ["lift", "mocap"])
def test_assemble_writes_valid_match(tmp_path, monkeypatch, motion):
    from vision import assemble

    fps, n = 12.5, 50  # 4 s
    start = np.array([[30.0, 20.0], [40.0, 30.0], [55.0, 25.0], [60.0, 40.0]])
    colours = np.array([[230, 128, 128], [230, 128, 128], [30, 128, 128], [30, 128, 128]], float)  # white vs black kits (Lab)
    rows = {k: [] for k in ("frame", "t", "half", "box", "score", "xy", "colour", "kp", "kps")}
    for f in range(n):
        for p in range(4):
            xy = start[p] + [0.15 * f, 0.05 * f * (-1) ** p]
            box = np.array([1000.0 + 40 * p, 400.0, 1020.0 + 40 * p, 460.0])
            kp = np.stack([np.full(17, 1010.0 + 40 * p), np.linspace(402, 458, 17)], 1)
            for k, v in zip(rows, (f, f / fps, 0, box, 0.9, xy, colours[p], kp, np.full(17, 0.8)), strict=True):
                rows[k].append(v)
    meta = {"start": 0.0, "duration": n / fps, "fps": fps, "frames": n, "camera": "cam.json"}
    np.savez(tmp_path / "ex.npz", **{k: np.array(v) for k, v in rows.items()}, ball=np.zeros((0, 8)), meta=np.array(json.dumps(meta)))
    (tmp_path / "cam.json").write_text(json.dumps({"params": CAM}))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["assemble", str(tmp_path / "ex.npz"), str(tmp_path / "cam.json"), str(out), "--motion", motion])
    assemble.main()

    d = out / "real-clip"
    m = Match.model_validate_json((d / "match.json").read_text())
    w = TrackingWindow.model_validate_json((d / "windows" / "w000.json").read_text())
    sk = SkeletonWindow.model_validate_json((d / "skeletons" / "w000.json").read_text())
    assert m.meta.source == "video" and w.n_frames == 100 and len(w.player_ids) == 4
    assert sorted(len(t.players) for t in m.teams[:2]) == [2, 2]  # kits split into two teams
    frames = np.array(sk.frames)
    assert frames.shape == (100, 4, 17, 3)
    assert sk.source == ("video-mocap" if motion == "mocap" else "video")
    assert (frames[:, :, 0, 2] > 1.2).mean() > 0.9  # heads above the grass: standing figures
