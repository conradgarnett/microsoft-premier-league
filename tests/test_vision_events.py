"""Video event heuristics, scored against the synthetic engine's ground-truth events."""

import numpy as np

from vision.events import detect_events


def test_events_recover_synthetic_passes_and_shots(bundle):
    team = {p.id: p.team_id for p in bundle.match.players()}
    keepers = {p.id for p in bundle.match.players() if p.position.value == "GK"}
    hit = det = gt_n = shots_hit = shots_gt = 0
    for w in bundle.windows.values():
        ev = detect_events(w.t_start, w.fps, np.array(w.ball), np.array(w.players), w.player_ids, team, keepers=keepers)
        t0, t1 = w.t_start + 0.3, w.t_start + w.n_frames / w.fps - 0.5
        gt = [e for e in bundle.match.events if t0 <= e.t <= t1]
        for kind in ("pass", "shot"):
            g = [e for e in gt if e.type.value == kind]
            d = [e for e in ev if e.type.value == kind and t0 <= e.t <= t1]
            used: set[int] = set()
            for e in d:
                k = next((i for i, x in enumerate(g) if i not in used and abs(x.t - e.t) < 0.6 and x.player_id == e.player_id), None)
                if k is not None:
                    used.add(k)
            if kind == "pass":
                hit, det, gt_n = hit + len(used), det + len(d), gt_n + len(g)
            else:
                shots_hit, shots_gt = shots_hit + len(used), shots_gt + len(g)
        assert [e.t for e in ev] == sorted(e.t for e in ev) and len({e.id for e in ev}) == len(ev)
    assert gt_n > 10 and shots_gt > 0
    assert hit / det >= 0.8 and hit / gt_n >= 0.7, (hit, det, gt_n)  # precision, recall
    assert shots_hit / shots_gt >= 0.6, (shots_hit, shots_gt)


def test_goal_detected_from_ball_crossing_line():
    fps, F = 25.0, 50
    players = np.zeros((F, 2, 2))
    players[:, 0] = [90.0, 34.0]  # shooter
    players[:, 1] = [60.0, 20.0]  # far away opponent
    ball = np.zeros((F, 3))
    ball[:10] = [90.5, 34.0, 0.11]  # at the shooter's feet
    for f in range(10, F):
        ball[f] = [min(90.5 + 25 * (f - 9) / fps, 106.0), 34.0, 0.5]
    ev = detect_events(0.0, fps, ball, players, ["a1", "b1"], {"a1": "a", "b1": "b"})
    assert [e.type.value for e in ev] == ["shot", "goal"]
    assert ev[0].outcome == "goal" and ev[0].player_id == "a1"
