import numpy as np

from sim.engine import MatchSimulator
from sim.schema import GOAL_HEIGHT, JOINT_NAMES, EventType, Match
from sim.skeleton import build_skeleton


def test_deterministic():
    a = MatchSimulator(seed=11, duration_min=4).run().match.model_dump_json()
    b = MatchSimulator(seed=11, duration_min=4).run().match.model_dump_json()
    assert a == b


def test_events_ordered_and_on_pitch(bundle):
    ev = bundle.match.events
    assert [e.t for e in ev] == sorted(e.t for e in ev)
    assert len({e.id for e in ev}) == len(ev)
    for e in ev:
        if e.x is not None and e.type != EventType.GOAL:
            assert -2 <= e.x <= 107 and -2 <= e.y <= 70, e


def test_plausible_counts(bundle):
    types = [e.type for e in bundle.match.events]
    passes = [e for e in bundle.match.events if e.type == EventType.PASS]
    assert len(passes) > 200
    rate = sum(p.outcome == "complete" for p in passes) / len(passes)
    assert 0.6 < rate < 0.95
    assert EventType.SHOT in types and EventType.PRESSURE in types and EventType.TACKLE in types
    goals = [e for e in bundle.match.events if e.type == EventType.GOAL]
    assert bundle.match.meta.final_score.home + bundle.match.meta.final_score.away == len(goals)


def test_every_shot_has_a_window(bundle):
    shots = [e.id for e in bundle.match.events if e.type == EventType.SHOT]
    covered = {i for w in bundle.match.windows for i in w.trigger_event_ids}
    assert set(shots) == covered
    for w in bundle.windows.values():
        assert len(w.players) == w.n_frames == len(w.ball)
        assert np.asarray(w.players).shape[1:] == (22, 2)


def test_tracking_consistent_with_shot(bundle):
    """Ball in the tracking window must be at the shooter's feet at the shot time."""
    m = bundle.match
    for ref in m.windows[:3]:
        w = bundle.windows[ref.id]
        shot = next(e for e in m.events if e.id == ref.trigger_event_ids[0])
        f = int(round((shot.t - w.t_start) * w.fps))
        ball = np.asarray(w.ball[f])
        assert np.hypot(ball[0] - shot.x, ball[1] - shot.y) < 1.5
        assert ball[2] <= GOAL_HEIGHT + 2


def test_skeleton_shape_and_bones(bundle):
    w = next(iter(bundle.windows.values()))
    sk = build_skeleton(w)
    a = np.asarray(sk.frames)
    assert a.shape == (w.n_frames, 22, len(JOINT_NAMES), 3)
    assert a[..., 2].min() >= 0
    thigh = np.linalg.norm(a[:, :, 9] - a[:, :, 11], axis=-1)  # l_hip -> l_knee
    assert np.allclose(thigh, thigh.mean(), atol=0.06)  # rigid bones (rounding + height scale only)


def test_schema_roundtrip(bundle):
    m2 = Match.model_validate_json(bundle.match.model_dump_json())
    assert m2.meta.match_id == bundle.match.meta.match_id
    assert m2.attack_direction(m2.meta.home_team_id, 1) == 1
    assert m2.attack_direction(m2.meta.home_team_id, 2) == -1


def test_movement_calibrated_to_real_footage():
    """Aggregate targets measured on real high-school footage (vision/stats.py):
    median speed ~1.95 m/s, p99 ~7.45 m/s, ~1.4% of time above 7 m/s, team width ~26 m."""
    from vision.stats import movement_stats, shape_stats

    sim = MatchSimulator(seed=5, duration_min=20)
    sim.run()
    st = movement_stats(sim.hist_pos[::5].astype(float), 5)
    assert 1.5 < st["speed_p50_mps"] < 2.8
    assert 6.0 < st["speed_p99_mps"] < 9.0
    assert 0.003 < st["share_sprint_gt7"] < 0.05
    sh = shape_stats(sim.hist_pos[::25].astype(float), {0: list(range(1, 11)), 1: list(range(12, 22))})
    assert 20 < sh["team_width_p10_p90_m"] < 34
