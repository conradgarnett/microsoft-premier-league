import numpy as np

from analytics import full_report
from analytics.body import angle_at, joint_angles
from analytics.core import Tracking, pass_metrics, physical, xg
from sim.schema import EventType


def test_xg_geometry():
    assert xg(94, 34) > xg(80, 34) > xg(70, 34)  # closer is better
    assert xg(94, 34) > xg(94, 10)  # central beats tight angle
    assert 0.12 < xg(94, 34) < 0.35  # penalty-spot range
    assert xg(94, 34, under_pressure=True) < xg(94, 34)


def test_angle_helper():
    a, b, c = np.array([0, 1.0, 0]), np.zeros(3), np.array([1.0, 0, 0])
    assert abs(angle_at(a, b, c) - 90) < 1e-6
    straight = np.zeros((17, 3))
    straight[:, 2] = np.linspace(1.7, 0, 17)
    ang = joint_angles(straight)
    assert ang["trunk_lean_deg"] < 1


def test_report_consistency(bundle):
    m = bundle.match
    r = full_report(m, bundle.lowres)
    teams = r["teams"]
    assert abs(sum(t["possession_pct"] for t in teams.values()) - 100) < 0.5
    assert abs(sum(t["field_tilt_pct"] for t in teams.values()) - 100) < 0.5
    for t in teams.values():
        assert t["ppda"] > 0 and 0 < t["defensive_line_m"] < 60
    assert all(0 <= b["chaos_index"] <= 100 for b in r["rhythm"])
    goals = sum(e.type == EventType.GOAL for e in m.events)
    assert sum(k["kind"] == "goal" for k in r["key_moments"]) == goals
    ranks = sorted(k["rank"] for k in r["key_moments"])
    assert ranks == list(range(1, len(ranks) + 1))
    # every key moment's evidence ids exist
    ids = {e.id for e in m.events}
    assert all(set(k["evidence"]) <= ids for k in r["key_moments"])


def test_pass_difficulty_orders_sensibly(bundle):
    m = bundle.match
    trk = Tracking(m, bundle.lowres)
    pm = [pass_metrics(m, e, trk) for e in m.events if e.type == EventType.PASS]
    d = np.array([p["difficulty"] for p in pm])
    ok = np.array([p["complete"] for p in pm])
    assert 0 <= d.min() and d.max() <= 1
    assert d[~ok].mean() > d[ok].mean()  # failed passes were harder on average


def test_physical(bundle):
    phys = physical(Tracking(bundle.match, bundle.lowres))
    km_per_90 = np.array([v["distance_km"] for v in phys.values()]) * 90 / (bundle.match.meta.duration_s / 60)
    assert 4 < np.median(km_per_90) < 16  # real high-school footage: ~12.6 km/90 incl. detector jitter
    assert max(v["top_speed_kmh"] for v in phys.values()) < 40
