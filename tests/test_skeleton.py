"""Validation of mocap-driven skeletons: human joint limits, rigid bones, planted feet, no teleports."""

import numpy as np
import pytest

from analytics.body import joint_angles
from sim.schema import J, TrackingWindow
from sim.skeleton import CHAIN, build_skeleton

FPS = 25


def straight_run(speed: float, seconds: float = 4.0) -> TrackingWindow:
    n = int(seconds * FPS)
    x = 20 + speed * np.arange(n) / FPS
    players = [[[float(xi), 34.0]] for xi in x]
    ball = [[float(xi) + 30, 34.0, 0.1] for xi in x]  # ball far ahead: player faces his run
    return TrackingWindow(id="test", t_start=0, fps=FPS, n_frames=n, player_ids=["p1"], ball=ball, players=players)


def frames(win: TrackingWindow) -> np.ndarray:
    return np.asarray(build_skeleton(win).frames)


@pytest.mark.parametrize("speed", [1.2, 3.0, 4.5, 7.5])
def test_feet_planted_during_stance(speed):
    a = frames(straight_run(speed))[10:-10, 0]
    for foot in ("l_ankle", "r_ankle"):
        v = np.linalg.norm(np.diff(a[:, J[foot], :2], axis=0), axis=1) * FPS
        # each stride has a stance phase where the foot is (nearly) still on the ground
        assert np.percentile(v, 10) < 0.25 * speed + 0.3, (foot, speed, np.percentile(v, 10))
        assert v.max() > 1.5 * speed  # and a swing phase faster than the body


def test_bones_rigid_and_human_limits(bundle):
    for w in list(bundle.windows.values())[:3]:
        a = frames(w)
        for p_, c_ in CHAIN:
            L = np.linalg.norm(a[:, :, J[c_]] - a[:, :, J[p_]], axis=-1)
            assert (L.max(axis=0) - L.min(axis=0)).max() < 0.01, (p_, c_)
        ang = joint_angles(a)
        for side in "lr":
            knee = ang[f"{side}_knee_deg"]
            assert knee.min() > 25 and knee.max() <= 180.5  # knees never hyperextend or fold through
            hip = ang[f"{side}_hip_deg"]
            assert hip.min() > 40
        assert ang["trunk_lean_deg"].max() < 95  # only dives/slides approach horizontal


def test_no_teleporting_limbs(bundle):
    w = next(iter(bundle.windows.values()))
    a = frames(w)
    step = np.linalg.norm(np.diff(a, axis=0), axis=-1)
    assert step.max() < 1.0  # < 25 m/s for any joint between frames
    assert a[..., 2].min() >= 0.0
