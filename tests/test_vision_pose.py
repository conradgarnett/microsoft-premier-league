"""Depth recovery for single-view lifted poses, checked against synthetic mocap skeletons."""

import numpy as np

from sim.schema import J
from sim.skeleton import CHAIN, build_skeleton, library
from vision.pose import rigidify


def test_rigidify_restores_bone_lengths_and_depth(bundle):
    w = next(iter(bundle.windows.values()))
    sk = np.asarray(build_skeleton(w).frames)  # (F,P,17,3) ground truth
    cam = np.array([52.5, -20.0])
    lengths = library()["bone_len"]
    err_flat, err_rigid = [], []
    for p in range(sk.shape[1]):
        gt = sk[:, p]
        pel = gt[:, J["pelvis"]]
        ray = pel[:, :2] - cam
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        ray3 = np.concatenate([ray, np.zeros((len(ray), 1))], 1)
        off = gt - pel[:, None]
        flat = gt - (off * ray3[:, None]).sum(-1, keepdims=True) * ray3[:, None]  # what one camera sees
        hips = gt[:, J["l_hip"], :2] - gt[:, J["r_hip"], :2]
        facing = np.stack([hips[:, 1], -hips[:, 0]], 1)  # left-hip vector rotated -90 degrees
        facing /= np.linalg.norm(facing, axis=1, keepdims=True) + 1e-9
        scale = np.linalg.norm(gt[:, J["l_hip"]] - gt[:, J["l_knee"]], axis=1).mean() / lengths[("l_hip", "l_knee")]
        rig = rigidify(flat, cam, facing, scale)
        for a, b in CHAIN:
            got = np.linalg.norm(rig[:, J[b]] - rig[:, J[a]], axis=1)
            assert np.allclose(got, lengths[(a, b)] * scale, rtol=0.02, atol=0.02), (a, b)
        err_flat.append(np.linalg.norm(flat - gt, axis=-1).mean())
        err_rigid.append(np.linalg.norm(rig - gt, axis=-1).mean())
    assert np.mean(err_rigid) < 0.8 * np.mean(err_flat), (np.mean(err_flat), np.mean(err_rigid))
