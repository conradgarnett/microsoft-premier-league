"""2D pose with ViTPose (Apache-2.0, Hugging Face transformers) and a simple 3D lift.

The lift is deliberately simple and honest about it: a single wide-angle camera gives no
depth per joint, so joints are placed on a vertical plane through the player's ground
position, perpendicular to the camera ray, scaled by the player's pixel height.
"""

from __future__ import annotations

import numpy as np

MODEL = "usyd-community/vitpose-base-simple"
COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
C = {n: i for i, n in enumerate(COCO)}


class PoseEstimator:
    def __init__(self, model: str = MODEL, device: str | None = None) -> None:
        import torch  # heavy: only the extractor needs it; coco_to_schema/lift stay numpy-only
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = VitPoseForPoseEstimation.from_pretrained(model).to(self.device).eval()

    def __call__(self, image: np.ndarray, boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> keypoints (N,17,2) in image px and scores (N,17)."""
        with self.torch.no_grad():
            return self._run(image, boxes_xyxy)

    def _run(self, image: np.ndarray, boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes_xyxy) == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))
        b = boxes_xyxy.astype(float).copy()
        # pad boxes: small, tight detections cut off feet and hands
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        b[:, 0] -= 0.15 * w
        b[:, 2] += 0.15 * w
        b[:, 1] -= 0.08 * h
        b[:, 3] += 0.05 * h
        coco = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
        inp = self.proc(image, boxes=[coco.tolist()], return_tensors="pt").to(self.device)
        out = self.model(**inp)
        res = self.proc.post_process_pose_estimation(out, boxes=[coco.tolist()])[0]
        kp = np.stack([r["keypoints"].cpu().numpy() for r in res])
        sc = np.stack([r["scores"].cpu().numpy() for r in res])
        return kp, sc


def coco_to_schema(kp: np.ndarray, sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COCO-17 (N,17,2) -> schema-17 joint order (N,17,2) + confidence. Toes are not observed."""
    mid = lambda a, b: (kp[:, C[a]] + kp[:, C[b]]) / 2  # noqa: E731
    smid = lambda a, b: np.minimum(sc[:, C[a]], sc[:, C[b]])  # noqa: E731
    joints = [
        mid("l_ear", "r_ear") * 0.5 + kp[:, C["nose"]] * 0.5,  # head
        mid("l_shoulder", "r_shoulder"),  # neck
        kp[:, C["l_shoulder"]], kp[:, C["r_shoulder"]], kp[:, C["l_elbow"]], kp[:, C["r_elbow"]],
        kp[:, C["l_wrist"]], kp[:, C["r_wrist"]],
        mid("l_hip", "r_hip"),  # pelvis
        kp[:, C["l_hip"]], kp[:, C["r_hip"]], kp[:, C["l_knee"]], kp[:, C["r_knee"]],
        kp[:, C["l_ankle"]], kp[:, C["r_ankle"]], kp[:, C["l_ankle"]], kp[:, C["r_ankle"]],
    ]  # fmt: skip
    conf = [
        np.minimum(sc[:, C["nose"]], smid("l_ear", "r_ear")), smid("l_shoulder", "r_shoulder"),
        sc[:, C["l_shoulder"]], sc[:, C["r_shoulder"]], sc[:, C["l_elbow"]], sc[:, C["r_elbow"]],
        sc[:, C["l_wrist"]], sc[:, C["r_wrist"]], smid("l_hip", "r_hip"),
        sc[:, C["l_hip"]], sc[:, C["r_hip"]], sc[:, C["l_knee"]], sc[:, C["r_knee"]],
        sc[:, C["l_ankle"]], sc[:, C["r_ankle"]], sc[:, C["l_ankle"]] * 0.3, sc[:, C["r_ankle"]] * 0.3,
    ]  # fmt: skip
    return np.stack(joints, 1), np.stack(conf, 1)


def lift(joints_px: np.ndarray, ground_xy: np.ndarray, cam_xy: np.ndarray, height_m: float = 1.75) -> np.ndarray:
    """(17,2) image joints -> (17,3) pitch metres on a camera-facing vertical plane at ground_xy."""
    feet = joints_px[[13, 14]].mean(0)
    top = joints_px[0]
    px_h = max(feet[1] - top[1], 1.0) / 0.93  # head joint sits ~7% below the crown
    s = height_m / px_h
    ray = ground_xy - cam_xy
    ray /= np.linalg.norm(ray) + 1e-9
    right = np.array([-ray[1], ray[0]])  # image-right on the ground plane, seen from the camera
    out = np.zeros((17, 3))
    dx = (joints_px[:, 0] - feet[0]) * s
    dz = (feet[1] - joints_px[:, 1]) * s
    out[:, :2] = ground_xy + dx[:, None] * right[None]
    out[:, 2] = np.maximum(dz + 0.07, 0)
    return out


def rigidify(joints: np.ndarray, cam_xy: np.ndarray, facing: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Planar lift (F,17,3) -> rigid 3D skeleton with recovered depth (Taylor's method).

    A bone that looks shorter than its true length (CMU mocap lengths, scaled) is tilted
    out of the image plane by sqrt(L^2 - l^2) along the camera ray. The sign comes from the
    body: knees, toes, elbows, hands and a leaning trunk go towards the facing direction, shins trail the knee, left joints of the
    hips and shoulders towards the body's left. Bones that look too long are shortened.
    """
    from sim.schema import J
    from sim.skeleton import CHAIN, library

    lengths = library()["bone_len"]
    out = joints.copy()
    ray = joints[:, J["pelvis"], :2] - cam_xy
    ray /= np.linalg.norm(ray, axis=1, keepdims=True) + 1e-9
    ray3 = np.concatenate([ray, np.zeros((len(ray), 1))], 1)
    left = np.stack([-facing[:, 1], facing[:, 0]], 1)
    fwd_sign = np.sign((facing * ray).sum(1))
    left_sign = np.sign((left * ray).sum(1))
    for a, b in CHAIN:
        d = joints[:, J[b]] - joints[:, J[a]]  # in-plane bone as observed
        L = lengths[(a, b)] * scale
        n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        d = np.where(n > L, d / n * L, d)
        if a in ("pelvis", "neck") and b not in ("head", "neck"):
            sign = left_sign * (1 if b.startswith("l_") else -1)
        elif a.endswith("knee"):
            sign = -fwd_sign  # shins trail the knee in a stride
        else:
            sign = fwd_sign
        depth = np.sqrt(np.maximum(L**2 - (d**2).sum(1), 0))
        out[:, J[b]] = out[:, J[a]] + d + (sign * depth)[:, None] * ray3
    return out
