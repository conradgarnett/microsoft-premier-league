"""Minimal ASF/AMC reader and forward kinematics for the CMU mocap database."""

from __future__ import annotations

import numpy as np

INCH_PER_UNIT = 1 / 0.45  # CMU ASF "length 0.45": one ASF unit = 1/0.45 inches
M_PER_UNIT = INCH_PER_UNIT * 0.0254


def _euler(ax: float, ay: float, az: float) -> np.ndarray:
    """Static XYZ Euler angles (degrees) -> rotation matrix (Rz @ Ry @ Rx)."""
    a, b, c = np.radians([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def parse_asf(text: str) -> tuple[dict, dict]:
    bones: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    lines = [ln.strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        if lines[i] == "begin":
            b: dict = {"dof": []}
            i += 1
            while lines[i] != "end":
                tok = lines[i].split()
                if tok[0] == "name":
                    b["name"] = tok[1]
                elif tok[0] == "direction":
                    b["direction"] = np.array(tok[1:4], float)
                elif tok[0] == "length":
                    b["length"] = float(tok[1])
                elif tok[0] == "axis":
                    b["axis"] = np.array(tok[1:4], float)
                elif tok[0] == "dof":
                    b["dof"] = tok[1:]
                i += 1
            bones[b["name"]] = b
        elif lines[i] == ":hierarchy":
            i += 2  # skip "begin"
            while lines[i] != "end":
                tok = lines[i].split()
                children[tok[0]] = tok[1:]
                i += 1
        i += 1
    return bones, children


def parse_amc(text: str) -> list[dict[str, list[float]]]:
    frames: list[dict[str, list[float]]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln[0] in "#:":
            continue
        if ln.isdigit():
            frames.append({})
        elif frames:
            tok = ln.split()
            frames[-1][tok[0]] = [float(v) for v in tok[1:]]
    return frames


def forward_kinematics(asf_text: str, amc_text: str) -> dict[str, np.ndarray]:
    """World positions (metres, Y up as in CMU) of every bone's end point; 'root' is the pelvis."""
    bones, children = parse_asf(asf_text)
    frames = parse_amc(amc_text)
    out: dict[str, np.ndarray] = {name: np.zeros((len(frames), 3)) for name in [*bones, "root"]}
    C = {n: _euler(*b["axis"]) for n, b in bones.items()}
    for f, fr in enumerate(frames):
        r = fr["root"]
        pos = {"root": np.array(r[:3]) * M_PER_UNIT}
        rot = {"root": _euler(*r[3:6])}
        stack = ["root"]
        while stack:
            parent = stack.pop()
            for child in children.get(parent, []):
                b = bones[child]
                vals = dict(zip(b["dof"], fr.get(child, []), strict=False))
                m = C[child] @ _euler(vals.get("rx", 0), vals.get("ry", 0), vals.get("rz", 0)) @ C[child].T
                rot[child] = rot[parent] @ m
                pos[child] = pos[parent] + b["length"] * M_PER_UNIT * rot[child] @ b["direction"]
                stack.append(child)
        for n, p in pos.items():
            out[n][f] = p
    return out


# CMU bone end-points -> our 17 joints (bone "x" ends where the next joint starts)
CMU_TO_SCHEMA = {
    "head": "head",
    "neck": "lowerneck",
    "l_shoulder": "lclavicle",
    "r_shoulder": "rclavicle",
    "l_elbow": "lhumerus",
    "r_elbow": "rhumerus",
    "l_wrist": "lradius",
    "r_wrist": "rradius",
    "pelvis": "root",
    "l_hip": "lhipjoint",
    "r_hip": "rhipjoint",
    "l_knee": "lfemur",
    "r_knee": "rfemur",
    "l_ankle": "ltibia",
    "r_ankle": "rtibia",
    "l_toe": "lfoot",
    "r_toe": "rfoot",
}
