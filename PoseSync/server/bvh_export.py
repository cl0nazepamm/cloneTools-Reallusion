# -*- coding: utf-8 -*-
"""BVH export helpers for PoseSync."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _SciRot
except Exception:
    _SciRot = None

SERVER_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SERVER_DIR / "assets"


# Target BVH joint aliases -> source MHR joint aliases (normalized names).
# This is focused on the UE/CC chain and common CC hand/twist joints.
TARGET_TO_SOURCE_ALIASES = {
    # 3ds Max Biped template (PoseSync_TPose.bvh)
    "bip001pelvis": ["root"],
    "bip001spine": ["cspine0"],
    "bip001spine1": ["cspine1"],
    "bip001spine2": ["cspine3", "cspine2"],
    "bip001neck": ["cneck", "neck"],
    "bip001head": ["chead", "head"],
    "bip001lclavicle": ["lclavicle"],
    "bip001lupperarm": ["luparm"],
    "bip001lforearm": ["llowarm"],
    "bip001lhand": ["lwrist"],
    "bip001lfinger0": ["lindex1", "lthumb1"],
    "bip001rclavicle": ["rclavicle"],
    "bip001rupperarm": ["ruparm"],
    "bip001rforearm": ["rlowarm"],
    "bip001rhand": ["rwrist"],
    "bip001rfinger0": ["rindex1", "rthumb1"],
    "bip001lthigh": ["lupleg"],
    "bip001lcalf": ["llowleg"],
    "bip001lfoot": ["lfoot"],
    "bip001ltoe0": ["lball"],
    "bip001rthigh": ["rupleg"],
    "bip001rcalf": ["rlowleg"],
    "bip001rfoot": ["rfoot"],
    "bip001rtoe0": ["rball"],

    "root": ["root"],
    "pelvis": ["root"],
    "ccbasepelvis": ["root"],
    "spine01": ["cspine0"],
    "spine02": ["cspine1"],
    "spine03": ["cspine2"],
    "spine04": ["cspine3"],
    "spine05": ["cspine3", "cneck"],
    "neck01": ["cneck", "neck"],
    "neck02": ["cneck", "neck"],
    "head": ["chead", "head"],
    "claviclel": ["lclavicle"],
    "upperarml": ["luparm"],
    "lowerarml": ["llowarm"],
    "handl": ["lwrist"],
    "ccbaselforearmtwist01": ["llowarmtwist1proc"],
    "ccbaselforearmtwist02": ["llowarmtwist2proc"],
    "ccbaselupperarmtwist01": ["luparmtwist1proc", "luparmtwist0proc"],
    "ccbaselupperarmtwist02": ["luparmtwist2proc", "luparmtwist1proc"],
    "thighl": ["lupleg"],
    "calfl": ["llowleg"],
    "footl": ["lfoot"],
    "balll": ["lball"],
    "ccbaselcalftwist01": ["llowlegtwist1proc"],
    "ccbaselcalftwist02": ["llowlegtwist2proc"],
    "ccbaselthightwist01": ["luplegtwist1proc", "luplegtwist0proc"],
    "ccbaselthightwist02": ["luplegtwist2proc", "luplegtwist1proc"],
    "ccbaselpinkytoe1": ["lball"],
    "ccbaselringtoe1": ["lball"],
    "ccbaselindextoe1": ["lball"],
    "clavicler": ["rclavicle"],
    "upperarmr": ["ruparm"],
    "lowerarmr": ["rlowarm"],
    "handr": ["rwrist"],
    "ccbaserforearmtwist01": ["rlowarmtwist1proc"],
    "ccbaserforearmtwist02": ["rlowarmtwist2proc"],
    "ccbaserupperarmtwist01": ["ruparmtwist1proc", "ruparmtwist0proc"],
    "ccbaserupperarmtwist02": ["ruparmtwist2proc", "ruparmtwist1proc"],
    "thighr": ["rupleg"],
    "calfr": ["rlowleg"],
    "footr": ["rfoot"],
    "ballr": ["rball"],
    "ccbasercalftwist01": ["rlowlegtwist1proc"],
    "ccbasercalftwist02": ["rlowlegtwist2proc"],
    "ccbaserthightwist01": ["ruplegtwist1proc", "ruplegtwist0proc"],
    "ccbaserthightwist02": ["ruplegtwist2proc", "ruplegtwist1proc"],
    "ccbaserpinkytoe1": ["rball"],
    "ccbaserringtoe1": ["rball"],
    "ccbaserindextoe1": ["rball"],
    "pinkymetacarpall": ["lpinky0", "lpinky1"],
    "pinky01l": ["lpinky1", "lpinky0"],
    "pinky02l": ["lpinky2"],
    "pinky03l": ["lpinky3"],
    "ringmetacarpall": ["lring1"],
    "ring01l": ["lring1"],
    "ring02l": ["lring2"],
    "ring03l": ["lring3"],
    "middlemetacarpall": ["lmiddle1"],
    "middle01l": ["lmiddle1"],
    "middle02l": ["lmiddle2"],
    "middle03l": ["lmiddle3"],
    "indexmetacarpall": ["lindex1"],
    "index01l": ["lindex1"],
    "index02l": ["lindex2"],
    "index03l": ["lindex3"],
    "thumb01l": ["lthumb0"],
    "thumb02l": ["lthumb1"],
    "thumb03l": ["lthumb2", "lthumb3"],
    "pinkymetacarpalr": ["rpinky0", "rpinky1"],
    "pinky01r": ["rpinky1", "rpinky0"],
    "pinky02r": ["rpinky2"],
    "pinky03r": ["rpinky3"],
    "ringmetacarpalr": ["rring1"],
    "ring01r": ["rring1"],
    "ring02r": ["rring2"],
    "ring03r": ["rring3"],
    "middlemetacarpalr": ["rmiddle1"],
    "middle01r": ["rmiddle1"],
    "middle02r": ["rmiddle2"],
    "middle03r": ["rmiddle3"],
    "indexmetacarpalr": ["rindex1"],
    "index01r": ["rindex1"],
    "index02r": ["rindex2"],
    "index03r": ["rindex3"],
    "thumb01r": ["rthumb0"],
    "thumb02r": ["rthumb1"],
    "thumb03r": ["rthumb2", "rthumb3"],
}


def _normalize_name(name: str) -> str:
    if not name:
        return ""
    n = str(name).strip().lower()
    for prefix in ("rl_", "cc_base_", "cc_", "skel_"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    return re.sub(r"[^a-z0-9]", "", n)


def _quat_xyzw_to_mat3(quat_xyzw: Iterable[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    n = x * x + y * y + z * z + w * w
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _mat3_to_quat_xyzw(m: np.ndarray) -> Tuple[float, float, float, float]:
    t = float(m[0, 0] + m[1, 1] + m[2, 2])
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return None
    return np.asarray(v, dtype=np.float64) / n


def _rot_x_deg_matrix(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _axis_rot_mat(axis: str, deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    if axis.upper() == "X":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=np.float64,
        )
    if axis.upper() == "Y":
        return np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float64,
        )
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _channel_angles_to_mat3(rotation_channels: List[str], angle_map: Dict[str, float]) -> np.ndarray:
    order = "".join(ch[0].upper() for ch in rotation_channels if ch.lower().endswith("rotation"))
    vals = [float(angle_map.get(ch, 0.0)) for ch in rotation_channels]
    if _SciRot is not None and len(order) == len(vals):
        try:
            return np.asarray(_SciRot.from_euler(order, vals, degrees=True).as_matrix(), dtype=np.float64)
        except Exception:
            pass
    r = np.eye(3, dtype=np.float64)
    for ch in rotation_channels:
        axis = ch[0].upper()
        ang = float(angle_map.get(ch, 0.0))
        r = r @ _axis_rot_mat(axis, ang)
    return r


def _to_bool(v, default=False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on", "y"):
            return True
        if s in ("0", "false", "no", "off", "n", ""):
            return False
    return bool(default)


def _to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _runtime_config_candidates() -> List[str]:
    out = []
    cfg_path = str(os.environ.get("POSESYNC_BVH_CONFIG_PATH", "")).strip()
    if cfg_path:
        out.append(cfg_path)
    legacy_path = str(os.environ.get("POSESYNC_BVH_JOINT_OFFSETS_PATH", "")).strip()
    if legacy_path:
        out.append(legacy_path)
    out.append(str(ASSETS_DIR / "biped_endo.json"))
    return out


def _load_runtime_transform_settings() -> Dict[str, object]:
    src_path = ""
    for p in _runtime_config_candidates():
        if p and os.path.isfile(p):
            src_path = p
            break
    if not src_path:
        return {}

    try:
        raw = json.loads(Path(src_path).read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    # Preferred config layout:
    # { "biped_endo": { "posesync": { "transform": {...} } } }
    # or { "posesync": { "transform": {...} } }
    posesync_cfg = None
    if isinstance(raw.get("posesync"), dict):
        posesync_cfg = raw.get("posesync")
    if posesync_cfg is None:
        be = raw.get("biped_endo")
        if isinstance(be, dict):
            for key in ("posesync", "PoseSync", "POSESYNC"):
                if isinstance(be.get(key), dict):
                    posesync_cfg = be.get(key)
                    break

    if isinstance(posesync_cfg, dict):
        t = posesync_cfg.get("transform")
        if isinstance(t, dict):
            return t
        return posesync_cfg

    # Legacy fallback: a flat joint-offset JSON map.
    return {"joint_offsets_deg": raw}


def _cfg_float(settings: Optional[Dict[str, object]], key: str, env_key: str, default: float) -> float:
    if isinstance(settings, dict) and key in settings:
        return _to_float(settings.get(key), default)
    return _to_float(os.environ.get(env_key, default), default)


def _cfg_text(settings: Optional[Dict[str, object]], key: str, env_key: str, default: str) -> str:
    if isinstance(settings, dict) and key in settings:
        return str(settings.get(key, default)).strip()
    return str(os.environ.get(env_key, default)).strip()


def _flip_switch(settings: Optional[Dict[str, object]], key: str, default: bool = True) -> bool:
    if not isinstance(settings, dict):
        return default
    for group_key in ("flip_switches", "switches", "transform_flip_switches"):
        g = settings.get(group_key)
        if isinstance(g, dict) and key in g:
            return _to_bool(g.get(key), default)
    return default


def _is_root_joint_norm(joint_name_norm: str) -> bool:
    return joint_name_norm in {
        "root",
        "rlboneroot",
        "bip001",
        "bip01",
        "bip001root",
        "bip001pelvis",
    }


def _is_pelvis_joint_norm(joint_name_norm: str) -> bool:
    return joint_name_norm in {
        "pelvis",
        "ccbasepelvis",
        "bip001pelvis",
    }


def _locked_joint_names_norm(settings: Optional[Dict[str, object]] = None) -> set:
    raw_default = "root,bip001,pelvis,cc_base_pelvis,bip001_pelvis,bip001 pelvis,bip001pelvis"
    if isinstance(settings, dict) and "locked_joints" in settings:
        raw_locked = settings.get("locked_joints")
    else:
        raw_locked = os.environ.get("POSESYNC_BVH_LOCKED_JOINTS", raw_default)

    out = set()
    if isinstance(raw_locked, (list, tuple, set)):
        for tok in raw_locked:
            n = _normalize_name(str(tok))
            if n:
                out.add(n)
        return out

    raw = str(raw_locked or "").strip()
    if not raw:
        return out
    for tok in raw.split(","):
        n = _normalize_name(tok)
        if n:
            out.add(n)
    return out


def _joint_local_x_offset_deg(joint_name_norm: str, settings: Optional[Dict[str, object]] = None) -> float:
    # Global non-root correction for BVH local axis mismatch.
    # Root is intentionally excluded because template root orientation is already correct.
    off = 0.0
    if not _is_root_joint_norm(joint_name_norm) and _flip_switch(settings, "enable_non_root_x", True):
        off += _cfg_float(settings, "non_root_x_deg", "POSESYNC_BVH_NON_ROOT_X_DEG", 0.0)

    # Per-joint cleanup offsets after retarget solve.
    if _is_root_joint_norm(joint_name_norm) and _flip_switch(settings, "enable_root_x", True):
        off += _cfg_float(settings, "root_x_deg", "POSESYNC_BVH_ROOT_X_DEG", 0.0)
    if (
        _is_pelvis_joint_norm(joint_name_norm)
        and not _is_root_joint_norm(joint_name_norm)
        and _flip_switch(settings, "enable_pelvis_x", True)
    ):
        off += _cfg_float(settings, "pelvis_x_deg", "POSESYNC_BVH_PELVIS_X_DEG", 0.0)
    if (
        "toe" in joint_name_norm
        or joint_name_norm.startswith("ball")
        or joint_name_norm in ("balll", "ballr")
    ) and _flip_switch(settings, "enable_toes_x", True):
        off += _cfg_float(settings, "toes_x_deg", "POSESYNC_BVH_TOES_X_DEG", 0.0)
    return off


def _source_basis_matrix(settings: Optional[Dict[str, object]] = None) -> np.ndarray:
    # Convert source coordinate basis into target BVH basis.
    # Default -90deg around X (Y-up -> Z-up style conversion for this CC template).
    deg = _cfg_float(settings, "source_x_deg", "POSESYNC_BVH_SOURCE_X_DEG", -90.0)
    rot_x = _rot_x_deg_matrix(deg)
    mirror_axis = _cfg_text(settings, "source_mirror_axis", "POSESYNC_BVH_SOURCE_MIRROR_AXIS", "none").lower()
    mirror = np.eye(3, dtype=np.float64)
    if mirror_axis == "x":
        mirror[0, 0] = -1.0
    elif mirror_axis == "y":
        mirror[1, 1] = -1.0
    elif mirror_axis == "z":
        mirror[2, 2] = -1.0
    # Rotate first, then mirror into target handedness.
    return mirror @ rot_x


def _joint_override_offsets_deg(
    settings: Optional[Dict[str, object]] = None,
) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {}
    if settings is None:
        settings = _load_runtime_transform_settings()
    raw = None
    if isinstance(settings, dict):
        for key in ("joint_offsets_deg", "joint_offsets", "per_joint_offsets"):
            cand = settings.get(key)
            if isinstance(cand, dict):
                raw = cand
                break
    if not isinstance(raw, dict):
        return out

    for k, v in raw.items():
        jn = _normalize_name(str(k))
        if not jn:
            continue
        x = y = z = 0.0
        try:
            if isinstance(v, (int, float)):
                x = float(v)
            elif isinstance(v, (list, tuple)) and len(v) >= 3:
                x, y, z = float(v[0]), float(v[1]), float(v[2])
            elif isinstance(v, dict):
                x = float(v.get("x", v.get("X", 0.0)))
                y = float(v.get("y", v.get("Y", 0.0)))
                z = float(v.get("z", v.get("Z", 0.0)))
        except Exception:
            continue
        out[jn] = (x, y, z)
    return out


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av = _safe_normalize(a)
    bv = _safe_normalize(b)
    if av is None or bv is None:
        return np.eye(3, dtype=np.float64)
    c = float(np.dot(av, bv))
    c = max(-1.0, min(1.0, c))
    if c > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-8:
        axis = np.cross(av, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(av, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = _safe_normalize(axis)
        if axis is None:
            return np.eye(3, dtype=np.float64)
        u = axis.reshape(3, 1)
        return -np.eye(3, dtype=np.float64) + 2.0 * (u @ u.T)

    v = np.cross(av, bv)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def _solve_rotation_from_vector_sets(
    template_vectors: List[np.ndarray], source_vectors: List[np.ndarray]
) -> Optional[np.ndarray]:
    if not template_vectors or not source_vectors:
        return None
    if len(template_vectors) != len(source_vectors):
        return None

    t = []
    s = []
    for tv, sv in zip(template_vectors, source_vectors):
        tn = _safe_normalize(np.asarray(tv, dtype=np.float64))
        sn = _safe_normalize(np.asarray(sv, dtype=np.float64))
        if tn is None or sn is None:
            continue
        t.append(tn)
        s.append(sn)

    if not t:
        return None
    if len(t) == 1:
        return _rotation_between_vectors(t[0], s[0])

    a = np.stack(t, axis=0)
    b = np.stack(s, axis=0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if float(np.linalg.det(r)) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return np.asarray(r, dtype=np.float64)


def _mat3_to_euler_deg_zyx(m: np.ndarray) -> Tuple[float, float, float]:
    # R = Rz(z) * Ry(y) * Rx(x)
    sy = -float(m[2, 0])
    sy = max(-1.0, min(1.0, sy))
    y = math.asin(sy)
    cy = math.cos(y)
    if abs(cy) > 1e-8:
        x = math.atan2(float(m[2, 1]), float(m[2, 2]))
        z = math.atan2(float(m[1, 0]), float(m[0, 0]))
    else:
        x = 0.0
        z = math.atan2(-float(m[0, 1]), float(m[1, 1]))
    return math.degrees(x), math.degrees(y), math.degrees(z)


def _mat3_to_euler_deg_xyz(m: np.ndarray) -> Tuple[float, float, float]:
    # R = Rx(x) * Ry(y) * Rz(z)
    sy = float(m[0, 2])
    sy = max(-1.0, min(1.0, sy))
    y = math.asin(sy)
    cy = math.cos(y)
    if abs(cy) > 1e-8:
        x = math.atan2(-float(m[1, 2]), float(m[2, 2]))
        z = math.atan2(-float(m[0, 1]), float(m[0, 0]))
    else:
        x = math.atan2(float(m[2, 1]), float(m[1, 1]))
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


def _quat_to_channel_angles_deg(
    quat_xyzw: Iterable[float], rotation_channels: List[str]
) -> Dict[str, float]:
    rot_order = "".join(ch[0].upper() for ch in rotation_channels if ch.lower().endswith("rotation"))
    m = _quat_xyzw_to_mat3(quat_xyzw)

    by_axis = {"Xrotation": 0.0, "Yrotation": 0.0, "Zrotation": 0.0}
    if _SciRot is not None and len(rot_order) == len(rotation_channels):
        try:
            e = _SciRot.from_matrix(m).as_euler(rot_order, degrees=True)
            for axis, val in zip(rot_order, e):
                by_axis[f"{axis}rotation"] = float(val)
        except Exception:
            pass
    else:
        if rot_order == "ZYX":
            ex, ey, ez = _mat3_to_euler_deg_zyx(m)
        elif rot_order == "XYZ":
            ex, ey, ez = _mat3_to_euler_deg_xyz(m)
        else:
            ex, ey, ez = _mat3_to_euler_deg_xyz(m)
        by_axis = {"Xrotation": ex, "Yrotation": ey, "Zrotation": ez}

    return {ch: float(by_axis.get(ch, 0.0)) for ch in rotation_channels}


def _parse_bvh_template(template_text: str):
    lines = template_text.splitlines()
    motion_idx = -1
    for i, line in enumerate(lines):
        if line.strip().upper() == "MOTION":
            motion_idx = i
            break
    if motion_idx < 0:
        raise ValueError("Invalid BVH: MOTION section not found.")

    hierarchy_lines = lines[:motion_idx]
    motion_lines = lines[motion_idx + 1 :]

    joint_channels: Dict[str, List[str]] = {}
    channel_layout: List[Tuple[str, str]] = []
    joint_order: List[str] = []
    joint_parent: Dict[str, Optional[str]] = {}
    joint_children: Dict[str, List[str]] = {}
    joint_offsets: Dict[str, np.ndarray] = {}

    stack: List[Tuple[str, Optional[str]]] = []
    pending: Optional[Tuple[str, Optional[str]]] = None

    for line in hierarchy_lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("ROOT "):
            pending = ("joint", s.split(None, 1)[1].strip())
            continue
        if s.startswith("JOINT "):
            pending = ("joint", s.split(None, 1)[1].strip())
            continue
        if s.startswith("End Site"):
            pending = ("end", None)
            continue
        if s == "{":
            if pending is None:
                stack.append(("block", None))
            else:
                typ, nm = pending
                pending = None
                if typ == "joint" and nm:
                    parent_nm = None
                    for k, v in reversed(stack):
                        if k == "joint" and v:
                            parent_nm = v
                            break
                    joint_parent[nm] = parent_nm
                    joint_children.setdefault(nm, [])
                    if parent_nm:
                        joint_children.setdefault(parent_nm, []).append(nm)
                    joint_order.append(nm)
                    joint_channels.setdefault(nm, [])
                    stack.append(("joint", nm))
                else:
                    stack.append(("end", None))
            continue
        if s == "}":
            if stack:
                stack.pop()
            continue
        if s.startswith("OFFSET "):
            parts = s.split()
            if len(parts) >= 4 and stack:
                top_typ, top_name = stack[-1]
                if top_typ == "joint" and top_name and top_name not in joint_offsets:
                    try:
                        joint_offsets[top_name] = np.array(
                            [float(parts[1]), float(parts[2]), float(parts[3])],
                            dtype=np.float64,
                        )
                    except Exception:
                        pass
            continue
        if s.startswith("CHANNELS "):
            parts = s.split()
            if len(parts) >= 3 and stack and stack[-1][0] == "joint" and stack[-1][1]:
                current_joint = stack[-1][1]
                count = int(parts[1])
                chans = parts[2 : 2 + count]
                joint_channels[current_joint] = chans
                for ch in chans:
                    channel_layout.append((current_joint, ch))
            continue

    frame_time = 1.0 / 30.0
    frame_values: List[float] = []
    frame_count = 0
    started_data = False

    for line in motion_lines:
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("frames:"):
            try:
                frame_count = int(s.split(":", 1)[1].strip())
            except Exception:
                frame_count = 0
            continue
        if s.lower().startswith("frame time:"):
            try:
                frame_time = float(s.split(":", 1)[1].strip())
            except Exception:
                frame_time = 1.0 / 30.0
            started_data = True
            continue
        if started_data and not frame_values:
            try:
                frame_values = [float(v) for v in s.split()]
            except Exception:
                frame_values = []

    channel_count = len(channel_layout)
    if len(frame_values) < channel_count:
        frame_values.extend([0.0] * (channel_count - len(frame_values)))
    elif len(frame_values) > channel_count:
        frame_values = frame_values[:channel_count]

    world_rest: Dict[str, np.ndarray] = {}
    for jn in joint_order:
        parent = joint_parent.get(jn)
        off = joint_offsets.get(jn, np.zeros(3, dtype=np.float64))
        if parent is None:
            world_rest[jn] = off.copy()
        else:
            world_rest[jn] = world_rest.get(parent, np.zeros(3, dtype=np.float64)) + off

    return {
        "hierarchy_lines": hierarchy_lines,
        "frame_time": frame_time,
        "frame_count": frame_count,
        "channel_layout": channel_layout,
        "joint_channels": joint_channels,
        "joint_order": joint_order,
        "joint_parent": joint_parent,
        "joint_children": joint_children,
        "joint_offsets": joint_offsets,
        "joint_world_rest": world_rest,
        "template_frame_values": frame_values,
    }


def _build_source_index(joint_quats_xyzw: Dict[str, Iterable[float]]):
    exact = {}
    normalized = {}
    for k, v in (joint_quats_xyzw or {}).items():
        if v is None:
            continue
        key = str(k).strip()
        exact[key] = v
        exact[key.lower()] = v
        nk = _normalize_name(key)
        if nk and nk not in normalized:
            normalized[nk] = v
    return exact, normalized


def _build_coord_index(
    joint_coords_xyz: Dict[str, Iterable[float]],
    settings: Optional[Dict[str, object]] = None,
):
    exact = {}
    normalized = {}
    basis = _source_basis_matrix(settings)
    for k, v in (joint_coords_xyz or {}).items():
        if v is None:
            continue
        key = str(k).strip()
        vec = np.asarray(v, dtype=np.float64)
        if vec.shape[0] < 3:
            continue
        vec = basis @ vec[:3]
        exact[key] = vec
        exact[key.lower()] = vec
        nk = _normalize_name(key)
        if nk and nk not in normalized:
            normalized[nk] = vec
    return exact, normalized


def _find_quat_for_joint(
    joint_name: str,
    exact: Dict[str, Iterable[float]],
    normalized: Dict[str, Iterable[float]],
):
    if joint_name in exact:
        return exact[joint_name]
    lname = joint_name.lower()
    if lname in exact:
        return exact[lname]
    n = _normalize_name(joint_name)
    candidates = [n]
    if n.startswith("ccbase"):
        candidates.append(n[6:])
    if n.startswith("rl"):
        candidates.append(n[2:])
    candidates.extend(TARGET_TO_SOURCE_ALIASES.get(n, []))
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if c in normalized:
            return normalized[c]
    return None


def _find_coord_for_joint(
    joint_name: str,
    exact: Dict[str, np.ndarray],
    normalized: Dict[str, np.ndarray],
):
    if joint_name in exact:
        return exact[joint_name]
    lname = joint_name.lower()
    if lname in exact:
        return exact[lname]

    n = _normalize_name(joint_name)
    candidates = [n]
    if n.startswith("ccbase"):
        candidates.append(n[6:])
    if n.startswith("rl"):
        candidates.append(n[2:])
    candidates.extend(TARGET_TO_SOURCE_ALIASES.get(n, []))

    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if c in normalized:
            return normalized[c]
    return None


def _format_frame(values: List[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def build_pose_bvh_from_template(
    template_bvh_path: str,
    output_bvh_path: str,
    joint_quats_xyzw: Dict[str, Iterable[float]],
    joint_coords_xyz: Optional[Dict[str, Iterable[float]]] = None,
) -> Dict[str, object]:
    template_path = Path(template_bvh_path)
    output_path = Path(output_bvh_path)
    if not template_path.is_file():
        raise FileNotFoundError(f"BVH template not found: {template_path}")

    parsed = _parse_bvh_template(template_path.read_text(encoding="utf-8", errors="ignore"))
    values = list(parsed["template_frame_values"])
    channel_layout = parsed["channel_layout"]
    joint_channels = parsed["joint_channels"]
    joint_parent = parsed["joint_parent"]
    joint_children = parsed["joint_children"]
    joint_world_rest = parsed["joint_world_rest"]
    channel_index = {(jn, ch): idx for idx, (jn, ch) in enumerate(channel_layout)}
    transform_settings = _load_runtime_transform_settings()
    locked_norm = _locked_joint_names_norm(transform_settings)
    joint_override = _joint_override_offsets_deg(transform_settings)

    quat_exact, quat_norm = _build_source_index(joint_quats_xyzw)
    coord_exact, coord_norm = _build_coord_index(joint_coords_xyz or {}, transform_settings)

    updated_joints = 0
    updated_channels = 0
    solved_joints = 0
    quat_fallback_joints = 0
    missing_joints = []

    # Solve rotations by matching template child vectors to source pose vectors.
    # This avoids direct rest-pose mismatch artifacts from source local quaternions.
    joint_global_rot: Dict[str, np.ndarray] = {}
    joint_local_rot: Dict[str, np.ndarray] = {}
    joint_mode: Dict[str, str] = {}
    template_local_rot: Dict[str, np.ndarray] = {}
    for joint_name in parsed["joint_order"]:
        chans = joint_channels.get(joint_name, [])
        rot_chans = [c for c in chans if c.lower().endswith("rotation")]
        if not rot_chans:
            template_local_rot[joint_name] = np.eye(3, dtype=np.float64)
            continue
        amap = {}
        for ch in rot_chans:
            idx = channel_index.get((joint_name, ch))
            if idx is not None and idx < len(values):
                amap[ch] = float(values[idx])
        template_local_rot[joint_name] = _channel_angles_to_mat3(rot_chans, amap)

    for joint_name in parsed["joint_order"]:
        parent = joint_parent.get(joint_name)
        parent_global = (
            joint_global_rot[parent]
            if parent is not None and parent in joint_global_rot
            else np.eye(3, dtype=np.float64)
        )
        jnorm = _normalize_name(joint_name)
        if jnorm in locked_norm:
            local_rot = template_local_rot.get(joint_name, np.eye(3, dtype=np.float64))
            joint_local_rot[joint_name] = local_rot
            joint_global_rot[joint_name] = parent_global @ local_rot
            joint_mode[joint_name] = "locked"
            continue

        tvecs = []
        svecs = []
        src_joint = _find_coord_for_joint(joint_name, coord_exact, coord_norm)
        if src_joint is not None:
            # Parent vector helps disambiguate roll and prevents frequent limb flips.
            if parent:
                src_parent = _find_coord_for_joint(parent, coord_exact, coord_norm)
                if src_parent is not None:
                    tv = joint_world_rest.get(parent, np.zeros(3)) - joint_world_rest.get(
                        joint_name, np.zeros(3)
                    )
                    sv = src_parent - src_joint
                    if float(np.linalg.norm(tv)) > 1e-6 and float(np.linalg.norm(sv)) > 1e-6:
                        tvecs.append(tv)
                        svecs.append(sv)
            for child in joint_children.get(joint_name, []):
                src_child = _find_coord_for_joint(child, coord_exact, coord_norm)
                if src_child is None:
                    continue
                tv = joint_world_rest.get(child, np.zeros(3)) - joint_world_rest.get(joint_name, np.zeros(3))
                sv = src_child - src_joint
                if float(np.linalg.norm(tv)) > 1e-6 and float(np.linalg.norm(sv)) > 1e-6:
                    tvecs.append(tv)
                    svecs.append(sv)

        solved_global = _solve_rotation_from_vector_sets(tvecs, svecs)
        if solved_global is not None:
            solved_joints += 1
            local_rot = parent_global.T @ solved_global
            joint_global_rot[joint_name] = solved_global
            joint_local_rot[joint_name] = local_rot
            joint_mode[joint_name] = "solved"
            continue

        quat = _find_quat_for_joint(joint_name, quat_exact, quat_norm)
        if quat is not None:
            quat_fallback_joints += 1
            local_rot = _quat_xyzw_to_mat3(quat)
            joint_local_rot[joint_name] = local_rot
            joint_global_rot[joint_name] = parent_global @ local_rot
            joint_mode[joint_name] = "quat"
            continue

        joint_local_rot[joint_name] = np.eye(3, dtype=np.float64)
        joint_global_rot[joint_name] = parent_global
        joint_mode[joint_name] = "identity"

    for joint_name in parsed["joint_order"]:
        chans = joint_channels.get(joint_name, [])
        rot_chans = [c for c in chans if c.lower().endswith("rotation")]
        if not rot_chans:
            continue
        jnorm = _normalize_name(joint_name)
        xoff = _joint_local_x_offset_deg(jnorm, transform_settings)
        yoff = 0.0
        zoff = 0.0
        if jnorm in joint_override:
            ox, oy, oz = joint_override[jnorm]
            xoff += float(ox)
            yoff += float(oy)
            zoff += float(oz)
        mode = joint_mode.get(joint_name, "identity")
        if mode == "locked" and abs(xoff) <= 1e-6 and abs(yoff) <= 1e-6 and abs(zoff) <= 1e-6:
            # Keep template channels exactly as authored when no correction is requested.
            continue
        if mode == "identity" and abs(xoff) <= 1e-6 and abs(yoff) <= 1e-6 and abs(zoff) <= 1e-6:
            missing_joints.append(joint_name)
            continue

        if mode == "locked":
            local_rot = template_local_rot.get(joint_name, np.eye(3, dtype=np.float64))
        else:
            local_rot = joint_local_rot.get(joint_name, np.eye(3, dtype=np.float64))
        if abs(xoff) > 1e-6:
            # Apply as local-space post correction.
            local_rot = local_rot @ _rot_x_deg_matrix(xoff)
        if abs(yoff) > 1e-6:
            local_rot = local_rot @ _axis_rot_mat("Y", yoff)
        if abs(zoff) > 1e-6:
            local_rot = local_rot @ _axis_rot_mat("Z", zoff)
        quat = _mat3_to_quat_xyzw(local_rot)
        angles = _quat_to_channel_angles_deg(quat, rot_chans)

        wrote = 0
        for ch in rot_chans:
            idx = channel_index.get((joint_name, ch))
            if idx is None:
                continue
            values[idx] = float(angles.get(ch, 0.0))
            wrote += 1

        if wrote > 0:
            updated_joints += 1
            updated_channels += wrote

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_lines = []
    out_lines.extend(parsed["hierarchy_lines"])
    out_lines.append("MOTION")
    out_lines.append("Frames: 1")
    out_lines.append(f"Frame Time: {float(parsed['frame_time']):.6f}")
    out_lines.append(_format_frame(values))
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return {
        "template_bvh_path": str(template_path),
        "output_bvh_path": str(output_path),
        "frame_time": float(parsed["frame_time"]),
        "joint_count": len(parsed["joint_order"]),
        "channel_count": len(channel_layout),
        "updated_joint_count": updated_joints,
        "updated_channel_count": updated_channels,
        "solved_joint_count": solved_joints,
        "quat_fallback_joint_count": quat_fallback_joints,
        "missing_joint_count": len(missing_joints),
        "missing_joint_sample": missing_joints[:20],
    }
