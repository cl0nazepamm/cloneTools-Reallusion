# -*- coding: utf-8 -*-
"""Direct MHR-to-BVH export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial.transform import Rotation as _SciRot
except Exception:
    _SciRot = None

SERVER_DIR = Path(__file__).resolve().parent
DEFAULT_MHR_BONE_DATA_PATH = SERVER_DIR / "MHR" / "mhr_bone_data.json"

DEFAULT_PROFILE = {
    "profile_name": "mhr_raw_pose",
    "profile_version": 1,
    "root_joint_name": "root",
    "drop_world_root": True,
    "zero_root_offset": True,
    "rotation_mode": "local_pose",
    "rotation_order": "ZXY",
    "source_quat_key": "quat_xyzw_global",
    "source_coord_key": "coord",
    "root_translation_mode": "pose_minus_rest",
    "root_translation_scale": 100.0,
    "root_translation_axis_multipliers": [1.0, -1.0, 1.0],
    "fps": 30,
    "include_joint_names": [],
    "exclude_joint_names": [],
}


def list_bvh_profile_names(profile_dir: Path) -> List[str]:
    out = []
    if not profile_dir.is_dir():
        return out
    for p in sorted(profile_dir.glob("*.json")):
        out.append(p.stem)
    return out


def _as_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_str_list(v) -> List[str]:
    if not isinstance(v, list):
        return []
    out = []
    for item in v:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _as_vec3(v, default: Tuple[float, float, float]) -> np.ndarray:
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return np.array([_as_float(v[0]), _as_float(v[1]), _as_float(v[2])], dtype=np.float64)
    return np.array(default, dtype=np.float64)


def _quat_xyzw_to_mat3(quat_xyzw: Iterable[float]) -> np.ndarray:
    if _SciRot is not None:
        try:
            return np.asarray(_SciRot.from_quat(quat_xyzw).as_matrix(), dtype=np.float64)
        except Exception:
            pass
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


def _orthonormalize(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    r = u @ vt
    if float(np.linalg.det(r)) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _mat3_to_euler_deg(m: np.ndarray, order: str) -> np.ndarray:
    r = _orthonormalize(m)
    if _SciRot is not None:
        return np.asarray(_SciRot.from_matrix(r).as_euler(order, degrees=True), dtype=np.float64)
    # Fallback is XYZ only when scipy is unavailable.
    sy = float(r[0, 2])
    sy = max(-1.0, min(1.0, sy))
    y = np.arcsin(sy)
    cy = np.cos(y)
    if abs(cy) > 1e-8:
        x = np.arctan2(-float(r[1, 2]), float(r[2, 2]))
        z = np.arctan2(-float(r[0, 1]), float(r[0, 0]))
    else:
        x = np.arctan2(float(r[2, 1]), float(r[1, 1]))
        z = 0.0
    if order == "XYZ":
        return np.array([np.degrees(x), np.degrees(y), np.degrees(z)], dtype=np.float64)
    return np.array([0.0, 0.0, 0.0], dtype=np.float64)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_profile(raw: Optional[Dict]) -> Dict:
    out = dict(DEFAULT_PROFILE)
    if isinstance(raw, dict):
        out.update(raw)

    order = str(out.get("rotation_order", "ZXY")).strip().upper()
    if len(order) != 3 or len(set(order)) != 3 or any(ch not in "XYZ" for ch in order):
        order = "ZXY"
    out["rotation_order"] = order

    out["root_joint_name"] = str(out.get("root_joint_name", "root")).strip() or "root"
    out["drop_world_root"] = bool(out.get("drop_world_root", True))
    out["zero_root_offset"] = bool(out.get("zero_root_offset", True))
    out["rotation_mode"] = str(out.get("rotation_mode", "local_pose")).strip().lower() or "local_pose"
    out["source_quat_key"] = str(out.get("source_quat_key", "quat_xyzw_global")).strip() or "quat_xyzw_global"
    out["source_coord_key"] = str(out.get("source_coord_key", "coord")).strip() or "coord"
    out["root_translation_mode"] = (
        str(out.get("root_translation_mode", "pose_minus_rest")).strip().lower() or "pose_minus_rest"
    )
    out["root_translation_scale"] = _as_float(out.get("root_translation_scale", 100.0), 100.0)
    out["root_translation_axis_multipliers"] = _as_vec3(
        out.get("root_translation_axis_multipliers"), (1.0, -1.0, 1.0)
    )
    out["fps"] = max(1, int(_as_float(out.get("fps", 30), 30)))
    out["include_joint_names"] = _as_str_list(out.get("include_joint_names"))
    out["exclude_joint_names"] = _as_str_list(out.get("exclude_joint_names"))
    return out


def load_bvh_profile(
    profile_dir: Path,
    profile_name: str = "mhr_raw_pose",
    profile_inline: Optional[Dict] = None,
) -> Dict:
    if isinstance(profile_inline, dict):
        p = _normalize_profile(profile_inline)
        p.setdefault("profile_name", "inline")
        p.setdefault("profile_source", "inline")
        return p

    name = str(profile_name or "").strip() or "mhr_raw_pose"
    path = profile_dir / f"{name}.json"
    if not path.is_file():
        if name == DEFAULT_PROFILE["profile_name"]:
            p = _normalize_profile(DEFAULT_PROFILE)
            p["profile_source"] = "builtin-default"
            return p
        raise FileNotFoundError(f"BVH profile not found: {path}")

    p = _normalize_profile(_load_json(path))
    p.setdefault("profile_name", name)
    p["profile_source"] = str(path)
    return p


def _build_skeleton_layout(mhr_bone_data: Dict, profile: Dict):
    joints = list(mhr_bone_data.get("joints", []))
    if not joints:
        raise ValueError("mhr_bone_data has no joints")

    joints_sorted = sorted(joints, key=lambda x: int(x.get("index", 0)))
    max_idx = int(max(int(j.get("index", 0)) for j in joints_sorted))
    by_idx = [None] * (max_idx + 1)
    for j in joints_sorted:
        idx = int(j.get("index", -1))
        if idx >= 0:
            by_idx[idx] = j
    if any(x is None for x in by_idx):
        raise ValueError("mhr_bone_data has sparse joint indices")

    name_to_idx = {}
    parent_idx = []
    for idx, j in enumerate(by_idx):
        name_to_idx[str(j.get("name"))] = idx
        parent_idx.append(int(j.get("parent_index", -1)))

    children = {i: [] for i in range(len(by_idx))}
    for i, p in enumerate(parent_idx):
        if p >= 0 and p < len(by_idx):
            children[p].append(i)

    root_name = str(profile.get("root_joint_name", "root"))
    root_idx = name_to_idx.get(root_name)
    if root_idx is None:
        # Fallback to first top-level parent.
        root_idx = next((i for i, p in enumerate(parent_idx) if p < 0), 0)

    if profile.get("drop_world_root", True):
        if by_idx[root_idx].get("name") == "body_world" and children.get(root_idx):
            # Prefer the real character root when present.
            root_idx = children[root_idx][0]

    include_names = set(profile.get("include_joint_names") or [])
    exclude_names = set(profile.get("exclude_joint_names") or [])
    include_set = set()

    if include_names:
        for nm in include_names:
            idx = name_to_idx.get(nm)
            while idx is not None and idx >= 0:
                include_set.add(idx)
                p = parent_idx[idx]
                idx = p if p >= 0 else None
        include_set.add(root_idx)
    else:
        # Include full subtree from root.
        stack = [root_idx]
        while stack:
            i = stack.pop()
            if i in include_set:
                continue
            include_set.add(i)
            stack.extend(children.get(i, []))

    exclude_set = {name_to_idx[nm] for nm in exclude_names if nm in name_to_idx}
    if root_idx in exclude_set:
        exclude_set.remove(root_idx)

    out_full_indices: List[int] = []
    out_names: List[str] = []
    out_parent_out_idx: List[int] = []

    def add_node(full_idx: int, parent_out_idx: int):
        if full_idx in exclude_set:
            return
        if full_idx not in include_set:
            return
        out_idx = len(out_full_indices)
        out_full_indices.append(full_idx)
        out_names.append(str(by_idx[full_idx].get("name")))
        out_parent_out_idx.append(parent_out_idx)
        for c in children.get(full_idx, []):
            add_node(c, out_idx)

    add_node(root_idx, -1)
    if not out_full_indices:
        raise ValueError("No joints selected for BVH export")

    return by_idx, out_full_indices, out_names, out_parent_out_idx


def build_mhr_pose_bvh(
    output_bvh_path: str,
    source_joints: Dict[str, Dict],
    profile: Dict,
    mhr_bone_data_path: Optional[str] = None,
) -> Dict[str, object]:
    profile = _normalize_profile(profile)
    order = profile["rotation_order"]
    rot_channels = [f"{axis}rotation" for axis in order]

    mhr_path = Path(mhr_bone_data_path) if mhr_bone_data_path else DEFAULT_MHR_BONE_DATA_PATH
    if not mhr_path.is_file():
        raise FileNotFoundError(f"MHR bone data not found: {mhr_path}")
    mhr_bone_data = _load_json(mhr_path)
    by_idx, full_indices, names, parents_out = _build_skeleton_layout(mhr_bone_data, profile)

    # Source lookup from _infer_pose_common shape:
    # source_joints[name] => {"quat_xyzw_global": [...], "coord": [...]}
    quats_by_name: Dict[str, np.ndarray] = {}
    coords_by_name: Dict[str, np.ndarray] = {}
    q_key = profile["source_quat_key"]
    c_key = profile["source_coord_key"]
    for name, payload in (source_joints or {}).items():
        if not isinstance(payload, dict):
            continue
        q = payload.get(q_key)
        if isinstance(q, (list, tuple)) and len(q) >= 4:
            try:
                quats_by_name[name] = np.asarray([float(q[0]), float(q[1]), float(q[2]), float(q[3])], dtype=np.float64)
            except Exception:
                pass
        c = payload.get(c_key)
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            try:
                coords_by_name[name] = np.asarray([float(c[0]), float(c[1]), float(c[2])], dtype=np.float64)
            except Exception:
                pass

    rest_global: List[np.ndarray] = []
    rest_offset: List[np.ndarray] = []
    for fi in full_indices:
        j = by_idx[fi]
        rst = j.get("rest_pose_state") or []
        if isinstance(rst, list) and len(rst) >= 7:
            q = rst[3:7]
        else:
            q = j.get("prerotation_quaternion_xyzw") or [0.0, 0.0, 0.0, 1.0]
        rest_global.append(_quat_xyzw_to_mat3(q))
        rest_offset.append(_as_vec3(j.get("translation_offset"), (0.0, 0.0, 0.0)))
    rest_global = [np.asarray(m, dtype=np.float64) for m in rest_global]

    pose_global: List[np.ndarray] = []
    missing_source = []
    for i, fi in enumerate(full_indices):
        nm = names[i]
        q = quats_by_name.get(nm)
        if q is None:
            # Keep bind orientation when source joint is missing.
            pose_global.append(rest_global[i].copy())
            missing_source.append(nm)
        else:
            pose_global.append(_quat_xyzw_to_mat3(q))

    pose_local: List[np.ndarray] = []
    rest_local: List[np.ndarray] = []
    for i in range(len(full_indices)):
        p_out = parents_out[i]
        if p_out >= 0:
            pose_local.append(_orthonormalize(pose_global[p_out].T @ pose_global[i]))
            rest_local.append(_orthonormalize(rest_global[p_out].T @ rest_global[i]))
        else:
            pose_local.append(_orthonormalize(pose_global[i]))
            rest_local.append(_orthonormalize(rest_global[i]))

    mode = str(profile.get("rotation_mode", "local_pose")).strip().lower()
    final_local = []
    for i in range(len(full_indices)):
        if mode == "local_delta_from_rest":
            m = _orthonormalize(rest_local[i].T @ pose_local[i])
        elif mode == "rest_pose":
            m = np.eye(3, dtype=np.float64)
        else:
            m = pose_local[i]
        final_local.append(m)

    # Root translation
    root_translation = np.zeros(3, dtype=np.float64)
    root_name = names[0]
    root_mode = str(profile.get("root_translation_mode", "pose_minus_rest")).strip().lower()
    if root_mode in ("pose", "pose_minus_rest"):
        src = coords_by_name.get(root_name)
        if src is not None:
            root_translation = src * float(profile.get("root_translation_scale", 100.0))
            root_translation = root_translation * np.asarray(
                profile.get("root_translation_axis_multipliers", [1.0, -1.0, 1.0]), dtype=np.float64
            )
            if root_mode == "pose_minus_rest":
                root_rest = _as_vec3(by_idx[full_indices[0]].get("rest_pose_state", [0.0, 0.0, 0.0]), (0.0, 0.0, 0.0))
                root_translation = root_translation - root_rest

    # Build hierarchy lines
    lines: List[str] = ["HIERARCHY"]
    children_out = {i: [] for i in range(len(full_indices))}
    for i, p in enumerate(parents_out):
        if p >= 0:
            children_out[p].append(i)

    def write_joint(out_idx: int, indent: int):
        pref = "  " * indent
        nm = names[out_idx]
        if out_idx == 0:
            lines.append(f"{pref}ROOT {nm}")
        else:
            lines.append(f"{pref}JOINT {nm}")
        lines.append(f"{pref}" + "{")
        off = rest_offset[out_idx].copy()
        if out_idx == 0 and bool(profile.get("zero_root_offset", True)):
            off[:] = 0.0
        lines.append(f"{pref}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        if out_idx == 0:
            lines.append(
                f"{pref}  CHANNELS 6 Xposition Yposition Zposition {' '.join(rot_channels)}"
            )
        else:
            lines.append(f"{pref}  CHANNELS 3 {' '.join(rot_channels)}")
        kids = children_out.get(out_idx, [])
        if kids:
            for c in kids:
                write_joint(c, indent + 1)
        else:
            lines.append(f"{pref}  End Site")
            lines.append(f"{pref}  " + "{")
            lines.append(f"{pref}    OFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{pref}  " + "}")
        lines.append(f"{pref}" + "}")

    write_joint(0, 0)

    # Single-frame motion output for still-image pose.
    frame_values: List[float] = []
    for out_idx in range(len(full_indices)):
        if out_idx == 0:
            frame_values.extend([float(root_translation[0]), float(root_translation[1]), float(root_translation[2])])
        e = _mat3_to_euler_deg(final_local[out_idx], order)
        by_axis = {f"{axis}rotation": float(val) for axis, val in zip(order, e)}
        for ch in rot_channels:
            frame_values.append(float(by_axis.get(ch, 0.0)))

    lines.append("MOTION")
    lines.append("Frames: 1")
    lines.append(f"Frame Time: {1.0 / float(profile.get('fps', 30)):.6f}")
    lines.append(" ".join(f"{v:.6f}" for v in frame_values))

    out_path = Path(output_bvh_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "output_bvh_path": str(out_path),
        "profile_name": profile.get("profile_name", "unknown"),
        "mhr_bone_data_path": str(mhr_path),
        "joint_count": len(full_indices),
        "missing_source_joint_count": len(missing_source),
        "missing_source_joint_sample": missing_source[:20],
        "rotation_mode": mode,
        "rotation_order": order,
        "root_joint_name": root_name,
        "root_translation": [float(root_translation[0]), float(root_translation[1]), float(root_translation[2])],
    }
