# -*- coding: utf-8 -*-
"""
Profile-driven pose mapping utilities.

This module turns source joints (SAM-3D / MHR) into target-bone outputs
using a validated JSON profile. Profiles define structure and transforms
without hardcoding bone names in Python.
"""

import copy
import json
import math
from pathlib import Path


PROFILE_SCHEMA_VERSION = "1.0.0"


def list_profile_names(profile_dir):
    p = Path(profile_dir)
    if not p.is_dir():
        return []
    names = []
    for fp in sorted(p.glob("*.json")):
        names.append(fp.stem)
    return names


def _to_float3(value, default=None):
    if default is None:
        default = [0.0, 0.0, 0.0]
    if value is None:
        return [float(default[0]), float(default[1]), float(default[2])]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Expected float3 list [x, y, z].")
    return [float(value[0]), float(value[1]), float(value[2])]


def _normalize_bone_entry(raw):
    if not isinstance(raw, dict):
        raise ValueError("Each profile bone entry must be an object.")

    target = str(raw.get("target", "")).strip()
    if not target:
        raise ValueError("Bone entry is missing 'target'.")

    sources_raw = raw.get("sources", [])
    if isinstance(sources_raw, str):
        sources = [sources_raw]
    elif isinstance(sources_raw, list):
        sources = [str(x).strip() for x in sources_raw if str(x).strip()]
    else:
        raise ValueError("Bone entry 'sources' must be string or list.")
    if not sources:
        raise ValueError("Bone entry '%s' has no source candidates." % target)

    return {
        "target": target,
        "parent": str(raw.get("parent", "")).strip() or None,
        "sources": sources,
        "rotation_offset_deg": _to_float3(raw.get("rotation_offset_deg", [0.0, 0.0, 0.0])),
        "rotation_offset_quat_xyzw": _quat_identity_if_missing(raw.get("rotation_offset_quat_xyzw")),
        "rotation_offset_mode": str(raw.get("rotation_offset_mode", "post")).strip().lower() or "post",
        "translation_offset": _to_float3(raw.get("translation_offset", [0.0, 0.0, 0.0])),
        "copy_translation": bool(raw.get("copy_translation", False)),
        "enabled": bool(raw.get("enabled", True)),
    }


def normalize_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("Profile must be a JSON object.")

    out = {
        "schema_version": str(profile.get("schema_version", PROFILE_SCHEMA_VERSION)),
        "profile_name": str(profile.get("profile_name", "unnamed_profile")).strip() or "unnamed_profile",
        "profile_version": str(profile.get("profile_version", "1.0.0")).strip() or "1.0.0",
        "description": str(profile.get("description", "")).strip(),
        "rotation_space": str(profile.get("rotation_space", "local_parent")).strip() or "local_parent",
        "output_coordinate_system": profile.get("output_coordinate_system", {}),
        "passthrough_source_joints": bool(profile.get("passthrough_source_joints", False)),
        "global_transform": {
            "translation_scale": float(
                profile.get("global_transform", {}).get("translation_scale", 1.0)
            ),
            "translation_offset": _to_float3(
                profile.get("global_transform", {}).get("translation_offset", [0.0, 0.0, 0.0])
            ),
            "rotation_offset_deg": _to_float3(
                profile.get("global_transform", {}).get("rotation_offset_deg", [0.0, 0.0, 0.0])
            ),
            "rotation_offset_quat_xyzw": _quat_identity_if_missing(
                profile.get("global_transform", {}).get("rotation_offset_quat_xyzw")
            ),
            "rotation_offset_mode": str(
                profile.get("global_transform", {}).get("rotation_offset_mode", "post")
            ).strip().lower() or "post",
        },
        "source_rotation_offsets_quat_xyzw": {},
        "source_rotation_offset_mode": str(
            profile.get("source_rotation_offset_mode", "post")
        ).strip().lower() or "post",
    }

    if out["rotation_space"] not in ("local_parent", "global"):
        raise ValueError("rotation_space must be 'local_parent' or 'global'.")
    if out["global_transform"]["rotation_offset_mode"] not in ("pre", "post"):
        raise ValueError("global_transform.rotation_offset_mode must be 'pre' or 'post'.")
    if out["source_rotation_offset_mode"] not in ("pre", "post"):
        raise ValueError("source_rotation_offset_mode must be 'pre' or 'post'.")

    src_q = profile.get("source_rotation_offsets_quat_xyzw", {})
    if src_q is None:
        src_q = {}
    if not isinstance(src_q, dict):
        raise ValueError("source_rotation_offsets_quat_xyzw must be an object.")
    for k, v in src_q.items():
        out["source_rotation_offsets_quat_xyzw"][str(k)] = _quat_identity_if_missing(v)

    cs = out["output_coordinate_system"]
    if not isinstance(cs, dict):
        raise ValueError("output_coordinate_system must be an object.")
    out["output_coordinate_system"] = {
        "up": str(cs.get("up", "Y")),
        "forward": str(cs.get("forward", "Z")),
        "handedness": str(cs.get("handedness", "right")),
        "units": str(cs.get("units", "meters")),
    }

    bones_raw = profile.get("bones", [])
    if bones_raw is None:
        bones_raw = []
    if not isinstance(bones_raw, list):
        raise ValueError("bones must be an array.")
    bones = []
    seen_targets = set()
    for item in bones_raw:
        b = _normalize_bone_entry(item)
        if b["rotation_offset_mode"] not in ("pre", "post"):
            raise ValueError("Bone '%s' has invalid rotation_offset_mode." % b["target"])
        t_key = b["target"].lower()
        if t_key in seen_targets:
            raise ValueError("Duplicate target bone in profile: %s" % b["target"])
        seen_targets.add(t_key)
        bones.append(b)
    out["bones"] = bones

    if not out["passthrough_source_joints"] and not out["bones"]:
        raise ValueError("Profile must define bones or set passthrough_source_joints=true.")

    return out


def load_profile(profile_dir, profile_name=None, profile_inline=None):
    if profile_inline is not None:
        return normalize_profile(copy.deepcopy(profile_inline))

    p_dir = Path(profile_dir)
    if not profile_name:
        profile_name = "source_passthrough"
    p_file = p_dir / ("%s.json" % str(profile_name).strip())
    if not p_file.is_file():
        names = list_profile_names(p_dir)
        raise FileNotFoundError(
            "Profile not found: %s. Available profiles: %s"
            % (profile_name, ", ".join(names) if names else "none")
        )
    raw = json.loads(p_file.read_text(encoding="utf-8"))
    return normalize_profile(raw)


def _quat_normalize_xyzw(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    inv = 1.0 / n
    return [x * inv, y * inv, z * inv, w * inv]


def _quat_identity_if_missing(q):
    if q is None:
        return [0.0, 0.0, 0.0, 1.0]
    if not isinstance(q, (list, tuple)) or len(q) != 4:
        raise ValueError("Quaternion must be [x, y, z, w].")
    return _quat_normalize_xyzw([float(q[0]), float(q[1]), float(q[2]), float(q[3])])


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quat_from_euler_xyz_deg(euler_deg):
    ex, ey, ez = [math.radians(float(v)) for v in euler_deg]
    hx, hy, hz = ex * 0.5, ey * 0.5, ez * 0.5
    sx, cx = math.sin(hx), math.cos(hx)
    sy, cy = math.sin(hy), math.cos(hy)
    sz, cz = math.sin(hz), math.cos(hz)

    qx = [sx, 0.0, 0.0, cx]
    qy = [0.0, sy, 0.0, cy]
    qz = [0.0, 0.0, sz, cz]
    q = _quat_mul_xyzw(_quat_mul_xyzw(qx, qy), qz)
    return _quat_normalize_xyzw(q)


def _apply_rotation_offset(src_q_xyzw, offset_deg_xyz, mode):
    src = _quat_normalize_xyzw(src_q_xyzw)
    off = _quat_from_euler_xyz_deg(offset_deg_xyz)
    if str(mode).lower() == "pre":
        return _quat_normalize_xyzw(_quat_mul_xyzw(off, src))
    return _quat_normalize_xyzw(_quat_mul_xyzw(src, off))


def _apply_quat_offset(src_q_xyzw, offset_q_xyzw, mode):
    src = _quat_normalize_xyzw(src_q_xyzw)
    off = _quat_normalize_xyzw(offset_q_xyzw)
    if str(mode).lower() == "pre":
        return _quat_normalize_xyzw(_quat_mul_xyzw(off, src))
    return _quat_normalize_xyzw(_quat_mul_xyzw(src, off))


def _pick_source_joint(source_joints, sources):
    for name in sources:
        if name in source_joints:
            return name, source_joints[name]
    return None, None


def _build_bone_record(target, parent, source_name, source_data, profile, bone_cfg):
    gt = profile["global_transform"]
    out = {
        "target": target,
        "parent": parent,
        "source": source_name,
        "found": source_data is not None,
        "rotation_space": profile["rotation_space"],
        "rotation": None,
        "translation": None,
    }
    if source_data is None:
        return out

    src_local = source_data.get("quat_xyzw_local", [0.0, 0.0, 0.0, 1.0])
    src_global = source_data.get("quat_xyzw_global", [0.0, 0.0, 0.0, 1.0])
    src_rot = src_local if profile["rotation_space"] == "local_parent" else src_global
    src_off = profile.get("source_rotation_offsets_quat_xyzw", {}).get(source_name)
    if src_off is not None:
        src_rot = _apply_quat_offset(
            src_rot,
            src_off,
            profile.get("source_rotation_offset_mode", "post"),
        )
    q = _apply_rotation_offset(src_rot, gt["rotation_offset_deg"], gt["rotation_offset_mode"])
    q = _apply_quat_offset(q, gt["rotation_offset_quat_xyzw"], gt["rotation_offset_mode"])
    q = _apply_rotation_offset(q, bone_cfg["rotation_offset_deg"], bone_cfg["rotation_offset_mode"])
    q = _apply_quat_offset(q, bone_cfg["rotation_offset_quat_xyzw"], bone_cfg["rotation_offset_mode"])

    out["rotation"] = {
        "quat_xyzw": _quat_normalize_xyzw(q),
        "global_offset_deg": [float(v) for v in gt["rotation_offset_deg"]],
        "global_offset_quat_xyzw": [float(v) for v in gt["rotation_offset_quat_xyzw"]],
        "bone_offset_deg": [float(v) for v in bone_cfg["rotation_offset_deg"]],
        "bone_offset_quat_xyzw": [float(v) for v in bone_cfg["rotation_offset_quat_xyzw"]],
        "global_offset_mode": gt["rotation_offset_mode"],
        "bone_offset_mode": bone_cfg["rotation_offset_mode"],
        "source_offset_quat_xyzw": [float(v) for v in src_off] if src_off is not None else None,
        "source_offset_mode": profile.get("source_rotation_offset_mode", "post"),
    }

    if bone_cfg.get("copy_translation", False):
        src_t = source_data.get("coord", [0.0, 0.0, 0.0])
        tx = float(src_t[0]) * gt["translation_scale"] + gt["translation_offset"][0] + bone_cfg["translation_offset"][0]
        ty = float(src_t[1]) * gt["translation_scale"] + gt["translation_offset"][1] + bone_cfg["translation_offset"][1]
        tz = float(src_t[2]) * gt["translation_scale"] + gt["translation_offset"][2] + bone_cfg["translation_offset"][2]
        out["translation"] = {
            "xyz": [tx, ty, tz],
            "copied_from_source": True,
            "scale": float(gt["translation_scale"]),
        }
    else:
        out["translation"] = {
            "xyz": [float(v) for v in bone_cfg["translation_offset"]],
            "copied_from_source": False,
            "scale": float(gt["translation_scale"]),
        }
    return out


def build_structured_pose(source_joints, profile, source_meta=None):
    if source_meta is None:
        source_meta = {}
    if not isinstance(source_joints, dict):
        raise ValueError("source_joints must be dict.")

    bones_cfg = list(profile["bones"])
    if profile.get("passthrough_source_joints", False):
        ordered = sorted(
            source_joints.items(),
            key=lambda kv: int(kv[1].get("index", 10**9)),
        )
        bones_cfg = []
        for src_name, _ in ordered:
            bones_cfg.append(
                {
                    "target": src_name,
                    "parent": None,
                    "sources": [src_name],
                    "rotation_offset_deg": [0.0, 0.0, 0.0],
                    "rotation_offset_mode": "post",
                    "translation_offset": [0.0, 0.0, 0.0],
                    "copy_translation": True,
                    "enabled": True,
                }
            )

    bones_out = []
    missing = []
    for b in bones_cfg:
        if not b.get("enabled", True):
            continue
        source_name, source_data = _pick_source_joint(source_joints, b["sources"])
        if source_data is None:
            missing.append(b["target"])
        rec = _build_bone_record(
            target=b["target"],
            parent=b.get("parent"),
            source_name=source_name,
            source_data=source_data,
            profile=profile,
            bone_cfg=b,
        )
        bones_out.append(rec)

    mapped_count = sum(1 for b in bones_out if b.get("found"))

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_name": profile["profile_name"],
        "profile_version": profile["profile_version"],
        "description": profile.get("description", ""),
        "rotation_space": profile["rotation_space"],
        "output_coordinate_system": profile["output_coordinate_system"],
        "global_transform": profile["global_transform"],
        "source_meta": source_meta,
        "bones": bones_out,
        "stats": {
            "requested_bone_count": len(bones_out),
            "mapped_bone_count": int(mapped_count),
            "missing_bones": missing,
            "missing_bone_count": len(missing),
        },
    }
