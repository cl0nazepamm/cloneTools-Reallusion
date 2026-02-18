# -*- coding: utf-8 -*-
"""
PoseSync SAM-3D server.

GPU-only inference endpoint:
- POST /infer_pose_base64
- POST /infer_pose_bvh_base64
- POST /infer_pose_mhr_bvh_base64
- POST /infer_pose_raw_base64
- POST /infer_pose_structured_base64
- POST /infer_pose_max_package_base64
"""

import base64
import datetime
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

try:
    from .bvh_export import build_pose_bvh_from_template
except Exception:
    from bvh_export import build_pose_bvh_from_template

try:
    from .max_pose_export import export_max_pose_package
except Exception:
    from max_pose_export import export_max_pose_package

try:
    from .pose_profiles import (
        build_structured_pose,
        list_profile_names,
        load_profile,
    )
except Exception:
    from pose_profiles import (
        build_structured_pose,
        list_profile_names,
        load_profile,
    )

try:
    from .mhr_bvh_export import (
        build_mhr_pose_bvh,
        list_bvh_profile_names,
        load_bvh_profile,
    )
except Exception:
    from mhr_bvh_export import (
        build_mhr_pose_bvh,
        list_bvh_profile_names,
        load_bvh_profile,
    )


def _register_cuda_dlls():
    """Add CUDA runtime DLL folders for conda/pip installs on Windows."""
    dirs = []
    conda_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(conda_bin):
        dirs.append(conda_bin)
    nvidia_base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if os.path.isdir(nvidia_base):
        for pkg in os.listdir(nvidia_base):
            d = os.path.join(nvidia_base, pkg, "bin")
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*.dll")):
                dirs.append(d)
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass


_register_cuda_dlls()

SAM3D_SOURCE_DIR = Path(__file__).resolve().parent / "sam-3d-body"
if SAM3D_SOURCE_DIR.is_dir():
    sam3d_src = str(SAM3D_SOURCE_DIR)
    if sam3d_src not in sys.path:
        sys.path.insert(0, sam3d_src)

MODEL_REPO = "facebook/sam-3d-body-dinov3"
DEFAULT_LOCAL_MODEL_DIR = str(
    Path(__file__).resolve().parent / "checkpoints" / "sam-3d-body-dinov3"
)
LOCAL_MODEL_DIR = os.environ.get("SAM3D_MODEL_DIR", DEFAULT_LOCAL_MODEL_DIR).strip()
MHR_HIERARCHY_JSON = os.path.join(LOCAL_MODEL_DIR, "assets", "mhr_joint_hierarchy.json")
DEFAULT_BVH_TEMPLATE_CANDIDATES = [
    str(Path(__file__).resolve().parent / "assets" / "Biped_TPose.bvh"),
    str(Path(__file__).resolve().parent / "assets" / "PoseSync_TPose.bvh"),
    str(Path(__file__).resolve().parent / "PoseSync_TPose.bvh"),
    os.environ.get("POSESYNC_BVH_TEMPLATE", "").strip(),
]
GENERATED_BVH_DIR = Path(__file__).resolve().parent / "_generated_bvh"
POSE_PROFILE_DIR = Path(__file__).resolve().parent / "assets" / "pose_profiles"
BVH_PROFILE_DIR = Path(__file__).resolve().parent / "assets" / "bvh_profiles"
MAX_POSE_PACKAGE_DIR = GENERATED_BVH_DIR / "max_pose_packages"
MHR_BONE_DATA_JSON = Path(__file__).resolve().parent / "MHR" / "mhr_bone_data.json"

log = logging.getLogger("PoseSync")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_estimator = None
_joint_names = []
_joint_parents = []

app = Flask(__name__)


# MHR -> UE/CC core chain mapping used by plugin apply stage.
MHR_TO_UE_CORE = {
    "root": "pelvis",
    "c_spine0": "spine_01",
    "c_spine1": "spine_02",
    "c_spine2": "spine_03",
    "c_spine3": "spine_04",
    "c_neck": "neck_01",
    "c_head": "head",
    "l_clavicle": "clavicle_l",
    "l_uparm": "upperarm_l",
    "l_lowarm": "lowerarm_l",
    "l_wrist": "hand_l",
    "r_clavicle": "clavicle_r",
    "r_uparm": "upperarm_r",
    "r_lowarm": "lowerarm_r",
    "r_wrist": "hand_r",
    "l_upleg": "thigh_l",
    "l_lowleg": "calf_l",
    "l_foot": "foot_l",
    "l_ball": "ball_l",
    "r_upleg": "thigh_r",
    "r_lowleg": "calf_r",
    "r_foot": "foot_r",
    "r_ball": "ball_r",
}

LEGACY_UE_CORE_TO_MHR_SOURCES = {
    "pelvis": ["root"],
    "spine_01": ["c_spine0"],
    "spine_02": ["c_spine1"],
    "spine_03": ["c_spine2"],
    "spine_04": ["c_spine3"],
    "neck_01": ["c_neck", "neck"],
    "head": ["c_head", "head"],
    "clavicle_l": ["l_clavicle"],
    "upperarm_l": ["l_uparm"],
    "lowerarm_l": ["l_lowarm"],
    "hand_l": ["l_wrist"],
    "clavicle_r": ["r_clavicle"],
    "upperarm_r": ["r_uparm"],
    "lowerarm_r": ["r_lowarm"],
    "hand_r": ["r_wrist"],
    "thigh_l": ["l_upleg"],
    "calf_l": ["l_lowleg"],
    "foot_l": ["l_foot"],
    "ball_l": ["l_ball"],
    "thigh_r": ["r_upleg"],
    "calf_r": ["r_lowleg"],
    "foot_r": ["r_foot"],
    "ball_r": ["r_ball"],
}


def _assert_cuda_ready():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is required for SAM-3D loading.") from e
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU-only loading failed.")
    idx = torch.cuda.current_device()
    log.info("CUDA ready: %s (device %d)", torch.cuda.get_device_name(idx), idx)


def _load_joint_names():
    global _joint_names, _joint_parents
    if _joint_names:
        return _joint_names
    if os.path.isfile(MHR_HIERARCHY_JSON):
        try:
            data = json.loads(Path(MHR_HIERARCHY_JSON).read_text(encoding="utf-8"))
            data_sorted = sorted(data, key=lambda v: int(v["index"]))
            _joint_names = [x["name"] for x in data_sorted]
            _joint_parents = [int(x.get("parent_index", -1)) for x in data_sorted]
            log.info("Loaded %d joint names from %s", len(_joint_names), MHR_HIERARCHY_JSON)
            return _joint_names
        except Exception as e:
            log.warning("Failed to read hierarchy json: %s", e)
    _joint_names = []
    _joint_parents = []
    return _joint_names


def _mat3_to_quat_xyzw(m):
    """
    Convert 3x3 rotation matrix to quaternion [x, y, z, w].
    """
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
    return [float(x), float(y), float(z), float(w)]


def _decode_image_b64(image_b64):
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _to_json_safe(value, array_mode="full", sample_size=128):
    """
    Convert numpy / tensor-like values into JSON-serializable data.
    array_mode:
      - full: include full array values
      - summary: include dtype/shape and a flat sample only
    """
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()

    # Torch tensor-like support without importing torch globally.
    if hasattr(value, "detach") and callable(getattr(value, "detach", None)):
        try:
            value = value.detach().cpu().numpy()
        except Exception:
            return str(value)

    if isinstance(value, np.ndarray):
        if array_mode == "summary":
            flat = value.reshape(-1)
            n = max(0, int(sample_size))
            sample = flat[:n].tolist() if n > 0 else []
            return {
                "dtype": str(value.dtype),
                "shape": [int(x) for x in value.shape],
                "sample_flat": sample,
                "sample_count": len(sample),
                "total_count": int(value.size),
                "truncated": int(value.size) > len(sample),
            }
        return value.tolist()

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v, array_mode=array_mode, sample_size=sample_size) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v, array_mode=array_mode, sample_size=sample_size) for v in value]

    try:
        return value.tolist()
    except Exception:
        return str(value)


def _serialize_sam3d_person(person, array_mode="full", sample_size=128):
    if not isinstance(person, dict):
        return _to_json_safe(person, array_mode=array_mode, sample_size=sample_size)
    out = {}
    for k in sorted(person.keys()):
        out[k] = _to_json_safe(person[k], array_mode=array_mode, sample_size=sample_size)
    return out


def _pick_first_existing_joint(joints, names):
    for n in names:
        v = joints.get(n)
        if v is not None:
            return n, v
    return None, None


def _build_legacy_ue_pose_from_joints(joints):
    ue_pose = {}
    for ue_name, source_candidates in LEGACY_UE_CORE_TO_MHR_SOURCES.items():
        _, src = _pick_first_existing_joint(joints, source_candidates)
        if src is not None:
            ue_pose[ue_name] = src["quat_xyzw_local"]
    return ue_pose


def _pick_bvh_template_path():
    for p in DEFAULT_BVH_TEMPLATE_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return ""


def _to_local_rot_mats(global_rots):
    """
    Convert global rotation matrices to parent-local matrices using hierarchy.
    """
    n = len(global_rots)
    local = np.zeros_like(global_rots)
    parents = _joint_parents if len(_joint_parents) >= n else [-1] * n
    for i in range(n):
        p = int(parents[i]) if i < len(parents) else -1
        g = global_rots[i]
        if p < 0 or p >= n:
            local[i] = g
        else:
            # For pure rotation matrices, inverse(parent) = transpose(parent).
            local[i] = np.matmul(global_rots[p].T, g)
    return local


def load_sam3d_model(local_model_dir=None):
    """Load SAM-3D estimator and keep it in memory."""
    global _estimator
    if _estimator is not None:
        return _estimator

    _assert_cuda_ready()
    model_dir = (local_model_dir or LOCAL_MODEL_DIR or "").strip()

    if model_dir:
        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        ckpt_candidates = [
            os.path.join(model_dir, "model.ckpt"),
            os.path.join(model_dir, "model.safetensors"),
        ]
        ckpt_candidates.extend(sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))))
        ckpt = next((p for p in ckpt_candidates if os.path.isfile(p)), "")
        mhr = os.path.join(model_dir, "assets", "mhr_model.pt")

        if not ckpt:
            raise FileNotFoundError(
                "Missing checkpoint. Expected model.ckpt or *.safetensors in local model dir."
            )
        if not os.path.isfile(mhr):
            raise FileNotFoundError(f"Missing MHR asset: {mhr}")

        log.info("Loading SAM-3D from local dir: %s", model_dir)
        log.info("Using checkpoint: %s", ckpt)
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=ckpt,
            mhr_path=mhr,
            device="cuda",
        )
        _estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
    else:
        from notebook.utils import setup_sam_3d_body

        log.info("Loading SAM-3D from Hugging Face repo: %s", MODEL_REPO)
        _estimator = setup_sam_3d_body(hf_repo_id=MODEL_REPO)

    _load_joint_names()
    log.info("SAM-3D model loaded.")
    return _estimator


@app.get("/health")
def health():
    profiles = list_profile_names(POSE_PROFILE_DIR)
    bvh_profiles = list_bvh_profile_names(BVH_PROFILE_DIR)
    return jsonify(
        {
            "ok": True,
            "model_loaded": _estimator is not None,
            "model_dir": LOCAL_MODEL_DIR,
            "joint_name_count": len(_joint_names),
            "profile_dir": str(POSE_PROFILE_DIR),
            "profiles": profiles,
            "bvh_profile_dir": str(BVH_PROFILE_DIR),
            "bvh_profiles": bvh_profiles,
            "endpoints": [
                "/health",
                "/pose_profiles",
                "/bvh_profiles",
                "/infer_pose_base64",
                "/infer_pose_bvh_base64",
                "/infer_pose_mhr_bvh_base64",
                "/mhr_raw_to_bvh",
                "/infer_pose_raw_base64",
                "/infer_pose_structured_base64",
                "/infer_pose_max_package_base64",
            ],
        }
    )


@app.get("/pose_profiles")
def pose_profiles():
    return jsonify(
        {
            "ok": True,
            "profile_dir": str(POSE_PROFILE_DIR),
            "profiles": list_profile_names(POSE_PROFILE_DIR),
        }
    )


@app.get("/bvh_profiles")
def bvh_profiles():
    return jsonify(
        {
            "ok": True,
            "profile_dir": str(BVH_PROFILE_DIR),
            "profiles": list_bvh_profile_names(BVH_PROFILE_DIR),
        }
    )


@app.post("/infer_pose_base64")
def infer_pose_base64():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image_b64", "")
    person_index = int(payload.get("person_index", 0))
    include_raw = bool(payload.get("include_raw", False))
    raw_array_mode = str(payload.get("raw_array_mode", "summary")).strip().lower() or "summary"
    raw_sample_size = int(payload.get("raw_sample_size", 128))

    if not image_b64:
        return jsonify({"ok": False, "error": "Missing image_b64"}), 400

    try:
        result = _infer_pose_common(
            image_b64=image_b64,
            person_index=person_index,
            include_raw=include_raw,
            raw_array_mode=raw_array_mode,
            raw_sample_size=raw_sample_size,
        )
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception as e:
        log.exception("infer_pose_base64 failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/infer_pose_bvh_base64")
def infer_pose_bvh_base64():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image_b64", "")
    person_index = int(payload.get("person_index", 0))
    template_bvh_path = str(payload.get("template_bvh_path") or "").strip()
    output_bvh_path = str(payload.get("output_bvh_path") or "").strip()

    if not image_b64:
        return jsonify({"ok": False, "error": "Missing image_b64"}), 400

    try:
        result = _infer_pose_common(
            image_b64=image_b64,
            person_index=person_index,
            include_raw=False,
        )

        joint_quats = {}
        joint_coords = {}
        for mhr_name, data in result.get("joints", {}).items():
            q = data.get("quat_xyzw_local")
            if q is not None:
                joint_quats[mhr_name] = q
            c = data.get("coord")
            if c is not None:
                joint_coords[mhr_name] = c
        for ue_name, quat in result.get("ue_pose", {}).items():
            if quat is not None:
                joint_quats[ue_name] = quat

        if not template_bvh_path:
            template_bvh_path = _pick_bvh_template_path()
        if not template_bvh_path:
            return jsonify({"ok": False, "error": "No BVH template found."}), 500

        if not output_bvh_path:
            GENERATED_BVH_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_bvh_path = str(GENERATED_BVH_DIR / f"posesync_pose_{ts}.bvh")

        bvh_stats = build_pose_bvh_from_template(
            template_bvh_path=template_bvh_path,
            output_bvh_path=output_bvh_path,
            joint_quats_xyzw=joint_quats,
            joint_coords_xyz=joint_coords,
        )

        result["bvh"] = bvh_stats
        result["bvh_path"] = bvh_stats["output_bvh_path"]
        result["template_bvh_path"] = bvh_stats["template_bvh_path"]
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception as e:
        log.exception("infer_pose_bvh_base64 failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/infer_pose_mhr_bvh_base64")
def infer_pose_mhr_bvh_base64():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image_b64", "")
    person_index = int(payload.get("person_index", 0))
    output_bvh_path = str(payload.get("output_bvh_path") or "").strip()
    bvh_profile_name = str(payload.get("bvh_profile_name", "mhr_raw_pose")).strip() or "mhr_raw_pose"
    bvh_profile_inline = payload.get("bvh_profile")
    mhr_bone_data_path = str(payload.get("mhr_bone_data_path") or "").strip()

    if not image_b64:
        return jsonify({"ok": False, "error": "Missing image_b64"}), 400

    try:
        base = _infer_pose_common(
            image_b64=image_b64,
            person_index=person_index,
            include_raw=False,
        )
        profile = load_bvh_profile(
            profile_dir=BVH_PROFILE_DIR,
            profile_name=bvh_profile_name,
            profile_inline=bvh_profile_inline,
        )

        if not output_bvh_path:
            GENERATED_BVH_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_bvh_path = str(GENERATED_BVH_DIR / f"posesync_mhr_pose_{ts}.bvh")

        bvh_stats = build_mhr_pose_bvh(
            output_bvh_path=output_bvh_path,
            source_joints=base.get("joints", {}),
            profile=profile,
            mhr_bone_data_path=mhr_bone_data_path or str(MHR_BONE_DATA_JSON),
        )
        return jsonify(
            {
                "ok": True,
                "people_detected": int(base.get("people_detected", 0)),
                "person_index": int(base.get("person_index", 0)),
                "source_joint_count": int(base.get("joint_count", 0)),
                "profile_name": profile.get("profile_name"),
                "profile_source": profile.get("profile_source"),
                "bvh": bvh_stats,
                "bvh_path": bvh_stats.get("output_bvh_path"),
            }
        )
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception as e:
        log.exception("infer_pose_mhr_bvh_base64 failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/mhr_raw_to_bvh")
def mhr_raw_to_bvh():
    payload = request.get_json(silent=True) or {}
    source_joints = payload.get("source_joints", {})
    output_bvh_path = str(payload.get("output_bvh_path") or "").strip()
    bvh_profile_name = str(payload.get("bvh_profile_name", "mhr_raw_pose")).strip() or "mhr_raw_pose"
    bvh_profile_inline = payload.get("bvh_profile")
    mhr_bone_data_path = str(payload.get("mhr_bone_data_path") or "").strip()

    if not isinstance(source_joints, dict) or not source_joints:
        return jsonify({"ok": False, "error": "Missing source_joints"}), 400

    try:
        profile = load_bvh_profile(
            profile_dir=BVH_PROFILE_DIR,
            profile_name=bvh_profile_name,
            profile_inline=bvh_profile_inline,
        )
        if not output_bvh_path:
            GENERATED_BVH_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_bvh_path = str(GENERATED_BVH_DIR / f"posesync_mhr_raw_{ts}.bvh")

        bvh_stats = build_mhr_pose_bvh(
            output_bvh_path=output_bvh_path,
            source_joints=source_joints,
            profile=profile,
            mhr_bone_data_path=mhr_bone_data_path or str(MHR_BONE_DATA_JSON),
        )
        return jsonify(
            {
                "ok": True,
                "profile_name": profile.get("profile_name"),
                "profile_source": profile.get("profile_source"),
                "bvh": bvh_stats,
                "bvh_path": bvh_stats.get("output_bvh_path"),
            }
        )
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception as e:
        log.exception("mhr_raw_to_bvh failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/infer_pose_raw_base64")
def infer_pose_raw_base64():
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image_b64", "")
    person_index = int(payload.get("person_index", 0))
    include_all_people = bool(payload.get("include_all_people", False))
    raw_array_mode = str(payload.get("raw_array_mode", "full")).strip().lower() or "full"
    raw_sample_size = int(payload.get("raw_sample_size", 128))

    if not image_b64:
        return jsonify({"ok": False, "error": "Missing image_b64"}), 400

    try:
        estimator = load_sam3d_model()
        img_rgb = _decode_image_b64(image_b64)
        outputs = estimator.process_one_image(img_rgb, inference_type="body")
        if not outputs:
            raise RuntimeError("No person detected")

        if person_index < 0 or person_index >= len(outputs):
            person_index = 0

        selected = outputs[person_index]
        response = {
            "ok": True,
            "people_detected": len(outputs),
            "person_index": person_index,
            "inference_type": "body",
            "raw_array_mode": raw_array_mode,
            "raw_sample_size": raw_sample_size,
            "sam3d_raw": _serialize_sam3d_person(
                selected, array_mode=raw_array_mode, sample_size=raw_sample_size
            ),
        }
        if include_all_people:
            response["sam3d_raw_all_people"] = [
                _serialize_sam3d_person(p, array_mode=raw_array_mode, sample_size=raw_sample_size)
                for p in outputs
            ]
        return jsonify(response)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception as e:
        log.exception("infer_pose_raw_base64 failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/infer_pose_structured_base64")
def infer_pose_structured_base64():
    """
    New clean-start endpoint:
    - source joints from SAM-3D
    - profile-driven target bone output
    """
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image_b64", "")
    person_index = int(payload.get("person_index", 0))
    profile_name = str(payload.get("profile_name", "source_passthrough")).strip() or "source_passthrough"
    profile_inline = payload.get("profile")
    include_source_joints = bool(payload.get("include_source_joints", False))
    include_raw = bool(payload.get("include_raw", False))
    raw_array_mode = str(payload.get("raw_array_mode", "summary")).strip().lower() or "summary"
    raw_sample_size = int(payload.get("raw_sample_size", 128))

    if not image_b64:
        return jsonify({"ok": False, "error": "Missing image_b64"}), 400

    try:
        base = _infer_pose_common(
            image_b64=image_b64,
            person_index=person_index,
            include_raw=include_raw,
            raw_array_mode=raw_array_mode,
            raw_sample_size=raw_sample_size,
        )
        profile = load_profile(
            profile_dir=POSE_PROFILE_DIR,
            profile_name=profile_name,
            profile_inline=profile_inline,
        )
        source_meta = {
            "people_detected": int(base.get("people_detected", 0)),
            "person_index": int(base.get("person_index", 0)),
            "joint_count": int(base.get("joint_count", 0)),
            "joint_name_count": int(len(base.get("joints", {}))),
        }
        structured = build_structured_pose(
            source_joints=base.get("joints", {}),
            profile=profile,
            source_meta=source_meta,
        )

        out = {
            "ok": True,
            "people_detected": base.get("people_detected", 0),
            "person_index": base.get("person_index", 0),
            "source_joint_count": base.get("joint_count", 0),
            "profile_name": profile.get("profile_name"),
            "profile_version": profile.get("profile_version"),
            "structured_pose": structured,
        }
        if include_source_joints:
            out["source_joints"] = base.get("joints", {})
        if include_raw:
            out["sam3d_raw"] = base.get("sam3d_raw")
            out["raw_array_mode"] = base.get("raw_array_mode")
            out["raw_sample_size"] = base.get("raw_sample_size")
        return jsonify(out)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception as e:
        log.exception("infer_pose_structured_base64 failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/infer_pose_max_package_base64")
def infer_pose_max_package_base64():
    """
    Build a 3ds Max-friendly package:
    - structured pose JSON
    - MAXScript apply script
    - manifest
    """
    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image_b64", "")
    person_index = int(payload.get("person_index", 0))
    profile_name = str(payload.get("profile_name", "max_biped_basic")).strip() or "max_biped_basic"
    profile_inline = payload.get("profile")
    include_source_joints = bool(payload.get("include_source_joints", True))
    include_raw = bool(payload.get("include_raw", False))
    raw_array_mode = str(payload.get("raw_array_mode", "summary")).strip().lower() or "summary"
    raw_sample_size = int(payload.get("raw_sample_size", 128))
    package_name = str(payload.get("package_name", "")).strip()
    output_root = str(payload.get("output_root", "")).strip()
    apply_translation = bool(payload.get("apply_translation", True))

    if not image_b64:
        return jsonify({"ok": False, "error": "Missing image_b64"}), 400

    try:
        base = _infer_pose_common(
            image_b64=image_b64,
            person_index=person_index,
            include_raw=include_raw,
            raw_array_mode=raw_array_mode,
            raw_sample_size=raw_sample_size,
        )
        profile = load_profile(
            profile_dir=POSE_PROFILE_DIR,
            profile_name=profile_name,
            profile_inline=profile_inline,
        )
        source_meta = {
            "people_detected": int(base.get("people_detected", 0)),
            "person_index": int(base.get("person_index", 0)),
            "joint_count": int(base.get("joint_count", 0)),
            "joint_name_count": int(len(base.get("joints", {}))),
        }
        structured = build_structured_pose(
            source_joints=base.get("joints", {}),
            profile=profile,
            source_meta=source_meta,
        )
        package = export_max_pose_package(
            structured_pose=structured,
            output_root=output_root or MAX_POSE_PACKAGE_DIR,
            package_name=package_name,
            include_source_joints=include_source_joints,
            source_joints=base.get("joints", {}),
            include_raw=include_raw,
            raw_payload=base.get("sam3d_raw"),
            apply_translation=apply_translation,
        )
        return jsonify(
            {
                "ok": True,
                "people_detected": base.get("people_detected", 0),
                "person_index": base.get("person_index", 0),
                "source_joint_count": base.get("joint_count", 0),
                "profile_name": profile.get("profile_name"),
                "profile_version": profile.get("profile_version"),
                "package": package,
            }
        )
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 422
    except Exception as e:
        log.exception("infer_pose_max_package_base64 failed")
        return jsonify({"ok": False, "error": str(e)}), 500


def _infer_pose_common(
    image_b64,
    person_index=0,
    include_raw=False,
    raw_array_mode="summary",
    raw_sample_size=128,
):
    estimator = load_sam3d_model()
    img_rgb = _decode_image_b64(image_b64)
    outputs = estimator.process_one_image(img_rgb, inference_type="body")
    if not outputs:
        raise RuntimeError("No person detected")
    if person_index < 0 or person_index >= len(outputs):
        person_index = 0
    person = outputs[person_index]

    rots_global = np.asarray(person["pred_global_rots"], dtype=np.float64)
    coords = np.asarray(person["pred_joint_coords"], dtype=np.float64)
    names = _load_joint_names()

    joint_count = int(min(len(rots_global), len(coords)))
    if names and len(names) >= joint_count:
        joint_names = names[:joint_count]
    else:
        joint_names = [f"joint_{i}" for i in range(joint_count)]

    rots_global = rots_global[:joint_count]
    rots_local = _to_local_rot_mats(rots_global)

    joints = {}
    for idx, jn in enumerate(joint_names):
        quat_global = _mat3_to_quat_xyzw(rots_global[idx])
        quat_local = _mat3_to_quat_xyzw(rots_local[idx])
        joints[jn] = {
            "index": idx,
            "quat_xyzw_global": quat_global,
            "quat_xyzw_local": quat_local,
            "coord": [float(coords[idx, 0]), float(coords[idx, 1]), float(coords[idx, 2])],
        }

    # Legacy compatibility output for existing plugin paths.
    ue_pose = _build_legacy_ue_pose_from_joints(joints)

    result = {
        "ok": True,
        "people_detected": len(outputs),
        "person_index": person_index,
        "joint_count": joint_count,
        "joints": joints,
        "ue_pose": ue_pose,
        "rotation_space": "local_parent",
        "mhr_to_ue_core": MHR_TO_UE_CORE,
        "legacy_ue_core_sources": LEGACY_UE_CORE_TO_MHR_SOURCES,
    }
    if include_raw:
        result["sam3d_raw"] = _serialize_sam3d_person(
            person,
            array_mode=raw_array_mode,
            sample_size=raw_sample_size,
        )
        result["raw_array_mode"] = raw_array_mode
        result["raw_sample_size"] = raw_sample_size
    return result


def get_loaded_model():
    return _estimator


if __name__ == "__main__":
    load_sam3d_model()
    log.info("PoseSync server listening on http://127.0.0.1:8642")
    app.run(host="127.0.0.1", port=8642, debug=False, threaded=False)
