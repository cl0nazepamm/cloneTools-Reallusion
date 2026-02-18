# -*- coding: utf-8 -*-
"""
PoseSync: single-image pose apply via SAM-3D -> BVH.
"""

rl_plugin_info = {"ap": "iClone", "ap_version": "8.0"}

import base64
import datetime
import json
import math
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

import RLPy
from PySide2 import QtWidgets, QtCore
from shiboken2 import wrapInstance


SERVER_BASE = "http://127.0.0.1:8642"
POSE_ENDPOINT = SERVER_BASE + "/infer_pose_base64"
POSE_BVH_ENDPOINT = SERVER_BASE + "/infer_pose_bvh_base64"
POSE_MHR_BVH_ENDPOINT = SERVER_BASE + "/infer_pose_mhr_bvh_base64"
POSE_RAW_ENDPOINT = SERVER_BASE + "/infer_pose_raw_base64"
HEALTH_ENDPOINT = SERVER_BASE + "/health"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(BASE_DIR, "server")
ASSETS_DIR = os.path.join(SERVER_DIR, "assets")
ASSETS_DATA_DIR = os.path.join(ASSETS_DIR, "data")
GENERATED_BVH_DIR = os.path.join(SERVER_DIR, "_generated_bvh")
DEBUG_BVH_PATH = os.path.join(GENERATED_BVH_DIR, "posesync_latest.bvh")
DEBUG_RAW_JSON_PATH = os.path.join(GENERATED_BVH_DIR, "sam3d_raw_latest.json")
TEMPLATE_BVH_CANDIDATES = [
    os.path.join(ASSETS_DATA_DIR, "MHR_ReallusionExport.bvh"),
    os.path.join(ASSETS_DIR, "MHR_ReallusionExport.bvh"),
    os.path.join(ASSETS_DIR, "Biped_TPose.bvh"),
    os.path.join(ASSETS_DIR, "PoseSync_TPose.bvh"),
    os.path.join(SERVER_DIR, "PoseSync_TPose.bvh"),
]
MOTION_PROFILE_CANDIDATES = [
    os.path.join(ASSETS_DATA_DIR, "MHR.3dxProfile"),
    os.path.join(ASSETS_DIR, "MHR.3dxProfile"),
]
SERVER_SCRIPT = os.path.join(BASE_DIR, "server", "pose_server.py")
SERVER_LOG = os.path.join(BASE_DIR, "server", "pose_server_runtime.log")
PLUGIN_LOG = os.path.join(BASE_DIR, "PoseSync_runtime.log")
_pose_dock_ui = {}


def _safe_call(obj, method_name, *args):
    fn = getattr(obj, method_name, None)
    if callable(fn):
        try:
            return fn(*args)
        except Exception:
            return None
    return None


def _invoke(obj, method_name, *args):
    fn = getattr(obj, method_name, None)
    if not callable(fn):
        return False, None
    try:
        return True, fn(*args)
    except Exception as e:
        _log("Invoke failed: %s.%s args=%d err=%s" % (type(obj).__name__, method_name, len(args), str(e)))
        return False, None


def _get_prop(obj, *names):
    for n in names:
        try:
            v = getattr(obj, n)
            if v is not None:
                return v, n
        except Exception:
            continue
    return None, ""


def _set_prop(obj, name, value):
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _ret_is_success(ret):
    if ret is None:
        return True
    if isinstance(ret, bool):
        return ret
    try:
        status = getattr(RLPy, "RStatus", None)
        if status is not None:
            failure = getattr(status, "Failure", None)
            success = getattr(status, "Success", None)
            if failure is not None and ret == failure:
                return False
            if success is not None and ret == success:
                return True
    except Exception:
        pass
    s = str(ret).lower()
    if "failure" in s:
        return False
    return True


def _pick_template_bvh_path():
    for p in TEMPLATE_BVH_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return ""


def _pick_motion_profile_path():
    for p in MOTION_PROFILE_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return ""


def _load_hik_profile_on_avatar(avatar, profile_path):
    if avatar is None:
        return False, "no-avatar"
    if not profile_path:
        return False, "no-profile-path"
    if not os.path.isfile(profile_path):
        return False, "profile-not-found"

    fn = getattr(avatar, "LoadHikProfile", None)
    if not callable(fn):
        return False, "LoadHikProfile-not-available"

    try:
        ret = fn(
            strPath=profile_path,
            bApplyTpose=True,
            bApplyBoneMapping=True,
            bSendUpdateEvent=True,
        )
        if _ret_is_success(ret):
            return True, "LoadHikProfile(kwargs)"
        return False, "LoadHikProfile-failed:%s" % str(ret)
    except Exception as e:
        return False, "LoadHikProfile-exception:%s" % str(e)


def _import_bvh_on_avatar(avatar, bvh_path):
    if avatar is None:
        return False, "no-avatar"
    if not os.path.isfile(bvh_path):
        return False, "bvh-not-found"

    _select_only_avatar(avatar)

    # Strict import path: API motion import only.
    calls = [
        ("avatar", "ImportMotion", (bvh_path,)),
        ("avatar", "LoadMotion", (bvh_path,)),
        ("avatar", "LoadMotionFile", (bvh_path,)),
        ("avatar", "ImportMotionFile", (bvh_path,)),
        ("RFileIO", "LoadFile", (bvh_path,)),
    ]
    for owner_name, method_name, args in calls:
        owner = avatar if owner_name == "avatar" else RLPy.RFileIO
        fn = getattr(owner, method_name, None)
        if not callable(fn):
            continue
        try:
            ret = fn(*args)
            if _ret_is_success(ret):
                return True, "%s.%s" % (owner_name, method_name)
        except Exception:
            continue
    return False, "bvh-import-failed"


def _normalize_bone_name(name):
    if not name:
        return ""
    n = str(name).strip().lower()
    for prefix in ("rl_", "cc_base_", "cc_", "skel_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
    chars = []
    for ch in n:
        if ch.isalnum():
            chars.append(ch)
    return "".join(chars)


def _quat_to_euler_deg_xyz(x, y, z, w):
    # Quaternion (x,y,z,w) -> Euler XYZ in degrees.
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    ex = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    ey = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    ez = math.atan2(t3, t4)
    return [math.degrees(ex), math.degrees(ey), math.degrees(ez)]


def _clip_looks_valid(clip):
    if clip is None:
        return False
    for name in ("GetControl", "GetLength", "SetLength", "GetDataBlock"):
        if callable(getattr(clip, name, None)):
            return True
    return False


def _find_any_clip(skeleton):
    if skeleton is None:
        return None, "no-skeleton"
    indices = []
    count = _safe_call(skeleton, "GetClipCount")
    if isinstance(count, int) and count > 0:
        indices.extend(list(range(0, min(count, 16))))
        indices.extend(list(range(1, min(count + 1, 16))))
    else:
        indices.extend([0, 1, 2, 3, 4, 5, 6, 7])
    seen = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        clip = _safe_call(skeleton, "GetClip", idx)
        if _clip_looks_valid(clip):
            return clip, "getclip:%d" % idx
    return None, "no-existing-clip"


def _make_time(ms):
    t_cls = getattr(RLPy, "RTime", None)
    if t_cls is None:
        return None
    try:
        return t_cls(int(ms))
    except Exception:
        return None


def _current_time():
    return _safe_call(getattr(RLPy, "RGlobal", None), "GetTime")


def _one_frame_time():
    fps = _safe_call(getattr(RLPy, "RGlobal", None), "GetFps")
    try:
        fps = float(fps)
    except Exception:
        fps = 30.0
    if fps <= 0.0:
        fps = 30.0
    # RTime in iClone docs is millisecond-based.
    return _make_time(max(1, int(round(1000.0 / fps))))


def _ensure_motion_clip(skeleton):
    clip, source = _find_any_clip(skeleton)
    if clip is not None:
        return clip, source

    # Prefer documented API: AddClip(kTime).
    now = _current_time() or _make_time(0)
    one_frame = _one_frame_time() or _make_time(34)
    candidate_calls = [
        ("AddClip", (now,) if now is not None else ()),
        ("AddClip", (_make_time(0),) if _make_time(0) is not None else ()),
        ("InsertClip", (now,) if now is not None else ()),
        ("InsertClip", (_make_time(0),) if _make_time(0) is not None else ()),
        ("CreateMotionClip", ()),
        ("AddMotionClip", (now,) if now is not None else ()),
    ]
    for name, args in candidate_calls:
        if not args:
            continue
        if not callable(getattr(skeleton, name, None)):
            continue
        ok, ret = _invoke(skeleton, name, *args)
        _log("Clip create try: %s args=%d ok=%s ret=%s" % (name, len(args), ok, str(ret)))
        if ok and _ret_is_success(ret):
            # Best effort: fetch clip at insertion time first.
            clip = _safe_call(skeleton, "GetClipByTime", now) if now is not None else None
            source = "getclipbytime"
            if not _clip_looks_valid(clip):
                clip = None
                clip, source = _find_any_clip(skeleton)
            if clip is not None:
                # Single-frame clip as requested.
                if one_frame is not None and callable(getattr(clip, "SetLength", None)):
                    _invoke(clip, "SetLength", one_frame)
                return clip, "created:%s" % name

    # Last attempt: maybe creation returned clip object directly.
    for name, args in candidate_calls:
        if not callable(getattr(skeleton, name, None)):
            continue
        ok, ret = _invoke(skeleton, name, *args)
        if ok and _clip_looks_valid(ret):
            if one_frame is not None and callable(getattr(ret, "SetLength", None)):
                _invoke(ret, "SetLength", one_frame)
            return ret, "created-return:%s" % name

    # Surface helpful debug list.
    clip_methods = [m for m in dir(skeleton) if "clip" in m.lower()]
    clip_methods.sort()
    _log("No clip could be created. Skeleton clip methods: %s" % ", ".join(clip_methods[:80]))
    return None, "no-clip-created"


def _set_float_control_value(float_control, value):
    if float_control is None:
        return False
    now = _safe_call(getattr(RLPy, "RGlobal", None), "GetTime")
    tries = [
        ("SetValue", (value,)),
        ("SetValue", (now, value)),
        ("SetKey", (value,)),
        ("SetKey", (now, value)),
        ("AddKey", (now, value)),
        ("AddKey", (value,)),
    ]
    for m, args in tries:
        ok, _ = _invoke(float_control, m, *args)
        if ok and _ret_is_success(_):
            return True
    return False


def _set_bone_rotation_via_clip(skeleton, bone, quat_xyzw):
    if skeleton is None or bone is None:
        return False, "clip:no-skeleton-or-bone"
    clip, clip_source = _ensure_motion_clip(skeleton)
    if clip is None:
        return False, "clip:no-clip0"
    _log("Using clip source: %s" % clip_source)

    ok, layer_ctrl = _invoke(clip, "GetControl", "Layer", bone)
    if not ok or layer_ctrl is None or not _ret_is_success(layer_ctrl):
        return False, "clip:no-layer-control"

    ok, data_block = _invoke(layer_ctrl, "GetDataBlock")
    if not ok or data_block is None or not _ret_is_success(data_block):
        return False, "clip:no-datablock"

    rx_ctrl = _safe_call(data_block, "GetControl", "Rotation/RotationX")
    ry_ctrl = _safe_call(data_block, "GetControl", "Rotation/RotationY")
    rz_ctrl = _safe_call(data_block, "GetControl", "Rotation/RotationZ")
    if rx_ctrl is None or ry_ctrl is None or rz_ctrl is None:
        return False, "clip:no-rotation-controls"

    x, y, z, w = [float(v) for v in quat_xyzw]
    ex, ey, ez = _quat_to_euler_deg_xyz(x, y, z, w)
    okx = _set_float_control_value(rx_ctrl, ex)
    oky = _set_float_control_value(ry_ctrl, ey)
    okz = _set_float_control_value(rz_ctrl, ez)
    if okx and oky and okz:
        return True, "clip:RotationXYZ"
    return False, "clip:set-failed"


def _log(msg):
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = "[%s] %s\n" % (ts, str(msg))
        with open(PLUGIN_LOG, "a", encoding="utf-8") as fp:
            fp.write(line)
    except Exception:
        pass


def _log_exception(prefix):
    try:
        _log("%s\n%s" % (prefix, traceback.format_exc()))
    except Exception:
        pass


def _message(title, text):
    _log("MESSAGE %s: %s" % (title, text))
    rmsg = getattr(RLPy, "RMessageBox", None)
    if rmsg is not None:
        try:
            rmsg.Show(title, text, RLPy.EMsgButton_Ok)
            return
        except Exception:
            _log_exception("RMessageBox.Show failed")
    try:
        RLPy.RUi.ShowMessageBox(title, text, RLPy.EMsgButton_Ok)
        return
    except Exception:
        _log_exception("RUi.ShowMessageBox failed")
    try:
        print("PoseSync [%s] %s" % (title, text))
    except Exception:
        pass


def _main_window():
    ptr = RLPy.RUi.GetMainWindow()
    try:
        return wrapInstance(int(ptr), QtWidgets.QMainWindow)
    except Exception:
        _log_exception("Failed to wrap main window")
        return None


def _pick_image():
    _log("Opening image picker")
    parent = _main_window()
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        "Select Input Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
    )
    _log("Image picker result: %s" % path)
    return path or ""


def _pick_bvh_save(default_path=""):
    _log("Opening BVH save picker")
    parent = _main_window()
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        "Save BVH",
        default_path or DEBUG_BVH_PATH,
        "BVH (*.bvh)",
    )
    _log("BVH save picker result: %s" % path)
    if path and not path.lower().endswith(".bvh"):
        path += ".bvh"
    return path or ""


def _default_output_bvh_path():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(GENERATED_BVH_DIR, "posesync_pose_%s.bvh" % ts)


def _wait_for_fresh_bvh(bvh_path, request_started_ts, wait_seconds=8.0):
    deadline = time.time() + max(0.5, float(wait_seconds))
    while time.time() < deadline:
        try:
            if os.path.isfile(bvh_path):
                size = os.path.getsize(bvh_path)
                mtime = os.path.getmtime(bvh_path)
                if size > 0 and mtime >= float(request_started_ts):
                    return True, "size=%d mtime=%.3f" % (int(size), float(mtime))
        except Exception:
            pass
        time.sleep(0.1)
    return False, "fresh-bvh-not-ready: %s" % bvh_path


def _iter_children(obj):
    children = _safe_call(obj, "GetChildren")
    if children:
        try:
            for child in children:
                yield child
            return
        except Exception:
            pass
    count = _safe_call(obj, "GetChildCount")
    if isinstance(count, int) and count > 0:
        for i in range(count):
            child = _safe_call(obj, "GetChild", i)
            if child is not None:
                yield child


def _walk_descendants(root):
    stack = [root]
    visited = set()
    while stack:
        node = stack.pop()
        if node is None:
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        yield node
        for child in _iter_children(node):
            stack.append(child)


def _object_name(obj):
    name = _safe_call(obj, "GetName")
    if name:
        return str(name)
    name = getattr(obj, "name", None)
    if name:
        return str(name)
    return ""


def _selected_avatar():
    selected = RLPy.RScene.GetSelectedObjects()
    avatars = [o for o in selected if o.GetType() == RLPy.EObjectType_Avatar]
    if avatars:
        return avatars[0]
    scene_avatars = _safe_call(RLPy.RScene, "GetAvatars")
    if scene_avatars:
        try:
            return scene_avatars[0]
        except Exception:
            return None
    return None


def _avatar_bones_by_name(avatar):
    out = {}
    skeleton = _safe_call(avatar, "GetSkeletonComponent")
    if skeleton is None:
        return out

    def add_bone(bone, prefer=False):
        if bone is None:
            return
        n = (_object_name(bone) or "").strip()
        if not n:
            return
        raw = n.lower()
        norm = _normalize_bone_name(n)
        if prefer or raw not in out:
            out[raw] = bone
        if norm and (prefer or norm not in out):
            out[norm] = bone

    root_bone = _safe_call(skeleton, "GetRootBone")
    if root_bone is not None:
        for bone in _walk_descendants(root_bone):
            add_bone(bone, prefer=False)

    skin_bones = _safe_call(skeleton, "GetSkinBones")
    if skin_bones:
        try:
            for bone in skin_bones:
                add_bone(bone, prefer=False)
        except Exception:
            pass

    # Motion bones are animatable for clip layer controls; prefer these.
    motion_bones = _safe_call(skeleton, "GetMotionBones")
    if motion_bones:
        try:
            for bone in motion_bones:
                add_bone(bone, prefer=True)
        except Exception:
            pass
    return out


def _http_json(url, payload=None, timeout=240):
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _server_python_candidates():
    out = []
    override = os.environ.get("POSESYNC_SERVER_PYTHON", "").strip()
    if override:
        out.append(override)

    home = os.path.expanduser("~")
    out.append(os.path.join(home, "miniconda3", "envs", "posesync-gpu", "python.exe"))
    out.append(os.path.join(home, "anaconda3", "envs", "posesync-gpu", "python.exe"))
    out.append(sys.executable)
    return out


def _pick_server_python():
    for p in _server_python_candidates():
        if p and os.path.isfile(p):
            return p
    return sys.executable


def _is_server_alive():
    try:
        data = _http_json(HEALTH_ENDPOINT, timeout=2)
        return bool(data.get("ok"))
    except Exception:
        return False


def _spawn_server():
    _log("Spawning server process")
    if not os.path.isfile(SERVER_SCRIPT):
        return False, "Server script missing: %s" % SERVER_SCRIPT

    pyexe = _pick_server_python()
    try:
        log_dir = os.path.dirname(SERVER_LOG)
        if log_dir and not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_fp = open(SERVER_LOG, "ab")
    except Exception as e:
        return False, "Cannot open server log file: %s" % str(e)

    creationflags = 0
    creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        subprocess.Popen(
            [pyexe, SERVER_SCRIPT],
            cwd=BASE_DIR,
            stdout=log_fp,
            stderr=log_fp,
            creationflags=creationflags,
        )
        _log("Server spawn attempted with python: %s" % pyexe)
    except Exception as e:
        try:
            log_fp.close()
        except Exception:
            pass
        return False, "Failed to start server process: %s" % str(e)

    try:
        log_fp.close()
    except Exception:
        pass
    return True, pyexe


def _ensure_server_running(wait_seconds=180):
    _log("Ensuring server running")
    if _is_server_alive():
        _log("Server already healthy")
        return True, "already-running"

    ok, detail = _spawn_server()
    if not ok:
        return False, detail

    deadline = time.time() + float(wait_seconds)
    while time.time() < deadline:
        if _is_server_alive():
            _log("Server became healthy")
            return True, detail
        time.sleep(1.0)

    return False, "Server did not become healthy within %ss. Log: %s" % (wait_seconds, SERVER_LOG)


def _method_signature(obj, name):
    fn = getattr(obj, name, None)
    if not callable(fn):
        return ""
    doc = getattr(fn, "__doc__", "") or ""
    line = str(doc).strip().splitlines()
    return (line[0] if line else "")[:240]


def _select_only_avatar(avatar):
    try:
        if callable(getattr(RLPy.RScene, "ClearSelectObjects", None)):
            RLPy.RScene.ClearSelectObjects()
        if callable(getattr(RLPy.RScene, "SelectObject", None)):
            RLPy.RScene.SelectObject(avatar)
            return
        if callable(getattr(RLPy.RScene, "SelectObjects", None)):
            RLPy.RScene.SelectObjects([avatar])
            return
    except Exception:
        _log_exception("select avatar failed")


def _build_motion_argsets(bvh_path, motion_profile_path=""):
    t0 = _make_time(0)
    out = [(bvh_path,), (bvh_path, 0), (bvh_path, True), (bvh_path, False)]
    if motion_profile_path:
        out.extend(
            [
                (bvh_path, motion_profile_path),
                (bvh_path, motion_profile_path, 0),
                (bvh_path, motion_profile_path, True),
                (bvh_path, motion_profile_path, False),
                (motion_profile_path, bvh_path),
            ]
        )
    if t0 is not None:
        out.append((bvh_path, t0))
        out.append((t0, bvh_path))
        if motion_profile_path:
            out.append((bvh_path, motion_profile_path, t0))
    return out


def _try_method_calls(target, target_name, method_names, argsets, attempts):
    for name in method_names:
        fn = getattr(target, name, None)
        if not callable(fn):
            continue
        sig = _method_signature(target, name)
        for args in argsets:
            ok, ret = _invoke(target, name, *args)
            ret_str = str(ret)
            attempts.append("%s.%s(%d) ok=%s ret=%s sig=%s" % (target_name, name, len(args), ok, ret_str, sig))
            if ok and _ret_is_success(ret):
                return True, "%s.%s" % (target_name, name)
            # Some APIs may return a profile object that must be applied in a second step.
            if ok and ret is not None and not isinstance(ret, (bool, int, float, str)):
                for apply_name in ("ApplyMotionProfile", "LoadMotionProfile", "SetMotionProfile"):
                    apply_fn = getattr(target, apply_name, None)
                    if not callable(apply_fn):
                        continue
                    ok2, ret2 = _invoke(target, apply_name, ret)
                    attempts.append(
                        "%s.%s(profile) ok=%s ret=%s" % (target_name, apply_name, ok2, str(ret2))
                    )
                    if ok2 and _ret_is_success(ret2):
                        return True, "%s.%s(profile)" % (target_name, apply_name)
    return False, ""


def _apply_bvh_to_avatar(avatar, bvh_path):
    if avatar is None:
        return False, "No avatar selected."
    if not os.path.isfile(bvh_path):
        return False, "BVH file not found: %s" % bvh_path

    motion_profile_path = _pick_motion_profile_path()
    if motion_profile_path:
        ok_profile, profile_msg = _load_hik_profile_on_avatar(avatar, motion_profile_path)
        _log("HIK profile apply: %s (%s)" % ("OK" if ok_profile else "FAILED", profile_msg))
        if not ok_profile:
            return False, profile_msg
    else:
        _log("HIK profile apply skipped: no profile path found")

    ok_import, import_mode = _import_bvh_on_avatar(avatar, bvh_path)
    if ok_import:
        _log("BVH apply success via %s" % import_mode)
        return True, import_mode
    return False, import_mode


def _auto_apply_bvh_to_selected_avatar(bvh_path):
    avatar = _selected_avatar()
    if avatar is None:
        return False, "No avatar selected."

    ok, mode = _apply_bvh_to_avatar(avatar, bvh_path)
    if ok:
        return True, "import:%s" % mode
    return False, "import failed (%s)" % mode


def _parse_bvh_first_frame(bvh_path):
    txt = open(bvh_path, "r", encoding="utf-8", errors="ignore").read()
    lines = txt.splitlines()
    motion_idx = -1
    for i, line in enumerate(lines):
        if line.strip().upper() == "MOTION":
            motion_idx = i
            break
    if motion_idx < 0:
        return {}, "MOTION not found"

    hierarchy = lines[:motion_idx]
    motion = lines[motion_idx + 1 :]
    current_joint = ""
    joint_channels = {}
    channel_layout = []

    for line in hierarchy:
        s = line.strip()
        if s.startswith("ROOT ") or s.startswith("JOINT "):
            current_joint = s.split(None, 1)[1].strip()
            if current_joint not in joint_channels:
                joint_channels[current_joint] = []
            continue
        if s.startswith("CHANNELS "):
            parts = s.split()
            count = int(parts[1])
            chans = parts[2 : 2 + count]
            if current_joint:
                joint_channels[current_joint] = chans
                for ch in chans:
                    channel_layout.append((current_joint, ch))

    frame_values = None
    started = False
    for line in motion:
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("frames:"):
            continue
        if s.lower().startswith("frame time:"):
            started = True
            continue
        if started:
            try:
                frame_values = [float(v) for v in s.split()]
            except Exception:
                frame_values = None
            break
    if frame_values is None:
        return {}, "First frame not found"

    rot_map = {}
    n = min(len(channel_layout), len(frame_values))
    for i in range(n):
        joint_name, ch = channel_layout[i]
        if not ch.lower().endswith("rotation"):
            continue
        if joint_name not in rot_map:
            rot_map[joint_name] = {}
        rot_map[joint_name][ch] = float(frame_values[i])
    return rot_map, ""


def _set_bone_euler_via_clip(skeleton, bone, ex, ey, ez):
    if skeleton is None or bone is None:
        return False, "clip:no-skeleton-or-bone"
    clip, clip_source = _ensure_motion_clip(skeleton)
    if clip is None:
        return False, "clip:no-clip0"

    ok, layer_ctrl = _invoke(clip, "GetControl", "Layer", bone)
    if not ok or layer_ctrl is None or not _ret_is_success(layer_ctrl):
        return False, "clip:no-layer-control"

    ok, data_block = _invoke(layer_ctrl, "GetDataBlock")
    if not ok or data_block is None or not _ret_is_success(data_block):
        return False, "clip:no-datablock"

    rx_ctrl = _safe_call(data_block, "GetControl", "Rotation/RotationX")
    ry_ctrl = _safe_call(data_block, "GetControl", "Rotation/RotationY")
    rz_ctrl = _safe_call(data_block, "GetControl", "Rotation/RotationZ")
    if rx_ctrl is None or ry_ctrl is None or rz_ctrl is None:
        return False, "clip:no-rotation-controls"

    okx = _set_float_control_value(rx_ctrl, float(ex))
    oky = _set_float_control_value(ry_ctrl, float(ey))
    okz = _set_float_control_value(rz_ctrl, float(ez))
    if okx and oky and okz:
        return True, "clip:BVHRotationXYZ:%s" % clip_source
    return False, "clip:set-failed"


def _apply_bvh_frame_to_clip(avatar, bvh_path):
    rot_map, err = _parse_bvh_first_frame(bvh_path)
    if err:
        _log("BVH fallback parse failed: %s" % err)
        return False, "parse-failed"
    if not rot_map:
        return False, "no-rotations"

    bones = _avatar_bones_by_name(avatar)
    if not bones:
        return False, "no-bones"
    skeleton = _safe_call(avatar, "GetSkeletonComponent")
    if skeleton is None:
        return False, "no-skeleton"

    applied = 0
    missing = 0
    failed = 0
    for joint_name, r in rot_map.items():
        b = bones.get(joint_name.lower())
        if b is None:
            b = bones.get(_normalize_bone_name(joint_name))
        if b is None:
            missing += 1
            continue
        ex = float(r.get("Xrotation", 0.0))
        ey = float(r.get("Yrotation", 0.0))
        ez = float(r.get("Zrotation", 0.0))
        ok, _mode = _set_bone_euler_via_clip(skeleton, b, ex, ey, ez)
        if ok:
            applied += 1
        else:
            failed += 1
    _log("BVH fallback applied=%d missing=%d failed=%d" % (applied, missing, failed))
    if applied > 0 and failed == 0:
        return True, "applied=%d missing=%d" % (applied, missing)
    if applied > 0:
        return True, "partial applied=%d missing=%d failed=%d" % (applied, missing, failed)
    return False, "applied=0 missing=%d failed=%d" % (missing, failed)


def _make_quaternion(x, y, z, w):
    q_cls = getattr(RLPy, "RQuaternion", None)
    if q_cls is None:
        return None
    try:
        return q_cls(x, y, z, w)
    except Exception:
        pass
    try:
        q = q_cls()
    except Exception:
        return None
    for setter in ("SetValue", "Set", "SetXYZW"):
        fn = getattr(q, setter, None)
        if callable(fn):
            try:
                fn(x, y, z, w)
                return q
            except Exception:
                continue
    return q


def _set_bone_quaternion(skeleton, bone, quat_xyzw):
    # Prefer official animation route first (clip layer controls).
    ok, mode = _set_bone_rotation_via_clip(skeleton, bone, quat_xyzw)
    if ok:
        return True, mode

    x, y, z, w = [float(v) for v in quat_xyzw]
    q = _make_quaternion(x, y, z, w)

    if q is not None:
        for m in ("SetLocalRotation", "SetRotation", "SetWorldRotation"):
            ok, ret = _invoke(bone, m, q)
            if ok and _ret_is_success(ret):
                return True, m
        for m in ("SetLocalRotation", "SetRotation", "SetWorldRotation"):
            ok, ret = _invoke(bone, m, x, y, z, w)
            if ok and _ret_is_success(ret):
                return True, m

    tr = (
        _safe_call(bone, "GetLocalTransform")
        or _safe_call(bone, "GetTransform")
    )
    if tr is None:
        tr, _ = _get_prop(bone, "LocalTransform", "WorldTransform", "BasisTransform")

    if tr is not None:
        if q is not None:
            for m in ("SetR", "SetRotation", "SetQuaternion"):
                ok, _ = _invoke(tr, m, q)
                if ok:
                    break
            if _set_prop(tr, "R", q) or _set_prop(tr, "Rotation", q):
                pass
        else:
            # Last-resort: patch existing rotation object in-place.
            rot, _ = _get_prop(tr, "R", "Rotation", "r", "rotation")
            if rot is not None:
                for attr, val in (("x", x), ("y", y), ("z", z), ("w", w), ("X", x), ("Y", y), ("Z", z), ("W", w)):
                    try:
                        setattr(rot, attr, float(val))
                    except Exception:
                        pass
                _set_prop(tr, "R", rot)
                _set_prop(tr, "Rotation", rot)

        # Push transform back to bone using real setter methods only.
        for m in ("SetLocalTransform", "SetTransform", "SetWorldTransform"):
            ok, ret = _invoke(bone, m, tr)
            if ok and _ret_is_success(ret):
                return True, "transform:" + m

    return False, mode


def _check_server():
    _log("Action: check_server")
    try:
        ok, detail = _ensure_server_running(wait_seconds=180)
        if not ok:
            _message("PoseSync", "Server start failed:\n%s" % detail)
            return
        data = _http_json(HEALTH_ENDPOINT, payload=None, timeout=10)
        data["server_python"] = detail
        data["server_log"] = SERVER_LOG
        data["plugin_log"] = PLUGIN_LOG
        _message("PoseSync", "Server OK\n%s" % json.dumps(data, indent=2))
    except Exception as e:
        _log_exception("check_server failed")
        _message("PoseSync", "Server check failed:\n%s" % str(e))


def _request_bvh_from_server(image_path, output_bvh_path):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("ascii")

    payload_mhr = {
        "image_b64": image_b64,
        "person_index": 0,
        "output_bvh_path": output_bvh_path,
        "bvh_profile_name": "mhr_raw_pose",
    }
    try:
        _log("Sending MHR BVH pose request to server")
        result = _http_json(POSE_MHR_BVH_ENDPOINT, payload=payload_mhr, timeout=300)
        return result, "mhr_raw"
    except urllib.error.HTTPError as e:
        # Fallback to template endpoint when new endpoint is unavailable.
        if int(getattr(e, "code", 0) or 0) not in (404, 405):
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError("Server HTTP error: %s\n%s" % (e, body))
    except Exception:
        # Non-HTTP errors are handled by fallback below.
        pass

    payload_template = {
        "image_b64": image_b64,
        "person_index": 0,
        "output_bvh_path": output_bvh_path,
    }
    template_bvh_path = _pick_template_bvh_path()
    if template_bvh_path:
        payload_template["template_bvh_path"] = template_bvh_path

    try:
        _log("Sending template BVH pose request to server (fallback)")
        result = _http_json(POSE_BVH_ENDPOINT, payload=payload_template, timeout=300)
        return result, "template"
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError("Server HTTP error: %s\n%s" % (e, body))


def _apply_pose_from_image():
    _log("Action: apply_pose_from_image")
    try:
        ok, detail = _ensure_server_running(wait_seconds=180)
        if not ok:
            _message("PoseSync", "Server start failed:\n%s" % detail)
            return

        image_path = _pick_image()
        if not image_path:
            _log("Apply cancelled: no image selected")
            return
        if not os.path.isfile(image_path):
            _message("PoseSync", "Image not found:\n%s" % image_path)
            return

        out_bvh = _pick_bvh_save(_default_output_bvh_path())
        if not out_bvh:
            _log("Apply cancelled: no output BVH selected")
            return

        try:
            os.makedirs(os.path.dirname(out_bvh), exist_ok=True)
            try:
                if os.path.isfile(out_bvh):
                    os.remove(out_bvh)
            except Exception:
                pass
            request_started_ts = time.time()
            result, endpoint_mode = _request_bvh_from_server(image_path, out_bvh)
            _log("Pose request endpoint mode: %s" % endpoint_mode)
        except Exception as e:
            _log_exception("Pose request failed")
            _message("PoseSync", "Pose request failed:\n%s" % str(e))
            return

        if not result.get("ok"):
            _message("PoseSync", "Inference failed:\n%s" % result.get("error", "unknown error"))
            return

        bvh_path = result.get("bvh_path", "") or out_bvh
        if not bvh_path:
            _message("PoseSync", "Server did not return a BVH path.")
            return
        ok_fresh, fresh_msg = _wait_for_fresh_bvh(bvh_path, request_started_ts, wait_seconds=8.0)
        if not ok_fresh:
            _message("PoseSync", "BVH not ready yet, skipping apply:\n%s" % fresh_msg)
            return

        bvh_meta = result.get("bvh", {}) or {}

        msg = [
            "Pose BVH build finished.",
            "Image: %s" % os.path.basename(image_path),
            "Source joints from server: %d" % int(result.get("joint_count", result.get("source_joint_count", 0))),
            "BVH path: %s" % bvh_path,
            "Template: %s" % result.get("template_bvh_path", ""),
            "BVH joint count: %s" % str(bvh_meta.get("joint_count", "n/a")),
            "BVH updated joints: %s" % str(bvh_meta.get("updated_joint_count", "n/a")),
            "BVH updated channels: %s" % str(bvh_meta.get("updated_channel_count", "n/a")),
            "BVH solved joints: %s" % str(bvh_meta.get("solved_joint_count", "n/a")),
            "BVH quat fallback joints: %s" % str(bvh_meta.get("quat_fallback_joint_count", "n/a")),
            "BVH missing joints: %s" % str(
                bvh_meta.get("missing_joint_count", bvh_meta.get("missing_source_joint_count", "n/a"))
            ),
        ]
        missing_sample = bvh_meta.get("missing_joint_sample", []) or bvh_meta.get("missing_source_joint_sample", [])
        if missing_sample:
            msg.append("BVH missing sample: %s" % ", ".join(missing_sample[:8]))

        ok_apply, apply_mode = _auto_apply_bvh_to_selected_avatar(bvh_path)
        if ok_apply:
            msg.append("Apply to selected avatar: OK (%s)" % apply_mode)
        else:
            msg.append("Apply to selected avatar: SKIPPED/FAILED (%s)" % apply_mode)

        _log(
            "Build result: bvh=%s updated_joints=%s"
            % (
                bvh_path,
                str(bvh_meta.get("updated_joint_count", 0)),
            )
        )
        _message("PoseSync", "\n".join(msg))
    except Exception:
        _log_exception("Unhandled exception in apply_pose_from_image")
        _message("PoseSync", "Unhandled plugin error. Check log:\n%s" % PLUGIN_LOG)


def _debug_bone_api():
    _log("Action: debug_bone_api")
    try:
        avatar = _selected_avatar()
        if avatar is None:
            _message("PoseSync", "Select one avatar first.")
            return
        skeleton = _safe_call(avatar, "GetSkeletonComponent")
        root_bone = _safe_call(skeleton, "GetRootBone") if skeleton else None
        if root_bone is None:
            _message("PoseSync", "No root bone found.")
            return

        methods = [m for m in dir(root_bone) if "rot" in m.lower() or "transform" in m.lower() or "control" in m.lower()]
        methods.sort()

        tr, tr_name = _get_prop(root_bone, "LocalTransform", "WorldTransform", "BasisTransform")
        tr_methods = []
        tr_type = "None"
        if tr is not None:
            tr_type = type(tr).__name__
            tr_methods = [
                m for m in dir(tr)
                if ("rot" in m.lower() or "quat" in m.lower() or "transform" in m.lower() or m in ("R", "T"))
            ]
            tr_methods.sort()
        skeleton = _safe_call(avatar, "GetSkeletonComponent")
        skel_methods = [m for m in dir(skeleton)] if skeleton is not None else []
        skel_methods = [m for m in skel_methods if ("clip" in m.lower() or "bone" in m.lower() or "motion" in m.lower())]
        skel_methods.sort()
        motion_count = 0
        skin_count = 0
        motion_bones = _safe_call(skeleton, "GetMotionBones") if skeleton else None
        skin_bones = _safe_call(skeleton, "GetSkinBones") if skeleton else None
        try:
            motion_count = len(motion_bones) if motion_bones is not None else 0
        except Exception:
            motion_count = 0
        try:
            skin_count = len(skin_bones) if skin_bones is not None else 0
        except Exception:
            skin_count = 0
        clip = _safe_call(skeleton, "GetClip", 0) if skeleton else None
        clip_methods = []
        ctrl_methods = []
        db_methods = []
        float_methods = []
        if clip is not None:
            clip_methods = [m for m in dir(clip) if ("control" in m.lower() or "data" in m.lower() or "clip" in m.lower())]
            clip_methods.sort()
            ok, layer_ctrl = _invoke(clip, "GetControl", "Layer", root_bone)
            if ok and layer_ctrl is not None:
                ctrl_methods = [m for m in dir(layer_ctrl) if ("data" in m.lower() or "control" in m.lower() or "key" in m.lower() or "value" in m.lower())]
                ctrl_methods.sort()
                ok, db = _invoke(layer_ctrl, "GetDataBlock")
                if ok and db is not None:
                    db_methods = [m for m in dir(db) if ("control" in m.lower() or "data" in m.lower())]
                    db_methods.sort()
                    rx = _safe_call(db, "GetControl", "Rotation/RotationX")
                    if rx is not None:
                        float_methods = [m for m in dir(rx) if ("key" in m.lower() or "value" in m.lower())]
                        float_methods.sort()

        _message(
            "PoseSync Bone API",
            (
                "Root bone: %s\n"
                "Bone type: %s\n"
                "Motion bones: %d\n"
                "Skin bones: %d\n"
                "Transform prop: %s\n"
                "Transform type: %s\n\n"
                "Skeleton members:\n%s\n\n"
                "Bone members:\n%s\n\n"
                "Transform members:\n%s\n\n"
                "Clip members:\n%s\n\n"
                "Layer control members:\n%s\n\n"
                "DataBlock members:\n%s\n\n"
                "RotationX control members:\n%s"
            )
            % (
                _object_name(root_bone),
                type(root_bone).__name__,
                motion_count,
                skin_count,
                tr_name or "N/A",
                tr_type,
                "\n".join(skel_methods[:120]),
                "\n".join(methods[:120]),
                "\n".join(tr_methods[:120]),
                "\n".join(clip_methods[:120]),
                "\n".join(ctrl_methods[:120]),
                "\n".join(db_methods[:120]),
                "\n".join(float_methods[:120]),
            ),
        )
    except Exception:
        _log_exception("debug_bone_api failed")
        _message("PoseSync", "Debug bone API failed. Check log:\n%s" % PLUGIN_LOG)


def _debug_sam3d_raw():
    _log("Action: debug_sam3d_raw")
    try:
        ok, detail = _ensure_server_running(wait_seconds=180)
        if not ok:
            _message("PoseSync", "Server start failed:\n%s" % detail)
            return

        image_path = _pick_image()
        if not image_path:
            _log("Raw debug cancelled: no image selected")
            return
        if not os.path.isfile(image_path):
            _message("PoseSync", "Image not found:\n%s" % image_path)
            return

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")

        payload = {
            "image_b64": image_b64,
            "person_index": 0,
            "raw_array_mode": "full",
            "include_all_people": False,
        }
        _log("Sending raw SAM-3D request to server")
        result = _http_json(POSE_RAW_ENDPOINT, payload=payload, timeout=300)
        if not result.get("ok"):
            _message("PoseSync", "Raw inference failed:\n%s" % result.get("error", "unknown error"))
            return

        os.makedirs(os.path.dirname(DEBUG_RAW_JSON_PATH), exist_ok=True)
        with open(DEBUG_RAW_JSON_PATH, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)

        raw_person = result.get("sam3d_raw") or {}
        raw_keys = sorted(list(raw_person.keys())) if isinstance(raw_person, dict) else []
        msg = [
            "Raw SAM-3D output saved.",
            "Image: %s" % os.path.basename(image_path),
            "People detected: %d" % int(result.get("people_detected", 0)),
            "Person index: %d" % int(result.get("person_index", 0)),
            "Raw key count: %d" % len(raw_keys),
            "Raw keys sample: %s" % (", ".join(raw_keys[:12]) if raw_keys else "n/a"),
            "JSON path: %s" % DEBUG_RAW_JSON_PATH,
        ]
        _message("PoseSync", "\n".join(msg))
    except Exception:
        _log_exception("debug_sam3d_raw failed")
        _message("PoseSync", "Raw SAM-3D debug failed. Check log:\n%s" % PLUGIN_LOG)


class PoseSyncWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PoseSyncWindow, self).__init__(parent)
        self.setObjectName("posesync_window")
        self.setMinimumWidth(560)
        self._apply_style()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setObjectName("header")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)
        header_title = QtWidgets.QLabel("PoseSync")
        header_title.setObjectName("header_title")
        header_layout.addWidget(header_title, 1)
        layout.addWidget(header)

        self.lbl_info = QtWidgets.QLabel("Load image, build BVH, auto-apply to selected avatar.")
        self.lbl_info.setObjectName("subtle")
        layout.addWidget(self.lbl_info)

        grp_pose = QtWidgets.QGroupBox("Image → BVH")
        grp_pose_layout = QtWidgets.QVBoxLayout(grp_pose)
        grp_pose_layout.setContentsMargins(10, 14, 10, 10)
        grp_pose_layout.setSpacing(8)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grp_pose_layout.addLayout(grid)

        grid.addWidget(QtWidgets.QLabel("Image"), 0, 0)
        self.edit_image = QtWidgets.QLineEdit()
        grid.addWidget(self.edit_image, 0, 1)
        self.btn_browse_image = QtWidgets.QPushButton("Browse...")
        self.btn_browse_image.setObjectName("secondary_btn")
        self.btn_browse_image.clicked.connect(self._on_browse_image)
        grid.addWidget(self.btn_browse_image, 0, 2)

        grid.addWidget(QtWidgets.QLabel("Output BVH"), 1, 0)
        self.edit_output = QtWidgets.QLineEdit(_default_output_bvh_path())
        grid.addWidget(self.edit_output, 1, 1)
        self.btn_browse_output = QtWidgets.QPushButton("Save As...")
        self.btn_browse_output.setObjectName("secondary_btn")
        self.btn_browse_output.clicked.connect(self._on_browse_output)
        grid.addWidget(self.btn_browse_output, 1, 2)

        self.chk_auto_apply = QtWidgets.QCheckBox("Auto-apply BVH to selected avatar")
        self.chk_auto_apply.setChecked(True)
        grp_pose_layout.addWidget(self.chk_auto_apply)

        profile_hint = _pick_motion_profile_path() or "(not found)"
        self.lbl_profile = QtWidgets.QLabel("Character profile: %s" % profile_hint)
        self.lbl_profile.setObjectName("subtle")
        grp_pose_layout.addWidget(self.lbl_profile)

        self.btn_apply = QtWidgets.QPushButton("Apply Pose")
        self.btn_apply.setObjectName("primary_btn")
        self.btn_apply.clicked.connect(self._on_apply)
        grp_pose_layout.addWidget(self.btn_apply)

        layout.addWidget(grp_pose)

        grp_server = QtWidgets.QGroupBox("Server")
        grp_server_layout = QtWidgets.QVBoxLayout(grp_server)
        grp_server_layout.setContentsMargins(10, 14, 10, 10)
        self.btn_check = QtWidgets.QPushButton("Check Server")
        self.btn_check.setObjectName("secondary_btn")
        self.btn_check.clicked.connect(self._on_check_server)
        grp_server_layout.addWidget(self.btn_check)
        layout.addWidget(grp_server)

        grp_status = QtWidgets.QGroupBox("Status")
        grp_status_layout = QtWidgets.QVBoxLayout(grp_status)
        grp_status_layout.setContentsMargins(8, 14, 8, 8)
        self.txt_status = QtWidgets.QPlainTextEdit()
        self.txt_status.setObjectName("status_box")
        self.txt_status.setReadOnly(True)
        self.txt_status.setMinimumHeight(180)
        grp_status_layout.addWidget(self.txt_status)
        layout.addWidget(grp_status)

        self._append_status("Ready.")

    def _apply_style(self):
        self.setStyleSheet(
            """
            QWidget#posesync_window {
                background-color: #1f2126;
                color: #f2f2f2;
                font-size: 13px;
            }
            QFrame#header {
                background-color: #7cc20c;
                border: 1px solid #91d81e;
            }
            QLabel#header_title {
                color: #0f1308;
                font-size: 16px;
                font-weight: 700;
                letter-spacing: 0.3px;
            }
            QLabel#subtle {
                color: #d2d5da;
                padding-left: 2px;
            }
            QGroupBox {
                border: 1px solid #4a4d55;
                margin-top: 10px;
                color: #e8eaef;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit {
                background-color: #17191d;
                border: 1px solid #535862;
                color: #f5f7fb;
                padding: 5px 6px;
            }
            QPushButton {
                border: 1px solid #5b616c;
                color: #f6f8fc;
                padding: 7px 12px;
                min-height: 30px;
            }
            QPushButton#primary_btn {
                background-color: #4caf50;
                border-color: #6bc26e;
                font-weight: 700;
            }
            QPushButton#primary_btn:hover {
                background-color: #55b85a;
            }
            QPushButton#secondary_btn {
                background-color: #2f9fee;
                border-color: #5cb6f2;
                font-weight: 600;
            }
            QPushButton#secondary_btn:hover {
                background-color: #3ba8f0;
            }
            QPlainTextEdit#status_box {
                background-color: #15171b;
                border: 1px solid #4f555f;
                color: #e4e8ef;
                selection-background-color: #2f9fee;
            }
            """
        )

    def _append_status(self, text):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_status.appendPlainText("[%s] %s" % (ts, text))

    def _on_browse_image(self):
        path = _pick_image()
        if path:
            self.edit_image.setText(path)
            self._append_status("Image set: %s" % path)

    def _on_browse_output(self):
        path = _pick_bvh_save(self.edit_output.text().strip() or _default_output_bvh_path())
        if path:
            self.edit_output.setText(path)
            self._append_status("Output set: %s" % path)

    def _on_check_server(self):
        ok, detail = _ensure_server_running(wait_seconds=180)
        if ok:
            self._append_status("Server OK (%s)" % detail)
        else:
            self._append_status("Server failed: %s" % detail)
            _message("PoseSync", "Server start failed:\n%s" % detail)

    def _on_apply(self):
        image_path = self.edit_image.text().strip()
        if not image_path:
            image_path = _pick_image()
            if image_path:
                self.edit_image.setText(image_path)
        if not image_path:
            self._append_status("No image selected.")
            return
        if not os.path.isfile(image_path):
            self._append_status("Image not found: %s" % image_path)
            _message("PoseSync", "Image not found:\n%s" % image_path)
            return

        out_bvh = self.edit_output.text().strip() or _default_output_bvh_path()
        self.edit_output.setText(out_bvh)
        try:
            os.makedirs(os.path.dirname(out_bvh), exist_ok=True)
        except Exception:
            pass

        ok, detail = _ensure_server_running(wait_seconds=180)
        if not ok:
            self._append_status("Server failed: %s" % detail)
            _message("PoseSync", "Server start failed:\n%s" % detail)
            return

        self._append_status("Building BVH...")
        QtWidgets.QApplication.processEvents()

        try:
            try:
                if os.path.isfile(out_bvh):
                    os.remove(out_bvh)
            except Exception:
                pass
            request_started_ts = time.time()
            result, endpoint_mode = _request_bvh_from_server(image_path, out_bvh)
        except Exception as e:
            self._append_status("Build failed: %s" % str(e))
            _message("PoseSync", "Pose request failed:\n%s" % str(e))
            return

        if not result.get("ok"):
            err = result.get("error", "unknown error")
            self._append_status("Inference failed: %s" % err)
            _message("PoseSync", "Inference failed:\n%s" % err)
            return

        bvh_path = result.get("bvh_path", "") or out_bvh
        ok_fresh, fresh_msg = _wait_for_fresh_bvh(bvh_path, request_started_ts, wait_seconds=8.0)
        if not ok_fresh:
            self._append_status("Build not finalized: %s" % fresh_msg)
            _message("PoseSync", "BVH is not freshly written yet.\n%s" % fresh_msg)
            return
        bvh_meta = result.get("bvh", {}) or {}
        self.edit_output.setText(bvh_path)
        self._append_status("BVH built via %s: %s" % (endpoint_mode, bvh_path))
        self._append_status(
            "Joints=%s, Missing=%s"
            % (
                str(bvh_meta.get("joint_count", "n/a")),
                str(bvh_meta.get("missing_joint_count", bvh_meta.get("missing_source_joint_count", "n/a"))),
            )
        )

        if self.chk_auto_apply.isChecked():
            self._append_status("Applying to selected avatar...")
            QtWidgets.QApplication.processEvents()
            ok_apply, apply_mode = _auto_apply_bvh_to_selected_avatar(bvh_path)
            if ok_apply:
                self._append_status("Apply OK: %s" % apply_mode)
            else:
                self._append_status("Apply failed: %s" % apply_mode)
                _message("PoseSync", "BVH generated but apply failed:\n%s" % apply_mode)

        _message(
            "PoseSync",
            "BVH generated.\nImage: %s\nBVH: %s" % (os.path.basename(image_path), bvh_path),
        )


def _open_pose_sync():
    global _pose_dock_ui

    area_names = (
        "EDockWidgetAreas_LeftDockWidgetArea",
        "EDockWidgetAreas_RightDockWidgetArea",
        "EDockWidgetAreas_TopDockWidgetArea",
        "EDockWidgetAreas_BottomDockWidgetArea",
    )
    all_areas = 0
    for name in area_names:
        all_areas |= int(getattr(RLPy, name, 0))

    feature_names = (
        "EDockWidgetFeatures_Movable",
        "EDockWidgetFeatures_Closable",
        "EDockWidgetFeatures_Floatable",
    )
    all_features = 0
    for name in feature_names:
        all_features |= int(getattr(RLPy, name, 0))

    if "dock" not in _pose_dock_ui:
        dock = RLPy.RUi.CreateRDockWidget()
        dock.SetWindowTitle("PoseSync")
        dock.SetParent(RLPy.RUi.GetMainWindow())
        if all_areas:
            dock.SetAllowedAreas(all_areas)
        if all_features:
            dock.SetFeatures(all_features)

        qt_dock = wrapInstance(int(dock.GetWindow()), QtWidgets.QDockWidget)
        widget = PoseSyncWindow(qt_dock)
        qt_dock.setWidget(widget)

        _pose_dock_ui["dock"] = dock
        _pose_dock_ui["widget"] = widget

    # Re-apply docking behavior to avoid stale state after reload.
    dock = _pose_dock_ui["dock"]
    try:
        if all_areas:
            dock.SetAllowedAreas(all_areas)
        if all_features:
            dock.SetFeatures(all_features)
    except Exception:
        pass

    try:
        qt_dock = wrapInstance(int(dock.GetWindow()), QtWidgets.QDockWidget)
        qt_dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        qt_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
    except Exception:
        _log_exception("dock qt flags failed")

    dock.Show()
    try:
        widget = _pose_dock_ui.get("widget")
        if widget is not None:
            widget.raise_()
            widget.activateWindow()
    except Exception:
        pass


def _add_menu_entry():
    _log("Registering PoseSync menu")
    main_qt = _main_window()
    if main_qt is None:
        _log("main_qt is None")
        return

    menu = main_qt.menuBar().findChild(QtWidgets.QMenu, "clone_tools_menu")
    if menu is None:
        menu = wrapInstance(
            int(RLPy.RUi.AddMenu("CloneTools", RLPy.EMenu_Plugins)),
            QtWidgets.QMenu,
        )
        menu.setObjectName("clone_tools_menu")

    for act in list(menu.actions()):
        if act.text() in (
            "PoseSync Window",
            "PoseSync Apply Pose",
            "PoseSync Check Server",
            "PoseSync Bone API",
            "PoseSync Raw SAM3D",
        ):
            menu.removeAction(act)

    menu.addAction("PoseSync Window").triggered.connect(_open_pose_sync)
    menu.addAction("PoseSync Apply Pose").triggered.connect(_apply_pose_from_image)
    menu.addAction("PoseSync Check Server").triggered.connect(_check_server)
    menu.addAction("PoseSync Bone API").triggered.connect(_debug_bone_api)
    menu.addAction("PoseSync Raw SAM3D").triggered.connect(_debug_sam3d_raw)


def initialize_plugin():
    _add_menu_entry()


def run_script():
    _add_menu_entry()

