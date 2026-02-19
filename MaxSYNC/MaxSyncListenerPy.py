# -*- coding: utf-8 -*-
"""MAXsync Python listener backend for 3ds Max.

Socket transport is implemented in Python (no .NET TcpListener).
The MAXScript wrapper keeps the same public MaxSYNC API.
"""

import socket
import threading
import time
import math
import re
from urllib.parse import unquote
from collections import deque

import pymxs

try:
    from PySide2 import QtCore
except Exception:  # pragma: no cover
    from PySide6 import QtCore


HOST = "127.0.0.1"
LISTEN_PORT = 29999
ICLONE_PORT = 29998
PROTOCOL_VERSION = "MSYNC1"
DRAIN_INTERVAL_MS = 20

def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


class MaxSyncPythonBridge(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.running = False
        self.server_socket = None
        self.server_thread = None
        self.timer = None
        self.sync_live_timeline = False

        self._lock = threading.Lock()
        self._import_queue = deque(maxlen=128)
        self._latest_live = None
        self._last_quat_by_node = {}
        self._live_node_cache = {}
        self._morph_cache = {}
        self._morph_cache_built_at = 0.0
        self._morph_cache_interval = 2.0
        self._debug_last_live = {
            "frame": 0,
            "entries": 0,
            "resolved": 0,
            "samples": [],
            "direct_applied": 0,
            "quat_flipped": 0,
            "morph_entries": 0,
            "morph_matched": 0,
            "morph_applied": 0,
            "morph_unmatched": 0,
            "morph_received": [],
            "morph_unmatched_names": [],
        }

    def _log(self, msg):
        print(f"[MAXsync] {msg}")

    def _sanitize_affix(self, text):
        s = str(text or "").strip()
        for ch in ("|", ";", ",", "\r", "\n"):
            s = s.replace(ch, "_")
        return s

    def _snapshot_scene_handles(self):
        rt = pymxs.runtime
        handles = set()
        try:
            for node in rt.objects:
                try:
                    handles.add(int(node.handle))
                except Exception:
                    continue
        except Exception:
            pass
        return handles

    def _collect_imported_nodes(self, before_handles):
        rt = pymxs.runtime
        created = []
        try:
            for node in rt.objects:
                try:
                    handle = int(node.handle)
                except Exception:
                    continue
                if handle not in before_handles:
                    created.append(node)
        except Exception:
            pass
        return created

    def _apply_import_name_affixes(self, imported_nodes, prefix="", suffix=""):
        rt = pymxs.runtime
        prefix = self._sanitize_affix(prefix)
        suffix = self._sanitize_affix(suffix)
        if not imported_nodes or (not prefix and not suffix):
            return 0

        renamed = 0
        for node in imported_nodes:
            try:
                old_name = str(node.name)
            except Exception:
                continue

            new_name = f"{prefix}{old_name}{suffix}"
            if not new_name or new_name == old_name:
                continue

            try:
                node.name = rt.uniquename(new_name)
                renamed += 1
            except Exception:
                continue
        return renamed

    def _find_live_node(self, node_name):
        rt = pymxs.runtime
        raw_name = str(node_name).strip()
        if not raw_name:
            return None

        candidates = []
        seen = set()
        for candidate in (
            raw_name,
            raw_name.replace("|", ":"),
            raw_name.split(":")[-1],
        ):
            c = str(candidate).strip()
            if not c:
                continue
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(c)

        for candidate in candidates:
            cache_key = candidate.lower()
            cached = self._live_node_cache.get(cache_key)
            if cached is not None:
                try:
                    _ = cached.name
                    return cached
                except Exception:
                    self._live_node_cache.pop(cache_key, None)

        resolved = None
        for candidate in candidates:
            try:
                node = rt.getNodeByName(candidate)
                if node is not None:
                    resolved = node
                    break
            except Exception:
                pass
        if resolved is not None:
            for candidate in candidates:
                self._live_node_cache[candidate.lower()] = resolved
            if len(self._live_node_cache) > 8192:
                self._live_node_cache.clear()
        return resolved

    def _setup_fbx_import(self, mode="skeleton"):
        rt = pymxs.runtime
        include_mesh = mode in {"merge", "create", "prop_create"}

        mode_map = {
            "skeleton": rt.name("exmerge"),
            "merge": rt.name("merge"),
            "create": rt.name("create"),
            "prop": rt.name("exmerge"),
            "prop_create": rt.name("create"),
        }
        rt.FBXImporterSetParam("Mode", mode_map.get(mode, rt.name("exmerge")))
        rt.FBXImporterSetParam("Animation", True)
        rt.FBXImporterSetParam("Skin", include_mesh)
        rt.FBXImporterSetParam("Shape", include_mesh)
        rt.FBXImporterSetParam("Cameras", False)
        rt.FBXImporterSetParam("Lights", False)
        rt.FBXImporterSetParam("SmoothingGroups", True)
        rt.FBXImporterSetParam("AxisConversion", False)
        rt.FBXImporterSetParam("ScaleConversion", False)
        rt.FBXImporterSetParam("UpAxis", "Z")

    def _remove_import_root_nodes(self, imported_nodes):
        rt = pymxs.runtime
        if not imported_nodes:
            return 0

        imported_handles = set()
        for node in imported_nodes:
            try:
                imported_handles.add(int(node.handle))
            except Exception:
                continue

        candidates = []
        for node in imported_nodes:
            try:
                node_name = str(node.name).lower()
            except Exception:
                continue
            if "boneroot" not in node_name:
                continue

            try:
                parent = node.parent
                parent_handle = int(parent.handle) if parent is not None else None
            except Exception:
                parent_handle = None

            if parent_handle in imported_handles:
                continue
            candidates.append(node)

        removed = 0
        for root in candidates:
            try:
                parent = root.parent
            except Exception:
                parent = None

            children = []
            try:
                for child in root.children:
                    children.append(child)
            except Exception:
                children = []

            for child in children:
                try:
                    child.parent = parent
                except Exception:
                    continue

            try:
                rt.delete(root)
                removed += 1
            except Exception:
                continue

        return removed

    def _parse_message(self, raw):
        text = (raw or "").replace("\r", "").replace("\n", "").strip()
        if not text:
            return ("invalid", "", "", "", "", False)

        parts = text.split("|")
        if len(parts) >= 3 and parts[0] == PROTOCOL_VERSION:
            mode = parts[1].strip().lower()
            if mode == "live_data":
                payload = "|".join(parts[2:]).strip()
                sub = payload.split("|", 2)
                frame_str = sub[0] if sub else "0"
                data_blob = sub[1] if len(sub) > 1 else ""
                morph_blob = sub[2] if len(sub) > 2 else ""
                return ("live", frame_str, data_blob, morph_blob, "")

            fbx_path = parts[2].strip() if len(parts) > 2 else ""
            prefix = ""
            suffix = ""
            remove_root = False
            for token in parts[3:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip().lower()
                value = unquote(value.strip())
                if key == "prefix":
                    prefix = value
                elif key == "suffix":
                    suffix = value
                elif key == "remove_root":
                    remove_root = str(value).strip().lower() in {"1", "true", "yes", "on"}
            return ("import", mode, fbx_path, prefix, suffix, remove_root)

        fbx_path = parts[0].strip() if parts else ""
        mode = parts[1].strip().lower() if len(parts) > 1 else "skeleton"
        return ("import", mode, fbx_path, "", "", False)

    def _normalize_quat(self, qx, qy, qz, qw):
        mag2 = (qx * qx) + (qy * qy) + (qz * qz) + (qw * qw)
        if mag2 <= 1.0e-12:
            return (0.0, 0.0, 0.0, 1.0)
        inv_mag = 1.0 / math.sqrt(mag2)
        return (qx * inv_mag, qy * inv_mag, qz * inv_mag, qw * inv_mag)

    def _lock_quat_hemisphere(self, node_key, quat_vals):
        if len(self._last_quat_by_node) > 8192 and node_key not in self._last_quat_by_node:
            self._last_quat_by_node.clear()
        prev = self._last_quat_by_node.get(node_key)
        qx, qy, qz, qw = quat_vals
        flipped = False
        if prev is not None:
            dot = (prev[0] * qx) + (prev[1] * qy) + (prev[2] * qz) + (prev[3] * qw)
            if dot < 0.0:
                qx, qy, qz, qw = (-qx, -qy, -qz, -qw)
                flipped = True
        self._last_quat_by_node[node_key] = (qx, qy, qz, qw)
        return (qx, qy, qz, qw), flipped

    def _normalize_morph_key(self, name):
        return re.sub(r"[^a-z0-9]+", "", str(name).lower())

    def _extract_morph_channel_label(self, sub_anim_name):
        text = str(sub_anim_name).strip()
        if not text:
            return ""
        if text.startswith("#"):
            text = text[1:]
        text = re.sub(r"^_?\d+__", "", text)
        text = re.sub(r"___.*$", "", text)
        text = text.replace("__", "_").strip(" _")
        return text

    def _morph_alias_keys(self, label):
        raw = str(label).strip()
        if not raw:
            return []
        candidates = [raw]
        lowered = raw.lower()
        for prefix in ("viseme_", "viseme", "expression_", "expression", "expr_", "expr", "morph_", "morph"):
            if lowered.startswith(prefix):
                trimmed = raw[len(prefix):].lstrip("_")
                if trimmed:
                    candidates.append(trimmed)
        if lowered.startswith("v_"):
            candidates.append(raw[2:])
        else:
            candidates.append("V_" + raw)
        return list({self._normalize_morph_key(x) for x in candidates if x})

    def _refresh_morph_cache(self, force=False):
        now = time.time()
        if (not force) and self._morph_cache and (now - self._morph_cache_built_at) < self._morph_cache_interval:
            return

        rt = pymxs.runtime
        cache = {}
        try:
            for node in rt.objects:
                mods = getattr(node, "modifiers", None)
                if mods is None:
                    continue
                try:
                    mod_count = int(mods.count)
                except Exception:
                    mod_count = 0
                if mod_count <= 0:
                    continue

                try:
                    node_name = str(node.name)
                except Exception:
                    continue

                for mod_index in range(1, mod_count + 1):
                    try:
                        mod = mods[mod_index]
                    except Exception:
                        continue
                    try:
                        if str(rt.classOf(mod)).lower() != "morpher":
                            continue
                    except Exception:
                        continue

                    try:
                        sub_count = int(mod.numsubs)
                    except Exception:
                        sub_count = 0
                    if sub_count <= 0:
                        continue

                    for sub_index in range(1, sub_count + 1):
                        try:
                            sub_name = rt.getSubAnimName(mod, sub_index)
                        except Exception:
                            continue
                        label = self._extract_morph_channel_label(sub_name)
                        if not label:
                            continue

                        target_key = f"{node_name}|{mod_index}|{sub_index}"
                        target = (node_name, mod_index, sub_index, target_key)
                        for alias in self._morph_alias_keys(label):
                            bucket = cache.setdefault(alias, [])
                            exists = False
                            for existing in bucket:
                                if existing[3] == target_key:
                                    exists = True
                                    break
                            if not exists:
                                bucket.append(target)
        except Exception as exc:
            self._log(f"Morph cache build error: {exc}")

        self._morph_cache = cache
        self._morph_cache_built_at = now

    def _to_morph_percent(self, value):
        v = _safe_float(value, 0.0)
        if abs(v) <= 1.5:
            v *= 100.0
        if v > 100.0:
            return 100.0
        if v < -100.0:
            return -100.0
        return v

    def _apply_morph_blob(self, morph_blob):
        rt = pymxs.runtime
        raw_entries = [e for e in (morph_blob or "").split(";") if e]
        if not raw_entries:
            return (0, 0, 0, 0)

        parsed = []
        for entry in raw_entries:
            if "=" not in entry:
                continue
            name, value = entry.split("=", 1)
            name = name.strip()
            if not name:
                continue
            parsed.append((name, value))
        if not parsed:
            return (0, 0, 0, 0)

        self._refresh_morph_cache(force=False)
        applied_writes = 0
        matched_entries = 0

        def apply_one(morph_name, morph_value):
            nonlocal applied_writes
            key = self._normalize_morph_key(morph_name)
            targets = self._morph_cache.get(key)
            if not targets:
                return False

            value_percent = self._to_morph_percent(morph_value)
            wrote_any = False
            for node_name, mod_index, sub_index, _target_key in targets:
                try:
                    node = rt.getNodeByName(node_name)
                    if node is None:
                        continue
                    mods = getattr(node, "modifiers", None)
                    if mods is None or mods.count < mod_index:
                        continue
                    mod = mods[mod_index]
                    if mod is None or mod.numsubs < sub_index:
                        continue
                    # pymxs sub-anim indexing is 0-based, while getSubAnimName is 1-based.
                    mod[sub_index - 1].value = value_percent
                    applied_writes += 1
                    wrote_any = True
                except Exception:
                    continue
            return wrote_any

        unresolved = []
        for morph_name, morph_value in parsed:
            if apply_one(morph_name, morph_value):
                matched_entries += 1
            else:
                unresolved.append((morph_name, morph_value))

        if unresolved:
            self._refresh_morph_cache(force=True)
            second_unresolved = []
            for morph_name, morph_value in unresolved:
                if apply_one(morph_name, morph_value):
                    matched_entries += 1
                else:
                    second_unresolved.append((morph_name, morph_value))
            unresolved = second_unresolved

        return (
            len(parsed),
            matched_entries,
            applied_writes,
            len(unresolved),
            list(parsed[:64]),
            [n for n, _ in unresolved[:64]],
        )

    def _server_loop(self):
        while self.running:
            try:
                try:
                    client, _addr = self.server_socket.accept()
                except socket.timeout:
                    continue

                with client:
                    client.settimeout(1.0)
                    chunks = []
                    while self.running:
                        try:
                            data = client.recv(8192)
                        except socket.timeout:
                            continue
                        if not data:
                            break
                        chunks.append(data)

                msg = b"".join(chunks).decode("utf-8", errors="replace").strip()
                if not msg:
                    continue
                parsed = self._parse_message(msg)
                payload_type = parsed[0]
                with self._lock:
                    if payload_type == "live":
                        self._latest_live = (parsed[1], parsed[2], parsed[3])
                    elif payload_type == "import":
                        self._import_queue.append((parsed[1], parsed[2], parsed[3], parsed[4], parsed[5]))
            except Exception as exc:
                if self.running:
                    self._log(f"Listener socket error: {exc}")

    def _process_import(self, mode, fbx_path, prefix="", suffix="", remove_root=False):
        rt = pymxs.runtime
        if not fbx_path:
            self._log("Invalid message: missing FBX path.")
            return
        if not rt.doesFileExist(fbx_path):
            self._log(f"File not found: {fbx_path}")
            return

        try:
            self._log(f"Importing ({mode}): {fbx_path}")
            before_handles = self._snapshot_scene_handles()
            self._setup_fbx_import(mode=mode)
            rt.importFile(fbx_path, rt.name("noPrompt"))
            imported_nodes = self._collect_imported_nodes(before_handles)
            if remove_root:
                removed = self._remove_import_root_nodes(imported_nodes)
                if removed > 0:
                    self._log(f"Removed {removed} imported root bone node(s).")
            renamed = self._apply_import_name_affixes(imported_nodes, prefix, suffix)
            if renamed > 0:
                self._log(f"Applied prefix/suffix to {renamed} imported node(s).")
            self._log("Import Complete.")
        except Exception as exc:
            self._log(f"Import failed: {exc}")

    def _process_live_data(self, frame_str, data_blob, morph_blob=""):
        rt = pymxs.runtime
        try:
            frame_num = int(frame_str)
        except Exception:
            frame_num = 0

        entries = [e for e in data_blob.split(";") if e]
        has_morph_data = bool(morph_blob and morph_blob.strip())
        if not entries and not has_morph_data:
            return

        resolved_count = 0
        sample_pairs = []
        direct_applied = 0
        quat_flipped = 0
        morph_entry_count = 0
        morph_matched = 0
        morph_applied = 0
        morph_unmatched = 0
        morph_pairs = []
        morph_unmatched_names = []

        with pymxs.undo(False):
            prev_anim_button = None
            try:
                prev_anim_button = rt.animButtonState
                rt.animButtonState = False
            except Exception:
                prev_anim_button = None

            try:
                if self.sync_live_timeline and frame_num > 0 and rt.sliderTime != frame_num:
                    rt.sliderTime = frame_num
            except Exception:
                pass

            if entries:
                find_live_node = self._find_live_node
                normalize_quat = self._normalize_quat
                lock_quat = self._lock_quat_hemisphere
                point3 = rt.Point3
                quat_ctor = rt.quat
                for entry in entries:
                    fields = entry.split(",", 7)
                    if len(fields) < 8:
                        continue
                    node_name = fields[0]
                    node = find_live_node(node_name)
                    if node is None:
                        if len(sample_pairs) < 8:
                            sample_pairs.append(f"{node_name} -> <missing>")
                        continue
                    resolved_count += 1
                    if len(sample_pairs) < 8:
                        try:
                            sample_pairs.append(f"{node_name} -> {node.name}")
                        except Exception:
                            sample_pairs.append(f"{node_name} -> <resolved>")

                    px = _safe_float(fields[1], 0.0)
                    py = _safe_float(fields[2], 0.0)
                    pz = _safe_float(fields[3], 0.0)
                    qx = _safe_float(fields[4], 0.0)
                    qy = _safe_float(fields[5], 0.0)
                    qz = _safe_float(fields[6], 0.0)
                    qw = _safe_float(fields[7], 1.0)
                    node_key = ""
                    try:
                        node_key = str(node.name).lower()
                    except Exception:
                        node_key = str(node_name).lower()
                    quat_vals = normalize_quat(qx, qy, qz, qw)
                    quat_vals, was_flipped = lock_quat(node_key, quat_vals)
                    if was_flipped:
                        quat_flipped += 1
                    live_rot = quat_ctor(quat_vals[0], quat_vals[1], quat_vals[2], quat_vals[3])
                    node.pos = point3(px, py, pz)
                    node.rotation = live_rot
                    direct_applied += 1

            if has_morph_data:
                (
                    morph_entry_count,
                    morph_matched,
                    morph_applied,
                    morph_unmatched,
                    morph_pairs,
                    morph_unmatched_names,
                ) = self._apply_morph_blob(morph_blob)

            if prev_anim_button is not None:
                try:
                    rt.animButtonState = prev_anim_button
                except Exception:
                    pass

        self._debug_last_live = {
            "frame": frame_num,
            "entries": len(entries),
            "resolved": resolved_count,
            "samples": sample_pairs,
            "direct_applied": direct_applied,
            "quat_flipped": quat_flipped,
            "morph_entries": morph_entry_count,
            "morph_matched": morph_matched,
            "morph_applied": morph_applied,
            "morph_unmatched": morph_unmatched,
            "morph_received": morph_pairs,
            "morph_unmatched_names": morph_unmatched_names,
        }

    @QtCore.Slot()
    def _drain_queues(self):
        imports = []
        live_msg = None

        with self._lock:
            while self._import_queue:
                imports.append(self._import_queue.popleft())
            if self._latest_live is not None:
                live_msg = self._latest_live
                self._latest_live = None

        for item in imports:
            if len(item) >= 5:
                mode, fbx_path, prefix, suffix, remove_root = item[0], item[1], item[2], item[3], item[4]
            elif len(item) >= 4:
                mode, fbx_path, prefix, suffix = item[0], item[1], item[2], item[3]
                remove_root = False
            else:
                mode, fbx_path = item[0], item[1]
                prefix, suffix = "", ""
                remove_root = False
            self._process_import(mode, fbx_path, prefix, suffix, remove_root)

        if live_msg is not None:
            self._process_live_data(live_msg[0], live_msg[1], live_msg[2])

    def start(self):
        if self.running:
            return True

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((HOST, LISTEN_PORT))
            self.server_socket.listen(8)
            self.server_socket.settimeout(0.5)
        except Exception as exc:
            self._log(f"Failed to start listener on {HOST}:{LISTEN_PORT} - {exc}")
            self.server_socket = None
            return False

        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

        if self.timer is None:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self._drain_queues)
        self.timer.setInterval(DRAIN_INTERVAL_MS)
        self.timer.start()

        self._log(f"Listening on port {LISTEN_PORT}")
        return True

    def stop(self):
        self.running = False
        if self.timer is not None:
            self.timer.stop()

        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        if self.server_thread is not None:
            self.server_thread.join(timeout=1.0)
            self.server_thread = None

        with self._lock:
            self._import_queue.clear()
            self._latest_live = None

        self._last_quat_by_node = {}
        self._live_node_cache = {}
        self._morph_cache = {}
        self._morph_cache_built_at = 0.0
        self._log("Stopped.")
        return True

    def status(self):
        return bool(self.running)

    def send_to_iclone(self, fbx_path, mode="prop"):
        payload = f"{PROTOCOL_VERSION}|{mode}|{fbx_path}"
        try:
            with socket.create_connection((HOST, ICLONE_PORT), timeout=2.0) as sock:
                sock.sendall(payload.encode("utf-8"))
            return True
        except Exception:
            return False


_bridge = None


def _get_bridge():
    global _bridge
    if _bridge is None:
        _bridge = MaxSyncPythonBridge()
    return _bridge


def start_listener():
    return _get_bridge().start()


def stop_listener():
    return _get_bridge().stop()


def status_listener():
    return _get_bridge().status()


def send_to_iclone(fbx_path, mode="prop"):
    return _get_bridge().send_to_iclone(fbx_path, mode=mode)


def get_debug_snapshot():
    bridge = _get_bridge()
    data = bridge._debug_last_live
    samples = " | ".join(data.get("samples", []))
    received_parts = [f"{n}={v}" for n, v in data.get("morph_received", [])]
    unmatched_names = data.get("morph_unmatched_names", [])
    lines = [
        f"frame={data.get('frame', 0)}, entries={data.get('entries', 0)}, "
        f"resolved={data.get('resolved', 0)}, direct_applied={data.get('direct_applied', 0)}, "
        f"quat_flipped={data.get('quat_flipped', 0)}",
        f"morph_entries={data.get('morph_entries', 0)}, morph_matched={data.get('morph_matched', 0)}, "
        f"morph_applied={data.get('morph_applied', 0)}, morph_unmatched={data.get('morph_unmatched', 0)}",
    ]
    if received_parts:
        lines.append("morph values: " + "  |  ".join(received_parts))
    if unmatched_names:
        lines.append("UNMATCHED names (not found in Max Morpher): " + ", ".join(unmatched_names))
    lines.append(f"samples={samples}")
    return "\n".join(lines)


def get_morph_cache_keys():
    """Dump all Morpher channel keys currently in the Max-side cache.
    Call this to see what normalized names Max has found — compare with
    the 'morph values' line from get_debug_snapshot() to spot mismatches.
    """
    bridge = _get_bridge()
    bridge._refresh_morph_cache(force=True)
    cache = bridge._morph_cache
    if not cache:
        return "(morph cache is empty — no Morpher modifiers found in scene)"
    lines = []
    for key in sorted(cache.keys()):
        targets = cache[key]
        target_strs = [f"{t[0]} mod[{t[1]}] ch[{t[2]}]" for t in targets]
        lines.append(f"{key!r}: {', '.join(target_strs)}")
    return "\n".join(lines)


def get_timer_interval_ms():
    bridge = _get_bridge()
    if bridge.timer is None:
        return -1
    return int(bridge.timer.interval())
