# -*- coding: utf-8 -*-
"""
MAXsync – Bidirectional iClone <-> 3ds Max Live Link
Supports avatar animation sync, prop transfer both ways.
"""

# ───────── Reallusion plugin handshake ─────────
rl_plugin_info = {"ap": "iClone", "ap_version": "8.0"}

# ───────── Imports ─────────
import os
import tempfile
import socket
import threading
import time
from urllib.parse import quote
import RLPy
from PySide2 import QtWidgets, QtCore, QtGui
from shiboken2 import wrapInstance

# ───────── CONSTANTS ─────────
MAX_HOST = '127.0.0.1'
MAX_PORT_SEND = 29999      # Send to Max (avatar/prop)
MAX_PORT_RECEIVE = 29998   # Receive from Max (prop)
TEMP_AVATAR_FBX = "MaxSync_Avatar.fbx"
TEMP_PROP_FBX = "MaxSync_Prop.fbx"
PROTOCOL_VERSION = "MSYNC1"
BUFFER_SIZE = 4096
LIVE_UPDATE_RATE = 0.05
LIVE_CONNECT_TIMEOUT = 0.6
LIVE_SELECTION_REFRESH = 0.5
LIVE_POS_EPSILON = 1e-4
LIVE_ROT_EPSILON = 1e-4

# ───────── GLOBAL STATE ─────────
dock_ui = {}
listener_thread = None
listener_running = False
server_socket = None
live_stream_thread = None
live_stream_running = False


def _sanitize_affix(text):
    s = str(text or "").strip()
    for ch in ("|", ";", ",", "\r", "\n"):
        s = s.replace(ch, "_")
    return s


def _build_message(mode, fbx_path):
    return f"{PROTOCOL_VERSION}|{mode}|{fbx_path}"


def _status_to_text(status):
    """Return a readable RStatus label instead of a SWIG proxy repr."""
    if status is None:
        return "None"

    status_enum = getattr(RLPy, "RStatus", None)
    if status_enum is not None:
        for name in dir(status_enum):
            if name.startswith("_"):
                continue
            try:
                candidate = getattr(status_enum, name)
                if status == candidate:
                    try:
                        return f"{name} ({int(candidate)})"
                    except Exception:
                        return name
            except Exception:
                continue

    try:
        return str(status)
    except Exception:
        return repr(status)


def _flag_enabled(mask, flag):
    try:
        return (int(mask) & int(flag)) != 0
    except Exception:
        return False


def _clear_flag(mask, flag):
    try:
        return int(mask) & ~int(flag)
    except Exception:
        return mask


def _parse_message(data):
    """
    Protocol:
      New:  MSYNC1|<mode>|<fbx_path>
      Old:  <fbx_path>|<mode>   or just <fbx_path>
    """
    text = (data or "").strip()
    if not text:
        return None, None

    parts = text.split("|")
    if len(parts) >= 3 and parts[0] == PROTOCOL_VERSION:
        mode = parts[1].strip().lower()
        fbx_path = "|".join(parts[2:]).strip()
        return fbx_path, mode

    # Legacy compatibility
    fbx_path = parts[0].strip()
    mode = parts[1].strip().lower() if len(parts) > 1 else "prop"
    return fbx_path, mode


def _recv_all(client):
    chunks = []
    while True:
        chunk = client.recv(BUFFER_SIZE)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


def _safe_call(obj, method_name):
    fn = getattr(obj, method_name, None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return None
    return None


def _read_component(obj, names):
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _vector3_to_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except Exception:
            return None

    x = _read_component(value, ("x", "X", "XAxis"))
    y = _read_component(value, ("y", "Y", "YAxis"))
    z = _read_component(value, ("z", "Z", "ZAxis"))
    if x is None or y is None or z is None:
        return None
    return [x, y, z]


def _quat_to_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except Exception:
            return None

    x = _read_component(value, ("x", "X"))
    y = _read_component(value, ("y", "Y"))
    z = _read_component(value, ("z", "Z"))
    w = _read_component(value, ("w", "W"))
    if x is None or y is None or z is None or w is None:
        return None
    return [x, y, z, w]


def _extract_world_transform(obj):
    transform = None
    for method_name in ("GetWorldTransform", "WorldTransform", "GetTransform"):
        transform = _safe_call(obj, method_name)
        if transform is not None:
            break
    if transform is None:
        return None, None

    pos = None
    for method_name in ("T", "Translation", "GetTranslation"):
        pos = _safe_call(transform, method_name)
        if pos is not None:
            break
    if pos is None:
        for attr_name in ("translation", "t", "position", "pos"):
            pos = getattr(transform, attr_name, None)
            if pos is not None:
                break

    rot = None
    for method_name in ("R", "Rotation", "GetRotation"):
        rot = _safe_call(transform, method_name)
        if rot is not None:
            break
    if rot is None:
        for attr_name in ("rotation", "r", "rot"):
            rot = getattr(transform, attr_name, None)
            if rot is not None:
                break

    return _vector3_to_list(pos), _quat_to_list(rot)


def _object_name(obj):
    for method_name in ("GetName",):
        name = _safe_call(obj, method_name)
        if name:
            return str(name)
    for attr_name in ("name", "Name"):
        name = getattr(obj, attr_name, None)
        if name:
            return str(name)
    return None


def _iter_children(obj):
    """Best-effort child iterator across iClone object/bone APIs."""
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
            child = None
            try:
                getter = getattr(obj, "GetChild", None)
                if callable(getter):
                    child = getter(i)
            except Exception:
                child = None
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
        try:
            for child in _iter_children(node):
                stack.append(child)
        except Exception:
            continue


def _avatar_stream_nodes(avatar):
    """Collect avatar skeleton bones when possible; fallback to avatar object."""
    skeleton = _safe_call(avatar, "GetSkeletonComponent")
    if skeleton is None:
        return [avatar]

    root_bone = _safe_call(skeleton, "GetRootBone")
    if root_bone is None:
        return [avatar]

    nodes = [node for node in _walk_descendants(root_bone)]
    return nodes if nodes else [avatar]


def _selected_live_target():
    selected = RLPy.RScene.GetSelectedObjects()
    if not selected:
        return [], ""

    # Live streaming must bind to a single currently selected object only.
    # iClone returns a selection list; the latest/active pick is typically last.
    obj = selected[-1]
    selected_name = _resolve_stream_owner_name(obj) or ""

    obj_type = None
    try:
        obj_type = obj.GetType()
    except Exception:
        obj_type = None

    if obj_type == RLPy.EObjectType_Avatar:
        return _avatar_stream_nodes(obj), selected_name
    return [obj], selected_name


def _resolve_stream_owner_name(obj):
    if obj is None:
        return ""

    obj_type = None
    try:
        obj_type = obj.GetType()
    except Exception:
        obj_type = None
    if obj_type == RLPy.EObjectType_Avatar:
        return _object_name(obj) or ""

    for method_name in ("GetOwner", "GetOwnerObject", "GetAvatar", "GetRelatedAvatar"):
        owner = _safe_call(obj, method_name)
        if owner is None:
            continue
        try:
            if owner.GetType() == RLPy.EObjectType_Avatar:
                return _object_name(owner) or ""
        except Exception:
            pass

    parent = obj
    for _ in range(64):
        parent = _safe_call(parent, "GetParent")
        if parent is None:
            break
        try:
            if parent.GetType() == RLPy.EObjectType_Avatar:
                return _object_name(parent) or ""
        except Exception:
            pass

    return _object_name(obj) or ""


def _current_frame_index():
    try:
        current_time = RLPy.RGlobal.GetTime()
    except Exception:
        return 0

    value = _safe_call(current_time, "GetFrameIndex")
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass
    return 0


def _encode_live_entry(name, pos, rot, prefix=""):
    full_name = f"{prefix}{name}"
    safe_name = str(full_name).replace("|", "_").replace(";", "_").replace(",", "_")
    return (
        f"{safe_name},{pos[0]},{pos[1]},{pos[2]},"
        f"{rot[0]},{rot[1]},{rot[2]},{rot[3]}"
    )


def _transform_changed(prev_pos, prev_rot, pos, rot):
    if prev_pos is None or prev_rot is None:
        return True
    for a, b in zip(prev_pos, pos):
        if abs(a - b) > LIVE_POS_EPSILON:
            return True
    for a, b in zip(prev_rot, rot):
        if abs(a - b) > LIVE_ROT_EPSILON:
            return True
    return False


def live_stream_loop(widget):
    """Background thread: streams selected object transforms from iClone to 3ds Max."""
    global live_stream_running
    warned_connection = False
    packet_count = 0
    cached_nodes = []
    cached_selected_name = ""
    next_selection_refresh = 0.0
    last_sent_transforms = {}
    last_sent_prefix = ""

    try:
        while live_stream_running:
            time.sleep(LIVE_UPDATE_RATE)

            now = time.time()
            if (not cached_nodes) or now >= next_selection_refresh:
                cached_nodes, cached_selected_name = _selected_live_target()
                next_selection_refresh = now + LIVE_SELECTION_REFRESH

            stream_nodes = cached_nodes
            if not stream_nodes:
                last_sent_transforms = {}
                last_sent_prefix = ""
                continue

            prefix = _sanitize_affix(widget.get_live_name_prefix(cached_selected_name))
            current_transforms = {}
            entries = []
            for obj in stream_nodes:
                name = _object_name(obj)
                if not name:
                    continue
                pos, rot = _extract_world_transform(obj)
                if pos is None or rot is None:
                    continue
                current_transforms[name] = (pos, rot)
                entries.append(_encode_live_entry(name, pos, rot, prefix))

            if not entries:
                last_sent_transforms = {}
                last_sent_prefix = ""
                continue

            # Wait/send only when data changed (or selection/prefix changed).
            should_send = prefix != last_sent_prefix
            if not should_send:
                if set(current_transforms.keys()) != set(last_sent_transforms.keys()):
                    should_send = True
                else:
                    for name, (pos, rot) in current_transforms.items():
                        prev_pos, prev_rot = last_sent_transforms.get(name, (None, None))
                        if _transform_changed(prev_pos, prev_rot, pos, rot):
                            should_send = True
                            break

            if not should_send:
                continue

            frame_index = _current_frame_index()
            payload = f"{frame_index}|{';'.join(entries)}"
            message = _build_message("live_data", payload)

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                    client.settimeout(LIVE_CONNECT_TIMEOUT)
                    client.connect((MAX_HOST, MAX_PORT_SEND))
                    client.sendall(message.encode("utf-8"))
                warned_connection = False
                packet_count += 1
                last_sent_transforms = current_transforms
                last_sent_prefix = prefix
                if packet_count == 1 or packet_count % 100 == 0:
                    widget.update_status.emit(
                        f"Live stream active: frame={frame_index}, nodes={len(entries)}"
                    )
            except ConnectionRefusedError:
                if not warned_connection:
                    widget.update_status.emit("Live stream waiting for 3ds Max listener on port 29999...")
                    warned_connection = True
                time.sleep(0.5)
            except socket.timeout:
                widget.update_status.emit("Live stream timeout: 3ds Max listener is overloaded or not responding.")
                time.sleep(0.2)
            except Exception as e:
                widget.update_status.emit(f"Live stream socket error: {str(e)}")
                time.sleep(0.2)
    finally:
        live_stream_running = False
        widget.live_state_changed.emit(False, "Live stream stopped.")


# ───────── PROP LISTENER ─────────
def listener_loop(widget):
    """Background thread that listens for incoming props from Max."""
    global listener_running, server_socket

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((MAX_HOST, MAX_PORT_RECEIVE))
        server_socket.listen(1)
        server_socket.settimeout(1.0)

        widget.update_status.emit(f"Listening for props on {MAX_PORT_RECEIVE}")

        while listener_running:
            try:
                client, addr = server_socket.accept()
                data = _recv_all(client)
                client.close()

                if data:
                    widget.prop_received.emit(data)

            except socket.timeout:
                continue
            except OSError:
                if listener_running:
                    widget.update_status.emit("Listener socket error while receiving data.")
            except Exception as e:
                widget.update_status.emit(f"Listener receive error: {str(e)}")

    except Exception as e:
        widget.update_status.emit(f"Listener error: {str(e)}")
    finally:
        listener_running = False
        if server_socket:
            server_socket.close()
            server_socket = None


# ───────── UI WIDGET ─────────
class MaxSyncWidget(QtWidgets.QWidget):
    update_status = QtCore.Signal(str)
    prop_received = QtCore.Signal(str)
    live_state_changed = QtCore.Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # ─── Status ───
        self.lbl_info = QtWidgets.QLabel("Select avatar or prop and click SYNC.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        # ═══════════════════════════════════════════
        # ─── AVATAR SYNC SECTION ───
        # ═══════════════════════════════════════════
        grp_avatar = QtWidgets.QGroupBox("Avatar → 3ds Max")
        vbox_avatar = QtWidgets.QVBoxLayout(grp_avatar)

        self.btn_sync = QtWidgets.QPushButton("SYNC Avatar")
        self.btn_sync.setMinimumHeight(40)
        self.btn_sync.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_sync.clicked.connect(self.on_sync_avatar)
        vbox_avatar.addWidget(self.btn_sync)

        self.btn_live = QtWidgets.QPushButton("Start Live Stream (iClone → Max)")
        self.btn_live.setCheckable(True)
        self.btn_live.setMinimumHeight(34)
        self.btn_live.setStyleSheet("font-weight: bold; background-color: #455A64; color: white;")
        self.btn_live.clicked.connect(self.on_live_toggle)
        vbox_avatar.addWidget(self.btn_live)

        # Options
        self.chk_root_motion = QtWidgets.QCheckBox("Export Root Motion")
        self.chk_root_motion.setChecked(True)

        self.chk_remove_mesh = QtWidgets.QCheckBox("Remove Mesh (Skeleton Only)")
        self.chk_remove_mesh.setChecked(True)

        self.chk_create_char = QtWidgets.QCheckBox("Create Character (Fresh Import)")
        self.chk_create_char.setEnabled(False)
        self.chk_remove_mesh.stateChanged.connect(self._on_remove_mesh_changed)

        self.chk_tpose_first = QtWidgets.QCheckBox("T-Pose on First Frame")
        self.chk_zero_root = QtWidgets.QCheckBox("Zero Motion Root")
        self.chk_remove_root = QtWidgets.QCheckBox("Remove Bone Root")
        self.chk_name_prefix = QtWidgets.QCheckBox("Name_")
        self.chk_name_prefix.setToolTip("Use selected iClone object name + '_' as prefix")
        self.edit_name_prefix = QtWidgets.QLineEdit()
        self.edit_name_prefix.setPlaceholderText("Name Prefix (optional)")

        vbox_avatar.addWidget(self.chk_root_motion)
        vbox_avatar.addWidget(self.chk_remove_mesh)
        vbox_avatar.addWidget(self.chk_create_char)
        vbox_avatar.addWidget(self.chk_tpose_first)
        vbox_avatar.addWidget(self.chk_zero_root)
        vbox_avatar.addWidget(self.chk_remove_root)
        vbox_avatar.addWidget(QtWidgets.QLabel("Multi-character naming"))
        vbox_avatar.addWidget(self.chk_name_prefix)
        vbox_avatar.addWidget(self.edit_name_prefix)

        self.use_selected_name_prefix = False
        self.name_prefix_value = ""
        self.chk_name_prefix.stateChanged.connect(self._on_name_prefix_mode_changed)
        self.edit_name_prefix.textChanged.connect(self._on_name_prefix_changed)
        self._on_name_prefix_mode_changed()

        layout.addWidget(grp_avatar)

        # ═══════════════════════════════════════════
        # ─── PROP SYNC SECTION ───
        # ═══════════════════════════════════════════
        grp_prop = QtWidgets.QGroupBox("Prop Transfer")
        vbox_prop = QtWidgets.QVBoxLayout(grp_prop)

        # Send prop to Max
        self.btn_send_prop = QtWidgets.QPushButton("Send Prop → Max")
        self.btn_send_prop.setMinimumHeight(35)
        self.btn_send_prop.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white;")
        self.btn_send_prop.clicked.connect(self.on_send_prop)
        vbox_prop.addWidget(self.btn_send_prop)

        self.chk_create_prop = QtWidgets.QCheckBox("Create Prop (Fresh Import)")
        vbox_prop.addWidget(self.chk_create_prop)

        # Receive prop from Max
        self.btn_listen = QtWidgets.QPushButton("Listener: Always On (Max → iClone)")
        self.btn_listen.setCheckable(True)
        self.btn_listen.clicked.connect(self.on_listen_toggle)
        vbox_prop.addWidget(self.btn_listen)

        layout.addWidget(grp_prop)

        # ─── Axis Group ───
        grp_axis = QtWidgets.QGroupBox("Up Axis")
        vbox_axis = QtWidgets.QVBoxLayout(grp_axis)

        self.radio_none = QtWidgets.QRadioButton("Z Up (Default)")
        self.radio_yup = QtWidgets.QRadioButton("Y Up (Maya/Blender)")
        self.radio_ue4 = QtWidgets.QRadioButton("UE4 Bone Axis")

        self.radio_none.setChecked(True)

        axis_group = QtWidgets.QButtonGroup(self)
        axis_group.addButton(self.radio_none)
        axis_group.addButton(self.radio_yup)
        axis_group.addButton(self.radio_ue4)

        vbox_axis.addWidget(self.radio_none)
        vbox_axis.addWidget(self.radio_yup)
        vbox_axis.addWidget(self.radio_ue4)

        layout.addWidget(grp_axis)

        layout.addStretch()

        # Connect signals
        self.update_status.connect(self._on_status_update)
        self.prop_received.connect(self._on_prop_received)
        self.live_state_changed.connect(self._on_live_state_changed)

        # Keep Max -> iClone listener always running while plugin is open.
        self._start_listener_always_on()

    def _on_remove_mesh_changed(self, state):
        mesh_included = state == QtCore.Qt.Unchecked
        self.chk_create_char.setEnabled(mesh_included)
        if not mesh_included:
            self.chk_create_char.setChecked(False)

    def _on_name_prefix_changed(self, _text=""):
        self.name_prefix_value = _sanitize_affix(self.edit_name_prefix.text())

    def _on_name_prefix_mode_changed(self, _state=0):
        self.use_selected_name_prefix = self.chk_name_prefix.isChecked()
        self.edit_name_prefix.setEnabled(True)
        self._on_name_prefix_changed()

    def _get_name_prefix_for_object(self, obj=None):
        if not self.use_selected_name_prefix:
            return ""
        base = _sanitize_affix(_resolve_stream_owner_name(obj))
        if base:
            return f"{base}_"
        return self.name_prefix_value

    def get_live_name_prefix(self, selected_name=""):
        if not self.use_selected_name_prefix:
            return ""
        base = _sanitize_affix(selected_name)
        if base:
            return f"{base}_"
        return self.name_prefix_value

    def _on_status_update(self, msg):
        self.lbl_info.setText(msg)

    def _on_live_state_changed(self, running, message):
        self._set_live_button_state(running)
        if message:
            self.lbl_info.setText(message)

    def _set_live_button_state(self, running):
        self.btn_live.blockSignals(True)
        self.btn_live.setChecked(running)
        self.btn_live.blockSignals(False)
        if running:
            self.btn_live.setText("Stop Live Stream (iClone → Max)")
            self.btn_live.setStyleSheet("font-weight: bold; background-color: #C62828; color: white;")
        else:
            self.btn_live.setText("Start Live Stream (iClone → Max)")
            self.btn_live.setStyleSheet("font-weight: bold; background-color: #455A64; color: white;")

    def _set_listener_ui_always_on(self):
        self.btn_listen.blockSignals(True)
        self.btn_listen.setChecked(True)
        self.btn_listen.blockSignals(False)
        self.btn_listen.setText("Listener: Always On (Max → iClone)")
        self.btn_listen.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_listen.setEnabled(False)

    def _start_listener_always_on(self):
        global listener_thread, listener_running
        if listener_running and listener_thread and listener_thread.is_alive():
            self._set_listener_ui_always_on()
            return

        listener_running = True
        listener_thread = threading.Thread(target=listener_loop, args=(self,), daemon=True)
        listener_thread.start()
        self._set_listener_ui_always_on()

    def start_live_stream(self):
        global live_stream_thread, live_stream_running
        if live_stream_running and live_stream_thread and live_stream_thread.is_alive():
            self._set_live_button_state(True)
            return

        live_stream_running = True
        live_stream_thread = threading.Thread(target=live_stream_loop, args=(self,), daemon=True)
        live_stream_thread.start()
        self._set_live_button_state(True)
        self.lbl_info.setText("Live stream started (iClone → Max).")

    def stop_live_stream(self):
        global live_stream_running
        live_stream_running = False
        self._set_live_button_state(False)
        self.lbl_info.setText("Stopping live stream...")

    def _on_prop_received(self, data):
        """Load received payload from Max."""
        fbx_path, mode = _parse_message(data)
        if not fbx_path:
            self.lbl_info.setText("Invalid message from 3ds Max.")
            return

        if not os.path.exists(fbx_path):
            self.lbl_info.setText(f"File not found: {fbx_path}")
            return

        self.lbl_info.setText(f"Loading: {os.path.basename(fbx_path)}")
        result = RLPy.RFileIO.LoadObject(fbx_path)
        if not result:
            self.lbl_info.setText("Failed to load from 3ds Max.")
            return

        if mode == "single_frame":
            self.lbl_info.setText(f"Single-frame object loaded: {os.path.basename(fbx_path)}")
        else:
            self.lbl_info.setText(f"Prop loaded: {os.path.basename(fbx_path)}")

    def get_options_bitmask(self):
        flags = 0
        if self.chk_root_motion.isChecked():
            flags |= RLPy.EExportFbxOptions_ExportRootMotion
        if self.chk_zero_root.isChecked():
            flags |= RLPy.EExportFbxOptions_ZeroMotionRoot
        if self.chk_remove_mesh.isChecked():
            flags |= RLPy.EExportFbxOptions_RemoveAllMesh
        if self.chk_tpose_first.isChecked():
            flags |= RLPy.EExportFbxOptions_TPoseOnMotionFirstFrame
        if self.chk_remove_root.isChecked():
            flags |= RLPy.EExportFbxOptions_RemoveBoneRoot
        return flags

    def get_axis_option(self, name_prefix=""):
        flags = RLPy.EExportFbxOptions2_ResetBoneScale

        # Prefix/suffix is applied after import on the Max side.
        # Keeping FBX export PrefixAndPostfix off avoids exporter failures.

        if self.radio_yup.isChecked():
            flags |= RLPy.EExportFbxOptions2_YUp
        elif self.radio_ue4.isChecked():
            flags |= RLPy.EExportFbxOptions2_UnrealEngine4BoneAxis

        return flags

    def on_sync_avatar(self):
        """Sync avatar animation to 3ds Max."""
        avatars = [o for o in RLPy.RScene.GetSelectedObjects() if o.GetType() == RLPy.EObjectType_Avatar]
        if not avatars:
            self.lbl_info.setText("Error: No avatar selected!")
            return
        if len(avatars) > 1:
            self.lbl_info.setText("Error: Select only one avatar.")
            return

        avatar = avatars[0]

        temp_dir = tempfile.gettempdir()
        fbx_path = os.path.join(temp_dir, TEMP_AVATAR_FBX)
        if os.path.exists(fbx_path):
            try:
                os.remove(fbx_path)
            except OSError:
                self.lbl_info.setText("Warning: could not replace temp avatar FBX file.")

        # Bake Motion
        start_tick = RLPy.RGlobal.GetStartTime()
        end_tick = RLPy.RGlobal.GetEndTime()

        fd, motion_path = tempfile.mkstemp(suffix=".rlmotion")
        os.close(fd)

        save_opt = RLPy.RSaveFileSetting()
        save_opt.SetSaveType(RLPy.ESaveFileType_Motion)
        save_opt.SetSaveRange(start_tick, end_tick)
        RLPy.RFileIO.SaveFile(avatar, save_opt, motion_path)

        # Export FBX
        name_prefix = self._get_name_prefix_for_object(avatar)
        opt1 = self.get_options_bitmask()
        opt2 = self.get_axis_option(name_prefix)
        opt3 = RLPy.EExportFbxOptions3__None

        self.lbl_info.setText("Exporting avatar...")
        QtWidgets.QApplication.processEvents()

        status = RLPy.RFileIO.ExportFbxFile(
            avatar, fbx_path,
            opt1, opt2, opt3,
            RLPy.EExportTextureSize_Original,
            RLPy.EExportTextureFormat_Default,
            motion_path
        )

        remove_root_flag = getattr(RLPy, "EExportFbxOptions_RemoveBoneRoot", 0)
        if status != RLPy.RStatus.Success and _flag_enabled(opt1, remove_root_flag):
            retry_opt1 = _clear_flag(opt1, remove_root_flag)
            retry_status = RLPy.RFileIO.ExportFbxFile(
                avatar, fbx_path,
                retry_opt1, opt2, opt3,
                RLPy.EExportTextureSize_Original,
                RLPy.EExportTextureFormat_Default,
                motion_path
            )
            if retry_status == RLPy.RStatus.Success:
                status = retry_status
                self.lbl_info.setText(
                    "Warning: Remove Bone Root failed on this avatar; exported without removing root."
                )
                QtWidgets.QApplication.processEvents()

        try:
            os.remove(motion_path)
        except OSError:
            self.lbl_info.setText("Warning: could not remove temporary motion cache file.")

        if status != RLPy.RStatus.Success:
            self.lbl_info.setText(f"Export Failed: {_status_to_text(status)}")
            return

        # Send to Max
        self._send_to_max(fbx_path, "avatar", name_prefix)

    def on_send_prop(self):
        """Send selected prop to 3ds Max."""
        props = [o for o in RLPy.RScene.GetSelectedObjects() if o.GetType() == RLPy.EObjectType_Prop]
        if not props:
            self.lbl_info.setText("Error: No prop selected!")
            return

        prop = props[0]

        temp_dir = tempfile.gettempdir()
        fbx_path = os.path.join(temp_dir, TEMP_PROP_FBX)
        if os.path.exists(fbx_path):
            try:
                os.remove(fbx_path)
            except OSError:
                self.lbl_info.setText("Warning: could not replace temp prop FBX file.")

        self.lbl_info.setText("Exporting prop...")
        QtWidgets.QApplication.processEvents()

        # Export prop with minimal options
        name_prefix = self._get_name_prefix_for_object(prop)
        opt1 = RLPy.EExportFbxOptions__None
        opt2 = RLPy.EExportFbxOptions2_ResetBoneScale
        opt3 = RLPy.EExportFbxOptions3__None

        status = RLPy.RFileIO.ExportFbxFile(
            prop, fbx_path,
            opt1, opt2, opt3,
            RLPy.EExportTextureSize_Original,
            RLPy.EExportTextureFormat_Default,
            ""
        )

        if status != RLPy.RStatus.Success:
            self.lbl_info.setText(f"Export Failed: {_status_to_text(status)}")
            return

        # Send to Max
        self._send_to_max(fbx_path, "prop", name_prefix)

    def _send_to_max(self, fbx_path, obj_type, name_prefix=""):
        """Send FBX file path to 3ds Max."""
        self.lbl_info.setText("Sending to 3ds Max...")
        QtWidgets.QApplication.processEvents()

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5.0)
                s.connect((MAX_HOST, MAX_PORT_SEND))

                if obj_type == "avatar":
                    if self.chk_remove_mesh.isChecked():
                        mode = "skeleton"
                    elif self.chk_create_char.isChecked():
                        mode = "create"
                    else:
                        mode = "merge"
                else:
                    mode = "prop_create" if self.chk_create_prop.isChecked() else "prop"

                message = _build_message(mode, fbx_path)
                prefix = _sanitize_affix(name_prefix)
                if prefix:
                    message += f"|prefix={quote(prefix, safe='')}"
                if obj_type == "avatar" and self.chk_remove_root.isChecked():
                    message += "|remove_root=1"
                s.sendall(message.encode('utf-8'))

            self.lbl_info.setText(f"Sent ({mode})!")
        except ConnectionRefusedError:
            self.lbl_info.setText("Connection Failed: Is MAXsync running in 3ds Max?")
        except Exception as e:
            self.lbl_info.setText(f"Socket Error: {str(e)}")

    def on_listen_toggle(self, checked):
        """Listener is always-on; keep this as a compatibility handler."""
        self._start_listener_always_on()
        self.lbl_info.setText("MAXsync listener (Max → iClone) is always on.")

    def on_live_toggle(self, checked):
        if checked:
            self.start_live_stream()
        else:
            self.stop_live_stream()

    def closeEvent(self, event):
        self.stop_live_stream()
        super().closeEvent(event)


# ───────── BOOTSTRAP ─────────
def open_max_sync():
    global dock_ui

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

    if "dock" not in dock_ui:
        dock = RLPy.RUi.CreateRDockWidget()
        dock.SetWindowTitle("MAXsync")
        dock.SetParent(RLPy.RUi.GetMainWindow())
        if all_areas:
            dock.SetAllowedAreas(all_areas)
        if all_features:
            dock.SetFeatures(all_features)

        qt_dock = wrapInstance(int(dock.GetWindow()), QtWidgets.QDockWidget)

        widget = MaxSyncWidget()
        dock_ui["widget"] = widget
        qt_dock.setWidget(widget)

        dock_ui["dock"] = dock

    # Re-apply docking behavior every open; fixes stale undockable state after reloads.
    dock = dock_ui["dock"]
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
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetClosable |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )
    except Exception:
        pass

    dock.Show()


def _add_menu_entry():
    main_qt = wrapInstance(int(RLPy.RUi.GetMainWindow()), QtWidgets.QMainWindow)
    menu = main_qt.menuBar().findChild(QtWidgets.QMenu, "clone_tools_menu")
    if menu is None:
        menu = wrapInstance(
            int(RLPy.RUi.AddMenu("CloneTools", RLPy.EMenu_Plugins)),
            QtWidgets.QMenu)
        menu.setObjectName("clone_tools_menu")

    for act in menu.actions():
        if act.text() == "MAXsync":
            menu.removeAction(act)

    menu.addAction("MAXsync").triggered.connect(open_max_sync)


def initialize_plugin():
    _add_menu_entry()


def run_script():
    _add_menu_entry()
