# -*- coding: utf-8 -*-
# This script exports animations from iClone the lightest way possible.
# Comes with useful options like removing the root and t-pose first frame.




"""
CloneFBX – one-click avatar FBX exporter with option dialog
Adds a menu entry: Plugins ▶ CloneTools ▶ CloneFBX
"""

# ───────── Reallusion plugin handshake (kills the compatibility nag) ─────────
rl_plugin_info = {"ap": "iClone", "ap_version": "8.0"}  # iClone 8.x

# ───────── Imports ─────────
import os, tempfile, sys, subprocess, shlex
import RLPy
from PySide2 import QtWidgets, QtCore
from shiboken2 import wrapInstance


def _sanitize_affix(text):
    s = str(text or "").strip()
    for ch in ("|", ";", ",", "\r", "\n"):
        s = s.replace(ch, "_")
    return s


# ───────── FBX OPTION DIALOG ─────────
class ExportOptionsDialog(QtWidgets.QDialog):
    """Preset selector + manual overrides → RLPy FBX option bitmasks"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FBX Export Options")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        vbox = QtWidgets.QVBoxLayout(self)

        # Preset selector
        preset_label = QtWidgets.QLabel("Preset:")
        vbox.addWidget(preset_label)
        self.combo_preset = QtWidgets.QComboBox()
        self.combo_preset.addItems(["Unreal Engine", "Custom"])
        vbox.addWidget(self.combo_preset)

        # Separator
        line0 = QtWidgets.QFrame()
        line0.setFrameShape(QtWidgets.QFrame.HLine)
        line0.setFrameShadow(QtWidgets.QFrame.Sunken)
        vbox.addWidget(line0)

        # Export options checkboxes
        self.chk_root_motion   = QtWidgets.QCheckBox("Export Root Motion")
        self.chk_zero_root     = QtWidgets.QCheckBox("Zero Motion Root")
        self.chk_remove_mesh   = QtWidgets.QCheckBox("Remove All Mesh")
        self.chk_tpose_first   = QtWidgets.QCheckBox("T-Pose on Motion First Frame")
        self.chk_remove_root   = QtWidgets.QCheckBox("Remove Bone Root")
        self.chk_lightwave_yup = QtWidgets.QCheckBox("LightWave Y Up")
        self.chk_root_motion.setChecked(True)

        self._custom_widgets = []
        for w in (self.chk_root_motion, self.chk_zero_root,
                  self.chk_remove_mesh, self.chk_tpose_first,
                  self.chk_remove_root, self.chk_lightwave_yup):
            vbox.addWidget(w)
            self._custom_widgets.append(w)

        # Separator
        line1 = QtWidgets.QFrame()
        line1.setFrameShape(QtWidgets.QFrame.HLine)
        line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        vbox.addWidget(line1)

        # Up Axis selection (opt2)
        self.axis_label = QtWidgets.QLabel("Up Axis (opt2):")
        vbox.addWidget(self.axis_label)

        self.radio_none = QtWidgets.QRadioButton("None (Z Up default)")
        self.radio_xup  = QtWidgets.QRadioButton("X Up")
        self.radio_yup  = QtWidgets.QRadioButton("Y Up")
        self.radio_ue4  = QtWidgets.QRadioButton("Unreal Engine 4 Bone Axis")
        self.radio_none.setChecked(True)

        axis_group = QtWidgets.QButtonGroup(self)
        for r in (self.radio_none, self.radio_xup, self.radio_yup, self.radio_ue4):
            axis_group.addButton(r)
            vbox.addWidget(r)
            self._custom_widgets.append(r)
        self._custom_widgets.append(self.axis_label)

        self.lbl_prefix = QtWidgets.QLabel("Name Prefix (optional):")
        self.txt_prefix = QtWidgets.QLineEdit()
        self.txt_prefix.setPlaceholderText("e.g. CharA_")
        self.lbl_suffix = QtWidgets.QLabel("Name Suffix (optional):")
        self.txt_suffix = QtWidgets.QLineEdit()
        self.txt_suffix.setPlaceholderText("e.g. _A")
        vbox.addWidget(self.lbl_prefix)
        vbox.addWidget(self.txt_prefix)
        vbox.addWidget(self.lbl_suffix)
        vbox.addWidget(self.txt_suffix)
        self._custom_widgets.append(self.lbl_prefix)
        self._custom_widgets.append(self.txt_prefix)
        self._custom_widgets.append(self.lbl_suffix)
        self._custom_widgets.append(self.txt_suffix)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        vbox.addWidget(btns)

        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        self._on_preset_changed(0)  # apply default preset

    # ── preset handling ──
    def _is_unreal(self):
        return self.combo_preset.currentText() == "Unreal Engine"

    def _on_preset_changed(self, _idx):
        unreal = self._is_unreal()
        for w in self._custom_widgets:
            w.setEnabled(not unreal)
        # Remove All Mesh stays usable with Unreal preset
        self.chk_remove_mesh.setEnabled(True)

    # ── bitmask builders ──
    def options_bitmask(self):
        if self._is_unreal():
            flags = (RLPy.EExportFbxOptions_AutoSkinRigidMesh
                     | RLPy.EExportFbxOptions_ExportRootMotion
                     | RLPy.EExportFbxOptions_ZeroMotionRoot
                     | RLPy.EExportFbxOptions_ExportPbrTextureAsImageInFormatDirectory
                     | RLPy.EExportFbxOptions_InverseNormalY)
            if self.chk_remove_mesh.isChecked():
                flags |= RLPy.EExportFbxOptions_RemoveAllMesh
            return flags
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
        if self.chk_lightwave_yup.isChecked():
            flags |= RLPy.EExportFbxOptions_LightWaveYUp
        return flags

    def get_up_axis_opt2(self):
        prefix = _sanitize_affix(self.txt_prefix.text())
        suffix = _sanitize_affix(self.txt_suffix.text())
        if self._is_unreal():
            flags = (RLPy.EExportFbxOptions2_UnrealEngine4BoneAxis
                     | RLPy.EExportFbxOptions2_RenameDuplicateBoneName
                     | RLPy.EExportFbxOptions2_RenameDuplicateMaterialName
                     | RLPy.EExportFbxOptions2_RenameTransparencyWithPostFix
                     | RLPy.EExportFbxOptions2_RenameBoneRootToGameType
                     | RLPy.EExportFbxOptions2_RenameBoneToLowerCase
                     | RLPy.EExportFbxOptions2_ResetBoneScale
                     | RLPy.EExportFbxOptions2_ResetSelfillumination
                     | RLPy.EExportFbxOptions2_ExtraWordForUnityAndUnreal
                     | RLPy.EExportFbxOptions2_BakeMouthOpenMotionToMesh
                     | RLPy.EExportFbxOptions2_UnrealIkBone
                     | RLPy.EExportFbxOptions2_UnrealPreset)
            if (prefix or suffix) and hasattr(RLPy, "EExportFbxOptions2_PrefixAndPostfix"):
                flags |= RLPy.EExportFbxOptions2_PrefixAndPostfix
            return flags
        # Custom mode – always include ResetBoneScale (prevents 10x scale issue)
        flags = RLPy.EExportFbxOptions2_ResetBoneScale
        if (prefix or suffix) and hasattr(RLPy, "EExportFbxOptions2_PrefixAndPostfix"):
            flags |= RLPy.EExportFbxOptions2_PrefixAndPostfix
        if self.radio_xup.isChecked():
            flags |= RLPy.EExportFbxOptions2_XUp
        elif self.radio_yup.isChecked():
            flags |= RLPy.EExportFbxOptions2_YUp
        elif self.radio_ue4.isChecked():
            flags |= RLPy.EExportFbxOptions2_UnrealEngine4BoneAxis
        return flags


# ───────── MAIN EXPORT ROUTINE ─────────
def export_clone_fbx():
    """Run the dialog, bake motion, export FBX"""
    # ask where to save
    fbx_path = RLPy.RUi.SaveFileDialog("FBX Files (*.fbx)")
    if not fbx_path:
        RLPy.RMessageBox.Warning("CloneFBX", "Export cancelled – no file selected.")
        return
    if not fbx_path.lower().endswith(".fbx"):
        fbx_path += ".fbx"
    fbx_path = os.path.normpath(fbx_path)

    # checkbox dialog
    dlg = ExportOptionsDialog()
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        RLPy.RMessageBox.Warning("CloneFBX", "Export cancelled – no options selected.")
        return
    opt1 = dlg.options_bitmask()
    opt2 = dlg.get_up_axis_opt2()
    opt3 = RLPy.EExportFbxOptions3__None

    # selected avatar
    avatars = [o for o in RLPy.RScene.GetSelectedObjects()
               if o.GetType() == RLPy.EObjectType_Avatar]
    if not avatars:
        RLPy.RMessageBox.Warning("CloneFBX",
            "No avatar selected.\nPlease highlight ONE avatar and try again.")
        return
    if len(avatars) > 1:
        RLPy.RMessageBox.Warning("CloneFBX",
            "Multiple avatars selected.\nSelect only one avatar for export.")
        return
    avatar = avatars[0]

    # work-area range
    start_tick = RLPy.RGlobal.GetStartTime()
    end_tick   = RLPy.RGlobal.GetEndTime()

    # bake to temp .rlmotion
    fd, motion_path = tempfile.mkstemp(suffix=".rlmotion"); os.close(fd)
    save_opt = RLPy.RSaveFileSetting()
    save_opt.SetSaveType(RLPy.ESaveFileType_Motion)
    save_opt.SetSaveRange(start_tick, end_tick)
    RLPy.RFileIO.SaveFile(avatar, save_opt, motion_path)

    # export FBX
    status = RLPy.RFileIO.ExportFbxFile(
        avatar, fbx_path,
        opt1, opt2, opt3,
        RLPy.EExportTextureSize_Original,
        RLPy.EExportTextureFormat_Default,
        motion_path)

    # cleanup
    try: os.remove(motion_path)
    except OSError: pass

    if status != RLPy.RStatus.Success:
        RLPy.RMessageBox.Error("CloneFBX", f"Export failed – status: {status}")


# ───────── MENU HOOKUP ─────────
def _add_menu_entry():
    """Create Plugins ▶ CloneTools ▶ CloneFBX and connect it."""
    main_qt = wrapInstance(int(RLPy.RUi.GetMainWindow()), QtWidgets.QMainWindow)

    # top-level "CloneTools" submenu under Plugins
    menu = main_qt.menuBar().findChild(QtWidgets.QMenu, "clone_tools_menu")
    if menu is None:
        menu = wrapInstance(
            int(RLPy.RUi.AddMenu("CloneTools", RLPy.EMenu_Plugins)),
            QtWidgets.QMenu)
        menu.setObjectName("clone_tools_menu")

    # avoid duplicate
    for act in menu.actions():
        if act.text() == "CloneFBX":
            menu.removeAction(act)

    menu.addAction("CloneFBX").triggered.connect(export_clone_fbx)


# iClone calls this when loading a plugin folder
def initialize_plugin():
    _add_menu_entry()


# if you simply "Run" the .py once via Script ▶ Load Python
def run_script():
    _add_menu_entry()
