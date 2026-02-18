"""
SAM 3D Body → Unity Humanoid FBX Converter (Image Only)

Features:
- Generate Unity Humanoid-compatible FBX files from a single image
- 3D mesh + skeleton + 1-frame animation
"""

import os
import json
import subprocess
import tempfile
import shutil
import time

import gradio as gr
import numpy as np
import torch

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info


# =============================================================================
# Constants
# =============================================================================

UNITY_BONES = [
    "Hips", "Spine", "Chest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftWrist", "LeftHand",
    "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
    "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
    "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
    "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
    "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightWrist", "RightHand",
    "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
    "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
    "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
    "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
    "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
]

PARENT_MAP = {
    "Hips": None, "Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    "LeftShoulder": "Chest", "LeftUpperArm": "LeftShoulder",
    "LeftLowerArm": "LeftUpperArm", "LeftWrist": "LeftLowerArm", "LeftHand": "LeftWrist",
    "LeftThumbProximal": "LeftHand", "LeftThumbIntermediate": "LeftThumbProximal",
    "LeftThumbDistal": "LeftThumbIntermediate",
    "LeftIndexProximal": "LeftHand", "LeftIndexIntermediate": "LeftIndexProximal",
    "LeftIndexDistal": "LeftIndexIntermediate",
    "LeftMiddleProximal": "LeftHand", "LeftMiddleIntermediate": "LeftMiddleProximal",
    "LeftMiddleDistal": "LeftMiddleIntermediate",
    "LeftRingProximal": "LeftHand", "LeftRingIntermediate": "LeftRingProximal",
    "LeftRingDistal": "LeftRingIntermediate",
    "LeftLittleProximal": "LeftHand", "LeftLittleIntermediate": "LeftLittleProximal",
    "LeftLittleDistal": "LeftLittleIntermediate",
    "RightShoulder": "Chest", "RightUpperArm": "RightShoulder",
    "RightLowerArm": "RightUpperArm", "RightWrist": "RightLowerArm", "RightHand": "RightWrist",
    "RightThumbProximal": "RightHand", "RightThumbIntermediate": "RightThumbProximal",
    "RightThumbDistal": "RightThumbIntermediate",
    "RightIndexProximal": "RightHand", "RightIndexIntermediate": "RightIndexProximal",
    "RightIndexDistal": "RightIndexIntermediate",
    "RightMiddleProximal": "RightHand", "RightMiddleIntermediate": "RightMiddleProximal",
    "RightMiddleDistal": "RightMiddleIntermediate",
    "RightRingProximal": "RightHand", "RightRingIntermediate": "RightRingProximal",
    "RightRingDistal": "RightRingIntermediate",
    "RightLittleProximal": "RightHand", "RightLittleIntermediate": "RightLittleProximal",
    "RightLittleDistal": "RightLittleIntermediate",
    "LeftUpperLeg": "Hips", "LeftLowerLeg": "LeftUpperLeg",
    "LeftFoot": "LeftLowerLeg", "LeftToes": "LeftFoot",
    "RightUpperLeg": "Hips", "RightLowerLeg": "RightUpperLeg",
    "RightFoot": "RightLowerLeg", "RightToes": "RightFoot",
}

MHR_KEYPOINT_INDEX = {info["name"]: idx for idx, info in pose_info["keypoint_info"].items()}


# =============================================================================
# Blender Script (1-frame animation support)
# =============================================================================

BLENDER_SCRIPT = r'''
"""
Blender FBX Export - 1-frame animation support
"""

import bpy
import sys
import json
from mathutils import Vector, Matrix, Quaternion

# -----------------------------------------------------------------------------
# Load arguments
# -----------------------------------------------------------------------------
verts_path = sys.argv[-6]
faces_path = sys.argv[-5]
joints_unity_path = sys.argv[-4]
joints_mhr70_path = sys.argv[-3]
keypoint_index_path = sys.argv[-2]
fbx_path = sys.argv[-1]

with open(verts_path) as f:
    vertices = json.load(f)["vertices"]
with open(faces_path) as f:
    faces = json.load(f)["faces"]
with open(joints_unity_path) as f:
    joints_unity = json.load(f)["joints"]
with open(joints_mhr70_path) as f:
    joints_mhr70 = json.load(f)["joints"]
with open(keypoint_index_path) as f:
    MHR_KP_IDX = json.load(f)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BONE_NAMES = [
    "Hips", "Spine", "Chest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftWrist", "LeftHand",
    "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
    "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
    "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
    "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
    "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightWrist", "RightHand",
    "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
    "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
    "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
    "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
    "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
]

PARENT_MAP = {
    "Hips": None, "Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    "LeftShoulder": "Chest", "LeftUpperArm": "LeftShoulder",
    "LeftLowerArm": "LeftUpperArm", "LeftWrist": "LeftLowerArm", "LeftHand": "LeftWrist",
    "LeftThumbProximal": "LeftHand", "LeftThumbIntermediate": "LeftThumbProximal",
    "LeftThumbDistal": "LeftThumbIntermediate",
    "LeftIndexProximal": "LeftHand", "LeftIndexIntermediate": "LeftIndexProximal",
    "LeftIndexDistal": "LeftIndexIntermediate",
    "LeftMiddleProximal": "LeftHand", "LeftMiddleIntermediate": "LeftMiddleProximal",
    "LeftMiddleDistal": "LeftMiddleIntermediate",
    "LeftRingProximal": "LeftHand", "LeftRingIntermediate": "LeftRingProximal",
    "LeftRingDistal": "LeftRingIntermediate",
    "LeftLittleProximal": "LeftHand", "LeftLittleIntermediate": "LeftLittleProximal",
    "LeftLittleDistal": "LeftLittleIntermediate",
    "RightShoulder": "Chest", "RightUpperArm": "RightShoulder",
    "RightLowerArm": "RightUpperArm", "RightWrist": "RightLowerArm", "RightHand": "RightWrist",
    "RightThumbProximal": "RightHand", "RightThumbIntermediate": "RightThumbProximal",
    "RightThumbDistal": "RightThumbIntermediate",
    "RightIndexProximal": "RightHand", "RightIndexIntermediate": "RightIndexProximal",
    "RightIndexDistal": "RightIndexIntermediate",
    "RightMiddleProximal": "RightHand", "RightMiddleIntermediate": "RightMiddleProximal",
    "RightMiddleDistal": "RightMiddleIntermediate",
    "RightRingProximal": "RightHand", "RightRingIntermediate": "RightRingProximal",
    "RightRingDistal": "RightRingIntermediate",
    "RightLittleProximal": "RightHand", "RightLittleIntermediate": "RightLittleProximal",
    "RightLittleDistal": "RightLittleIntermediate",
    "LeftUpperLeg": "Hips", "LeftLowerLeg": "LeftUpperLeg",
    "LeftFoot": "LeftLowerLeg", "LeftToes": "LeftFoot",
    "RightUpperLeg": "Hips", "RightLowerLeg": "RightUpperLeg",
    "RightFoot": "RightLowerLeg", "RightToes": "RightFoot",
}

INDEX_MAP = {name: i for i, name in enumerate(BONE_NAMES)}
EPS = 1e-6

def get_children(name):
    return [c for c, p in PARENT_MAP.items() if p == name]

def get_hierarchical_order():
    """Return bone names in parent-to-child order"""
    ordered = []
    visited = set()
    def visit(name):
        if name in visited:
            return
        visited.add(name)
        ordered.append(name)
        for child in get_children(name):
            visit(child)
    for name in BONE_NAMES:
        if PARENT_MAP.get(name) is None:
            visit(name)
    return ordered

# -----------------------------------------------------------------------------
# Initialize scene
# -----------------------------------------------------------------------------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

hips_pos = Vector(joints_unity[INDEX_MAP["Hips"]])

# -----------------------------------------------------------------------------
# Create mesh
# -----------------------------------------------------------------------------
centered_verts = [[v[0] - hips_pos.x, v[1], v[2] - hips_pos.z] for v in vertices]
mesh = bpy.data.meshes.new("BodyMesh")
mesh.from_pydata(centered_verts, [], faces)
mesh.update()

mesh_obj = bpy.data.objects.new("BodyMeshObj", mesh)
bpy.context.collection.objects.link(mesh_obj)
bpy.context.view_layer.objects.active = mesh_obj
mesh_obj.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.object.select_all(action='DESELECT')

# -----------------------------------------------------------------------------
# Create armature
# -----------------------------------------------------------------------------
bpy.ops.object.armature_add(location=(0, 0, 0), enter_editmode=False)
arm = bpy.context.active_object
arm.name = "Armature"
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

bpy.ops.object.mode_set(mode="EDIT")
bones = arm.data.edit_bones
bones.remove(bones[0])

for name in BONE_NAMES:
    bones.new(name)

# Joint positions (centered at Hips)
joint_pos = [Vector([j[0] - hips_pos.x, j[1], j[2] - hips_pos.z]) for j in joints_unity]

# Torso chain
bones["Hips"].head = joint_pos[INDEX_MAP["Hips"]]
bones["Hips"].tail = joint_pos[INDEX_MAP["Spine"]]
bones["Spine"].head = joint_pos[INDEX_MAP["Spine"]]
bones["Spine"].tail = joint_pos[INDEX_MAP["Chest"]]
bones["Chest"].head = joint_pos[INDEX_MAP["Chest"]]
bones["Chest"].tail = joint_pos[INDEX_MAP["Neck"]]
bones["Neck"].head = joint_pos[INDEX_MAP["Neck"]]
bones["Neck"].tail = joint_pos[INDEX_MAP["Head"]]

head_pos = joint_pos[INDEX_MAP["Head"]]
neck_pos = joint_pos[INDEX_MAP["Neck"]]
head_dir = (head_pos - neck_pos)
if head_dir.length < EPS:
    head_dir = Vector((0, 0.1, 0))
else:
    head_dir = head_dir.normalized() * 0.1
bones["Head"].head = head_pos
bones["Head"].tail = head_pos + head_dir

# Other bones
for name in BONE_NAMES:
    if name in ["Hips", "Spine", "Chest", "Neck", "Head"]:
        continue
    
    head = joint_pos[INDEX_MAP[name]]
    bones[name].head = head
    children = get_children(name)
    
    if name in ["LeftToes", "RightToes"]:
        parent_name = PARENT_MAP.get(name)
        if parent_name:
            direction = (head - bones[parent_name].head).normalized()
            bones[name].tail = head + direction * 0.01
        continue
    
    if children:
        child_pos = joint_pos[INDEX_MAP[children[0]]]
        v = child_pos - head
        if v.length < EPS:
            v = Vector((0, 0.05, 0))
        bones[name].tail = head + v
    else:
        parent_name = PARENT_MAP.get(name)
        if parent_name:
            v = head - bones[parent_name].head
            if v.length < EPS:
                v = Vector((0, 0.03, 0))
            else:
                v = v.normalized() * 0.03
            bones[name].tail = head + v
        else:
            bones[name].tail = head + Vector((0, 0.03, 0))

# Parent-child relationships
for name, parent in PARENT_MAP.items():
    if parent:
        bones[name].parent = bones[parent]

bpy.ops.object.mode_set(mode="OBJECT")

# -----------------------------------------------------------------------------
# Skinning
# -----------------------------------------------------------------------------
mod = mesh_obj.modifiers.new("ArmatureMod", "ARMATURE")
mod.object = arm

# -----------------------------------------------------------------------------
# Create animation (1 frame)
# -----------------------------------------------------------------------------
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 1

action = bpy.data.actions.new("HumanoidAnimation")
arm.animation_data_create()
arm.animation_data.action = action

bpy.ops.object.mode_set(mode="POSE")

# MHR70 keypoint getter
def KP(name):
    idx = MHR_KP_IDX.get(name)
    if idx is None:
        return None
    raw = joints_mhr70[idx]
    return Vector((raw[0] - hips_pos.x, raw[1], raw[2] - hips_pos.z))

def J(idx):
    j = joints_unity[idx]
    return Vector((j[0] - hips_pos.x, j[1], j[2] - hips_pos.z))

# -----------------------------------------------------------------------------
# Rotation computation helpers
# -----------------------------------------------------------------------------
def build_quaternion_from_axes(primary_dir, secondary_hint):
    """Build quaternion from primary axis and secondary hint"""
    primary = primary_dir.normalized()
    secondary_raw = secondary_hint.normalized()
    
    dot = secondary_raw.dot(primary)
    secondary = secondary_raw - (primary * dot)
    if secondary.length < EPS:
        if abs(primary.x) < 0.9:
            secondary = Vector((1, 0, 0)) - (primary * primary.x)
        else:
            secondary = Vector((0, 1, 0)) - (primary * primary.y)
    secondary = secondary.normalized()
    
    tertiary = primary.cross(secondary).normalized()
    
    mat = Matrix((
        (secondary.x, primary.x, tertiary.x),
        (secondary.y, primary.y, tertiary.y),
        (secondary.z, primary.z, tertiary.z)
    ))
    return mat.to_4x4().to_quaternion()

def compute_secondary_hint(name):
    """Compute secondary hint for bone"""
    side = "left" if name.startswith("Left") else "right"
    
    if name in ["LeftWrist", "RightWrist", "LeftHand", "RightHand"]:
        thumb = KP(f"{side}_thumb_third_joint")
        pinky = KP(f"{side}_pinky_finger_third_joint")
        if thumb and pinky:
            palm = pinky - thumb
            if palm.length > EPS:
                return palm.normalized()
        return Vector((1, 0, 0)) if side == "left" else Vector((-1, 0, 0))
    
    if name in ["LeftFoot", "RightFoot"]:
        toe = KP(f"{side}_big_toe")
        heel = KP(f"{side}_heel")
        if toe and heel:
            forward = toe - heel
            if forward.length > EPS:
                return forward.normalized()
        return Vector((0, 0, 1))
    
    if name in ["LeftLowerArm", "RightLowerArm"]:
        olecranon = KP(f"{side}_olecranon")
        if olecranon:
            parent_pos = J(INDEX_MAP[name])
            elbow_side = olecranon - parent_pos
            if elbow_side.length > EPS:
                return elbow_side.normalized()
        return Vector((1, 0, 0)) if side == "left" else Vector((-1, 0, 0))
    
    if name in ["Hips", "Spine"]:
        ll = J(INDEX_MAP.get("LeftUpperLeg", 0))
        rl = J(INDEX_MAP.get("RightUpperLeg", 0))
        ls = J(INDEX_MAP.get("LeftShoulder", 0))
        rs = J(INDEX_MAP.get("RightShoulder", 0))
        if name == "Hips":
            hint = rl - ll
        else:
            hint = rs - ls
        if hint.length > EPS:
            return hint.normalized()
        return Vector((1, 0, 0))
    
    if name in ["LeftShoulder", "RightShoulder"]:
        acromion = KP(f"{side}_acromion")
        if acromion:
            parent_pos = J(INDEX_MAP[name])
            hint = acromion - parent_pos
            if hint.length > EPS:
                return hint.normalized()
    
    return Vector((1, 0, 0)) if side == "left" else Vector((-1, 0, 0))

def compute_bone_world_rotation(name):
    """Compute bone world rotation"""
    if name == "Head":
        return None
    
    # Special handling for Chest/Neck
    if name in ["Chest", "Neck"]:
        chest_idx = INDEX_MAP["Chest"]
        neck_idx = INDEX_MAP["Neck"]
        head_idx = INDEX_MAP["Head"]
        ls_idx = INDEX_MAP["LeftShoulder"]
        rs_idx = INDEX_MAP["RightShoulder"]
        
        current_chest = J(chest_idx)
        current_neck = J(neck_idx)
        
        ls = J(ls_idx)
        rs = J(rs_idx)
        torso_right = (rs - ls)
        if torso_right.length < EPS:
            torso_right = Vector((1, 0, 0))
        else:
            torso_right = torso_right.normalized()
        
        torso_up = (current_neck - current_chest)
        if torso_up.length < EPS:
            torso_up = Vector((0, 1, 0))
        else:
            torso_up = torso_up.normalized()
        
        # Face direction detection
        left_ear = KP("left_ear")
        right_ear = KP("right_ear")
        nose = KP("nose")
        
        face_forward = None
        if left_ear and right_ear and nose:
            ctr = (left_ear + right_ear) * 0.5
            v = nose - ctr
            if v.length > EPS:
                face_forward = v.normalized()
        
        if face_forward is None and nose:
            v = nose - current_neck
            if v.length > EPS:
                face_forward = v.normalized()
        
        if face_forward is None:
            face_forward = Vector((0, 0, 1))
        
        tmp = face_forward - torso_up * face_forward.dot(torso_up)
        if tmp.length < EPS:
            tmp = face_forward
        chest_forward = tmp.normalized()
        
        if torso_right.cross(chest_forward).dot(torso_up) < 0:
            chest_forward = -chest_forward
        
        chest_matrix = Matrix((
            (torso_right.x, torso_up.x, chest_forward.x),
            (torso_right.y, torso_up.y, chest_forward.y),
            (torso_right.z, torso_up.z, chest_forward.z),
        )).to_4x4()
        
        if name == "Chest":
            return chest_matrix.to_quaternion()
        
        # Neck processing
        current_head = J(head_idx)
        neck_up = (current_head - current_neck)
        if neck_up.length < EPS:
            neck_up = torso_up.copy()
        neck_up = neck_up.normalized()
        
        if left_ear and right_ear:
            head_right = (right_ear - left_ear)
            if head_right.length > EPS:
                head_right = head_right.normalized()
            else:
                head_right = torso_right
        else:
            head_right = torso_right
        
        head_right_ortho = head_right - neck_up * head_right.dot(neck_up)
        if head_right_ortho.length < EPS:
            head_right_ortho = Vector((1, 0, 0))
        head_right_ortho = head_right_ortho.normalized()
        
        head_forward = neck_up.cross(head_right_ortho)
        if head_forward.length < EPS:
            head_forward = Vector((0, 0, 1))
        else:
            head_forward = head_forward.normalized()
        
        head_matrix = Matrix((
            (head_right_ortho.x, neck_up.x, head_forward.x),
            (head_right_ortho.y, neck_up.y, head_forward.y),
            (head_right_ortho.z, neck_up.z, head_forward.z),
        )).to_4x4()
        return head_matrix.to_quaternion()
    
    # Normal bones
    children = get_children(name)
    if not children:
        return None
    
    i = INDEX_MAP[name]
    ci = INDEX_MAP[children[0]]
    current_parent = J(i)
    current_child = J(ci)
    
    primary = current_child - current_parent
    if primary.length < EPS:
        return None
    
    secondary_hint = compute_secondary_hint(name)
    return build_quaternion_from_axes(primary, secondary_hint)

# -----------------------------------------------------------------------------
# Apply animation
# -----------------------------------------------------------------------------
scene.frame_set(1)

# Hips position (always at origin)
pb_hips = arm.pose.bones["Hips"]
pb_hips.location = (0, 0, 0)
pb_hips.keyframe_insert(data_path="location", group="Hips")

# Compute rest pose rotations
rest_rotations = {}
for name in get_hierarchical_order():
    if name == "Head":
        continue
    rot = compute_bone_world_rotation(name)
    if rot:
        rest_rotations[name] = rot

# Apply world delta and local rotation
world_deltas = {}
for name in get_hierarchical_order():
    if name == "Head":
        continue
    
    current_rot = compute_bone_world_rotation(name)
    if current_rot is None or name not in rest_rotations:
        continue
    
    rest_rot = rest_rotations[name]
    world_delta = current_rot @ rest_rot.inverted()
    world_deltas[name] = world_delta
    
    pb = arm.pose.bones[name]
    parent_name = PARENT_MAP.get(name)
    
    if parent_name and parent_name in world_deltas:
        parent_delta = world_deltas[parent_name]
        local_rot = parent_delta.inverted() @ world_delta
    else:
        local_rot = world_delta
    
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = local_rot
    pb.keyframe_insert(data_path="rotation_quaternion", group=name)

bpy.ops.object.mode_set(mode="OBJECT")

# -----------------------------------------------------------------------------
# Save to NLA
# -----------------------------------------------------------------------------
if arm.animation_data and arm.animation_data.action:
    action = arm.animation_data.action
    track = arm.animation_data.nla_tracks.new()
    track.name = "HumanoidAnimation"
    strip = track.strips.new(action.name, start=1, action=action)
    arm.animation_data.action = None
    print("✓ Animation saved to NLA")

# Reset pose
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = 'QUATERNION'
    pb.location = (0, 0, 0)
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.scale = (1, 1, 1)
bpy.ops.object.mode_set(mode="OBJECT")

# -----------------------------------------------------------------------------
# FBX Export
# -----------------------------------------------------------------------------
bpy.ops.object.select_all(action='DESELECT')
mesh_obj.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm

bpy.ops.export_scene.fbx(
    filepath=fbx_path,
    use_selection=False,
    bake_anim=True,
    bake_anim_use_all_bones=True,
    bake_anim_use_nla_strips=True,
    bake_anim_use_all_actions=False,
    bake_anim_force_startend_keying=True,
    bake_anim_step=1.0,
    bake_anim_simplify_factor=0.0,
    apply_scale_options='FBX_SCALE_NONE',
    apply_unit_scale=True,
    axis_forward='-Z',
    axis_up='Y',
    primary_bone_axis='Y',
    secondary_bone_axis='X',
    use_armature_deform_only=True,
    add_leaf_bones=True,
)

print(f"✓ FBX export complete: {fbx_path}")
print("✓ 1-frame animation: HumanoidAnimation")
'''


# =============================================================================
# Joint Conversion
# =============================================================================

def convert_to_blender_coords(vec):
    """SAM3D coordinate system → Blender coordinate system"""
    x, y, z = vec
    return np.array([x, z, -y])


def get_keypoint(joints, name):
    """Get joint position from keypoint name"""
    idx = MHR_KEYPOINT_INDEX.get(name)
    if idx is None:
        return None
    return joints[idx]


def safe_get_keypoint(joints, name):
    """Safely get keypoint"""
    pt = get_keypoint(joints, name)
    if pt is None:
        return None
    if np.any(np.isnan(pt)) or np.any(np.abs(pt) > 100):
        return None
    return pt


def compute_unity_joints(joints_3d):
    """Compute Unity Humanoid bone positions from MHR70 joints"""
    kp = lambda name: get_keypoint(joints_3d, name)
    safe_kp = lambda name: safe_get_keypoint(joints_3d, name)
    
    # Reference points
    left_hip = kp("left_hip")
    right_hip = kp("right_hip")
    hips = (left_hip + right_hip) * 0.5
    neck = kp("neck")
    
    # Body up direction
    body_up = neck - hips
    torso_len = np.linalg.norm(body_up)
    if torso_len < 1e-6:
        body_up = np.array([0, 1, 0], float)
        torso_len = 1.0
    body_up = body_up / torso_len
    
    # Spine
    spine = hips + body_up * (torso_len * 0.30)
    
    # Chest
    left_acr = kp("left_acromion")
    right_acr = kp("right_acromion")
    shoulder_center = (left_acr + right_acr) * 0.5
    chest_hint = (shoulder_center * 2 + neck) / 3
    chest = _interpolate_curve(spine, chest_hint, neck, 2/3)
    
    # Head
    head = _compute_head_position(joints_3d, body_up, neck)
    
    # Shoulders
    left_shoulder = kp("left_shoulder")
    right_shoulder = kp("right_shoulder")
    left_shoulder_root = neck * 0.4 + left_shoulder * 0.6
    right_shoulder_root = neck * 0.4 + right_shoulder * 0.6
    
    # Fingers
    left_fingers = _get_finger_positions(joints_3d, "left")
    right_fingers = _get_finger_positions(joints_3d, "right")
    
    unity = {
        "Hips": hips, "Spine": spine, "Chest": chest, "Neck": neck, "Head": head,
        "LeftShoulder": left_shoulder_root, "LeftUpperArm": left_shoulder,
        "LeftLowerArm": kp("left_elbow"), "LeftWrist": kp("left_wrist"), "LeftHand": kp("left_wrist"),
        "RightShoulder": right_shoulder_root, "RightUpperArm": right_shoulder,
        "RightLowerArm": kp("right_elbow"), "RightWrist": kp("right_wrist"), "RightHand": kp("right_wrist"),
        "LeftThumbProximal": left_fingers[0], "LeftThumbIntermediate": left_fingers[1], "LeftThumbDistal": left_fingers[2],
        "LeftIndexProximal": left_fingers[3], "LeftIndexIntermediate": left_fingers[4], "LeftIndexDistal": left_fingers[5],
        "LeftMiddleProximal": left_fingers[6], "LeftMiddleIntermediate": left_fingers[7], "LeftMiddleDistal": left_fingers[8],
        "LeftRingProximal": left_fingers[9], "LeftRingIntermediate": left_fingers[10], "LeftRingDistal": left_fingers[11],
        "LeftLittleProximal": left_fingers[12], "LeftLittleIntermediate": left_fingers[13], "LeftLittleDistal": left_fingers[14],
        "RightThumbProximal": right_fingers[0], "RightThumbIntermediate": right_fingers[1], "RightThumbDistal": right_fingers[2],
        "RightIndexProximal": right_fingers[3], "RightIndexIntermediate": right_fingers[4], "RightIndexDistal": right_fingers[5],
        "RightMiddleProximal": right_fingers[6], "RightMiddleIntermediate": right_fingers[7], "RightMiddleDistal": right_fingers[8],
        "RightRingProximal": right_fingers[9], "RightRingIntermediate": right_fingers[10], "RightRingDistal": right_fingers[11],
        "RightLittleProximal": right_fingers[12], "RightLittleIntermediate": right_fingers[13], "RightLittleDistal": right_fingers[14],
        "LeftUpperLeg": kp("left_hip"), "LeftLowerLeg": kp("left_knee"),
        "LeftFoot": kp("left_ankle"), "LeftToes": kp("left_big_toe"),
        "RightUpperLeg": kp("right_hip"), "RightLowerLeg": kp("right_knee"),
        "RightFoot": kp("right_ankle"), "RightToes": kp("right_big_toe"),
    }
    
    return [unity[name].tolist() for name in UNITY_BONES]


def _interpolate_curve(p0, p1, p2, t):
    """3-point curve interpolation"""
    seg0 = np.linalg.norm(p1 - p0)
    seg1 = np.linalg.norm(p2 - p1)
    total = max(seg0 + seg1, 1e-6)
    s = t * total
    if s <= seg0:
        return (1 - s/seg0) * p0 + (s/seg0) * p1 if seg0 > 1e-6 else p0.copy()
    else:
        s2 = s - seg0
        return (1 - s2/seg1) * p1 + (s2/seg1) * p2 if seg1 > 1e-6 else p1.copy()


def _compute_head_position(joints_3d, body_up, neck):
    """Compute head position"""
    safe_kp = lambda name: safe_get_keypoint(joints_3d, name)
    kp = lambda name: get_keypoint(joints_3d, name)
    
    ls = kp("left_shoulder")
    rs = kp("right_shoulder")
    torso_right = rs - ls
    if np.linalg.norm(torso_right) > 1e-6:
        torso_right = torso_right / np.linalg.norm(torso_right)
    else:
        torso_right = np.array([1, 0, 0], float)
    
    torso_forward = np.cross(body_up, torso_right)
    if np.linalg.norm(torso_forward) > 1e-6:
        torso_forward = torso_forward / np.linalg.norm(torso_forward)
    else:
        torso_forward = np.array([0, 0, 1], float)
    
    left_ear = safe_kp("left_ear")
    right_ear = safe_kp("right_ear")
    nose = safe_kp("nose")
    
    if left_ear is not None and right_ear is not None:
        return (left_ear + right_ear) * 0.5 + body_up * 0.06
    elif left_ear is not None:
        return left_ear + torso_right * 0.07 + body_up * 0.06
    elif right_ear is not None:
        return right_ear - torso_right * 0.07 + body_up * 0.06
    elif nose is not None:
        return nose - torso_forward * 0.08 + body_up * 0.06
    else:
        return neck + body_up * 0.15


def _get_finger_positions(joints_3d, side):
    """Get finger joint positions"""
    kp = lambda name: get_keypoint(joints_3d, name)
    prefixes = [f"{side}_thumb", f"{side}_forefinger", f"{side}_middle_finger",
                f"{side}_ring_finger", f"{side}_pinky_finger"]
    fingers = []
    for prefix in prefixes:
        fingers.extend([kp(f"{prefix}_third_joint"), kp(f"{prefix}2"), kp(f"{prefix}3")])
    return fingers


# =============================================================================
# FBX Export
# =============================================================================

def export_to_fbx(vertices, joints_unity, joints_mhr70, faces):
    """Generate FBX with Blender"""
    tmp_dir = tempfile.mkdtemp(prefix="sam3d_fbx_")
    
    try:
        verts_path = os.path.join(tmp_dir, "verts.json")
        faces_path = os.path.join(tmp_dir, "faces.json")
        joints_unity_path = os.path.join(tmp_dir, "joints_unity.json")
        joints_mhr70_path = os.path.join(tmp_dir, "joints_mhr70.json")
        keypoint_path = os.path.join(tmp_dir, "keypoint_index.json")
        script_path = os.path.join(tmp_dir, "blender_script.py")
        fbx_path = os.path.join(tmp_dir, "output.fbx")
        
        with open(verts_path, "w") as f:
            json.dump({"vertices": vertices}, f)
        with open(faces_path, "w") as f:
            json.dump({"faces": faces.tolist()}, f)
        with open(joints_unity_path, "w") as f:
            json.dump({"joints": joints_unity}, f)
        with open(joints_mhr70_path, "w") as f:
            json.dump({"joints": joints_mhr70}, f)
        with open(keypoint_path, "w") as f:
            json.dump(MHR_KEYPOINT_INDEX, f)
        with open(script_path, "w") as f:
            f.write(BLENDER_SCRIPT)
        
        subprocess.run([
            "blender", "-b",
            "--python", script_path,
            "--",
            verts_path, faces_path, joints_unity_path, joints_mhr70_path, keypoint_path, fbx_path
        ], check=True, cwd=tmp_dir)
        
        timestamp = int(time.time())
        final_path = f"/tmp/pose_{timestamp}.fbx"
        shutil.copyfile(fbx_path, final_path)
        
        return final_path
    
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except:
            pass


# =============================================================================
# SAM 3D Body Estimation
# =============================================================================

class PoseEstimator:
    """SAM 3D Body pose estimator"""
    
    def __init__(self, checkpoint_path, mhr_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM 3D Body on {self.device}...")
        
        model, cfg = load_sam_3d_body(
            checkpoint_path=checkpoint_path,
            device=self.device,
            mhr_path=mhr_path
        )
        
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        self.faces = self.estimator.faces
    
    def process_image(self, image_path):
        """Estimate 3D body from image"""
        outputs_raw = self.estimator.process_one_image(image_path)
        outputs = self._pick_largest_person(outputs_raw)
        
        vertices = outputs["pred_vertices"]
        joints_mhr70 = outputs["pred_keypoints_3d"]
        
        vertices = np.array([convert_to_blender_coords(v) for v in vertices])
        joints_mhr70 = np.array([convert_to_blender_coords(j) for j in joints_mhr70])
        joints_unity = compute_unity_joints(joints_mhr70)
        
        return {
            "vertices": vertices.tolist(),
            "joints_mhr70": joints_mhr70.tolist(),
            "joints_unity": joints_unity,
        }
    
    def _pick_largest_person(self, outputs):
        """Select the largest person"""
        if isinstance(outputs, dict):
            return outputs
        sizes = []
        for o in outputs:
            verts = o["pred_vertices"]
            sizes.append((np.max(verts[:,0]) - np.min(verts[:,0])) +
                        (np.max(verts[:,1]) - np.min(verts[:,1])))
        return outputs[int(np.argmax(sizes))]


# =============================================================================
# Gradio UI
# =============================================================================

def create_app(estimator):
    """Create Gradio app"""
    
    def process_image(file_obj, progress=gr.Progress()):
        if file_obj is None:
            return None
        
        progress(0, desc="🖼️ Analyzing image...")
        result = estimator.process_image(file_obj.name)
        
        progress(0.5, desc="📦 Generating FBX file...")
        fbx_path = export_to_fbx(
            result["vertices"],
            result["joints_unity"],
            result["joints_mhr70"],
            estimator.faces
        )
        
        progress(1.0, desc="✅ Complete!")
        return fbx_path
    
    with gr.Blocks(title="SAM 3D Body → Unity FBX") as app:
        gr.Markdown("## 🧍‍♂️ SAM 3D Body → Unity Humanoid FBX")
        gr.Markdown("""
        ### Features
        - Generate Unity Humanoid-compatible FBX files from a single image
        - 3D mesh + skeleton + **1-frame animation**
        
        ### How to Use
        1. Upload an image
        2. Click "Generate FBX" button
        3. Download the generated FBX → Import into Unity
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.File(label="📁 Image File", file_types=["image"])
                generate_btn = gr.Button("🚀 Generate FBX", variant="primary")
            with gr.Column():
                output_file = gr.File(label="📦 Generated FBX", interactive=False)
        
        generate_btn.click(fn=process_image, inputs=[input_image], outputs=output_file)
    
    return app


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    CHECKPOINT_PATH = "/mnt/e/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
    MHR_PATH = "/mnt/e/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    
    estimator = PoseEstimator(CHECKPOINT_PATH, MHR_PATH)
    app = create_app(estimator)
    app.launch()