# Pose Profile JSON

These files define how source SAM-3D joints map into target bones.

## Core idea

- `source_passthrough.json`: no mapping assumptions, emits every source joint.
- `ue_core_clean.json`: clean UE/CC core-chain mapping.
- `max_biped_basic.json`: starter mapping for 3ds Max Biped naming.

## Endpoint

Use `POST /infer_pose_structured_base64` with:

- `image_b64` (required)
- `profile_name` (optional, default: `source_passthrough`)
- `profile` (optional inline profile object)
- `include_source_joints` (optional bool)
- `include_raw` (optional bool)

## Profile fields

- `schema_version`
- `profile_name`
- `profile_version`
- `description`
- `rotation_space`: `local_parent` or `global`
- `output_coordinate_system`: metadata only (`up`, `forward`, `handedness`, `units`)
- `global_transform`:
  - `translation_scale`
  - `translation_offset` `[x,y,z]`
  - `rotation_offset_deg` `[x,y,z]`
  - `rotation_offset_quat_xyzw` `[x,y,z,w]` (optional)
  - `rotation_offset_mode`: `pre` or `post`
- `source_rotation_offsets_quat_xyzw`: optional per-source-joint quaternion offsets
- `source_rotation_offset_mode`: `pre` or `post`
- `passthrough_source_joints`: if true, all source joints are emitted
- `bones`: list of entries
  - `target`
  - `parent` (or `null`)
  - `sources` (priority list)
  - `copy_translation` (bool)
  - `rotation_offset_deg` `[x,y,z]`
  - `rotation_offset_quat_xyzw` `[x,y,z,w]` (optional)
  - `rotation_offset_mode` `pre|post`
  - `translation_offset` `[x,y,z]`
  - `enabled` (bool)
