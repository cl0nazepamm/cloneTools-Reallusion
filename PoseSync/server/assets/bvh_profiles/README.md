# BVH Profiles

`mhr_raw_pose` exports a direct MHR hierarchy BVH from SAM3DBody joints.

## Keys

- `root_joint_name`: root joint name from MHR skeleton (default `root`)
- `drop_world_root`: drop `body_world` and start from `root`
- `zero_root_offset`: force root OFFSET to `0 0 0`
- `rotation_mode`: `local_pose` or `local_delta_from_rest`
- `rotation_order`: Euler channel order (default `ZXY`)
- `source_quat_key`: source quaternion key in joint payload (default `quat_xyzw_global`)
- `source_coord_key`: source coordinate key in joint payload (default `coord`)
- `root_translation_mode`: `none`, `pose`, or `pose_minus_rest`
- `root_translation_scale`: coordinate scale (SAM3D joints are meters; use `100.0` for cm)
- `root_translation_axis_multipliers`: axis conversion for root translation
- `include_joint_names`: optional subset list (ancestors auto-included)
- `exclude_joint_names`: optional exclusion list
- `fps`: output frame rate
