# 3ds Max Pose Package Export

Use `POST /infer_pose_max_package_base64` to generate a Max-friendly package.

## Request fields

- `image_b64` (required)
- `person_index` (default `0`)
- `profile_name` (default `max_biped_basic`)
- `profile` (optional inline profile JSON)
- `package_name` (optional folder name)
- `output_root` (optional output root path)
- `include_source_joints` (default `true`)
- `include_raw` (default `false`)
- `apply_translation` (default `true`)

## Output package files

- `structured_pose.json`: normalized profile-driven pose data
- `apply_pose.ms`: generated MaxScript to apply rotations by node name
- `manifest.json`: file references + mapping stats
- `source_joints.json` (optional)
- `sam3d_raw.json` (optional)

## Max usage

1. Open your scene in 3ds Max.
2. Ensure node names match the profile target names.
3. Run `apply_pose.ms` in MaxScript.
4. Review listener output for missing node names.
