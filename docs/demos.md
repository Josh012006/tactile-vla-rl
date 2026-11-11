# Demonstrations (SFT): collection, teleoperation, checks, conversion

The FORGE PegInsert demonstrations are **the same as expo-ft's**: `demos` is a link to `/network/scratch/m/monganj/expo-ft/demos`, so the existing
folders are read as they are, with the same layout:

| Folder | Content |
|---|---|
| `demos/isaaclab/ForgePegInsert_Robotiq_scripted` | clean scripted demonstrations |
| `demos/isaaclab/ForgePegInsert_Robotiq_noisy` | scripted demonstrations, 35 % of them with noise injection (clean labels) |
| `demos/isaaclab/ForgePegInsert_Robotiq_teleop` | teleoperated demonstrations |
| `demos/isaaclab/lerobot/expo_ft/forge_peg_insert_robotiq_*` | the same, in LeRobot format (what openpi reads for the SFT) |

An `ep_NNNN_seedS.npz` contains `exterior_image`, `wrist_image`, `joint_position`, `gripper_position`, `ft_force`, `peg_socket_offset`, `actions`
(7 absolute joint targets + gripper command), `prompt`, `seed`, and `tactile` (T, 2, 7, 4) and `gripper_effort` when the server had a robot with
taxels. To check what the existing demonstrations contain:
`python -c "import numpy as np; print(np.load('<file>.npz').files)"`.

The SFT reads these `.npz` files directly (touch included): `docs/algos/sft.md`.

## The server must be set up for these tools

One environment, an episode that ends after 5 consecutive "inserted" steps (the demonstration rule: the script stops recording at the same
step), gripper closed on the peg:

```bash
bash jobs/env_server.sh --set num_envs=1 --set success_hold_steps=5 --set gripper_can_open=false
```

The scripts check this at start-up (`scripts/demos/single_env.py`) and stop with the command to run if the server is set up differently. The jobs
below do it themselves.

## Commands

| What | Command |
|---|---|
| scripted demonstrations | `sbatch jobs/job_demos.sh collect --out demos/isaaclab/ForgePegInsert_Robotiq_scripted --num-episodes 50` |
| noisy demonstrations | `sbatch jobs/job_demos.sh collect --out demos/isaaclab/ForgePegInsert_Robotiq_noisy --num-episodes 50 --noise-fraction 0.35 --action-noise-sigma 0.06` |
| videos of the collection | add `--video-dir reports/demo_videos` (exterior, wrist, taxel panel, banner) |
| replay demonstrations, with videos | `sbatch jobs/job_demos.sh replay --demos-dir <folder> --save-videos --video-first 20 --video-dir reports/replay_videos --report reports/replay_report.json` |
| robot: images per phase and taxel video | `sbatch jobs/job_demos.sh robot_check --out reports/robot_check_tsf` |
| teleoperation | `sbatch jobs/job_teleop_server.sh`, then the two commands it prints (tunnel, `teleop_client.py` on your PC) |
| LeRobot conversion | `python -m scripts.demos.convert_waypoint_demos_to_lerobot --demos-dir <folder> --output-dir demos/isaaclab/lerobot --repo-name expo_ft/<name>` (`.venv`, needs `lerobot`) |
| normalisation statistics | `python -m scripts.demos.compute_demo_norm_stats --demos-dir <folder> --output-dir assets/forge_demo_norm --asset-id <name>` |
| analyses without a GPU | `scripts.demos.noise_effect`, `analyze_waypoint_failure`, `demo_z_bands` |

Without `--video-dir` and `--report`, `replay` writes its videos and its JSON report **inside the demonstration folder** (`<demos-dir>/replay_videos`,
`<demos-dir>/replay_report.json`): pass both when the folder is the one shared with expo-ft.

The videos, the teleoperation and `robot_check` display the tactile information: two 7×4 grids (left finger, right finger, row 0 = fingertip at the
bottom), colour scale 0 to 5 N, a verdict (both fingers, one finger, no contact), the maximum and the sum over the zones, the number of zones in
contact, the wrist force, the gripper effort. The drawing is defined once, in `src/eval/video_panels.py`.

## What changed compared with expo-ft

- The scripts talk to the new server (`VectorEnvClient`), not to the old `EnvClient`. `reset(seed)` is round 0 of that seed; every step is sent with
  `auto_reset=False`, so a finished episode is never restarted behind the script's back (the old code had to fetch the last observation before the
  reset).
- Server ops ported: `ik_joint_target` (FORGE's IK, for every environment), `finger_width`, `finger_distance`, `teleop_step` (IK + step in one round trip,
  for teleoperation through the tunnel) and `save_teleop_demo` (the server writes the file, through the `demos` link, and never writes outside the
  project).
- The server check (`num_envs`, `success_hold_steps`, `gripper_can_open`) comes from the server's spec, no longer from a client-side YAML.

## Not ported

`grasp_check`, `grip_check`, `grip_stats` (one-off calibration of the Robotiq grasp), `repro_check`, `reset_history` (replaced by
`tests/test_isaac_fidelity.py`), `smoke_test_env_server` (replaced by the tests), `analyze_peg_metrics` (reads the outputs of the old evaluation),
`capture_forge_cameras` (builds its own Isaac environment: the cameras are now defined in `src/envs/isaaclab/forge_env.py`) and the USD asset tools
(`robot_assets.py` is already in `src/envs/isaaclab`).

## To validate on the cluster

The ops that read the simulator (`ik_joint_target`, `finger_width`, `finger_distance`) can only be tested with Isaac: run
`sbatch jobs/job_demos.sh collect --out demos/isaaclab/test_collect --num-episodes 2 --video-dir reports/demo_videos` first. The expo-ft
demonstrations were recorded with the old server: a `replay` in the new environment measures whether the same seed gives the same start
(the "reset dev" column and the drift in mm).
