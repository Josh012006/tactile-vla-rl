"""Builds the FORGE environment from a task YAML and serves it as a vector of environments driven in rounds.

Imports Isaac Lab: import this module only AFTER AppLauncher exists (server.py does it).

Layers (each testable on its own):
  forge_env.py       the Isaac Lab environment (cameras, joint-position actions, Robotiq, taxels, the real reset)
  ForgeSim (here)    the SimAdapter of episode_manager.py for that environment
  episode_manager    rounds, per-environment episodes, held environments, final observations (numpy only)
  backend.py         protocol <-> manager;  ws_server.py: the WebSocket

Isaac Lab's own episode logic is switched off (episode_length_s = NATIVE_TIMEOUT_DISABLED_S): episodes are ended by the
manager, per environment, so nothing is ever reset behind the client's back and the success flag of the step that times out
is a real one. The teleoperation / demo-saving / grasp-geometry debug ops of the old server are not ported.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
import torch

import isaaclab_tasks  # noqa: F401  (registers Isaac-Forge-*-Direct-v0)
from isaaclab_tasks.utils import parse_env_cfg

from . import protocol as P
from .backend import ManagerBackend
from .episode_manager import VectorEnvManager
from .forge_env import CAMERA_SIZE, GRIPPER_OPEN_WIDTH, ForgeEnvJointPosPi05
from .robot_assets import ensure_robot_usd
from .robots import TACTILE_COLS, TACTILE_ROWS, apply_to_env_cfg, profile_from_yaml, robot_info

log = logging.getLogger("forge_backend")

# Isaac Lab ends an episode when episode_length_buf >= max_episode_length - 1 (_get_dones of Factory) and resets inside
# step(). A value this large (seconds) makes that never happen: max_episode_length ~ 1.5e7 control steps.
NATIVE_TIMEOUT_DISABLED_S = 1.0e6

GRIPPER_OBS_MODES = ("raw", "normalized", "closed")


def gripper_obs_value(width: np.ndarray, mode: str) -> np.ndarray:
    """Gripper observation from the finger opening `width` (m per finger, FORGE's convention: 0 closed, 0.04 open).

    pi05_droid_jointpos expects the DROID convention: 0 = open, 1 = closed.
      normalized : 1 - width / 0.04, in [0, 1] (holding the peg it reads ~0.9)       <- recommended
      closed     : constant 1.0 (only sensible together with gripper_can_open: false)
      raw        : the width in metres, NOT the model's convention (legacy)
    """
    if mode == "raw":
        return width
    if mode == "normalized":
        return np.clip(1.0 - width / GRIPPER_OPEN_WIDTH, 0.0, 1.0)
    return np.ones_like(width)


def build_env(cfg: dict, device: str = "cuda:0"):
    """The FORGE environment configured by the task YAML `cfg` (see configs/task/peg_insert_forge.yaml)."""
    task_name = cfg["task_name"]
    num_envs = int(cfg.get("num_envs", 1))
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    # FORGE's stock scene is built with clone_in_fabric=True: env_1..env_{N-1} then exist only in Fabric, NOT in the USD stage.
    # Everything that finds its prims through the stage (every Isaac Lab sensor: cameras, contact sensors) then sees ONE
    # environment (buffers of size 1, a device-side assert at the first reset). Clones must be real USD prims.
    env_cfg.scene.clone_in_fabric = bool(cfg.get("clone_in_fabric", False))

    # Robot: YAML field `robot`. The Robotiq profiles spawn a USD derived from NVIDIA's (built once, see robot_assets.py).
    profile = profile_from_yaml(cfg)
    usd_path = profile.robot_usd
    if profile.build_usd:
        usd_path = ensure_robot_usd(profile, exclude_zone_joints=bool(cfg.get("tactile_exclude_zone_joints", False)),
                                    log=lambda m: print(f"[env_server] {m}", flush=True))
    apply_to_env_cfg(env_cfg, profile, usd_path, cfg)

    # reset_grip_open_width (m per finger; None = FORGE's own 1.25 x peg radius). See the task YAML.
    open_width = cfg.get("reset_grip_open_width")
    if open_width is not None:
        env_cfg.task.held_asset_cfg.diameter = 2.0 * float(open_width) / 1.25

    env_cfg.action_space = P.ACTION_DIM            # 7 absolute joint targets + gripper, not FORGE's 7D task-space action

    # FORGE zeroes the arm gains (its own task-space controller applies torques); the joint-position action needs a PD.
    stiffness, damping = float(cfg.get("arm_stiffness", 80.0)), float(cfg.get("arm_damping", 40.0))
    for group in ("panda_arm1", "panda_arm2"):
        env_cfg.robot.actuators[group].stiffness = stiffness
        env_cfg.robot.actuators[group].damping = damping

    if cfg.get("disable_sleep", False):
        for name in ("articulation_props", "rigid_props"):
            props = getattr(env_cfg.robot.spawn, name, None)
            if props is not None and hasattr(props, "sleep_threshold"):
                props.sleep_threshold = 0.0

    if cfg.get("enhanced_determinism", False):
        if not hasattr(env_cfg.sim.physx, "enable_enhanced_determinism"):
            raise ValueError("enhanced_determinism: this Isaac Lab's PhysxCfg has no enable_enhanced_determinism field")
        env_cfg.sim.physx.enable_enhanced_determinism = True

    env_cfg.sim.render_interval = env_cfg.decimation      # one render per env step

    # Episodes are ended by the server, never by Isaac Lab (see NATIVE_TIMEOUT_DISABLED_S).
    env_cfg.episode_length_s = NATIVE_TIMEOUT_DISABLED_S

    env = ForgeEnvJointPosPi05(
        cfg=env_cfg, render_mode=None, gripper_can_open=cfg.get("gripper_can_open", True), robot_profile=profile,
        camera_size=int(cfg.get("camera_size", CAMERA_SIZE)))
    print(f"[env_server] robot={profile.name}  num_envs={env.num_envs}  camera={env._camera_size}px  "
          f"arm PD={stiffness:g}/{damping:g}  step_dt={env.step_dt:.4f}s ({1.0 / env.step_dt:.1f} Hz)", flush=True)
    return env, profile


class ForgeSim:
    """episode_manager.SimAdapter for a ForgeEnvJointPosPi05."""

    def __init__(self, env, profile, gripper_obs: str, reset_passes: int):
        if gripper_obs not in GRIPPER_OBS_MODES:
            raise ValueError(f"gripper_obs must be one of {GRIPPER_OBS_MODES}, got {gripper_obs!r}")
        if int(reset_passes) < 1:
            raise ValueError(f"reset_passes must be >= 1, got {reset_passes}")
        self.env, self.profile, self.gripper_obs, self.reset_passes = env, profile, gripper_obs, int(reset_passes)
        self.num_envs = env.num_envs

    # -- SimAdapter ------------------------------------------------------------------------------------------------------
    def reset(self, seed: int):
        self.env.reset_lockstep(seed=int(seed), passes=self.reset_passes)
        torch.cuda.synchronize()

    def step_physics(self, actions: np.ndarray):
        env = self.env
        action = torch.from_numpy(np.array(actions, dtype=np.float32)).to(env.device)
        env.step(action)
        env.assert_no_native_reset()
        torch.cuda.synchronize()

    def inserted(self) -> np.ndarray:
        return self.env.inserted().cpu().numpy()

    def hold_actions(self) -> np.ndarray:
        return self.env.hold_action().cpu().numpy().astype(np.float32)

    def observe(self) -> Dict[str, np.ndarray]:
        env = self.env
        cpu = lambda x: x.detach().cpu().numpy()
        width = cpu(env.gripper_width())[:, None]
        obs = {
            P.EXTERIOR_IMAGE: cpu(env._exterior_cam.data.output["rgb"]).astype(np.uint8),
            P.WRIST_IMAGE: cpu(env._wrist_cam.data.output["rgb"]).astype(np.uint8),
            P.JOINT_POSITION: cpu(env.joint_pos[:, 0:7]).astype(np.float32),
            P.GRIPPER_POSITION: gripper_obs_value(width, self.gripper_obs).astype(np.float32),
            P.FT_FORCE: cpu(env.force_sensor_smooth[:, 0:3]).astype(np.float32),
            P.FORCE_THRESHOLD: cpu(env.contact_penalty_thresholds).astype(np.float32),
            P.PEG_SOCKET_OFFSET: cpu(env.peg_socket_offset()).astype(np.float32),
        }
        if env._is_robotiq:           # PD torque demanded of the finger joint (N.m, last sub-step)
            obs[P.GRIPPER_EFFORT] = cpu(env.gripper_effort()).astype(np.float32)
        if env._tactile_sensors:      # (N, 2, 7, 4) N, [:, 0] left finger, [:, 1] right, row 0 = fingertip
            obs[P.TACTILE] = cpu(env.tactile_grids()).astype(np.float32)
        return obs

    # -- diagnostics (op `debug`) -----------------------------------------------------------------------------------------
    def debug(self, request: dict) -> Any:
        kind = request.get("kind")
        env = self.env
        np_ = lambda d: {k: v.detach().cpu().numpy() for k, v in d.items()}
        if kind == "hold_action":                 # (N, 8): keeps every robot where it is (clients use it to idle)
            return self.hold_actions()
        if kind == "joint_names":
            return list(env._robot.data.joint_names)
        if kind == "robot_info":
            return {**robot_info(self.profile), "gripper_can_open": bool(env._gripper_can_open),
                    "joint_names": list(env._robot.data.joint_names), "gripper_adapter": env.gripper_info()}
        if kind == "task_geometry":
            task = env.cfg_task
            hole_depth = float(task.fixed_asset_cfg.height)
            return {"hole_depth": hole_depth, "success_xy": 0.0025, "success_z": hole_depth * float(task.success_threshold)}
        if kind == "fingertip_pose":              # FORGE's own computation from the simulated rigid-body pose
            return {"pos": env.fingertip_midpoint_pos.cpu().numpy(), "quat": env.fingertip_midpoint_quat.cpu().numpy()}
        if kind == "vram":
            return self._vram()
        if kind == "peg_pose":                    # held peg, env frame: position (m) and direction of its axis (unit vector)
            from isaaclab.utils.math import quat_apply
            axis = torch.zeros(env.num_envs, 3, device=env.device)
            axis[:, 2] = 1.0
            held = env._held_asset.data
            return {"pos": np_({"p": held.root_pos_w - env.scene.env_origins})["p"],
                    "axis": np_({"a": quat_apply(held.root_quat_w, axis)})["a"]}
        return {"error": f"unknown debug kind: {kind}"}

    def _vram(self) -> dict:
        """GPU memory of THIS process, from NVML (Kit / PhysX / RTX allocate outside torch's allocator, so torch's own
        counters only see a part). `device_used_mib` is the whole GPU (other jobs included)."""
        mib = 1.0 / 2 ** 20
        out = {"num_envs": self.num_envs, "pid": os.getpid(),
               "torch_allocated_mib": torch.cuda.memory_allocated() * mib, "torch_reserved_mib": torch.cuda.memory_reserved() * mib,
               "torch_max_reserved_mib": torch.cuda.max_memory_reserved() * mib}
        try:
            import pynvml
            pynvml.nvmlInit()
            try:       # NVML ignores CUDA_VISIBLE_DEVICES: find the handle of the GPU torch uses by its UUID
                handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(torch.cuda.get_device_properties(self.env.device).uuid))
            except Exception:  # noqa: BLE001
                handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            out.update(device_used_mib=mem.used * mib, device_total_mib=mem.total * mib, process_mib=None)
            for list_procs in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
                for proc in list_procs(handle):
                    if proc.pid == os.getpid() and proc.usedGpuMemory is not None:
                        out["process_mib"] = max(out["process_mib"] or 0.0, proc.usedGpuMemory * mib)
        except Exception as e:  # noqa: BLE001
            out["nvml_error"] = f"{type(e).__name__}: {e} (pip install nvidia-ml-py)"
        return out


def build_backend(cfg: dict, device: str = "cuda:0"):
    """(ManagerBackend, env): the environment, the manager and the protocol layer."""
    env, profile = build_env(cfg, device)
    sim = ForgeSim(env, profile, cfg.get("gripper_obs", "normalized"), int(cfg.get("reset_passes", 1)))
    manager = VectorEnvManager(
        sim, max_steps=int(cfg["max_steps_per_episode"]), success_hold_steps=int(cfg.get("success_hold_steps", 1)),
        auto_reset=bool(cfg.get("auto_reset", True)), seed=int(cfg.get("env_seed", 0)))
    obs_spec = P.build_obs_spec(env._camera_size, gripper_effort=env._is_robotiq, tactile=bool(env._tactile_sensors),
                                tactile_shape=(2, TACTILE_ROWS, TACTILE_COLS))
    robot = {**robot_info(profile), "gripper_can_open": bool(env._gripper_can_open),
             "joint_names": list(env._robot.data.joint_names)}
    backend = ManagerBackend(manager, obs_spec, prompt=cfg.get("language_instruction", ""),
                             control_hz=float(cfg.get("control_hz", 1.0 / env.step_dt)), robot=robot, debug_fn=sim.debug)
    return backend, env
