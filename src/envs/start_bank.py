"""Bank of start states: how "reset environment i" works without disturbing the other environments.

FORGE's reset (forge_env.reset_lockstep) acts on ALL environments and steps the physics (see forge_env.py, section 0): it
cannot run while other environments are collecting. So it runs only at start-up, a few times, and every start state it produces
is captured COMPLETELY. Resetting environment i later = writing one captured state into environment i: no physics step, the
other environments are not touched.

A captured start state (everything FORGE's reset writes, taken after the reset and the one hold step that renders the cameras,
exactly the state the old protocol handed to the policy):
  sim/*     robot joint positions and velocities (every DOF, Robotiq mimic joints included); pose and velocity of the held
            (peg) and fixed (socket) assets, in the ENVIRONMENT frame (world minus env origin), so that a state captured in
            slot 3 can be written in slot 0.
  env/*     the FORGE / controller tensors the reset sets and the step reads: control targets, gains, thresholds, noise
            offsets, previous-step buffers, success bookkeeping... (STATE_NAMES), plus the intermediate values recomputed at
            every step (DERIVED_NAMES), stored so that the window between a restore and the next step is consistent.
  obs       the full observation (images, tactile, force...) of that state, sent as the first observation of the episode.

What is lost by a restore: PhysX's contact warm-start (the solver restarts from zero impulses), hence a small transient in the
first step. tests/test_isaac_bank.py measures it against the repeatability floor of the simulator itself.

The bank is cached on disk, keyed by a digest of the task YAML (any field that changes the physics, the robot, the cameras or
the build seed changes it, and so does the number of environments of the build: FORGE draws the randomisation of all
environments in one batch, so the content depends on it).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

log = logging.getLogger("start_bank")

BANK_FORMAT = 1

# Written by FORGE's reset (and read by its step): MUST be captured. Absent or not per-environment => error at build time.
STATE_NAMES = (
    "ctrl_target_joint_pos", "fixed_pos_obs_frame", "init_fixed_pos_obs_noise", "prev_fingertip_pos", "prev_fingertip_quat",
    "prev_joint_pos", "ep_succeeded", "ep_success_times", "flip_quats", "force_sensor_smooth", "force_sensor_world_smooth",
    "pos_threshold", "rot_threshold", "task_prop_gains", "task_deriv_gains", "ema_factor", "contact_penalty_thresholds",
    "dead_zone_thresholds", "actions", "prev_actions", "episode_length_buf",
)
# Recomputed at every step from the simulation: captured when present, so a read between restore and step is coherent.
DERIVED_NAMES = (
    "fixed_pos", "fixed_quat", "held_pos", "held_quat", "fingertip_midpoint_pos", "fingertip_midpoint_quat",
    "fingertip_midpoint_linvel", "fingertip_midpoint_angvel", "fingertip_midpoint_jacobian", "left_finger_jacobian",
    "right_finger_jacobian", "joint_pos", "joint_vel", "joint_vel_fd", "ee_linvel_fd", "ee_angvel_fd", "arm_mass_matrix",
    "force_sensor_world", "noisy_fingertip_pos", "noisy_fingertip_quat", "noisy_force", "joint_torque", "applied_wrench",
)
DICT_NAMES = ("first_pred_success_tx",)        # {threshold: (N,) tensor}
# Per-environment-shaped attributes that are NOT start state (Isaac Lab bookkeeping, constants, outputs): never captured.
IGNORED_NAMES = frozenset({
    "reset_buf", "reset_terminated", "reset_time_outs", "reward_buf", "obs_buf", "extras", "default_gains",
    "default_pos_threshold", "default_rot_threshold", "default_dead_zone", "delta_pos", "delta_yaw", "_target_joint_pos",
    "success_pred_scale",
})
# YAML fields that do not change a start state (everything else is part of the cache key).
NOT_IN_DIGEST = frozenset({"name", "server_host", "server_port", "max_steps_per_episode", "success_hold_steps", "control_hz",
                           "language_instruction", "auto_reset", "bank_cache_dir", "env_seed"})


def bank_digest(cfg: dict) -> str:
    relevant = {k: v for k, v in sorted(cfg.items()) if k not in NOT_IN_DIGEST}
    blob = json.dumps({"format": BANK_FORMAT, "cfg": relevant, "state": STATE_NAMES, "derived": DERIVED_NAMES,
                       "dict": DICT_NAMES}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _is_per_env(value, n: int) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == n


def scan_unlisted(env) -> List[str]:
    """Attributes of `env` that look like per-environment tensors (or dicts of them) and are in none of the lists above. The
    completeness test fails on a non-empty answer: each name is either start state (add it to STATE_NAMES), recomputed
    (DERIVED_NAMES) or irrelevant (IGNORED_NAMES). Constants that merely have N rows are caught by the restore test."""
    n, known = env.num_envs, set(STATE_NAMES) | set(DERIVED_NAMES) | set(DICT_NAMES) | IGNORED_NAMES
    out = []
    for name, value in vars(env).items():
        if name in known:
            continue
        if _is_per_env(value, n) or (isinstance(value, dict) and value and all(_is_per_env(v, n) for v in value.values())):
            out.append(name)
    return sorted(out)


@dataclass
class BankState:
    state: Dict[str, torch.Tensor]          # name -> (K, ...) on the simulation device
    obs: Dict[str, np.ndarray]              # key -> (K, ...) host arrays
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.meta["size"])

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"state": {k: v.cpu() for k, v in self.state.items()}, "obs": self.obs, "meta": self.meta}, tmp)
        tmp.replace(path)

    @staticmethod
    def load(path: Path, device: str) -> "BankState":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return BankState({k: v.to(device) for k, v in blob["state"].items()}, blob["obs"], blob["meta"])


# ---------------------------------------------------------------------------------------------------------------- capture
def capture(env) -> Dict[str, torch.Tensor]:
    """The complete state of every environment of `env`, as tensors (N, ...), env-frame poses. Clones: safe to keep."""
    n, out, skipped = env.num_envs, {}, []
    robot = env._robot
    out["sim/joint_pos"] = robot.data.joint_pos.clone()
    out["sim/joint_vel"] = robot.data.joint_vel.clone()
    origins = env.scene.env_origins
    for tag, asset in (("held", env._held_asset), ("fixed", env._fixed_asset)):
        out[f"sim/{tag}_pose"] = torch.cat([asset.data.root_pos_w - origins, asset.data.root_quat_w], dim=-1).clone()
        out[f"sim/{tag}_vel"] = torch.cat([asset.data.root_lin_vel_w, asset.data.root_ang_vel_w], dim=-1).clone()
    for names, required in ((STATE_NAMES, True), (DERIVED_NAMES, False)):
        for name in names:
            value = getattr(env, name, None)
            if _is_per_env(value, n):
                out[f"env/{name}"] = value.detach().clone()
            elif required:
                skipped.append(name)
    for name in DICT_NAMES:
        for key, value in (getattr(env, name, None) or {}).items():
            if _is_per_env(value, n):
                out[f"envdict/{name}/{key}"] = value.detach().clone()
            else:
                skipped.append(f"{name}[{key}]")
    if skipped:
        raise RuntimeError(f"state that FORGE's reset writes is missing or not per-environment: {skipped}. Isaac Lab changed, "
                           "or STATE_NAMES in start_bank.py is wrong for this version: fix the list (the completeness "
                           "test, tests/test_isaac_bank.py, prints what the environment holds).")
    return out


def restore(env, bank: BankState, env_ids: torch.Tensor, bank_ids: torch.Tensor):
    """Write bank entries `bank_ids` into environments `env_ids` (index tensors on the simulation device). No physics step."""
    s, ids, b = bank.state, env_ids, bank_ids
    robot = env._robot
    env.scene.reset(ids)       # sensors outdated + zeroed history, asset external forces / actuator states cleared
    env.write_robot_state_raw(s["sim/joint_pos"][b], s["sim/joint_vel"][b], ids)
    origins = env.scene.env_origins[ids]
    for tag, asset in (("held", env._held_asset), ("fixed", env._fixed_asset)):
        pose = s[f"sim/{tag}_pose"][b].clone()
        pose[:, 0:3] += origins
        asset.write_root_pose_to_sim(pose, env_ids=ids)
        asset.write_root_velocity_to_sim(s[f"sim/{tag}_vel"][b], env_ids=ids)
    for key, value in s.items():
        if key.startswith("env/"):
            getattr(env, key[4:])[ids] = value[b]
        elif key.startswith("envdict/"):
            _, name, thresh = key.split("/", 2)
            for k, target in getattr(env, name).items():
                if str(k) == thresh:
                    target[ids] = value[b]
    robot.set_joint_position_target(env.ctrl_target_joint_pos[ids], env_ids=ids)
    robot.set_joint_effort_target(torch.zeros(len(ids), robot.num_joints, device=env.device), env_ids=ids)
    env.refresh_kinematics()


# ------------------------------------------------------------------------------------------------------------------- build
def build_bank(env, observe: Callable[[], Dict[str, np.ndarray]], size: int, seed: int, passes: int,
               log_every_s: float = 20.0) -> BankState:
    """`size` start states from ceil(size / N) real resets of all N environments (seed, seed + 1, ...)."""
    n, rounds = env.num_envs, math.ceil(size / env.num_envs)
    states: List[Dict[str, torch.Tensor]] = []
    observations: List[Dict[str, np.ndarray]] = []
    t0 = last = time.time()
    for r in range(rounds):
        env.reset_lockstep(seed=seed + r, passes=passes)
        states.append(capture(env))
        observations.append({k: np.array(v) for k, v in observe().items()})
        if time.time() - last > log_every_s:
            last = time.time()
            print(f"[start_bank] {r + 1}/{rounds} resets  ({last - t0:.0f} s)", flush=True)
    state = {k: torch.cat([c[k] for c in states], dim=0)[:size] for k in states[0]}
    obs = {k: np.concatenate([c[k] for c in observations], axis=0)[:size] for k in observations[0]}
    meta = {"size": int(size), "build_num_envs": n, "seed": int(seed), "passes": int(passes), "rounds": rounds,
            "build_seconds": time.time() - t0, "format": BANK_FORMAT}
    return BankState(state, obs, meta)


def load_or_build(env, observe: Callable[[], Dict[str, np.ndarray]], cfg: dict) -> BankState:
    size, seed = int(cfg.get("bank_size", 256)), int(cfg.get("bank_seed", 0))
    if size < 1:
        raise ValueError(f"bank_size must be >= 1, got {size}")
    passes = int(cfg.get("reset_passes", 1))
    cache_dir = cfg.get("bank_cache_dir")
    digest = bank_digest(cfg)
    path = Path(cache_dir) / f"start_bank_{digest}.pt" if cache_dir else None
    if path is not None and path.exists():
        bank = BankState.load(path, str(env.device))
        if bank.meta.get("format") == BANK_FORMAT and bank.size == size:
            bank.meta.update(from_cache=True, path=str(path), digest=digest)
            print(f"[start_bank] loaded {size} start states from {path}", flush=True)
            # FORGE creates some tensors (task_prop_gains, ...) only inside its first reset, and a restore writes into them in
            # place: with a bank from the cache this process never ran a real reset, so it runs one now (a few seconds).
            env.reset_lockstep(seed=seed, passes=passes)
            return bank
        print(f"[start_bank] {path} does not match (format/size): rebuilding", flush=True)
    print(f"[start_bank] building {size} start states ({math.ceil(size / env.num_envs)} real resets of {env.num_envs} "
          f"environments x {passes} pass(es))...", flush=True)
    bank = build_bank(env, observe, size, seed, passes)
    bank.meta.update(from_cache=False, digest=digest, path=str(path) if path else None)
    if path is not None:
        bank.save(path)
        print(f"[start_bank] saved to {path}", flush=True)
    print(f"[start_bank] built in {bank.meta['build_seconds']:.0f} s", flush=True)
    return bank
