"""Diagnostic ops of the Isaac Lab server used by tests/test_isaac_bank.py (not by training).

`probe` plays a scripted sequence of actions from a known start and returns what the simulator did, for four starts:
    real                the REAL FORGE reset of build round r (same seed as the bank build), after whatever ran before
    real_history        the real resets of rounds 0, 1, ..., r in a row: exactly the sequence that built the bank, so the
                        episode that precedes round r is the one that preceded it at build time (the reference of the test)
    restore             bank entries of round r written back into the same slots
    restore_permuted    the same entries written into the NEXT slot (slot i gets the entry captured in slot i + 1)
Comparing them measures (a) how faithful a restore is to the real reset, (b) whether a state captured in one slot is
independent of that slot (env-frame content), (c) how much the real reset itself depends on what ran before it (`real` vs
`real_history`). Repeating a mode gives its own repeatability floor (a different history precedes each repetition).
It moves the simulator: the episodes in flight are invalid afterwards (the client must reset).
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from isaaclab.utils.math import quat_apply

from . import protocol as P


def scripted_actions(hold: torch.Tensor, step: int, amplitude: float = 0.02, period: int = 24) -> torch.Tensor:
    """hold action + a small deterministic sinusoidal offset of the arm joints (different phase per joint)."""
    a = hold.clone()
    phase = torch.arange(7, device=hold.device, dtype=hold.dtype)
    a[:, 0:7] += amplitude * torch.sin(2.0 * math.pi * step / period + phase)
    return a


def _record(sim) -> Dict[str, np.ndarray]:
    env = sim.env
    cpu = lambda x: x.detach().float().cpu().numpy()
    origins = env.scene.env_origins
    axis_z = torch.zeros(env.num_envs, 3, device=env.device)
    axis_z[:, 2] = 1.0
    rec = {
        "joint_pos": cpu(env._robot.data.joint_pos), "joint_vel": cpu(env._robot.data.joint_vel),
        # The peg is round: a spin about its own axis changes the quaternion and nothing physical. So the pose is recorded as
        # its position (env frame, m), the direction of its axis (local z rotated into the world: a tilt of x rad moves it by
        # ~x) and, for information only, the raw quaternion.
        "held_pos": cpu(env._held_asset.data.root_pos_w - origins),
        "held_axis": cpu(quat_apply(env._held_asset.data.root_quat_w, axis_z)),
        "held_quat": cpu(env._held_asset.data.root_quat_w),
        "peg_socket_offset": cpu(env.peg_socket_offset()), "ft_force": cpu(env.force_sensor_smooth[:, 0:3]),
        "inserted": cpu(env.inserted()),
    }
    if env._tactile_sensors:
        rec["tactile"] = cpu(env.tactile_grids())
    for name, cam in (("exterior", env._exterior_cam), ("wrist", env._wrist_cam)):
        rec[f"image_mean_{name}"] = cpu(cam.data.output["rgb"]).mean(axis=(1, 2, 3))
    return rec


def probe(sim, request: dict) -> Dict[str, np.ndarray]:
    env, bank = sim.env, sim.bank
    mode, rnd, steps = str(request["mode"]), int(request["round"]), int(request["steps"])
    n = env.num_envs
    if bank.meta["build_num_envs"] != n:
        raise RuntimeError(f"the bank was built with {bank.meta['build_num_envs']} environments, this server has {n}")
    first = rnd * n
    if first + n > bank.size:
        raise ValueError(f"round {rnd} needs bank entries {first}..{first + n - 1}, the bank has {bank.size}")
    ids = np.arange(n)
    entries = first + ids
    seed, passes = int(bank.meta["seed"]), int(bank.meta["passes"])
    if mode == "real":
        env.reset_lockstep(seed=seed + rnd, passes=passes)
    elif mode == "real_history":
        for k in range(rnd + 1):                       # as in start_bank.build_bank
            env.reset_lockstep(seed=seed + k, passes=passes)
    elif mode in ("restore", "restore_permuted"):
        order = ids if mode == "restore" else np.roll(ids, -1)
        sim.restore(ids, entries[order])
    else:
        raise ValueError(f"unknown probe mode {mode!r}")
    hold = env.hold_action()
    traj = [_record(sim)]
    for s in range(steps):
        env.step(scripted_actions(hold, s))
        env.assert_no_native_reset()
        traj.append(_record(sim))
    torch.cuda.synchronize()
    return {k: np.stack([r[k] for r in traj], axis=0) for k in traj[0]}


PROBE_KINDS = ("probe",)
