"""Glue between the WebSocket server and a VectorEnvManager (numpy only: no Isaac Sim, usable with a fake simulator).

`ManagerBackend` implements ws_server.Backend: it turns the requests of protocol.py into manager calls and describes the task
(`spec`). The Isaac Lab side (isaaclab/forge_backend.py) builds a manager around the real simulator and wraps it in this class; the
tests wrap a fake simulator in the SAME class, so the protocol logic that runs on the cluster is the one the CPU tests run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import protocol as P
from .demo_store import save_teleop_demo
from .episode_manager import VectorEnvManager


class ManagerBackend:
    def __init__(self, manager: VectorEnvManager, obs_spec: P.ObsSpec, prompt: str = "", control_hz: float = 15.0,
                 robot: Optional[dict] = None, debug_fn: Optional[Callable[[dict], Any]] = None,
                 project_root: Optional[Path] = None):
        self.manager = manager
        self.obs_spec = obs_spec
        self.prompt, self.control_hz, self.robot = prompt, float(control_hz), dict(robot or {})
        self.debug_fn = debug_fn
        self.project_root = None if project_root is None else Path(project_root)    # where `save_teleop_demo` may write (inside it only)
        self._checked = False

    def spec(self) -> dict:
        m = self.manager
        return P.EnvSpec(
            protocol_version=P.PROTOCOL_VERSION, num_envs=m.n, action_dim=P.ACTION_DIM, max_steps=m.tracker.max_steps,
            success_hold_steps=m.tracker.hold_steps, control_hz=self.control_hz, prompt=self.prompt,
            auto_reset=m.auto_reset, obs=self.obs_spec, robot=self.robot).to_wire()

    def _check_obs(self, obs: dict):
        """Once: what the spec announces is what is sent."""
        if self._checked:
            return
        n = self.manager.n
        for key, (shape, dtype) in self.obs_spec.items():
            got = obs.get(key)
            if got is None or tuple(got.shape) != (n, *shape) or str(got.dtype) != dtype:
                raise RuntimeError(f"observation {key!r}: spec says {(n, *shape)} {dtype}, "
                                   f"got {None if got is None else (tuple(got.shape), str(got.dtype))}")
        extra = set(obs) - set(self.obs_spec)
        if extra:
            raise RuntimeError(f"observation keys not announced in the spec: {sorted(extra)}")
        self._checked = True

    def reset(self, request: dict) -> dict:
        out = self.manager.reset(seed=request.get("seed"), round=request.get("round"))
        self._check_obs(out["obs"])
        return out

    def step(self, request: dict) -> dict:
        out = self.manager.step(request["actions"], auto_reset=request.get("auto_reset"))
        self._check_obs(out["obs"])
        return out

    def observe(self) -> dict:
        return self.manager.observe()

    def debug(self, request: dict) -> Any:
        kind = request.get("kind")
        if kind == "timings":                      # server-side time per phase since the last call with reset=True
            return self.manager.timings(reset=bool(request.get("reset", False)))
        if kind == "teleop_step":                  # one control step of a human operator, in ONE round trip (see _teleop_step)
            return self._teleop_step(request)
        if kind == "save_teleop_demo":             # write a demonstration sent by the operator's PC (demo_store.py)
            if self.project_root is None:
                return {"error": "this server was started without a project root: it cannot save demonstrations"}
            return save_teleop_demo(self.project_root, request)
        if self.debug_fn is None:
            return {"error": f"unknown debug kind: {kind}"}
        return self.debug_fn(request)

    def _teleop_step(self, request: dict) -> dict:
        """IK + step in one message, for the operator of scripts/demos/teleop_client.py (the tunnel costs ~0.3 s per round trip).

        request: delta_pos (N, 3) m in the env frame (or (3,): the same for every environment), target_quat (N, 4) or (4,), cmd_gain,
        max_joint_step. The peg is rigidly held, so moving the fingertip by delta_pos moves the peg by it. The simulator's op
        `ik_joint_target` gives the 7 absolute joint targets of FORGE's damped-least-squares IK; the action is
            measured angles + clip(cmd_gain * (IK target - measured angles), +-max_joint_step),   gripper 1.0 (closed)
        exactly like scripts/demos/collect_waypoint_demos.py builds it (the action is the label of the observation BEFORE this call).
        Never auto-resets: when the episode ends the operator saves it, then calls `reset`. Returns the usual step response plus
        `action` (N, 8) and `clipped` (N,) bool (max_joint_step was hit)."""
        if self.debug_fn is None:
            return {"error": "this server has no simulator ops (no ik_joint_target)"}
        ik = self.debug_fn({"kind": "ik_joint_target", "delta_pos": request["delta_pos"], "target_quat": request["target_quat"]})
        q_now, q_ik = np.asarray(ik["q_now"], dtype=np.float64), np.asarray(ik["q_target"], dtype=np.float64)
        dq = float(request.get("cmd_gain", 1.0)) * (q_ik - q_now)
        max_step = float(request.get("max_joint_step", 0.05))
        clipped = np.abs(dq).max(axis=1) > max_step
        action = np.concatenate([q_now + np.clip(dq, -max_step, max_step), np.ones((len(q_now), 1))], axis=1).astype(np.float32)
        if not np.isfinite(action).all():
            raise RuntimeError(f"non-finite teleop action: {action}")
        return {**self.step({"actions": action, "auto_reset": False}), "action": action, "clipped": clipped}
