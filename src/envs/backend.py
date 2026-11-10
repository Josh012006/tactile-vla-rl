"""Glue between the WebSocket server and a VectorEnvManager (numpy only: no Isaac Sim, usable with a fake simulator).

`ManagerBackend` implements ws_server.Backend: it turns the requests of protocol.py into manager calls and describes the task
(`spec`). The Isaac Lab side (forge_backend.py) builds a manager around the real simulator and wraps it in this class; the
tests wrap a fake simulator in the SAME class, so the protocol logic that runs on the cluster is the one the CPU tests run.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from . import protocol as P
from .episode_manager import VectorEnvManager


class ManagerBackend:
    def __init__(self, manager: VectorEnvManager, obs_spec: P.ObsSpec, prompt: str = "", control_hz: float = 15.0,
                 robot: Optional[dict] = None, debug_fn: Optional[Callable[[dict], Any]] = None):
        self.manager = manager
        self.obs_spec = obs_spec
        self.prompt, self.control_hz, self.robot = prompt, float(control_hz), dict(robot or {})
        self.debug_fn = debug_fn
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
        if self.debug_fn is None:
            return {"error": f"unknown debug kind: {kind}"}
        return self.debug_fn(request)
