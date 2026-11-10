"""Client side of the env protocol: one connection, or several servers driven as ONE vector of environments.

    env = VectorEnvClient([("localhost", 8102)])                      # one server with num_envs=16
    env = VectorEnvClient([("node1", 8102), ("node2", 8102)])         # two servers (e.g. two GPUs): 2 x num_envs
    res = env.reset(seed=0)                                           # every environment starts an episode
    while True:
        res = env.step(policy(res.obs))                               # actions: (env.num_envs, 8) float32
        # res.done rows ended an episode and were AUTO-RESET: res.obs holds their new first observation, res.final_obs the
        # last one of the ended episode. Nothing waits for the slowest environment.
    env.reset(env_ids=[3], bank_ids=[17])                             # restart ONE environment from start state 17

Differences with the old EnvClientWrapper, on purpose:
  * no silent recovery: when a call fails the exception propagates. The old wrapper re-created the environment, reset
    it and retried, which hides a lost episode in the data; let the job fail and resume from a checkpoint instead.
  * the wait for a server that is still starting (Isaac Sim takes ~2 minutes) only applies to the first connection.
  * a step costs ONE round trip (the old protocol used three: step, get_info_for_step, get_observation).
"""
from __future__ import annotations

import inspect
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import websockets.exceptions
import websockets.sync.client

from . import wire
from .protocol import EnvSpec, StepResult, concat_results

log = logging.getLogger("env_client")

SEED_STRIDE = 1_000_003       # seed of server k = seed + k * SEED_STRIDE (several servers must not share a seed)


class EnvServerError(RuntimeError):
    """The server answered with an error, or the link broke."""


class EnvConnection:
    """One persistent WebSocket connection to ONE env server."""

    def __init__(self, host: str = "localhost", port: int = 8102, connect_timeout_s: float = 900.0):
        self.host, self.port = host, int(port)
        self._conn = self._connect(connect_timeout_s)
        self.spec: EnvSpec = EnvSpec.from_wire(self.call("spec")["spec"])

    def _connect(self, timeout_s: float):
        uri = f"ws://{self.host}:{self.port}"
        kwargs = dict(compression=None, max_size=None, close_timeout=100)
        params = inspect.signature(websockets.sync.client.connect).parameters
        if "ping_interval" in params:
            kwargs["ping_interval"] = None
        if "legacy" in params:           # recent websockets: connect() without `with` is deprecated unless legacy=True
            kwargs["legacy"] = True
        deadline = time.time() + timeout_s
        while True:
            try:
                return websockets.sync.client.connect(uri, **kwargs)
            except (ConnectionRefusedError, OSError, EOFError, websockets.exceptions.InvalidMessage) as e:
                if time.time() > deadline:
                    raise EnvServerError(f"no env server at {uri} after {timeout_s:.0f} s ({e!r})") from e
                log.info("waiting for the env server at %s (%s)...", uri, type(e).__name__)
                time.sleep(5)

    def call(self, op: str, **payload) -> dict:
        try:
            self._conn.send(wire.pack({"op": op, **payload}))
            response = wire.unpack(self._conn.recv())
        except websockets.exceptions.ConnectionClosed as e:
            raise EnvServerError(f"connection to {self.host}:{self.port} closed during {op!r}: {e}") from e
        if response.get("status") != "ok":
            raise EnvServerError(f"{op!r} failed on {self.host}:{self.port}: {response.get('message')}")
        return response

    def reset(self, env_ids=None, bank_ids=None, seed: Optional[int] = None) -> StepResult:
        payload: Dict[str, Any] = {}
        if env_ids is not None:
            payload["env_ids"] = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if bank_ids is not None:
            payload["bank_ids"] = np.asarray(bank_ids, dtype=np.int64).reshape(-1)
        if seed is not None:
            payload["seed"] = int(seed)
        return StepResult.from_wire(self.call("reset", **payload))

    def step(self, actions: np.ndarray, auto_reset: Optional[bool] = None) -> StepResult:
        payload: Dict[str, Any] = {"actions": np.ascontiguousarray(actions, dtype=np.float32)}
        if auto_reset is not None:
            payload["auto_reset"] = bool(auto_reset)
        return StepResult.from_wire(self.call("step", **payload))

    def observe(self) -> Dict[str, np.ndarray]:
        return dict(self.call("observe")["obs"])

    def debug(self, kind: str, **kwargs) -> Any:
        return self.call("debug", kind=kind, **kwargs)["result"]

    def close(self):
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


class VectorEnvClient:
    """Several env servers (or one) seen as one batch of `num_envs` environments, index = position in `endpoints` order."""

    def __init__(self, endpoints: Sequence[Tuple[str, int]], connect_timeout_s: float = 900.0):
        if not endpoints:
            raise ValueError("endpoints is empty")
        self.conns: List[EnvConnection] = [EnvConnection(h, p, connect_timeout_s) for h, p in endpoints]
        specs = [c.spec for c in self.conns]
        for c, s in zip(self.conns[1:], specs[1:]):
            if not specs[0].same_task_as(s):
                raise EnvServerError(f"{c.host}:{c.port} does not run the same task as {self.conns[0].host}:{self.conns[0].port}")
        self.per_server = [s.num_envs for s in specs]
        self.num_envs = int(sum(self.per_server))
        self.spec: EnvSpec = EnvSpec(**{**specs[0].__dict__, "num_envs": self.num_envs})
        self.prompt = self.spec.prompt
        self._bounds = np.cumsum([0] + self.per_server)
        self._pool = ThreadPoolExecutor(max_workers=len(self.conns)) if len(self.conns) > 1 else None

    def _all(self, fn):
        """fn(conn, index) on every server, concurrently when there are several; results in server order."""
        if self._pool is None:
            return [fn(self.conns[0], 0)]
        return list(self._pool.map(lambda k: fn(self.conns[k], k), range(len(self.conns))))

    def reset(self, env_ids=None, bank_ids=None, seed: Optional[int] = None) -> StepResult:
        """Start new episodes in `env_ids` (default: every environment of every server); the others are not touched.

        bank_ids: start state of each reset environment (default: drawn by each environment's own stream, see
        episode_manager.py). seed (env_ids=None only): server k is seeded with seed + k * SEED_STRIDE, so servers never
        replay the same start-state order. A server that has none of `env_ids` is only asked for its current state."""
        if env_ids is None:
            if bank_ids is not None:
                raise ValueError("bank_ids needs env_ids")
            return concat_results(self._all(lambda c, k: c.reset(seed=None if seed is None else int(seed) + k * SEED_STRIDE)),
                                  self.per_server)
        if seed is not None:
            raise ValueError("seed reseeds every environment: pass it with env_ids=None only")
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0 or ids.min() < 0 or ids.max() >= self.num_envs or len(np.unique(ids)) != len(ids):
            raise ValueError(f"env_ids must be distinct integers in [0, {self.num_envs}), got {ids.tolist()}")
        bank = None if bank_ids is None else np.asarray(bank_ids, dtype=np.int64).reshape(-1)
        if bank is not None and len(bank) != len(ids):
            raise ValueError(f"bank_ids has {len(bank)} entries for {len(ids)} environments")
        b = self._bounds

        def one(c, k):
            mine = (ids >= b[k]) & (ids < b[k + 1])
            return c.reset(env_ids=ids[mine] - b[k], bank_ids=None if bank is None else bank[mine])
        return concat_results(self._all(one), self.per_server)

    def step(self, actions: np.ndarray, auto_reset: Optional[bool] = None) -> StepResult:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.spec.action_dim):
            raise ValueError(f"actions must have shape {(self.num_envs, self.spec.action_dim)}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("non-finite action")
        b = self._bounds
        return concat_results(self._all(lambda c, k: c.step(actions[b[k]:b[k + 1]], auto_reset)), self.per_server)

    def observe(self) -> Dict[str, np.ndarray]:
        parts = self._all(lambda c, k: c.observe())
        return {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}

    def hold_actions(self) -> np.ndarray:
        """(num_envs, 8) actions that keep every robot where it is (simulation servers only)."""
        return np.concatenate(self._all(lambda c, k: np.asarray(c.debug("hold_action"), dtype=np.float32)), axis=0)

    def timings(self, reset: bool = False) -> List[dict]:
        """Server-side time per phase (step_physics / inserted / observe / restore), one dict per server."""
        return self._all(lambda c, k: c.debug("timings", reset=reset))

    def debug(self, kind: str, **kwargs) -> List[Any]:
        return self._all(lambda c, k: c.debug(kind, **kwargs))

    def close(self):
        for c in self.conns:
            c.close()
        if self._pool is not None:
            self._pool.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
