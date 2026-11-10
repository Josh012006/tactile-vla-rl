"""The logic of a vector of independent environments, with no simulator in it (numpy only, tested without a GPU).

`VectorEnvManager` is everything the server does between two physics steps: which start state each environment gets, when an
episode ends, the auto-reset, the final observation, the frozen environments. The simulator is behind the small `SimAdapter`
interface: the real one (forge_backend.ForgeSim) drives Isaac Lab, the fake one (testing.FakeSim) is a few lines of numpy, so
the SAME manager code is what the CPU tests exercise and what runs on the cluster.

Start states: environment slot s draws its start states from its OWN random stream, seeded by (seed, s): a permutation of the
bank, consumed in order, reshuffled when exhausted. So the initialisation of an environment depends on nothing but its seed
and its own episode count, never on the other environments, and every bank entry is used once before any is repeated.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Protocol

import numpy as np

from .episode_tracker import EpisodeTracker, StepFlags

Obs = Dict[str, np.ndarray]


class SimAdapter(Protocol):
    num_envs: int

    def step_physics(self, actions: np.ndarray) -> None:
        """Advance ALL environments by one control step with (N, 8) actions."""

    def inserted(self) -> np.ndarray:
        """(N,) bool: FORGE's insertion flag at the current state."""

    def observe(self) -> Obs:
        """Observation of ALL environments, numpy, environment index first."""

    def restore(self, env_ids: np.ndarray, bank_ids: np.ndarray) -> None:
        """Write bank entry bank_ids[i] into environment env_ids[i]. No physics step; the other environments are untouched."""

    def hold_actions(self) -> np.ndarray:
        """(N, 8) actions that keep every robot where it is now."""


class VectorEnvManager:
    def __init__(self, sim: SimAdapter, bank_obs: Obs, max_steps: int, success_hold_steps: int = 1,
                 auto_reset: bool = True, seed: int = 0):
        self.sim = sim
        self.n = int(sim.num_envs)
        self.bank_obs = bank_obs
        sizes = {len(v) for v in bank_obs.values()}
        if len(sizes) != 1 or min(sizes) < 1:
            raise ValueError(f"bank_obs must hold the same, non-zero number of entries for every key, got {sizes}")
        self.bank_size = sizes.pop()
        self.tracker = EpisodeTracker(self.n, max_steps, success_hold_steps)
        self.auto_reset = bool(auto_reset)
        self.start_id = np.full(self.n, -1, dtype=np.int64)
        self._last_action: Optional[np.ndarray] = None
        self._started = False
        self._timings: Dict[str, list] = {}
        self.reseed(seed)

    # -- start-state sampling ----------------------------------------------------------------------------------------
    def reseed(self, seed: int):
        if int(seed) < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self.seed = int(seed)
        self._rngs = [np.random.default_rng([self.seed, slot]) for slot in range(self.n)]
        self._perm = [None] * self.n
        self._pos = [0] * self.n

    def _next_bank_id(self, slot: int) -> int:
        if self._perm[slot] is None or self._pos[slot] >= self.bank_size:
            self._perm[slot] = self._rngs[slot].permutation(self.bank_size)
            self._pos[slot] = 0
        value = int(self._perm[slot][self._pos[slot]])
        self._pos[slot] += 1
        return value

    # -- helpers -----------------------------------------------------------------------------------------------------
    def _timed(self, name: str, fn, *args):
        """fn(*args), its wall time added to timings()[name] (the simulator adapters synchronise the GPU before returning)."""
        t0 = time.perf_counter()
        out = fn(*args)
        slot = self._timings.setdefault(name, [0.0, 0])
        slot[0] += time.perf_counter() - t0
        slot[1] += 1
        return out

    def timings(self, reset: bool = False) -> Dict[str, dict]:
        """{name: {"total_s", "calls", "mean_ms"}} for step_physics / inserted / observe / restore since the last reset."""
        out = {k: {"total_s": v[0], "calls": v[1], "mean_ms": 1e3 * v[0] / max(v[1], 1)} for k, v in self._timings.items()}
        if reset:
            self._timings = {}
        return out

    def invalidate(self):
        """The simulator was touched behind the manager's back (debug ops): reset() is needed before the next step."""
        self._started = False

    def _check_ids(self, env_ids) -> np.ndarray:
        ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.n or len(np.unique(ids)) != len(ids)):
            raise ValueError(f"env_ids must be distinct integers in [0, {self.n}), got {ids.tolist()}")
        return ids

    def _empty_final(self, obs: Obs) -> Obs:
        return {k: v[:0] for k, v in obs.items()}

    def _wire(self, obs: Obs, flags: Optional[StepFlags], final_obs: Obs, final_ids: np.ndarray) -> dict:
        n = self.n
        zeros_b = np.zeros(n, dtype=bool)
        if flags is None:           # response of a reset: nothing happened yet
            body = dict(reward=np.zeros(n, np.float32), terminated=zeros_b, truncated=zeros_b.copy(),
                        acting=self.tracker.active.copy(), inserted=zeros_b.copy(), success=self.tracker.success.copy(),
                        t=self.tracker.t.copy(), all_done=self.tracker.all_done)
        else:
            body = dict(reward=flags.reward, terminated=flags.terminated, truncated=flags.truncated, acting=flags.acting,
                        inserted=flags.inserted, success=flags.success, t=flags.t, all_done=self.tracker.all_done)
        return {"obs": obs, "start_id": self.start_id.copy(), "final_obs": final_obs,
                "final_ids": np.asarray(final_ids, dtype=np.int64), **body}

    def _splice_bank_obs(self, obs: Obs, ids: np.ndarray, bank: np.ndarray) -> Obs:
        out = {}
        for k, v in obs.items():
            row = v if v.flags.writeable else np.array(v)
            row[ids] = self.bank_obs[k][bank]
            out[k] = row
        return out

    # -- the operations ----------------------------------------------------------------------------------------------
    def reset(self, env_ids=None, bank_ids=None, seed: Optional[int] = None) -> dict:
        """Start new episodes in `env_ids` (default all) from `bank_ids` (default: drawn by each slot's own stream).

        `env_ids=[]` is allowed and does nothing but report the current state (a client driving several servers uses it for
        the servers it does not reset). `seed` reseeds the start-state streams of EVERY slot: only with env_ids=None."""
        if seed is not None and env_ids is not None:
            raise ValueError("seed reseeds the start-state stream of every environment: pass it with env_ids=None only")
        ids = np.arange(self.n) if env_ids is None else self._check_ids(env_ids)
        if not self._started and len(ids) != self.n:
            raise RuntimeError("the first reset must reset every environment (reset(env_ids=None))")
        if len(ids) == 0:
            obs = self._timed("observe", self.sim.observe)
            return self._wire(obs, None, self._empty_final(obs), np.zeros(0, np.int64))
        if seed is not None:
            self.reseed(seed)
        if bank_ids is None:
            bank = np.array([self._next_bank_id(int(i)) for i in ids], dtype=np.int64)
        else:
            bank = np.asarray(bank_ids, dtype=np.int64).reshape(-1)
            if len(bank) != len(ids):
                raise ValueError(f"bank_ids has {len(bank)} entries for {len(ids)} environments")
            if bank.min() < 0 or bank.max() >= self.bank_size:
                raise ValueError(f"bank_ids must be in [0, {self.bank_size}), got {bank.tolist()}")
        self._timed("restore", self.sim.restore, ids, bank)
        self.tracker.reset_envs(ids)
        self.start_id[ids] = bank
        hold = self.sim.hold_actions()
        self._last_action = hold.copy() if self._last_action is None else self._last_action
        self._last_action[ids] = hold[ids]
        if len(ids) == self.n:      # every row comes from the bank: no need to read the simulator
            obs = {k: np.empty((self.n, *v.shape[1:]), v.dtype) for k, v in self.bank_obs.items()}
        else:
            obs = self._timed("observe", self.sim.observe)
        obs = self._splice_bank_obs(obs, ids, bank)
        self._started = True
        return self._wire(obs, None, self._empty_final(obs), np.zeros(0, np.int64))

    def step(self, actions, auto_reset: Optional[bool] = None) -> dict:
        if not self._started:
            raise RuntimeError("step() before reset(): call reset() first")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n, self._last_action.shape[1]):
            raise ValueError(f"actions must have shape {(self.n, self._last_action.shape[1])}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("non-finite action")
        auto = self.auto_reset if auto_reset is None else bool(auto_reset)

        if self.tracker.all_done:       # only with auto_reset=False: nothing to simulate, report it (no physics step is spent)
            obs = self._timed("observe", self.sim.observe)
            zeros = np.zeros(self.n, dtype=bool)
            flags = StepFlags(acting=zeros, inserted=zeros.copy(), reward=np.zeros(self.n, np.float32),
                              terminated=zeros.copy(), truncated=zeros.copy(), success=self.tracker.success.copy(),
                              t=self.tracker.t.copy(), all_done=True)
            return self._wire(obs, flags, self._empty_final(obs), np.zeros(0, np.int64))

        running = self.tracker.active
        actions = np.where(running[:, None], actions, self._last_action)     # ended episodes repeat their last action
        self._last_action = actions.copy()
        self._timed("step_physics", self.sim.step_physics, actions)
        flags = self.tracker.update(self._timed("inserted", self.sim.inserted))
        obs = self._timed("observe", self.sim.observe)

        done = flags.terminated | flags.truncated
        final_obs, final_ids = self._empty_final(obs), np.zeros(0, np.int64)
        if auto and done.any():
            final_ids = np.flatnonzero(done)
            final_obs = {k: np.array(v[final_ids]) for k, v in obs.items()}    # BEFORE the restore overwrites the rows
            bank = np.array([self._next_bank_id(int(i)) for i in final_ids], dtype=np.int64)
            self._timed("restore", self.sim.restore, final_ids, bank)
            self.tracker.reset_envs(final_ids)
            self.start_id[final_ids] = bank
            self._last_action[final_ids] = self.sim.hold_actions()[final_ids]
            obs = self._splice_bank_obs(obs, final_ids, bank)
        return self._wire(obs, flags, final_obs, final_ids)

    def observe(self) -> dict:
        return {"obs": self._timed("observe", self.sim.observe)}
