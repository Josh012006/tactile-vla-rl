"""The logic of a vector of environments driven in rounds, with no simulator in it (numpy only, tested without a GPU).

`VectorEnvManager` is everything the server does between two physics steps: when a round starts, when an episode ends, which
environments are held, the final observations. The simulator is behind the small `SimAdapter` interface: the real one
(forge_backend.ForgeSim) drives Isaac Lab, the fake one (tests/fakes.py) is a few lines of numpy, so the SAME manager code is
what the CPU tests exercise and what runs on the cluster.

Rounds. FORGE's reset acts on ALL environments and steps the physics, so the environments of a server start their episodes
together (a round) and the next round starts when the last of them has ended; an environment that ended earlier is held in place
and ignored (`acting` False). The real reset of round r is seeded with `seed + r`: the start state of every environment is a
function of (seed, r, num_envs) up to a small dependence on the episode that ran just before (Robotiq, measured in
tests/test_isaac_fidelity.py). The same sequence of resets and steps from a fresh server is bit-identical: a run is replayed
exactly by running the same sequence again.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Protocol

import numpy as np

from .episode_tracker import EpisodeTracker, StepFlags
from .protocol import MAX_ROUND

Obs = Dict[str, np.ndarray]


class SimAdapter(Protocol):
    num_envs: int

    def reset(self, seed: int) -> None:
        """The real reset of ALL environments, seeded with `seed`, followed by the step that renders the cameras."""

    def step_physics(self, actions: np.ndarray) -> None:
        """Advance ALL environments by one control step with (N, 8) actions."""

    def inserted(self) -> np.ndarray:
        """(N,) bool: FORGE's insertion flag at the current state."""

    def observe(self) -> Obs:
        """Observation of ALL environments, numpy, environment index first."""

    def hold_actions(self) -> np.ndarray:
        """(N, 8) actions that keep every robot where it is now."""


class VectorEnvManager:
    def __init__(self, sim: SimAdapter, max_steps: int, success_hold_steps: int = 1, auto_reset: bool = True, seed: int = 0):
        self.sim = sim
        self.n = int(sim.num_envs)
        self.tracker = EpisodeTracker(self.n, max_steps, success_hold_steps)
        self.auto_reset = bool(auto_reset)
        self.seed = self._check_seed(seed)
        self.round = -1                         # round of the episodes in flight (-1: nothing started yet)
        self._last_action: Optional[np.ndarray] = None
        self._started = False
        self._timings: Dict[str, list] = {}

    # -- helpers -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_seed(seed) -> int:
        if int(seed) < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        return int(seed)

    def _timed(self, name: str, fn, *args):
        """fn(*args), its wall time added to timings()[name] (the simulator adapters synchronise the GPU before returning)."""
        t0 = time.perf_counter()
        out = fn(*args)
        slot = self._timings.setdefault(name, [0.0, 0])
        slot[0] += time.perf_counter() - t0
        slot[1] += 1
        return out

    def timings(self, reset: bool = False) -> Dict[str, dict]:
        """{name: {"total_s", "calls", "mean_ms"}} for reset / step_physics / inserted / observe since the last reset=True."""
        out = {k: {"total_s": v[0], "calls": v[1], "mean_ms": 1e3 * v[0] / max(v[1], 1)} for k, v in self._timings.items()}
        if reset:
            self._timings = {}
        return out

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
        return {"obs": obs, "round_id": np.full(n, self.round, dtype=np.int64), "final_obs": final_obs,
                "final_ids": np.asarray(final_ids, dtype=np.int64), **body}

    def _start_round(self, round_id: int) -> Obs:
        """The real reset of every environment for round `round_id`; returns the first observation of the round."""
        if not 0 <= round_id < MAX_ROUND:
            raise ValueError(f"round must be in [0, {MAX_ROUND}), got {round_id}")
        self._timed("reset", self.sim.reset, self.seed + round_id)
        self.tracker.reset_envs()
        self._last_action = self.sim.hold_actions().copy()
        self.round = round_id
        self._started = True
        return self._timed("observe", self.sim.observe)

    # -- the operations ----------------------------------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, round: Optional[int] = None) -> dict:
        """Start a round in every environment (real FORGE reset).

        seed  : sets the seed of the whole sequence of rounds (round 0 unless `round` says otherwise);
        round : the position in the sequence (same seed, same round: the same start states up to the small dependence on the previous episode, see the top of the file);
        neither: the next round."""
        if seed is not None:
            self.seed = self._check_seed(seed)
            target = 0 if round is None else int(round)
        else:
            target = self.round + 1 if round is None else int(round)
        obs = self._start_round(target)
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
        actions = np.where(running[:, None], actions, self._last_action)     # held environments repeat their last action
        self._last_action = actions.copy()
        self._timed("step_physics", self.sim.step_physics, actions)
        flags = self.tracker.update(self._timed("inserted", self.sim.inserted))
        obs = self._timed("observe", self.sim.observe)

        final_ids = np.flatnonzero(flags.terminated | flags.truncated)
        final_obs = {k: np.array(v[final_ids]) for k, v in obs.items()}      # BEFORE a new round overwrites the observation
        if auto and self.tracker.all_done:
            obs = self._start_round(self.round + 1)
        return self._wire(obs, flags, final_obs, final_ids)

    def observe(self) -> dict:
        return {"obs": self._timed("observe", self.sim.observe)}
