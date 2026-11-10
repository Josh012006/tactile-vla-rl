"""Per-environment episode bookkeeping (numpy only: no Isaac Sim, unit-tested without a GPU).

Why it lives on the SERVER side: the old client kept ONE scalar state (done, success, streak) and Isaac Lab ended episodes by
itself and reset inside env.step(), so the step that timed out returned a post-reset observation and an always-False success
flag. Here Isaac Lab never ends or resets anything (its timeout is disabled, see isaaclab/forge_env.py): this class decides, per
environment, when an episode is over, from the "inserted" flag FORGE computes at each step.

Rules, for each environment e whose episode is running before a step (`acting[e]`):
    streak[e]  = streak[e] + 1 if inserted[e] else 0
    terminated = streak[e] >= success_hold_steps                  (success: the episode ends at this step)
    t[e]      += 1
    truncated  = (not terminated) and t[e] >= max_steps           (timeout)
An environment whose episode ended stays frozen (nothing about it changes) until reset_envs() restarts it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepFlags:
    acting: np.ndarray        # (N,) bool  running BEFORE this step (the transition is valid)
    inserted: np.ndarray      # (N,) bool  FORGE's flag, False for rows that were not acting
    reward: np.ndarray        # (N,) float32  1.0 where inserted
    terminated: np.ndarray    # (N,) bool  ended by success on this step
    truncated: np.ndarray     # (N,) bool  ended by timeout on this step
    success: np.ndarray       # (N,) bool  the episode succeeded (cumulative)
    t: np.ndarray             # (N,) int32  steps of the episode so far
    all_done: bool            # no episode is running any more


class EpisodeTracker:
    def __init__(self, num_envs: int, max_steps: int, success_hold_steps: int = 1):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if success_hold_steps < 1:
            raise ValueError(
                f"success_hold_steps must be >= 1, got {success_hold_steps} (the legacy value 0 of the old client, "
                "'flag of the last step', always read False after a timeout and is not supported)")
        self.num_envs, self.max_steps, self.hold_steps = int(num_envs), int(max_steps), int(success_hold_steps)
        self.active = np.ones(self.num_envs, dtype=bool)
        self.streak = np.zeros(self.num_envs, dtype=np.int32)
        self.t = np.zeros(self.num_envs, dtype=np.int32)
        self.success = np.zeros(self.num_envs, dtype=bool)

    def reset_envs(self, env_ids=None):
        """Start new episodes in `env_ids` (default: all). The other environments are not touched."""
        ids = slice(None) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
        self.active[ids] = True
        self.streak[ids] = 0
        self.t[ids] = 0
        self.success[ids] = False

    @property
    def all_done(self) -> bool:
        return not bool(self.active.any())

    def update(self, inserted) -> StepFlags:
        """Account for one step of every environment. `inserted`: (N,) bool read after the step."""
        if self.all_done:
            raise RuntimeError("no episode is running (every environment ended and none was reset): call reset first")
        inserted = np.asarray(inserted, dtype=bool).reshape(self.num_envs)
        acting = self.active.copy()
        inserted = inserted & acting
        self.streak = np.where(acting, np.where(inserted, self.streak + 1, 0), self.streak).astype(np.int32)
        terminated = acting & (self.streak >= self.hold_steps)
        self.t = self.t + acting.astype(np.int32)
        truncated = acting & ~terminated & (self.t >= self.max_steps)
        self.success = self.success | terminated
        self.active = self.active & ~(terminated | truncated)
        return StepFlags(acting=acting, inserted=inserted, reward=inserted.astype(np.float32), terminated=terminated,
                         truncated=truncated, success=self.success.copy(), t=self.t.copy(), all_done=self.all_done)
