"""The replay buffer of the RL methods: the transitions of the online episodes (and of the warm-up ones), one plan = one transition. Pure numpy. Design: docs/algos/rl.md.

    buf = ReplayBuffer(capacity=(warmup_episodes + total_episodes) * (max_plans + 1), horizon=16, gamma=0.99, step_reward=-1.0)
    ids = buf.add_episode(episode, warmup=False)          # T + 1 rows: the states s_0 .. s_T of an episode of T plans
    batch = buf.transition_batch(buf.sample_transitions(rng, 256, mixing_ratio=0.5))
    obs, prompts = buf.observations(ids)                  # the env's observation format: what `CandidateSampler.sample` and `refresh` take (the `source` of a refresh)

A ROW is a state. The rows of an episode are contiguous (s_0 .. s_T): the transition of row i (a plan: the chunk executed at s_i) goes to row i + 1, so the next state needs no storage of its own, and the
state ids are the ids of `CandidateCache` (the VLA's candidates of a state are cached under its row). The last row of an episode has no action.

Reward and ending (docs/algos/rl.md, decision): a plan costs `step_reward` (-1) until the plan that reaches the success, whose reward is 0 and which is TERMINAL (its target is the reward alone). An episode that
ends without success (the 150-step horizon) is a time-out, NOT terminal: the target of its last plan bootstraps from the last state, so that state needs candidates too (`needs_candidates`).
V^mu (the reference value of Cal-QL) is the Monte-Carlo return-to-go of the episode's own plans, with the plan-level reward and gamma, of EVERY stored trajectory (warm-up and online).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.envs import protocol as P

CAMERA_KEYS = {"exterior": P.EXTERIOR_IMAGE, "wrist": P.WRIST_IMAGE}
IMAGE_SHAPE = (224, 224, 3)
ACTION_DIM = P.ACTION_DIM


@dataclasses.dataclass
class Episode:
    """One finished episode of T plans: the T + 1 states s_0 .. s_T seen at the plans (and after the last one) and the T chunks executed."""
    exterior: np.ndarray         # (T + 1, 224, 224, 3) uint8
    wrist: np.ndarray            # (T + 1, 224, 224, 3) uint8
    joints: np.ndarray           # (T + 1, 7) float32: the joint positions (the offsets of the chunks are from them)
    gripper: np.ndarray          # (T + 1, 1) float32
    actions: np.ndarray          # (T, H, 8) float32: the chunk executed at each plan, in the model's NORMALISED space (what the critic reads)
    success: bool                # the last state is a success; otherwise the episode was cut (time-out)
    prompt: str
    touch: Optional[np.ndarray] = None   # (T + 1, touch_dim) float32: the critic's touch vector of each state (TouchSpec.build_batch: already scaled), when the critic sees touch

    @property
    def plans(self) -> int:
        return int(len(self.actions))

    def check(self, horizon: int, touch_dim: int = 0) -> None:
        t = self.plans
        if t < 1:
            raise ValueError("an episode has at least one plan")
        expected = {"exterior": (t + 1, *IMAGE_SHAPE), "wrist": (t + 1, *IMAGE_SHAPE), "joints": (t + 1, 7), "gripper": (t + 1, 1), "actions": (t, horizon, ACTION_DIM)}
        for name, shape in expected.items():
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"episode.{name} has shape {np.shape(getattr(self, name))}, expected {shape}")
        if touch_dim and (self.touch is None or np.shape(self.touch) != (t + 1, touch_dim)):
            raise ValueError(f"episode.touch must be ({t + 1}, {touch_dim}) (the critic sees touch), got {None if self.touch is None else np.shape(self.touch)}")
        if not touch_dim and self.touch is not None:
            raise ValueError("episode.touch is given but the buffer has no touch (touch_dim = 0)")


class ReplayBuffer:
    def __init__(self, capacity: int, horizon: int, gamma: float, step_reward: float = -1.0, touch_dim: int = 0):
        if capacity < 2 or horizon < 1 or not 0 < gamma <= 1:
            raise ValueError("capacity >= 2, horizon >= 1 and gamma in (0, 1] are needed")
        self.capacity, self.horizon, self.gamma, self.step_reward, self.touch_dim = capacity, horizon, gamma, step_reward, touch_dim
        self.exterior = np.zeros((capacity, *IMAGE_SHAPE), np.uint8)          # untouched rows cost no memory until written
        self.wrist = np.zeros((capacity, *IMAGE_SHAPE), np.uint8)
        self.joints = np.zeros((capacity, 7), np.float32)
        self.gripper = np.zeros((capacity, 1), np.float32)
        self.touch = np.zeros((capacity, touch_dim), np.float32)
        self.action = np.zeros((capacity, horizon, ACTION_DIM), np.float32)
        self.has_action = np.zeros(capacity, bool)
        self.reward = np.zeros(capacity, np.float32)
        self.terminal = np.zeros(capacity, bool)
        self.vmu = np.zeros(capacity, np.float32)
        self.needs_candidates = np.zeros(capacity, bool)
        self.warmup = np.zeros(capacity, bool)
        self.episode = np.full(capacity, -1, np.int32)
        self.plan_index = np.zeros(capacity, np.int32)
        self.prompt_id = np.zeros(capacity, np.int32)
        self.prompts: List[str] = []
        self.episodes: List[dict] = []                    # per episode: {"first": row, "plans": T, "success": bool, "warmup": bool, "return": discounted return of the plans}
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.exterior, self.wrist, self.joints, self.gripper, self.touch, self.action, self.has_action, self.reward, self.terminal, self.vmu,
                                          self.needs_candidates, self.warmup, self.episode, self.plan_index, self.prompt_id)))

    # -- writing --------------------------------------------------------------------------------------------------------------
    def add_episode(self, ep: Episode, warmup: bool) -> np.ndarray:
        """Store an episode: its T + 1 rows (contiguous). Returns their ids."""
        ep.check(self.horizon, self.touch_dim)
        t = ep.plans
        first = self.size
        if first + t + 1 > self.capacity:
            raise ValueError(f"the buffer is full: {first} rows of {self.capacity}, an episode of {t} plans needs {t + 1}")
        ids = np.arange(first, first + t + 1)
        rewards = np.full(t, self.step_reward, np.float32)
        if ep.success:
            rewards[-1] = 0.0
        returns = np.zeros(t, np.float32)                  # Monte-Carlo return-to-go, plan level
        running = 0.0
        for i in range(t - 1, -1, -1):
            running = float(rewards[i]) + self.gamma * running
            returns[i] = running
        self.exterior[ids], self.wrist[ids] = ep.exterior, ep.wrist
        self.joints[ids], self.gripper[ids] = ep.joints, ep.gripper
        if self.touch_dim:
            self.touch[ids] = ep.touch
        self.action[ids[:-1]] = ep.actions
        self.has_action[ids[:-1]] = True
        self.reward[ids[:-1]] = rewards
        self.terminal[ids[:-1]] = False
        self.terminal[ids[-2]] = bool(ep.success)
        self.vmu[ids[:-1]] = returns
        self.needs_candidates[ids[:-1]] = True
        self.needs_candidates[ids[-1]] = not ep.success      # a time-out's last state is bootstrapped from; a success is terminal
        self.warmup[ids] = bool(warmup)
        self.episode[ids] = len(self.episodes)
        self.plan_index[ids] = np.arange(t + 1)
        self.prompt_id[ids] = self._prompt_id(ep.prompt)
        self.episodes.append({"first": first, "plans": t, "success": bool(ep.success), "warmup": bool(warmup), "return": float(returns[0])})
        self.size = first + t + 1
        return ids

    def _prompt_id(self, prompt: str) -> int:
        if prompt not in self.prompts:
            self.prompts.append(prompt)
        return self.prompts.index(prompt)

    # -- reading --------------------------------------------------------------------------------------------------------------
    def transition_ids(self, warmup: Optional[bool] = None) -> np.ndarray:
        """The rows that carry a transition (a chunk executed at that state); `warmup` True / False keeps the warm-up / online ones only."""
        mask = self.has_action[:self.size]
        if warmup is not None:
            mask = mask & (self.warmup[:self.size] == warmup)
        return np.flatnonzero(mask)

    def candidate_ids(self) -> np.ndarray:
        """The rows whose VLA candidates are needed: every transition's state (the chunk executed came from them, and the critic's regulariser uses them) and the last state of a time-out."""
        return np.flatnonzero(self.needs_candidates[:self.size])

    def sample_transitions(self, rng: np.random.Generator, batch_size: int, mixing_ratio: float) -> np.ndarray:
        """`batch_size` rows with a transition, drawn with replacement: a `mixing_ratio` share from the warm-up data, the rest from the online data (Cal-QL's mixing ratio, paper B.1). While one
        of the two is empty (the warm-up of the critic: no online data yet) the whole batch comes from the other."""
        warm, online = self.transition_ids(True), self.transition_ids(False)
        if not len(warm) and not len(online):
            raise ValueError("the buffer holds no transition")
        n_warm = 0 if not len(warm) else (batch_size if not len(online) else int(round(mixing_ratio * batch_size)))
        parts = []
        if n_warm:
            parts.append(rng.choice(warm, n_warm))
        if batch_size - n_warm:
            parts.append(rng.choice(online, batch_size - n_warm))
        return np.concatenate(parts)

    def observations(self, ids: Sequence[int]):
        """(the env's observation dict of these rows, one prompt per row): the format of `CandidateSampler.sample` and `rl_candidates.refresh`."""
        ids = np.asarray(ids, np.int64)
        obs = {P.EXTERIOR_IMAGE: self.exterior[ids], P.WRIST_IMAGE: self.wrist[ids], P.JOINT_POSITION: self.joints[ids], P.GRIPPER_POSITION: self.gripper[ids]}
        return obs, [self.prompts[i] for i in self.prompt_id[ids]]

    def critic_state(self, ids: Sequence[int]) -> Dict[str, np.ndarray]:
        """What the critic reads of a state: the images by camera name (uint8, 224 x 224), the proprioception (7 joints + the gripper) and, when the critic sees touch, its touch vector."""
        ids = np.asarray(ids, np.int64)
        state = {"exterior": self.exterior[ids], "wrist": self.wrist[ids], "proprio": np.concatenate([self.joints[ids], self.gripper[ids]], axis=1)}
        if self.touch_dim:
            state["touch"] = self.touch[ids]
        return state

    def transition_batch(self, ids: Sequence[int]) -> Dict[str, np.ndarray]:
        """The transitions at rows `ids` (all rows with an action): the state, the chunk executed, the reward, whether it was terminal, the next state, V^mu, and the rows themselves."""
        ids = np.asarray(ids, np.int64)
        if not self.has_action[ids].all():
            raise ValueError("some rows carry no transition (the last state of an episode)")
        return {"ids": ids, "next_ids": ids + 1, "state": self.critic_state(ids), "next_state": self.critic_state(ids + 1), "action": self.action[ids], "reward": self.reward[ids],
                "terminal": self.terminal[ids], "vmu": self.vmu[ids], "warmup": self.warmup[ids]}

    # -- statistics -----------------------------------------------------------------------------------------------------------
    def success_rate(self, warmup: Optional[bool] = None, last: int = 0) -> float:
        """Share of the episodes that succeeded (`warmup` filters the kind, `last` keeps the last N episodes of that kind; 0 = all)."""
        eps = [e for e in self.episodes if warmup is None or e["warmup"] == warmup]
        eps = eps[-last:] if last else eps
        return float(np.mean([e["success"] for e in eps])) if eps else float("nan")
