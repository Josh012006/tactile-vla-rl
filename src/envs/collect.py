"""Data collection and fixed-set evaluation on top of a vector of independent environments (numpy only).

    collector = Collector(env, policy, mode="steps")         # or mode="episodes"
    data = collector.collect(80)           # steps mode: 80 steps in EVERY environment -> Rollout, arrays (T=80, N, ...)
    update(data); collector.version += 1   # the learner works while the environments wait (they are not stepped meanwhile)
    data = collector.collect(80)           # the episodes in flight simply continue with the new policy

    collector = Collector(env, policy, mode="episodes")
    episodes = collector.collect(16)       # episodes mode: until >= 16 episodes have ended -> list[Trajectory]

`env` is a VectorEnvClient / EnvConnection (anything with reset / step / num_envs / spec). `policy(obs)` takes the observation
dict of ALL environments, arrays (N, ...), and returns actions (N, 8) float32, or (actions, extras) where `extras` is a dict of
(N, ...) arrays (log-prob, value, ...) stored next to the actions.

What both modes guarantee
  * no environment ever waits for another: an episode that ends is replaced at once (auto-reset), the batch keeps its size;
  * transitions are never mixed up across episodes: `next_obs` of a step that ended an episode is the LAST observation of that
    episode (final_obs), never the first one of the next; `terminated` (success) and `truncated` (timeout) are separate;
  * every transition carries the `version` of the policy that chose its action and the `start_id` of its episode;
  * `episodes` mode returns every episode that ended up to the step where the target is reached, so it may return a few more
    than asked (up to num_envs - 1). Keeping exactly the first E would over-represent short episodes (the successful ones):
    that is the cost of not letting environments wait. Fixed-set evaluation has no such bias: use run_fixed_episodes.
  * the episodes still in flight when a call returns are kept and continue in the next call (with the updated policy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

Obs = Dict[str, np.ndarray]


@dataclass
class EpisodeInfo:
    env: int
    start_id: int
    length: int
    success: bool
    terminated: bool
    truncated: bool
    episode_return: float


@dataclass
class Rollout:
    """`steps` mode. Arrays have the time index first, then the environment: (T, N, ...)."""
    obs: Obs
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    start_id: np.ndarray                 # (T, N) bank entry of the episode each transition belongs to
    t: np.ndarray                        # (T, N) step index inside the episode AFTER the transition
    version: np.ndarray                  # (T,) policy version that acted at each step
    next_obs: Optional[Obs]              # (T, N, ...) true next observation (final obs where an episode ended), or None
    last_obs: Obs                        # (N, ...) observation the next call starts from (bootstrap value for the cut)
    extras: Dict[str, np.ndarray]
    episodes: List[EpisodeInfo]          # episodes that ended during this call

    @property
    def done(self) -> np.ndarray:
        return self.terminated | self.truncated


@dataclass
class Trajectory:
    """`episodes` mode: one complete episode of one environment. Arrays have the time index first: (L, ...)."""
    env: int
    start_id: int
    obs: Obs                             # (L, ...) what the policy saw before each action
    action: np.ndarray                   # (L, 8)
    reward: np.ndarray                   # (L,)
    version: np.ndarray                  # (L,) policy version of each action (an update may fall inside an episode)
    final_obs: Obs                       # last observation, each (...) (after the last action)
    terminated: bool
    truncated: bool
    success: bool
    extras: Dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return len(self.reward)

    @property
    def episode_return(self) -> float:
        return float(self.reward.sum())


def _call_policy(policy, obs: Obs, n: int):
    out = policy(obs)
    actions, extras = out if isinstance(out, tuple) else (out, {})
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] != n:
        raise ValueError(f"policy must return (num_envs={n}, action_dim) actions, got {actions.shape}")
    return actions, {k: np.asarray(v) for k, v in dict(extras).items()}


class Collector:
    def __init__(self, env, policy: Callable, mode: str = "steps", store_next_obs: bool = True, seed: Optional[int] = None):
        if mode not in ("steps", "episodes"):
            raise ValueError(f"mode must be 'steps' or 'episodes', got {mode!r}")
        self.env, self.policy, self.mode = env, policy, mode
        self.store_next_obs = bool(store_next_obs)
        self.n = int(env.num_envs)
        self.version = 0
        self.total_env_steps = 0
        self._seed = seed
        self._obs: Optional[Obs] = None
        self._start_id: Optional[np.ndarray] = None
        self._ep_return = np.zeros(self.n, dtype=np.float64)
        self._rows: List[List[dict]] = [[] for _ in range(self.n)]      # episodes mode: rows of the episode in flight

    def start(self, seed: Optional[int] = None):
        """Start an episode in every environment (done by the first collect() if you do not call it)."""
        res = self.env.reset(seed=self._seed if seed is None else seed)
        self._obs, self._start_id = dict(res.obs), res.start_id.copy()
        self._ep_return[:] = 0.0
        self._rows = [[] for _ in range(self.n)]

    def collect(self, amount: int):
        if int(amount) < 1:
            raise ValueError(f"amount must be >= 1, got {amount}")
        if self._obs is None:
            self.start()
        return self._collect_steps(int(amount)) if self.mode == "steps" else self._collect_episodes(int(amount))

    # -- steps mode --------------------------------------------------------------------------------------------------
    def _collect_steps(self, steps: int) -> Rollout:
        n, rec = self.n, {k: [] for k in
                          ("obs", "action", "reward", "terminated", "truncated", "start_id", "t", "next_obs", "version")}
        extras: Dict[str, list] = {}
        episodes: List[EpisodeInfo] = []
        for _ in range(steps):
            actions, ex = _call_policy(self.policy, self._obs, n)
            res = self.env.step(actions, auto_reset=True)
            rec["obs"].append(self._obs)
            rec["action"].append(actions)
            rec["reward"].append(res.reward)
            rec["terminated"].append(res.terminated)
            rec["truncated"].append(res.truncated)
            rec["start_id"].append(self._start_id.copy())      # start state of the episode this transition belongs to
            rec["t"].append(res.t)
            rec["version"].append(self.version)
            if self.store_next_obs:
                rec["next_obs"].append(res.next_obs())
            for k, v in ex.items():
                extras.setdefault(k, []).append(v)
            self._ep_return += res.reward
            for e in np.flatnonzero(res.done):
                episodes.append(EpisodeInfo(int(e), int(self._start_id[e]), int(res.t[e]), bool(res.success[e]),
                                            bool(res.terminated[e]), bool(res.truncated[e]), float(self._ep_return[e])))
                self._ep_return[e] = 0.0
            self._obs, self._start_id = dict(res.obs), res.start_id.copy()
            self.total_env_steps += n
        stack = lambda seq: np.stack(seq, axis=0)
        stack_obs = lambda seq: {k: np.stack([o[k] for o in seq], axis=0) for k in seq[0]}
        return Rollout(
            obs=stack_obs(rec["obs"]), action=stack(rec["action"]), reward=stack(rec["reward"]),
            terminated=stack(rec["terminated"]), truncated=stack(rec["truncated"]), start_id=stack(rec["start_id"]),
            t=stack(rec["t"]), version=np.asarray(rec["version"], dtype=np.int64),
            next_obs=stack_obs(rec["next_obs"]) if self.store_next_obs else None, last_obs=dict(self._obs),
            extras={k: stack(v) for k, v in extras.items()}, episodes=episodes)

    # -- episodes mode -----------------------------------------------------------------------------------------------
    def _collect_episodes(self, target: int) -> List[Trajectory]:
        n, done_eps = self.n, []
        while len(done_eps) < target:
            actions, ex = _call_policy(self.policy, self._obs, n)
            res = self.env.step(actions, auto_reset=True)
            for e in range(n):
                self._rows[e].append({"obs": {k: v[e] for k, v in self._obs.items()}, "action": actions[e],
                                      "reward": res.reward[e], "version": self.version,
                                      "extras": {k: v[e] for k, v in ex.items()}})
            self.total_env_steps += n
            for j, e in enumerate(res.final_ids):
                rows, final = self._rows[e], {k: v[j] for k, v in res.final_obs.items()}
                done_eps.append(Trajectory(
                    env=int(e), start_id=int(self._start_id[e]),
                    obs={k: np.stack([r["obs"][k] for r in rows]) for k in rows[0]["obs"]},
                    action=np.stack([r["action"] for r in rows]), reward=np.asarray([r["reward"] for r in rows], np.float32),
                    version=np.asarray([r["version"] for r in rows], np.int64), final_obs=final,
                    terminated=bool(res.terminated[e]), truncated=bool(res.truncated[e]), success=bool(res.success[e]),
                    extras={k: np.stack([r["extras"][k] for r in rows]) for k in rows[0]["extras"]}))
                self._rows[e] = []
            self._obs, self._start_id = dict(res.obs), res.start_id.copy()
        return done_eps


def run_fixed_episodes(env, policy: Callable, bank_ids: Sequence[int], max_loops: Optional[int] = None) -> List[dict]:
    """Run exactly one episode from each start state in `bank_ids` (evaluation on a FIXED set), as many in parallel as there
    are environments, a free environment taking the next start state at once. No auto-reset: an ended environment is frozen
    until the client restarts it, so no episode is cut or repeated. Returns one dict per bank id, in order:
    {bank_id, success, terminated, truncated, length, episode_return, env}."""
    ids = [int(b) for b in bank_ids]
    if not ids:
        return []
    n = int(env.num_envs)
    results: List[Optional[dict]] = [None] * len(ids)
    res = env.reset()                                       # first reset must cover every environment
    queue = list(range(len(ids)))
    slot_of: Dict[int, int] = {}                            # environment -> index in `ids`
    returns = np.zeros(n, dtype=np.float64)

    def assign(free_envs: List[int]):
        nonlocal res
        take = free_envs[:len(queue)]
        if not take:
            return
        picked = [queue.pop(0) for _ in take]
        res = env.reset(env_ids=take, bank_ids=[ids[i] for i in picked])
        for e, i in zip(take, picked):
            slot_of[e], returns[e] = i, 0.0

    assign(list(range(min(n, len(ids)))))
    loops, limit = 0, max_loops if max_loops is not None else 10 ** 9
    while slot_of:
        loops += 1
        if loops > limit:
            raise RuntimeError(f"run_fixed_episodes: not finished after {limit} steps")
        actions, _ = _call_policy(policy, res.obs, n)
        res = env.step(actions, auto_reset=False)
        returns += np.where(res.acting, res.reward, 0.0)
        finished = [e for e in list(slot_of) if res.done[e]]
        for e in finished:
            i = slot_of.pop(e)
            results[i] = {"bank_id": ids[i], "success": bool(res.success[e]), "terminated": bool(res.terminated[e]),
                          "truncated": bool(res.truncated[e]), "length": int(res.t[e]), "episode_return": float(returns[e]),
                          "env": int(e)}
        if finished and queue:
            assign(finished)
    return results  # type: ignore[return-value]
