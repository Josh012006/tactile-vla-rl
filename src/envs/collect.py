"""Data collection and fixed-set evaluation on top of a vector of environments driven in rounds (numpy only).

    collector = Collector(env, policy, mode="steps")         # or mode="episodes"
    data = collector.collect(80)           # steps mode: 80 steps in EVERY environment -> Rollout, arrays (T=80, N, ...)
    update(data); collector.version += 1   # the learner works while the environments wait (they are not stepped meanwhile)
    data = collector.collect(80)           # the episodes in flight simply continue with the new policy

    collector = Collector(env, policy, mode="episodes")
    episodes = collector.collect(16)       # episodes mode: until >= 16 episodes have ended -> list[Trajectory]

`env` is a VectorEnvClient / EnvConnection (anything with reset / step / num_envs / spec). `policy(obs)` takes the observation
dict of ALL environments, arrays (N, ...), and returns actions (N, 8) float32, or (actions, extras) where `extras` is a dict of
(N, ...) arrays (log-prob, value, ...) stored next to the actions.

Rounds (protocol.py). The environments of a server start their episodes together and the next round starts when the last of
them has ended: an environment that ended early waits, held in place. Its rows are NOT transitions: `acting` is False there and
the learner must ignore them (steps mode keeps them so that the arrays stay rectangular; episodes mode never stores them).
`total_env_steps` counts valid transitions, `total_idle_steps` the steps spent waiting: the price of synchronous rounds.

What both modes guarantee
  * transitions are never mixed up across episodes: `next_obs` of a step that ended an episode is the LAST observation of that
    episode (final_obs), never the first one of the next round; `terminated` (success) and `truncated` (timeout) are separate;
  * every transition carries the `version` of the policy that chose its action and the `round` of its episode;
  * `episodes` mode returns every episode that ended up to the step where the target is reached, so it may return a few more
    than asked (up to num_envs - 1). Asking for a multiple of num_envs gives whole rounds. Stopping inside a round keeps the
    episodes that ended first, which over-represents the short (successful) ones: ask for whole rounds, or use run_fixed_rounds
    for an evaluation, which has no such bias;
  * the episodes still in flight when a call returns are kept and continue in the next call (with the updated policy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

Obs = Dict[str, np.ndarray]


@dataclass
class EpisodeInfo:
    env: int
    round: int
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
    acting: np.ndarray                   # (T, N) bool: the row is a valid transition (False: the env waits for the others)
    round: np.ndarray                    # (T, N) round of the episode each transition belongs to
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
    round: int
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
        self.total_env_steps = 0            # valid transitions
        self.total_idle_steps = 0           # steps an environment spent waiting for the others to end their episode
        self._seed = seed
        self._obs: Optional[Obs] = None
        self._round: Optional[np.ndarray] = None
        self._ep_return = np.zeros(self.n, dtype=np.float64)
        self._rows: List[List[dict]] = [[] for _ in range(self.n)]      # episodes mode: rows of the episode in flight

    def start(self, seed: Optional[int] = None):
        """Start round 0 of `seed` in every environment (done by the first collect() if you do not call it)."""
        res = self.env.reset(seed=self._seed if seed is None else seed)
        self._obs, self._round = dict(res.obs), res.round_id.copy()
        self._ep_return[:] = 0.0
        self._rows = [[] for _ in range(self.n)]

    def collect(self, amount: int):
        if int(amount) < 1:
            raise ValueError(f"amount must be >= 1, got {amount}")
        if self._obs is None:
            self.start()
        return self._collect_steps(int(amount)) if self.mode == "steps" else self._collect_episodes(int(amount))

    def _account(self, res):
        valid = int(res.acting.sum())
        self.total_env_steps += valid
        self.total_idle_steps += self.n - valid

    # -- steps mode --------------------------------------------------------------------------------------------------
    def _collect_steps(self, steps: int) -> Rollout:
        n, rec = self.n, {k: [] for k in
                          ("obs", "action", "reward", "terminated", "truncated", "acting", "round", "t", "next_obs", "version")}
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
            rec["acting"].append(res.acting)
            rec["round"].append(self._round.copy())            # round of the episode this transition belongs to
            rec["t"].append(res.t)
            rec["version"].append(self.version)
            if self.store_next_obs:
                rec["next_obs"].append(res.next_obs())
            for k, v in ex.items():
                extras.setdefault(k, []).append(v)
            self._ep_return += res.reward
            for e in np.flatnonzero(res.done):
                episodes.append(EpisodeInfo(int(e), int(self._round[e]), int(res.t[e]), bool(res.success[e]),
                                            bool(res.terminated[e]), bool(res.truncated[e]), float(self._ep_return[e])))
                self._ep_return[e] = 0.0
            self._obs, self._round = dict(res.obs), res.round_id.copy()
            self._account(res)
        stack = lambda seq: np.stack(seq, axis=0)
        stack_obs = lambda seq: {k: np.stack([o[k] for o in seq], axis=0) for k in seq[0]}
        return Rollout(
            obs=stack_obs(rec["obs"]), action=stack(rec["action"]), reward=stack(rec["reward"]),
            terminated=stack(rec["terminated"]), truncated=stack(rec["truncated"]), acting=stack(rec["acting"]),
            round=stack(rec["round"]), t=stack(rec["t"]), version=np.asarray(rec["version"], dtype=np.int64),
            next_obs=stack_obs(rec["next_obs"]) if self.store_next_obs else None, last_obs=dict(self._obs),
            extras={k: stack(v) for k, v in extras.items()}, episodes=episodes)

    # -- episodes mode -----------------------------------------------------------------------------------------------
    def _collect_episodes(self, target: int) -> List[Trajectory]:
        n, done_eps = self.n, []
        while len(done_eps) < target:
            actions, ex = _call_policy(self.policy, self._obs, n)
            res = self.env.step(actions, auto_reset=True)
            for e in np.flatnonzero(res.acting):                # a waiting environment has no transition
                self._rows[e].append({"obs": {k: v[e] for k, v in self._obs.items()}, "action": actions[e],
                                      "reward": res.reward[e], "version": self.version,
                                      "extras": {k: v[e] for k, v in ex.items()}})
            self._account(res)
            for j, e in enumerate(res.final_ids):
                rows, final = self._rows[e], {k: v[j] for k, v in res.final_obs.items()}
                done_eps.append(Trajectory(
                    env=int(e), round=int(self._round[e]),
                    obs={k: np.stack([r["obs"][k] for r in rows]) for k in rows[0]["obs"]},
                    action=np.stack([r["action"] for r in rows]), reward=np.asarray([r["reward"] for r in rows], np.float32),
                    version=np.asarray([r["version"] for r in rows], np.int64), final_obs=final,
                    terminated=bool(res.terminated[e]), truncated=bool(res.truncated[e]), success=bool(res.success[e]),
                    extras={k: np.stack([r["extras"][k] for r in rows]) for k in rows[0]["extras"]}))
                self._rows[e] = []
            self._obs, self._round = dict(res.obs), res.round_id.copy()
        return done_eps


def run_fixed_rounds(env, policy: Callable, rounds: Sequence[int], seed: int = 0) -> List[dict]:
    """Evaluation on a FIXED set of start states: one episode in every environment for each round in `rounds` of `seed`.

    The real reset is deterministic: (seed, round, num_envs) always gives the same starts, so two evaluations (two policies,
    two checkpoints) see exactly the same episodes. A round ends when every environment has ended; nothing starts by itself
    (auto_reset=False) and no episode is cut or repeated. Returns one dict per (round, environment), rounds in the order given and
    environments in index order: {round, env, success, terminated, truncated, length, episode_return}."""
    out: List[dict] = []
    n = int(env.num_envs)
    for r in rounds:
        res = env.reset(seed=seed, round=int(r))
        returns = np.zeros(n, dtype=np.float64)
        results: Dict[int, dict] = {}
        while not res.all_done:
            actions, _ = _call_policy(policy, res.obs, n)
            res = env.step(actions, auto_reset=False)
            returns += np.where(res.acting, res.reward, 0.0)
            for e in np.flatnonzero(res.done):
                results[int(e)] = {"round": int(r), "env": int(e), "success": bool(res.success[e]),
                                   "terminated": bool(res.terminated[e]), "truncated": bool(res.truncated[e]),
                                   "length": int(res.t[e]), "episode_return": float(returns[e])}
        out.extend(results[e] for e in sorted(results))
    return out
