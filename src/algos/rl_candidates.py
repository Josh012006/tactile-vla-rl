"""The candidates of the RL methods: k chunks sampled from the VLA at a state, and the cache that keeps them. Pure numpy (the model is behind `policy.sample_candidates`, pi0.5 / pi0): it can be read and tested
anywhere. Design and decisions: docs/algos/rl.md (sections 2 and 3).

    sampler = CandidateSampler(policy, cfg)                           # cfg: src/algos/rl_config.RLConfig
    cands = sampler.sample(obs, prompt, row_seeds)                    # k chunks per environment, normalised + env space (src.models.interface.Candidates)
    cache = CandidateCache(cfg.num_candidates, horizon=16, state_dim=32)
    cache.put(state_ids, cands, version=0)                            # the candidates of a state are kept with the transition's id
    refresh(cache, sampler, source, ids, version=1, seed=...)         # PA-RL: after every distillation round, recompute them with the updated VLA

What is cached is the VLA's RAW candidates of every state (paper 5.2.2 and Appendix E); the optimisation (global ranking, local steps) is done on the fly with the current critic. The cache of
Best-of-N never expires (the VLA never changes: one version); PA-RL's gets a new version after every distillation epoch, and the states of the old one are `stale`.

The chunks are the model's NORMALISED ones (the first 8 dimensions of what it outputs, before the output transforms): the critic reads them and the local steps move them (alpha = 3e-4 in that
space, docs/algos/rl.md); `policy.normalized_to_targets` brings one back to the env's space to be executed or distilled.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.algos.rl_config import RLConfig
from src.models.interface import Candidates, num_envs_of, take

Prompts = Union[str, Sequence[str]]
Source = Callable[[np.ndarray], Tuple[Dict[str, np.ndarray], Prompts]]      # state ids -> (their observations, their prompt(s))


def candidate_seed(seed: int, state_id: int, version: int) -> int:
    """The sampling noise of the candidates of state `state_id` under the VLA of `version`: a function of the three alone (32 bits, what `plan(row_seeds=...)` takes), so a refresh can be repeated."""
    return int(np.random.SeedSequence([int(seed), int(state_id), int(version)]).generate_state(1, np.uint32)[0])


def concat_candidates(parts: Sequence[Candidates]) -> Candidates:
    """Candidates of consecutive slices of environments, as one."""
    if not parts:
        raise ValueError("no candidates to join")
    return Candidates(normalized=np.concatenate([p.normalized for p in parts]), states=np.concatenate([p.states for p in parts]), targets=np.concatenate([p.targets for p in parts]),
                      infer_ms=float(sum(p.infer_ms for p in parts)))


class CandidateSampler:
    """k candidates per environment from a pi0.5 / pi0 policy (`sample_candidates`), in model calls of a fixed shape: `rows_per_sample_call` environments per call (50 x 32 samples does not fit
    on an L40S, measured, docs/algos/rl.md section 5), every call padded to exactly that many rows, so ONE compilation whatever the number of environments."""

    def __init__(self, policy, cfg: RLConfig):
        self.policy, self.k, self.envs_per_call = policy, cfg.num_candidates, cfg.rows_per_sample_call

    def sample(self, obs: Dict[str, np.ndarray], prompt: Prompts, row_seeds: Sequence[int]) -> Candidates:
        """k candidates for every environment of `obs`. `row_seeds`: one int per environment (the noise of an environment's candidates is a function of ITS seed alone)."""
        n = num_envs_of(obs)
        if len(row_seeds) != n:
            raise ValueError(f"row_seeds has {len(row_seeds)} entries for {n} environments")
        prompts = [prompt] * n if isinstance(prompt, str) else list(prompt)
        if len(prompts) != n:
            raise ValueError(f"{len(prompts)} prompts for {n} environments")
        parts: List[Candidates] = []
        for start in range(0, n, self.envs_per_call):
            idx = list(range(start, min(n, start + self.envs_per_call)))
            parts.append(self.policy.sample_candidates(take(obs, idx), [prompts[i] for i in idx], self.k, row_seeds=[int(row_seeds[i]) for i in idx], pad_to=self.envs_per_call))
        return concat_candidates(parts)


class CandidateCache:
    """The VLA's candidates of every stored state, by state id (the dense index of the replay buffer), with the version of the VLA that drew them (-1 = nothing stored).

    Per state: k x H x 8 float32 normalised chunks (pi0.5: 16 x 8 x 32 x 4 B = 16 KiB; 20 000 states = about 330 MB) and the state row the output transforms need."""

    def __init__(self, num_candidates: int, horizon: int, state_dim: int, capacity: int = 4096, action_dim: int = 8):
        if min(num_candidates, horizon, state_dim, capacity, action_dim) < 1:
            raise ValueError("num_candidates, horizon, state_dim, capacity and action_dim must be >= 1")
        self.k, self.h, self.ds, self.d = num_candidates, horizon, state_dim, action_dim
        self._alloc(capacity)
        self.size = 0                                                  # ids 0 .. size - 1 have been written or skipped (the highest id put + 1)

    def _alloc(self, capacity: int) -> None:
        self.normalized = np.zeros((capacity, self.k, self.h, self.d), np.float32)
        self.states = np.zeros((capacity, self.ds), np.float32)
        self.version = np.full(capacity, -1, np.int64)

    @property
    def capacity(self) -> int:
        return len(self.version)

    @property
    def nbytes(self) -> int:
        return int(self.normalized.nbytes + self.states.nbytes + self.version.nbytes)

    def _grow(self, need: int) -> None:
        if need <= self.capacity:
            return
        old = (self.normalized, self.states, self.version)
        self._alloc(max(need, 2 * self.capacity))
        for new, was in zip((self.normalized, self.states, self.version), old):
            new[:len(was)] = was

    def put(self, ids: Sequence[int], cands: Candidates, version: int) -> None:
        """Store the candidates of the states `ids` (replacing any older ones) as drawn by the VLA of `version` (>= 0)."""
        ids = np.asarray(ids, np.int64)
        if version < 0:
            raise ValueError("version must be >= 0")
        if ids.ndim != 1 or len(ids) != len(cands.normalized) or len(set(ids.tolist())) != len(ids) or (len(ids) and ids.min() < 0):
            raise ValueError(f"ids must be {len(cands.normalized)} distinct non-negative integers, got {ids.tolist()[:8]}")
        if cands.normalized.shape[1:] != (self.k, self.h, self.d) or cands.states.shape[1:] != (self.ds,):
            raise ValueError(f"candidates {cands.normalized.shape} / states {cands.states.shape} do not fit the cache (k {self.k}, H {self.h}, {self.d} dims, state {self.ds})")
        if not len(ids):
            return
        self._grow(int(ids.max()) + 1)
        self.normalized[ids], self.states[ids], self.version[ids] = cands.normalized, cands.states, version
        self.size = max(self.size, int(ids.max()) + 1)

    def has(self, ids: Sequence[int]) -> np.ndarray:
        ids = np.asarray(ids, np.int64)
        return (ids >= 0) & (ids < self.capacity) & (self.version[np.clip(ids, 0, self.capacity - 1)] >= 0)

    def get(self, ids: Sequence[int]) -> np.ndarray:
        """The normalised candidates (N, k, H, 8) of the states `ids`; a state with nothing stored is an error, not zeros."""
        ids = np.asarray(ids, np.int64)
        missing = ids[~self.has(ids)]
        if len(missing):
            raise KeyError(f"no candidates stored for states {missing.tolist()[:8]}")
        return self.normalized[ids]

    def states_of(self, ids: Sequence[int]) -> np.ndarray:
        self.get(ids)
        return self.states[np.asarray(ids, np.int64)]

    def versions_of(self, ids: Sequence[int]) -> np.ndarray:
        return self.version[np.asarray(ids, np.int64)]

    def stale_ids(self, version: int) -> np.ndarray:
        """The ids below `size` whose candidates are not those of `version` (older ones, or never written): what a refresh has to recompute."""
        return np.flatnonzero(self.version[:self.size] != version)


def refresh(cache: CandidateCache, sampler: CandidateSampler, source: Source, ids: Sequence[int], version: int, seed: int, chunk: int = 0,
            progress: Optional[Callable[[int, int], None]] = None, clock: Callable[[], float] = time.monotonic) -> dict:
    """Recompute the candidates of the states `ids` with the VLA as it is NOW and store them under `version` (PA-RL: after every distillation epoch; paper Appendix E).
    `source(ids)` gives the observations and prompt(s) of those states (the replay buffer); they are read `chunk` states at a time (default: 4 model calls' worth) so the buffer is never
    held whole in memory. The noise of a state is `candidate_seed(seed, id, version)`: a refresh can be repeated. Returns {"states", "seconds", "model_ms"}."""
    ids = np.asarray(sorted(int(i) for i in ids), np.int64)
    chunk = chunk or 4 * sampler.envs_per_call
    t0, model_ms = clock(), 0.0
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        obs, prompts = source(part)
        cands = sampler.sample(obs, prompts, [candidate_seed(seed, i, version) for i in part])
        cache.put(part, cands, version)
        model_ms += cands.infer_ms
        if progress is not None:
            progress(min(start + chunk, len(ids)), len(ids))
    return {"states": int(len(ids)), "seconds": clock() - t0, "model_ms": model_ms}
