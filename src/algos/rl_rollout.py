"""Acting, collecting and evaluating with the RL methods (docs/algos/rl.md): the VLA's k candidates at a state, the critic's choice among them, a round of K environments playing together,
and the storage of what they lived (replay buffer + candidate cache).

    actor = RLActor(policy, cfg, optimiser)                                  # policy: the pi0.5 policy (`sample_candidates`, `normalized_to_targets`); optimiser: `ActionOptimiser.for_critic(net, cfg)`
    collected, stats = collect_round(env, actor, mode="softmax", params=critic.params, seed=cfg.seed, round_id=r, key=key, replan=8)
    ids = store_round(buffer, cache, collected, warmup=False, version=0)     # the episodes into the buffer, their candidates into the cache (under the VLA's version)
    result = evaluate(actor, critic.params, env, seeds, replan=8, max_steps=150, master_seed=42)      # 100 fixed episodes with the optimisation at inference (src/eval/episodes.py)

Modes of `act` (what chooses the chunk that is executed among the k candidates of the VLA):
  "vla"      the VLA alone: its first candidate (the warm-up episodes of the base VLA, docs/algos/rl.md: 50 episodes, no critic yet);
  "softmax"  PA-RL's rollouts: a sample of pi^Opt over the optimised set (`rl_action_opt.pick`);
  "argmax"   Best-of-N's rollouts, and every evaluation (paper B.1: the argmax of pi^Opt at inference).
The candidates drawn to ACT are the candidates CACHED for that state (the cache holds the VLA's raw candidates of every stored state): nothing is sampled twice. A state that ended an episode by a time-out
gets its candidates in one extra call (its value is bootstrapped from, `ReplayBuffer.needs_candidates`); a success is terminal and needs none.

A round is K environments of one server that start together (`reset(seed, round)`) and play until they have all ended (success, or time-out at the server's horizon). The replan schedule is the same for all
(a new chunk every `replan` steps, the first at step 0), so every environment that still runs asks for a plan at the same time; the model is called for those only, but the critic's batch is padded to `pad_to`
rows (default K) so that it is compiled once. Seeds: the noise of the candidates of plan k of environment e of round r is a function of (seed, r, e, k) alone (`rollout_seed`); the softmax draws come from `key`.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.algos.rl_action_opt import ActionOptimiser, Optimised, pick
from src.algos.rl_buffer import Episode, ReplayBuffer
from src.algos.rl_candidates import CandidateCache, CandidateSampler
from src.algos.rl_config import RLConfig
from src.envs import protocol as P
from src.models.interface import Candidates, Plan, num_envs_of, take

MODES = ("vla", "softmax", "argmax")


def rollout_seed(seed: int, round_id: int, env: int, plan: int) -> int:
    """The sampling noise of the candidates of plan `plan` of environment `env` in round `round_id`: a function of the four alone (32 bits, `sample_candidates(row_seeds=...)`)."""
    return int(np.random.SeedSequence([int(seed), int(round_id), int(env), int(plan)]).generate_state(1, np.uint32)[0])


def critic_state_from_obs(obs: Dict[str, np.ndarray], touch=None) -> Dict[str, np.ndarray]:
    """What the critic reads of the observations of N environments: the same dict as `ReplayBuffer.critic_state` (images by camera, proprioception = 7 joints + gripper, touch when the critic sees it)."""
    state = {"exterior": np.asarray(obs[P.EXTERIOR_IMAGE]), "wrist": np.asarray(obs[P.WRIST_IMAGE]),
             "proprio": np.concatenate([np.asarray(obs[P.JOINT_POSITION], np.float32), np.asarray(obs[P.GRIPPER_POSITION], np.float32).reshape(len(obs[P.JOINT_POSITION]), -1)], axis=1)}
    if touch is not None and touch.enabled:
        state["touch"] = touch.build_batch(obs)
    return state


def _pad_rows(array: np.ndarray, rows: int) -> np.ndarray:
    """`array` (n, ...) with its last row repeated up to `rows` rows."""
    n = len(array)
    if n == rows:
        return array
    return np.concatenate([array, np.repeat(array[-1:], rows - n, axis=0)], axis=0)


@dataclasses.dataclass
class Acted:
    normalized: np.ndarray            # (n, H, 8) the chosen chunk, in the model's normalised space (what the critic reads and the buffer keeps)
    targets: np.ndarray               # (n, H, 8) the same chunk in the env's action space (what is executed)
    candidates: Candidates            # the VLA's k candidates of the n states (to be cached)
    q: Optional[np.ndarray]           # (n,) the critic's Q of the chosen chunk (None for the VLA alone)
    source: Optional[np.ndarray]      # (n,) index of the candidate the chosen chunk started from (None for the VLA alone)
    infer_ms: float                   # the VLA's time for the candidates
    logs: Dict[str, float]            # the optimisation metrics of `ActionOptimiser.summarise` (empty for the VLA alone)


class RLActor:
    def __init__(self, policy, cfg: RLConfig, optimiser: Optional[ActionOptimiser] = None):
        self.policy, self.cfg, self.optimiser = policy, cfg, optimiser
        self.sampler = CandidateSampler(policy, cfg)
        self.touch = cfg.critic_touch

    def candidates(self, obs: Dict[str, np.ndarray], prompt, row_seeds: Sequence[int]) -> Candidates:
        return self.sampler.sample(obs, prompt, row_seeds)

    def act(self, obs: Dict[str, np.ndarray], prompt, row_seeds: Sequence[int], mode: str, params=None, key=None, pad_to: Optional[int] = None) -> Acted:
        """The chunk to execute at each environment of `obs`, chosen among the VLA's k candidates by `mode`. `params`: the CURRENT critic's parameters; `key`: a jax PRNG key (softmax); `pad_to`:
        the critic's batch is padded to that many rows (one compilation whatever the number of environments that still play)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        cands = self.candidates(obs, prompt, row_seeds)
        n = len(cands.normalized)
        if mode == "vla":
            return Acted(normalized=cands.normalized[:, 0], targets=cands.targets[:, 0], candidates=cands, q=None, source=None, infer_ms=cands.infer_ms, logs={})
        if self.optimiser is None or params is None:
            raise ValueError(f"mode {mode!r} needs the critic (an ActionOptimiser and its parameters)")
        if mode == "softmax" and key is None:
            raise ValueError("softmax needs a jax PRNG key")
        rows = max(n, pad_to or n)
        state = {name: _pad_rows(arr, rows) for name, arr in critic_state_from_obs(obs, self.touch).items()}
        result = self.optimiser(params, state, _pad_rows(cands.normalized, rows))
        chosen = pick(result, mode, key, self.cfg.softmax_temperature)
        normalized = np.asarray(chosen.actions)[:n]
        targets = self.policy.normalized_to_targets(normalized[:, None], cands.states, np.asarray(obs[P.JOINT_POSITION], np.float32))[:, 0]
        first_n = Optimised(*[x[:n] for x in result])                                                       # the padding rows are not states: out of the metrics
        logs = self.optimiser.summarise(first_n, type(chosen)(*[x[:n] for x in chosen]), cands.normalized)
        return Acted(normalized=normalized, targets=targets, candidates=cands, q=np.asarray(chosen.q)[:n], source=np.asarray(chosen.source)[:n], infer_ms=cands.infer_ms, logs=logs)


class OptimisingPolicy:
    """The evaluation's policy: the VLA's candidates + the critic's argmax, behind the `plan` of `VLAPolicy` so that `src/eval/episodes.run_batch` plays it unchanged (the paper's evaluation: the VLA
    WITH action optimisation at inference)."""

    touch: dict = {}

    def __init__(self, actor: RLActor, params, mode: str = "argmax"):
        if mode == "softmax":
            raise ValueError("an evaluation takes the argmax (or the VLA alone), not a sample")
        self.actor, self.params, self.mode = actor, params, mode
        self.info = actor.policy.info
        self.logs: List[Dict[str, float]] = []

    def plan(self, obs, prompt, num_samples: int = 1, *, row_seeds=None, pad_to=None) -> Plan:
        if num_samples != 1:
            raise ValueError("the evaluation policy gives one chunk per environment")
        n = num_envs_of(obs)
        acted = self.actor.act(obs, prompt, row_seeds if row_seeds is not None else list(range(n)), self.mode, self.params, pad_to=pad_to)
        self.logs.append(acted.logs)
        return Plan(targets=acted.targets[:, None], raw=acted.normalized[:, None], infer_ms=acted.infer_ms)


# -- one round of training rollouts ---------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Collected:
    """An episode and the VLA's candidates of the states that need them: the T states of its plans and, for a time-out, its last state (T + 1)."""
    episode: Episode
    candidates: Candidates            # rows = those states, in order
    env: int
    round_id: int
    steps: int = 0                    # environment steps the episode lasted (the critic's UTD counts these)


def collect_round(env, actor: RLActor, *, mode: str, params=None, seed: int, round_id: int, key=None, replan: int, max_steps: Optional[int] = None, pad_to: Optional[int] = None,
                  clock: Callable[[], float] = time.monotonic) -> Tuple[List[Collected], dict]:
    """K = `env.num_envs` episodes played together from `env.reset(seed=seed, round=round_id)` (a VectorEnvClient), until they have all ended. Returns the episodes with their candidates and the round's
    statistics (success rate, plans, time in the model / the simulator, the mean of the optimisation metrics)."""
    import jax
    n, prompt = env.num_envs, env.spec.prompt
    horizon = actor.policy.info.action_horizon
    replan = max(1, min(int(replan), horizon))
    max_steps = int(max_steps or env.spec.max_steps)
    pad_to = pad_to or n
    started = clock()
    res = env.reset(seed=seed, round=round_id)
    keys = ("exterior", "wrist", "joints", "gripper", "touch")
    rows = [{k: [] for k in keys} for _ in range(n)]            # the states of each environment at its plans, then its last state
    chunks: List[List[np.ndarray]] = [[] for _ in range(n)]     # normalised chunk executed at each plan
    cand_parts: List[List[Candidates]] = [[] for _ in range(n)]
    targets_now: List[Optional[np.ndarray]] = [None] * n
    used = [0] * n
    ended = np.zeros(n, bool)
    ep_steps = np.zeros(n, np.int64)
    success = np.zeros(n, bool)
    actions = np.zeros((n, P.ACTION_DIM), np.float32)
    model_ms, logs = 0.0, []
    touch = actor.touch

    def record(obs, idx):
        sub = take(obs, idx)
        state = critic_state_from_obs(sub, touch)
        for j, e in enumerate(idx):
            rows[e]["exterior"].append(state["exterior"][j])
            rows[e]["wrist"].append(state["wrist"][j])
            rows[e]["joints"].append(state["proprio"][j, :7])
            rows[e]["gripper"].append(state["proprio"][j, 7:])
            if "touch" in state:
                rows[e]["touch"].append(state["touch"][j])

    t = 0
    while t < max_steps and not ended.all():
        need = [e for e in range(n) if not ended[e] and (targets_now[e] is None or used[e] >= replan)]
        if need:
            obs_need = take(res.obs, need)
            step_key = None if key is None else jax.random.fold_in(key, t)
            acted = actor.act(obs_need, prompt, [rollout_seed(seed, round_id, e, len(chunks[e])) for e in need], mode, params, step_key, pad_to=pad_to)
            model_ms += acted.infer_ms
            logs.append(acted.logs)
            record(res.obs, need)
            for j, e in enumerate(need):
                chunks[e].append(acted.normalized[j])
                cand_parts[e].append(Candidates(normalized=acted.candidates.normalized[j:j + 1], states=acted.candidates.states[j:j + 1], targets=acted.candidates.targets[j:j + 1], infer_ms=0.0))
                targets_now[e], used[e] = acted.targets[j], 0
        for e in range(n):
            if not ended[e]:
                actions[e] = targets_now[e][used[e]]
                used[e] += 1
        ep_steps += ~ended
        res = env.step(actions, auto_reset=False)
        t += 1
        newly = [e for e in range(n) if not ended[e] and bool(res.done[e])]
        if newly:
            record(res.next_obs(), newly)
            success[newly] = np.asarray(res.success)[newly]
            ended[newly] = True
    if not ended.all():
        raise RuntimeError(f"{int((~ended).sum())} episodes did not end within {max_steps} steps: is the server's max_steps_per_episode larger than the horizon of the rollout?")
    timed_out = [e for e in range(n) if not success[e]]
    if timed_out:                                                # the last state of a time-out is bootstrapped from: its candidates, in one more call
        last = {k: np.stack([rows[e][k][-1] for e in timed_out]) for k in ("exterior", "wrist", "joints", "gripper")}
        obs_last = {P.EXTERIOR_IMAGE: last["exterior"], P.WRIST_IMAGE: last["wrist"], P.JOINT_POSITION: last["joints"], P.GRIPPER_POSITION: last["gripper"]}
        extra = actor.candidates(obs_last, prompt, [rollout_seed(seed, round_id, e, len(chunks[e])) for e in timed_out])
        model_ms += extra.infer_ms
        for j, e in enumerate(timed_out):
            cand_parts[e].append(Candidates(normalized=extra.normalized[j:j + 1], states=extra.states[j:j + 1], targets=extra.targets[j:j + 1], infer_ms=0.0))
    collected = []
    for e in range(n):
        r = rows[e]
        episode = Episode(exterior=np.stack(r["exterior"]), wrist=np.stack(r["wrist"]), joints=np.stack(r["joints"]).astype(np.float32), gripper=np.stack(r["gripper"]).astype(np.float32),
                          actions=np.stack(chunks[e]).astype(np.float32), success=bool(success[e]), prompt=prompt, touch=np.stack(r["touch"]).astype(np.float32) if r["touch"] else None)
        parts = cand_parts[e]
        cands = Candidates(normalized=np.concatenate([p.normalized for p in parts]), states=np.concatenate([p.states for p in parts]), targets=np.concatenate([p.targets for p in parts]),
                           infer_ms=0.0)
        collected.append(Collected(episode=episode, candidates=cands, env=e, round_id=round_id, steps=int(ep_steps[e])))
    stats = {"success_rate": float(success.mean()), "episodes": n, "plans_mean": float(np.mean([c.episode.plans for c in collected])), "steps": t, "env_steps": int(ep_steps.sum()), "model_s": model_ms / 1e3,
             "wall_s": clock() - started}
    if logs and logs[0]:
        stats.update({f"opt_{k}": float(np.mean([lg[k] for lg in logs if k in lg])) for k in logs[0]})
    return collected, stats


def store_round(buffer: ReplayBuffer, cache: CandidateCache, collected: Sequence[Collected], *, warmup: bool, version: int) -> List[np.ndarray]:
    """The episodes of a round into the buffer and the VLA's candidates of their states into the cache (under the VLA's `version`). Returns the rows of each episode."""
    all_ids = []
    for item in collected:
        ids = buffer.add_episode(item.episode, warmup=warmup)
        with_cands = ids[:len(item.candidates.normalized)]
        cache.put(with_cands, item.candidates, version)
        all_ids.append(ids)
    return all_ids


# -- evaluation ----------------------------------------------------------------------------------------------------------------------------
def evaluate(actor: RLActor, params, env, seeds: Sequence[int], *, replan: int, max_steps: int, master_seed: int, label: str = "eval", step: int = 0, mode: str = "argmax",
             on_episode: Optional[Callable[[dict], None]] = None) -> dict:
    """The fixed episodes of the SFT evaluations with the optimisation at inference: `src/eval/episodes.run_batch` on `env` (a FRESH server of K >= len(seeds) environments: one batch, round 0 of `master_seed`,
    so the start states are those of every SFT evaluation on the same GPU), the critic's argmax over the VLA's candidates for each chunk (`mode="vla"` evaluates the VLA alone). Returns `summarize`'s dict plus
    the optimisation metrics and the time."""
    from src.eval import episodes as E
    k = env.num_envs
    if len(seeds) > k:
        raise ValueError(f"{len(seeds)} episodes in a server of {k} environments: they would not all start from the same reset (docs/eval/sft_eval.md)")
    policy = OptimisingPolicy(actor, params, mode)
    records: Dict[int, dict] = {}

    def keep(record: dict) -> None:
        records[int(record["episode"])] = record
        if on_episode is not None:
            on_episode(record)

    batch = E.run_batch(env, policy, {i: (i, int(s)) for i, s in enumerate(seeds)}, master_seed=master_seed, batch=0, replan=replan, max_steps=max_steps, on_episode=keep, pad_to=k)
    result = E.summarize(label, step, records, list(seeds))
    result["wall_s_batch"] = batch["wall_s"]
    logs = [lg for lg in policy.logs if lg]
    if logs:
        result["optimisation"] = {name: float(np.mean([lg[name] for lg in logs if name in lg])) for name in logs[0]}
    return result
