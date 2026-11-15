"""The critic's update (docs/algos/rl.md, paper Eq. 3.3 and Alg. 2): Cal-QL with pi^Opt targets, one Adam step + the delayed target copy, and the loop that runs N of them on batches of the replay buffer.

    trainer = CriticTrainer(net, cfg, horizon=16)                              # the critic, its target copy, Adam, the action optimiser; `trainer.critic` / `trainer.opt_state` are the state
    batch = make_batch(buffer, cache, rng, cfg)                                # critic_batch_size transitions: half warm-up, half online (mixing_ratio) + the VLA's cached candidates of s and s'
    metrics = trainer.update(batch)                                            # one gradient step, then target <- tau * params + (1 - tau) * target; device scalars (`to_floats`)
    logs = run_updates(trainer, buffer, cache, rng, steps=cfg.warmup_critic_steps)    # the warm-up: 5000 steps on the warm-up episodes; online, the caller asks for utd * (environment steps) of them

The loss, for a batch of transitions (s, a, r, s') of the buffer (a = the chunk that was executed, in the model's normalised space):
    y        = r + gamma * (1 - terminal) * Qbar(s', a'),     a' = the action of pi^Opt at s' chosen with the CURRENT critic Q_theta (global top-m, local ascent, then `rollout_selection`: a sample of the
               softmax over the optimised set for PA-RL, the argmax for Best-of-N: the policy that is executed), Qbar = the delayed copy, which only EVALUATES it (group, docs/algos/rl.md);
    TD       = mean (Q_theta(s, a) - y)^2                      (MSE: HL-Gauss only if Q explode, paper B.1);
    Cal-QL   = mean_s [ mean_k max(Q_theta(s, a_k), V^mu(s)) ] - mean Q_theta(s, a)      a_k = the k candidates of the VLA cached at s, V^mu = the Monte-Carlo return-to-go stored with the row;
    loss     = TD + cal_ql_alpha * Cal-QL                      (alpha = 0.01, paper B.1).
A success is terminal (y = r, no bootstrap, so s' needs no candidates: s' is replaced by s in the candidate lookup and the value is masked); a time-out is not (its last state has candidates: `needs_candidates`).
The image of s is augmented (random shift 4 px, DrQ), the one of s' is not (proposal: the ranking and the target see the clean state; the paper gives nothing on this). The three evaluations of
the critic at s (data action, the k candidates) share one encoding of the shifted images.
Every number the metrics of docs/algos/rl.md section 4 ask of the critic update is in `metrics` (TD loss, Q mean / min / max on the data actions, Q minus the Monte-Carlo return, the candidates' Q spread, the target, ...).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.algos import rl_critic as rc
from src.algos.rl_action_opt import ActionOptimiser, pick
from src.algos.rl_buffer import ReplayBuffer
from src.algos.rl_candidates import CandidateCache
from src.algos.rl_config import RLConfig


def make_batch(buffer: ReplayBuffer, cache: CandidateCache, rng: np.random.Generator, cfg: RLConfig, batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
    """`batch_size` (default `critic_batch_size`) transitions drawn with the Cal-QL mixing ratio, their states, the next states and the VLA's cached candidates of both (host arrays)."""
    ids = buffer.sample_transitions(rng, batch_size or cfg.critic_batch_size, cfg.mixing_ratio)
    batch = buffer.transition_batch(ids)
    terminal = batch["terminal"]
    batch["cand"] = cache.get(ids)
    batch["next_cand"] = cache.get(np.where(terminal, ids, ids + 1))                   # a terminal transition has no candidates at s': the value is masked, any valid ones do
    return batch


def to_floats(metrics: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    return {k: float(v) for k, v in jax.device_get(metrics).items()}


class CriticTrainer:
    def __init__(self, net: rc.CriticNet, cfg: RLConfig, horizon: int, seed: Optional[int] = None):
        self.net, self.cfg = net, cfg
        self.optimiser = ActionOptimiser.for_critic(net, cfg)
        key = jax.random.key(cfg.seed if seed is None else seed)
        init_key, self.key = jax.random.split(key)
        self.critic = rc.init_critic(net, init_key, horizon=horizon)
        self.tx = optax.adam(cfg.critic_lr)
        self.opt_state = self.tx.init(self.critic.params)
        self.updates = 0
        self._step = jax.jit(self._train_step)

    # -- one update -----------------------------------------------------------------------------------------------------------
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
        """One Adam step on the batch (`make_batch`), then the target copy's soft update. Returns the metrics as device scalars."""
        self.key, step_key = jax.random.split(self.key)
        arrays = {k: batch[k] for k in ("state", "next_state", "action", "reward", "terminal", "vmu", "cand", "next_cand")}
        params, target, self.opt_state, metrics = self._step(self.critic.params, self.critic.target_params, self.opt_state, step_key, arrays)
        self.critic = rc.CriticState(params=params, target_params=target)
        self.updates += 1
        return metrics

    def _train_step(self, params, target_params, opt_state, key, batch):
        cfg, net = self.cfg, self.net
        shift_key, pick_key = jax.random.split(key)
        terminal = batch["terminal"].astype(jnp.float32)
        # the target: the action of pi^Opt at s' (ranked / optimised / chosen by the CURRENT critic), evaluated by the delayed copy
        optimised = self.optimiser(params, batch["next_state"], batch["next_cand"])
        chosen = pick(optimised, cfg.rollout_selection, pick_key, cfg.softmax_temperature)
        q_next = rc.q_values(net, target_params, batch["next_state"], chosen.actions[:, None])[:, 0]
        y = jax.lax.stop_gradient(batch["reward"] + cfg.gamma * (1.0 - terminal) * q_next)

        def loss_fn(p):
            features = rc.encode_state(net, p, batch["state"], train=True, rng=shift_key)
            q_data = rc.score(net, p, features, batch["action"][:, None])[:, 0]                       # (B,)
            q_cand = rc.score(net, p, features, batch["cand"])                                          # (B, k)
            td = jnp.mean((q_data - y) ** 2)
            regulariser = jnp.mean(jnp.mean(jnp.maximum(q_cand, batch["vmu"][:, None]), axis=1) - q_data)
            return td + cfg.cal_ql_alpha * regulariser, (td, regulariser, q_data, q_cand)

        (loss, (td, regulariser, q_data, q_cand)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = self.tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        target_params = jax.tree.map(lambda t, p: (1.0 - cfg.tau) * t + cfg.tau * p, target_params, params)
        metrics = {"loss": loss, "td_loss": td, "cal_ql_term": regulariser, "grad_norm": jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads))),
                   "q_data_mean": q_data.mean(), "q_data_min": q_data.min(), "q_data_max": q_data.max(),
                   "q_minus_return": (q_data - batch["vmu"]).mean(),                                  # over- (> 0) / under- (< 0) estimation against the Monte-Carlo return-to-go
                   "q_candidates_mean": q_cand.mean(), "candidate_q_std": q_cand.std(axis=1).mean(),
                   "candidates_below_vmu_share": (q_cand < batch["vmu"][:, None]).mean(),             # where Cal-QL's max clips the regulariser
                   "target_mean": y.mean(), "next_candidate_q_std": optimised.candidate_q.std(axis=1).mean(), "next_chosen_q_mean": chosen.q.mean()}
        if cfg.local_steps > 0:
            metrics["next_local_q_gain"] = (optimised.q - optimised.q_start).mean()
            metrics["next_local_accepted_share"] = optimised.accepted.mean() / cfg.local_steps
        return params, target_params, opt_state, metrics


def run_updates(trainer: CriticTrainer, buffer: ReplayBuffer, cache: CandidateCache, rng: np.random.Generator, steps: int, log_every: int = 0,
                log: Optional[Callable[[int, Dict[str, float]], None]] = None, clock: Callable[[], float] = time.monotonic) -> Dict[str, float]:
    """`steps` updates, each on a fresh batch of `make_batch`. Returns the mean of the metrics over the steps (and the wall time per step, `seconds_per_update`); with `log_every`, `log(update count,
    mean metrics of the last log_every steps)` is called every that many steps (the metrics are read from the device only then, so the steps are not synchronised one by one)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    window: list = []
    total: list = []
    start = clock()
    for i in range(1, steps + 1):
        window.append(trainer.update(make_batch(buffer, cache, rng, trainer.cfg)))
        if log_every and i % log_every == 0:
            means = _mean([to_floats(m) for m in window])
            total.extend(window)
            window = []
            if log is not None:
                log(trainer.updates, means)
    total.extend(window)
    jax.block_until_ready(total[-1])
    result = _mean([to_floats(m) for m in total])
    result["seconds_per_update"] = (clock() - start) / steps
    return result


def _mean(rows) -> Dict[str, float]:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
