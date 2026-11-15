"""Stage I of PA-RL: action optimisation at a state with the CURRENT critic Q_theta (docs/algos/rl.md, paper Alg. 1 and Eq. 4.3). Used for the rollouts, the distillation labels, the critic's targets (the action of pi^Opt at s')
and the evaluation. Best-of-N is the same code with keep_top = 1 and no local step.

    opt = ActionOptimiser.for_critic(net, cfg)                       # keep_top, local_steps, local_lr come from the RLConfig
    result = opt(params, state, candidates)                          # candidates (B, k, H, 8): the VLA's cached normalised chunks of the B states; state: the buffer's critic state dict
    chosen = pick(result, mode, rng, temperature)                    # pi^Opt: "softmax" (a sample) or "argmax"; mode is cfg.rollout_selection / distill_selection / eval_selection
    chosen.actions                                                   # (B, H, 8), normalised space: `normalized_to_targets` brings them back to the env's space
    logs = opt.summarise(result, chosen, candidates)                 # floats for the metrics of docs/algos/rl.md section 4

What it does, per state:
  - global: Q of the k candidates, keep the top m (`keep_top`);
  - local (`local_steps` T > 0): on each kept chunk, T steps a <- a + alpha * grad_a Q(s, a) (alpha = `local_lr`, in the model's normalised space, no clipping), a step being KEPT ONLY IF Q rises strictly
    (Alg. 1 lines 5-9). The ascent is deterministic, so a refused step would be proposed again and refused again: "stop the loop of this action" and "keep the action while the later steps are refused" are the
    same thing, and the code just refuses every step that does not raise Q. A step whose Q is not finite (NaN, inf) is refused too;
  - the optimised set is the m chunks with their Q; pi^Opt is the softmax of their Q (temperature 1, Eq. 4.3) or the argmax.
The state is encoded ONCE (images through the CNNs); the T + 1 evaluations of the ascent run the MLP only (`rl_critic.score`).
The result is not differentiable (no gradient flows through the selection).
"""
from __future__ import annotations

import functools
from typing import Callable, Dict, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.algos import rl_critic
from src.algos.rl_config import RLConfig

SELECTIONS = ("softmax", "argmax")


class Optimised(NamedTuple):
    """The optimised set of B states (m = keep_top chunks each; order = descending Q BEFORE the local steps)."""
    actions: jnp.ndarray            # (B, m, H, 8) after the local steps
    q: jnp.ndarray                  # (B, m) Q of `actions`
    q_start: jnp.ndarray            # (B, m) Q of the same chunks before the local steps (= the candidates' Q)
    source: jnp.ndarray             # (B, m) int, index of each chunk among the k candidates
    candidate_q: jnp.ndarray        # (B, k) Q of all the candidates
    accepted: jnp.ndarray           # (B, m) int, number of local steps kept (0 .. T)
    displacement: jnp.ndarray       # (B, m) L1 norm of (actions - the starting chunk), summed over the whole chunk


class Chosen(NamedTuple):
    actions: jnp.ndarray            # (B, H, 8) the action of pi^Opt, normalised space
    q: jnp.ndarray                  # (B,)
    index: jnp.ndarray              # (B,) int, position in the optimised set
    source: jnp.ndarray             # (B,) int, index among the k candidates it started from


class ActionOptimiser:
    """encode(params, state) -> features and score(params, features, actions (B, N, H, 8)) -> Q (B, N) are the two halves of the critic (`rl_critic.encode_state`, `rl_critic.score`)."""

    def __init__(self, encode: Callable, score: Callable, keep_top: int, local_steps: int, local_lr: float):
        if keep_top < 1 or local_steps < 0 or local_lr <= 0:
            raise ValueError(f"keep_top >= 1, local_steps >= 0 and local_lr > 0 are needed, got {keep_top}, {local_steps}, {local_lr}")
        self.encode, self.score = encode, score
        self.keep_top, self.local_steps, self.local_lr = keep_top, local_steps, local_lr
        self._run = jax.jit(self._optimise)

    @classmethod
    def for_critic(cls, net: rl_critic.CriticNet, cfg: RLConfig) -> "ActionOptimiser":
        return cls(functools.partial(rl_critic.encode_state, net), functools.partial(rl_critic.score, net), cfg.keep_top, cfg.local_steps, cfg.local_lr)

    def __call__(self, params, state: Dict[str, jnp.ndarray], candidates) -> Optimised:
        """`params`: the CURRENT critic's (Q_theta, not the target network); candidates (B, k, H, 8) normalised chunks."""
        candidates = jnp.asarray(candidates, jnp.float32)
        if candidates.ndim != 4:
            raise ValueError(f"candidates must be (B, k, H, 8), got {candidates.shape}")
        if self.keep_top > candidates.shape[1]:
            raise ValueError(f"keep_top ({self.keep_top}) cannot exceed the number of candidates ({candidates.shape[1]})")
        return self._run(params, state, candidates)

    def _optimise(self, params, state, candidates) -> Optimised:
        features = self.encode(params, state)
        candidate_q = self.score(params, features, candidates)                                          # (B, k)
        q, source = jax.lax.top_k(candidate_q, self.keep_top)                                           # global: the m best, by Q
        start = jnp.take_along_axis(candidates, source[:, :, None, None], axis=1)                       # (B, m, H, 8)
        accepted = jnp.zeros(source.shape, jnp.int32)
        actions = start
        if self.local_steps > 0:
            def q_and_grad(a):
                def total(x):
                    values = self.score(params, features, x)
                    return values.sum(), values                                                          # a chunk's Q depends on that chunk only: the gradient of the sum is each chunk's own
                (_, values), grad = jax.value_and_grad(total, has_aux=True)(a)
                return values, grad

            def step(carry, _):
                a, current_q, grad, n = carry
                proposal = a + self.local_lr * grad
                new_q, new_grad = q_and_grad(proposal)
                keep = new_q > current_q                                                                 # strict; False for NaN
                pick_new = lambda new, old: jnp.where(keep[..., None, None], new, old)
                return (pick_new(proposal, a), jnp.where(keep, new_q, current_q), pick_new(new_grad, grad), n + keep.astype(jnp.int32)), None

            q0, grad0 = q_and_grad(start)
            (actions, q, _, accepted), _ = jax.lax.scan(step, (start, q0, grad0, accepted), None, length=self.local_steps)
        displacement = jnp.abs(actions - start).sum(axis=(-2, -1))
        return jax.lax.stop_gradient(Optimised(actions=actions, q=q, q_start=jnp.take_along_axis(candidate_q, source, axis=1), source=source, candidate_q=candidate_q,
                                               accepted=accepted, displacement=displacement))

    def summarise(self, result: Optimised, chosen: Optional[Chosen] = None, candidates=None) -> Dict[str, float]:
        """The metrics of docs/algos/rl.md section 4 for one batch of states, as floats (the caller averages them over batches)."""
        candidate_q = np.asarray(result.candidate_q)
        spread = candidate_q.std(axis=1)                                                                 # per state, over the k candidates
        logs = {"candidate_q_std_mean": float(spread.mean()), "candidate_q_std_p10": float(np.percentile(spread, 10)), "candidate_q_std_p50": float(np.percentile(spread, 50)),
                "candidate_q_std_p90": float(np.percentile(spread, 90)), "candidate_q_mean": float(candidate_q.mean()), "optimised_q_mean": float(np.asarray(result.q).mean())}
        if self.local_steps > 0:
            logs["local_q_gain_mean"] = float((np.asarray(result.q) - np.asarray(result.q_start)).mean())
            logs["local_accepted_share"] = float(np.asarray(result.accepted).mean() / self.local_steps)
            logs["local_displacement_l1_mean"] = float(np.asarray(result.displacement).mean())
        if chosen is not None:
            logs["chosen_q_mean"] = float(np.asarray(chosen.q).mean())
            if candidates is not None:                                                                   # the VLA's first sample is the action it would have executed alone
                first = np.asarray(candidates)[:, 0]
                logs["chosen_differs_from_first_sample_share"] = float((np.abs(np.asarray(chosen.actions) - first).reshape(len(first), -1).max(axis=1) > 0).mean())
        return logs


@functools.partial(jax.jit, static_argnames=("mode",))
def pick(result: Optimised, mode: str, rng=None, temperature: float = 1.0) -> Chosen:
    """pi^Opt over the optimised set: "argmax" (evaluation, Best-of-N) or "softmax" of Q / temperature (paper Eq. 4.3, temperature 1), a sample that needs `rng`."""
    if mode == "argmax":
        index = jnp.argmax(result.q, axis=-1)
    elif mode == "softmax":
        if rng is None:
            raise ValueError("softmax selection needs an rng")
        index = jax.random.categorical(rng, result.q / temperature, axis=-1)
    else:
        raise ValueError(f"selection must be one of {SELECTIONS}, got {mode!r}")
    rows = jnp.arange(index.shape[0])
    return Chosen(actions=result.actions[rows, index], q=result.q[rows, index], index=index, source=result.source[rows, index])
