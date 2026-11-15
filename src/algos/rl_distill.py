"""PA-RL's Stage II: distil the critic's best actions into the VLA (docs/algos/rl.md, paper Alg. 2 lines 9-12 and Appendix B.1 / E).

    distiller = Distiller(policy, optimiser, cfg, model_cfg)                 # policy: the pi0.5 policy of the run; optimiser: `ActionOptimiser.for_critic(net, cfg)`
    metrics = distiller.distill(critic_params, key, rng, buffer, cache)      # annotate every buffer state with pi^Opt (distill_selection), then `distill_epochs` epochs of SFT on them
    distiller.save_state(dir) / distiller.load_state(dir)                    # the VLA's trainable parameters and Adam's moments (resume)
    distiller.save_checkpoint(dir, meta)                                     # the trainable parameters, as an SFT checkpoint: `sft.load_checkpoint` reads it

1. ANNOTATION. Every state whose candidates are cached (`buffer.candidate_ids()`: the states of the plans and the last state of a time-out, warm-up included) gets the action of pi^Opt under the CURRENT
   critic: the VLA's cached candidates, global top-m by Q, local ascent, then the softmax sample (or the argmax) of `distill_selection` (the same `ActionOptimiser` and `pick` as the rollouts).
   The labels are fixed for the whole round (the critic does not change during the distillation) and are in the model's NORMALISED space, so they are the training targets as they are
   (`policy.training_batch_from_normalized`: no round trip through the env's space).
2. TRAINING. The VLA's own flow-matching loss on (observation, label) pairs, with the recipe of the SFT (src/algos/sft.py: openpi's AdamW, the clipping, the micro-batches, the draws of a step made once
   for the whole batch, the trainable parameters in float32 and the frozen ones in bfloat16): lr `distill_lr` constant, batch `distill_batch_size` in micro-batches of `distill_micro_batch_size`, `distill_epochs`
   epochs over the annotated states (a new random order each epoch, the incomplete tail dropped). Adam's state is kept from one round to the next (one continuous fine-tuning cut in rounds).
   The functions of the loop are those of sft._run, copied here and not shared: that loop is tested against openpi's own and is not touched.
After the epochs the policy is given the new weights (`policy.set_state`); refreshing the candidate cache with them is the caller's (RLRun).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.algos import sft
from src.algos.rl_action_opt import ActionOptimiser, pick
from src.algos.rl_buffer import ReplayBuffer
from src.algos.rl_candidates import CandidateCache
from src.algos.rl_config import RLConfig
from src.algos.rl_rollout import pad_rows

ANNOTATE_BATCH = 256            # states per call of the critic when annotating (the last batch is padded: one compilation)
PREFETCH_DEPTH = 3
DRAW_SEED_OFFSET = 2            # the flow-matching draws of update number n come from fold_in(key(cfg.seed + 2), n)


class VLATrainer:
    """The VLA's trainable parameters (float32), the frozen ones, Adam, and the jitted gradient step of the SFT's recipe."""

    def __init__(self, policy, cfg: RLConfig):
        import flax.nnx as nnx
        self.policy, self.cfg = policy, cfg
        sft_cfg = sft.load_config(overrides={"lr": cfg.distill_lr, "lr_end": cfg.distill_lr, "warmup_steps": 0, "steps": 1, "batch_size": cfg.distill_batch_size,
                                             "micro_batch_size": cfg.distill_micro_batch_size})
        original, self.frozen = sft.split_state(policy)
        self.trainable = sft.tree_cast(original, jnp.float32, copy=True)
        self.tx, _ = sft.make_optimizer(sft_cfg)                                   # AdamW + clipping of configs/algo/sft.yaml, constant learning rate
        self.opt_state = self.tx.init(self.trainable)
        self.updates = 0
        self.n_micro = cfg.distill_batch_size // cfg.distill_micro_batch_size
        self.draw_key = jax.random.key(cfg.seed + DRAW_SEED_OFFSET)
        self._prepare = jax.jit(lambda rng, observation, actions: policy.prepare_loss_batch(rng, observation, actions, train=True))

        @jax.jit
        def grad_fn(trainable, frozen, observation, actions, draws):
            loss_of = lambda t: policy.loss_of_rows(nnx.State.merge(t, frozen), observation, actions, draws)     # noqa: E731
            return jax.value_and_grad(loss_of)(trainable)

        @jax.jit
        def apply_fn(trainable, opt_state, grads):
            grads = jax.tree.map(lambda g: g / self.n_micro, grads)
            updates, opt_state = self.tx.update(grads, opt_state, trainable)
            return optax.apply_updates(trainable, updates), opt_state, optax.global_norm(grads)

        self._grad, self._apply = grad_fn, apply_fn

    def step(self, observation, actions) -> Dict[str, jnp.ndarray]:
        """One update on a batch of `distill_batch_size` rows (`policy.training_batch_from_normalized`)."""
        observation, draws = self._prepare(jax.random.fold_in(self.draw_key, self.updates), observation, actions)
        micro, total, losses = self.cfg.distill_micro_batch_size, None, []
        for i in range(self.n_micro):
            rows = slice(i * micro, (i + 1) * micro)
            cut = lambda x, rows=rows: x[rows]                                                          # noqa: E731
            loss, grads = self._grad(self.trainable, self.frozen, jax.tree.map(cut, observation), actions[rows], jax.tree.map(cut, draws))
            total = grads if total is None else jax.tree.map(jnp.add, total, grads)
            losses.append(loss)
        self.trainable, self.opt_state, grad_norm = self._apply(self.trainable, self.opt_state, total)
        self.updates += 1
        return {"loss": jnp.mean(jnp.stack(losses)), "grad_norm": grad_norm}

    def sync_policy(self) -> None:
        """The policy runs the weights as deployed (everything bfloat16): what samples the candidates and acts."""
        self.policy.set_state(sft.deployed_state(self.trainable, self.frozen))


class Distiller:
    def __init__(self, policy, optimiser: ActionOptimiser, cfg: RLConfig, model_cfg: Optional[dict] = None):
        self.policy, self.optimiser, self.cfg, self.model_cfg = policy, optimiser, cfg, model_cfg
        self.trainer = VLATrainer(policy, cfg)

    @property
    def updates(self) -> int:
        return self.trainer.updates

    # -- the labels -------------------------------------------------------------------------------------------------------------------------
    def annotate(self, critic_params, key, ids: np.ndarray, buffer: ReplayBuffer, cache: CandidateCache) -> tuple:
        """(labels (N, H, 8), metrics): the action of pi^Opt at each of the states `ids`, under the critic `critic_params`."""
        cfg, h = self.cfg, self.policy.info.action_horizon
        labels = np.zeros((len(ids), h, 8), np.float32)
        logs: list = []
        for i, start in enumerate(range(0, len(ids), ANNOTATE_BATCH)):
            part = ids[start:start + ANNOTATE_BATCH]
            n = len(part)
            state = {name: pad_rows(arr, ANNOTATE_BATCH) for name, arr in buffer.critic_state(part).items()}
            cands = pad_rows(cache.get(part), ANNOTATE_BATCH)
            result = self.optimiser(critic_params, state, cands)
            chosen = pick(result, cfg.distill_selection, jax.random.fold_in(key, i), cfg.softmax_temperature)
            labels[start:start + n] = np.asarray(chosen.actions)[:n]
            first_n = type(result)(*[x[:n] for x in result])
            logs.append((n, self.optimiser.summarise(first_n, type(chosen)(*[x[:n] for x in chosen]), cands[:n])))
        weights = np.array([n for n, _ in logs], np.float64)
        return labels, {k: float(np.average([lg[k] for _, lg in logs], weights=weights)) for k in logs[0][1]}

    # -- the epochs ---------------------------------------------------------------------------------------------------------------------------
    def _batches(self, ids: np.ndarray, labels: np.ndarray, rng: np.random.Generator, buffer: ReplayBuffer):
        """(Observation, actions) of the batches of `distill_epochs` epochs over the annotated states: a new permutation each epoch, the incomplete tail dropped. Built in a thread (openpi's input
        transforms are CPU work)."""
        size = self.cfg.distill_batch_size
        steps = len(ids) // size

        def make():
            for _ in range(self.cfg.distill_epochs):
                order = rng.permutation(len(ids))
                for b in range(steps):
                    rows = order[b * size:(b + 1) * size]
                    obs, prompts = buffer.observations(ids[rows])
                    yield self.policy.training_batch_from_normalized(obs, prompts, labels[rows])
        return sft.prefetch(make(), PREFETCH_DEPTH), steps * self.cfg.distill_epochs

    def distill(self, critic_params, key, rng: np.random.Generator, buffer: ReplayBuffer, cache: CandidateCache) -> Dict[str, float]:
        """Annotate every state that has cached candidates, then train. Returns the metrics of docs/algos/rl.md section 4 (the labels' statistics, the loss, the gradient norm, the times)."""
        ids = buffer.candidate_ids()
        if len(ids) < self.cfg.distill_batch_size:
            raise ValueError(f"{len(ids)} states to distil, fewer than a batch ({self.cfg.distill_batch_size})")
        t0 = time.monotonic()
        labels, metrics = self.annotate(critic_params, key, ids, buffer, cache)
        t1 = time.monotonic()
        feed, steps = self._batches(ids, labels, rng, buffer)
        losses, norms = [], []
        for observation, actions in feed:
            out = self.trainer.step(observation, actions)
            losses.append(out["loss"])
            norms.append(out["grad_norm"])
        jax.block_until_ready(self.trainer.trainable)
        self.trainer.sync_policy()
        t2 = time.monotonic()
        loss = np.asarray(jax.device_get(losses), np.float64)
        return {**{f"label_{k}": v for k, v in metrics.items()}, "states": int(len(ids)), "steps": steps, "loss_first": float(loss[0]), "loss_mean": float(loss.mean()), "loss_last": float(loss[-1]),
                "grad_norm_mean": float(np.mean(jax.device_get(norms))), "annotate_s": round(t1 - t0, 1), "train_s": round(t2 - t1, 1), "step_s": round((t2 - t1) / steps, 3)}

    # -- files --------------------------------------------------------------------------------------------------------------------------------
    def save_state(self, directory: Path) -> None:
        """The training state (trainable parameters, Adam) in <directory>/resume (the SFT's snapshot writer); the update count goes in the caller's state file."""
        sft.save_resume(Path(directory), self.trainer.updates, self.trainer.trainable, self.trainer.opt_state, 0)

    def load_state(self, directory: Path, updates: int) -> None:
        snap = Path(directory) / sft.RESUME_DIR
        meta = json.loads((snap / sft.META_FILE).read_text())
        self.trainer.trainable, self.trainer.opt_state = sft.load_resume(snap, meta, self.trainer.trainable, self.trainer.opt_state, 0)
        self.trainer.updates = int(updates)
        self.trainer.sync_policy()

    def save_checkpoint(self, directory: Path, meta: dict) -> None:
        """The trainable parameters as an SFT checkpoint (+ the normalisation statistics): `scripts/eval_sft.py`-style loading with `sft.load_checkpoint` on a policy of the same model YAML."""
        sft.save_checkpoint(directory, self.trainer.trainable, meta)
        sft.save_norm_stats(self.model_cfg, directory)
