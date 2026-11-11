"""Supervised fine-tuning (SFT) of a VLA on the FORGE demonstrations. Needs the `vla` dependency group (JAX, openpi); imported lazily.

    python -m scripts.sft --model pi0_fast                     # LoRA (configs/model/pi0_fast.yaml), the demonstrations of configs/algo/sft.yaml
    python -m scripts.sft --model pi05 --full --demos demos/isaaclab/ForgePegInsert_Robotiq_teleop

One model, one set of demonstrations, one run. The model is an `OpenPIPolicy` (src/models/openpi_policy.py): its input transforms (the
same as at inference), its loss (flow matching for pi0.5, next-token cross entropy for pi0-FAST) and its freeze filter are used as they
are; what this file adds is the optimiser, the loop, the evaluation and the checkpoints.

What is trained is `policy.trainable_filter`: openpi's freeze filter of the model YAML (LoRA matrices + SigLIP + action projections with
LoRA, everything without it). The trainable parameters are kept in float32 and the frozen ones in bfloat16, as openpi's own train.py does;
the optimiser is openpi's (AdamW, b2 0.95, gradient clipping at 1) with the schedule of configs/algo/sft.yaml. One device (no FSDP): the
batch is cut into micro-batches of `micro_batch_size` rows whose gradients are summed, so the memory depends on the micro-batch and the
optimisation on the batch.

Checkpoints hold the TRAINABLE parameters only, in bfloat16 (what the policy runs with): a LoRA checkpoint is small, and `load_checkpoint`
puts it on top of a freshly loaded policy built with the same model YAML (and the same overrides: the rank is part of the structure).
There is no resume: a run that stops is run again.

The evaluation inside the run (`eval_interval`) is not the success rate, which only the env gives: it is the loss on a fixed set of samples
(training ones, and held-out ones with `val_episodes`) and the error of the policy's own chunks against the demonstration's, with, for
pi0-FAST, what the generated token strings were like (`OpenPIPolicy.decode_stats`).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import yaml

from src.algos.demos import DemoSet

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "algo" / "sft.yaml"
PARAMS_DIR, META_FILE = "params", "meta.json"


# -- configuration ------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SFTConfig:
    """configs/algo/sft.yaml (every key documented there)."""
    demos: Tuple[str, ...]
    num_demos: int = 0
    val_episodes: int = 0
    prompt: str = "Insert the peg into the socket."
    steps: int = 4000
    batch_size: int = 64
    micro_batch_size: int = 16
    lr: float = 2.5e-5
    lr_end: float = 2.5e-5
    warmup_steps: int = 0
    weight_decay: float = 1e-10
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    clip_grad_norm: float = 1.0
    log_interval: int = 50
    eval_interval: int = 200
    eval_samples: int = 16
    save_interval: int = 200
    prefetch: int = 3
    seed: int = 0
    output_root: str = "runs/sft"

    def __post_init__(self):
        if not self.demos:
            raise ValueError("demos: at least one folder")
        positive = ("steps", "batch_size", "micro_batch_size", "log_interval", "eval_interval", "eval_samples", "save_interval", "prefetch")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.batch_size % self.micro_batch_size:
            raise ValueError(f"batch_size ({self.batch_size}) must be a multiple of micro_batch_size ({self.micro_batch_size})")
        if self.warmup_steps < 0 or self.num_demos < 0 or self.val_episodes < 0:
            raise ValueError("warmup_steps, num_demos and val_episodes must be >= 0")
        if self.lr <= 0 or self.lr_end < 0:
            raise ValueError("lr must be > 0 and lr_end >= 0")


def load_config(path: Optional[Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> SFTConfig:
    """configs/algo/sft.yaml with `overrides` (the CLI's `--set key=value`) applied. An unknown key is an error, never ignored."""
    raw = yaml.safe_load(Path(path or CONFIG_PATH).read_text()) or {}
    raw.update(overrides or {})
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ValueError(f"unknown SFT keys {unknown}; known: {sorted(fields)}")
    if isinstance(raw.get("demos"), str):
        raw["demos"] = [raw["demos"]]
    raw["demos"] = tuple(raw.get("demos") or ())
    return SFTConfig(**raw)


# -- state: trainable (float32) and frozen (bfloat16) parts ---------------------------------------------------------------------------
def tree_cast(tree, dtype):
    import jax
    return jax.tree.map(lambda x: x.astype(dtype), tree)


def split_state(policy):
    """(trainable, frozen): the policy's weights split by its trainable filter. The two merge back with `nnx.State.merge`."""
    import flax.nnx as nnx
    state = policy.state
    return state.filter(policy.trainable_filter), state.filter(nnx.Not(policy.trainable_filter))


def deployed_state(trainable, frozen):
    """The weights as the policy runs them (everything bfloat16): what a checkpoint holds and what the evaluation samples with."""
    import flax.nnx as nnx
    import jax.numpy as jnp
    return nnx.State.merge(tree_cast(trainable, jnp.bfloat16), frozen)


def make_optimizer(cfg: SFTConfig):
    import optax
    schedule = optax.warmup_cosine_decay_schedule(init_value=cfg.lr / (cfg.warmup_steps + 1), peak_value=cfg.lr, warmup_steps=cfg.warmup_steps,
                                                  decay_steps=max(cfg.steps, cfg.warmup_steps + 1), end_value=cfg.lr_end)
    tx = optax.chain(optax.clip_by_global_norm(cfg.clip_grad_norm),
                     optax.adamw(schedule, b1=cfg.b1, b2=cfg.b2, eps=cfg.eps, weight_decay=cfg.weight_decay))
    return tx, schedule


# -- checkpoints ---------------------------------------------------------------------------------------------------------------------
def save_checkpoint(directory: Path, trainable, meta: Mapping[str, Any]) -> None:
    """<directory>/params (orbax, the trainable parameters in bfloat16) and <directory>/meta.json."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    host = jax.device_get(tree_cast(trainable, jnp.bfloat16).to_pure_dict())
    ocp.PyTreeCheckpointer().save(directory / PARAMS_DIR, host)
    (directory / META_FILE).write_text(json.dumps(dict(meta), indent=2, default=str))


def load_checkpoint(policy, directory) -> dict:
    """Put the trained parameters of a checkpoint into `policy` (loaded from the same model YAML and overrides). Returns the checkpoint's meta.
    Refuses a checkpoint that does not fit the policy: a parameter the policy lacks, or of another shape (another LoRA rank)."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    directory = Path(directory).resolve()
    meta = json.loads((directory / META_FILE).read_text())
    saved = tu.flatten_dict(ocp.PyTreeCheckpointer().restore(directory / PARAMS_DIR))
    new = nnx.state(nnx.merge(policy.graphdef, policy.state))
    flat = tu.flatten_dict(new.to_pure_dict())
    problems = [f"{'/'.join(map(str, k))}: not in the policy" for k in saved if k not in flat]
    problems += [f"{'/'.join(map(str, k))}: {tuple(v.shape)} in the checkpoint, {tuple(flat[k].shape)} in the policy"
                 for k, v in saved.items() if k in flat and tuple(v.shape) != tuple(flat[k].shape)]
    if problems:
        raise ValueError(f"{directory} does not fit this policy (same model YAML and overrides as the run, rank included?):\n  "
                         + "\n  ".join(problems[:8]) + ("\n  ..." if len(problems) > 8 else ""))
    for k, v in saved.items():
        flat[k] = jnp.asarray(v).astype(flat[k].dtype)
    new.replace_by_pure_dict(tu.unflatten_dict(flat))
    policy.set_state(new)
    return meta


# -- data feed -----------------------------------------------------------------------------------------------------------------------
def row_batches(n_rows: int, batch_size: int, steps: int, seed: int) -> Iterator[np.ndarray]:
    """`steps` batches of `batch_size` distinct rows: a new random permutation per epoch, the incomplete tail of an epoch is dropped."""
    if n_rows < batch_size:
        raise ValueError(f"{n_rows} training steps but batch_size {batch_size}")
    rng = np.random.default_rng(seed)
    produced = 0
    while produced < steps:
        order = rng.permutation(n_rows)
        for i in range(0, n_rows - batch_size + 1, batch_size):
            if produced == steps:
                return
            produced += 1
            yield order[i:i + batch_size]


def prefetch(iterator: Iterator, depth: int) -> Iterator:
    """Run `iterator` in a thread, `depth` items ahead: the transforms of openpi (resize, tokens, FAST) are CPU work that would
    otherwise stop the GPU between two steps. An exception of the thread is raised in the consumer."""
    end, q, stop = object(), queue.Queue(maxsize=depth), threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def run():
        try:
            for item in iterator:
                if not put(item):
                    return
            put(end)
        except BaseException as exc:                      # noqa: BLE001  re-raised in the consumer
            put(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


def micro_batches(policy, dataset: DemoSet, rows: np.ndarray, micro: int) -> List[tuple]:
    """[(Observation, actions)] of `micro` rows each, in the model's own input and target space (OpenPIPolicy.training_batch)."""
    out = []
    for i in range(0, len(rows), micro):
        obs, prompts, target = dataset.sample(rows[i:i + micro])
        out.append(policy.training_batch(obs, prompts, target))
    return out


# -- evaluation inside the run -------------------------------------------------------------------------------------------------------
def evaluate_chunks(policy, dataset: DemoSet, rows: np.ndarray) -> dict:
    """The policy's own chunks (`plan`, sampled as at deployment) against the demonstration's, for `rows` of `dataset`: mean absolute error
    of the 7 joints over the chunk (rad), of the first step, of the gripper command; and, for pi0-FAST, the generated token strings
    (a chunk that could not be decoded is counted, and left out of the errors: its content is openpi's all-zero fallback)."""
    obs, prompts, target = dataset.sample(rows)
    stats = getattr(policy, "decode_stats", None)
    if stats is not None:
        stats.reset()
    plan = policy.plan(obs, prompts)
    chunk = plan.targets[:, 0]
    valid = np.ones(len(rows), bool) if policy.last_valid is None else np.asarray(policy.last_valid)[:, 0]
    out: Dict[str, Any] = {"infer_ms": plan.infer_ms, "chunks": int(len(rows)), "chunks_valid": int(valid.sum())}
    if valid.any():
        err = np.abs(chunk[valid] - target[valid])
        out.update(joint_abs_err=float(err[..., :7].mean()), joint_abs_err_step0=float(err[:, 0, :7].mean()), gripper_abs_err=float(err[..., 7].mean()))
    if stats is not None and stats.chunks:
        out["decode"] = stats.summary()
    return out


def device_memory_mib() -> Optional[dict]:
    """JAX's view of the first device's memory (None on the CPU)."""
    import jax
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:                                    # noqa: BLE001
        return None
    if not stats:
        return None
    return {"in_use": round(stats.get("bytes_in_use", 0) / 2**20), "peak": round(stats.get("peak_bytes_in_use", 0) / 2**20)}


class MetricsLog:
    """One JSON object per line in <run>/metrics.jsonl, and the same on the standard output. Every line carries `elapsed_s`, the seconds since the log was opened."""

    def __init__(self, path: Path):
        self.file = open(path, "a", buffering=1)
        self.t0 = time.monotonic()

    def write(self, **fields) -> None:
        line = json.dumps({**fields, "elapsed_s": round(time.monotonic() - self.t0, 1)}, default=float)
        self.file.write(line + "\n")
        print(line, flush=True)

    def close(self) -> None:
        self.file.close()


# -- the run ---------------------------------------------------------------------------------------------------------------------------
def train(policy, train_set: DemoSet, val_set: Optional[DemoSet], cfg: SFTConfig, run_dir: Path, meta: Optional[Mapping[str, Any]] = None) -> dict:
    """Run the SFT of `policy` on `train_set`. Writes <run_dir>/config.json, metrics.jsonl and checkpoints/<step>/. Returns a summary
    (final loss, checkpoints, time). `policy` ends with the final weights (`set_state`)."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    if train_set.horizon != policy.info.action_horizon:
        raise ValueError(f"the demonstrations are cut in chunks of {train_set.horizon} steps, the model's action_horizon is {policy.info.action_horizon}")
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    meta = {**(meta or {}), "model": policy.info.name, "family": policy.info.family, "action_horizon": policy.info.action_horizon}
    (run_dir / "config.json").write_text(json.dumps({"sft": dataclasses.asdict(cfg), **meta}, indent=2, default=str))
    log = MetricsLog(run_dir / "metrics.jsonl")

    trainable, frozen = split_state(policy)
    trainable = tree_cast(trainable, jnp.float32)
    n_trainable = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(trainable))
    n_frozen = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(frozen))
    tx, schedule = make_optimizer(cfg)
    opt_state = tx.init(trainable)
    n_micro = cfg.batch_size // cfg.micro_batch_size
    log.write(event="start", run_dir=str(run_dir), trainable_m=round(n_trainable / 1e6, 1), frozen_m=round(n_frozen / 1e6, 1),
              train_steps=len(train_set), val_steps=len(val_set) if val_set else 0, batch_size=cfg.batch_size,
              micro_batch_size=cfg.micro_batch_size, memory_mib=device_memory_mib())

    @jax.jit
    def grad_fn(trainable, frozen, rng, observation, actions):
        loss_of = lambda t: policy.loss(nnx.State.merge(t, frozen), rng, observation, actions, train=True)     # noqa: E731
        return jax.value_and_grad(loss_of)(trainable)

    @jax.jit
    def add_fn(total, grads):
        return jax.tree.map(jnp.add, total, grads)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def apply_fn(trainable, opt_state, grads):
        grads = jax.tree.map(lambda g: g / n_micro, grads)
        updates, opt_state = tx.update(grads, opt_state, trainable)
        return optax.apply_updates(trainable, updates), opt_state, optax.global_norm(grads)

    @jax.jit
    def eval_loss_fn(state, rng, observation, actions):
        return policy.loss(state, rng, observation, actions, train=False)

    eval_rows = {"train": np.random.default_rng(cfg.seed + 7).choice(len(train_set), min(cfg.eval_samples, len(train_set)), replace=False)}
    if val_set is not None:
        eval_rows["val"] = np.random.default_rng(cfg.seed + 8).choice(len(val_set), min(cfg.eval_samples, len(val_set)), replace=False)
    sets = {"train": train_set, "val": val_set}

    def evaluate(step: int) -> None:
        t_eval = time.monotonic()
        state = deployed_state(trainable, frozen)
        policy.set_state(state)
        policy.seed(cfg.seed + 1)                          # the same noise at every evaluation
        result: Dict[str, Any] = {}
        for name, rows in eval_rows.items():
            losses = [float(eval_loss_fn(state, jax.random.key(cfg.seed + 2), *b))
                      for b in micro_batches(policy, sets[name], rows, min(cfg.micro_batch_size, len(rows)))]
            result[name] = {"loss": float(np.mean(losses)), **evaluate_chunks(policy, sets[name], rows)}
        log.write(event="eval", step=step, eval_s=round(time.monotonic() - t_eval, 1), **result)

    def checkpoint(step: int) -> str:
        path = run_dir / "checkpoints" / str(step)
        t_save = time.monotonic()
        save_checkpoint(path, trainable, {**meta, "step": step, "sft": dataclasses.asdict(cfg)})
        size_mib = round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20)
        log.write(event="checkpoint", step=step, path=str(path), save_s=round(time.monotonic() - t_save, 1), size_mib=size_mib)
        return str(path)

    base_key = jax.random.key(cfg.seed)
    feed = prefetch((micro_batches(policy, train_set, rows, cfg.micro_batch_size)
                     for rows in row_batches(len(train_set), cfg.batch_size, cfg.steps, cfg.seed)), cfg.prefetch)
    t_start, t_window, losses, grad_norms, checkpoints, last_loss = time.monotonic(), time.monotonic(), [], [], [], float("nan")
    for step, micros in enumerate(feed, start=1):
        total, step_losses = None, []
        for i, (observation, actions) in enumerate(micros):
            loss, grads = grad_fn(trainable, frozen, jax.random.fold_in(base_key, step * n_micro + i), observation, actions)
            total = grads if total is None else add_fn(total, grads)
            step_losses.append(loss)
        trainable, opt_state, grad_norm = apply_fn(trainable, opt_state, total)
        losses.append(jnp.mean(jnp.stack(step_losses)))
        grad_norms.append(grad_norm)
        if step == 1:
            jax.block_until_ready(trainable)
            log.write(event="first_step", seconds=round(time.monotonic() - t_start, 1), note="includes the compilation", memory_mib=device_memory_mib())
            t_window = time.monotonic()
        if step % cfg.log_interval == 0 or step == cfg.steps:
            last_loss = float(np.mean(jax.device_get(losses)))
            now = time.monotonic()
            log.write(event="train", step=step, loss=last_loss, grad_norm=float(np.mean(jax.device_get(grad_norms))), lr=float(schedule(step - 1)),
                      step_s=round((now - t_window) / len(losses), 3), memory_mib=device_memory_mib())
            losses, grad_norms, t_window = [], [], now
        if step % cfg.eval_interval == 0 or step == cfg.steps:
            evaluate(step)
        if step % cfg.save_interval == 0 or step == cfg.steps:
            checkpoints.append(checkpoint(step))
    policy.set_state(deployed_state(trainable, frozen))
    summary = {"steps": cfg.steps, "final_loss": last_loss, "checkpoints": checkpoints, "seconds": round(time.monotonic() - t_start, 1)}
    log.write(event="done", **summary)
    log.close()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
