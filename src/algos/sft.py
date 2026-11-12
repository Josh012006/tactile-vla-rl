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
A run can be RESUMED (section "Resume" below): every `resume_interval` steps, at the last step and when the job is told to stop (SIGTERM), the
full training state is written to <run>/resume/, and `--resume <run>` continues from it with the same data order and the same noise.

The evaluation inside the run (`eval_interval`) is not the success rate, which only the env gives: it is the loss on a fixed set of samples
(training ones, and held-out ones with `val_episodes`) and the error of the policy's own chunks against the demonstration's, with, for
pi0-FAST, what the generated token strings were like (`OpenPIPolicy.decode_stats`).

Resume. The snapshot holds what the checkpoints do not: the trainable parameters in float32 (the bfloat16 checkpoints are rounded), the optimiser
state (AdamW moments, step counts) and the step. Everything else is a function of the step: the batch of step s comes from a permutation seeded by
`seed` (the batches already done are skipped without being prepared) and its noise from `fold_in(key(seed), s)`, so a resumed run sees what an
uninterrupted one would have seen (up to the non-determinism of the GPU kernels). A snapshot is about 3 times the float32 trainable parameters
(4.5 GiB measured with pi0.5 LoRA, 29 GiB measured with pi0-FAST full fine-tuning, written in 77 s), written aside and swapped in once complete, so the disk holds two of them for a moment.
What may change on resume: `steps` (only when the learning rate is constant, lr = lr_end: a decay depends on the total), the intervals, `eval_samples`,
`prefetch`. Everything that shapes the optimisation or the data may not (FROZEN_ON_RESUME), nor the model YAML.

wandb (src/algos/tracking.py): the events of metrics.jsonl are also logged to a wandb run (`wandb_mode`, `wandb_project`), whose id is kept in
<run>/wandb_id.txt; a resumed run continues that same wandb run (in the project it was created in).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import math
import queue
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import yaml

from src.algos import tracking
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
    resume_interval: int = 0
    prefetch: int = 3
    seed: int = 0
    wandb_mode: str = "online"                              # "online", "offline" or "disabled" (src/algos/tracking.py)
    wandb_project: str = "tactile-vla-rl"
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
        if self.warmup_steps < 0 or self.num_demos < 0 or self.val_episodes < 0 or self.resume_interval < 0:
            raise ValueError("warmup_steps, num_demos, val_episodes and resume_interval must be >= 0")
        if self.lr <= 0 or self.lr_end < 0:
            raise ValueError("lr must be > 0 and lr_end >= 0")
        if self.wandb_mode not in tracking.MODES:
            raise ValueError(f"wandb_mode must be one of {tracking.MODES}, got {self.wandb_mode!r}")


def load_config(path: Optional[Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> SFTConfig:
    """configs/algo/sft.yaml with `overrides` (the CLI's `--set key=value`) applied. An unknown key is an error, never ignored."""
    return config_from_dict(yaml.safe_load(Path(path or CONFIG_PATH).read_text()) or {}, overrides)


def config_from_dict(raw: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> SFTConfig:
    """An SFTConfig from a dict of its keys (the YAML, or the `sft` block of a run's config.json) and overrides."""
    raw = dict(raw)
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
def tree_cast(tree, dtype, copy: bool = False):
    """The tree with every leaf in `dtype`. copy=True gives new buffers even when the dtype already matches (the training state is donated to the
    optimiser step, so it must not share buffers with the policy's own state)."""
    import jax
    import jax.numpy as jnp
    return jax.tree.map((lambda x: jnp.array(x, dtype=dtype, copy=True)) if copy else (lambda x: x.astype(dtype)), tree)


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


# -- resume ---------------------------------------------------------------------------------------------------------------------------
RESUME_DIR = "resume"
# What may not differ between a run and its resume: it would change the optimisation or the data. (`steps` is checked apart.)
FROZEN_ON_RESUME = ("demos", "num_demos", "val_episodes", "prompt", "batch_size", "micro_batch_size", "lr", "lr_end", "warmup_steps", "weight_decay",
                    "b1", "b2", "eps", "clip_grad_norm", "seed")


def check_resume_compatible(saved: Mapping[str, Any], cfg: SFTConfig, saved_model: Optional[Any] = None, model: Optional[Any] = None) -> None:
    """Refuse a resume whose settings (or model YAML) differ from the run's in what shapes the optimisation. `saved` is the `sft` block of config.json."""
    now = dataclasses.asdict(cfg)
    now["demos"], saved = list(now["demos"]), {**saved, "demos": list(saved.get("demos", ()))}
    diff = [f"{k}: {saved.get(k)!r} in the run, {now[k]!r} now" for k in FROZEN_ON_RESUME if saved.get(k) != now[k]]
    if saved.get("steps") != now["steps"] and saved.get("lr") != saved.get("lr_end"):
        diff.append(f"steps: {saved.get('steps')} in the run, {now['steps']} now, and the learning rate decays (lr != lr_end): the schedule depends on the total")
    if saved_model is not None and model is not None and json.loads(json.dumps(saved_model, default=str)) != json.loads(json.dumps(model, default=str)):
        diff.append("the model YAML (and its overrides) differs from the run's")
    if diff:
        raise ValueError("cannot resume: " + "; ".join(diff))


def find_resume(run_dir: Path) -> Optional[Tuple[Path, dict]]:
    """The newest COMPLETE snapshot of a run (meta.json is written last), or None. A crash while one is being swapped in leaves the other behind."""
    found = []
    for name in (RESUME_DIR, RESUME_DIR + ".new", RESUME_DIR + ".old"):
        meta_path = Path(run_dir) / name / META_FILE
        if meta_path.exists():
            found.append((json.loads(meta_path.read_text()), Path(run_dir) / name))
    if not found:
        return None
    meta, directory = max(found, key=lambda item: item[0]["step"])
    return directory, meta


def save_resume(run_dir: Path, step: int, trainable, opt_state, n_rows: int) -> int:
    """Write the training state at `step` to <run_dir>/resume, aside first and swapped in once complete. Returns its size in MiB."""
    import jax
    import orbax.checkpoint as ocp
    run_dir = Path(run_dir).resolve()
    new, cur, old = (run_dir / (RESUME_DIR + s) for s in (".new", "", ".old"))
    if new.exists():
        shutil.rmtree(new)
    new.mkdir(parents=True)
    leaves = jax.tree.leaves((trainable, opt_state))
    ocp.PyTreeCheckpointer().save(new / PARAMS_DIR, {f"{i:06d}": leaf for i, leaf in enumerate(leaves)})
    (new / META_FILE).write_text(json.dumps({"step": step, "n_rows": n_rows, "n_leaves": len(leaves)}))
    if old.exists():
        shutil.rmtree(old)
    if cur.exists():
        cur.rename(old)
    new.rename(cur)
    if old.exists():
        shutil.rmtree(old)
    return round(sum(f.stat().st_size for f in cur.rglob("*") if f.is_file()) / 2**20)


def load_resume(directory: Path, meta: Mapping[str, Any], trainable, opt_state, n_rows: int):
    """(trainable, opt_state) of a snapshot, in the structure of the ones given (freshly built from the same model and settings)."""
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    if meta["n_rows"] != n_rows:
        raise ValueError(f"the snapshot was taken with {meta['n_rows']} training rows, there are {n_rows} now: the demonstrations changed")
    saved = ocp.PyTreeCheckpointer().restore(Path(directory).resolve() / PARAMS_DIR)
    leaves, treedef = jax.tree.flatten((trainable, opt_state))
    if len(leaves) != meta["n_leaves"] or len(saved) != len(leaves):
        raise ValueError(f"the snapshot holds {len(saved)} arrays, the training state has {len(leaves)}: another model or optimiser")
    restored = []
    for i, leaf in enumerate(leaves):
        value = saved[f"{i:06d}"]
        if tuple(value.shape) != tuple(leaf.shape):
            raise ValueError(f"array {i}: {tuple(value.shape)} in the snapshot, {tuple(leaf.shape)} in the training state")
        restored.append(jnp.asarray(value).astype(leaf.dtype))
    return jax.tree.unflatten(treedef, restored)


# -- data feed -----------------------------------------------------------------------------------------------------------------------
def row_batches(n_rows: int, batch_size: int, steps: int, seed: int, skip: int = 0) -> Iterator[np.ndarray]:
    """Batches `skip + 1` to `steps` of the sequence of `batch_size` distinct rows: a new random permutation per epoch, the incomplete tail of an epoch is
    dropped. The `skip` first ones are walked over, not given (a resumed run does not prepare the batches it already trained on)."""
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
            if produced > skip:
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


# Rows per forward pass of the evaluation loss: the logits of pi0-FAST are (rows, tokens, 257152 words) in float32, 7.8 GB for 16 rows of 474 tokens, which
# ran out of memory on an L40S (the first probe). The evaluation is a few rows, so small passes cost nothing.
EVAL_MICRO_BATCH = 4


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
    """JAX's view of the first device's memory in MiB (None on the CPU): bytes in use, peak, and the limit of its allocator."""
    import jax
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:                                    # noqa: BLE001
        return None
    if not stats:
        return None
    return {"in_use": round(stats.get("bytes_in_use", 0) / 2**20), "peak": round(stats.get("peak_bytes_in_use", 0) / 2**20),
            "limit": round(stats.get("bytes_limit", 0) / 2**20)}                  # the allocator's cap: XLA_PYTHON_CLIENT_MEM_FRACTION of the card


class MetricsLog:
    """One JSON object per line in <run>/metrics.jsonl, and the same on the standard output. Every line carries `elapsed_s`, the seconds since the log was opened.
    `sink` (tracking.WandbSink) gets each event too; a failure of the sink is reported once and drops it, it never stops the run."""

    def __init__(self, path: Path, sink=None):
        self.file = open(path, "a", buffering=1)
        self.t0 = time.monotonic()
        self.sink = sink

    def write(self, **fields) -> None:
        line = json.dumps({**fields, "elapsed_s": round(time.monotonic() - self.t0, 1)}, default=float)
        self.file.write(line + "\n")
        print(line, flush=True)
        if self.sink is not None:
            try:
                self.sink(json.loads(line))
            except Exception as error:                      # noqa: BLE001
                print(f"[wandb] logging stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                self.sink = None

    def camera_views(self, step: int, images: Mapping[str, Any]) -> None:
        """A few training rows as the model sees them, one picture per row with its cameras side by side (openpi's `camera_views`), to wandb only."""
        if self.sink is None:
            return
        try:
            self.sink.log_camera_views(step, images)
        except Exception as error:                          # noqa: BLE001
            print(f"[wandb] camera_views not logged: {type(error).__name__}: {error}", file=sys.stderr, flush=True)

    def close(self, exit_code: int = 0) -> None:
        self.file.close()
        if self.sink is not None:
            try:
                self.sink.close(exit_code)
            except Exception as error:                      # noqa: BLE001
                print(f"[wandb] closing failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)


# -- the run ---------------------------------------------------------------------------------------------------------------------------
def train(policy, train_set: DemoSet, val_set: Optional[DemoSet], cfg: SFTConfig, run_dir: Path, meta: Optional[Mapping[str, Any]] = None,
          resume: bool = False) -> dict:
    """Run the SFT of `policy` on `train_set`. Writes <run_dir>/config.json, metrics.jsonl, checkpoints/<step>/ and resume/. Returns a summary
    (final loss, checkpoints, time, `stopped` when a SIGTERM ended it early). `policy` ends with the final weights (`set_state`).
    resume=True continues the run of <run_dir> from its newest complete snapshot (ValueError if there is none, or if the settings differ in
    what shapes the optimisation, see FROZEN_ON_RESUME); `policy` must be freshly loaded from the same model YAML.
    A run that fails writes an `error` line (exception, memory) to metrics.jsonl before the exception goes on."""
    if train_set.horizon != policy.info.action_horizon:
        raise ValueError(f"the demonstrations are cut in chunks of {train_set.horizon} steps, the model's action_horizon is {policy.info.action_horizon}")
    run_dir = Path(run_dir)
    meta = {**(meta or {}), "model": policy.info.name, "family": policy.info.family, "action_horizon": policy.info.action_horizon}
    snapshot_found = find_resume(run_dir) if resume else None
    wandb_project = cfg.wandb_project
    if resume:
        if snapshot_found is None:
            raise ValueError(f"{run_dir} has no complete resume snapshot (resume_interval was 0 and the run was not stopped by a SIGTERM?)")
        saved_cfg = json.loads((run_dir / "config.json").read_text())
        wandb_project = saved_cfg["sft"].get("wandb_project", wandb_project)       # the wandb run lives in the project it was created in
        check_resume_compatible(saved_cfg["sft"], cfg, saved_cfg.get("model_cfg"), meta.get("model_cfg"))
        if snapshot_found[1]["step"] >= cfg.steps:
            raise ValueError(f"the run is already at step {snapshot_found[1]['step']} (steps = {cfg.steps}): give a larger `steps` to train further")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"sft": dataclasses.asdict(cfg), **meta}, indent=2, default=str))
    sink = tracking.start_tracking(run_dir, resume, wandb_project, run_dir.name, cfg.wandb_mode, {"sft": dataclasses.asdict(cfg), **meta},
                                   tags=[str(meta.get("model_name", meta["model"])), str(meta.get("mode", ""))])
    log = MetricsLog(run_dir / "metrics.jsonl", sink)
    try:
        return _run(policy, train_set, val_set, cfg, run_dir, meta, resume, snapshot_found, log)
    except BaseException as error:
        # The reason of a failure (an out-of-memory above all) in metrics.jsonl and in wandb, not only in the job's log.
        log.write(event="error", error=f"{type(error).__name__}: {str(error)[:600]}", memory_mib=device_memory_mib())
        log.close(exit_code=1)
        raise


def _run(policy, train_set: DemoSet, val_set: Optional[DemoSet], cfg: SFTConfig, run_dir: Path, meta: Mapping[str, Any], resume: bool, snapshot_found,
         log: MetricsLog) -> dict:
    """The body of `train`, once the run is validated and its log is open."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    original, frozen = split_state(policy)
    trainable = tree_cast(original, jnp.float32, copy=True)
    # The policy's own (bfloat16) copy of the trainable weights is not needed any more: 6 GiB for pi0.5 fine-tuned in full, on a card where every GiB counts.
    # `policy.state` is stale until the first evaluation (or the end of the run) sets the state again.
    jax.block_until_ready(trainable)                        # the copy is done before its source goes
    kept = {id(x) for x in jax.tree.leaves(frozen)}
    for leaf in jax.tree.leaves(original):
        if id(leaf) not in kept and hasattr(leaf, "delete"):
            leaf.delete()
    del original
    n_trainable = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(trainable))
    n_frozen = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(frozen))
    tx, schedule = make_optimizer(cfg)
    opt_state = tx.init(trainable)
    start_step = 0
    if snapshot_found is not None:
        start_step = snapshot_found[1]["step"]
        trainable, opt_state = load_resume(snapshot_found[0], snapshot_found[1], trainable, opt_state, len(train_set))
    n_micro = cfg.batch_size // cfg.micro_batch_size
    log.write(event="start", run_dir=str(run_dir), trainable_m=round(n_trainable / 1e6, 1), frozen_m=round(n_frozen / 1e6, 1),
              train_steps=len(train_set), val_steps=len(val_set) if val_set else 0, batch_size=cfg.batch_size,
              micro_batch_size=cfg.micro_batch_size, memory_mib=device_memory_mib())
    if resume:
        log.write(event="resume", from_step=start_step, snapshot=str(snapshot_found[0]), target_steps=cfg.steps)
    elif log.sink is not None:                              # a fresh run shows what the model gets (the first rows of the run's wandb page, like openpi's)
        picture_rows = np.random.default_rng(cfg.seed).choice(len(train_set), min(5, len(train_set)), replace=False)
        (picture_obs, _), = micro_batches(policy, train_set, picture_rows, len(picture_rows))
        log.camera_views(0, picture_obs.images)

    @jax.jit
    def grad_fn(trainable, frozen, rng, observation, actions):
        loss_of = lambda t: policy.loss(nnx.State.merge(t, frozen), rng, observation, actions, train=True)     # noqa: E731
        return jax.value_and_grad(loss_of)(trainable)

    # Memory of a full fine-tuning (a float32 copy of the gradients is 12.5 GiB for pi0.5): the running sum is donated (it becomes the new sum), and the python
    # names of the gradient trees are dropped as soon as they are used (`del grads`, `total = None` below). The probes showed about TWO gradient copies still allocated
    # after each step, and the evaluation then ran out of memory; only the trees that are donated AND reused in an output are given back, so nothing else is donated.
    @functools.partial(jax.jit, donate_argnums=(0,))
    def add_fn(total, grads):
        return jax.tree.map(jnp.add, total, grads)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def apply_fn(trainable, opt_state, grads):
        grads = jax.tree.map(lambda g: g / n_micro, grads)
        updates, opt_state = tx.update(grads, opt_state, trainable)
        return optax.apply_updates(trainable, updates), opt_state, optax.global_norm(grads)

    # openpi's `param_norm`: the global norm of the kernels of the WHOLE model (not the biases, scales or embeddings), at the logged steps only. The frozen part
    # never changes: its sum of squares is computed once (leaf by leaf, so that no float32 copy of the model exists), the trainable part at each log.
    # A failure here (a filter that was only tested on a tiny model) must not stop a training: the value is then left out.
    import openpi.shared.nnx_utils as nnx_utils
    kernels = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")), lambda _, x: x.value.ndim > 1)

    def kernel_sq(state):
        return sum((jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree.leaves(state.filter(kernels))), jnp.zeros((), jnp.float32))

    trainable_sq_fn = jax.jit(kernel_sq)
    try:
        frozen_sq = float(kernel_sq(frozen))
    except Exception as error:                              # noqa: BLE001
        frozen_sq = None
        print(f"[sft] param_norm not computed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)

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
                      for b in micro_batches(policy, sets[name], rows, min(cfg.micro_batch_size, len(rows), EVAL_MICRO_BATCH))]
            result[name] = {"loss": float(np.mean(losses)), **evaluate_chunks(policy, sets[name], rows)}
        log.write(event="eval", step=step, eval_s=round(time.monotonic() - t_eval, 1), **result)

    def checkpoint(step: int) -> str:
        path = run_dir / "checkpoints" / str(step)
        if path.exists():                                   # saved by the attempt this one resumed: same step, same content
            shutil.rmtree(path)
        t_save = time.monotonic()
        save_checkpoint(path, trainable, {**meta, "step": step, "sft": dataclasses.asdict(cfg)})
        size_mib = round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20)
        log.write(event="checkpoint", step=step, path=str(path), save_s=round(time.monotonic() - t_save, 1), size_mib=size_mib)
        return str(path)

    def snapshot(step: int) -> None:
        t_snap = time.monotonic()
        size_mib = save_resume(run_dir, step, trainable, opt_state, len(train_set))
        log.write(event="resume_snapshot", step=step, save_s=round(time.monotonic() - t_snap, 1), size_mib=size_mib)

    # A SIGTERM (Slurm's warning before the time limit, `--signal=B:TERM@...` forwarded by the job script) ends the run after the current step, with a snapshot.
    stop = threading.Event()
    try:
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    except ValueError:                                      # not the main thread: nothing to catch
        previous_handler = None
    base_key = jax.random.key(cfg.seed)
    feed = prefetch((micro_batches(policy, train_set, rows, cfg.micro_batch_size)
                     for rows in row_batches(len(train_set), cfg.batch_size, cfg.steps, cfg.seed, skip=start_step)), cfg.prefetch)
    t_start, t_window, losses, grad_norms, checkpoints, last_loss = time.monotonic(), time.monotonic(), [], [], [], float("nan")
    stopped, last_snapshot = False, start_step
    for step, micros in enumerate(feed, start=start_step + 1):
        total, step_losses = None, []
        for i, (observation, actions) in enumerate(micros):
            loss, grads = grad_fn(trainable, frozen, jax.random.fold_in(base_key, step * n_micro + i), observation, actions)
            total = grads if total is None else add_fn(total, grads)
            del grads
            step_losses.append(loss)
        trainable, opt_state, grad_norm = apply_fn(trainable, opt_state, total)
        total = None                                        # not kept through the evaluation, the checkpoint and the snapshot
        losses.append(jnp.mean(jnp.stack(step_losses)))
        grad_norms.append(grad_norm)
        if step == start_step + 1:
            jax.block_until_ready(trainable)
            log.write(event="first_step", seconds=round(time.monotonic() - t_start, 1), note="includes the compilation", memory_mib=device_memory_mib())
            t_window = time.monotonic()
        if step % cfg.log_interval == 0 or step == cfg.steps:
            last_loss = float(np.mean(jax.device_get(losses)))
            now = time.monotonic()
            norm = {}
            if frozen_sq is not None:
                try:
                    norm["param_norm"] = math.sqrt(frozen_sq + float(trainable_sq_fn(trainable)))
                except Exception as error:                  # noqa: BLE001
                    frozen_sq = None
                    print(f"[sft] param_norm not computed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            log.write(event="train", step=step, loss=last_loss, grad_norm=float(np.mean(jax.device_get(grad_norms))), **norm, lr=float(schedule(step - 1)),
                      step_s=round((now - t_window) / len(losses), 3), memory_mib=device_memory_mib())
            losses, grad_norms, t_window = [], [], now
        if step % cfg.eval_interval == 0 or step == cfg.steps:
            evaluate(step)
        if step % cfg.save_interval == 0 or step == cfg.steps:
            checkpoints.append(checkpoint(step))
        if cfg.resume_interval and (step % cfg.resume_interval == 0 or step == cfg.steps):
            snapshot(step)
            last_snapshot = step
        if stop.is_set() and step < cfg.steps:
            if last_snapshot != step:
                snapshot(step)
            stopped = True
            feed.close()
            break
    if previous_handler is not None:
        signal.signal(signal.SIGTERM, previous_handler)
    policy.set_state(deployed_state(trainable, frozen))
    summary = {"steps": cfg.steps, "final_loss": last_loss, "checkpoints": checkpoints, "seconds": round(time.monotonic() - t_start, 1),
               "stopped": stopped, "last_step": step if cfg.steps > start_step else start_step}
    log.write(event="stopped" if stopped else "done", **summary)
    log.close()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
