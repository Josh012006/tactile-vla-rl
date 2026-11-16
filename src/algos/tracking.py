"""Weights & Biases logging of a training run, next to its metrics.jsonl (which stays the reference: nothing here may stop a run).

Modelled on expo-ft's `init_wandb` (and openpi's): the id of the wandb run is written to <run>/wandb_id.txt when the run starts, and a resumed run
(`--resume`) re-attaches to THAT run with `resume="must"`, so one training is one wandb run, however many times it was stopped and continued.

The events of metrics.jsonl are forwarded as they are written (WandbSink): `train/*`, `eval/<train|val>/*`, `checkpoint/*`, `snapshot/*`, `perf/*`, and, like openpi's SFT, `camera_views` (5 training rows as the model sees them) at the start of a fresh run.
Every point carries our own `step` and the x axis of every metric is `step` (`define_metric`), not wandb's internal counter: a resumed run goes back to
the step of its snapshot (some steps after it were already logged by the attempt that died), and wandb's own step may only increase, it drops the points
that go back. The steps that were run twice therefore appear twice on the curve; with a SIGTERM stop (snapshot at the stop step) there are none.

The events of the RL runs (src/algos/rl_run.py) have their own view, `rl_metrics`: grouped by what they measure, without constants or copies, the critic's windows on the number of critic updates (docs/algos/rl.md section 4).

Modes: "online", "offline" (written to <run>/wandb/, uploaded later with `wandb sync <run>/wandb/offline-run-*`), "disabled". An online run that cannot
start (no login, no network on the node) falls back to offline, with a warning; a package that is not installed disables the logging. wandb is
imported only when it is used.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

WANDB_ID_FILE = "wandb_id.txt"
MODES = ("online", "offline", "disabled")
# metrics.jsonl event -> prefix of its wandb keys (the fields that are not numbers are left out)
PREFIX = {"train": "train", "eval": "eval", "checkpoint": "checkpoint", "resume_snapshot": "snapshot", "first_step": "perf", "warmup": "warmup", "round": "round", "distill": "distill", "refresh": "refresh"}      # the last four: the RL runs (src/algos/rl_run.py)
RL_AXES = {"critic/*": "critic_update"}         # the RL runs: the critic's windows are plotted against the number of critic updates
NOT_LOGGED = {"event", "step", "elapsed_s", "from_step"}


def _warn(message: str) -> None:
    print(f"[wandb] {message}", file=sys.stderr, flush=True)


def flatten(prefix: str, value: Any, out: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """{'a': {'b': 1, 'c': 'text'}, 'd': 2.5} -> {'<prefix>/a/b': 1, '<prefix>/d': 2.5}: nested dictionaries are walked, only numbers are kept."""
    out = {} if out is None else out
    if isinstance(value, Mapping):
        for key, item in value.items():
            flatten(f"{prefix}/{key}", item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value
    return out


def _se(rate: float, n: Any) -> Optional[float]:
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / n) if n else None


def rl_metrics(event: str, fields: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """The wandb view of the events of an RL run (src/algos/rl_run.py), or None for an event that is not one of theirs (the SFT's events keep the generic view above).
    metrics.jsonl keeps every field, as the reference; wandb gets the ones that say something, grouped by what they measure, each on its natural axis:
        eval/*        the fixed evaluation episodes (success_rate +- success_se, lengths), eval/opt/* the critic's view of the states visited by the evaluation   x = step (the round)
        rollout/*     the round's own rollouts (the ones the critic is trained on), rollout/opt/* the same critic statistics                                      x = step
        warmup/*      the base VLA's episodes of the warm-up (the baseline to compare the curves to)                                                          x = step 0
        distill/*     PA-RL's distillation: its labels (distill/labels/*), loss, gradient, size                                                                  x = step
        refresh/*, buffer/*, time/*                                                                                                                              x = step
        critic/*      the critic's losses and Q statistics, the mean of every window of CRITIC_LOG_EVERY updates                                                   x = critic_update (the number of updates)
    Left out of wandb on purpose: constants (episode counts, round ids), the loop's own step counter, three copies of the same wall time, the per-collection lines of the warm-up
    (their sum is the warm-up line) and the critic's mean over a whole round (the windows are the finer curve)."""
    m: Dict[str, float] = {}
    get = fields.get
    if event == "critic":
        flatten("critic", {k: v for k, v in fields.items() if k not in ("event", "stage", "update", "seconds_per_update")}, m)
        if "seconds_per_update" in fields:
            m["time/critic_update_s"] = fields["seconds_per_update"]
        m["critic_update"] = fields["update"]
        return m
    if event == "warmup":
        if not fields.get("summary"):
            return {}
        m.update({"warmup/success_rate": fields["success_rate"], "warmup/env_steps": fields["env_steps"], "warmup/buffer_rows": fields["buffer_rows"], "time/warmup_s": fields["wall_s"]})
        if _se(fields["success_rate"], fields["episodes"]) is not None:
            m["warmup/success_se"] = _se(fields["success_rate"], fields["episodes"])
        return m
    if event == "round":
        m.update({"rollout/success_rate": fields["success_rate"], "rollout/online_success_rate_cumulative": fields["online_success_rate"], "rollout/plans_per_episode": fields["plans_mean"],
                  "rollout/env_steps": fields["env_steps"], "buffer/rows": fields["buffer_rows"], "buffer/online_episodes": fields["online_episodes"],
                  "time/round_s": fields["round_wall_s"], "time/collect_s": fields["wall_s"], "time/model_s": fields["model_s"]})
        if _se(fields["success_rate"], get("episodes")) is not None:
            m["rollout/success_se"] = _se(fields["success_rate"], get("episodes"))
        for key, value in fields.items():
            if key.startswith("opt_"):
                m[f"rollout/opt/{key[4:]}"] = value
        return m
    if event == "eval" and "success_rate" in fields:                       # an RL evaluation (the SFT's carry `train` / `val` dictionaries)
        for key in ("success_rate", "success_se", "steps_mean", "steps_mean_success"):
            if key in fields:
                m[f"eval/{key}"] = fields[key]
        flatten("eval/opt", get("optimisation", {}), m)
        if "eval_wall_s" in fields:
            m["time/eval_s"] = fields["eval_wall_s"]
        return m
    if event == "distill":
        for key, value in fields.items():
            if key in NOT_LOGGED or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if key.startswith("label_"):
                m[f"distill/labels/{key[6:]}"] = value
            elif key in ("annotate_s", "train_s", "step_s"):
                m[f"time/distill_{key}"] = value
            else:
                m[f"distill/{key}"] = value
        return m
    if event == "refresh":
        m.update({"refresh/states": fields["states"], "time/refresh_s": fields["seconds"], "time/refresh_model_s": fields["model_ms"] / 1e3})
        return m
    return None


def read_run_id(run_dir: Path) -> Optional[str]:
    path = Path(run_dir) / WANDB_ID_FILE
    return path.read_text().strip() or None if path.exists() else None


def init_wandb(run_dir: Path, resuming: bool, project: str, name: str, mode: str = "online", config: Optional[Mapping[str, Any]] = None,
               tags: Optional[Sequence[str]] = None, axes: Optional[Mapping[str, str]] = None):
    """Start (or, with resuming=True, re-attach to) the wandb run of `run_dir`; returns it, or None when nothing is logged.
    `axes`: {"prefix*": "metric used as the x axis of the keys that start with it", ...}, for the keys that are not on `step` (the critic's windows are on the number of updates).
    The run goes to <run_dir>/wandb/ and its id to <run_dir>/wandb_id.txt. A resumed run is a run that already has an id there: without one (the first
    attempt had no wandb) a new run starts."""
    if mode not in MODES:
        raise ValueError(f"wandb_mode must be one of {MODES}, got {mode!r}")
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        _warn("the package is not installed (uv sync --group vla): no wandb logging, metrics.jsonl only")
        return None
    run_dir = Path(run_dir)
    run_id = read_run_id(run_dir) if resuming else None
    if resuming and run_id is None:
        _warn(f"{run_dir / WANDB_ID_FILE} not found: this run was not logged to wandb before, a new wandb run starts")
    kwargs: Dict[str, Any] = {"project": project, "name": name, "dir": str(run_dir), "tags": [t[:64] for t in (tags or []) if t]}      # wandb refuses empty or longer than 64 characters tags
    if run_id is not None:
        kwargs.update(id=run_id, resume="must")          # the same run: an error rather than a new run if it does not exist
    run = None
    for attempt in ([mode] if mode == "offline" else [mode, "offline"]):
        try:
            run = wandb.init(mode=attempt, **kwargs)
            break
        except Exception as error:                       # noqa: BLE001  (no login, no network, run unknown to the server...)
            _warn(f"wandb.init(mode={attempt!r}) failed: {type(error).__name__}: {error}")
            try:
                wandb.finish(exit_code=1)
            except Exception:                            # noqa: BLE001
                pass
            if attempt == "online":
                _warn("falling back to offline mode: `wandb sync` the run later")
    if run is None:
        _warn("no wandb logging for this run, metrics.jsonl only")
        return None
    if run_id is None:
        (run_dir / WANDB_ID_FILE).write_text(run.id)
    if config is not None:
        run.config.update(json.loads(json.dumps(config, default=str)), allow_val_change=resuming)
    run.define_metric("step")
    for pattern, axis in (axes or {}).items():                  # BEFORE the "*" below: wandb applies the first pattern that matches a key, in the order of definition
        run.define_metric(axis)
        run.define_metric(pattern, step_metric=axis)
    run.define_metric("*", step_metric="step")
    return run


class WandbSink:
    """Receives the dictionaries MetricsLog writes and logs them. `step` is the last step seen (the point of the events that have none)."""

    def __init__(self, run):
        self.run = run
        self.step = 0

    def __call__(self, fields: Mapping[str, Any]) -> None:
        event = fields["event"]
        if event == "start":
            self.run.summary.update({k: fields[k] for k in ("trainable_m", "frozen_m", "train_steps", "val_steps") if k in fields})
        elif event in ("done", "stopped"):
            self.run.summary.update({k: fields[k] for k in ("final_loss", "seconds", "stopped", "last_step") if k in fields})
        elif event == "resume":
            self.step = int(fields["from_step"])
            self.run.log({"step": self.step, "resume/from_step": self.step})
        elif (view := rl_metrics(event, fields)) is not None:         # an event of an RL run: its own view (rl_metrics)
            if event == "critic":                                   # on its own axis (the number of updates): no `step`
                self.run.log(view)
            else:
                step = int(fields["step"])
                self.step = step
                if view:
                    self.run.log({"step": step, **view})
        elif event in PREFIX:
            step = int(fields["step"]) if "step" in fields else self.step + 1       # first_step: the first step of this attempt is done
            self.step = step
            metrics: Dict[str, float] = {}
            for key, value in fields.items():
                if key not in NOT_LOGGED:
                    flatten(f"{PREFIX[event]}/{key}", value, metrics)
            if metrics:
                self.run.log({"step": step, **metrics})

    def log_camera_views(self, step: int, images: Mapping[str, Any], count: int = 5) -> None:
        """openpi's `camera_views`: for the first `count` rows, one picture with the cameras side by side. `images`: name -> (rows, H, W, 3), uint8 or in [-1, 1]."""
        import numpy as np
        import wandb

        def to_uint8(x):
            x = np.asarray(x)
            return x if x.dtype == np.uint8 else np.clip((x.astype(np.float32) + 1.0) * 127.5, 0, 255).astype(np.uint8)

        names = list(images)
        rows = min(count, len(images[names[0]]))
        pictures = [wandb.Image(np.concatenate([to_uint8(images[name][i]) for name in names], axis=1)) for i in range(rows)]
        self.run.log({"step": step, "camera_views": pictures})

    def close(self, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)


def start_tracking(run_dir: Path, resuming: bool, project: str, name: str, mode: str, config: Optional[Mapping[str, Any]] = None,
                   tags: Optional[Sequence[str]] = None, axes: Optional[Mapping[str, str]] = None) -> Optional[WandbSink]:
    """init_wandb wrapped as the sink of a MetricsLog (None when nothing is logged)."""
    run = init_wandb(run_dir, resuming, project, name, mode, config, tags, axes)
    return None if run is None else WandbSink(run)
