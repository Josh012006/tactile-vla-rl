"""Weights & Biases logging of a training run, next to its metrics.jsonl (which stays the reference: nothing here may stop a run).

Modelled on expo-ft's `init_wandb` (and openpi's): the id of the wandb run is written to <run>/wandb_id.txt when the run starts, and a resumed run
(`--resume`) re-attaches to THAT run with `resume="must"`, so one training is one wandb run, however many times it was stopped and continued.

The events of metrics.jsonl are forwarded as they are written (WandbSink): `train/*`, `eval/<train|val>/*`, `checkpoint/*`, `snapshot/*`, `perf/*`.
Every point carries our own `step` and the x axis of every metric is `step` (`define_metric`), not wandb's internal counter: a resumed run goes back to
the step of its snapshot (some steps after it were already logged by the attempt that died), and wandb's own step may only increase, it drops the points
that go back. The steps that were run twice therefore appear twice on the curve; with a SIGTERM stop (snapshot at the stop step) there are none.

Modes: "online", "offline" (written to <run>/wandb/, uploaded later with `wandb sync <run>/wandb/offline-run-*`), "disabled". An online run that cannot
start (no login, no network on the node) falls back to offline, with a warning; a package that is not installed disables the logging. wandb is
imported only when it is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

WANDB_ID_FILE = "wandb_id.txt"
MODES = ("online", "offline", "disabled")
# metrics.jsonl event -> prefix of its wandb keys (the fields that are not numbers are left out)
PREFIX = {"train": "train", "eval": "eval", "checkpoint": "checkpoint", "resume_snapshot": "snapshot", "first_step": "perf"}
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


def read_run_id(run_dir: Path) -> Optional[str]:
    path = Path(run_dir) / WANDB_ID_FILE
    return path.read_text().strip() or None if path.exists() else None


def init_wandb(run_dir: Path, resuming: bool, project: str, name: str, mode: str = "online", config: Optional[Mapping[str, Any]] = None,
               tags: Optional[Sequence[str]] = None):
    """Start (or, with resuming=True, re-attach to) the wandb run of `run_dir`; returns it, or None when nothing is logged.
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
        elif event in PREFIX:
            step = int(fields["step"]) if "step" in fields else self.step + 1       # first_step: the first step of this attempt is done
            self.step = step
            metrics: Dict[str, float] = {}
            for key, value in fields.items():
                if key not in NOT_LOGGED:
                    flatten(f"{PREFIX[event]}/{key}", value, metrics)
            if metrics:
                self.run.log({"step": step, **metrics})

    def close(self, exit_code: int = 0) -> None:
        self.run.finish(exit_code=exit_code)


def start_tracking(run_dir: Path, resuming: bool, project: str, name: str, mode: str, config: Optional[Mapping[str, Any]] = None,
                   tags: Optional[Sequence[str]] = None) -> Optional[WandbSink]:
    """init_wandb wrapped as the sink of a MetricsLog (None when nothing is logged)."""
    run = init_wandb(run_dir, resuming, project, name, mode, config, tags)
    return None if run is None else WandbSink(run)
