"""The configuration of the RL fine-tuning (Best-of-N and PA-RL): configs/algo/rl.yaml and the experiment files of configs/experiment/rl/. Pure Python (no JAX): it can be read, checked and
tested anywhere. Design, sources and reasons of every value: docs/algos/rl.md.

    cfg = load_rl_config(overrides={"num_candidates": 16})                          # configs/algo/rl.yaml with `--set` style overrides
    exp, cfg = load_rl_experiment("configs/experiment/rl/pi05_best_of_n.yaml")      # an experiment: its `set` block on top of the YAML (and the command line's on top of that)

An unknown key, a wrong type or a combination the method cannot do (Best-of-N with a local step, a total that is not a whole number of batches) is an error, never ignored.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "algo" / "rl.yaml"
METHODS = ("best_of_n", "parl")
SELECTIONS = ("softmax", "argmax")
CAMERAS = ("exterior", "wrist")
WANDB_MODES = ("online", "offline", "disabled")        # src/algos/tracking.py MODES, repeated here to stay free of the training imports (a test keeps them equal)
EXPERIMENT_KEYS = {"name", "description", "model", "set", "model_set", "start"}


@dataclasses.dataclass(frozen=True)
class RLConfig:
    """configs/algo/rl.yaml (every key documented there; the defaults are the paper's, Appendix B.1)."""
    method: str = "parl"
    num_candidates: int = 32
    keep_top: int = 10
    local_steps: int = 10
    local_lr: float = 3e-4
    rollout_selection: str = "softmax"
    distill_selection: str = "softmax"
    eval_selection: str = "argmax"
    softmax_temperature: float = 1.0
    envs_per_batch: int = 50
    total_episodes: int = 1000
    replan_steps: int = 0
    rows_per_sample_call: int = 8
    critic_cameras: Tuple[str, ...] = ("exterior", "wrist")
    critic_image_size: int = 128
    random_shift_px: int = 4
    critic_hidden: Tuple[int, ...] = (512, 512, 512)
    critic_lr: float = 3e-4
    critic_batch_size: int = 256
    utd: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    step_reward: float = -1.0
    cal_ql_alpha: float = 0.01
    mixing_ratio: float = 0.5
    hl_gauss: bool = False
    use_touch: bool = False
    warmup_episodes: int = 50
    warmup_critic_steps: int = 5000
    distill_lr: float = 2.5e-5
    distill_batch_size: int = 64
    distill_micro_batch_size: int = 16
    distill_epochs: int = 1
    eval_episodes: int = 100
    seed: int = 0
    wandb_mode: str = "online"
    wandb_project: str = "tactile-vla-rl"
    output_root: str = "runs/rl/probing"

    def __post_init__(self):
        def need(ok: bool, message: str):
            if not ok:
                raise ValueError(message)

        need(self.method in METHODS, f"method must be one of {METHODS}, got {self.method!r}")
        for name in ("rollout_selection", "distill_selection", "eval_selection"):
            need(getattr(self, name) in SELECTIONS, f"{name} must be one of {SELECTIONS}, got {getattr(self, name)!r}")
        need(self.softmax_temperature > 0, "softmax_temperature must be > 0")
        positive = ("num_candidates", "keep_top", "envs_per_batch", "total_episodes", "rows_per_sample_call", "critic_image_size", "critic_batch_size", "utd",
                    "distill_batch_size", "distill_micro_batch_size", "distill_epochs", "eval_episodes")
        for name in positive:
            need(isinstance(getattr(self, name), int) and getattr(self, name) >= 1, f"{name} must be an integer >= 1, got {getattr(self, name)!r}")
        need(self.keep_top <= self.num_candidates, f"keep_top ({self.keep_top}) cannot exceed num_candidates ({self.num_candidates})")
        need(self.local_steps >= 0 and self.replan_steps >= 0 and self.random_shift_px >= 0 and self.warmup_episodes >= 0 and self.warmup_critic_steps >= 0,
             "local_steps, replan_steps, random_shift_px, warmup_episodes and warmup_critic_steps must be >= 0")
        need(self.local_lr > 0 and self.critic_lr > 0 and self.distill_lr > 0, "local_lr, critic_lr and distill_lr must be > 0")
        need(0 < self.gamma <= 1 and 0 < self.tau <= 1, "gamma and tau must be in (0, 1]")
        need(0 <= self.mixing_ratio < 1, "mixing_ratio must be in [0, 1)")
        need(self.cal_ql_alpha >= 0, "cal_ql_alpha must be >= 0")
        need(len(self.critic_cameras) >= 1 and set(self.critic_cameras) <= set(CAMERAS) and len(set(self.critic_cameras)) == len(self.critic_cameras),
             f"critic_cameras: a non-empty list without repeats from {CAMERAS}, got {list(self.critic_cameras)}")
        need(len(self.critic_hidden) >= 1 and all(isinstance(h, int) and h >= 1 for h in self.critic_hidden), "critic_hidden: a non-empty list of integers >= 1")
        need(self.total_episodes % self.envs_per_batch == 0, f"total_episodes ({self.total_episodes}) must be a multiple of envs_per_batch ({self.envs_per_batch}): the batches are synchronous")
        need(self.warmup_episodes % self.envs_per_batch == 0, f"warmup_episodes ({self.warmup_episodes}) must be a multiple of envs_per_batch ({self.envs_per_batch})")
        need(self.distill_batch_size % self.distill_micro_batch_size == 0,
             f"distill_batch_size ({self.distill_batch_size}) must be a multiple of distill_micro_batch_size ({self.distill_micro_batch_size})")
        need(self.wandb_mode in WANDB_MODES, f"wandb_mode must be one of {WANDB_MODES}, got {self.wandb_mode!r}")
        if self.method == "best_of_n":
            need(self.keep_top == 1, "best_of_n takes the argmax of the k candidates: keep_top must be 1")
            need(self.local_steps == 0, "best_of_n has no local optimisation: local_steps must be 0")
            need(self.rollout_selection == "argmax", "best_of_n executes the argmax: rollout_selection must be argmax")

    @property
    def variant(self) -> str:
        """best_of_n, parl_global (variant A) or parl_global_local (variant B)."""
        return self.method if self.method == "best_of_n" else ("parl_global_local" if self.local_steps > 0 else "parl_global")

    @property
    def distills(self) -> bool:
        """Only PA-RL changes the VLA."""
        return self.method == "parl"

    @property
    def rounds(self) -> int:
        """Collection rounds of the run: batches of envs_per_batch episodes."""
        return self.total_episodes // self.envs_per_batch


def config_from_dict(raw: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> RLConfig:
    """An RLConfig from a dict of its keys (the YAML, or the `rl` block of a run's config.json) and overrides. An unknown key is an error."""
    raw = dict(raw)
    raw.update(overrides or {})
    fields = {f.name for f in dataclasses.fields(RLConfig)}
    unknown = sorted(set(raw) - fields)
    if unknown:
        raise ValueError(f"unknown RL keys {unknown}; known: {sorted(fields)}")
    for name in ("critic_cameras", "critic_hidden"):
        if name in raw:
            if isinstance(raw[name], (str, bytes)) or not isinstance(raw[name], (list, tuple)):
                raise ValueError(f"{name} is a list, got {raw[name]!r}")
            raw[name] = tuple(raw[name])
    return RLConfig(**raw)


def load_rl_config(path: Optional[Path] = None, overrides: Optional[Mapping[str, Any]] = None) -> RLConfig:
    """configs/algo/rl.yaml with `overrides` (the CLI's `--set key=value`) applied."""
    return config_from_dict(yaml.safe_load(Path(path or CONFIG_PATH).read_text()) or {}, overrides)


def load_experiment(path) -> dict:
    """An experiment file (configs/experiment/rl/*.yaml): `name` (the run's folder), `model`, `description`, `set` (keys of configs/algo/rl.yaml), `model_set` (keys of the model YAML) and,
    once the run is launched from one, `start` ({run, step}: the SFT checkpoint the VLA starts from). Unknown keys and wrong types are refused."""
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise ValueError(f"no experiment file {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: an experiment is a mapping")
    unknown = sorted(set(raw) - EXPERIMENT_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown} (allowed: {sorted(EXPERIMENT_KEYS)})")
    for key in ("name", "model"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise ValueError(f"{path}: `{key}` is required (a string)")
    for key in ("set", "model_set"):
        if not isinstance(raw.get(key) or {}, dict):
            raise ValueError(f"{path}: `{key}` is a mapping")
    start = raw.get("start")
    if start is not None:
        if not isinstance(start, dict) or set(start) - {"run", "step"} or not isinstance(start.get("run"), str) or not isinstance(start.get("step"), int):
            raise ValueError(f"{path}: `start` is {{run: <SFT run folder>, step: <checkpoint step>}}")
    return raw


def load_rl_experiment(path, overrides: Optional[Mapping[str, Any]] = None) -> Tuple[dict, RLConfig]:
    """(the experiment's dict, its RLConfig): configs/algo/rl.yaml, then the experiment's `set`, then `overrides` (the command line's, which win)."""
    exp = load_experiment(path)
    return exp, load_rl_config(overrides={**(exp.get("set") or {}), **(overrides or {})})
