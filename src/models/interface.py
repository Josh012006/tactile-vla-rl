"""The interface between the algorithms (src/algo, later) and a VLA model. Pure numpy + yaml: no JAX, no openpi, so the algorithms,
the rollout code and their tests do not need the `vla` dependency group.

    policy = load_model("pi05")                                   # src/models/__init__.py: an OpenPIPolicy (needs the vla group)
    res = env.reset(seed=0, round=0)                              # VectorEnvClient: res.obs = the env's observation dict, (N, ...) arrays
    plan = policy.plan(res.obs, env.spec.prompt)                  # Plan: targets (N, 1, H, 8) = what to send to env.step, one chunk per env
    actions = plan.targets[:, 0, t]                               # (N, 8): 7 ABSOLUTE joint-position targets + gripper command

What the model receives is the env's observation dict itself (src/envs/protocol.py keys), N environments at once. The model reads
the keys it needs and ignores the others: the images, the joint state and the gripper. Touch (`extra/tactile`, `extra/ft_force`,
`extra/gripper_effort`) is in the dict and stays out of the model unless the model YAML switches it on for a component (`vlm.use_tactile`,
`action_head.use_force`, ...: src/models/settings.py). When it does, `rows_of` adds the component's touch vector to every row
(`touch/<component>`, built by src/touch/features.py), and how the component uses it is src/touch/injection/. So the same env stream
feeds the baseline and the tactile runs, only the YAML differs, and the algorithms do not change.

What comes out is always in the ENV's action space (absolute joint targets + gripper), whatever the model returns natively: the
YAML's `output_semantics` says how to read the model's chunk (`chunk_to_targets`), and the inverse (`targets_to_chunk`) turns
demonstrations or targets of the env into the model's own space for training.

`Plan.targets` has a sample axis: `plan(obs, prompt, num_samples=S)` draws S independent chunks per environment (flow matching:
independent noise; the prefix is computed once). pi0-FAST decodes the most likely tokens unless its sampling temperature is set,
so its S samples are identical by default.

A policy is one of:
    OpenPIPolicy    src/models/openpi_policy.py: pi0.5 / pi0 / pi0-FAST through openpi, batched, with the training hooks.
    CallablePolicy  any openpi-style `infer(obs_dict) -> {"actions": ...}` function (a websocket client, a test double).
    StaticPolicy    holds the arm where it is (dry runs of an algorithm without a model).
"""
from __future__ import annotations

import abc
import copy
import dataclasses
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from src.envs import protocol as P
from src.models.settings import BLOCK_FIELDS, COMPONENTS, parse_model_settings
from src.touch.features import TouchSpec

ROOT = Path(__file__).resolve().parents[2]
SEMANTICS = ("offsets_from_chunk_start_state", "absolute_targets")
NORM_STATS_FIELDS = ("dir", "asset_id")
TOUCH_ROW_PREFIX = "touch/"           # a row carries the touch vector of component C under "touch/C" (only for components that use touch)


# -- model YAML ---------------------------------------------------------------------------------------------------------------
def norm_stats_source(cfg: Mapping) -> Optional[Tuple[Path, str]]:
    """(folder, asset id) of the normalisation statistics the YAML's `norm_stats` block names: <folder>/<asset id>/norm_stats.json, the folder
    relative to the repository root unless absolute. None (no block): the statistics saved with the checkpoint (DROID's), as openpi has it.
    The block holds both keys or it is refused. Does not check that the file exists (src/models/loading.py does, where it is read)."""
    block = cfg.get("norm_stats")
    if block is None:
        return None
    if not isinstance(block, Mapping) or set(block) != set(NORM_STATS_FIELDS) or not all(isinstance(v, str) and v for v in block.values()):
        raise ValueError(f"norm_stats: a mapping with the two keys {list(NORM_STATS_FIELDS)} (non-empty strings), or nothing for the checkpoint's own, got {block!r}")
    folder = Path(block["dir"])
    return (folder if folder.is_absolute() else ROOT / folder), block["asset_id"]


def load_model_cfg(name: str) -> dict:
    """configs/model/<name>.yaml, validated (output semantics, per-component settings, normalisation statistics)."""
    cfg = yaml.safe_load((ROOT / "configs" / "model" / f"{name}.yaml").read_text())
    if cfg["output_semantics"] not in SEMANTICS:
        raise ValueError(f"configs/model/{name}.yaml: output_semantics must be one of {SEMANTICS}, got {cfg['output_semantics']!r}")
    parse_model_settings(cfg)
    norm_stats_source(cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: Optional[Mapping[str, Any]] = None) -> dict:
    """A validated copy of the model config with `overrides` applied: keys `<component>.<key>`, e.g.
    {"vlm.use_lora": False, "action_head.lora_rank": 8, "action_head.use_tactile": True} (the probe's `--set`, an algorithm's own config).
    The normalisation statistics: {"norm_stats.dir": "assets/...", "norm_stats.asset_id": "..."}, or {"norm_stats": None} for the checkpoint's own."""
    cfg = copy.deepcopy(dict(cfg))
    for key, value in (overrides or {}).items():
        component, _, field = key.partition(".")
        if component == "norm_stats":
            if not field:
                cfg["norm_stats"] = value
                continue
            if field not in NORM_STATS_FIELDS:
                raise ValueError(f"override {key!r}: the keys of norm_stats are {list(NORM_STATS_FIELDS)}")
            cfg["norm_stats"] = {**(cfg.get("norm_stats") or {}), field: value}
            continue
        if component not in COMPONENTS or field not in BLOCK_FIELDS:
            raise ValueError(f"override {key!r}: expected <component>.<key>, component in {list(COMPONENTS)}, key in {list(BLOCK_FIELDS)}")
        cfg[component] = {**(cfg.get(component) or {}), field: value}
    parse_model_settings(cfg)
    norm_stats_source(cfg)
    return cfg


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """What an algorithm needs to know about a model (from its YAML)."""
    name: str
    family: str
    action_horizon: int
    action_dim: int
    output_semantics: str

    @staticmethod
    def from_cfg(cfg: dict) -> "ModelInfo":
        if cfg["output_semantics"] not in SEMANTICS:
            raise ValueError(f"output_semantics must be one of {SEMANTICS}, got {cfg['output_semantics']!r}")
        horizon = int(cfg["action_horizon"])
        if horizon < 1:
            raise ValueError(f"action_horizon must be >= 1, got {horizon}")
        dim = int(cfg.get("action_dim", P.ACTION_DIM))
        if dim != P.ACTION_DIM:
            raise ValueError(f"action_dim is {dim}: the env's actions have {P.ACTION_DIM} dimensions (7 joints + gripper)")
        return ModelInfo(name=str(cfg.get("name", "model")), family=str(cfg.get("family", "unknown")), action_horizon=horizon,
                         action_dim=dim, output_semantics=cfg["output_semantics"])


# -- observation in, chunk out (pure numpy) -------------------------------------------------------------------------------------
def to_openpi_obs(obs: Dict[str, np.ndarray], env: int, prompt: str) -> dict:
    """The observation of environment `env` in the format openpi's DROID policies read (droid_policy.DroidInputs): the five keys it
    needs, the prompt included. The env keys already are DROID's (images 224 x 224 uint8, 7 joints, 1 gripper)."""
    return {
        "observation/exterior_image_1_left": obs[P.EXTERIOR_IMAGE][env],
        "observation/wrist_image_left": obs[P.WRIST_IMAGE][env],
        "observation/joint_position": np.asarray(obs[P.JOINT_POSITION][env], dtype=np.float32),
        "observation/gripper_position": np.asarray(obs[P.GRIPPER_POSITION][env], dtype=np.float32).reshape(-1),
        "prompt": prompt,
    }


def num_envs_of(obs: Dict[str, np.ndarray]) -> int:
    return int(np.asarray(obs[P.JOINT_POSITION]).shape[0])


def rows_of(obs: Dict[str, np.ndarray], prompt: Union[str, Sequence[str]], touch: Optional[Mapping[str, TouchSpec]] = None) -> List[dict]:
    """One model observation per environment. `prompt`: one string for all the environments, or one per environment.
    `touch`: component name -> TouchSpec; each ENABLED spec adds its vector to every row under `touch/<component>` (KeyError when the
    observation lacks a signal it needs, see src/touch/features.py). Without it, or with all specs off, a row is the task alone."""
    n = num_envs_of(obs)
    prompts = [prompt] * n if isinstance(prompt, str) else list(prompt)
    if len(prompts) != n:
        raise ValueError(f"{len(prompts)} prompts for {n} environments")
    rows = [to_openpi_obs(obs, e, prompts[e]) for e in range(n)]
    for name, spec in (touch or {}).items():
        if spec.enabled:
            features = spec.build_batch(obs)
            for e, row in enumerate(rows):
                row[TOUCH_ROW_PREFIX + name] = features[e]
    return rows


def take(obs: Dict[str, np.ndarray], idx: Sequence[int]) -> Dict[str, np.ndarray]:
    """The observation of the environments `idx` only (an algorithm plans for the environments that need a new chunk)."""
    idx = np.asarray(idx, dtype=np.int64)
    return {k: np.asarray(v)[idx] for k, v in obs.items()}


def chunk_to_targets(chunk: np.ndarray, q_plan: np.ndarray, semantics: str) -> np.ndarray:
    """Model output (H, >= 8) -> env actions (H, 8): 7 ABSOLUTE joint targets + gripper.
    offsets_from_chunk_start_state: the 7 joints are offsets from the joint state at the moment of the plan (q_plan): add it back once.
    absolute_targets: the model already returns absolute targets. The gripper (column 7) is passed through."""
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] < P.ACTION_DIM:
        raise ValueError(f"expected a chunk (H, >= {P.ACTION_DIM}), got {chunk.shape}")
    out = np.zeros((len(chunk), P.ACTION_DIM), np.float32)
    out[:, :7] = chunk[:, :7]
    if semantics == "offsets_from_chunk_start_state":
        out[:, :7] += np.asarray(q_plan, dtype=np.float32)[:7]
    elif semantics != "absolute_targets":
        raise ValueError(semantics)
    out[:, 7] = chunk[:, 7]
    return out


def targets_to_chunk(targets: np.ndarray, q_plan: np.ndarray, semantics: str) -> np.ndarray:
    """The inverse of `chunk_to_targets`: env actions (H, 8) (a demonstration, an executed chunk) -> what the model is trained to output
    (H, 8). Offsets: the joint state at the moment of the plan is subtracted; this is exactly openpi's DeltaActions (the delta config
    used for SFT in expo-ft), done here so that the YAML's `output_semantics` is the only place that says it."""
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != P.ACTION_DIM:
        raise ValueError(f"expected targets (H, {P.ACTION_DIM}), got {targets.shape}")
    out = targets.copy()
    if semantics == "offsets_from_chunk_start_state":
        out[:, :7] -= np.asarray(q_plan, dtype=np.float32)[:7]
    elif semantics != "absolute_targets":
        raise ValueError(semantics)
    return out


def padded_batch_size(n: int) -> int:
    """The batch size a jitted model is called with for `n` rows: the next power of two. The environments that need a new chunk vary
    from one step to the next, and every new batch shape costs a compilation (tens of seconds): padding bounds them to log2(N)."""
    if n < 1:
        raise ValueError("empty batch")
    return 1 << math.ceil(math.log2(n))


# -- the interface -------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Plan:
    """What `VLAPolicy.plan` returns."""
    targets: np.ndarray       # (N, S, H, 8) float32: the chunks in the env's action space (absolute joint targets + gripper)
    raw: np.ndarray           # (N, S, H, D >= 8) float32: what the model returned (offsets or absolute, see ModelInfo.output_semantics)
    infer_ms: float           # model time of the whole call (device time included); NaN when the model does not report it


class VLAPolicy(abc.ABC):
    """Batched policy over the env's observation dict. Subclasses implement `raw_chunks`; the conversion to env actions is here."""

    info: ModelInfo
    touch: Mapping[str, TouchSpec] = {}        # component -> TouchSpec of the components that use touch (set by the subclass from its YAML)

    @abc.abstractmethod
    def raw_chunks(self, rows: List[dict], num_samples: int, row_seeds: Optional[Sequence[int]] = None, pad_to: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """`rows`: one openpi-format observation per environment (`rows_of`: `to_openpi_obs` + the touch vectors). Returns the model's own chunks (N, S, H, D >= 8)
        in the order [environment][sample], and the model time in ms.
        `row_seeds` (one int per row): the sampling noise of row i is a function of row_seeds[i] ONLY, not of the other rows, of the batch size or of the policy's own stream (an
        evaluation gives each episode its own noise this way). A model with no sampling noise (pi0-FAST decodes greedily) ignores it.
        `pad_to`: call the model with exactly this many rows (the rows are padded by repeating the last one) instead of the next power of two: one shape whatever the number of rows."""

    def seed(self, seed: int) -> None:
        """Restart the policy's random stream (sampling noise). Same seed + same inputs = same chunks. Default: nothing to seed."""

    def plan(self, obs: Dict[str, np.ndarray], prompt: Union[str, Sequence[str]], num_samples: int = 1, *,
             row_seeds: Optional[Sequence[int]] = None, pad_to: Optional[int] = None) -> Plan:
        """Chunks for every row of `obs`. `row_seeds` / `pad_to`: see `raw_chunks`; a policy that does not take them is not called with them unless they are given."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        rows = rows_of(obs, prompt, self.touch)
        n, h = len(rows), self.info.action_horizon
        if row_seeds is not None and len(row_seeds) != n:
            raise ValueError(f"row_seeds has {len(row_seeds)} entries for {n} rows")
        extra = {k: v for k, v in (("row_seeds", row_seeds), ("pad_to", pad_to)) if v is not None}
        raw, infer_ms = self.raw_chunks(rows, num_samples, **extra)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 4 or raw.shape[:2] != (n, num_samples) or raw.shape[3] < P.ACTION_DIM:
            raise ValueError(f"{type(self).__name__}.raw_chunks returned {raw.shape}, expected ({n}, {num_samples}, H, >= {P.ACTION_DIM})")
        if raw.shape[2] != h:
            raise ValueError(f"the model returned chunks of {raw.shape[2]} steps but its YAML says action_horizon: {h}")
        q = np.asarray(obs[P.JOINT_POSITION], dtype=np.float32)
        targets = np.stack([[chunk_to_targets(raw[i, s], q[i], self.info.output_semantics) for s in range(num_samples)] for i in range(n)])
        return Plan(targets=targets, raw=raw, infer_ms=float(infer_ms))

    def act(self, obs: Dict[str, np.ndarray], prompt: Union[str, Sequence[str]]) -> np.ndarray:
        """One chunk per environment: (N, H, 8) absolute targets."""
        return self.plan(obs, prompt, 1).targets[:, 0]

    def sample(self, obs: Dict[str, np.ndarray], prompt: Union[str, Sequence[str]], num_samples: int) -> np.ndarray:
        """`num_samples` independent chunks per environment: (N, S, H, 8)."""
        return self.plan(obs, prompt, num_samples).targets


class CallablePolicy(VLAPolicy):
    """Wraps an openpi-style function `infer(obs_dict) -> {"actions": (H, >= 8), "policy_timing": {"infer_ms": ...}}`, one call per row
    and per sample (a websocket client of an openpi policy server, a test double)."""

    def __init__(self, infer: Callable[[dict], dict], cfg: dict):
        self._infer = infer
        self.info = ModelInfo.from_cfg(cfg)

    def raw_chunks(self, rows, num_samples, row_seeds=None, pad_to=None):
        out, ms = [], 0.0
        for row in rows:
            samples = []
            for _ in range(num_samples):
                res = self._infer(dict(row))
                samples.append(np.asarray(res["actions"], dtype=np.float32))
                ms += float(res.get("policy_timing", {}).get("infer_ms", float("nan")))
            out.append(np.stack(samples))
        return np.stack(out), ms


class StaticPolicy(VLAPolicy):
    """Holds every environment where it is: the joint targets are the current joints, the gripper command its current position.
    For dry runs of an algorithm and of the env code without a model."""

    def __init__(self, action_horizon: int = 16, output_semantics: str = "absolute_targets"):
        self.info = ModelInfo.from_cfg({"name": "static", "family": "static", "action_horizon": action_horizon,
                                        "output_semantics": output_semantics})

    def raw_chunks(self, rows, num_samples, row_seeds=None, pad_to=None):
        h, offsets = self.info.action_horizon, self.info.output_semantics == "offsets_from_chunk_start_state"
        out = np.zeros((len(rows), num_samples, h, P.ACTION_DIM), np.float32)
        for i, row in enumerate(rows):
            if not offsets:
                out[i, :, :, :7] = row["observation/joint_position"][:7]
            out[i, :, :, 7] = float(np.asarray(row["observation/gripper_position"]).reshape(-1)[0])
        return out, 0.0
