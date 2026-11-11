"""The interface between the algorithms (src/algo, later) and a VLA model. Pure numpy + yaml: no JAX, no openpi, so the algorithms,
the rollout code and their tests do not need the `vla` dependency group.

    policy = load_model("pi05")                                   # src/models/__init__.py: an OpenPIPolicy (needs the vla group)
    res = env.reset(seed=0, round=0)                              # VectorEnvClient: res.obs = the env's observation dict, (N, ...) arrays
    plan = policy.plan(res.obs, env.spec.prompt)                  # Plan: targets (N, 1, H, 8) = what to send to env.step, one chunk per env
    actions = plan.targets[:, 0, t]                               # (N, 8): 7 ABSOLUTE joint-position targets + gripper command

What the model receives is the env's observation dict itself (src/envs/protocol.py keys), N environments at once. The model reads the
keys it needs and ignores the others: the images, the joint state and the gripper today. `extra/tactile` and the force keys are in the
dict and are ignored until a touch config tells a model to read them (configs/touch, later): the same env stream then feeds the
baseline and the tactile runs, only the config differs. Adding a touch input means extending `to_openpi_obs` (what the model is
given) and the openpi input transform (openpi's DroidInputs builds the state from joints + gripper only): nothing in the algorithms.

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
import dataclasses
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from src.envs import protocol as P

ROOT = Path(__file__).resolve().parents[2]
SEMANTICS = ("offsets_from_chunk_start_state", "absolute_targets")


# -- model YAML ---------------------------------------------------------------------------------------------------------------
def load_model_cfg(name: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "model" / f"{name}.yaml").read_text())
    if cfg["output_semantics"] not in SEMANTICS:
        raise ValueError(f"configs/model/{name}.yaml: output_semantics must be one of {SEMANTICS}, got {cfg['output_semantics']!r}")
    return cfg


def apply_overrides(cfg: dict, use_lora: Optional[bool] = None, lora_rank_vlm: Optional[int] = None,
                    lora_rank_action_head: Optional[int] = None) -> dict:
    """A copy of the model config with the LoRA keys overridden (command-line flags of the probe, the algorithms' own configs)."""
    cfg = dict(cfg)
    if use_lora is not None:
        cfg["use_lora"] = use_lora
    for key, val in (("vlm", lora_rank_vlm), ("action_head", lora_rank_action_head)):
        if val is not None:
            cfg["lora_rank"] = {**(cfg.get("lora_rank") or {}), key: val}
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


def rows_of(obs: Dict[str, np.ndarray], prompt: Union[str, Sequence[str]]) -> List[dict]:
    """One model observation per environment. `prompt`: one string for all the environments, or one per environment."""
    n = num_envs_of(obs)
    prompts = [prompt] * n if isinstance(prompt, str) else list(prompt)
    if len(prompts) != n:
        raise ValueError(f"{len(prompts)} prompts for {n} environments")
    return [to_openpi_obs(obs, e, prompts[e]) for e in range(n)]


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

    @abc.abstractmethod
    def raw_chunks(self, rows: List[dict], num_samples: int) -> Tuple[np.ndarray, float]:
        """`rows`: one openpi-format observation per environment (`to_openpi_obs`). Returns the model's own chunks (N, S, H, D >= 8)
        in the order [environment][sample], and the model time in ms."""

    def seed(self, seed: int) -> None:
        """Restart the policy's random stream (sampling noise). Same seed + same inputs = same chunks. Default: nothing to seed."""

    def plan(self, obs: Dict[str, np.ndarray], prompt: Union[str, Sequence[str]], num_samples: int = 1) -> Plan:
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        rows = rows_of(obs, prompt)
        n, h = len(rows), self.info.action_horizon
        raw, infer_ms = self.raw_chunks(rows, num_samples)
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

    def raw_chunks(self, rows, num_samples):
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

    def raw_chunks(self, rows, num_samples):
        h, offsets = self.info.action_horizon, self.info.output_semantics == "offsets_from_chunk_start_state"
        out = np.zeros((len(rows), num_samples, h, P.ACTION_DIM), np.float32)
        for i, row in enumerate(rows):
            if not offsets:
                out[i, :, :, :7] = row["observation/joint_position"][:7]
            out[i, :, :, 7] = float(np.asarray(row["observation/gripper_position"]).reshape(-1)[0])
        return out, 0.0
