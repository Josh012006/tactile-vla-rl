"""Which touch signals a component of the system sees, and the vector built from them. numpy only.

The same three switches, in the YAML block of every component that may use touch (configs/model/<model>.yaml: `vlm`, `action_head`;
the algorithm configs, later: `residual`, `critic`):

    use_tactile   bool                    the taxels
    use_force     bool                    the force signals
    force_mode    "measured" | "proxy"    where the force comes from (only meaningful with use_force)

All off (the default) = the task alone: nothing here is built, stored or fed to anything. They are the switches of expo-ft's AuxSpec
(expo_ft/utils/aux_features.py); what changes is that every component has its own, so that ablations can give touch to the action
head only, to the residual and the critic only, to several, or to none.

Signals (keys of the env observation, src/envs/protocol.py):

    use_force, force_mode "measured":
        extra/ft_force         (3,)       wrist force sensor, N
        extra/gripper_effort   scalar     PD torque demanded of the Robotiq finger joint, N.m (Robotiq robots only)
    use_tactile:
        extra/tactile          (2, 7, 4)  net contact force of each taxel zone, N (robot "robotiq_2f85_tsf" only)
    use_force, force_mode "proxy":
        routed but NOT implemented (the force is estimated instead of read, arXiv 2610.23849): NotImplementedError when the spec is
        created, before anything is built or any GPU time is spent.

Layout of the vector, only the enabled parts present, in this order:   [ft_force (3) | gripper_effort (1) | tactile flattened in C order (56)]
    dim = 4 * use_force + 56 * use_tactile

Scaling: a single parameter-free map applied to every entry, sign(x) * log1p(|x|) (expo-ft's): it compresses the large forces of a hard
contact without a clip or a hand-tuned scale, keeps the sign and maps 0 to 0.

A missing key raises: a zero vector would be a silent, wrong "no contact". The ways to get here without the signals are a robot
without taxels / gripper effort (the server only sends `extra/tactile` for `robotiq_2f85_tsf` and `extra/gripper_effort` for Robotiq
robots) and demonstrations recorded without them.

What a component does with the vector (the injection) is not decided here: src/touch/injection/.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from src.envs import protocol as P

FT_FORCE_SHAPE = (3,)
GRIPPER_EFFORT_SHAPE = (1,)
TACTILE_SHAPE = (2, 7, 4)
FORCE_DIM = int(np.prod(FT_FORCE_SHAPE)) + int(np.prod(GRIPPER_EFFORT_SHAPE))      # 3 + 1
TACTILE_DIM = int(np.prod(TACTILE_SHAPE))                                          # 2 * 7 * 4

FORCE_MODES = ("measured", "proxy")
IMPLEMENTED_FORCE_MODES = ("measured",)
FIELDS = ("use_tactile", "use_force", "force_mode")                                # the keys of a component's YAML block


@dataclasses.dataclass(frozen=True)
class TouchSpec:
    """Which touch signals one component sees. Immutable and hashable."""

    use_tactile: bool = False
    use_force: bool = False
    force_mode: str = "measured"
    where: str = "touch"                    # the YAML path, for error messages (not part of the comparison)

    def __post_init__(self):
        for name in ("use_tactile", "use_force"):
            if not isinstance(getattr(self, name), bool):
                # a YAML string such as "false" would be truthy: refuse instead of silently turning a signal on
                raise ValueError(f"{self.where}.{name} must be true or false, got {getattr(self, name)!r}")
        if self.force_mode not in FORCE_MODES:
            raise ValueError(f"{self.where}.force_mode must be one of {FORCE_MODES}, got {self.force_mode!r}")
        if not self.use_force and self.force_mode != "measured":
            raise ValueError(f"{self.where}.force_mode: {self.force_mode!r} has no effect while use_force is false: set use_force: true, or remove it")
        if self.use_force and self.force_mode not in IMPLEMENTED_FORCE_MODES:
            raise NotImplementedError(f"{self.where}.force_mode: {self.force_mode!r} is routed but not implemented yet (the force proxy, "
                                      f"arXiv 2610.23849). Implemented: {IMPLEMENTED_FORCE_MODES}.")

    def __eq__(self, other):
        return isinstance(other, TouchSpec) and (self.use_tactile, self.use_force, self.force_mode) == (other.use_tactile, other.use_force, other.force_mode)

    def __hash__(self):
        return hash((self.use_tactile, self.use_force, self.force_mode))

    @classmethod
    def from_mapping(cls, block: Optional[Mapping], where: str) -> "TouchSpec":
        """From the touch keys of a YAML block (absent = off). `where` names the block in errors: "action_head", "residual", ..."""
        block = block or {}
        return cls(use_tactile=block.get("use_tactile", False), use_force=block.get("use_force", False),
                   force_mode=block.get("force_mode", "measured"), where=where)

    # -- description --------------------------------------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.use_tactile or self.use_force

    @property
    def dim(self) -> int:
        return (FORCE_DIM if self.use_force else 0) + (TACTILE_DIM if self.use_tactile else 0)

    @property
    def keys(self) -> Tuple[str, ...]:
        """Observation keys that `build_batch` reads, in layout order."""
        return ((P.FT_FORCE, P.GRIPPER_EFFORT) if self.use_force else ()) + ((P.TACTILE,) if self.use_tactile else ())

    def describe(self) -> str:
        if not self.enabled:
            return "none (task only)"
        parts = []
        if self.use_force:
            parts.append(f"force[{self.force_mode}] (ft_force 3 + gripper_effort 1)")
        if self.use_tactile:
            parts.append("tactile (2x7x4)")
        return " + ".join(parts) + f" = {self.dim} values"

    # -- data -----------------------------------------------------------------------------------------------------------------
    def build_batch(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """The env observation (N environments) -> (N, dim) float32, see the module docstring."""
        if not self.enabled:
            raise RuntimeError(f"{self.where}: build called while use_tactile and use_force are both false")
        missing = [k for k in self.keys if k not in obs]
        if missing:
            raise KeyError(f"{self.where}: {self.describe()} is enabled but the observation has no {missing}. Robot 'robotiq_2f85_tsf' is "
                           f"required for extra/tactile and a Robotiq robot for extra/gripper_effort; demonstrations must have been "
                           f"recorded with them.")
        n = int(np.asarray(obs[P.JOINT_POSITION]).shape[0])
        parts = []
        if self.use_force:
            force = np.asarray(obs[P.FT_FORCE], dtype=np.float32)
            effort = np.asarray(obs[P.GRIPPER_EFFORT], dtype=np.float32).reshape(n, -1)
            _check(P.FT_FORCE, force, (n,) + FT_FORCE_SHAPE)
            _check(P.GRIPPER_EFFORT, effort, (n,) + GRIPPER_EFFORT_SHAPE)
            parts += [force, effort]
        if self.use_tactile:
            tactile = np.asarray(obs[P.TACTILE], dtype=np.float32)
            _check(P.TACTILE, tactile, (n,) + TACTILE_SHAPE)
            parts.append(tactile.reshape(n, -1))
        raw = np.concatenate(parts, axis=1)
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"{self.where}: non-finite value in the tactile / force observation")
        return (np.sign(raw) * np.log1p(np.abs(raw))).astype(np.float32)


def _check(name: str, value: np.ndarray, shape: Tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")


def required_keys(specs: Mapping[str, TouchSpec]) -> Tuple[str, ...]:
    """Observation keys needed by all the specs (each once, in order): what a data collector or a replay buffer must store."""
    out = []
    for spec in specs.values():
        out += [k for k in spec.keys if k not in out]
    return tuple(out)
