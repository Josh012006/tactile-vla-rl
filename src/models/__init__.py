"""The VLA models (pi0.5, pi0-FAST) as openpi provides them (third_party/openpi), behind one interface (src/models/interface.py).

    from src.models import load_model
    policy = load_model("pi05")                                          # configs/model/pi05.yaml, needs the `vla` dependency group
    policy = load_model("pi05", overrides={"vlm.use_lora": False})       # keys of the YAML overridden: <component>.<key>
    plan = policy.plan(env_obs, prompt)                                  # chunks in the env's action space

What is in this package is what makes a model work in the architecture: the interface, the YAML settings, the openpi adapter and its
loading. Measuring and debugging tools are elsewhere: src/eval (rollouts, progress, videos) and scripts/ (probe_vla, compare_reports,
fetch_models).

Importing this package does not import JAX.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Union

OPENPI_FAMILIES = ("pi05", "pi0", "pi0_fast")


def load_model(model: Union[str, dict], *, seed: int = 0, overrides: Optional[Mapping[str, Any]] = None):
    """`model`: the name of a configs/model/<name>.yaml, or an already loaded config dict. `overrides`: keys `<component>.<key>` of the
    YAML to change for this load (`vlm.use_lora`, `action_head.lora_rank`, `action_head.use_tactile`, ...). Returns a `VLAPolicy`
    (src/models/interface.py)."""
    from src.models.interface import apply_overrides, load_model_cfg
    cfg = apply_overrides(load_model_cfg(model) if isinstance(model, str) else dict(model), overrides)
    family = cfg["family"]
    if family in OPENPI_FAMILIES:
        from src.models.openpi_policy import OpenPIPolicy
        return OpenPIPolicy.load(cfg, seed=seed)
    raise ValueError(f"unknown model family {family!r} (known: {OPENPI_FAMILIES})")
