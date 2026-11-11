"""The VLA models (pi0.5, pi0-FAST) as openpi provides them (third_party/openpi), behind one interface (src/models/interface.py).

    from src.models import load_model
    policy = load_model("pi05")                          # configs/model/pi05.yaml, needs the `vla` dependency group
    policy = load_model("pi05", use_lora=False)          # LoRA keys of the YAML overridden
    plan = policy.plan(env_obs, prompt)                  # chunks in the env's action space

Importing this package does not import JAX.
"""
from __future__ import annotations

from typing import Optional, Union

OPENPI_FAMILIES = ("pi05", "pi0", "pi0_fast")


def load_model(model: Union[str, dict], *, seed: int = 0, use_lora: Optional[bool] = None, lora_rank_vlm: Optional[int] = None,
               lora_rank_action_head: Optional[int] = None):
    """`model`: the name of a configs/model/<name>.yaml, or an already loaded config dict. The keyword arguments override the LoRA
    keys of the YAML. Returns a `VLAPolicy` (src/models/interface.py)."""
    from src.models.interface import apply_overrides, load_model_cfg
    cfg = apply_overrides(load_model_cfg(model) if isinstance(model, str) else dict(model), use_lora, lora_rank_vlm, lora_rank_action_head)
    family = cfg["family"]
    if family in OPENPI_FAMILIES:
        from src.models.openpi_policy import OpenPIPolicy
        return OpenPIPolicy.load(cfg, seed=seed)
    raise ValueError(f"unknown model family {family!r} (known: {OPENPI_FAMILIES})")
