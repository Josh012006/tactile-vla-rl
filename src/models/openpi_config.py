"""LoRA settings of a configs/model/<model>.yaml, applied to an openpi TrainConfig.

YAML keys (all optional):

  use_lora: true            # default true
                            #   true : the Gemma weights are frozen and LoRA matrices are trained, with openpi's own freeze filter
                            #          (Pi0Config.get_freeze_filter: everything outside the Gemma LLMs, e.g. the SigLIP vision tower and
                            #          the action projections, stays trainable: this is openpi's behaviour, kept as is)
                            #   false: no LoRA, everything is trained (full fine-tuning)
  lora_rank:                # defaults = openpi's (gemma_2b_lora: 16, gemma_300m_lora: 32)
    vlm: 16                 # the VLM (PaliGemma's Gemma 2B)
    action_head: 32         # the action expert (Gemma 300M); pi0.5 / pi0 only: pi0-FAST has no action expert

The default ranks give EXACTLY openpi's variants (`gemma_2b_lora`, `gemma_300m_lora`): nothing is patched. For another rank the variant is
named `<base>_lora_r<rank>` and `openpi.models.gemma.get_config` / `gemma_fast.get_config` are wrapped once, in this process, to build it:
same LoRA as openpi (alpha = rank, as in openpi's variants), only the rank changes. The fork is not modified. The same YAML must be used
wherever a checkpoint trained with a given rank is loaded: the rank is part of the model structure.

`parse_lora_settings` needs nothing but the standard library (it is what tests/test_models_pin.py runs on every model YAML).
"""
from __future__ import annotations

import dataclasses
import re
from typing import Optional

# openpi's ranks (openpi/models/gemma.py: gemma_2b_lora, gemma_300m_lora; gemma_fast.py: gemma_2b_lora)
DEFAULT_RANK = {"gemma_2b": 16, "gemma_300m": 32}
FAMILIES_WITH_ACTION_HEAD = ("pi05", "pi0")
_RANKED = re.compile(r"^(gemma_2b|gemma_300m)_lora_r([0-9]+)$")


@dataclasses.dataclass(frozen=True)
class LoraSettings:
    use_lora: bool
    vlm_rank: int
    action_head_rank: Optional[int]          # None: the family has no action expert


def _rank(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer, got {value!r}")
    return value


def parse_lora_settings(model_cfg: dict) -> LoraSettings:
    use_lora = model_cfg.get("use_lora", True)
    if not isinstance(use_lora, bool):
        raise ValueError(f"use_lora must be true or false, got {use_lora!r}")
    ranks = model_cfg.get("lora_rank") or {}
    if not isinstance(ranks, dict):
        raise ValueError("lora_rank must be a mapping with the keys vlm and action_head")
    unknown = set(ranks) - {"vlm", "action_head"}
    if unknown:
        raise ValueError(f"lora_rank: unknown keys {sorted(unknown)} (expected vlm, action_head)")
    has_head = model_cfg.get("family") in FAMILIES_WITH_ACTION_HEAD
    if "action_head" in ranks and not has_head:
        raise ValueError(f"lora_rank.action_head: the family {model_cfg.get('family')!r} has no action expert")
    vlm = _rank(ranks.get("vlm", DEFAULT_RANK["gemma_2b"]), "lora_rank.vlm")
    head = _rank(ranks.get("action_head", DEFAULT_RANK["gemma_300m"]), "lora_rank.action_head") if has_head else None
    return LoraSettings(use_lora=use_lora, vlm_rank=vlm, action_head_rank=head)


def variant_name(base: str, use_lora: bool, rank: int) -> str:
    """`gemma_2b` | `gemma_2b_lora` (openpi's own name when the rank is openpi's) | `gemma_2b_lora_r<rank>`."""
    if not use_lora:
        return base
    return f"{base}_lora" if rank == DEFAULT_RANK[base] else f"{base}_lora_r{rank}"


def _install_ranked_variants() -> None:
    """Make `gemma_*_lora_r<rank>` known to openpi's Gemma configs (idempotent)."""
    import openpi.models.gemma as gemma
    import openpi.models.gemma_fast as gemma_fast

    for module in (gemma, gemma_fast):
        real = module.get_config
        if getattr(real, "_ranked_variants", False):
            continue

        def wrapped(variant, _real=real):
            m = _RANKED.match(variant)
            if m is None:
                return _real(variant)
            cfg = _real(f"{m.group(1)}_lora")
            cfg.lora_configs = {k: v.replace(rank=int(m.group(2)), alpha=float(m.group(2))) for k, v in cfg.lora_configs.items()}
            if hasattr(cfg, "keys") and "variant" in cfg:                       # gemma_fast: the dict is handed to the Module, which knows the base name only
                cfg.variant = f"{m.group(1)}_lora"
            return cfg

        wrapped._ranked_variants = True                # noqa: SLF001
        module.get_config = wrapped


def apply_lora_settings(train_cfg, settings: LoraSettings):
    """The TrainConfig with the Gemma variants chosen by `settings` and the freeze filter openpi derives from them."""
    model = train_cfg.model
    fields = {"paligemma_variant": variant_name("gemma_2b", settings.use_lora, settings.vlm_rank)}
    if hasattr(model, "action_expert_variant"):
        if settings.action_head_rank is None:
            raise ValueError("this model has an action expert but the YAML family says it has none")
        fields["action_expert_variant"] = variant_name("gemma_300m", settings.use_lora, settings.action_head_rank)
    if any(_RANKED.match(v) for v in fields.values()):
        _install_ranked_variants()
    model = dataclasses.replace(model, **fields)
    return dataclasses.replace(train_cfg, model=model, freeze_filter=model.get_freeze_filter())
