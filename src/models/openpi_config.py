"""The LoRA settings of a model YAML (src/models/settings.py) applied to an openpi TrainConfig.

The default ranks give EXACTLY openpi's variants (`gemma_2b_lora`, `gemma_300m_lora`): nothing is patched. For another rank the variant is
named `<base>_lora_r<rank>` and `openpi.models.gemma.get_config` / `gemma_fast.get_config` are wrapped once, in this process, to build it:
same LoRA as openpi (alpha = rank, as in openpi's variants), only the rank changes. The fork is not modified.

The freeze filter is openpi's own (`get_freeze_filter` of the model config: Pi0Config for pi0.5 / pi0, Pi0FASTConfig for pi0-FAST). For pi0.5
it handles the components independently (LoRA on the VLM only, on the action expert only, on both); for pi0-FAST it freezes the Gemma
weights that are not LoRA when the variant is a LoRA one, and nothing otherwise.

Needs openpi (the `vla` dependency group), imported lazily: `import src.models.openpi_config` alone does not import it.
"""
from __future__ import annotations

import dataclasses
import re

from src.models.settings import ModelSettings, component_variant

_RANKED = re.compile(r"^(gemma_2b|gemma_300m)_lora_r([0-9]+)$")


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


def apply_settings(train_cfg, settings: ModelSettings):
    """The TrainConfig with the Gemma variants chosen by `settings` and the freeze filter openpi derives from them."""
    model = train_cfg.model
    fields = {"paligemma_variant": component_variant(settings, "vlm")}
    if hasattr(model, "action_expert_variant"):
        if settings.action_head is None:
            raise ValueError("this model has an action expert but the YAML family says it has none")
        fields["action_expert_variant"] = component_variant(settings, "action_head")
    elif settings.action_head is not None:
        raise ValueError("the YAML family has an action expert but the openpi model has none")
    if any(_RANKED.match(v) for v in fields.values()):
        _install_ranked_variants()
    model = dataclasses.replace(model, **fields)
    return dataclasses.replace(train_cfg, model=model, freeze_filter=model.get_freeze_filter())
