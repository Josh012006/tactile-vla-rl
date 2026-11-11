"""The per-component settings of a configs/model/<model>.yaml. Standard library + numpy only (no JAX, no openpi): this is what
tests/test_models_pin.py parses on every model YAML, and what the algorithms read to know which observation keys a model needs.

A VLA model is made of components, each with its own block of the YAML:

    vlm:                        # the vision-language model (PaliGemma's Gemma 2B + SigLIP)
      use_lora: true            #   LoRA on the Gemma weights                                   default true
      lora_rank: 16             #   openpi's default for gemma_2b_lora                          default 16
      use_tactile: false        #   touch given to this component (src/touch/features.py)       default false
      use_force: false          #   ...
      force_mode: measured      #   ...
    action_head:                # the action expert (Gemma 300M); pi0.5 / pi0 only: pi0-FAST has none, the key is refused there
      use_lora: true            #                                                               default true
      lora_rank: 32             #   openpi's default for gemma_300m_lora                        default 32
      use_tactile: false
      use_force: false
      force_mode: measured

Every key is optional; an absent block is all defaults. A new component (a residual network, a critic) will be one more block with the
same keys, read by the algorithm config that owns it (configs/algo/...), through `TouchSpec.from_mapping`.

LoRA, as openpi has it (nothing re-implemented):
    use_lora true : the component's Gemma weights are frozen and LoRA matrices are trained, with openpi's own freeze filter
                    (Pi0Config.get_freeze_filter: the SigLIP vision tower and the action projections stay trainable, it is openpi's behaviour).
                    The two components are independent: LoRA on the VLM only, on the action head only, on both, or on none.
    use_lora false: the component is trained in full (or left as it is when nothing is trained).
    lora_rank     : the default ranks give EXACTLY openpi's variants (`gemma_2b_lora`, `gemma_300m_lora`). Another rank is a new variant
                    (src/models/openpi_config.py) and changes the model structure: use the same value wherever a checkpoint trained with it
                    is loaded.

Touch: the switches are parsed and validated here and the features reach the model's input rows (src/models/interface.py), but WHERE
they enter a component is decided in src/touch/injection/: a component that asks for touch before an injection exists for it raises
NotImplementedError when the model is built, it is never silently ignored.

The settings of an older YAML (a top-level `use_lora` / `lora_rank: {vlm, action_head}`) are refused with a message that says what to write.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Mapping, Optional, Tuple

from src.touch.features import FIELDS as TOUCH_FIELDS
from src.touch.features import TouchSpec, required_keys

# openpi's ranks (openpi/models/gemma.py: gemma_2b_lora, gemma_300m_lora; gemma_fast.py: gemma_2b_lora)
DEFAULT_RANK = {"gemma_2b": 16, "gemma_300m": 32}
FAMILIES_WITH_ACTION_HEAD = ("pi05", "pi0")
COMPONENTS = ("vlm", "action_head")
_BASE_OF = {"vlm": "gemma_2b", "action_head": "gemma_300m"}
BLOCK_FIELDS = ("use_lora", "lora_rank") + TOUCH_FIELDS


@dataclasses.dataclass(frozen=True)
class ComponentSettings:
    use_lora: bool
    lora_rank: int
    touch: TouchSpec


@dataclasses.dataclass(frozen=True)
class ModelSettings:
    vlm: ComponentSettings
    action_head: Optional[ComponentSettings]          # None: the family has no action expert

    @property
    def components(self) -> Dict[str, ComponentSettings]:
        return {"vlm": self.vlm, **({"action_head": self.action_head} if self.action_head is not None else {})}

    @property
    def any_lora(self) -> bool:
        return any(c.use_lora for c in self.components.values())

    @property
    def touch_specs(self) -> Dict[str, TouchSpec]:
        """Component name -> TouchSpec, for ALL the components (a disabled spec has `enabled == False`)."""
        return {name: c.touch for name, c in self.components.items()}

    @property
    def active_touch(self) -> Dict[str, TouchSpec]:
        return {name: spec for name, spec in self.touch_specs.items() if spec.enabled}

    @property
    def required_obs_keys(self) -> Tuple[str, ...]:
        """Env observation keys beyond images / joints / gripper that this model reads: what the data must carry."""
        return required_keys(self.active_touch)


def _rank(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where} must be a positive integer, got {value!r}")
    return value


def _parse_block(model_cfg: Mapping, name: str) -> ComponentSettings:
    block = model_cfg.get(name) or {}
    if not isinstance(block, Mapping):
        raise ValueError(f"{name}: must be a mapping with the keys {list(BLOCK_FIELDS)}, got {block!r}")
    unknown = set(block) - set(BLOCK_FIELDS)
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)} (expected {list(BLOCK_FIELDS)})")
    use_lora = block.get("use_lora", True)
    if not isinstance(use_lora, bool):
        raise ValueError(f"{name}.use_lora must be true or false, got {use_lora!r}")
    rank = _rank(block.get("lora_rank", DEFAULT_RANK[_BASE_OF[name]]), f"{name}.lora_rank")
    return ComponentSettings(use_lora=use_lora, lora_rank=rank, touch=TouchSpec.from_mapping(block, name))


def parse_model_settings(model_cfg: Mapping) -> ModelSettings:
    """The `vlm` / `action_head` blocks of a model YAML (already loaded), validated. Raises ValueError (NotImplementedError for a force
    mode that is routed but not implemented) at the first problem, before anything heavy is built."""
    legacy = [k for k in ("use_lora", "lora_rank") if k in model_cfg]
    if legacy:
        raise ValueError(f"{legacy}: the model YAML now has one block per component. Write\n"
                         f"  vlm:          {{use_lora: true, lora_rank: 16}}\n  action_head:  {{use_lora: true, lora_rank: 32}}   (pi0.5 / pi0 only)")
    has_head = model_cfg.get("family") in FAMILIES_WITH_ACTION_HEAD
    if "action_head" in model_cfg and not has_head:
        raise ValueError(f"action_head: the family {model_cfg.get('family')!r} has no action expert")
    return ModelSettings(vlm=_parse_block(model_cfg, "vlm"), action_head=_parse_block(model_cfg, "action_head") if has_head else None)


def variant_name(base: str, use_lora: bool, rank: int) -> str:
    """`gemma_2b` | `gemma_2b_lora` (openpi's own name when the rank is openpi's) | `gemma_2b_lora_r<rank>`."""
    if not use_lora:
        return base
    return f"{base}_lora" if rank == DEFAULT_RANK[base] else f"{base}_lora_r{rank}"


def component_variant(settings: ModelSettings, component: str) -> str:
    """openpi's Gemma variant name for a component of `settings`."""
    c = settings.components[component]
    return variant_name(_BASE_OF[component], c.use_lora, c.lora_rank)


def describe(settings: ModelSettings) -> str:
    """One line per component, for logs and reports."""
    lines = []
    for name, c in settings.components.items():
        lora = f"LoRA rank {c.lora_rank}" if c.use_lora else "no LoRA"
        lines.append(f"{name}: {lora}, touch: {c.touch.describe()}")
    return "; ".join(lines)
