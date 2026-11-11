"""Where and how touch enters a component: one module per method, registered here. NONE is written yet.

A component whose YAML block says `use_tactile: true` / `use_force: true` (configs/model/<model>.yaml: `vlm`, `action_head`) gets its
touch vector (src/touch/features.py) with every observation row (src/models/interface.py, `rows_of`): the data path is there. What is
missing is the injection itself, which has to be decided (extra tokens for the VLM, extra input of the action expert, a residual
network, the critic ...). Until a method is registered for a component, creating a model that asks for touch there stops with an
error that says so (`build_injection`), instead of silently ignoring the switch.

To add a method:
    1. write src/touch/injection/<method>.py with a factory `build(spec: TouchSpec) -> injection object`;
    2. register it: INJECTIONS["action_head"] = build   (the component names are the YAML block names);
    3. make the model apply it (src/models/openpi_policy.py: the features are in the rows, the injection is built in __init__).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from src.touch.features import TouchSpec

INJECTIONS: Dict[str, Callable[[TouchSpec], object]] = {}          # component name -> factory


def build_injection(component: str, spec: TouchSpec) -> Optional[object]:
    """None when the component does not use touch; otherwise the injection registered for it."""
    if not spec.enabled:
        return None
    factory = INJECTIONS.get(component)
    if factory is None:
        raise NotImplementedError(
            f"{component}: touch is switched on ({spec.describe()}) but no injection is written for this component yet "
            f"(src/touch/injection/__init__.py: INJECTIONS is empty). The switches are parsed and the features are built; "
            f"set use_tactile / use_force to false, or register an injection.")
    return factory(spec)
