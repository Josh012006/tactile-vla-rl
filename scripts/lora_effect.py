"""Is the LoRA of a model really in its forward pass? One load, a few seconds of compute.

    python -m scripts.lora_effect --model pi0_fast --obs-file reports/obs_<tag>.npz
    python -m scripts.lora_effect --model pi05

Why: pi0-FAST decodes its tokens greedily, so a small LoRA perturbation can leave the first chunk of an observation unchanged (probe job of
2026-11-11: identical chunks with and without LoRA). A chunk that does not move does not say whether the LoRA is connected. The training loss
does, and it is what this script looks at, on ONE batch, with the weights as loaded:

  * loss with the LoRA as initialised (openpi: lora_a and lora_b ~ normal(0.01));
  * loss with every `lora_b` set to zero: the LoRA then adds exactly nothing, this is the base model;
  * the gradient of the loss with respect to the LoRA parameters, in both situations.

Reading: the LoRA is in the forward pass when the two losses differ and the gradient of `lora_b` is non-zero (it is non-zero only if the
LoRA branch is connected to the loss). With lora_b = 0 the gradient of `lora_a` must be exactly zero (the branch is multiplied by lora_b):
a non-zero value there would mean the LoRA is not the usual A.B product. The same loss path (`OpenPIPolicy.training_batch` / `loss`) is
the one SFT will use, so this also runs it once on the real model. The target chunk is "hold the arm where it is": any valid chunk does,
only the comparison between the two losses matters.

One model per process, like the probe. Output: printed and reports/lora_effect_<model>_<time>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.envs import protocol as P
from src.models.interface import ROOT, apply_overrides, load_model_cfg
from src.models.settings import describe as describe_settings
from src.models.settings import parse_model_settings


def _is_lora(path) -> bool:
    return any("lora" in str(part) for part in path)


def measure(policy, obs: Dict[str, np.ndarray], prompt: str, seed: int = 0) -> dict:
    """Losses and LoRA gradient norms of `policy` (an OpenPIPolicy, N = 1 or more environments) on `obs`, see the module docstring."""
    import flax.traverse_util as tu
    import jax
    import jax.numpy as jnp

    n, h = int(np.asarray(obs[P.JOINT_POSITION]).shape[0]), policy.info.action_horizon
    hold = np.concatenate([np.asarray(obs[P.JOINT_POSITION], np.float32).reshape(n, 7), np.asarray(obs[P.GRIPPER_POSITION], np.float32).reshape(n, 1)], axis=1)
    target = np.repeat(hold[:, None, :], h, axis=1)                                      # (N, H, 8): hold the arm
    observation, actions = policy.training_batch(obs, prompt, target)

    state = policy.state
    lora = {k: v for k, v in tu.flatten_dict(state.to_pure_dict()).items() if _is_lora(k)}
    if not lora:
        raise ValueError("this model has no LoRA parameters (use_lora is false in every component): nothing to measure")
    zero_b = {k: (jnp.zeros_like(v) if "lora_b" in str(k[-1]) else v) for k, v in lora.items()}
    if all(zero_b[k] is v for k, v in lora.items()):
        raise ValueError("no `lora_b` parameter found among the LoRA parameters")
    key = jax.random.key(seed)

    def loss_of(lora_params, state_):
        new = jax.tree.map(lambda x: x, state_)                                            # a copy: the policy's own state is not touched
        new.replace_by_pure_dict(tu.unflatten_dict({**tu.flatten_dict(new.to_pure_dict()), **lora_params}))
        return policy.loss(new, key, observation, actions, train=False)                   # train=False: no augmentation, comparable losses

    value_and_grad = jax.jit(jax.value_and_grad(loss_of))

    def run(params):
        value, grads = value_and_grad(params, state)
        norm = lambda part: float(jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for k, g in grads.items() if part in str(k[-1]))))      # noqa: E731
        return {"loss": float(value), "grad_norm_lora_a": norm("lora_a"), "grad_norm_lora_b": norm("lora_b")}

    full, base = run(lora), run(zero_b)
    out = {"n_lora_tensors": len(lora), "with_lora_as_initialised": full, "lora_b_zeroed_base_model": base,
           "loss_difference": full["loss"] - base["loss"]}
    finite = all(np.isfinite(v) for part in (full, base) for v in part.values())
    connected = finite and full["grad_norm_lora_b"] > 0 and full["loss"] != base["loss"]
    out["lora_a_gradient_is_zero_when_b_is_zero"] = bool(base["grad_norm_lora_a"] == 0.0)
    out["verdict"] = ("LoRA is in the forward pass" if connected else "LoRA does NOT change the loss or has no gradient: not connected") \
        if finite else "non-finite value: see the numbers"
    return out


def main(argv: Optional[list] = None) -> int:
    from scripts.probe_vla import env_obs_of, load_obs, parse_sets, synthetic_obs

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="configs/model/<model>.yaml: pi05 | pi0_fast")
    ap.add_argument("--obs-file", default=None, help="an observation saved by scripts/probe_vla.py (reports/obs_<tag>.npz); default: a synthetic one")
    ap.add_argument("--set", action="append", default=[], metavar="COMPONENT.KEY=VALUE", help="override a key of the model YAML, as in probe_vla")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    a = ap.parse_args(argv)

    cfg = apply_overrides(load_model_cfg(a.model), parse_sets(a.set))
    settings = parse_model_settings(cfg)
    if not settings.any_lora:
        ap.error("no LoRA in this configuration (use_lora is false in every component): nothing to measure")
    row = load_obs(a.obs_file) if a.obs_file else synthetic_obs("Insert the peg into the socket.")

    from src.models.openpi_policy import OpenPIPolicy
    t0 = time.monotonic()
    policy = OpenPIPolicy.load(cfg, seed=a.seed)
    load_s = round(time.monotonic() - t0, 1)
    t0 = time.monotonic()
    result = {"model": a.model, "settings_text": describe_settings(settings), "observation": a.obs_file or "synthetic", "load_s": load_s,
              **measure(policy, env_obs_of(row), row["prompt"], seed=a.seed)}
    result["measure_s"] = round(time.monotonic() - t0, 1)

    full, base = result["with_lora_as_initialised"], result["lora_b_zeroed_base_model"]
    print(f"=== {a.model}: {result['settings_text']}   observation: {result['observation']}   {result['n_lora_tensors']} LoRA tensors")
    print(f"loss with the LoRA as initialised : {full['loss']:.6f}   |grad lora_a| {full['grad_norm_lora_a']:.3e}   |grad lora_b| {full['grad_norm_lora_b']:.3e}")
    print(f"loss with lora_b = 0 (base model)  : {base['loss']:.6f}   |grad lora_a| {base['grad_norm_lora_a']:.3e}   |grad lora_b| {base['grad_norm_lora_b']:.3e}")
    print(f"difference {result['loss_difference']:+.3e}   gradient of lora_a is zero when lora_b is zero: {result['lora_a_gradient_is_zero_when_b_is_zero']}")
    print(f"VERDICT: {result['verdict']}   (load {load_s} s, measure {result['measure_s']} s)")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"lora_effect_{a.model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"report: {path}")
    return 0 if result["verdict"].startswith("LoRA is in") else 1


if __name__ == "__main__":
    raise SystemExit(main())
