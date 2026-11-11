"""scripts/lora_effect.py on CPU, with a tiny pi0.5 that has a small LoRA (gemma `dummy` variants) and openpi's own DroidInputs / DroidOutputs.
Needs the `vla` dependency group (skipped otherwise). What is tested is the logic of the measurement, not a real model: on the cluster the
script runs on pi0-FAST and pi0.5 themselves."""
import dataclasses
import types

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("openpi")

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import openpi.models.gemma as gemma  # noqa: E402
import openpi.models.lora as lora  # noqa: E402
from openpi import transforms  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.policies import droid_policy  # noqa: E402
from openpi.policies import policy as openpi_policy  # noqa: E402

from scripts import lora_effect  # noqa: E402
from src.models.openpi_policy import OpenPIPolicy  # noqa: E402
from test_openpi_policy import H, MODEL_DIM, TinyTokens, env_obs  # noqa: E402


def tiny_policy(monkeypatch, with_lora=True):
    real = gemma.get_config

    def get_config(variant):
        if variant == "dummy_lora":
            cfg = real("dummy")
            cfg.lora_configs = {"attn": lora.LoRAConfig(rank=4, alpha=4.0), "ffn": lora.LoRAConfig(rank=4, alpha=4.0)}
            return cfg
        return real(variant)

    monkeypatch.setattr(gemma, "get_config", get_config)
    variant = "dummy_lora" if with_lora else "dummy"
    base = pi0_config.Pi0Config(pi05=True, paligemma_variant=variant, action_expert_variant=variant, action_horizon=H)
    model = base.create(jax.random.key(0))
    # A freshly initialised pi0.5 is blind to its Gemma weights (LoRA included): the adaRMS modulation layers start at ZERO ("adaLN-zero"),
    # so the gates of the action expert's blocks are 0 and nothing reaches its output. The released checkpoints have trained, non-zero ones:
    # make them non-zero here, as in a real model.
    noise = jax.random.key(1)
    state = nnx.state(model)
    state = jax.tree_util.tree_map_with_path(
        lambda path, x: 0.1 * jax.random.normal(jax.random.fold_in(noise, hash(str(path)) % 2**31), x.shape, x.dtype) if "Dense_0" in str(path) and "llm" in str(path) else x, state)
    nnx.update(model, state)
    pol = openpi_policy.Policy(model, transforms=[droid_policy.DroidInputs(model_type=base.model_type), TinyTokens(), transforms.PadStatesAndActions(MODEL_DIM)],
                               output_transforms=[droid_policy.DroidOutputs(action_dim=8)])
    cfg = {"name": "tiny", "family": "pi05", "action_horizon": H, "action_dim": 8, "output_semantics": "offsets_from_chunk_start_state"}
    return OpenPIPolicy(pol, types.SimpleNamespace(freeze_filter=base.get_freeze_filter()), cfg, seed=0)


def test_a_connected_lora_changes_the_loss_and_has_gradients(monkeypatch):
    policy = tiny_policy(monkeypatch)
    before = jax.tree.map(np.asarray, policy.state.to_pure_dict())
    out = lora_effect.measure(policy, env_obs(1), "insert the peg")
    full, base = out["with_lora_as_initialised"], out["lora_b_zeroed_base_model"]
    assert out["verdict"] == "LoRA is in the forward pass", out
    assert full["loss"] != base["loss"] and np.isfinite(out["loss_difference"])
    assert full["grad_norm_lora_b"] > 0 and full["grad_norm_lora_a"] > 0
    assert base["grad_norm_lora_b"] > 0 and out["lora_a_gradient_is_zero_when_b_is_zero"] is True       # a path to b, none to a while b = 0
    assert out["n_lora_tensors"] > 0
    after = jax.tree.map(np.asarray, policy.state.to_pure_dict())
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(before), jax.tree.leaves(after))), "the policy's own weights must not change"


def test_a_model_blind_to_its_weights_is_reported_as_not_connected(monkeypatch):
    """The same tiny model with the adaRMS layers left at zero: the script must say so instead of claiming the LoRA works."""
    policy = tiny_policy(monkeypatch)
    state = nnx.state(policy.policy._model)
    state = jax.tree_util.tree_map_with_path(lambda path, x: x * 0 if "Dense_0" in str(path) and "llm" in str(path) else x, state)
    policy.set_state(state)
    out = lora_effect.measure(policy, env_obs(1), "insert the peg")
    assert out["verdict"].startswith("LoRA does NOT"), out


def test_a_model_without_lora_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="no LoRA"):
        lora_effect.measure(tiny_policy(monkeypatch, with_lora=False), env_obs(1), "insert the peg")
