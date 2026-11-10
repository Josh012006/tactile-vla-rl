"""src/models/loading.py on CPU, with a tiny pi0.5 (gemma `dummy` variants, a small LoRA) and a fake checkpoint.

It reproduces the failure seen on the first probe job (a LoRA model, a checkpoint without LoRA tensors) and checks the way out.
Needs the `vla` dependency group (skipped otherwise). The SigLIP tower of openpi is not downsized by any variant, so the model
creation takes some seconds and ~2 GB of RAM.
"""
import dataclasses
import types

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("openpi")

import flax.nnx as nnx  # noqa: E402
import flax.traverse_util as tu  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import openpi.models.gemma as gemma  # noqa: E402
import openpi.models.lora as lora  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402
from openpi.models import pi0_config  # noqa: E402

from src.models import loading  # noqa: E402


@pytest.fixture
def tiny(monkeypatch):
    """(train config with a LoRA model, the checkpoint's params, the non-LoRA config the checkpoint comes from)."""
    real = gemma.get_config

    def get_config(variant):
        if variant == "dummy_lora":
            cfg = real("dummy")
            cfg.lora_configs = {"attn": lora.LoRAConfig(rank=4, alpha=4.0), "ffn": lora.LoRAConfig(rank=4, alpha=4.0)}
            return cfg
        return real(variant)

    monkeypatch.setattr(gemma, "get_config", get_config)

    base = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4)
    with_lora = dataclasses.replace(base, paligemma_variant="dummy_lora", action_expert_variant="dummy_lora")
    ckpt = jax.tree.map(np.asarray, nnx.state(base.create(jax.random.key(1))).to_pure_dict())     # what restore_params(np.ndarray) returns
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *a, **k: ckpt)            # noqa: SLF001
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda p, **k: p)
    train_cfg = types.SimpleNamespace(model=with_lora, weight_loader=weight_loaders.CheckpointWeightLoader("fake/params"))
    return train_cfg, ckpt, base


def test_openpi_strict_loading_fails_on_a_checkpoint_without_lora_tensors(tiny):
    """The error of the first job: the model has lora_a / lora_b, the checkpoint does not."""
    train_cfg, ckpt, _ = tiny
    with pytest.raises(ValueError, match="lora"):
        train_cfg.model.load(ckpt)


def test_expo_ft_way_keeps_the_checkpoint_weights_and_initialises_the_lora(tiny):
    train_cfg, ckpt, _ = tiny
    model = loading.build_params_like_expo_ft(train_cfg)
    got = tu.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    want = tu.flatten_dict(ckpt, sep="/")
    lora_keys = {k for k in got if "lora" in k}
    assert lora_keys and not (lora_keys & set(want)), "the LoRA tensors must be the only ones absent from the checkpoint"
    assert set(got) - lora_keys == set(want)
    for k, v in want.items():
        assert got[k].dtype == jnp.bfloat16, k
        assert np.array_equal(np.asarray(got[k], dtype=np.float32), np.asarray(jnp.asarray(v).astype(jnp.bfloat16), dtype=np.float32)), k
    for k in lora_keys:
        assert got[k].dtype == jnp.bfloat16, k
    # openpi initialises lora_b with normal(0.01), NOT zeros: the loaded model is the base model plus a small perturbation
    # (documented in loading.py and configs/model/pi05.yaml). If openpi ever changes this, the documentation must change too.
    assert any(np.any(np.asarray(got[k], dtype=np.float32) != 0) for k in lora_keys if "lora_b" in k)


def test_the_lora_only_init_gives_the_same_values_as_a_full_model_init(tiny):
    """What makes the low-memory loading legitimate: the LoRA tensors are those of `create(key)`, as in expo-ft."""
    train_cfg, _, _ = tiny
    key = jax.random.key(0)
    full = tu.flatten_dict(nnx.state(train_cfg.model.create(key)).to_pure_dict())
    only = loading.init_lora_params(train_cfg.model, key)
    assert only and all(any("lora" in part for part in k) for k in only)
    assert {k for k in full if any("lora" in part for part in k)} == set(only)
    for k, v in only.items():
        assert np.array_equal(np.asarray(v, dtype=np.float32), np.asarray(jnp.asarray(full[k]).astype(jnp.bfloat16), dtype=np.float32)), k


def test_the_loaded_arrays_are_on_the_device(tiny):
    """numpy arrays left in the model would be transferred again at every call."""
    train_cfg, _, _ = tiny
    model = loading.build_params_like_expo_ft(train_cfg)
    assert all(isinstance(x, jax.Array) for x in jax.tree.leaves(nnx.state(model)))


def test_a_model_without_lora_loads_through_the_same_path(tiny):
    """use_lora: false with `load: weight_loader`: nothing is missing from the checkpoint, the result is the checkpoint."""
    _, ckpt, base = tiny
    model = loading.build_params_like_expo_ft(types.SimpleNamespace(model=base, weight_loader=weight_loaders.CheckpointWeightLoader("fake/params")))
    got = tu.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    want = tu.flatten_dict(ckpt, sep="/")
    assert set(got) == set(want)
    for k, v in want.items():
        assert np.array_equal(np.asarray(got[k], dtype=np.float32), np.asarray(jnp.asarray(v).astype(jnp.bfloat16), dtype=np.float32)), k


def test_load_key_is_validated():
    with pytest.raises(ValueError, match="load:"):
        loading.load_policy({"family": "pi05", "use_lora": False, "load": "magic", "openpi_config": "pi05_droid", "checkpoint": "gs://openpi-assets/x"})
    with pytest.raises(ValueError, match="strict"):                      # a LoRA model cannot be loaded strictly
        loading.load_policy({"family": "pi05", "use_lora": True, "load": "strict", "openpi_config": "pi05_droid", "checkpoint": "gs://openpi-assets/x"})


def test_the_probe_counts_total_trainable_and_lora_parameters(tiny):
    from src.models import probe
    train_cfg, _, _ = tiny
    model = loading.build_params_like_expo_ft(train_cfg)
    counts = probe.trainable_counts(types.SimpleNamespace(_model=model), types.SimpleNamespace(freeze_filter=train_cfg.model.get_freeze_filter()))
    assert "error" not in counts, counts
    assert counts["total_m"] > counts["trainable_m"] > counts["lora_m"] > 0       # the SigLIP tower is trainable too (openpi's filter)
