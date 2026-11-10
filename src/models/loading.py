"""Build an openpi policy from a configs/model/<model>.yaml. Needs the `vla` dependency group (jax, openpi); nothing else in src imports it.

The YAML names an openpi TrainConfig (`openpi_config`: data pipeline, norm stats, weights), whose Gemma variants are then set by the
LoRA keys of the YAML (`use_lora`, `lora_rank`: see src/models/openpi_config.py). Two ways to load the weights (`load:`):

  strict        openpi's own `create_trained_policy`: the checkpoint's `params/` must have EXACTLY the structure of the model. Only possible
                without LoRA (`use_lora: false`): pi0-FAST (checkpoint trained with this very config).

  weight_loader the way expo-ft loads pi0.5 (expo_ft/agents/vla/pi05.py: `_load_weights_and_validate` + `pi05_init_train_state`):
                the TrainConfig's `weight_loader` (CheckpointWeightLoader) returns the checkpoint weights MERGED with the model's own
                initialisation for every parameter whose name matches `.*lora.*`. This is what makes a LoRA model loadable from a
                checkpoint that has no LoRA tensors (pi05_droid_jointpos). The strict path cannot do that: it fails with
                "symmetric difference of key sets: {'lora_a', 'lora_b'}". Works without LoRA too (nothing is missing).

                Same loading as expo-ft, with a lower peak in memory (see `build_params_like_expo_ft`).

                openpi initialises BOTH lora_a and lora_b with normal(0.01) (openpi/models/lora.py, LoRAConfig.init_fn), whereas the LoRA
                paper starts from B = 0. Kept as openpi has it: a freshly initialised LoRA model is the base model plus a random
                perturbation. `use_lora: false` gives the exact base function.
"""
from __future__ import annotations

import pathlib

GCS_ANON = {"token": "anon"}


def _download(uri: str) -> pathlib.Path:
    import openpi.shared.download as download
    return download.maybe_download(uri, gs=GCS_ANON)       # already fetched by src.models.fetch: a cache hit, no network


def init_lora_params(model_cfg, key):
    """Only the LoRA parameters of `model_cfg.create(key)`, in bfloat16, on the device. Same values as in a full `create(key)` (the random
    draws are functional in the key), but jit only computes what is returned: the rest of the initialisation is dropped by XLA."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax
    import jax.numpy as jnp

    @jax.jit
    def init(rng):
        _, state = nnx.split(model_cfg.create(rng))
        flat = tu.flatten_dict(state.to_pure_dict())
        return {k: v.astype(jnp.bfloat16) for k, v in flat.items() if any("lora" in part for part in k)}

    return init(key)


def build_params_like_expo_ft(train_cfg, seed: int = 0):
    """The model of `train_cfg.model`, weights from `train_cfg.weight_loader`, LoRA parameters (absent from the checkpoint) from the
    model's own init, every parameter in bfloat16 (as the strict path does). Returns the nnx model.

    Memory: expo-ft passes the whole float32 checkpoint through a jit that also creates the model (`pi05_init_train_state.init`), which
    puts ~13 GB of float32 on the device on top of the bfloat16 result (measured: JAX peak 19.4 GB for a model of 6.5 GB). Here the
    checkpoint is cast to bfloat16 on the HOST, only the LoRA parameters are initialised on the device, and the peak is the model."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax
    import jax.numpy as jnp
    import numpy as np
    import openpi.shared.array_typing as at

    key = jax.random.key(seed)
    model_cfg = train_cfg.model
    abstract = nnx.eval_shape(model_cfg.create, key)
    graphdef, state = nnx.split(abstract)

    # expo-ft `_load_weights_and_validate`: the loader returns the checkpoint weights + ShapeDtypeStruct for what it does not have
    # (the LoRA parameters); the structure must match the model's; then the ShapeDtypeStruct are dropped.
    loaded = train_cfg.weight_loader.load(state.to_pure_dict())
    at.check_pytree_equality(expected=state.to_pure_dict(), got=loaded, check_shapes=True, check_dtypes=False)
    flat = {k: v for k, v in tu.flatten_dict(loaded).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    flat = {k: np.asarray(v).astype(jnp.bfloat16) for k, v in flat.items()}                      # on the host, before any transfer
    flat = jax.device_put(flat)
    flat.update(init_lora_params(model_cfg, key))

    state.replace_by_pure_dict(tu.unflatten_dict(flat))
    return nnx.merge(graphdef, state)


def _policy_from(train_cfg, model, checkpoint_dir: pathlib.Path):
    """The tail of openpi's `create_trained_policy` (same transforms, same norm stats from the checkpoint's assets/), for a model we built."""
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_config.asset_id is None:
        raise ValueError("asset id is required to load the norm stats")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    return _policy.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs=None,
        metadata=train_cfg.policy_metadata,
    )


def load_policy(model_cfg: dict):
    """(policy, openpi TrainConfig actually used: Gemma variants and freeze filter as set by the YAML)."""
    from src.models.openpi_config import apply_lora_settings, parse_lora_settings

    settings = parse_lora_settings(model_cfg)
    mode = model_cfg.get("load", "strict")
    if mode not in ("strict", "weight_loader"):
        raise ValueError(f"load: must be 'strict' or 'weight_loader', got {mode!r}")
    if mode == "strict" and settings.use_lora:
        raise ValueError("load: strict cannot load a LoRA model from a checkpoint without LoRA tensors: use `load: weight_loader` "
                         "(or `use_lora: false`)")
    from openpi.policies import policy_config
    from openpi.training import config as _config

    train_cfg = apply_lora_settings(_config.get_config(model_cfg["openpi_config"]), settings)
    checkpoint = _download(model_cfg["checkpoint"])
    if mode == "strict":
        return policy_config.create_trained_policy(train_cfg, checkpoint), train_cfg
    return _policy_from(train_cfg, build_params_like_expo_ft(train_cfg), checkpoint), train_cfg
