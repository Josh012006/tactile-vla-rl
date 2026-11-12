"""Build an openpi policy from a configs/model/<model>.yaml. Needs the `vla` dependency group (jax, openpi); nothing else in src imports it.

The YAML names an openpi TrainConfig (`openpi_config`: data pipeline, norm stats), whose Gemma variants are then set by the per-component
LoRA keys of the YAML (`vlm.use_lora`, `vlm.lora_rank`, `action_head.*`: src/models/settings.py, src/models/openpi_config.py). The
NORM STATS are the checkpoint's (DROID's) unless the YAML has a `norm_stats` block (`dir`, `asset_id`: <dir>/<asset_id>/norm_stats.json), see `load_norm_stats`.
The WEIGHTS are always those of the YAML's `checkpoint`, never those the openpi config happens to name: one source of truth. (Today the two
agree for pi0.5 and for pi0-FAST PolaRiS; the replacement protects a YAML that is changed to another checkpoint from loading the old one.)
Two ways to load them (`load:`):

  strict        openpi's own `create_trained_policy`: the checkpoint's `params/` must have EXACTLY the structure of the model. Only possible
                without LoRA anywhere (`use_lora: false` in every component).

  weight_loader the way expo-ft loads pi0.5 (expo_ft/agents/vla/pi05.py: `_load_weights_and_validate` + `pi05_init_train_state`), and
                the way openpi's own LoRA recipe for pi0-FAST does (`pi0_fast_libero_low_mem_finetune`: `CheckpointWeightLoader(...)`
                with a LoRA variant): a CheckpointWeightLoader on the YAML's checkpoint returns the checkpoint weights MERGED with the
                model's own initialisation for every parameter whose name matches `.*lora.*`. This is what makes a LoRA model loadable
                from a checkpoint that has no LoRA tensors (pi05_droid_jointpos, pi0-FAST polaris). The strict path cannot do that: it
                fails with "symmetric difference of key sets: {'lora_a', 'lora_b'}". Works without LoRA too (nothing is missing).

                Same loading as expo-ft, with a lower peak in memory (see `build_params_like_expo_ft`).

                openpi initialises BOTH lora_a and lora_b with normal(0.01) (openpi/models/lora.py, LoRAConfig.init_fn), whereas the LoRA
                paper starts from B = 0. Kept as openpi has it: a freshly initialised LoRA model is the base model plus a random
                perturbation. `use_lora: false` gives the exact base function.
"""
from __future__ import annotations

import dataclasses
import pathlib

GCS_ANON = {"token": "anon"}


def _download(uri: str) -> pathlib.Path:
    import openpi.shared.download as download
    return download.maybe_download(uri, gs=GCS_ANON)       # already fetched by scripts.fetch_models: a cache hit, no network


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
    model's own init (and ONLY those: a checkpoint that holds LoRA tensors, e.g. an expo-ft SFT step, keeps its trained ones), every parameter in bfloat16 (as the strict path does). Returns the nnx model.

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
    for k, v in init_lora_params(model_cfg, key).items():                  # only the LoRA tensors the checkpoint LACKS: one that it holds (an expo-ft SFT step) is the trained one
        flat.setdefault(k, v)

    state.replace_by_pure_dict(tu.unflatten_dict(flat))
    return nnx.merge(graphdef, state)


def load_norm_stats(model_cfg: dict, train_cfg, checkpoint_dir: pathlib.Path):
    """The normalisation statistics of the policy: the YAML's `norm_stats` block when it has one (<dir>/<asset_id>/norm_stats.json, e.g. computed on
    the demonstrations: scripts/demos/compute_demo_norm_stats.py), otherwise the ones saved with the checkpoint (DROID's), as openpi has it."""
    import openpi.shared.normalize as normalize
    from src.models.interface import norm_stats_source

    source = norm_stats_source(model_cfg)
    if source is None:
        data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
        if data_config.asset_id is None:
            raise ValueError("asset id is required to load the norm stats")
        folder, asset_id = checkpoint_dir / "assets", data_config.asset_id
    else:
        folder, asset_id = source
    if not (folder / asset_id / "norm_stats.json").is_file():
        raise FileNotFoundError(f"normalisation statistics not found: {folder / asset_id / 'norm_stats.json'}"
                                + ("" if source is None else " (the `norm_stats` block of the model YAML; scripts/demos/compute_demo_norm_stats.py writes them)"))
    return normalize.load(folder / asset_id)                 # what openpi's checkpoints.load_norm_stats does


def _policy_from(train_cfg, model, norm_stats):
    """The tail of openpi's `create_trained_policy` (same transforms), for a model we built, with the given normalisation statistics."""
    import openpi.policies.policy as _policy
    import openpi.transforms as transforms

    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
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


def with_yaml_weights(train_cfg, checkpoint_dir: pathlib.Path):
    """The TrainConfig whose weight loader reads the `params/` of `checkpoint_dir` (the YAML's checkpoint, already downloaded)."""
    from openpi.training import weight_loaders

    return dataclasses.replace(train_cfg, weight_loader=weight_loaders.CheckpointWeightLoader(str(checkpoint_dir / "params")))


def load_policy(model_cfg: dict):
    """(policy, openpi TrainConfig actually used: Gemma variants, freeze filter and weight loader as set by the YAML)."""
    from src.models.openpi_config import apply_settings
    from src.models.settings import parse_model_settings

    settings = parse_model_settings(model_cfg)
    mode = model_cfg.get("load", "strict")
    if mode not in ("strict", "weight_loader"):
        raise ValueError(f"load: must be 'strict' or 'weight_loader', got {mode!r}")
    if mode == "strict" and settings.any_lora:
        raise ValueError("load: strict cannot load a LoRA model from a checkpoint without LoRA tensors: use `load: weight_loader` "
                         "(or `use_lora: false` in every component)")
    from openpi.policies import policy_config
    from openpi.training import config as _config

    train_cfg = apply_settings(_config.get_config(model_cfg["openpi_config"]), settings)
    checkpoint = _download(model_cfg["checkpoint"])
    norm_stats = load_norm_stats(model_cfg, train_cfg, checkpoint)
    if mode == "strict":
        return policy_config.create_trained_policy(train_cfg, checkpoint, norm_stats=norm_stats), train_cfg
    train_cfg = with_yaml_weights(train_cfg, checkpoint)
    return _policy_from(train_cfg, build_params_like_expo_ft(train_cfg), norm_stats), train_cfg
