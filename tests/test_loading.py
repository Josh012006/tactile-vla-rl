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


def test_a_checkpoint_that_holds_lora_tensors_keeps_them(tiny):
    """A checkpoint trained with LoRA (an expo-ft SFT step: `params/` holds the base weights AND the trained lora_a / lora_b) must come back
    as it is: only the LoRA tensors that the checkpoint LACKS are initialised. (A first version initialised every LoRA tensor and so replaced
    the trained ones by random ones: expo-ft's own step-4000 teleop checkpoint scored 0 % in this repository's evaluation.)"""
    train_cfg, _, _ = tiny
    full = jax.tree.map(np.asarray, nnx.state(train_cfg.model.create(jax.random.key(7))).to_pure_dict())      # a LoRA checkpoint, other values than the init of seed 0
    flat_full = tu.flatten_dict(full, sep="/")
    lora_keys = {k for k in flat_full if "lora" in k}
    assert lora_keys
    train_cfg.weight_loader = weight_loaders.CheckpointWeightLoader("fake/params")
    weight_loaders._model.restore_params = lambda *a, **k: full                                      # noqa: SLF001 (monkeypatched by the fixture, restored with it)
    model = loading.build_params_like_expo_ft(train_cfg)
    got = tu.flatten_dict(nnx.state(model).to_pure_dict(), sep="/")
    assert set(got) == set(flat_full)
    for k in lora_keys:
        assert np.array_equal(np.asarray(got[k], dtype=np.float32), np.asarray(jnp.asarray(flat_full[k]).astype(jnp.bfloat16), dtype=np.float32)), f"{k} was replaced"


def test_our_checkpoint_route_and_expo_fts_full_params_route_give_the_same_weights(tiny, monkeypatch, tmp_path):
    """The two ways an evaluation rebuilds a fine-tuned model, from the SAME trained values, end with the SAME weights:
    (1) ours: the initial model (base weights + LoRA init) with the TRAINABLE parameters of `<ckpt>/params` laid over it (`sft.load_checkpoint`);
    (2) expo-ft's: a `params/` that holds every weight, frozen and trained (what `scripts/adopt_expo_ft_checkpoint.py` points at), through `build_params_like_expo_ft`.
    The trained values are the initial ones plus a perturbation, in float32 as in training; `save_checkpoint` rounds them to bfloat16 (route 1), the loader casts
    the full params to bfloat16 (route 2): the same rounding, so the weights must be identical, leaf by leaf. (Written to share buffers: the SigLIP tower makes
    every copy of the model ~1 GB.)"""
    from openpi.shared import nnx_utils

    from src.algos import sft

    train_cfg, _, _ = tiny
    trainable_filter = nnx.Any(nnx_utils.PathRegex(".*lora.*"), nnx_utils.PathRegex(".*action_out_proj.*"))    # LoRA + a non-LoRA trained part (SigLIP / projections in the real runs)

    initial = loading.build_params_like_expo_ft(train_cfg)                       # the initial model of the run: base weights + LoRA init (the checkpoint has no LoRA)
    graphdef, state = nnx.split(initial)
    del initial
    trainable = state.filter(trainable_filter)
    rng = np.random.default_rng(0)
    trained_flat = {k: np.asarray(v, dtype=np.float32) + 0.05 * rng.standard_normal(v.shape).astype(np.float32)          # float32, as in the training
                    for k, v in tu.flatten_dict(trainable.to_pure_dict(), sep="/").items()}
    assert any("lora" in k for k in trained_flat) and any("action_out_proj" in k for k in trained_flat)
    trained = jax.tree.map(lambda x: x, trainable)
    trained.replace_by_pure_dict(tu.unflatten_dict(trained_flat, sep="/"))

    # route 1: our checkpoint (trainable only, bfloat16), laid over the initial model by sft.load_checkpoint
    sft.save_checkpoint(tmp_path / "ckpt", trained, {"step": 4000})
    holder = types.SimpleNamespace(graphdef=graphdef, state=state)
    holder.set_state = lambda new: setattr(holder, "state", new)
    sft.load_checkpoint(holder, tmp_path / "ckpt")
    ours = tu.flatten_dict(nnx.state(nnx.merge(holder.graphdef, holder.state)).to_pure_dict(), sep="/")

    # route 2: expo-ft's params/: every weight (the frozen ones as they are, the trained ones in float32), restored by the weight loader
    full = {k: np.asarray(v) for k, v in ours.items()}                           # views of the same buffers
    full.update(trained_flat)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *a, **k: tu.unflatten_dict(full, sep="/"))     # noqa: SLF001
    theirs = tu.flatten_dict(nnx.state(loading.build_params_like_expo_ft(train_cfg)).to_pure_dict(), sep="/")

    assert set(ours) == set(theirs)
    initial_values = tu.flatten_dict(state.to_pure_dict(), sep="/")
    differ_from_initial = set()
    for k in ours:
        a, b = np.asarray(ours[k]), np.asarray(theirs[k])
        assert a.dtype == b.dtype == jnp.bfloat16, k
        assert np.array_equal(a.astype(np.float32), b.astype(np.float32)), f"{k} differs between the two routes"
        if not np.array_equal(a.astype(np.float32), np.asarray(initial_values[k]).astype(np.float32)):
            differ_from_initial.add(k)
    assert differ_from_initial == set(trained_flat)                              # the trained tensors, and only those, are not the initial ones


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
    no_lora = {"vlm": {"use_lora": False}, "action_head": {"use_lora": False}}
    with pytest.raises(ValueError, match="load:"):
        loading.load_policy({"family": "pi05", **no_lora, "load": "magic", "openpi_config": "pi05_droid", "checkpoint": "gs://openpi-assets/x"})
    with pytest.raises(ValueError, match="strict"):                      # a LoRA model cannot be loaded strictly
        loading.load_policy({"family": "pi05", "load": "strict", "openpi_config": "pi05_droid", "checkpoint": "gs://openpi-assets/x"})
    with pytest.raises(ValueError, match="strict"):                      # ... in ANY component
        loading.load_policy({"family": "pi05", "vlm": {"use_lora": False}, "load": "strict", "openpi_config": "pi05_droid", "checkpoint": "gs://openpi-assets/x"})
    with pytest.raises(ValueError, match="one block per component"):     # the old global keys are refused before anything is downloaded
        loading.load_policy({"family": "pi05", "use_lora": False, "load": "weight_loader", "openpi_config": "pi05_droid", "checkpoint": "gs://openpi-assets/x"})


def test_the_probe_counts_total_trainable_and_lora_parameters(tiny):
    from scripts import probe_vla as probe
    train_cfg, _, _ = tiny
    model = loading.build_params_like_expo_ft(train_cfg)
    counts = probe.trainable_counts(types.SimpleNamespace(_model=model), types.SimpleNamespace(freeze_filter=train_cfg.model.get_freeze_filter()))
    assert "error" not in counts, counts
    assert counts["total_m"] > counts["trainable_m"] > counts["lora_m"] > 0       # the SigLIP tower is trainable too (openpi's filter)


# -- the weights are the YAML's checkpoint, also for pi0-FAST ---------------------------------------------------------------------------
def test_the_weight_loader_reads_the_yamls_checkpoint_not_the_one_the_openpi_config_names(tmp_path):
    """One source of truth for the weights: whatever loader the openpi config carries, `with_yaml_weights` reads the YAML's checkpoint."""
    @dataclasses.dataclass(frozen=True)
    class Cfg:
        weight_loader: object

    got = loading.with_yaml_weights(Cfg(weight_loaders.CheckpointWeightLoader("gs://somewhere/else/params")), tmp_path / "ckpt")
    assert isinstance(got.weight_loader, weight_loaders.CheckpointWeightLoader) and got.weight_loader.params_path == str(tmp_path / "ckpt" / "params")


def test_pi0_fast_with_lora_is_structurally_loadable_from_its_checkpoint_without_lora():
    """pi0-FAST through the real openpi config: the checkpoint (here: the abstract structure of the model WITHOUT LoRA, which is what the
    released checkpoint has) merged by openpi's CheckpointWeightLoader gives exactly the structure of the LoRA model, the LoRA tensors
    being the only ones to come from the model. Abstract (eval_shape): nothing is allocated, nothing is downloaded. What only the cluster
    can show is the real file: scripts/probe_vla.py on the job."""
    import openpi.shared.array_typing as at
    from openpi.training import config as _config
    from openpi.training.weight_loaders import _merge_params           # noqa: PLC2701

    from src.models.openpi_config import apply_settings
    from src.models.settings import parse_model_settings

    cfg = {"family": "pi0_fast"}
    lora = apply_settings(_config.get_config("pi0_fast_droid_jointpos_polaris"), parse_model_settings(cfg))
    plain = apply_settings(_config.get_config("pi0_fast_droid_jointpos_polaris"), parse_model_settings({**cfg, "vlm": {"use_lora": False}}))
    key = jax.random.key(0)
    expected = nnx.state(nnx.eval_shape(lora.model.create, key)).to_pure_dict()
    checkpoint = nnx.state(nnx.eval_shape(plain.model.create, key)).to_pure_dict()           # what the released checkpoint holds
    merged = _merge_params(checkpoint, expected, missing_regex=".*lora.*")
    at.check_pytree_equality(expected=expected, got=merged, check_shapes=True, check_dtypes=False)
    missing = {k for k in tu.flatten_dict(expected) if k not in tu.flatten_dict(checkpoint)}
    assert missing and all(any("lora" in part for part in k) for k in missing)               # only LoRA tensors come from the model
    with pytest.raises(ValueError):                                                         # and the strict structure check would have failed
        at.check_pytree_equality(expected=expected, got=checkpoint, check_shapes=True, check_dtypes=False)


def test_the_policy_reads_the_yamls_norm_stats_and_names_a_missing_file(tmp_path):
    import numpy as np
    import openpi.shared.normalize as normalize
    from src.models.loading import load_norm_stats
    stats = {"state": normalize.NormStats(mean=np.zeros(8), std=np.ones(8), q01=-np.ones(8), q99=np.full(8, 2.0))}
    normalize.save(tmp_path / "stats" / "mine", stats)
    got = load_norm_stats({"norm_stats": {"dir": str(tmp_path / "stats"), "asset_id": "mine"}}, None, tmp_path)
    assert np.allclose(got["state"].q99, 2.0)
    with pytest.raises(FileNotFoundError, match="normalisation statistics not found"):
        load_norm_stats({"norm_stats": {"dir": str(tmp_path / "stats"), "asset_id": "absent"}}, None, tmp_path)
