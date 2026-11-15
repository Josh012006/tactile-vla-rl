"""What is random in our SFT is what is random in openpi's train.py (the SFT of expo-ft), number for number, for the same `seed`:

  1. the ORDER of the rows: `sft.row_batches` against openpi's own `TorchDataLoader` (its source, run as it is);
  2. the INITIALISATION of the LoRA matrices: `loading.openpi_model_key` / `init_lora_params` against the model that openpi's own `init_train_state` builds;
  3. the NOISE of an update (image augmentation, flow-matching noise and time): the micro-batches of a batch, from `prepare_loss_batch` / `loss_of_rows`, against openpi's
     `compute_loss(rng, whole batch)` (what `train_step` calls), for the loss and for the gradient;
  4. the CHECKPOINT: what `save_checkpoint` writes is read by openpi's own `restore_params`, and `load_checkpoint` reads it back (and the older layout of this repo).

openpi's train.py and data_loader.py import lerobot (not installed here): the functions and the class under test are taken from the files' source and run in a namespace holding what
they use. Tiny pi0.5 with LoRA (gemma `dummy` variants and a small SigLIP, as in tests/test_sft_step_equals_openpi_train_step.py).
"""
import ast
import dataclasses
import json
import logging
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("openpi")
pytest.importorskip("openpi_client")
torch = pytest.importorskip("torch")

import flax.nnx as nnx  # noqa: E402
import flax.traverse_util as tu  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import openpi  # noqa: E402
import openpi.models.gemma as gemma  # noqa: E402
import openpi.models.lora as lora  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.policies import droid_policy  # noqa: E402
from openpi.policies import policy as openpi_policy  # noqa: E402

from src.algos import sft  # noqa: E402
from src.models import loading  # noqa: E402
from src.models.openpi_policy import OpenPIPolicy  # noqa: E402
from test_openpi_policy import TinyTokens  # noqa: E402

H = 4
OPENPI_SRC = Path(openpi.__file__).resolve().parent


def _source(path: Path, names, namespace):
    """Functions / classes `names` of the file `path`, compiled from its source into `namespace` (the module itself cannot be imported without lerobot)."""
    if not path.is_file():
        pytest.skip(f"{path} not found")
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names), (names, [n.name for n in nodes])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)          # noqa: S102
    return namespace


@pytest.fixture
def tiny(monkeypatch):
    """(base config, model) of a tiny pi0.5 with LoRA."""
    real = gemma.get_config

    def get_config(variant):
        if variant == "dummy_lora":
            cfg = real("dummy")
            cfg.lora_configs = {"attn": lora.LoRAConfig(rank=4, alpha=4.0), "ffn": lora.LoRAConfig(rank=4, alpha=4.0)}
            return cfg
        return real(variant)

    monkeypatch.setattr(gemma, "get_config", get_config)
    import openpi.models.siglip as siglip
    real_siglip = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda *a, **k: real_siglip(*a, **{**k, "variant": "mu/14"}))
    base = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy_lora", action_expert_variant="dummy_lora", action_horizon=H)
    return base


# -- 1. the order of the rows --------------------------------------------------------------------------------------------------------------
def test_the_order_of_the_rows_is_the_one_of_openpis_data_loader_for_the_same_seed(tmp_path, monkeypatch):
    # openpi's loader as a module of its own (a file in tmp_path): its workers are spawned, and what they load has to be importable
    tree = ast.parse((OPENPI_SRC / "training" / "data_loader.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("_collate_fn", "_worker_init_fn", "TorchDataLoader")]
    assert len(nodes) == 3
    header = ("import logging, multiprocessing, os, typing\nfrom collections.abc import Iterator, Sequence\nimport jax\nimport jax.numpy as jnp\nimport numpy as np\nimport torch\n"
              "Dataset = object\n")
    (tmp_path / "openpi_loader_under_test.py").write_text(header + ast.unparse(ast.Module(body=nodes, type_ignores=[])))
    (tmp_path / "index_dataset_under_test.py").write_text("import numpy as np\n\nclass IndexDataset:\n    def __init__(self, n):\n        self.n = n\n    def __len__(self):\n        return self.n\n"
                                                         "    def __getitem__(self, i):\n        return {'i': np.int64(i)}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import index_dataset_under_test
    import openpi_loader_under_test
    namespace = {"TorchDataLoader": openpi_loader_under_test.TorchDataLoader}
    IndexDataset = index_dataset_under_test.IndexDataset
    n, batch, steps = 70, 8, 50                                                    # 8 batches an epoch (the 6 rows left over are dropped): 6 epochs and a bit
    loader = iter(namespace["TorchDataLoader"](IndexDataset(n), batch, shuffle=True, num_workers=2, seed=42))     # openpi's TrainConfig defaults: seed 42, workers
    theirs = [np.asarray(next(loader)["i"]).tolist() for _ in range(steps)]
    ours = [b.tolist() for b in sft.row_batches(n, batch, steps, seed=42)]
    assert ours == theirs
    assert [b.tolist() for b in sft.row_batches(n, batch, steps, seed=7)] != theirs                                # and the seed is what decides
    assert [b.tolist() for b in sft.row_batches(n, batch, steps, seed=42, skip=13)] == theirs[13:]               # a resumed run skips, it does not reorder
    assert all(len(set(b)) == batch for b in ours) and len({r for b in ours[:n // batch] for r in b}) == (n // batch) * batch


# -- 2. the initialisation of the LoRA matrices -----------------------------------------------------------------------------------------------
def test_the_lora_matrices_start_from_the_values_openpis_init_train_state_draws_for_the_same_seed(tiny):
    import openpi.shared.array_typing as at
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import openpi.training.sharding as sharding
    import openpi.training.utils as training_utils
    import openpi.training.weight_loaders as weight_loaders
    from typing import Any
    namespace = {"jax": jax, "jnp": jnp, "nnx": nnx, "at": at, "nnx_utils": nnx_utils, "_config": _config, "_optimizer": _optimizer, "sharding": sharding,
                 "training_utils": training_utils, "Any": Any, "logging": logging, "_weight_loaders": weight_loaders, "traverse_util": tu}
    _source(OPENPI_SRC.parents[1] / "scripts" / "train.py", ["_load_weights_and_validate", "init_train_state"], namespace)
    config = dataclasses.replace(_config.get_config("expo_pi05_droid_lora_finetune_sft_joint_state_delta"), model=tiny, weight_loader=weight_loaders.NoOpWeightLoader(),
                                 freeze_filter=tiny.get_freeze_filter(), ema_decay=None)
    seed = 42
    _, init_rng = jax.random.split(jax.random.key(seed))                         # train.py main(): `train_rng, init_rng = split(key(config.seed))`
    mesh = sharding.make_mesh(1)
    with sharding.set_mesh(mesh):
        state, _ = namespace["init_train_state"](config, init_rng, mesh, resume=False)
    theirs = {k: np.asarray(v) for k, v in tu.flatten_dict(state.params.to_pure_dict()).items() if any("lora" in part for part in k)}
    ours = {k: np.asarray(v) for k, v in loading.init_lora_params(tiny, loading.openpi_model_key(seed), jnp.float32).items()}
    assert theirs and set(ours) == set(theirs)
    for k in theirs:
        assert theirs[k].dtype == np.float32 and ours[k].dtype == np.float32
        np.testing.assert_array_equal(ours[k], theirs[k], err_msg="/".join(k))
    other = loading.init_lora_params(tiny, loading.openpi_model_key(seed + 1), jnp.float32)                      # the seed decides the draw
    assert any(not np.array_equal(np.asarray(other[k]), theirs[k]) for k in theirs)
    plain = loading.init_lora_params(tiny, jax.random.key(seed), jnp.float32)                                    # (the key this repo used before)
    assert any(not np.array_equal(np.asarray(plain[k]), theirs[k]) for k in theirs)


# -- 3. the noise of an update ----------------------------------------------------------------------------------------------------------------
def _random_batch(base, n, seed=0):
    rng = np.random.default_rng(seed)
    obs = base.fake_obs(n)
    images = {k: jnp.asarray(rng.uniform(-1, 1, v.shape).astype(np.float32)) for k, v in obs.images.items()}        # not constant: the augmentation has something to change
    obs = dataclasses.replace(obs, images=images, state=jnp.asarray(rng.normal(size=obs.state.shape).astype(np.float32)))
    return obs, jnp.asarray(rng.normal(size=(n, H, base.action_dim)).astype(np.float32))


def test_micro_batches_draw_what_a_single_pass_of_the_whole_batch_draws(tiny):
    model = tiny.create(jax.random.key(0))
    noise = jax.random.key(1)
    state0 = jax.tree_util.tree_map_with_path(          # the adaRMS layers start at zero in a fresh model, which blinds it to its Gemma weights
        lambda path, x: 0.1 * jax.random.normal(jax.random.fold_in(noise, hash(str(path)) % 2**31), x.shape, x.dtype) if "Dense_0" in str(path) and "llm" in str(path) else x,
        nnx.state(model))
    nnx.update(model, state0)
    graphdef, state = nnx.split(model)
    from openpi.shared import nnx_utils
    trainable_filter = nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora.*"))
    trainable, frozen = state.filter(trainable_filter), state.filter(nnx.Not(trainable_filter))
    pol = openpi_policy.Policy(model, transforms=[droid_policy.DroidInputs(model_type=tiny.model_type), TinyTokens()], output_transforms=[droid_policy.DroidOutputs(action_dim=8)])
    cfg = {"name": "tiny", "family": "pi05", "action_horizon": H, "action_dim": 8, "output_semantics": "offsets_from_chunk_start_state"}
    policy = OpenPIPolicy(pol, types.SimpleNamespace(freeze_filter=nnx.Not(trainable_filter)), cfg, seed=0)

    n, micro = 8, 2
    observation, actions = _random_batch(tiny, n)
    rng = jax.random.fold_in(jax.random.key(42), 3)                                         # the key of update 3

    # openpi: the loss of the whole batch, one key (train_step: `loss = mean(model.compute_loss(train_rng, observation, actions, train=True))`)
    def whole(t):
        return jnp.mean(nnx.merge(graphdef, nnx.State.merge(t, frozen)).compute_loss(rng, observation, actions, train=True))
    want_loss, want_grads = jax.value_and_grad(whole)(trainable)

    # ours: the draws of the whole batch once, the loss and gradient of each micro-batch from them, averaged
    prepared, draws = policy.prepare_loss_batch(rng, observation, actions, train=True)
    cut = lambda tree, i: jax.tree.map(lambda x: x[i * micro:(i + 1) * micro], tree)            # noqa: E731
    losses, grads = [], []
    for i in range(n // micro):
        loss, g = jax.value_and_grad(lambda t: policy.loss_of_rows(nnx.State.merge(t, frozen), cut(prepared, i), actions[i * micro:(i + 1) * micro], cut(draws, i)))(trainable)
        losses.append(loss)
        grads.append(g)
    got_loss = jnp.mean(jnp.stack(losses))
    got_grads = jax.tree.map(lambda *g: sum(g) / len(g), *grads)
    assert float(got_loss) == pytest.approx(float(want_loss), rel=1e-5)
    rel = lambda a, b: float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))                          # noqa: E731
    flat = lambda tree: jnp.concatenate([jnp.ravel(x).astype(jnp.float32) for x in jax.tree.leaves(tree)])       # noqa: E731
    assert rel(flat(got_grads), flat(want_grads)) < 1e-2           # bfloat16 compute: another batch shape reduces in another order (about 1 % per element), not other numbers
    assert max(float(jnp.max(jnp.abs(g))) for g in jax.tree.leaves(want_grads)) > 1e-6          # there is a gradient to compare

    # what the micro-batches used to draw (their own small shapes, a key per micro-batch) is NOT what the whole batch draws: the test can fail
    old = jnp.mean(jnp.stack([jnp.mean(nnx.merge(graphdef, state).compute_loss(jax.random.fold_in(rng, i), cut(observation, i), actions[i * micro:(i + 1) * micro], train=True))
                              for i in range(n // micro)]))
    assert abs(float(old) - float(want_loss)) > 1e-4 * abs(float(want_loss))
    old_grads = jax.tree.map(lambda *g: sum(g) / len(g), *[jax.grad(lambda t, i=i: jnp.mean(nnx.merge(graphdef, nnx.State.merge(t, frozen)).compute_loss(
        jax.random.fold_in(rng, i), cut(observation, i), actions[i * micro:(i + 1) * micro], train=True)))(trainable) for i in range(n // micro)])
    assert rel(flat(old_grads), flat(want_grads)) > 5 * rel(flat(got_grads), flat(want_grads))


# -- 4. the checkpoint -----------------------------------------------------------------------------------------------------------------------
class Module(nnx.Module):
    def __init__(self, rngs):
        self.layer = nnx.Linear(3, 4, rngs=rngs)
        self.lora_a = nnx.Param(jnp.full((2, 2), 0.5, jnp.float32))


def test_a_checkpoint_is_read_by_openpis_restore_params_and_by_load_checkpoint(tmp_path):
    import openpi.models.model as _model
    model = Module(nnx.Rngs(0))
    graphdef, state = nnx.split(model)
    trainable, frozen = state.filter(lambda path, _: "lora" in "/".join(map(str, path))), state.filter(lambda path, _: "lora" not in "/".join(map(str, path)))
    trainable = jax.tree.map(lambda x: x + 0.25, trainable)                                          # "trained"
    frozen = jax.tree.map(lambda x: x.astype(jnp.bfloat16), frozen)                                  # as in openpi's train state
    sft.save_checkpoint(tmp_path / "ckpt", trainable, frozen, {"step": 4000})
    # openpi's reader, as expo-ft's and openpi's inference use it
    params = _model.restore_params(tmp_path / "ckpt" / "params", restore_type=np.ndarray)
    flat = tu.flatten_dict(params)
    assert set(flat) == {("layer", "kernel"), ("layer", "bias"), ("lora_a",)}
    np.testing.assert_array_equal(flat[("lora_a",)], np.full((2, 2), 0.75, np.float32))              # trainable: float32, as trained
    assert flat[("lora_a",)].dtype == np.float32 and flat[("layer", "kernel")].dtype == jnp.bfloat16.dtype      # frozen: bfloat16
    # ours
    fresh = types.SimpleNamespace(graphdef=graphdef, state=jax.tree.map(lambda x: x.astype(jnp.bfloat16), nnx.state(Module(nnx.Rngs(5)))),
                                  trainable_filter=lambda path, _: "lora" in "/".join(map(str, path)))
    fresh.set_state = lambda s: setattr(fresh, "state", s)
    kernel_before = np.asarray(fresh.state.to_pure_dict()["layer"]["kernel"]).copy()
    meta = sft.load_checkpoint(fresh, tmp_path / "ckpt")
    assert meta["step"] == 4000
    got = tu.flatten_dict(fresh.state.to_pure_dict())
    np.testing.assert_array_equal(np.asarray(got[("lora_a",)]).astype(np.float32), np.full((2, 2), 0.75, np.float32))
    assert got[("lora_a",)].dtype == jnp.bfloat16                                                    # the policy runs in bfloat16
    np.testing.assert_array_equal(np.asarray(got[("layer", "kernel")]), kernel_before)                  # only the TRAINABLE parameters are replaced: the frozen ones are the policy's own (shared with its forks)


def test_a_checkpoint_of_the_older_layout_is_still_read(tmp_path):
    """The checkpoints written before openpi's layout: the trainable parameters only, in bfloat16, no `params` level."""
    import orbax.checkpoint as ocp
    graphdef, state = nnx.split(Module(nnx.Rngs(0)))
    (tmp_path / "old").mkdir()
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save((tmp_path / "old" / sft.PARAMS_DIR).resolve(), {"lora_a": np.full((2, 2), 0.75, dtype=jnp.bfloat16)})
    (tmp_path / "old" / sft.META_FILE).write_text(json.dumps({"step": 200}))
    fresh = types.SimpleNamespace(graphdef=graphdef, state=jax.tree.map(lambda x: x.astype(jnp.bfloat16), state),
                                  trainable_filter=lambda path, _: "lora" in "/".join(map(str, path)))
    fresh.set_state = lambda s: setattr(fresh, "state", s)
    assert sft.load_checkpoint(fresh, tmp_path / "old")["step"] == 200
    np.testing.assert_array_equal(np.asarray(fresh.state.to_pure_dict()["lora_a"]).astype(np.float32), np.full((2, 2), 0.75, np.float32))
