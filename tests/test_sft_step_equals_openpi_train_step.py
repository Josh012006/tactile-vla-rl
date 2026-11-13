"""One update of our SFT (the pieces of `src.algos.sft._run`: trainable parameters in float32 / frozen in bfloat16, `make_optimizer`, the gradient of `OpenPIPolicy.loss`) against
openpi's own `train_step` (scripts/train.py of the pinned fork, which expo-ft's SFT runs), from the same weights, on the same batches, with the same noise: the parameters must
come out the same after several updates. Tiny pi0.5 with LoRA (gemma `dummy` variants; the SigLIP tower is the real one, a few minutes on CPU).

openpi's train.py imports its data loader (torch, lerobot): `train_step` is taken from the file's source and run in a namespace holding what it uses.
"""
import ast
import dataclasses
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("openpi")
pytest.importorskip("openpi_client")

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import openpi  # noqa: E402
import openpi.models.gemma as gemma  # noqa: E402
import openpi.models.lora as lora  # noqa: E402
import optax  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.policies import droid_policy  # noqa: E402
from openpi.policies import policy as openpi_policy  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import optimizer as _optimizer  # noqa: E402

from src.algos import sft  # noqa: E402
from src.models.openpi_policy import OpenPIPolicy  # noqa: E402
from test_openpi_policy import TinyTokens  # noqa: E402

H, LR = 4, 2.5e-5


def _openpi_train_step():
    """`train_step` of openpi's scripts/train.py, compiled from the file's source (the module cannot be imported without torch / lerobot)."""
    path = Path(openpi.__file__).resolve().parents[2] / "scripts" / "train.py"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_step")
    import dataclasses as dc

    import openpi.models.model as _model
    import openpi.shared.array_typing as at
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.utils as training_utils
    namespace = {"jax": jax, "jnp": jnp, "nnx": nnx, "optax": optax, "dataclasses": dc, "at": at, "_config": _config, "_model": _model,
                 "nnx_utils": nnx_utils, "training_utils": training_utils}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)          # noqa: S102
    return namespace["train_step"], training_utils


def test_four_updates_of_our_sft_give_the_parameters_of_openpis_train_step(monkeypatch):
    real = gemma.get_config

    def get_config(variant):
        if variant == "dummy_lora":
            cfg = real("dummy")
            cfg.lora_configs = {"attn": lora.LoRAConfig(rank=4, alpha=4.0), "ffn": lora.LoRAConfig(rank=4, alpha=4.0)}
            return cfg
        return real(variant)

    monkeypatch.setattr(gemma, "get_config", get_config)
    import openpi.models.siglip as siglip                                                  # a tiny vision tower too (the real one, 400M parameters, does not fit in 7 GB with its gradients)
    real_siglip = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda *a, **k: real_siglip(*a, **{**k, "variant": "mu/14"}))
    base = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy_lora", action_expert_variant="dummy_lora", action_horizon=H)
    model = base.create(jax.random.key(0))
    # as in the other tiny-model tests: the adaRMS layers start at zero in a fresh model, which blinds it to its Gemma weights
    noise = jax.random.key(1)
    state0 = jax.tree_util.tree_map_with_path(
        lambda path, x: 0.1 * jax.random.normal(jax.random.fold_in(noise, hash(str(path)) % 2**31), x.shape, x.dtype) if "Dense_0" in str(path) and "llm" in str(path) else x,
        nnx.state(model))
    nnx.update(model, state0)

    # What is trained here: the LoRA matrices, the action projection and the (tiny) vision tower. The partition itself is openpi's `get_freeze_filter` on both sides in the real runs
    # (src/algos/sft.py uses the config's), not under test.
    from openpi.shared import nnx_utils
    freeze = nnx.Not(nnx.Any(nnx_utils.PathRegex(".*lora.*"), nnx_utils.PathRegex(".*action_out_proj.*"), nnx_utils.PathRegex(".*PaliGemma/img.*")))

    # ---- openpi's side: its config (AdamW defaults, constant 2.5e-5), its TrainState as `init_train_state` builds it, its train_step
    train_step, training_utils = _openpi_train_step()
    config = dataclasses.replace(_config.get_config("expo_pi05_droid_lora_finetune_sft_joint_state_delta"), model=base, freeze_filter=freeze,
                                 lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=LR, decay_steps=100_000, decay_lr=LR), ema_decay=None)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    params = nnx.state(model)
    params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))        # "Convert frozen params to bfloat16"
    train_state = training_utils.TrainState(step=0, params=params, model_def=nnx.graphdef(model), tx=tx, opt_state=tx.init(params.filter(config.trainable_filter)),
                                            ema_decay=None, ema_params=None)

    # ---- our side: a policy over the same model, the weights as the policy runs them (everything bfloat16), then split and cast as `_run` does
    pol = openpi_policy.Policy(model, transforms=[droid_policy.DroidInputs(model_type=base.model_type), TinyTokens()], output_transforms=[droid_policy.DroidOutputs(action_dim=8)])
    cfg = {"name": "tiny", "family": "pi05", "action_horizon": H, "action_dim": 8, "output_semantics": "offsets_from_chunk_start_state"}
    policy = OpenPIPolicy(pol, types.SimpleNamespace(freeze_filter=freeze), cfg, seed=0)
    policy.set_state(jax.tree.map(lambda x: x.astype(jnp.bfloat16), nnx.state(model)))       # what `load_policy` hands to the trainer
    # openpi starts from the float32 values of the model, ours from their bfloat16 rounding (the checkpoint is cast to bfloat16 on the host): here both start from the SAME
    # float32 numbers, so that what is compared is the update itself (the rounding of the start is measured in another test)
    _, frozen = sft.split_state(policy)
    start = train_state.params.filter(config.trainable_filter)
    trainable = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.float32, copy=True), start)
    scfg = sft.SFTConfig(demos=("unused",), lr=LR, lr_end=LR, warmup_steps=0, steps=4)
    ours_tx, _ = sft.make_optimizer(scfg)
    opt_state = ours_tx.init(trainable)

    @jax.jit                                                                                 # compiled, as in `_run` (eager and compiled bfloat16 arithmetic differ at the 1e-4 level)
    def ours_step(trainable, opt_state, key, observation, actions):
        loss, grads = jax.value_and_grad(lambda t: policy.loss(nnx.State.merge(t, frozen), key, observation, actions, train=True))(trainable)
        updates, opt_state = ours_tx.update(grads, opt_state, trainable)
        return optax.apply_updates(trainable, updates), opt_state, loss, optax.global_norm(grads)

    rng = jax.random.key(7)
    for step in range(4):
        observation, actions = base.fake_obs(2), base.fake_act(2)
        key = jax.random.fold_in(rng, step)                                                    # openpi: fold_in(rng, state.step), state.step = step
        train_state, info = jax.jit(lambda s, o, a, _r=rng: train_step(config, _r, s, (o, a)))(train_state, observation, actions)
        trainable, opt_state, loss, gnorm = ours_step(trainable, opt_state, key, observation, actions)
        assert float(loss) == pytest.approx(float(info["loss"]), rel=1e-4), f"loss at update {step}"
        assert float(gnorm) == pytest.approx(float(info["grad_norm"]), rel=1e-3), f"gradient norm at update {step}"
    theirs = train_state.params.filter(config.trainable_filter)
    assert jax.tree.structure(trainable) == jax.tree.structure(theirs)
    for a, b in zip(jax.tree.leaves(trainable), jax.tree.leaves(theirs)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=2e-6)       # four updates of about lr = 2.5e-5: a wrong optimiser would be off by 1e-5 or more
    # and they moved: the test is not comparing two untouched copies
    moved = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(trainable), jax.tree.leaves(start)))
    assert moved > 1e-6, moved
