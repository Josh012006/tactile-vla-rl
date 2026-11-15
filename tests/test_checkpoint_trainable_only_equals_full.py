"""Does a checkpoint of the TRAINABLE parameters only give the same evaluated model as the checkpoint of every parameter?

openpi's train.py writes EVERY parameter (the trainable ones in float32, the frozen ones in bfloat16). For pi0.5 with LoRA that is 467.0 M trainable parameters (1781 MiB in float32,
891 MiB in bfloat16) and 2936.5 M frozen ones (5601 MiB), so 7382 MiB against 891 or 1781 MiB: the frozen ones are the base model's, which every loader of this repo already has
(`sft.load_checkpoint` reads and replaces the trainable parameters only). `sft.save_checkpoint` writes the trainable parameters in float32 (the whole checkpoint when it is given the frozen
ones too). These tests compare it, and a bfloat16 variant, with the whole checkpoint on the tiny pi0.5 of tests/test_sft.py with its weights in bfloat16 as the evaluation runs them:

  1. the whole checkpoint holds the trainable parameters in float32 and the frozen ones in bfloat16 (what expo-ft's SFT writes), the one `save_checkpoint` writes only the trainable ones, in float32;
  2. loaded into a policy, each trainable-only checkpoint gives EXACTLY the weights of the whole one (every leaf, bit for bit) and EXACTLY the same chunks (same noise);
  3. openpi's own reader for inference (`restore_params(dtype=bfloat16)`, which is what serves expo-ft's checkpoints) gives, for the trainable parameters, the weights of the policy
     that loaded the bfloat16 partial checkpoint: expo-ft's evaluation sees the same numbers;
  4. what is lost: the float32 values of the trainable parameters (a relative error of at most 2**-8 per weight), which only matter to a training that CONTINUES from the checkpoint
     (the resume snapshot, which holds them in float32, is for that), never to an evaluation;
  5. the size on disk.

Not covered here (it needs the real model): that the frozen part of a full checkpoint is the base checkpoint's, in bfloat16 (that is what makes it redundant).

Needs the `vla` dependency group (skipped otherwise)."""
import json

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("openpi")
pytest.importorskip("orbax.checkpoint")

import flax.nnx as nnx  # noqa: E402
import flax.traverse_util as tu  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import openpi.models.model as _model  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

from src.algos import sft  # noqa: E402
from test_openpi_policy import env_obs  # noqa: E402
from test_sft import PROMPT, release_memory, vision_frozen_policy  # noqa: E402

PARTIALS = ("partial_float32", "partial_bfloat16")


def deployed_policy():
    """A policy with every weight in bfloat16, as the evaluation runs it (the tiny model is created in float32)."""
    policy = vision_frozen_policy()
    policy.set_state(sft.tree_cast(policy.state, jnp.bfloat16))
    return policy


def write_trainable_only(directory, trainable):
    """What a save of the trainable parameters only would write: openpi's layout (`{"params": ...}`) without the frozen ones."""
    directory.mkdir(parents=True)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(directory / sft.PARAMS_DIR, {"params": jax.device_get(trainable)})
    (directory / sft.META_FILE).write_text(json.dumps({"step": 3, "updates": 4}))


def size_mib(directory):
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) / 2 ** 20


def flat(state):
    return tu.flatten_dict(state.to_pure_dict())


def chunks(policy):
    policy.seed(3)
    plan = policy.plan(env_obs(2), PROMPT)
    return np.asarray(plan.raw), np.asarray(plan.targets)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    release_memory()
    policy = vision_frozen_policy()
    trainable, frozen = sft.split_state(policy)                                     # float32 in the tiny model
    rng = np.random.default_rng(0)
    trained = jax.tree.map(lambda x: x + jnp.asarray(rng.normal(0.0, 0.05, x.shape), jnp.float32), trainable)       # "trained": values that bfloat16 cannot hold exactly
    root = tmp_path_factory.mktemp("checkpoints")
    sft.save_checkpoint(root / "full", trained, {"step": 3, "updates": 4}, frozen=sft.tree_cast(frozen, jnp.bfloat16))      # openpi's whole checkpoint: the frozen ones in bfloat16, as in its train state
    sft.save_checkpoint(root / "partial_float32", trained, {"step": 3, "updates": 4})                                # what the SFT writes
    write_trainable_only(root / "partial_bfloat16", sft.tree_cast(trained, jnp.bfloat16))
    del policy
    release_memory()
    return root, trained


def test_the_whole_checkpoint_holds_the_trainable_parameters_in_float32_and_the_frozen_ones_in_bfloat16_and_the_one_the_sft_writes_the_trainable_ones_only(saved):
    root, trained = saved
    params = tu.flatten_dict(_model.restore_params(root / "full" / "params", restore_type=np.ndarray))
    trainable_keys = set(flat(trained))
    assert trainable_keys and trainable_keys < set(params)                            # trainable ones and, besides, frozen ones
    assert all(params[k].dtype == np.float32 for k in trainable_keys)
    assert all(params[k].dtype == jnp.bfloat16.dtype for k in set(params) - trainable_keys)
    written = tu.flatten_dict(_model.restore_params(root / "partial_float32" / "params", restore_type=np.ndarray))        # what `save_checkpoint` writes by itself
    assert set(written) == trainable_keys and all(written[k].dtype == np.float32 and np.array_equal(written[k], params[k]) for k in trainable_keys)


@pytest.mark.parametrize("partial", PARTIALS)
def test_a_trainable_only_checkpoint_gives_the_same_weights_and_the_same_chunks_as_the_full_one(saved, partial):
    root, _ = saved
    full = deployed_policy()
    sft.load_checkpoint(full, root / "full")
    other = deployed_policy()
    sft.load_checkpoint(other, root / partial)
    want, got = flat(full.state), flat(other.state)
    assert set(got) == set(want)
    assert all(got[k].dtype == jnp.bfloat16 and np.array_equal(np.asarray(got[k]), np.asarray(want[k])) for k in want)         # every leaf, bit for bit
    raw_full, targets_full = chunks(full)
    raw_other, targets_other = chunks(other)
    assert np.array_equal(raw_full, raw_other) and np.array_equal(targets_full, targets_other)
    untouched = deployed_policy()
    assert not np.array_equal(chunks(untouched)[0], raw_full)                        # the checkpoint does change the model: the comparison is not empty
    release_memory()


def test_openpis_reader_for_inference_sees_the_weights_the_bfloat16_checkpoint_gives(saved):
    root, trained = saved
    served = tu.flatten_dict(_model.restore_params(root / "full" / "params", restore_type=np.ndarray, dtype=jnp.bfloat16))      # how openpi/expo-ft load a checkpoint to evaluate it
    other = deployed_policy()
    sft.load_checkpoint(other, root / "partial_bfloat16")
    ours = flat(other.state)
    keys = set(flat(trained))
    assert all(np.array_equal(np.asarray(ours[k]), served[k]) for k in keys)
    release_memory()


def test_what_the_bfloat16_checkpoint_loses_is_the_float32_value_of_the_trainable_parameters(saved):
    root, trained = saved
    exact = flat(trained)
    rounded = {k: np.asarray(v.astype(jnp.bfloat16)).astype(np.float32) for k, v in exact.items()}
    assert any(not np.array_equal(np.asarray(exact[k]), rounded[k]) for k in exact)                    # bfloat16 does lose something ...
    worst = max(float(np.max(np.abs(rounded[k] - np.asarray(exact[k])) / np.maximum(np.abs(np.asarray(exact[k])), 1e-30))) for k in exact)
    assert worst <= 2 ** -8                                                              # ... at most 2**-8 of each weight (8 bits of mantissa, round to nearest)
    params = tu.flatten_dict(_model.restore_params(root / "partial_float32" / "params", restore_type=np.ndarray))
    assert all(np.array_equal(params[k], np.asarray(exact[k])) for k in exact)         # the float32 partial checkpoint loses nothing


def test_the_size_on_disk(saved, capsys):
    root, _ = saved
    sizes = {name: size_mib(root / name) for name in ("full",) + PARTIALS}
    with capsys.disabled():
        print("\nsize on disk (tiny model, MiB): " + ", ".join(f"{name} {mib:.1f}" for name, mib in sizes.items()))
    assert sizes["partial_bfloat16"] < sizes["partial_float32"] < sizes["full"]
