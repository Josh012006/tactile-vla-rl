"""scripts/check_frozen_is_base.py: the comparison of the frozen part of a checkpoint with the base model's, on the tiny pi0.5 of tests/test_sft.py (the base is written as
a checkpoint of its own, in float32, the way the real base is stored), and the frozen / trainable split of the REAL pi0.5 (abstract model: no weights are built).

Needs the `vla` dependency group (skipped otherwise)."""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("openpi")
pytest.importorskip("orbax.checkpoint")

import flax.traverse_util as tu  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from scripts import check_frozen_is_base as CF  # noqa: E402
from src.algos import sft  # noqa: E402
from test_sft import release_memory, vision_frozen_policy  # noqa: E402


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    release_memory()
    policy = vision_frozen_policy()
    trainable, frozen = sft.split_state(policy)
    rng = np.random.default_rng(0)
    base_frozen = jax.tree.map(lambda x: x + jnp.asarray(rng.normal(0.0, 0.05, x.shape), jnp.float32), frozen)         # float32 values that bfloat16 cannot hold exactly
    trained = jax.tree.map(lambda x: x + 0.1, trainable)
    root = tmp_path_factory.mktemp("files")
    sft.save_checkpoint(root / "base", trainable, base_frozen, {"step": 0})                                          # the base: float32 throughout
    sft.save_checkpoint(root / "ckpt", trained, sft.tree_cast(base_frozen, jnp.bfloat16), {"step": 3})              # a checkpoint: trained part in float32, frozen part the base's in bfloat16
    sft.save_checkpoint(root / "unrounded", trained, base_frozen, {"step": 3})                                       # the frozen part kept in float32
    groups = {"trainable": set(tu.flatten_dict(trainable.to_pure_dict())), "frozen": set(tu.flatten_dict(frozen.to_pure_dict()))}
    del policy
    release_memory()
    return root, groups


def test_a_checkpoint_whose_frozen_part_is_the_base_in_bfloat16_is_equal(files):
    root, groups = files
    report = CF.compare_frozen(CF.read_params(root / "ckpt" / "params"), CF.read_params(root / "base" / "params"), groups["frozen"])
    assert report["verdict"] == "EQUAL" and report["frozen"] == len(groups["frozen"]) > 0 and report["equal"] == report["frozen"]
    assert report["checkpoint_dtypes"] == ["bfloat16"] and not report["different"] and not report["missing_in_base"] and not report["missing_in_checkpoint"]
    assert CF.format_report(report)[0].endswith("EQUAL")


def test_a_frozen_tensor_that_differs_is_found_with_its_difference(files):
    root, groups = files
    checkpoint, base = CF.read_params(root / "ckpt" / "params"), CF.read_params(root / "base" / "params")
    key = sorted(groups["frozen"])[0]
    changed = dict(checkpoint)
    value = np.asarray(changed[key]).astype(np.float32).copy()
    value.flat[0] += 0.5
    changed[key] = value.astype(jnp.bfloat16.dtype)
    report = CF.compare_frozen(changed, base, groups["frozen"])
    assert report["verdict"] == "NOT EQUAL" and report["equal"] == report["frozen"] - 1
    (found,) = report["different"]
    assert found["key"] == "/".join(key) and found["max_abs"] > 0.4 and 0 < found["share"] < 1 and not found["same_values"]


def test_missing_tensors_and_other_shapes_are_reported(files):
    root, groups = files
    checkpoint, base = CF.read_params(root / "ckpt" / "params"), CF.read_params(root / "base" / "params")
    first, second = sorted(groups["frozen"])[:2]
    short_base = {k: v for k, v in base.items() if k != first}
    short_checkpoint = {k: v for k, v in checkpoint.items() if k != second}
    reshaped = dict(checkpoint)
    reshaped[first] = np.zeros((1,), jnp.bfloat16.dtype)
    assert CF.compare_frozen(checkpoint, short_base, groups["frozen"])["missing_in_base"] == ["/".join(first)]
    assert CF.compare_frozen(short_checkpoint, base, groups["frozen"])["missing_in_checkpoint"] == ["/".join(second)]
    assert len(CF.compare_frozen(reshaped, base, groups["frozen"])["other_shape"]) == 1


def test_a_frozen_part_kept_in_float32_is_not_equal_even_though_it_is_the_base(files):
    root, groups = files
    report = CF.compare_frozen(CF.read_params(root / "unrounded" / "params"), CF.read_params(root / "base" / "params"), groups["frozen"])
    assert report["verdict"] == "NOT EQUAL" and report["checkpoint_dtypes"] == ["float32"] and len(report["different"]) == report["frozen"]    # unrounded: not the base in bfloat16


def test_the_precision_a_checkpoint_is_stored_in_comes_from_its_metadata(files):
    root, groups = files
    table = CF.precision_table(CF.stored_arrays(root / "ckpt" / "params"), groups)
    assert set(table) == {"trainable", "frozen"} and set(table["trainable"]) == {"float32"} and set(table["frozen"]) == {"bfloat16"}
    assert table["trainable"]["float32"]["tensors"] == len(groups["trainable"]) and table["frozen"]["bfloat16"]["tensors"] == len(groups["frozen"])
    base = CF.precision_table(CF.stored_arrays(root / "base" / "params"), groups)
    assert set(base["frozen"]) == {"float32"}
    assert any("frozen" in line and "bfloat16" in line for line in CF.format_precision(table))


def test_the_split_of_the_real_pi05_is_51_trainable_tensors_and_20_frozen_ones():
    """Abstract model (`nnx.eval_shape`): nothing is allocated. The counts are those of the real model with the YAML's LoRA ranks."""
    from openpi.training import config as _config
    from src.models.interface import load_model_cfg
    from src.models.openpi_config import apply_settings
    from src.models.settings import parse_model_settings
    model_cfg = load_model_cfg("pi05")
    groups = CF.model_groups(apply_settings(_config.get_config(model_cfg["openpi_config"]), parse_model_settings(model_cfg)))
    assert len(groups["trainable"]) == 51 and len(groups["frozen"]) == 20 and not groups["trainable"] & groups["frozen"]
    assert not any("lora" in part for k in groups["frozen"] for part in k)             # a LoRA tensor is never frozen
