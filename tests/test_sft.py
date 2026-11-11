"""src/algos/sft.py on CPU, with the tiny pi0.5 of tests/test_openpi_policy.py (dummy Gemma variants, tokenizer and normalisation stubbed).
openpi's SigLIP tower is not downsized by any variant and a float32 copy with its optimiser state does not fit the CI memory, so here the
vision tower is FROZEN (the freeze filter of the policy) and only the language model and the action expert are trained: what is tested is
the loop (micro-batches, accumulation, optimiser, evaluation, checkpoints), not the size of the model.

Needs the `vla` dependency group (skipped otherwise)."""
import json
import types

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
import openpi.shared.nnx_utils as nnx_utils  # noqa: E402

from src.algos import demos, sft  # noqa: E402
from test_openpi_policy import H, build  # noqa: E402
from test_sft_demos import write_demo  # noqa: E402

PROMPT = "insert the peg"


def vision_frozen_policy(seed_offset=0):
    policy, _ = build()
    policy.train_cfg = types.SimpleNamespace(freeze_filter=nnx_utils.PathRegex(".*img.*"))
    return policy


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demos")
    for i, steps in enumerate((6, 5, 7)):
        write_demo(folder, f"ep_{i:04d}_seed{i}.npz", steps, seed=i, tactile=False)
    return demos.load_demos([str(folder)], horizon=H, val_episodes=1, default_prompt=PROMPT)


@pytest.fixture(scope="module")
def run(sets, tmp_path_factory):
    """One short run, shared by the tests below (the compilation is the slow part)."""
    train_set, val_set = sets
    policy = vision_frozen_policy()
    trainable0, frozen0 = sft.split_state(policy)
    before = {k: np.asarray(v) for k, v in tu.flatten_dict(policy.state.to_pure_dict()).items()}
    cfg = sft.load_config(overrides={"demos": ["x"], "steps": 4, "batch_size": 4, "micro_batch_size": 2, "lr": 1e-3, "lr_end": 1e-3, "log_interval": 2,
                                     "eval_interval": 2, "eval_samples": 2, "save_interval": 2, "prefetch": 1})
    run_dir = tmp_path_factory.mktemp("run") / "r"
    summary = sft.train(policy, train_set, val_set, cfg, run_dir, {"model_name": "tiny"})
    return policy, run_dir, summary, before, (trainable0, frozen0)


def test_the_run_trains_the_trainable_part_only_and_writes_its_log(run):
    policy, run_dir, summary, before, (trainable0, frozen0) = run
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    events = [line["event"] for line in lines]
    assert events[0] == "start" and events[1] == "first_step" and events[-1] == "done"
    assert events.count("train") == 2 and events.count("eval") == 2 and events.count("checkpoint") == 2
    assert all(np.isfinite(line["loss"]) and line["grad_norm"] > 0 for line in lines if line["event"] == "train")
    evals = [line for line in lines if line["event"] == "eval"]
    assert set(evals[0]) >= {"train", "val"} and np.isfinite(evals[0]["val"]["loss"]) and evals[0]["train"]["chunks"] == 2
    assert lines[0]["trainable_m"] > 0 and lines[0]["frozen_m"] > 100                       # SigLIP (frozen) is the bulk of this tiny model
    after = {k: np.asarray(v) for k, v in tu.flatten_dict(policy.state.to_pure_dict()).items()}
    frozen_keys = {k for k in tu.flatten_dict(frozen0.to_pure_dict())}
    trainable_keys = {k for k in tu.flatten_dict(trainable0.to_pure_dict())}
    assert frozen_keys and trainable_keys and not frozen_keys & trainable_keys
    assert all(np.array_equal(after[k], before[k]) for k in frozen_keys)                      # the vision tower did not move
    assert any(not np.allclose(after[k].astype(np.float32), before[k].astype(np.float32)) for k in trainable_keys)
    assert (run_dir / "config.json").exists() and json.loads((run_dir / "summary.json").read_text())["steps"] == 4


def test_a_checkpoint_per_save_interval_with_its_meta(run):
    _, run_dir, summary, *_ = run
    assert [p.rsplit("/", 1)[1] for p in summary["checkpoints"]] == ["2", "4"]
    meta = json.loads((run_dir / "checkpoints" / "4" / "meta.json").read_text())
    assert meta["step"] == 4 and meta["model_name"] == "tiny" and meta["action_horizon"] == H and meta["sft"]["batch_size"] == 4


def test_load_checkpoint_gives_a_fresh_policy_the_trained_weights(run):
    trained, run_dir, *_ = run
    fresh = vision_frozen_policy()
    trainable_before = tu.flatten_dict(sft.split_state(fresh)[0].to_pure_dict())
    meta = sft.load_checkpoint(fresh, run_dir / "checkpoints" / "4")
    assert meta["step"] == 4
    want = tu.flatten_dict(sft.split_state(trained)[0].to_pure_dict())
    got = tu.flatten_dict(sft.split_state(fresh)[0].to_pure_dict())
    assert set(got) == set(want)
    assert all(np.array_equal(np.asarray(got[k]), np.asarray(want[k])) for k in want)         # `trained` ends with the bfloat16 weights it saved
    assert any(not np.allclose(np.asarray(got[k]).astype(np.float32), np.asarray(trainable_before[k]).astype(np.float32)) for k in got)
    obs = {"observation/exterior_image_1_left": np.zeros((1, 224, 224, 3), np.uint8), "observation/wrist_image_left": np.zeros((1, 224, 224, 3), np.uint8),
           "observation/joint_position": np.zeros((1, 7), np.float32), "observation/gripper_position": np.zeros((1, 1), np.float32)}
    trained.seed(3)
    fresh.seed(3)
    # same weights, same noise: the same chunk. Not bit for bit HERE: this tiny model's weights are float32 and `trained` holds its trained part in
    # bfloat16 while `fresh` holds the same (rounded) values in float32, so the arithmetic differs; the real policies are bfloat16 throughout.
    assert np.allclose(trained.plan(obs, PROMPT).raw, fresh.plan(obs, PROMPT).raw, atol=1e-2)


def test_a_checkpoint_that_does_not_fit_the_policy_is_refused(run, tmp_path):
    import orbax.checkpoint as ocp
    bad = tmp_path / "bad"
    bad.mkdir()
    ocp.PyTreeCheckpointer().save(bad.resolve() / sft.PARAMS_DIR, {"not_a_layer": {"w": np.zeros((3,), np.float32)}})
    (bad / sft.META_FILE).write_text("{}")
    with pytest.raises(ValueError, match="does not fit this policy"):
        sft.load_checkpoint(vision_frozen_policy(), bad)


def test_the_chunks_of_the_demonstrations_and_the_models_horizon_must_agree(sets, tmp_path):
    train_set, _ = sets
    other = demos.DemoSet(train_set.episodes, horizon=H + 1)
    with pytest.raises(ValueError, match="action_horizon"):
        sft.train(vision_frozen_policy(), other, None, sft.load_config(overrides={"demos": ["x"]}), tmp_path / "r")
