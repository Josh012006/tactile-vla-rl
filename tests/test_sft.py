"""src/algos/sft.py on CPU, with the tiny pi0.5 of tests/test_openpi_policy.py (dummy Gemma variants, tokenizer and normalisation stubbed).
openpi's SigLIP tower has no tiny variant by name (here its "mu/14" is patched in at the model's creation) and is FROZEN (the freeze filter of the policy), so that only
the language model and the action expert are trained: what is tested is the loop (micro-batches, accumulation, optimiser, evaluation, checkpoints), not the size of the model.

Needs the `vla` dependency group (skipped otherwise)."""
import gc
import importlib.util
import json
import os
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

from src.algos import demos, sft, tracking  # noqa: E402
from test_openpi_policy import H, build  # noqa: E402
from test_sft_demos import write_demo  # noqa: E402

PROMPT = "insert the peg"
META = {"model_name": "tiny", "model_cfg": {"name": "tiny"}}
os.environ.setdefault("WANDB_SILENT", "true")
HAS_WANDB = importlib.util.find_spec("wandb") is not None
COMMON = {"wandb_mode": "disabled", "demos": ["x"], "batch_size": 4, "micro_batch_size": 2, "lr": 1e-3, "lr_end": 1e-3, "log_interval": 2, "eval_interval": 100, "eval_samples": 2,
          "save_interval": 2, "resume_interval": 2, "prefetch": 1}


def vision_frozen_policy(seed_offset=0):
    # a tiny vision tower (openpi's variant "mu/14"): the real SigLIP (400M) in float32, its host copy and its serialisation do not fit the 7 GB of the test machine
    import openpi.models.siglip as siglip
    real_siglip = siglip.Module
    patch = pytest.MonkeyPatch()
    patch.setattr(siglip, "Module", lambda *a, **k: real_siglip(*a, **{**k, "variant": "mu/14"}))
    try:
        policy, _ = build()
    finally:
        patch.undo()
    policy.train_cfg = types.SimpleNamespace(freeze_filter=nnx_utils.PathRegex(".*img.*"))
    return policy


def release_memory():
    """Drop the compiled programs and garbage of the runs that are over: every `train` compiles its own (the test machine has 7 GB)."""
    jax.clear_caches()
    gc.collect()


def fingerprint(state):
    """Per parameter its sum and its first values: enough to see whether it moved, without a second copy of the model in memory (the test machine has 7 GB)."""
    out = {}
    for k, v in tu.flatten_dict(state.to_pure_dict()).items():
        x = np.asarray(v).astype(np.float32)
        out[k] = (float(x.sum(dtype=np.float64)), x.ravel()[:64].copy())
    return out


def same(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1])


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
    release_memory()
    policy = vision_frozen_policy()
    trainable0, frozen0 = sft.split_state(policy)
    before = fingerprint(policy.state)
    cfg = sft.load_config(overrides={"wandb_mode": "disabled", "demos": ["x"], "steps": 4, "batch_size": 4, "micro_batch_size": 2, "lr": 1e-3, "lr_end": 1e-3, "log_interval": 2,
                                     "eval_interval": 2, "eval_samples": 2, "save_interval": 2, "resume_interval": 2, "prefetch": 1})
    run_dir = tmp_path_factory.mktemp("run") / "r"
    summary = sft.train(policy, train_set, val_set, cfg, run_dir, META)
    return policy, run_dir, summary, before, (trainable0, frozen0)


def test_the_run_trains_the_trainable_part_only_and_writes_its_log(run):
    policy, run_dir, summary, before, (trainable0, frozen0) = run
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    events = [line["event"] for line in lines]
    assert events[0] == "start" and events[1] == "first_step" and events[-1] == "done"
    assert events.count("train") == 2 and events.count("eval") == 2 and events.count("checkpoint") == 2
    assert all(np.isfinite(line["loss"]) and line["grad_norm"] > 0 for line in lines if line["event"] == "train")
    assert all(np.isfinite(line["param_norm"]) and line["param_norm"] > 0 for line in lines if line["event"] == "train")     # openpi's param_norm
    assert all(line["batch_prep_s"] > 0 for line in lines if line["event"] == "train")                                        # what the feed thread costs per batch
    evals = [line for line in lines if line["event"] == "eval"]
    assert set(evals[0]) >= {"train", "val"} and np.isfinite(evals[0]["val"]["loss"]) and evals[0]["train"]["chunks"] == 2
    assert all("elapsed_s" in line for line in lines) and evals[0]["eval_s"] > 0                  # time and size of what a run costs
    assert all(line["save_s"] >= 0 and line["size_mib"] > 0 for line in lines if line["event"] == "checkpoint")
    assert lines[0]["trainable_m"] > 0                                             # (the frozen part, the tiny vision tower, rounds to 0.0 M)
    after = fingerprint(policy.state)
    assert all(x.is_deleted() for x in jax.tree.leaves(trainable0))        # the policy's own copy of the trainable weights was freed during the run (memory)
    assert not any(x.is_deleted() for x in jax.tree.leaves(policy.state))   # and the policy ends with valid weights
    frozen_keys = {k for k in tu.flatten_dict(frozen0.to_pure_dict())}
    trainable_keys = {k for k in tu.flatten_dict(trainable0.to_pure_dict())}
    assert frozen_keys and trainable_keys and not frozen_keys & trainable_keys
    assert all(same(after[k], before[k]) for k in frozen_keys)                                # the vision tower did not move
    assert any(not same(after[k], before[k]) for k in trainable_keys)
    assert (run_dir / "config.json").exists() and json.loads((run_dir / "summary.json").read_text())["steps"] == 4


def test_a_checkpoint_per_save_interval_with_its_meta(run):
    _, run_dir, summary, *_ = run
    assert [p.rsplit("/", 1)[1] for p in summary["checkpoints"]] == ["2", "3"]                  # openpi's rule: step % 2 == 0 and step > 0, and the last step (the loop runs 0 .. steps - 1)
    meta = json.loads((run_dir / "checkpoints" / "3" / "meta.json").read_text())
    assert meta["step"] == 3 and meta["updates"] == 4 and meta["model_name"] == "tiny" and meta["action_horizon"] == H and meta["sft"]["batch_size"] == 4


def test_load_checkpoint_gives_a_fresh_policy_the_trained_weights(run):
    trained, run_dir, *_ = run
    fresh = vision_frozen_policy()
    trainable_before = tu.flatten_dict(sft.split_state(fresh)[0].to_pure_dict())
    meta = sft.load_checkpoint(fresh, run_dir / "checkpoints" / "3")
    assert meta["step"] == 3
    want = tu.flatten_dict(sft.split_state(trained)[0].to_pure_dict())
    got = tu.flatten_dict(sft.split_state(fresh)[0].to_pure_dict())
    assert set(got) == set(want)
    bf16 = lambda x: np.asarray(x).astype(jnp.bfloat16)                                          # noqa: E731
    assert all(np.array_equal(bf16(got[k]), bf16(want[k])) for k in want)         # the checkpoint holds the trained weights in float32; `trained` ends with their bfloat16 rounding
    assert any(not np.allclose(np.asarray(got[k]).astype(np.float32), np.asarray(trainable_before[k]).astype(np.float32)) for k in got)
    obs = {"observation/exterior_image_1_left": np.zeros((1, 224, 224, 3), np.uint8), "observation/wrist_image_left": np.zeros((1, 224, 224, 3), np.uint8),
           "observation/joint_position": np.zeros((1, 7), np.float32), "observation/gripper_position": np.zeros((1, 1), np.float32)}
    trained.seed(3)
    fresh.seed(3)
    # same weights, same noise: the same chunk. Not bit for bit HERE: this tiny model's weights are float32 and `trained` holds its trained part in
    # bfloat16 while `fresh` holds the (unrounded) values in float32, so the arithmetic differs; the real policies are bfloat16 throughout.
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


# -- resume ---------------------------------------------------------------------------------------------------------------------------------
def snapshot_leaves(run_dir):
    import orbax.checkpoint as ocp
    directory, meta = sft.find_resume(run_dir)
    saved = ocp.PyTreeCheckpointer().restore(directory.resolve() / sft.PARAMS_DIR)
    return meta, [np.asarray(saved[k]) for k in sorted(saved)]


@pytest.fixture(scope="module")
def resumed(sets, tmp_path_factory):
    """The run of `run` (4 steps in one go) done in two parts: 2 steps, then a resume to 4."""
    train_set, val_set = sets
    release_memory()
    run_dir = tmp_path_factory.mktemp("resumed") / "r"
    # wandb offline (when installed), the second attempt asking for another project: what tracking.start_tracking gets is recorded
    mode, started, real = ("offline" if HAS_WANDB else "disabled"), [], tracking.start_tracking

    def recording(run_dir, resuming, project, *args, **kwargs):
        started.append((resuming, project))
        return real(run_dir, resuming, project, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(tracking, "start_tracking", recording)
    try:
        first = sft.train(vision_frozen_policy(), train_set, val_set,
                          sft.load_config(overrides={**COMMON, "wandb_mode": mode, "wandb_project": "proj_a", "steps": 2}), run_dir, META)
        release_memory()
        second = sft.train(vision_frozen_policy(), train_set, val_set,
                           sft.load_config(overrides={**COMMON, "wandb_mode": mode, "wandb_project": "proj_b", "steps": 4}), run_dir, META, resume=True)
    finally:
        patch.undo()
    return run_dir, first, second, started


def test_a_resumed_run_ends_in_the_same_state_as_an_uninterrupted_one(run, resumed):
    """The parameters (float32) and the whole optimiser state after step 4 are the same whether the run was cut at step 2 or not."""
    _, run_dir_a, *_ = run
    run_dir_b, first, second, _ = resumed
    meta_a, leaves_a = snapshot_leaves(run_dir_a)
    meta_b, leaves_b = snapshot_leaves(run_dir_b)
    assert meta_a["step"] == meta_b["step"] == 4 and len(leaves_a) == len(leaves_b) > 10
    for a, b in zip(leaves_a, leaves_b):
        assert a.shape == b.shape and np.allclose(a, b, rtol=1e-5, atol=1e-7)
    assert first["last_step"] == 1 and second["last_step"] == 3 and not second["stopped"]                 # the loop variable: the first part did 0 and 1, the second 2 and 3


def test_the_log_of_a_resume_continues_the_numbering(resumed):
    run_dir, *_ = resumed
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    resume = [line for line in lines if line["event"] == "resume"]
    assert len(resume) == 1 and resume[0]["from_step"] == 2 and resume[0]["target_steps"] == 4
    assert [line["step"] for line in lines if line["event"] == "train"] == [0, 2]                     # log every 2 steps from 0 (openpi logs the step 0 alone)
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["1", "3"]                  # the first part's last step (1); then step 2 is not > the step the resume started at, and 3 is the last
    assert sft.find_resume(run_dir)[1]["step"] == 4


@pytest.mark.skipif(not HAS_WANDB, reason="wandb is not installed")
def test_a_resumed_run_continues_the_same_wandb_run_in_its_project(resumed):
    run_dir, _, _, started = resumed
    run_id = (run_dir / "wandb_id.txt").read_text().strip()
    assert started == [(False, "proj_a"), (True, "proj_a")]                              # `proj_b` of the second attempt is ignored: the run lives in proj_a
    attempts = sorted((run_dir / "wandb").glob("offline-run-*"))
    assert len(attempts) == 2 and all(p.name.endswith(run_id) for p in attempts)         # two attempts, one wandb run (same id)


def test_a_resume_that_would_change_the_optimisation_or_has_nothing_to_resume_is_refused(resumed, sets, tmp_path):
    run_dir, *_ = resumed
    train_set, val_set = sets
    with pytest.raises(ValueError, match="no complete resume snapshot"):
        sft.train(vision_frozen_policy(), train_set, val_set, sft.load_config(overrides={**COMMON, "steps": 4}), tmp_path / "empty", META, resume=True)
    with pytest.raises(ValueError, match="already at step 4"):
        sft.train(vision_frozen_policy(), train_set, val_set, sft.load_config(overrides=COMMON | {"steps": 4}), run_dir, META, resume=True)
    with pytest.raises(ValueError, match=r"lr: 0.001 in the run"):
        sft.train(vision_frozen_policy(), train_set, val_set, sft.load_config(overrides={**COMMON, "steps": 6, "lr": 5e-4, "lr_end": 1e-3}), run_dir, META, resume=True)
    other = demos.DemoSet(train_set.episodes[:1], horizon=H)
    with pytest.raises(ValueError, match="demonstrations changed"):
        sft.train(vision_frozen_policy(), other, None, sft.load_config(overrides={**COMMON, "steps": 6, "batch_size": 4}), run_dir, META, resume=True)


def test_a_sigterm_stops_the_run_after_the_current_step_with_a_snapshot(sets, tmp_path):
    import os
    import signal
    import threading
    import time
    train_set, val_set = sets
    run_dir = tmp_path / "r"

    def terminate_after_the_first_step():
        log = run_dir / "metrics.jsonl"
        for _ in range(6000):
            if log.exists() and "first_step" in log.read_text():
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.05)

    threading.Thread(target=terminate_after_the_first_step, daemon=True).start()
    cfg = sft.load_config(overrides={**COMMON, "steps": 12, "resume_interval": 0, "save_interval": 100})
    summary = sft.train(vision_frozen_policy(), train_set, val_set, cfg, run_dir, META)
    assert summary["stopped"] and 0 <= summary["last_step"] < 11
    directory, meta = sft.find_resume(run_dir)
    assert meta["step"] == summary["last_step"] + 1                                         # a snapshot was written although resume_interval is 0
    events = [json.loads(line)["event"] for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert events[-1] == "stopped" and "resume_snapshot" in events
    assert signal.getsignal(signal.SIGTERM) is not None


def test_a_failing_run_writes_an_error_line_with_the_reason_and_raises(sets, tmp_path, monkeypatch):
    """The first probes of the full fine-tuning died without a word in metrics.jsonl: the reason (an out-of-memory above all) must be in it."""
    train_set, val_set = sets

    def out_of_memory(*args, **kwargs):
        raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 13412358144 bytes")

    monkeypatch.setattr(sft, "evaluate_chunks", out_of_memory)
    cfg = sft.load_config(overrides={**COMMON, "steps": 2, "eval_interval": 1, "save_interval": 100, "resume_interval": 0})
    run_dir = tmp_path / "r"
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        sft.train(vision_frozen_policy(), train_set, val_set, cfg, run_dir, META)
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "error" and "RESOURCE_EXHAUSTED" in events[-1]["error"] and "RuntimeError" in events[-1]["error"]
    assert "done" not in [e["event"] for e in events] and not (run_dir / "summary.json").exists()


def test_a_checkpoint_loaded_into_a_fork_leaves_the_base_policy_alone_and_shares_the_frozen_weights(run):
    """The evaluation of several checkpoints at once (src/eval/sweep.py): each worker's fork gets its checkpoint, the initial state stays the base policy's."""
    _, run_dir, *_ = run
    base = vision_frozen_policy()
    fork = base.fork(1)
    before = fingerprint(base.state)
    initial = base.state
    sft.load_checkpoint(fork, run_dir / "checkpoints" / "3")
    after = fingerprint(base.state)
    assert base.state is initial and all(same(before[k], after[k]) for k in before)          # the base is untouched
    frozen_base, frozen_fork = (tu.flatten_dict(sft.split_state(p)[1].to_pure_dict()) for p in (base, fork))
    assert frozen_base and all(frozen_base[k] is frozen_fork[k] for k in frozen_base)         # the frozen weights are the same device arrays: no second base model
    trained_base, trained_fork = (tu.flatten_dict(sft.split_state(p)[0].to_pure_dict()) for p in (base, fork))
    assert any(not np.array_equal(np.asarray(trained_base[k]).astype(np.float32), np.asarray(trained_fork[k]).astype(np.float32)) for k in trained_base)
    fork.set_state(initial)                                                                   # and a worker goes back to the initial state for the next job
    assert all(same(fingerprint(fork.state)[k], before[k]) for k in before)
