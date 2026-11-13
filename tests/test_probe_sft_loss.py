"""scripts/probe_sft_loss.py without a GPU: the arguments, the statistics, the choice of runs and rows, and the loop that draws the noise."""
import json

import numpy as np
import pytest

from scripts import probe_sft_loss as PS


def test_parse_checkpoint_takes_the_step_after_the_last_colon():
    assert PS.parse_checkpoint("peg_insert_pi05_lora_teleop:4000") == ("peg_insert_pi05_lora_teleop", 4000)
    assert PS.parse_checkpoint("/scratch/a:b/run:0") == ("/scratch/a:b/run", 0)
    for bad in ("run", "run:", ":4000", "run:-1", "run:x"):
        with pytest.raises(ValueError, match="RUN:STEP"):
            PS.parse_checkpoint(bad)


def test_fixed_rows_are_sorted_distinct_and_reproducible():
    a = PS.fixed_rows(1000, 64, seed=3)
    assert list(a) == sorted(set(a)) and len(a) == 64 and a.max() < 1000
    assert np.array_equal(a, PS.fixed_rows(1000, 64, seed=3)) and not np.array_equal(a, PS.fixed_rows(1000, 64, seed=4))
    assert len(PS.fixed_rows(10, 64, seed=0)) == 10                                  # fewer rows than asked: all of them
    with pytest.raises(ValueError):
        PS.fixed_rows(0, 5, 0)


def test_summary_and_paired_difference():
    s = PS.summarise([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5 and s["rows"] == 4 and abs(s["se"] - np.std([1, 2, 3, 4], ddof=1) / 2) < 1e-12
    assert PS.summarise([5.0])["se"] == 0.0
    a, b = np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 2.5])
    d = PS.paired_difference(a, b)
    assert d["mean"] == pytest.approx(0.5) and d["se"] == pytest.approx(0.0, abs=1e-12)      # a constant offset: no noise left once paired
    with pytest.raises(ValueError):
        PS.paired_difference([1.0], [1.0, 2.0])


def test_group_by_run_keeps_the_order_and_loads_each_run_once():
    got = PS.group_by_run([("ours", 4000), ("theirs", 0), ("ours", 0), ("ours", 4000)])
    assert got == [("ours", [4000, 0]), ("theirs", [0])]


def test_sft_settings_come_from_the_first_run_that_has_them():
    adopted = {"adopted_from": "x", "model_cfg": {}}
    own = {"sft": {"demos": ["demos/a", "demos/b"], "num_demos": 50, "prompt": "Insert."}, "model_cfg": {}}
    assert PS.sft_settings([adopted, own]) == {"demos": ["demos/a", "demos/b"], "num_demos": 50, "prompt": "Insert."}
    assert PS.sft_settings([adopted]) is None


def _run(tmp_path, name, config, steps=()):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    for s in steps:
        (d / "checkpoints" / str(s)).mkdir(parents=True)
        (d / "checkpoints" / str(s) / "meta.json").write_text(json.dumps({"step": s}))
    return str(d)


CFG = {"name": "pi05", "family": "pi05", "action_horizon": 16}


def test_dry_run_checks_the_runs_and_prints_the_rows(tmp_path, capsys):
    ours = _run(tmp_path, "ours", {"sft": {"demos": ["demos/t"], "num_demos": 50, "prompt": "p"}, "model_cfg": CFG}, steps=(2000, 4000))
    theirs = _run(tmp_path, "theirs", {"adopted_from": "expo-ft", "model": "pi05", "model_cfg": CFG})
    assert PS.main(["--checkpoint", f"{ours}:0", f"{ours}:4000", f"{theirs}:0", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "256 (seed 0)" in out and "demos/t" in out and "steps [0, 4000]" in out and "steps [0]" in out


def test_errors_before_anything_is_loaded(tmp_path):
    ours = _run(tmp_path, "ours", {"sft": {"demos": ["demos/t"]}, "model_cfg": CFG}, steps=(4000,))
    theirs = _run(tmp_path, "theirs", {"adopted_from": "expo-ft", "model": "pi05", "model_cfg": CFG})
    with pytest.raises(SystemExit, match="no checkpoint \\[1234\\]"):
        PS.main(["--checkpoint", f"{ours}:1234", "--dry-run"])
    with pytest.raises(SystemExit, match="adopted checkpoint: its only weights state is step 0"):
        PS.main(["--checkpoint", f"{theirs}:4000", "--dry-run"])
    with pytest.raises(SystemExit, match="give --demos"):
        PS.main(["--checkpoint", f"{theirs}:0", "--dry-run"])
    other = _run(tmp_path, "other", {"sft": {"demos": ["demos/t"]}, "model_cfg": {**CFG, "action_horizon": 10}}, steps=(4000,))
    with pytest.raises(SystemExit, match="action horizon"):
        PS.main(["--checkpoint", f"{ours}:4000", f"{other}:4000", "--dry-run"])


def test_losses_of_uses_the_same_keys_for_every_checkpoint_and_one_row_per_demonstration_row():
    jax = pytest.importorskip("jax")
    import types
    seen = []

    def fn(state, key, observation, actions):                                        # the loss of a row = a function of the key and of the row's own value
        seen.append(np.asarray(jax.random.key_data(key)).tolist())
        return observation * 0 + float(jax.random.uniform(key)) + actions

    batches = [(np.zeros(3), np.array([1.0, 2.0, 3.0])), (np.zeros(2), np.array([4.0, 5.0]))]
    policy = types.SimpleNamespace(state=None)
    a = PS.losses_of(policy, fn, batches, draws=2, seed=0)
    first_keys, seen[:] = list(seen), []
    b = PS.losses_of(policy, fn, batches, draws=2, seed=0)
    assert a.shape == (2, 5) and np.array_equal(a, b) and first_keys == seen          # nothing depends on what was loaded before
    assert len({tuple(k) for k in first_keys}) == 4                                    # one key per (draw, micro-batch)
    assert np.allclose(a[0, :3] - a[0, 0], [0.0, 1.0, 2.0])                            # rows keep their order inside a micro-batch


def test_row_loss_of_a_tiny_pi05_is_one_value_per_row_and_its_mean_is_the_policys_loss():
    """`make_row_loss` on a real (tiny) pi0.5: shape (rows,), and the mean over rows equals the model's own mean loss for the same key (what `OpenPIPolicy.loss` returns)."""
    pytest.importorskip("jax")
    pytest.importorskip("openpi")
    import types

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0_config

    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4)
    graphdef, state = nnx.split(cfg.create(jax.random.key(0)))
    obs, actions = cfg.fake_obs(2), cfg.fake_act(2)
    policy = types.SimpleNamespace(graphdef=graphdef, state=state)
    key = jax.random.key(5)
    rows = PS.make_row_loss(policy, augment=False)(state, key, obs, actions)
    assert rows.shape == (2,) and bool(jnp.all(jnp.isfinite(rows)))
    reference = jnp.mean(nnx.merge(graphdef, state).compute_loss(key, obs, actions, train=False))
    assert float(jnp.mean(rows)) == pytest.approx(float(reference), rel=1e-5)
    both = PS.losses_of(policy, PS.make_row_loss(policy, False), [(obs, actions)], draws=2, seed=0)
    assert both.shape == (2, 2) and not np.array_equal(both[0], both[1])               # the draws differ (other time and noise), the rows are the same
