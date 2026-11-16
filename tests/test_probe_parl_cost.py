"""scripts/probe_parl_cost.py on CPU: the stages measured with a fake trainable VLA and a tiny critic, and the arithmetic of the estimate."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from rl_fakes import FakeTrainablePolicy  # noqa: E402

from scripts import probe_parl_cost as PP  # noqa: E402
from src.algos.rl_config import load_rl_config, load_rl_experiment  # noqa: E402

TINY = {"method": "parl", "keep_top": 2, "local_steps": 2, "local_lr": 1e-2, "rollout_selection": "softmax", "num_candidates": 4, "critic_hidden": [16, 16], "critic_image_size": 16,
        "critic_cameras": ["wrist"], "random_shift_px": 0, "critic_batch_size": 8, "distill_batch_size": 4, "distill_micro_batch_size": 2, "distill_lr": 0.05, "rows_per_sample_call": 3}


def test_the_stages_are_timed_with_the_code_of_the_run_and_the_unit_costs_are_derived():
    cfg = load_rl_config(overrides=TINY)
    lines = []
    m = PP.measure_parl(FakeTrainablePolicy(), cfg, None, states=16, refresh_states=6, state_dim=3, plans=3, say=lines.append)
    n = m["states"]                                                                                                          # the states with candidates: 3 plans per episode (+ the last state of a time-out)
    assert 12 <= n <= 16 and m["distill"]["steps"] == m["distill_first"]["steps"] == n // 4                                 # batches of 4
    assert m["annotate_s_per_state"] == pytest.approx(m["distill"]["annotate_s"] / n) and m["step_s"] == m["distill"]["step_s"] and m["step_s"] > 0
    assert m["refresh"]["states"] == 6 and m["refresh_s_per_state"] == pytest.approx(m["refresh"]["seconds"] / 6) and m["refresh_s_per_state"] > 0
    assert "save_state_s" not in m and len(lines) == 4                                                                      # no --save-dir: no save timed


def test_a_buffer_smaller_than_a_distillation_batch_is_refused():
    with pytest.raises(SystemExit, match="batch"):
        PP.measure_parl(FakeTrainablePolicy(), load_rl_config(overrides={**TINY, "distill_batch_size": 64, "distill_micro_batch_size": 16}), None, states=4, refresh_states=2, state_dim=3, plans=3, say=lambda s: None)


def test_the_states_of_each_stage_are_the_buffer_after_each_round():
    cfg = load_rl_config(overrides={"envs_per_batch": 50, "total_episodes": 150, "warmup_episodes": 50})
    assert PP.stage_states(cfg, plans=10) == [1000, 1500, 2000]                                                             # (50 + 50 r) episodes x 10 plans


def test_the_estimate_adds_up_the_stages_over_the_rounds():
    cfg = load_rl_config(overrides={"envs_per_batch": 50, "total_episodes": 100, "warmup_episodes": 50, "distill_batch_size": 64, "distill_micro_batch_size": 16})      # 2 rounds, evals 0 and 2
    m = {"annotate_s_per_state": 0.01, "step_s": 8.0, "refresh_s_per_state": 0.2, "save_state_s": 30.0, "save_checkpoint_s": 30.0}
    est = PP.estimate_run(m, cfg, plans=10, collect_round_s=100.0, eval_s=600.0, critic_h=1.0)
    n = [1000, 1500]
    assert est["state_recomputations"] == sum(n) and est["vla_steps"] == 1000 // 64 + 1500 // 64                              # 15 + 23, one epoch
    assert est["annotate_h"] == round(2500 * 0.01 / 3600, 2) and est["train_h"] == round(38 * 8.0 / 3600, 2) and est["refresh_h"] == round(2500 * 0.2 / 3600, 2)
    assert est["saves_h"] == round(2 * 60.0 / 3600, 2) and est["collect_h"] == round(3 * 100.0 / 3600, 2) and est["eval_h"] == round(2 * 600.0 / 3600, 2) and est["critic_h"] == 1.0
    assert est["total_h"] == round(sum(est[k] for k in ("annotate_h", "train_h", "refresh_h", "saves_h", "collect_h", "eval_h", "critic_h")), 1)


def test_the_memory_verdict_adds_the_env_server_to_the_peak():
    gib = 2 ** 30
    assert "fits" in PP.memory_verdict({"peak_bytes_in_use": 20 * gib}) and "DOES NOT FIT" in PP.memory_verdict({"peak_bytes_in_use": 40 * gib})
    assert "no memory statistics" in PP.memory_verdict({})


def test_the_two_parl_experiments_have_a_distillation_and_the_default_one_is_variant_b():
    assert load_rl_experiment(PP.DEFAULT_EXPERIMENT)[1].variant == "parl_global_local"
    assert load_rl_experiment("configs/experiment/rl/pi05_parl_global.yaml")[1].distills
    assert not load_rl_experiment("configs/experiment/rl/pi05_best_of_n.yaml")[1].distills


def test_the_compiler_is_asked_for_the_memory_of_each_micro_batch_and_the_largest_that_fits_is_used():
    cfg = load_rl_config(overrides=TINY)
    gib = 2 ** 30
    lines = []
    m = PP.measure_parl(FakeTrainablePolicy(), cfg, None, states=16, refresh_states=4, state_dim=3, plans=3, micros=(4, 2, 3), memory=lambda: {"bytes_in_use": gib, "bytes_limit": 8 * gib}, say=lines.append)
    gm = m["grad_memory"]
    assert set(gm["micro"]) == {4, 2} and gm["chosen"] == 4 == m["micro_used"]                                              # 3 does not divide the batch of 4
    assert all(r["fits"] and r["need_gib"] >= gm["resident_gib"] for r in gm["micro"].values()) and any("micro-batch 4" in line for line in lines)


def test_nothing_runs_when_no_micro_batch_fits():
    cfg = load_rl_config(overrides=TINY)
    with pytest.raises(SystemExit, match="no micro-batch fits"):
        PP.measure_parl(FakeTrainablePolicy(), cfg, None, states=16, refresh_states=4, state_dim=3, plans=3, micros=(4, 2), memory=lambda: {"bytes_in_use": 8 * 2 ** 30, "bytes_limit": 8 * 2 ** 30},
                        say=lambda s: None)


def test_the_memory_is_read_from_arguments_that_type_check_their_leaves():
    """openpi's `Observation` refuses placeholders for its leaves: `jit(f).lower(obs)` failed on the cluster; `compiled_memory` hands the compiler flat arrays."""
    import dataclasses
    import jax.numpy as jnp

    @jax.tree_util.register_pytree_node_class
    @dataclasses.dataclass
    class Strict:
        x: object
        y: object = None

        def __post_init__(self):
            for leaf in jax.tree.leaves(self):
                assert isinstance(leaf, (jax.Array, np.ndarray)) or hasattr(leaf, "aval"), f"not an array: {type(leaf)}"

        def tree_flatten(self):
            return (self.x, self.y), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(*children)

    strict = Strict(jnp.ones((4, 8)))
    f = lambda w, s: jnp.tanh(s.x @ w).sum()                                                                               # noqa: E731
    with pytest.raises(Exception):
        jax.jit(f).lower(jnp.ones((8, 8)), strict)
    stats = PP.compiled_memory(f, jnp.ones((8, 8)), strict)
    assert stats is not None and getattr(stats, "argument_size_in_bytes", 0) > 0
