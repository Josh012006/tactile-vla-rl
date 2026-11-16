"""scripts/probe_critic_cost.py on CPU: the synthetic data, the measurement loop on a tiny critic and the arithmetic of the estimate."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from scripts import probe_critic_cost as PC  # noqa: E402
from src.algos import rl_critic as rc  # noqa: E402
from src.algos.rl_config import load_rl_config, load_rl_experiment  # noqa: E402
from src.algos.rl_critic_update import CriticTrainer  # noqa: E402

TINY = {"num_candidates": 4, "keep_top": 2, "critic_hidden": [16, 16, 16], "critic_image_size": 32, "critic_batch_size": 8}


def test_the_synthetic_data_fills_the_buffer_and_every_state_that_needs_candidates():
    cfg = load_rl_config(overrides=TINY)
    buf, cache = PC.synthetic_data(cfg, episodes=4, plans=3)
    assert len(buf) == 4 * 4 and len(buf.transition_ids(True)) == 6 and len(buf.transition_ids(False)) == 6        # half warm-up, half online
    assert cache.has(buf.candidate_ids()).all() and cache.get(buf.candidate_ids()[:2]).shape == (2, 4, PC.HORIZON, 8)
    assert 0 < buf.success_rate() < 1


def test_the_synthetic_data_carries_touch_when_the_critic_sees_it():
    cfg = load_rl_config(overrides={**TINY, "critic_use_tactile": True})
    buf, _ = PC.synthetic_data(cfg, episodes=2, plans=2)
    assert buf.touch_dim == cfg.critic_touch.dim > 0 and buf.critic_state([0, 1])["touch"].shape == (2, buf.touch_dim)


def test_measure_times_the_host_and_the_device_parts_after_the_compiling_calls():
    cfg = load_rl_config(overrides={**TINY, "critic_cameras": ["exterior"]})
    buf, cache = PC.synthetic_data(cfg, episodes=2, plans=3)
    trainer = CriticTrainer(rc.CriticNet.from_config(cfg), cfg, horizon=PC.HORIZON)
    ticks = iter(np.arange(0, 100, 0.5))
    row = PC.measure(trainer, buf, cache, steps=3, warmup_steps=2, clock=lambda: float(next(ticks)))
    assert trainer.updates == 2 + 3 + 3 and row["steps"] == 3                                                     # + the 3 of the run's loop (batches prepared ahead), timed as a whole
    assert row["overlapped_ms"] == pytest.approx(500.0 / 3, abs=0.01)
    assert row["host_batch_ms"]["median"] == 500.0 and row["update_ms"]["median"] == 500.0 and row["total_ms"]["median"] == 1000.0         # the fake clock moves 0.5 s per reading


def test_the_estimate_is_the_runs_updates_times_the_median_time():
    cfg = load_rl_config()                                                                       # defaults: utd 1, 5000 warm-up steps
    est = PC.estimate([{"variant": "parl_global_local", "total_ms": {"median": 100.0}}], {"parl_global_local": cfg}, episodes=1000, steps_per_episode=150)
    assert est == [{"variant": "parl_global_local", "updates": 155_000, "hours": round(155_000 * 100.0 / 3.6e6, 2)}]


def test_the_three_experiments_of_the_probe_exist_and_load():
    for name, path in PC.EXPERIMENTS.items():
        _, cfg = load_rl_experiment(path)
        assert cfg.variant == name
