"""The arithmetic of the cost report (tests/cost_model.py). Pure (no GPU)."""
import math

import numpy as np
import pytest

from cost_model import derive, fit


def rows(ns, vram_fixed=8000.0, vram_per=30.0, step_fixed=500.0, step_per=3.5, reset_fixed=4.0, reset_per=0.02):
    return [{"num_envs": n, "vram_mib": vram_fixed + vram_per * n, "reset_s": reset_fixed + reset_per * n,
             "step_ms_steps_mode": step_fixed + step_per * n, "step_ms_episodes_mode": step_fixed + step_per * n + 0.1 * n}
            for n in ns]


def test_fit_recovers_a_line_and_needs_two_points():
    f = fit([1, 2, 4, 8], [10 + 3 * n for n in (1, 2, 4, 8)])
    assert f["fixed"] == pytest.approx(10) and f["per_env"] == pytest.approx(3) and f["r2"] == pytest.approx(1)
    assert fit([1], [2]) is None


def test_step_mode_gain_matches_the_hand_computation():
    out = derive(rows([1, 4, 16]), episode_steps=150, rollout_steps=80)
    one, sixteen = out["steps"][0], out["steps"][2]
    t1 = 80 * 0.5035 + (80 / 150) * 4.02
    assert one["update_s"] == pytest.approx(t1) and one["samples_per_update"] == 80
    t16 = 80 * 0.556 + (80 / 150) * 4.32
    assert sixteen["update_s"] == pytest.approx(t16) and sixteen["samples_per_update"] == 16 * 80
    assert sixteen["gain"] == pytest.approx((16 * 80 / t16) / (80 / t1))
    assert one["gain"] == pytest.approx(1.0) and one["efficiency"] == pytest.approx(1.0)
    assert 1.0 < sixteen["gain"] < 16.0 and 0.0 < sixteen["efficiency"] < 1.0


def test_episode_mode_gain_matches_the_hand_computation():
    out = derive(rows([1, 4, 16]), episode_steps=150, rollout_steps=80)
    one, four, sixteen = out["episodes"]
    assert one["update_s"] == pytest.approx(4.02 + 150 * 0.5036) and one["episodes_per_update"] == 1
    assert sixteen["time_for_target_s"] == pytest.approx(1 * (4.32 + 150 * (0.556 + 0.0016)))
    assert one["time_for_target_s"] == pytest.approx(16 * one["update_s"]) and out["target_episodes"] == 16
    assert sixteen["gain"] == pytest.approx(one["time_for_target_s"] / sixteen["time_for_target_s"])
    assert four["gain"] > 1 and sixteen["gain"] > four["gain"]


def test_target_not_a_multiple_of_N_rounds_up():
    out = derive(rows([1, 4]), episode_steps=10, rollout_steps=5, target_episodes=6)
    assert out["episodes"][1]["time_for_target_s"] == pytest.approx(2 * out["episodes"][1]["update_s"])      # ceil(6 / 4) = 2


def test_memory_cost_of_the_gain():
    out = derive(rows([1, 16]), episode_steps=150, rollout_steps=80)
    s = out["steps"][1]
    assert s["gain_per_extra_gib"] == pytest.approx((s["gain"] - 1) / (15 * 30.0 / 1024))
    assert out["steps"][0]["gain_per_extra_gib"] is None          # nothing was added at the baseline


def test_a_slower_environment_count_is_visible_as_a_gain_below_one():
    bad = rows([1, 4])
    bad[1]["step_ms_steps_mode"] = 5000.0                           # 4 environments 10x slower per step than 1
    assert derive(bad, 150, 80)["steps"][1]["gain"] < 1.0


@pytest.mark.parametrize("kw", [dict(episode_steps=0, rollout_steps=5), dict(episode_steps=5, rollout_steps=0)])
def test_bad_arguments(kw):
    with pytest.raises(ValueError):
        derive(rows([1, 2]), **kw)
    with pytest.raises(ValueError):
        derive([], 5, 5)
    with pytest.raises(ValueError, match="increasing"):
        derive(rows([4, 1]), 5, 5)
