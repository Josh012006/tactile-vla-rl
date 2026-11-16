"""scripts/probe_batch_cost.py without a GPU: the measurement loop on a fake policy and the arithmetic of the estimate."""
import types

import numpy as np
import pytest

from scripts import probe_batch_cost as PB
from src.envs import protocol as P
from src.models.interface import Plan


class Fake:
    def __init__(self, native=True):
        self.calls = []
        self.samples = []
        self._native_samples = native

    def plan(self, obs, prompt, num_samples=1, *, row_seeds=None, pad_to=None):
        n = len(obs[P.JOINT_POSITION])
        self.calls.append((n, list(row_seeds), pad_to))
        self.samples.append(num_samples)
        raw = np.zeros((n, num_samples, 4, 8), np.float32)
        return Plan(targets=raw.copy(), raw=raw, infer_ms=1.0)


def test_every_batch_size_is_called_once_to_compile_then_repeats_times_with_pad_to_and_one_seed_per_row():
    policy, clock_values = Fake(), iter(range(0, 10_000))
    rows = PB.measure(policy, [8, 1, 8], 3, clock=lambda: next(clock_values) / 1e3, memory=lambda: {"peak_bytes_in_use": 2 ** 30}, say=lambda m: None)
    assert [r["batch"] for r in rows] == [1, 8]                                       # sorted, no duplicates
    assert [c[0] for c in policy.calls] == [1] * 4 + [8] * 4 and all(c[2] == c[0] and len(set(c[1])) == c[0] for c in policy.calls)
    assert all(r["calls"] == 3 and r["median_ms"] == 1.0 and r["first_call_ms"] == 1.0 for r in rows)      # the fake clock advances by 1 ms per reading
    assert rows[1]["median_ms_per_row"] == 0.12 and rows[0]["memory"] == {"peak_bytes_in_use": 2 ** 30}


def test_the_rows_of_a_batch_observation_differ():
    obs = PB.batch_obs(3)
    assert obs[P.EXTERIOR_IMAGE].shape == (3, 224, 224, 3) and obs[P.JOINT_POSITION].shape == (3, 7)
    assert not np.array_equal(obs[P.EXTERIOR_IMAGE][0], obs[P.EXTERIOR_IMAGE][1])


def test_the_estimate_adds_the_simulators_steps_and_the_models_plans():
    est = PB.estimate([{"batch": 100, "median_ms": 400.0}, {"batch": 1, "median_ms": None}], steps=150, replan=8)
    assert len(est) == 1 and est[0]["plans"] == 19
    sim = 150 * (531.0 + 2.29 * 100) / 1e3
    assert abs(est[0]["sim_s"] - round(sim, 1)) < 1e-9 and est[0]["model_s"] == 7.6
    assert abs(est[0]["batch_s"] - (round(sim, 1) + 7.6)) < 0.11 and abs(est[0]["s_per_episode"] * 100 - est[0]["batch_s"]) < 0.5


def test_a_batch_that_does_not_fit_is_recorded_and_the_larger_ones_are_not_tried():
    class Small(Fake):
        def plan(self, obs, prompt, num_samples=1, *, row_seeds=None, pad_to=None):
            if pad_to > 8:
                raise RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 7.8GiB\nmore text")
            return super().plan(obs, prompt, num_samples, row_seeds=row_seeds, pad_to=pad_to)

    rows = PB.measure(Small(), [1, 8, 32, 64], 2, memory=lambda: {}, say=lambda m: None)
    assert [r["batch"] for r in rows] == [1, 8, 32] and rows[2]["failed"].startswith("RESOURCE_EXHAUSTED") and "median_ms" not in rows[2]
    assert [e["batch"] for e in PB.estimate(rows, 150, 8)] == [1, 8]


def test_split_measures_one_plan_of_many_rows_cut_in_equal_calls_and_puts_max_rows_back():
    policy = Fake()
    policy.max_rows = 7
    rows = PB.measure(policy, [100], 2, memory=lambda: {}, say=lambda m: None, max_rows=64)
    assert policy.max_rows == 7                                                              # the policy's own setting is back
    assert rows[0]["batch"] == 100 and rows[0]["max_rows"] == 64 and (rows[0]["call_rows"], rows[0]["calls_per_plan"]) == (50, 2)
    assert PB.split_calls(64, 64) == {"call_rows": 64, "calls_per_plan": 1} and PB.split_calls(24, 16) == {"call_rows": 12, "calls_per_plan": 2}
    assert PB.split_calls(100, 16) == {"call_rows": 15, "calls_per_plan": 7} and PB.estimate(rows, 150, 8)[0]["batch"] == 100


def test_num_samples_goes_to_the_policy_with_one_row_per_environment_and_pi0_fast_refuses_it():
    policy = Fake()
    rows = PB.measure(policy, [4], 2, memory=lambda: {}, say=lambda m: None, num_samples=32)
    assert set(policy.samples) == {32} and all(c[2] == 4 for c in policy.calls)                 # the samples are inside the call: B rows
    assert rows[0]["num_samples"] == 32 and rows[0]["median_ms_per_candidate"] == round(rows[0]["median_ms"] / (4 * 32), 3)
    with pytest.raises(ValueError, match="pi0.5"):
        PB.measure(Fake(native=False), [4], 2, memory=lambda: {}, say=lambda m: None, num_samples=8)
    assert PB.measure(Fake(native=False), [4], 2, memory=lambda: {}, say=lambda m: None)[0]["num_samples"] == 1      # one candidate per row: as before, pi0-FAST included


def test_the_estimate_of_the_cache_refresh_uses_the_time_per_row_and_the_runs_states():
    assert PB.refresh_states(19) == 19 * sum(50 + 50 * r for r in range(1, 10)) == 51_300           # 10 rounds, none after the last: the buffer grows from 100 to 500 episodes
    est = PB.estimate([{"batch": 50, "num_samples": 32, "median_ms": 5_000.0}, {"batch": 50, "num_samples": 1, "median_ms": 100.0}], steps=150, replan=8)
    assert est[0]["refresh_states"] == 51_300 and est[0]["refresh_h"] == round(51_300 * 5_000.0 / 50 / 3.6e6, 1)      # 100 ms per state: 1.4 h
    assert "refresh_h" not in est[1]                                                                 # one candidate per row: no refresh to estimate
    assert PB.estimate([{"batch": 50, "num_samples": 32, "median_ms": 5_000.0}], 150, 8, states_to_refresh=1000)[0]["refresh_states"] == 1000
