"""scripts/probe_determinism.py on CPU: the comparison of traces, the scripted open-loop run on a fake environment, and the model's ways of being called on the tiny test policy."""
import types

import numpy as np
import pytest

from scripts import probe_determinism as PD
from src.envs import protocol as P
from test_openpi_policy import build


# -- the simulator part -------------------------------------------------------------------------------------------------------------------------
class FakeEnv:
    """One environment whose state is q += 0.1 * (target - q): deterministic, unless `noise` perturbs the step `noise_from` on."""
    num_envs = 1

    def __init__(self, length=12, noise=0.0, noise_from=5):
        self.length, self.noise, self.noise_from = length, noise, noise_from
        self.actions = []

    def _res(self, done):
        image = np.zeros((1, 8, 8, 3), np.uint8)
        if self.noise and self.t >= self.noise_from:
            image[0, 0, 0, 0] = 3                                           # one pixel value, once the perturbation has started
        obs = {P.JOINT_POSITION: self.q[None].astype(np.float32), P.GRIPPER_POSITION: np.full((1, 1), 0.9, np.float32), P.PEG_SOCKET_OFFSET: np.full((1, 3), 0.01, np.float32),
               P.FT_FORCE: np.zeros((1, 3), np.float32), P.EXTERIOR_IMAGE: image, P.WRIST_IMAGE: image.copy()}
        return types.SimpleNamespace(obs=obs, all_done=done)

    def reset(self, seed, round):
        self.q, self.t = np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.7, 0.8]), 0
        return self._res(False)

    def step(self, action, auto_reset=False):
        assert action.shape == (1, P.ACTION_DIM) and auto_reset is False
        self.actions.append(action.copy())
        self.q = self.q + 0.1 * (action[0, :7] - self.q) + (self.noise if self.t >= self.noise_from else 0.0)
        self.t += 1
        return self._res(self.t >= self.length)


def test_the_scripted_targets_start_at_the_pose_and_are_the_same_every_time():
    q0 = np.linspace(-1, 1, 7)
    a, b = PD.scripted_targets(q0, 20), PD.scripted_targets(q0, 20)
    assert a.shape == (20, 7) and a.dtype == np.float32 and np.array_equal(a, b)
    assert np.abs(a[0] - q0).max() <= 0.15 + 1e-6 and np.abs(a - q0).max() > 0.05                    # moves, within the amplitude


def test_a_trace_holds_the_reading_after_the_reset_and_after_every_step_and_the_same_actions_are_sent_each_time():
    env = FakeEnv(length=12)
    trace = PD.run_trace(env, seed=1, steps=10)
    assert trace.shape[0] == 11 and trace.shape[1] == 7 + 1 + 3 + 3                                   # reset + 10 steps; joints, gripper, peg-socket offset, force
    first = [a.copy() for a in env.actions]
    again = PD.run_trace(env, seed=1, steps=10)
    assert np.array_equal(trace, again) and all(np.array_equal(x, y) for x, y in zip(first, env.actions[10:]))
    assert PD.run_trace(FakeEnv(length=4), 1, 10).shape[0] == 5                                        # an episode that ends stops the trace


def test_identical_traces_are_identical_and_a_perturbation_is_found_at_its_first_step_and_grows():
    clean = PD.run_trace(FakeEnv(), 1, 10)
    shaken = PD.run_trace(FakeEnv(noise=1e-6, noise_from=5), 1, 10)
    same = PD.compare_traces({"a": clean, "b": clean.copy()}, "a")
    assert same["traces"]["b"]["identical"] and same["traces"]["b"]["first_differing_step"] is None
    diff = PD.compare_traces({"a": clean, "b": shaken}, "a")["traces"]["b"]
    assert not diff["identical"] and diff["first_differing_step"] == 6                                 # step 5 is the 6th row: the reset is row 0
    assert 0 < diff["max_abs_diff_at"]["10"] and "25" not in diff["max_abs_diff_at"] and diff["max_abs_diff_final"] > 0


def test_the_verdict_separates_fresh_servers_from_a_second_play_in_the_same_server():
    clean = PD.run_trace(FakeEnv(), 1, 10)
    shaken = PD.run_trace(FakeEnv(noise=1e-6, noise_from=3), 1, 10)
    ok = PD.verdict_env({"s0.r0": clean, "s0.r1": clean.copy(), "s1.r0": clean.copy(), "s1.r1": clean.copy()})
    assert ok["fresh_servers_identical"] is True and ok["replays_identical"] is True
    bad = PD.verdict_env({"s0.r0": clean, "s0.r1": shaken, "s1.r0": clean.copy(), "s1.r1": clean.copy()})
    assert bad["fresh_servers_identical"] is True and bad["replays_identical"] is False                                                   # a replay differs, the fresh servers do not
    assert not bad["same_server_replays"]["s0.r1"]["identical"] and bad["same_server_replays"]["s1.r1"]["identical"]
    worse = PD.verdict_env({"s0.r0": clean, "s1.r0": shaken})
    assert worse["fresh_servers_identical"] is False and worse["replays_identical"] is None


# -- the model part -----------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pol():
    return build()[0]


def test_the_model_is_the_same_bits_every_way_it_is_called_on_the_tiny_policy(pol):
    result = PD.policy_checks(pol, PD.synthetic_obs(), "insert the peg", seed=5, repeats=2, threads=2)
    assert result["repeat_identical"] and result["chain_identical"] and result["fork_identical"] and result["threads_identical"], result
    assert len(result["threads"]) == 4 and len(set(result["repeat"] + [result["reference"], result["fork"]])) == 1
    pol.seed(6)
    assert PD.chunk_sha(pol.plan(PD.synthetic_obs(), "insert the peg")) != result["reference"]         # another seed is another chunk: the check can tell


def test_the_camera_frames_of_two_plays_are_compared_by_first_step_pixels_and_size_of_the_difference():
    a, b, c = {}, {}, {}
    PD.run_trace(FakeEnv(), 1, 10, images=a)
    PD.run_trace(FakeEnv(), 1, 10, images=b)
    PD.run_trace(FakeEnv(noise=1e-6, noise_from=4), 1, 10, images=c)
    assert len(a[P.EXTERIOR_IMAGE]) == 11 and a[P.EXTERIOR_IMAGE][0].shape == (8, 8, 3)
    same = PD.compare_images(a, b)
    assert all(v["identical"] and v["first_differing_step"] is None for v in same.values())
    diff = PD.compare_images(a, c)[P.EXTERIOR_IMAGE]
    assert not diff["identical"] and diff["first_differing_step"] == 4 and diff["max_abs_diff_at"]["5"] == 3
    assert diff["differing_fraction_at"]["5"] == pytest.approx(1 / (8 * 8 * 3)) and "25" not in diff["max_abs_diff_at"]
    assert diff["mean_abs_diff_at"]["5"] == pytest.approx(3 / (8 * 8 * 3), abs=1e-3) and diff["share_over_2_at"]["5"] == pytest.approx(1 / (8 * 8 * 3), abs=1e-4)
    assert diff["share_over_8_at"]["5"] == 0.0


def test_the_frames_are_written_as_pngs_with_the_difference_against_server_0(tmp_path):
    PIL = pytest.importorskip("PIL")
    a, c = {}, {}
    PD.run_trace(FakeEnv(), 1, 10, images=a)
    PD.run_trace(FakeEnv(noise=1e-6, noise_from=4), 1, 10, images=c)
    PD.save_frame_pngs(tmp_path / "images", {"s0.r0": a, "s1.r0": c})
    names = sorted(f.name for f in (tmp_path / "images").iterdir())
    assert any(n.startswith("s0.r0_") and n.endswith("_step1.png") for n in names)
    assert any(n.startswith("s1.r0_") and n.endswith("_step1_diff_x8.png") for n in names)
    assert not any(n.startswith("s0.r0_") and "diff" in n for n in names)          # nothing to compare server 0 with


def test_the_server_arguments_follow_the_set_overrides(tmp_path):
    from src.eval import sweep as S
    server = S.make_server_factory(tmp_path / "task.yaml", ["max_steps_per_episode=100"], tmp_path / "logs", {"PATH": "/usr/bin"},
                                   server_args=["--rendering_mode", "performance"])("a")
    assert server.command[-2:] == ["--rendering_mode", "performance"] and "max_steps_per_episode=100" in server.command
    plain = S.make_server_factory(tmp_path / "task.yaml", [], tmp_path / "logs", {"PATH": "/usr/bin"})("a")
    assert "--rendering_mode" not in plain.command
