"""The logic of src/models/probe.py that does not need openpi, JAX or a GPU: the observation handed to openpi, the conversion of a
model chunk into env actions (the offsets-or-absolute point that must be right), and the rollout loop, driven over a real socket
against the fake simulator by a fake model."""
import numpy as np
import pytest

from fakes import tag_of
from src.envs import protocol as P
from src.envs.client import VectorEnvClient
from src.models import probe
from test_client_server import Running, servers  # noqa: F401  (the `servers` fixture)

# What openpi's DroidInputs reads at inference (third_party/openpi/src/openpi/policies/droid_policy.py).
DROID_KEYS = {"observation/exterior_image_1_left", "observation/wrist_image_left", "observation/joint_position",
              "observation/gripper_position", "prompt"}


def obs_of(n=2):
    return {P.EXTERIOR_IMAGE: np.full((n, 4, 4, 3), 7, np.uint8), P.WRIST_IMAGE: np.full((n, 4, 4, 3), 9, np.uint8),
            P.JOINT_POSITION: np.arange(n * 7, dtype=np.float32).reshape(n, 7), P.GRIPPER_POSITION: np.full((n, 1), 0.5, np.float32)}


def test_the_observation_has_exactly_the_keys_openpi_reads():
    o = probe.to_openpi_obs(obs_of(), env=1, prompt="insert the peg")
    assert set(o) == DROID_KEYS and o["prompt"] == "insert the peg"
    assert np.array_equal(o["observation/joint_position"], np.arange(7, 14, dtype=np.float32))      # environment 1, not 0
    assert o["observation/gripper_position"].shape == (1,) and o["observation/exterior_image_1_left"][0, 0, 0] == 7
    assert o["observation/wrist_image_left"][0, 0, 0] == 9


def test_offsets_are_added_to_the_plan_state_once_and_absolute_targets_are_left_alone():
    q = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    chunk = np.tile(np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 1.0], np.float32), (5, 1))
    off = probe.chunk_to_targets(chunk, q, "offsets_from_chunk_start_state")
    assert np.allclose(off[:, :7], chunk[:, :7] + q) and (off[:, 7] == 1.0).all() and off.shape == (5, P.ACTION_DIM)
    ab = probe.chunk_to_targets(chunk, q, "absolute_targets")
    assert np.array_equal(ab[:, :7], chunk[:, :7]) and (ab[:, 7] == 1.0).all()
    longer = np.zeros((3, 32), np.float32)                          # a model that returns more columns: only the first 8 are used
    assert probe.chunk_to_targets(longer, q, "absolute_targets").shape == (3, P.ACTION_DIM)
    with pytest.raises(ValueError):
        probe.chunk_to_targets(np.zeros((3, 7), np.float32), q, "absolute_targets")
    with pytest.raises(ValueError):
        probe.chunk_to_targets(chunk, q, "velocity")


def test_the_indicator_tells_offsets_from_absolute_targets():
    q = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 2.0, 0.8], np.float32)
    assert probe.looks_like(np.tile(0.02 * np.ones(8), (10, 1)), q)["looks_like"] == "offsets"
    assert probe.looks_like(np.tile(q + 0.02, (10, 1)), q)["looks_like"] == "absolute"


def test_a_model_yaml_must_state_how_to_read_its_output():
    for name in ("pi05", "pi0_fast"):
        assert probe.load_model_cfg(name)["output_semantics"] in probe.SEMANTICS


class FakeModel:
    """Returns a chunk of `h` rows that drives joint 0 to 1.0 (the fake simulator's 'inserted'); counts its calls."""

    def __init__(self, semantics, h=10):
        self.semantics, self.h, self.calls = semantics, h, []

    def __call__(self, obs):
        self.calls.append(obs)
        q0 = float(obs["observation/joint_position"][0])
        chunk = np.zeros((self.h, 8), np.float32)
        chunk[:, 0] = (1.0 - q0) if self.semantics == "offsets_from_chunk_start_state" else 1.0
        return {"actions": chunk, "policy_timing": {"infer_ms": 12.5}}


@pytest.mark.parametrize("semantics", probe.SEMANTICS)
def test_rollout_succeeds_with_either_reading_when_the_yaml_matches_the_model(servers, semantics):  # noqa: F811
    s = servers(num_envs=2, max_steps=20, seed=0)
    model = FakeModel(semantics)
    cfg = {"output_semantics": semantics, "action_horizon": 10}
    with VectorEnvClient([s.endpoint]) as env:
        out = probe.run_rollout(env, model, cfg, steps=12, replan=4)
    assert all(e["success"] and e["terminated"] for e in out["envs"]), out["envs"]
    assert out["all_done"] and out["semantics_used"] == semantics
    assert {frozenset(c) for c in model.calls} == {frozenset(DROID_KEYS)}
    assert all(p["infer_ms"] == 12.5 and p["raw_shape"] == [10, 8] for p in out["plans"])
    assert out["plans"][0]["looks_like"] in ("offsets", "absolute")


def test_rollout_calls_the_model_every_replan_steps_and_clips_replan_to_the_horizon(servers):  # noqa: F811
    s = servers(num_envs=1, max_steps=50, seed=0)

    class Idle:                                    # never inserts: holds joint 0 where it is
        calls = 0

        def __call__(self, obs):
            Idle.calls += 1
            chunk = np.zeros((6, 8), np.float32)
            chunk[:, 0] = float(obs["observation/joint_position"][0])
            return {"actions": chunk}

    with VectorEnvClient([s.endpoint]) as env:
        out = probe.run_rollout(env, Idle(), {"output_semantics": "absolute_targets", "action_horizon": 6}, steps=24, replan=100)
    assert out["replan"] == 6 and Idle.calls == 4 and out["steps_run"] == 24      # 24 steps, a new chunk every 6
    assert not out["envs"][0]["success"] and out["envs"][0]["steps"] == 24
    assert out["plans"][0]["step"] == 0 and [p["step"] for p in out["plans"]] == [0, 6, 12, 18]


def test_rollout_is_the_same_episode_for_the_same_seed(servers):  # noqa: F811
    s = servers(num_envs=2, max_steps=20, seed=3)
    cfg = {"output_semantics": "absolute_targets", "action_horizon": 10}
    with VectorEnvClient([s.endpoint]) as env:
        a = probe.run_rollout(env, FakeModel("absolute_targets"), cfg, steps=12, replan=4, seed=3)
        b = probe.run_rollout(env, FakeModel("absolute_targets"), cfg, steps=12, replan=4, seed=3)
    assert [e["steps"] for e in a["envs"]] == [e["steps"] for e in b["envs"]]
    assert [p["q_plan"] for p in a["plans"]] == [p["q_plan"] for p in b["plans"]] and tag_of(3, 0) == 9
