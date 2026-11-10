"""EpisodeTracker rules, wire format, protocol dataclasses. Pure (no GPU)."""
import numpy as np
import pytest

from src.envs import wire
from src.envs.episode_tracker import EpisodeTracker
from src.envs.protocol import EnvSpec, StepResult, build_obs_spec, concat_results, PROTOCOL_VERSION, ACTION_DIM


def run(tr, rows):
    return [tr.update(r) for r in rows]


# ----------------------------------------------------------------------------------------------------------- tracker
def test_success_ends_only_that_env():
    tr = EpisodeTracker(3, max_steps=5)
    f = run(tr, [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert f[1].terminated.tolist() == [False, True, False] and f[1].reward.tolist() == [0, 1, 0]
    assert f[2].acting.tolist() == [True, False, True]            # env 1 frozen afterwards
    assert f[2].t.tolist() == [3, 2, 3]                           # and its counter stops
    assert f[2].success.tolist() == [False, True, False]


def test_timeout_is_exactly_max_steps_and_not_a_success():
    tr = EpisodeTracker(1, max_steps=4)
    f = run(tr, [[0]] * 4)
    assert [x.truncated[0] for x in f] == [False, False, False, True]
    assert not f[3].terminated[0] and not f[3].success[0] and f[3].t[0] == 4 and f[3].all_done


def test_success_on_the_last_step_is_success_not_timeout():
    tr = EpisodeTracker(1, max_steps=2)
    f = run(tr, [[0], [1]])
    assert f[1].terminated[0] and not f[1].truncated[0]


def test_hold_needs_consecutive_steps_and_a_break_resets_the_streak():
    tr = EpisodeTracker(2, max_steps=10, success_hold_steps=3)
    f = run(tr, [[1, 1], [1, 0], [1, 1], [0, 1], [0, 1]])
    assert f[2].terminated.tolist() == [True, False]
    assert f[4].terminated.tolist() == [False, True]               # env 1: streak broken at step 2, 3 in a row at steps 3..5


def test_frozen_env_ignores_the_simulator_flag():
    tr = EpisodeTracker(2, max_steps=3)
    tr.update([1, 0])
    f = tr.update([1, 0])
    assert f.inserted.tolist() == [False, False] and f.reward.tolist() == [0, 0] and f.t.tolist() == [1, 2]


def test_reset_envs_restarts_only_those():
    tr = EpisodeTracker(3, max_steps=3)
    run(tr, [[1, 0, 0], [0, 0, 0]])
    tr.reset_envs([0])
    assert tr.active.tolist() == [True, True, True] and tr.t.tolist() == [0, 2, 2] and tr.success.tolist() == [False] * 3
    f = tr.update([0, 0, 0])
    assert f.t.tolist() == [1, 3, 3] and f.truncated.tolist() == [False, True, True]


def test_update_when_nothing_runs_is_an_error():
    tr = EpisodeTracker(1, max_steps=1)
    tr.update([0])
    with pytest.raises(RuntimeError, match="no episode is running"):
        tr.update([0])


@pytest.mark.parametrize("kw", [dict(num_envs=0, max_steps=3), dict(num_envs=1, max_steps=0),
                                dict(num_envs=1, max_steps=3, success_hold_steps=0)])
def test_tracker_rejects_bad_arguments(kw):
    with pytest.raises(ValueError):
        EpisodeTracker(**kw)


def test_terminated_wins_over_truncated_on_the_same_step():
    tr = EpisodeTracker(1, max_steps=1)
    f = tr.update([1])
    assert f.terminated[0] and not f.truncated[0]


# --------------------------------------------------------------------------------------------------------------- wire
def test_wire_roundtrip_including_empty_and_scalars():
    msg = {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.array([True, False]), "c": np.float32(1.5),
           "d": np.zeros((2, 8, 8, 3), np.uint8), "e": np.array(3, np.int32), "empty": np.zeros((0, 4, 4, 3), np.uint8),
           "s": "text", "n": None, "nested": {"x": np.ones(2, np.int64)}, "i": np.int64(7)}
    out = wire.unpack(wire.pack(msg))
    for k in ("a", "b", "d", "e", "empty"):
        assert out[k].dtype == msg[k].dtype and out[k].shape == msg[k].shape and (out[k] == msg[k]).all()
    assert out["c"] == np.float32(1.5) and out["s"] == "text" and out["n"] is None and out["i"] == 7
    assert out["nested"]["x"].tolist() == [1, 1]


def test_wire_rejects_object_arrays():
    with pytest.raises(TypeError):
        wire.pack({"x": np.array([object()])})


# ----------------------------------------------------------------------------------------------------------- protocol
def spec(**kw):
    base = dict(protocol_version=PROTOCOL_VERSION, num_envs=4, action_dim=ACTION_DIM, max_steps=10, success_hold_steps=1,
                control_hz=15.0, prompt="p", auto_reset=True, bank_size=8,
                obs=build_obs_spec(4, gripper_effort=True, tactile=True), robot={"name": "x"})
    base.update(kw)
    return EnvSpec(**base)


def test_spec_wire_roundtrip_and_compatibility():
    s = spec()
    assert EnvSpec.from_wire(wire.unpack(wire.pack(s.to_wire()))) == s
    assert s.same_task_as(spec(num_envs=16, robot={"name": "y"}))                 # different size/robot dict: still one task
    assert not s.same_task_as(spec(max_steps=11))
    assert not s.same_task_as(spec(obs=build_obs_spec(4, gripper_effort=False, tactile=False)))


def make_result(n, final_ids=(), reward=None):
    obs = {"x": np.arange(n, dtype=np.float32)[:, None] * 10.0}
    fin = {"x": np.array([[-1.0 - i] for i in range(len(final_ids))], np.float32).reshape(len(final_ids), 1)}
    z = np.zeros(n, bool)
    return StepResult(obs=obs, reward=np.zeros(n, np.float32) if reward is None else reward, terminated=z.copy(),
                      truncated=z.copy(), acting=np.ones(n, bool), inserted=z.copy(), success=z.copy(),
                      t=np.zeros(n, np.int32), start_id=np.arange(n, dtype=np.int64), final_obs=fin,
                      final_ids=np.asarray(final_ids, np.int64), all_done=False)


def test_next_obs_puts_the_final_observation_where_the_episode_ended():
    r = make_result(3, final_ids=[1])
    nxt = r.next_obs()["x"][:, 0]
    assert nxt.tolist() == [0.0, -1.0, 20.0] and r.obs["x"][1, 0] == 10.0           # obs untouched, copy returned


def test_concat_results_offsets_final_ids():
    a, b = make_result(2, final_ids=[1]), make_result(3, final_ids=[0, 2])
    c = concat_results([a, b], [2, 3])
    assert c.final_ids.tolist() == [1, 2, 4] and c.obs["x"].shape == (5, 1) and c.final_obs["x"].shape == (3, 1)
    assert c.next_obs()["x"][:, 0].tolist() == [0.0, -1.0, -1.0, 10.0, -2.0]
    assert concat_results([a], [2]) is a
