"""Collector (steps / episodes modes) and run_fixed_episodes against a fake simulator through the real manager. Pure."""
import numpy as np
import pytest

from fakes import ScriptedPolicy, expected_length, make_manager
from src.envs import protocol as P
from src.envs.collect import Collector, run_fixed_episodes


class DirectEnv:
    """The client interface (reset / step / num_envs / spec) straight on a manager: no network, StepResult objects."""

    def __init__(self, mgr):
        self.mgr, self.num_envs = mgr, mgr.n

    def reset(self, env_ids=None, bank_ids=None, seed=None):
        return P.StepResult.from_wire(self.mgr.reset(env_ids, bank_ids, seed))

    def step(self, actions, auto_reset=None):
        return P.StepResult.from_wire(self.mgr.step(actions, auto_reset))


def env_of(**kw):
    mgr, sim = make_manager(**kw)
    return DirectEnv(mgr), mgr, sim


# ------------------------------------------------------------------------------------------------------------ steps mode
def test_steps_mode_shapes_and_every_slot_gets_exactly_T_steps():
    env, mgr, sim = env_of(num_envs=4, bank_size=12, max_steps=10)
    c = Collector(env, ScriptedPolicy(), mode="steps")
    r = c.collect(25)
    assert r.action.shape == (25, 4, 8) and r.reward.shape == (25, 4) and r.obs[P.JOINT_POSITION].shape == (25, 4, 7)
    assert r.next_obs[P.JOINT_POSITION].shape == (25, 4, 7) and r.last_obs[P.JOINT_POSITION].shape == (4, 7)
    assert sim.physics_steps == 25 and c.total_env_steps == 100                      # nobody waited, nobody ran extra


def test_next_obs_is_the_true_next_observation_and_never_crosses_an_episode_boundary():
    env, mgr, sim = env_of(num_envs=3, bank_size=9, max_steps=6)
    r = Collector(env, ScriptedPolicy(), mode="steps").collect(40)
    j = P.JOINT_POSITION
    for t in range(40):
        for e in range(3):
            if r.done[t, e]:
                assert r.next_obs[j][t, e, 1] == r.t[t, e]                           # last obs of the ended episode: clock == length
                assert r.next_obs[j][t, e, 2] == r.obs[j][t, e, 2]                   # same start state: same episode
                if t + 1 < 40:
                    assert r.obs[j][t + 1, e, 1] == 0                                # the following transition starts a new episode
            elif t + 1 < 40:
                assert (r.next_obs[j][t, e] == r.obs[j][t + 1, e]).all()             # otherwise next_obs[t] == obs[t+1]


def test_flags_are_consistent_with_the_predicted_episode_lengths():
    env, mgr, sim = env_of(num_envs=4, bank_size=12, max_steps=10)
    r = Collector(env, ScriptedPolicy(), mode="steps").collect(60)
    assert len(r.episodes) >= 12
    for ep in r.episodes:
        length, term = expected_length(ep.start_id, 10)
        assert ep.length == length and ep.terminated == term and ep.truncated == (not term) and ep.success == term
        assert ep.episode_return == (1.0 if term else 0.0)                           # sparse reward: 1 at the inserting step
    assert (r.reward[r.terminated] == 1.0).all() and (r.reward[~r.terminated] == 0.0).all()
    assert not (r.terminated & r.truncated).any()


def test_start_id_of_a_transition_is_the_start_state_of_its_own_episode():
    env, mgr, sim = env_of(num_envs=3, bank_size=9, max_steps=6)
    r = Collector(env, ScriptedPolicy(), mode="steps").collect(40)
    assert (r.start_id == r.obs[P.JOINT_POSITION][:, :, 2]).all()


def test_the_episodes_in_flight_continue_across_calls_and_the_policy_version_is_stamped():
    env, mgr, sim = env_of(num_envs=2, bank_size=6, max_steps=10)
    c = Collector(env, ScriptedPolicy(), mode="steps")
    a = c.collect(4)
    c.version += 1                                                                    # the learner "updated"
    b = c.collect(4)
    assert a.version.tolist() == [0] * 4 and b.version.tolist() == [1] * 4
    assert (b.obs[P.JOINT_POSITION][0] == a.last_obs[P.JOINT_POSITION]).all()        # continues exactly where it stopped
    j = P.JOINT_POSITION
    for e in range(2):                                                                # an episode spans the two calls: clock is continuous
        clocks = np.concatenate([a.obs[j][:, e, 1], b.obs[j][:, e, 1]])
        resets = np.flatnonzero(np.diff(clocks) != 1)
        assert all(clocks[i + 1] == 0 for i in resets)


def test_extras_from_the_policy_are_stored_per_step():
    env, mgr, sim = env_of(num_envs=2)
    pol = ScriptedPolicy()
    r = Collector(env, lambda obs: (pol(obs), {"value": obs[P.JOINT_POSITION][:, 1]}), mode="steps").collect(5)
    assert r.extras["value"].shape == (5, 2) and (r.extras["value"] == r.obs[P.JOINT_POSITION][:, :, 1]).all()


def test_store_next_obs_false_saves_the_memory():
    env, mgr, sim = env_of(num_envs=2)
    assert Collector(env, ScriptedPolicy(), mode="steps", store_next_obs=False).collect(3).next_obs is None


def test_bad_policy_output_and_arguments():
    env, mgr, sim = env_of(num_envs=2)
    with pytest.raises(ValueError, match="actions"):
        Collector(env, lambda obs: np.zeros((3, 8), np.float32), mode="steps").collect(1)
    with pytest.raises(ValueError):
        Collector(env, ScriptedPolicy(), mode="nope")
    with pytest.raises(ValueError):
        Collector(env, ScriptedPolicy(), mode="steps").collect(0)


# ---------------------------------------------------------------------------------------------------------- episodes mode
def test_episodes_mode_returns_complete_episodes_with_their_final_observation():
    env, mgr, sim = env_of(num_envs=4, bank_size=12, max_steps=10)
    eps = Collector(env, ScriptedPolicy(), mode="episodes").collect(10)
    assert len(eps) >= 10 and len(eps) <= 10 + 3                                       # may overshoot by < num_envs
    j = P.JOINT_POSITION
    for ep in eps:
        length, term = expected_length(ep.start_id, 10)
        assert ep.length == length and ep.terminated == term and ep.success == term and ep.truncated == (not term)
        assert ep.obs[j].shape == (length, 7) and ep.action.shape == (length, 8)
        assert ep.obs[j][0, 1] == 0 and ep.obs[j][0, 2] == ep.start_id                  # starts at the start state, clock 0
        assert (np.diff(ep.obs[j][:, 1]) == 1).all()                                    # contiguous steps of ONE episode
        assert ep.final_obs[j][1] == length and ep.final_obs[j][2] == ep.start_id       # final obs: after the last action
        assert ep.episode_return == (1.0 if term else 0.0)


def test_episodes_in_flight_continue_after_the_update_and_mix_policy_versions():
    env, mgr, sim = env_of(num_envs=3, bank_size=9, max_steps=10)
    c = Collector(env, ScriptedPolicy(), mode="episodes")
    first = c.collect(3)
    c.version += 1
    second = c.collect(3)
    mixed = [ep for ep in second if ep.version.min() != ep.version.max()]
    assert mixed, "at least one episode was in flight during the update"
    for ep in mixed:
        assert ep.version[0] == 0 and ep.version[-1] == 1 and (np.diff(ep.version) >= 0).all()
        assert (np.diff(ep.obs[P.JOINT_POSITION][:, 1]) == 1).all()                      # still one continuous episode


def test_no_environment_waits_in_episodes_mode_either():
    env, mgr, sim = env_of(num_envs=4, bank_size=12, max_steps=10)
    c = Collector(env, ScriptedPolicy(), mode="episodes")
    c.collect(8)
    assert c.total_env_steps == 4 * sim.physics_steps


# ---------------------------------------------------------------------------------------------- fixed-set evaluation
def reference(bank_ids, max_steps=10):
    return [expected_length(b, max_steps) for b in bank_ids]


@pytest.mark.parametrize("num_envs", [1, 3, 8])
def test_fixed_episodes_give_the_same_results_whatever_the_number_of_environments(num_envs):
    ids = [0, 4, 3, 7, 1, 2, 11, 5, 9, 10]
    env, mgr, sim = env_of(num_envs=num_envs, bank_size=12, max_steps=10)
    res = run_fixed_episodes(env, ScriptedPolicy(), ids)
    assert [r["bank_id"] for r in res] == ids                                           # one result per id, in order
    for r, (length, term) in zip(res, reference(ids)):
        assert r["length"] == length and r["terminated"] == term and r["success"] == term
        assert r["episode_return"] == (1.0 if term else 0.0)


def test_fixed_episodes_fewer_than_environments_and_empty():
    env, mgr, sim = env_of(num_envs=5, bank_size=12)
    res = run_fixed_episodes(env, ScriptedPolicy(), [2, 3])
    assert [r["bank_id"] for r in res] == [2, 3] and res[1]["truncated"]
    assert run_fixed_episodes(env, ScriptedPolicy(), []) == []


def test_fixed_episodes_repeated_ids_are_run_separately():
    env, mgr, sim = env_of(num_envs=2, bank_size=12)
    res = run_fixed_episodes(env, ScriptedPolicy(), [0, 0, 0])
    assert len(res) == 3 and len({r["length"] for r in res}) == 1


def test_fixed_episodes_loop_guard():
    env, mgr, sim = env_of(num_envs=1, bank_size=12, max_steps=10)
    with pytest.raises(RuntimeError, match="not finished"):
        run_fixed_episodes(env, ScriptedPolicy(), [3], max_loops=4)
