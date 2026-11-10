"""Collector (steps / episodes modes) and run_fixed_rounds against a fake simulator through the real manager. Pure (no GPU)."""
import numpy as np
import pytest

from fakes import ScriptedPolicy, expected_length, make_manager, tag_of
from src.envs import protocol as P
from src.envs.collect import Collector, run_fixed_rounds

J = P.JOINT_POSITION


class DirectEnv:
    """The client interface (reset / step / num_envs / spec) straight on a manager: no network, StepResult objects."""

    def __init__(self, mgr):
        self.mgr, self.num_envs = mgr, mgr.n

    def reset(self, seed=None, round=None):
        return P.StepResult.from_wire(self.mgr.reset(seed, round))

    def step(self, actions, auto_reset=None):
        return P.StepResult.from_wire(self.mgr.step(actions, auto_reset))


def env_of(**kw):
    mgr, sim = make_manager(**kw)
    return DirectEnv(mgr), mgr, sim


# ------------------------------------------------------------------------------------------------------------ steps mode
def test_steps_mode_shapes_and_every_environment_gets_exactly_T_steps():
    env, mgr, sim = env_of(num_envs=4, max_steps=10)
    c = Collector(env, ScriptedPolicy(), mode="steps")
    r = c.collect(25)
    assert r.reward.shape == (25, 4) and r.acting.shape == (25, 4) and r.round.shape == (25, 4)
    assert r.obs[J].shape == (25, 4, 7) and r.next_obs[J].shape == (25, 4, 7) and r.last_obs[J].shape == (4, 7)
    assert sim.physics_steps == 25
    assert c.total_env_steps + c.total_idle_steps == 25 * 4 and c.total_env_steps == int(r.acting.sum())


def test_waiting_environments_are_marked_not_acting_and_a_round_restarts_all_of_them():
    env, mgr, sim = env_of(num_envs=4, max_steps=10, seed=1)
    r = Collector(env, ScriptedPolicy(), mode="steps").collect(12)
    lengths = [expected_length(tag_of(1, e), 10)[0] for e in range(4)]
    for e in range(4):
        k = np.arange(1, 11)                                       # steps of round 0 (transition k is row k - 1)
        assert (r.acting[:10, e] == (k <= lengths[e])).all()
    assert r.acting[10:].all()                                     # rows 10 and 11: round 1 is running everywhere
    assert (r.round[:10] == 0).all() and (r.round[10:] == 1).all()
    assert c_idle(r) == sum(10 - n for n in lengths)


def c_idle(r):
    return int((~r.acting).sum())


def test_next_obs_is_the_true_next_observation_and_never_crosses_a_round():
    env, mgr, sim = env_of(num_envs=3, max_steps=6, seed=2)
    r = Collector(env, ScriptedPolicy(), mode="steps").collect(30)
    done = r.done
    assert done.any()
    assert (r.next_obs[J][done][:, 1] == r.t[done]).all()           # clock of the last observation = steps of the episode
    first = ~done & r.acting & (r.t > 0)
    assert (r.next_obs[J][first][:, 1] == r.t[first]).all()         # inside an episode, next obs is the following observation
    ok = r.acting                                                   # (a waiting row is not a transition: its obs is the next round's)
    assert (r.next_obs[J][..., 2][ok] == r.obs[J][..., 2][ok]).all()   # same start state: never the first obs of another round


def test_flags_and_lengths_are_consistent_with_the_predicted_episodes():
    env, mgr, sim = env_of(num_envs=4, max_steps=10, seed=5)
    r = Collector(env, ScriptedPolicy(), mode="steps").collect(40)
    assert len(r.episodes) >= 8
    for ep in r.episodes:
        length, term = expected_length(tag_of(5 + ep.round, ep.env), 10)
        assert (ep.length, ep.terminated, ep.truncated, ep.success) == (length, term, not term, term)
        assert ep.episode_return == (1.0 if term else 0.0)


def test_the_round_of_a_transition_is_the_round_of_its_own_episode():
    env, mgr, sim = env_of(num_envs=3, max_steps=6, seed=4)
    r = Collector(env, ScriptedPolicy(), mode="steps").collect(30)
    seed_of_row = 4 + r.round
    tags = r.obs[J][:, :, 2].astype(np.int64)
    expected = np.array([[tag_of(seed_of_row[k, e], e) for e in range(3)] for k in range(30)])
    assert (tags == expected).all()


def test_the_episodes_in_flight_continue_across_calls_and_the_policy_version_is_stamped():
    env, mgr, sim = env_of(num_envs=2, max_steps=10)
    c = Collector(env, ScriptedPolicy(), mode="steps")
    a = c.collect(3)
    c.version += 1
    b = c.collect(3)
    assert a.version.tolist() == [0, 0, 0] and b.version.tolist() == [1, 1, 1]
    running = b.acting[0]
    assert running.any() and not running.all()                      # env 0 (3 steps) ended in the first call, env 1 is still going
    assert (b.t[0][running] == a.t[-1][running] + 1).all()          # the call continued the episode: t did not restart
    assert (b.obs[J][0] == a.last_obs[J]).all()


def test_extras_from_the_policy_are_stored_per_step():
    env, mgr, sim = env_of(num_envs=2)
    inner = ScriptedPolicy()
    r = Collector(env, lambda o: (inner(o), {"logp": np.full(2, -1.5, np.float32)}), mode="steps").collect(4)
    assert r.extras["logp"].shape == (4, 2) and (r.extras["logp"] == -1.5).all()


def test_store_next_obs_false_saves_the_memory():
    env, mgr, sim = env_of(num_envs=2)
    assert Collector(env, ScriptedPolicy(), mode="steps", store_next_obs=False).collect(3).next_obs is None


def test_bad_policy_output_and_arguments():
    env, mgr, sim = env_of(num_envs=2)
    with pytest.raises(ValueError, match="num_envs"):
        Collector(env, lambda o: np.zeros((3, 8), np.float32)).collect(1)
    with pytest.raises(ValueError, match="amount"):
        Collector(env, ScriptedPolicy()).collect(0)
    with pytest.raises(ValueError, match="mode"):
        Collector(env, ScriptedPolicy(), mode="chunks")


# -------------------------------------------------------------------------------------------------------- episodes mode
def test_episodes_mode_returns_complete_episodes_with_their_final_observation():
    env, mgr, sim = env_of(num_envs=4, max_steps=10, seed=3)
    eps = Collector(env, ScriptedPolicy(), mode="episodes").collect(8)
    assert len(eps) >= 8
    for ep in eps:
        length, term = expected_length(tag_of(3 + ep.round, ep.env), 10)
        assert ep.length == length and ep.terminated == term and ep.truncated == (not term) and ep.success == term
        assert ep.obs[J][0, 1] == 0 and ep.obs[J][0, 2] == tag_of(3 + ep.round, ep.env)       # starts at the start state, clock 0
        assert ep.final_obs[J][1] == length and ep.final_obs[J][2] == ep.obs[J][0, 2]         # final obs: after the last action
        assert ep.obs[J].shape == (length, 7) and ep.action.shape == (length, 8) and ep.reward.shape == (length,)
        assert ep.reward.sum() == (1.0 if term else 0.0)


def test_a_whole_round_is_N_episodes_one_per_environment_and_waiting_adds_no_rows():
    env, mgr, sim = env_of(num_envs=4, max_steps=10, seed=1)
    lengths = sorted(expected_length(tag_of(1, e), 10)[0] for e in range(4))
    assert lengths == [3, 4, 7, 10]
    c = Collector(env, ScriptedPolicy(), mode="episodes")
    eps = c.collect(4)
    assert sorted(ep.env for ep in eps) == [0, 1, 2, 3] and {ep.round for ep in eps} == {0}
    assert sorted(ep.length for ep in eps) == lengths               # nobody was padded to the length of the slowest
    assert c.total_env_steps == sum(lengths) and c.total_idle_steps == 4 * 10 - sum(lengths)
    assert mgr.round == 1                                           # the next round has already started


def test_episodes_in_flight_continue_after_the_update_and_mix_policy_versions():
    env, mgr, sim = env_of(num_envs=3, max_steps=10, seed=1)       # tags 3, 4, 5: lengths 10, 7, 3
    assert [expected_length(tag_of(1, e), 10)[0] for e in range(3)] == [10, 7, 3]
    c = Collector(env, ScriptedPolicy(), mode="episodes")
    first = c.collect(2)                                            # stops when the second episode ends, env 0 is still running
    assert {ep.env for ep in first} == {1, 2}
    c.version += 1
    rest = c.collect(1)
    assert [ep.env for ep in rest] == [0] and rest[0].length == 10
    assert set(rest[0].version.tolist()) == {0, 1}                  # the update fell inside this episode


# ----------------------------------------------------------------------------------------------------------- fixed rounds
def test_fixed_rounds_give_the_same_episodes_every_time_and_in_order():
    env, mgr, sim = env_of(num_envs=3, max_steps=10)
    res = run_fixed_rounds(env, ScriptedPolicy(), [0, 2, 5], seed=7)
    assert [(r["round"], r["env"]) for r in res] == [(r, e) for r in (0, 2, 5) for e in range(3)]
    for r in res:
        length, term = expected_length(tag_of(7 + r["round"], r["env"]), 10)
        assert (r["length"], r["terminated"], r["truncated"], r["success"]) == (length, term, not term, term)
    env2, mgr2, sim2 = env_of(num_envs=3, max_steps=10)
    env2.reset(seed=99)                                             # something else ran before
    assert run_fixed_rounds(env2, ScriptedPolicy(), [0, 2, 5], seed=7) == res
    assert sim.reset_calls == [7, 9, 12]                            # each round is reset explicitly, never skipped or repeated


def test_fixed_rounds_repeated_rounds_are_run_separately_and_empty_is_empty():
    env, mgr, sim = env_of(num_envs=2)
    res = run_fixed_rounds(env, ScriptedPolicy(), [1, 1], seed=0)
    assert len(res) == 4 and res[:2] == [{**r} for r in res[2:]]
    assert run_fixed_rounds(env, ScriptedPolicy(), [], seed=0) == []
