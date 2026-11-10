"""VectorEnvManager: rounds, held environments, final observations, replay by (seed, round). Pure (no GPU)."""
import numpy as np
import pytest

from fakes import ScriptedPolicy, expected_length, make_manager, tag_of, trigger
from src.envs import protocol as P

J = P.JOINT_POSITION


def episode_ends(mgr, rounds, auto_reset=None):
    """Run `rounds` complete rounds with the scripted policy: {(round, env): (length, terminated)} as the manager reports it."""
    policy, res, out = ScriptedPolicy(), mgr.reset(), {}
    while mgr.round < rounds:
        r = mgr.round
        res = mgr.step(policy(res["obs"]), auto_reset=auto_reset)
        for e in res["final_ids"]:
            out[(r, int(e))] = (int(res["t"][e]), bool(res["terminated"][e]))
    return out


# ------------------------------------------------------------------------------------------------------------- reset
def test_first_reset_is_a_real_reset_of_round_0_and_takes_no_step():
    mgr, sim = make_manager(num_envs=4, seed=7)
    res = mgr.reset()
    assert sim.reset_calls == [7] and sim.physics_steps == 0
    assert (res["round_id"] == 0).all() and mgr.round == 0
    assert (res["obs"][J][:, 2] == [tag_of(7, e) for e in range(4)]).all()               # the observation is the simulator's
    assert not res["terminated"].any() and res["acting"].all() and (res["t"] == 0).all() and not res["all_done"]
    assert res["final_ids"].size == 0


def test_reset_arguments_select_the_seed_and_the_round():
    mgr, sim = make_manager(num_envs=2, seed=7)
    mgr.reset()                                    # round 0 of seed 7
    mgr.reset()                                    # the next round
    mgr.reset(seed=100)                            # a new seed starts at round 0
    mgr.reset(round=4)                             # same seed, another position
    mgr.reset(seed=100, round=1)
    mgr.reset(seed=5, round=2)
    assert sim.reset_calls == [7, 8, 100, 104, 101, 7]
    assert mgr.round == 2 and mgr.seed == 5


def test_same_seed_and_round_give_the_same_start_whatever_ran_before():
    a, _ = make_manager(num_envs=3)
    b, _ = make_manager(num_envs=3)
    a.reset(seed=11, round=5)
    b.reset(seed=2)
    for _ in range(4):
        b.step(np.zeros((3, 8), np.float32))
    b.reset(seed=11, round=5)
    assert (a.observe()["obs"][J] == b.observe()["obs"][J]).all()


def test_reset_argument_validation():
    mgr, _ = make_manager(num_envs=3)
    with pytest.raises(ValueError, match="seed"):
        mgr.reset(seed=-1)
    with pytest.raises(ValueError, match="round"):
        mgr.reset(seed=0, round=-1)
    with pytest.raises(ValueError, match="round"):
        mgr.reset(seed=0, round=P.MAX_ROUND)
    with pytest.raises(ValueError, match="seed"):
        make_manager(seed=-3)


def test_step_argument_validation():
    mgr, _ = make_manager(num_envs=2)
    with pytest.raises(RuntimeError, match="before reset"):
        mgr.step(np.zeros((2, 8), np.float32))
    mgr.reset()
    with pytest.raises(ValueError, match="shape"):
        mgr.step(np.zeros((3, 8), np.float32))
    bad = np.zeros((2, 8), np.float32)
    bad[1, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mgr.step(bad)


# ---------------------------------------------------------------------------------------------------- episodes and rounds
@pytest.mark.parametrize("hold", [1, 2])
def test_every_episode_has_the_predicted_length_and_outcome(hold):
    mgr, _ = make_manager(num_envs=4, max_steps=10, hold=hold, seed=3)
    ends = episode_ends(mgr, rounds=5)
    assert len(ends) == 4 * 5
    for (r, e), got in ends.items():
        assert got == expected_length(tag_of(3 + r, e), 10, hold), (r, e)


def test_an_ended_environment_waits_and_the_round_restarts_when_the_last_one_ends():
    mgr, sim = make_manager(num_envs=4, max_steps=10, seed=1)
    policy = ScriptedPolicy()
    res = mgr.reset()
    tags = [tag_of(1, e) for e in range(4)]
    lengths = [expected_length(t, 10)[0] for t in tags]
    assert len(set(lengths)) > 1                                  # the test needs environments that end at different steps
    last = max(lengths)
    for k in range(1, last + 1):
        before_q0 = sim.q0.copy()
        res = mgr.step(policy(res["obs"]))
        for e in range(4):
            if k > lengths[e]:                                    # ended earlier: held, not acting, nothing about it changes
                assert not res["acting"][e] and res["round_id"][e] == (0 if k < last else 1)
        ended_now = [e for e in range(4) if lengths[e] == k]
        assert res["final_ids"].tolist() == ended_now
        if k < last:
            assert res["round_id"].tolist() == [0] * 4 and sim.reset_calls == [1] and not res["all_done"]
            for e in ended_now:                                   # its row of obs is its last observation, not a new start
                assert res["obs"][J][e, 1] == k
                assert (res["final_obs"][J][ended_now.index(e)] == res["obs"][J][e]).all()
            for e in range(4):
                if k > lengths[e]:
                    assert sim.q0[e] == before_q0[e]              # held: the action it was given is ignored
    # the last step of the round: every environment starts over, in the SAME call
    assert sim.reset_calls == [1, 2] and (res["round_id"] == 1).all() and mgr.round == 1
    assert (res["obs"][J][:, 1] == 0).all() and (res["obs"][J][:, 2] == [tag_of(2, e) for e in range(4)]).all()
    final_envs = [e for e in range(4) if lengths[e] == last]
    assert res["final_ids"].tolist() == final_envs
    assert (res["final_obs"][J][:, 1] == last).all()              # the last observation of the episodes that ended on this step
    assert res["terminated"].sum() + res["truncated"].sum() == len(final_envs)
    assert res["acting"].tolist() == [lengths[e] >= last for e in range(4)]          # who was running before this step
    assert (mgr.tracker.t == 0).all() and mgr.tracker.active.all()


def test_t_counts_the_valid_steps_of_each_episode_and_stays_frozen_after_it_ends():
    mgr, _ = make_manager(num_envs=3, max_steps=10, seed=2)
    res = mgr.reset(seed=2)
    lengths = [expected_length(tag_of(2, e), 10)[0] for e in range(3)]
    seen = {}
    for k in range(1, max(lengths) + 1):
        res = mgr.step(ScriptedPolicy()(res["obs"]), auto_reset=False)
        for e in range(3):
            seen[(k, e)] = int(res["t"][e])
    for (k, e), t in seen.items():
        assert t == min(k, lengths[e])


def test_without_auto_reset_nothing_starts_by_itself():
    mgr, sim = make_manager(num_envs=3, max_steps=10, seed=0, auto_reset=False)
    res = mgr.reset()
    policy, steps = ScriptedPolicy(), 0
    while not res["all_done"]:
        res = mgr.step(policy(res["obs"]))
        steps += 1
    assert sim.reset_calls == [0] and mgr.round == 0
    assert steps == max(expected_length(tag_of(0, e), 10)[0] for e in range(3))
    physics = sim.physics_steps
    res = mgr.step(policy(res["obs"]))                            # a no-op: no physics step, nobody acting
    assert sim.physics_steps == physics and not res["acting"].any() and res["all_done"]
    res = mgr.reset()                                             # the client starts the next round
    assert sim.reset_calls == [0, 1] and mgr.round == 1 and res["acting"].all() and not res["all_done"]


def test_per_step_auto_reset_override():
    mgr, sim = make_manager(num_envs=2, max_steps=3, auto_reset=True)
    mgr.reset(seed=3)
    res = None
    for _ in range(3):
        res = mgr.step(np.zeros((2, 8), np.float32), auto_reset=False)        # zero action: never inserted, timeout at step 3
    assert res["all_done"] and res["truncated"].all() and sim.reset_calls == [3]
    mgr2, sim2 = make_manager(num_envs=2, max_steps=3, auto_reset=False)
    mgr2.reset(seed=3)
    for _ in range(3):
        res = mgr2.step(np.zeros((2, 8), np.float32), auto_reset=True)        # the override restarts the round
    assert sim2.reset_calls == [3, 4] and not res["all_done"] and (res["round_id"] == 1).all()


# ------------------------------------------------------------------------------------------------------------- replay
def test_the_whole_run_is_a_function_of_the_seed():
    a, _ = make_manager(num_envs=3, max_steps=8, seed=4)
    b, _ = make_manager(num_envs=3, max_steps=8, seed=4)
    c, _ = make_manager(num_envs=3, max_steps=8, seed=5)
    assert episode_ends(a, 6) == episode_ends(b, 6) != episode_ends(c, 6)


def test_the_seeds_of_the_rounds_are_consecutive_whatever_the_policy_does():
    a, sa = make_manager(num_envs=2, max_steps=6, seed=9)
    b, sb = make_manager(num_envs=2, max_steps=6, seed=9)
    episode_ends(a, 4)                                            # scripted policy
    res = b.reset()
    while b.round < 4:                                            # a policy that never inserts: every round times out
        res = b.step(np.zeros((2, 8), np.float32))
    assert sa.reset_calls[:5] == sb.reset_calls[:5] == [9, 10, 11, 12, 13]


# ------------------------------------------------------------------------------------------------------------ plumbing
def test_a_debug_free_manager_reports_timings_per_phase_and_resets_them():
    mgr, _ = make_manager(num_envs=2, max_steps=2)
    mgr.reset()
    mgr.step(np.zeros((2, 8), np.float32))
    t = mgr.timings()
    assert t["reset"]["calls"] == 1 and t["step_physics"]["calls"] == 1 and t["inserted"]["calls"] == 1
    assert t["observe"]["calls"] == 2                             # one after the reset, one after the step
    assert mgr.timings(reset=True) and mgr.timings() == {}
    mgr.step(np.zeros((2, 8), np.float32))                        # second step ends the round: a reset inside the step
    assert mgr.timings()["reset"]["calls"] == 1
