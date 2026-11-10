"""VectorEnvManager: independence, auto-reset, final observations, frozen environments, start-state sampling. Pure."""
import numpy as np
import pytest

from fakes import ScriptedPolicy, expected_length, make_manager, trigger
from src.envs import protocol as P


def run_until(mgr, steps, policy=None, auto_reset=None):
    policy = policy or ScriptedPolicy()
    res = mgr.reset()
    out = [res]
    for _ in range(steps):
        res = mgr.step(policy(res["obs"]), auto_reset=auto_reset)
        out.append(res)
    return out


# ------------------------------------------------------------------------------------------------------------- reset
def test_first_reset_writes_the_bank_and_takes_no_physics_step():
    mgr, sim = make_manager(num_envs=4, bank_size=12)
    res = mgr.reset()
    assert sim.physics_steps == 0
    ids = res["start_id"]
    assert ((0 <= ids) & (ids < 12)).all() and (sim.tag == ids).all()                  # entries written in the sim (slots draw independently: repeats across slots are allowed)
    assert (res["obs"][P.JOINT_POSITION][:, 2] == ids).all()                           # the observation IS the bank's
    assert not res["terminated"].any() and res["acting"].all() and (res["t"] == 0).all() and not res["all_done"]
    assert res["final_ids"].size == 0


def test_partial_reset_touches_only_those_environments_and_does_not_step_physics():
    mgr, sim = make_manager(num_envs=4, bank_size=12)
    pol = ScriptedPolicy()
    res = mgr.reset()
    for _ in range(3):
        res = mgr.step(pol(res["obs"]))
    before = sim.snapshot()
    steps_before, t_before = sim.physics_steps, mgr.tracker.t.copy()
    res2 = mgr.reset(env_ids=[1, 3], bank_ids=[7, 8])
    after = sim.snapshot()
    for a, b in zip(before, after):
        assert (a[[0, 2]] == b[[0, 2]]).all()                                            # untouched environments: identical
    assert after[2][[1, 3]].tolist() == [7, 8] and after[1][[1, 3]].tolist() == [0, 0]
    assert sim.physics_steps == steps_before
    assert res2["t"].tolist() == [t_before[0], 0, t_before[2], 0]                         # counters: only the reset ones restart
    assert res2["start_id"][[1, 3]].tolist() == [7, 8]
    assert (res2["obs"][P.JOINT_POSITION][[1, 3], 2] == [7, 8]).all()                    # reset rows from the bank
    assert (res2["obs"][P.JOINT_POSITION][[0, 2], 1] == t_before[[0, 2]]).all()          # other rows: their real current state


def test_reset_with_no_ids_is_a_noop_that_reports_the_state():
    mgr, sim = make_manager()
    mgr.reset()
    n_calls = len(sim.restore_calls)
    res = mgr.reset(env_ids=[])
    assert len(sim.restore_calls) == n_calls and res["obs"][P.JOINT_POSITION].shape[0] == 4


def test_reset_argument_validation():
    mgr, _ = make_manager(num_envs=3, bank_size=5)
    with pytest.raises(RuntimeError, match="first reset"):
        mgr.reset(env_ids=[0])
    with pytest.raises(RuntimeError, match="before reset"):
        mgr.step(np.zeros((3, 8), np.float32))
    mgr.reset()
    with pytest.raises(ValueError, match="distinct"):
        mgr.reset(env_ids=[0, 0])
    with pytest.raises(ValueError, match="distinct"):
        mgr.reset(env_ids=[3])
    with pytest.raises(ValueError, match="bank_ids"):
        mgr.reset(env_ids=[0], bank_ids=[5])
    with pytest.raises(ValueError, match="bank_ids"):
        mgr.reset(env_ids=[0, 1], bank_ids=[1])
    with pytest.raises(ValueError, match="seed"):
        mgr.reset(env_ids=[0], seed=3)
    with pytest.raises(ValueError):
        mgr.reset(seed=-1)


def test_step_argument_validation():
    mgr, _ = make_manager(num_envs=3)
    mgr.reset()
    with pytest.raises(ValueError, match="shape"):
        mgr.step(np.zeros((2, 8), np.float32))
    bad = np.zeros((3, 8), np.float32)
    bad[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mgr.step(bad)


# ------------------------------------------------------------------------------------------------------- auto-reset
@pytest.mark.parametrize("hold", [1, 2])
def test_every_episode_has_the_predicted_length_and_outcome(hold):
    mgr, sim = make_manager(num_envs=4, bank_size=12, max_steps=10, hold=hold, seed=3)
    out = run_until(mgr, 60)
    started = {e: [int(out[0]["start_id"][e])] for e in range(4)}
    seen = 0
    for k, res in enumerate(out[1:], start=1):
        for e in res["final_ids"]:
            tag = started[e][-1]
            length, term = expected_length(tag, 10, hold)
            assert res["t"][e] == length, (tag, length, res["t"][e])
            assert bool(res["terminated"][e]) == term and bool(res["truncated"][e]) == (not term)
            assert bool(res["success"][e]) == term
            started[e].append(int(res["start_id"][e]))
            seen += 1
    assert seen >= 12


def test_auto_reset_returns_the_new_start_in_obs_and_the_last_observation_in_final_obs():
    mgr, sim = make_manager(num_envs=3, bank_size=9, max_steps=6)
    pol = ScriptedPolicy()
    res = mgr.reset()
    tags = res["start_id"].copy()
    for _ in range(40):
        prev_obs = {k: np.array(v) for k, v in res["obs"].items()}
        res = mgr.step(pol(res["obs"]))
        done = res["terminated"] | res["truncated"]
        for e in np.flatnonzero(done):
            j = list(res["final_ids"]).index(e)
            fin = res["final_obs"][P.JOINT_POSITION][j]
            assert fin[2] == tags[e]                                                    # final obs: still the ENDED episode
            assert fin[1] == prev_obs[P.JOINT_POSITION][e, 1] + 1                       # one step after the previous observation
            assert res["obs"][P.JOINT_POSITION][e, 1] == 0                              # new episode: clock 0
            assert res["obs"][P.JOINT_POSITION][e, 2] == res["start_id"][e]
            assert res["t"][e] == fin[1]                                                # t describes the ended episode
            assert (res["obs"][P.EXTERIOR_IMAGE][e] == res["start_id"][e] % 251).all()  # images are the bank's too
            tags[e] = res["start_id"][e]
        assert set(res["final_ids"].tolist()) == set(np.flatnonzero(done).tolist())


def test_a_step_that_ends_one_episode_does_not_disturb_the_others():
    mgr, sim = make_manager(num_envs=4, bank_size=12, max_steps=10)
    pol = ScriptedPolicy()
    res = mgr.reset()
    ends = 0
    for _ in range(25):
        before = sim.clock.copy()
        res = mgr.step(pol(res["obs"]))
        ended = np.zeros(4, bool)
        ended[res["final_ids"]] = True
        ends += int(ended.sum())
        # a restored env restarts at clock 0; every other env's clock grows by exactly 1 per step
        assert ((sim.clock == before + 1) | (ended & (sim.clock == 0))).all()
    assert ends >= 4


def test_t_counts_valid_steps_of_the_current_episode_and_start_id_describes_obs():
    mgr, sim = make_manager(num_envs=2, bank_size=6, max_steps=5)
    pol = ScriptedPolicy()
    res = mgr.reset()
    for _ in range(30):
        res = mgr.step(pol(res["obs"]))
        assert (res["start_id"] == sim.tag).all()                                      # start_id always matches what runs now


# -------------------------------------------------------------------------------------------- start-state sampling
def sequences(mgr, n_resets):
    """The start ids that env slots got, in order, by repeated full resets."""
    seqs = [[] for _ in range(mgr.n)]
    for _ in range(n_resets):
        r = mgr.reset()
        for e in range(mgr.n):
            seqs[e].append(int(r["start_id"][e]))
    return seqs


def test_each_slot_uses_every_bank_entry_once_before_repeating():
    mgr, _ = make_manager(num_envs=3, bank_size=7, seed=5)
    for seq in sequences(mgr, 14):
        assert sorted(seq[:7]) == list(range(7)) and sorted(seq[7:]) == list(range(7))


def test_start_ids_of_a_slot_do_not_depend_on_how_many_other_environments_exist():
    a = sequences(make_manager(num_envs=2, bank_size=10, seed=9)[0], 6)
    b = sequences(make_manager(num_envs=8, bank_size=10, seed=9)[0], 6)
    assert a[0] == b[0] and a[1] == b[1]
    assert b[0] != b[1] and b[2] != b[3]                                               # slots differ from one another


def test_seed_reproduces_and_changes_the_order():
    s1 = sequences(make_manager(seed=1)[0], 4)
    s1b = sequences(make_manager(seed=1)[0], 4)
    s2 = sequences(make_manager(seed=2)[0], 4)
    assert s1 == s1b and s1 != s2
    mgr, _ = make_manager(seed=1)
    mgr.reset()
    again = mgr.reset(seed=1)["start_id"].tolist()
    assert again == [s[0] for s in s1]                                                  # reset(seed=) restarts the streams


def test_the_order_does_not_depend_on_when_other_slots_reset():
    ref, _ = make_manager(num_envs=3, bank_size=10, max_steps=4, seed=4)
    other, _ = make_manager(num_envs=3, bank_size=10, max_steps=4, seed=4)
    ref.reset(), other.reset()
    for _ in range(2):
        ref.reset(env_ids=[0])
    other.reset(env_ids=[1]), other.reset(env_ids=[2]), other.reset(env_ids=[0]), other.reset(env_ids=[0])
    assert ref.start_id[0] == other.start_id[0]                                         # slot 0's 3rd draw, whatever the others did


def test_a_slots_sequence_is_unaffected_by_reshuffles_of_other_slots():
    """Small bank: slots exhaust their permutation (reshuffle) at different times; a shared generator would couple them."""
    a, _ = make_manager(num_envs=3, bank_size=3, seed=2)
    b, _ = make_manager(num_envs=3, bank_size=3, seed=2)
    a.reset(), b.reset()
    seq_a, seq_b = [int(a.start_id[0])], [int(b.start_id[0])]
    for _ in range(8):
        seq_a.append(int(a.reset(env_ids=[0])["start_id"][0]))
    for _ in range(8):
        b.reset(env_ids=[1]), b.reset(env_ids=[2]), b.reset(env_ids=[2])
        seq_b.append(int(b.reset(env_ids=[0])["start_id"][0]))
    assert seq_a == seq_b


def test_explicit_bank_ids_do_not_consume_the_random_stream():
    a, _ = make_manager(num_envs=2, bank_size=10, seed=4)
    b, _ = make_manager(num_envs=2, bank_size=10, seed=4)
    a.reset(), b.reset()
    a.reset(env_ids=[0], bank_ids=[9])
    nxt_a = a.reset(env_ids=[0])["start_id"][0]
    nxt_b = b.reset(env_ids=[0])["start_id"][0]
    assert nxt_a == nxt_b


# ---------------------------------------------------------------------------------------- auto_reset=False (evaluation)
def test_without_auto_reset_an_ended_env_is_held_until_the_client_resets_it():
    mgr, sim = make_manager(num_envs=3, bank_size=12, max_steps=10, auto_reset=False)
    pol = ScriptedPolicy()
    res = mgr.reset()
    res = mgr.reset(env_ids=[0, 1, 2], bank_ids=[0, 4, 3])      # tags: 0 -> trigger 2 (ends at 3), 4 -> trigger 6, 3 -> never
    assert [trigger(t) for t in (0, 4, 3)] == [2, 6, 10 ** 9]
    done_at = {}
    for k in range(1, 12):
        res = mgr.step(pol(res["obs"]))
        for e in np.flatnonzero(res["terminated"] | res["truncated"]):
            done_at[int(e)] = (k, bool(res["terminated"][e]), int(res["t"][e]))
        assert res["final_ids"].size == 0                                               # nothing is reset by itself
        if 0 in done_at and k > done_at[0][0]:
            assert not res["acting"][0] and res["t"][0] == done_at[0][2] and res["success"][0]
            assert sim.q0[0] == 1.0                                                     # held: last action repeated
        if res["all_done"]:
            break
    assert done_at == {0: (3, True, 3), 1: (7, True, 7), 2: (10, False, 10)}
    assert res["all_done"]
    steps = sim.physics_steps
    idle = mgr.step(pol(res["obs"]))                                                    # nothing runs: a no-op, no physics step
    assert sim.physics_steps == steps and not idle["acting"].any() and idle["all_done"] and not idle["reward"].any()
    assert idle["t"].tolist() == res["t"].tolist()
    res = mgr.reset(env_ids=[1], bank_ids=[0])                                          # restart one: it runs, the others stay frozen
    assert res["acting"].tolist() == [False, True, False] and not res["all_done"]
    res = mgr.step(pol(res["obs"]))
    assert res["acting"].tolist() == [False, True, False] and res["t"].tolist() == [3, 1, 10]


def test_a_held_env_ignores_new_actions_while_the_running_ones_follow_them():
    mgr, sim = make_manager(num_envs=2, bank_size=12, max_steps=10, auto_reset=False)
    pol = ScriptedPolicy()
    mgr.reset()
    res = mgr.reset(env_ids=[0, 1], bank_ids=[0, 3])         # env 0 succeeds at step 3, env 1 never
    for _ in range(3):
        res = mgr.step(pol(res["obs"]))
    assert res["terminated"].tolist() == [True, False] and sim.q0.tolist() == [1.0, 0.0]
    res = mgr.step(np.full((2, 8), 0.5, np.float32))
    assert sim.q0.tolist() == [1.0, 0.5]                      # env 0 kept its last action, env 1 took the new one


def test_per_step_auto_reset_override():
    mgr, _ = make_manager(num_envs=2, bank_size=6, max_steps=3, auto_reset=True)
    res = mgr.reset(env_ids=None, bank_ids=None)
    res = mgr.reset(env_ids=[0, 1], bank_ids=[3, 3])                                   # never succeed -> time out at step 3
    pol = ScriptedPolicy()
    for _ in range(2):
        res = mgr.step(pol(res["obs"]), auto_reset=False)
    res = mgr.step(pol(res["obs"]), auto_reset=False)
    assert res["truncated"].all() and res["final_ids"].size == 0 and res["all_done"]


def test_a_debug_op_that_moves_the_sim_invalidates_the_episodes():
    mgr, _ = make_manager()
    mgr.reset()
    mgr.invalidate()
    with pytest.raises(RuntimeError, match="before reset"):
        mgr.step(np.zeros((4, 8), np.float32))


def test_timings_accumulate_and_reset():
    mgr, _ = make_manager()
    mgr.reset()
    mgr.step(np.zeros((4, 8), np.float32))
    t = mgr.timings()
    assert t["step_physics"]["calls"] == 1 and t["observe"]["calls"] == 1 and t["restore"]["calls"] == 1
    assert mgr.timings(reset=True) and mgr.timings() == {}
