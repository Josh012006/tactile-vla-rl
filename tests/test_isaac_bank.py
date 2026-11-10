"""Start-state bank on the REAL simulator (needs Isaac Lab + GPU:  pytest tests/test_isaac_bank.py --isaac -s).

What is verified, in order of importance:
  1. a restore writes exactly the captured state (and nothing else moves): test_restore_writes_exactly_the_bank_state,
     test_restore_leaves_the_other_environments_bit_identical;
  2. the state captured is COMPLETE and slot-independent: test_capture_is_complete, test_restore_matches_the_real_reset
     (the trajectory after a restore follows the one after the real FORGE reset, within the repeatability floor of the
     simulator itself), and the same with the entries written in the NEXT slot (env-frame content);
  3. the running environments keep behaving when another one is reset: test_resets_do_not_disturb_running_environments;
  4. the per-environment episode logic works on the real thing: test_asynchronous_auto_reset_on_the_real_simulator,
     test_long_run_is_stable.

TOLERANCES are first guesses (nobody has run this on the cluster yet). The floor (two REAL resets with the same seed, same
slot) is always measured and printed; a comparison passes if its error is within max(absolute tolerance, FLOOR_FACTOR x floor).
After the first run, tighten TOL to what you see. Every number is also written to reports/restore_equivalence.json.
"""
import json

import numpy as np
import pytest

from src.envs import protocol as P

pytestmark = pytest.mark.isaac

N = 4
BANK = 16                                   # 4 real resets of 4 environments
FLOOR_FACTOR = 10.0
TOL = {"joint_pos": 5e-3, "joint_vel": 5e-2, "held_pose": 2e-3, "peg_socket_offset": 2e-3, "ft_force": 3.0, "tactile": 3.0,
       "image_mean_exterior": 2.0, "image_mean_wrist": 4.0}
STATE_TOL = {"sim/": 1e-4, "env/": 1e-5, "envdict/": 0.0}        # restore vs bank, read back from the simulator
PROBE_STEPS = 24


@pytest.fixture(scope="module")
def server(isaac_pool):
    return isaac_pool.get(N, bank_size=BANK)


@pytest.fixture
def env(server):
    c = server.client()
    yield c
    c.close()


def scripted(hold0, k, amplitude=0.02, period=24):
    a = hold0.copy()
    a[:, 0:7] += amplitude * np.sin(2 * np.pi * k / period + np.arange(7))
    return a


def run_script(env, steps, interrupt=None):
    """Hold + sinusoid for `steps` steps from the CURRENT episodes; returns {key: (steps+1, N, ...)}. interrupt(k) is called
    before step k (to reset an environment mid-way)."""
    hold0 = env.hold_actions()
    res_obs = env.observe()
    traj = {k: [np.array(v)] for k, v in res_obs.items() if "image" not in k}
    for k in range(steps):
        if interrupt is not None:
            interrupt(k)
        res = env.step(scripted(hold0, k), auto_reset=False)
        for key in traj:
            traj[key].append(np.array(res.obs[key]))
    return {k: np.stack(v) for k, v in traj.items()}


def worst(a, b, rows=None):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if rows is not None:
        d = d[:, rows]
    return float(d.max())


def passes(err, floor, key):
    return err <= max(TOL[key], FLOOR_FACTOR * floor)


# ------------------------------------------------------------------------------------------------------------ 1. exact
def test_capture_is_complete(env):
    info = env.debug("bank_info")[0]
    assert info["unlisted"] == [], (
        f"per-environment tensors of the FORGE environment that start_bank.py neither captures nor ignores: {info['unlisted']}. "
        "Each one is start state (add to STATE_NAMES), recomputed every step (DERIVED_NAMES) or irrelevant (IGNORED_NAMES).")
    must = {"sim/joint_pos", "sim/joint_vel", "sim/held_pose", "sim/fixed_pose", "env/ctrl_target_joint_pos",
            "env/fixed_pos_obs_frame", "env/init_fixed_pos_obs_noise", "env/contact_penalty_thresholds", "env/ema_factor",
            "env/task_prop_gains", "env/prev_fingertip_pos", "env/force_sensor_smooth", "env/episode_length_buf"}
    assert must <= set(info["captured"]), sorted(must - set(info["captured"]))
    assert info["size"] == BANK and info["build_num_envs"] == N


def test_bank_entries_are_distinct_and_finite(env):
    ids = np.arange(BANK)
    state = env.conns[0].debug("bank_state", bank_ids=ids)
    obs = env.conns[0].debug("bank_obs", bank_ids=ids)
    for k, v in {**state, **obs}.items():
        assert np.isfinite(v.astype(np.float64)).all(), k
    assert np.unique(np.round(state["sim/held_pose"][:, :3], 6), axis=0).shape[0] == BANK        # peg placed differently each time
    assert np.unique(np.round(state["sim/joint_pos"], 6), axis=0).shape[0] == BANK
    assert np.unique(np.round(state["env/init_fixed_pos_obs_noise"], 7), axis=0).shape[0] == BANK


@pytest.mark.parametrize("permute", [False, True], ids=["same_slots", "next_slot"])
def test_restore_writes_exactly_the_bank_state(env, permute):
    from src.envs.start_bank import DERIVED_NAMES
    env.reset(seed=0)
    bank_ids = np.array([3, 1, 2, 0]) if not permute else np.array([13, 6, 9, 14])
    res = env.reset(env_ids=np.arange(N), bank_ids=bank_ids)
    assert res.start_id.tolist() == bank_ids.tolist()
    now = env.conns[0].debug("env_state", env_ids=np.arange(N))
    ref = env.conns[0].debug("bank_state", bank_ids=bank_ids)
    for key in ref:
        tol = next(t for prefix, t in STATE_TOL.items() if key.startswith(prefix))
        a, b = now[key].astype(np.float64), ref[key].astype(np.float64)
        if key.startswith("env/") and key[4:] in DERIVED_NAMES:
            continue                                  # recomputed from the simulation at the next step, compared by the trajectory tests
        assert np.abs(a - b).max() <= tol, f"{key}: restored state differs from the bank by {np.abs(a - b).max():.3e}"
    bank_obs = env.conns[0].debug("bank_obs", bank_ids=bank_ids)
    for key, v in bank_obs.items():
        assert np.array_equal(res.obs[key], v), key                          # first observation == the bank's, bit for bit


def test_restore_leaves_the_other_environments_bit_identical(env):
    env.reset(seed=0)
    env.reset(env_ids=np.arange(N), bank_ids=[0, 1, 2, 3])
    env.step(env.hold_actions())
    before = env.conns[0].debug("env_state", env_ids=np.arange(N))
    env.reset(env_ids=[1], bank_ids=[9])
    after = env.conns[0].debug("env_state", env_ids=np.arange(N))
    for key in before:
        for other in (0, 2, 3):
            assert np.array_equal(before[key][other], after[key][other]), f"{key}: environment {other} changed"
    assert not np.array_equal(before["sim/held_pose"][1], after["sim/held_pose"][1])        # the reset one did change


# ------------------------------------------------------------------------------------------------- 2. equivalence (probe)
def probe(env, mode, rnd):
    return env.conns[0].debug("probe", mode=mode, round=rnd, steps=PROBE_STEPS)


def compare(a, b):
    out = {}
    for key in TOL:
        if key not in a:
            continue
        sl = slice(1, None) if key.startswith("image") else slice(None)           # images at step 0 are the reset state's, not rendered
        out[key] = worst(a[key][sl], b[key][sl])
    return out


def test_restore_matches_the_real_reset(env, reports_dir):
    report = {"steps": PROBE_STEPS, "n": N, "floor_factor": FLOOR_FACTOR, "tol": TOL, "rounds": {}}
    failures = []
    for rnd in (0, 2):
        real = probe(env, "real", rnd)
        real2 = probe(env, "real", rnd)                                      # same seed, same slots: the simulator's own floor
        restored = probe(env, "restore", rnd)
        permuted = probe(env, "restore_permuted", rnd)
        floor = compare(real, real2)
        err = compare(real, restored)
        rolled = {k: np.roll(v, -1, axis=1) if v.ndim > 1 and v.shape[1] == N else v for k, v in real.items()}
        err_perm = compare(rolled, permuted)           # slot i of `permuted` holds entry of slot i+1: compare with real's slot i+1
        report["rounds"][rnd] = {"floor": floor, "restore_vs_real": err, "restore_next_slot_vs_real": err_perm}
        for label, e in (("restore", err), ("next-slot restore", err_perm)):
            for key, value in e.items():
                if not passes(value, floor[key], key):
                    failures.append(f"round {rnd} {label} {key}: {value:.3e} > max({TOL[key]}, {FLOOR_FACTOR}x floor {floor[key]:.3e})")
        for name, traj in (("restored", restored), ("permuted", permuted)):
            for key, v in traj.items():
                assert np.isfinite(v.astype(np.float64)).all(), f"{name} {key} not finite"
    (reports_dir / "restore_equivalence.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report["rounds"], indent=1))
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------------------------------- 3. isolation of running environments
def test_resets_do_not_disturb_running_environments(env, reports_dir):
    ids = [0, 1, 2, 3]

    def start():
        env.reset(seed=0)
        env.reset(env_ids=np.arange(N), bank_ids=ids)

    start()
    run_a = run_script(env, 30)
    start()
    run_a2 = run_script(env, 30)                                              # floor: identical protocol, no interruption
    start()
    calls = []

    def interrupt(k):
        if k in (10, 20):
            calls.append(k)
            env.reset(env_ids=[0], bank_ids=[5 + k // 10])

    run_b = run_script(env, 30, interrupt)
    assert calls == [10, 20]
    others = [1, 2, 3]
    rows = {}
    for key in ("observation/joint_position", "extra/peg_socket_offset", "extra/ft_force"):
        floor = worst(run_a[key], run_a2[key], others)
        err = worst(run_a[key], run_b[key], others)
        rows[key] = {"floor": floor, "with_resets": err}
        tol = {"observation/joint_position": TOL["joint_pos"], "extra/peg_socket_offset": TOL["peg_socket_offset"],
               "extra/ft_force": TOL["ft_force"]}[key]
        assert err <= max(tol, FLOOR_FACTOR * floor), f"{key}: running environments moved by {err:.3e} (floor {floor:.3e})"
    (reports_dir / "isolation.json").write_text(json.dumps(rows, indent=2))
    print("\n" + json.dumps(rows, indent=1))
    assert worst(run_a["observation/joint_position"], run_b["observation/joint_position"], [0]) > 0     # sanity: env 0 itself did change


def test_reset_order_is_reproducible_by_seed(env):
    a = env.reset(seed=7).start_id
    b = env.reset(seed=7).start_id
    c = env.reset(seed=8).start_id
    assert a.tolist() == b.tolist() and a.tolist() != c.tolist()


# ------------------------------------------------------------------------------------------- 4. per-environment episodes
@pytest.fixture(scope="module")
def short_server(isaac_pool):
    return isaac_pool.get(N, bank_size=BANK, max_steps_per_episode=6)


def test_asynchronous_auto_reset_on_the_real_simulator(short_server):
    with short_server.client() as env:
        res = env.reset(seed=0)
        res = env.reset(env_ids=np.arange(N), bank_ids=[0, 1, 2, 3])
        ended_at = {}
        for k in range(1, 20):
            if k == 4:                                                         # env 0 restarted 3 steps late
                res = env.reset(env_ids=[0], bank_ids=[4])
            res = env.step(env.hold_actions())
            for j, e in enumerate(res.final_ids):
                ended_at.setdefault(int(e), []).append(k)
                assert res.truncated[e] and not res.terminated[e] and res.t[e] == 6
                bank = env.conns[0].debug("bank_obs", bank_ids=[int(res.start_id[e])])
                for key, v in bank.items():                                    # obs row = the new episode's bank observation
                    assert np.array_equal(res.obs[key][e], v[0]), key
                assert not np.array_equal(res.final_obs[P.WRIST_IMAGE][j], res.obs[P.WRIST_IMAGE][e])
            assert all(np.isfinite(v.astype(np.float64)).all() for v in res.obs.values())
        assert ended_at[1][0] == ended_at[2][0] == ended_at[3][0] == 6          # synchronous starts end together...
        assert ended_at[0][0] == 6 + 3                                          # ...the late one ends 3 steps later, alone
        assert all(b - a == 6 for e in ended_at for a, b in zip(ended_at[e], ended_at[e][1:]))      # then every 6 steps


def test_long_run_is_stable(server):
    from src.envs.collect import Collector
    with server.client() as env:
        rng = np.random.default_rng(0)
        hold = {"v": None}

        def policy(obs):
            if hold["v"] is None:
                hold["v"] = env.hold_actions()
            return hold["v"] + rng.normal(0, 0.01, hold["v"].shape).astype(np.float32)

        data = Collector(env, policy, mode="steps", store_next_obs=False).collect(330)
        assert len(data.episodes) >= N * 2                                      # max_steps 150: every env finished twice
        assert all(ep.length == 150 and ep.truncated for ep in data.episodes if not ep.terminated)
        for k, v in data.obs.items():
            assert np.isfinite(v.astype(np.float64)).all(), k
        assert np.abs(data.obs[P.PEG_SOCKET_OFFSET]).max() < 1.0                # the peg never flies away
