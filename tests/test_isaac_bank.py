"""Start-state bank on the REAL simulator (needs Isaac Lab + GPU:  pytest tests/test_isaac_bank.py --isaac -s).

What is verified, in order of importance:
  1. a restore writes exactly the captured state (and nothing else moves): test_restore_writes_exactly_the_bank_state,
     test_restore_leaves_the_other_environments_bit_identical;
  2. the state captured is COMPLETE and slot-independent: test_capture_is_complete, test_restore_matches_the_real_reset
     (the trajectory after a restore follows the one after the real FORGE reset replayed in build order, within the
     repeatability floors of the real reset and of the restore themselves), and the same with the entries written in the
     NEXT slot (env-frame content);
  3. the running environments keep behaving when another one is reset: test_resets_do_not_disturb_running_environments;
  4. the per-environment episode logic works on the real thing: test_asynchronous_auto_reset_on_the_real_simulator,
     test_long_run_is_stable.

TOLERANCES are first guesses (first runs on the cluster: the peg's quaternion and the next-slot images were not comparable
quantities, see TOL and compare()). The floors (the real reset repeated, the restore repeated, each after another history)
are always measured and printed; a comparison passes if its error is within max(absolute tolerance, FLOOR_FACTOR x the larger floor).
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
# held_pos: peg position, env frame (m). held_axis: direction of the peg's axis (a tilt of x rad moves it by ~x). The peg is
# round, so its quaternion (held_quat) is only INFORMATION: a spin about its own axis changes it and nothing physical.
TOL = {"joint_pos": 5e-3, "joint_vel": 5e-2, "held_pos": 2e-3, "held_axis": 2e-2, "peg_socket_offset": 2e-3, "ft_force": 3.0,
       "tactile": 3.0, "image_mean_exterior": 2.0, "image_mean_wrist": 4.0}
INFO = ("held_quat",)                       # compared and written to the report, never asserted
STATE_TOL = {"sim/": 1e-4, "env/": 1e-5, "envdict/": 0.0}        # restore vs bank, read back from the simulator
PROBE_STEPS = 24


# NOT module-scoped: the pool keeps ONE server alive, so a fixture that cached its server would hold a stopped one as soon as
# another test (short_server below) asked for a different configuration. pool.get() is cheap while the server is alive.
@pytest.fixture
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
    for key in (*TOL, *INFO):
        if key not in a:
            continue
        sl = slice(1, None) if key.startswith("image") else slice(None)           # images at step 0 are the reset state's, not rendered
        out[key] = worst(a[key][sl], b[key][sl])
    return out


CURVE_KEYS = ("tactile", "ft_force", "held_pos", "held_axis", "joint_pos")      # where the restore differs most / matters most
ROUNDS = (1, 3)               # not 0: round 0 follows the construction of the environment, which a replay cannot recreate


def curve(a, b, key):
    """Error of `key` at EACH step (max over environments and components): a list of PROBE_STEPS + 1 numbers (step 0 = the state
    right after the reset or restore, before any control step)."""
    d = np.abs(a[key].astype(np.float64) - b[key].astype(np.float64))
    return [float(x) for x in d.reshape(d.shape[0], -1).max(axis=1)]


def settle_step(err_curve, floor_curves, factor=2.0):
    """First step from which the restore error stays within `factor` x the worst floor error: the length of the transient after
    a restore. None = it never settles within the probe."""
    limit = factor * max(max(c) for c in floor_curves)
    for k in range(len(err_curve)):
        if all(e <= limit for e in err_curve[k:]):
            return k
    return None


def print_curves(rnd, curves):
    print(f"\n[round {rnd}] error per step (steps 0..{PROBE_STEPS}). restore = restore vs the REAL resets in build order; "
          "floor_ref / floor_restore = each repeated after a different history; history = real reset without the replay vs with it")
    for key in CURVE_KEYS:
        c = curves[key]
        for name in ("restore", "floor_ref", "floor_restore", "history"):
            print(f"  {key if name == 'restore' else '':10s} {name:13s} " + " ".join(f"{x:.2g}" for x in c[name]))
        print(f"  {'':10s} settles at step {c['settle_step']}")


def print_summary(rnd, floor_ref, floor_restore, history, err, err_perm):
    print(f"\n[round {rnd}] worst error over the whole probe (floor_ref: the real reset repeated; floor_restore: the restore repeated)")
    print(f"  {'':18s} {'floor_ref':>10s} {'floor_rest':>10s} {'history':>10s} {'restore':>10s} {'next_slot':>10s} {'restore/floor':>14s}")
    for key in (*TOL, *INFO):
        if key in err:
            fl = max(floor_ref[key], floor_restore[key])
            print(f"  {key:18s} {floor_ref[key]:10.3g} {floor_restore[key]:10.3g} {history[key]:10.3g} {err[key]:10.3g} "
                  f"{err_perm.get(key, float('nan')):10.3g} {err[key] / max(fl, 1e-12):14.1f}")


def test_restore_matches_the_real_reset(env, reports_dir):
    """The reference is the real reset replayed in BUILD ORDER (rounds 0..r in a row), because the start of a real reset depends
    a little on the episode that ran before it. Two floors are measured: the reference repeated after another history, and the
    restore repeated after another history. A restore is faithful when its error is not larger than the floors."""
    report = {"steps": PROBE_STEPS, "n": N, "floor_factor": FLOOR_FACTOR, "tol": TOL, "rounds": {}}
    failures = []
    for rnd in ROUNDS:
        real = probe(env, "real", rnd)                                       # reset with whatever history precedes it
        ref = probe(env, "real_history", rnd)                                # reset in build order: what the bank captured
        ref2 = probe(env, "real_history", rnd)                               # repeated (preceded by the end of the first run)
        restored = probe(env, "restore", rnd)
        permuted = probe(env, "restore_permuted", rnd)
        restored2 = probe(env, "restore", rnd)                               # repeated (preceded by another entry's run)
        floor_ref, floor_restore, history = compare(ref, ref2), compare(restored, restored2), compare(real, ref)
        err = compare(ref, restored)
        rolled = {k: np.roll(v, -1, axis=1) if v.ndim > 1 and v.shape[1] == N else v for k, v in ref.items()}
        err_perm = compare(rolled, permuted)            # slot i of `permuted` holds entry of slot i+1: compare with ref's slot i+1
        # An image also depends on WHERE the environment is (neighbours, light), so it is only comparable in the same slot.
        err_perm = {k: v for k, v in err_perm.items() if not k.startswith("image")}
        floor = {k: max(floor_ref[k], floor_restore[k]) for k in floor_ref}
        curves = {}
        for key in CURVE_KEYS:
            c = {"restore": curve(ref, restored, key), "floor_ref": curve(ref, ref2, key),
                 "floor_restore": curve(restored, restored2, key), "history": curve(real, ref, key)}
            c["settle_step"] = settle_step(c["restore"], (c["floor_ref"], c["floor_restore"]))
            curves[key] = c
        print_summary(rnd, floor_ref, floor_restore, history, err, err_perm)
        print_curves(rnd, curves)
        if "tactile" in ref:
            print(f"\n[round {rnd}] tactile at step 0, largest taxel: real reset {np.abs(ref['tactile'][0]).max():.3g} N, "
                  f"restore {np.abs(restored['tactile'][0]).max():.3g} N, at step 1: {np.abs(ref['tactile'][1]).max():.3g} N vs "
                  f"{np.abs(restored['tactile'][1]).max():.3g} N")
        report["rounds"][rnd] = {"floor_ref": floor_ref, "floor_restore": floor_restore, "real_vs_history": history,
                                 "restore_vs_real": err, "restore_next_slot_vs_real": err_perm, "per_step": curves}
        for label, e in (("restore", err), ("next-slot restore", err_perm)):
            for key, value in e.items():
                if key in TOL and not passes(value, floor[key], key):
                    failures.append(f"round {rnd} {label} {key}: {value:.3e} > max({TOL[key]}, {FLOOR_FACTOR}x floor {floor[key]:.3e})")
        for name, traj in (("restored", restored), ("permuted", permuted), ("restored2", restored2)):
            for key, v in traj.items():
                assert np.isfinite(v.astype(np.float64)).all(), f"{name} {key} not finite"
    (reports_dir / "restore_equivalence.json").write_text(json.dumps(report, indent=2))
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
@pytest.fixture
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
