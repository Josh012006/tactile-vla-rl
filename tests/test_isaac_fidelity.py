"""Fidelity of the real simulator (needs Isaac Lab + GPU:  pytest tests/test_isaac_fidelity.py --isaac -s).

Every round starts with FORGE's own reset, so there is no approximation to measure against a reference. What is verified:
  1. REPRODUCIBILITY, two different guarantees:
     a. the same SEQUENCE of resets and steps from a fresh server is bit-identical, every observation included
        (test_the_same_sequence_on_a_fresh_server_is_bit_exact): this is what replays a run;
     b. the start of a (seed, round) does NOT forget the episode that ran just before (Robotiq only; the Franka does). It is
        measured, not claimed away: test_start_depends_little_on_the_previous_episode bounds it, test_what_survives_a_reset and
        test_where_a_reset_diverges say where it comes from (the reset is bit-identical until the gripper closes on the peg;
        there contact chaos amplifies float-level differences, in about 1 environment out of 4). Same level as expo-ft's;
  2. the environments of a server are INDEPENDENT: what one does does not change what another does
     (test_the_environments_of_a_round_are_independent), and their starts are valid and different
     (test_start_states_are_valid_and_diverse);
  3. the rounds work on the real thing: all environments end, the next round starts in the same call, with the start a
     reset(seed, round) gives (test_a_round_ends_and_the_next_one_starts_by_itself, test_long_run_is_stable).

What is NOT claimed: the start of environment i depends on (seed, round, num_envs, i), so it is not the start a single
environment would get from the same seed (FORGE draws all its randomisation in one batch); the physics of an environment
in a batch is not compared with the same environment alone; trajectories after a start that differs by the dependence above
are not expected to be identical (contact-rich dynamics amplify it).

Physical quantities (joints, forces, peg pose, taxels) are compared with EXACT where an exact guarantee exists; images with
IMAGE_TOL (the renderer is not bit-exact: mean pixel value, out of 255). Numbers go to reports/*.json.
"""
import json

import numpy as np
import pytest

from src.envs import protocol as P

pytestmark = pytest.mark.isaac

N = 4
PROBE_STEPS = 24
EXACT = 1e-6                    # physics must reproduce exactly; the margin is float printing, not simulation noise
IMAGE_TOL = 3.0                 # mean pixel value (0..255) of an image, same state rendered twice
# Independence of the environments: another environment doing something violent must not move this one. Exactly 0 is expected
# (separate scenes, no shared contacts); these tolerances only keep a last-bit difference of the batched solver from failing it.
ISOLATION_TOL = {"observation/joint_position": 1e-4, "extra/ft_force": 0.05, "extra/peg_socket_offset": 1e-4,
                 "extra/tactile": 0.05, "extra/gripper_effort": 1e-3, "peg_pos": 1e-4, "peg_axis": 1e-3}


# NOT module-scoped: the pool keeps ONE server alive, so a fixture that cached its server would hold a stopped one as soon as
# another test (short_server below) asked for a different configuration. pool.get() is cheap while the server is alive.
@pytest.fixture
def server(isaac_pool):
    return isaac_pool.get(N)


@pytest.fixture
def env(server):
    c = server.client()
    yield c
    c.close()


def scripted(hold, k, amplitude=0.02, period=24, only=None, extra=0.0):
    """hold + a small sinusoid on the arm joints; `extra` adds a large offset to the environments in `only`."""
    a = hold.copy()
    a[:, 0:7] += amplitude * np.sin(2 * np.pi * k / period + np.arange(7))
    if only is not None:
        a[only, 0:7] += extra * np.cos(2 * np.pi * k / period + np.arange(7))
    return a


def snapshot(env, obs):
    """Everything physical about the current state, as arrays (N, ...), plus the mean pixel value of each camera."""
    pose = env.debug("peg_pose")[0]
    out = {k: np.array(v) for k, v in obs.items() if "image" not in k}
    out["peg_pos"], out["peg_axis"] = np.array(pose["pos"]), np.array(pose["axis"])
    for k, v in obs.items():
        if "image" in k:
            out["mean_" + k.split("/")[-1]] = np.asarray(v, dtype=np.float64).mean(axis=(1, 2, 3))
    return out


def run_script(env, seed, rnd, steps=PROBE_STEPS, only=None, extra=0.0):
    """reset(seed, round), then `steps` scripted steps; {key: (steps + 1, N, ...)}. Step 0 is the state right after the reset."""
    res = env.reset(seed=seed, round=rnd)
    hold = env.hold_actions()
    traj = [snapshot(env, res.obs)]
    for k in range(steps):
        res = env.step(scripted(hold, k, only=only, extra=extra), auto_reset=False)
        traj.append(snapshot(env, res.obs))
    return {k: np.stack([t[k] for t in traj]) for k in traj[0]}


def worst(a, b, rows=None):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if rows is not None:
        d = d[:, rows]
    return float(d.max())


def compare(a, b, rows=None):
    return {k: worst(a[k], b[k], rows) for k in a}


def is_image(key):
    return key.startswith("mean_")


# ----------------------------------------------------------------------------------------------------- 1. reproducibility
LEAK_STARTS = [(21, 0), (11, 2), (1, 0)]       # (seed, round) of the start whose state after the reset is measured


def differing(a, b):
    """{tensor name: per-environment worst difference} for the tensors of two state dumps that are not identical."""
    out = {}
    for key in a:
        if key in b and a[key].shape == b[key].shape:
            d = np.abs(np.nan_to_num(a[key].astype(np.float64)) - np.nan_to_num(b[key].astype(np.float64)))
            if d.size == 0:                 # a buffer with no element (e.g. a sensor without bodies of this kind): nothing to compare
                continue
            worst_per_env = d.reshape(d.shape[0], -1).max(axis=1)
            if worst_per_env.max() > 0:
                out[key] = [float(x) for x in worst_per_env]
    return out


def test_what_survives_a_reset(isaac_pool, reports_dir):
    """MEASUREMENT (no threshold): which tensor keeps something of the episode that ran before a reset?
    For each start (seed, round) of LEAK_STARTS the reset of that start is done after three different histories: a bare reset
    (a), a 24-step run of another seed (b), a 7-step run of the next round (c); (a) is done twice (the floor: identical history,
    everything must be identical). After each reset every per-environment tensor of the env, the robot, the two assets and the
    contact sensors is dumped, and the dumps of (a again), (b), (c) are compared with the first (a), tensor by tensor and
    environment by environment. The same method as expo-ft's scripts/isaaclab/reset_history.py, on the tensors and not only on
    the start state. A first run and a first dump on the server are thrown away: the very first read of some derived buffers
    (centre-of-mass poses) differs from the later ones whatever the history, which a first measurement showed.
    Result (Robotiq, L40S): about 124 tensors differ, only poses, velocities, accelerations and contact forces of the affected
    environments, no control buffer; `--isaac-set robot=franka`: nothing differs; `--isaac-set disable_sleep=true`: no change
    (not the sleeping state). test_where_a_reset_diverges says when the difference appears."""
    srv = isaac_pool.get(N, fresh=True)
    report = {}
    with srv.client() as env:
        run_script(env, seed=1, rnd=0, steps=5)
        env.debug("state_dump")                                         # thrown away: first read of the lazy buffers
        for seed, rnd in LEAK_STARTS:
            histories = {"a": lambda: env.reset(seed=9, round=1), "a again": lambda: env.reset(seed=9, round=1),
                         "b": lambda: run_script(env, seed=5, rnd=0), "c": lambda: run_script(env, seed=seed, rnd=rnd + 1, steps=7)}
            dumps = {}
            for tag, history in histories.items():
                history()
                env.reset(seed=seed, round=rnd)
                dumps[tag] = env.debug("state_dump")[0]
            report[f"start seed={seed} round={rnd}"] = {
                "tensors_compared": len(dumps["a"]), "floor": differing(dumps["a"], dumps["a again"]),
                "after_24_steps_of_another_seed": differing(dumps["a"], dumps["b"]),
                "after_7_steps_of_the_next_round": differing(dumps["a"], dumps["c"])}
    (reports_dir / "state_leak.json").write_text(json.dumps(report, indent=2))
    for start, parts in report.items():
        print(f"\n== {start}: {parts['tensors_compared']} tensors per dump; floor (identical history): {len(parts['floor'])} differ")
        for name in ("after_24_steps_of_another_seed", "after_7_steps_of_the_next_round"):
            rows = parts[name]
            print(f"-- {name}: {len(rows)} tensors differ from the bare-reset history (largest first; one value per environment)")
            for key, per_env in sorted(rows.items(), key=lambda kv: -max(kv[1]))[:30]:
                print(f"  {key:60s} " + " ".join(f"{v:9.2e}" for v in per_env))
            for key in ("robot.joint_pos", "held.root_pos_w", "held.root_quat_w"):      # the start state itself (0 = identical)
                print(f"  START STATE {key:48s} " + " ".join(f"{v:9.2e}" for v in rows.get(key, [0.0] * N)))
            print("  every tensor that differs: " + ", ".join(sorted(rows)))


TRACE_GROUPS = ("dof_pos", "dof_vel", "link_pose", "link_vel", "held_pose", "held_vel", "fixed_pose")


def trace_of_reset(env, seed, rnd):
    env.debug("trace_next_reset")
    env.reset(seed=seed, round=rnd)
    trace = env.debug("reset_trace")[0]
    assert "error" not in trace, trace
    return trace


def trace_differences(a, b):
    """{group: (T, N) worst absolute difference per physics step and environment} between two traces (shortest length)."""
    t = min(len(a["phases"]), len(b["phases"]))
    out = {}
    for g in TRACE_GROUPS:
        d = np.abs(a[g][:t].astype(np.float64) - b[g][:t].astype(np.float64))
        out[g] = d.reshape(t, d.shape[1], -1).max(axis=2)
    return out


def test_where_a_reset_diverges(isaac_pool, reports_dir):
    """MEASUREMENT (no threshold): at which physics step of the reset do two histories start to differ, and in which joint or link?
    The reset is traced (forge_backend.py _traced: the PhysX state after every physics step it takes). For each start of
    LEAK_STARTS it is run after a bare reset (a), again after the same (a again: the floor, must be 0) and after 24 steps of another
    seed (b). Record 0 is the state left by the previous episode, so it differs by construction; record 1 is the first physics
    step after FORGE has written the robot and the assets to their default pose: a difference THERE means the teleport did not
    erase something (state inside PhysX); a difference that appears later and grows means a tiny one was amplified by the dynamics
    (the closed-loop IK, the grasp). For every environment that differs: the first step, the joints and links, then the growth."""
    srv = isaac_pool.get(N, fresh=True)
    report = {}
    with srv.client() as env:
        run_script(env, seed=1, rnd=0, steps=5)
        for seed, rnd in LEAK_STARTS:
            env.reset(seed=9, round=1)
            a = trace_of_reset(env, seed, rnd)
            env.reset(seed=9, round=1)
            a2 = trace_of_reset(env, seed, rnd)
            run_script(env, seed=5, rnd=0)
            b = trace_of_reset(env, seed, rnd)
            floor = max(float(v.max()) for g, v in trace_differences(a, a2).items())
            diff = trace_differences(a, b)
            joints, bodies, phases = a["joint_names"], a["body_names"], a["phases"]
            print(f"\n== start seed={seed} round={rnd}: {len(phases) - 2} physics steps in the reset; floor (identical history): {floor:.1e};"
                  f" lengths a={len(a['phases'])} b={len(b['phases'])}")
            per_env = {}
            for e in range(N):
                steps = [t for t in range(1, diff["dof_pos"].shape[0]) if any(diff[g][t, e] > 0 for g in TRACE_GROUPS)]
                if not steps:
                    print(f"  env {e}: identical at every physics step of the reset")
                    per_env[e] = None
                    continue
                t0 = steps[0]
                first = [g for g in TRACE_GROUPS if diff[g][t0, e] > 0]
                print(f"  env {e}: FIRST difference at physics step {t0} of {len(phases) - 2} ({phases[t0]}), in {first}")
                for g, names, axis in (("dof_pos", joints, 1), ("dof_vel", joints, 1), ("link_pose", bodies, 1), ("link_vel", bodies, 1)):
                    d = np.abs(a[g][t0, e].astype(np.float64) - b[g][t0, e].astype(np.float64))
                    d = d.reshape(d.shape[0], -1).max(axis=1)
                    top = [(names[i], float(d[i])) for i in np.argsort(-d)[:4] if d[i] > 0]
                    if top:
                        print(f"      {g:9s} " + "  ".join(f"{n}={v:.1e}" for n, v in top))
                shown = sorted(set(steps[:6]) | set(range(t0, diff["dof_pos"].shape[0], 10)) | {diff["dof_pos"].shape[0] - 1})
                print("      growth (step, phase, worst over joints / links / peg):")
                for t in shown:
                    print(f"        {t:4d} {phases[t]:45s} dof_pos {diff['dof_pos'][t, e]:.1e}  dof_vel {diff['dof_vel'][t, e]:.1e}  "
                          f"link_pose {diff['link_pose'][t, e]:.1e}  held_pose {diff['held_pose'][t, e]:.1e}  held_vel {diff['held_vel'][t, e]:.1e}")
                per_env[e] = {"first_step": t0, "phase": phases[t0], "first_groups": first,
                              "final": {g: float(diff[g][-1, e]) for g in TRACE_GROUPS}}
            report[f"start seed={seed} round={rnd}"] = {"floor": floor, "physics_steps": len(phases) - 2, "envs": per_env}
    (reports_dir / "reset_trace.json").write_text(json.dumps(report, indent=2))


# What the start state of a (seed, round) may differ by after another episode: measured maxima (Robotiq, 4 envs, 3 starts) were
# 1.4e-4 m on the peg and 1.8e-4 rad on the joints (expo-ft: 2e-4 m); the bounds leave a factor ~10 so only a regression fails.
START_TOL_M, START_TOL_RAD = 2e-3, 2e-3


def test_start_depends_little_on_the_previous_episode(env, reports_dir):
    """The start of (seed 11, round 2) after three different histories, against the one after a bare reset: how much does it
    differ? (This is NOT bit-exactness: see the module docstring.) Bounded by START_TOL_*, and the start does change with seed and round."""
    env.reset(seed=9, round=1)
    reference = run_script(env, seed=11, rnd=2, steps=0)
    after = {}
    env.reset(seed=9, round=1)
    after["bare reset, again"] = run_script(env, seed=11, rnd=2, steps=0)
    run_script(env, seed=5, rnd=0)
    after["24 steps of another seed"] = run_script(env, seed=11, rnd=2, steps=0)
    run_script(env, seed=11, rnd=3, steps=7)
    after["7 steps of the next round"] = run_script(env, seed=11, rnd=2, steps=0)
    report = {}
    for name, start in after.items():
        d = compare(reference, start)
        report[name] = {"peg_m": d["peg_pos"], "joints_rad": d["observation/joint_position"], "peg_socket_offset_m": d["extra/peg_socket_offset"]}
    other_round = run_script(env, seed=11, rnd=3, steps=0)
    other_seed = run_script(env, seed=12, rnd=2, steps=0)
    report["other_round_step0"] = worst(reference["extra/peg_socket_offset"][0], other_round["extra/peg_socket_offset"][0])
    report["other_seed_step0"] = worst(reference["extra/peg_socket_offset"][0], other_seed["extra/peg_socket_offset"][0])
    (reports_dir / "reset_fidelity.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=1))
    for name, r in report.items():
        if isinstance(r, dict):
            assert r["peg_m"] <= START_TOL_M and r["peg_socket_offset_m"] <= START_TOL_M, f"{name}: the peg start moved by {r}"
            assert r["joints_rad"] <= START_TOL_RAD, f"{name}: the joint start moved by {r}"
    assert report["other_round_step0"] > 1e-4 and report["other_seed_step0"] > 1e-4      # sanity: the start does change


def test_next_round_is_the_next_seed(env):
    env.reset(seed=7, round=3)
    nxt = env.reset()                                               # no argument: the next round
    assert (nxt.round_id == 4).all()
    a = snapshot(env, nxt.obs)
    explicit = env.reset(seed=7, round=4)
    b = snapshot(env, explicit.obs)
    for key in a:
        assert worst(a[key], b[key]) <= (IMAGE_TOL if is_image(key) else EXACT), key


def test_start_states_are_valid_and_diverse(env):
    offsets = []
    for rnd in range(4):
        res = env.reset(seed=1, round=rnd)
        assert res.acting.all() and (res.t == 0).all() and not res.done.any() and (res.round_id == rnd).all()
        for key, v in res.obs.items():
            assert np.isfinite(np.asarray(v, dtype=np.float64)).all(), key
        offsets.append(np.array(res.obs[P.PEG_SOCKET_OFFSET]))
    rows = np.concatenate(offsets).round(6)
    assert np.unique(rows, axis=0).shape[0] == 4 * N                # every environment of every round starts somewhere else


# --------------------------------------------------------------------------------------------------------- 2. independence
def test_the_environments_of_a_round_are_independent(env, reports_dir):
    others = list(range(1, N))
    a = run_script(env, seed=21, rnd=0)
    a2 = run_script(env, seed=21, rnd=0)                            # the floor: same protocol, nothing changed
    b = run_script(env, seed=21, rnd=0, only=[0], extra=0.25)       # environment 0 is driven hard, the others as in `a`
    floor, moved, own = compare(a, a2, others), compare(a, b, others), compare(a, b, [0])
    rows = {}
    for key in ISOLATION_TOL:
        if key in a:
            rows[key] = {"floor": floor[key], "others_with_env0_driven_hard": moved[key], "env0_itself": own[key]}
    (reports_dir / "isolation.json").write_text(json.dumps(rows, indent=2))
    print("\n" + json.dumps(rows, indent=1))
    for key, r in rows.items():
        assert r["others_with_env0_driven_hard"] <= max(ISOLATION_TOL[key], 10 * r["floor"]), \
            f"{key}: environments 1.. moved by {r['others_with_env0_driven_hard']:.3e} when environment 0 did something else (floor {r['floor']:.3e})"
    assert own["peg_pos"] > 1e-3 or own["observation/joint_position"] > 1e-2        # sanity: environment 0 itself did change


# ------------------------------------------------------------------------------------------------------------ 3. the rounds
def test_long_run_is_stable(server):
    from src.envs.collect import Collector
    with server.client() as env:
        rng = np.random.default_rng(0)
        hold = {"v": None}

        def policy(obs):
            if hold["v"] is None:
                hold["v"] = env.hold_actions()
            return hold["v"] + rng.normal(0, 0.01, hold["v"].shape).astype(np.float32)

        data = Collector(env, policy, mode="steps", store_next_obs=False, seed=0).collect(330)      # seed given: starts at round 0
        assert len(data.episodes) >= N * 2                                      # max_steps 150: every env finished twice
        assert all(ep.length == 150 and ep.truncated for ep in data.episodes if not ep.terminated)
        assert data.round.max() == 2
        for k, v in data.obs.items():
            assert np.isfinite(v.astype(np.float64)).all(), k
        assert np.abs(data.obs[P.PEG_SOCKET_OFFSET]).max() < 1.0                # the peg never flies away


@pytest.fixture
def short_server(isaac_pool):
    return isaac_pool.get(N, max_steps_per_episode=6)


def test_a_round_ends_and_the_next_one_starts_by_itself(short_server):
    with short_server.client() as env:
        res = env.reset(seed=3)
        hold = env.hold_actions()
        for k in range(1, 7):
            assert (res.round_id == 0).all() and res.acting.all()
            res = env.step(hold)
            assert (res.t == k).all() or res.done.all()
        # 6 steps without insertion: every episode timed out on the same step, the next round started inside the call
        assert res.truncated.all() and not res.terminated.any() and (res.t == 6).all() and not res.all_done
        assert (res.round_id == 1).all() and res.final_ids.tolist() == list(range(N))
        for key in (P.JOINT_POSITION, P.PEG_SOCKET_OFFSET):         # (constants such as the gripper effort are equal on both sides)
            assert not np.array_equal(res.final_obs[key], res.obs[key]), key   # the last observation of round 0 is not the start of round 1
        started = snapshot(env, res.obs)
        explicit = snapshot(env, env.reset(seed=3, round=1).obs)
        for key in started:
            assert worst(started[key], explicit[key]) <= (IMAGE_TOL if is_image(key) else EXACT), key
        again = env.step(hold)                                       # and the new round runs
        assert again.acting.all() and (again.t == 1).all() and (again.round_id == 1).all()


def run_sequence(env, steps=20):
    """A fixed sequence of resets and steps (episodes of 6 steps: three rounds start inside it): every non-image observation."""
    res = env.reset(seed=5)
    hold = env.hold_actions()
    traj = [res.obs]
    for k in range(steps):
        res = env.step(scripted(hold, k))
        traj.append(res.obs)
    return {key: np.stack([np.asarray(o[key], dtype=np.float64) for o in traj]) for key in traj[0] if "image" not in key}


def test_the_same_sequence_on_a_fresh_server_is_bit_exact(isaac_pool, reports_dir):
    """The guarantee expo-ft had (its YAML: 'restarting the server and running the same collection command gives the same
    demos'): the same sequence of resets and steps, run twice from a freshly started server, gives identical observations, bit
    for bit, resets of the rounds included. It is NOT the same as 'a reset does not depend on what ran before it' (the two
    tests above), which expo-ft did not have either with the Robotiq."""
    runs = []
    for _ in range(2):
        srv = isaac_pool.get(N, fresh=True, max_steps_per_episode=6)
        with srv.client() as env:
            runs.append(run_sequence(env))
    diff = {key: worst(runs[0][key], runs[1][key]) for key in runs[0]}
    (reports_dir / "sequence_determinism.json").write_text(json.dumps(diff, indent=2))
    print("\n" + json.dumps(diff, indent=1))
    for key, d in diff.items():
        assert d == 0.0, f"{key}: the same sequence gave a different result on a fresh server, by {d:.3e}"

