"""Fidelity of the real simulator (needs Isaac Lab + GPU:  pytest tests/test_isaac_fidelity.py --isaac -s).

Every round starts with FORGE's own reset, so there is no approximation to measure against a reference. What is verified:
  1. the reset is REPRODUCIBLE: the same (seed, round) gives the same start and the same trajectory, bit for bit, whatever ran
     before (test_reset_is_bit_exact_for_a_seed_and_round, test_next_round_is_the_next_seed, test_a_collected_run_is_replayed_exactly);
  2. the environments of a server are INDEPENDENT: what one does does not change what another does
     (test_the_environments_of_a_round_are_independent), and their starts are valid and different
     (test_start_states_are_valid_and_diverse);
  3. the rounds work on the real thing: all environments end, the next round starts in the same call, with exactly the start a
     reset(seed, round) gives (test_a_round_ends_and_the_next_one_starts_by_itself, test_long_run_is_stable).

What is NOT claimed: the start of environment i depends on (seed, round, num_envs, i), so it is not the start a single
environment would get from the same seed (FORGE draws all its randomisation in one batch); and the physics of an environment
in a batch is not compared with the same environment alone.

Physical quantities (joints, forces, peg pose, taxels) are compared with EXACT; images with IMAGE_TOL (the renderer is not
bit-exact: mean pixel value, out of 255). Every number is written to reports/reset_fidelity.json and reports/isolation.json.
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
def test_reset_is_bit_exact_for_a_seed_and_round(env, reports_dir):
    first = run_script(env, seed=11, rnd=2)
    run_script(env, seed=5, rnd=0)                                  # something else in between: another seed, another round
    run_script(env, seed=11, rnd=3, steps=7)
    again = run_script(env, seed=11, rnd=2)
    diff = compare(first, again)
    other_round = run_script(env, seed=11, rnd=3, steps=0)
    other_seed = run_script(env, seed=12, rnd=2, steps=0)
    report = {"same_seed_same_round": diff, "other_round_step0": worst(first["extra/peg_socket_offset"][0], other_round["extra/peg_socket_offset"][0]),
              "other_seed_step0": worst(first["extra/peg_socket_offset"][0], other_seed["extra/peg_socket_offset"][0])}
    (reports_dir / "reset_fidelity.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=1))
    for key, d in diff.items():
        assert d <= (IMAGE_TOL if is_image(key) else EXACT), f"{key}: the same (seed, round) gave a different run, by {d:.3e}"
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

        data = Collector(env, policy, mode="steps", store_next_obs=False).collect(330)
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
        for key, v in res.final_obs.items():
            assert not np.array_equal(v, res.obs[key]), key        # the last observation of round 0 is not the start of round 1
        started = snapshot(env, res.obs)
        explicit = snapshot(env, env.reset(seed=3, round=1).obs)
        for key in started:
            assert worst(started[key], explicit[key]) <= (IMAGE_TOL if is_image(key) else EXACT), key
        again = env.step(hold)                                       # and the new round runs
        assert again.acting.all() and (again.t == 1).all() and (again.round_id == 1).all()


def test_a_collected_run_is_replayed_exactly(short_server):
    """The strongest check: a run of several rounds, its actions recorded, replayed from the same seed gives the same
    observations at every step, including across the resets that start the rounds."""
    from src.envs.collect import Collector
    with short_server.client() as env:
        rng = np.random.default_rng(0)
        hold = {"v": None}
        sent = []

        def policy(obs):
            if hold["v"] is None:
                hold["v"] = env.hold_actions()
            a = hold["v"] + rng.normal(0, 0.01, hold["v"].shape).astype(np.float32)
            sent.append(a)
            return a

        first = Collector(env, policy, mode="steps", seed=5).collect(20)          # rounds of 6 steps: 3 resets inside
        assert first.round[-1].max() >= 2
        it = iter(sent)
        second = Collector(env, lambda obs: next(it), mode="steps", seed=5).collect(20)
        assert (first.round == second.round).all()
        for key in first.obs:
            if "image" in key:
                continue
            assert np.abs(first.obs[key].astype(np.float64) - second.obs[key].astype(np.float64)).max() <= EXACT, key
            assert np.abs(first.next_obs[key].astype(np.float64) - second.next_obs[key].astype(np.float64)).max() <= EXACT, key
