"""Sweep of the Robotiq finger settings against the start-state residual (needs Isaac Lab + GPU).

    SLURM_SUBMIT_DIR=$PWD bash jobs/job_isaac_tests.sh tests/test_isaac_finger_sweep.py 2>&1 | tee logs/isaac_finger_sweep.log
    SWEEP="default;finger_damping=50;finger_effort=3,finger_stiffness=2000" SWEEP_TAG=b SWEEP_STARTS=6 ...   # other settings
Settings are `field=value` (several, comma-separated, form one setting; settings are separated by ;): the finger fields of the task
YAML (finger_stiffness, finger_damping, finger_effort, fingerpad_length) and physx_<field> = any field of Isaac Lab's PhysxCfg
(physx_solver_type=0, physx_friction_correlation_distance=0.0005, ...). About 7 minutes per setting: keep a job to 3 settings
(the interactive allocation is killed at its time limit) and give each job its own SWEEP_TAG: reports/finger_sweep_<tag>.json.

MEASUREMENT, no threshold. Why: the start state of a (seed, round) depends a little on the episode that ran before, because the
gripper closing on the peg amplifies float-level differences (tests/test_isaac_fidelity.py::test_where_a_reset_diverges). Softer
or more damped fingers may close without that chaos. For each setting (a server restart each: these are YAML fields):

  1. RESIDUAL: for SWEEP_STARTS starts, the start of (seed, round 0) after a bare reset against the same start after 24 steps of
     another seed; every environment of every start counts (N x starts "env-starts"). Reported: how many differ at all, the worst
     difference on the peg (m) and on the joints (rad). 0 differing env-starts is the goal.
  2. WHAT THE SETTING DOES TO THE GRASP (a setting that removes the residual by changing the task is not a solution): the peg
     against the fingertip at the start (mean distance and its spread over env-starts), how the peg slips in the hand during 24
     scripted steps (mm), the finger opening, the finger motor effort, the tactile force (N). Compare each row with `default`.
Not measured here: the crush / loss rate over hundreds of reset seeds (the criterion that chose finger_effort = 6): repeat it on the
setting(s) that pass this first screen.
"""
import json
import os

import numpy as np
import pytest

from src.envs import protocol as P
from test_isaac_fidelity import N, scripted, snapshot

pytestmark = pytest.mark.isaac

DEFAULT_SWEEP = "finger_effort=3;physx_friction_correlation_distance=0.0005;physx_solver_type=0"
STARTS = int(os.environ.get("SWEEP_STARTS", "6"))
STEPS = 24                       # scripted steps used to see the peg slip in the hand
SLIP_MM = 5.0                    # slip beyond this (mm) counts as a lost peg
ALLOWED = {"finger_stiffness", "finger_damping", "finger_effort", "fingerpad_length"}


def parse_sweep(spec):
    """"default;finger_damping=50;finger_damping=20,finger_stiffness=2000" -> [(name, {field: value}), ...]"""
    out = []
    for item in (s.strip() for s in spec.split(";")):
        if not item:
            continue
        if item == "default":
            out.append((item, {}))
            continue
        fields = {k.strip(): float(v) for k, v in (p.split("=") for p in item.split(","))}
        unknown = {k for k in fields if k not in ALLOWED and not k.startswith("physx_")}
        if unknown:
            raise ValueError(f"SWEEP: unknown field(s) {sorted(unknown)} in {item!r}; allowed: {sorted(ALLOWED)} and physx_<field>")
        out.append((item, fields))
    return out


def run(env, seed, rnd, steps):
    """reset(seed, round), then `steps` scripted steps; list of frames (snapshot + fingertip position), frame 0 = just after the reset."""
    res = env.reset(seed=seed, round=rnd)
    hold = env.hold_actions()
    frames = []

    def frame(obs):
        f = snapshot(env, obs)
        f["tip_pos"] = np.array(env.debug("fingertip_pose")[0]["pos"])
        return f

    frames.append(frame(res.obs))
    for k in range(steps):
        res = env.step(scripted(hold, k), auto_reset=False)
        frames.append(frame(res.obs))
    return frames


def worst(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def measure(env, starts):
    """Per env-start metrics of one running server. {metric: list over env-starts}."""
    rows = {k: [] for k in ("differs", "peg_m", "joints_rad", "slip_mm", "peg_tip_m", "grip", "effort", "tactile_n", "tactile_max_n")}
    for seed in range(1, starts + 1):
        env.reset(seed=9, round=1)
        ref = run(env, seed, 0, 0)[0]                           # after a bare reset
        run(env, seed=5, rnd=0, steps=STEPS)                    # another episode in between
        frames = run(env, seed, 0, STEPS)                       # the same start after it, then the scripted steps
        start = frames[0]
        for e in range(N):
            peg = float(np.abs(start["peg_pos"][e] - ref["peg_pos"][e]).max())
            joints = float(np.abs(start[P.JOINT_POSITION][e].astype(np.float64) - ref[P.JOINT_POSITION][e]).max())
            offset = float(np.abs(start[P.PEG_SOCKET_OFFSET][e].astype(np.float64) - ref[P.PEG_SOCKET_OFFSET][e]).max())
            rel = np.stack([f["peg_pos"][e] - f["tip_pos"][e] for f in frames])          # peg against the fingertip, env frame
            rows["differs"].append(bool(max(peg, joints, offset) > 0))
            rows["peg_m"].append(max(peg, offset))
            rows["joints_rad"].append(joints)
            rows["slip_mm"].append(1e3 * float(np.linalg.norm(rel - rel[0], axis=1).max()))
            rows["peg_tip_m"].append(float(np.linalg.norm(rel[0])))
            rows["grip"].append(float(np.asarray(start[P.GRIPPER_POSITION][e]).reshape(-1)[0]))
            rows["effort"].append(float(np.asarray(start[P.GRIPPER_EFFORT][e]).reshape(-1)[0]) if P.GRIPPER_EFFORT in start else float("nan"))
            if P.TACTILE in start:
                rows["tactile_n"].append(float(np.asarray(start[P.TACTILE][e]).sum()))
                rows["tactile_max_n"].append(max(float(np.asarray(f[P.TACTILE][e]).sum()) for f in frames))
            else:
                rows["tactile_n"].append(float("nan"))
                rows["tactile_max_n"].append(float("nan"))
    return rows


def summarize(rows):
    mean = lambda k: float(np.nanmean(rows[k]))
    return {"env_starts": len(rows["differs"]), "differing": int(sum(rows["differs"])),
            "worst_peg_m": float(max(rows["peg_m"])), "worst_joints_rad": float(max(rows["joints_rad"])),
            "median_peg_m_when_differs": float(np.median([m for m, d in zip(rows["peg_m"], rows["differs"]) if d] or [0.0])),
            "peg_tip_mm_mean": 1e3 * mean("peg_tip_m"), "peg_tip_mm_std": 1e3 * float(np.std(rows["peg_tip_m"])),
            "slip_mm_mean": mean("slip_mm"), "slip_mm_max": float(max(rows["slip_mm"])),
            "lost": int(sum(s > SLIP_MM for s in rows["slip_mm"])),
            "grip_mean": mean("grip"), "effort_mean": mean("effort"),
            "tactile_n_mean": mean("tactile_n"), "tactile_n_max": float(np.nanmax(rows["tactile_max_n"]))}


HEAD = (f"{'setting':50s} {'differ':>8s} {'peg mm worst/med':>17s} {'joint rad':>9s} | {'peg-tip mm':>14s} {'slip mm':>8s} {'lost':>5s} "
        f"{'grip':>5s} {'effort':>7s} {'tactile N':>10s}")


def row(name, r):
    if "error" in r:
        return f"{name:50s} ERROR {r['error']}"
    return (f"{name:50s} {r['differing']:3d}/{r['env_starts']:<4d} {1e3 * r['worst_peg_m']:8.2f}/{1e3 * r['median_peg_m_when_differs']:<8.3f} "
            f"{r['worst_joints_rad']:9.1e} | {r['peg_tip_mm_mean']:7.2f}+-{r['peg_tip_mm_std']:5.2f} {r['slip_mm_mean']:8.2f} "
            f"{r['lost']:5d} {r['grip_mean']:5.2f} {r['effort_mean']:7.2f} {r['tactile_n_mean']:5.1f}/{r['tactile_n_max']:<4.1f}")


def test_finger_settings_sweep(isaac_pool, reports_dir):
    sweep = parse_sweep(os.environ.get("SWEEP", DEFAULT_SWEEP))
    out = reports_dir / f"finger_sweep_{os.environ.get('SWEEP_TAG', 'last')}.json"
    report = {}
    print("\n" + HEAD + "\n" + "-" * len(HEAD), flush=True)
    for name, overrides in sweep:
        try:
            srv = isaac_pool.get(N, fresh=True, **overrides)
            with srv.client() as env:
                run(env, seed=1, rnd=0, steps=5)                # warm-up, thrown away
                report[name] = summarize(measure(env, STARTS))
        except Exception as e:  # noqa: BLE001  -- one setting that crashes (PhysX refuses it) must not stop the sweep
            report[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(row(name, report[name]), flush=True)             # one line per setting as soon as it is done
        out.write_text(json.dumps(report, indent=2))
    print("\ndiffer = environments whose start changed at all after another episode (goal 0/24); peg mm = worst, and median over the "
          "environments that differ; the columns after | say what the setting does to the grasp: compare with the `default` row "
          "(8/24, worst 5.2 mm, peg-tip 45.08+-1.55, slip 2.35, grip 0.87, effort 6.0, tactile 106.9/127.6).")
    assert report, "empty sweep"
