"""Time and GPU memory per environment, and how they grow when environments are added (needs Isaac Lab + GPU).

    pytest tests/test_isaac_cost.py --isaac -s
    COST_NUM_ENVS=1,2,4,8,16,32,64 COST_STEPS=100 pytest tests/test_isaac_cost.py --isaac -s

For every N a FRESH server is started (one Isaac Sim start-up each, ~2 min) so memory is not polluted by a previous size;
only one server is alive at a time. Measured, all on THIS machine:
  vram         process memory (NVML) after start-up, after the first steps, peak; torch's own counters for reference
  step         client round-trip per step with N environments stepping, and its server-side split: physics (Isaac Lab), observe
               (images / tactile to numpy), the rest = serialisation + network + the manager's bookkeeping
  restore      cost of resetting 1, N/2 and N environments (client round-trip), i.e. what an auto-reset costs
  throughput   environment-steps per second
Then linear fits  vram(N) = fixed + N * per_env  and  step(N) = fixed + N * per_env, the marginal cost of adding environments,
and the largest N that fits in the GPU memory. Everything goes to reports/env_cost_<timestamp>.json (+ a printed table).
The test fails only on nonsense (non-finite numbers, memory that does not grow, N environments slower than N x one).

Note on concurrency: this measures ONE server process driven by one client, the way training uses it. A collection phase of T
steps costs T x step(N); the update phase costs nothing on the simulator (it is not stepped while the learner works).
"""
import json
import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.isaac

SIZES = [int(x) for x in os.environ.get("COST_NUM_ENVS", "1,2,4,8,16,32").split(",")]
STEPS = int(os.environ.get("COST_STEPS", "60"))
WARMUP = 5


def stats(ms):
    a = np.asarray(ms, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "min": float(a.min()),
            "max": float(a.max())}


def fit(xs, ys):
    """least squares  y = a + b x ; returns (a, b, r2)."""
    x, y = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    if len(x) < 2:
        return None
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return {"fixed": float(a), "per_env": float(b), "r2": r2}


def measure(pool, n):
    srv = pool.get(n, fresh=True, bank_size=max(4, n), bank_seed=1)
    out = {"num_envs": n, "start_seconds": srv.start_seconds}
    with srv.client() as env:
        out["vram_ready"] = env.debug("vram")[0]
        env.reset(seed=0)
        out["vram_after_reset"] = env.debug("vram")[0]
        action = env.hold_actions()
        for _ in range(WARMUP):
            env.step(action)
        env.timings(reset=True)
        rtt = []
        for _ in range(STEPS):
            t0 = time.perf_counter()
            env.step(action)
            rtt.append(1e3 * (time.perf_counter() - t0))
        server = env.timings(reset=True)[0]
        out["vram_after_steps"] = env.debug("vram")[0]
        out["step_ms"] = stats(rtt)
        out["server_ms_per_step"] = {k: v["mean_ms"] for k, v in server.items() if k in ("step_physics", "inserted", "observe")}
        out["overhead_ms_per_step"] = out["step_ms"]["mean"] - sum(out["server_ms_per_step"].values())
        out["env_steps_per_s"] = 1e3 * n / out["step_ms"]["mean"]

        restore = {}
        for k in sorted({1, max(1, n // 2), n}):
            ids = np.arange(k)
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                env.reset(env_ids=ids, bank_ids=np.arange(k) % max(4, n))
                times.append(1e3 * (time.perf_counter() - t0))
            restore[str(k)] = stats(times[2:])
        out["restore_ms_by_num_reset_envs"] = restore
        out["restore_server_ms_per_call"] = env.timings(reset=True)[0].get("restore", {}).get("mean_ms")
        out["vram_final"] = env.debug("vram")[0]
    return out


def test_cost_per_environment(isaac_pool, reports_dir):
    rows, errors = [], []
    for n in SIZES:
        try:
            rows.append(measure(isaac_pool, n))
        except Exception as e:             # noqa: BLE001  -- a server that dies at large N is a result (GPU limit), not a crash
            errors.append({"num_envs": n, "error": f"{type(e).__name__}: {str(e)[:400]}"})
            break
    isaac_pool.stop()
    report = {"sizes_requested": SIZES, "steps": STEPS, "rows": rows, "errors": errors}

    proc = [(r["num_envs"], r["vram_final"].get("process_mib")) for r in rows if r["vram_final"].get("process_mib")]
    if len(proc) >= 2:
        f = fit(*zip(*proc))
        report["vram_fit_mib"] = f
        total = rows[-1]["vram_final"].get("device_total_mib")
        if total and f["per_env"] > 0:
            report["max_envs_in_gpu_memory"] = int((0.9 * total - f["fixed"]) // f["per_env"])      # 10 % margin
    step = [(r["num_envs"], r["step_ms"]["mean"]) for r in rows]
    if len(step) >= 2:
        report["step_ms_fit"] = fit(*zip(*step))
    for key in ("step_physics", "observe"):
        pts = [(r["num_envs"], r["server_ms_per_step"].get(key)) for r in rows if r["server_ms_per_step"].get(key) is not None]
        if len(pts) >= 2:
            report[f"{key}_ms_fit"] = fit(*zip(*pts))
    marginal = []
    for a, b in zip(rows, rows[1:]):
        dn = b["num_envs"] - a["num_envs"]
        pa, pb = a["vram_final"].get("process_mib"), b["vram_final"].get("process_mib")
        marginal.append({"from": a["num_envs"], "to": b["num_envs"], "step_ms_per_added_env": (b["step_ms"]["mean"] - a["step_ms"]["mean"]) / dn,
                         "vram_mib_per_added_env": None if pa is None or pb is None else (pb - pa) / dn})
    report["marginal_cost"] = marginal

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"env_cost_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"\n{'N':>4} {'start s':>8} {'VRAM MiB':>9} {'MiB/env':>8} {'step ms':>8} {'physics':>8} {'observe':>8} {'other':>7} {'env-steps/s':>12} {'restore 1 env ms':>17}")
    for r in rows:
        vram = r["vram_final"].get("process_mib") or float("nan")
        print(f"{r['num_envs']:>4} {r['start_seconds']:>8.0f} {vram:>9.0f} {vram / r['num_envs']:>8.0f} {r['step_ms']['mean']:>8.1f} "
              f"{r['server_ms_per_step'].get('step_physics', float('nan')):>8.1f} {r['server_ms_per_step'].get('observe', float('nan')):>8.1f} "
              f"{r['overhead_ms_per_step']:>7.1f} {r['env_steps_per_s']:>12.0f} {r['restore_ms_by_num_reset_envs']['1']['mean']:>17.1f}")
    for k in ("vram_fit_mib", "step_ms_fit", "step_physics_ms_fit", "observe_ms_fit", "max_envs_in_gpu_memory"):
        if k in report:
            print(f"{k}: {report[k]}")
    for e in errors:
        print(f"STOPPED at N={e['num_envs']}: {e['error']}")
    print(f"report: {path}")

    assert len(rows) >= 1, f"no size could be measured: {errors}"
    for r in rows:
        assert np.isfinite(r["step_ms"]["mean"]) and r["step_ms"]["mean"] > 0
    if len(proc) >= 2:
        assert proc[-1][1] > proc[0][1], "GPU memory does not grow with the number of environments: NVML figure is wrong?"
    if len(rows) >= 2:
        one, last = rows[0], rows[-1]
        assert last["step_ms"]["mean"] < 1.5 * last["num_envs"] / one["num_envs"] * one["step_ms"]["mean"], \
            "N parallel environments are slower than N x one environment: parallelism brings nothing"
