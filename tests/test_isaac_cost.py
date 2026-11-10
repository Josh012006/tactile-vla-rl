"""Memory and time as a function of the number of environments N (needs Isaac Lab + GPU).

    pytest tests/test_isaac_cost.py --isaac -s
    COST_NUM_ENVS=1,2,4,8,16,32,64,128 COST_ROLLOUT_STEPS=80 pytest tests/test_isaac_cost.py --isaac -s

For every N a FRESH server is started (one Isaac Sim start-up each, ~2 min) so memory is not polluted by a previous size; only one
server is alive at a time. Per N, measured on THIS machine:
  vram     process memory (NVML) after start-up, after the first reset, after the steps, final; torch's counters for reference
  reset    the real reset of every environment (one round starts): client round trip and server side, mean of COST_RESETS
  step     client round trip of one step with N environments, and its server-side split: physics (Isaac Lab, rendering included),
           observe (images / taxels to numpy), the rest = serialisation + network + the manager's bookkeeping
  direct   one whole round through the Collector in each of its two modes ("episodes": N complete episodes; "steps": L steps in
           every environment), wall clock, resets included. The server runs with max_steps_per_episode = COST_MEASURED_EPISODE
           (short, to keep the test minutes long); the per-step cost it yields (the reset taken out) is what the projection uses.

Then tests/cost_model.py turns that into what N buys, for the two ways of feeding a learner (read its docstring):
  update_per = "step"      update every T steps of every environment: time per update, samples per second, gain over N = 1
  update_per = "episode"   update every round (N complete episodes): time per update, episodes per hour, time to collect E
                           episodes (E = the largest N), gain over N = 1
projected to the real episode length L (the YAML's max_steps_per_episode, or COST_EPISODE_STEPS) and rollout length T
(COST_ROLLOUT_STEPS, default 80), with the VRAM of each N next to it, so "what it costs in space and what it gains in time"
is one table. Linear fits  vram(N), step(N), reset(N) = fixed + N x per_env  give the marginal cost of one more environment and
the largest N that fits in the GPU memory (an extrapolation beyond the largest N measured). Everything goes to
reports/env_cost_<timestamp>.json. The test fails only on nonsense (non-finite numbers, memory that does not grow, N environments
slower than N x one).

What is not in the numbers: the policy (a VLA forward pass for N environments) and the learner update. They come on top of the
simulator time and usually depend on N, so read the gain as an upper bound of the speed-up of the whole loop.
"""
import json
import os
import time

import numpy as np
import pytest

from cost_model import derive, fit

pytestmark = pytest.mark.isaac

SIZES = [int(x) for x in os.environ.get("COST_NUM_ENVS", "1,2,4,8,16,32,64").split(",")]
TIMED_STEPS = int(os.environ.get("COST_STEPS", "20"))
RESETS = int(os.environ.get("COST_RESETS", "3"))
MEASURED_EPISODE = int(os.environ.get("COST_MEASURED_EPISODE", "30"))
ROLLOUT_STEPS = int(os.environ.get("COST_ROLLOUT_STEPS", "80"))
WARMUP = 3


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "min": float(a.min()),
            "max": float(a.max())}


def episode_steps_of(config):
    if os.environ.get("COST_EPISODE_STEPS"):
        return int(os.environ["COST_EPISODE_STEPS"])
    import yaml
    with open(config) as f:
        return int(yaml.safe_load(f)["max_steps_per_episode"])


def measure(pool, n):
    from src.envs.collect import Collector
    assert TIMED_STEPS + WARMUP < MEASURED_EPISODE, "the timed steps must fit inside one episode"
    srv = pool.get(n, fresh=True, max_steps_per_episode=MEASURED_EPISODE)
    out = {"num_envs": n, "start_seconds": srv.start_seconds, "measured_episode_steps": MEASURED_EPISODE}
    with srv.client() as env:
        out["vram_ready"] = env.debug("vram")[0]

        # -- resets (the first one also creates FORGE's lazily built tensors: reported apart)
        t0 = time.perf_counter()
        env.reset(seed=0, round=0)
        out["first_reset_s"] = time.perf_counter() - t0
        out["vram_after_reset"] = env.debug("vram")[0]
        env.timings(reset=True)
        times = []
        for i in range(RESETS):
            t0 = time.perf_counter()
            env.reset(seed=0, round=i + 1)
            times.append(time.perf_counter() - t0)
        out["reset_s"] = stats(times)
        out["reset_server_s"] = env.timings(reset=True)[0]["reset"]["mean_ms"] / 1e3

        # -- steps inside one episode (auto_reset off: no reset inside the timed loop)
        env.reset(seed=0, round=0)
        action = env.hold_actions()
        for _ in range(WARMUP):
            env.step(action, auto_reset=False)
        env.timings(reset=True)
        rtt = []
        for _ in range(TIMED_STEPS):
            t0 = time.perf_counter()
            env.step(action, auto_reset=False)
            rtt.append(1e3 * (time.perf_counter() - t0))
        server = env.timings(reset=True)[0]
        out["vram_after_steps"] = env.debug("vram")[0]
        out["step_ms"] = stats(rtt)
        out["server_ms_per_step"] = {k: v["mean_ms"] for k, v in server.items() if k in ("step_physics", "inserted", "observe")}
        out["overhead_ms_per_step"] = out["step_ms"]["mean"] - sum(out["server_ms_per_step"].values())
        out["env_steps_per_s"] = 1e3 * n / out["step_ms"]["mean"]

        # -- one whole round through the Collector, in each mode
        # The start of round 0 is outside the timing; the wall clock is MEASURED_EPISODE steps plus the reset that starts the
        # next round (auto_reset), which is subtracted later with the reset measured above. The policy returns "hold" at once.
        hold = env.hold_actions()
        direct = {}
        for mode, amount in (("episodes", n), ("steps", MEASURED_EPISODE)):
            collector = Collector(env, lambda obs: hold, mode=mode, store_next_obs=False, seed=0)
            collector.start()
            t0 = time.perf_counter()
            data = collector.collect(amount)
            direct[mode] = time.perf_counter() - t0
            if mode == "episodes":
                assert len(data) == n and all(ep.length == MEASURED_EPISODE for ep in data if not ep.terminated)
        out["direct_round_s"] = direct
        out["vram_final"] = env.debug("vram")[0]
    return out


def test_cost_as_a_function_of_the_number_of_environments(isaac_pool, isaac_config, reports_dir):
    rows, errors = [], []
    for n in SIZES:
        try:
            rows.append(measure(isaac_pool, n))
        except Exception as e:             # noqa: BLE001  -- a server that dies at large N is a result (GPU limit), not a crash
            errors.append({"num_envs": n, "error": f"{type(e).__name__}: {str(e)[:400]}"})
            break
    isaac_pool.stop()
    L, T = episode_steps_of(isaac_config), ROLLOUT_STEPS
    report = {"sizes_requested": SIZES, "timed_steps": TIMED_STEPS, "resets": RESETS, "measured_episode_steps": MEASURED_EPISODE,
              "episode_steps": L, "rollout_steps": T, "rows": rows, "errors": errors}

    # per-step cost through the Collector, the reset taken out: what the projection uses
    for r in rows:
        reset_s = r["reset_s"]["mean"]
        r["vram_mib"] = r["vram_final"].get("process_mib") or float("nan")
        r["step_ms_steps_mode"] = 1e3 * (r["direct_round_s"]["steps"] - reset_s) / MEASURED_EPISODE
        r["step_ms_episodes_mode"] = 1e3 * (r["direct_round_s"]["episodes"] - reset_s) / MEASURED_EPISODE
        r["reset_s_mean"] = reset_s
    usable = [{**r, "reset_s": r["reset_s_mean"]} for r in rows if np.isfinite(r["vram_mib"])]
    if usable:
        report["derived"] = derive(usable, episode_steps=L, rollout_steps=T)

    proc = [(r["num_envs"], r["vram_mib"]) for r in rows if np.isfinite(r["vram_mib"])]
    if len(proc) >= 2:
        f = fit(*zip(*proc))
        report["vram_fit_mib"] = f
        total = rows[-1]["vram_final"].get("device_total_mib")
        if total and f["per_env"] > 0:
            report["max_envs_in_gpu_memory"] = int((0.9 * total - f["fixed"]) // f["per_env"])      # 10 % margin, extrapolated
    if len(rows) >= 2:
        report["step_ms_fit"] = fit([r["num_envs"] for r in rows], [r["step_ms"]["mean"] for r in rows])
        report["reset_s_fit"] = fit([r["num_envs"] for r in rows], [r["reset_s_mean"] for r in rows])
        for key in ("step_physics", "observe"):
            pts = [(r["num_envs"], r["server_ms_per_step"][key]) for r in rows if key in r["server_ms_per_step"]]
            if len(pts) >= 2:
                report[f"{key}_ms_fit"] = fit(*zip(*pts))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"env_cost_{stamp}.json"
    path.write_text(json.dumps(report, indent=2))

    print(f"\n=== COST (what one more environment costs)   measured on episodes of {MEASURED_EPISODE} steps")
    print(f"{'N':>4} {'start s':>8} {'VRAM MiB':>9} {'MiB/env':>8} {'step ms':>8} {'physics':>8} {'observe':>8} {'other':>7} {'reset s':>8} {'env-steps/s':>12}")
    for r in rows:
        print(f"{r['num_envs']:>4} {r['start_seconds']:>8.0f} {r['vram_mib']:>9.0f} {r['vram_mib'] / r['num_envs']:>8.0f} "
              f"{r['step_ms']['mean']:>8.1f} {r['server_ms_per_step'].get('step_physics', float('nan')):>8.1f} "
              f"{r['server_ms_per_step'].get('observe', float('nan')):>8.1f} {r['overhead_ms_per_step']:>7.1f} "
              f"{r['reset_s_mean']:>8.2f} {r['env_steps_per_s']:>12.0f}")
    for k in ("vram_fit_mib", "step_ms_fit", "reset_s_fit", "step_physics_ms_fit", "observe_ms_fit", "max_envs_in_gpu_memory"):
        if k in report:
            print(f"{k}: {report[k]}")
    if "derived" in report:
        d = report["derived"]
        print(f"\n=== update_per = step   (T = {T} steps per environment per update, episodes of L = {L} steps, gain over N = {d['baseline_num_envs']})")
        print(f"{'N':>4} {'VRAM MiB':>9} {'update s':>9} {'samples/upd':>12} {'samples/s':>10} {'gain x':>7} {'efficiency':>11} {'reset share':>12} {'gain/extra GiB':>15}")
        for s in d["steps"]:
            g = s["gain_per_extra_gib"]
            print(f"{s['num_envs']:>4} {s['vram_mib']:>9.0f} {s['update_s']:>9.1f} {s['samples_per_update']:>12} {s['samples_per_s']:>10.2f} "
                  f"{s['gain']:>7.2f} {s['efficiency']:>10.0%} {s['reset_share']:>11.0%} {'-' if g is None else f'{g:.2f}':>15}")
        print(f"\n=== update_per = episode   (one update per round of N episodes of L = {L} steps; target {d['target_episodes']} episodes)")
        print(f"{'N':>4} {'VRAM MiB':>9} {'update s':>9} {'episodes/upd':>13} {'episodes/h':>11} {'time for target s':>18} {'gain x':>7} {'efficiency':>11} {'gain/extra GiB':>15}")
        for e in d["episodes"]:
            g = e["gain_per_extra_gib"]
            print(f"{e['num_envs']:>4} {e['vram_mib']:>9.0f} {e['update_s']:>9.1f} {e['episodes_per_update']:>13} {e['episodes_per_hour']:>11.1f} "
                  f"{e['time_for_target_s']:>18.0f} {e['gain']:>7.2f} {e['efficiency']:>10.0%} {'-' if g is None else f'{g:.2f}':>15}")
    for e in errors:
        print(f"STOPPED at N={e['num_envs']}: {e['error']}")
    print(f"report: {path}")

    assert len(rows) >= 1, f"no size could be measured: {errors}"
    for r in rows:
        assert np.isfinite(r["step_ms"]["mean"]) and r["step_ms"]["mean"] > 0 and r["reset_s_mean"] > 0
    if len(proc) >= 2:
        assert proc[-1][1] > proc[0][1], "GPU memory does not grow with the number of environments: NVML figure is wrong?"
    if len(rows) >= 2:
        one, last = rows[0], rows[-1]
        assert last["step_ms"]["mean"] < 1.5 * last["num_envs"] / one["num_envs"] * one["step_ms"]["mean"], \
            "N parallel environments are slower than N x one environment: parallelism brings nothing"
