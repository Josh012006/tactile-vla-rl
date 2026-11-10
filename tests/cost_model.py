"""Arithmetic of the cost report (pure numpy, tested without a GPU in tests/test_cost_model.py).

Two ways of feeding a learner, and what N parallel environments buy for each:

  update_per = "step"      the learner updates every T steps of every environment (collector mode "steps", Rollout of shape
                           (T, N)). The time between two updates is T steps of the simulation plus the resets that fall inside:
                           a round lasts L steps, so T / L resets per update on average. Each update sees N x T samples.
  update_per = "episode"   the learner updates once per round, i.e. after N complete episodes (collector mode "episodes"). The time
                           between two updates is one reset plus L steps (L = max_steps, reached when at least one environment
                           does not succeed, which is the usual case for a policy that is still learning; successes end earlier
                           but wait for the others). Each update sees N episodes.

Gain = how many times faster than the smallest N measured (normally 1 environment) the same amount of data is collected:
  "step"   : samples per second, N x T / (T x step + T / L x reset)
  "episode": time to collect E episodes, ceil(E / N) x (reset + L x step), E = the largest N measured.
Time per step and per reset are the MEASURED ones (client side, through the Collector), so the gain includes the client's own work;
what is NOT included is the policy (a VLA forward pass for N environments) and the learner update: both come on top and are
the same for the two columns being compared only if they do not depend on N, which is usually false for the policy.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def fit(xs: Sequence[float], ys: Sequence[float]) -> Optional[dict]:
    """least squares  y = fixed + per_env x ; returns {"fixed", "per_env", "r2"}, or None with fewer than 2 points."""
    x, y = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    if len(x) < 2:
        return None
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    return {"fixed": float(a), "per_env": float(b), "r2": r2}


def derive(rows: List[dict], episode_steps: int, rollout_steps: int, target_episodes: Optional[int] = None) -> Dict[str, List[dict]]:
    """Cost and gain as a function of N.

    rows: one dict per N, increasing N, with
        num_envs, vram_mib, reset_s, step_ms_steps_mode (per step through the Collector in "steps" mode, reset not included),
        step_ms_episodes_mode (same in "episodes" mode)
    episode_steps L (the YAML's max_steps_per_episode), rollout_steps T. Returns {"steps": [...], "episodes": [...]}."""
    if not rows:
        raise ValueError("no measurement")
    L, T = int(episode_steps), int(rollout_steps)
    if L < 1 or T < 1:
        raise ValueError("episode_steps and rollout_steps must be >= 1")
    ns = [int(r["num_envs"]) for r in rows]
    if ns != sorted(set(ns)):
        raise ValueError(f"rows must have distinct, increasing num_envs, got {ns}")
    target = int(target_episodes) if target_episodes else ns[-1]
    base = rows[0]
    base_vram = float(base["vram_mib"])

    steps_rows, episode_rows = [], []
    for r in rows:
        n, reset_s, vram = int(r["num_envs"]), float(r["reset_s"]), float(r["vram_mib"])

        step_s = float(r["step_ms_steps_mode"]) / 1e3
        t_update = T * step_s + (T / L) * reset_s
        samples = n * T
        steps_rows.append({"num_envs": n, "vram_mib": vram, "update_s": t_update, "samples_per_update": samples,
                           "samples_per_s": samples / t_update, "reset_share": (T / L) * reset_s / t_update})

        step_e = float(r["step_ms_episodes_mode"]) / 1e3
        t_batch = reset_s + L * step_e
        rounds = math.ceil(target / n)
        episode_rows.append({"num_envs": n, "vram_mib": vram, "update_s": t_batch, "episodes_per_update": n,
                             "episodes_per_hour": 3600.0 * n / t_batch, "time_for_target_s": rounds * t_batch,
                             "target_episodes": target, "reset_share": reset_s / t_batch})

    for rows_out, key, better_when_higher in ((steps_rows, "samples_per_s", True), (episode_rows, "time_for_target_s", False)):
        ref = rows_out[0][key]
        for r, src in zip(rows_out, rows):
            gain = r[key] / ref if better_when_higher else ref / r[key]
            r["gain"] = gain
            r["efficiency"] = gain * rows_out[0]["num_envs"] / r["num_envs"]          # 1.0 = perfectly linear
            extra_gib = (r["vram_mib"] - base_vram) / 1024.0
            r["gain_per_extra_gib"] = (gain - 1.0) / extra_gib if extra_gib > 1e-9 else None
    return {"steps": steps_rows, "episodes": episode_rows, "baseline_num_envs": ns[0],
            "episode_steps": L, "rollout_steps": T, "target_episodes": target}
