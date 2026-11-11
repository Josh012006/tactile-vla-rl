"""One round of a FORGE rollout driven by a VLA policy through src/models/interface.py: chunks executed `replan` steps at a time, the
task progress recorded at every step. numpy only (no JAX): it works with any `VLAPolicy`, openpi or a test double.

Used by scripts/probe_vla.py today; an algorithm's evaluation will use it too.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np

from src.envs import protocol as P
from src.eval.progress import progress_sample, progress_summary
from src.models.interface import VLAPolicy, take


def looks_like(raw_chunk: np.ndarray, q_plan: np.ndarray) -> dict:
    """Indicator only: is the raw chunk closer to zero (offsets) or to the current joint state (absolute targets)?"""
    raw = np.asarray(raw_chunk, dtype=np.float64)[:, :7]
    q = np.asarray(q_plan, dtype=np.float64)[:7]
    d_off, d_abs = float(np.abs(raw).mean()), float(np.abs(raw - q).mean())
    return {"mean_abs_raw": d_off, "mean_abs_raw_minus_state": d_abs, "looks_like": "offsets" if d_off < d_abs else "absolute"}


def run_rollout(env, policy: VLAPolicy, steps: int, replan: int, seed: int = 0,
                on_step: Optional[Callable[[int, object, List[int]], None]] = None) -> dict:
    """One round (one episode per environment) driven by `policy` (src/models/interface.py: one batched call for the environments that
    need a new chunk). `env` is a VectorEnvClient. A chunk is executed `replan` steps, then the model is called again from the new
    observation. Stops at `steps` steps or when every episode has ended. auto_reset is off: nothing starts by itself.
    `on_step(t, res, plans_per_env)` is called after the reset (t = 0) and after every step: used to record the video."""
    n, prompt = env.num_envs, env.spec.prompt
    semantics = policy.info.output_semantics
    replan = max(1, min(int(replan), policy.info.action_horizon))
    res = env.reset(seed=seed, round=0)
    actions = np.zeros((n, P.ACTION_DIM), np.float32)         # rows of environments that ended are ignored by the server (not acting)
    chunk: List[Optional[np.ndarray]] = [None] * n
    used = [0] * n
    ended = np.zeros(n, bool)
    plans: List[dict] = []
    track = [[] for _ in range(n)]              # |target - joint state after the step| (rad, max over the 7 joints), per step
    first_q = np.array(res.obs[P.JOINT_POSITION], dtype=np.float32)
    n_plans = [0] * n
    progress = [[{"t": 0, **progress_sample(res.obs, e)}] for e in range(n)]
    if on_step:
        on_step(0, res, n_plans)
    t = 0
    while t < steps and not res.all_done:
        need = [e for e in range(n) if not ended[e] and (chunk[e] is None or used[e] >= replan)]
        if need:
            t_call = time.monotonic()
            plan = policy.plan(take(res.obs, need), prompt)
            wall_ms = 1e3 * (time.monotonic() - t_call)           # includes the conversions around the model call
            for k, e in enumerate(need):
                q_plan = np.asarray(res.obs[P.JOINT_POSITION][e], dtype=np.float32)
                raw = plan.raw[k, 0]
                chunk[e], used[e] = plan.targets[k, 0], 0
                n_plans[e] += 1
                plans.append({"step": t, "env": e, "batch": len(need), "raw_shape": list(raw.shape), "q_plan": q_plan.tolist(),
                              "raw_first": raw[0].tolist(), "raw_last": raw[-1].tolist(),
                              "infer_ms": plan.infer_ms, "wall_ms": round(wall_ms, 1), **looks_like(raw, q_plan)})
        for e in range(n):
            if not ended[e]:
                actions[e] = chunk[e][used[e]]
                used[e] += 1
        res = env.step(actions, auto_reset=False)
        for e in range(n):
            if not ended[e]:
                track[e].append(float(np.abs(actions[e, :7] - res.obs[P.JOINT_POSITION][e][:7]).max()))
                progress[e].append({"t": t + 1, **progress_sample(res.obs, e)})
        ended |= res.done
        t += 1
        if on_step:
            on_step(t, res, n_plans)
    per_env = [{"env": e, "success": bool(res.success[e]), "terminated": bool(res.terminated[e]), "truncated": bool(res.truncated[e]),
                "steps": len(track[e]), "tracking_err_max_rad": max(track[e]) if track[e] else None,
                "tracking_err_mean_rad": float(np.mean(track[e])) if track[e] else None,
                "joint_motion_rad": float(np.abs(np.asarray(res.obs[P.JOINT_POSITION][e]) - first_q[e]).max()),
                "progress": progress_summary(progress[e], bool(res.success[e])), "progress_series": progress[e]} for e in range(n)]
    return {"steps_run": t, "replan": replan, "semantics_used": semantics, "all_done": bool(res.all_done), "envs": per_env, "plans": plans}
