"""One round of a FORGE rollout driven by a VLA policy through src/models/interface.py: chunks executed `replan` steps at a time, the
task progress recorded at every step. numpy only (no JAX): it works with any `VLAPolicy`, openpi or a test double.

Used by scripts/probe_vla.py today; an algorithm's evaluation will use it too.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence

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
                on_step: Optional[Callable[[int, object, List[int]], None]] = None, *, round_id: int = 0, active: Optional[Sequence[bool]] = None,
                row_seeds: Optional[Callable[[int, int], int]] = None, pad_to: Optional[int] = None,
                on_env_done: Optional[Callable[[int, dict], None]] = None) -> dict:
    """One round (one episode per environment) driven by `policy` (src/models/interface.py: one batched call for the environments that
    need a new chunk). `env` is a VectorEnvClient. A chunk is executed `replan` steps, then the model is called again from the new
    observation. Stops at `steps` steps or when every episode has ended. auto_reset is off: nothing starts by itself.
    `on_step(t, res, plans_per_env)` is called after the reset (t = 0) and after every step: used to record the video.

    The round is `reset(seed=seed, round=round_id)`: the start of every environment is a function of those, of the number of environments and of its index.
    `active` (default: all): the environments that play. The others are held where they are (their joint targets are their own joint positions) and the policy
    is never called for them; they still exist in the simulator, so the start states of the others do not depend on which ones play. The rollout ends when every
    ACTIVE episode has ended.
    `row_seeds(env, k)` -> the seed of the sampling noise of the k-th plan (0-based) of environment `env` (`VLAPolicy.plan(row_seeds=...)`); `pad_to`: the batch size
    of every model call (`VLAPolicy.plan(pad_to=...)`). `on_env_done(env, summary)` is called when the episode of an environment ends, with its entry of `envs`."""
    n, prompt = env.num_envs, env.spec.prompt
    semantics = policy.info.output_semantics
    replan = max(1, min(int(replan), policy.info.action_horizon))
    play = np.ones(n, bool) if active is None else np.asarray(active, bool)
    if play.shape != (n,):
        raise ValueError(f"active must have one entry per environment ({n}), got shape {play.shape}")
    res = env.reset(seed=seed, round=round_id)
    actions = np.zeros((n, P.ACTION_DIM), np.float32)         # rows of environments that ended are ignored by the server (not acting)
    chunk: List[Optional[np.ndarray]] = [None] * n
    used = [0] * n
    ended = ~play                                              # an environment that does not play never asks the policy for anything
    plans: List[dict] = []
    track = [[] for _ in range(n)]              # |target - joint state after the step| (rad, max over the 7 joints), per step
    first_q = np.array(res.obs[P.JOINT_POSITION], dtype=np.float32)
    n_plans = [0] * n
    plan_ms = [[] for _ in range(n)]            # wall time of the model calls that planned for each environment (shared with the others of the batch)
    decoded = [[0, 0] for _ in range(n)]       # pi0-FAST: [chunks, chunks that could not be decoded] per environment
    progress = [[{"t": 0, **progress_sample(res.obs, e)}] for e in range(n)]
    finished: dict = {}                        # env -> its summary, taken on the step its episode ended (the flags of `res` only say what happened on the LAST step)

    def summary(e: int) -> dict:
        out = {"env": e, "success": bool(res.success[e]), "terminated": bool(res.terminated[e]), "truncated": bool(res.truncated[e]),
               "steps": len(track[e]), "tracking_err_max_rad": max(track[e]) if track[e] else None,
               "tracking_err_mean_rad": float(np.mean(track[e])) if track[e] else None,
               "joint_motion_rad": float(np.abs(np.asarray(res.obs[P.JOINT_POSITION][e]) - first_q[e]).max()),
               "progress": progress_summary(progress[e], bool(res.success[e])), "progress_series": progress[e], "plan_wall_ms": list(plan_ms[e])}
        if decoded[e][0]:
            out["decode"] = {"chunks": decoded[e][0], "failed": decoded[e][1], "failed_rate": decoded[e][1] / decoded[e][0]}
        return out

    if on_step:
        on_step(0, res, n_plans)
    t = 0
    while t < steps and not bool(ended[play].all()) and not res.all_done:
        need = [e for e in range(n) if not ended[e] and (chunk[e] is None or used[e] >= replan)]
        if need:
            kwargs = {}
            if row_seeds is not None:
                kwargs["row_seeds"] = [row_seeds(e, n_plans[e]) for e in need]
            if pad_to is not None:
                kwargs["pad_to"] = pad_to
            t_call = time.monotonic()
            plan = policy.plan(take(res.obs, need), prompt, **kwargs)
            wall_ms = 1e3 * (time.monotonic() - t_call)           # includes the conversions around the model call
            valid = getattr(policy, "last_valid", None)
            for k, e in enumerate(need):
                q_plan = np.asarray(res.obs[P.JOINT_POSITION][e], dtype=np.float32)
                raw = plan.raw[k, 0]
                chunk[e], used[e] = plan.targets[k, 0], 0
                n_plans[e] += 1
                plan_ms[e].append(round(wall_ms, 1))
                if valid is not None and len(valid) == len(need):
                    decoded[e][0] += 1
                    decoded[e][1] += int(not valid[k, 0])
                plans.append({"step": t, "env": e, "batch": len(need), "raw_shape": list(raw.shape), "q_plan": q_plan.tolist(),
                              "raw_first": raw[0].tolist(), "raw_last": raw[-1].tolist(),
                              "infer_ms": plan.infer_ms, "wall_ms": round(wall_ms, 1), **looks_like(raw, q_plan)})
        for e in range(n):
            if not play[e]:                                       # held where it is
                actions[e, :7] = np.asarray(res.obs[P.JOINT_POSITION][e], dtype=np.float32)[:7]
                actions[e, 7] = float(np.asarray(res.obs[P.GRIPPER_POSITION][e]).reshape(-1)[0])
            elif not ended[e]:
                actions[e] = chunk[e][used[e]]
                used[e] += 1
        res = env.step(actions, auto_reset=False)
        for e in range(n):
            if not ended[e]:
                track[e].append(float(np.abs(actions[e, :7] - res.obs[P.JOINT_POSITION][e][:7]).max()))
                progress[e].append({"t": t + 1, **progress_sample(res.obs, e)})
        newly = [e for e in range(n) if play[e] and not ended[e] and bool(res.done[e])]
        ended = ended | res.done
        t += 1
        if on_step:
            on_step(t, res, n_plans)
        for e in newly:
            finished[e] = summary(e)
            if on_env_done:
                on_env_done(e, finished[e])
    per_env = [finished.get(e) or summary(e) for e in range(n)]
    return {"steps_run": t, "replan": replan, "semantics_used": semantics, "all_done": bool(ended[play].all()), "envs": per_env, "plans": plans}
