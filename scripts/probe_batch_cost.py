"""What does the MODEL cost when it plans for K environments at once? (no simulator: a few minutes on a GPU)

    python -m scripts.probe_batch_cost --model pi05 --batch-sizes 1 8 32 64 100
    python -m scripts.probe_batch_cost --model pi0_fast --batch-sizes 1 8 32 64 100

Run it through jobs/job_probe_batch_cost.sh. The evaluation (scripts/eval_sft.py) plays K episodes per server and calls the model ONCE per replan for all of them
(`plan(..., pad_to=K)`: the same batch shape every time, so one compilation). What that costs is not known until measured; this probe measures it, per batch size B:
  - the first call (compilation included) and then `--repeats` more calls of `VLAPolicy.plan(obs of B rows, row_seeds=..., pad_to=B)` (the way the evaluation calls it): median, min, max, in
    ms for the call and per row; a batch that does not fit (out of memory) is recorded as failed and the larger ones are not tried;
  - the memory the card holds after it (`jax` device statistics: the peak is the largest so far, and the sizes run in increasing order, so each peak is the one of the largest batch).
The simulator's side (docs/envs/isaaclab_env.md: a step costs about 531 ms + 2.29 ms per environment, 8.1 GiB + 26-29.5 MiB per environment) is NOT measured here: the estimate printed
at the end combines the two with `--steps` and `--replan` and is an ESTIMATE (it assumes the simulator figures of the docs and no overlap). The real number is a real evaluation:
    sbatch jobs/job_eval_sft.sh --run <run> --steps 0 --n-episodes 100 --envs-per-server 100 --output-dir <a probe folder>
whose log prints the wall time of every batch.
The result is printed and stored in reports/probe_batch_cost_<model>_<time>_<job id>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from scripts.eval_sft import with_deterministic_xla
from scripts.probe_determinism import PROMPT, synthetic_obs
from scripts.probe_eval_parallel import probe_tag
from scripts.sft import enable_compilation_cache, parse_assignments
from src.envs import protocol as P
from src.eval.episodes import plan_seed
from src.eval.sweep import gpu_name
from src.models.interface import ROOT, apply_overrides, load_model_cfg

SIM_STEP_FIXED_MS = 531.0        # docs/envs/isaaclab_env.md: step = 531 ms + 2.29 ms per environment (measured earlier, not here)
SIM_STEP_PER_ENV_MS = 2.29


def batch_obs(rows: int, seed: int = 0) -> dict:
    """`rows` observations in the env's format: every row has its own images (synthetic_obs of seed + row), the joint positions of a Franka-like pose."""
    singles = [synthetic_obs(seed + r) for r in range(rows)]
    return {k: np.concatenate([o[k] for o in singles], axis=0) for k in (P.EXTERIOR_IMAGE, P.WRIST_IMAGE, P.JOINT_POSITION, P.GRIPPER_POSITION)}


def device_memory() -> dict:
    """The card's memory according to JAX (bytes in use and peak), or {} where the backend does not say (CPU)."""
    try:
        import jax
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:                                                  # a backend without statistics is not an error here
        return {}
    return {k: int(stats[k]) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in stats}


def measure(policy, batch_sizes: Sequence[int], repeats: int, prompt: str = PROMPT, clock: Callable[[], float] = time.monotonic,
            memory: Callable[[], dict] = device_memory, say: Callable[[str], None] = print) -> List[dict]:
    """For every batch size B (in increasing order): the time of the first call (it compiles) and of `repeats` more, the way an evaluation calls the policy."""
    out = []
    for b in sorted(set(int(x) for x in batch_sizes)):
        obs = batch_obs(b)
        seeds = [plan_seed(1000 + r, 0) for r in range(b)]
        times = []
        try:
            for call in range(repeats + 1):
                t0 = clock()
                plan = policy.plan(obs, prompt, row_seeds=seeds, pad_to=b)
                times.append(1e3 * (clock() - t0))
                if plan.targets.shape[0] != b:
                    raise RuntimeError(f"asked for {b} rows, the policy returned {plan.targets.shape[0]}")
        except RuntimeError as e:                                       # XLA's out-of-memory is a RuntimeError (XlaRuntimeError): the largest batch that fits is the result
            if "returned" in str(e):
                raise
            out.append({"batch": b, "failed": str(e).strip().splitlines()[0][:300], "memory": memory()})
            say(f"  B={b:<4} FAILED: {out[-1]['failed']}")
            break
        first, rest = times[0], times[1:]
        row = {"batch": b, "first_call_ms": round(first, 1), "calls": len(rest), "median_ms": round(statistics.median(rest), 1) if rest else None,
               "min_ms": round(min(rest), 1) if rest else None, "max_ms": round(max(rest), 1) if rest else None,
               "median_ms_per_row": round(statistics.median(rest) / b, 2) if rest else None, "memory": memory()}
        out.append(row)
        say(f"  B={b:<4} first call {first:9.1f} ms | median {row['median_ms']} ms (min {row['min_ms']}, max {row['max_ms']}) = {row['median_ms_per_row']} ms per row | memory {row['memory'].get('peak_bytes_in_use', 0) / 2 ** 30:.2f} GiB peak")
    return out


def estimate(rows: List[dict], steps: int, replan: int, episodes: Optional[int] = None) -> List[dict]:
    """The time of one batch of B episodes (every one running `steps` steps, planning every `replan`) and of one episode in it: the simulator figures of the docs plus the measured model time.
    An ESTIMATE: no overlap between the model and the simulator, the simulator's cost as measured with other batch sizes."""
    plans = -(-steps // max(1, replan))
    out = []
    for r in rows:
        if r.get("failed") or r["median_ms"] is None:
            continue
        b = r["batch"]
        sim_s = steps * (SIM_STEP_FIXED_MS + SIM_STEP_PER_ENV_MS * b) / 1e3
        model_s = plans * r["median_ms"] / 1e3
        total = sim_s + model_s
        out.append({"batch": b, "plans": plans, "sim_s": round(sim_s, 1), "model_s": round(model_s, 1), "batch_s": round(total, 1), "s_per_episode": round(total / b, 2),
                    "episodes_per_hour": round(3600 * b / total, 1)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="configs/model/<name>.yaml (pi05 or pi0_fast)")
    ap.add_argument("--model-set", action="append", default=[], metavar="KEY=VALUE", help="override a key of the model YAML: <component>.<key>")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64, 100], help="rows per model call (default 1 8 32 64 100)")
    ap.add_argument("--repeats", type=int, default=5, help="timed calls per batch size after the first, which compiles (default 5)")
    ap.add_argument("--steps", type=int, default=150, help="episode length of the estimate (default 150)")
    ap.add_argument("--replan", type=int, default=8, help="steps per chunk of the estimate (default 8)")
    ap.add_argument("--no-deterministic-xla", action="store_true", help="leave XLA_FLAGS as they are")
    args = ap.parse_args(argv)
    if min(args.batch_sizes) < 1 or args.repeats < 1:
        raise SystemExit("--batch-sizes must be 1 or more and --repeats 1 or more")
    if not args.no_deterministic_xla:
        os.environ["XLA_FLAGS"] = with_deterministic_xla(os.environ.get("XLA_FLAGS", ""))
    cfg = apply_overrides(load_model_cfg(args.model), parse_assignments(args.model_set))
    enable_compilation_cache()
    from src.models import load_model
    policy = load_model(cfg, seed=0)
    print(f"batch cost of {args.model} on {gpu_name()}, XLA_FLAGS='{os.environ.get('XLA_FLAGS', '')}', memory before: {device_memory()}", flush=True)
    rows = measure(policy, args.batch_sizes, args.repeats)
    est = estimate(rows, args.steps, args.replan)
    print(f"\n=== ESTIMATE of one batch of B episodes ({args.steps} steps, a chunk every {args.replan}; the simulator's cost is the docs' figure, not measured here)")
    for e in est:
        print(f"  B={e['batch']:<4} simulator {e['sim_s']:7.1f} s + model {e['model_s']:7.1f} s = {e['batch_s']:7.1f} s -> {e['s_per_episode']:6.2f} s per episode, {e['episodes_per_hour']} episodes/hour")
    tag = probe_tag(args.model, time.strftime("%Y%m%d_%H%M%S"), os.environ.get("SLURM_JOB_ID"))
    path = ROOT / "reports" / f"probe_batch_cost_{tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"model": args.model, "gpu": gpu_name(), "xla_flags": os.environ.get("XLA_FLAGS", ""), "measured": rows, "estimate": est,
                                "estimate_assumptions": {"steps": args.steps, "replan": args.replan, "sim_step_fixed_ms": SIM_STEP_FIXED_MS, "sim_step_per_env_ms": SIM_STEP_PER_ENV_MS}}, indent=1))
    print(f"report: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
