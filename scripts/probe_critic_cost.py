"""What does the critic's update cost? (no VLA, no simulator: a few minutes on a GPU)

    python -m scripts.probe_critic_cost                                   # the three methods of docs/algos/rl.md, 50 timed updates each
    python -m scripts.probe_critic_cost --variants parl_global_local --steps 200 --set critic_use_tactile=true

The RL loop does one critic update per environment step (UTD = 1: about 150 000 per run of 1000 episodes of 150 steps, plus the warm-up) and each one ranks (and, in variant B, optimises) the
k = 32 candidates of the next states of its batch of 256 transitions, so what it costs decides how long a run is (docs/algos/rl.md: target under 10 h for the whole run). This measures it on the
real network at the real sizes, on SYNTHETIC data (random images and candidates: the cost does not depend on the values), for each variant of configs/experiment/rl/ (best_of_n, parl_global,
parl_global_local; the experiment's `set` and your `--set` give the RLConfig):
  - `host_batch_ms`: `make_batch` (sampling the transitions, gathering 2 x 256 states of 2 x 224 x 224 x 3 bytes and the candidates of s and s' from the numpy buffer), `update_ms`: the jitted update
    from the host arrays (transfer included) up to the end of the device work, `total_ms` = their sum, median / min / max over `--steps` updates after `--warmup-steps` which compile;
  - the card's memory (JAX statistics, peak = the largest so far, so the variants run in order of increasing work);
  - an ESTIMATE of the updates of a whole run: (utd x episodes x steps per episode + warm-up critic steps) x the median total time.
The result is printed and stored in reports/probe_critic_cost_<time>_<job id>.json. Run it through jobs/job_probe_critic_cost.sh (the XLA settings of the other probes: the critic shares the card).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from scripts.probe_batch_cost import device_memory
from scripts.sft import enable_compilation_cache, parse_assignments
from src.eval.sweep import gpu_name
from src.models.interface import ROOT

EXPERIMENTS = {"best_of_n": "configs/experiment/rl/pi05_best_of_n.yaml", "parl_global": "configs/experiment/rl/pi05_parl_global.yaml",
               "parl_global_local": "configs/experiment/rl/pi05_parl_global_local.yaml"}
HORIZON = 16                       # pi0.5's chunk (docs/algos/rl.md)


def synthetic_data(cfg, episodes: int, plans: int, seed: int = 0, horizon: int = HORIZON, state_dim: int = 8):
    """A replay buffer of `episodes` episodes of `plans` plans (random images / joints / chunks, half of them warm-up) and a cache holding random candidates for every state that needs them."""
    from src.algos.rl_buffer import Episode, ReplayBuffer
    from src.algos.rl_candidates import CandidateCache
    from src.models.interface import Candidates
    rng = np.random.default_rng(seed)
    touch_dim = cfg.critic_touch.dim
    buf = ReplayBuffer(episodes * (plans + 1), horizon, cfg.gamma, cfg.step_reward, touch_dim=touch_dim)
    for i in range(episodes):
        shape = (plans + 1, 224, 224, 3)
        buf.add_episode(Episode(exterior=rng.integers(0, 255, shape, dtype=np.uint8), wrist=rng.integers(0, 255, shape, dtype=np.uint8),
                                joints=rng.normal(size=(plans + 1, 7)).astype(np.float32), gripper=rng.random((plans + 1, 1)).astype(np.float32),
                                actions=rng.normal(size=(plans, horizon, 8)).astype(np.float32), success=bool(i % 2), prompt="Insert the peg into the socket.",
                                touch=rng.normal(size=(plans + 1, touch_dim)).astype(np.float32) if touch_dim else None), warmup=i < episodes // 2)
    cache = CandidateCache(cfg.num_candidates, horizon, state_dim, capacity=len(buf))
    ids = buf.candidate_ids()
    cands = rng.normal(size=(len(ids), cfg.num_candidates, horizon, 8)).astype(np.float32)
    cache.put(ids, Candidates(normalized=cands, states=np.zeros((len(ids), state_dim), np.float32), targets=cands, infer_ms=0.0), version=0)
    return buf, cache


def summarise_ms(values: List[float]) -> Dict[str, float]:
    return {"median": round(statistics.median(values), 2), "min": round(min(values), 2), "max": round(max(values), 2)}


def measure(trainer, buf, cache, steps: int, warmup_steps: int, clock: Callable[[], float] = time.perf_counter, block: Optional[Callable] = None) -> dict:
    """`warmup_steps` updates (the first compiles), then `steps` timed ones; the host and device parts apart."""
    import jax
    from src.algos.rl_critic_update import make_batch
    block = block or jax.block_until_ready
    rng = np.random.default_rng(1)
    first_start = clock()
    for i in range(warmup_steps):
        block(trainer.update(make_batch(buf, cache, rng, trainer.cfg)))
        if i == 0:
            first_ms = (clock() - first_start) * 1e3
    host, device = [], []
    for _ in range(steps):
        t0 = clock()
        batch = make_batch(buf, cache, rng, trainer.cfg)
        t1 = clock()
        block(trainer.update(batch))
        t2 = clock()
        host.append((t1 - t0) * 1e3)
        device.append((t2 - t1) * 1e3)
    total = [h + d for h, d in zip(host, device)]
    return {"first_update_ms": round(first_ms, 1), "host_batch_ms": summarise_ms(host), "update_ms": summarise_ms(device), "total_ms": summarise_ms(total), "steps": steps}


def estimate(rows: List[dict], cfg_by_name: Dict[str, object], episodes: int, steps_per_episode: int) -> List[dict]:
    """Hours of critic updates over a whole run for each measured variant: (utd x environment steps + warm-up critic steps) x the median total time of an update."""
    out = []
    for row in rows:
        cfg = cfg_by_name[row["variant"]]
        updates = cfg.utd * episodes * steps_per_episode + cfg.warmup_critic_steps
        out.append({"variant": row["variant"], "updates": updates, "hours": round(updates * row["total_ms"]["median"] / 3.6e6, 2)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS), help="which methods to measure (default all three, in this order)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a key of configs/algo/rl.yaml (applied to every variant)")
    ap.add_argument("--steps", type=int, default=50, help="timed updates per variant (default 50)")
    ap.add_argument("--warmup-steps", type=int, default=3, help="updates before timing, the first compiles (default 3)")
    ap.add_argument("--episodes", type=int, default=40, help="episodes of the synthetic buffer (default 40; its size does not change the cost per update)")
    ap.add_argument("--plans", type=int, default=19, help="plans per synthetic episode (default 19: 150 steps, a chunk every 8)")
    ap.add_argument("--run-episodes", type=int, default=1000, help="online episodes of the run to estimate (default 1000)")
    ap.add_argument("--run-steps", type=int, default=150, help="environment steps per episode of that run (default 150)")
    args = ap.parse_args(argv)
    if min(args.steps, args.warmup_steps, args.episodes, args.plans) < 1:
        raise SystemExit("--steps, --warmup-steps, --episodes and --plans must be 1 or more")
    enable_compilation_cache()
    from src.algos.rl_config import load_rl_experiment
    from src.algos import rl_critic
    from src.algos.rl_critic_update import CriticTrainer
    overrides = parse_assignments(args.set)
    print(f"critic update cost on {gpu_name()}, memory before: {device_memory()}", flush=True)
    rows, cfgs = [], {}
    for name in args.variants:
        _, cfg = load_rl_experiment(EXPERIMENTS[name], overrides)
        cfgs[name] = cfg
        buf, cache = synthetic_data(cfg, args.episodes, args.plans)
        trainer = CriticTrainer(rl_critic.CriticNet.from_config(cfg), cfg, horizon=HORIZON)
        print(f"[{name}] k={cfg.num_candidates} keep_top={cfg.keep_top} local_steps={cfg.local_steps} batch={cfg.critic_batch_size} parameters={rl_critic.num_parameters(trainer.critic.params)}", flush=True)
        row = {"variant": name, **measure(trainer, buf, cache, args.steps, args.warmup_steps), "memory": device_memory()}
        rows.append(row)
        print(f"  first update (compilation) {row['first_update_ms']} ms | host batch {row['host_batch_ms']} | update {row['update_ms']} | total {row['total_ms']} | memory {row['memory']}", flush=True)
        del trainer, buf, cache
    est = estimate(rows, cfgs, args.run_episodes, args.run_steps)
    print(f"\n=== ESTIMATE of the critic updates of a run ({args.run_episodes} episodes of {args.run_steps} steps, UTD per environment step, warm-up included)")
    for e in est:
        print(f"  {e['variant']:<18} {e['updates']} updates -> {e['hours']} h")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = ROOT / "reports" / f"probe_critic_cost_{stamp}_{os.environ.get('SLURM_JOB_ID') or 'pid' + str(os.getpid())}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"gpu": gpu_name(), "overrides": overrides, "measured": rows, "estimate": est,
                                "assumptions": {"run_episodes": args.run_episodes, "run_steps": args.run_steps, "synthetic_data": True}}, indent=1))
    print(f"report: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
