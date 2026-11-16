"""What does a PA-RL run cost, stage by stage? (no simulator: the real pi0.5 of an SFT checkpoint, synthetic states, a few minutes on a GPU)

    python -m scripts.probe_parl_cost --start peg_insert_pi05_lora_teleop 4000                                   # variant B (global + local), the dearer of the two
    python -m scripts.probe_parl_cost --experiment configs/experiment/rl/pi05_parl_global.yaml --start <run> <step>   # variant A
    python -m scripts.probe_parl_cost --start <run> <step> --states 256 --refresh-states 32 --save-dir $SLURM_TMPDIR

Run it through jobs/job_probe_parl_cost.sh (same XLA settings as a run: the VLA and the critic share half of the card with the env server).

What PA-RL does after every round (src/algos/rl_run.py `_distill`), on every stored state, and what this measures PER UNIT with the code of the run itself (`Distiller`, `refresh`), on synthetic
states (random images / joints / candidates: the cost does not depend on the values; the buffer is small, the costs are linear in the number of states):
  1. ANNOTATION  `Distiller.annotate`: the critic ranks (and, in B, optimises) the 32 cached candidates of each state                         -> seconds per state
  2. TRAINING    `VLATrainer.step` through `Distiller.distill`, prefetch included, batch `distill_batch_size` in micro-batches of `distill_micro_batch_size`   -> seconds per step
  3. REFRESH     `rl_candidates.refresh`: the new VLA draws the 32 candidates of every state again (calls of `rows_per_sample_call` rows)                  -> seconds per state
  4. SAVES (only with --save-dir): the VLA's checkpoint (~1.7 GB) and the resume state (VLA + Adam) written once each; a run does it after every round, on the cluster's disk (slower than a node's
     $SLURM_TMPDIR: this is a floor)
and the card's memory after each (JAX statistics: the peak is what the VLA + Adam + the critic need, the env server's 9-11 GB comes on top).
The first call of each stage compiles and is not timed (it is printed: it is paid once per run). The ESTIMATE then adds up, for each round of the run (the buffer holds the warm-up and the rounds so far, `--plans`
states per episode), annotation + steps + refresh [+ saves], and, as constants taken from the earlier measurements (docs/algos/rl.md section 5, probe_critic_cost), the collection, the evaluations and the
critic updates: an ESTIMATE of the whole run, not a measurement of it. Stored in reports/probe_parl_cost_<time>_<job id>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from scripts.probe_batch_cost import device_memory
from scripts.probe_critic_cost import synthetic_data
from scripts.sft import enable_compilation_cache, parse_assignments
from src.eval.sweep import gpu_name
from src.models.interface import ROOT

DEFAULT_EXPERIMENT = "configs/experiment/rl/pi05_parl_global_local.yaml"
SYNTHETIC_PLANS = 19               # plans per synthetic episode (150 steps, a chunk every 8)
# What the earlier measurements say of the rest of a run (docs/algos/rl.md section 5, the critic probe of 15 Nov. 2026): used by the estimate, change them on the command line.
COLLECT_ROUND_S = 280.0            # a round of 50 episodes: about 182 s of model + 97 s of simulator
EVAL_S = 480.0                     # an evaluation of 100 episodes
CRITIC_H = 1.92                    # 155 000 critic updates at 44.7 ms in the run's loop (batches prepared ahead; 78.9 ms one after the other = 3.4 h), variant B, probe of 15 Nov. 2026
SERVER_GIB, CARD_GIB = 11.0, 46.0  # an env server of 50-100 environments takes 9-11 GB; the L40S has 46 GB


def stage_states(cfg, plans: int) -> List[int]:
    """The states the stages work on after each round r = 1..R: the buffer holds the warm-up episodes and the r rounds so far, `plans` states per episode."""
    return [(cfg.warmup_episodes + r * cfg.envs_per_batch) * plans for r in range(1, cfg.rounds + 1)]


def compiled_memory(fn, *args):
    """The compiler's memory statistics of `fn(*args)`, compiled and not run. The arguments go in as flat arrays and are put back together INSIDE the traced function: `jit(fn).lower(obs)` rebuilds
    `obs` with placeholders (`ArgInfo`) for its leaves, and openpi's `Observation` type-checks its leaves in its constructor and refuses them (the first version of this probe failed there)."""
    import jax
    leaves, treedef = jax.tree.flatten(args)
    return jax.jit(lambda *xs: fn(*jax.tree.unflatten(treedef, xs))).lower(*leaves).compile().memory_analysis()


def grad_memory(distiller, buf, ids, micros: Sequence[int], memory: Callable[[], dict] = device_memory, margin: float = 0.03) -> Dict[str, object]:
    """What XLA itself says the gradient pass needs at each micro-batch size, WITHOUT running it (`lower().compile().memory_analysis()`: the compiler's buffer plan, exact for the compiled program).
    A pass needs, on top of what is resident (the VLA, the trainable copy, Adam, the critic: `bytes_in_use` now), its temporaries and its output (the loss and one gradient tree), and the running sum of the
    micro-batches is a second tree: need = resident + temp + 2 x output. The largest micro-batch whose need is under the allocator's limit (less `margin`) is `chosen` (None if even the smallest does not fit)."""
    import jax
    trainer, batch = distiller.trainer, distiller.cfg.distill_batch_size
    h = distiller.policy.info.action_horizon
    obs, prompts = buf.observations(ids[:batch])
    observation, actions = distiller.policy.training_batch_from_normalized(obs, prompts, np.zeros((batch, h, 8), np.float32))
    observation, draws = trainer._prepare(jax.random.key(0), observation, actions)
    mem = memory()
    resident, limit = mem.get("bytes_in_use", 0), mem.get("bytes_limit", 0)
    rows, chosen = {}, None
    for micro in sorted({m for m in micros if 0 < m <= batch and batch % m == 0}, reverse=True):
        cut = lambda x: x[:micro]                                                                                                  # noqa: E731
        stats = compiled_memory(trainer._grad, trainer.trainable, trainer.frozen, jax.tree.map(cut, observation), actions[:micro], jax.tree.map(cut, draws))
        temp, out = int(getattr(stats, "temp_size_in_bytes", 0) or 0), int(getattr(stats, "output_size_in_bytes", 0) or 0)
        need = resident + temp + 2 * out
        rows[micro] = {"temp_gib": round(temp / 2 ** 30, 2), "output_gib": round(out / 2 ** 30, 2), "need_gib": round(need / 2 ** 30, 2), "fits": bool(limit and need <= limit * (1 - margin))}
        if chosen is None and rows[micro]["fits"]:
            chosen = micro
    return {"resident_gib": round(resident / 2 ** 30, 2), "limit_gib": round(limit / 2 ** 30, 2), "micro": rows, "chosen": chosen}


def measure_parl(policy, cfg, model_cfg: Optional[dict], *, states: int, refresh_states: int, state_dim: int = 32, plans: int = SYNTHETIC_PLANS, seed: int = 0,
                 save_dir: Optional[Path] = None, micros: Sequence[int] = (), clock: Callable[[], float] = time.monotonic, memory: Callable[[], dict] = device_memory, say: Callable[[str], None] = print) -> Dict[str, object]:
    """Time the stages of PA-RL's Stage II on `policy` with synthetic states. `states`: about how many the annotation and the training work on (the buffer holds the episodes that make them);
    `refresh_states`: how many the timed refresh redraws."""
    import jax
    from src.algos import rl_critic
    from src.algos.rl_candidates import refresh
    from src.algos.rl_critic_update import CriticTrainer
    from src.algos.rl_distill import Distiller
    from src.algos.rl_rollout import RLActor

    horizon = policy.info.action_horizon                                              # `state_dim`: the width of the state row the sampler returns with the candidates (pi0.5: 32, the padded state; the run reads it off the first collected batch)
    buf, cache = synthetic_data(cfg, episodes=max(1, math.ceil(states / (plans + 1))), plans=plans, seed=seed, horizon=horizon, state_dim=state_dim)
    ids = buf.candidate_ids()
    if len(ids) < cfg.distill_batch_size:
        raise SystemExit(f"{len(ids)} states: fewer than a distillation batch ({cfg.distill_batch_size}); raise --states")
    trainer = CriticTrainer(rl_critic.CriticNet.from_config(cfg), cfg, horizon)
    actor = RLActor(policy, cfg, trainer.optimiser)
    distiller = Distiller(policy, trainer.optimiser, cfg, model_cfg)
    params, rng, key = trainer.critic.params, np.random.default_rng(seed), jax.random.key(seed)
    out: Dict[str, object] = {"variant": cfg.variant, "states": int(len(ids)), "memory_before": memory()}
    say(f"[{cfg.variant}] {len(ids)} synthetic states, batch {cfg.distill_batch_size} (micro {cfg.distill_micro_batch_size}), k={cfg.num_candidates} keep_top={cfg.keep_top} local_steps={cfg.local_steps}")

    out["memory_resident"] = memory()
    if micros:
        gm = out["grad_memory"] = grad_memory(distiller, buf, ids, micros, memory)
        say(f"  gradient pass, from the compiler (resident {gm['resident_gib']} GiB of {gm['limit_gib']}; need = resident + temporaries + 2 gradient trees):")
        for m_, r in gm["micro"].items():
            say(f"    micro-batch {m_:>2}: temporaries {r['temp_gib']:>6} GiB, gradient tree {r['output_gib']} GiB -> need {r['need_gib']:>6} GiB: " + ("fits" if r["fits"] else "DOES NOT FIT"))
        if gm["chosen"] is None:
            raise SystemExit("no micro-batch fits in the memory JAX may use (see above): raise XLA_PYTHON_CLIENT_MEM_FRACTION (the env server needs 9-11 GB of the card) or recompute activations")
        distiller.trainer.set_micro(gm["chosen"])
        out["micro_used"] = gm["chosen"]
        say(f"  -> micro-batch {gm['chosen']} for the timed passes (set `distill_micro_batch_size: {gm['chosen']}` for the run)")
    first = distiller.distill(params, key, rng, buf, cache)                         # compiles the annotation and the training step
    out["distill_first"] = {k: first[k] for k in ("states", "steps", "annotate_s", "train_s", "step_s")}
    out["memory_after_first_distill"] = memory()
    say(f"  distillation, first pass (compilation included, paid once): annotate {first['annotate_s']} s, {first['steps']} steps in {first['train_s']} s | memory {out['memory_after_first_distill']}")
    second = distiller.distill(params, key, rng, buf, cache)
    out["distill"] = {k: second[k] for k in ("states", "steps", "annotate_s", "train_s", "step_s")}
    out["annotate_s_per_state"] = second["annotate_s"] / second["states"]
    out["step_s"] = second["step_s"]
    out["memory_after_distill"] = memory()
    say(f"  distillation, timed: annotate {second['annotate_s']} s for {second['states']} states ({out['annotate_s_per_state'] * 1e3:.1f} ms per state), "
        f"{second['steps']} steps in {second['train_s']} s ({second['step_s']} s per step) | memory {out['memory_after_distill']}")

    chosen = ids[:min(refresh_states, len(ids))]
    t0 = clock()
    refresh(cache, actor.sampler, buf.observations, chosen[:cfg.rows_per_sample_call], version=1, seed=seed, clock=clock)           # one padded call: compiles the sampling of the new weights
    out["refresh_first_s"] = clock() - t0
    info = refresh(cache, actor.sampler, buf.observations, chosen, version=2, seed=seed, clock=clock)
    out["refresh"] = info
    out["refresh_s_per_state"] = info["seconds"] / info["states"]
    out["memory_after_refresh"] = memory()
    say(f"  refresh: {info['states']} states in {info['seconds']:.1f} s ({out['refresh_s_per_state'] * 1e3:.0f} ms per state; model alone {info['model_ms'] / info['states']:.0f} ms) | memory {out['memory_after_refresh']}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        t0 = clock()
        distiller.save_state(save_dir / "state")
        t1 = clock()
        distiller.save_checkpoint(save_dir / "vla", {"step": 0, "probe": True})
        t2 = clock()
        out["save_state_s"], out["save_checkpoint_s"] = t1 - t0, t2 - t1
        out["save_size_gib"] = round(sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file()) / 2 ** 30, 2)
        say(f"  saves ({out['save_size_gib']} GiB): resume state {t1 - t0:.1f} s, VLA checkpoint {t2 - t1:.1f} s (local disk: a floor)")
    return out


def estimate_run(m: Dict[str, object], cfg, plans: int, *, collect_round_s: float = COLLECT_ROUND_S, eval_s: float = EVAL_S, critic_h: float = CRITIC_H) -> Dict[str, float]:
    """The hours of a whole run from the measured unit costs `m` (`measure_parl`): per round, annotation + training steps + refresh [+ saves] on every stored state, plus the constants of the other stages."""
    n = stage_states(cfg, plans)
    steps = [(s // cfg.distill_batch_size) * cfg.distill_epochs for s in n]
    annotate_h = sum(n) * m["annotate_s_per_state"] / 3600
    train_h = sum(steps) * m["step_s"] / 3600
    refresh_h = sum(n[:-1]) * m["refresh_s_per_state"] / 3600                                   # no refresh after the last round (src/algos/rl_run.py `_distill`)
    saves_h = cfg.rounds * (m.get("save_state_s", 0.0) + m.get("save_checkpoint_s", 0.0)) / 3600
    collect_h = (cfg.warmup_episodes // cfg.envs_per_batch + cfg.rounds) * collect_round_s / 3600
    eval_h = len(cfg.eval_rounds) * eval_s / 3600
    parts = {"annotate_h": annotate_h, "train_h": train_h, "refresh_h": refresh_h, "saves_h": saves_h, "collect_h": collect_h, "eval_h": eval_h, "critic_h": critic_h}
    parts = {k: round(v, 2) for k, v in parts.items()}
    return {"plans_per_episode": plans, "state_recomputations": int(sum(n[:-1])), "vla_steps": int(sum(steps)), **parts, "total_h": round(sum(parts.values()), 1)}


def memory_verdict(memory: Dict[str, int], server_gib: float = SERVER_GIB, card_gib: float = CARD_GIB) -> str:
    peak = memory.get("peak_bytes_in_use")
    if peak is None:
        return "no memory statistics on this backend"
    gib = peak / 2 ** 30
    return f"peak of the VLA + Adam + critic {gib:.1f} GiB + env server {server_gib:.0f} GiB = {gib + server_gib:.1f} of {card_gib:.0f} GiB: " + ("fits" if gib + server_gib < card_gib else "DOES NOT FIT")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, metavar="FILE", help=f"the PA-RL experiment to measure (default {DEFAULT_EXPERIMENT}, variant B: the dearer)")
    ap.add_argument("--start", nargs=2, metavar=("RUN", "STEP"), help="the SFT run (folder name or path) and the checkpoint step the VLA starts from (default: the experiment's `start`)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a key of configs/algo/rl.yaml")
    ap.add_argument("--micro-sweep", type=int, nargs="*", default=[16, 8, 4, 2], metavar="M", help="micro-batch sizes whose memory the compiler is asked for before anything runs; the largest that fits is used "
                    "(default 16 8 4 2; no value: skip and use the experiment's `distill_micro_batch_size`)")
    ap.add_argument("--states", type=int, default=512, help="synthetic states annotated and trained on (default 512: 8 steps of 64)")
    ap.add_argument("--refresh-states", type=int, default=64, help="states whose candidates the timed refresh redraws (default 64)")
    ap.add_argument("--plans", type=int, nargs="+", default=[19, 10], help="plans per episode for the estimate (default 19 = 150 steps / 8, the upper bound, and 10)")
    ap.add_argument("--save-dir", metavar="DIR", help="also time the writing of the VLA checkpoint and of the resume state in a temporary folder under DIR (deleted afterwards)")
    ap.add_argument("--collect-round-s", type=float, default=COLLECT_ROUND_S, help=f"seconds of a collection round of 50 episodes (default {COLLECT_ROUND_S}, docs/algos/rl.md section 5)")
    ap.add_argument("--eval-s", type=float, default=EVAL_S, help=f"seconds of an evaluation of 100 episodes (default {EVAL_S})")
    ap.add_argument("--critic-h", type=float, default=CRITIC_H, help=f"hours of critic updates of the run (default {CRITIC_H}: probe_critic_cost, variant B)")
    args = ap.parse_args(argv)
    if min(args.states, args.refresh_states) < 1 or not args.plans or min(args.plans) < 1:
        raise SystemExit("--states, --refresh-states and --plans must be 1 or more")
    from scripts.eval_sft import run_seed
    from scripts.rl import check_start, resolve_start
    from src.algos import sft as sft_mod
    from src.algos.rl_config import load_rl_experiment
    from src.models import load_model

    enable_compilation_cache()
    overrides = parse_assignments(args.set)
    exp, cfg = load_rl_experiment(args.experiment, overrides)
    if not cfg.distills:
        raise SystemExit(f"{args.experiment} is {cfg.variant}: it has no distillation or refresh (nothing to measure here; the critic probe covers Best-of-N)")
    start = resolve_start(exp, tuple(args.start) if args.start else None)
    model_cfg = check_start(exp, start)
    print(f"PA-RL stage costs on {gpu_name()}; starting from {start['run']} step {start['step']}; memory before: {device_memory()}", flush=True)
    policy = load_model(model_cfg, seed=run_seed(start["run_config"]))
    sft_mod.load_checkpoint(policy, Path(start["checkpoint"]))
    print(f"model loaded; memory: {device_memory()}", flush=True)
    scratch = tempfile.mkdtemp(prefix="probe_parl_", dir=args.save_dir) if args.save_dir else None
    try:
        measured = measure_parl(policy, cfg, model_cfg, states=args.states, refresh_states=args.refresh_states, save_dir=Path(scratch) if scratch else None, micros=args.micro_sweep)
    except (ValueError, RuntimeError) as err:
        if "RESOURCE_EXHAUSTED" not in str(err):
            raise
        print(f"\n=== OUT OF MEMORY with distill_micro_batch_size={cfg.distill_micro_batch_size}: {str(err)[:200]}\nmemory: {device_memory()}\n"
              f"JAX is limited to half of the card (the env server has the other half): lower it, e.g. --set distill_micro_batch_size=4 (the optimisation is the same, the memory is not).", flush=True)
        return 2
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    print(f"\n=== MEMORY: {memory_verdict(measured['memory_after_refresh'])} (the nvidia-smi report at the end of the log gives the process's real maximum)")
    estimates: List[dict] = []
    for plans in args.plans:
        est = estimate_run(measured, cfg, plans, collect_round_s=args.collect_round_s, eval_s=args.eval_s, critic_h=args.critic_h)
        estimates.append(est)
        print(f"\n=== ESTIMATE of a whole {cfg.variant} run ({cfg.rounds} rounds, {plans} states per episode): {est['state_recomputations']} state recomputations, {est['vla_steps']} VLA steps")
        for k in ("annotate_h", "train_h", "refresh_h", "saves_h", "collect_h", "eval_h", "critic_h"):
            print(f"  {k[:-2]:<9} {est[k]:>6} h" + ("   (not measured: pass --save-dir)" if k == "saves_h" and "save_state_s" not in measured else ""))
        print(f"  TOTAL     {est['total_h']:>6} h")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = ROOT / "reports" / f"probe_parl_cost_{stamp}_{os.environ.get('SLURM_JOB_ID') or 'pid' + str(os.getpid())}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"gpu": gpu_name(), "experiment": args.experiment, "start": {k: start[k] for k in ("run", "step")}, "overrides": overrides, "measured": measured, "estimates": estimates,
                                "constants": {"collect_round_s": args.collect_round_s, "eval_s": args.eval_s, "critic_h": args.critic_h, "synthetic_data": True}}, indent=1, default=float))
    print(f"report: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
