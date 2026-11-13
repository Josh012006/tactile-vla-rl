"""The TRAINING loss of several checkpoints on the SAME demonstration rows, through this repository's training path (no simulator: minutes on a GPU).

    python -m scripts.probe_sft_loss --checkpoint peg_insert_pi05_lora_teleop:0 peg_insert_pi05_lora_teleop:4000 _adopted_expoft_teleop_4000:0

Why. The success-rate curve of our pi0.5 teleop SFT stays near 45 % from step 2200 to 4000 while expo-ft's SFT, evaluated by the same evaluation, reaches 60-66 % (docs/eval/sft_eval.md
section 6.3). Both models are rebuilt by the same code and evaluated by the same code, so the difference is in the TRAINING (or in its seed). Our training path builds the inputs and the targets
of a batch with `OpenPIPolicy.training_batch` (demonstration chunk -> model space -> openpi's input transforms -> normalisation); expo-ft's goes through LeRobot and openpi's data loader.
The loss of expo-ft's weights on batches built by OUR path says whether the two build the same thing:
  - their loss on our batches is much HIGHER than ours: what their model was trained on is not what our path builds (inputs or targets differ: a pipeline mismatch to find);
  - their loss is LOWER than ours: the pipeline is the same and our optimisation fits the demonstrations less well (same loss, worse fit);
  - about EQUAL: the loss does not tell them apart (look at sampling order, augmentation, the seed).
(Their norm_stats are the ones saved next to the checkpoint; they are identical to ours, checked by sha256. The model rows are TRAINING rows: a lower loss is a better fit, not better generalisation.)

`--checkpoint RUN:STEP` (several): RUN is an SFT run (a path, or its folder name under runs/sft/peg_insert/ or runs/sft/probing/), STEP a checkpoint of it, 0 = the run's initial state.
An adopted expo-ft checkpoint (scripts/adopt_expo_ft_checkpoint.py) has one weights state, called step 0: `_adopted_expoft_teleop_4000:0`. The rows are drawn once (`--rows`, `--seed`) from the demonstrations of
the first run that carries its SFT settings (or `--demos`), the same for every checkpoint; the flow-matching noise and time (pi0.5 / pi0) are drawn from the same keys for every checkpoint
(`--draws` of them, averaged per row), so the checkpoints are compared PAIRED: the difference to the first checkpoint is printed with its standard error over rows. Without augmentation (as the evaluation
inside the SFT: `train=False`; `--augment` switches openpi's on).
Run it through jobs/job_probe_sft_loss.sh. Output: the table, and reports/probe_sft_loss_<tag>.json (per-row losses included).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.eval_sft import discover_checkpoints, find_run_dir, with_deterministic_xla
from scripts.probe_eval_parallel import probe_tag
from scripts.sft import enable_compilation_cache
from src.eval.sweep import gpu_name
from src.models.interface import ROOT

DEFAULT_PROMPT = "Insert the peg into the socket."
# Rows per forward pass: pi0-FAST's logits are (rows, tokens, 257152 words) in float32 (sft.EVAL_MICRO_BATCH).
DEFAULT_MICRO_BATCH = {"pi05": 16, "pi0": 16, "pi0_fast": 4}


# -- pure helpers (tested without a GPU) --------------------------------------------------------------------------------------------------
def parse_checkpoint(text: str) -> Tuple[str, int]:
    """'RUN:STEP' -> (RUN, STEP). The step is after the LAST colon (a path may contain one)."""
    run, sep, step = text.rpartition(":")
    if not sep or not run or not step.isdigit():
        raise ValueError(f"--checkpoint {text!r}: expected RUN:STEP, e.g. peg_insert_pi05_lora_teleop:4000 (STEP 0 = the run's initial state)")
    return run, int(step)


def fixed_rows(total: int, n: int, seed: int) -> np.ndarray:
    """`n` distinct rows of a set of `total`, sorted, the same for the same arguments."""
    if total < 1 or n < 1:
        raise ValueError("fixed_rows: the set and the number of rows must be positive")
    return np.sort(np.random.default_rng(seed).choice(total, min(n, total), replace=False))


def summarise(values: Sequence[float]) -> Dict[str, float]:
    """mean and standard error of the mean over rows (0 for a single row)."""
    v = np.asarray(values, dtype=np.float64)
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    return {"mean": float(v.mean()), "se": se, "rows": int(len(v))}


def paired_difference(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """a - b, row by row: mean and standard error over rows (the same rows and the same noise in both, so the pairing removes most of the variance)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired_difference: {a.shape} against {b.shape}")
    return summarise(a - b)


def sft_settings(configs: Sequence[dict]) -> Optional[dict]:
    """The `sft` block (demos, num_demos, prompt) of the first run config that has one, else None (an adopted run has none)."""
    for config in configs:
        block = config.get("sft")
        if block and block.get("demos"):
            return {"demos": list(block["demos"]), "num_demos": int(block.get("num_demos", 0)), "prompt": block.get("prompt", DEFAULT_PROMPT)}
    return None


def group_by_run(checkpoints: Sequence[Tuple[str, int]]) -> List[Tuple[str, List[int]]]:
    """[(run, [steps in the order given])], the runs in the order of their first appearance (one model load per run)."""
    order: Dict[str, List[int]] = {}
    for run, step in checkpoints:
        order.setdefault(run, [])
        if step not in order[run]:
            order[run].append(step)
    return list(order.items())


# -- the loss (needs jax and openpi) --------------------------------------------------------------------------------------------------
def make_row_loss(policy, augment: bool):
    """jitted (state, rng, observation, actions) -> the loss of every row of the batch (the mean of `OpenPIPolicy.loss` is over rows too)."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    graphdef = policy.graphdef

    @jax.jit
    def fn(state, rng, observation, actions):
        loss = nnx.merge(graphdef, state).compute_loss(rng, observation, actions, train=augment)
        return jnp.mean(jnp.reshape(loss, (loss.shape[0], -1)), axis=1)

    return fn


def losses_of(policy, fn, batches, draws: int, seed: int) -> np.ndarray:
    """(draws, rows): the loss of every row with the weights now in `policy`, for `draws` fixed noise keys (the same for every checkpoint)."""
    import jax
    base = jax.random.key(seed + 2)
    out = np.zeros((draws, sum(len(b[1]) for b in batches)), dtype=np.float64)
    for d in range(draws):
        done = 0
        for m, (observation, actions) in enumerate(batches):
            key = jax.random.fold_in(jax.random.fold_in(base, d), m)
            row_loss = np.asarray(fn(policy.state, key, observation, actions), dtype=np.float64)
            out[d, done:done + len(row_loss)] = row_loss
            done += len(row_loss)
    return out


# -- command line ---------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True, metavar="RUN:STEP", help="checkpoints to compare, the first one is the reference of the paired differences")
    ap.add_argument("--demos", nargs="+", metavar="FOLDER", help="folders of ep_*.npz (default: those of the first run that has its SFT settings)")
    ap.add_argument("--num-demos", type=int, help="the first N episodes of each folder (default: the run's)")
    ap.add_argument("--rows", type=int, default=256, help="rows of the demonstrations to take the loss on (default 256)")
    ap.add_argument("--draws", type=int, default=4, help="noise draws (flow-matching time and noise) per row, averaged (default 4)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the rows and of the noise (default 0)")
    ap.add_argument("--micro-batch", type=int, help=f"rows per forward pass (default by family: {DEFAULT_MICRO_BATCH})")
    ap.add_argument("--augment", action="store_true", help="openpi's image augmentation on (train=True); default off, as the evaluation inside the SFT")
    ap.add_argument("--tag", help="name of the report (default: the time and the job id)")
    ap.add_argument("--dry-run", action="store_true", help="check the runs and their checkpoints, print what would be computed; load nothing")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.rows < 1 or args.draws < 1:
        raise SystemExit("--rows and --draws must be at least 1")
    try:
        wanted = [parse_checkpoint(c) for c in args.checkpoint]
    except ValueError as e:
        raise SystemExit(str(e))
    groups = group_by_run(wanted)
    runs: Dict[str, Tuple[Path, dict]] = {}
    for name, steps in groups:
        run_dir = find_run_dir(name)
        config = json.loads((run_dir / "config.json").read_text())
        found = discover_checkpoints(run_dir)
        missing = [s for s in steps if s != 0 and s not in found]
        if missing and "adopted_from" not in config:
            raise SystemExit(f"{run_dir}: no checkpoint {missing} (found: {sorted(found)})")
        if "adopted_from" in config and any(s != 0 for s in steps):
            raise SystemExit(f"{run_dir} is an adopted checkpoint: its only weights state is step 0")
        runs[name] = (run_dir, config)
    horizons = {int(cfg["model_cfg"]["action_horizon"]) for _, cfg in runs.values()}
    if len(horizons) != 1:
        raise SystemExit(f"the runs do not share one action horizon ({sorted(horizons)}): their batches cannot be the same rows")
    horizon = horizons.pop()
    settings = sft_settings([cfg for _, cfg in runs.values()])
    demos = args.demos or (settings or {}).get("demos")
    if not demos:
        raise SystemExit("no SFT settings in any run (adopted runs have none): give --demos FOLDER ...")
    num_demos = args.num_demos if args.num_demos is not None else (settings or {}).get("num_demos", 0)
    prompt = (settings or {}).get("prompt", DEFAULT_PROMPT)
    print(f"rows: {args.rows} (seed {args.seed}) of the demonstrations {demos} (num_demos {num_demos}), {args.draws} noise draws, augmentation {'on' if args.augment else 'off'}")
    for name, steps in groups:
        print(f"  {runs[name][0]}  steps {steps}   model {runs[name][1]['model_cfg']['name']}")
    if args.dry_run:
        return 0

    os.environ["XLA_FLAGS"] = with_deterministic_xla(os.environ.get("XLA_FLAGS", ""))
    enable_compilation_cache()
    from src.algos import demos as demos_mod
    from src.algos import sft as sft_mod
    from src.models import load_model

    train, _ = demos_mod.load_demos(demos, horizon, num_demos=num_demos, val_episodes=0, default_prompt=prompt, root=ROOT)
    rows = fixed_rows(len(train), args.rows, args.seed)
    print(f"{len(train)} demonstration steps, {len(rows)} rows taken")

    per_row: Dict[str, np.ndarray] = {}          # "RUN:STEP" -> (rows,) mean over the noise draws
    spread: Dict[str, float] = {}                # "RUN:STEP" -> mean over rows of the standard deviation over draws (the noise of the loss itself)
    for name, steps in groups:
        run_dir, config = runs[name]
        model_cfg = config["model_cfg"]
        print(f"\nloading {run_dir.name} ({model_cfg['name']})", flush=True)
        policy = load_model(model_cfg, seed=0)
        initial = policy.state
        micro = args.micro_batch or DEFAULT_MICRO_BATCH.get(model_cfg.get("family", ""), 4)
        batches = sft_mod.micro_batches(policy, train, rows, min(micro, len(rows)))
        fn = make_row_loss(policy, args.augment)
        for step in steps:
            policy.set_state(initial)
            if step != 0:
                meta = sft_mod.load_checkpoint(policy, discover_checkpoints(run_dir)[step])
                if int(meta.get("step", step)) != step:
                    raise SystemExit(f"{run_dir}/checkpoints/{step} says it is step {meta.get('step')}")
            draws = losses_of(policy, fn, batches, args.draws, args.seed)
            label = f"{run_dir.name}:{step}"
            per_row[label] = draws.mean(axis=0)
            spread[label] = float(draws.std(axis=0, ddof=1).mean()) if args.draws > 1 else 0.0
            s = summarise(per_row[label])
            print(f"  {label:<60} loss {s['mean']:.5f} +- {s['se']:.5f} (SE over {s['rows']} rows)", flush=True)
        del fn, batches, policy, initial
        gc.collect()

    labels = list(per_row)
    reference = labels[0]
    print(f"\n{'checkpoint':<62} {'loss':>10} {'SE':>9}   {'difference to the first':>26}")
    table = []
    for label in labels:
        s = summarise(per_row[label])
        diff = None if label == reference else paired_difference(per_row[label], per_row[reference])
        table.append({"checkpoint": label, **s, "difference_to_first": diff, "noise_sd_over_draws": spread[label]})
        d = "-" if diff is None else f"{diff['mean']:+.5f} +- {diff['se']:.5f} ({diff['mean'] / diff['se']:+.1f} SE)" if diff["se"] > 0 else f"{diff['mean']:+.5f}"
        print(f"{label:<62} {s['mean']:>10.5f} {s['se']:>9.5f}   {d:>26}")
    print(f"\nfirst = {reference}. A paired difference of a few SE is a real difference of fit on these rows; read it with the three cases at the top of this file.")

    tag = args.tag or probe_tag("sft", time.strftime("%Y%m%d_%H%M%S"), os.environ.get("SLURM_JOB_ID"))
    path = ROOT / "reports" / f"probe_sft_loss_{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"gpu": gpu_name(), "xla_flags": os.environ.get("XLA_FLAGS", ""), "demos": demos, "num_demos": num_demos, "rows": rows.tolist(), "draws": args.draws,
                                "seed": args.seed, "augment": args.augment, "table": table, "per_row": {k: v.tolist() for k, v in per_row.items()}}, indent=2))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
