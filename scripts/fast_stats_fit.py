"""How well do given normalisation statistics fit OUR demonstrations, as pi0-FAST sees them? And what did the evaluations of the runs trained with them show?

    source .venv/bin/activate && export HF_HUB_OFFLINE=1
    python -m scripts.fast_stats_fit \
        --stats checkpoint --stats assets/forge_demo_norm:forge_demos_robotiq_noisy_h10 \
        --runs runs/sft/peg_insert/peg_insert_pi0_fast_lora_teleop_droid_stats runs/sft/peg_insert/peg_insert_pi0_fast_lora_teleop_demo_stats

Part 1 (--stats, repeatable; CPU, about a minute; the default demonstrations are the teleoperated ones, the horizon the YAML's 10): every chunk of the demonstrations goes through EXACTLY the transforms
of the pi0-FAST pipeline (scripts/fast_tokens.py: `DroidInputs`, `DeltaActions`, quantile normalisation) with each statistics; what comes out is what the tokenizer is given. `checkpoint` is the
checkpoint's own (DROID / PolaRiS) statistics, `DIR:ASSET_ID` is `<DIR>/<ASSET_ID>/norm_stats.json`. Per dimension of the state and of the actions it prints
  - `out %`: the share of the values outside [-1, 1], i.e. beyond the statistics' 1st / 99th percentile (what the quantile normalisation maps onto -1 / +1);
  - `p1`, `p99`: the 1st and 99th percentile of the normalised values, `use`: (p99 - p1) / 2, the share of the statistics' range that the demonstrations occupy (1 = all of it, 0.1 = a tenth: the values
    then sit in a narrow band around `mid` = (p99 + p1) / 2);
  - for the state, `bins`: how many of the 256 bins in which pi0-FAST writes the state as text (bins of [-1, 1], values outside fall in the end bins) the demonstrations occupy.
Part 2 (--runs, optional): for each SFT run, per checkpoint, the success rate and the share of chunks the FAST decoder could not decode (read from `<run>/eval/results/<step>.json`: nothing is simulated).
A chunk that cannot be decoded is replaced by openpi with a zero chunk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DEMOS = "demos/isaaclab/ForgePegInsert_Robotiq_teleop"
STATE_BINS = 256                  # openpi's FASTTokenizer: the state is discretised into 256 bins of [-1, 1]
JOINT_NAMES = [f"joint{i + 1}" for i in range(7)] + ["gripper"]


# -- pure helpers (tested without openpi) ---------------------------------------------------------------------------------------------
def normalised_inputs(pipe, folders: Sequence[str], horizon: int, limit: int = 0, tail: str = "pad") -> Tuple[np.ndarray, np.ndarray]:
    """(state (N, D), actions (N, H, D)) as the pipeline normalises them for the tokenizer, for every chunk of the demonstrations."""
    from scripts import fast_tokens as ft
    states, chunks = [], []
    for folder in folders:
        for _, demo in ft.load_demos(folder, limit):
            for item in ft.chunk_items(demo["joint_position"], demo["gripper_position"], demo["actions"], horizon, tail):
                state, actions = pipe.to_model(item, demo["prompt"])
                states.append(np.asarray(state, np.float64))
                chunks.append(np.asarray(actions, np.float64))
    return np.stack(states), np.stack(chunks)


def fit_of(values: np.ndarray) -> Dict[str, float]:
    """Statistics of one dimension's normalised values (any shape): the share outside [-1, 1], p1, p99, the range used, its middle, the std."""
    v = np.asarray(values, np.float64).reshape(-1)
    p1, p99 = np.percentile(v, [1, 99])
    return {"out_pct": float(100 * np.mean(np.abs(v) > 1.0)), "p1": float(p1), "p99": float(p99), "use": float((p99 - p1) / 2), "mid": float((p99 + p1) / 2), "std": float(v.std())}


def state_bins_used(values: np.ndarray) -> int:
    """How many of the 256 bins of openpi's state discretisation (`np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1`) the values fall in."""
    return int(len(np.unique(np.digitize(np.asarray(values).reshape(-1), bins=np.linspace(-1, 1, STATE_BINS + 1)[:-1]) - 1)))


def fit_report(states: np.ndarray, actions: np.ndarray) -> dict:
    """{"state": {dim: fit + bins}, "actions": {dim: fit}, "chunks": N, "chunks_out_pct": share of chunks with a value outside [-1, 1] in some action dimension}."""
    dims = states.shape[1]
    names = JOINT_NAMES[:dims] if dims <= len(JOINT_NAMES) else [f"d{i}" for i in range(dims)]
    return {"chunks": int(len(states)),
            "state": {n: {**fit_of(states[:, i]), "bins": state_bins_used(states[:, i])} for i, n in enumerate(names)},
            "actions": {n: fit_of(actions[:, :, i]) for i, n in enumerate(names)},
            "chunks_out_pct": float(100 * np.mean(np.any(np.abs(actions) > 1.0, axis=(1, 2))))}


def format_fit(report: dict) -> List[str]:
    lines = [f"  {report['chunks']} chunks; {report['chunks_out_pct']:.1f} % of them have an action value outside [-1, 1]",
             f"  {'':<9} {'out %':>6} {'p1':>7} {'p99':>7} {'use':>6} {'mid':>7} {'std':>6} {'bins':>5}"]
    for part in ("state", "actions"):
        for name, f in report[part].items():
            bins = f"{f['bins']:>5}" if "bins" in f else f"{'':>5}"
            lines.append(f"  {part[:5]:<5}{name:<5}{f['out_pct']:>6.1f} {f['p1']:>7.3f} {f['p99']:>7.3f} {f['use']:>6.3f} {f['mid']:>7.3f} {f['std']:>6.3f} {bins}")
    return lines


def eval_rows(results_dir: Path) -> List[dict]:
    """[{step, success_pct, se_pct, decode_failed_pct (None for a model without token strings)}] of `<results_dir>/<step>.json`, by step."""
    rows = []
    for path in Path(results_dir).glob("*.json"):
        if not path.stem.isdigit():
            continue
        data = json.loads(path.read_text())
        chunks, failed = data.get("decode_chunks"), data.get("decode_failed")
        rows.append({"step": int(data.get("step", path.stem)), "success_pct": 100 * float(data["success_rate"]), "se_pct": 100 * float(data["success_se"]),
                     "decode_failed_pct": (100 * failed / chunks) if chunks else None, "decode_chunks": chunks})
    return sorted(rows, key=lambda r: r["step"])


def run_summary(rows: Sequence[dict], from_step: int = 1000) -> dict:
    """The mean success and the mean share of undecodable chunks over the checkpoints from `from_step` on (the plateau)."""
    sel = [r for r in rows if r["step"] >= from_step]
    if not sel:
        return {}
    failed = [r["decode_failed_pct"] for r in sel if r["decode_failed_pct"] is not None]
    return {"from_step": from_step, "checkpoints": len(sel), "success_pct": float(np.mean([r["success_pct"] for r in sel])),
            "decode_failed_pct": float(np.mean(failed)) if failed else None}


def format_runs(names: Sequence[str], tables: Sequence[Sequence[dict]]) -> List[str]:
    steps = sorted({r["step"] for t in tables for r in t})
    by = [{r["step"]: r for r in t} for t in tables]
    short = [n[-28:] for n in names]
    head = f"  {'step':>5} " + " ".join(f"{s:>28}" for s in short)
    lines = [head, f"  {'':>5} " + " ".join(f"{'success %  / not decoded %':>28}" for _ in names)]
    for step in steps:
        cells = []
        for b in by:
            r = b.get(step)
            cells.append(f"{'':>28}" if r is None else f"{r['success_pct']:>8.0f} +- {r['se_pct']:<3.0f} / " + (f"{r['decode_failed_pct']:>5.1f}" if r["decode_failed_pct"] is not None else f"{'-':>5}") + f"{'':>4}")
        lines.append(f"  {step:>5} " + " ".join(cells))
    for name, t in zip(names, tables):
        s = run_summary(t)
        if s:
            failed = f", {s['decode_failed_pct']:.1f} % of the chunks not decoded" if s["decode_failed_pct"] is not None else ""
            lines.append(f"  {name}: from step {s['from_step']} ({s['checkpoints']} checkpoints) mean success {s['success_pct']:.1f} %{failed}")
    return lines


# -- the command ---------------------------------------------------------------------------------------------------------------------
def pipeline_for(model: str, stats: str):
    """(model YAML, pipeline) for `stats`: 'checkpoint' (the checkpoint's own) or '<dir>:<asset id>'."""
    from scripts import fast_tokens as ft
    from src.models.interface import apply_overrides, load_model_cfg
    cfg = load_model_cfg(model)
    if stats != "checkpoint":
        directory, sep, asset = stats.rpartition(":")
        if not sep or not directory or not asset:
            raise SystemExit(f"--stats {stats!r}: expected 'checkpoint' or '<dir>:<asset id>'")
        cfg = apply_overrides(cfg, {"norm_stats.dir": directory, "norm_stats.asset_id": asset})
    if cfg.get("family") != "pi0_fast":
        raise SystemExit(f"configs/model/{model}.yaml is a '{cfg.get('family')}' model: this is about pi0-FAST")
    return cfg, ft.openpi_pipeline(cfg, 1024)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="append", help="statistics to test, repeatable: 'checkpoint' or '<dir>:<asset id>'")
    ap.add_argument("--model", default="pi0_fast", help="configs/model/<model>.yaml (a pi0-FAST one)")
    ap.add_argument("--demos-dir", action="append", help=f"folder with ep_*.npz, repeatable (default {DEFAULT_DEMOS})")
    ap.add_argument("--horizon", type=int, default=None, help="chunk length (default: the YAML's action_horizon)")
    ap.add_argument("--limit-demos", type=int, default=0, help="only the first N demos of each folder (0 = all)")
    ap.add_argument("--runs", nargs="+", metavar="RUN", help="SFT runs whose evaluations to tabulate (a path, or a folder name under runs/sft/peg_insert or runs/sft/probing)")
    ap.add_argument("--out", default="reports/fast_stats_fit.json", help="JSON report")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.stats and not args.runs:
        raise SystemExit("give --stats (the fit of statistics to the demonstrations) and / or --runs (the evaluations of runs)")
    report: dict = {}
    if args.stats:
        import os
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        folders = args.demos_dir or [DEFAULT_DEMOS]
        report["fit"] = {}
        for stats in args.stats:
            cfg, pipe = pipeline_for(args.model, stats)
            horizon = args.horizon or int(cfg["action_horizon"])
            print(f"\nstatistics: {stats}   (demonstrations {', '.join(folders)}, chunks of {horizon})")
            fit = fit_report(*normalised_inputs(pipe, folders, horizon, args.limit_demos))
            print("\n".join(format_fit(fit)))
            report["fit"][stats] = fit
    if args.runs:
        from scripts.eval_sft import find_run_dir
        tables, names = [], []
        for run in args.runs:
            run_dir = find_run_dir(run)
            tables.append(eval_rows(run_dir / "eval" / "results"))
            names.append(run_dir.name)
        print("\nevaluations (100 episodes per checkpoint, the same starting states in every run):")
        print("\n".join(format_runs(names, tables)))
        report["runs"] = {n: {"rows": t, "plateau": run_summary(t)} for n, t in zip(names, tables)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
