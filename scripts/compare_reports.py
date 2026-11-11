"""Compare the first action chunk of two probe reports (scripts/probe_vla.py) made on the SAME observation (`--obs-file`).

    python -m scripts.compare_reports reports/probe_pi05_<time>.json reports/probe_pi05_nolora_<time>.json

The first inference of a fresh process always draws the same flow-matching / sampling noise (openpi's Policy starts from key(0)), so two
reports on the same observation differ only through the model (LoRA or not, another loader, another checkpoint) and the numerics of the
GPU. Two reports of the SAME model on the SAME observation give the numerical floor. Reports on different observations are refused:
the observation of a FORGE rollout changes from one run to the next (the Robotiq start state is not reproducible) and moves the
action by far more than a LoRA does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def compare_chunks(a_rows, b_rows) -> dict:
    a, b = np.asarray(a_rows, dtype=np.float64), np.asarray(b_rows, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = np.abs(a - b)
    scale = 0.5 * (np.abs(a).mean() + np.abs(b).mean())
    return {"rows": n, "max_abs": float(d.max()), "mean_abs": float(d.mean()), "mean_abs_of_values": float(scale),
            "relative_mean": float(d.mean() / scale) if scale > 0 else float("nan"),
            "max_abs_per_column": [round(float(x), 5) for x in d.max(axis=0)]}


def label(r: dict) -> str:
    """`settings_text` of the report (what was loaded: LoRA per component, touch); reports from before the per-component settings carry `lora`."""
    if "settings_text" in r:
        what = r["settings_text"]
    else:
        lora = r.get("lora") or {}
        what = "no LoRA" if not lora.get("use_lora", True) else f"LoRA r{lora.get('vlm_rank')}/{lora.get('action_head_rank')}"
    return f"{r['model']} ({what}, load: {r.get('load_mode')})"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report_a")
    ap.add_argument("report_b")
    ap.add_argument("--allow-different-observations", action="store_true")
    args = ap.parse_args(argv)
    ra, rb = (json.loads(Path(p).read_text()) for p in (args.report_a, args.report_b))
    sa, sb = ra.get("observation_sha256"), rb.get("observation_sha256")
    print(f"A: {label(ra)}   {args.report_a}\nB: {label(rb)}   {args.report_b}")
    if not sa or sa != sb:
        print(f"observation A {sa}\nobservation B {sb}")
        if not args.allow_different_observations:
            print("REFUSED: the two reports were not made on the same observation (use --obs-file); the difference would mostly be the observation.")
            return 2
    res = compare_chunks(ra["first_chunk"]["rows"], rb["first_chunk"]["rows"])
    print(f"first chunk, {res['rows']} rows x {len(res['max_abs_per_column'])} columns: max |A-B| {res['max_abs']:.4f}   mean |A-B| {res['mean_abs']:.4f}   "
          f"mean |value| {res['mean_abs_of_values']:.4f}   relative {100 * res['relative_mean']:.1f} %")
    print(f"max |A-B| per column (7 joints, gripper): {res['max_abs_per_column']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

