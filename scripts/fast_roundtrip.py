"""What does the FAST round trip do to OUR demonstrations' motion, with given normalisation statistics?

    source .venv/bin/activate && export HF_HUB_OFFLINE=1
    python -m scripts.fast_roundtrip --stats checkpoint --stats assets/forge_demo_norm:forge_demos_robotiq_noisy_h10

pi0-FAST writes a chunk as FAST tokens: the (delta, normalised) actions go through a DCT, the coefficients are rounded (scale 10) and compressed. When the normalised actions are tiny (the DROID statistics
are much wider than our teleoperated motion: scripts.fast_stats_fit shows the demonstrations use 1 to 7 % of their range), most coefficients can round to 0 and the decoded chunk is a smoothed, shrunk
version of the real one (down to no motion at all). This measures it, per chunk of the demonstrations, with EACH statistics, through the real transforms and the real tokenizer (scripts/fast_tokens.py,
the chunk truncated at the YAML's `max_token_len` as at training):

  - the MOTION of a chunk is `actions[:, :7] - current joints` (what the policy has to command: the delta the tokenizer sees, in rad), the decoded motion is `decoded[:, :7] - current joints`;
  - `error / motion`: the norm of (decoded - original) over the norm of the original motion, per chunk (0 = lossless, 1 = as large as the motion itself); mean, median, 90th percentile, and the share above 0.5;
  - `kept`: the norm of the decoded motion over the norm of the original one (1 = same amplitude, 0.3 = the decoded chunk moves 30 % as much); mean and median, and the share of chunks below 0.1
    (`killed`: the decoded chunk hardly moves);
  - `gain`: the projection of the decoded motion on the original one (what the decoded chunk keeps in the SAME direction, 1 = all of it);
  - per joint, the RMS of the motion and of the error, in rad;
  - the tokens per chunk (prefix + postfix), the chunks over `max_token_len`, the chunks the decoder could not decode, those it replaced by a zero chunk.
Chunks whose original motion is nil (the arm stays still) have no ratio: they are counted apart (`still`).

CPU only, about a minute per statistics. Needs the `vla` dependency group and the files cached by scripts.fetch_models. The default demonstrations are the teleoperated ones, the horizon the YAML's.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts import fast_tokens as ft

DEFAULT_DEMOS = "demos/isaaclab/ForgePegInsert_Robotiq_teleop"
JOINTS = [f"joint{i + 1}" for i in range(7)]
EPS = 1e-6                    # rad: below this a chunk does not move
KILLED_BELOW = 0.1            # a decoded chunk that keeps less than this share of the motion's amplitude is "killed"
BAD_ERROR = 0.5               # an error larger than this share of the motion is "bad"


# -- one chunk --------------------------------------------------------------------------------------------------------------------
def round_trip(pipe: ft.Pipeline, item: ft.Item, prompt: str, max_token_len: int) -> dict:
    """{"tokens": prefix + postfix, "failed": decoding raised, "zeros": decoded to a zero chunk although the chunk is not one, "motion": (H, 7) original, "decoded": (H, 7) decoded motion
    (None when failed), "gripper_err": max |error| of the gripper command (None when failed)}; the motion is relative to the current joints."""
    horizon, dim = item.actions.shape
    state, actions = pipe.to_model(item, prompt)
    tokens, mask, ar, _ = pipe.tokenizer.tokenize(prompt, state, actions)
    tokens, mask, ar = np.asarray(tokens), np.asarray(mask, bool), np.asarray(ar)
    total = int(mask.sum())
    if total >= len(mask):
        raise ValueError(f"the measuring tokenizer is too short ({len(mask)} tokens): raise --measure-len")
    keep = min(total, max_token_len)                                   # the real pipeline truncates at max_token_len
    generated = tokens[:keep][(ar[:keep] == 1) & mask[:keep]].astype(np.int32)
    motion = np.asarray(item.actions[:, :7], np.float64) - np.asarray(item.joints, np.float64)[None]
    try:
        chunk = np.asarray(pipe.tokenizer.extract_actions(generated, horizon, dim))
        absolute = np.asarray(pipe.from_model(state, chunk), np.float64)
    except Exception:                                                  # a chunk that cannot be decoded is a result, not a crash
        return {"tokens": total, "failed": True, "zeros": False, "motion": motion, "decoded": None, "gripper_err": None}
    return {"tokens": total, "failed": False, "zeros": bool(not np.any(chunk) and np.any(actions)), "motion": motion,
            "decoded": absolute[:, :7] - np.asarray(item.joints, np.float64)[None], "gripper_err": float(np.max(np.abs(absolute[:, 7] - item.actions[:, 7])))}


# -- summary ----------------------------------------------------------------------------------------------------------------------
def _spread(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, np.float64)
    return {"mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90))} if len(v) else {"mean": math.nan, "p50": math.nan, "p90": math.nan}


def summarize(records: Sequence[dict], max_token_len: int) -> dict:
    """The statistics of the module docstring over the records of `round_trip`."""
    n = len(records)
    decoded = [r for r in records if not r["failed"]]
    moving = [r for r in decoded if np.linalg.norm(r["motion"]) > EPS]
    rel_err, kept, gain = [], [], []
    for r in moving:
        m, d = r["motion"], r["decoded"]
        nm = float(np.linalg.norm(m))
        rel_err.append(float(np.linalg.norm(d - m)) / nm)
        kept.append(float(np.linalg.norm(d)) / nm)
        gain.append(float(np.sum(d * m)) / nm ** 2)
    joints = {}
    if decoded:
        m_all = np.concatenate([r["motion"] for r in decoded])
        e_all = np.concatenate([r["decoded"] - r["motion"] for r in decoded])
        joints = {name: {"rms_motion_rad": float(np.sqrt(np.mean(m_all[:, j] ** 2))), "rms_error_rad": float(np.sqrt(np.mean(e_all[:, j] ** 2)))} for j, name in enumerate(JOINTS)}
    tokens = [r["tokens"] for r in records]
    return {"chunks": n, "still": len(decoded) - len(moving), "decode_failed": float(np.mean([r["failed"] for r in records])), "decoded_all_zeros": float(np.mean([r["zeros"] for r in records])),
            "tokens": {"mean": float(np.mean(tokens)), "p99": float(np.percentile(tokens, 99)), "max": int(max(tokens))},
            "over_max_token_len": float(np.mean([t > max_token_len for t in tokens])),
            "error_over_motion": {**_spread(rel_err), "share_above": float(np.mean(np.asarray(rel_err) > BAD_ERROR)) if rel_err else math.nan},
            "kept": {**_spread(kept), "share_killed": float(np.mean(np.asarray(kept) < KILLED_BELOW)) if kept else math.nan},
            "gain": _spread(gain), "joints": joints,
            "gripper_err_max": float(max(r["gripper_err"] for r in decoded)) if decoded else math.nan}


def measure(pipe: ft.Pipeline, folders: Sequence[str], horizon: int, max_token_len: int, limit: int = 0, tail: str = "pad") -> dict:
    records = []
    for folder in folders:
        for _, demo in ft.load_demos(folder, limit):
            for item in ft.chunk_items(demo["joint_position"], demo["gripper_position"], demo["actions"], horizon, tail):
                records.append(round_trip(pipe, item, demo["prompt"], max_token_len))
    return summarize(records, max_token_len)


def format_summary(name: str, s: dict) -> List[str]:
    e, k, g, t = s["error_over_motion"], s["kept"], s["gain"], s["tokens"]
    lines = [f"statistics: {name}",
             f"  {s['chunks']} chunks ({s['still']} still); tokens/chunk mean {t['mean']:.0f}, p99 {t['p99']:.0f}, max {t['max']}; over max_token_len {100 * s['over_max_token_len']:.2f} %, "
             f"not decoded {100 * s['decode_failed']:.2f} %, decoded to zeros {100 * s['decoded_all_zeros']:.2f} %",
             f"  error / motion : mean {e['mean']:.2f}, median {e['p50']:.2f}, p90 {e['p90']:.2f}; {100 * e['share_above']:.1f} % of the chunks above {BAD_ERROR}",
             f"  kept (amplitude): mean {k['mean']:.2f}, median {k['p50']:.2f}, p90 {k['p90']:.2f}; {100 * k['share_killed']:.1f} % of the chunks below {KILLED_BELOW} (killed)",
             f"  gain (same direction): mean {g['mean']:.2f}, median {g['p50']:.2f}",
             f"  gripper command: largest error {s['gripper_err_max']:.4f}",
             f"  {'':<8} {'motion rms rad':>15} {'error rms rad':>14} {'error/motion':>13}"]
    for j, v in s["joints"].items():
        ratio = v["rms_error_rad"] / v["rms_motion_rad"] if v["rms_motion_rad"] > 0 else math.nan
        lines.append(f"  {j:<8} {v['rms_motion_rad']:>15.4f} {v['rms_error_rad']:>14.4f} {ratio:>13.2f}")
    return lines


# -- the command ------------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="append", required=True, help="statistics to test, repeatable: 'checkpoint' or '<dir>:<asset id>'")
    ap.add_argument("--model", default="pi0_fast", help="configs/model/<model>.yaml (a pi0-FAST one)")
    ap.add_argument("--demos-dir", action="append", help=f"folder with ep_*.npz, repeatable (default {DEFAULT_DEMOS})")
    ap.add_argument("--horizon", type=int, default=None, help="chunk length (default: the YAML's action_horizon)")
    ap.add_argument("--max-token-len", type=int, default=None, help="truncation limit (default: the YAML's max_token_len)")
    ap.add_argument("--tail", choices=("pad", "drop"), default="pad", help="the chunks of the last steps: repeat the last action (pad) or skip them (drop)")
    ap.add_argument("--limit-demos", type=int, default=0, help="only the first N demos of each folder (0 = all)")
    ap.add_argument("--out", default="reports/fast_roundtrip.json", help="JSON report")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from scripts.fast_stats_fit import pipeline_for
    folders = args.demos_dir or [DEFAULT_DEMOS]
    report: Dict[str, object] = {"demos": folders, "results": {}}
    for stats in args.stats:
        cfg, pipe = pipeline_for(args.model, stats)
        horizon = args.horizon or int(cfg["action_horizon"])
        max_token_len = args.max_token_len or int(cfg["max_token_len"])
        print(f"\n(demonstrations {', '.join(folders)}; chunks of {horizon}; truncation at {max_token_len} tokens)")
        summary = measure(pipe, folders, horizon, max_token_len, args.limit_demos, args.tail)
        print("\n".join(format_summary(stats, summary)))
        report["results"][stats] = {"horizon": horizon, "max_token_len": max_token_len, **summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
