"""How many tokens does pi0-FAST need for the chunks of OUR demonstrations, and does the FAST round trip keep them?

pi0-FAST writes an action chunk as text: `[prefix: task + state][postfix: "Action: " FAST tokens "|" EOS]`. The FAST tokens are a
variable-length compression of the chunk, and three limits of openpi depend on that length (configs/model/pi0_fast.yaml):

  max_token_len      prefix + postfix must fit: a longer sequence is TRUNCATED when it is tokenised, so the chunk cannot be decoded
  max_decoding_steps the model generates the postfix token by token (256 in openpi's `sample_actions`): a postfix longer than that is
                     cut before the end token
  action_horizon     the number of steps of a chunk (the longer the chunk, the more tokens)

This script measures them on the real chunks of the demonstrations: for every step t of every demo, the chunk of the next `horizon` actions
(the last steps repeat the final action, as the LeRobot loader of openpi does) goes through EXACTLY the transforms of the openpi config
(`DroidInputs`, `DeltaActions`, quantile normalisation with the checkpoint's DROID statistics) and the FAST tokenizer. It reports the token
counts, how many chunks exceed the limits, and, by decoding the tokens back (as at inference, after the truncation at `max_token_len`),
how much of the chunk survives (error in rad after `AbsoluteActions`).

    source .venv/bin/activate && export HF_HUB_OFFLINE=1          # the tokenizers were fetched by scripts.fetch_models
    python -m scripts.fast_tokens                                 # the three demo folders, the YAML's horizon and limits
    python -m scripts.fast_tokens --horizon 10 --horizon 16 --max-token-len 250 --demos-dir demos/isaaclab/ForgePegInsert_Robotiq_teleop

    # with other normalisation statistics than the checkpoint's (the `norm_stats` block of the model YAML: <dir>/<asset id>/norm_stats.json)
    python -m scripts.fast_tokens --norm-stats-dir assets/forge_demo_norm --norm-stats-asset-id forge_demos_robotiq_noisy_h10 --out reports/fast_tokens_demo_stats.json

CPU only, about a minute. Needs the `vla` dependency group (openpi, transformers) and the files cached by scripts.fetch_models.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # nothing here needs a GPU (set before jax is imported by openpi)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_DIRS = tuple(f"demos/isaaclab/ForgePegInsert_Robotiq_{kind}" for kind in ("scripted", "noisy", "teleop"))
MAX_DECODING_STEPS = 256            # openpi's default in Pi0FAST.sample_actions
IMAGE = np.zeros((224, 224, 3), np.uint8)      # DroidInputs wants images; only the state and the actions matter here


# -- the demonstrations -> chunks -------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Item:
    """One chunk: the joint state + gripper at step t (as DroidInputs builds the state) and the next `horizon` ABSOLUTE actions."""
    joints: np.ndarray            # (7,)
    gripper: np.ndarray           # (1,)
    actions: np.ndarray           # (horizon, 8)


def chunk_items(joint_position: np.ndarray, gripper_position: np.ndarray, actions: np.ndarray, horizon: int, tail: str = "pad") -> List[Item]:
    """The chunks of one demonstration. `tail="pad"`: one chunk per step, the steps past the end repeat the last action (what the LeRobot
    loader does by clamping the index); `tail="drop"`: only the chunks that fit inside the demonstration."""
    if tail not in ("pad", "drop"):
        raise ValueError(f"tail must be 'pad' or 'drop', got {tail!r}")
    actions = np.asarray(actions, np.float32)
    n = len(actions)
    last = n if tail == "pad" else n - horizon + 1
    items = []
    for t in range(max(last, 0)):
        idx = np.minimum(np.arange(t, t + horizon), n - 1)
        items.append(Item(np.asarray(joint_position[t], np.float32), np.asarray(gripper_position[t], np.float32).reshape(-1), actions[idx]))
    return items


def load_demos(folder: str, limit: int = 0) -> List[Tuple[str, dict]]:
    """[(file name, {joint_position, gripper_position, actions, prompt})] of the ep_*.npz of `folder` (the failed ones are not there)."""
    path = Path(folder)
    files = sorted(path.glob("ep_*.npz"))
    if not files:
        raise FileNotFoundError(f"no ep_*.npz in {path}")
    out = []
    for f in files[: limit or None]:
        with np.load(f, allow_pickle=True) as z:
            out.append((f.name, {"joint_position": z["joint_position"], "gripper_position": z["gripper_position"], "actions": z["actions"],
                                 "prompt": str(z["prompt"])}))
    return out


# -- the model side: transforms + tokenizer ---------------------------------------------------------------------------------------
@dataclasses.dataclass
class Pipeline:
    """What the measurement needs from openpi, as plain callables (so that tests can give fakes).
    to_model(item, prompt) -> (normalised state (D,), normalised actions (H, D)): the inputs of the tokenizer.
    from_model(normalised state, normalised chunk) -> ABSOLUTE chunk (H, 8): the output transforms of the policy.
    tokenizer: openpi's FASTTokenizer (`tokenize(prompt, state, actions)`, `extract_actions(tokens, horizon, dim)`)."""
    to_model: Callable
    from_model: Callable
    tokenizer: object


def make_pipeline(data_transforms, norm_stats, use_quantiles: bool, tokenizer) -> Pipeline:
    """The transforms of an openpi data config (`DataConfig.data_transforms`) around the normalisation: the same order as
    src/models/loading.py `_policy_from` (inputs: data transforms then Normalize; outputs: Unnormalize then data transforms)."""
    import openpi.transforms as transforms

    normalize = transforms.Normalize(norm_stats, use_quantiles=use_quantiles)
    unnormalize = transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)

    def to_model(item: Item, prompt: str):
        # copies: openpi's DeltaActions subtracts the state IN PLACE, which would turn the item's absolute chunk (what the round trip is
        # compared with) into a delta chunk
        data = {"observation/exterior_image_1_left": IMAGE, "observation/wrist_image_left": IMAGE,
                "observation/joint_position": item.joints.copy(), "observation/gripper_position": item.gripper.copy(),
                "actions": item.actions.copy(), "prompt": prompt}
        for t in data_transforms.inputs:
            data = t(data)
        data = normalize(data)
        return np.asarray(data["state"]), np.asarray(data["actions"])

    def from_model(state, chunk):
        data = unnormalize({"state": np.asarray(state), "actions": np.asarray(chunk)})
        for t in data_transforms.outputs:
            data = t(data)
        return np.asarray(data["actions"])

    return Pipeline(to_model, from_model, tokenizer)


def openpi_pipeline(model_cfg: dict, measure_len: int) -> Pipeline:
    """The real pipeline of configs/model/<model>.yaml: its openpi TrainConfig (data transforms), the DROID statistics of the YAML's
    checkpoint, and a FAST tokenizer long enough that nothing is truncated while measuring."""
    from openpi.models import tokenizer as _tokenizer
    from openpi.training import config as _config

    from src.models.loading import _download, load_norm_stats

    train_cfg = _config.get_config(model_cfg["openpi_config"])
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    norm_stats = load_norm_stats(model_cfg, train_cfg, _download(model_cfg["checkpoint"]))       # the YAML's `norm_stats` block, else the checkpoint's
    kwargs = getattr(train_cfg.model, "fast_model_tokenizer_kwargs", None) or {}
    tokenizer = _tokenizer.FASTTokenizer(measure_len, **kwargs)
    return make_pipeline(data_config.data_transforms, norm_stats, data_config.use_quantile_norm, tokenizer)


# -- measurement ----------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Record:
    total: int                # prefix + postfix tokens (what max_token_len limits)
    prefix: int               # task + state tokens
    postfix: int              # "Action: " + FAST tokens + "|" + EOS (what max_decoding_steps limits)
    failed: bool              # the decoding raised
    zeros: bool               # the decoding returned an all-zero chunk although the chunk is not (openpi's fallback when "Action: " is lost)
    joint_err: float          # max over the chunk of |decoded - original| on the 7 joints, rad (NaN if failed)
    gripper_err: float        # same on the gripper command


def measure_item(pipe: Pipeline, item: Item, prompt: str, max_token_len: int) -> Record:
    horizon, dim = item.actions.shape
    state, actions = pipe.to_model(item, prompt)
    tokens, mask, ar, _ = pipe.tokenizer.tokenize(prompt, state, actions)
    tokens, mask, ar = np.asarray(tokens), np.asarray(mask, bool), np.asarray(ar)
    total, prefix, postfix = int(mask.sum()), int(((ar == 0) & mask).sum()), int(((ar == 1) & mask).sum())
    if total >= len(mask):
        raise ValueError(f"the measuring tokenizer is too short ({len(mask)} tokens): raise --measure-len")
    keep = min(total, max_token_len)                       # what the real pipeline keeps: it truncates at max_token_len
    generated = tokens[:keep][(ar[:keep] == 1) & mask[:keep]].astype(np.int32)
    try:
        chunk = np.asarray(pipe.tokenizer.extract_actions(generated, horizon, dim))
        absolute = pipe.from_model(state, chunk)
        err = np.abs(absolute - item.actions)
        zeros = bool(not np.any(chunk) and np.any(actions))
        return Record(total, prefix, postfix, False, zeros, float(err[:, :7].max()), float(err[:, 7].max()))
    except Exception:                                      # a chunk that cannot be decoded is a result, not a crash
        return Record(total, prefix, postfix, True, False, math.nan, math.nan)


def pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, float), q))


def summarize(records: Sequence[Record], max_token_len: int, max_decoding_steps: int) -> dict:
    n = len(records)
    total = [r.total for r in records]
    post = [r.postfix for r in records]
    ok = [r for r in records if not r.failed]
    jerr = [r.joint_err for r in ok]
    return {
        "chunks": n,
        "tokens_total": {"mean": float(np.mean(total)), "p50": pct(total, 50), "p90": pct(total, 90), "p99": pct(total, 99), "max": int(max(total))},
        "tokens_prefix": {"mean": float(np.mean([r.prefix for r in records])), "max": int(max(r.prefix for r in records))},
        "tokens_postfix": {"mean": float(np.mean(post)), "p99": pct(post, 99), "max": int(max(post))},
        "over_max_token_len": float(np.mean([t > max_token_len for t in total])),
        "over_max_decoding_steps": float(np.mean([p > max_decoding_steps for p in post])),
        "decode_failed": float(np.mean([r.failed for r in records])),
        "decoded_all_zeros": float(np.mean([r.zeros for r in records])),
        "joint_err_rad": {"mean": float(np.mean(jerr)), "p99": pct(jerr, 99), "max": float(max(jerr))} if jerr else None,
        "gripper_err": {"mean": float(np.mean([r.gripper_err for r in ok])), "max": float(max(r.gripper_err for r in ok))} if ok else None,
    }


def recommend(all_records: Sequence[Record], margin: float) -> dict:
    """The smallest limits that cover every measured chunk, and with a safety margin (a policy under RL can write longer chunks than the
    demonstrations: it explores)."""
    need_total = max(r.total for r in all_records)
    need_post = max(r.postfix for r in all_records)
    return {"max_token_len_covering_all": need_total, "max_token_len_with_margin": int(math.ceil(need_total * (1 + margin) / 10.0) * 10),
            "max_decoding_steps_covering_all": need_post, "margin": margin}


# -- command line ---------------------------------------------------------------------------------------------------------------
def run(model_cfg: dict, pipe: Pipeline, folders: Sequence[str], horizons: Sequence[int], max_token_len: int, max_decoding_steps: int,
        tail: str, limit: int, margin: float) -> dict:
    """Measure every (folder, horizon). Returns the report (also what main prints)."""
    report: Dict[str, object] = {"model": model_cfg.get("name"), "max_token_len": max_token_len, "max_decoding_steps": max_decoding_steps,
                                 "tail": tail, "results": []}
    everything: List[Record] = []
    for folder in folders:
        demos = load_demos(folder, limit)
        for horizon in horizons:
            records = []
            for _, demo in demos:
                for item in chunk_items(demo["joint_position"], demo["gripper_position"], demo["actions"], horizon, tail):
                    records.append(measure_item(pipe, item, demo["prompt"], max_token_len))
            everything += records
            report["results"].append({"folder": folder, "demos": len(demos), "horizon": horizon, **summarize(records, max_token_len, max_decoding_steps)})
    report["recommend"] = recommend(everything, margin)
    return report


def print_report(report: dict) -> None:
    print(f"\nmax_token_len = {report['max_token_len']}   max_decoding_steps = {report['max_decoding_steps']}   tail = {report['tail']}\n")
    head = (f"{'folder':44s} {'demos':>5s} {'H':>3s} {'chunks':>7s} | {'tokens mean/p99/max':>20s} | {'prefix':>6s} {'postfix max':>11s} | "
            f"{'> max_tok':>9s} {'> dec':>6s} {'fail':>5s} {'zeros':>5s} | {'joint err rad mean/p99/max':>28s} {'grip max':>8s}")
    print(head)
    for r in report["results"]:
        t, e, g = r["tokens_total"], r["joint_err_rad"], r["gripper_err"]
        err = f"{e['mean']:.4f} / {e['p99']:.4f} / {e['max']:.4f}" if e else "-"
        print(f"{Path(r['folder']).name:44s} {r['demos']:5d} {r['horizon']:3d} {r['chunks']:7d} | {t['mean']:6.1f} /{t['p99']:5.0f} /{t['max']:5d} | "
              f"{r['tokens_prefix']['max']:6d} {r['tokens_postfix']['max']:11d} | {100 * r['over_max_token_len']:8.2f}% "
              f"{100 * r['over_max_decoding_steps']:5.1f}% {100 * r['decode_failed']:4.1f}% {100 * r['decoded_all_zeros']:4.1f}% | "
              f"{err:>28s} {(g['max'] if g else float('nan')):8.4f}")
    rec = report["recommend"]
    print(f"\nsmallest max_token_len covering every measured chunk: {rec['max_token_len_covering_all']}   "
          f"with a {100 * rec['margin']:.0f} % margin: {rec['max_token_len_with_margin']}   "
          f"(needed max_decoding_steps: {rec['max_decoding_steps_covering_all']})")


def main(argv: Optional[Sequence[str]] = None, pipeline: Optional[Pipeline] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="pi0_fast", help="configs/model/<model>.yaml (a pi0-FAST one)")
    ap.add_argument("--demos-dir", action="append", help=f"folder with ep_*.npz, repeatable (default: {', '.join(DEFAULT_DEMO_DIRS)})")
    ap.add_argument("--horizon", type=int, action="append", help="chunk length, repeatable (default: the YAML's action_horizon)")
    ap.add_argument("--max-token-len", type=int, default=None, help="limit to test (default: the YAML's max_token_len)")
    ap.add_argument("--max-decoding-steps", type=int, default=MAX_DECODING_STEPS, help="limit to test (openpi's default)")
    ap.add_argument("--tail", choices=("pad", "drop"), default="pad", help="the chunks of the last steps: repeat the last action (pad) or skip them (drop)")
    ap.add_argument("--limit-demos", type=int, default=0, help="only the first N demos of each folder (0 = all)")
    ap.add_argument("--margin", type=float, default=0.2, help="safety margin of the recommended max_token_len")
    ap.add_argument("--measure-len", type=int, default=1024, help="length of the measuring tokenizer: must be above every chunk, nothing is truncated")
    ap.add_argument("--norm-stats-dir", default=None, help="other normalisation statistics than the checkpoint's: <dir>/<asset id>/norm_stats.json (with --norm-stats-asset-id)")
    ap.add_argument("--norm-stats-asset-id", default=None)
    ap.add_argument("--out", default="reports/fast_tokens.json", help="JSON report")
    a = ap.parse_args(argv)

    from src.models.interface import apply_overrides, load_model_cfg, norm_stats_source
    cfg = load_model_cfg(a.model)
    if a.norm_stats_dir or a.norm_stats_asset_id:
        cfg = apply_overrides(cfg, {"norm_stats.dir": a.norm_stats_dir, "norm_stats.asset_id": a.norm_stats_asset_id})        # refused unless both are given
    if cfg.get("family") != "pi0_fast":
        print(f"configs/model/{a.model}.yaml is a '{cfg.get('family')}' model: this measures FAST tokens (family pi0_fast)", file=sys.stderr)
        return 2
    horizons = a.horizon or [int(cfg["action_horizon"])]
    max_token_len = a.max_token_len or int(cfg["max_token_len"])
    folders = a.demos_dir or list(DEFAULT_DEMO_DIRS)
    pipe = pipeline if pipeline is not None else openpi_pipeline(cfg, a.measure_len)
    report = run(cfg, pipe, folders, horizons, max_token_len, a.max_decoding_steps, a.tail, a.limit_demos, a.margin)
    source = norm_stats_source(cfg)
    report["norm_stats"] = str(source[0] / source[1] / "norm_stats.json") if source else "the checkpoint's own"
    print(f"normalisation statistics: {report['norm_stats']}")
    print_report(report)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
