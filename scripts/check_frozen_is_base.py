"""Is the FROZEN part of a checkpoint the base model's, in bfloat16? (and in which precision is every part of a checkpoint stored?)

    python -m scripts.check_frozen_is_base --checkpoint <expo-ft run>/sft/<exp>/<dataset>/4000 [--model pi05] [--base <dir with params/>]

A checkpoint written like openpi's (expo-ft's SFT) holds EVERY parameter (7382 MiB for pi0.5 with LoRA): the trainable ones (467 M, float32) and the frozen ones (2936 M, bfloat16). The frozen ones
are never trained, so they must be the base checkpoint's (`checkpoint:` of the model YAML, downloaded by scripts.fetch_models) rounded to bfloat16, and a checkpoint of the trainable parameters only
loses nothing (tests/test_checkpoint_trainable_only_equals_full.py shows it on a tiny model; this script checks the premise on the real files: on expo-ft's step-4000 checkpoint, 20 / 20 frozen tensors
are equal, which is why `sft.save_checkpoint` writes the trainable parameters only). The checkpoints of scripts/sft.py are of that kind: given one, this script reports every frozen tensor as missing.

For each checkpoint it prints
  - the precision it is STORED in, from the files' metadata (no weights read): how many tensors and parameters, per group (trainable / frozen) and dtype, for the checkpoint and the base;
  - the comparison of every frozen tensor with the base's, rounded to bfloat16 the way the loaders do: how many are bit for bit equal, which ones are missing, of another shape or different
    (largest difference, share of the values that differ). Verdict EQUAL when all of them are equal.
Exit code 0 when every checkpoint is EQUAL, 1 otherwise.

Which parameters are frozen is the model's (`--model`, default pi05: its LoRA ranks must be those the checkpoint was trained with, 16 / 32 for expo-ft's), from openpi's freeze filter.
Memory: the base and one checkpoint are read whole, on the host (about 20 GB for pi0.5 with a float32 base): run it with `srun --mem=48G --cpus-per-task=4 -t 00:30:00`, no GPU
(JAX_PLATFORMS=cpu). Nothing is written.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

PARAMS_DIR = "params"
Key = Tuple[str, ...]


# -- pure helpers (tested on a tiny model, no GPU) -------------------------------------------------------------------------------------------
def stored_arrays(params_dir: Path) -> Dict[Key, Tuple[Tuple[int, ...], str]]:
    """{parameter path: (shape, dtype name)} of an orbax `params` folder, from its metadata (no weights are read). Both layouts: openpi's (`{"params": ...}`, each variable a `value`) and a plain tree."""
    import flax.traverse_util as tu
    import orbax.checkpoint as ocp
    with ocp.PyTreeCheckpointer() as checkpointer:
        tree = checkpointer.metadata(Path(params_dir)).tree
    if set(tree) == {"params"}:
        tree = tree["params"]
    out = {}
    for key, meta in tu.flatten_dict(tree).items():
        out[key[:-1] if key[-1] == "value" else key] = (tuple(meta.shape), str(meta.dtype))
    return out


def precision_table(stored: Mapping[Key, Tuple[Tuple[int, ...], str]], groups: Mapping[str, Set[Key]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """{group: {dtype: {"tensors": n, "parameters": n}}} for the parameters of `stored`; those in no group are called "other"."""
    names = {k: g for g, keys in groups.items() for k in keys}
    table: Dict[str, Dict[str, Dict[str, int]]] = {}
    for key, (shape, dtype) in stored.items():
        entry = table.setdefault(names.get(key, "other"), {}).setdefault(dtype, {"tensors": 0, "parameters": 0})
        entry["tensors"] += 1
        entry["parameters"] += int(np.prod(shape, dtype=np.int64))
    return table


def compare_frozen(checkpoint: Mapping[Key, np.ndarray], base: Mapping[Key, np.ndarray], frozen: Iterable[Key]) -> dict:
    """The frozen tensors of `checkpoint` against `base` rounded to bfloat16 (what `build_params_like_expo_ft` and openpi's `init_train_state` make of the base). Returns
    {"frozen": n, "equal": n, "missing_in_checkpoint": [...], "missing_in_base": [...], "other_shape": [...], "different": [{"key", "max_abs", "share"}], "checkpoint_dtypes": [...], "verdict": ...}."""
    import jax.numpy as jnp
    bf16 = jnp.bfloat16.dtype
    report = {"frozen": 0, "equal": 0, "missing_in_checkpoint": [], "missing_in_base": [], "other_shape": [], "different": [], "checkpoint_dtypes": set()}
    for key in sorted(frozen):
        report["frozen"] += 1
        name = "/".join(map(str, key))
        if key not in checkpoint:
            report["missing_in_checkpoint"].append(name)
            continue
        if key not in base:
            report["missing_in_base"].append(name)
            continue
        have, want = np.asarray(checkpoint[key]), np.asarray(base[key]).astype(bf16)
        report["checkpoint_dtypes"].add(str(have.dtype))
        if have.shape != want.shape:
            report["other_shape"].append(f"{name}: {have.shape} in the checkpoint, {want.shape} in the base")
        elif have.dtype == bf16 and np.array_equal(have, want):
            report["equal"] += 1
        else:
            h, w = have.astype(np.float32), want.astype(np.float32)
            differ = h != w
            report["different"].append({"key": name, "max_abs": float(np.max(np.abs(h - w))) if h.size else 0.0, "share": float(differ.mean()) if h.size else 0.0, "dtype": str(have.dtype),
                                        "same_values": bool(not differ.any())})
    report["checkpoint_dtypes"] = sorted(report["checkpoint_dtypes"])
    ok = report["equal"] == report["frozen"]
    report["verdict"] = "EQUAL" if ok else "NOT EQUAL"
    return report


def format_precision(table: Mapping[str, Mapping[str, Mapping[str, int]]]) -> List[str]:
    lines = []
    for group in sorted(table):
        for dtype, e in sorted(table[group].items()):
            lines.append(f"    {group:<10} {dtype:<10} {e['tensors']:>5} tensors {e['parameters'] / 1e6:>10.1f} M parameters")
    return lines


def format_report(report: Mapping) -> List[str]:
    lines = [f"    frozen tensors: {report['frozen']}, bit for bit equal to the base in bfloat16: {report['equal']}  ->  {report['verdict']}"]
    for field, text in (("missing_in_checkpoint", "missing in the checkpoint"), ("missing_in_base", "missing in the base"), ("other_shape", "of another shape")):
        if report[field]:
            lines.append(f"    {len(report[field])} {text}: " + "; ".join(report[field][:5]) + (" ..." if len(report[field]) > 5 else ""))
    for d in sorted(report["different"], key=lambda d: -d["max_abs"])[:10]:
        what = f"same values, stored as {d['dtype']}" if d["same_values"] else f"max |difference| {d['max_abs']:.3g}, {100 * d['share']:.2f} % of the values ({d['dtype']})"
        lines.append(f"    different: {d['key']}  {what}")
    if len(report["different"]) > 10:
        lines.append(f"    ... and {len(report['different']) - 10} more different tensors")
    return lines


# -- the model's groups and the files (needs openpi) ------------------------------------------------------------------------------------------
def model_groups(train_cfg) -> Dict[str, Set[Key]]:
    """{"trainable": paths, "frozen": paths} of the model, from openpi's freeze filter (abstract model: no weights)."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax
    state = nnx.split(nnx.eval_shape(train_cfg.model.create, jax.random.key(0)))[1]
    trainable = set(tu.flatten_dict(state.filter(train_cfg.trainable_filter).to_pure_dict()))
    everything = set(tu.flatten_dict(state.to_pure_dict()))
    return {"trainable": trainable, "frozen": everything - trainable}


def read_params(params_dir: Path) -> Dict[Key, np.ndarray]:
    """All the tensors of a `params` folder as stored, on the host (openpi's own reader, which both the base and the checkpoints go through)."""
    import flax.traverse_util as tu
    import openpi.models.model as _model
    return tu.flatten_dict(_model.restore_params(Path(params_dir), restore_type=np.ndarray))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True, metavar="DIR", help="checkpoint folders (each holds params/): an expo-ft step folder (or any whole openpi checkpoint)")
    ap.add_argument("--model", default="pi05", help="model YAML (configs/model/<name>.yaml) giving the frozen / trainable split and the base checkpoint (default pi05)")
    ap.add_argument("--base", metavar="DIR", help="the base checkpoint folder (holds params/); default: the YAML's `checkpoint`, as scripts.fetch_models downloaded it")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for c in args.checkpoint:
        if not (Path(c) / PARAMS_DIR).is_dir():
            raise SystemExit(f"{c} has no {PARAMS_DIR}/ folder: give the step folder of a checkpoint")
    from openpi.training import config as _config
    from src.models.interface import load_model_cfg
    from src.models.loading import _download
    from src.models.openpi_config import apply_settings
    from src.models.settings import parse_model_settings

    model_cfg = load_model_cfg(args.model)
    train_cfg = apply_settings(_config.get_config(model_cfg["openpi_config"]), parse_model_settings(model_cfg))
    base_dir = Path(args.base) if args.base else _download(model_cfg["checkpoint"])
    groups = model_groups(train_cfg)
    print(f"model {args.model}: {len(groups['trainable'])} trainable tensors, {len(groups['frozen'])} frozen ones; base {base_dir}")
    print("  base, as stored:")
    print("\n".join(format_precision(precision_table(stored_arrays(base_dir / PARAMS_DIR), groups))))
    base = read_params(base_dir / PARAMS_DIR)
    exit_code = 0
    for c in args.checkpoint:
        directory = Path(c).resolve()
        print(f"\n{directory}")
        print("  stored as:")
        print("\n".join(format_precision(precision_table(stored_arrays(directory / PARAMS_DIR), groups))))
        report = compare_frozen(read_params(directory / PARAMS_DIR), base, groups["frozen"])
        print("  frozen part against the base:")
        print("\n".join(format_report(report)))
        exit_code |= int(report["verdict"] != "EQUAL")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
