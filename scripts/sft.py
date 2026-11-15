"""SFT of a VLA on the FORGE demonstrations: `python -m scripts.sft --model pi0_fast` (algorithm: src/algos/sft.py, settings: configs/algo/sft.yaml).

    python -m scripts.sft --experiment configs/experiment/sft/pi05_lora_scripted.yaml           # a main run: everything it needs is in the experiment file
    python -m scripts.sft --experiment configs/experiment/sft/pi05_lora_scripted.yaml --dry-run # reads the demonstrations and prints what would be trained on: no GPU, no model
    python -m scripts.sft --model pi0_fast --dry-run                    # the same for a run given on the command line
    python -m scripts.sft --model pi0_fast                              # LoRA (the model YAML's), the demonstrations of configs/algo/sft.yaml
    python -m scripts.sft --model pi05 --full                           # full fine-tuning: use_lora false in every component
    python -m scripts.sft --model pi05 --demos demos/isaaclab/ForgePegInsert_Robotiq_noisy --set steps=1500 --run-name pi05_noisy
    python -m scripts.sft --model pi05 --model-set vlm.lora_rank=8 --model-set action_head.lora_rank=16

    python -m scripts.sft --resume runs/sft/peg_insert/peg_insert_pi05_lora_scripted            # continue a stopped run (its path, or its name under output_root)
    python -m scripts.sft --resume <run> --set steps=6000                                       # ... and train longer (constant learning rate only)

--resume: the model, its overrides and the demonstrations are the run's own (config.json), so only --set is accepted next to it, and only for what
may change on resume (steps, intervals, eval_samples, prefetch: see src/algos/sft.py). The snapshot is <run>/resume/, written every `resume_interval`
steps, at the last step and when the job gets a SIGTERM. The wandb run of the first attempt (<run>/wandb_id.txt) is continued, not a new one.

wandb: `wandb login` once on the login node; `--set wandb_mode=offline` (then `wandb sync <run>/wandb/offline-run-*`) or `--set wandb_mode=disabled` to change it.

--experiment FILE (configs/experiment/sft/*.yaml) gives the model, the demonstrations, the run name and the overrides of a MAIN run, whose output_root is
runs/sft/peg_insert; what is given on the command line is added to it and wins. Without an experiment, a run is a probe: output_root is runs/sft/probing.

--set KEY=VALUE overrides a key of configs/algo/sft.yaml, --model-set KEY=VALUE a key of the model YAML (`<component>.<key>`, as the probe's
--set); values are read as YAML (`--set val_episodes=5`, `--model-set vlm.use_lora=false`). The run goes to <output_root>/<run name>/.

Run it in the learner venv (`.venv`, `uv sync --group vla`) with the models fetched (python -m scripts.fetch_models); on a compute node
`export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false` (the jobs do it).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.algos import demos as demos_mod
from src.algos import sft as sft_mod
from src.models.interface import ROOT, apply_overrides, load_model_cfg, norm_stats_source
from src.models.settings import FAMILIES_WITH_ACTION_HEAD, parse_model_settings


def parse_assignments(items: Optional[List[str]]) -> Dict[str, object]:
    """["a=1", "b.c=false"] -> {"a": 1, "b.c": False} (values are YAML)."""
    out: Dict[str, object] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def model_overrides(model_cfg: dict, full: bool, model_set: Dict[str, object]) -> Dict[str, object]:
    """--full: use_lora false in every component the model has (the exact base function, every weight trained), then --model-set on top."""
    overrides: Dict[str, object] = {}
    if full:
        overrides["vlm.use_lora"] = False
        if model_cfg["family"] in FAMILIES_WITH_ACTION_HEAD:
            overrides["action_head.use_lora"] = False
    overrides.update(model_set)
    return overrides


def load_output_root(config: str) -> str:
    return sft_mod.load_config(config).output_root


def find_run_dir(name_or_path: str, output_root: str) -> Path:
    """A run given as a path (absolute, or relative to where the command runs) or as its folder name under output_root."""
    for candidate in (Path(name_or_path), ROOT / output_root / name_or_path):
        if (candidate / "config.json").exists():
            return candidate.resolve()
    raise SystemExit(f"no run {name_or_path!r} (looked for config.json at that path and under {output_root}/)")


def run_name_of(model: str, folders, mode: str) -> str:
    tags = "+".join(Path(f).name.replace("ForgePegInsert_Robotiq_", "") for f in folders)
    return f"{model}_{tags}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"


EXPERIMENT_KEYS = {"name", "description", "model", "full", "demos", "set", "model_set"}


def load_experiment(path) -> dict:
    """An experiment file (configs/experiment/sft/*.yaml): `name` (the run's folder), `model`, `full`, `demos`, `set` (keys of the algorithm YAML),
    `model_set` (keys of the model YAML), `description`. Unknown keys and wrong types are refused, not ignored."""
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"no experiment file {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: an experiment is a mapping")
    unknown = sorted(set(raw) - EXPERIMENT_KEYS)
    if unknown:
        raise SystemExit(f"{path}: unknown keys {unknown} (allowed: {sorted(EXPERIMENT_KEYS)})")
    for key in ("name", "model"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise SystemExit(f"{path}: `{key}` is required (a string)")
    if "demos" in raw and not (isinstance(raw["demos"], list) and raw["demos"] and all(isinstance(d, str) for d in raw["demos"])):
        raise SystemExit(f"{path}: `demos` is a list of folders")
    if not isinstance(raw.get("full", False), bool):
        raise SystemExit(f"{path}: `full` is true or false")
    for key in ("set", "model_set"):
        if not isinstance(raw.get(key) or {}, dict):
            raise SystemExit(f"{path}: `{key}` is a mapping")
    return raw


def stats_description(model_cfg: dict) -> str:
    """Where the normalisation statistics come from; a file that is not there stops the command before anything heavy is loaded."""
    source = norm_stats_source(model_cfg)
    if source is None:
        return "the checkpoint's own (DROID)"
    file = source[0] / source[1] / "norm_stats.json"
    if not file.is_file():
        raise SystemExit(f"normalisation statistics not found: {file} (the `norm_stats` block of the model YAML)")
    return str(file)


def print_dry_run(train, val, cfg, model_cfg, mode) -> None:
    info = train.describe()
    print(f"model {model_cfg['name']} ({model_cfg['family']}), horizon {train.horizon}, {mode}")
    print(f"normalisation statistics: {stats_description(model_cfg)}")
    print(f"training set: {info['episodes']} episodes, {info['steps']} steps (min / mean / max per episode: {info['steps_per_episode']}), "
          f"{info['bytes'] / 2**30:.2f} GiB in memory")
    print(f"  observation keys: {info['keys']}\n  prompts: {info['prompts']}")
    if val is not None:
        print(f"held-out set: {val.describe()['episodes']} episodes, {len(val)} steps")
    epochs = cfg.steps * cfg.batch_size / len(train)
    print(f"{cfg.steps} updates of {cfg.batch_size} rows ({cfg.batch_size // cfg.micro_batch_size} micro-batches of {cfg.micro_batch_size}) = {epochs:.1f} epochs; "
          f"checkpoints every {cfg.save_interval} steps, evaluation every {cfg.eval_interval}, wandb {cfg.wandb_mode} (project {cfg.wandb_project})")
    obs, prompts, target = train.sample([0, len(train) // 2, len(train) - 1])
    print(f"a sample: first chunk target (env space, absolute) {target[0, 0].round(3).tolist()}; last row's chunk repeats the last action: "
          f"{bool((target[2] == target[2, -1]).all())}")


def enable_compilation_cache() -> None:
    """JAX's persistent compilation cache, as openpi's scripts/train.py and so expo-ft have it (~/.cache/jax): the first step of every launch compiled for about 100 s
    (97 to 99 s measured for pi0.5 full, and again in the first evaluation), and so does every resume. The cache only stores what XLA compiled: results do not change.
    JAX_COMPILATION_CACHE_DIR, when set, is the folder instead."""
    import jax
    if not os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", metavar="FILE", help="experiment file (configs/experiment/sft/*.yaml): model, demonstrations, run name and overrides of a main run")
    ap.add_argument("--model", help="configs/model/<name>.yaml (not needed with --experiment or --resume)")
    ap.add_argument("--full", action="store_true", help="full fine-tuning (use_lora false in every component) instead of the YAML's LoRA")
    ap.add_argument("--demos", nargs="+", metavar="FOLDER", help="folders of ep_*.npz, instead of the `demos` of the config")
    ap.add_argument("--config", default=str(sft_mod.CONFIG_PATH), help="algorithm YAML (default configs/algo/sft.yaml)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a key of the algorithm YAML")
    ap.add_argument("--model-set", action="append", metavar="KEY=VALUE", help="override a key of the model YAML: <component>.<key>")
    ap.add_argument("--resume", metavar="RUN", help="continue the run RUN (a path, or a folder name under output_root) from its resume snapshot")
    ap.add_argument("--run-name", help="folder name of the run (default: <model>_<demos>_<lora|full>_<date>)")
    ap.add_argument("--dry-run", action="store_true", help="read the demonstrations and print what would be trained on; load no model")
    args = ap.parse_args(argv)

    exp = load_experiment(args.experiment) if args.experiment else {}
    if args.resume and exp:
        raise SystemExit("--resume takes everything from the run: no --experiment next to it")
    if exp:
        if args.model and args.model != exp["model"]:
            raise SystemExit(f"{args.experiment} is an experiment of {exp['model']}, not of {args.model}")
        args.model = exp["model"]
    if not args.resume and not args.model:
        ap.error("--experiment or --model is required (or --resume)")
    args.full = args.full or bool(exp.get("full", False))
    args.demos = args.demos or exp.get("demos")
    args.run_name = args.run_name or exp.get("name")
    sets = {**(exp.get("set") or {}), **parse_assignments(args.set)}               # the command line wins over the experiment
    model_set = {**(exp.get("model_set") or {}), **parse_assignments(args.model_set)}
    saved = None
    if args.resume:
        if args.full or args.demos or args.model_set or args.dry_run:
            raise SystemExit("--resume takes the model, its overrides and the demonstrations from the run: only --set is accepted next to it")
        run_dir = find_run_dir(args.resume, load_output_root(args.config))
        saved = json.loads((run_dir / "config.json").read_text())
        if args.model and saved["model_name"] != args.model:
            raise SystemExit(f"{run_dir} is a run of {saved['model_name']}, not of {args.model}")
        args.model = saved["model_name"]
        cfg = sft_mod.config_from_dict(saved["sft"], sets)
        model_cfg, mode = saved["model_cfg"], saved["mode"]
    else:
        if args.demos:
            sets["demos"] = args.demos
        cfg = sft_mod.load_config(args.config, sets)
        base_cfg = load_model_cfg(args.model)
        model_cfg = apply_overrides(base_cfg, model_overrides(base_cfg, args.full, model_set))
        mode = "lora" if parse_model_settings(model_cfg).any_lora else "full"
        run_dir = ROOT / cfg.output_root / (args.run_name or run_name_of(args.model, cfg.demos, mode))

    train, val = demos_mod.load_demos(cfg.demos, int(model_cfg["action_horizon"]), num_demos=cfg.num_demos, val_episodes=cfg.val_episodes,
                                      default_prompt=cfg.prompt, root=ROOT)
    if args.dry_run:
        print_dry_run(train, val, cfg, model_cfg, mode)
        print(f"run folder: {run_dir}" + (" (EXISTS: choose another name)" if run_dir.exists() else ""))
        return 0

    enable_compilation_cache()
    from src.models import load_model
    if not args.resume and run_dir.exists():
        raise SystemExit(f"{run_dir} exists: choose another --run-name, or continue it with --resume")
    print(f"run: {run_dir}" + (" (resumed)" if args.resume else ""), flush=True)
    policy = load_model(model_cfg, seed=cfg.seed, for_training=True)       # the state openpi's train.py starts from: LoRA initialised with its key, trainable parameters in float32
    meta = {"model_name": args.model, "mode": mode, "model_cfg": model_cfg, "demos": list(cfg.demos)}
    summary = sft_mod.train(policy, train, val, cfg, run_dir, meta, resume=bool(args.resume))
    print(json.dumps(summary, indent=2))
    if summary["stopped"]:
        shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
        print(f"stopped at step {summary['last_step']} on a SIGTERM; continue with: --resume {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
