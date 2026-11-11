"""SFT of a VLA on the FORGE demonstrations: `python -m scripts.sft --model pi0_fast` (algorithm: src/algos/sft.py, settings: configs/algo/sft.yaml).

    python -m scripts.sft --model pi0_fast --dry-run                    # reads the demonstrations and prints what would be trained on: no GPU, no model
    python -m scripts.sft --model pi0_fast                              # LoRA (the model YAML's), the demonstrations of configs/algo/sft.yaml
    python -m scripts.sft --model pi05 --full                           # full fine-tuning: use_lora false in every component
    python -m scripts.sft --model pi05 --demos demos/isaaclab/ForgePegInsert_Robotiq_noisy --set steps=1500 --run-name pi05_noisy
    python -m scripts.sft --model pi05 --model-set vlm.lora_rank=8 --model-set action_head.lora_rank=16

--set KEY=VALUE overrides a key of configs/algo/sft.yaml, --model-set KEY=VALUE a key of the model YAML (`<component>.<key>`, as the probe's
--set); values are read as YAML (`--set val_episodes=5`, `--model-set vlm.use_lora=false`). The run goes to <output_root>/<run name>/.

Run it in the learner venv (`.venv`, `uv sync --group vla`) with the models fetched (python -m scripts.fetch_models); on a compute node
`export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false` (the jobs do it).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from src.algos import demos as demos_mod
from src.algos import sft as sft_mod
from src.models.interface import ROOT, apply_overrides, load_model_cfg
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


def run_name_of(model: str, folders, mode: str) -> str:
    tags = "+".join(Path(f).name.replace("ForgePegInsert_Robotiq_", "") for f in folders)
    return f"{model}_{tags}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"


def print_dry_run(train, val, cfg, model_cfg, mode) -> None:
    info = train.describe()
    print(f"model {model_cfg['name']} ({model_cfg['family']}), horizon {train.horizon}, {mode}")
    print(f"training set: {info['episodes']} episodes, {info['steps']} steps (min / mean / max per episode: {info['steps_per_episode']}), "
          f"{info['bytes'] / 2**30:.2f} GiB in memory")
    print(f"  observation keys: {info['keys']}\n  prompts: {info['prompts']}")
    if val is not None:
        print(f"held-out set: {val.describe()['episodes']} episodes, {len(val)} steps")
    epochs = cfg.steps * cfg.batch_size / len(train)
    print(f"{cfg.steps} updates of {cfg.batch_size} rows ({cfg.batch_size // cfg.micro_batch_size} micro-batches of {cfg.micro_batch_size}) = {epochs:.1f} epochs; "
          f"checkpoints every {cfg.save_interval} steps, evaluation every {cfg.eval_interval}")
    obs, prompts, target = train.sample([0, len(train) // 2, len(train) - 1])
    print(f"a sample: first chunk target (env space, absolute) {target[0, 0].round(3).tolist()}; last row's chunk repeats the last action: "
          f"{bool((target[2] == target[2, -1]).all())}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="configs/model/<name>.yaml")
    ap.add_argument("--full", action="store_true", help="full fine-tuning (use_lora false in every component) instead of the YAML's LoRA")
    ap.add_argument("--demos", nargs="+", metavar="FOLDER", help="folders of ep_*.npz, instead of the `demos` of the config")
    ap.add_argument("--config", default=str(sft_mod.CONFIG_PATH), help="algorithm YAML (default configs/algo/sft.yaml)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a key of the algorithm YAML")
    ap.add_argument("--model-set", action="append", metavar="KEY=VALUE", help="override a key of the model YAML: <component>.<key>")
    ap.add_argument("--run-name", help="folder name of the run (default: <model>_<demos>_<lora|full>_<date>)")
    ap.add_argument("--dry-run", action="store_true", help="read the demonstrations and print what would be trained on; load no model")
    args = ap.parse_args(argv)

    sets = parse_assignments(args.set)
    if args.demos:
        sets["demos"] = args.demos
    cfg = sft_mod.load_config(args.config, sets)
    base_cfg = load_model_cfg(args.model)
    model_cfg = apply_overrides(base_cfg, model_overrides(base_cfg, args.full, parse_assignments(args.model_set)))
    mode = "lora" if parse_model_settings(model_cfg).any_lora else "full"

    train, val = demos_mod.load_demos(cfg.demos, int(model_cfg["action_horizon"]), num_demos=cfg.num_demos, val_episodes=cfg.val_episodes,
                                      default_prompt=cfg.prompt, root=ROOT)
    if args.dry_run:
        print_dry_run(train, val, cfg, model_cfg, mode)
        return 0

    from src.models import load_model
    run_dir = ROOT / cfg.output_root / (args.run_name or run_name_of(args.model, cfg.demos, mode))
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists: choose another --run-name")
    print(f"run: {run_dir}", flush=True)
    policy = load_model(model_cfg, seed=cfg.seed)
    meta = {"model_name": args.model, "mode": mode, "model_cfg": model_cfg, "demos": list(cfg.demos)}
    summary = sft_mod.train(policy, train, val, cfg, run_dir, meta)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
