"""Put an expo-ft SFT checkpoint in the form `scripts/eval_sft.py` evaluates, to run it through THIS repository's evaluation.

    python -m scripts.adopt_expo_ft_checkpoint --checkpoint <expo-ft run>/sft/<exp name>/<dataset>/4000 --name expoft_teleop_4000
    sbatch --time=00:45:00 jobs/job_eval_sft.sh --run _adopted_expoft_teleop_4000 --steps 0 --n-episodes 100 --envs-per-server 100 --output-dir runs/sft/probing/_cost_expoft_teleop_4000

Why: to know whether a result of this repository's SFT (e.g. 41 % at step 4000 for the teleop pi0.5) differs from expo-ft's (66 % at 4000 on its curve) because of the TRAINING or because of the EVALUATION.
The same weights, evaluated here: about 66 % says the evaluation is fine and the training differs; about 41 % says the evaluation differs from expo-ft's.

What it does: writes `runs/sft/probing/_adopted_<name>/config.json` (no checkpoint folder, nothing is copied): the model YAML (`--model`, default pi05) whose `checkpoint` is the expo-ft step folder (its `params/` hold the base weights
AND the LoRA tensors: `src/models/loading.py` loads them through the same weight loader as expo-ft) and whose `norm_stats` block is the one saved with that checkpoint (`<checkpoint>/assets/<asset id>/norm_stats.json`: the statistics it was trained with).
The evaluation then calls it **step 0** ("the run's initial state" = the model as loaded). Nothing is trained or changed. The LoRA ranks of the model YAML must be those of the expo-ft run (pi05.yaml: 16 / 32, the config expo-ft used).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.models.interface import ROOT, load_model_cfg, norm_stats_source

PROBING = ROOT / "runs" / "sft" / "probing"


def adopted_config(model: str, checkpoint: Path, asset_id: Optional[str] = None) -> dict:
    """The run config (`model_cfg` as scripts/sft.py saves it) of an expo-ft checkpoint folder: its weights as the YAML's `checkpoint`, its normalisation statistics as the `norm_stats` block."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not (checkpoint / "params").is_dir():
        raise SystemExit(f"{checkpoint} has no params/ folder: give the STEP folder of an expo-ft SFT checkpoint (<run>/sft/<experiment>/<dataset>/<step>)")
    assets = checkpoint / "assets"
    ids = sorted(d.name for d in assets.iterdir() if (d / "norm_stats.json").is_file()) if assets.is_dir() else []
    if asset_id is not None:
        if asset_id not in ids:
            raise SystemExit(f"{assets} has no norm_stats.json for the asset id {asset_id!r} (it has {ids})")
        ids = [asset_id]
    if len(ids) != 1:
        raise SystemExit(f"{assets} holds {len(ids)} sets of normalisation statistics {ids}: give --asset-id" if ids else f"{assets} holds no <asset id>/norm_stats.json: the statistics this checkpoint was trained with are needed")
    cfg = load_model_cfg(model)
    cfg["checkpoint"] = str(checkpoint)
    cfg["norm_stats"] = {"dir": str(assets), "asset_id": ids[0]}
    norm_stats_source(cfg)                                                           # validates the block
    return {"adopted_from": str(checkpoint), "model": model, "model_cfg": cfg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="an expo-ft SFT checkpoint STEP folder (holds params/ and assets/)")
    ap.add_argument("--name", required=True, help="the run is written as runs/sft/probing/_adopted_<name>/")
    ap.add_argument("--model", default="pi05", help="configs/model/<name>.yaml (default pi05)")
    ap.add_argument("--asset-id", help="which assets/<asset id> when the checkpoint holds several")
    args = ap.parse_args(argv)
    config = adopted_config(args.model, Path(args.checkpoint), args.asset_id)
    run = PROBING / f"_adopted_{args.name}"
    if run.exists():
        raise SystemExit(f"{run} exists: use another --name (nothing is overwritten)")
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(config, indent=1))
    print(f"written: {run / 'config.json'}\nweights: {config['model_cfg']['checkpoint']}\nnormalisation statistics: {config['model_cfg']['norm_stats']['dir']}/{config['model_cfg']['norm_stats']['asset_id']}/norm_stats.json\n"
          f"evaluate it (its only checkpoint is called step 0):\n  sbatch --time=00:45:00 jobs/job_eval_sft.sh --run _adopted_{args.name} --steps 0 --n-episodes 100 --envs-per-server 100 --output-dir runs/sft/probing/_cost_{args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
