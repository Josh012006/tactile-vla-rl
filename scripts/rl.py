"""Online RL fine-tuning of pi0.5 on the FORGE PegInsert task: Best-of-N (the VLA stays frozen) and PA-RL (variants A and B: the VLA is distilled after every round) (docs/algos/rl.md).

    python -m scripts.rl --experiment configs/experiment/rl/pi05_best_of_n.yaml --start peg_insert_pi05_lora_teleop 4000      # SFT run (folder name or path) and checkpoint step to start from
    python -m scripts.rl --experiment configs/experiment/rl/pi05_best_of_n.yaml --start <run> <step> --dry-run                # what would run: no GPU, no model, no server
    python -m scripts.rl --experiment <file> --start <run> <step> --set total_episodes=100 --set warmup_critic_steps=500 --run-name probe       # a short probe, in the experiment's output_root
    python -m scripts.rl --resume runs/rl/peg_insert/<run name>                                                               # after a stop or a failure: continues after the last finished stage

Run it through jobs/job_rl_best_of_n.sh (Best-of-N) or jobs/job_rl_parl.sh --variant A|B (PA-RL) (Isaac Sim, the VLA and the critic share one GPU). The experiment file (configs/experiment/rl/*.yaml) says the method and the name of the run; everything else is configs/algo/rl.yaml.
`--start` (or the experiment's `start: {run, step}`) is the SFT checkpoint the VLA starts from: the model is the SFT run's own (its model YAML, its LoRA, its normalisation statistics, compared byte for byte with the copy
saved in the checkpoint, as in scripts/eval_sft.py).

The run (src/algos/rl_run.py): the warm-up (`warmup_episodes` of the base VLA, then `warmup_critic_steps` critic updates), then `total_episodes / envs_per_batch` rounds of collection with the critic and critic
updates (`utd` per environment step) and, for PA-RL, the distillation of the VLA on every stored state and the refresh of the whole candidate cache; with an evaluation after the warm-up, every `eval_every_rounds`
rounds and after the last (the 100 fixed episodes of the SFT evaluations, the critic's argmax over the VLA's candidates, on a fresh server).
Output <output_root>/<run name>/: config.json, metrics.jsonl (+ wandb), episodes/, resume/, critics/round_<r>.msgpack (every round), vlas/round_<r>/ (PA-RL: the VLA's trainable parameters after every round, an SFT
checkpoint layout, about 1.7 GB each), eval/results/<round>.json and eval/curve.*, logs/server_*.log.
The saved rounds are evaluated again, as the SFT checkpoints are, by scripts/eval_rl.py (jobs/job_eval_rl.sh).
XLA's deterministic GPU ops are NOT switched on here (the flag is global to the process and would slow the critic's training; the SFT evaluations are comparable, not bit-reproducible, because of the cameras: same starts,
same episode seeds, a standard error). SIGTERM / SIGINT: the run ends after the stage in progress (a second signal kills the servers at once).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from scripts.eval_sft import DEFAULT_TASK, check_norm_stats, discover_checkpoints, find_run_dir, git_commit, run_seed, task_settings
from scripts.sft import enable_compilation_cache, parse_assignments
from src.algos import rl_config
from src.models.interface import ROOT

FROZEN_ON_RESUME_EXCEPT = {"wandb_mode", "wandb_project", "rows_per_sample_call"}      # what only changes how the run is computed or logged: the rest is the run
UNSUPPORTED_FAMILIES = ("pi0_fast",)


# -- what the run starts from ------------------------------------------------------------------------------------------------------------------------
def resolve_start(exp: dict, cli: Optional[Tuple[str, int]]) -> dict:
    """The SFT checkpoint the VLA starts from: `--start RUN STEP` wins over the experiment's `start`. Returns {"run": <resolved run folder>, "step": int, "checkpoint": <its folder>, "run_config": <its config.json>}."""
    start = {"run": cli[0], "step": int(cli[1])} if cli else exp.get("start")
    if not start:
        raise SystemExit("no SFT checkpoint to start from: give --start RUN STEP, or a `start: {run, step}` in the experiment file")
    run_dir = find_run_dir(start["run"])
    found = discover_checkpoints(run_dir)
    if start["step"] not in found:
        raise SystemExit(f"{run_dir} has no complete checkpoint at step {start['step']} (found: {sorted(found)})")
    return {"run": str(run_dir), "step": int(start["step"]), "checkpoint": str(found[start["step"]]), "run_config": json.loads((run_dir / "config.json").read_text())}


def check_start(exp: dict, start: dict) -> dict:
    """The model of the run is the SFT run's: refuse what does not fit, and check the normalisation statistics. Returns its model config."""
    sft = start["run_config"]
    model_cfg = sft["model_cfg"]
    if model_cfg["family"] in UNSUPPORTED_FAMILIES:
        raise SystemExit(f"{model_cfg['family']} is not part of the RL methods (samples are not drawn inside one model call: docs/algos/rl.md)")
    if sft.get("model_name") != exp["model"]:
        raise SystemExit(f"the experiment is for {exp['model']}, the SFT run {start['run']} is a run of {sft.get('model_name')}")
    if exp.get("model_set"):
        raise SystemExit("`model_set` is not allowed in an RL experiment: the model is the SFT run's")
    from src.eval.sweep import Job
    check_norm_stats(model_cfg, [Job(label=str(start["step"]), step=start["step"], checkpoint=Path(start["checkpoint"]))])
    return model_cfg


def run_config(exp: dict, cfg: rl_config.RLConfig, start: dict, model_cfg: dict, task: dict, name: str) -> dict:
    return {"name": name, "description": exp.get("description"), "experiment": exp, "rl": dataclasses.asdict(cfg), "start": {k: start[k] for k in ("run", "step", "checkpoint")},
            "model_name": exp["model"], "model_cfg": model_cfg, "sft_seed": run_seed(start["run_config"]), "task": task, "git": git_commit()}


def resume_config(saved: dict, sets: Dict[str, object]) -> rl_config.RLConfig:
    """The run's own settings; `--set` may change only what does not shape the run (logging, how many environments are sampled per model call)."""
    forbidden = sorted(set(sets) - FROZEN_ON_RESUME_EXCEPT)
    if forbidden:
        raise SystemExit(f"--resume continues the run with its own settings: {forbidden} cannot be changed (only {sorted(FROZEN_ON_RESUME_EXCEPT)})")
    return rl_config.config_from_dict(saved["rl"], sets)


# -- the env servers -------------------------------------------------------------------------------------------------------------------------------
def make_opener(task_config: Path, env_set, logs_dir: Path, max_steps: int, ready_timeout: float, stop: threading.Event):
    """open_env(num_envs, label): a context manager that starts an env server of `num_envs` environments (own port and process group), waits for it, gives the client, and stops the server."""
    from src.envs.client import VectorEnvClient
    from src.eval.sweep import ServerError, make_server_factory
    factories: Dict[int, object] = {}

    @contextlib.contextmanager
    def open_env(num_envs: int, label: str):
        if num_envs not in factories:
            factories[num_envs] = make_server_factory(task_config, env_set, logs_dir, num_envs=num_envs)
        server = factories[num_envs](label)
        try:
            server.start()
            server.wait_ready(ready_timeout, stop)
            env = VectorEnvClient([server.endpoint], connect_timeout_s=120.0)
            try:
                if env.spec.max_steps != max_steps or env.num_envs != num_envs:
                    raise ServerError(f"the server has {env.num_envs} environments of {env.spec.max_steps} steps, the run needs {num_envs} of {max_steps}")
                print(f"env server '{label}' ready on port {server.endpoint[1]} ({num_envs} environments)", flush=True)
                yield env
            finally:
                env.close()
        finally:
            server.stop()

    return open_env


# -- command line --------------------------------------------------------------------------------------------------------------------------------
def print_plan(cfg: rl_config.RLConfig, run_dir: Path, start: Optional[dict], model_cfg: Optional[dict], max_steps: int) -> None:
    horizon = int((model_cfg or {}).get("action_horizon", 16))
    replan = cfg.replan_steps or max(1, horizon // 2)
    plans = -(-max_steps // replan)
    rows = (cfg.warmup_episodes + cfg.total_episodes) * (plans + 1)
    print(f"run folder: {run_dir}" + (" (EXISTS: choose another --run-name, or continue it with --resume)" if run_dir.exists() else ""))
    if start:
        print(f"starts from the SFT run {start['run']} at step {start['step']}; model {model_cfg['name']} ({model_cfg['family']}), chunk {horizon}, replan {replan}")
    print(f"method {cfg.variant}: k = {cfg.num_candidates} candidates, keep {cfg.keep_top}, {cfg.local_steps} local steps; rollouts '{cfg.rollout_selection}', evaluation '{cfg.eval_selection}'")
    print(f"warm-up: {cfg.warmup_episodes} episodes of the base VLA + {cfg.warmup_critic_steps} critic updates; then {cfg.rounds} rounds of {cfg.envs_per_batch} episodes, {cfg.utd} critic update(s) per environment step")
    if cfg.distills:
        print(f"distillation after every round: batch {cfg.distill_batch_size} (micro {cfg.distill_micro_batch_size}), lr {cfg.distill_lr}, {cfg.distill_epochs} epoch(s) over every stored state; labels '{cfg.distill_selection}'; "
              f"the cache of all of them is drawn again after each one; a VLA checkpoint (~1.7 GB) is kept per round")
    print(f"evaluations ({cfg.eval_episodes} episodes each) after the rounds {list(cfg.eval_rounds)}; replay buffer of up to {rows} states (about {rows * 2 * 150528 / 2**30:.1f} GiB of images)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", metavar="FILE", help="experiment file (configs/experiment/rl/*.yaml)")
    ap.add_argument("--start", nargs=2, metavar=("RUN", "STEP"), help="the SFT run (folder name or path) and the checkpoint step the VLA starts from (default: the experiment's `start`)")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a key of configs/algo/rl.yaml (the command line wins over the experiment)")
    ap.add_argument("--resume", metavar="RUN", help="continue the run RUN (a path, or a folder name under its output_root) after its last finished stage")
    ap.add_argument("--run-name", help="folder name of the run (default: the experiment's `name`)")
    ap.add_argument("--task-config", default=DEFAULT_TASK, help=f"env server YAML (default {DEFAULT_TASK})")
    ap.add_argument("--env-set", action="append", default=[], metavar="KEY=VALUE", help="override a key of the task YAML for the servers (not num_envs / server_port)")
    ap.add_argument("--dry-run", action="store_true", help="print what would run; load nothing")
    ap.add_argument("--ready-timeout", type=float, default=1800.0, help="seconds an env server may take to start (default 1800)")
    args = ap.parse_args(argv)
    if bool(args.resume) == bool(args.experiment):
        ap.error("give --experiment (a new run) or --resume (a run that stopped), not both and not neither")
    sets = parse_assignments(args.set)
    task_config = Path(args.task_config) if Path(args.task_config).is_absolute() else ROOT / args.task_config
    if not task_config.is_file():
        raise SystemExit(f"no task YAML {task_config}")

    if args.resume:
        base = rl_config.load_rl_config()
        candidates = [Path(args.resume), ROOT / args.resume, ROOT / base.output_root / args.resume, ROOT / "runs/rl/peg_insert" / args.resume, ROOT / "runs/rl/probing" / args.resume]
        run_dir = next((c.resolve() for c in candidates if (c / "config.json").exists()), None)
        if run_dir is None:
            raise SystemExit(f"no RL run {args.resume!r} (looked for config.json at that path and under runs/rl/)")
        saved = json.loads((run_dir / "config.json").read_text())
        cfg = resume_config(saved, sets)
        model_cfg, task, config = saved["model_cfg"], saved["task"], saved
        start = {**saved["start"], "run_config": json.loads((Path(saved["start"]["run"]) / "config.json").read_text())}
        resuming = True
    else:
        exp, cfg = rl_config.load_rl_experiment(args.experiment, sets)
        start = resolve_start(exp, tuple(args.start) if args.start else None)
        model_cfg = check_start(exp, start)
        task = task_settings(task_config, args.env_set, cfg.envs_per_batch)
        name = args.run_name or exp["name"]
        run_dir = ROOT / cfg.output_root / name
        config = {**run_config(exp, cfg, start, model_cfg, task, name), "env_set": list(args.env_set)}
        resuming = False
        if run_dir.exists() and not args.dry_run:
            raise SystemExit(f"{run_dir} exists: choose another --run-name, or continue it with --resume")
    print_plan(cfg, run_dir, start, model_cfg, int(task["max_steps_per_episode"]))
    if args.dry_run:
        return 0

    enable_compilation_cache()
    from src.algos import rl_run, tracking
    from src.algos import sft as sft_mod
    from src.eval.sweep import kill_all_servers
    from src.models import load_model

    stop = threading.Event()

    def on_signal(signum, _frame):
        if stop.is_set():
            kill_all_servers(2.0)
            sys.exit(130)
        print(f"\nsignal {signum}: the run ends after the stage in progress (a second signal kills the servers at once)", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resuming:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"loading the model ({model_cfg['name']}) and the SFT checkpoint {start['checkpoint']}", flush=True)
    policy = load_model(model_cfg, seed=run_seed(start["run_config"]))
    meta = sft_mod.load_checkpoint(policy, Path(start["checkpoint"]))
    if int(meta.get("step", start["step"])) != start["step"]:
        raise SystemExit(f"{start['checkpoint']} says it is step {meta.get('step')}")
    sink = tracking.start_tracking(run_dir, resuming, cfg.wandb_project, run_dir.name, cfg.wandb_mode, config, tags=[model_cfg["name"], cfg.variant], axes=tracking.RL_AXES)
    log = sft_mod.MetricsLog(run_dir / "metrics.jsonl", sink)
    open_env = make_opener(task_config, config.get("env_set", []), run_dir / "logs", int(task["max_steps_per_episode"]), args.ready_timeout, stop)
    run = rl_run.RLRun(cfg, policy, run_dir, open_env, log, max_steps=int(task["max_steps_per_episode"]), stop=stop, title=f"{run_dir.name}: success rate vs collection round",
                         model_cfg=model_cfg, checkpoint_meta={"start_run": start["run"], "start_step": start["step"]})
    try:
        summary = run.run()
    except BaseException as error:
        log.write(event="error", error=f"{type(error).__name__}: {str(error)[:600]}", memory_mib=sft_mod.device_memory_mib())
        log.close(exit_code=1)
        raise
    log.close()
    print(json.dumps(summary, indent=2))
    if summary["stopped"]:
        shown = run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir
        print(f"stopped after stage {summary['stage']}; continue with: --resume {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
