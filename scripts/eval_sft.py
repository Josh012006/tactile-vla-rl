"""Success-rate curve of an SFT run: every checkpoint, and the run's initial state (step 0), evaluated in the FORGE environment on the same fixed episodes.

    python -m scripts.eval_sft --run runs/sft/peg_insert/peg_insert_pi05_lora_teleop --n-episodes 200 --workers 3
    python -m scripts.eval_sft --run <run> --dry-run                      # what would be evaluated and what is already done: no GPU, no model, no server
    python -m scripts.eval_sft --run <run> --steps 0 1000 4000            # only these checkpoints (0 = the run's initial state)
    python -m scripts.eval_sft --run <run> --plot-only --title "Teleop demos"      # rebuild curve.json / .png / .pdf from the results on disk: no GPU, no server

Run it through jobs/job_eval_sft.sh (an env server is started for each checkpoint: Isaac Sim on the same GPU as the policy).

What is evaluated. `--run` is a run folder of scripts/sft.py (runs/sft/peg_insert/<name>, or a probe under runs/sft/probing/): its config.json gives the model YAML with its
overrides and normalisation statistics, and checkpoints/<step>/ the trained parameters. Step 0 is the run's own initial state: the same model, the same LoRA initialisation
(openpi's, seed 0), nothing trained: the point the curve starts from (`--no-initial` leaves it out).

The episodes. `--seed` (default 42) generates the `--n-episodes` seeds ONCE (the way expo-ft does: episode_seeds.json, kept in the output folder); every checkpoint plays the same
ones. Each episode is `reset(seed=<its seed>)` in a server with ONE environment, a chunk is executed `--replan` steps (default: half the model's chunk, as expo-ft's 8 of 16) and
the episode lasts at most `max_steps_per_episode` of the task YAML.

Deterministic evaluation (docs/eval/sft_eval.md). pi0-FAST decodes greedily; pi0.5's flow noise restarts from the episode's seed at every episode; the model is always called with a
batch of 1; XLA's GPU kernels are set deterministic (XLA_FLAGS --xla_gpu_deterministic_ops=true, added here); each checkpoint gets a FRESH env server (the physics is bit-identical for
an identical sequence of calls from a fresh server, on the same GPU model). Every episode records `start_sha` and `traj_sha`, every checkpoint an `eval_sha`: two evaluations are
identical when these are.

Resume. Re-run the SAME command. Each finished episode is a line of <output-dir>/results/<step>.jsonl: a checkpoint that was cut is continued from its missing episodes only, and its
result and the curve are recomputed from the file as if nothing had happened. (A checkpoint evaluated in several attempts is statistically equivalent to one done in a single
attempt, not bit-identical: docs/envs/isaaclab_env.md N1. `single_attempt` in its result says which one it is.) Another --seed / --replan / task YAML / model than the folder's is refused.
`--n-episodes` may grow: the first seeds do not change, only the missing episodes are run. `--force` sets aside (`.prev1`, ...) the results of the checkpoints it evaluates and starts them over.

Parallel. `--workers N` evaluates N checkpoints at the same time: one thread per checkpoint, each with its own env server (a subprocess, own port and process group) and its own
policy (they share the compiled model and the base weights, so N workers cost N env servers plus one model in GPU memory). SIGTERM / SIGINT stop it cleanly: the episodes in progress are dropped,
every server is killed, the finished episodes are kept.

Output (--output-dir, default <run>/eval): eval_protocol.json (what the results were made with, and one entry per attempt), episode_seeds.json, results/<step>.jsonl and .json,
logs/server_<step>.log, videos/<step>/ with --save-videos; curve.json / curve.png / curve.pdf in --curve-dir (default: the output folder).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from scripts.sft import enable_compilation_cache, parse_assignments
from src.eval import episodes as E
from src.eval.curve import write_curve
from src.eval.sweep import Job, Sweep, gpu_name, kill_all_servers, make_server_factory
from src.models.interface import ROOT

RECIPE = "deterministic-v1"                  # policy seeded per episode, batch of 1, fresh server per checkpoint, one environment per server
DEFAULT_TASK = "configs/task/peg_insert_forge.yaml"
XLA_DETERMINISTIC = "--xla_gpu_deterministic_ops=true"
FORBIDDEN_ENV_SET = ("num_envs", "server_port")


# -- the run and its checkpoints -------------------------------------------------------------------------------------------------------
def find_run_dir(name_or_path: str) -> Path:
    """A run given as a path (absolute, or relative to where the command runs or to the repository) or as its folder name under runs/sft/peg_insert or runs/sft/probing."""
    for candidate in (Path(name_or_path), ROOT / name_or_path, ROOT / "runs/sft/peg_insert" / name_or_path, ROOT / "runs/sft/probing" / name_or_path):
        if (candidate / "config.json").exists():
            return candidate.resolve()
    raise SystemExit(f"no SFT run {name_or_path!r} (looked for config.json at that path and under runs/sft/peg_insert/ and runs/sft/probing/)")


def discover_checkpoints(run_dir: Path) -> Dict[int, Path]:
    """step -> folder, for the numeric folders of <run>/checkpoints that are complete (meta.json is written last)."""
    found: Dict[int, Path] = {}
    folder = run_dir / "checkpoints"
    if folder.is_dir():
        for entry in folder.iterdir():
            if entry.is_dir() and entry.name.isdigit() and (entry / "meta.json").exists():
                found[int(entry.name)] = entry
    return dict(sorted(found.items()))


def select_jobs(found: Dict[int, Path], steps: Optional[List[int]], no_initial: bool) -> List[Job]:
    """The checkpoints to evaluate, by step. Without --steps: the initial state (step 0) and every checkpoint found. With --steps: exactly those (an unknown one is an error)."""
    if steps:
        wanted = sorted(set(steps))
        missing = [s for s in wanted if s != 0 and s not in found]
        if missing:
            raise SystemExit(f"--steps {missing} not found among the checkpoints: {sorted(found)}")
    else:
        wanted = ([] if no_initial else [0]) + list(found)
    return [Job(label=str(s), step=s, checkpoint=None if s == 0 else found[s]) for s in wanted]


# -- protocol ---------------------------------------------------------------------------------------------------------------------------
def task_settings(task_config: Path, env_set: List[str]) -> dict:
    """The task YAML with the `--env-set KEY=VALUE` overrides applied, as the server will read it (one environment)."""
    task = yaml.safe_load(Path(task_config).read_text()) or {}
    for key, value in parse_assignments(env_set).items():
        if key in FORBIDDEN_ENV_SET:
            raise SystemExit(f"--env-set {key}: set by this script (one environment per server, a free port)")
        task[key] = value
    task["num_envs"] = 1
    task.pop("server_port", None)
    return task


def build_protocol(args, run_dir: Path, model_cfg: dict, task: dict) -> dict:
    horizon = int(model_cfg["action_horizon"])
    replan = int(args.replan) if args.replan else max(1, horizon // 2)
    if not 1 <= replan <= horizon:
        raise SystemExit(f"--replan {replan}: between 1 and the model's chunk length ({horizon})")
    return {"recipe": RECIPE, "master_seed": int(args.seed), "replan": replan, "max_steps": int(task["max_steps_per_episode"]), "num_envs": 1,
            "model_cfg": model_cfg, "task": task, "run": str(run_dir)}


def with_deterministic_xla(flags: str) -> str:
    """`flags` (the XLA_FLAGS of the environment) with XLA's deterministic GPU ops added once."""
    return flags if XLA_DETERMINISTIC in flags.split() else (flags + " " + XLA_DETERMINISTIC).strip()


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def acquire_run_lock(output_dir: Path):
    """One evaluation at a time per output folder (two would append to the same files). Returns the open file (keep it open); a filesystem without locks only warns."""
    import fcntl
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = open(output_dir / "eval.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another evaluation is running on {output_dir} (eval.lock is held): wait for it, or use another --output-dir")
    except OSError as e:
        print(f"WARNING: could not lock {output_dir / 'eval.lock'} ({e!r}): do not start two evaluations on the same folder", flush=True)
    return handle


def set_aside(path: Path) -> Optional[Path]:
    """Rename `path` to <name>.prev<k> (the first k that is free): nothing is ever overwritten or deleted. None when `path` does not exist."""
    if not path.exists():
        return None
    k = 1
    while path.with_name(f"{path.name}.prev{k}").exists():
        k += 1
    target = path.with_name(f"{path.name}.prev{k}")
    path.rename(target)
    return target


# -- report ------------------------------------------------------------------------------------------------------------------------------
def progress_of(output_dir: Path, jobs: List[Job], seeds: List[int]) -> Dict[str, int]:
    return {j.label: sum(1 for i in E.EpisodeLog(output_dir / "results" / f"{j.label}.jsonl").load() if i < len(seeds)) for j in jobs}


def print_table(results_dir: Path) -> None:
    from src.eval.curve import load_entries
    entries = load_entries(results_dir)
    if not entries:
        return
    print(f"\n{'step':>6} {'success':>9} {'SE':>6} {'Wilson 95 %':>16} {'episodes':>9}  attempts  eval_sha")
    for e in entries:
        lo, hi = e["wilson95"]
        print(f"{e['step']:>6} {100 * e['success_rate']:>8.1f}% {100 * e['success_se']:>5.1f}% {f'[{100 * lo:.0f}, {100 * hi:.0f}]':>16} {e['n_episodes']:>9}  "
              f"{'1' if e['single_attempt'] else 'several':>8}  {e['eval_sha']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, metavar="RUN", help="an SFT run: its path, or its folder name under runs/sft/peg_insert/ or runs/sft/probing/")
    ap.add_argument("--n-episodes", type=int, default=50, help="episodes per checkpoint (default 50, expo-ft's); may grow between runs")
    ap.add_argument("--seed", type=int, default=E.DEFAULT_SEED, help="master seed of the fixed episode list (default 42)")
    ap.add_argument("--steps", type=int, nargs="+", metavar="STEP", help="only these checkpoints (0 = the run's initial state); default: 0 and every checkpoint of the run")
    ap.add_argument("--no-initial", action="store_true", help="leave the initial state (step 0) out of the default list")
    ap.add_argument("--output-dir", metavar="DIR", help="where the evaluation data goes (default: <run>/eval)")
    ap.add_argument("--curve-dir", metavar="DIR", help="where curve.json / curve.png / curve.pdf go (default: the output folder)")
    ap.add_argument("--workers", type=int, default=1, help="checkpoints evaluated at the same time (default 1): one thread, one env server and one policy each")
    ap.add_argument("--replan", type=int, help="steps executed per chunk before the model is called again (default: half the model's chunk)")
    ap.add_argument("--task-config", default=DEFAULT_TASK, help=f"env server YAML (default {DEFAULT_TASK})")
    ap.add_argument("--env-set", action="append", default=[], metavar="KEY=VALUE", help="override a key of the task YAML for the servers (not num_envs / server_port)")
    ap.add_argument("--save-videos", action="store_true", help="one video per episode in <output-dir>/videos/<step>/")
    ap.add_argument("--video-dir", metavar="DIR", help="parent folder of the videos (implies --save-videos)")
    ap.add_argument("--title", help="title of the plot (small font, wrapped); default: 'Eval success rate vs. checkpoint — <run name>'")
    ap.add_argument("--plot-only", action="store_true", help="only rebuild curve.json / .png / .pdf from the results on disk")
    ap.add_argument("--force", action="store_true", help="set aside the results of the selected checkpoints (.prev1, ...) and evaluate them from scratch")
    ap.add_argument("--dry-run", action="store_true", help="print what would be evaluated and what is already done; load nothing")
    ap.add_argument("--ready-timeout", type=float, default=1800.0, help="seconds an env server may take to start (default 1800)")
    args = ap.parse_args(argv)

    run_dir = find_run_dir(args.run)
    run_config = json.loads((run_dir / "config.json").read_text())
    model_cfg = run_config["model_cfg"]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval"
    curve_dir = Path(args.curve_dir).resolve() if args.curve_dir else output_dir
    results_dir = output_dir / "results"
    title = args.title

    if args.plot_only:
        entries = write_curve(results_dir, curve_dir, title, name=run_dir.name)
        if not entries:
            raise SystemExit(f"no complete result in {results_dir}")
        print_table(results_dir)
        print(f"curve: {curve_dir / 'curve.png'}, {curve_dir / 'curve.pdf'}, {curve_dir / 'curve.json'}")
        return 0

    found = discover_checkpoints(run_dir)
    jobs = select_jobs(found, args.steps, args.no_initial)
    if not jobs:
        raise SystemExit(f"{run_dir} has no checkpoint yet (checkpoints/<step>/meta.json)")
    task_config = Path(args.task_config) if Path(args.task_config).is_absolute() else ROOT / args.task_config
    if not task_config.is_file():
        raise SystemExit(f"no task YAML {task_config}")
    task = task_settings(task_config, args.env_set)
    protocol = build_protocol(args, run_dir, model_cfg, task)
    protocol_path = output_dir / E.PROTOCOL_FILE
    saved = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    try:
        if saved is not None:
            E.check_protocol(saved, protocol)
        seeds = E.make_seeds(args.seed, args.n_episodes) if args.dry_run else E.load_or_create_seeds(output_dir, args.seed, args.n_episodes)
    except ValueError as e:
        raise SystemExit(str(e))

    done = progress_of(output_dir, jobs, seeds)
    print(f"run {run_dir}\nmodel {model_cfg['name']} ({model_cfg['family']}), chunk {model_cfg['action_horizon']}, replan {protocol['replan']}, episodes of at most {protocol['max_steps']} steps")
    print(f"{len(seeds)} episodes per checkpoint, master seed {args.seed}, {args.workers} worker(s); data in {output_dir}, curve in {curve_dir}")
    for j in jobs:
        where = "initial state of the run" if j.checkpoint is None else str(j.checkpoint.relative_to(run_dir))
        print(f"  step {j.step:>6}: {done[j.label]:>4}/{len(seeds)} episodes done   ({where})")
    if args.dry_run:
        return 0

    lock = acquire_run_lock(output_dir)                                      # noqa: F841  (held until the process ends)
    if args.force:
        for j in jobs:
            for name in (f"{j.label}.jsonl", f"{j.label}.json"):
                moved = set_aside(results_dir / name)
                if moved:
                    print(f"--force: {name} set aside as {moved.name}")
    attempts = list((saved or {}).get("attempts", []))
    attempt = len(attempts) + 1
    os.environ["XLA_FLAGS"] = with_deterministic_xla(os.environ.get("XLA_FLAGS", ""))                # before JAX is imported (load_model below)
    gpu = gpu_name()
    attempts.append({"attempt": attempt, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "gpu": gpu, "xla_flags": os.environ["XLA_FLAGS"],
                     "git": git_commit(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "labels": [j.label for j in jobs], "workers": args.workers, "n_episodes": len(seeds)})
    E.atomic_write_json(protocol_path, {**protocol, "attempts": attempts})
    previous_gpus = sorted({a["gpu"] for a in attempts[:-1]} - {gpu})
    if previous_gpus:
        print(f"WARNING: earlier attempts of this folder ran on {previous_gpus}, this one on {gpu}: the bits of the physics may differ between GPU models (N4); keep one curve on one kind of GPU")

    stop = threading.Event()

    def on_signal(signum, _frame):
        if stop.is_set():                                                    # a second signal: no more patience
            kill_all_servers(2.0)
            os._exit(130)
        print(f"\nsignal {signum}: stopping (episodes in progress are dropped, the servers are killed, what is finished is kept) ...", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    enable_compilation_cache()
    from src.algos import sft as sft_mod
    from src.models import load_model
    print(f"loading the model ({model_cfg['name']}), XLA_FLAGS={os.environ['XLA_FLAGS']}", flush=True)
    base = load_model(model_cfg, seed=0)
    initial = base.state                                                     # the run's initial state: the policy as loaded

    def prepare(policy, job: Job) -> None:
        policy.set_state(initial)                                            # the weights of no other checkpoint leak in
        if job.checkpoint is not None:
            meta = sft_mod.load_checkpoint(policy, job.checkpoint)
            if int(meta.get("step", job.step)) != job.step:
                raise ValueError(f"{job.checkpoint} says it is step {meta.get('step')}")

    def on_complete(job: Job, summary: dict) -> None:
        try:
            write_curve(results_dir, curve_dir, title, name=run_dir.name)
        except Exception as e:  # noqa: BLE001  a plotting problem must not stop the evaluation
            print(f"[{job.label}] the curve could not be drawn: {e!r} (python -m scripts.eval_sft --run ... --plot-only redoes it)", flush=True)

    def say(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    video_dir = Path(args.video_dir).resolve() if args.video_dir else (output_dir / "videos" if args.save_videos else None)
    sweep = Sweep(output_dir=output_dir, jobs=jobs, seeds=seeds, replan=protocol["replan"], max_steps=protocol["max_steps"], workers=args.workers,
                  policy_factory=lambda: base.fork(0), prepare=prepare, server_factory=make_server_factory(task_config, args.env_set, output_dir / "logs"),
                  attempt=attempt, gpu=gpu, stop=stop, on_complete=on_complete, video_dir=video_dir, ready_timeout_s=args.ready_timeout, say=say)
    result = sweep.run()
    try:
        write_curve(results_dir, curve_dir, title, name=run_dir.name)
    except Exception as e:  # noqa: BLE001
        print(f"the curve could not be drawn: {e!r}", flush=True)
    print_table(results_dir)
    print(f"\ncomplete: {result.complete}; incomplete: {result.incomplete}; failed: {result.failed}")
    if result.failed:
        return 1
    if result.stopped and result.incomplete:
        print("stopped before the end: run the same command again, only the missing episodes are evaluated")
        return 0
    print(f"curve: {curve_dir / 'curve.png'}, {curve_dir / 'curve.pdf'}, {curve_dir / 'curve.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
