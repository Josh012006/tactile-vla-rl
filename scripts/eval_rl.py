"""Success-rate curve of an RL run, evaluated again the way the SFT checkpoints are (scripts/eval_sft.py): every round of the run (or the rounds asked for), on the same fixed episodes of the FORGE environment.

    python -m scripts.eval_rl --run runs/rl/peg_insert/peg_insert_pi05_best_of_n                    # every round whose critic (and VLA) were kept, 100 episodes each
    python -m scripts.eval_rl --run <run> --dry-run                                                  # what would be evaluated and what is already done: no GPU, no model, no server
    python -m scripts.eval_rl --run <run> --rounds 0 10 20                                           # only these rounds (0 = after the warm-up: the SFT checkpoint and the first critic)
    python -m scripts.eval_rl --run <run> --rounds 0 2 --selection vla --output-dir <run>/eval_rl_vla  # the VLA ALONE of these rounds (no critic): its first candidate, to tell what the distillation did
    python -m scripts.eval_rl --run <run> --plot-only --title "PA-RL B"                              # rebuild curve.json / .png / .pdf from the results on disk: no GPU, no server

Run it through jobs/job_eval_rl.sh (an env server is started for each round: Isaac Sim on the same GPU as the policy).

What is evaluated. `--run` is a run folder of scripts/rl.py (runs/rl/peg_insert/<name>, or a probe under runs/rl/probing/). Round r is the pair (VLA, critic) the run had at the end of round r:
the critic is critics/round_<r>.msgpack (every round keeps one); the VLA is vlas/round_<r>/ for PA-RL (r >= 1: the trainable parameters after that round's distillation) and the SFT checkpoint the run
started from for Best-of-N and for round 0 (the VLA never changed then). The policy is the one of the run's own evaluations (src/algos/rl_rollout.py): the VLA's k candidates at each chunk, the optimisation
of the method (variant B: its local ascent too) and the critic's ARGMAX over the optimised set, one chunk executed per plan. The model is the run's (its model YAML with its overrides, its LoRA, the SFT run's seed
for the LoRA initialisation), the critic's network is built from the run's own settings.
`--selection vla` evaluates the VLA alone instead (the first of its k candidates, no optimisation, no critic: the policy of the warm-up): with a PA-RL run it separates what the distillation did to the VLA
from what the critic's selection adds. The selection is part of the protocol of the output folder, so such an evaluation needs its own --output-dir (its episodes are the same fixed ones: the paired
comparisons with the evaluation with the critic stay valid on the first batch of --envs-per-server episodes).
The normalisation statistics are the run's too: the file its model YAML names, which the SFT copied into every checkpoint (assets/), including the ones this run writes (a PA-RL round's VLA): the two are compared
byte for byte before anything starts, and a difference is refused (scripts/eval_sft.py check_norm_stats). The RL never computes statistics of its own.

The episodes, the determinism, the resume, --workers and the output are those of scripts/eval_sft.py (read its docstring; docs/eval/sft_eval.md): one FRESH env server per round with `--envs-per-server` K
environments (default 100) that play their episodes together from the real reset seeded `--seed` (default 42), so the starts and the episode seeds are those of every SFT evaluation on the same GPU and of the
evaluations made during the run; XLA's deterministic GPU ops are switched on (XLA_FLAGS --xla_gpu_deterministic_ops=true, added here); the cameras are not reproducible, so an evaluation is comparable and not
bit-reproducible (same starts, same episode seeds, a standard error). One difference: the model is called as the run calls it, `rows_per_sample_call` environments (of the run's settings) x k candidates per call, every
call padded to that many rows (one compilation); that and the settings of the optimisation are part of the protocol of the output folder, like the model and the task (a folder cannot mix them).

Output (--output-dir, default <run>/eval_rl: NOT the run's own eval/ folder, which holds the evaluations made during the run, a different kind of result): eval_protocol.json, episode_seeds.json, results/<round>.jsonl and
.json (the format of the SFT evaluations: scripts/compare_eval_paired.py reads them), logs/server_<round>.log, videos/<round>/ with --save-videos; curve.json / curve.png / curve.pdf (x = the collection round).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from scripts.eval_sft import (DEFAULT_ENVS_PER_SERVER, DEFAULT_TASK, resolve_task_config, run_roots, acquire_run_lock, check_norm_stats, file_sha, git_commit, norm_stats_path, print_table, progress_of, set_aside, task_settings,
                              with_deterministic_xla)
from scripts.sft import enable_compilation_cache, parse_assignments
from src.algos import rl_config
from src.eval import episodes as E
from src.eval.curve import write_curve
from src.eval.sweep import Job, Sweep, gpu_name, kill_all_servers, make_server_factory
from src.models.interface import ROOT

RECIPE = "rl-batched-v1"                       # K environments per server, one reset per round, noise per (episode, chunk), candidates by calls of rows_per_sample_call environments, fresh server per round
X_LABEL = "collection round (0 = after the warm-up)"
# What of the run's RL settings shapes a result (the candidates, their optimisation, the critic's network): fixed by the output folder, like the model and the task.
PROTOCOL_KEYS = ("variant", "num_candidates", "keep_top", "local_steps", "local_lr", "eval_selection", "softmax_temperature", "rows_per_sample_call", "critic_cameras", "critic_image_size",
                 "critic_hidden", "critic_use_tactile", "critic_use_force", "critic_force_mode")


# -- the run and its rounds -----------------------------------------------------------------------------------------------------------------------
def find_rl_run_dir(name_or_path: str) -> Path:
    """A run given as a path (absolute, or relative to where the command runs or to the repository) or as its folder name under runs/rl/<task>/ (peg_insert, ..., probing)."""
    for candidate in (Path(name_or_path), ROOT / name_or_path, *(folder / name_or_path for folder in run_roots("rl", ROOT))):
        if (candidate / "config.json").exists():
            return candidate.resolve()
    raise SystemExit(f"no RL run {name_or_path!r} (looked for config.json at that path and under runs/rl/*/)")


def vla_checkpoint(run_dir: Path, cfg: rl_config.RLConfig, start_checkpoint: str, r: int) -> Path:
    """The VLA of round r: the PA-RL run's own parameters after the round's distillation, otherwise the SFT checkpoint the run started from (Best-of-N never changes the VLA, nor does the warm-up)."""
    return Path(run_dir) / "vlas" / f"round_{r:03d}" if cfg.distills and r >= 1 else Path(start_checkpoint)


def critic_file(run_dir: Path, r: int) -> Path:
    return Path(run_dir) / "critics" / f"round_{r:03d}.msgpack"


def discover_rounds(run_dir: Path, cfg: rl_config.RLConfig, start_checkpoint: str) -> Dict[int, Path]:
    """round -> the folder of its VLA, for the rounds whose critic AND VLA are complete and that the run finished (resume/state.json: a stage cut before it was written is not one)."""
    state = Path(run_dir) / "resume" / "state.json"
    last = int(json.loads(state.read_text())["stage"]) if state.exists() else None
    found: Dict[int, Path] = {}
    folder = Path(run_dir) / "critics"
    if folder.is_dir():
        for path in folder.glob("round_*.msgpack"):
            r = int(path.stem.split("_")[1])
            vla = vla_checkpoint(run_dir, cfg, start_checkpoint, r)
            if (last is None or r <= last) and (vla / "meta.json").exists():
                found[r] = vla
    return dict(sorted(found.items()))


def select_jobs(found: Dict[int, Path], rounds: Optional[List[int]]) -> List[Job]:
    """The rounds to evaluate, by round (the job's `checkpoint` is the round's VLA folder). Without --rounds: every round found. With --rounds: exactly those (an unknown one is an error)."""
    wanted = sorted(set(rounds)) if rounds else list(found)
    missing = [r for r in wanted if r not in found]
    if missing:
        raise SystemExit(f"--rounds {missing} not found (a round needs its critic in critics/ and its VLA, and must be finished): found {sorted(found)}")
    return [Job(label=str(r), step=r, checkpoint=found[r]) for r in wanted]


# -- protocol -----------------------------------------------------------------------------------------------------------------------------------
def rl_block(cfg: rl_config.RLConfig, selection: Optional[str] = None) -> dict:
    """The RL settings that shape a result. `selection`: what the evaluation executes ('argmax' of the critic, or 'vla' alone), the run's `eval_selection` by default."""
    import dataclasses
    raw = dataclasses.asdict(cfg)
    return {**{k: raw[k] for k in PROTOCOL_KEYS if k in raw}, "variant": cfg.variant, "eval_selection": selection or cfg.eval_selection}


def build_protocol(args, run_dir: Path, saved: dict, cfg: rl_config.RLConfig, task: dict) -> dict:
    model_cfg = saved["model_cfg"]
    horizon = int(model_cfg["action_horizon"])
    replan = int(args.replan) if args.replan else max(1, min(cfg.replan_steps or horizon // 2, horizon))      # as RLRun
    if not 1 <= replan <= horizon:
        raise SystemExit(f"--replan {replan}: between 1 and the model's chunk length ({horizon})")
    stats = norm_stats_path(model_cfg)
    return {"recipe": RECIPE, "master_seed": int(args.seed), "replan": replan, "max_steps": int(task["max_steps_per_episode"]), "num_envs": int(args.envs_per_server), "model_batch": None,
            "model_cfg": model_cfg, "norm_stats_sha": file_sha(stats) if stats else None, "task": task, "run": str(run_dir), "rl": rl_block(cfg, getattr(args, "selection", None))}


def check_rl_block(saved: Optional[dict], now: dict) -> None:
    """Refuse an output folder whose earlier results were made with other candidates / optimisation / critic settings (E.check_protocol covers the rest)."""
    if saved is None or saved.get("rl") is None:
        return
    diff = [f"{k}: {saved['rl'].get(k)!r} in the folder, {now['rl'].get(k)!r} now" for k in now["rl"] if json.loads(json.dumps(saved["rl"].get(k))) != json.loads(json.dumps(now["rl"][k]))]
    if diff:
        raise ValueError("this --output-dir holds results made with other RL settings (use another --output-dir):\n  " + "\n  ".join(diff))


def task_differences(saved_task: dict, task: dict) -> List[str]:
    """What the task of this evaluation has that the run's own did not (the run's servers have envs_per_batch environments, the evaluation's K: `num_envs` is expected to differ)."""
    keys = (set(saved_task) | set(task)) - {"num_envs"}
    return [f"{k}: {saved_task.get(k)!r} in the run, {task.get(k)!r} now" for k in sorted(keys) if json.loads(json.dumps(saved_task.get(k), default=str)) != json.loads(json.dumps(task.get(k), default=str))]


def load_critic(path: Path, net, horizon: int):
    """The parameters of a saved critic (`rl_run.critic_bytes`): the ones the run's evaluations use (not the delayed copy)."""
    import flax.serialization
    import jax
    from src.algos import rl_critic as rc
    template = rc.init_critic(net, jax.random.key(0), horizon=horizon)
    return flax.serialization.from_bytes({"params": template.params, "target_params": template.target_params}, Path(path).read_bytes())["params"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, metavar="RUN", help="an RL run: its path, or its folder name under runs/rl/<task>/ or runs/rl/probing/")
    ap.add_argument("--n-episodes", type=int, default=None, help="episodes per round (default: the run's own `eval_episodes`, 100); may grow between runs")
    ap.add_argument("--seed", type=int, default=E.DEFAULT_SEED, help="master seed of the fixed episode list (default 42, the SFT evaluations' and the run's)")
    ap.add_argument("--rounds", type=int, nargs="+", metavar="R", help="only these rounds (0 = after the warm-up); default: every finished round whose critic and VLA were kept")
    ap.add_argument("--output-dir", metavar="DIR", help="where the evaluation data goes (default: <run>/eval_rl)")
    ap.add_argument("--curve-dir", metavar="DIR", help="where curve.json / curve.png / curve.pdf go (default: the output folder)")
    ap.add_argument("--envs-per-server", type=int, default=DEFAULT_ENVS_PER_SERVER, metavar="K",
                    help=f"environments in each server, playing together (default {DEFAULT_ENVS_PER_SERVER}); at least --n-episodes keeps a round in ONE batch. Fixed by the folder's protocol")
    ap.add_argument("--workers", type=int, default=1, help="rounds evaluated at the same time (default 1, as for the SFT): one thread, one env server and one policy each")
    ap.add_argument("--replan", type=int, help="steps executed per chunk before the model is called again (default: the run's)")
    ap.add_argument("--selection", choices=("argmax", "vla"), default=None,
                    help="what is executed: 'argmax' = the critic's argmax over the optimised candidates (default: the run's eval_selection); 'vla' = the VLA alone (its first candidate, no critic). "
                         "Part of the folder's protocol: use another --output-dir")
    ap.add_argument("--task-config", default=None, help=f"env server YAML (default: the one the RL run recorded, then {DEFAULT_TASK}, the PegInsert one that every run without a `task_config` used); the run's own `env_set` is applied on top, then --env-set")
    ap.add_argument("--env-set", action="append", default=[], metavar="KEY=VALUE", help="override a key of the task YAML for the servers (not num_envs / server_port)")
    ap.add_argument("--save-videos", action="store_true", help="videos of the first --video-episodes episodes of each round in <output-dir>/videos/<round>/")
    ap.add_argument("--video-episodes", type=int, default=5, help="how many episodes of a round are filmed with --save-videos (default 5)")
    ap.add_argument("--video-dir", metavar="DIR", help="parent folder of the videos (implies --save-videos)")
    ap.add_argument("--title", help="title of the plot (small font, wrapped); default: '<run name>: success rate vs collection round (evaluated again)'")
    ap.add_argument("--plot-only", action="store_true", help="only rebuild curve.json / .png / .pdf from the results on disk")
    ap.add_argument("--force", action="store_true", help="set aside the results of the selected rounds (.prev1, ...) and evaluate them from scratch")
    ap.add_argument("--dry-run", action="store_true", help="print what would be evaluated and what is already done; load nothing")
    ap.add_argument("--ready-timeout", type=float, default=1800.0, help="seconds an env server may take to start (default 1800)")
    args = ap.parse_args(argv)

    run_dir = find_rl_run_dir(args.run)
    saved = json.loads((run_dir / "config.json").read_text())
    cfg = rl_config.config_from_dict(saved["rl"])
    model_cfg = saved["model_cfg"]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "eval_rl"
    curve_dir = Path(args.curve_dir).resolve() if args.curve_dir else output_dir
    results_dir = output_dir / "results"
    title = args.title or f"{run_dir.name}: success rate vs collection round (evaluated again, deterministic)"

    if args.plot_only:
        entries = write_curve(results_dir, curve_dir, title, name=run_dir.name, x_label=X_LABEL)
        if not entries:
            raise SystemExit(f"no complete result in {results_dir}")
        print_table(results_dir)
        print(f"curve: {curve_dir / 'curve.png'}, {curve_dir / 'curve.pdf'}, {curve_dir / 'curve.json'}")
        return 0

    selection = args.selection or cfg.eval_selection
    if selection == "softmax":
        raise SystemExit("eval_selection is softmax: an evaluation takes the argmax (or --selection vla)")
    found = discover_rounds(run_dir, cfg, saved["start"]["checkpoint"])
    jobs = select_jobs(found, args.rounds)
    if not jobs:
        raise SystemExit(f"{run_dir} has no finished round with its critic yet (critics/round_<r>.msgpack)")
    n_episodes = args.n_episodes or cfg.eval_episodes
    if args.envs_per_server < 1:
        raise SystemExit("--envs-per-server must be at least 1")
    task_config = resolve_task_config(args.task_config, saved.get("task_config"))
    task = task_settings(task_config, list(saved.get("env_set") or []) + list(args.env_set), args.envs_per_server)
    protocol = build_protocol(args, run_dir, saved, cfg, task)
    norm = check_norm_stats(model_cfg, jobs)
    protocol_path = output_dir / E.PROTOCOL_FILE
    previous = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    try:
        if previous is not None:
            E.check_protocol(previous, protocol)
            check_rl_block(previous, protocol)
        seeds = E.make_seeds(args.seed, n_episodes) if args.dry_run else E.load_or_create_seeds(output_dir, args.seed, n_episodes)
    except ValueError as e:
        raise SystemExit(str(e))

    done = progress_of(output_dir, jobs, seeds)
    print(f"run {run_dir} ({cfg.variant}), started from the SFT checkpoint {saved['start']['checkpoint']}")
    print(f"model {model_cfg['name']} ({model_cfg['family']}), chunk {model_cfg['action_horizon']}, replan {protocol['replan']}, episodes of at most {protocol['max_steps']} steps; {cfg.num_candidates} candidates per chunk "
          f"in calls of {cfg.rows_per_sample_call} environments, keep {cfg.keep_top}, {cfg.local_steps} local steps, '{selection}'" + (" (the VLA alone: its first candidate, no critic)" if selection == "vla" else ""))
    print(f"{len(seeds)} episodes per round, master seed {args.seed}, {args.envs_per_server} environments per server, {args.workers} worker(s); data in {output_dir}, curve in {curve_dir}")
    differences = task_differences(saved.get("task") or {}, task)
    if differences:
        print("WARNING: the task of this evaluation is not the one of the run (the run's evaluations and this one would not be on the same physics):\n  " + "\n  ".join(differences))
    if len(seeds) > args.envs_per_server:
        print(f"WARNING: {len(seeds)} episodes in servers of {args.envs_per_server} environments: batches after the first start after other episodes (small history dependence, "
              f"docs/envs/isaaclab_env.md N1); --envs-per-server {len(seeds)} or more keeps one batch (set it from the first run on: it is fixed by the protocol)")
    if norm["source"] is None:
        print("normalisation statistics: the base checkpoint's own (the model YAML has no norm_stats block), as in the SFT")
    else:
        print(f"normalisation statistics: {norm['source']} (sha256 {norm['sha256'][:12]}), identical to the copy saved in the {len(norm['verified'])} VLA checkpoint(s) to evaluate, as in the SFT")
    for j in jobs:
        try:
            where = str(j.checkpoint.relative_to(run_dir))
        except ValueError:
            where = f"the SFT start {j.checkpoint}"
        print(f"  round {j.step:>3}: {done[j.label]:>4}/{len(seeds)} episodes done   (VLA: {where}; critic: {critic_file(run_dir, j.step).relative_to(run_dir)})")
    if args.dry_run:
        return 0

    lock = acquire_run_lock(output_dir)                                      # noqa: F841  (held until the process ends)
    if args.force:
        for j in jobs:
            for name in (f"{j.label}.jsonl", f"{j.label}.json"):
                moved = set_aside(results_dir / name)
                if moved:
                    print(f"--force: {name} set aside as {moved.name}")
    attempts = list((previous or {}).get("attempts", []))
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
    from src.algos import rl_critic as rc
    from src.algos import sft as sft_mod
    from src.algos.rl_action_opt import ActionOptimiser
    from src.algos.rl_rollout import OptimisingPolicy, RLActor
    from src.models import load_model
    print(f"loading the model ({model_cfg['name']}), XLA_FLAGS={os.environ['XLA_FLAGS']}", flush=True)
    base = load_model(model_cfg, seed=int(saved["sft_seed"]))                # the LoRA initialisation of the SFT run the RL run started from, as scripts/rl.py
    initial = base.state
    horizon = int(model_cfg["action_horizon"])
    net = rc.CriticNet.from_config(cfg)
    optimiser = ActionOptimiser.for_critic(net, cfg)
    start_step = int(saved["start"]["step"])

    def make_policy():
        return OptimisingPolicy(RLActor(base.fork(0), cfg, optimiser), None, selection)

    def prepare(policy, job: Job) -> None:
        vla = policy.actor.policy
        vla.set_state(initial)                                               # the weights of no other round leak in
        meta = sft_mod.load_checkpoint(vla, job.checkpoint)
        distilled = cfg.distills and job.step >= 1
        if distilled and int(meta.get("round", -1)) != job.step:
            raise ValueError(f"{job.checkpoint} says it is round {meta.get('round')}, not {job.step}")
        if not distilled and int(meta.get("step", start_step)) != start_step:
            raise ValueError(f"{job.checkpoint} says it is step {meta.get('step')}, the run started from step {start_step}")
        policy.params = load_critic(critic_file(run_dir, job.step), net, horizon)
        policy.logs = []

    def on_complete(job: Job, summary: dict) -> None:
        try:
            write_curve(results_dir, curve_dir, title, name=run_dir.name, x_label=X_LABEL)
        except Exception as e:  # noqa: BLE001  a plotting problem must not stop the evaluation
            print(f"[{job.label}] the curve could not be drawn: {e!r} (python -m scripts.eval_rl --run ... --plot-only redoes it)", flush=True)

    def say(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    video_dir = Path(args.video_dir).resolve() if args.video_dir else (output_dir / "videos" if args.save_videos else None)
    sweep = Sweep(output_dir=output_dir, jobs=jobs, seeds=seeds, replan=protocol["replan"], max_steps=protocol["max_steps"], workers=args.workers,
                  policy_factory=make_policy, prepare=prepare,
                  server_factory=make_server_factory(task_config, list(saved.get("env_set") or []) + list(args.env_set), output_dir / "logs", num_envs=args.envs_per_server),
                  attempt=attempt, gpu=gpu, stop=stop, on_complete=on_complete, video_dir=video_dir, ready_timeout_s=args.ready_timeout, say=say,
                  master_seed=args.seed, envs_per_server=args.envs_per_server, video_episodes=args.video_episodes)
    result = sweep.run()
    try:
        write_curve(results_dir, curve_dir, title, name=run_dir.name, x_label=X_LABEL)
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
