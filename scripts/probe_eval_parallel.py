"""Probe of the parallel evaluation (src/eval/sweep.py): how many checkpoints one GPU can evaluate at the same time, and whether the results stay bit-identical.

    python -m scripts.probe_eval_parallel --model pi05 --workers 1 2 --episodes 4
    python -m scripts.probe_eval_parallel --model pi0_fast --workers 1 2 --episodes 4 --model-set vlm.lora_rank=8
    python -m scripts.probe_eval_parallel --model pi05 --workers 1 --episodes 4 --no-resume-check --no-deterministic-xla      # what XLA's deterministic ops cost

Run it through jobs/job_probe_eval_parallel.sh. It needs no SFT run: the policy is the model's initial state (openpi's LoRA initialisation, seed 0), which is also what step 0
of a run is. For each worker count W it evaluates W IDENTICAL "checkpoints" (the same weights, the same episodes) at the same time, with the real code path of
scripts/eval_sft.py (one env server per checkpoint, a policy fork per worker), and reports:

  * cost: the median time of an episode with W workers running (the steady state, start-up and compilation excluded), hence the throughput in episodes per hour and the
    speed-up against W = 1; how much of that time is the model (plans x median plan time) and how much the simulator; the GPU memory and utilisation (nvidia-smi, every 5 s)
    at the peak; the wall time including the servers' start-up;
  * determinism, checked on the trajectories (`traj_sha`: every joint / gripper / peg-socket reading of every episode):
      - between the W replicas of one run: are the episodes of two checkpoints evaluated at the same time on the same GPU bit-identical?
      - between the runs with different W: does adding workers change a result?
      - after a CUT: the first W = 1 replica is copied with only its first episodes kept, as if the job had been killed, and the evaluation is resumed on a fresh server:
        how many of the episodes run after the cut differ from the uninterrupted ones? (docs/envs/isaaclab_env.md N1 says some may, slightly.)
    The verdict of each is printed and stored in the report (reports/probe_eval_parallel_<time>.json).

Every run writes in its OWN folder, <output-root>/_probe_eval_parallel_<model>_<time>_<job id>/ (two probes started in the same second once shared one folder and mixed their
episodes: the results of that first pair were worth nothing), under a lock; the report has the same tag.

What it cannot say: other GPU models (N4), longer evaluations than `--episodes`, other checkpoints than the initial state.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from scripts.eval_sft import DEFAULT_TASK, RECIPE, acquire_run_lock, task_settings, with_deterministic_xla
from scripts.sft import enable_compilation_cache, parse_assignments
from src.eval import episodes as E
from src.eval.sweep import Job, Sweep, gpu_name, kill_all_servers, make_server_factory
from src.models.interface import ROOT, apply_overrides, load_model_cfg


# -- GPU sampling ------------------------------------------------------------------------------------------------------------------------
class GpuSampler:
    """memory.used (MiB) and utilization.gpu (%) of the first GPU every `period_s`, from a thread."""

    def __init__(self, period_s: float = 5.0):
        self.period_s, self.memory, self.utilization = period_s, [], []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                     check=True, timeout=30).stdout.strip().splitlines()[0].split(",")
                self.memory.append(int(out[0]))
                self.utilization.append(int(out[1]))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.period_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict:
        return {"samples": len(self.memory), "memory_peak_mib": max(self.memory) if self.memory else None,
                "utilization_mean": round(statistics.mean(self.utilization), 1) if self.utilization else None}


# -- analysis (pure: tested on CPU) -----------------------------------------------------------------------------------------------------------
def episode_shas(records: Dict[int, dict]) -> Dict[int, str]:
    return {i: r["traj_sha"] for i, r in records.items()}


def differing_episodes(reference: Dict[int, str], other: Dict[int, str], only_from: int = 0) -> List[int]:
    """Episodes (>= only_from) present in both whose trajectories are not bit-identical."""
    return sorted(i for i in reference if i >= only_from and i in other and reference[i] != other[i])


def episode_walls(records_by_label: Dict[str, Dict[int, dict]]) -> List[float]:
    """Wall time of every episode of every replica except the first episode of each (it holds the compilation or the warm-up)."""
    return [r["wall_s"] for recs in records_by_label.values() for i, r in recs.items() if i > 0]


def model_seconds(records_by_label: Dict[str, Dict[int, dict]]) -> List[float]:
    """Seconds an episode spends in the model: plans x the median time of a plan (same episodes as `episode_walls`; records without the two numbers are left out).
    The rest of the wall time is the simulator, the transport and the image handling."""
    return [r["plans"] * r["plan_ms_median"] / 1000.0 for recs in records_by_label.values() for i, r in recs.items()
            if i > 0 and r.get("plans") is not None and r.get("plan_ms_median") is not None]


def probe_tag(model: str, stamp: str, job_id: Optional[str]) -> str:
    """The name of this probe's folder and report: the model, the time and the Slurm job (or the process) make it unique to ONE probe."""
    return f"{model}_{stamp}_{job_id or 'pid' + str(os.getpid())}"


def analyse(runs: Dict[int, Dict[str, Dict[int, dict]]], resume: Optional[dict] = None) -> dict:
    """`runs`: W -> label -> records by episode. Compares every replica of every run to the first replica of the first run (`reference`)."""
    first_w = min(runs)
    ref_label = sorted(runs[first_w])[0]
    reference = episode_shas(runs[first_w][ref_label])
    out: dict = {"reference": f"W={first_w} {ref_label}", "workers": {}}
    for w, by_label in sorted(runs.items()):
        walls = episode_walls(by_label)
        med = statistics.median(walls) if walls else None
        mismatches = {label: differing_episodes(reference, episode_shas(recs)) for label, recs in sorted(by_label.items())}
        model_s = model_seconds(by_label)
        out["workers"][str(w)] = {"replicas": len(by_label), "episode_wall_s_median": med, "episodes_per_hour": round(3600 * w / med, 1) if med else None,
                                  "model_s_per_episode_median": round(statistics.median(model_s), 1) if model_s else None,
                                  "identical_to_reference": all(not m for m in mismatches.values()), "differing_episodes": mismatches}
    base = out["workers"][str(first_w)]["episodes_per_hour"]
    for w in out["workers"].values():
        w["speedup_vs_first"] = round(w["episodes_per_hour"] / base, 2) if base and w["episodes_per_hour"] else None
    if resume is not None:
        cut = resume["cut_after"]
        differing = differing_episodes(reference, episode_shas(resume["records"]), only_from=cut)
        out["resume"] = {"cut_after": cut, "episodes_after_the_cut": len(reference) - cut, "differing_after_the_cut": differing,
                         "identical_after_the_cut": not differing}
    out["all_workers_identical"] = all(w["identical_to_reference"] for w in out["workers"].values())
    return out


# -- the runs -------------------------------------------------------------------------------------------------------------------------------
def load_records(output_dir: Path, label: str) -> Dict[int, dict]:
    return E.EpisodeLog(output_dir / "results" / f"{label}.jsonl").load()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="configs/model/<name>.yaml (pi05 or pi0_fast)")
    ap.add_argument("--model-set", action="append", metavar="KEY=VALUE", help="override a key of the model YAML: <component>.<key>")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2], help="worker counts to try, each running that many identical checkpoints at once (default 1 2)")
    ap.add_argument("--episodes", type=int, default=4, help="episodes per checkpoint (default 4; measured on an L40S: about 90 s each with one worker for pi0.5, 110 s for pi0-FAST)")
    ap.add_argument("--seed", type=int, default=E.DEFAULT_SEED)
    ap.add_argument("--replan", type=int, help="default: half the model's chunk")
    ap.add_argument("--task-config", default=DEFAULT_TASK)
    ap.add_argument("--no-resume-check", action="store_true", help="skip the cut-and-resume check")
    ap.add_argument("--no-deterministic-xla", action="store_true", help="leave XLA_FLAGS as they are (no --xla_gpu_deterministic_ops): measures what that flag costs; NOT an evaluation setting")
    ap.add_argument("--output-root", default="runs/sft/probing", help="the probe writes in <output-root>/_probe_eval_parallel_<model>_<time>_<job id>/")
    ap.add_argument("--ready-timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)
    if args.episodes < 2:
        raise SystemExit("--episodes must be at least 2 (the first episode of every replica carries the compilation and is left out of the timings)")

    cfg = apply_overrides(load_model_cfg(args.model), parse_assignments(args.model_set))
    task_config = Path(args.task_config) if Path(args.task_config).is_absolute() else ROOT / args.task_config
    task = task_settings(task_config, [])
    replan = int(args.replan) if args.replan else max(1, int(cfg["action_horizon"]) // 2)
    max_steps = int(task["max_steps_per_episode"])
    tag = probe_tag(args.model, time.strftime("%Y%m%d_%H%M%S"), os.environ.get("SLURM_JOB_ID"))
    root = ROOT / args.output_root / f"_probe_eval_parallel_{tag}"
    root.mkdir(parents=True, exist_ok=False)                         # one folder per probe: never share one with another
    lock = acquire_run_lock(root)                                    # noqa: F841  (held until the process ends)
    seeds = E.load_or_create_seeds(root, args.seed, args.episodes)

    if not args.no_deterministic_xla:
        os.environ["XLA_FLAGS"] = with_deterministic_xla(os.environ.get("XLA_FLAGS", ""))
    enable_compilation_cache()

    def on_signal(signum, _frame):                          # a probe is not resumable: stop the servers and leave
        print(f"signal {signum}: killing the env servers and leaving", flush=True)
        kill_all_servers(5.0)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    from src.models import load_model
    gpu = gpu_name()
    print(f"probe of the parallel evaluation: {args.model} on {gpu}, workers {args.workers}, {args.episodes} episodes, replan {replan}, XLA_FLAGS='{os.environ.get('XLA_FLAGS', '')}'", flush=True)
    base = load_model(cfg, seed=0)

    def say(msg: str) -> None:
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def sweep_for(output_dir: Path, jobs: List[Job], workers: int, attempt: int = 1) -> Sweep:
        return Sweep(output_dir=output_dir, jobs=jobs, seeds=seeds, replan=replan, max_steps=max_steps, workers=workers, policy_factory=lambda: base.fork(0),
                     prepare=lambda policy, job: None, server_factory=make_server_factory(task_config, [], output_dir / "logs"), attempt=attempt, gpu=gpu,
                     ready_timeout_s=args.ready_timeout, say=say)

    report: dict = {"model": args.model, "model_cfg": cfg, "gpu": gpu, "episodes": args.episodes, "replan": replan, "recipe": RECIPE, "xla_flags": os.environ.get("XLA_FLAGS", ""),
                    "deterministic_xla": not args.no_deterministic_xla, "runs": {}}
    runs: Dict[int, Dict[str, Dict[int, dict]]] = {}
    failures: List[str] = []
    for w in args.workers:
        out = root / f"w{w}"
        jobs = [Job(label=f"r{j}", step=0) for j in range(w)]
        say(f"==== {w} worker(s): {w} identical checkpoints at the same time")
        t0 = time.monotonic()
        with GpuSampler() as gpus:
            result = sweep_for(out, jobs, w).run()
        wall = time.monotonic() - t0
        report["runs"][str(w)] = {"wall_s_with_startup": round(wall, 1), "gpu": gpus.summary(), "failed": result.failed, "incomplete": result.incomplete}
        if not result.ok:
            failures.append(f"W={w}: failed {result.failed}, incomplete {result.incomplete}")
            say(f"W={w} did not finish ({result.failed or result.incomplete}): the next counts are not tried (they would only need more memory)")
            break
        runs[w] = {j.label: load_records(out, j.label) for j in jobs}

    resume = None
    if runs and not args.no_resume_check:
        w0 = min(runs)
        cut = max(1, args.episodes // 2)
        out = root / "resume"
        (out / "results").mkdir(parents=True)
        source = (root / f"w{w0}" / "results" / "r0.jsonl").read_text().splitlines(keepends=True)
        (out / "results" / "r0.jsonl").write_text("".join(source[:cut]))
        say(f"==== resume check: the first {cut} episodes of W={w0} r0 are kept, the rest is evaluated again on a fresh server")
        result = sweep_for(out, [Job(label="r0", step=0)], 1, attempt=2).run()
        if result.ok:
            resume = {"cut_after": cut, "records": load_records(out, "r0")}
        else:
            failures.append(f"resume check: failed {result.failed}, incomplete {result.incomplete}")

    if runs:
        analysis = analyse(runs, resume)
        report["analysis"] = analysis
        print("\n=== results")
        for w, a in analysis["workers"].items():
            run = report["runs"][w]
            print(f"W={w}: episode {a['episode_wall_s_median']:.1f} s (median, of which {a['model_s_per_episode_median']} s in the model), {a['episodes_per_hour']} episodes/hour, "
                  f"x{a['speedup_vs_first']} vs W={min(runs)}; "
                  f"GPU memory peak {run['gpu']['memory_peak_mib']} MiB, utilisation {run['gpu']['utilization_mean']} %, wall with start-up {run['wall_s_with_startup']} s; "
                  f"bit-identical to the reference: {a['identical_to_reference']}" + ("" if a["identical_to_reference"] else f" (differing episodes {a['differing_episodes']})"))
        if "resume" in analysis:
            r = analysis["resume"]
            print(f"resume after {r['cut_after']} episodes: {len(r['differing_after_the_cut'])} of {r['episodes_after_the_cut']} later episodes differ from the uninterrupted ones "
                  f"({r['differing_after_the_cut']})")
        print(f"all worker counts bit-identical: {analysis['all_workers_identical']}")
    report["failures"] = failures
    path = ROOT / "reports" / f"probe_eval_parallel_{tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str))
    print(f"report: {path}\ndata: {root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
