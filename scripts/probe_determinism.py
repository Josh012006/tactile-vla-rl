"""Where does the non-reproducibility of an evaluation come from: the model, or the simulator?

    python -m scripts.probe_determinism --part policy --model pi05 --processes 2       # the model alone, no Isaac: a few minutes
    python -m scripts.probe_determinism --part policy --model pi0_fast --processes 2
    python -m scripts.probe_determinism --part env --servers 3                          # the simulator alone, fixed actions: about 15 minutes

Run them through jobs/job_probe_determinism.sh. Two evaluations of the same checkpoint with the same seeds, one worker, a fresh server and the same XLA flags gave different trajectories
(eval_sha 1d6af170... and aba0a88d... for pi0.5, 12 Nov. 2026), although the first observation of each episode was bit-identical: so either the model's actions differ, or the physics
differs over 150 steps (docs/envs/isaaclab_env.md G1 was measured over 20 steps only). Each part isolates one:

  --part policy   (no simulator) The same observation and seed through the real model, several ways, comparing the bits of the action chunk (sha256):
                    repeat   the same policy object, re-seeded, called again;
                    chain    the same sequence of calls WITHOUT re-seeding (the way an episode calls it), done twice;
                    fork     a `fork` of the policy (what the sweep gives each worker);
                    threads  several forks called at the same time from several threads;
                    process  `--processes N`: N more fresh Python processes (a new compilation each): the reference chunk of each.
                  All identical: the model is bit-reproducible, so its inputs (images, state) are what differ between evaluations. Some different: the model is not, and which of the ways differ says where to look.
  --part env      (no model) The same open-loop sequence of 150 joint targets (the first observation's joint positions plus a fixed sinusoid: the arm moves, the peg stays in the
                  gripper, as in an episode) played on `--servers` FRESH servers, each one twice in a row (a second reset of the same seed in the same server). Per-step readings of every
                  trace (joints, gripper, peg-socket offset, force) are compared with the first: the first step that differs, and how far apart they are at 1, 5, 10, 25, 50, 100, 150 steps.
                  Identical across fresh servers: the physics is reproducible over 150 steps. Different: the physics is not, and the first step says when (an early step: the contact
                  solver; growing slowly from a late step: chaos amplifying float noise). The two CAMERA IMAGES of the first play of every server are compared the same way (first differing
                  step, fraction of pixels that differ, largest difference): the model sees the images, so images that differ between fresh servers change its actions (docs/envs/isaaclab_env.md N3).
                  A second play in the same server starts from a (reproducibly) different state: N1.

The verdict is printed and stored in reports/probe_determinism_<part>_<model or env>_<time>_<job id>.json; the data (server logs) in runs/sft/probing/_probe_determinism_<same tag>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from scripts.eval_sft import DEFAULT_TASK, task_settings, with_deterministic_xla
from scripts.probe_eval_parallel import probe_tag
from scripts.sft import enable_compilation_cache, parse_assignments
from src.envs import protocol as P
from src.eval.episodes import HASHED_KEYS
from src.eval.sweep import gpu_name, kill_all_servers, make_server_factory
from src.models.interface import ROOT, apply_overrides, load_model_cfg

TRACE_KEYS = HASHED_KEYS + (P.FT_FORCE,)         # what every step of a trace holds (environment 0): no image
IMAGE_KEYS = (P.EXTERIOR_IMAGE, P.WRIST_IMAGE)   # compared separately (`compare_images`): they are what the model sees
CHECK_STEPS = (1, 5, 10, 25, 50, 100, 150)
PROMPT = "insert the peg"


# -- the model alone ----------------------------------------------------------------------------------------------------------------------------
def synthetic_obs(seed: int = 0) -> dict:
    """An observation in the env's format (one environment): fixed random images, joint positions of a Franka-like pose, a closed gripper."""
    rng = np.random.default_rng(seed)
    return {P.EXTERIOR_IMAGE: rng.integers(0, 255, (1, 224, 224, 3), dtype=np.uint8), P.WRIST_IMAGE: rng.integers(0, 255, (1, 224, 224, 3), dtype=np.uint8),
            P.JOINT_POSITION: np.array([[0.0, -0.5, 0.0, -2.2, 0.0, 1.7, 0.8]], np.float32), P.GRIPPER_POSITION: np.full((1, 1), 0.9, np.float32)}


def chunk_sha(plan) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(plan.raw)).tobytes()).hexdigest()[:16]


def policy_checks(base, obs: dict, prompt: str, seed: int, repeats: int = 3, threads: int = 2) -> dict:
    """The ways of calling the model listed in the module's doc, on one observation. Every value is a sha256 (16 hex) of the raw chunk(s); `*_identical` says whether they all agree
    with the reference (the first seeded call of the first way)."""
    base.seed(seed)
    reference = chunk_sha(base.plan(obs, prompt))
    repeat = []
    for _ in range(repeats):
        base.seed(seed)
        repeat.append(chunk_sha(base.plan(obs, prompt)))
    chains = []
    for _ in range(2):                                                    # the same sequence of calls twice, never re-seeded in between
        base.seed(seed)
        chains.append([chunk_sha(base.plan(obs, prompt)) for _ in range(repeats)])
    fork = base.fork(seed)
    fork_sha = chunk_sha(fork.plan(obs, prompt))
    out: Dict[str, List[str]] = {}

    def worker(k: int) -> None:
        try:
            f = base.fork(seed)
            shas = []
            for _ in range(repeats):
                f.seed(seed)
                shas.append(chunk_sha(f.plan(obs, prompt)))
            out[str(k)] = shas
        except Exception as e:  # noqa: BLE001  a failing thread is a result, not a crash of the probe
            out[str(k)] = [f"ERROR {e!r}"]

    pool = [threading.Thread(target=worker, args=(k,)) for k in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    threaded = [s for k in sorted(out) for s in out[k]]
    return {"reference": reference, "repeat": repeat, "repeat_identical": all(s == reference for s in repeat),
            "chain": chains, "chain_identical": chains[0] == chains[1],
            "fork": fork_sha, "fork_identical": fork_sha == reference,
            "threads": threaded, "threads_identical": len(threaded) == threads * repeats and all(s == reference for s in threaded)}


def other_processes(model: str, model_sets: List[str], seed: int, n: int) -> List[Optional[str]]:
    """The reference chunk sha computed by `n` fresh Python processes (a new compilation each), one after the other. None when a child fails."""
    shas: List[Optional[str]] = []
    for _ in range(n):
        command = [sys.executable, "-m", "scripts.probe_determinism", "--part", "policy", "--model", model, "--seed", str(seed), "--child"]
        for item in model_sets:
            command += ["--model-set", item]
        done = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        line = [ln for ln in done.stdout.splitlines() if ln.startswith("REFERENCE_SHA ")]
        shas.append(line[-1].split()[1] if line else None)
        if not line:
            print(f"a child process failed (exit {done.returncode}):\n{done.stderr[-1500:]}", flush=True)
    return shas


# -- the simulator alone ------------------------------------------------------------------------------------------------------------------------
def scripted_targets(q0: np.ndarray, steps: int, amplitude: float = 0.15, period: int = 60) -> np.ndarray:
    """(steps, 7) joint targets: the start pose plus a fixed sinusoid with a phase per joint. Open loop: the same for every replay, whatever the simulator does."""
    t = np.arange(steps)[:, None]
    phase = np.linspace(0.0, np.pi, 7)[None]
    return (np.asarray(q0, np.float64)[None] + amplitude * np.sin(2 * np.pi * t / period + phase)).astype(np.float32)


def reading(obs: dict) -> np.ndarray:
    """The readings of environment 0 that a trace keeps, as one float64 vector."""
    return np.concatenate([np.asarray(obs[k], np.float64)[0].ravel() for k in TRACE_KEYS if k in obs])


def run_trace(env, seed: int, steps: int, amplitude: float = 0.15, period: int = 60, images: Optional[Dict[str, list]] = None) -> np.ndarray:
    """reset(seed, round 0) then `steps` steps of the scripted targets; returns (steps + 1, D): the reading after the reset and after every step (shorter if the episode ends).
    `images`, a dict, is filled with the list of environment 0's camera frames of every reading, by image key."""
    res = env.reset(seed=seed, round=0)
    targets = scripted_targets(np.asarray(res.obs[P.JOINT_POSITION][0][:7]), steps, amplitude, period)
    rows = [reading(res.obs)]

    def keep(obs: dict) -> None:
        if images is not None:
            for key in IMAGE_KEYS:
                if key in obs:
                    images.setdefault(key, []).append(np.array(np.asarray(obs[key])[0], copy=True))

    keep(res.obs)
    for t in range(steps):
        action = np.zeros((env.num_envs, P.ACTION_DIM), np.float32)
        action[:, :7] = targets[t]
        action[:, 7] = 1.0                                                # ignored while the task YAML has gripper_can_open: false
        res = env.step(action, auto_reset=False)
        rows.append(reading(res.obs))
        keep(res.obs)
        if res.all_done:
            break
    return np.stack(rows)


def compare_traces(traces: Dict[str, np.ndarray], reference: str, check_steps=CHECK_STEPS) -> dict:
    """Every trace against `reference`: its length, whether it is bit-identical, the first step that differs and the largest absolute difference at `check_steps`."""
    ref = traces[reference]
    out: dict = {"reference": reference, "traces": {}}
    for name, trace in traces.items():
        n = min(len(ref), len(trace))
        same = np.array([np.array_equal(ref[i], trace[i]) for i in range(n)])
        first = None if same.all() else int(np.argmax(~same))
        out["traces"][name] = {"length": int(len(trace)), "identical": bool(same.all() and len(trace) == len(ref)), "first_differing_step": first,
                               "max_abs_diff_at": {str(s): float(np.abs(ref[s] - trace[s]).max()) for s in check_steps if s < n},
                               "max_abs_diff_final": float(np.abs(ref[n - 1] - trace[n - 1]).max())}
    return out


def compare_images(reference: Dict[str, list], other: Dict[str, list], check_steps=CHECK_STEPS) -> dict:
    """Per camera: whether the frames are bit-identical, the first frame that differs, and at `check_steps` the fraction of pixel values that differ and the largest difference (0-255)."""
    out: dict = {}
    for key, ref in reference.items():
        frames = other.get(key, [])
        n = min(len(ref), len(frames))
        differs = [bool(not np.array_equal(ref[i], frames[i])) for i in range(n)]
        out[key] = {"frames": n, "identical": not any(differs) and len(ref) == len(frames), "first_differing_step": differs.index(True) if any(differs) else None,
                    "differing_fraction_at": {str(s): float(np.mean(ref[s] != frames[s])) for s in check_steps if s < n},
                    "max_abs_diff_at": {str(s): int(np.abs(ref[s].astype(np.int16) - frames[s].astype(np.int16)).max()) for s in check_steps if s < n}}
    return out


def verdict_env(traces: Dict[str, np.ndarray]) -> dict:
    """Fresh server against fresh server (`s<i>.r0`), and a replay in the same server against the first play there (`s<i>.r1` against `s<i>.r0`)."""
    names = sorted(traces)
    firsts = [n for n in names if n.endswith(".r0")]
    fresh = compare_traces({n: traces[n] for n in firsts}, firsts[0]) if len(firsts) > 1 else None
    replays = {}
    for n in names:
        if n.endswith(".r1") and n[:-3] + ".r0" in traces:
            replays[n] = compare_traces({n[:-3] + ".r0": traces[n[:-3] + ".r0"], n: traces[n]}, n[:-3] + ".r0")["traces"][n]
    return {"fresh_servers": fresh, "same_server_replays": replays,
            "fresh_servers_identical": None if fresh is None else all(t["identical"] for t in fresh["traces"].values()),
            "replays_identical": all(t["identical"] for t in replays.values()) if replays else None}


# -- the script ---------------------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", required=True, choices=["policy", "env"])
    ap.add_argument("--model", help="configs/model/<name>.yaml (pi05 or pi0_fast): --part policy")
    ap.add_argument("--model-set", action="append", default=[], metavar="KEY=VALUE", help="override a key of the model YAML: <component>.<key>")
    ap.add_argument("--seed", type=int, default=42, help="seed of the policy's noise (--part policy) and of the environment's reset (--part env)")
    ap.add_argument("--repeats", type=int, default=3, help="--part policy: calls per way (default 3)")
    ap.add_argument("--threads", type=int, default=2, help="--part policy: forks called at the same time (default 2)")
    ap.add_argument("--processes", type=int, default=2, help="--part policy: fresh Python processes whose reference chunk is compared (default 2; 0 for none)")
    ap.add_argument("--servers", type=int, default=3, help="--part env: fresh servers (default 3)")
    ap.add_argument("--replays", type=int, default=2, help="--part env: plays per server (default 2: the second is a reset of the same seed in the same server)")
    ap.add_argument("--amplitude", type=float, default=0.15, help="--part env: amplitude of the sinusoid, rad (default 0.15)")
    ap.add_argument("--period", type=int, default=60, help="--part env: its period in steps (default 60)")
    ap.add_argument("--task-config", default=DEFAULT_TASK)
    ap.add_argument("--no-deterministic-xla", action="store_true", help="--part policy: leave XLA_FLAGS as they are (to see what the flag changes)")
    ap.add_argument("--ready-timeout", type=float, default=1800.0)
    ap.add_argument("--output-root", default="runs/sft/probing")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.part == "policy" and not args.model:
        raise SystemExit("--part policy needs --model")
    if args.part == "env" and (args.servers < 2 or args.replays < 1):
        raise SystemExit("--part env needs --servers 2 or more (fresh against fresh) and --replays 1 or more")

    if args.part == "policy":
        return policy_part(args)
    return env_part(args)


def policy_part(args) -> int:
    if not args.no_deterministic_xla:
        os.environ["XLA_FLAGS"] = with_deterministic_xla(os.environ.get("XLA_FLAGS", ""))
    cfg = apply_overrides(load_model_cfg(args.model), parse_assignments(args.model_set))
    enable_compilation_cache()
    from src.models import load_model
    base = load_model(cfg, seed=0)
    obs = synthetic_obs()
    if args.child:                                                         # a fresh process: only its reference chunk
        base.seed(args.seed)
        print(f"REFERENCE_SHA {chunk_sha(base.plan(obs, PROMPT))}", flush=True)
        return 0
    print(f"policy determinism of {args.model} on {gpu_name()}, XLA_FLAGS='{os.environ.get('XLA_FLAGS', '')}'", flush=True)
    result = policy_checks(base, obs, PROMPT, args.seed, args.repeats, args.threads)
    processes = other_processes(args.model, args.model_set, args.seed, args.processes) if args.processes else []
    result["processes"] = processes
    result["processes_identical"] = all(s == result["reference"] for s in processes) if processes else None
    flags = ["repeat_identical", "chain_identical", "fork_identical", "threads_identical", "processes_identical"]
    print("\n=== results (the bits of the action chunk for one observation and one seed)")
    print(f"reference chunk: {result['reference']}")
    for f in flags:
        print(f"  {f[:-10]:<9} {'-' if result[f] is None else ('identical' if result[f] else 'DIFFERENT')}")
    if processes:
        print(f"  other processes: {processes}")
    verdict = "the model is bit-reproducible here: what differs between two evaluations is its input (the images or the state), not the model" if all(result[f] in (True, None) for f in flags) \
        else "the model is NOT bit-reproducible in the ways marked DIFFERENT"
    print(verdict)
    write_report(args, {"part": "policy", "model": args.model, "xla_flags": os.environ.get("XLA_FLAGS", ""), "gpu": gpu_name(), "result": result, "verdict": verdict}, args.model)
    return 0 if all(result[f] in (True, None) for f in flags) else 1


def env_part(args) -> int:
    from src.envs.client import VectorEnvClient
    tag = probe_tag("env", time.strftime("%Y%m%d_%H%M%S"), os.environ.get("SLURM_JOB_ID"))
    root = ROOT / args.output_root / f"_probe_determinism_{tag}"
    root.mkdir(parents=True, exist_ok=False)
    task_config = Path(args.task_config) if Path(args.task_config).is_absolute() else ROOT / args.task_config
    task = task_settings(task_config, [])
    steps = int(task["max_steps_per_episode"])

    def on_signal(signum, _frame):
        print(f"signal {signum}: killing the env servers and leaving", flush=True)
        kill_all_servers(5.0)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    factory = make_server_factory(task_config, [], root / "logs")
    print(f"simulator determinism on {gpu_name()}: {args.servers} fresh servers x {args.replays} plays of {steps} scripted steps (seed {args.seed})", flush=True)
    traces: Dict[str, np.ndarray] = {}
    frames: Dict[str, Dict[str, list]] = {}                                 # the camera frames of the first play of every server
    for i in range(args.servers):
        server = factory(f"s{i}")
        env = None
        try:
            server.start()
            server.wait_ready(args.ready_timeout, threading.Event())
            env = VectorEnvClient([server.endpoint], connect_timeout_s=120.0)
            for k in range(args.replays):
                t0 = time.monotonic()
                collected: Optional[Dict[str, list]] = {} if k == 0 else None
                traces[f"s{i}.r{k}"] = run_trace(env, args.seed, steps, args.amplitude, args.period, images=collected)
                if collected is not None:
                    frames[f"s{i}.r0"] = collected
                print(f"{time.strftime('%H:%M:%S')} server {i}, play {k}: {len(traces[f's{i}.r{k}']) - 1} steps in {time.monotonic() - t0:.0f} s", flush=True)
        finally:
            if env is not None:
                env.close()
            server.stop()
    result = verdict_env(traces)
    print("\n=== results (readings of environment 0: joints, gripper, peg-socket offset, force)")
    if result["fresh_servers"]:
        for name, t in result["fresh_servers"]["traces"].items():
            print(f"  {name} against {result['fresh_servers']['reference']}: " + ("identical" if t["identical"] else f"first differing step {t['first_differing_step']}, max difference {t['max_abs_diff_at']} (final {t['max_abs_diff_final']:.3g})"))
    for name, t in result["same_server_replays"].items():
        print(f"  {name} against {name[:-3]}.r0 (same server, second reset): " + ("identical" if t["identical"] else f"first differing step {t['first_differing_step']}, final difference {t['max_abs_diff_final']:.3g}"))
    verdict = ("the physics is bit-reproducible over this sequence from fresh servers" if result["fresh_servers_identical"]
               else "the physics is NOT bit-reproducible from fresh servers over this sequence: see the first differing step")
    result["images"] = {name: compare_images(frames["s0.r0"], frames[name]) for name in sorted(frames) if name != "s0.r0"}
    print("\n=== camera images of the first play of each server, against server 0")
    for name, cams in result["images"].items():
        for key, c in cams.items():
            print(f"  {name} {key.split('/')[-1]}: " + ("identical" if c["identical"] else f"first differing step {c['first_differing_step']}, differing pixel values {c['differing_fraction_at']}, "
                                                                                       f"largest difference {c['max_abs_diff_at']}"))
    images_identical = all(c["identical"] for cams in result["images"].values() for c in cams.values())
    result["images_identical"] = images_identical
    verdict += ("; the camera images are bit-identical too" if images_identical else "; the CAMERA IMAGES differ between fresh servers (the model's input is not reproducible)")
    print(verdict)
    np.savez_compressed(root / "traces.npz", **{k.replace(".", "_"): v for k, v in traces.items()})
    write_report(args, {"part": "env", "gpu": gpu_name(), "steps": steps, "seed": args.seed, "result": result, "verdict": verdict, "data": str(root)}, "env", tag)
    return 0 if result["fresh_servers_identical"] and images_identical else 1


def write_report(args, report: dict, name: str, tag: Optional[str] = None) -> None:
    tag = tag or probe_tag(name, time.strftime("%Y%m%d_%H%M%S"), os.environ.get("SLURM_JOB_ID"))
    path = ROOT / "reports" / f"probe_determinism_{args.part}_{tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=1, default=str))
    print(f"report: {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
