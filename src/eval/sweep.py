"""Evaluate several checkpoints, several AT THE SAME TIME, each in its own thread with its own env server.

    sweep = Sweep(output_dir=..., jobs=[Job("0", 0), Job("500", 500, ckpt_dir), ...], seeds=seeds, replan=8, max_steps=150, workers=3,
                  policy_factory=..., prepare=..., server_factory=..., attempt=1, gpu="NVIDIA L40S", stop=stop_event)
    result = sweep.run()

One WORKER is one thread. It takes the next checkpoint from the queue, sets its weights in its own policy, starts a FRESH env server (a subprocess with its own port and
process group), runs the episodes the checkpoint still lacks, stops the server, and takes the next one. Why a fresh server for each checkpoint, and not one per worker kept alive:
the start of an episode is bit-identical only for an identical sequence of calls from a fresh server (docs/envs/isaaclab_env.md G1), and a server that has already run another
checkpoint's episodes would give different bits. The start-up costs a few minutes against two hours of episodes.

Why a subprocess and not a thread for the simulator: Isaac Sim allows one SimulationApp per process. The threads of this file only supervise the subprocesses and drive the policy.

Clean termination (the part that needs care when processes outlive their parent):
  * every server is started in its own process group (start_new_session) and stopped with SIGTERM to the GROUP, then SIGKILL after a grace period: Isaac spawns children;
  * `stop` (a threading.Event, set by the signal handlers of scripts/eval_sft.py) is checked between steps (an episode in progress is dropped, not recorded), while waiting
    for a server and between episodes; each worker stops its server in a `finally`;
  * if the workers do not finish within `grace_s` of the stop, the servers are killed from the main thread (their env calls then fail and the workers leave); `atexit` kills
    whatever is left when the interpreter ends.

Servers start ONE AT A TIME (a lock held from the start to the `ready` line): the first start of all builds the Robotiq USD, and two first starts would write it together.

The policy is NOT loaded here: `policy_factory()` gives each worker a policy of its own (OpenPIPolicy.fork: its own random stream and weights, the same compiled sampler and the
same frozen weights in memory), `prepare(policy, job)` puts the job's weights in it. numpy only: the tests drive it with doubles.
"""
from __future__ import annotations

import atexit
import dataclasses
import os
import queue
import signal
import socket
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from src.envs.client import VectorEnvClient
from src.eval import episodes as E

READY_LINE = "[env_server] ready"
ROOT = Path(__file__).resolve().parents[2]


class ServerError(RuntimeError):
    """An env server did not start, or died."""


# -- the env server as a subprocess -------------------------------------------------------------------------------------------------
_LIVE: set = set()
_LIVE_LOCK = threading.Lock()


def kill_all_servers(grace_s: float = 10.0) -> None:
    """Stop every server this process started and has not stopped yet (SIGTERM to the group, SIGKILL after `grace_s`)."""
    with _LIVE_LOCK:
        live = list(_LIVE)
    for server in live:
        server.stop(grace_s)


atexit.register(kill_all_servers, 5.0)


def free_port() -> int:
    """A port nobody listens on right now (the OS picks it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def clean_server_env(env: Mapping[str, str]) -> Dict[str, str]:
    """The environment of an env server started from the learner's venv: the learner venv is taken out (jobs/env_server.sh activates .venv-isaaclab itself) and the JAX / XLA
    settings of the learner do not reach Isaac Sim."""
    out = {k: v for k, v in env.items() if k not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH") and not k.startswith(("XLA_", "JAX_"))}
    venv = env.get("VIRTUAL_ENV")
    if venv and "PATH" in out:
        out["PATH"] = os.pathsep.join(p for p in out["PATH"].split(os.pathsep) if not p.startswith(venv.rstrip("/") + "/"))
    return out


class ServerProcess:
    """One env server: `command` run in its own process group, its output in `log_path`, ready when `READY_LINE` appears in it."""

    def __init__(self, command: Sequence[str], log_path: Path, endpoint: Tuple[str, int], env: Optional[Mapping[str, str]] = None, cwd: Optional[Path] = None):
        self.command, self.log_path, self.endpoint = list(command), Path(log_path), endpoint
        self.env, self.cwd = (dict(env) if env is not None else None), (str(cwd) if cwd else None)
        self.proc: Optional[subprocess.Popen] = None
        self._log = None
        self._offset = 0
        self._tail = b""
        self._lock = threading.Lock()

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        self._log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(self.command)}\n".encode())
        self._log.flush()
        self._offset = self.log_path.stat().st_size                      # the log is appended to: a restart must not read an old `ready` line
        self._tail = b""
        self.proc = subprocess.Popen(self.command, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=self.cwd, env=self.env, start_new_session=True)
        with _LIVE_LOCK:
            _LIVE.add(self)

    def _ready_in_log(self) -> bool:
        with open(self.log_path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        self._offset += len(data)
        window = self._tail + data
        self._tail = window[-len(READY_LINE):]
        return READY_LINE.encode() in window

    def tail(self, lines: int = 30) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def wait_ready(self, timeout_s: float, stop: Optional[threading.Event] = None, poll_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            if stop is not None and stop.is_set():
                raise E.EvalAborted("stopped while the env server was starting")
            code = self.proc.poll()
            if self._ready_in_log():
                return
            if code is not None:
                raise ServerError(f"the env server exited with code {code} before it was ready; end of {self.log_path}:\n{self.tail()}")
            if time.monotonic() > deadline:
                raise ServerError(f"the env server was not ready after {timeout_s:.0f} s; end of {self.log_path}:\n{self.tail()}")
            (stop.wait(poll_s) if stop is not None else time.sleep(poll_s))

    def stop(self, grace_s: float = 30.0) -> None:
        """SIGTERM to the whole process group, SIGKILL when it is still there after `grace_s`. Safe to call twice, from any thread."""
        with self._lock:
            proc, self.proc = self.proc, None
        if proc is not None:
            for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 15.0)):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=wait_s)
                    break
                except subprocess.TimeoutExpired:
                    continue
            try:
                os.killpg(proc.pid, signal.SIGKILL)                  # whatever the group still holds once the leader is gone
            except ProcessLookupError:
                pass
        if self._log is not None:
            self._log.close()
            self._log = None
        with _LIVE_LOCK:
            _LIVE.discard(self)


def make_server_factory(task_config: Path, env_set: Sequence[str], logs_dir: Path, base_env: Optional[Mapping[str, str]] = None) -> Callable[[str], ServerProcess]:
    """label -> a ServerProcess for one environment, on a port nobody listens on, running jobs/env_server.sh with the task YAML and `--set KEY=VALUE` overrides."""
    env = clean_server_env(base_env if base_env is not None else os.environ)
    env["ENV_CONFIG"] = str(task_config)

    def factory(label: str) -> ServerProcess:
        port = free_port()
        command = ["bash", str(ROOT / "jobs" / "env_server.sh"), "--set", "num_envs=1", "--set", f"server_port={port}"]
        for item in env_set:
            command += ["--set", item]
        return ServerProcess(command, Path(logs_dir) / f"server_{label}.log", ("127.0.0.1", port), env=env, cwd=ROOT)

    return factory


def gpu_name() -> str:
    """Name of the first GPU nvidia-smi sees (recorded with every episode: a curve should come from one kind of GPU, N4 of docs/envs/isaaclab_env.md)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=30).stdout
        return out.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# -- the sweep --------------------------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Job:
    """One checkpoint to evaluate. `label` names its files (results/<label>.jsonl / .json), `step` is its place on the curve, `checkpoint` what `prepare` loads (None: the initial state)."""
    label: str
    step: int
    checkpoint: Optional[Path] = None


@dataclasses.dataclass
class SweepResult:
    complete: List[str]                 # every episode done (now or before)
    incomplete: List[str]               # stopped before the last episode
    failed: Dict[str, str]              # label -> what went wrong
    stopped: bool                       # the stop event was set

    @property
    def ok(self) -> bool:
        return not self.failed and not self.incomplete


class Sweep:
    def __init__(self, *, output_dir: Path, jobs: Sequence[Job], seeds: Sequence[int], replan: int, max_steps: int, workers: int,
                 policy_factory: Callable[[], object], prepare: Callable[[object, Job], None], server_factory: Callable[[str], ServerProcess],
                 attempt: int, gpu: str, stop: Optional[threading.Event] = None, on_complete: Optional[Callable[[Job, dict], None]] = None,
                 video_dir: Optional[Path] = None, fps: int = 15, ready_timeout_s: float = 1800.0, connect_timeout_s: float = 120.0, grace_s: float = 90.0,
                 say: Callable[[str], None] = print):
        self.output_dir, self.jobs, self.seeds = Path(output_dir), list(jobs), list(seeds)
        self.replan, self.max_steps, self.workers = int(replan), int(max_steps), max(1, int(workers))
        self.policy_factory, self.prepare, self.server_factory = policy_factory, prepare, server_factory
        self.attempt, self.gpu = int(attempt), gpu
        self.stop = stop if stop is not None else threading.Event()
        self.on_complete, self.video_dir, self.fps = on_complete, (Path(video_dir) if video_dir else None), fps
        self.ready_timeout_s, self.connect_timeout_s, self.grace_s = ready_timeout_s, connect_timeout_s, grace_s
        self.say = say
        self.results_dir = self.output_dir / "results"
        self._start_lock, self._finish_lock, self._state_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self._complete: List[str] = []
        self._incomplete: List[str] = []
        self._failed: Dict[str, str] = {}

    # -- bookkeeping ------------------------------------------------------------------------------------------------------------------------
    def log_for(self, job: Job) -> E.EpisodeLog:
        return E.EpisodeLog(self.results_dir / f"{job.label}.jsonl")

    def _acquire(self, lock: threading.Lock) -> None:
        """Take `lock`, giving up (EvalAborted) when the sweep is told to stop while waiting for it."""
        while not lock.acquire(timeout=1.0):
            if self.stop.is_set():
                raise E.EvalAborted("stopped while waiting for another server to start")

    def _finish(self, job: Job, records: Mapping[int, dict]) -> None:
        summary = E.summarize(job.label, job.step, records, self.seeds)
        E.atomic_write_json(self.results_dir / f"{job.label}.json", summary)
        with self._finish_lock:
            if self.on_complete is not None:
                self.on_complete(job, summary)
        attempts = "one attempt" if summary["single_attempt"] else f"{len(summary['attempts'])} attempts"
        self.say(f"[{job.label}] done: {summary['n_success']}/{summary['n_episodes']} = {100 * summary['success_rate']:.1f} % "
                 f"(+- {100 * summary['success_se']:.1f} SE), {attempts}, eval_sha {summary['eval_sha']}")

    # -- one checkpoint ------------------------------------------------------------------------------------------------------------------------
    def _run_job(self, job: Job, policy) -> str:
        log = self.log_for(job)
        done = log.load()
        if log.duplicates:
            self.say(f"[{job.label}] WARNING: {log.duplicates} episodes appear twice in {log.path.name} (two runs on the same folder?): the first record of each is kept")
        pending = [i for i in range(len(self.seeds)) if i not in done]
        if not pending:
            self._finish(job, done)
            return "complete"
        self.say(f"[{job.label}] {len(done)}/{len(self.seeds)} episodes already done, {len(pending)} to run")
        self.prepare(policy, job)
        server = self.server_factory(job.label)
        env = None
        try:
            self._acquire(self._start_lock)
            try:
                server.start()
                server.wait_ready(self.ready_timeout_s, self.stop)
            finally:
                self._start_lock.release()
            env = VectorEnvClient([server.endpoint], connect_timeout_s=self.connect_timeout_s)
            if env.spec.max_steps != self.max_steps:
                raise ServerError(f"the server's episodes are {env.spec.max_steps} steps long, the protocol's are {self.max_steps} (max_steps_per_episode of the task YAML)")
            self.say(f"[{job.label}] env server ready on port {server.endpoint[1]}")
            for i in pending:
                if self.stop.is_set():
                    break
                video = self.video_dir / job.label / f"ep_{i:04d}_seed{self.seeds[i]}.mp4" if self.video_dir else None
                try:
                    record = E.run_episode(env, policy, episode=i, seed=self.seeds[i], replan=self.replan, max_steps=self.max_steps, stop=self.stop,
                                           video_path=video, fps=self.fps)
                except E.EvalAborted:
                    break
                record.update(attempt=self.attempt, gpu=self.gpu)
                log.append(record)
                done[i] = record
                counted = [r for k, r in done.items() if k < len(self.seeds)]
                self.say(f"[{job.label}] episode {i} (seed {self.seeds[i]}): {'SUCCESS' if record['success'] else 'failure'} in {record['steps']} steps, "
                         f"{record['wall_s']:.0f} s; {len(counted)}/{len(self.seeds)} done, {sum(bool(r['success']) for r in counted)} successes")
        finally:
            if env is not None:
                env.close()
            server.stop()
        if all(i in done for i in range(len(self.seeds))):
            self._finish(job, done)
            return "complete"
        return "incomplete"

    # -- the workers ---------------------------------------------------------------------------------------------------------------------------------
    def _worker(self, jobs: "queue.Queue[Job]") -> None:
        try:
            policy = self.policy_factory()
        except Exception as e:  # noqa: BLE001
            self.say(f"a worker could not make its policy: {e!r}\n{traceback.format_exc()}")
            with self._state_lock:
                self._failed[threading.current_thread().name] = repr(e)
            return
        while not self.stop.is_set():
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                status = self._run_job(job, policy)
            except E.EvalAborted:
                status = "incomplete"
            except Exception as e:  # noqa: BLE001  one checkpoint failing must not take the others with it
                self.say(f"[{job.label}] FAILED: {e!r}\n{traceback.format_exc()}")
                with self._state_lock:
                    self._failed[job.label] = repr(e)
                continue
            with self._state_lock:
                (self._complete if status == "complete" else self._incomplete).append(job.label)

    def run(self) -> SweepResult:
        jobs: "queue.Queue[Job]" = queue.Queue()
        for job in self.jobs:
            jobs.put(job)
        threads = [threading.Thread(target=self._worker, args=(jobs,), name=f"eval-worker-{k}", daemon=True) for k in range(min(self.workers, len(self.jobs)))]
        for t in threads:
            t.start()
        stopped_at: Optional[float] = None
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
            if self.stop.is_set():
                stopped_at = stopped_at if stopped_at is not None else time.monotonic()
                waited = time.monotonic() - stopped_at
                if waited > self.grace_s:
                    kill_all_servers()                                        # their env calls fail and the workers leave
                if waited > self.grace_s + 30:
                    self.say("some workers did not leave in time: giving up on them (their servers are killed)")
                    break
        with self._state_lock:
            queued = [j.label for j in self.jobs if j.label not in self._complete and j.label not in self._failed]
            return SweepResult(complete=sorted(self._complete, key=lambda l: next(j.step for j in self.jobs if j.label == l)),
                               incomplete=[l for l in queued], failed=dict(self._failed), stopped=self.stop.is_set())
