"""src/eval/sweep.py: the env server as a subprocess (ready line, death, timeout, stop, the whole process group is killed) and the sweep of several checkpoints in
threads, against the fake simulator over real sockets: resume of what is missing, same results with one or several workers, one server at a time, a failing checkpoint
that does not stop the others, every server stopped. The servers hold K environments that play a checkpoint's episodes together (batches)."""
import json
import os
import signal
import sys
import threading
import time

import pytest

from src.eval import episodes as E
from src.eval import sweep as S
from test_client_server import Running  # noqa: F401
from test_eval_episodes import FakePolicy, will_succeed

N = 6                      # episodes per checkpoint, and the environments of a server unless a test says otherwise
MASTER = 42


# -- the subprocess -------------------------------------------------------------------------------------------------------------------------------
def py(code):
    return [sys.executable, "-u", "-c", code]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        return "Z" not in open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0]        # a zombie is dead for our purpose
    except OSError:
        return False


def test_a_server_is_ready_when_its_log_says_so_and_stop_kills_the_whole_group(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = f"import subprocess,sys,time; c=subprocess.Popen(['sleep','300']); open({str(pid_file)!r},'w').write(str(c.pid)); print('[env_server] ready: 1 environments',flush=True); time.sleep(300)"
    server = S.ServerProcess(py(code), tmp_path / "logs" / "server.log", ("127.0.0.1", 1))
    server.start()
    server.wait_ready(30, poll_s=0.1)
    child = int(pid_file.read_text())
    assert alive(server.proc.pid) and alive(child)
    leader = server.proc.pid
    t0 = time.monotonic()
    server.stop(grace_s=5)
    assert time.monotonic() - t0 < 20 and not alive(leader) and not alive(child)               # the child Isaac would have spawned is gone too
    server.stop()                                                                              # twice is fine
    assert server not in S._LIVE


def test_a_server_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('[env_server] ready',flush=True); time.sleep(300)"
    server = S.ServerProcess(py(code), tmp_path / "s.log", ("127.0.0.1", 1))
    server.start()
    server.wait_ready(30, poll_s=0.1)
    pid = server.proc.pid
    server.stop(grace_s=1)
    assert not alive(pid)


def test_a_server_that_dies_before_it_is_ready_is_reported_with_its_log(tmp_path):
    server = S.ServerProcess(py("print('Isaac Sim failed: no GPU',flush=True); raise SystemExit(3)"), tmp_path / "s.log", ("127.0.0.1", 1))
    server.start()
    with pytest.raises(S.ServerError, match=r"(?s)exited with code 3.*no GPU"):
        server.wait_ready(30, poll_s=0.1)
    server.stop()


def test_a_server_that_is_never_ready_times_out_and_a_stop_ends_the_wait(tmp_path):
    server = S.ServerProcess(py("import time; time.sleep(300)"), tmp_path / "s.log", ("127.0.0.1", 1))
    server.start()
    with pytest.raises(S.ServerError, match="not ready after"):
        server.wait_ready(0.5, poll_s=0.1)
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    with pytest.raises(E.EvalAborted):
        server.wait_ready(60, stop, poll_s=0.1)
    server.stop(grace_s=2)


def test_an_old_ready_line_in_an_appended_log_is_not_mistaken_for_the_new_servers(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("[env_server] ready: from an earlier start\n")
    server = S.ServerProcess(py("import time; time.sleep(300)"), log, ("127.0.0.1", 1))
    server.start()
    with pytest.raises(S.ServerError, match="not ready after"):
        server.wait_ready(0.5, poll_s=0.1)
    server.stop(grace_s=2)


def test_kill_all_servers_stops_what_is_left(tmp_path):
    servers = [S.ServerProcess(py("import time; time.sleep(300)"), tmp_path / f"s{k}.log", ("127.0.0.1", 1)) for k in range(2)]
    for s in servers:
        s.start()
    pids = [s.proc.pid for s in servers]
    S.kill_all_servers(grace_s=2)
    assert not any(alive(p) for p in pids) and not S._LIVE


def test_the_server_environment_loses_the_learners_venv_and_the_xla_settings(tmp_path):
    env = S.clean_server_env({"VIRTUAL_ENV": "/p/.venv", "PATH": "/p/.venv/bin:/usr/bin:/bin", "PYTHONPATH": "/x", "XLA_FLAGS": "--a", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
                              "JAX_COMPILATION_CACHE_DIR": "/c", "HOME": "/h", "SLURM_JOB_ID": "7"})
    assert env == {"PATH": "/usr/bin:/bin", "HOME": "/h", "SLURM_JOB_ID": "7"}
    factory = S.make_server_factory(tmp_path / "task.yaml", ["max_steps_per_episode=100"], tmp_path / "logs", {"PATH": "/usr/bin"})
    a, b = factory("0"), factory("500")
    assert a.command[:2] == ["bash", str(S.ROOT / "jobs" / "env_server.sh")] and "num_envs=1" in a.command and "max_steps_per_episode=100" in a.command
    assert a.env["ENV_CONFIG"] == str(tmp_path / "task.yaml") and a.env["PATH"] == "/usr/bin" and b.log_path.name == "server_500.log"
    assert a.log_path == tmp_path / "logs" / "server_0.log" and f"server_port={a.endpoint[1]}" in a.command


# -- the sweep ---------------------------------------------------------------------------------------------------------------------------------------------
class FakeServer:
    """What a ServerProcess is for the sweep: `start`, `wait_ready`, `stop` and an endpoint, here a fake simulator behind a real socket. The registry counts what runs."""

    def __init__(self, registry, label, max_steps, num_envs):
        self.registry, self.label, self.max_steps, self.num_envs = registry, label, max_steps, num_envs
        self.running = None
        self.endpoint = None

    def start(self):
        with self.registry.lock:
            self.registry.starting += 1
            self.registry.max_starting = max(self.registry.max_starting, self.registry.starting)
        time.sleep(0.05)                                                                   # long enough for two workers to overlap, if nothing serialised them
        self.running = Running(num_envs=self.num_envs, max_steps=self.max_steps, seed=0)
        self.endpoint = self.running.endpoint
        with self.registry.lock:
            self.registry.starting -= 1
            self.registry.started += 1
            self.registry.live.add(self.label)

    def wait_ready(self, timeout_s, stop=None):
        time.sleep(0.02)

    def stop(self, grace_s=0):
        if self.running is not None:
            self.registry.reset_calls.append(list(self.running.sim.reset_calls))                 # the seeds of the real resets this server did
            self.running.stop()
            self.running = None
            with self.registry.lock:
                self.registry.live.discard(self.label)
                self.registry.stopped += 1


class Registry:
    def __init__(self, max_steps=20, num_envs=N):
        self.lock = threading.Lock()
        self.starting = self.max_starting = self.stopped = self.started = 0
        self.live = set()
        self.reset_calls = []
        self.max_steps, self.num_envs = max_steps, num_envs

    def factory(self, label):
        return FakeServer(self, label, self.max_steps, self.num_envs)


class CountingPolicy(FakePolicy):
    """A FakePolicy that counts the episodes it plays (the first chunk of an episode is asked for with the noise seed `plan_seed(episode seed, 0)`: one entry, the episode's seed, per episode)
    in a counter shared by all the workers, and can stop the sweep after a number of model calls."""

    def __init__(self, shared, stop=None, stop_after_calls=None):
        super().__init__()
        self.shared, self.stop, self.stop_after_calls = shared, stop, stop_after_calls

    def raw_chunks(self, rows, num_samples, row_seeds=None, pad_to=None):
        with self.shared["lock"]:
            for s in row_seeds or ():
                if s in self.shared["firsts"]:
                    self.shared["episodes"].append(self.shared["firsts"][s])
            self.shared["calls"] += 1
            if self.stop_after_calls is not None and self.shared["calls"] >= self.stop_after_calls:
                self.stop.set()
        return super().raw_chunks(rows, num_samples, row_seeds, pad_to)


def make_sweep(tmp_path, labels, workers, shared, registry, stop=None, stop_after_calls=None, attempt=1, prepare=None, seeds=None, completed=None, envs=N):
    stop = stop or threading.Event()
    seeds = seeds or E.make_seeds(MASTER, N)
    shared["firsts"].update({E.plan_seed(s, 0): s for s in E.make_seeds(MASTER, 50)})
    jobs = [S.Job(label=l, step=int(l)) for l in labels]
    return S.Sweep(output_dir=tmp_path, jobs=jobs, seeds=seeds, replan=4, max_steps=20, workers=workers,
                   policy_factory=lambda: CountingPolicy(shared, stop, stop_after_calls), prepare=prepare or (lambda policy, job: None),
                   server_factory=registry.factory, attempt=attempt, gpu="fake gpu", stop=stop, on_complete=(lambda job, s: completed.append(job.label)) if completed is not None else None,
                   say=lambda m: None, grace_s=5, master_seed=MASTER, envs_per_server=envs), stop


def shared():
    return {"lock": threading.Lock(), "episodes": [], "firsts": {}, "calls": 0}


def outcome(i, envs=N):
    """Whether episode i succeeds: environment i % envs in round i // envs of the master seed."""
    return will_succeed(MASTER, i // envs, i % envs)


def results(tmp_path, label):
    return json.loads((tmp_path / "results" / f"{label}.json").read_text())


def test_a_sweep_evaluates_every_checkpoint_on_the_same_episodes_and_writes_what_the_curve_needs(tmp_path):
    reg, sh, done = Registry(), shared(), []
    sw, _ = make_sweep(tmp_path, ["0", "500", "1000"], 2, sh, reg, completed=done)
    out = sw.run()
    assert out.ok and out.complete == ["0", "500", "1000"] and sorted(done) == ["0", "1000", "500"] and not out.stopped
    seeds = E.make_seeds(MASTER, N)
    for label in ("0", "500", "1000"):
        r = results(tmp_path, label)
        assert r["seeds"] == seeds and r["n_episodes"] == N and r["single_attempt"] and r["gpus"] == ["fake gpu"]
        assert r["successes"] == [outcome(i) for i in range(N)]                         # the fake simulator's outcome is known from the start state of the environment
        assert len((tmp_path / "results" / f"{label}.jsonl").read_text().splitlines()) == N
    assert len({results(tmp_path, l)["eval_sha"] for l in ("0", "500", "1000")}) == 1       # the same policy on the same episodes: the same fingerprint
    assert reg.live == set() and reg.stopped == 3                                         # a fresh server per checkpoint, all stopped
    assert sorted(reg.reset_calls) == [[MASTER]] * 3                                      # ONE real reset per checkpoint: round 0 of the master seed


def test_one_worker_or_several_give_the_same_results(tmp_path):
    shas = {}
    for workers in (1, 3):
        reg = Registry()
        sw, _ = make_sweep(tmp_path / f"w{workers}", ["0", "100", "200"], workers, shared(), reg)
        assert sw.run().ok
        shas[workers] = {l: results(tmp_path / f"w{workers}", l)["eval_sha"] for l in ("0", "100", "200")}
    assert shas[1] == shas[3]


def test_servers_start_one_at_a_time_even_with_several_workers(tmp_path):
    reg = Registry()
    sw, _ = make_sweep(tmp_path, ["0", "1", "2", "3"], 4, shared(), reg)
    assert sw.run().ok and reg.max_starting == 1


def test_a_sweep_that_is_stopped_keeps_its_episodes_and_the_same_command_finishes_it_as_if_nothing_had_happened(tmp_path):
    labels = ["0", "500", "1000"]
    reference_dir = tmp_path / "reference"
    ref, _ = make_sweep(reference_dir, labels, 1, shared(), Registry())
    assert ref.run().ok

    reg, sh = Registry(), shared()
    cut, stop = make_sweep(tmp_path / "cut", labels, 2, sh, reg, stop_after_calls=5)            # SIGTERM at the fifth model call, in the middle of two checkpoints
    first = cut.run()
    assert first.stopped and not first.ok and first.incomplete and not first.complete
    assert reg.live == set() and reg.started >= 1 and reg.stopped == reg.started                # every server that was started is stopped
    logged = {l: len(E.EpisodeLog(tmp_path / "cut" / "results" / f"{l}.jsonl").load()) for l in labels}
    assert 0 < sum(logged.values()) and all(v < N for v in logged.values())                    # what had ended is on disk; the episodes still running were dropped
    assert not list((tmp_path / "cut" / "results").glob("*.json"))                              # no result for a checkpoint that is not complete

    sh2, reg2 = shared(), Registry()
    resume, _ = make_sweep(tmp_path / "cut", labels, 2, sh2, reg2, attempt=2)
    second = resume.run()
    assert second.ok and sorted(second.complete) == sorted(labels)
    assert len(sh2["episodes"]) == N * len(labels) - sum(logged.values())                         # ONLY the missing episodes were played (the others of the batch were held)
    for label in labels:
        r, ref_r = results(tmp_path / "cut", label), results(reference_dir, label)
        assert r["successes"] == ref_r["successes"] and r["eval_sha"] == ref_r["eval_sha"]       # the fake simulator has no history: the same bits as an uncut run
        assert r["single_attempt"] == (logged[label] == 0 or logged[label] == N)
        assert len(E.EpisodeLog(tmp_path / "cut" / "results" / f"{label}.jsonl").load()) == N
    assert any(not results(tmp_path / "cut", l)["single_attempt"] for l in labels)               # at least one checkpoint was finished by a second attempt, and says so
    assert reg2.live == set()


def test_a_finished_checkpoint_is_not_run_again(tmp_path):
    sw, _ = make_sweep(tmp_path, ["0", "500"], 1, shared(), Registry())
    assert sw.run().ok
    sh, reg = shared(), Registry()
    again, _ = make_sweep(tmp_path, ["0", "500"], 1, sh, reg, attempt=2)
    assert again.run().ok and sh["episodes"] == [] and reg.started == 0                            # no episode, no server
    assert results(tmp_path, "0")["single_attempt"]


def test_more_episodes_than_before_only_runs_the_new_ones(tmp_path):
    short, _ = make_sweep(tmp_path, ["0"], 1, shared(), Registry(), seeds=E.make_seeds(MASTER, 3))
    assert short.run().ok
    sh = shared()
    longer, _ = make_sweep(tmp_path, ["0"], 1, sh, Registry(), seeds=E.make_seeds(MASTER, 5), attempt=2)
    assert longer.run().ok
    assert sh["episodes"] == E.make_seeds(MASTER, 5)[3:] and results(tmp_path, "0")["n_episodes"] == 5
    assert results(tmp_path, "0")["successes"] == [outcome(i) for i in range(5)]                     # the new episodes start from the same states as in one go


def test_a_failing_checkpoint_is_reported_and_the_others_are_evaluated(tmp_path):
    def prepare(policy, job):
        if job.label == "500":
            raise ValueError("checkpoint 500 does not fit this policy")

    reg = Registry()
    sw, _ = make_sweep(tmp_path, ["0", "500", "1000"], 2, shared(), reg, prepare=prepare)
    out = sw.run()
    assert not out.ok and set(out.failed) == {"500"} and "does not fit" in out.failed["500"]
    assert sorted(out.complete) == ["0", "1000"] and (tmp_path / "results" / "0.json").exists() and not (tmp_path / "results" / "500.json").exists()
    assert reg.live == set()


def test_the_server_is_stopped_when_an_episode_blows_up_and_a_wrong_episode_length_is_refused(tmp_path):
    class Boom(CountingPolicy):
        def raw_chunks(self, rows, n, row_seeds=None, pad_to=None):
            raise RuntimeError("the model crashed")

    reg = Registry()
    sw = S.Sweep(output_dir=tmp_path, jobs=[S.Job("0", 0)], seeds=E.make_seeds(MASTER, 2), replan=4, max_steps=20, workers=1, policy_factory=lambda: Boom(shared()),
                 prepare=lambda p, j: None, server_factory=reg.factory, attempt=1, gpu="g", say=lambda m: None, master_seed=MASTER, envs_per_server=N)
    out = sw.run()
    assert "the model crashed" in out.failed["0"] and reg.live == set() and reg.started == 1 and reg.stopped == 1
    reg2 = Registry(max_steps=30)                                                                   # a server whose episodes are not the protocol's
    sw2, _ = make_sweep(tmp_path / "b", ["0"], 1, shared(), reg2)
    out2 = sw2.run()
    assert "30 steps long" in out2.failed["0"] and reg2.live == set()


def test_a_worker_that_cannot_make_its_policy_is_a_failure_and_a_stop_before_the_start_runs_nothing(tmp_path):
    reg = Registry()

    def broken():
        raise MemoryError("no room for the policy")

    sw = S.Sweep(output_dir=tmp_path, jobs=[S.Job("0", 0)], seeds=E.make_seeds(MASTER, 2), replan=4, max_steps=20, workers=1, policy_factory=broken,
                 prepare=lambda p, j: None, server_factory=reg.factory, attempt=1, gpu="g", say=lambda m: None, master_seed=MASTER, envs_per_server=N)
    out = sw.run()
    assert out.failed and not out.ok and "0" in out.incomplete and reg.started == 0
    stop = threading.Event()
    stop.set()
    sh = shared()
    sw2, _ = make_sweep(tmp_path / "b", ["0", "1"], 2, sh, reg, stop=stop)
    out2 = sw2.run()
    assert out2.stopped and sh["episodes"] == [] and sorted(out2.incomplete) == ["0", "1"]


def test_a_stop_while_waiting_for_the_start_lock_leaves_at_once(tmp_path):
    reg = Registry()
    sw, stop = make_sweep(tmp_path, ["0"], 1, shared(), reg)
    sw._start_lock.acquire()                                                                       # another worker is starting its server, for a long time
    threading.Timer(0.3, stop.set).start()
    t0 = time.monotonic()
    out = sw.run()
    assert time.monotonic() - t0 < 10 and out.stopped and reg.live == set()
    sw._start_lock.release()


def test_fewer_environments_than_episodes_means_several_batches_each_with_its_own_round_and_the_same_outcomes_as_the_batch_semantics_say(tmp_path):
    reg, sh = Registry(num_envs=2), shared()
    sw, _ = make_sweep(tmp_path, ["0"], 1, sh, reg, envs=2)
    assert sw.run().ok
    assert reg.reset_calls == [[MASTER, MASTER + 1, MASTER + 2]]                                   # three batches of two: rounds 0, 1, 2 of the master seed, in one server
    r = results(tmp_path, "0")
    assert r["successes"] == [outcome(i, envs=2) for i in range(N)]
    recs = E.EpisodeLog(tmp_path / "results" / "0.jsonl").load()
    assert sorted((rec["batch"], rec["env"]) for rec in recs.values()) == [(b, e) for b in range(3) for e in range(2)]


def test_a_batch_that_was_cut_is_resumed_in_the_same_round_in_a_fresh_server(tmp_path):
    reg, sh = Registry(num_envs=2), shared()
    sw, _ = make_sweep(tmp_path, ["0"], 1, sh, reg, seeds=E.make_seeds(MASTER, 4), envs=2)
    assert sw.run().ok
    log = E.EpisodeLog(tmp_path / "results" / "0.jsonl")
    kept = {i: r for i, r in log.load().items() if i in (0, 1, 2)}                                  # episode 3 (batch 1, environment 1) is lost, as in a kill
    (tmp_path / "results" / "0.jsonl").unlink()
    (tmp_path / "results" / "0.json").unlink()
    for r in kept.values():
        log.append(r)
    reg2, sh2 = Registry(num_envs=2), shared()
    again, _ = make_sweep(tmp_path, ["0"], 1, sh2, reg2, seeds=E.make_seeds(MASTER, 4), envs=2, attempt=2)
    assert again.run().ok
    assert reg2.reset_calls == [[MASTER + 1]]                                                       # only batch 1 runs again, with ITS round
    assert sh2["episodes"] == [E.make_seeds(MASTER, 4)[3]] and results(tmp_path, "0")["successes"] == [outcome(i, envs=2) for i in range(4)]


def test_a_server_with_another_number_of_environments_than_the_protocols_is_refused(tmp_path):
    reg = Registry(num_envs=3)
    sw, _ = make_sweep(tmp_path, ["0"], 1, shared(), reg, envs=N)
    out = sw.run()
    assert not out.ok and "3 environments" in out.failed["0"] and reg.live == set()
