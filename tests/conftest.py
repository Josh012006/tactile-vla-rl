"""Shared pytest setup.

Two families of tests:
  * pure tests (default): numpy only, no GPU, no Isaac Sim. They test the logic that runs on the cluster, with a fake simulator
    behind the same SimAdapter (tests/fakes.py).
  * Isaac tests (`--isaac`): start real env servers (python -m src.envs.server) in subprocesses, so they need the Isaac Lab
    virtualenv and a GPU. They check the part that cannot be tested without the simulator (state restore, isolation) and
    measure time and GPU memory per environment.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_addoption(parser):
    parser.addoption("--isaac", action="store_true", default=False, help="also run the tests that start Isaac Lab servers (GPU)")
    parser.addoption("--isaac-config", default="configs/task/peg_insert_forge.yaml", help="task YAML of the Isaac tests")
    parser.addoption("--isaac-set", action="append", default=[], metavar="KEY=VALUE",
                     help="YAML field overridden on EVERY server of the Isaac tests (repeatable), "
                          "e.g. --isaac-set robot=robotiq_2f85 --isaac-set tactile_exclude_zone_joints=true")


def pytest_configure(config):
    config.addinivalue_line("markers", "isaac: needs Isaac Lab and a GPU (run with --isaac)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--isaac"):
        return
    skip = pytest.mark.skip(reason="needs Isaac Lab + GPU: run with --isaac")
    for item in items:
        if "isaac" in item.keywords:
            item.add_marker(skip)


# ------------------------------------------------------------------------------------------------------------ Isaac servers
class IsaacServer:
    """One env server process. `start()` blocks until it answers (Isaac Sim start-up + bank build can take minutes)."""

    def __init__(self, config, num_envs, port, tmp_path, sets=()):
        self.config, self.num_envs, self.port = config, num_envs, port
        self.log_path = Path(tmp_path) / f"server_n{num_envs}_p{port}.log"
        self.sets = [f"num_envs={num_envs}", f"server_port={port}", *sets]
        self.proc = None
        self.start_seconds = None

    def _crash_traceback(self):
        """The log's traceback if the server crashed but its process is still alive (Kit may not exit after an exception)."""
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return None
        return text if "Traceback (most recent call last)" in text else None

    def start(self, timeout_s=1800):
        env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        cmd = [sys.executable, "-m", "src.envs.server", "--config", self.config, *sum((["--set", s] for s in self.sets), [])]
        t0 = time.time()
        with open(self.log_path, "w") as log:
            self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        from src.envs.client import EnvConnection, EnvServerError
        conn = None
        while conn is None:
            if self.proc.poll() is not None:
                raise RuntimeError(f"env server exited with code {self.proc.returncode}; log: {self.log_path}\n"
                                   + self.log_path.read_text()[-3000:])
            text = self._crash_traceback()
            if text is not None:
                time.sleep(10)          # let a normal exit happen first (its exit code is then reported above)
                self.stop()
                raise RuntimeError(f"env server crashed (process kept alive by Kit); log: {self.log_path}\n" + text[-3000:])
            if time.time() - t0 > timeout_s:
                self.stop()
                raise TimeoutError(f"env server not ready after {timeout_s} s; log: {self.log_path}")
            try:
                conn = EnvConnection("127.0.0.1", self.port, connect_timeout_s=5)
            except EnvServerError:
                pass
        self.start_seconds = time.time() - t0
        conn.close()
        return self

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def client(self):
        from src.envs.client import VectorEnvClient
        return VectorEnvClient([("127.0.0.1", self.port)], connect_timeout_s=60)


@pytest.fixture(scope="session")
def isaac_config(request):
    return request.config.getoption("--isaac-config")


class ServerPool:
    """At most ONE Isaac server alive at a time (two would compete for the GPU and distort every measurement). A server is
    reused by the tests that ask for the same (num_envs, overrides); asking for another one stops the current one first."""

    def __init__(self, config, log_dir, extra_sets=()):
        self.config, self.log_dir, self.current, self.key, self._port = config, log_dir, None, None, 8150
        self.extra_sets = list(extra_sets)      # --isaac-set KEY=VALUE, applied to every server (a test's own overrides come after)

    def get(self, num_envs, fresh=False, **overrides):
        key = (num_envs, tuple(sorted(overrides.items())))
        if self.current is not None and self.key == key and not fresh and self.current.proc.poll() is None:
            return self.current
        self.stop()
        self._port += 1
        sets = [f"bank_cache_dir={ROOT / 'cache' / 'start_banks_tests'}", *self.extra_sets,
                *[f"{k}={v}" for k, v in overrides.items()]]
        srv = IsaacServer(self.config, num_envs, self._port, self.log_dir, sets)
        self.current, self.key = srv, key
        try:
            return srv.start()
        except Exception:
            self.stop()
            raise

    def stop(self):
        if self.current is not None:
            self.current.stop()
        self.current, self.key = None, None


@pytest.fixture(scope="session")
def isaac_pool(isaac_config, tmp_path_factory, request):
    pool = ServerPool(isaac_config, tmp_path_factory.mktemp("isaac_servers"), request.config.getoption("--isaac-set"))
    yield pool
    pool.stop()


@pytest.fixture(scope="session")
def reports_dir():
    d = ROOT / "reports"
    d.mkdir(exist_ok=True)
    return d
