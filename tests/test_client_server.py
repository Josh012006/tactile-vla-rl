"""Client <-> WebSocket server <-> ManagerBackend <-> fake simulator, over real sockets. Pure (no GPU).

Checks that the protocol carries everything (auto-reset rows, final_obs, partial resets across several servers), that bad
requests fail loudly, and that a Collector driven through the network gives what it gives in-process."""
import threading

import numpy as np
import pytest

from fakes import ScriptedPolicy, expected_length, make_backend
from src.envs import protocol as P
from src.envs.client import EnvConnection, EnvServerError, VectorEnvClient
from src.envs.collect import Collector, run_fixed_episodes
from src.envs.ws_server import WsEnvServer


class Running:
    def __init__(self, **kw):
        self.backend, self.sim = make_backend(**kw)
        self.server = WsEnvServer(self.backend, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        assert self.server.ready.wait(10)

    @property
    def endpoint(self):
        return ("127.0.0.1", self.server.port)

    def stop(self):
        self.server.stop()


@pytest.fixture
def servers():
    made = []

    def make(**kw):
        s = Running(**kw)
        made.append(s)
        return s
    yield make
    for s in made:
        s.stop()


def test_spec_reaches_the_client(servers):
    s = servers(num_envs=4, bank_size=12, max_steps=10, hold=2)
    with VectorEnvClient([s.endpoint]) as env:
        sp = env.spec
        assert (sp.num_envs, sp.max_steps, sp.success_hold_steps, sp.bank_size, sp.auto_reset) == (4, 10, 2, 12, True)
        assert sp.prompt == "insert the peg" and P.JOINT_POSITION in sp.obs


def test_auto_reset_over_the_wire_matches_the_prediction(servers):
    s = servers(num_envs=3, bank_size=9, max_steps=8)
    with VectorEnvClient([s.endpoint]) as env:
        c = Collector(env, ScriptedPolicy(), mode="steps")
        r = c.collect(40)
        assert len(r.episodes) >= 6
        for ep in r.episodes:
            length, term = expected_length(ep.start_id, 8)
            assert (ep.length, ep.terminated) == (length, term)
        j = P.JOINT_POSITION
        assert (r.next_obs[j][r.done][:, 1] == r.t[r.done]).all()
    assert s.sim.physics_steps == 40                                   # one physics step per client step, restores cost none


def test_partial_reset_over_the_wire_leaves_other_environments_alone(servers):
    s = servers(num_envs=4, bank_size=12, max_steps=10)
    with VectorEnvClient([s.endpoint]) as env:
        res = env.reset(seed=0)
        res = env.step(ScriptedPolicy()(res.obs))
        before = s.sim.snapshot()
        res = env.reset(env_ids=[2], bank_ids=[9])
        after = s.sim.snapshot()
        for a, b in zip(before, after):
            assert (a[[0, 1, 3]] == b[[0, 1, 3]]).all()
        assert res.start_id[2] == 9 and res.obs[P.JOINT_POSITION][2, 2] == 9


def test_two_servers_are_one_vector_with_routed_partial_resets(servers):
    a, b = servers(num_envs=2, bank_size=8, max_steps=10), servers(num_envs=3, bank_size=8, max_steps=10)
    with VectorEnvClient([a.endpoint, b.endpoint]) as env:
        assert env.num_envs == 5 and env.per_server == [2, 3]
        res = env.reset(seed=11)
        assert res.obs[P.JOINT_POSITION].shape == (5, 7) and res.start_id.shape == (5,)
        res = env.step(np.zeros((5, 8), np.float32))
        res = env.reset(env_ids=[1, 4], bank_ids=[5, 6])               # global ids 1 (server a, slot 1) and 4 (server b, slot 2)
        assert res.start_id[[1, 4]].tolist() == [5, 6]
        assert a.sim.tag[1] == 5 and b.sim.tag[2] == 6
        assert a.sim.restore_calls[-1][0].tolist() == [1] and b.sim.restore_calls[-1][0].tolist() == [2]
        res = env.reset(env_ids=[0], bank_ids=[1])                     # only server a is asked to restore
        assert len(b.sim.restore_calls[-1][0]) == 1 and b.sim.restore_calls[-1][0].tolist() == [2]    # server b: unchanged
        assert res.obs[P.JOINT_POSITION].shape == (5, 7)               # ...yet the answer covers all 5 environments


def test_servers_get_different_seeds(servers):
    a, b = servers(num_envs=2, bank_size=50), servers(num_envs=2, bank_size=50)
    with VectorEnvClient([a.endpoint, b.endpoint]) as env:
        res = env.reset(seed=0)
        assert res.start_id[:2].tolist() != res.start_id[2:].tolist()


def test_servers_with_different_tasks_are_refused(servers):
    a, b = servers(num_envs=2, max_steps=10), servers(num_envs=2, max_steps=11)
    with pytest.raises(EnvServerError, match="same task"):
        VectorEnvClient([a.endpoint, b.endpoint])


def test_fixed_episode_evaluation_over_two_servers(servers):
    a, b = servers(num_envs=2, bank_size=12, max_steps=10), servers(num_envs=2, bank_size=12, max_steps=10)
    ids = [0, 4, 3, 7, 1, 2, 11]
    with VectorEnvClient([a.endpoint, b.endpoint]) as env:
        res = run_fixed_episodes(env, ScriptedPolicy(), ids)
    for r, i in zip(res, ids):
        length, term = expected_length(i, 10)
        assert (r["length"], r["terminated"]) == (length, term)


def test_errors_are_reported_and_the_connection_survives(servers):
    s = servers(num_envs=2)
    conn = EnvConnection(*s.endpoint)
    with pytest.raises(EnvServerError, match="before reset"):
        conn.step(np.zeros((2, 8), np.float32))
    with pytest.raises(EnvServerError, match="unknown op"):
        conn.call("fly")
    with pytest.raises(EnvServerError, match="distinct"):
        conn.reset(env_ids=[5])
    conn.reset()
    assert conn.step(np.zeros((2, 8), np.float32)).obs[P.JOINT_POSITION].shape == (2, 7)
    conn.close()


def test_a_second_client_is_refused(servers):
    s = servers(num_envs=2)
    first = EnvConnection(*s.endpoint)
    with pytest.raises(Exception):
        EnvConnection(*s.endpoint, connect_timeout_s=3)
    first.close()


def test_timings_debug_op(servers):
    s = servers(num_envs=2)
    with VectorEnvClient([s.endpoint]) as env:
        env.reset()
        env.step(np.zeros((2, 8), np.float32))
        t = env.timings(reset=True)[0]
        assert t["step_physics"]["calls"] == 1
        assert env.timings()[0] == {}
