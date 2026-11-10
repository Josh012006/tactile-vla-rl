"""Client <-> WebSocket server <-> ManagerBackend <-> fake simulator, over real sockets. Pure (no GPU).

Checks that the protocol carries everything (rounds, final_obs, replay by (seed, round), several servers as one vector), that
bad requests fail loudly, and that a Collector driven through the network gives what it gives in-process."""
import threading

import numpy as np
import pytest

from fakes import ScriptedPolicy, expected_length, make_backend, tag_of
from src.envs import protocol as P
from src.envs.client import SEED_STRIDE, EnvConnection, EnvServerError, VectorEnvClient
from src.envs.collect import Collector, run_fixed_rounds
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
    s = servers(num_envs=4, max_steps=10, hold=2)
    with VectorEnvClient([s.endpoint]) as env:
        sp = env.spec
        assert (sp.num_envs, sp.max_steps, sp.success_hold_steps, sp.auto_reset) == (4, 10, 2, True)
        assert sp.prompt == "insert the peg" and P.JOINT_POSITION in sp.obs


def test_rounds_over_the_wire_match_the_prediction(servers):
    s = servers(num_envs=3, max_steps=8, seed=2)
    with VectorEnvClient([s.endpoint]) as env:
        c = Collector(env, ScriptedPolicy(), mode="steps")
        r = c.collect(40)
        assert len(r.episodes) >= 6
        for ep in r.episodes:
            length, term = expected_length(tag_of(2 + ep.round, ep.env), 8)
            assert (ep.length, ep.terminated) == (length, term)
        j = P.JOINT_POSITION
        assert (r.next_obs[j][r.done][:, 1] == r.t[r.done]).all()
    assert s.sim.physics_steps == 40                                   # one physics step per client step, a reset costs none here
    assert s.sim.reset_calls == [2 + k for k in range(len(s.sim.reset_calls))]      # consecutive rounds of one seed


def test_reset_with_seed_and_round_replays_the_same_start(servers):
    s = servers(num_envs=4, max_steps=10)
    with VectorEnvClient([s.endpoint]) as env:
        a = env.reset(seed=11, round=5)
        env.step(np.zeros((4, 8), np.float32))
        env.reset(seed=3)
        b = env.reset(seed=11, round=5)
        assert (a.obs[P.JOINT_POSITION] == b.obs[P.JOINT_POSITION]).all() and (a.round_id == 5).all() and (b.round_id == 5).all()
        c = env.reset()                                                # the next round of seed 11
        assert (c.round_id == 6).all() and s.sim.reset_calls[-3:] == [3, 16, 17]


def test_two_servers_are_one_vector_each_with_its_own_rounds(servers):
    a, b = servers(num_envs=2, max_steps=6), servers(num_envs=3, max_steps=6)
    with VectorEnvClient([a.endpoint, b.endpoint]) as env:
        assert env.num_envs == 5 and env.per_server == [2, 3]
        res = env.reset(seed=11)
        assert res.obs[P.JOINT_POSITION].shape == (5, 7) and res.round_id.shape == (5,)
        assert (res.obs[P.JOINT_POSITION][:2, 2] == [tag_of(11, e) for e in range(2)]).all()
        assert (res.obs[P.JOINT_POSITION][2:, 2] == [tag_of(11 + SEED_STRIDE, e) for e in range(3)]).all()
        policy = ScriptedPolicy()
        for _ in range(40):
            res = env.step(policy(res.obs))
            assert len(set(res.round_id[:2])) == 1 and len(set(res.round_id[2:])) == 1       # a round is per server
        assert a.sim.reset_calls[0] == 11 and b.sim.reset_calls[0] == 11 + SEED_STRIDE
        assert len(a.sim.reset_calls) == res.round_id[0] + 1 and len(b.sim.reset_calls) == res.round_id[2] + 1


def test_servers_get_different_seeds(servers):
    a, b = servers(num_envs=2), servers(num_envs=2)
    with VectorEnvClient([a.endpoint, b.endpoint]) as env:
        env.reset(seed=0)
        assert a.sim.reset_calls != b.sim.reset_calls


def test_servers_with_different_tasks_are_refused(servers):
    a, b = servers(num_envs=2, max_steps=10), servers(num_envs=2, max_steps=11)
    with pytest.raises(EnvServerError, match="same task"):
        VectorEnvClient([a.endpoint, b.endpoint])


def test_fixed_round_evaluation_over_two_servers(servers):
    a, b = servers(num_envs=2, max_steps=10), servers(num_envs=2, max_steps=10)
    with VectorEnvClient([a.endpoint, b.endpoint]) as env:
        res = run_fixed_rounds(env, ScriptedPolicy(), [0, 4, 3], seed=1)
    assert [(r["round"], r["env"]) for r in res] == [(r, e) for r in (0, 4, 3) for e in range(4)]
    for r in res:
        seed = 1 + (0 if r["env"] < 2 else SEED_STRIDE) + r["round"]
        length, term = expected_length(tag_of(seed, r["env"] % 2), 10)
        assert (r["length"], r["terminated"]) == (length, term)


def test_errors_are_reported_and_the_connection_survives(servers):
    s = servers(num_envs=2)
    conn = EnvConnection(*s.endpoint)
    with pytest.raises(EnvServerError, match="before reset"):
        conn.step(np.zeros((2, 8), np.float32))
    with pytest.raises(EnvServerError, match="unknown op"):
        conn.call("fly")
    with pytest.raises(EnvServerError, match="seed"):
        conn.reset(seed=-4)
    with pytest.raises(EnvServerError, match="round"):
        conn.reset(seed=0, round=P.MAX_ROUND)
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
        assert t["reset"]["calls"] == 1 and t["step_physics"]["calls"] == 1
        assert env.timings()[0] == {}
