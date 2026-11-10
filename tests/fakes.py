"""A fake simulator for the tests that do not need Isaac Sim, behind the same SimAdapter as the real one.

Per-environment state: q0 (the last absolute joint-0 target = "position"), clock (physics steps since the round started) and
tag (identifies the start state of the environment in its round). Behaviour:
  * reset(seed): the REAL reset of every environment, reproducible from `seed` alone: tag = tag_of(seed, env), q0 from the tag,
                 clock 0. The seeds it was called with are recorded (`reset_calls`);
  * step_physics: q0 <- action[0], clock += 1 for EVERY environment (a physics step is global, like the real one);
  * inserted    : q0 >= 1.0.
`expected_length(tag, max_steps, hold)` is the episode length the scripted policy below produces from a given start state, so
every test can predict the exact outcome of an episode.
"""
from __future__ import annotations

import numpy as np

from src.envs import protocol as P
from src.envs.backend import ManagerBackend
from src.envs.episode_manager import VectorEnvManager

CAMERA = 4
NEVER = 10 ** 9


def tag_of(seed: int, env: int) -> int:
    """The start state the fake reset gives environment `env` for `seed` (the real one: a function of seed, num_envs, env)."""
    return 3 * int(seed) + int(env)


def trigger(tag: int) -> int:
    """Clock value at which the scripted policy starts pushing q0 to 1.0 from start state `tag` (NEVER for tag % 4 == 3)."""
    return NEVER if tag % 4 == 3 else 2 + (tag % 5)


def expected_length(tag: int, max_steps: int, hold: int = 1):
    """(length, terminated) of an episode started from `tag` and driven by `ScriptedPolicy`."""
    trig = trigger(tag)
    length = trig + hold             # the first push is seen at clock == trig, inserted from the step after that observation
    return (length, True) if length <= max_steps else (max_steps, False)


class FakeSim:
    def __init__(self, num_envs: int, with_tactile: bool = False):
        self.num_envs = num_envs
        self.with_tactile = with_tactile
        self.q0 = np.zeros(num_envs, np.float32)
        self.clock = np.zeros(num_envs, np.int64)
        self.tag = np.full(num_envs, -1, np.int64)
        self.physics_steps = 0
        self.reset_calls = []                # seed of every real reset
        self.observe_calls = 0

    # -- observations ------------------------------------------------------------------------------------------------------
    def obs_of(self, q0, clock, tag):
        n = len(q0)
        joint = np.zeros((n, 7), np.float32)
        joint[:, 0], joint[:, 1], joint[:, 2] = q0, clock, tag          # joint[:,1]=clock and [:,2]=tag are readable by policies
        obs = {
            P.EXTERIOR_IMAGE: np.broadcast_to((tag % 251).astype(np.uint8)[:, None, None, None], (n, CAMERA, CAMERA, 3)).copy(),
            P.WRIST_IMAGE: np.broadcast_to((clock % 251).astype(np.uint8)[:, None, None, None], (n, CAMERA, CAMERA, 3)).copy(),
            P.JOINT_POSITION: joint, P.GRIPPER_POSITION: np.zeros((n, 1), np.float32),
            P.FT_FORCE: np.zeros((n, 3), np.float32), P.FORCE_THRESHOLD: np.zeros(n, np.float32),
            P.PEG_SOCKET_OFFSET: np.zeros((n, 3), np.float32),
        }
        if self.with_tactile:
            obs[P.TACTILE] = np.zeros((n, 2, 7, 4), np.float32) + clock[:, None, None, None].astype(np.float32)
        return obs

    def obs_spec(self):
        return P.build_obs_spec(CAMERA, gripper_effort=False, tactile=self.with_tactile)

    # -- SimAdapter --------------------------------------------------------------------------------------------------------
    def reset(self, seed):
        self.reset_calls.append(int(seed))
        self.tag = np.array([tag_of(seed, e) for e in range(self.num_envs)], np.int64)
        self.q0 = (self.tag % 3).astype(np.float32) * 0.1
        self.clock = np.zeros(self.num_envs, np.int64)

    def step_physics(self, actions):
        self.q0 = actions[:, 0].astype(np.float32).copy()
        self.clock = self.clock + 1
        self.physics_steps += 1

    def inserted(self):
        return self.q0 >= 1.0

    def observe(self):
        self.observe_calls += 1
        return self.obs_of(self.q0, self.clock, self.tag)

    def hold_actions(self):
        a = np.zeros((self.num_envs, P.ACTION_DIM), np.float32)
        a[:, 0] = self.q0
        return a

    def snapshot(self):
        return self.q0.copy(), self.clock.copy(), self.tag.copy()


class ScriptedPolicy:
    """Pushes joint 0 to 1.0 once the environment's clock reached the trigger of its start state, else holds it at 0."""

    def __call__(self, obs):
        clock, tag = obs[P.JOINT_POSITION][:, 1], obs[P.JOINT_POSITION][:, 2]
        a = np.zeros((len(clock), P.ACTION_DIM), np.float32)
        push = np.array([c >= trigger(int(t)) for c, t in zip(clock, tag)])
        a[:, 0] = np.where(push, 1.0, 0.0)
        return a


def make_manager(num_envs=4, max_steps=10, hold=1, auto_reset=True, seed=0, with_tactile=False):
    sim = FakeSim(num_envs, with_tactile)
    mgr = VectorEnvManager(sim, max_steps=max_steps, success_hold_steps=hold, auto_reset=auto_reset, seed=seed)
    return mgr, sim


def make_backend(num_envs=4, max_steps=10, hold=1, auto_reset=True, seed=0):
    mgr, sim = make_manager(num_envs, max_steps, hold, auto_reset, seed)
    return ManagerBackend(mgr, sim.obs_spec(), prompt="insert the peg", control_hz=15.0, robot={"name": "fake"}), sim
