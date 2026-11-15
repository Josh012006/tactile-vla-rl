"""Doubles of the RL tests: a fake pi0.5-like policy (candidates drawn from a seed, targets = joints + 0.1 x chunk) and K fake environments that follow the protocol of src/envs/protocol.py
(`reset(seed, round)`, `step(actions, auto_reset=False)`, StepResult). Environment e succeeds at its `success_at[e]`-th step, never (a time-out at MAX_STEPS) when it is larger."""
import numpy as np

from src.envs import protocol as P
from src.models.interface import Candidates, ModelInfo

H, K = 8, 4
MAX_STEPS, REPLAN = 20, 4
CFG = {"num_candidates": K, "keep_top": 2, "local_steps": 2, "local_lr": 1e-2, "rows_per_sample_call": 3, "critic_hidden": [16, 16, 16]}


def fake_targets(normalized, joints):
    out = normalized * 0.1
    joints = np.asarray(joints, np.float32)
    out[..., :7] += joints.reshape(len(joints), *([1] * (out.ndim - 2)), 7)
    return out.astype(np.float32)


class FakePolicy:
    info = ModelInfo(name="fake", family="pi05", action_horizon=H, action_dim=8, output_semantics="offsets_from_chunk_start_state")

    def __init__(self):
        self.sample_calls = []

    def sample_candidates(self, obs, prompt, num_samples, *, row_seeds=None, pad_to=None):
        joints = np.asarray(obs[P.JOINT_POSITION], np.float32)
        self.sample_calls.append(len(joints))
        norm = np.stack([np.random.default_rng(int(s)).normal(size=(num_samples, H, 8)) for s in row_seeds]).astype(np.float32)
        return Candidates(normalized=norm, states=joints[:, :3].copy(), targets=fake_targets(norm, joints), infer_ms=1.0)

    def normalized_to_targets(self, normalized, states, joints):
        return fake_targets(np.array(normalized, np.float32), joints)


class FakeSpec:
    prompt = "Insert the peg into the socket."
    max_steps = MAX_STEPS


class FakeEnv:
    """K environments; environment e succeeds on its `success_at[e]`-th step (never = times out at MAX_STEPS). The observation's joints are the last action's first 7 values."""

    spec = FakeSpec()

    def __init__(self, success_at):
        self.success_at = list(success_at)
        self.num_envs = len(success_at)
        self.received = []
        self.resets = []

    def _obs(self, q, t):
        n = self.num_envs
        img = np.zeros((n, 224, 224, 3), np.uint8)
        img[:, 0, 0, 0] = t                                                   # the step number marks the images
        return {P.EXTERIOR_IMAGE: img, P.WRIST_IMAGE: img.copy(), P.JOINT_POSITION: q.copy(), P.GRIPPER_POSITION: np.zeros((n, 1), np.float32), P.FT_FORCE: np.zeros((n, 3), np.float32),
                P.PEG_SOCKET_OFFSET: np.zeros((n, 3), np.float32), P.GRIPPER_EFFORT: np.full(n, 0.5, np.float32)}

    def reset(self, seed=0, round=0):
        self.resets.append((seed, round))
        n = self.num_envs
        self.q, self.t, self.ended = np.zeros((n, 7), np.float32), 0, np.zeros(n, bool)
        return self._result(np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool))

    def _result(self, term, trunc, acting):
        n = self.num_envs
        obs = self._obs(self.q, self.t)
        done = np.flatnonzero(term | trunc)
        return P.StepResult(obs=obs, reward=term.astype(np.float32), terminated=term, truncated=trunc, acting=acting, inserted=term, success=self.succeeded.copy() if hasattr(self, "succeeded") else np.zeros(n, bool),
                            t=np.full(n, self.t, np.int32), round_id=np.zeros(n, np.int64), final_obs={k: v[done] for k, v in obs.items()}, final_ids=done.astype(np.int64),
                            all_done=bool(self.ended.all()))

    def step(self, actions, auto_reset=False):
        self.received.append(np.array(actions))
        acting = ~self.ended
        self.q = np.where(acting[:, None], actions[:, :7], self.q).astype(np.float32)
        self.t += 1
        term = acting & (np.asarray(self.success_at) == self.t)
        trunc = acting & ~term & (self.t >= MAX_STEPS)
        self.succeeded = getattr(self, "succeeded", np.zeros(self.num_envs, bool)) | term
        self.ended = self.ended | term | trunc
        return self._result(term, trunc, acting)
