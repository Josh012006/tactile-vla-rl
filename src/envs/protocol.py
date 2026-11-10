"""Contract between the Isaac Lab environment server and its clients.

No Isaac Sim / torch import here: the learner process (JAX) imports this file too.

One server owns `num_envs` INDEPENDENT environments (one Isaac Lab scene, one process), driven in ROUNDS:

    round       FORGE's own reset acts on all the environments together and steps the physics, so the environments of a
                server always start their episodes TOGETHER, with the real reset. That is a round. Each environment then runs
                its own episode (it succeeds or times out on its own clock); an environment whose episode ended is held in
                place (`acting` is False for it) until the others have ended too, then the next round starts.
    reset(seed, round)   start a round. `seed` fixes the whole sequence of rounds, `round` the position in it: the start state of
                every environment is a function of (seed, round, num_envs) up to a small dependence on the episode that ran
                just before (Robotiq: peg <= 0.14 mm in the measurements, see tests/test_isaac_fidelity.py). The same SEQUENCE
                of resets and steps from a fresh server is bit-identical. Without arguments: the next round. With `seed` only: round 0.
    step(actions)        step ALL environments once. (num_envs, ACTION_DIM) actions in, everything back in ONE message. When the
                last running environment ends on this step and auto_reset is on, the next round starts inside the same call:
                the returned `obs` is the first observation of the new round, the real last observation of every episode that
                ended on this step is in `final_obs` (rows `final_ids`). Nothing is ever lost or stale.

Episode semantics (applied per environment by the server):
    inserted     FORGE's own flag at this step (xy < 2.5 mm and z below its threshold).
    reward       1.0 where `inserted`, else 0.0 (sparse, ground truth).
    terminated   the episode ended by SUCCESS: `inserted` for `success_hold_steps` consecutive steps (1 = first time).
    truncated    the episode ended by TIMEOUT: `max_steps` steps without success. Not a terminal state for bootstrapping.
    success      this episode has succeeded (cumulative, of the episode that just ended when `done`).
    t            steps of the episode so far (of the ended episode when `done`).
    acting       the environment's episode was running BEFORE the step: the row is a valid transition. False = the environment
                 ended earlier in this round and waits for the others (its row must be ignored).
    round_id     the round `obs` belongs to (after a possible new round: it describes `obs`; the episodes that just ended
                 belong to the `round_id` of the previous step).

`auto_reset=False` (spec default or per step): when every environment has ended, nothing starts by itself: all_done is True,
a step is a no-op (no physics step, `acting` False everywhere) until the client calls reset. This is how a FIXED set of
evaluation episodes is run: reset(seed, round) always gives the same starts (collect.run_fixed_rounds).

Wire format: one msgpack message per request / response (see wire.py), a dict with an "op" key
(spec | reset | step | observe | debug) and, in the response, "status": "ok" | "error".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

PROTOCOL_VERSION = 3
MAX_ROUND = 1_000_000   # rounds per seed (the seed of a round is seed + round; servers of one client are SEED_STRIDE apart)
ACTION_DIM = 8      # 7 ABSOLUTE joint-position targets (rad) + gripper command (0 = open, 1 = closed)

# Observation keys. Kept identical to the DROID / pi05 convention (image + state) and to the old env server (extra/*), so
# the model adapters, the demonstrations and the old analysis scripts keep working.
EXTERIOR_IMAGE = "observation/exterior_image_1_left"
WRIST_IMAGE = "observation/wrist_image_left"
JOINT_POSITION = "observation/joint_position"
GRIPPER_POSITION = "observation/gripper_position"
FT_FORCE = "extra/ft_force"
FORCE_THRESHOLD = "extra/force_threshold"
PEG_SOCKET_OFFSET = "extra/peg_socket_offset"
GRIPPER_EFFORT = "extra/gripper_effort"     # robots with a motorised finger joint (Robotiq) only
TACTILE = "extra/tactile"                   # robots with taxels only: (2, 7, 4) N, [0] left finger, [1] right, row 0 = fingertip

ObsSpec = Dict[str, Tuple[Tuple[int, ...], str]]    # key -> (shape of ONE environment, numpy dtype name)


def build_obs_spec(camera_size: int, gripper_effort: bool, tactile: bool, tactile_shape=(2, 7, 4)) -> ObsSpec:
    """Per-environment shape and dtype of every observation key a server sends."""
    spec: ObsSpec = {
        EXTERIOR_IMAGE: ((camera_size, camera_size, 3), "uint8"),
        WRIST_IMAGE: ((camera_size, camera_size, 3), "uint8"),
        JOINT_POSITION: ((7,), "float32"),
        GRIPPER_POSITION: ((1,), "float32"),
        FT_FORCE: ((3,), "float32"),
        FORCE_THRESHOLD: ((), "float32"),
        PEG_SOCKET_OFFSET: ((3,), "float32"),
    }
    if gripper_effort:
        spec[GRIPPER_EFFORT] = ((), "float32")
    if tactile:
        spec[TACTILE] = (tuple(tactile_shape), "float32")
    return spec


@dataclass(frozen=True)
class EnvSpec:
    """What a server tells a client when it connects (op `spec`)."""
    protocol_version: int
    num_envs: int
    action_dim: int
    max_steps: int
    success_hold_steps: int
    control_hz: float
    prompt: str
    auto_reset: bool
    obs: ObsSpec
    robot: Dict[str, Any]

    def to_wire(self) -> dict:
        return {
            "protocol_version": self.protocol_version, "num_envs": self.num_envs, "action_dim": self.action_dim,
            "max_steps": self.max_steps, "success_hold_steps": self.success_hold_steps, "control_hz": self.control_hz,
            "prompt": self.prompt, "auto_reset": self.auto_reset, "robot": self.robot,
            "obs": {k: {"shape": list(shape), "dtype": dtype} for k, (shape, dtype) in self.obs.items()},
        }

    @staticmethod
    def from_wire(d: dict) -> "EnvSpec":
        return EnvSpec(
            protocol_version=int(d["protocol_version"]), num_envs=int(d["num_envs"]), action_dim=int(d["action_dim"]),
            max_steps=int(d["max_steps"]), success_hold_steps=int(d["success_hold_steps"]),
            control_hz=float(d["control_hz"]), prompt=str(d["prompt"]), auto_reset=bool(d["auto_reset"]),
            robot=dict(d["robot"]),
            obs={k: (tuple(int(n) for n in v["shape"]), str(v["dtype"])) for k, v in d["obs"].items()},
        )

    def same_task_as(self, other: "EnvSpec") -> bool:
        """True when two servers can be driven together as one vector of environments."""
        keys = ("protocol_version", "action_dim", "max_steps", "success_hold_steps", "control_hz", "prompt",
                "auto_reset", "obs")
        return all(getattr(self, k) == getattr(other, k) for k in keys)


@dataclass
class StepResult:
    """Everything one `step` (or `reset`) returns. Arrays have the environment index first, except `final_*`."""
    obs: Dict[str, np.ndarray]          # (N, ...) observation each environment acts on next
    reward: np.ndarray                  # (N,) float32
    terminated: np.ndarray              # (N,) bool   episode ended by success on this step
    truncated: np.ndarray               # (N,) bool   episode ended by timeout on this step
    acting: np.ndarray                  # (N,) bool   the env's episode was running BEFORE the step: the row is a valid transition
    inserted: np.ndarray                # (N,) bool
    success: np.ndarray                 # (N,) bool
    t: np.ndarray                       # (N,) int32
    round_id: np.ndarray                # (N,) int64  round `obs` belongs to (the same for every row)
    final_obs: Dict[str, np.ndarray]    # (n_final, ...) last observation of the episodes that ended on this step
    final_ids: np.ndarray               # (n_final,) int64  which environments those rows are
    all_done: bool                      # no environment is running (only possible with auto_reset=False)

    @property
    def done(self) -> np.ndarray:
        """(N,) bool: the episode ended on this step (success or timeout)."""
        return self.terminated | self.truncated

    def next_obs(self) -> Dict[str, np.ndarray]:
        """The TRUE next observation of every transition: `obs`, with the rows of the episodes that ended on this step replaced
        by their final observation (the one a value function must bootstrap from / the one the last action led to)."""
        out = {}
        for k, v in self.obs.items():
            row = np.array(v)
            if len(self.final_ids):
                row[self.final_ids] = self.final_obs[k]
            out[k] = row
        return out

    @staticmethod
    def from_wire(d: dict) -> "StepResult":
        return StepResult(
            obs=dict(d["obs"]), reward=np.asarray(d["reward"]), terminated=np.asarray(d["terminated"]),
            truncated=np.asarray(d["truncated"]), acting=np.asarray(d["acting"]), inserted=np.asarray(d["inserted"]),
            success=np.asarray(d["success"]), t=np.asarray(d["t"]), round_id=np.asarray(d["round_id"]),
            final_obs=dict(d["final_obs"]), final_ids=np.asarray(d["final_ids"], dtype=np.int64),
            all_done=bool(d["all_done"]))


def concat_results(parts: Sequence[StepResult], sizes: Sequence[int]) -> StepResult:
    """Stack the results of several servers (environment axis) into one; `sizes[k]` = number of envs of server k."""
    if len(parts) == 1:
        return parts[0]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts], axis=0)
    return StepResult(
        obs={k: np.concatenate([p.obs[k] for p in parts], axis=0) for k in parts[0].obs},
        reward=cat("reward"), terminated=cat("terminated"), truncated=cat("truncated"), acting=cat("acting"),
        inserted=cat("inserted"), success=cat("success"), t=cat("t"), round_id=cat("round_id"),
        final_obs={k: np.concatenate([p.final_obs[k] for p in parts], axis=0) for k in parts[0].final_obs},
        final_ids=np.concatenate([p.final_ids + off for p, off in zip(parts, offsets)]).astype(np.int64),
        all_done=all(p.all_done for p in parts))
