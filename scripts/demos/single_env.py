"""ONE environment of an env server, with the small API the demo scripts were written for.

The scripts of this folder (collect_waypoint_demos, replay_demos, robot_check) drive a single robot, one episode at a time: the old
expo-ft server had exactly one. The new server owns `num_envs` environments (src/envs/protocol.py); these scripts need a server started
with ONE of them, and this class hides the vector protocol:

    env = SingleEnv("localhost", 8102, hold_steps=5, gripper_can_open=False)    # checks the server, exits with the fix if it is wrong
    obs = env.reset(seed)               # observation of the first frame: {key: array of ONE environment}
    env.step(action)                    # action: (8,) = 7 absolute joint targets + gripper command
    obs = env.observation()             # observation after the step: with auto_reset off it is the real last one, even when the episode ended
    st = env.status()                   # done / inserted / terminated / truncated / success / t of that step
    env.debug("finger_width")           # simulation-only ops (the rows of environment 0)

`reset(seed)` is round 0 of that seed (the start state is a function of (seed, num_envs, the episode that ran before: see
docs/isaaclab_env.md)); the server must not auto-reset: every step is sent with auto_reset=False, so nothing starts behind the script's back.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional

import numpy as np

from src.envs.client import VectorEnvClient

START_COMMAND = "bash jobs/env_server.sh --set num_envs=1"


class Status(NamedTuple):
    done: bool            # the episode ended on this step (success or timeout)
    inserted: bool        # FORGE's own "inserted" flag at this step
    terminated: bool      # ended by success (the server saw `inserted` for success_hold_steps steps in a row)
    truncated: bool       # ended by timeout (max_steps steps without success)
    success: bool
    t: int


def _row0(x: Any) -> Any:
    """The row of environment 0 of what a debug op returned: arrays (1, ...) -> (...), dicts element by element, the rest as is."""
    if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == 1:
        return x[0]
    if isinstance(x, dict):
        return {k: _row0(v) for k, v in x.items()}
    return x


class SingleEnv:
    def __init__(self, host: str = "localhost", port: int = 8102, *, hold_steps: Optional[int] = None,
                 gripper_can_open: Optional[bool] = None):
        """hold_steps / gripper_can_open: what the script needs the server to be set to (None = no requirement). A server that is not
        exits with the command that starts a right one."""
        self.client = VectorEnvClient([(host, port)])
        self.spec = self.client.spec
        problems = []
        if self.spec.num_envs != 1:
            problems.append(f"it runs {self.spec.num_envs} environments (these scripts drive one): --set num_envs=1")
        if hold_steps is not None and self.spec.success_hold_steps != hold_steps:
            problems.append(f"its success_hold_steps is {self.spec.success_hold_steps}, the script counts {hold_steps} 'inserted' steps in a row "
                            f"and the server would end the episode earlier or later than it: --set success_hold_steps={hold_steps}")
        if gripper_can_open is not None and bool(self.spec.robot.get("gripper_can_open", True)) != gripper_can_open:
            problems.append(f"its gripper_can_open is {not gripper_can_open}, the script needs {gripper_can_open}: "
                            f"--set gripper_can_open={str(gripper_can_open).lower()}")
        if problems:
            raise SystemExit(f"the env server at {host}:{port} is not set up for this script:\n  - " + "\n  - ".join(problems)
                             + f"\nRestart it, e.g. `{START_COMMAND} ...` (jobs/env_server.sh passes --set to the server).")
        self.prompt, self.max_steps = self.spec.prompt, self.spec.max_steps
        self._res = None

    def reset(self, seed: int) -> dict:
        self._res = self.client.reset(seed=int(seed))
        return self.observation()

    def step(self, action) -> None:
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        self._res = self.client.step(action, auto_reset=False)

    def observation(self) -> dict:
        if self._res is None:
            raise RuntimeError("reset() first")
        return {k: np.array(v[0]) for k, v in self._res.obs.items()}

    def status(self) -> Status:
        r = self._res
        return Status(bool(r.done[0]), bool(r.inserted[0]), bool(r.terminated[0]), bool(r.truncated[0]), bool(r.success[0]), int(r.t[0]))

    def debug(self, kind: str, **kwargs) -> Any:
        result = _row0(self.client.debug(kind, **kwargs)[0])
        if isinstance(result, dict) and set(result) == {"error"}:
            raise RuntimeError(f"{result['error']}  (is the env server an old version? restart it from the current code)")
        return result

    def close(self) -> None:
        self.client.close()
