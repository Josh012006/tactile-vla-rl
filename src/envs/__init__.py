"""Isaac Lab environments (FORGE tasks) served to the learner over a WebSocket.

Importing this package pulls in ONLY the client side (numpy, msgpack, websockets): the learner's JAX environment needs no
Isaac Sim. The simulator-specific side lives in subpackages (src/envs/isaaclab/: FORGE env, robots, USD assets and the server,
run as `python -m src.envs.isaaclab.server` inside the Isaac Lab virtualenv); what is here is independent of the simulator.
"""
from .client import EnvConnection, EnvServerError, VectorEnvClient
from .collect import Collector, EpisodeInfo, Rollout, Trajectory, run_fixed_rounds
from .protocol import ACTION_DIM, PROTOCOL_VERSION, EnvSpec, StepResult

__all__ = ["ACTION_DIM", "PROTOCOL_VERSION", "Collector", "EnvConnection", "EnvServerError", "EnvSpec", "EpisodeInfo",
           "Rollout", "StepResult", "Trajectory", "VectorEnvClient", "run_fixed_rounds"]
