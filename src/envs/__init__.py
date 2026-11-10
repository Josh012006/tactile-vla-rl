"""Isaac Lab environments (FORGE tasks) served to the learner over a WebSocket.

Importing this package pulls in ONLY the client side (numpy, msgpack, websockets): the learner's JAX environment needs no
Isaac Sim. The simulation side (forge_env.py, forge_backend.py, start_bank.py, server.py, robot_assets.py) imports Isaac Lab
and is run as `python -m src.envs.server` inside the Isaac Lab virtualenv.
"""
from .client import EnvConnection, EnvServerError, VectorEnvClient
from .collect import Collector, EpisodeInfo, Rollout, Trajectory, run_fixed_episodes
from .protocol import ACTION_DIM, PROTOCOL_VERSION, EnvSpec, StepResult

__all__ = ["ACTION_DIM", "PROTOCOL_VERSION", "Collector", "EnvConnection", "EnvServerError", "EnvSpec", "EpisodeInfo",
           "Rollout", "StepResult", "Trajectory", "VectorEnvClient", "run_fixed_episodes"]
