#!/bin/bash
# Start ONE Isaac Lab env server on a GPU node (L40S):   jobs/env_server.sh [--set KEY=VALUE ...]
#   e.g.  jobs/env_server.sh --set num_envs=16 --set server_port=8103
# Several servers (one per GPU): start the FIRST one alone and wait until it prints "ready": the first start builds the Robotiq
# USD in third_party/robotiq/_build/ and two simultaneous first starts would write it at the same time.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # reproducible resets (src/envs/server.py explains)
exec python -m src.envs.server --config "${ENV_CONFIG:-configs/task/peg_insert_forge.yaml}" "$@"
