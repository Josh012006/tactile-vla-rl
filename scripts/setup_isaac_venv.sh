#!/bin/bash
# Creates .venv-isaaclab: Isaac Sim 5.1 + torch + Isaac Lab (third_party/IsaacLab submodule) + what the env server needs.
# Run it ONCE, on the cluster (login node is fine, no GPU needed), from anywhere:   bash scripts/setup_isaac_venv.sh
# It is idempotent: running it again only completes what is missing. Takes 10-20 min (Isaac Sim wheels are ~10 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f third_party/IsaacLab/isaaclab.sh ] || git submodule update --init third_party/IsaacLab
[ -d third_party/robotiq ] || echo "WARNING: third_party/robotiq/ is missing (NVIDIA Robotiq USD + tactile repo): the Robotiq robots cannot be built until you copy it, see README."

[ -d .venv-isaaclab ] || uv venv --python 3.11 --seed .venv-isaaclab
source .venv-isaaclab/bin/activate

pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
(cd third_party/IsaacLab && ./isaaclab.sh --install none)
pip install numpy msgpack websockets pyyaml nvidia-ml-py pytest

python - <<'PY'
import torch, msgpack, websockets, yaml
print("torch", torch.__version__, "| cuda available on this node:", torch.cuda.is_available())
PY
echo "OK: .venv-isaaclab ready. Isaac Sim itself is only checked when a server starts (needs a GPU node)."
