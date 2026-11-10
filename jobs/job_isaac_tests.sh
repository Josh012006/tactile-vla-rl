#!/bin/bash
# Isaac tests (fidelity of the real reset, independence of the environments, time and VRAM as a function of N) on an L40S.
#   sbatch jobs/job_isaac_tests.sh                       # everything (fidelity tests, then the cost sweep)
#   sbatch jobs/job_isaac_tests.sh tests/test_isaac_fidelity.py
#   COST_NUM_ENVS=1,4,16,64 sbatch --export=ALL jobs/job_isaac_tests.sh tests/test_isaac_cost.py
# Output: logs/slurm/isaac_tests_<jobid>.out and reports/*.json (reset_fidelity, isolation, env_cost_*).
#
#SBATCH --job-name=isaac_tests
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue

set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports
source .venv-isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TARGET=("$@"); [ ${#TARGET[@]} -gt 0 ] || TARGET=(tests)
pytest "${TARGET[@]}" --isaac -s -p no:cacheprovider
