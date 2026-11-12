#!/bin/bash
# What does the model cost when it plans for K environments at once? (scripts/probe_batch_cost.py: no simulator, a few minutes)
#   sbatch jobs/job_probe_batch_cost.sh --model pi05 --batch-sizes 1 8 32 64 100
#   sbatch jobs/job_probe_batch_cost.sh --model pi0_fast --batch-sizes 1 8 32 64 100
# Every option goes to `python -m scripts.probe_batch_cost` (see its --help). Output: logs/slurm/probe_batch_cost_<jobid>.out, reports/probe_batch_cost_<model>_<tag>.json.
# Use the GPU of the evaluation (an L40S). No mail (not a training job).
#
#SBATCH --job-name=probe_batch_cost
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --requeue

set -u
[ $# -eq 0 ] && { echo "usage: sbatch jobs/job_probe_batch_cost.sh --model <pi05 | pi0_fast> [--batch-sizes 1 8 32 ...]" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # same settings as jobs/job_eval_sft.sh: what is checked is what the evaluation does
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.probe_batch_cost "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
