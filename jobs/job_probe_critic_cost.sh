#!/bin/bash
# What does the critic's update cost? (scripts/probe_critic_cost.py: no VLA, no simulator, a few minutes)
#   sbatch jobs/job_probe_critic_cost.sh                                                      # the three methods, 50 timed updates each
#   sbatch jobs/job_probe_critic_cost.sh --variants parl_global_local --steps 200 --set critic_use_tactile=true
# Every option goes to `python -m scripts.probe_critic_cost` (see its --help). Output: logs/slurm/probe_critic_cost_<jobid>.out, reports/probe_critic_cost_<time>_<jobid>.json.
# Use the GPU of the runs (an L40S). No mail (not a training job).
#
#SBATCH --job-name=probe_critic_cost
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --requeue

set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # same settings as the other probes: the critic shares the card with the VLA (half of it)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
python -m scripts.probe_critic_cost "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
