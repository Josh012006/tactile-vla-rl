#!/bin/bash
# What does each stage of PA-RL cost? (scripts/probe_parl_cost.py: the real pi0.5 of an SFT checkpoint, synthetic states, no simulator, a few minutes after the model is loaded)
#   sbatch jobs/job_probe_parl_cost.sh --start peg_insert_pi05_lora_teleop 4000                                           # variant B (the dearer), with the saves timed in $SLURM_TMPDIR
#   sbatch jobs/job_probe_parl_cost.sh --experiment configs/experiment/rl/pi05_parl_global.yaml --start <run> <step>      # variant A
# Every option goes to `python -m scripts.probe_parl_cost` (see its --help). Output: logs/slurm/probe_parl_cost_<jobid>.out, reports/probe_parl_cost_<time>_<jobid>.json.
# Same card and XLA settings as a run (an L40S, JAX limited to half of it: the env server takes the rest). No mail (not a training job).
#
#SBATCH --job-name=probe_parl_cost
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --requeue

set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.probe_parl_cost --save-dir "${SLURM_TMPDIR:-/tmp}" "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
