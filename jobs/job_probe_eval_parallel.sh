#!/bin/bash
# Probe of the parallel evaluation: how many checkpoints one GPU evaluates at the same time, what that costs, and whether the results stay bit-identical.
#   sbatch jobs/job_probe_eval_parallel.sh --model pi05 --workers 1 2 3 --episodes 4
#   sbatch jobs/job_probe_eval_parallel.sh --model pi0_fast --workers 1 2 3 --episodes 4
#   sbatch --gres=gpu:a100l:1 jobs/job_probe_eval_parallel.sh --model pi05 --workers 1 2 4 6 --episodes 4        # a bigger card
# Every option goes to `python -m scripts.probe_eval_parallel` (see its --help). It needs no SFT run: the policy is the model's initial state.
# For each worker count W it evaluates W identical checkpoints at the same time with the real code path of the evaluation, then cuts one evaluation after its first episodes and
# resumes it. Output: logs/slurm/probe_eval_parallel_<jobid>.out, reports/probe_eval_parallel_<time>.json, the data in runs/sft/probing/_probe_eval_parallel_<time>/.
# Time: each worker count costs the start of W env servers (one after the other, 2 to 3 minutes each) and the episodes (about 35 s each with one worker): an hour or two for 1 2 3.
# Use the same kind of GPU as the evaluation you are choosing --workers for.
#
#SBATCH --job-name=probe_eval_parallel
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue

set -u
[ $# -eq 0 ] && { echo "usage: sbatch jobs/job_probe_eval_parallel.sh --model <pi05 | pi0_fast> [--workers 1 2 3] [--episodes N]" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports runs/sft/probing
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # same settings as jobs/job_eval_sft.sh: what is measured is what the evaluation will do
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.probe_eval_parallel "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
