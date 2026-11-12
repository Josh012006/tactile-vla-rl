#!/bin/bash
# OBSOLETE since the protocol batched-v1 (K environments per server): the probe still runs with one environment per server (--envs-per-server is not an option of this probe); see jobs/job_probe_batch_cost.sh and docs/eval/sft_eval.md section 6.1.
# Probe of the parallel evaluation: how many checkpoints one GPU evaluates at the same time, what that costs, and whether the results stay bit-identical.
#   sbatch jobs/job_probe_eval_parallel.sh --model pi05 --workers 1 2 --episodes 4
#   sbatch jobs/job_probe_eval_parallel.sh --model pi0_fast --workers 1 2 --episodes 4
#   sbatch jobs/job_probe_eval_parallel.sh --model pi05 --workers 1 --episodes 4 --no-resume-check --no-deterministic-xla      # what XLA's deterministic ops cost (not an evaluation setting)
# Every option goes to `python -m scripts.probe_eval_parallel` (see its --help). It needs no SFT run: the policy is the model's initial state.
# For each worker count W it evaluates W identical checkpoints at the same time with the real code path of the evaluation, then cuts one evaluation after its first episodes and
# resumes it. Output: logs/slurm/probe_eval_parallel_<jobid>.out, reports/probe_eval_parallel_<model>_<time>_<jobid>.json, the data in
# runs/sft/probing/_probe_eval_parallel_<model>_<time>_<jobid>/ (one folder per probe: two probes never share one).
# Time: each worker count costs the start of W env servers (2 to 3 minutes each) and the episodes (about 90-110 s each with one worker, measured): about 35 minutes for 1 2.
# Use the same kind of GPU as the evaluation you are choosing --workers for. XLA's deterministic ops are set by the script (not here) so that --no-deterministic-xla can measure without them.
#
#SBATCH --job-name=probe_eval_parallel
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue

set -u
[ $# -eq 0 ] && { echo "usage: sbatch jobs/job_probe_eval_parallel.sh --model <pi05 | pi0_fast> [--workers 1 2] [--episodes N]" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports runs/sft/probing
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # same settings as jobs/job_eval_sft.sh: what is measured is what the evaluation will do
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.probe_eval_parallel "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
