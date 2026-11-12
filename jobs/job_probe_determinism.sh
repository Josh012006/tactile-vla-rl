#!/bin/bash
# Where does the non-reproducibility of an evaluation come from, the model or the simulator? (docs/eval/sft_eval.md section 3)
#   sbatch jobs/job_probe_determinism.sh --part policy --model pi05 --processes 2          # the model alone: no Isaac, a few minutes
#   sbatch jobs/job_probe_determinism.sh --part policy --model pi0_fast --processes 2
#   sbatch jobs/job_probe_determinism.sh --part env --servers 3                            # the simulator alone, 150 scripted steps on fresh servers: about 20 minutes
#   sbatch jobs/job_probe_determinism.sh --part policy --model pi05 --no-deterministic-xla  # what XLA's deterministic ops change
# Every option goes to `python -m scripts.probe_determinism` (see its --help). Output: logs/slurm/probe_determinism_<jobid>.out, reports/probe_determinism_<part>_<tag>.json, the data in
# runs/sft/probing/_probe_determinism_<tag>/ (--part env). No mail (not a training job). The exit code is 1 when something is not bit-reproducible: that is the result, not a failure.
# Use the same kind of GPU as the evaluation (the physics is bit-identical only on the same one).
#
#SBATCH --job-name=probe_determinism
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --requeue

set -u
[ $# -eq 0 ] && { echo "usage: sbatch jobs/job_probe_determinism.sh --part <policy | env> [--model <pi05 | pi0_fast>] [--servers N] [--processes N]" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports runs/sft/probing
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # same settings as jobs/job_eval_sft.sh: what is checked is what the evaluation does
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.probe_determinism "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
