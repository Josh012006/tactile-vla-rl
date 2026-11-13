#!/bin/bash
# The TRAINING loss of several checkpoints on the same demonstration rows (scripts/probe_sft_loss.py: no simulator, minutes). Why: docs/eval/sft_eval.md section 6.4.
#   sbatch jobs/job_probe_sft_loss.sh --checkpoint peg_insert_pi05_lora_teleop:0 peg_insert_pi05_lora_teleop:2000 peg_insert_pi05_lora_teleop:4000 _adopted_expoft_teleop_4000:0
# The first checkpoint is the reference of the paired differences. Check first, on the login node, that the runs and steps exist:
#   python -m scripts.probe_sft_loss --checkpoint ... --dry-run
# Every option goes to `python -m scripts.probe_sft_loss` (see its --help: --rows, --draws, --demos, --augment). Output: logs/slurm/probe_sft_loss_<jobid>.out, reports/probe_sft_loss_sft_<time>_<job id>.json.
# One policy of pi0.5 at a time (16 rows per pass): the card of the evaluation (an L40S) is more than enough. No mail (not a training job).
#
#SBATCH --job-name=probe_sft_loss
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --requeue

set -u
[ $# -eq 0 ] && { echo "usage: sbatch jobs/job_probe_sft_loss.sh --checkpoint RUN:STEP [RUN:STEP ...] [--rows 256] [--draws 4] [--demos FOLDER ...]" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.probe_sft_loss "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
