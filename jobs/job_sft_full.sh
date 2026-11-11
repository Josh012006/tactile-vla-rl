#!/bin/bash
# FULL fine-tuning (use_lora false in every component: every weight trained, the exact base function at the start) of pi0.5 or pi0-FAST on
# the FORGE demonstrations, on an A100 80 GB. LoRA runs on an L40S: jobs/job_sft_lora.sh.
#   sbatch jobs/job_sft_full.sh pi05
#   sbatch jobs/job_sft_full.sh pi0_fast --demos demos/isaaclab/ForgePegInsert_Robotiq_teleop --run-name pi0fast_teleop_full
#   sbatch jobs/job_sft_full.sh pi05 --set micro_batch_size=4          # if it runs out of memory
#   sbatch jobs/job_sft_full.sh pi05 --resume <run name>               # continue a run that stopped: its own settings are used, none is added here
# Everything after the model name goes to `python -m scripts.sft` (see its --help). Added here, except with --resume: `--full`, a micro-batch of 8 and
# a resume snapshot every 1000 steps (40 GB each, see configs/algo/sft.yaml). A later --set wins.
# Memory is NOT measured yet: float32 weights + gradients + AdamW state is about 16 bytes per parameter (3.4 G parameters for pi0.5 = 54 GB, 2.9 G
# for pi0-FAST = 47 GB) before the activations. The `first_step` and `train` lines of metrics.jsonl give the real numbers (memory_mib).
# Before the first run, on the login node: see jobs/job_sft_lora.sh.
# Output: runs/sft/<run name>/ (a checkpoint holds the whole model in bfloat16: ~6.7 GB for pi0.5, 5.8 GB for pi0-FAST; a save every 200 steps is 20
# of them: raise save_interval with `--set save_interval=1000` when the disk matters; resume/ holds one snapshot, 35-40 GB, 70-80 GB while it is swapped),
# logs/slurm/sft_full_<jobid>.out
# Time limit: 10 minutes before it, Slurm sends a SIGTERM; the run writes its snapshot after the current step and stops cleanly (writing 40 GB takes minutes).
#
#SBATCH --job-name=sft_full
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue

set -u
MODEL="${1:-}"
[ -z "$MODEL" ] || [[ "$MODEL" == --* ]] && { echo "usage: sbatch jobs/job_sft_full.sh <model: pi05 | pi0_fast> [scripts.sft options]" >&2; exit 1; }
shift
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm runs/sft
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DEFAULTS="--full --set micro_batch_size=8 --set resume_interval=1000"
for arg in "$@"; do [ "$arg" = "--resume" ] && DEFAULTS=""; done      # a resume takes everything from the run
# Run in the background and forward the SIGTERM Slurm sends 10 minutes before the time limit (--signal=B:TERM@600): the SFT then writes a resume
# snapshot after its current step and exits 0 ("stopped at step N ... continue with ..."). `wait` returns early on a trapped signal: wait again.
python -m scripts.sft --model "$MODEL" $DEFAULTS "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
