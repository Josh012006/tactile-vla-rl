#!/bin/bash
# FULL fine-tuning (use_lora false in every component: every weight trained, the exact base function at the start) of pi0.5 or pi0-FAST on
# the FORGE demonstrations, on an A100 80 GB. LoRA runs on an L40S: jobs/job_sft_lora.sh.
#   sbatch jobs/job_sft_full.sh pi05
#   sbatch jobs/job_sft_full.sh pi0_fast --demos demos/isaaclab/ForgePegInsert_Robotiq_teleop --run-name pi0fast_teleop_full
#   sbatch jobs/job_sft_full.sh pi05 --resume <run name>               # continue a run that stopped: its own settings are used, none is added here
# Everything after the model name goes to `python -m scripts.sft` (see its --help). Added here, except with --resume: `--full`, a micro-batch of 8 and
# a resume snapshot every 1000 steps (29 GiB each for pi0-FAST, see configs/algo/sft.yaml). A later --set wins.
# Measured on the A100 (docs/algos/sft.md, section 4), pi0-FAST: peak 61 GiB of the 75 the allocator may use (0.95 of the card; the default 0.75 killed the first probes), 7.3 s per step,
# so 4001 steps are about 8.1 h (about 8.6 h with the evaluations, saves and snapshots, hence the 12 h below). pi0.5: first step measured only (peak 67.5 GiB, it fits). A failure writes an `error` line.
# Before the first run, on the login node (fetch_models, wandb login): see jobs/job_sft_lora.sh.
# Output: runs/sft/<run name>/ (a checkpoint holds the whole model in bfloat16: 4.3 GiB for pi0-FAST, written in 17 s; a save every 200 steps is 20 of them, 87 GiB: raise
# save_interval with `--set save_interval=1000` when the disk matters; resume/ holds one snapshot, 29 GiB for pi0-FAST, twice that while it is swapped), logs/slurm/sft_full_<jobid>.out
# Time limit: 10 minutes before it, Slurm sends a SIGTERM; the run writes its snapshot after the current step (77 s measured for pi0-FAST) and stops cleanly.
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
#SBATCH --mail-user=josue.mongan@mila.quebec

set -u
MODEL="${1:-}"
[ -z "$MODEL" ] || [[ "$MODEL" == --* ]] && { echo "usage: sbatch jobs/job_sft_full.sh <model: pi05 | pi0_fast> [scripts.sft options]" >&2; exit 1; }
shift
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm runs/sft
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95      # XLA's allocator stops at 75 % of the card by default: the first full probes failed at 58-60 GiB of 80. 0.95: the memory is the constraint here
export WANDB__SERVICE_WAIT=300                  # wandb's helper process can be slow to start on a loaded node
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
