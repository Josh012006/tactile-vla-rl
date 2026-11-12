#!/bin/bash
# SFT with LoRA (the model YAML's: openpi's own LoRA) of pi0.5 or pi0-FAST on the FORGE demonstrations, on an L40S (46 GB).
#   sbatch jobs/job_sft_lora.sh pi0_fast                                    # the demonstrations of configs/algo/sft.yaml (scripted)
#   sbatch jobs/job_sft_lora.sh pi05 --demos demos/isaaclab/ForgePegInsert_Robotiq_noisy --run-name pi05_noisy
#   sbatch jobs/job_sft_lora.sh pi05 --set steps=1500 --set micro_batch_size=8 --model-set vlm.lora_rank=8
#   sbatch jobs/job_sft_lora.sh pi05 --resume <run name>                     # continue a run that stopped (time limit, failure): its own settings are used
# Everything after the model name goes to `python -m scripts.sft` (see its --help; --set overrides configs/algo/sft.yaml, --model-set the model YAML).
# Full fine-tuning does not fit here: jobs/job_sft_full.sh (A100 80 GB).
# Before the first run, on the login node (internet needed, the job itself runs offline):
#   uv sync --group vla && source .venv/bin/activate && python -m scripts.fetch_models
#   python -m scripts.sft --model <model> --dry-run                         # reads the demonstrations, loads no model
#   wandb login                                                             # once; the job logs to wandb (--set wandb_mode=offline | disabled to change)
# Output: runs/sft/<run name>/ (metrics.jsonl, wandb_id.txt, checkpoints/<step>/, resume/), logs/slurm/sft_lora_<jobid>.out
# Time: 4000 steps are about 10 h on an L40S (8.7 s per step, measured) and 7 h on an A100 (6.0 s; `sbatch --gres=gpu:a100l:1 jobs/job_sft_lora.sh ...`), hence 14 h. Time limit: 5 minutes before it, Slurm sends a SIGTERM; the run writes its resume snapshot (4.5 GiB with LoRA) and stops cleanly.
#
#SBATCH --job-name=sft_lora
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec

set -u
MODEL="${1:-}"
[ -z "$MODEL" ] || [[ "$MODEL" == --* ]] && { echo "usage: sbatch jobs/job_sft_lora.sh <model: pi05 | pi0_fast> [scripts.sft options]" >&2; exit 1; }
shift
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm runs/sft
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # allocate what is used (the default preallocates 75 % of the GPU and hides the real size)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9      # XLA's allocator stops at 75 % of the card by default (a failure 'RESOURCE_EXHAUSTED' with 15 of 46 GB in use): 0.9, as openpi recommends
export WANDB__SERVICE_WAIT=300                  # wandb's helper process can be slow to start on a loaded node
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # everything was fetched by scripts.fetch_models: a missing file fails instead of downloading
# Run in the background and forward the SIGTERM Slurm sends 5 minutes before the time limit (--signal=B:TERM@300): the SFT then writes a resume
# snapshot after its current step and exits 0 ("stopped at step N ... continue with ..."). `wait` returns early on a trapped signal: wait again.
python -m scripts.sft --model "$MODEL" "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
