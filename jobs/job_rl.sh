#!/bin/bash
# Online RL fine-tuning of pi0.5 on the FORGE PegInsert task (scripts/rl.py): Best-of-N now; PA-RL's distillation is the next step. Isaac Sim (one env server at a time), the VLA and the critic share ONE L40S (46 GB).
#   sbatch jobs/job_rl.sh --experiment configs/experiment/rl/pi05_best_of_n.yaml --start <SFT run> <step>      # a main run: output in runs/rl/peg_insert/<experiment name>
#   sbatch jobs/job_rl.sh --experiment configs/experiment/rl/pi05_best_of_n.yaml --start <SFT run> <step> --run-name probe --set total_episodes=100 --set warmup_critic_steps=500     # a short probe
#   sbatch jobs/job_rl.sh --resume runs/rl/peg_insert/<run name>      # after a stop or a failure: continues after the last finished stage (a stage = the warm-up, or one round with its critic updates)
# Every option goes to `python -m scripts.rl` (see its --help). Look first, on the login node, at what would run:  python -m scripts.rl --experiment <file> --start <run> <step> --dry-run
# Before the first run: the learner venv and the models fetched (uv sync --group vla; python -m scripts.fetch_models), the Isaac Lab venv (scripts/setup_isaac_venv.sh), wandb login (--set wandb_mode=offline | disabled to change).
# Output: runs/rl/peg_insert/<name>/ (config.json, metrics.jsonl, wandb_id.txt, episodes/, resume/, critics/, eval/results/<round>.json and eval/curve.png, logs/server_*.log), logs/slurm/rl_<jobid>.out, logs/slurm/gpu_<jobid>.csv.
# Disk: about 0.25 GB of episodes per round (images, uncompressed) + the critic's state (about 30 MB) rewritten every round.
# Time: NOT MEASURED end to end. Parts measured before (docs/algos/rl.md section 5): a round of 50 episodes about 280 s of collection, an evaluation of 100 episodes about 8 min; the critic's updates (about 5000 per round) are
# unmeasured (jobs/job_probe_critic_cost.sh). The time limit is a first guess; a run that hits it ends cleanly on the SIGTERM sent 5 minutes before (after the stage in progress) and continues with --resume.
#
#SBATCH --job-name=rl
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec

set -u
[ $# -eq 0 ] && { echo "usage: sbatch jobs/job_rl.sh --experiment configs/experiment/rl/<name>.yaml --start <SFT run> <step> | --resume <run> [scripts.rl options]" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm runs/rl/peg_insert runs/rl/probing
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1                       # the log is read while the job runs
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # the VLA and the critic take what they use: the env server needs the rest of the card
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5       # and JAX may not grow past half of it (as the evaluation jobs: an env server of 50-100 environments takes 9-11 GB)
export WANDB__SERVICE_WAIT=300                  # wandb's helper process can be slow to start on a loaded node
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # everything was fetched by scripts.fetch_models: a missing file fails instead of downloading
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > "logs/slurm/gpu_${SLURM_JOB_ID:-local}.csv" 2>/dev/null &
smi=$!
# The SIGTERM Slurm sends 5 minutes before the time limit goes to the run: it ends after the stage in progress and exits 0 ("continue with --resume ..."). `wait` returns early on a trapped signal: wait again.
python -m scripts.rl "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
kill $smi 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
