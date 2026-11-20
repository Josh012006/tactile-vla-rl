#!/bin/bash
# Online RL fine-tuning of pi0.5 on the FORGE PegInsert task, BEST-OF-N (scripts/rl.py): the VLA stays frozen, the online critic picks the argmax of the k = 32 candidates of the VLA. Isaac Sim (one env server at a time), the VLA and the critic share ONE L40S (46 GB).
#   sbatch jobs/job_rl_best_of_n.sh --start <SFT run> <step>                          # a main run of configs/experiment/rl/pi05_best_of_n.yaml: output in runs/rl/peg_insert/<experiment name>
#   sbatch jobs/job_rl_best_of_n.sh --start <SFT run> <step> --run-name probe --set total_episodes=100 --set warmup_critic_steps=500     # a short probe (the experiment's output_root)
#   sbatch jobs/job_rl_best_of_n.sh --experiment configs/experiment/rl/pi05_best_of_n_touch.yaml --start <SFT run> <step> --run-name <name>      # the same with tactile + measured force in the critic
#   sbatch jobs/job_rl_best_of_n.sh --resume runs/rl/peg_insert/<run name>            # after a stop or a failure: continues after the last finished stage (a stage = the warm-up, or one round with its critic updates)
# Every other option goes to `python -m scripts.rl` (see its --help); `--experiment FILE` replaces the default experiment (it must be a best_of_n one: PA-RL has jobs/job_rl_parl.sh). Look first, on the login node, at what would run:
#   python -m scripts.rl --experiment configs/experiment/rl/pi05_best_of_n.yaml --start <run> <step> --dry-run
# Before the first run: the learner venv and the models fetched (uv sync --group vla; python -m scripts.fetch_models), the Isaac Lab venv (scripts/setup_isaac_venv.sh), wandb login (--set wandb_mode=offline | disabled to change).
# Output: runs/rl/peg_insert/<name>/ (config.json, metrics.jsonl, wandb_id.txt, episodes/, resume/, critics/, eval/results/<round>.json and eval/curve.png, logs/server_*.log), logs/slurm/rl_bon_<jobid>.out, logs/slurm/gpu_<jobid>.csv.
# Disk: about 0.25 GB of episodes per round (images, uncompressed) + the critic's state (about 30 MB) rewritten every round: a few GB in all.
# Time: for the budget written in the experiment files (600 warm-up episodes + 120 000 critic updates, then 20 rounds of 50 episodes): about 5.9 h, CALCULATED from the parts measured (warm-up: 12 collection batches of about 5 min
# + 120 000 updates at 0.041 s = about 142 min; 20 rounds of about 9.7 min = about 194 min; 2 in-run evaluations of about 8 min; saves and server starts), so 8 h leaves room (20 rounds with 150 episodes / 30 000 updates took about 4.5 h).
# A SIGTERM is honoured between stages only; a run that hits the limit ends cleanly on the SIGTERM sent 5 minutes before (after the stage in progress) and continues with --resume.
#
#SBATCH --job-name=rl_bon
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec

set -u
usage() { echo "usage: sbatch jobs/job_rl_best_of_n.sh --start <SFT run> <step> [--run-name NAME] [--set KEY=VALUE ...] | --resume <run> [scripts.rl options]" >&2; exit 1; }
[ $# -eq 0 ] && usage

# The arguments of scripts.rl: the Best-of-N experiment unless another one is given (it must be a best_of_n one); a --resume needs none (the run carries its own).
args=("$@")
experiment=""; resume=0
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --experiment) experiment="${args[i+1]:-}" ;;
        --experiment=*) experiment="${args[i]#--experiment=}" ;;
        --resume|--resume=*) resume=1 ;;
    esac
done
default_experiment=configs/experiment/rl/pi05_best_of_n.yaml
if [ -n "$experiment" ]; then
    root="${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}"
    file="$experiment"; [ -f "$file" ] || file="$root/$experiment"
    grep -Eq '^[[:space:]]+method:[[:space:]]*best_of_n\b' "$file" 2>/dev/null || { echo "$experiment is not a best_of_n experiment (PA-RL: jobs/job_rl_parl.sh)" >&2; exit 1; }
elif [ "$resume" -eq 0 ]; then
    args=(--experiment "$default_experiment" "${args[@]}")
fi
[ -n "${RL_JOB_PRINT_ARGS:-}" ] && { echo "python -m scripts.rl ${args[*]}"; exit 0; }       # the translation of the arguments only (tests/test_rl_jobs.py)

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
python -m scripts.rl "${args[@]}" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
kill $smi 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
