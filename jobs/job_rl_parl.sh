#!/bin/bash
# Online RL fine-tuning of pi0.5 on the FORGE PegInsert task, PA-RL (scripts/rl.py): after every round the VLA is distilled on the critic's best actions of every stored state, then the candidate cache is drawn again with the new VLA (not after the last round).
# Isaac Sim (one env server at a time), the VLA, its optimiser and the critic share ONE L40S (46 GB). The two versions are chosen by --variant:
#   A  global      = configs/experiment/rl/pi05_parl_global.yaml        (top-m of the k = 32 candidates by Q, then distillation)
#   B  global+local = configs/experiment/rl/pi05_parl_global_local.yaml  (the same + 10 steps of gradient ascent on Q of the kept chunks)
#   sbatch jobs/job_rl_parl.sh --variant A --start <SFT run> <step>                   # a main run of variant A: output in runs/rl/peg_insert/<experiment name>
#   sbatch jobs/job_rl_parl.sh --variant B --start <SFT run> <step> --run-name probe --set total_episodes=100 --set warmup_critic_steps=500     # a short probe of variant B (the experiment's output_root)
#   sbatch jobs/job_rl_parl.sh --resume runs/rl/peg_insert/<run name>                 # after a stop or a failure: no --variant (the run carries its own); continues after the last finished stage (a stage = the warm-up, or one round with its critic updates AND its distillation / refresh)
# `--variant` also takes `global` / `global_local` (or `a` / `b`); `--experiment FILE` instead of `--variant` runs another PA-RL experiment (it must be a parl one: Best-of-N has jobs/job_rl_best_of_n.sh). Every other option goes to
# `python -m scripts.rl` (see its --help). Look first, on the login node, at what would run:  python -m scripts.rl --experiment configs/experiment/rl/pi05_parl_global.yaml --start <run> <step> --dry-run
# Before the first run: the learner venv and the models fetched (uv sync --group vla; python -m scripts.fetch_models), the Isaac Lab venv (scripts/setup_isaac_venv.sh), wandb login (--set wandb_mode=offline | disabled to change).
# Output: runs/rl/peg_insert/<name>/ (config.json, metrics.jsonl, wandb_id.txt, episodes/, resume/, critics/, vlas/, eval/results/<round>.json and eval/curve.png, logs/server_*.log), logs/slurm/rl_parl_<jobid>.out, logs/slurm/gpu_<jobid>.csv.
# Disk: about 0.25 GB of episodes per round (images, uncompressed) + the critic's state (about 30 MB) rewritten every round + the VLA's trainable parameters of EVERY round (vlas/round_<r>/, about 1.7 GB each: about 17 GB for 10 rounds)
# and, for the resume, the last stage's cache (up to a few GB) and VLA + Adam state (about 5 GB): about 25 GB in all per run.
# Memory (probe of 15 Nov. 2026): the distillation runs in micro-batches of 4 (`distill_micro_batch_size`), the peak of the VLA + Adam + critic is 22.3 GiB, JAX is limited to half of the card (below) and the env server has the rest.
# Time: NOT MEASURED end to end; estimated from the parts measured on 15 Nov. 2026 (docs/algos/rl.md sections 3 and 5) for the 10 rounds of 50 episodes of the defaults: about 7.5 h (5 h if the episodes average 10 plans) plus saves and server
# starts, the same for A and B. A SIGTERM is honoured between stages only: a PA-RL stage is up to an hour at the end of a run, so a stage cut by the time limit is played again on --resume. A run that hits the limit ends cleanly on the
# SIGTERM sent 5 minutes before (after the stage in progress) and continues with --resume.
#
#SBATCH --job-name=rl_parl
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
usage() { echo "usage: sbatch jobs/job_rl_parl.sh --variant A|B --start <SFT run> <step> [--run-name NAME] [--set KEY=VALUE ...] | --resume <run> [scripts.rl options]" >&2; exit 1; }
[ $# -eq 0 ] && usage

# The arguments of scripts.rl: --variant becomes the experiment file of that version; a --resume needs neither (the run carries its own); another PA-RL experiment can be given with --experiment instead.
args=(); variant=""; experiment=""; resume=0
while [ $# -gt 0 ]; do
    case "$1" in
        --variant) [ $# -ge 2 ] || usage; variant="$2"; shift 2 ;;
        --variant=*) variant="${1#--variant=}"; shift ;;
        --experiment) [ $# -ge 2 ] || usage; experiment="$2"; args+=("$1" "$2"); shift 2 ;;
        --experiment=*) experiment="${1#--experiment=}"; args+=("$1"); shift ;;
        --resume|--resume=*) resume=1; args+=("$1"); shift ;;
        *) args+=("$1"); shift ;;
    esac
done
if [ -n "$variant" ]; then
    [ -z "$experiment" ] || { echo "--variant and --experiment both given: choose one" >&2; exit 1; }
    [ "$resume" -eq 0 ] || { echo "--variant is for a new run: a --resume continues the run's own experiment" >&2; exit 1; }
    case "${variant,,}" in
        a|global) experiment=configs/experiment/rl/pi05_parl_global.yaml ;;
        b|global_local|global+local) experiment=configs/experiment/rl/pi05_parl_global_local.yaml ;;
        *) echo "unknown --variant '$variant': A (global) or B (global_local)" >&2; exit 1 ;;
    esac
    args=(--experiment "$experiment" "${args[@]}")
elif [ -z "$experiment" ] && [ "$resume" -eq 0 ]; then
    echo "PA-RL has two versions: say which with --variant A (global) or --variant B (global_local)" >&2; usage
fi
if [ -n "$experiment" ]; then
    root="${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}"
    file="$experiment"; [ -f "$file" ] || file="$root/$experiment"
    grep -Eq '^[[:space:]]+method:[[:space:]]*parl\b' "$file" 2>/dev/null || { echo "$experiment is not a parl experiment (Best-of-N: jobs/job_rl_best_of_n.sh)" >&2; exit 1; }
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
