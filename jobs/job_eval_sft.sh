#!/bin/bash
# Success-rate curve of an SFT run: its initial state (step 0) and every checkpoint, on the same fixed episodes of the FORGE environment (Isaac Sim and the policy on ONE GPU).
#   sbatch jobs/job_eval_sft.sh --run runs/sft/peg_insert/peg_insert_pi05_lora_teleop --n-episodes 50
#   sbatch jobs/job_eval_sft.sh --run <run> --steps 0 2000 4000 --n-episodes 100           # only these checkpoints (0 = the run's initial state)
#   sbatch jobs/job_eval_sft.sh --run <run> --n-episodes 50 --save-videos                    # one video per episode
#   sbatch jobs/job_eval_sft.sh --run <run> --n-episodes 50                                  # the SAME command again after a stop or a failure: only the missing episodes are run
# Every option goes to `python -m scripts.eval_sft` (see its --help: --seed 42, --output-dir, --curve-dir, --title, --replan, --force, ...).
# Look first, on the login node, at what would be evaluated and what is already done:  python -m scripts.eval_sft --run <run> --dry-run
# Rebuild the curve only (no GPU):                                                      python -m scripts.eval_sft --run <run> --plot-only --title "..."
# Before the first run: the learner venv and the models fetched (uv sync --group vla; python -m scripts.fetch_models), the Isaac Lab venv (scripts/setup_isaac_venv.sh).
#
# One worker = one checkpoint at a time, with its own env server (about 8 GB of GPU memory each, plus the one policy the workers share). Measured on an L40S (probe of 12 Nov. 2026, one
# env per server, replan 8 / 5): one worker uses 17 GB of the card and 65 % of its utilisation; 2 or 3 workers only saturate it (92-97 %) and give the SAME throughput as one (42 episodes
# per hour for pi0.5, 34 for pi0-FAST). So --workers stays 1: more would cost memory for nothing; for speed, use several GPUs, i.e. several jobs on different runs. Keep ONE kind of GPU for a
# whole curve (the physics is bit-identical only on the same one).
# Time: a 150-step episode takes about 86 s for pi0.5 and 107 s for pi0-FAST (measured, XLA deterministic ops on), so 50 episodes are about 1.2 h (pi0.5) or 1.5 h (pi0-FAST) per
# checkpoint and 200 episodes about 4.8 h or 6 h: the limit below holds about 11 checkpoints of 50 episodes for pi0.5 and 9 for pi0-FAST. Five minutes before the limit Slurm sends a SIGTERM:
# the episodes in progress are dropped, the servers are killed, the finished episodes are kept, and the same command continues from there.
# Requeue: the job is requeueable (a preemption or a node failure restarts the script, which is the same command: the evaluation continues where it stopped). A time limit is not
# a requeue: send the same command again.
# Resources, for ONE worker: 6 CPUs and 40 G of host memory (an Isaac server and the model); the GPU memory is the 17 GB above. Measured (12 Nov. 2026, sacct): an Isaac server alone peaks at 8.3 GB
# of host memory and uses about 1.4 cores on average; the model alone (pi0.5 / pi0-FAST, load and sampling) peaks at 21.7 / 19.9 GB and uses about 1.7 cores: together about 30 GB and 3 cores,
# the rest is margin (Kit start-up and the XLA compilation use more cores for a short time). Check on the first real evaluation. `sacct -j <jobid> --format=JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem`
# after a job says what it really used; logs/slurm/gpu_<jobid>.csv has the card's utilisation every 30 s.
# Output: <run>/eval/ (eval_protocol.json, episode_seeds.json, results/, logs/server_<step>.log, curve.json / curve.png / curve.pdf), logs/slurm/eval_sft_<jobid>.out,
# logs/slurm/gpu_<jobid>.csv.
#
#SBATCH --job-name=eval_sft
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josue.mongan@mila.quebec

set -u
[ $# -eq 0 ] && { echo "usage: sbatch jobs/job_eval_sft.sh --run <SFT run> [--n-episodes N] [scripts.eval_sft options]" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1                       # the log is read while the job runs
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # the policy takes what it uses: the env server needs the rest of the card
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5       # and it may not grow past half of it (the default 75 % would leave the servers too little; measured: 17 GB in all for one worker)
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"     # deterministic GPU kernels: part of what makes an evaluation reproducible
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # everything was fetched by scripts.fetch_models: a missing file fails instead of downloading
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > "logs/slurm/gpu_${SLURM_JOB_ID:-local}.csv" 2>/dev/null &
smi=$!
# Run in the background and forward the SIGTERM Slurm sends before the time limit: the evaluation stops cleanly (servers killed, finished episodes kept) and exits 0.
# `wait` returns early on a trapped signal: wait again.
python -m scripts.eval_sft "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
kill $smi 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
