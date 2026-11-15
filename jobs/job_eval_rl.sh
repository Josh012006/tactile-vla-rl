#!/bin/bash
# Success-rate curve of an RL run, evaluated again the way the SFT checkpoints are (jobs/job_eval_sft.sh): every finished round of the run (its critic and its VLA: PA-RL's own after the round's distillation,
# the SFT checkpoint it started from for Best-of-N and for round 0), on the same fixed episodes of the FORGE environment (Isaac Sim and the policy on ONE GPU). Protocol of scripts/eval_sft.py: --envs-per-server K
# (default 100) environments per server, a FRESH server per round, master seed 42, XLA deterministic GPU ops, the normalisation statistics of the SFT compared byte for byte with the copies saved in the checkpoints.
#   sbatch jobs/job_eval_rl.sh --run runs/rl/peg_insert/peg_insert_pi05_best_of_n                       # every finished round, 100 episodes each (the run's `eval_episodes`)
#   sbatch jobs/job_eval_rl.sh --run <run> --rounds 0 5 10 15 20                                        # only these rounds (0 = after the warm-up)
#   sbatch jobs/job_eval_rl.sh --run <run> --rounds 20 --save-videos                                    # one video per episode (default: the first 5, --video-episodes)
#   sbatch jobs/job_eval_rl.sh --run <run>                                                              # the SAME command again after a stop or a failure: only the missing episodes are run
# Every option goes to `python -m scripts.eval_rl` (see its --help: --seed 42, --output-dir, --curve-dir, --title, --replan, --force, ...).
# Look first, on the login node, at what would be evaluated and what is already done:  python -m scripts.eval_rl --run <run> --dry-run
# Rebuild the curve only (no GPU):                                                      python -m scripts.eval_rl --run <run> --plot-only --title "..."
# Before the first run: the learner venv and the models fetched (uv sync --group vla; python -m scripts.fetch_models), the Isaac Lab venv (scripts/setup_isaac_venv.sh).
#
# One worker = one round at a time, with its own env server of K environments (about 11 GiB for K = 100) and the one policy the workers share (as the SFT evaluation; --workers stays 1). Keep ONE kind of GPU for a whole curve.
# Output: <run>/eval_rl/ (NOT the run's own eval/ folder, which holds the evaluations made during the run: eval_protocol.json, episode_seeds.json, results/<round>.jsonl and .json, logs/server_<round>.log,
# curve.json / curve.png / curve.pdf), logs/slurm/eval_rl_<jobid>.out, logs/slurm/gpu_<jobid>.csv.
# Time: NOT MEASURED for the RL policy. The SFT evaluation took 6.7 min per checkpoint (pi0.5, 100 episodes, K = 100); here each chunk also samples k = 32 candidates and runs the critic, so expect more
# (about 8 min per round was the estimate for the evaluations of the run): the 21 rounds of a run about 3 h. The limit below is a guess; check it on the first evaluation. Five minutes before the limit Slurm sends a SIGTERM:
# the episodes still running are dropped, the servers are killed, the episodes that had ended are kept, and the same command continues from there. The job is requeueable (a preemption restarts the script,
# which is the same command). A time limit is not a requeue: send the same command again. No mail (not a training job).
#
#SBATCH --job-name=eval_rl
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --requeue

set -u
[ $# -eq 0 ] && { echo "usage: sbatch jobs/job_eval_rl.sh --run <RL run> [--rounds R ...] [scripts.eval_rl options]" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

source .venv/bin/activate
export PYTHONUNBUFFERED=1                       # the log is read while the job runs
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # the policy takes what it uses: the env server needs the rest of the card
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5       # and it may not grow past half of it
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true"     # deterministic GPU kernels: part of what makes an evaluation reproducible
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # everything was fetched by scripts.fetch_models: a missing file fails instead of downloading
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 30 > "logs/slurm/gpu_${SLURM_JOB_ID:-local}.csv" 2>/dev/null &
smi=$!
# Run in the background and forward the SIGTERM Slurm sends before the time limit: the evaluation stops cleanly (servers killed, finished episodes kept) and exits 0.
# `wait` returns early on a trapped signal: wait again.
python -m scripts.eval_rl "$@" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM INT
wait $child; status=$?
while kill -0 $child 2>/dev/null; do wait $child; status=$?; done
trap - TERM INT
kill $smi 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory still in use at the end: /'
exit $status
