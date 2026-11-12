#!/bin/bash
# Success-rate curve of an SFT run: its initial state (step 0) and every checkpoint, on the same fixed episodes of the FORGE environment (Isaac Sim and the policy on ONE GPU).
# Protocol batched-v1: --envs-per-server K (default 100) environments per server; the episodes of a checkpoint play together in batches of K (docs/eval/sft_eval.md section 2).
#   sbatch jobs/job_eval_sft.sh --run runs/sft/peg_insert/peg_insert_pi05_lora_teleop --n-episodes 50
#   sbatch jobs/job_eval_sft.sh --run <run> --steps 0 2000 4000 --n-episodes 100           # only these checkpoints (0 = the run's initial state)
#   sbatch jobs/job_eval_sft.sh --run <run> --n-episodes 50 --save-videos                    # one video per episode (default: the first 5 of each batch, --video-episodes)
#   sbatch jobs/job_eval_sft.sh --run <run> --steps 0 --n-episodes 100 --output-dir <probe dir>   # MEASURE the cost of a batch of 100 on the real simulator and model (nothing else), see below
#   sbatch jobs/job_eval_sft.sh --run <run> --n-episodes 50                                  # the SAME command again after a stop or a failure: only the missing episodes are run
# Every option goes to `python -m scripts.eval_sft` (see its --help: --seed 42, --output-dir, --curve-dir, --title, --replan, --force, ...).
# Look first, on the login node, at what would be evaluated and what is already done:  python -m scripts.eval_sft --run <run> --dry-run
# Rebuild the curve only (no GPU):                                                      python -m scripts.eval_sft --run <run> --plot-only --title "..."
# Before the first run: the learner venv and the models fetched (uv sync --group vla; python -m scripts.fetch_models), the Isaac Lab venv (scripts/setup_isaac_venv.sh).
#
# One worker = one checkpoint at a time, with its own env server of K environments (about 8.1 GiB + 26-29.5 MiB per environment: about 11 GiB for K = 100; plus the one policy the workers share,
# whose memory is 14.2 GiB at 64 rows for pi0.5 and 9.4 GiB at 8 rows for pi0-FAST, scripts/probe_batch_cost.py). Measured on an L40S before batching (probe of 12 Nov. 2026, ONE environment per server): one worker used 17 GB of the card and 65 % of its utilisation; 2 or 3 workers only
# saturated it and gave the SAME throughput as one. So --workers stays 1; for speed, use several GPUs, i.e. several jobs on different runs. Keep ONE kind of GPU for a whole curve (the physics is bit-identical only on the same one).
# The model is called with the K rows split in equal calls of at most --model-batch rows (default 64: pi0.5 does not fit 100 rows in one call; pi0-FAST fits 16 and not 20: its default is 16, i.e. 7 calls of 15).
# Time: simulator side NOT MEASURED with K = 100 (model side measured, scripts/probe_batch_cost.py: pi0.5 87 ms per row and plan, linear). Before, an episode alone took about 86 s (pi0.5) or 107 s (pi0-FAST), i.e. 50 episodes 1.2-1.5 h per checkpoint. A batch of K plays its episodes together: one step costs about
# 0.53 s + 0.0023 K s, so 150 steps of K = 100 are about 2 minutes of simulator plus the model's calls (a chunk every 8 steps: 19 calls at batch K, `scripts/probe_batch_cost.py` measures it), i.e. a checkpoint of
# 100 episodes in minutes instead of 2.4 h IF those estimates hold. Measure it (second line of the usage above) before trusting the time limit below. Five minutes before the limit Slurm sends a SIGTERM:
# the episodes still running are dropped, the servers are killed, the episodes that had ended are kept (written the moment they end), and the same command continues from there.
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
