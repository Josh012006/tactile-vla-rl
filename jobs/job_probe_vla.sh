#!/bin/bash
# Probe the VLA models on an L40S: what they receive, what they return, what they cost in VRAM, plus a short FORGE rollout.
#   sbatch jobs/job_probe_vla.sh                                  # pi05 (expo-ft loading, LoRA), pi05 without LoRA, pi0_fast; Isaac Lab env server (1 env)
#   PROBE_MODELS="pi05" sbatch --export=ALL jobs/job_probe_vla.sh   # a model name; `pi05:nolora` = `--use-lora false` (the exact base function)
#   PROBE_SYNTHETIC=1 sbatch --export=ALL jobs/job_probe_vla.sh   # no env server: fake observation (shapes, timing, VRAM only)
#   PROBE_STEPS=60 PROBE_REPLAN=8 sbatch --export=ALL jobs/job_probe_vla.sh
# Before the first run, on the login node (internet needed, the job itself runs offline):
#   uv sync --group vla && source .venv/bin/activate && python -m src.models.fetch
# Output: logs/slurm/probe_vla_<jobid>.out, reports/probe_<model>_<time>.json and the two camera images of the observation.
#
#SBATCH --job-name=probe_vla
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue

set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports
MODELS="${PROBE_MODELS:-pi05 pi05:nolora pi0_fast}"
PORT="${PROBE_PORT:-8102}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

if [ -z "${PROBE_SYNTHETIC:-}" ]; then
    # 1. the env server, in its own venv, one environment (a probe, not a benchmark). Its VRAM is counted in the GPU total, not in the JAX numbers.
    SERVER_LOG="logs/slurm/probe_vla_server_${SLURM_JOB_ID:-local}.log"
    bash jobs/env_server.sh --set num_envs=1 --set server_port="$PORT" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "env server starting (pid $SERVER_PID, log $SERVER_LOG) ..."
    for _ in $(seq 1 360); do
        grep -q "\[env_server\] ready" "$SERVER_LOG" && break
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: the env server died:"; tail -30 "$SERVER_LOG"; exit 1; }
        sleep 5
    done
    grep -q "\[env_server\] ready" "$SERVER_LOG" || { echo "ERROR: the env server did not become ready in 30 min"; tail -30 "$SERVER_LOG"; exit 1; }
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/GPU memory used with the env server alone: /'
    EXTRA=(--port "$PORT")
else
    EXTRA=(--synthetic)
fi

# 2. the models, one process each (a model that is deleted does not give its memory back to the next one), in the learner venv
source .venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # allocate what is used (the default preallocates 75 % of the GPU and hides the real size)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # everything was fetched by src.models.fetch: a missing file fails instead of downloading
status=0
for m in $MODELS; do
    echo; echo "################ $m"
    name="${m%%:*}"; flags=()
    [ "$m" != "$name" ] && [ "${m#*:}" = "nolora" ] && flags=(--use-lora false)
    python -m src.models.probe --model "$name" ${flags[@]+"${flags[@]}"} "${EXTRA[@]}" --steps "${PROBE_STEPS:-40}" --replan "${PROBE_REPLAN:-8}" || status=1
done
exit $status
