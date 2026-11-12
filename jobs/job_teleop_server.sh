#!/bin/bash
# The env server for TELEOPERATION, alone on an L40S: you drive it from your PC with scripts/demos/teleop_client.py, through an ssh tunnel.
#   sbatch jobs/job_teleop_server.sh                       # 8 h, port 8102
#   sbatch --time=03:00:00 jobs/job_teleop_server.sh        # shorter
# Then read logs/slurm/teleop_server_<jobid>.out: it prints the node and the two commands to run on your PC (tunnel, then the client).
# The demos you record are written by the server in `demos/...` (the link to scratch), whatever --out-dir you give the client (it is relative
# to the project root and must stay inside it). Cancel the job when you are done (scancel <jobid>).
# Extra server settings: DEMO_SERVER_SETS="--set robot=robotiq_2f85" sbatch --export=ALL jobs/job_teleop_server.sh
#
#SBATCH --job-name=teleop_server
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue
#SBATCH --mail-user=josue.mongan@mila.quebec

set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm
PORT="${DEMO_PORT:-8102}"
SERVER_LOG="logs/slurm/teleop_server_server_${SLURM_JOB_ID:-local}.log"

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT TERM INT

# shellcheck disable=SC2086
bash jobs/env_server.sh --set num_envs=1 --set success_hold_steps=5 --set gripper_can_open=false --set server_port="$PORT" ${DEMO_SERVER_SETS:-} \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "env server starting (pid $SERVER_PID, log $SERVER_LOG) ..."
for _ in $(seq 1 360); do
    grep -q "\[env_server\] ready" "$SERVER_LOG" && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: the env server died:"; tail -30 "$SERVER_LOG"; exit 1; }
    sleep 5
done
grep -q "\[env_server\] ready" "$SERVER_LOG" || { echo "ERROR: the env server did not become ready in 30 min"; tail -30 "$SERVER_LOG"; exit 1; }
NODE="$(hostname)"
echo
echo "READY on $NODE, port $PORT. On your PC:"
echo "  1. tunnel (leave it open):   ssh -J mila -L $PORT:localhost:$PORT $NODE"
echo "  2. client:                   python teleop_client.py --host localhost --port $PORT --seed 100"
echo "(copy scripts/demos/teleop_client.py to the PC once; pip install numpy pygame websockets msgpack)"
wait "$SERVER_PID"
