#!/bin/bash
# Demonstrations and checks of the robot on an L40S: the Isaac Lab env server (ONE environment) and the client script, in one job.
#   sbatch jobs/job_demos.sh collect    --out demos/isaaclab/ForgePegInsert_Robotiq_scripted --num-episodes 50
#   sbatch jobs/job_demos.sh collect    --out demos/isaaclab/ForgePegInsert_Robotiq_noisy --num-episodes 50 --noise-fraction 0.35 --action-noise-sigma 0.06
#   sbatch jobs/job_demos.sh replay     --demos-dir demos/isaaclab/ForgePegInsert_Robotiq_teleop --save-videos --video-first 20
#   sbatch jobs/job_demos.sh robot_check --out reports/robot_check_tsf
# The first word is the tool (scripts/demos/<tool>.py: collect = collect_waypoint_demos); everything after it goes to the tool as is, see its --help.
# The server is started with what the tool needs (single_env.py checks it): num_envs=1 for all of them, and
#   collect, replay : success_hold_steps=5, gripper_can_open=false   (the peg stays in the closed gripper; an episode ends 5 'inserted' steps in a row)
#   robot_check     : gripper_can_open=true                          (it opens and closes the gripper)
# Extra server settings: DEMO_SERVER_SETS="--set robot=robotiq_2f85 --set reset_passes=3" sbatch --export=ALL jobs/job_demos.sh ...
# The task YAML is configs/task/peg_insert_forge.yaml (ENV_CONFIG=... to change it): its robot is robotiq_2f85_tsf, so the demos carry the taxels.
# Teleoperation is not a job: see jobs/job_teleop_server.sh (the client runs on your PC).
# Output: logs/slurm/demos_<jobid>.out (the tool), logs/slurm/demos_server_<jobid>.log (Isaac Lab), and what the tool writes (demos in
# the `demos` link, videos and reports where its flags say).
#
#SBATCH --job-name=demos
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --no-requeue
#SBATCH --mail-user=josue.mongan@mila.quebec

set -u
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/tactile-vla-rl}" || exit 1
mkdir -p logs/slurm reports
[ $# -ge 1 ] || { echo "usage: sbatch jobs/job_demos.sh <collect|replay|robot_check> [tool arguments]" >&2; exit 1; }
TOOL="$1"; shift
case "$TOOL" in
    collect)     MODULE=collect_waypoint_demos; SETS=(--set num_envs=1 --set success_hold_steps=5 --set gripper_can_open=false) ;;
    replay)      MODULE=replay_demos;           SETS=(--set num_envs=1 --set success_hold_steps=5 --set gripper_can_open=false) ;;
    robot_check) MODULE=robot_check;            SETS=(--set num_envs=1 --set gripper_can_open=true) ;;
    *) echo "unknown tool '$TOOL' (collect | replay | robot_check)" >&2; exit 1 ;;
esac
PORT="${DEMO_PORT:-8102}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

SERVER_LOG="logs/slurm/demos_server_${SLURM_JOB_ID:-local}.log"
# shellcheck disable=SC2086
bash jobs/env_server.sh "${SETS[@]}" --set server_port="$PORT" ${DEMO_SERVER_SETS:-} > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "env server starting (pid $SERVER_PID, log $SERVER_LOG) ..."
for _ in $(seq 1 360); do
    grep -q "\[env_server\] ready" "$SERVER_LOG" && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: the env server died:"; tail -30 "$SERVER_LOG"; exit 1; }
    sleep 5
done
grep -q "\[env_server\] ready" "$SERVER_LOG" || { echo "ERROR: the env server did not become ready in 30 min"; tail -30 "$SERVER_LOG"; exit 1; }

source .venv/bin/activate
python -m "scripts.demos.$MODULE" --port "$PORT" "$@"
