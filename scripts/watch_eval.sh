#!/usr/bin/env bash
# Active evaluation: watch SFT runs (or RL runs) and submit an evaluation job for every checkpoint (round) as soon as it is ready.
#
#   scripts/watch_eval.sh sft --run peg_insert_pi05_lora_teleop_s43 peg_insert_pi05_lora_noisy_s43 --n-episodes 500
#   scripts/watch_eval.sh sft --glob 'peg_insert_pi0_fast_lora_*_s4[34]' --n-episodes 500 --eval-subdir eval_500 --every 400
#   scripts/watch_eval.sh rl  --run peg_insert_pi05_best_of_n --n-episodes 500 --eval-subdir eval_rl_500
#   scripts/watch_eval.sh sft --run <run> --dry-run --once                 # what it would submit now, nothing submitted
#
# MODE
#   sft  a checkpoint is ready when <run>/checkpoints/<step>/meta.json exists (written last: scripts/eval_sft.py's own rule); the initial state (step 0)
#        is ready from the start (--no-initial leaves it out). Evaluated with jobs/job_eval_sft.sh. The run is finished when its summary.json says "stopped": false.
#   rl   a round r is ready when critics/round_<r>.msgpack exists, r <= resume/state.json "stage", and (PA-RL, r >= 1) vlas/round_<r>/meta.json exists
#        (scripts/eval_rl.py's own rule). Evaluated with jobs/job_eval_rl.sh, again, the way the SFT checkpoints are.
#        The run is finished when its stage reaches total_episodes // envs_per_batch.
#
# WHICH RUNS
#   --run NAME...        run folder names (under runs/sft/peg_insert, runs/sft/probing | runs/rl/peg_insert, runs/rl/probing) or paths; several allowed
#   --glob PATTERN...    shell patterns on those folder names, EXPANDED AGAIN at every look: a run started later is picked up (quote them)
# WHICH CHECKPOINTS (steps for sft, rounds for rl)
#   --steps N...         only these (alias: --rounds)
#   --every N            only the multiples of N (0 is one: the initial state / the warm-up)
#   --from N, --until M  only N <= x <= M
#   --no-initial         sft: leave step 0 out
# EVALUATION (passed to jobs/job_eval_{sft,rl}.sh)
#   --n-episodes N       episodes per checkpoint (default 100 for sft; for rl, the run's own `eval_episodes`)
#   --envs-per-server K  environments per server (default 100). Frozen by the protocol of the output folder: the same K for the whole curve.
#   --eval-subdir NAME   output folder <run>/NAME (default eval for sft, eval_rl for rl). One folder = one protocol (K, seed; the episodes may only grow).
#   --extra "ARGS"       more options for scripts.eval_sft / scripts.eval_rl, e.g. "--seed 7" or "--selection vla"
#   --sbatch-args "ARGS" more options for sbatch, e.g. "--time=10:00:00" (the GPU stays the job script's L40S: ONE kind of GPU per curve)
#   --max-per-job M      at most M checkpoints in one job (default 0: all that are ready); the next ones go in the next job
# WATCHING
#   --interval S         seconds between two looks (default 300)
#   --min-new M          wait for M ready checkpoints before submitting, except when the run is finished (default 1)
#   --max-jobs J         at most J evaluation jobs of the watcher queued or running at once, all runs together (default 5)
#   --job-prefix P       job names are P_<run> (default watch_eval_<mode>); ONE job per run at a time: the output folder is locked
#   --log FILE           default logs/watch_eval_<mode>.log
#   --once               look once, submit, exit            --dry-run   print the sbatch commands, submit nothing
#   --keep-going         do not stop when every run is finished and evaluated (default: stop)
#
# At each look, for each run: the ready checkpoints that pass the filters, minus those already evaluated (results/<label>.json with at least --n-episodes
# episodes); if some are left and no job of that run is queued or running, ONE job is submitted on them. A job resumes where an earlier one stopped (time limit,
# preemption): the next look submits the same command again. A run whose jobs end three times in a row without completing a checkpoint is left alone (read
# logs/slurm/ for the error); the other runs go on. Never start an evaluation by hand on an output folder the watcher follows.
# Let it live as long as the training does: tmux/screen on the login node (it only reads files and calls squeue/sbatch), e.g.
#   tmux new -s watch_eval 'cd ~/projects/tactile-vla-rl && scripts/watch_eval.sh sft --glob "peg_insert_pi05_lora_*_s4*" --n-episodes 500'
# Needs bash >= 4.4, python3 (json only), squeue/sbatch. It works in the repository root, wherever it is started from.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

usage() { sed -n '2,/^set -u/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

[ $# -ge 1 ] || { usage; exit 1; }
case "$1" in
    sft|rl) MODE="$1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "first argument: sft or rl (got '$1'); --help for the usage" >&2; exit 1 ;;
esac

RUNS=(); GLOBS=(); STEPS=(); EVERY=0; FROM=""; UNTIL=""; NO_INITIAL=0
N_EP=""; K=100; SUBDIR=""; EXTRA=""; SBATCH_ARGS=""; MAX_PER_JOB=0
INTERVAL=300; MIN_NEW=1; MAX_JOBS=5; PREFIX=""; LOG=""; ONCE=0; DRY=0; STOP_WHEN_DONE=1

take_values() { # $1 = name of an array; then the remaining arguments: appends the values up to the next option; sets TAKEN (how many)
    local -n arr="$1"; shift
    TAKEN=0
    while [ $# -gt 0 ] && [[ "$1" != --* ]]; do arr+=("$1"); shift; TAKEN=$((TAKEN + 1)); done
}
while [ $# -gt 0 ]; do
    opt="$1"; shift
    case "$opt" in
        --run)             take_values RUNS "$@"; shift "$TAKEN" ;;
        --glob)            take_values GLOBS "$@"; shift "$TAKEN" ;;
        --steps|--rounds)  take_values STEPS "$@"; shift "$TAKEN" ;;
        --every)           EVERY="${1:-}"; shift ;;
        --from)            FROM="${1:-}"; shift ;;
        --until)           UNTIL="${1:-}"; shift ;;
        --no-initial)      NO_INITIAL=1 ;;
        --n-episodes)      N_EP="${1:-}"; shift ;;
        --envs-per-server) K="${1:-}"; shift ;;
        --eval-subdir)     SUBDIR="${1:-}"; shift ;;
        --extra)           EXTRA="${1:-}"; shift ;;
        --sbatch-args)     SBATCH_ARGS="${1:-}"; shift ;;
        --max-per-job)     MAX_PER_JOB="${1:-}"; shift ;;
        --interval)        INTERVAL="${1:-}"; shift ;;
        --min-new)         MIN_NEW="${1:-}"; shift ;;
        --max-jobs)        MAX_JOBS="${1:-}"; shift ;;
        --job-prefix)      PREFIX="${1:-}"; shift ;;
        --log)             LOG="${1:-}"; shift ;;
        --once)            ONCE=1 ;;
        --dry-run)         DRY=1 ;;
        --keep-going)      STOP_WHEN_DONE=0 ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "unknown option $opt (--help for the usage)" >&2; exit 1 ;;
    esac
done
[ ${#RUNS[@]} -gt 0 ] || [ ${#GLOBS[@]} -gt 0 ] || { echo "give the runs to follow: --run NAME... and/or --glob PATTERN..." >&2; exit 1; }
for v in EVERY K MAX_PER_JOB INTERVAL MIN_NEW MAX_JOBS; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be a whole number (got '${!v}')" >&2; exit 1; }
done
for v in FROM UNTIL N_EP; do
    [ -z "${!v}" ] || [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be a whole number (got '${!v}')" >&2; exit 1; }
done
for s in "${STEPS[@]:-}"; do [ -z "$s" ] || [[ "$s" =~ ^[0-9]+$ ]] || { echo "--steps/--rounds: whole numbers (got '$s')" >&2; exit 1; }; done
{ [ "$K" -ge 1 ] && [ "$INTERVAL" -ge 1 ] && [ "$MIN_NEW" -ge 1 ] && [ "$MAX_JOBS" -ge 1 ]; } || { echo "--envs-per-server, --interval, --min-new and --max-jobs must be at least 1" >&2; exit 1; }

if [ "$MODE" = sft ]; then
    ROOTS=(runs/sft/peg_insert runs/sft/probing); JOB=jobs/job_eval_sft.sh; UNIT=step; FLAG=--steps; SUBDIR="${SUBDIR:-eval}"; N_EP="${N_EP:-100}"
else
    ROOTS=(runs/rl/peg_insert runs/rl/probing); JOB=jobs/job_eval_rl.sh; UNIT=round; FLAG=--rounds; SUBDIR="${SUBDIR:-eval_rl}"
fi
[ -f "$JOB" ] || { echo "no $JOB in $ROOT" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is needed (it reads the JSON files)" >&2; exit 1; }
PREFIX="${PREFIX:-watch_eval_$MODE}"
LOG="${LOG:-logs/watch_eval_$MODE.log}"
mkdir -p "$(dirname "$LOG")"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
trap 'log "stopped (signal)"; exit 0' INT TERM

# -- the runs to follow -----------------------------------------------------------------------------------------------------------------
resolve_run() { # a name or a path -> the absolute folder holding config.json
    local c r
    for c in "$1" "$ROOT/$1" $(for r in "${ROOTS[@]}"; do echo "$ROOT/$r/$1"; done); do
        [ -f "$c/config.json" ] && { (cd "$c" && pwd); return 0; }
    done
    return 1
}
list_runs() { # one absolute folder per line, no duplicates; "WARNING ..." lines for names that match nothing
    local n g r d
    {
        for n in "${RUNS[@]:-}"; do
            [ -z "$n" ] && continue
            resolve_run "$n" || echo "WARNING no run '$n' (no config.json at that path nor under ${ROOTS[*]})"
        done
        for g in "${GLOBS[@]:-}"; do
            [ -z "$g" ] && continue
            for r in "${ROOTS[@]}"; do
                for d in "$ROOT/$r"/$g; do [ -f "$d/config.json" ] && echo "$d"; done
            done
        done
    } | awk '!seen[$0]++'
}

# -- the state of one run ---------------------------------------------------------------------------------------------------------------
run_state() { # $1 = run folder. Prints three lines: FINISHED 0|1, READY <list>, PENDING <list> (sorted; READY and PENDING already filtered)
    python3 - "$MODE" "$1" "$1/$SUBDIR" "$N_EP" "$EVERY" "$FROM" "$UNTIL" "$NO_INITIAL" "${STEPS[*]:-}" <<'PY'
import json, os, sys
mode, run, evaldir, n_ep, every, lo, hi, no_initial, only = sys.argv[1:10]
every = int(every); lo = int(lo) if lo else None; hi = int(hi) if hi else None
only = {int(x) for x in only.split()}
config = json.load(open(os.path.join(run, "config.json")))
def exists(*p): return os.path.exists(os.path.join(run, *p))
def load(*p):
    try:
        return json.load(open(os.path.join(run, *p)))
    except Exception:
        return None
ready, finished, default_eps = [], False, None
if mode == "sft":
    if not int(no_initial):
        ready.append(0)
    folder = os.path.join(run, "checkpoints")
    if os.path.isdir(folder):
        ready += [int(d) for d in os.listdir(folder) if d.isdigit() and os.path.exists(os.path.join(folder, d, "meta.json"))]
    summary = load("summary.json")
    finished = bool(summary) and summary.get("stopped") is False
else:
    rl = config["rl"]
    rounds = int(rl["total_episodes"]) // int(rl["envs_per_batch"])
    state = load("resume", "state.json")
    last = int(state["stage"]) if state else None
    distills = rl.get("method") == "parl"
    folder = os.path.join(run, "critics")
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.startswith("round_") and f.endswith(".msgpack"):
                r = int(f[len("round_"):-len(".msgpack")])
                vla_ok = (not distills) or r == 0 or exists("vlas", "round_%03d" % r, "meta.json")
                if (last is None or r <= last) and vla_ok:
                    ready.append(r)
    finished = last is not None and last >= rounds
    default_eps = int(rl.get("eval_episodes", 100))
need = int(n_ep) if n_ep else (default_eps or 100)
def wanted(x):
    return ((not only or x in only) and (every == 0 or x % every == 0)
            and (lo is None or x >= lo) and (hi is None or x <= hi))
ready = sorted(x for x in set(ready) if wanted(x))
def evaluated(x):
    try:
        d = json.load(open(os.path.join(evaldir, "results", "%d.json" % x)))
    except Exception:
        return False
    return int(d.get("n_episodes", 0)) >= need
pending = [x for x in ready if not evaluated(x)]
print("FINISHED", 1 if finished else 0)
print("READY", *ready)
print("PENDING", *pending)
PY
}

queued_names() { squeue -u "${USER:-$(id -un)}" -h -o '%j' 2>/dev/null | grep "^${PREFIX}_" || true; }

declare -A LAST_DONE STRIKES DONE_LOGGED WARNED
names="$(list_runs 2>/dev/null | grep -v '^WARNING' | xargs -r -n1 basename | tr '\n' ' ')"
eps_txt="${N_EP:-each run eval_episodes}"
dry_txt=""; [ "$DRY" = 1 ] && dry_txt=" | DRY RUN"
log "watching ($MODE): ${names}| ${UNIT}s every ${EVERY} (0 = all) | ${eps_txt} episodes, K=$K | output <run>/$SUBDIR | look every ${INTERVAL}s | max $MAX_JOBS jobs${dry_txt}"

while :; do
    runs=()
    while IFS= read -r line; do
        case "$line" in
            WARNING*) [ -n "${WARNED[$line]:-}" ] || { WARNED[$line]=1; log "$line"; } ;;
            "") ;;
            *) runs+=("$line") ;;
        esac
    done < <(list_runs)
    queue="$(queued_names)"
    active=$(printf '%s\n' "$queue" | grep -c . || true)
    all_done=1
    if [ ${#runs[@]} -eq 0 ]; then all_done=0; log "no run to follow yet (a --glob may match later)"; fi
    for run in "${runs[@]:-}"; do
        [ -z "$run" ] && continue
        name="$(basename "$run")"
        state="$(run_state "$run" 2>&1)" || { log "$name: could not read its state: $state"; all_done=0; continue; }
        finished=$(awk '$1=="FINISHED"{print $2}' <<<"$state")
        ready=$(awk '$1=="READY"{$1="";print}' <<<"$state" | xargs)
        pending=$(awk '$1=="PENDING"{$1="";print}' <<<"$state" | xargs)
        npend=$(wc -w <<<"$pending")
        ndone=$(( $(wc -w <<<"$ready") - npend ))
        if [ "$npend" -eq 0 ]; then
            if [ "$finished" = 1 ]; then
                [ -n "${DONE_LOGGED[$name]:-}" ] || { log "$name: finished and every $UNIT evaluated ($ready)"; DONE_LOGGED[$name]=1; }
                continue
            fi
            all_done=0; continue
        fi
        all_done=0
        unset 'DONE_LOGGED[$name]'
        job="${PREFIX}_${name}"
        if grep -qx "$job" <<<"$queue"; then continue; fi                                     # its job is queued or running: the output folder is locked
        if [ "$finished" != 1 ] && [ "$npend" -lt "$MIN_NEW" ]; then continue; fi             # wait for more ready checkpoints
        if [ "${STRIKES[$name]:-0}" -ge 3 ]; then continue; fi                                # left alone
        # a job of this run has ended since the last submission: did it complete at least one checkpoint?
        if [ -n "${LAST_DONE[$name]:-}" ]; then
            if [ "$ndone" -le "${LAST_DONE[$name]}" ]; then
                STRIKES[$name]=$(( ${STRIKES[$name]:-0} + 1 ))
                log "$name: the last job completed no checkpoint (${STRIKES[$name]}/3): read its log in logs/slurm/"
                if [ "${STRIKES[$name]}" -ge 3 ]; then log "$name: LEFT ALONE (three jobs without progress); the other runs go on. Fix it, then restart the watcher."; continue; fi
            else
                STRIKES[$name]=0
            fi
        fi
        if [ "$active" -ge "$MAX_JOBS" ]; then log "$name: $npend $UNIT(s) pending ($pending), but $active/$MAX_JOBS jobs of the watcher are queued: next look"; continue; fi
        todo=($pending)
        if [ "$MAX_PER_JOB" -gt 0 ] && [ "${#todo[@]}" -gt "$MAX_PER_JOB" ]; then todo=("${todo[@]:0:$MAX_PER_JOB}"); fi
        cmd=(sbatch "--job-name=$job")
        # shellcheck disable=SC2206
        [ -z "$SBATCH_ARGS" ] || cmd+=($SBATCH_ARGS)
        cmd+=("$JOB" --run "$run" "$FLAG" "${todo[@]}" --envs-per-server "$K" --output-dir "$run/$SUBDIR")
        [ -z "$N_EP" ] || cmd+=(--n-episodes "$N_EP")
        # shellcheck disable=SC2206
        [ -z "$EXTRA" ] || cmd+=($EXTRA)
        if [ "$DRY" = 1 ]; then
            log "$name: DRY RUN, $npend pending of [$ready], would submit: ${cmd[*]}"
            continue
        fi
        if out="$("${cmd[@]}" 2>&1)"; then
            LAST_DONE[$name]=$ndone; active=$((active + 1)); queue+=$'\n'"$job"
            log "$name: submitted ${todo[*]} -> $out"
        else
            log "$name: sbatch failed: $out"
        fi
    done
    [ "$ONCE" = 1 ] && break
    if [ "$all_done" = 1 ] && [ "$STOP_WHEN_DONE" = 1 ]; then log "every run is finished and evaluated: stop"; break; fi
    sleep "$INTERVAL" & wait $!
done
