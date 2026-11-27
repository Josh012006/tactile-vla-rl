#!/usr/bin/env bash
# Active evaluation: watch SFT runs (or RL runs) and submit an evaluation job for every checkpoint (round) as soon as it is ready.
#
#   scripts/watch_eval.sh sft --run peg_insert_pi05_lora_teleop_s43 peg_insert_pi05_lora_noisy_s43 --n-episodes 500
#   scripts/watch_eval.sh sft --glob 'peg_insert_pi0_fast_lora_*_s4[34]' --n-episodes 500 --eval-subdir eval_500 --every 400
#   scripts/watch_eval.sh rl  --run peg_insert_pi05_best_of_n --n-episodes 500 --eval-subdir eval_rl_500 --extra "--selection vla"
#   scripts/watch_eval.sh sft --run <run> --dry-run --once                 # what it would submit now, nothing submitted
# Run it under nohup (or tmux) on the login node; it only reads files and calls squeue/sbatch:
#   nohup scripts/watch_eval.sh sft --glob 'peg_insert_pi05_lora_*_s4[34]' --n-episodes 500 --eval-subdir eval_500 > logs/watch_eval_nohup.out 2>&1 &
#
# ONE JOB PER CHECKPOINT. Each ready checkpoint that is not evaluated yet gets its own job of jobs/job_eval_sft.sh (or job_eval_rl.sh), named eval_sft (eval_rl) like
# any other: sbatch --time=... jobs/job_eval_sft.sh --run <run> --steps <step> --n-episodes N --output-dir <run>/<subdir>. The jobs of ONE run are chained
# (--dependency=afterany: on that run's previous job, when it is still queued or running) because they share an output folder, which one evaluation locks at a time;
# the jobs of different runs are independent. A job that is only waiting for the previous one holds no GPU.
#   sft  a checkpoint is ready when <run>/checkpoints/<step>/meta.json exists (scripts/eval_sft.py's own rule); the initial state (step 0) is ready from the start
#        (--no-initial leaves it out). The run is finished when its summary.json says "stopped": false.
#   rl   a round r is ready when critics/round_<r>.msgpack exists, r <= resume/state.json "stage", and (PA-RL, r >= 1) vlas/round_<r>/meta.json exists
#        (scripts/eval_rl.py's own rule). The run is finished when its stage reaches total_episodes // envs_per_batch.
# Every submission is written to logs/watch_eval_<mode>.jobs ("<run> <step> <job id>"): a restart goes on from there and does not submit twice. A checkpoint
# is evaluated when results/<label>.json of the output folder holds at least --n-episodes episodes: a job that ended without it (time limit, preemption,
# failure) is submitted again, up to --max-tries times, then the checkpoint is left alone (read logs/slurm/). When every run is finished and evaluated it stops.
#
# WHICH RUNS
#   --run NAME...        run folder names (under runs/sft/<task>/ | runs/rl/<task>/: peg_insert, gear_mesh, nut_thread, probing, ...) or paths; several allowed
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
#   --time HH:MM:SS      time limit of every job (default 03:00:00)
#   --sbatch-args "ARGS" more options for sbatch (the GPU stays the job script's L40S: ONE kind of GPU per curve)
# WATCHING
#   --interval S         seconds between two looks (default 300)
#   --pause S            seconds between two submissions (default 30)
#   --max-jobs J         at most J jobs of the watcher queued or running at once, all runs together (default 5)
#   --max-tries T        submissions per checkpoint before it is left alone (default 3)
#   --log FILE           default logs/watch_eval_<mode>.log
#   --once               look once, submit, exit            --dry-run   print the sbatch commands, submit nothing
#   --keep-going         do not stop when every run is finished and evaluated (default: stop)
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
N_EP=""; K=100; SUBDIR=""; EXTRA=""; SBATCH_ARGS=""; TIME="03:00:00"
INTERVAL=300; PAUSE=30; MAX_TRIES=3; MAX_JOBS=5; LOG=""; ONCE=0; DRY=0; STOP_WHEN_DONE=1

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
        --time)            TIME="${1:-}"; shift ;;
        --interval)        INTERVAL="${1:-}"; shift ;;
        --pause)           PAUSE="${1:-}"; shift ;;
        --max-tries)       MAX_TRIES="${1:-}"; shift ;;
        --max-jobs)        MAX_JOBS="${1:-}"; shift ;;
        --log)             LOG="${1:-}"; shift ;;
        --once)            ONCE=1 ;;
        --dry-run)         DRY=1 ;;
        --keep-going)      STOP_WHEN_DONE=0 ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "unknown option $opt (--help for the usage)" >&2; exit 1 ;;
    esac
done
[ ${#RUNS[@]} -gt 0 ] || [ ${#GLOBS[@]} -gt 0 ] || { echo "give the runs to follow: --run NAME... and/or --glob PATTERN..." >&2; exit 1; }
for v in EVERY K INTERVAL PAUSE MAX_TRIES MAX_JOBS; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be a whole number (got '${!v}')" >&2; exit 1; }
done
for v in FROM UNTIL N_EP; do
    [ -z "${!v}" ] || [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be a whole number (got '${!v}')" >&2; exit 1; }
done
for s in "${STEPS[@]:-}"; do [ -z "$s" ] || [[ "$s" =~ ^[0-9]+$ ]] || { echo "--steps/--rounds: whole numbers (got '$s')" >&2; exit 1; }; done
{ [ "$K" -ge 1 ] && [ "$INTERVAL" -ge 1 ] && [ "$MAX_TRIES" -ge 1 ] && [ "$MAX_JOBS" -ge 1 ]; } || { echo "--envs-per-server, --interval, --max-tries and --max-jobs must be at least 1" >&2; exit 1; }

if [ "$MODE" = sft ]; then
    KIND=sft; JOB=jobs/job_eval_sft.sh; UNIT=step; FLAG=--steps; SUBDIR="${SUBDIR:-eval}"; N_EP="${N_EP:-100}"
else
    KIND=rl; JOB=jobs/job_eval_rl.sh; UNIT=round; FLAG=--rounds; SUBDIR="${SUBDIR:-eval_rl}"
fi
[ -f "$JOB" ] || { echo "no $JOB in $ROOT" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is needed (it reads the JSON files)" >&2; exit 1; }
LOG="${LOG:-logs/watch_eval_$MODE.log}"
mkdir -p "$(dirname "$LOG")"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
trap 'log "stopped (signal)"; exit 0' INT TERM

# -- the runs to follow -----------------------------------------------------------------------------------------------------------------
roots() { # the folders that hold runs of this kind: runs/<kind>/<task>/ for every task that has some (looked up at every call: a new task's first run is seen)
    (cd "$ROOT" && ls -d "runs/$KIND"/*/ 2>/dev/null | sed 's:/$::')
}
resolve_run() { # a name or a path -> the absolute folder holding config.json
    local c r
    for c in "$1" "$ROOT/$1" $(for r in $(roots); do echo "$ROOT/$r/$1"; done); do
        [ -f "$c/config.json" ] && { (cd "$c" && pwd); return 0; }
    done
    return 1
}
list_runs() { # one absolute folder per line, no duplicates; "WARNING ..." lines for names that match nothing
    local n g r d
    {
        for n in "${RUNS[@]:-}"; do
            [ -z "$n" ] && continue
            resolve_run "$n" || echo "WARNING no run '$n' (no config.json at that path nor under runs/$KIND/*/)"
        done
        for g in "${GLOBS[@]:-}"; do
            [ -z "$g" ] && continue
            for r in $(roots); do
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

JOBFILE="logs/watch_eval_$MODE.jobs"
declare -A JOB_OF TRIES LAST_JOB LEGACY_OF GAVE_UP DONE_LOGGED WARNED
if [ -f "$JOBFILE" ]; then
    while read -r n x id; do
        [ -n "$n" ] && [ -n "$id" ] || continue
        JOB_OF["$n $x"]=$id; TRIES["$n $x"]=$(( ${TRIES["$n $x"]:-0} + 1 )); LAST_JOB[$n]=$id
    done < "$JOBFILE"
fi
# jobs of the first version of this script (named watch_eval_<mode>_<run>) still queued or running: the run is left to them until they end
while read -r id jn; do case "$jn" in "watch_eval_${MODE}_"*) LEGACY_OF[${jn#"watch_eval_${MODE}_"}]=$id ;; esac; done < <(squeue -u "${USER:-$(id -un)}" -h -o '%i %j' 2>/dev/null)
job_alive() { [ -n "$1" ] && [ -n "$(squeue -h -j "$1" -o %i 2>/dev/null)" ]; }      # queued or running
count_alive() { local k c=0; for k in "${!JOB_OF[@]}"; do job_alive "${JOB_OF[$k]}" && c=$((c + 1)); done; echo "$c"; }

names="$(list_runs 2>/dev/null | grep -v '^WARNING' | xargs -r -n1 basename | tr '\n' ' ')"
eps_txt="${N_EP:-each run eval_episodes}"
dry_txt=""; [ "$DRY" = 1 ] && dry_txt=" | DRY RUN"
log "watching ($MODE): ${names}| ${UNIT}s every ${EVERY} (0 = all) | ${eps_txt} episodes, K=$K | output <run>/$SUBDIR | one job per ${UNIT}, ${TIME} each | look every ${INTERVAL}s | max $MAX_JOBS jobs${dry_txt}"

while :; do
    runs=()
    while IFS= read -r line; do
        case "$line" in
            WARNING*) [ -n "${WARNED[$line]:-}" ] || { WARNED[$line]=1; log "$line"; } ;;
            "") ;;
            *) runs+=("$line") ;;
        esac
    done < <(list_runs)
    active=$(count_alive)
    all_done=1
    if [ ${#runs[@]} -eq 0 ]; then all_done=0; log "no run to follow yet (a --glob may match later)"; fi
    for run in "${runs[@]:-}"; do
        [ -z "$run" ] && continue
        name="$(basename "$run")"
        state="$(run_state "$run" 2>&1)" || { log "$name: could not read its state: $state"; all_done=0; continue; }
        finished=$(awk '$1=="FINISHED"{print $2}' <<<"$state")
        ready=$(awk '$1=="READY"{$1="";print}' <<<"$state" | xargs)
        pending=$(awk '$1=="PENDING"{$1="";print}' <<<"$state" | xargs)
        if [ -z "$pending" ]; then
            if [ "$finished" = 1 ]; then
                [ -n "${DONE_LOGGED[$name]:-}" ] || { log "$name: finished and every $UNIT evaluated ($ready)"; DONE_LOGGED[$name]=1; }
                continue
            fi
            all_done=0; continue
        fi
        all_done=0
        unset 'DONE_LOGGED[$name]'
        if job_alive "${LEGACY_OF[$name]:-}"; then continue; fi                      # a job of the first version is still on this run
        for x in $pending; do
            key="$name $x"
            if job_alive "${JOB_OF[$key]:-}"; then continue; fi                      # its job is queued or running
            if [ "${TRIES[$key]:-0}" -ge "$MAX_TRIES" ]; then
                [ -n "${GAVE_UP[$key]:-}" ] || { log "$name $UNIT $x: $MAX_TRIES jobs without a result, LEFT ALONE (read logs/slurm/; restart the watcher after the fix)"; GAVE_UP[$key]=1; }
                continue
            fi
            if [ "$active" -ge "$MAX_JOBS" ]; then break; fi                          # the cap is for all runs: next look
            cmd=(sbatch)
            [ -z "$TIME" ] || cmd+=("--time=$TIME")
            prev="${LAST_JOB[$name]:-}"; after=""
            if job_alive "$prev"; then cmd+=("--dependency=afterany:$prev"); after=", after job $prev"; fi     # one evaluation at a time on the output folder of the run
            # shellcheck disable=SC2206
            [ -z "$SBATCH_ARGS" ] || cmd+=($SBATCH_ARGS)
            cmd+=("$JOB" --run "$run" "$FLAG" "$x" --envs-per-server "$K" --output-dir "$run/$SUBDIR")
            [ -z "$N_EP" ] || cmd+=(--n-episodes "$N_EP")
            # shellcheck disable=SC2206
            [ -z "$EXTRA" ] || cmd+=($EXTRA)
            if [ "$DRY" = 1 ]; then
                log "$name $UNIT $x: DRY RUN, would submit: ${cmd[*]}"
                continue
            fi
            if out="$("${cmd[@]}" 2>&1)"; then
                id="$(grep -o 'Submitted batch job [0-9]*' <<<"$out" | grep -o '[0-9]*$')"
                [ -n "$id" ] || { log "$name $UNIT $x: sbatch said '$(tail -1 <<<"$out")' (no job id read)"; continue; }
                JOB_OF[$key]=$id; TRIES[$key]=$(( ${TRIES[$key]:-0} + 1 )); LAST_JOB[$name]=$id; active=$((active + 1))
                echo "$name $x $id" >> "$JOBFILE"
                log "$name $UNIT $x: submitted job $id ($(basename "$JOB" .sh)${after})"
                sleep "$PAUSE"
            else
                log "$name $UNIT $x: sbatch failed: $out"
            fi
        done
    done
    [ "$ONCE" = 1 ] && break
    if [ "$all_done" = 1 ] && [ "$STOP_WHEN_DONE" = 1 ]; then log "every run is finished and evaluated: stop"; break; fi
    sleep "$INTERVAL" & wait $!
done
