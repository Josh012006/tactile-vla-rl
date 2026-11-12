# Evaluation of the SFT checkpoints: success-rate curve, deterministic, resumable, parallel

`python -m scripts.eval_sft --run <SFT run>` evaluates the initial state of a run (step 0) and every checkpoint of it in the FORGE environment, on the same fixed episodes, and draws the
success rate against the training step. Code: `src/eval/episodes.py` (seeds, one episode, the log, the numbers), `src/eval/sweep.py` (several checkpoints at once: threads, env servers),
`src/eval/curve.py` (the plot), `scripts/eval_sft.py` (command line), `scripts/probe_eval_parallel.py` (measures the parallelism), `jobs/job_eval_sft.sh`,
`jobs/job_probe_eval_parallel.sh`. Tests: `tests/test_eval_episodes.py`, `tests/test_eval_sweep.py`, `tests/test_eval_sft_script.py`, and the two `fork` tests of
`tests/test_openpi_policy.py` and `tests/test_sft.py`. **Status: tested on CPU (fake simulator over real sockets, real subprocesses); a first probe ran on an L40S on 12 Nov. 2026 (cost and parallelism, section 6); no real checkpoint has been evaluated yet** (section 9).

It follows expo-ft's `scripts/eval_curve.py` (fixed seeds generated from `--seed` 42, `--n-episodes`, a curve in % with one standard error, curve.json / png / pdf) and adds what that script does not
have: a resume at the episode, a deterministic policy for both models, several checkpoints at the same time, and a record that shows whether two evaluations are bit-identical.

## 1. Commands

| What | Command |
|---|---|
| the curve of a run, 50 episodes per checkpoint (about 1.2 h per checkpoint for pi0.5, 1.5 h for pi0-FAST) | `sbatch jobs/job_eval_sft.sh --run runs/sft/peg_insert/<run> --n-episodes 50` |
| what would be evaluated, and what is already done (login node, no GPU, no model, no server) | `python -m scripts.eval_sft --run <run> --n-episodes 200 --dry-run` |
| continue an evaluation that stopped (time limit, failure, `scancel`) | **the same command again**: only the missing episodes are run (a preemption or a node failure requeues the job, which does it by itself) |
| only some checkpoints (0 = the run's initial state) | `--steps 0 1000 4000` |
| more episodes than before | the same command with a larger `--n-episodes`: the first seeds do not change, only the new episodes are run |
| another seed, another folder | `--seed 7 --output-dir <dir>` (a folder made with another seed is refused) |
| redo a checkpoint from scratch | `--force --steps 1000` (its files are set aside as `.prev1`, never overwritten) |
| videos | `--save-videos` (`<output-dir>/videos/<step>/ep_<i>_seed<seed>.mp4`), `--video-dir <dir>` |
| the plot only (no GPU, no server) | `python -m scripts.eval_sft --run <run> --plot-only --title "Teleop demos, pi0.5"` |
| where the files go | `--output-dir` (data, default `<run>/eval`), `--curve-dir` (curve.json / png / pdf, default the output folder) |
| what a worker costs, and whether more workers pay off (measured: they do not on one L40S) | `sbatch jobs/job_probe_eval_parallel.sh --model pi05 --workers 1 2 --episodes 4` (section 6) |

Before the first run: `uv sync --group vla` (matplotlib is in that group since this file), the models fetched, the Isaac Lab venv (`scripts/setup_isaac_venv.sh`), the SFT run's checkpoints.
`--run` is a path, or the folder name under `runs/sft/peg_insert/` or `runs/sft/probing/`. The job mails (`--mail-user`, `--mail-type=ALL`, like the SFT jobs) and is requeueable (`--requeue`): a preemption or a node failure restarts the script, which is the same command, so the evaluation continues; a time limit is not
a requeue, send the command again. Resources for one worker: 1 L40S, 8 CPUs, 48 G, 14 h (`sacct -j <jobid> --format=JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem` says what a job really used).

## 2. What is evaluated

- **The run.** Its `config.json` gives the model YAML with its overrides and normalisation statistics (`model_cfg`): the evaluation builds the policy from that, never from the current YAML, so the
  weights, the rank and the statistics are the ones the run trained with. `checkpoints/<step>/` (complete ones only: `meta.json` is written last) holds the trained parameters.
- **The normalisation statistics are checked.** The policy reads the file the run's `norm_stats` block names (`<dir>/<asset id>/norm_stats.json`, the way the SFT did); the SFT also copied that file into every
  checkpoint it saved (`checkpoints/<step>/assets/<asset id>/norm_stats.json`). Before anything starts (and in `--dry-run`), the evaluation compares the two **byte for byte** for every checkpoint it is going
  to evaluate and refuses if one differs or has no copy (the folder was recomputed or edited after the training: put the checkpoint's copy back, or evaluate with the YAML the run used). The file's sha256
  is printed and frozen in `eval_protocol.json` (`norm_stats_sha`), so a folder cannot be resumed with other statistics. A model YAML without a `norm_stats` block uses the base checkpoint's own (DROID's)
  statistics on both sides: nothing to compare. Step 0 is built from the same YAML, so it has the run's statistics by construction.
- **Step 0** is the run's own initial state: the same model with the same LoRA initialisation (openpi's, seed 0) and nothing trained. This is the point the curve starts from, and also where an RL run
  starts from SFT's checkpoint. (`--no-initial` leaves it out of the default list; a checkpoint list given with `--steps` is exact.)
- **The episodes.** `--seed` (default 42) generates the `--n-episodes` seeds once (`np.random.default_rng(seed).integers(0, 2**31 - 1, n)`, expo-ft's recipe), kept in `<output-dir>/episode_seeds.json`.
  Episode *i* of every checkpoint is `reset(seed=seeds[i], round=0)` in a server with **one** environment. Why one: the start state depends on (seed, round, `num_envs`, env index)
  (`docs/envs/isaaclab_env.md` N2), the model is then always called with a batch of 1 (so its compiled shapes and arithmetic never depend on how many environments are still running), and the parallelism
  is across checkpoints, not across environments.
- **The episode.** A chunk is executed `--replan` steps (default: half of the model's chunk: 8 for pi0.5, 5 for pi0-FAST; expo-ft's ratio is 8 of 16), then the model is called again. At most
  `max_steps_per_episode` of the task YAML (150) steps; success is the environment's `success` (`success_hold_steps`). An episode that does not end within that length is an error, never a recorded failure.
  `--task-config` / `--env-set KEY=VALUE` change the server's settings; they are part of the protocol (section 5).

## 3. Deterministic evaluation

The evaluation is deterministic when the same weights give the same episodes, bit for bit. The sources of randomness, and what is done about each:

| source | pi0.5 | pi0-FAST | what the evaluation does |
|---|---|---|---|
| sampling of the action chunk | flow matching: the noise comes from the policy's key, which is **split at every call**, so it depends on everything the policy did before | **none**: greedy decoding (temperature 0: `Pi0FAST.sample_actions` takes the argmax) | `policy.seed(episode seed)` before every episode, for both models: the noise is a function of the episode alone, whatever ran before it, in which order and in which thread |
| shape of the model call | the batch is padded to a power of two | the same | one environment per server: the batch is always 1 |
| GPU kernels | reductions and scatters may differ from run to run | the same | `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (added by the script and the jobs) |
| compilation | autotuning could pick another algorithm in another compilation | the same | all the workers share **one** compiled sampler (`OpenPIPolicy.fork`) |
| LoRA initialisation (step 0) | `build_params_like_expo_ft(seed=0)`: fixed | the same | nothing to do |
| physics of the environment | bit-identical for an identical sequence of calls from a **fresh server** on the same machine (G1 of `docs/envs/isaaclab_env.md`); after other episodes the start of an episode can differ slightly (N1) | the same | a **fresh server for every checkpoint**; one GPU model for a whole curve |
| images | not compared bit for bit between two servers (N3) | the same | the fingerprints below do not cover them |

**Answer to "is pi0-FAST purely deterministic and pi0.5 not?"** On the policy side both are, with this protocol: pi0-FAST because it has no randomness at all, pi0.5 because its noise is restarted from the
episode's seed. What decides the rest is the environment, the same for both. The difference that remains is in the meaning: a pi0.5 evaluation is *one* fixed draw of the noise per episode, not an average over
the noise (to average over it, evaluate with several master seeds); pi0-FAST has no noise to average over.

**Fingerprints.** Every episode records `start_sha` (the first observation) and `traj_sha` (every joint, gripper and peg-socket reading of the episode, images excluded); a checkpoint's result carries
`eval_sha`, one hash over its episodes. Two evaluations of the same weights with the same `eval_sha` are the same episodes bit for bit; a different `eval_sha` says they diverged, and `traj_sha` says where.
`scripts/probe_eval_parallel.py` uses them to check that parallel evaluations and a resumed one give the same bits (section 6).

**What is not guaranteed, and said so in the files.**
- A checkpoint evaluated in **several attempts** (a resume) is statistically equivalent to one evaluated in a single attempt, **not bit-identical**: the episodes run after the cut start from a server that has not
  played the earlier ones, and the start of an episode after other episodes can differ by hundredths of a millimetre (N1; ~1 environment start in 3, median 0.01 to 0.06 mm, up to 5 to 11 mm). Every line of the
  log has its `attempt`, and the result has `attempts` and `single_attempt`; the probe measures how often a resumed episode really differs.
- Another GPU model, driver or Isaac version may change the bits (N4). Each record has its `gpu`, `eval_protocol.json` has one entry per attempt (host, GPU, XLA flags, git commit, Slurm job), and a run on another
  GPU than the earlier attempts prints a warning.
- The determinism flag's cost and the bit-identity between workers on one GPU are **not measured yet** (section 9).

**For the RL evaluation (to keep).** The same rules apply: restart the policy's randomness from the episode's seed (an RL policy's exploration noise too, or its deterministic mode when it has one, as expo-ft's
`--deterministic`), one environment and a batch of 1, a fresh server per evaluated checkpoint, the fixed seeds of the same `--seed`, one kind of GPU per curve, and the fingerprints. `Sweep`, `run_episode` and
`EpisodeLog` are generic: an RL evaluation only supplies its own `prepare(policy, job)` (load an RL checkpoint) and its own step 0 (the SFT checkpoint the RL started from).

## 4. Resume

Each finished episode is **one line** of `<output-dir>/results/<step>.jsonl`, appended, flushed and fsynced:

```
{"episode": 17, "seed": 1405681631, "success": true, "terminated": true, "truncated": false, "steps": 84, "start_sha": "...", "traj_sha": "...", "plans": 11,
 "plan_ms_median": 520.1, "tracking_err_max_rad": 0.04, "progress": {...}, "wall_s": 36.4, "decode": null, "video": null, "attempt": 1, "gpu": "NVIDIA L40S"}
```

The same command again reads these files:

- the episodes that are there are **skipped**; the missing ones are run, in order, on a fresh server; a line cut by a kill (no newline at its end) is dropped and the file is cut back to the last whole line,
  so the episode that was in progress when the job stopped is simply run again;
- the checkpoint's result (`results/<step>.json`: success rate, standard error, Wilson 95 % interval, per-episode outcomes, `eval_sha`, attempts, GPUs) and the curve are **recomputed from the files**, so they
  are what an evaluation that was never cut would give (up to N1 above, which `single_attempt` records);
- a checkpoint whose `.json` exists and whose episodes are all there is not run again (no server is started for it); `--n-episodes` can grow, and only the new episodes are run;
- a log from another evaluation (an episode whose seed is not the one of the list) and a damaged whole line are errors, not silently dropped; an episode that appears twice keeps its first record, with a warning;
- one evaluation at a time per output folder (`eval.lock`, `flock`): two would append to the same files.

**What may not change between two runs on a folder** (`eval_protocol.json`): `--seed`, `--replan`, the episode length, the number of environments, the model YAML as the run saved it, the sha256 of its
normalisation statistics, the task YAML with its `--env-set` overrides, and the recipe name (`deterministic-v1`). A change is refused with the list of what differs. The number of episodes, the checkpoints, `--workers`, the videos and the title may change.

## 5. Parallel evaluation of several checkpoints

`--workers N` evaluates N checkpoints at the same time.

- **One worker is one thread.** It takes the next checkpoint of a queue (by step), puts its weights in its own policy (`OpenPIPolicy.fork`, then `load_checkpoint`), starts a **fresh env server**, runs the missing
  episodes, stops the server, takes the next one. Fresh for each checkpoint, not one server kept per worker: the physics is bit-identical only for an identical sequence of calls from a fresh server (G1), and
  a start-up (a few minutes) is small against two hours of episodes.
- **The server is a subprocess**, not a thread: Isaac Sim allows one `SimulationApp` per process. It runs `jobs/env_server.sh` (one environment, a free port) in its **own process group**, with the learner's
  venv and XLA / JAX variables taken out of its environment, its log in `<output-dir>/logs/server_<step>.log`. The threads only supervise the subprocesses and drive the policy.
- **Memory.** The policies of all the workers share the compiled sampler and the base weights in GPU memory (`fork`; `load_checkpoint` replaces only the trainable parameters; checked in the tests: the frozen
  weights are the same device arrays): N workers cost about N env servers (about 8 GB each, `docs/envs/isaaclab_env.md` section 6) plus **one** model (about 8 GB for pi0-FAST at inference) plus the LoRA
  parameters of each. The policy stays under half of the card (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` in the job) so that the servers have room.
- **Servers start one at a time** (a lock from the start to the `ready` line): the first start of all builds the Robotiq USD, and two first starts would write it together.
- **Termination and clean-up.** SIGTERM and SIGINT (a `scancel`, or Slurm's warning 5 minutes before the time limit, forwarded by the job) set a stop flag. It is checked at every step of an episode (an
  episode in progress is **dropped**, not recorded), while a server starts, and between episodes; each worker stops its server in a `finally`. A server is stopped by SIGTERM to its whole process group, then SIGKILL
  after the grace period. If the workers have not left 90 s after the stop, the servers are killed from the main thread (their calls fail and the workers leave); `atexit` kills what is left; a second signal
  kills at once. A checkpoint that fails (a checkpoint that does not fit, a server that dies, a crash) is reported, does not stop the others, and the exit code is 1; a stop leaves exit code 0 with the message
  to run the same command again.
- **Choosing `--workers`: 1.** Measured on an L40S (section 6): two or three workers saturate the card (92-97 % utilisation) and give **the same throughput as one** (42 episodes per hour for pi0.5, 34 for pi0-FAST),
  while one worker already uses 17 GB. The likely reason (not checked) is that processes sharing one GPU without MPS take turns instead of overlapping; whatever the reason, more workers cost memory and nothing else. The default is 1, and `--workers` stays for a card
  or a setting where it pays off (the probe measures it). To go faster, use **several GPUs**, i.e. one job per run (or, if needed, one job per group of checkpoints: a worker pool that spans jobs needs a lock per
  checkpoint instead of one per folder, which is not written).

## 6. The probe

`sbatch jobs/job_probe_eval_parallel.sh --model pi05 --workers 1 2 --episodes 4` (same with `--model pi0_fast`; `--no-deterministic-xla --workers 1 --no-resume-check` measures what XLA's deterministic ops cost).
It needs no SFT run: the policy is the model's initial state. For each worker count W it evaluates W identical checkpoints (same weights, same episodes) at the same time with the real code path, and the
report (`reports/probe_eval_parallel_<model>_<time>_<jobid>.json`, printed at the end) gives:

- the median time of an episode (the first episode of each replica carries the compilation and is left out), how much of it is the model (plans x median plan time) and the rest (simulator), the throughput
  `3600 * W / median` and the speed-up against the first count;
- the GPU memory peak and utilisation (nvidia-smi, every 5 s) and the wall time with the servers' start-up;
- whether every replica of every count is **bit-identical** to the first one (`traj_sha` per episode): parallel evaluation on one GPU and the number of workers must not change a result;
- the **cut-and-resume check**: the first replica is copied with only its first half kept, as if the job had been killed, and resumed on a fresh server: how many of the episodes run after the cut differ
  from the uninterrupted ones (N1 says some may, slightly; this puts a number on it).

Every probe writes in its own folder (`runs/sft/probing/_probe_eval_parallel_<model>_<time>_<jobid>/`, under a lock).

**First measurements (L40S, 12 Nov. 2026, initial state, 4 episodes, XLA deterministic ops on, read from the two jobs' logs).** Seconds per episode of one worker, and throughput:

| | W = 1 | W = 2 | W = 3 |
|---|---|---|---|
| pi0.5 (replan 8): s per episode of each worker | 86-87 (95 with the compilation) | 172 | 259 |
| pi0.5: episodes per hour | 42 | 42 | 42 |
| pi0.5: GPU memory peak / mean utilisation | 17.0 GB / 65 % | 25.2 GB / 92 % | 33.3 GB / 97 % |
| pi0-FAST (replan 5): s per episode of each worker | 104-118 (107 typical) | 212 | 320-330 |
| pi0-FAST: episodes per hour | 34 | 34 | 33-34 |

An episode is 86 s for pi0.5 and 107 s for pi0-FAST, not the 35 s of expo-ft (a separate server, another protocol): 50 episodes are about 1.2 h and 1.5 h, 200 episodes about 4.8 h and 6 h.
What part of it is the model, and whether XLA's deterministic ops cost anything, is what the next probe prints. The 0/4 successes are expected (nothing is trained), and the `Error decoding tokens ... shape (0, 8)`
lines of pi0-FAST are openpi's message for an initial-state model that writes no action tokens: the chunk is counted as not decoded (`decode_failed`) and the previous one is kept.

**The determinism part of that first probe is void.** The two jobs (pi0.5 and pi0-FAST) started in the same second and so wrote in the same folder: their episodes were mixed in the same files (the log's
"1 episodes appear twice", the pi0-FAST job's crash on the existing `resume/` folder), and every "bit-identical: False" compares one model with the other. It has to be run again (the probe now refuses to
share a folder); the speeds above do not depend on it, they are the per-episode times of each job's own log lines.

## 7. What is written

`<output-dir>/` (default `<run>/eval/`): `eval_protocol.json` (what the results were made with, and one entry per attempt), `episode_seeds.json`, `results/<step>.jsonl` and `.json`,
`logs/server_<step>.log`, `videos/<step>/` with `--save-videos`, `eval.lock`; `<curve-dir>/curve.json`, `curve.png`, `curve.pdf` (rebuilt after every finished checkpoint, so a stopped evaluation still has an
up-to-date plot of what is complete). The job's output is `logs/slurm/eval_sft_<jobid>.out`; the table at its end lists, per step, the success rate, the standard error, the Wilson interval, the number of
episodes, whether one attempt and the `eval_sha`.

**The curve** is expo-ft's: success rate in %, error bars of one standard error (`sqrt(p (1 - p) / n)`), the rate written above each point, x = SFT iteration (0 = initial state of the run). **The title is set in a small
font** (9 pt, bold, wrapped to 90 characters per line) so that it stays inside the picture and the PDF; only the title is small, the axes, ticks and annotations keep expo-ft's sizes (`--title` sets it; the default is
"Eval success rate vs. checkpoint — <run name>"). The Wilson interval is in curve.json, not on the plot (at 0 % and 100 % the standard error is 0).

## 8. Differences from expo-ft's `eval_curve.py`

| | expo-ft | here |
|---|---|---|
| episode seeds | `episode_seeds.json`, `--seed` 42, `--n-episodes` | the same generation, so the same seeds for the same `--seed` and `--n-episodes`; the file also records the master seed and a longer list extends a shorter one |
| resume | a checkpoint whose result file exists is skipped; **one cut in the middle is redone whole** | at the episode: a line per episode, only the missing ones are run, the result recomputed from the file |
| one checkpoint | a subprocess per checkpoint (`eval_policy.py`), episodes one after the other | threads in one process (one shared compiled model), one env server per checkpoint |
| several checkpoints | one after the other | `--workers N` at the same time |
| randomness of the policy | not seeded per episode (`--deterministic` only for the RL's residual actor) | restarted from the episode's seed for both models, XLA deterministic ops, batch of 1 |
| bit-level check | none | `start_sha`, `traj_sha`, `eval_sha`, `attempt`, `gpu`, the protocol file and its check on resume |
| curve and data location | `--output-dir` for everything (default: the checkpoints folder) | `--output-dir` (default `<run>/eval`) and `--curve-dir` |
| title | 15 pt | 9 pt, wrapped |
| initial point | `base` (the base model) | `0`: the run's own initial state |

## 9. Checked and not checked

Checked (CPU): the seeds (expo-ft's, a longer list extends a shorter one, a seed file of another master seed refused); the log (a line cut by a kill, a damaged line, duplicates, concurrent appends); the
numbers; the protocol check; one episode on the fake simulator over a real socket (record, reproducible on a fresh server, the noise of the policy a function of the episode and not of the history, a stop in the
middle of an episode drops it, an episode that does not end is an error); the sweep (every checkpoint on the same seeds, one or three workers give the same results, servers start one at a time, a stopped sweep
keeps its episodes and the same command finishes it with the results of an uncut one, a finished checkpoint starts no server, more episodes run only the new ones, a failing checkpoint or policy does not stop the
others, every server is stopped); the server subprocess (ready line, an old ready line not mistaken, a death, a timeout, a stop while waiting, SIGTERM then SIGKILL, the whole process group killed); the script
(`--dry-run`, `--plot-only`, the refusal of another protocol, the lock, `--force`); the curve (rebuilt from complete checkpoints only, the title small and inside the figure); the probe's analysis; and
on a tiny pi0.5 that `OpenPIPolicy.fork` gives the same bits as its parent for the same seed, has its own stream and weights, and shares the device arrays, and that `load_checkpoint` on a fork leaves the base
policy untouched and the frozen weights shared.

**Not checked, to be seen on the cluster:**
- everything that needs Isaac Sim and the real models together: a first evaluation of a real checkpoint, the server subprocess started from the learner's venv through `jobs/env_server.sh`, the `ready` line, the
  clean-up of Isaac's child processes;
- that `--xla_gpu_deterministic_ops` gives the same bits in two compilations and two threads, and its cost (the corrected probe compares replicas, `--no-deterministic-xla` the cost);
- that two servers and a model on one GPU give episodes bit-identical to one alone (G1 was measured with one server); the speed-up is measured (none on an L40S); the host memory of an Isaac server
  (`--mem=48G` for one worker is an estimate, to be checked with `sacct` after the first evaluation job);
- the size of N1 for resumed episodes (the probe's cut-and-resume check), and the effect on a success rate over hundreds of episodes (N6 of the env's reproducibility section);
- the compilation of the shared sampler by several threads at once;
- what the model and the simulator each take of the 86 s / 107 s of an episode (the corrected probe prints it);
- the default `--replan` of pi0-FAST (5 of 10): expo-ft's 8 of 16 transposed, not measured.
