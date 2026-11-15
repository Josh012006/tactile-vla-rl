# Evaluation of the SFT checkpoints: success-rate curve, K environments per server, resumable, parallel

`python -m scripts.eval_sft --run <SFT run>` evaluates the initial state of a run (step 0) and every checkpoint of it in the FORGE environment, on the same fixed episodes, and draws the
success rate against the training step. Code: `src/eval/episodes.py` (seeds, one batch of episodes, the log, the numbers), `src/eval/rollout.py` (the loop over K environments), `src/eval/sweep.py` (several checkpoints at once: threads, env servers),
`src/eval/curve.py` (the plot), `scripts/eval_sft.py` (command line), `scripts/probe_eval_parallel.py` (measures the parallelism), `scripts/probe_determinism.py` (model alone, simulator alone), `scripts/probe_batch_cost.py` (what the model costs at batch K), `jobs/job_eval_sft.sh`, `jobs/job_probe_batch_cost.sh`,
`jobs/job_probe_eval_parallel.sh`, `jobs/job_probe_determinism.sh`. Tests: `tests/test_eval_episodes.py`, `tests/test_eval_sweep.py`, `tests/test_eval_sft_script.py`, `tests/test_eval_rollout.py`, `tests/test_probe_determinism.py`, `tests/test_probe_batch_cost.py`, and the two `fork` tests of
`tests/test_openpi_policy.py` and `tests/test_sft.py`. **Status: tested on CPU (fake simulator over real sockets, real subprocesses); probes ran on an L40S on 12 Nov. 2026 (cost, parallelism, determinism, section 6). The protocol is now `batched-v1` (K environments per server, section 2); its cost on the real simulator and the real models is NOT measured yet (section 6.1); no real checkpoint has been evaluated yet** (section 9).

It follows expo-ft's `scripts/eval_curve.py` (fixed seeds generated from `--seed` 42, `--n-episodes`, a curve in % with one standard error, curve.json / png / pdf) and adds what that script does not
have: a resume at the episode, a deterministic policy for both models, several checkpoints at the same time, and a record that shows whether two evaluations are bit-identical.

## 1. Commands

| What | Command |
|---|---|
| the curve of a run, 50 episodes per checkpoint (all of them in ONE batch of a server of `--envs-per-server` = 100 environments: time to be measured, section 6.1) | `sbatch jobs/job_eval_sft.sh --run runs/sft/peg_insert/<run> --n-episodes 50` |
| what the model costs at batch K, and an estimate of a batch (no simulator, a few minutes) | `sbatch jobs/job_probe_batch_cost.sh --model pi05 --batch-sizes 1 8 32 64 100` |
| the real cost of a batch: one checkpoint (the initial state), 100 episodes, in a folder of its own | `sbatch jobs/job_eval_sft.sh --run <run> --steps 0 --n-episodes 100 --envs-per-server 100 --output-dir <probe dir>` |
| what would be evaluated, and what is already done (login node, no GPU, no model, no server) | `python -m scripts.eval_sft --run <run> --n-episodes 200 --dry-run` |
| continue an evaluation that stopped (time limit, failure, `scancel`) | **the same command again**: only the missing episodes are run (a preemption or a node failure requeues the job, which does it by itself) |
| only some checkpoints (0 = the run's initial state) | `--steps 0 1000 4000` |
| more episodes than before | the same command with a larger `--n-episodes`: the first seeds do not change, only the new episodes are run |
| another seed, another folder | `--seed 7 --output-dir <dir>` (a folder made with another seed is refused) |
| redo a checkpoint from scratch | `--force --steps 1000` (its files are set aside as `.prev1`, never overwritten) |
| videos | `--save-videos` (every episode) or, by default, the first `--video-episodes` 5 episodes of each batch: `<output-dir>/videos/<step>/ep_<i>_seed<seed>.mp4`, `--video-dir <dir>` |
| environments per server (part of the protocol, default 100) | `--envs-per-server K` |
| most rows of one model call (part of the protocol; default 64 for pi0.5 / pi0, 16 for pi0-FAST: both the largest that fit, section 6.1) | `--model-batch M` |
| an expo-ft SFT checkpoint through this evaluation (to compare the two SFT pipelines on the same weights, section 6.3) | `python -m scripts.adopt_expo_ft_checkpoint --checkpoint <expo-ft step folder> --name <name>`, then `--run _adopted_<name> --steps 0` |
| the plot only (no GPU, no server) | `python -m scripts.eval_sft --run <run> --plot-only --title "Teleop demos, pi0.5"` |
| where the files go | `--output-dir` (data, default `<run>/eval`), `--curve-dir` (curve.json / png / pdf, default the output folder) |
| what a worker costs, and whether more workers pay off (measured: they do not on one L40S) | `sbatch jobs/job_probe_eval_parallel.sh --model pi05 --workers 1 2 --episodes 4` (section 6) |

Before the first run: `uv sync --group vla` (matplotlib is in that group since this file), the models fetched, the Isaac Lab venv (`scripts/setup_isaac_venv.sh`), the SFT run's checkpoints.
`--run` is a path, or the folder name under `runs/sft/peg_insert/` or `runs/sft/probing/`. The job mails (`--mail-user`, `--mail-type=ALL`, like the SFT jobs) and is requeueable (`--requeue`): a preemption or a node failure restarts the script, which is the same command, so the evaluation continues; a time limit is not
a requeue, send the command again. Resources for one worker: 1 L40S, 6 CPUs, 40 G, 14 h (`sacct -j <jobid> --format=JobID,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem` says what a job really used).

## 2. What is evaluated

- **The run.** Its `config.json` gives the model YAML with its overrides and normalisation statistics (`model_cfg`): the evaluation builds the policy from that, never from the current YAML, so the
  weights, the rank and the statistics are the ones the run trained with. `checkpoints/<step>/` (complete ones only: `meta.json` is written last) holds the trained parameters.
- **The normalisation statistics are checked.** The policy reads the file the run's `norm_stats` block names (`<dir>/<asset id>/norm_stats.json`, the way the SFT did); the SFT also copied that file into every
  checkpoint it saved (`checkpoints/<step>/assets/<asset id>/norm_stats.json`). Before anything starts (and in `--dry-run`), the evaluation compares the two **byte for byte** for every checkpoint it is going
  to evaluate and refuses if one differs or has no copy (the folder was recomputed or edited after the training: put the checkpoint's copy back, or evaluate with the YAML the run used). The file's sha256
  is printed and frozen in `eval_protocol.json` (`norm_stats_sha`), so a folder cannot be resumed with other statistics. A model YAML without a `norm_stats` block uses the base checkpoint's own (DROID's)
  statistics on both sides: nothing to compare. Step 0 is built from the same YAML, so it has the run's statistics by construction.
- **Step 0** is the run's own initial state: the same model with the same LoRA initialisation (openpi's, drawn from the `seed` of the run, read from the `sft` block of its config.json; 0 for a run without one) and nothing trained. This is the point the curve starts from, and also where an RL run
  starts from SFT's checkpoint. (`--no-initial` leaves it out of the default list; a checkpoint list given with `--steps` is exact.)
- **The episodes.** `--seed` (default 42) generates the `--n-episodes` seeds once (`np.random.default_rng(seed).integers(0, 2**31 - 1, n)`, expo-ft's recipe), kept in `<output-dir>/episode_seeds.json`.
  **K environments per server** (`--envs-per-server`, default 100, frozen in `eval_protocol.json` as `num_envs`). Episode *i* is environment `i % K` of batch `i // K`; a batch is one
  `reset(seed=master seed, round=batch)` of a **fresh** server: the start state of environment *e* is a function of (master seed, K, e) (`docs/envs/isaaclab_env.md` N2), the same for every checkpoint on the same GPU
  model. The K episodes of a batch play **together**: one physics step for all of them, one model call per replan for the environments that need a chunk (`plan(..., pad_to=K)`: always the same batch shape, so one
  compilation and the same arithmetic whichever episodes are still running). The K rows are **split in equal model calls of at most `--model-batch` rows** (`OpenPIPolicy.max_rows`:
  100 rows with 64 are two calls of 50; the last call is padded, one that would hold only padding is not made), because the model's memory grows with the rows of ONE call (section 6.1); the time does not
  suffer (it is linear in the rows) and a row's noise is its own seed's whichever call it falls in. An episode that has ended is no longer planned for; one that is not played (already done in an earlier attempt) is **held** where it is (its
  joint targets are its own joint positions) and the others do not notice. The rollout ends when every played episode has ended. With `--n-episodes` N: N <= K is one batch (the others of the K environments are held); N > K
  several batches, the first of them in the first reset, the next ones after other episodes in the same server (N1: their starts can differ slightly, reproducibly for the same sequence; a script prints a WARNING when N > K, and N <= K is recommended).
  The **noise** of the model for episode *i*: `plan_seed(seed_i, k)` for its *k*-th chunk, a function of the episode alone (section 3), whichever episodes share the call.
  The seeds of `episode_seeds.json` are not used to start the simulator any more (the start is a function of the master seed and the batch); they seed the model's noise.
- **An episode.** A chunk is executed `--replan` steps (default: half of the model's chunk: 8 for pi0.5, 5 for pi0-FAST; expo-ft's ratio is 8 of 16), then the model is called again. At most
  `max_steps_per_episode` of the task YAML (150) steps; success is the environment's `success` (`success_hold_steps`). An episode that does not end within that length is an error, never a recorded failure.
  `--task-config` / `--env-set KEY=VALUE` change the server's settings; they are part of the protocol (section 5).

## 3. Deterministic evaluation

The aim is that the same weights give the same episodes, bit for bit. **Measured on 12 Nov. 2026, this is NOT reached end to end**: two evaluations of the same initial state (pi0.5, one worker, a fresh server, same seeds,
same XLA flags, same GPU) gave different `eval_sha` (`1d6af170...` and `aba0a88d...`). The first observation of an episode was bit-identical in 3 episodes out of 4 (`start_sha`; the 4th started 0.003 mm
elsewhere: N1), then the trajectories diverged by tens of millimetres at the end of 150 steps (the untrained model moves the arm far, which amplifies any difference). `scripts/probe_determinism.py` (section 6) then isolated the two halves, on an L40S: **the model is bit-reproducible** (the same observation and seed give the same action chunk,
bit for bit, in the same object, after other calls, in a fork, in several threads at once and in fresh processes, for pi0.5 and pi0-FAST) and **the physics is bit-reproducible over 150 scripted steps** from fresh servers
(joints, gripper, peg-socket offset and force identical in 3 servers). What differs is the input of the model that the first test did not cover: **the camera images** (N3). Measured (13:04, L40S, 3 fresh servers, same 150 scripted steps):
**the images are not reproducible between fresh servers, from the very first frame** (35 to 57 % of the pixel values differ, by up to 49 levels out of 255, at every checked step up to 150), while the readings of the same run are bit-identical.
The model therefore receives a slightly different picture at step 0 of every evaluation and its actions drift apart from there. Size of the difference between two fresh servers (default rendering, 13:19): mean 0.4 to 0.8 levels out of 255, 2 to 5 % of the values above 2 levels, 0.03 to 0.4 % above 8 levels:
sensor-level noise, at every step. `--rendering_mode performance`: larger in the first 5 frames (mean up to 1.3), then it falls to 0.04 to 0.2 (3 to 10 % of the values still differ) from step 25 on, never bit-identical;
`--kit_args=--/rtx/post/aa/op=0` (anti-aliasing off): no better (wrist camera worse at the start). No setting tried makes the images reproducible. Changing the preset also changes what the model sees
relative to the demonstrations it was trained on, so it would need its own check; it is not adopted. A second reset of the same seed in the same server starts from a state that differs from the first (0.0256 at the end of the same scripted play) **identically in every server**, so it is reproducible
for a given sequence of episodes, and it is the history dependence of N1. Until the images are settled an evaluation is **statistically** reproducible (fixed seeds, fixed starts, a standard error), not bit-reproducible. The sources of randomness, and what is done about each:

| source | pi0.5 | pi0-FAST | what the evaluation does |
|---|---|---|---|
| sampling of the action chunk | flow matching: the noise of the policy's own key is **split at every call**, so it depends on everything the policy did before | **none**: greedy decoding (temperature 0: `Pi0FAST.sample_actions` takes the argmax; `row_seeds` is ignored) | `plan(row_seeds=...)`: the flow noise of every row is drawn from `plan_seed(episode seed, chunk number)` (`jax.random.key(seed)`, a function of that seed alone) and passed to the sampler: a row's chunk does not depend on the other rows, on the batch composition, on what ran before or on the thread |
| shape of the model call | the batch is padded to a power of two | the same | `pad_to=K`: always K rows (padding repeats the last row), whatever the number of episodes still running |
| GPU kernels | reductions and scatters may differ from run to run | the same | `XLA_FLAGS=--xla_gpu_deterministic_ops=true` (added by the script and the jobs) |
| compilation | autotuning could pick another algorithm in another compilation | the same | all the workers share **one** compiled sampler (`OpenPIPolicy.fork`) |
| LoRA initialisation (step 0) | `build_params_like_expo_ft(seed=<seed of the run>)`: fixed by the run's config.json | the same | nothing to do |
| physics of the environment | bit-identical for an identical sequence of calls from a **fresh server** over the 20 steps that were tested (G1 of `docs/envs/isaaclab_env.md`); **not measured over 150 steps, and an evaluation of 150 steps was not reproducible** (above); after other episodes the start of an episode can differ slightly (N1) | the same | a **fresh server for every checkpoint** (and one real reset for a checkpoint of at most K episodes); one GPU model for a whole curve (kept: it is what G1 needs, and costs nothing) |
| images | **differ between fresh servers** (sensor-level noise, above) | the same | accepted: kept the default rendering, the comparison is statistical. The fingerprints below do not cover the images |

**Answer to "is pi0-FAST purely deterministic and pi0.5 not?"** On the policy side both are, with this protocol: pi0-FAST because it has no randomness at all, pi0.5 because its noise is restarted from the
episode's seed. What decides the rest is the environment, the same for both. The difference that remains is in the meaning: a pi0.5 evaluation is *one* fixed draw of the noise per episode, not an average over
the noise (to average over it, evaluate with several master seeds); pi0-FAST has no noise to average over.

**Fingerprints.** Every episode records `start_sha` (the first observation of its environment: joints, gripper, peg-socket offset, **no image**, so it IS identical for the same episode in two evaluations on one GPU) and `traj_sha` (every joint, gripper and peg-socket reading of the episode, images excluded; it differs between evaluations because the images do); a checkpoint's result carries
`eval_sha`, one hash over its episodes. Two evaluations of the same weights with the same `eval_sha` are the same episodes bit for bit; a different `eval_sha` says they diverged, and `traj_sha` says where.
`scripts/probe_eval_parallel.py` uses them to check that parallel evaluations and a resumed one give the same bits (section 6).

**What is not guaranteed, and said so in the files.**
- A checkpoint evaluated in **several attempts** (a resume) is statistically equivalent to one evaluated in a single attempt, **not bit-identical**: the episodes run after the cut start from a server that has not
  played the earlier ones, and the start of an episode after other episodes can differ by hundredths of a millimetre (N1; ~1 environment start in 3, median 0.01 to 0.06 mm, up to 5 to 11 mm). Every line of the
  log has its `attempt`, and the result has `attempts` and `single_attempt`; the probe measures how often a resumed episode really differs.
- Another GPU model, driver or Isaac version may change the bits (N4). Each record has its `gpu`, `eval_protocol.json` has one entry per attempt (host, GPU, XLA flags, git commit, Slurm job), and a run on another
  GPU than the earlier attempts prints a warning.
- The determinism flag's cost and the bit-identity between workers on one GPU are **not measured yet** (section 9).

**For the RL evaluation (to keep).** The same rules apply: the policy's randomness drawn per row from the episode's seed (an RL policy's exploration noise too, or its deterministic mode when it has one, as expo-ft's
`--deterministic`), K environments in a fresh server per evaluated checkpoint, the same master seed (the same starts), one kind of GPU per curve, and the fingerprints. `Sweep`, `run_batch` and
`EpisodeLog` are generic: an RL evaluation only supplies its own `prepare(policy, job)` (load an RL checkpoint) and its own step 0 (the SFT checkpoint the RL started from).

## 4. Resume

Each finished episode is **one line** of `<output-dir>/results/<step>.jsonl`, appended, flushed and fsynced:

```
{"episode": 17, "seed": 1405681631, "batch": 0, "env": 17, "success": true, "terminated": true, "truncated": false, "steps": 84, "start_sha": "...", "traj_sha": "...", "plans": 11,
 "plan_ms_median": 520.1, "tracking_err_max_rad": 0.04, "progress": {...}, "wall_s": 36.4, "decode": null, "video": null, "attempt": 1, "gpu": "NVIDIA L40S"}
```

The same command again reads these files:

- the episodes that are there are **skipped**; the missing ones are run on a fresh server, batch by batch (the same rounds, so the same start states; the episodes already done are held); a line is written
  **the moment its episode ends**, so a stop keeps every episode that had ended and drops the ones still running (a kill of a batch of 100 loses at most the episodes in progress, not the batch); a line cut by a kill (no newline at its end) is dropped and the file is cut back to the last whole line;
- the checkpoint's result (`results/<step>.json`: success rate, standard error, Wilson 95 % interval, per-episode outcomes, `eval_sha`, attempts, GPUs) and the curve are **recomputed from the files**, so they
  are what an evaluation that was never cut would give (up to N1 above, which `single_attempt` records);
- a checkpoint whose `.json` exists and whose episodes are all there is not run again (no server is started for it); `--n-episodes` can grow, and only the new episodes are run;
- a log from another evaluation (an episode whose seed is not the one of the list) and a damaged whole line are errors, not silently dropped; an episode that appears twice keeps its first record, with a warning;
- one evaluation at a time per output folder (`eval.lock`, `flock`): two would append to the same files.

**What may not change between two runs on a folder** (`eval_protocol.json`): `--seed`, `--replan`, the episode length, the number of environments per server (K), the model YAML as the run saved it, the sha256 of its
normalisation statistics, the task YAML with its `--env-set` overrides, and the recipe name (`batched-v1`). A change is refused with the list of what differs. The number of episodes, the checkpoints, `--workers`, the videos and the title may change.

## 5. Parallel evaluation of several checkpoints

`--workers N` evaluates N checkpoints at the same time.

- **One worker is one thread.** It takes the next checkpoint of a queue (by step), puts its weights in its own policy (`OpenPIPolicy.fork`, then `load_checkpoint`), starts a **fresh env server** of K environments, runs the missing
  episodes batch by batch, stops the server, takes the next one. Fresh for each checkpoint, not one server kept per worker: the physics is bit-identical only for an identical sequence of calls from a fresh server (G1), and
  a start-up (a few minutes) is small against two hours of episodes.
- **The server is a subprocess**, not a thread: Isaac Sim allows one `SimulationApp` per process. It runs `jobs/env_server.sh` (K environments, a free port) in its **own process group**, with the learner's
  venv and XLA / JAX variables taken out of its environment, its log in `<output-dir>/logs/server_<step>.log`. The threads only supervise the subprocesses and drive the policy.
- **Memory.** The policies of all the workers share the compiled sampler and the base weights in GPU memory (`fork`; `load_checkpoint` replaces only the trainable parameters; checked in the tests: the frozen
  weights are the same device arrays): N workers cost about N env servers (about 8.1 GiB + 26-29.5 MiB per environment each, `docs/envs/isaaclab_env.md` section 6: about 11 GiB for K = 100; the model at batch K needs more than at batch 1: not measured, section 6.1) plus **one** model (about 8 GB for pi0-FAST at inference) plus the LoRA
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

**Where the divergence comes from: `jobs/job_probe_determinism.sh`** (`scripts/probe_determinism.py`).
- `--part policy --model pi05 --processes 2` (and pi0_fast): no simulator. The same observation and seed through the real model, five ways (the same policy re-seeded, the same calls without re-seeding done twice, a `fork`,
  several forks in several threads, fresh Python processes), comparing the bits of the action chunk. All identical: the model is bit-reproducible and the physics diverges.
- `--part env --servers 3`: no model. The same 150 open-loop joint targets (start pose plus a fixed sinusoid) played on fresh servers, each twice, comparing the readings step by step: the first step that differs and
  how far apart the traces are at 1, 5, 10, 25, 50, 100, 150 steps; and, for the first play of every server, the two camera images (first differing step, fraction of pixel values that differ, largest difference).
  Measured (12 Nov. 2026): the readings are identical across 3 fresh servers over 150 steps; the second play in each server differs from the first, by the same amount in all three (N1, reproducible). The images (13:04): different between fresh servers from step 0 on, 35 to 57 % of the values, up to 49 levels; the script now also prints the mean difference and the shares above 2 and 8 levels, writes PNGs
  of the frames and of their difference (x8), and takes `--rendering-mode {performance,balanced,quality}` and `--server-arg=ARG` (e.g. `--server-arg=--kit_args=--/rtx/post/aa/op=0`) to restart the servers with another renderer setting.

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

**The second probe (separate folders, valid) confirmed the speeds and measured the rest.** pi0.5, one worker: 89 s per episode of which 1.8 s in the model; pi0-FAST: 107 s of which 17.9 s. **The simulator is
the cost** (about 0.58 s per step, 150 steps, the same for both models; the env server alone uses about 80 % of the GPU): this is the measured cost model of `docs/envs/isaaclab_env.md` (a step is 531 ms fixed plus 2.29 ms per
environment), and why two servers on one GPU take twice as long each. XLA's deterministic ops cost nothing (89.2 s with and without). Bit-identity failed: the two replicas of two workers differ from one worker and from each other
from episode 0 on, for both models, and a third run with one worker alone differs from the first (above). Not concurrency, then. Next: `jobs/job_probe_determinism.sh` (below).

**The determinism part of the very first probe is void.** The two jobs (pi0.5 and pi0-FAST) started in the same second and so wrote in the same folder: their episodes were mixed in the same files (the log's
"1 episodes appear twice", the pi0-FAST job's crash on the existing `resume/` folder), and every "bit-identical: False" compares one model with the other. It has to be run again (the probe now refuses to
share a folder); the speeds above do not depend on it, they are the per-episode times of each job's own log lines.

### 6.1 Cost of K environments per server

`jobs/job_probe_batch_cost.sh` (`scripts/probe_batch_cost.py`, no simulator), 12 Nov. 2026, L40S, policy limited to half the card (22.3 GiB), XLA deterministic ops. Median ms per call (ms per row), peak memory:

| rows in the call | 1 | 8 | 12 | 16 | 20 | 32 | 50 | 64 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| pi0.5 | 102 (102) | 680 (85) | | | | 2 731 (85), 10.4 GiB | 4 425 (88), 12.6 GiB | 5 748 (90), 14.2 GiB | **out of memory** in one call (11.8 GiB allocation) |
| pi0-FAST | 439 | 1 108 (139), 9.4 GiB | 1 582 (132), 11.4 GiB | 1 007 (63), 13.3 GiB | **out of memory** (9.6 GiB allocation) | out of memory | | | |

**100 rows split by `--model-batch` (what the evaluation does):** pi0.5 (64 -> two calls of 50): **9.19 s per plan** (92 ms per row), 14.2 GiB peak. pi0-FAST (16 -> seven calls of 15): 41.2 s per plan, 13.3 GiB peak, **not representative** (below).
- pi0.5's time is **linear in the rows** (about 90 ms per row: the card is saturated from B = 8), so grouping rows does not make the model faster and splitting costs nothing.
- pi0-FAST: the logits of the prefill (rows x 948 tokens x 257,152 words in bf16) limit one call to **16 rows** (13.3 GiB; 20 fails): `--model-batch 16` (the default) gives 100 rows as 7 calls of 15. Its **time is not a function of the rows**: a call lasts as long as the longest
  token string it generates (up to 256 steps), and the probe feeds random-noise images, whose strings are unpredictable (12 rows took longer than 16; the 7 calls of 15 took 5.9 s each against 1.0 s for 16). So the probe's 41 s per plan is NOT an estimate of an evaluation (the SFT models write about 11 tokens);
  only a real batch says it.
- Host memory of the model alone: 4.9 GB (pi0.5) and 6.6 GB (pi0-FAST), about one core (`sacct`).
- **Several servers per checkpoint do not pay off on one card**: pi0.5's model is about 60 % of a batch and compute-bound; the simulator's fixed cost (about 80 s) is paid once per batch.
- **Estimate of a checkpoint of 100 episodes** (simulator from `docs/envs/isaaclab_env.md`: 150 steps x (0.53 + 0.0023 x 100) s = 114 s, not measured with K = 100; every episode running its 150 steps, so an upper bound; about one minute more to start the server and load the checkpoint):
  pi0.5: 114 s + 19 plans x 9.19 s = 175 s = **4.8 min, about 5.8 min with the start** (86 s alone before: 2.4 h); 21 checkpoints (step 0 and a save every 200 of 4000 steps) **about 2 h**.
  pi0-FAST (30 plans): 114 s + 30 x (7 calls of 1.0-1.6 s if the strings are short = 210-340 s) = 5.4-7.5 min, **21 checkpoints about 2.2-3 h**; if its strings are long like the probe's, 22 min per checkpoint, about 8 h. The 14 h limit of the job holds either way.
To measure the real thing (the simulator at K = 100, the strings of a real pi0-FAST): `sbatch jobs/job_eval_sft.sh --run <run> --steps 0 --n-episodes 100 --envs-per-server 100 --output-dir <probe dir>`: the log prints the wall time of the batch, `sacct` the host memory, `gpu_<jobid>.csv` the card's memory and utilisation.

### 6.2 First real batches (12 Nov. 2026, L40S, step 4000 of the teleop runs, 100 episodes, K = 100, `jobs/job_eval_sft.sh --steps 4000 ...`)

| | pi0.5 (`peg_insert_pi05_lora_teleop`) | pi0-FAST (`..._pi0_fast_lora_teleop_demo_stats`, the demonstrations' statistics, seed 0: since replaced by `peg_insert_pi0_fast_lora_teleop`, the checkpoint's statistics) |
|---|---|---|
| success | **41 / 100** (SE 4.9, Wilson 32-51 %) | **46 / 100** (SE 5.0, Wilson 37-56 %) |
| chunks planned | 1 529 (of at most 1 900) | 2 427 (of at most 3 000) |
| server ready after | 86 s | 90 s |
| the batch (150 steps) | **309 s** | **1 462 s** (a model call of 15 rows lasts as long as the longest token string: about 50 s per plan) |
| one checkpoint, from the start of the evaluation to its result | 402 s = 6.7 min | 1 558 s = 26 min |
| job (`sacct`) | 7 min 28, 1.5 cores, MaxRSS 29.1 GB of 40 | 26 min 43, 1.2 cores, MaxRSS 29.3 GB of 40 |
| GPU memory (accounting) | policy 16.8 GiB + server 10.6 GiB | policy 16.8 GiB + server 10.8 GiB |

**All 21 checkpoints of a run** (step 0 and a save every 200 of 4000 steps): pi0.5 about **2.4 h**, pi0-FAST about **9.1 h**, both inside the 14 h of the job. These are the figures of the final checkpoint, which ends 41-46 % of its episodes early: a weaker checkpoint runs more
episodes to the full 150 steps and takes longer (pi0.5 at most about 8 min). Two GPUs for pi0-FAST (two jobs on different checkpoints) need a lock per checkpoint instead of per folder, which is not written.
- The 41 % of pi0.5 matches expo-ft's evaluation of the same kind of start (teleop SFT, step 1200: 40 %, 100 fixed episodes); nothing in the log points to a fault (every episode ended, the failures all ran the full 150 steps, the successes are spread from 34 to 144 steps).
- pi0-FAST: about 4.5 % of the chunks could not be decoded in this run (the warnings the tokenizer printed, 109 distinct ones out of 2 427 chunks; the exact count is the `decode` of every episode, and the batch line now prints it). Those chunks are the model's zero action, executed for 5 steps.
  The tokenizer's warnings (printed twice per chunk) are now dropped: the failures are counted instead.

### 6.3 Is a difference with expo-ft the training or the evaluation? (answered: the training, or its seed; see 6.4)

Expo-ft's curve for the teleop pi0.5 SFT (its step 1200, 40 %, was the RL start) reaches **66 % at step 4000**; this repository's teleop SFT (same demos, same noisy normalisation statistics, same optimiser settings: constant 2.5e-5, no EMA, batch 64, LoRA 16/32) gives **41 % at step 4000**
(12 Nov. 2026, 100 episodes, SE 4.9; the difference is about 3.7 standard errors of a difference). Evaluating expo-ft's own step-4000 weights **with this evaluation** separates the two explanations:
`python -m scripts.adopt_expo_ft_checkpoint --checkpoint <expo-ft run>/sft/expo_pi05_droid_lora_finetune_sft_joint_state_delta/<dataset>/4000 --name expoft_teleop_4000`, then
`sbatch --time=00:45:00 jobs/job_eval_sft.sh --run _adopted_expoft_teleop_4000 --steps 0 --n-episodes 100 --envs-per-server 100 --output-dir runs/sft/probing/_cost_expoft_teleop_4000` (its only checkpoint is called step 0).
About 66 %: the evaluation agrees with expo-ft's and the difference is in the training here (data pipeline, prompt, augmentation, ...). About 41 %: this evaluation differs from expo-ft's (rendering of 100 environments, replan, sampling, ...). Between the two: look at both.

**First run (12 Nov. 2026, job 11676974): 0 / 100, neither answer.** Cause, found the same day: `build_params_like_expo_ft` (src/models/loading.py) initialised EVERY LoRA tensor of the model and put the result over the loaded weights. That is right for a checkpoint without LoRA tensors (pi05_droid_jointpos: the LoRA starts from openpi's init), but an expo-ft SFT step holds the TRAINED lora_a / lora_b: they were replaced by random ones, which gave the base model plus a perturbation (about 0 % on FORGE). Fixed: only the LoRA tensors the checkpoint lacks are initialised
(`tests/test_loading.py::test_a_checkpoint_that_holds_lora_tensors_keeps_them`). Nothing else is affected: for a checkpoint without LoRA tensors the result is identical to before, so every other evaluation of this repository (step 0, the SFT checkpoints, which load their trained parameters on top of the initial model) stands. The test was run again (job 11677187, 12 Nov. 2026): **60 / 100** (SE 4.9, Wilson 50-69) for expo-ft's step-4000 teleop weights, against 66 % on expo-ft's own curve: the evaluation reproduces expo-ft's. The remaining gap to this repository's own SFT (41 % at step 4000, same evaluation, same starting states) is a difference of the TRAINING, or training noise (one seed per configuration, not measured): the full curve of the run decides (a plateau of about 40 % between steps 2600 and 4000 against expo-ft's 60-70 % points to a systematic difference; scattered points, to noise).

Two ways of rebuilding a fine-tuned model, which must agree: ours (the run's initial model, with the TRAINABLE parameters of `<ckpt>/params` laid over it, `sft.load_checkpoint`) and expo-ft's (a `params/` with every weight, through `build_params_like_expo_ft`). `tests/test_loading.py::test_our_checkpoint_route_and_expo_fts_full_params_route_give_the_same_weights` builds a tiny pi0.5, trains (perturbs) its LoRA and a non-LoRA part in float32, saves them with `save_checkpoint` (the trainable parameters only) and as a full `params/`, loads both ways and requires identical bfloat16 weights, leaf by leaf (it fails with the first version of `build_params_like_expo_ft`).

### 6.4 Full curve of the teleop SFT and the loss test (12 Nov. 2026)

Full curve of `peg_insert_pi05_lora_teleop` (21 checkpoints, 100 episodes each, K = 100, seed 42): 0, 7, 21, 19, 38, 35, 32, 29, 28, 35, 31 % for steps 0 to 2000, then 45, 40, 46, 55, 47, 40, 50, 45, 37, 43 % for steps 2200 to 4000. The ten points of the plateau have a mean of 44.8 % and a standard deviation of 5.2 points, which is the binomial noise of one point (SE about 5): the curve is flat around 45 %, no checkpoint is an unlucky draw.
Step 4000 evaluated twice (same weights, starting states and seeds, camera images not bit-reproducible): 41 %, then 43 %. expo-ft's own step-4000 weights through this evaluation: 60 % (their curve: 66 %). So the evaluation reproduces expo-ft's and is repeatable, and the 15-20 points that remain between our SFT and theirs come from the TRAINING (the recipe) or from the seed of ONE training run (not measured: a second seed is about 9.7 h).
Compared and found identical: AdamW settings, constant learning rate, batch 64 (gradients of the micro-batches summed then divided by their number = the mean), no EMA, float32 trainable parameters, openpi's image augmentation, chunk targets (offsets from the chunk-start state, last action repeated), lossless images (LeRobot `image` features are PNG), the prompt (the one of the .npz), the normalisation statistics (sha256). Compared afterwards, see 6.5: the exact content of a training batch against openpi's data path, one update of our loop against openpi's `train_step`, the set of trainable parameters. Not compared: the order in which rows are drawn (random in both), the content of the logs of the two cluster runs.

`scripts/probe_sft_loss.py` (`jobs/job_probe_sft_loss.sh`) makes the first of these comparisons without a new training: the loss of expo-ft's step-4000 weights on batches built by OUR training path (`OpenPIPolicy.training_batch`), against the loss of our checkpoints on the same rows and the same noise keys (paired, with a standard error over rows).

    sbatch jobs/job_probe_sft_loss.sh --checkpoint peg_insert_pi05_lora_teleop:0 peg_insert_pi05_lora_teleop:2000 peg_insert_pi05_lora_teleop:4000 _adopted_expoft_teleop_4000:0

Reading: their loss much HIGHER than ours = what their model was trained on is not what our path builds (a mismatch of inputs or targets to find); LOWER = same pipeline, our optimisation fits the demonstrations less well; about EQUAL = the loss does not tell them apart (sampling order, augmentation, the seed). The rows are training rows: a lower loss is a better fit, not better generalisation. Result: theirs LOWER on training rows by 9 % (paired, +4.3 SE of the rows), equal on unseen rows; the standard error is over ROWS of ONE run each, it does not contain the run-to-run variation of the training, so it does not say that the recipes differ (see 6.5).

### 6.5 Everything that could differ between our SFT and expo-ft's, checked in the sandbox (12 Nov. 2026)

Same openpi commit (f7cfe2d, fork `expo_ft`), same jax 0.5.3 / flax 0.10.2 pins. Each point below is a test that fails when the item is made different (`tests/test_training_data_equals_expo_ft.py`, `tests/test_sft_step_equals_openpi_train_step.py`), plus one count.

- **The training samples are the same, tensor by tensor.** Synthetic episodes (including the last frames, whose chunk repeats the last action) go through expo-ft's chain (LeRobot sample with float32 CHW images, `Repack`, `DroidInputs`, `DeltaActions`, `Normalize` with the quantiles of `forge_demos_robotiq_noisy`, resize, padding to 32) and through ours (`DemoSet.sample`, `rows_of`, `targets_to_chunk`, the input transforms of the plain config, again through `OpenPIPolicy.training_batch` itself): state, normalised actions, images, image masks and prompt are equal. (The tokenizer is left out of both: it needs a download, and its tokens are a function of the prompt and of the normalised state.) The transforms return float64, JAX rounds to float32 on the way to the device in both loaders. The LeRobot round trip of an image (uint8 to float32 / 255 and back to `uint8` in `DroidInputs`) is exact for the 256 values.
- **One update is the same.** From the same weights, batches and noise key, four updates of our pieces (`make_optimizer`, trainable float32 / frozen bfloat16, the gradient of `OpenPIPolicy.loss`) give the loss, the gradient norm and the parameters of openpi's own `train_step` (taken from its `scripts/train.py`). A learning rate 20 % off makes the test fail. This covers the loss function, the clip, AdamW, the constant schedule, the dtypes and the freeze mechanism. Accumulating micro-batches (sum, then division by their number) is the mean of the batch; the noise of a row differs between the two loops (different keys), which is not a difference of distribution.
- **The same parameters are trained.** `expo_pi05_droid_lora_finetune_sft_joint_state_delta` and our YAML (through `apply_settings`) give the same 51 trainable tensors: 466 957 072 trainable and 2 936 464 384 frozen parameters (SigLIP and the projections included). The optimiser, the schedule, the batch and the absence of EMA were already read identical.
- **Not different in a way that matters.** Our trainable parameters start from the bfloat16 rounding of the checkpoint (it is cast on the host), expo-ft's from its float32 values: the forward pass rounds them to bfloat16 anyway, the updates accumulate in float32 on both sides, and what is evaluated is bfloat16 on both sides. expo-ft's 4001 iterations against our 4000 updates. `model.train()` only sets a flag no module reads (all dropout rates are 0).
- **The demonstrations are the same ones.** On OUR 50 episodes the loss of expo-ft's weights is about 0.003, as ours; on unseen noisy demonstrations 0.036 for both. A model trained on other frames would not sit at 0.003 on these.

What is left is not in the code: (1) the seed of ONE training run (the 9 % lower training loss and the 15-20 points of success are single draws; the run-to-run spread of an SFT on 50 demonstrations is not measured here), (2) what the two cluster runs actually used, which only their logs show: number of frames, number of trainable parameters, the loss and gradient-norm trajectories, `config.json`. `scripts/compare_eval_paired.py` adds the episode-by-episode comparison of two evaluations (same starting states): whether the failures are on the same states, and an exact test on the discordant episodes.

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
| one checkpoint | a subprocess per checkpoint (`eval_policy.py`), episodes one after the other | threads in one process (one shared compiled model), one env server of K environments per checkpoint, the episodes of a batch in parallel |
| several checkpoints | one after the other | `--workers N` at the same time |
| randomness of the policy | not seeded per episode (`--deterministic` only for the RL's residual actor) | per row from the episode's seed for both models, XLA deterministic ops, fixed batch K |
| bit-level check | none | `start_sha`, `traj_sha`, `eval_sha`, `attempt`, `gpu`, the protocol file and its check on resume |
| curve and data location | `--output-dir` for everything (default: the checkpoints folder) | `--output-dir` (default `<run>/eval`) and `--curve-dir` |
| title | 15 pt | 9 pt, wrapped |
| initial point | `base` (the base model) | `0`: the run's own initial state |

## 9. Checked and not checked

Checked (CPU): the seeds (expo-ft's, a longer list extends a shorter one, a seed file of another master seed refused); the log (a line cut by a kill, a damaged line, duplicates, concurrent appends); the
numbers; the protocol check; a batch of episodes on the fake simulator over a real socket (a record the moment each episode ends, a fresh server gives the same bits, held environments change nothing for the others, the noise of the policy a function of the episode and not of the batch or the history, a stop keeps the episodes that had ended, an episode that does not end is an error); on a tiny pi0.5 that the noise of a row depends only on its seed and that `pad_to` fixes the batch; the sweep (every checkpoint on the same seeds, one or three workers give the same results, servers start one at a time, a stopped sweep
keeps its episodes and the same command finishes it with the results of an uncut one, several batches use rounds 0, 1, 2 of the master seed and a cut batch is resumed in its own round, a server with another number of environments than the protocol's is refused, a finished checkpoint starts no server, more episodes run only the new ones, a failing checkpoint or policy does not stop the
others, every server is stopped); the server subprocess (ready line, an old ready line not mistaken, a death, a timeout, a stop while waiting, SIGTERM then SIGKILL, the whole process group killed); the script
(`--dry-run`, `--plot-only`, the refusal of another protocol, the lock, `--force`); the curve (rebuilt from complete checkpoints only, the title small and inside the figure); the probe's analysis; and
on a tiny pi0.5 that `OpenPIPolicy.fork` gives the same bits as its parent for the same seed, has its own stream and weights, and shares the device arrays, and that `load_checkpoint` on a fork leaves the base
policy untouched and the frozen weights shared.

**Not checked, to be seen on the cluster:**
- everything that needs Isaac Sim and the real models together: a first evaluation of a real checkpoint, the server subprocess started from the learner's venv through `jobs/env_server.sh`, the `ready` line, the
  clean-up of Isaac's child processes;
- that the model gives the same bits in two compilations, two threads and two processes (`--part policy`); its flag `--xla_gpu_deterministic_ops` costs nothing (measured);
- the cost of K environments per server on the real simulator (section 6.1: the model's side is measured, the simulator's at K = 100 is the docs' figure), pi0-FAST's `--model-batch`;
- the physics over 150 steps from a fresh server (`--part env`); the speed-up of several workers is measured (none on an L40S); the host memory of an Isaac server
  (`--mem=48G` for one worker is an estimate, to be checked with `sacct` after the first evaluation job);
- the size of N1 for resumed episodes (the probe's cut-and-resume check), and the effect on a success rate over hundreds of episodes (N6 of the env's reproducibility section);
- the compilation of the shared sampler by several threads at once;
- what the model and the simulator each take of the 86 s / 107 s of an episode (the corrected probe prints it);
- the default `--replan` of pi0-FAST (5 of 10): expo-ft's 8 of 16 transposed, not measured.
