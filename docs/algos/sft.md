# SFT: supervised fine-tuning on the FORGE demonstrations

`python -m scripts.sft --experiment <file> | --model <pi05 | pi0_fast>` fine-tunes a VLA on the demonstrations of `docs/demos.md` (the same as expo-ft's). The result is a
checkpoint the RL starts from. Settings: `configs/algo/sft.yaml`. Code: `src/algos/demos.py` (data), `src/algos/sft.py` (loop, checkpoints), `src/algos/tracking.py` (wandb),
`scripts/sft.py` (command line). **Status: tested on a tiny model on the CPU; probed 30 steps on the cluster, complete for pi0.5 LoRA (both cards), pi0-FAST LoRA, pi0-FAST full and pi0.5 full, wandb online included (section 4)** (section 9).

## 1. Commands

| What | Command |
|---|---|
| a main run (LoRA, on an L40S) | `sbatch jobs/job_sft_lora.sh --experiment configs/experiment/sft/pi05_lora_scripted.yaml`: see "Main runs and probes" below |
| look at the data and the run folder, load no model (login node, no GPU) | `python -m scripts.sft --experiment configs/experiment/sft/pi05_lora_scripted.yaml --dry-run` |
| a probe: LoRA on an L40S, demonstrations of the YAML | `sbatch jobs/job_sft_lora.sh --model pi0_fast` (goes to `runs/sft/probing`) |
| a probe, other demonstrations | `sbatch jobs/job_sft_lora.sh --model pi05 --demos demos/isaaclab/ForgePegInsert_Robotiq_noisy --run-name pi05_noisy` |
| full fine-tuning on an A100 80 GB | `sbatch jobs/job_sft_full.sh --model pi05` |
| another rank | `sbatch jobs/job_sft_lora.sh --model pi05 --model-set vlm.lora_rank=8 --model-set action_head.lora_rank=16` |
| shorter run, smaller micro-batch | `sbatch jobs/job_sft_lora.sh --model pi05 --set steps=1500 --set micro_batch_size=8` |
| continue a run that stopped | `sbatch jobs/job_sft_lora.sh --resume runs/sft/peg_insert/<run name>` (or `job_sft_full.sh`): see section 6 |
| the success-rate curve of a run (step 0 and every checkpoint, in the env) | `sbatch jobs/job_eval_sft.sh --run runs/sft/peg_insert/<run> --n-episodes 50`: `docs/eval/sft_eval.md` |
| follow a run | wandb (project `tactile-vla-rl`, section 5b), or `tail -f runs/sft/peg_insert/<run>/metrics.jsonl` (or `logs/slurm/sft_lora_<jobid>.out`) |
| no wandb, or wandb offline | `--set wandb_mode=disabled`, `--set wandb_mode=offline` (then `wandb sync <run>/wandb/offline-run-*`) |

### Main runs and probes

- **Main runs** are the experiment files of `configs/experiment/sft/`: one file = one run (`name`, `model`, `full`, `demos`, `set`, `model_set`, `description`; any other key is refused). They go to
  **`runs/sft/peg_insert/<name>/`** (`output_root` in their `set`). What is given on the command line is added to the file and wins (`--set steps=100`, `--run-name`).
- **Probes** (a run to measure or look at something, not a main run) have no experiment file: their default `output_root` is **`runs/sft/probing`**.
- The first five SFTs of PegInsert, all LoRA, without touch (the SFT of expo-ft: cameras, joints, gripper, prompt; the taxels in the demonstrations are not read by the model). The three pi0.5 runs, and
  a PAIR of pi0-FAST runs on the teleoperated demonstrations (the best ones) that differ only by the normalisation statistics (section 2b). The scripted and noisy pi0-FAST runs come after the pair:

| experiment file | run folder under `runs/sft/peg_insert/` | model | demonstrations | normalisation statistics |
|---|---|---|---|---|
| `pi05_lora_scripted.yaml` | `peg_insert_pi05_lora_scripted` | pi0.5 | `ForgePegInsert_Robotiq_scripted` | noisy demonstrations, 16 steps |
| `pi05_lora_noisy.yaml` | `peg_insert_pi05_lora_noisy` | pi0.5 | `ForgePegInsert_Robotiq_noisy` | noisy demonstrations, 16 steps |
| `pi05_lora_teleop.yaml` | `peg_insert_pi05_lora_teleop` | pi0.5 | `ForgePegInsert_Robotiq_teleop` | noisy demonstrations, 16 steps |
| `pi0_fast_lora_teleop_droid_stats.yaml` | `peg_insert_pi0_fast_lora_teleop_droid_stats` | pi0-FAST | `ForgePegInsert_Robotiq_teleop` | the checkpoint's (DROID) |
| `pi0_fast_lora_teleop_demo_stats.yaml` | `peg_insert_pi0_fast_lora_teleop_demo_stats` | pi0-FAST | `ForgePegInsert_Robotiq_teleop` | noisy demonstrations, 10 steps |

`--set KEY=VALUE` overrides a key of `configs/algo/sft.yaml`; `--model-set KEY=VALUE` a key of the model YAML (`<component>.<key>`, the
probe's `--set`); `--full` is `use_lora: false` in every component the model has. Before the first job, on the login node:
`uv sync --group vla`, `python -m scripts.fetch_models`, `wandb login` (once), then the `--dry-run`. The jobs run offline for the models (`HF_HUB_OFFLINE`); wandb needs the network of the node (section 5b).

## 2. The data

The `ep_*.npz` files are read directly (`src/algos/demos.py`), not the LeRobot copy the expo-ft SFT used: the same episodes and frames, but the
`.npz` files also hold the touch signals (`tactile`, `ft_force`, `gripper_effort`, `peg_socket_offset`), which are served under the env's
observation keys, and the `lerobot` package is not needed. `fail_seed*.npz` are never used.

A sample is a step `t` of an episode:

- **observation**: what the env sent at `t`, before the action: the two images (224 × 224), the 7 joints, the gripper position, the prompt of the file;
- **target**: the chunk `actions[t : t + horizon]` in the env's action space (absolute joint targets + gripper command), the last action repeated past the end of
  the episode (as openpi's LeRobot loader does, and as `scripts/fast_tokens.py` measured).

`horizon` is the model's `action_horizon`: **16 for pi0.5, 10 for pi0-FAST**. The conversion to the model's own space is not done here: `OpenPIPolicy.training_batch`
applies `output_semantics` (offsets from the chunk-start state for pi0.5, absolute targets for pi0-FAST) and then the same input transforms as at inference
(DROID normalisation with the **checkpoint's** statistics, resize, prompt and, for pi0-FAST, FAST tokens). The SFT therefore trains on exactly what the model
receives when it acts. Folders and options: `demos`, `num_demos` (first N of each folder), `val_episodes` (last V of each folder held out). Several folders can
be trained on together; episodes with and without taxels are not mixed.

### 2b. Normalisation statistics

Which statistics the SFT normalises with is the `norm_stats` block of the model YAML (`dir`, `asset_id`: `<dir>/<asset_id>/norm_stats.json`, `src/models/interface.py`
`norm_stats_source`, read by `src/models/loading.py` `load_norm_stats`). Without the block: the checkpoint's own (DROID's `assets/droid`).

- **pi0.5: the statistics of the noisy demonstrations, for the three demonstration types** (`assets/forge_demo_norm/forge_demos_robotiq_noisy`), as in the SFT of
  expo-ft (`configs/task/isaaclab/peg_insert_forge_sft_robotiq_*.yaml` there): the same file for the three runs, so that the demonstrations are the only difference
  between them; the noisy set was the one that covered the other two best. The three pi0.5 experiment files carry the block (`model_set: norm_stats.*`). The file is
  the one expo-ft wrote (`scripts/demos/compute_demo_norm_stats.py`, identical in both repositories); copy it, do not recompute it.
- **pi0-FAST: undecided, a pair of runs decides.** Its openpi pipeline normalises OFFSETS from the chunk-start state too (`RLDSDroidDataConfig`, JOINT_POSITION: `DeltaActions` before `Normalize`,
  `AbsoluteActions` after the inverse), so statistics computed on the demonstrations have the same kind as pi0.5's and only the chunk length differs (10 steps:
  `compute_demo_norm_stats --horizon 10 --droid-stats <the polaris checkpoint's norm_stats.json>`, asset `forge_demos_robotiq_noisy_h10`). Measured on the 150 demonstrations with
  `scripts/fast_tokens.py` (CPU, 2026-11-12; `--norm-stats-dir` / `--norm-stats-asset-id` give the other statistics):

  | | the checkpoint's (DROID) | noisy demonstrations, 10 steps |
  |---|---|---|
  | reconstruction error of FAST, mean / p99 | 0.0112 / 0.021 rad | **0.0017 / 0.0036 rad** |
  | largest error (the undecodable chunks) | 0.030 to 0.046 rad | 0.070 rad |
  | tokens per chunk, mean / maximum (limit 180, 256 generated) | 56 / 62 | 64 / 92 |
  | chunks openpi cannot decode (zero chunk instead) | 3 of 8913 (0.03 %) | 20 of 8913 (0.22 %; teleop about 16 of 4006) |

  The statistics of the demonstrations are about 6.5 times more precise, with 7 times more undecodable chunks (about 4 % of the episodes would have one, with 19 chunks each). The cause of the
  failures is not established (hypothesis: the tails of the noise, up to ±4 once normalised, give coefficients that a tokenizer built on DROID's ±1 reads badly). A measurement cannot say which
  wins in the environment, so the first pi0-FAST runs are the PAIR `pi0_fast_lora_teleop_droid_stats` / `pi0_fast_lora_teleop_demo_stats`, on the teleoperated demonstrations (the best ones); only the
  success rate of the two settles it (a loss or an error computed with other statistics is not comparable). The scripted and noisy pi0-FAST runs wait for that.
- **The checkpoints carry them**: `checkpoints/<step>/assets/<asset_id>/norm_stats.json` (as openpi does) and the block in `meta.json` (`model_cfg`). A policy that loads
  such a checkpoint (evaluation, RL) must be built with the same `norm_stats` block, otherwise the actions come back un-normalised with the wrong scale. A resume refuses a run
  whose model YAML differs. The dry run prints which file is used and stops if it is missing.

Why not DROID's for pi0.5 (measured on the cluster, 2026-11-12, `convert_waypoint_demos_to_lerobot --dry-run` and `compute_demo_norm_stats`, the three demonstration types):

| | DROID statistics | statistics of the noisy demonstrations |
|---|---|---|
| std of the normalised actions (7 joints) | 0.003 to 0.03 (e.g. scripted: 0.016 0.028 0.008 0.030 0.003 0.028 0.013) | 0.15 to 0.43 |
| range reached by the normalised actions | about -0.21 to +0.19 of the full scale [-1, 1] (teleop: -0.18 to +0.09) | -3.3 to +2.0 (quantile normalisation does not clip: the tails go past ±1) |
| demonstration range / DROID range (q99 - q01, offsets) | 0.01 to 0.12 (joint 5: 0.01, a floor of 0.0184 is applied) | |
| state outside [-1, 1] | none | 2 % per joint (by construction of the quantiles) |

With DROID's statistics the whole motion lives in a few percent of the interval the pretrained model works in: a millimetre-level insertion is a tiny part of the output
range. The demo statistics keep the gripper's DROID ones (the gripper command is constant, 1.0, in the demonstrations) and widen a joint's range to at least 2 % of DROID's
(`--min-range-frac`). The expo-ft SFT checkpoints the RL started from were trained this way (clean 1200, noisy 1400, teleop 1200): this SFT stays comparable with them.

## 3. What is trained, and how

- **Trainable parameters** = `policy.trainable_filter` = openpi's freeze filter for the model YAML, unchanged: with LoRA, the LoRA matrices **and** SigLIP (and,
  for pi0.5, the action projections); without it, everything. Parameter counts are in `docs/models/pi05.md` and `pi0_fast.md` (e.g. pi0-FAST with LoRA: 442.7 M trainable
  out of 2951.2 M).
- **Precision**: the trainable parameters are float32 during the run and the frozen ones bfloat16, as openpi's own `train.py` does. Checkpoints and evaluation use
  bfloat16, what the policy runs with.
- **Loss**: `policy.loss`, i.e. openpi's `compute_loss` with `train=True` (image augmentation included): flow matching for pi0.5, next-token cross entropy for pi0-FAST.
- **Optimiser**: openpi's AdamW (b1 0.9, b2 0.95, weight decay 1e-10, gradient clipping at 1.0), no EMA, learning rate 2.5e-5 constant (no warmup; `lr`, `lr_end`,
  `warmup_steps` give a cosine decay if wanted). These are the values of expo-ft's pi0.5 SFT.
- **Batches**: 64 rows per update, 4000 updates, a new random permutation of the steps every epoch. With 50 demonstrations (about 2 400 steps for scripted) that is about 100
  epochs; expo-ft started the RL from the step 1200 to 1400 checkpoints, which is why a checkpoint is saved every 200 steps.
- **Memory knob**: `micro_batch_size` cuts the batch into passes whose gradients are summed (`batch_size` must be a multiple). It sets the memory, not the optimisation. One
  device, no FSDP.
- A thread prepares the next batches (`prefetch`): openpi's transforms (resize, tokens, FAST) are CPU work.

## 4. LoRA or full fine-tuning

| | LoRA (default) | Full |
|---|---|---|
| model YAML | `use_lora: true`, ranks 16 (VLM) and 32 (action expert) | `--full`: `use_lora: false` in every component |
| job | `jobs/job_sft_lora.sh`, L40S 46 GB, 64 GB of RAM | `jobs/job_sft_full.sh`, A100 80 GB (`a100l`), 128 GB of RAM, micro-batch 8 |
| starting point | the checkpoint plus a small random LoRA (openpi's normal(0.01) init for both matrices: not exactly the base function) | exactly the checkpoint |
| checkpoint | the trainable part only | the whole model |

The experiments use LoRA, as in expo-ft. Measured by 30-step probes (2026-11-11, batch 64, with one checkpoint, one evaluation and, in the second round, one resume snapshot at the end):

| LoRA, micro-batch 16 | pi0.5, L40S | pi0.5, A100 80 GB | pi0-FAST, L40S |
|---|---|---|---|
| memory once loaded | 11.8 GiB | 11.9 GiB | 10.6 GiB |
| memory peak (training) | 23.9 GiB (of 45) | 23.7 GiB | 22.0 GiB |
| first step (compilation) | 108 s | 114 s | 87 s |
| seconds per step | 7.7, then 8.7 | 5.4, then 6.0 | 7.9, then 8.9 |
| evaluation (16 chunks) | 34 s (first, compilation included) | 35 s | out of memory (below) |
| checkpoint | 698 MiB, 2.5 s | 698 MiB, 3.4 s | |
| resume snapshot | | 4.5 GiB, 15 s | |

- **Time.** The step is compute-bound and the card decides: **6.0 s on the A100, 8.7 s on the L40S** (x1.44). 4000 steps are about **7 h on an A100** (6.7 h of steps, a quarter of an hour of evaluations and saves) and
  **10 h on an L40S** (the LoRA job asks for 14 h, which holds for both). expo-ft's SFT, same model, batch, steps and learning rate, ran on an A100 (`job_forge_sft.sh`) in about 6 h: the same order, so the loop
  is not slower than openpi's. The two models cost the same per step. A larger micro-batch would not speed it up much, 16 stays. After 30 steps the loss went 1.56 -> 0.14 (pi0.5, flow matching) and
  1.22 -> 0.41 (pi0-FAST, tokens), and pi0.5's own chunks were 0.048 rad (mean over the 7 joints) from the demonstration's, 16/16 valid. 20 checkpoints of 698 MiB (the size at step 30) are 14 GB.
- **Round 1 failures** were the allocator's 75 % cap (`XLA_PYTHON_CLIENT_MEM_FRACTION`, 0.75 by default even with `PREALLOCATE=false`): the full runs died at 57 to 59 GiB in use (75 % of 80 GB with the next buffer),
  and the jobs now raise it (0.9 LoRA, 0.95 full; `memory_mib.limit` shows it: 77,094 MiB on the A100, 41,138 on the L40S).

Round 3 (after the cap was raised, the policy's copy of the weights freed and the evaluation loss cut in 4-row passes), 30 steps, batch 64:

| | pi0-FAST LoRA, L40S | pi0-FAST full, A100 | pi0.5 full, A100 |
|---|---|---|---|
| state once loaded | 10.1 GiB | 33.2 GiB | 38.1 GiB (was 44.2: the policy's copy is gone) |
| after a step (`in_use`) | 14.0 GiB | **55.7 GiB** | |
| peak | 21.8 GiB (of 40) | **72.1 GiB** (of 75.3) | out of memory at the first step: 75.2 GiB in use of 75.3 |
| seconds per step | 8.0, then 8.9 | 6.5, then 7.25 (4001 steps: about 8 h) | |
| first step | 86 s | 77 s | |
| outcome | complete: evaluation 66 s (16 chunks decoded, 0 failures, 11 tokens each), checkpoint 662 MiB, snapshot 4.0 GiB in 8.7 s | **out of memory in the evaluation** (a 7.8 GB allocation) | |

- **What the numbers say.** The state is what it should be (`in_use` at the start = float32 parameters + the two AdamW moments, 12 bytes per parameter). But **after each step about two gradient trees (2 x 10.9 GiB for pi0-FAST full, 2 x 1.6 GiB
  for LoRA) stay allocated**, in every probe, old code and new. Within a step the peak is then the state + those 2 + the new gradients (3 trees) + activations: 72 GiB for pi0-FAST full, and for pi0.5 (12.5 GiB per tree) more than the
  75 GiB the card allows; and the evaluation of pi0-FAST starts from 55 GiB, so its 7.8 GB allocation (the logits of the 16-row prefill of `plan`, 16 x 948 tokens x 257,152 words in bfloat16) fails. The two failures are not a node
  problem: `in_use` is the process's own allocation and reached its own limit (76,964 of 77,094 MiB for pi0.5), which another job could not cause.
- **Correction.** The gradient trees were kept alive by their python names (`grads` of the last micro-batch, `total` after the update: a donated buffer that no output reuses is not given back). They are now
  dropped as soon as they are used (`del grads`, `total = None`) and only the buffers that an output reuses are donated.

Round 4 (with that correction), 30 steps, batch 64, A100:

| | pi0-FAST full | pi0.5 full (round 5, below) |
|---|---|---|
| state once loaded | 33.2 GiB | 38.1 GiB |
| after a step (`in_use`) | **33.7 GiB** (was 55.7) | **38.6 GiB** |
| peak | **61.0 GiB** of 75.3 (was 72.1) | **68.1 GiB** of 75.3 (69,733 MiB of 77,094 at step 30) |
| first step | 77 s | 99 s |
| seconds per step | 6.5, then 7.3: 4001 steps about **8.1 h**, about 8.6 h with the evaluations, saves and snapshots | 6.4, then 7.1 to 7.3: 4000 steps about **8.0 h**, about **8.7 h** with the evaluations, saves and snapshots |
| evaluation (16 chunks) | 56 s; 0 decoding failures, 11.4 tokens per chunk | 54 s (compilation included); 16/16 valid chunks |
| checkpoint (the whole model, bfloat16) | 4.3 GiB, 17 s | 4.9 GiB (5,008 MiB), 41 s |
| resume snapshot | **29.0 GiB, 77 s** | **29.0 GiB (29,722 MiB), 160 s** |
| `param_norm` | 1819.6, moving in the 3rd decimal | 1832.708 -> 1832.721, moving in the 3rd decimal |

- The hypothesis is confirmed: `in_use` after a step is the state again, the peak fell by 11 GiB and the evaluation fits.
- Round 5 (pi0.5 full, A100, job 11665035, 30 steps, batch 64, micro-batch 8) is complete: the first attempt of round 4 had been **preempted by Slurm** after 3 minutes (state PREEMPTED, no error in the log: not a failure of the code).
  Loss 1.08 (step 10) -> 0.070 -> 0.014 (step 30); gradient norm 11.7 -> 3.4 -> 1.1; at step 30 the evaluation (16 training rows) gives a loss of 0.0054, 0.0105 rad mean error over the 7 joints of the whole chunk (0.0084 at its first step) and 0.043 on the gripper.
  The peak grows slowly (69,086 -> 69,733 MiB over 30 steps) and leaves 7.3 GiB under the allocator's limit.
- Disk of a full run: pi0-FAST 20 checkpoints of 4.3 GiB = 87 GiB; pi0.5 20 of 4.9 GiB = 98 GiB. Plus the snapshot (29 GiB, twice that while it is swapped): `--set save_interval=1000` when the disk matters.
  The snapshot sizes were measured after 30 steps, when the AdamW moments compress well; the upper bound is 3 times the float32 parameters (37.5 GiB for pi0.5, 32.6 GiB for pi0-FAST), and a longer run may write more
  and take longer than the 160 s measured (pi0.5). Even at twice that it is far inside the 10 minutes between the SIGTERM and the end of the job. It is longer than the roughly 30 seconds Slurm leaves after a
  **preemption**: a preempted full run only has the periodic snapshots (every 1000 steps, about 2 h), not a final one.

## 4b. Where the GPU waits (the cluster's under-use notice, 2026-11-12)

The 30-step probes are short, and most of their time is not training: the pi0.5 full probe (job 11665035) ran 605 s on the card; the nvidia-smi report gave 37 % utilisation, and the 210 s of its 30 steps
are 35 % of that. Compilation (99 s), the evaluation (54 s), the checkpoint (41 s) and the snapshot (160 s) keep the card nearly idle, and so does the loading of the model before them. A notice about such a job
measures the probe, not the real run. For a real run of 4000 steps, from the measured times:

| | steps | compilation | evaluations | checkpoints | snapshots | card busy |
|---|---|---|---|---|---|---|
| pi0.5 LoRA, L40S | 9.7 h | 108 s, once | 20, at most 34 s each | 20 x 2.5 s | 8 x 15 s | about 97 % |
| pi0.5 full, A100 | 8.0 h | 99 s, once | 20, at most 54 s each | 20 x 41 s | 4 x 160 s | about 91 % |

During the steps the card is busy: the batches are prepared by a thread ahead of the step (`prefetch`), JAX dispatches without waiting (the first host synchronisation is at the first `log_interval`), and a rough count of
the work (about 770 tokens per sample through the 2 B Gemma and the three SigLIP image slots, the forward done twice because of openpi's remat, once more backwards) gives about half of the A100's bf16 peak during a full step, which is
what XLA reaches on such models. So the loop has no large margin left; what remains is below.

Done (no change to what is trained or computed):

- **Compilation cache.** `scripts/sft.py` turns on JAX's persistent cache (`~/.cache/jax`, openpi's and expo-ft's folder; `JAX_COMPILATION_CACHE_DIR` to choose another): the 100 s of the first step, and of the first
  evaluation, are paid once per program and per type of card, not at every launch and every resume.
- **Measures, to see the real run instead of guessing.** Each `train` line has `batch_prep_s` (seconds the feed thread takes to prepare one batch; the GPU waits for data only if it exceeds `step_s`), and both SFT jobs write
  the card's utilisation and memory every 30 s to `logs/slurm/gpu_<jobid>.csv`.

Looked at and not done:

- **Saving in the background.** Checkpoints and snapshots stop the card for about 25 minutes of a full run (3 % of its time) and for 3 minutes of a LoRA run (0.5 %): the experiments are LoRA, and the change (host copies of 29 GiB,
  a thread to join at every stop) is not worth that.
- **Two runs on one card.** The steps are compute-bound: two LoRA runs on one A100 would each take about twice as long.
- **A larger micro-batch.** The noise and the augmentation of a micro-batch come from `fold_in(key, step * n_micro + i)`: a different number of micro-batches draws other numbers, so the run would no longer be the same one.
- **Remat policy (needs a decision).** openpi's Gemma recomputes the forward pass in the backward (`nothing_saveable`, hard-coded in `gemma.py` of the pinned fork). Keeping activations would remove that
  recomputation, a quarter to a third of the step if the memory allows it (LoRA on an A100 has room, the L40S probably not), with the same maths. It means changing the pinned openpi commit (checked by `tests/test_models_pin.py`),
  so it is not done; a probe on the cluster would tell the gain and the memory.

## 5. What a run writes

`<output_root>/<run name>/`: `runs/sft/peg_insert/<name>/` for an experiment file, `runs/sft/probing/<model>_<demos>_<lora|full>_<date>/` for a probe (`runs` is a link to the scratch disk; `--run-name` chooses the folder):

| File | Content |
|---|---|
| `config.json` | the algorithm settings and the model YAML as run (overrides included); read back by `--resume` |
| `metrics.jsonl` | one JSON line per event: `start` (parameter counts, sizes), `error` (the exception and the memory, when the run fails), `first_step` (seconds including the compilation, memory), `train` (every `log_interval`: loss, gradient norm, `param_norm`, learning rate, seconds per step, `batch_prep_s`, memory), `eval` (with `eval_s`), `checkpoint` (with `save_s` and `size_mib`), `resume_snapshot` (same), `resume`, `done` or `stopped`; every line has `elapsed_s` |
| `wandb_id.txt` | the id of the wandb run of this training (section 5b); `--resume` reads it to continue the same run |
| `wandb/` | the local files of wandb (the whole run when it is offline) |
| `resume/` | the training state to resume from (section 6), replaced at each snapshot |
| `checkpoints/<step>/` | `params` (orbax: the trainable parameters in bfloat16) and `meta.json` (step, model, overrides, demos, settings) |
| `summary.json` | final loss, checkpoint list, duration |

`eval` lines (every `eval_interval`) hold, for a **fixed** set of `eval_samples` rows, once on training rows and once on held-out rows when `val_episodes > 0`: the loss, and
the policy's own chunks (`plan`, sampled as at deployment) against the demonstration's: mean absolute error of the 7 joints over the chunk, of the first step, of the
gripper. For pi0-FAST the line also has `decode`: the failures of the FAST decoder, the chunks without an end token and the token-string lengths
(`OpenPIPolicy.decode_stats`); a chunk that could not be decoded is counted and left out of the errors. This is not the success rate, which only the env gives.

## 5b. wandb

The events of `metrics.jsonl` are also logged to Weights & Biases (`src/algos/tracking.py`, settings `wandb_mode` and `wandb_project` in `configs/algo/sft.yaml`; the run is named after its folder),
as expo-ft did (openpi's `loss`, `grad_norm`, `param_norm`, and the `camera_views` of 5 training rows at the start of a fresh run): `train/*` (loss, `grad_norm`, `param_norm`, `lr`, `step_s`, `memory_mib/*`), `eval/<train|val>/*` (loss, own-chunk errors, FAST decoding statistics), `checkpoint/*`, `snapshot/*` (`save_s`, `size_mib`),
`perf/*` (the first step, compilation included), and the parameter counts and final numbers in the run's summary. The config (settings + model YAML) is the run's config in wandb. Tags: the model and `lora`/`full`.

- **Resume = the same wandb run.** The id is written to `<run>/wandb_id.txt` when the training starts; `--resume` calls `wandb.init(id=..., resume="must")`, in the project the run was created in
  (a `wandb_project` given on resume is ignored), so the curves continue in the same run instead of a new one. `resume="must"` is expo-ft's choice: an error rather than a new run if that one does not exist.
  A run that was never logged to wandb (no `wandb_id.txt`) gets a new wandb run on resume, with a warning.
- **x axis: `step`.** Every point carries our `step` and `define_metric("*", step_metric="step")` makes it the axis, because wandb's own step may only increase and a resumed run goes back to its snapshot
  (the points that go back would be dropped). Consequence: steps that were run twice (a crash, then a resume from an older snapshot) are on the curve twice. After a SIGTERM stop the snapshot is
  at the stop step, so there is no overlap.
- **Never stops a run.** `wandb_mode: online` (default) that cannot start (no login, no network on the node) falls back to **offline** with a warning, in the same run id: `wandb sync <run>/wandb/offline-run-*`
  uploads it later. A package that is not installed, or an error while logging, leaves `metrics.jsonl` alone (it stays the reference). `disabled` logs nothing and does not import wandb.
- Not known yet: whether the compute nodes reach wandb's servers (the first job says it: `[wandb] ... falling back to offline` in the log if not). `wandb login` on the login node is needed once.

## 6. Resume

A run can be continued after a time limit, a failure or a preemption: `--resume <run>` (the run's path, e.g. `runs/sft/peg_insert/peg_insert_pi05_lora_scripted`, or a folder name under `output_root`), with the same job script and without `--model`. Nothing else
has to be repeated: the model, its overrides, the demonstrations and the settings come from the run's `config.json`, so only `--set` is accepted next to `--resume`.

What is written, in `<run>/resume/` (the checkpoints cannot do it: they are bfloat16 and hold no optimiser state):

- the trainable parameters **in float32**, the whole **AdamW state** and the step. Size: about 3 times the float32 trainable parameters, i.e. 4.5 GiB measured with LoRA (15 s to write) and 29 GiB measured for pi0-FAST full fine-tuning (77 s to write) and 29 GiB measured for pi0.5 full fine-tuning (160 s to write).
  full fine-tuning. It is written aside and swapped in when complete, so the disk holds twice that for a moment, and a crash while writing leaves the previous snapshot usable;
- when: every `resume_interval` steps (500 in the YAML, 1000 in the full job; 0 = never), at the last step (so that `--set steps=<larger>` can train further), and **when the job gets a
  SIGTERM**.

**The SIGTERM**: Slurm sends it before the time limit (`--signal=B:TERM@300` in the LoRA job, `@600` in the full one, whose snapshot is longer to write); the job scripts forward it to Python,
which finishes its current step, writes a snapshot and ends with the status 0 and the line `stopped at step N ... continue with ...`. A job killed without warning (node failure, `scancel -9`)
resumes from the last periodic snapshot.

**Same run**: the batch of step `s` comes from a permutation seeded by `seed` (the batches already done are skipped, not prepared) and its noise from `fold_in(key(seed), s)`, so a resumed run
trains on what an uninterrupted one would have. The test compares the parameters and the optimiser state after 4 steps done in one go and as 2 + 2: equal (up to 1e-5, the GPU kernels are not
bit-reproducible). The checkpoints of the steps already passed are kept; those the resume repeats are written again.

**What may change on resume**: `steps` (only with a constant learning rate, `lr = lr_end`, since a decay depends on the total), `log_interval`, `eval_interval`, `eval_samples`, `save_interval`,
`resume_interval`, `prefetch`. **What may not**: the demonstrations (and their number), `batch_size`, `micro_batch_size`, the optimiser and its rates, `seed`, the prompt, the model YAML: the
resume is refused with the list of the differences. A run already at its `steps` is refused too (raise `steps`).

Not done: restoring on several devices, and a resume after the demonstrations changed (the number of training rows must be the same).

## 7. Loading a checkpoint

```python
from src.models import load_model
from src.algos.sft import load_checkpoint
policy = load_model("pi05")                       # the SAME model YAML and overrides as the run (the rank is part of the structure; --full too; the `norm_stats` block too:
                                                  # a pi0.5 run of configs/experiment/sft/ needs `model_set` of its experiment file, or the checkpoint's meta.json `model_cfg`)
meta = load_checkpoint(policy, "runs/sft/peg_insert/<run>/checkpoints/1200")
```

`load_checkpoint` puts the trained parameters into the policy (`set_state`: nothing recompiles) and returns `meta.json`. It refuses a checkpoint with a parameter the policy lacks or
of another shape (another rank), naming the first ones. A stopped run is continued with `--resume` (section 6).

## 8. Differences from the SFT of expo-ft

| | expo-ft | here |
|---|---|---|
| training code | openpi's `scripts/train.py` through `run_pipeline.py --stage sft`, with a config added to the fork | our own loop on `OpenPIPolicy` (one code for pi0.5 and pi0-FAST, and for the touch inputs later); openpi's loss, filter and optimiser |
| data | LeRobot dataset converted from the `.npz` | the `.npz` directly (touch included) |
| pi0.5 chunk | 16 steps, offsets (DeltaActions) | the same |
| normalisation statistics | the noisy demonstrations' (`sft_norm_stats_dir` / `sft_norm_stats_asset_id`), for the three demonstration types | the same file, through the `norm_stats` block of the model YAML (section 2b); pi0-FAST: the checkpoint's own |
| pi0-FAST | not run | 10-step chunks, absolute targets (openpi's pipeline for this checkpoint) |
| steps | `num_train_steps: 4001` (4001 iterations) | `steps: 4000` updates |
| evaluation | none during the run | loss and own-chunk error on a fixed set, FAST decoding statistics |
| resume | yes (openpi's) | yes (section 6) |
| wandb | yes, the same run on resume (`wandb_id.txt`) | yes, same logic (section 5b), x axis `step` |
| FSDP | yes (openpi's) | no (one device) |

## 9. Checked and not checked

Checked (`tests/test_sft_demos.py`, `tests/test_sft.py`, CPU): the chunks (last action repeated, never across episodes), the options of the data and of the command line, the refusal of broken
demonstrations, the loop on a tiny pi0.5 (micro-batches, accumulation, only the trainable part moves, the log, the checkpoints), the loading of a checkpoint into a fresh policy and the refusal of one
that does not fit, the resume (2 + 2 steps end in the state of 4 in one go, the refusals, the SIGTERM stop, a crash while swapping snapshots). wandb (`tests/test_tracking.py`: a fake `wandb` module for the id file, the resume, the fallbacks and the events; and a real offline run of 2 + 2 steps: one id, two attempts). The vision tower of that tiny model is frozen (its optimiser state does not fit the test machine).

**Not checked, to be seen in the first runs:**

- the memory, the time per step and the snapshot size of the full runs after the first 30 steps (measured at 30 steps; the snapshots may grow towards 3 times the float32 parameters);
- statistics for pi0-FAST computed on the demonstrations (offsets over 10 steps), and what they do to the FAST token lengths and to the 0.011 rad reconstruction error;
- pi0-FAST training on real FAST tokens: the path through the real tokenizer (the tests use a stub) has only been exercised by `scripts/fast_tokens.py` for the lengths;
- that the 4000 steps and the learning rate give a policy as good as expo-ft's SFT (success rate in the env: `scripts/eval_sft.py`, `docs/eval/sft_eval.md`);
