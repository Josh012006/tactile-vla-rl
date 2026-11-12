# SFT: supervised fine-tuning on the FORGE demonstrations

`python -m scripts.sft --model <pi05 | pi0_fast>` fine-tunes a VLA on the demonstrations of `docs/demos.md` (the same as expo-ft's). The result is a
checkpoint the RL starts from. Settings: `configs/algo/sft.yaml`. Code: `src/algos/demos.py` (data), `src/algos/sft.py` (loop, checkpoints), `src/algos/tracking.py` (wandb),
`scripts/sft.py` (command line). **Status: tested on a tiny model on the CPU; probed 30 steps on the cluster (LoRA pi0.5 complete on both cards; full and pi0-FAST ran out of memory, corrected, to be re-probed: section 4)** (section 9).

## 1. Commands

| What | Command |
|---|---|
| look at the data, load no model (login node, no GPU) | `python -m scripts.sft --model pi0_fast --dry-run` |
| LoRA on an L40S | `sbatch jobs/job_sft_lora.sh pi0_fast` |
| LoRA, other demonstrations | `sbatch jobs/job_sft_lora.sh pi05 --demos demos/isaaclab/ForgePegInsert_Robotiq_noisy --run-name pi05_noisy` |
| full fine-tuning on an A100 80 GB | `sbatch jobs/job_sft_full.sh pi05` |
| another rank | `sbatch jobs/job_sft_lora.sh pi05 --model-set vlm.lora_rank=8 --model-set action_head.lora_rank=16` |
| shorter run, smaller micro-batch | `sbatch jobs/job_sft_lora.sh pi05 --set steps=1500 --set micro_batch_size=8` |
| continue a run that stopped | `sbatch jobs/job_sft_lora.sh pi05 --resume <run name>` (or `job_sft_full.sh`): see section 6 |
| follow a run | wandb (project `tactile-vla-rl`, section 5b), or `tail -f runs/sft/<run>/metrics.jsonl` (or `logs/slurm/sft_lora_<jobid>.out`) |
| no wandb, or wandb offline | `--set wandb_mode=disabled`, `--set wandb_mode=offline` (then `wandb sync runs/sft/<run>/wandb/offline-run-*`) |

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

Normalisation statistics computed on the demonstrations (`scripts/demos/compute_demo_norm_stats.py`) are **not** used by this SFT: it keeps the checkpoint's, as the
policy does at inference and as the token lengths of `docs/models/pi0_fast.md` were measured. (expo-ft had a variant with demo statistics; it would have to be
added to the policy's transforms, not only to the SFT.)

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
- **The three failures** (the `error` line of `metrics.jsonl` gives them) are the same cause: **XLA's allocator stops at 75 % of the card** (`XLA_PYTHON_CLIENT_MEM_FRACTION`, 0.75 by default even with
  `PREALLOCATE=false`), and the jobs did not raise it. The two full runs failed with 57.3 GiB (pi0.5) and 59.1 GiB (pi0-FAST) in use, at the first step, which is 75 % of 80 GB once the next buffer is added.
  The pi0-FAST LoRA run failed in its **evaluation** (allocation of 7.8 GB = the float32 logits of 16 rows x 474 tokens x 257,152 words) with 24 GiB peak: with the cap it also lacked room.
- **Corrections.** (1) The jobs set `XLA_PYTHON_CLIENT_MEM_FRACTION` (0.9 LoRA, 0.95 full). (2) The evaluation loss is computed 4 rows at a time (`EVAL_MICRO_BATCH`): 2 GB of logits. (3) The accumulation donates its
  buffers (two float32 copies of the gradients, not three). (4) The policy's own bfloat16 copy of the trainable weights is freed once the float32 copy exists (6.2 GiB for pi0.5 full). (5) `memory_mib` now has the
  allocator's `limit`. Full fine-tuning of pi0.5 then needs about 38 GiB (state) + 2 x 12.5 GiB (gradients) + a few GiB of activations, **about 67 GiB**, against 0.95 x 80 GB = 76 GiB; pi0-FAST about 60 GiB.
  **Not measured since**: the full runs have not run with these corrections. A smaller `micro_batch_size` would not have helped: it only changes the activations.
  If pi0.5 still does not fit: keep the first AdamW moment in bfloat16 (6 GiB less; a change of the optimiser, so a decision).

## 5. What a run writes

`runs/sft/<model>_<demos>_<lora|full>_<date>/` (`runs` is a link to the scratch disk; `--run-name` chooses the folder):

| File | Content |
|---|---|
| `config.json` | the algorithm settings and the model YAML as run (overrides included); read back by `--resume` |
| `metrics.jsonl` | one JSON line per event: `start` (parameter counts, sizes), `error` (the exception and the memory, when the run fails), `first_step` (seconds including the compilation, memory), `train` (every `log_interval`: loss, gradient norm, learning rate, seconds per step, memory), `eval` (with `eval_s`), `checkpoint` (with `save_s` and `size_mib`), `resume_snapshot` (same), `resume`, `done` or `stopped`; every line has `elapsed_s` |
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
- **Never stops a run.** `wandb_mode: online` (default) that cannot start (no login, no network on the node) falls back to **offline** with a warning, in the same run id: `wandb sync runs/sft/<run>/wandb/offline-run-*`
  uploads it later. A package that is not installed, or an error while logging, leaves `metrics.jsonl` alone (it stays the reference). `disabled` logs nothing and does not import wandb.
- Not known yet: whether the compute nodes reach wandb's servers (the first job says it: `[wandb] ... falling back to offline` in the log if not). `wandb login` on the login node is needed once.

## 6. Resume

A run can be continued after a time limit, a failure or a preemption: `--resume <run name>` (a folder name under `runs/sft`, or a path), with the same job script. Nothing else
has to be repeated: the model, its overrides, the demonstrations and the settings come from the run's `config.json`, so only `--set` is accepted next to `--resume`.

What is written, in `<run>/resume/` (the checkpoints cannot do it: they are bfloat16 and hold no optimiser state):

- the trainable parameters **in float32**, the whole **AdamW state** and the step. Size: about 3 times the float32 trainable parameters, i.e. 4.5 GiB measured with LoRA (15 s to write) and 35 to 40 GB estimated for
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
policy = load_model("pi05")                       # the SAME model YAML and overrides as the run (the rank is part of the structure; --full too)
meta = load_checkpoint(policy, "runs/sft/<run>/checkpoints/1200")
```

`load_checkpoint` puts the trained parameters into the policy (`set_state`: nothing recompiles) and returns `meta.json`. It refuses a checkpoint with a parameter the policy lacks or
of another shape (another rank), naming the first ones. A stopped run is continued with `--resume` (section 6).

## 8. Differences from the SFT of expo-ft

| | expo-ft | here |
|---|---|---|
| training code | openpi's `scripts/train.py` through `run_pipeline.py --stage sft`, with a config added to the fork | our own loop on `OpenPIPolicy` (one code for pi0.5 and pi0-FAST, and for the touch inputs later); openpi's loss, filter and optimiser |
| data | LeRobot dataset converted from the `.npz` | the `.npz` directly (touch included) |
| pi0.5 chunk | 16 steps, offsets (DeltaActions) | the same |
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

- wandb ONLINE from a compute node (the tests are offline or fake);
- full fine-tuning on the A100 after the memory corrections (section 4): memory and seconds per step, and its resume snapshot (35 to 40 GB estimated, 15 s for 4.5 GiB with LoRA: minutes, to be seen against the 10-minute warning); the evaluation of pi0-FAST, which ran out of memory in the probe;
- pi0-FAST training on real FAST tokens: the path through the real tokenizer (the tests use a stub) has only been exercised by `scripts/fast_tokens.py` for the lengths;
- that the 4000 steps and the learning rate give a policy as good as expo-ft's SFT (success rate in the env: this is the RL's evaluation, not this file's);
- the time to write a resume snapshot and its size on the real models (`resume_snapshot` lines: `save_s`, `size_mib`), above all the 35 to 40 GB of the full fine-tuning against the 10 minutes of the SIGTERM warning;
- the bfloat16 checkpoint of a full fine-tuning (about 6.7 GB for pi0.5) against the disk: 20 of them for `save_interval: 200`.
