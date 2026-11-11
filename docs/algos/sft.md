# SFT: supervised fine-tuning on the FORGE demonstrations

`python -m scripts.sft --model <pi05 | pi0_fast>` fine-tunes a VLA on the demonstrations of `docs/demos.md` (the same as expo-ft's). The result is a
checkpoint the RL starts from. Settings: `configs/algo/sft.yaml`. Code: `src/algos/demos.py` (data), `src/algos/sft.py` (loop, checkpoints),
`scripts/sft.py` (command line). **Status: written and tested on a tiny model on the CPU; no run on the cluster yet** (section 8).

## 1. Commands

| What | Command |
|---|---|
| look at the data, load no model (login node, no GPU) | `python -m scripts.sft --model pi0_fast --dry-run` |
| LoRA on an L40S | `sbatch jobs/job_sft_lora.sh pi0_fast` |
| LoRA, other demonstrations | `sbatch jobs/job_sft_lora.sh pi05 --demos demos/isaaclab/ForgePegInsert_Robotiq_noisy --run-name pi05_noisy` |
| full fine-tuning on an A100 80 GB | `sbatch jobs/job_sft_full.sh pi05` |
| another rank | `sbatch jobs/job_sft_lora.sh pi05 --model-set vlm.lora_rank=8 --model-set action_head.lora_rank=16` |
| shorter run, smaller micro-batch | `sbatch jobs/job_sft_lora.sh pi05 --set steps=1500 --set micro_batch_size=8` |
| follow a run | `tail -f runs/sft/<run>/metrics.jsonl` (or `logs/slurm/sft_lora_<jobid>.out`) |

`--set KEY=VALUE` overrides a key of `configs/algo/sft.yaml`; `--model-set KEY=VALUE` a key of the model YAML (`<component>.<key>`, the
probe's `--set`); `--full` is `use_lora: false` in every component the model has. Before the first job, on the login node:
`uv sync --group vla`, `python -m scripts.fetch_models`, then the `--dry-run`. The jobs run offline.

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

The experiments use LoRA, as in expo-ft. Memory is **not measured** for either mode. A float32 model with gradients and AdamW state is about 16 bytes per parameter,
so full fine-tuning needs roughly 54 GB (pi0.5) or 47 GB (pi0-FAST) before the activations, which is why it is on the 80 GB card; LoRA on the L40S is a first guess. The
`first_step` and `train` lines of `metrics.jsonl` carry `memory_mib` (JAX's in-use and peak bytes): that is the measurement, from the first job.

## 5. What a run writes

`runs/sft/<model>_<demos>_<lora|full>_<date>/` (`runs` is a link to the scratch disk; `--run-name` chooses the folder):

| File | Content |
|---|---|
| `config.json` | the algorithm settings and the model YAML as run (overrides included) |
| `metrics.jsonl` | one JSON line per event: `start` (parameter counts, sizes), `first_step` (seconds including the compilation, memory), `train` (every `log_interval`: loss, gradient norm, learning rate, seconds per step, memory), `eval`, `checkpoint`, `done` |
| `checkpoints/<step>/` | `params` (orbax: the trainable parameters in bfloat16) and `meta.json` (step, model, overrides, demos, settings) |
| `summary.json` | final loss, checkpoint list, duration |

`eval` lines (every `eval_interval`) hold, for a **fixed** set of `eval_samples` rows, once on training rows and once on held-out rows when `val_episodes > 0`: the loss, and
the policy's own chunks (`plan`, sampled as at deployment) against the demonstration's: mean absolute error of the 7 joints over the chunk, of the first step, of the
gripper. For pi0-FAST the line also has `decode`: the failures of the FAST decoder, the chunks without an end token and the token-string lengths
(`OpenPIPolicy.decode_stats`); a chunk that could not be decoded is counted and left out of the errors. This is not the success rate, which only the env gives.

## 6. Loading a checkpoint

```python
from src.models import load_model
from src.algos.sft import load_checkpoint
policy = load_model("pi05")                       # the SAME model YAML and overrides as the run (the rank is part of the structure; --full too)
meta = load_checkpoint(policy, "runs/sft/<run>/checkpoints/1200")
```

`load_checkpoint` puts the trained parameters into the policy (`set_state`: nothing recompiles) and returns `meta.json`. It refuses a checkpoint with a parameter the policy lacks or
of another shape (another rank), naming the first ones. There is no resume of a stopped run: it is run again.

## 7. Differences from the SFT of expo-ft

| | expo-ft | here |
|---|---|---|
| training code | openpi's `scripts/train.py` through `run_pipeline.py --stage sft`, with a config added to the fork | our own loop on `OpenPIPolicy` (one code for pi0.5 and pi0-FAST, and for the touch inputs later); openpi's loss, filter and optimiser |
| data | LeRobot dataset converted from the `.npz` | the `.npz` directly (touch included) |
| pi0.5 chunk | 16 steps, offsets (DeltaActions) | the same |
| pi0-FAST | not run | 10-step chunks, absolute targets (openpi's pipeline for this checkpoint) |
| steps | `num_train_steps: 4001` (4001 iterations) | `steps: 4000` updates |
| evaluation | none during the run | loss and own-chunk error on a fixed set, FAST decoding statistics |
| resume, FSDP, wandb | yes (openpi's) | no |

## 8. Checked and not checked

Checked (`tests/test_sft_demos.py`, `tests/test_sft.py`, CPU): the chunks (last action repeated, never across episodes), the options of the data and of the command line, the refusal of broken
demonstrations, the loop on a tiny pi0.5 (micro-batches, accumulation, only the trainable part moves, the log, the checkpoints), the loading of a checkpoint into a fresh policy and the refusal of one
that does not fit. The vision tower of that tiny model is frozen (its optimiser state does not fit the test machine).

**Not checked, to be seen in the first runs:**

- the whole thing on the real models and the GPUs: memory (LoRA on the L40S, full on the A100), seconds per step, the best micro-batch;
- pi0-FAST training on real FAST tokens: the path through the real tokenizer (the tests use a stub) has only been exercised by `scripts/fast_tokens.py` for the lengths;
- that the 4000 steps and the learning rate give a policy as good as expo-ft's SFT (success rate in the env: this is the RL's evaluation, not this file's);
- the bfloat16 checkpoint of a full fine-tuning (about 6.7 GB for pi0.5) against the disk: 20 of them for `save_interval: 200`.
