# pi0-FAST (autoregressive over FAST action tokens), DROID joint-position weights (PolaRiS)

Reference sheet for the `pi0_fast` model of `tactile-vla-rl`: what goes in, what comes out, what the network is made of, how it is
fine-tuned (full or LoRA), and what is parameterised in `configs/model/pi0_fast.yaml`. It is the counterpart of `docs/models/pi05.md`;
the two models differ in how they produce actions, not in the vision-language backbone.

Everything below was read in the pinned openpi fork (`third_party/openpi`, commit `f7cfe2d2`, the same commit as pi0.5: pi0-FAST is part of
openpi, there is no second repository) or measured on it. Parameter counts come from `jax.eval_shape` on the real openpi configs (no
weights loaded, no GPU needed). What is a consequence drawn from those facts rather than a fact is said so. What was not checked is in the last section.

## 1. At a glance

| | |
|---|---|
| Family | pi0-FAST (`Pi0FASTConfig`), **autoregressive**: the action chunk is a sequence of discrete tokens produced one at a time by the language model. No diffusion, no flow matching, **no action expert** |
| Backbone | PaliGemma: **SigLIP So400m/14** vision tower + **Gemma 2B** language model. This one network also writes the actions |
| Action tokenizer | **FAST** (`physical-intelligence/fast`, Hugging Face Hub): DCT of the normalised chunk, then byte-pair encoding. Not part of the checkpoint |
| Weights used | `gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris` (`params/` and `assets/`): DROID (90 %) co-trained with the PolaRiS simulation data (10 %), joint **position** actions |
| Parameters | 2.92 G without LoRA, 2.95 G with LoRA (section 6) |
| Input | 2 camera images (224×224) + a black third image, joint state (7) + gripper (1) written as text tokens, a language prompt |
| Output | a chunk of **10** steps × 8 values (7 joint targets + gripper), **absolute** targets, decoded from tokens |
| Decoding | token by token, greedy by default (temperature 0), at most 256 steps, stops at the end-of-sequence token |
| Measured on our demonstrations | at most **62** tokens per chunk (limit 180) and at most **16** generated tokens (limit 256): the limits are not a constraint. FAST is lossy: **0.011 rad** mean error on the joints, and **3 chunks out of 8 913** fail to decode (section 5) |
| Training loss | next-token cross entropy, on the action tokens only |
| Default LoRA | rank **16** on the VLM (`gemma_2b_lora`); there is no second component |
| openpi config used for data and norm stats | `pi0_fast_droid_jointpos_polaris` |
| Our files | `configs/model/pi0_fast.yaml`, `src/models/` (same wrapper as pi0.5) |

Why this checkpoint: the only joint-**position** pi0-FAST that openpi defines is the PolaRiS one. `pi0_fast_droid` is joint **velocity** and does
not fit the absolute joint targets of our environment.

## 2. Architecture

| Component | Architecture | Parameters (measured) |
|---|---|---:|
| Vision tower | SigLIP So400m/14, 224×224 input → 16×16 = **256 tokens per image**, projected to the Gemma width (2048) | 414.8 M |
| Language model | Gemma 2B: 18 layers, width 2048, MLP 16 384, 8 heads, 1 KV head, head dim 256, vocabulary 257 152 (`gemma_fast.py`) | 2508.5 M (of which the embedding table, also used to write the output logits, is 526.6 M) |
| **Total, no LoRA** | | **2923.3 M** |

- It is the **same VLM as pi0.5** (same SigLIP, same Gemma 2B), without the action expert and without the projections or the time MLP: the
  action path is the language model itself, which reads and writes tokens.
- **Sequence and attention** (`pi0_fast.py`, `make_attn_mask`): `[image tokens][prefix text][action text]`. The images and the prefix are
  **bidirectional** (autoregressive mask 0); the part that starts at `Action:` is **causal** (mask 1): each token sees the whole prefix and the
  previous action tokens. Inference fills a key/value cache with the prefix, then decodes one token per step.
- Weights are `bfloat16` (2.9 G parameters ≈ 5.8 GB).

## 3. Observation space

### 3.1 What the environment sends and what the model reads

`src/models/interface.py` (`to_openpi_obs`) takes the same five things as for pi0.5; only the image slots differ (`DroidInputs`, branch
`PI0_FAST`):

| Environment key | Shape (one environment) | Model input |
|---|---|---|
| `observation/exterior_image_1_left` | (224, 224, 3) uint8 | image `base_0_rgb` |
| (none) | zeros (black) | image `base_1_rgb`, **not masked out** (`image_mask = True`: "we don't mask out padding images for FAST models") |
| `observation/wrist_image_left` | (224, 224, 3) uint8 | image `wrist_0_rgb` |
| `observation/joint_position` | (7,) float32, rad | state, dims 0 to 6 |
| `observation/gripper_position` | (1,) float32, 0 = open, 1 = closed | state, dim 7 |
| the `prompt` (`EnvSpec.prompt`, "Insert the peg into the socket.") | string | text |

Not read by the baseline: `extra/tactile` (2, 7, 4), `extra/ft_force`, `extra/gripper_effort`, `extra/force_threshold`,
`extra/peg_socket_offset`. They reach the model only when `vlm.use_tactile` / `vlm.use_force` is switched on (section 8).

### 3.2 Processing

1. **Images**: resized with padding to 224×224 (a no-op here), scaled to [-1, 1]; same training augmentation as pi0.5 when `train=True` (random
   crop 95 % and rotation ±5° on the non-wrist images, colour jitter on all).
2. **State**: 8 values, **normalised with quantiles** (DROID q01/q99, `use_quantile_norm` is on for every model but pi0), **discretised into
   256 bins over [-1, 1]** and written as integers in the text. Same convention as pi0.5.
3. **Text** (`FASTTokenizer.tokenize`):

        prefix : "Task: <prompt>, State: <8 integers>;\n"      tokens with BOS, mask 0 (bidirectional), no loss
        postfix: "Action: " <FAST tokens> "|"  + EOS             mask 1 (causal), the loss is computed here

   The prompt is **lower-cased**, stripped, `_` replaced by a space.
4. **Token budget**: `max_token_len = 180` for the prefix text **plus** the action tokens (the config of the checkpoint sets it; the openpi
   default is 250). A longer sequence is **truncated** with a warning: a chunk that needs more tokens is cut and cannot be decoded
   correctly. Image tokens (3 × 256 = 768) come on top. **Measured on our demonstrations: at most 62 text tokens** (section 5), so 180 leaves a
   factor of about 3.

## 4. Action space

| | |
|---|---|
| Native model output | a token sequence; decoded to `(action_horizon = 10, action_dim = 8)` (no padding: the model is built with `action_dim = 8`) |
| Columns | 0 to 6 = joints, 7 = gripper |
| Joints | **absolute joint-position targets**, in rad. Inside openpi the model works on **offsets from the joint state** (the RLDS DROID data pipeline for joint-position actions applies `DeltaActions` on the way in and `AbsoluteActions` on the way out), and the output transform **adds the state back**: what `plan` returns is already absolute |
| Gripper | absolute command, 0 = open, 1 = closed |
| Normalisation | quantile-normalised (DROID statistics of the checkpoint's `assets/`) before tokenisation, un-normalised after decoding |
| Chunk length in time | 10 steps; at the environment's 15 Hz control rate that is about 0.67 s. How many steps are executed before planning again is a choice of the algorithm |

In the YAML: `output_semantics: absolute_targets`. This is the **opposite** of pi0.5 (`offsets_from_chunk_start_state`): adding the chunk-start
state again here would add it twice. It was checked on 2026-11-10 with `scripts/probe_vla` (5 chunks of a FORGE rollout all look absolute).
`chunk_to_targets` passes the 8 columns through; `targets_to_chunk` (training) is the identity.

## 5. Generation: autoregressive decoding of FAST tokens

**FAST tokenisation of a chunk** (`FASTTokenizer`): the normalised chunk (10 × 8) is compressed by a discrete cosine transform along time,
quantised, and byte-pair encoded into a short sequence of integer tokens (the method of the FAST paper; the tokenizer is loaded from the Hub with
`trust_remote_code=True`, so its code is not in the openpi fork). Each FAST token id `k` is mapped to the PaliGemma token `vocab_size − 1 − 128 − k`, i.e. to the **last tokens of the
vocabulary** (the last 128 special tokens are skipped), so no new embedding is added. The number of tokens per chunk varies with the chunk's
content. Recorded revision of the tokenizer used by our experiments: `ec4d7aa71691cac0b8bed6942be45684db2110f4` (a record, not enforced).

**Training** (`Pi0FAST.compute_loss`): the whole sequence is fed with a one-token shift; the loss is the **cross entropy of the next token**,
averaged over the **postfix tokens only** (`token_loss_mask`: the prompt and the state are not trained on). It is teacher forcing: no
sampling during training.

**Decoding** (`Pi0FAST.sample_actions`): prefill, then at each step the token is the **argmax** of the last logits (temperature 0, the default)
or a sample of `softmax(logits / temperature)`; the loop stops when every row has produced the end token or after
`max_decoding_steps = 256`. The text is then decoded: what follows `Action: ` up to `|` goes back through the FAST tokenizer to a (10, 8) chunk.
If `Action: ` is not found in the output, **the chunk is all zeros** (openpi's `extract_actions`).

- The temperature is **not exposed in our YAML**: with the default 0 the S samples of `plan(..., num_samples=S)` are identical. The wrapper
  repeats the row S times because this model has no `num_samples` argument.
- Cost: one prefill over 768 image tokens + the prefix, then one Gemma 2B forward per generated token, so the cost grows with the number
  of action tokens (at most 256 by the limit, at most 16 on our demonstrations), unlike pi0.5 where it is a fixed 10 expert passes.

### Measured on the FORGE demonstrations

`python -m scripts.fast_tokens` (2026-11-11, `HF_HUB_OFFLINE=1`, CPU, about a minute): every step of every demonstration gives a chunk of the next 10
actions (the last steps repeat the final action, as the LeRobot loader does). The chunk goes through the transforms of the openpi config
(`DroidInputs`, `DeltaActions`, quantile normalisation with the DROID statistics of the checkpoint) and the FAST tokenizer; the tokens are decoded back
after the truncation at `max_token_len`, and the result goes through `AbsoluteActions`. Report: `reports/fast_tokens.json`.

| Folder | Demos | Chunks | Tokens per chunk, mean / p99 / max | Longest prefix | Longest postfix | Above 180 | Joint error (rad), mean / p99 / max | Gripper error, max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ForgePegInsert_Robotiq_scripted` | 50 | 2422 | 56.4 / 60 / 62 | 46 | 16 | 0 | 0.0112 / 0.0211 / 0.0299 | 0.0058 |
| `ForgePegInsert_Robotiq_noisy` | 50 | 2485 | 56.4 / 60 / 62 | 46 | 16 | 0 | 0.0113 / 0.0216 / 0.0463 | 0.5001 (decode failures, below) |
| `ForgePegInsert_Robotiq_teleop` | 50 | 4006 | 56.2 / 59 / 60 | 46 | 14 | 0 | 0.0112 / 0.0163 / 0.0204 | 0.0058 |

- **The limits are not a constraint.** The longest sequence is 62 tokens (limit `max_token_len = 180`) and the longest postfix 16 (limit
  `max_decoding_steps = 256`): no chunk of the 8 913 is truncated. The smallest `max_token_len` that covers all of them is 62, 80 with a 20 % margin.
  `configs/model/pi0_fast.yaml` keeps 180, the value of the checkpoint's config: a smaller value would only shorten the padding, and it was not tested.
  `action_horizon = 10` is kept.
- **FAST is lossy.** Decoding the tokens of a demonstration gives back the chunk with a **mean error of 0.011 rad** on the worst joint of each chunk
  (p99 0.016 to 0.022, max 0.020 to 0.046), the same for the three folders. The gripper is reproduced to 0.006. Whether this matters for the insertion
  is **not measured** (it is an error on the targets of a whole chunk, followed by the next replanning).
- **A few chunks do not decode.** The FAST decoder printed `cannot reshape array of size 76 into shape (8)` three times: 3 chunks out of 8 913 (0.03 %),
  all in the noisy folder, each with only 5 to 7 FAST tokens (the shortest sequences). The decoder then returns an all-zero chunk, and zeros in the normalised
  space are not a zero action: they un-normalise to the middle of the DROID statistics, consistent with the gripper error of 0.5 in that folder. A policy under
  RL can write such token strings too, so `OpenPIPolicy` counts them (below): the algorithm must not execute a chunk that failed to decode.
- Not covered by these numbers: the chunks a **policy** writes (they may be longer than the demonstrations' and may fail to decode more often), and
  the tokens of other horizons (`--horizon`, repeatable). That rate depends on the weights, so it cannot be settled before training: it is **measured by
  the policy itself** at every call (next paragraph).

### Counting the failures of a policy

For a pi0-FAST `OpenPIPolicy` (`src/models/openpi_policy.py`), every `plan` decodes the generated token strings once more to check them and records:

| Attribute | Content |
|---|---|
| `last_valid` | `(N, S)` bool of the last `plan`: `False` = the chunk could not be decoded (the all-zero fallback). The algorithm must not execute it |
| `decode_stats.summary()` | cumulative since `decode_stats.reset()`: `chunks`, `failed` and `failed_rate`, `no_eos` (no end token within 256 steps), `tokens_mean`, `tokens_max` |

`python -m scripts.probe_vla --model pi0_fast` resets the counters before its FORGE rollout and prints them at the end (one episode is about 15 chunks: it
shows the order of magnitude of the base model, not a rate). The SFT and the RL log the same numbers with every evaluation and every update.

## 6. Where the LoRA is, and what is trained

### 6.1 The LoRA of openpi (not modified by us)

Same implementation as for pi0.5 (`lora.py`), one component only: the **Gemma 2B**, in every one of the 18 layers, on the attention projections and on the
feed-forward matrices (`gemma_fast.py`: `gemma_2b_lora`).

- `W + (alpha / rank) · A·B` with **alpha = rank** (scaling 1.0), no rank-stabilised variant.
- **Initialisation: normal(0, 0.01) for both `lora_a` and `lora_b`** (the LoRA paper starts from B = 0): a fresh LoRA model is the base model plus a small random
  perturbation, not exactly the base function. `vlm.use_lora: false` is the exact base function.
- **Default rank: 16**, 27.9 M parameters. The parameters are linear in the rank: **1.74 M per unit** (4 → 7.0 M, 8 → 14.0 M, 16 → 27.9 M,
  32 → 55.8 M, 64 → 111.6 M; derived by linearity from the measured rank-16 point). `vlm.lora_rank` changes the structure of the model: a
  checkpoint trained with a rank must be loaded with the same one (`gemma_2b_lora_r<rank>` in `src/models/openpi_config.py`, same LoRA with
  `alpha = rank`; the fork is not modified).
- An `action_head` block in the YAML is **refused**: there is no action expert.

### 6.2 What is trainable (`Pi0FASTConfig.get_freeze_filter`)

With LoRA, the filter freezes everything under the language model (`.*llm.*`) except the LoRA matrices; the vision tower is **not** frozen.
Without LoRA, nothing is frozen.

| `vlm.use_lora` | Trainable | Parameters trained | Total parameters |
|:--:|---|---:|---:|
| true (default) | LoRA (27.9 M) + SigLIP (414.8 M) | **442.7 M** (15.0 %) | 2951.2 M |
| false | everything (full fine-tuning) | **2923.3 M** | 2923.3 M |

With LoRA the embedding table (526.6 M) is frozen along with the rest of Gemma. Full fine-tuning needs its own job on an A100 80 GB, LoRA runs on an
L40S; memory was not measured here.

## 7. Loading the weights

The checkpoint is a **full fine-tune of the VLM**, with no LoRA tensors. As for pi0.5, `load: weight_loader` keeps the checkpoint weights and takes
the LoRA parameters from the model's own initialisation; this is also how openpi's own LoRA recipe for pi0-FAST
(`pi0_fast_libero_low_mem_finetune`) loads its checkpoint. Without LoRA nothing is missing and the result is the checkpoint itself;
`load: strict` is possible only in that case. The FAST tokenizer and the PaliGemma tokenizer (`gs://big_vision/paligemma_tokenizer.model`)
are **not in the checkpoint**: they are fetched at load time (`python -m scripts.fetch_models` caches them). The norm statistics come from the
checkpoint's `assets/` (asset id `droid`).

## 8. Where touch can enter (nothing is implemented yet)

`configs/model/pi0_fast.yaml` has a single component block, `vlm` (with `use_tactile`, `use_force`, `force_mode`, all off). A component that
asks for touch stops with an explicit error until an injection exists (`src/touch/injection`). As facts about the model: everything goes through the
language model's input, and the state is already **text**, so a touch vector can be given either as tokens in the prefix (discretised like the joints)
or as continuous embeddings added to the sequence by a new trainable projection. There is no separate action head to feed.

## 9. Fine-tuning defaults of the openpi config we start from

`pi0_fast_droid_jointpos_polaris`: `Pi0FASTConfig(action_dim=8, action_horizon=10, max_token_len=180)`, learning rate 5e-5 (1 000 warm-up steps, cosine
schedule whose end value equals its peak), batch 128, 1 000 steps, RLDS data (DROID 90 % + PolaRiS 10 %). These are the values written in the
fork for this checkpoint's config, not a recipe for our demonstrations (LeRobot format); they are not necessarily what was used to
train the released weights. openpi's own LoRA recipe for this family is `pi0_fast_libero_low_mem_finetune`.

## 10. What it means for reinforcement learning (consequences, not openpi facts)

- The likelihood of a chunk is **exact**: the product of the probabilities of its tokens, so policy-gradient style updates and importance ratios are
  directly available, which is not the case for the flow-matching pi0.5.
- A chunk is a **discrete** object: the log-probability sums over a variable number of tokens (at most 16 on our demonstrations, 256 by the limit), so a
  long chunk weighs more than a short one unless the algorithm normalises it.
- Decoding at temperature 0 is deterministic: for exploration the algorithm must sample (temperature > 0), which our YAML does not expose yet.
- A chunk can fail to decode (3 of 8 913 demonstration chunks did; a policy can produce more) and the fallback is an all-zero chunk in normalised space,
  not a hold: count these failures and do not execute them. Truncation at `max_token_len` did not occur on the demonstrations.

## 11. Not checked

- The tokens and the decoding failures of chunks written by a **fine-tuned** policy: they are counted by `decode_stats` during the SFT and the RL, not known in advance.
- Whether the FAST reconstruction error (0.011 rad on average) limits the success rate of the insertion.
- Memory use and step time, LoRA versus full fine-tuning, on our GPUs.
- The training recipe and the data of the PolaRiS checkpoint (released by Physical Intelligence, not in this repo).
- The effect of the rank on the success rate (no experiment yet).
