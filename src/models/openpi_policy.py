"""pi0.5 / pi0 / pi0-FAST through openpi, behind src/models/interface.py. Needs the `vla` dependency group (jax, openpi).

    policy = OpenPIPolicy.load(cfg)             # or src.models.load_model("pi05"); loading itself: src/models/loading.py
    plan = policy.plan(obs, prompt, num_samples=8)

What this adds to the `Policy` of openpi, which is built for ONE observation at a time and freezes the weights it was created with:

  * batched sampling over N environments (and S samples each). The per-row transforms of openpi (normalisation, resize, tokens) are
    applied row by row, then the rows are stacked. pi0.5 / pi0 draw the S samples inside `sample_actions` (`num_samples`: the prefix
    is computed once, the noise is independent); pi0-FAST has no such argument, its rows are repeated. The batch is padded to the next
    power of two (padded_batch_size) so that a varying number of environments does not recompile the model at every step.
  * the weights are an ARGUMENT of the jitted sampler, not frozen into it: `set_state(new_state)` after a training step is enough,
    nothing is recompiled. (openpi's `Policy` builds its sampler with `nnx_utils.module_jit`, which freezes the module's state when the
    policy is created: after a training step it would keep sampling from the old weights.)
  * the training hooks an algorithm needs: `training_batch` (env observations + env-space target chunks -> what `compute_loss`
    takes, through the SAME input transforms as inference, the offsets of `output_semantics` included), `loss`, `trainable_filter`.

The random stream is the policy's own (`seed`): the same seed and the same inputs give the same chunks. openpi's Policy starts from
jax.random.key(0), and so does this class by default. An evaluation does not use that stream: `plan(..., row_seeds=...)` draws the flow noise of every row from
that row's own seed (a function of the seed alone, whatever the other rows or the batch size), and `pad_to` fixes the number of rows the model is called with.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import time
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from src.envs import protocol as P
from src.models.interface import Candidates, ModelInfo, VLAPolicy, chunk_to_targets, num_envs_of, padded_batch_size, rows_of, targets_to_chunk
from src.models.settings import ModelSettings, parse_model_settings
from src.touch.injection import build_injection

FAMILIES_WITH_NATIVE_SAMPLES = ("pi05", "pi0")        # Pi0.sample_actions(num_samples=...) of the pinned fork; Pi0FAST has no such argument
FAST_EOS_TOKEN = 1                                    # PaliGemma's end-of-sequence id, what Pi0FAST.sample_actions stops on


@dataclasses.dataclass
class DecodeStats:
    """pi0-FAST only: what the generated token strings were like, summed over the calls of `plan` since the last `reset()`.
    A policy writes its own token strings (unlike the demonstrations measured by scripts/fast_tokens.py): one that cannot be decoded
    gives openpi's fallback, an all-zero chunk in normalised space, which is NOT a hold and must not be executed (`OpenPIPolicy.last_valid`)."""
    chunks: int = 0
    failed: int = 0           # the FAST decoder returned an all-zero chunk: the tokens could not be decoded
    no_eos: int = 0           # no end token within the decoding limit (256): the string was cut
    tokens_sum: int = 0       # generated tokens, the end token included (the decoding limit when there is none)
    tokens_max: int = 0

    def add(self, tokens: int, no_eos: bool, failed: bool) -> None:
        self.chunks += 1
        self.failed += int(failed)
        self.no_eos += int(no_eos)
        self.tokens_sum += tokens
        self.tokens_max = max(self.tokens_max, tokens)

    def reset(self) -> None:
        self.chunks = self.failed = self.no_eos = self.tokens_sum = self.tokens_max = 0

    def summary(self) -> dict:
        n = max(self.chunks, 1)
        return {"chunks": self.chunks, "failed": self.failed, "failed_rate": self.failed / n, "no_eos": self.no_eos,
                "no_eos_rate": self.no_eos / n, "tokens_mean": self.tokens_sum / n, "tokens_max": self.tokens_max}


class _QuietFastDecoder(logging.Filter):
    """openpi's FAST tokenizer logs two warnings through the ROOT logger every time a token string cannot be decoded (`Error decoding tokens: ...` and `Tokens: [...]`), and a chunk is decoded
    twice here (by openpi's output transform and by `_record_decoding`): hundreds of lines per evaluation. They are dropped; the failures are COUNTED instead (`decode_stats`, `last_valid`, and the
    `decode` of every episode record)."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (message.startswith("Error decoding tokens") or message.startswith("Tokens: ["))


def silence_fast_decoder_warnings() -> None:
    root = logging.getLogger()
    if not any(isinstance(f, _QuietFastDecoder) for f in root.filters):
        root.addFilter(_QuietFastDecoder())


def fast_extractor(policy):
    """openpi's `ExtractFASTActions` in the policy's output transforms (it holds the FAST tokenizer), or None when there is none."""
    for transform in getattr(getattr(policy, "_output_transform", None), "transforms", ()):
        if hasattr(transform, "tokenizer") and hasattr(transform.tokenizer, "extract_actions"):
            return transform
    return None


class OpenPIPolicy(VLAPolicy):
    def __init__(self, policy, train_cfg, cfg: dict, seed: int = 0):
        """`policy`: an openpi `Policy` (its transforms, norm stats and model are used); `train_cfg`: the openpi TrainConfig it was built
from (Gemma variants and freeze filter as set by the YAML); `cfg`: the model YAML (its touch switches are read here)."""
        import flax.nnx as nnx
        import jax

        self.policy, self.train_cfg, self.cfg = policy, train_cfg, cfg
        self.info = ModelInfo.from_cfg(cfg)
        self.settings: ModelSettings = parse_model_settings(cfg)
        self.touch = self.settings.touch_specs                                       # rows_of adds the vectors of the enabled ones
        # a component that asks for touch needs an injection (src/touch/injection): none is written yet, so this raises rather than ignore it
        self.injections = {name: build_injection(name, spec) for name, spec in self.touch.items()}
        self._native_samples = self.info.family in FAMILIES_WITH_NATIVE_SAMPLES
        self.max_rows: Optional[int] = None                  # most rows of ONE model call: a larger batch is split in equal calls (see `raw_chunks`); None = never split
        self.decode_stats = DecodeStats()                    # pi0-FAST: what the generated tokens were like (cumulative, reset() to restart)
        self.last_valid: Optional[np.ndarray] = None         # pi0-FAST: (N, S) bool of the last `plan`: False = the chunk could not be decoded
        self.last_sample = None                              # (model outputs, state rows) of the last `raw_chunks`, before the output transforms: what `sample_candidates` reads
        self._fast = None if self._native_samples else fast_extractor(policy)
        if self._fast is not None:
            silence_fast_decoder_warnings()
        self._graphdef, self._state = nnx.split(policy._model)                      # noqa: SLF001
        graphdef, kwargs, native = self._graphdef, dict(policy._sample_kwargs), self._native_samples   # noqa: SLF001

        def sample(state, rng, observation, num_samples, noise=None):
            model = nnx.merge(graphdef, state)
            if not native:
                return model.sample_actions(rng, observation, **kwargs)
            return model.sample_actions(rng, observation, num_samples=num_samples, **({} if noise is None else {"noise": noise}), **kwargs)

        def row_noise(seeds, shape):
            """(B, *shape) standard normals, row i from the key made of seeds[i] alone."""
            return jax.vmap(lambda s: jax.random.normal(jax.random.key(s), shape))(seeds)

        self._sample = jax.jit(sample, static_argnames=("num_samples",))
        self._row_noise = jax.jit(row_noise, static_argnames=("shape",))
        self._noise_shape = (int(policy._model.action_horizon), int(policy._model.action_dim))          # noqa: SLF001
        self.seed(seed)

    @classmethod
    def load(cls, cfg: dict, seed: int = 0, *, for_training: bool = False) -> "OpenPIPolicy":
        """`seed` also draws the LoRA initialisation, as openpi's train.py does for `TrainConfig.seed`. `for_training`: see `loading.build_params_like_expo_ft`."""
        from src.models.loading import load_policy
        policy, train_cfg = load_policy(cfg, seed, for_training=for_training)
        return cls(policy, train_cfg, cfg, seed)

    def fork(self, seed: int = 0) -> "OpenPIPolicy":
        """A second policy over the same model, for another thread (the evaluation of several checkpoints at once, src/eval/sweep.py). It SHARES what is heavy and stateless: the
        openpi `Policy` (transforms, norm stats), the jitted sampler (one compilation, and so one arithmetic, for all the forks) and, until `set_state` replaces them, the weights
        in device memory (`load_checkpoint` on a fork replaces the trainable parameters only: the frozen ones stay shared, so N forks hold one base model). It does NOT share what
        a call mutates: its own random stream (`seed`), its own weights reference (`set_state`) and its own `decode_stats` / `last_valid`."""
        other = copy.copy(self)
        other.decode_stats = DecodeStats()
        other.last_valid = None
        other.last_sample = None
        other.seed(seed)
        return other

    # -- randomness, weights ----------------------------------------------------------------------------------------------
    def seed(self, seed: int) -> None:
        import jax
        self._key = jax.random.key(seed)

    @property
    def graphdef(self):
        return self._graphdef

    @property
    def state(self):
        """The current weights (an nnx.State), the ones `plan` samples with."""
        return self._state

    def set_state(self, state) -> None:
        """Sample from these weights from now on (same structure and shapes as `state`: no recompilation)."""
        import jax
        if jax.tree.structure(state) != jax.tree.structure(self._state):
            raise ValueError("set_state: the state does not have the structure of the model's")
        self._state = state

    @property
    def trainable_filter(self):
        """nnx filter of the parameters a training step updates: openpi's freeze filter of the config (the `use_lora` / `lora_rank` of the YAML's vlm and action_head blocks)."""
        import flax.nnx as nnx
        return nnx.All(nnx.Param, nnx.Not(self.train_cfg.freeze_filter))

    # -- inference ---------------------------------------------------------------------------------------------------------
    def _stack(self, items: List[dict]):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        batch = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *items)
        return batch, _model.Observation.from_dict(copy.copy(batch))

    def _call(self, transformed: List[dict], source: List[int], seeds: Optional[List[int]], padded: int, static_s: int):
        """ONE model call over the rows `source` (indices into `transformed`), padded to `padded` rows by repeating the last one (its noise too). Returns the outputs and the states of the
        real rows only (padding trimmed) and the time of the call in ms."""
        import jax
        import jax.numpy as jnp
        real = len(source)
        source = source + [source[-1]] * (padded - real)
        batch, observation = self._stack([transformed[i] for i in source])
        noise = None
        if seeds is not None and self._native_samples:
            seeds = seeds + [seeds[-1]] * (padded - real)
            noise = self._row_noise(jnp.asarray(seeds, dtype=jnp.uint32), (static_s, *self._noise_shape))
            key = jax.random.key(0)                                                          # unused by the sampler once the noise is given (no augmentation at inference)
        else:
            self._key, key = jax.random.split(self._key)
        t0 = time.monotonic()
        out = np.asarray(self._sample(self._state, key, observation, num_samples=static_s, **({} if noise is None else {"noise": noise})))   # asarray waits for the device
        return out[:real * static_s], np.asarray(batch["state"])[:real], 1e3 * (time.monotonic() - t0)

    def raw_chunks(self, rows: List[dict], num_samples: int, row_seeds: Optional[Sequence[int]] = None, pad_to: Optional[int] = None):
        """The batch is `pad_to` rows (default: the next power of two). If `max_rows` is set and the batch is larger, it is split in `ceil(batch / max_rows)` calls of EQUAL size
        `ceil(batch / calls)` (100 rows, `max_rows` 64: two calls of 50), the last one padded; a call that would hold only padding is not made. The model's time is linear in the number of
        rows (measured), so the split costs nothing; it exists because the memory of one call grows with its rows (pi0-FAST: the prefill logits). The noise of a row is its own seed's,
        whatever call it falls in."""
        n, s = len(rows), num_samples
        if row_seeds is not None and len(row_seeds) != n:
            raise ValueError(f"row_seeds has {len(row_seeds)} entries for {n} rows")
        if self.max_rows is not None and int(self.max_rows) < 1:
            raise ValueError(f"max_rows must be 1 or more, got {self.max_rows}")
        transformed = [self.policy._input_transform(copy.deepcopy(r)) for r in rows]          # noqa: SLF001  one per environment
        if self._native_samples:
            source = list(range(n))                                                           # S samples drawn inside the model
            static_s = s
        else:
            source = [i for i in range(n) for _ in range(s)]                                  # FAST: one row per sample
            static_s = 1
        if pad_to is not None and pad_to < len(source):
            raise ValueError(f"pad_to={pad_to} is smaller than the {len(source)} rows of the call")
        total = int(pad_to) if pad_to is not None else padded_batch_size(len(source))
        if self.max_rows is None or total <= int(self.max_rows):
            size = total
        else:
            size = -(-total // -(-total // int(self.max_rows)))                               # equal calls: ceil(total / ceil(total / max_rows))
        seeds = None if row_seeds is None else [int(x) % (2 ** 32) for x in row_seeds]
        outs, states, infer_ms = [], [], 0.0
        for start in range(0, len(source), size):
            part = source[start:start + size]
            part_seeds = seeds[start:start + size] if seeds is not None and self._native_samples else None       # (pi0-FAST decodes greedily: no noise)
            out, state, ms = self._call(transformed, part, part_seeds, size, static_s)
            outs.append(out)
            states.append(state)
            infer_ms += ms
        out, state = np.concatenate(outs), np.concatenate(states)
        self.last_sample = (out, state)
        if self._fast is not None:
            self._record_decoding(out, n * s, n, s)
        chunks = []
        for i in range(n):
            for k in range(s):
                j = i * s + k                                      # row of `out`, (environment, sample) major, in both layouts
                src = i if self._native_samples else j             # the batch row that carries this environment's state
                res = self.policy._output_transform({"state": state[src], "actions": out[j]})  # noqa: SLF001
                chunks.append(np.asarray(res["actions"], dtype=np.float32))
        return np.stack(chunks).reshape(n, s, *chunks[0].shape), infer_ms

    # -- candidates for the RL methods (docs/algos/rl.md) ----------------------------------------------------------------------
    @property
    def native_samples(self) -> bool:
        """True when the S samples of a state are drawn inside ONE model call (pi0.5 / pi0: the prefix is computed once); False for pi0-FAST (one row per sample)."""
        return self._native_samples

    def sample_candidates(self, obs: Dict[str, np.ndarray], prompt: Union[str, Sequence[str]], num_samples: int, *,
                          row_seeds: Optional[Sequence[int]] = None, pad_to: Optional[int] = None) -> Candidates:
        """`plan` that also returns the chunks in the model's NORMALISED space (the first 8 dimensions of what the model outputs, before the output transforms) and the state rows that bring
        them back to the env's space (`normalized_to_targets`). The env-space chunks are `plan`'s own, unchanged. pi0.5 / pi0 only: pi0-FAST's output is a token string whose normalised chunk
        is the middle of its output transforms (ExtractFASTActions), to be taken out at the step of docs/algos/rl.md that prepares its path."""
        if not self._native_samples:
            raise NotImplementedError("sample_candidates: pi0-FAST's candidates are token strings; their normalised chunks are decoded inside the output transforms (not wired yet)")
        plan = self.plan(obs, prompt, num_samples, row_seeds=row_seeds, pad_to=pad_to)
        out, state = self.last_sample
        n = plan.targets.shape[0]
        out = np.asarray(out, dtype=np.float32)
        normalized = out.reshape(n, num_samples, *out.shape[1:])[..., :P.ACTION_DIM]            # (environment, sample)-major, as in `raw_chunks`
        valid = np.ones((n, num_samples), bool) if self.last_valid is None else np.asarray(self.last_valid, bool)
        return Candidates(normalized=np.ascontiguousarray(normalized), states=np.asarray(state[:n], dtype=np.float32), targets=plan.targets, valid=valid, infer_ms=plan.infer_ms)

    def normalized_to_targets(self, normalized: np.ndarray, states: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """The env's chunks (N, S, H, 8) (absolute joint targets + gripper) of normalised chunks (N, S, H, 8), through the SAME output transforms as `plan`: for the chunks `sample_candidates`
        returned (with its `states`) the result is its `targets`. `joints`: (N, >= 7) the joint positions of the observations (the offsets are from there). Used to execute or to
        distill a chunk chosen or moved (gradient ascent) in the normalised space."""
        normalized = np.asarray(normalized, dtype=np.float32)
        n, s, h, d = normalized.shape
        if d != P.ACTION_DIM or len(states) != n or len(joints) != n:
            raise ValueError(f"normalized {normalized.shape}, states {np.shape(states)}, joints {np.shape(joints)}: expected (N, S, H, {P.ACTION_DIM}) and N states and joints")
        model_dim = self._noise_shape[1]
        padded = np.zeros((h, model_dim), np.float32)                                          # the model's chunks are wider (padding); what is past the env's 8 dimensions is dropped by the output transforms
        targets = np.zeros((n, s, h, P.ACTION_DIM), np.float32)
        for i in range(n):
            for k in range(s):
                padded[:, :d] = normalized[i, k]
                chunk = np.asarray(self.policy._output_transform({"state": np.asarray(states[i]), "actions": padded.copy()})["actions"], dtype=np.float32)      # noqa: SLF001
                targets[i, k] = chunk_to_targets(chunk, np.asarray(joints[i], dtype=np.float32), self.info.output_semantics)
        return targets

    def _record_decoding(self, tokens: np.ndarray, rows: int, n: int, s: int) -> None:
        """Count the token strings of the first `rows` rows of `tokens` (the padding rows are not chunks) and set `last_valid`."""
        ex = self._fast
        valid = np.ones(rows, bool)
        for j in range(rows):
            row = np.asarray(tokens[j])
            eos = np.flatnonzero(row == FAST_EOS_TOKEN)
            length = int(eos[0]) + 1 if len(eos) else len(row)
            decoded = np.asarray(ex.tokenizer.extract_actions(row.astype(np.int32), ex.action_horizon, ex.action_dim))
            valid[j] = bool(np.any(decoded))                 # a real chunk is never exactly zero in normalised space
            self.decode_stats.add(length, no_eos=not len(eos), failed=not valid[j])
        self.last_valid = valid.reshape(n, s)

    # -- training hooks ------------------------------------------------------------------------------------------------------
    def training_batch(self, obs: Dict[str, np.ndarray], prompt: Union[str, Sequence[str]], target_actions: np.ndarray):
        """(Observation, actions) for `loss`, from the env's observations and the env-space target chunks (N, H, 8): absolute joint
        targets + gripper, e.g. a demonstration chunk. The targets are put in the model's own space (`output_semantics`) and go through
        the same input transforms as at inference, plus the normalisation of the actions."""
        import jax
        import jax.numpy as jnp
        n, h = num_envs_of(obs), self.info.action_horizon
        target = np.asarray(target_actions, dtype=np.float32)
        if target.shape != (n, h, P.ACTION_DIM):
            raise ValueError(f"target_actions must be ({n}, {h}, {P.ACTION_DIM}) (N, action_horizon, 8), got {target.shape}")
        q = np.asarray(obs[P.JOINT_POSITION], dtype=np.float32)
        items = []
        for i, row in enumerate(rows_of(obs, prompt, self.touch)):
            row["actions"] = targets_to_chunk(target[i], q[i], self.info.output_semantics)
            items.append(self.policy._input_transform(row))                                    # noqa: SLF001
        batch, observation = self._stack(items)
        return observation, jnp.asarray(batch["actions"])

    def loss(self, state, rng, observation, actions, *, train: bool = True):
        """Mean training loss of the model with weights `state` (an nnx.State: differentiable with respect to it). Flow matching for
        pi0.5 / pi0, next-token cross entropy for pi0-FAST. `train=True` switches on openpi's image augmentation."""
        import flax.nnx as nnx
        import jax.numpy as jnp
        return jnp.mean(nnx.merge(self._graphdef, state).compute_loss(rng, observation, actions, train=train))

    # -- the loss of a batch computed in slices ------------------------------------------------------------------------------
    # openpi's `compute_loss(rng, observation, actions)` draws everything random (the image augmentation of every row, the flow-matching noise and time of every row) from ONE
    # key for the WHOLE batch. A batch cut into micro-batches (one device, less memory) would draw from smaller shapes, other numbers: `prepare_loss_batch` makes the draws
    # of the whole batch once, `loss_of_rows` computes the loss of a slice of rows from them. The mean of the losses of equal slices is `compute_loss`'s mean, with the same randomness.
    def prepare_loss_batch(self, rng, observation, actions, *, train: bool = True):
        """(observation after openpi's preprocessing and image augmentation, what else `compute_loss` draws): both for the whole batch, from `rng` as `compute_loss` splits it."""
        import jax
        import openpi.models.model as _model
        if not self._native_samples:                                   # pi0-FAST: the augmentation is the only draw, from `rng` itself
            return _model.preprocess_observation(rng, observation, train=train, image_keys=list(observation.images.keys())), None
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        return observation, (noise, time)

    def loss_of_rows(self, state, observation, actions, draws):
        """Mean loss of some rows of a batch prepared by `prepare_loss_batch` (`observation`, `actions` and `draws` cut to the same rows). Pi0.5 / pi0: the body of openpi's
        `Pi0.compute_loss` after its draws, pi0-FAST: its `compute_loss` on the already augmented observation."""
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        model = nnx.merge(self._graphdef, state)
        if draws is None:
            return jnp.mean(model.compute_loss(jax.random.key(0), observation, actions, train=False))       # train=False: nothing is drawn, the images are already augmented
        from openpi.models.pi0 import make_attn_mask
        noise, time = draws
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = model.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond])
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
        return jnp.mean(jnp.mean(jnp.square(v_t - u_t), axis=-1))
