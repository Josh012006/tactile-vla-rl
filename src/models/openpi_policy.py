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
import time
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from src.envs import protocol as P
from src.models.interface import ModelInfo, VLAPolicy, num_envs_of, padded_batch_size, rows_of, targets_to_chunk
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
        self.decode_stats = DecodeStats()                    # pi0-FAST: what the generated tokens were like (cumulative, reset() to restart)
        self.last_valid: Optional[np.ndarray] = None         # pi0-FAST: (N, S) bool of the last `plan`: False = the chunk could not be decoded
        self._fast = None if self._native_samples else fast_extractor(policy)
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
    def load(cls, cfg: dict, seed: int = 0) -> "OpenPIPolicy":
        from src.models.loading import load_policy
        policy, train_cfg = load_policy(cfg)
        return cls(policy, train_cfg, cfg, seed)

    def fork(self, seed: int = 0) -> "OpenPIPolicy":
        """A second policy over the same model, for another thread (the evaluation of several checkpoints at once, src/eval/sweep.py). It SHARES what is heavy and stateless: the
        openpi `Policy` (transforms, norm stats), the jitted sampler (one compilation, and so one arithmetic, for all the forks) and, until `set_state` replaces them, the weights
        in device memory (`load_checkpoint` on a fork replaces the trainable parameters only: the frozen ones stay shared, so N forks hold one base model). It does NOT share what
        a call mutates: its own random stream (`seed`), its own weights reference (`set_state`) and its own `decode_stats` / `last_valid`."""
        other = copy.copy(self)
        other.decode_stats = DecodeStats()
        other.last_valid = None
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

    def raw_chunks(self, rows: List[dict], num_samples: int, row_seeds: Optional[Sequence[int]] = None, pad_to: Optional[int] = None):
        import jax
        import jax.numpy as jnp
        n, s = len(rows), num_samples
        if row_seeds is not None and len(row_seeds) != n:
            raise ValueError(f"row_seeds has {len(row_seeds)} entries for {n} rows")
        transformed = [self.policy._input_transform(copy.deepcopy(r)) for r in rows]          # noqa: SLF001  one per environment
        if self._native_samples:
            source = list(range(n))                                                           # S samples drawn inside the model
            static_s = s
        else:
            source = [i for i in range(n) for _ in range(s)]                                  # FAST: one row per sample
            static_s = 1
        if pad_to is not None and pad_to < len(source):
            raise ValueError(f"pad_to={pad_to} is smaller than the {len(source)} rows of the call")
        padded = int(pad_to) if pad_to is not None else padded_batch_size(len(source))
        source = source + [source[-1]] * (padded - len(source))
        batch, observation = self._stack([transformed[i] for i in source])
        noise = None
        if row_seeds is not None and self._native_samples:
            seeds = [int(x) % (2 ** 32) for x in row_seeds]
            seeds = seeds + [seeds[-1]] * (padded - n)                                       # the padding rows repeat the last row, noise included
            noise = self._row_noise(jnp.asarray(seeds, dtype=jnp.uint32), (static_s, *self._noise_shape))
            key = jax.random.key(0)                                                          # unused by the sampler once the noise is given (no augmentation at inference)
        else:
            self._key, key = jax.random.split(self._key)
        t0 = time.monotonic()
        out = np.asarray(self._sample(self._state, key, observation, num_samples=static_s, **({} if noise is None else {"noise": noise})))   # asarray waits for the device
        infer_ms = 1e3 * (time.monotonic() - t0)
        state = np.asarray(batch["state"])
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
