"""The critic of the RL methods: Q(s, a) of a chunk a at a state s (images, joints, gripper). JAX / Flax, one network + a delayed copy for the targets. Design and sources: docs/algos/rl.md.

    net = CriticNet.from_config(cfg)
    state = init_critic(net, jax.random.key(0), horizon=16)                # CriticState: params, target_params
    q = q_values(net, state.params, s, actions)                            # s: the state dict of the buffer (images by camera, "proprio", "touch"); actions (B, N, H * 8) or (B, N, H, 8): N candidates per state -> (B, N); a state is encoded ONCE
    z = encode_state(net, state.params, s)                                 # the state's features alone (the CNNs)
    q = score(net, state.params, z, actions)                               # the MLP alone: Q of N chunks given the features (q_values = the two in a row); the action optimisation repeats only this
    state = soft_update(state, tau=0.005)                                  # target <- tau * params + (1 - tau) * target

Architecture (paper B.1 for the shape of the MLP and the random shift; the rest is ours, the paper gives none):
  - one small CNN per camera, trained from scratch with the critic: the uint8 image is resized to `critic_image_size` (128), scaled to [-1, 1], shifted at random by up to `random_shift_px` (4) pixels
    in training only (DrQ-style: replicate padding then a random crop), 4 convolutions 3 x 3 with stride 2 (32, 64, 128, 128 channels, GroupNorm + ReLU), average pooling of the last map to 4 x 4,
    flatten, Dense(256), LayerNorm;
  - the state feature is the cameras' features, the proprioception (7 joints + gripper, as they are) and, in the tactile variant, the touch vector (src/touch/features.py: already scaled);
  - the MLP of `critic_hidden` (3 x 512): the (flattened, normalised) action chunk is concatenated to the input of EVERY layer, the output layer included; LayerNorm + ReLU after each hidden Dense;
  - the output is the scalar Q.
The chunk is in the model's NORMALISED space (docs/algos/rl.md): it is what the VLA outputs before its output transforms, so the local ascent steps (alpha = 3e-4) move it directly.
This file holds the network, its initialisation and the target copy; the loss (Cal-QL) and the update are in the critic-update step.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.algos.rl_config import RLConfig

ENCODER_CHANNELS = (32, 64, 128, 128)
ENCODER_FEATURES = 256
PROPRIO_DIM = 8                    # 7 joint positions + the gripper position


def preprocess(images: jnp.ndarray, size: int) -> jnp.ndarray:
    """uint8 (B, H, W, 3) -> float32 (B, size, size, 3) in [-1, 1], resized with antialiasing."""
    x = images.astype(jnp.float32) / 127.5 - 1.0
    if x.shape[1] != size or x.shape[2] != size:
        x = jax.image.resize(x, (x.shape[0], size, size, x.shape[3]), method="linear", antialias=True)
    return x


def random_shift(rng, images: jnp.ndarray, pad: int) -> jnp.ndarray:
    """DrQ's augmentation: every image of the batch is padded by `pad` pixels (edge replication) and cropped back at a random offset (0 .. 2 * pad in each direction)."""
    if pad <= 0:
        return images
    b, h, w, _ = images.shape
    padded = jnp.pad(images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    offsets = jax.random.randint(rng, (b, 2), 0, 2 * pad + 1)
    crop = jax.vmap(lambda img, o: jax.lax.dynamic_slice(img, (o[0], o[1], 0), (h, w, img.shape[-1])))
    return crop(padded, offsets)


class ImageEncoder(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for channels in ENCODER_CHANNELS:
            x = nn.Conv(channels, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.relu(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        return nn.LayerNorm()(nn.Dense(ENCODER_FEATURES)(x))


class CriticNet(nn.Module):
    cameras: Tuple[str, ...] = ("exterior", "wrist")
    hidden: Tuple[int, ...] = (512, 512, 512)
    image_size: int = 128
    shift_px: int = 4
    touch_dim: int = 0                 # 0 = the critic does not see touch

    @classmethod
    def from_config(cls, cfg: RLConfig) -> "CriticNet":
        return cls(cameras=tuple(cfg.critic_cameras), hidden=tuple(cfg.critic_hidden), image_size=cfg.critic_image_size, shift_px=cfg.random_shift_px, touch_dim=cfg.critic_touch.dim)

    @nn.compact
    def __call__(self, state: Optional[Dict[str, jnp.ndarray]], actions: Optional[jnp.ndarray], train: bool = False, features: Optional[jnp.ndarray] = None, return_features: bool = False) -> jnp.ndarray:
        """state: camera -> uint8 (B, H, W, 3), "proprio" (B, 8) and, with `touch_dim`, "touch" (B, touch_dim); actions (B, N, A): N chunks per state, A = H * 8 flattened. Returns Q (B, N).
        `train` switches on the random shift (it draws from the `shift` RNG stream: `apply(..., rngs={"shift": key})`).
        The two halves can be run apart (the action optimisation changes the chunk many times at a fixed state): `return_features=True` stops after the state's encoding and returns it (B, F);
        `features=` gives that encoding back and skips the encoders (then `state` is not read). `encode_state` and `score` below wrap the two."""
        if features is None:
            features_list = []
            for name in self.cameras:
                x = preprocess(state[name], self.image_size)
                if train and self.shift_px > 0:
                    x = random_shift(self.make_rng("shift"), x, self.shift_px)
                features_list.append(ImageEncoder(name=f"encoder_{name}")(x))
            features_list.append(state["proprio"].astype(jnp.float32))
            if self.touch_dim:
                if "touch" not in state or state["touch"].shape[-1] != self.touch_dim:
                    raise ValueError(f"the critic sees touch ({self.touch_dim} values) but the state has {None if 'touch' not in state else state['touch'].shape}")
                features_list.append(state["touch"].astype(jnp.float32))
            elif "touch" in state:
                raise ValueError("the state carries touch but the critic was built without it (touch_dim = 0)")
            features = jnp.concatenate(features_list, axis=-1)                                 # (B, F)
        if return_features:
            return features
        n = actions.shape[1]
        h = jnp.broadcast_to(features[:, None, :], (features.shape[0], n, features.shape[-1]))  # the state is encoded once, shared by the N chunks
        for i, width in enumerate(self.hidden):
            h = nn.relu(nn.LayerNorm()(nn.Dense(width, name=f"dense_{i}")(jnp.concatenate([h, actions], axis=-1))))
        return nn.Dense(1, name="out")(jnp.concatenate([h, actions], axis=-1))[..., 0]


@dataclasses.dataclass
class CriticState:
    params: dict
    target_params: dict


def flatten_actions(actions: jnp.ndarray) -> jnp.ndarray:
    """(B, N, H, 8) -> (B, N, H * 8); an array that is already (B, N, A) is returned as it is."""
    return actions.reshape(*actions.shape[:2], -1) if actions.ndim == 4 else actions


def init_critic(net: CriticNet, rng, horizon: int, image_hw: Tuple[int, int] = (224, 224), action_dim: int = 8) -> CriticState:
    """Parameters initialised on dummy inputs; the target copy starts equal to them."""
    state = {name: jnp.zeros((1, *image_hw, 3), jnp.uint8) for name in net.cameras}
    state["proprio"] = jnp.zeros((1, PROPRIO_DIM))
    if net.touch_dim:
        state["touch"] = jnp.zeros((1, net.touch_dim))
    params = net.init({"params": rng}, state, jnp.zeros((1, 1, horizon * action_dim)), train=False)["params"]
    return CriticState(params=params, target_params=jax.tree.map(jnp.array, params))


def encode_state(net: CriticNet, params: dict, state: Dict[str, jnp.ndarray], *, train: bool = False, rng=None) -> jnp.ndarray:
    """The state's feature vector (B, F): images through the encoders (random shift when `train`, which needs `rng`), proprioception, touch."""
    if train and net.shift_px > 0 and rng is None:
        raise ValueError("train=True with a random shift needs an rng")
    return net.apply({"params": params}, state, None, train=train, return_features=True, rngs={"shift": rng} if train and net.shift_px > 0 else None)


def score(net: CriticNet, params: dict, features: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    """Q of N chunks at each of B states whose features are given: (B, N). Only the MLP runs: this is what the local ascent repeats."""
    return net.apply({"params": params}, None, flatten_actions(actions), features=features)


def q_values(net: CriticNet, params: dict, state: Dict[str, jnp.ndarray], actions: jnp.ndarray, *, train: bool = False, rng=None) -> jnp.ndarray:
    """Q of N chunks at each of B states: (B, N). `train` needs `rng` (the random shift)."""
    return score(net, params, encode_state(net, params, state, train=train, rng=rng), actions)


def soft_update(state: CriticState, tau: float) -> CriticState:
    """The delayed target network: target <- tau * params + (1 - tau) * target (paper Eq. 3.3's delayed target Q; tau = 0.005, a decision: the paper gives none)."""
    if not 0 < tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    return CriticState(params=state.params, target_params=jax.tree.map(lambda t, p: (1 - tau) * t + tau * p, state.target_params, state.params))


def num_parameters(params: dict) -> int:
    return int(sum(x.size for x in jax.tree.leaves(params)))
