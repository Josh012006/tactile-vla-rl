"""The FORGE demonstrations (the ep_*.npz files of scripts/demos, see docs/demos.md) as a training set. numpy only: no JAX, no openpi.

A demonstration holds, for every step t of an episode, the observation BEFORE the action `actions[t]` (7 absolute joint targets + the
gripper command). A training sample is a step t of an episode: the observation at t, and the chunk of the next `horizon` actions,
`actions[t : t + horizon]`, with the last action repeated past the end of the episode (what the LeRobot loader of openpi does, and what
scripts/fast_tokens.py measured). The chunk stays in the ENV's action space (absolute targets): the model's own space (offsets from the
chunk-start state for pi0.5, absolute for pi0-FAST) is applied by `OpenPIPolicy.training_batch`, through `output_semantics`.

The samples are served in the format of the env's observation dict (src/envs/protocol.py keys, one row per sample), which is what
`OpenPIPolicy.training_batch` and `VLAPolicy.plan` take: the SFT, the evaluation of a checkpoint and the RL read the same thing. The touch
signals the demonstrations hold (`tactile`, `ft_force`, `gripper_effort`, `peg_socket_offset`) are served as well, under the env's keys, so
an SFT with touch needs no new data path.

The demonstrations are read from the .npz files, not from the LeRobot dataset the expo-ft SFT used: same episodes, same frames, but the
LeRobot copy has no touch and needs the `lerobot` package.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.envs import protocol as P

IMAGE_SIZE = (224, 224, 3)
# env observation key -> array of the .npz, and the shape of ONE step. The first four are required, the others are served when the file has them.
REQUIRED = {
    P.EXTERIOR_IMAGE: ("exterior_image", IMAGE_SIZE),
    P.WRIST_IMAGE: ("wrist_image", IMAGE_SIZE),
    P.JOINT_POSITION: ("joint_position", (7,)),
    P.GRIPPER_POSITION: ("gripper_position", (1,)),
}
OPTIONAL = {
    P.FT_FORCE: ("ft_force", (3,)),
    P.PEG_SOCKET_OFFSET: ("peg_socket_offset", (3,)),
    P.GRIPPER_EFFORT: ("gripper_effort", ()),
    P.TACTILE: ("tactile", (2, 7, 4)),
}


@dataclasses.dataclass
class Episode:
    obs: Dict[str, np.ndarray]      # env key -> (T, ...)
    actions: np.ndarray             # (T, 8) float32
    prompt: str
    source: str                     # "<folder name>/<file name>"


def episode_files(folder: Path) -> List[Path]:
    """The successful demonstrations of a folder, in order (failed episodes, `fail_seed*.npz`, are never trained on)."""
    return sorted(Path(folder).glob("ep_*.npz"))


def read_episode(path: Path, default_prompt: str) -> Episode:
    """One .npz, checked: every required array present, shapes and dtypes as the env sends them, values finite."""
    with np.load(path) as z:
        files = set(z.files)
        missing = [name for name, _ in REQUIRED.values() if name not in files] + (["actions"] if "actions" not in files else [])
        if missing:
            raise ValueError(f"{path.name}: missing arrays {missing}")
        actions = z["actions"].astype(np.float32)
        steps = len(actions)
        obs: Dict[str, np.ndarray] = {}
        for key, (name, shape) in {**REQUIRED, **{k: v for k, v in OPTIONAL.items() if v[0] in files}}.items():
            arr = z[name]
            if key == P.GRIPPER_POSITION:
                arr = arr.reshape(len(arr), -1)
            if arr.shape != (steps, *shape):
                raise ValueError(f"{path.name}: {name} has shape {arr.shape}, expected {(steps, *shape)}")
            if key in (P.EXTERIOR_IMAGE, P.WRIST_IMAGE):
                if arr.dtype != np.uint8:
                    raise ValueError(f"{path.name}: {name} must be uint8, got {arr.dtype}")
            else:
                arr = arr.astype(np.float32)
                if not np.isfinite(arr).all():
                    raise ValueError(f"{path.name}: non-finite values in {name}")
            obs[key] = arr
        prompt = str(z["prompt"]) if "prompt" in files else default_prompt
    if actions.shape != (steps, P.ACTION_DIM) or steps < 1 or not np.isfinite(actions).all():
        raise ValueError(f"{path.name}: actions must be finite, shape (T, {P.ACTION_DIM}) with T >= 1, got {actions.shape}")
    return Episode(obs=obs, actions=actions, prompt=prompt, source=f"{path.parent.name}/{path.name}")


class DemoSet:
    """The steps of a list of episodes, as training samples. `len` = the number of steps; `sample(rows)` = a batch."""

    def __init__(self, episodes: Sequence[Episode], horizon: int):
        if not episodes:
            raise ValueError("no episode")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        keys = set(episodes[0].obs)
        for ep in episodes:
            if set(ep.obs) != keys:
                raise ValueError(f"{ep.source} has the arrays {sorted(ep.obs)} but {episodes[0].source} has {sorted(keys)}: "
                                 "do not mix demonstrations recorded with and without taxels in one set")
        self.horizon = horizon
        self.episodes = list(episodes)
        lengths = np.array([len(ep.actions) for ep in episodes])
        self.episode_of = np.repeat(np.arange(len(episodes)), lengths)                  # (N,) episode of every row
        first = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self.last_row = np.repeat(first + lengths - 1, lengths)                           # (N,) last row of the row's own episode
        self.obs = {k: np.concatenate([ep.obs[k] for ep in episodes]) for k in sorted(keys)}
        self.actions = np.concatenate([ep.actions for ep in episodes])                    # (N, 8)
        self.prompts = [ep.prompt for ep in episodes]

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self.obs)

    def chunk_rows(self, rows: np.ndarray) -> np.ndarray:
        """(B, horizon): the rows of the actions that follow each row, the last one repeated past the end of the episode."""
        rows = np.asarray(rows)
        return np.minimum(rows[:, None] + np.arange(self.horizon)[None, :], self.last_row[rows][:, None])

    def sample(self, rows: Sequence[int]) -> Tuple[Dict[str, np.ndarray], List[str], np.ndarray]:
        """(observation dict with B rows, one prompt per row, target chunks (B, horizon, 8) in the env's action space)."""
        rows = np.asarray(rows, dtype=np.int64)
        return ({k: v[rows] for k, v in self.obs.items()}, [self.prompts[e] for e in self.episode_of[rows]],
                self.actions[self.chunk_rows(rows)])

    def describe(self) -> dict:
        lengths = np.bincount(self.episode_of)
        return {"episodes": len(self.episodes), "steps": len(self), "steps_per_episode": [int(lengths.min()), float(lengths.mean()), int(lengths.max())],
                "horizon": self.horizon, "keys": list(self.keys), "prompts": sorted(set(self.prompts)),
                "bytes": int(sum(v.nbytes for v in self.obs.values()) + self.actions.nbytes)}


def load_demos(folders: Sequence[str], horizon: int, *, num_demos: int = 0, val_episodes: int = 0, default_prompt: str,
               root: Optional[Path] = None) -> Tuple[DemoSet, Optional[DemoSet]]:
    """(training set, held-out set or None) from folders of ep_*.npz.
    num_demos: the first N episodes of EACH folder (0 = all). val_episodes: the LAST V episodes of each folder (of those kept) are held
    out of the training set (0 = none: the SFT of expo-ft trained on every demonstration, and the real test is the success rate in the env)."""
    if not folders:
        raise ValueError("no demonstration folder")
    if num_demos < 0 or val_episodes < 0:
        raise ValueError("num_demos and val_episodes must be >= 0")
    train: List[Episode] = []
    held: List[Episode] = []
    for folder in folders:
        path = Path(folder) if Path(folder).is_absolute() or root is None else Path(root) / folder
        files = episode_files(path)
        if not files:
            raise FileNotFoundError(f"no ep_*.npz in {path}")
        if num_demos:
            files = files[:num_demos]
        if val_episodes >= len(files):
            raise ValueError(f"{path}: val_episodes={val_episodes} leaves no training episode out of {len(files)}")
        cut = len(files) - val_episodes
        episodes = [read_episode(f, default_prompt) for f in files]
        train += episodes[:cut]
        held += episodes[cut:]
    return DemoSet(train, horizon), (DemoSet(held, horizon) if held else None)
