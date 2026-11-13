"""The training samples of our SFT are the ones expo-ft's SFT trains on, tensor by tensor.

expo-ft: LeRobot sample -> openpi's data loader transforms of the config `expo_pi05_droid_lora_finetune_sft_joint_state_delta`
    (Repack -> DroidInputs -> DeltaActions -> Normalize(quantiles) -> ResizeImages -> TokenizePrompt -> PadStatesAndActions).
Ours: `DemoSet.sample` -> `rows_of` -> `targets_to_chunk` -> the input transforms of the config `expo_pi05_droid_lora_finetune_sft_joint_state`
    (DroidInputs -> Normalize -> ResizeImages -> TokenizePrompt -> PadStatesAndActions), the ones `OpenPIPolicy.training_batch` uses.

Both start from the same synthetic episodes. The tokenizer is left out of both chains (it needs a download): the prompt STRING is compared, and the
tokens are a function of the prompt and of the normalised state, which are compared.
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openpi")
pytest.importorskip("openpi_client")
from openpi import transforms as T                                                            # noqa: E402
from openpi.training import config as C                                                      # noqa: E402

from src.algos.demos import DemoSet, Episode                                                  # noqa: E402
from src.envs import protocol as P                                                            # noqa: E402
from src.models.interface import rows_of, targets_to_chunk                                    # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NORM = REPO / "assets/forge_demo_norm/forge_demos_robotiq_noisy/norm_stats.json"
HORIZON = 16
PROMPT = "insert the peg into the hole"


@pytest.fixture(autouse=True)
def no_tokenizer_download(monkeypatch):
    """openpi's config builds the PaliGemma tokenizer (a download from gs://) when it creates the transforms; the chains below drop TokenizePrompt anyway."""
    from openpi.models import tokenizer
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", lambda *a, **k: None)


def _local(cfg):
    """The config with its norm stats read from OUR folder instead of gs:// (what the SFT of expo-ft does with `sft_norm_stats_dir` / `sft_norm_stats_asset_id`)."""
    assets = dataclasses.replace(cfg.data.assets, assets_dir=str(NORM.parent.parent), asset_id=NORM.parent.name)
    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, assets=assets))


def _episodes(n=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for e in range(n):
        steps = int(rng.integers(20, 40))                                                     # shorter than 2 chunks sometimes: the end-of-episode padding is exercised
        q = np.cumsum(rng.normal(0, 0.03, (steps, 7)), axis=0).astype(np.float32) + rng.normal(0, 0.3, 7).astype(np.float32)
        grip = rng.uniform(0, 0.8, (steps, 1)).astype(np.float32)
        act = np.concatenate([q + rng.normal(0, 0.02, (steps, 7)).astype(np.float32), (rng.uniform(size=(steps, 1)) > 0.5).astype(np.float32)], axis=1)
        obs = {P.EXTERIOR_IMAGE: rng.integers(0, 256, (steps, 224, 224, 3), dtype=np.uint8),
               P.WRIST_IMAGE: rng.integers(0, 256, (steps, 224, 224, 3), dtype=np.uint8),
               P.JOINT_POSITION: q, P.GRIPPER_POSITION: grip}
        out.append(Episode(obs=obs, actions=act, prompt=PROMPT, source=f"synthetic/ep_{e}.npz"))
    return out


def _without_tokenizer(transforms):
    return [t for t in transforms if not isinstance(t, T.TokenizePrompt)]


def _expo_ft_chain():
    cfg = _local(C.get_config("expo_pi05_droid_lora_finetune_sft_joint_state_delta"))
    data = cfg.data.create(cfg.assets_dirs, cfg.model)                                      # data.norm_stats: loaded by openpi from the folder above, as train.py does
    return T.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs, T.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                      *_without_tokenizer(data.model_transforms.inputs)])


def _our_transforms():
    from src.models.loading import load_norm_stats
    cfg = _local(C.get_config("expo_pi05_droid_lora_finetune_sft_joint_state"))
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = load_norm_stats({"norm_stats": {"dir": str(NORM.parent.parent), "asset_id": NORM.parent.name}}, cfg, Path("unused"))
    return [T.InjectDefaultPrompt(None), *data.data_transforms.inputs, T.Normalize(norm_stats, use_quantiles=data.use_quantile_norm),
            *_without_tokenizer(data.model_transforms.inputs)]


def _our_chain():
    return T.compose(_our_transforms())


def _lerobot_sample(ds: DemoSet, row: int) -> dict:
    """What LeRobotDataset returns for frame `row` of the dataset convert_waypoint_demos_to_lerobot.py writes: images float32 CHW in [0, 1] (PNG decoded,
    torchvision to_tensor), the 16 following actions with the last one repeated (delta_timestamps), the task string."""
    ep = int(ds.episode_of[row])
    img = lambda key: (ds.obs[key][row].astype(np.float32) / np.float32(255)).transpose(2, 0, 1)             # noqa: E731
    return {"exterior_image_1_left": img(P.EXTERIOR_IMAGE), "exterior_image_2_left": np.zeros((3, 224, 224), np.float32),
            "wrist_image_left": img(P.WRIST_IMAGE), "joint_position": ds.obs[P.JOINT_POSITION][row],
            "gripper_position": ds.obs[P.GRIPPER_POSITION][row], "actions": ds.actions[ds.chunk_rows(np.array([row]))[0]].copy(),
            "prompt": ds.prompts[ep]}


def test_our_training_sample_equals_the_one_of_expo_fts_loader_tensor_by_tensor():
    ds = DemoSet(_episodes(), HORIZON)
    theirs, ours = _expo_ft_chain(), _our_chain()
    rows = np.arange(len(ds))                                                                  # every frame of every episode, the last ones (padded chunks) included
    obs, prompts, targets = ds.sample(rows)
    q = obs[P.JOINT_POSITION]
    checked = 0
    for i, row in enumerate(rows):
        a = theirs(_lerobot_sample(ds, int(row)))
        r = rows_of({k: v[i:i + 1] for k, v in obs.items()}, prompts[i])[0]
        r["actions"] = targets_to_chunk(targets[i], q[i], "offsets_from_chunk_start_state")
        b = ours(r)
        assert a["prompt"] == b["prompt"] == PROMPT
        for key in ("state", "actions"):
            assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype, key
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{key} of row {row}")
        for cam in a["image"]:
            np.testing.assert_array_equal(a["image"][cam], b["image"][cam], err_msg=f"image {cam} of row {row}")
            assert bool(a["image_mask"][cam]) == bool(b["image_mask"][cam]), cam
        checked += 1
    assert checked == len(ds) > 50


def test_the_gripper_command_and_the_offsets_look_like_what_the_loss_should_see():
    """Sanity of the comparison itself: the transformed actions are not trivially equal because both are zero or constant."""
    ds = DemoSet(_episodes(1), HORIZON)
    a = _expo_ft_chain()(_lerobot_sample(ds, 0))
    assert a["actions"].shape == (HORIZON, 32) and np.abs(a["actions"][:, :8]).std() > 0.05 and a["state"].shape == (32,)


def test_training_batch_of_the_policy_equals_expo_fts_samples_stacked():
    """The same through `OpenPIPolicy.training_batch` itself (stacking, observation, actions array), on a tiny pi0.5 whose input transforms are OUR real list (the tokenizer
    apart): the Observation and the actions the loss receives are the stacked samples of expo-ft's loader."""
    pytest.importorskip("jax")
    import types

    import jax
    from openpi.models import pi0_config
    from openpi.policies import policy as openpi_policy
    from test_openpi_policy import TinyTokens

    from src.models.openpi_policy import OpenPIPolicy

    base = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=HORIZON)
    pol = openpi_policy.Policy(base.create(jax.random.key(0)), transforms=[*_our_transforms(), TinyTokens()], output_transforms=[])
    cfg = {"name": "tiny", "family": "pi05", "action_horizon": HORIZON, "action_dim": 8, "output_semantics": "offsets_from_chunk_start_state"}
    policy = OpenPIPolicy(pol, types.SimpleNamespace(freeze_filter=base.get_freeze_filter()), cfg, seed=0)

    ds = DemoSet(_episodes(2, seed=5), HORIZON)
    rows = np.array([0, 3, len(ds) - 1, len(ds) // 2, 7, len(ds) - 2])                           # the last frames of an episode among them
    obs, prompts, targets = ds.sample(rows)
    observation, actions = policy.training_batch(obs, prompts, targets)
    theirs = _expo_ft_chain()
    samples = [theirs(_lerobot_sample(ds, int(r))) for r in rows]
    # The transforms return float64 arrays (openpi's norm stats are float64); JAX (x64 off) rounds them to float32 on the way to the device, in expo-ft's loader as here.
    np.testing.assert_array_equal(np.asarray(actions), np.stack([x["actions"] for x in samples]).astype(np.float32))
    np.testing.assert_array_equal(np.asarray(observation.state), np.stack([x["state"] for x in samples]).astype(np.float32))
    # openpi's Observation holds images as float32 in [-1, 1]: the same conversion as in the training loop of openpi (uint8 / 255 * 2 - 1)
    for cam in samples[0]["image"]:
        want = np.stack([x["image"][cam] for x in samples]).astype(np.float32) / 255.0 * 2.0 - 1.0
        np.testing.assert_allclose(np.asarray(observation.images[cam]), want, atol=1e-6, err_msg=cam)
        np.testing.assert_array_equal(np.asarray(observation.image_masks[cam]), np.stack([x["image_mask"][cam] for x in samples]), err_msg=cam)
