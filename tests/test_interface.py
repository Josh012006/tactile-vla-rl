"""src/models/interface.py: what an algorithm sees of a model. numpy only (no JAX, no openpi, no GPU)."""
import numpy as np
import pytest

from src.envs import protocol as P
from src.models import interface as I
from src.models.interface import CallablePolicy, ModelInfo, StaticPolicy

# What openpi's DroidInputs reads at inference (third_party/openpi/src/openpi/policies/droid_policy.py).
DROID_KEYS = {"observation/exterior_image_1_left", "observation/wrist_image_left", "observation/joint_position",
              "observation/gripper_position", "prompt"}


def obs_of(n=2):
    return {P.EXTERIOR_IMAGE: np.full((n, 4, 4, 3), 7, np.uint8), P.WRIST_IMAGE: np.full((n, 4, 4, 3), 9, np.uint8),
            P.JOINT_POSITION: np.arange(n * 7, dtype=np.float32).reshape(n, 7), P.GRIPPER_POSITION: np.full((n, 1), 0.5, np.float32),
            P.TACTILE: np.ones((n, 2, 7, 4), np.float32)}


def cfg_of(semantics="absolute_targets", horizon=4):
    return {"name": "m", "family": "pi05", "action_horizon": horizon, "output_semantics": semantics}


# -- observation in ------------------------------------------------------------------------------------------------------------
def test_the_observation_has_exactly_the_keys_openpi_reads():
    o = I.to_openpi_obs(obs_of(), env=1, prompt="insert the peg")
    assert set(o) == DROID_KEYS and o["prompt"] == "insert the peg"          # the tactile key of the env is not handed to the model
    assert np.array_equal(o["observation/joint_position"], np.arange(7, 14, dtype=np.float32))      # environment 1, not 0
    assert o["observation/gripper_position"].shape == (1,) and o["observation/exterior_image_1_left"][0, 0, 0] == 7
    assert o["observation/wrist_image_left"][0, 0, 0] == 9


def test_rows_take_one_prompt_for_all_or_one_per_environment():
    assert [r["prompt"] for r in I.rows_of(obs_of(3), "a")] == ["a", "a", "a"]
    assert [r["prompt"] for r in I.rows_of(obs_of(2), ["a", "b"])] == ["a", "b"]
    with pytest.raises(ValueError, match="prompts"):
        I.rows_of(obs_of(3), ["a", "b"])


def test_take_keeps_the_chosen_environments_of_every_key():
    sub = I.take(obs_of(4), [3, 1])
    assert sub[P.JOINT_POSITION].shape == (2, 7) and sub[P.TACTILE].shape == (2, 2, 7, 4)
    assert np.array_equal(sub[P.JOINT_POSITION][0], obs_of(4)[P.JOINT_POSITION][3])


@pytest.mark.parametrize("n,want", [(1, 1), (2, 2), (3, 4), (5, 8), (8, 8), (9, 16), (17, 32)])
def test_the_batch_is_padded_to_a_power_of_two(n, want):
    assert I.padded_batch_size(n) == want


# -- chunk out ----------------------------------------------------------------------------------------------------------------------
def test_offsets_are_added_to_the_plan_state_once_and_absolute_targets_are_left_alone():
    q = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    chunk = np.tile(np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 1.0], np.float32), (5, 1))
    off = I.chunk_to_targets(chunk, q, "offsets_from_chunk_start_state")
    assert np.allclose(off[:, :7], chunk[:, :7] + q) and (off[:, 7] == 1.0).all() and off.shape == (5, P.ACTION_DIM)
    ab = I.chunk_to_targets(chunk, q, "absolute_targets")
    assert np.array_equal(ab[:, :7], chunk[:, :7]) and (ab[:, 7] == 1.0).all()
    longer = np.zeros((3, 32), np.float32)                          # a model that returns more columns: only the first 8 are used
    assert I.chunk_to_targets(longer, q, "absolute_targets").shape == (3, P.ACTION_DIM)
    with pytest.raises(ValueError):
        I.chunk_to_targets(np.zeros((3, 7), np.float32), q, "absolute_targets")
    with pytest.raises(ValueError):
        I.chunk_to_targets(chunk, q, "velocity")


@pytest.mark.parametrize("semantics", I.SEMANTICS)
def test_targets_to_chunk_is_the_inverse_of_chunk_to_targets(semantics):
    rng = np.random.default_rng(0)
    q, targets = rng.normal(size=7).astype(np.float32), rng.normal(size=(6, 8)).astype(np.float32)
    back = I.chunk_to_targets(I.targets_to_chunk(targets, q, semantics), q, semantics)
    assert np.allclose(back, targets, atol=1e-6)


def test_the_training_target_of_an_offsets_model_is_the_target_minus_the_state_and_the_gripper_is_left_alone():
    """openpi's DeltaActions with make_bool_mask(7, -1): the 7 joints are offsets, the gripper stays absolute."""
    q = np.arange(7, dtype=np.float32)
    targets = np.tile(np.arange(8, dtype=np.float32) + 10, (3, 1))
    chunk = I.targets_to_chunk(targets, q, "offsets_from_chunk_start_state")
    assert np.array_equal(chunk[:, :7], np.full((3, 7), 10, np.float32)) and (chunk[:, 7] == 17).all()
    assert np.array_equal(targets[:, :7], np.tile(np.arange(7, dtype=np.float32) + 10, (3, 1)))      # the input is not modified
    with pytest.raises(ValueError):
        I.targets_to_chunk(np.zeros((3, 7), np.float32), q, "absolute_targets")


# -- the policies ---------------------------------------------------------------------------------------------------------------------
class Recorder:
    """An openpi-style infer(): a chunk whose first joint is the call number, so that samples can be told apart."""

    def __init__(self, h=4):
        self.h, self.calls = h, []

    def __call__(self, row):
        self.calls.append(row)
        chunk = np.zeros((self.h, 8), np.float32)
        chunk[:, 0] = len(self.calls)
        return {"actions": chunk, "policy_timing": {"infer_ms": 2.0}}


def test_plan_returns_env_actions_with_a_sample_axis_and_the_raw_chunks():
    model = Recorder()
    policy = CallablePolicy(model, cfg_of("offsets_from_chunk_start_state"))
    plan = policy.plan(obs_of(2), "p", num_samples=3)
    assert plan.targets.shape == (2, 3, 4, P.ACTION_DIM) and plan.raw.shape == (2, 3, 4, 8) and len(model.calls) == 6
    assert plan.infer_ms == 12.0
    q = obs_of(2)[P.JOINT_POSITION]
    for i in range(2):
        for s in range(3):                                                  # [environment][sample] order, state of THAT environment added
            assert np.allclose(plan.targets[i, s, :, 0], plan.raw[i, s, :, 0] + q[i, 0])
    assert policy.act(obs_of(2), "p").shape == (2, 4, P.ACTION_DIM)
    assert policy.sample(obs_of(2), "p", 5).shape == (2, 5, 4, P.ACTION_DIM)
    with pytest.raises(ValueError, match="num_samples"):
        policy.plan(obs_of(2), "p", num_samples=0)


def test_a_chunk_of_the_wrong_length_is_refused_and_not_silently_executed():
    with pytest.raises(ValueError, match="action_horizon"):
        CallablePolicy(Recorder(h=3), cfg_of(horizon=4)).plan(obs_of(1), "p")


def test_a_static_policy_targets_the_current_joints_in_both_readings():
    for semantics in I.SEMANTICS:
        plan = StaticPolicy(action_horizon=5, output_semantics=semantics).plan(obs_of(2), "p", num_samples=2)
        assert plan.targets.shape == (2, 2, 5, P.ACTION_DIM)
        assert np.allclose(plan.targets[1, 0, :, :7], obs_of(2)[P.JOINT_POSITION][1]) and np.allclose(plan.targets[:, :, :, 7], 0.5)


# -- the YAML -----------------------------------------------------------------------------------------------------------------------------
def test_model_info_is_validated():
    assert ModelInfo.from_cfg(cfg_of()).action_horizon == 4
    with pytest.raises(ValueError, match="output_semantics"):
        ModelInfo.from_cfg(cfg_of("velocity"))
    with pytest.raises(ValueError, match="action_dim"):
        ModelInfo.from_cfg({**cfg_of(), "action_dim": 7})
    with pytest.raises(ValueError, match="action_horizon"):
        ModelInfo.from_cfg(cfg_of(horizon=0))


@pytest.mark.parametrize("name", ["pi05", "pi0_fast"])
def test_the_shipped_model_yamls_give_a_valid_model_info(name):
    info = ModelInfo.from_cfg(I.load_model_cfg(name))
    assert info.name == name and info.action_dim == P.ACTION_DIM and info.action_horizon > 0


def test_overrides_change_a_copy_not_the_loaded_config():
    cfg = {"use_lora": True, "lora_rank": {"vlm": 16, "action_head": 32}}
    out = I.apply_overrides(cfg, use_lora=False, lora_rank_vlm=8)
    assert out == {"use_lora": False, "lora_rank": {"vlm": 8, "action_head": 32}}
    assert cfg == {"use_lora": True, "lora_rank": {"vlm": 16, "action_head": 32}}
    assert I.apply_overrides(cfg) == cfg


def test_load_model_refuses_an_unknown_family_before_importing_anything_heavy():
    from src.models import load_model
    with pytest.raises(ValueError, match="unknown model family"):
        load_model({**cfg_of(), "family": "mystery"})
