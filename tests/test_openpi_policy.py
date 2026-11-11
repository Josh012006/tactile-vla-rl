"""src/models/openpi_policy.py on CPU, with a tiny pi0.5 (gemma `dummy` variants) and openpi's own DroidInputs / DroidOutputs / padding.
The prompt tokenizer (it downloads PaliGemma's) and the DROID normalisation (it needs the checkpoint's norm stats) are replaced by stubs:
what is tested is the plumbing of the batched path (order of environments and samples, padding, state per row, weights as an argument,
training batch), not the real transforms, which are openpi's and run on the cluster (`python -m scripts.probe_vla` prints the
check of this path against openpi's own `Policy.infer` on the real model).

Needs the `vla` dependency group (skipped otherwise). The SigLIP tower of openpi is not downsized by any variant: a few minutes of
compilation on CPU for the four batch shapes used here.
"""
import dataclasses
import types

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("openpi")

import flax.nnx as nnx  # noqa: E402
import flax.traverse_util as tu  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from openpi import transforms  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.policies import droid_policy  # noqa: E402
from openpi.policies import policy as openpi_policy  # noqa: E402

from src.envs import protocol as P  # noqa: E402
from src.models.interface import ModelInfo  # noqa: E402
from src.models.openpi_policy import OpenPIPolicy  # noqa: E402

H, MODEL_DIM = 4, 32


@dataclasses.dataclass(frozen=True)
class TinyTokens(transforms.DataTransformFn):
    """What the PaliGemma tokenizer would add: a prompt as 8 token ids (and no string left in the dict)."""

    def __call__(self, data):
        data = dict(data)
        data.pop("prompt", None)
        data["tokenized_prompt"], data["tokenized_prompt_mask"] = np.arange(8, dtype=np.int32), np.ones(8, bool)
        return data


@dataclasses.dataclass(frozen=True)
class StateTag(transforms.DataTransformFn):
    """Output transform that returns, as the actions, the first element of the state the row carries: tells which row went where."""

    def __call__(self, data):
        return {"actions": np.zeros((H, 8), np.float32) + float(np.asarray(data["state"])[0])}


def build(family="pi05", outputs=None):
    base = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=H)
    model = base.create(jax.random.key(0))
    pol = openpi_policy.Policy(
        model,
        transforms=[droid_policy.DroidInputs(model_type=base.model_type), TinyTokens(), transforms.PadStatesAndActions(MODEL_DIM)],
        output_transforms=outputs or [droid_policy.DroidOutputs(action_dim=8)])
    cfg = {"name": "tiny", "family": family, "action_horizon": H, "action_dim": 8, "output_semantics": "offsets_from_chunk_start_state"}
    return OpenPIPolicy(pol, types.SimpleNamespace(freeze_filter=base.get_freeze_filter()), cfg, seed=3), base


def env_obs(n, seed=0):
    rng = np.random.default_rng(seed)
    return {P.EXTERIOR_IMAGE: rng.integers(0, 255, (n, 224, 224, 3), dtype=np.uint8), P.WRIST_IMAGE: rng.integers(0, 255, (n, 224, 224, 3), dtype=np.uint8),
            P.JOINT_POSITION: (np.arange(n)[:, None] + 1 + 0.1 * np.arange(7)[None]).astype(np.float32), P.GRIPPER_POSITION: np.full((n, 1), 0.25, np.float32)}


@pytest.fixture(scope="module")
def pol():
    return build()[0]


def test_the_batched_path_gives_the_chunk_of_openpis_own_policy_for_the_same_key(pol):
    """N = 1, S = 1: the same transforms, the same sampler, the same random key: the same chunk."""
    from scripts import probe_vla as probe
    row = probe.to_openpi_obs(env_obs(1), 0, "insert the peg")
    check = probe.interface_check(pol, row, seed=7)
    assert check["shape_reference"] == check["shape_batched"] == [H, 8]
    assert check["max_abs_diff"] < 1e-5, check


def test_every_environment_gets_its_own_row_and_the_samples_are_independent(pol):
    plan = pol.plan(env_obs(3), "insert the peg", num_samples=2)               # 3 environments -> padded to 4, 2 samples each
    assert plan.targets.shape == (3, 2, H, P.ACTION_DIM) and plan.raw.shape == (3, 2, H, 8) and plan.infer_ms > 0
    assert not np.allclose(plan.raw[:, 0], plan.raw[:, 1])                       # different noise: different chunks
    q = env_obs(3)[P.JOINT_POSITION]
    assert np.allclose(plan.targets[:, :, :, :7], plan.raw[:, :, :, :7] + q[:, None, None, :])      # the state of THAT environment, once


def test_the_state_of_each_row_follows_the_environment_in_both_layouts(pol):
    """Order and padding of the batched path, without running the model (a stub sampler: the compilation is the slow part).
    Native samples (pi0.5 / pi0: S inside the model) and repeated rows (pi0-FAST): environment i, sample s <- the state of environment i."""
    seen = []

    def stub_sampler(state, rng, observation, num_samples):
        seen.append((observation.state.shape[0], num_samples))
        return jnp.zeros((observation.state.shape[0] * num_samples, H, MODEL_DIM))

    shell = types.SimpleNamespace(_model=pol.policy._model, _sample_kwargs={}, _input_transform=pol.policy._input_transform,     # noqa: SLF001
                                  _output_transform=transforms.compose([StateTag()]))
    for family, expected_call in (("pi05", (4, 2)), ("pi0_fast", (8, 1))):          # 3 environments x 2 samples
        tagged = OpenPIPolicy(shell, pol.train_cfg, {**pol.cfg, "family": family})
        tagged._sample = stub_sampler                                                  # noqa: SLF001
        plan = tagged.plan(env_obs(3), "p", num_samples=2)
        assert seen[-1] == expected_call, (family, seen)                              # padded to a power of two, samples inside or as rows
        want = np.repeat(env_obs(3)[P.JOINT_POSITION][:, 0], 2).reshape(3, 2)        # the first joint of environment i, for both samples
        assert plan.raw.shape == (3, 2, H, 8) and np.allclose(plan.raw[:, :, 0, 0], want), (family, plan.raw[:, :, 0, 0])


def test_set_state_changes_what_is_sampled_without_rebuilding_the_policy(pol):
    """openpi's Policy freezes the weights into its jitted sampler; this one takes them as an argument."""
    obs = env_obs(1)
    pol.seed(5)
    before = pol.plan(obs, "p").raw
    pure = pol.state.to_pure_dict()
    flat = tu.flatten_dict(pure)
    zeroed = {k: (jnp.zeros_like(v) if "action_out_proj" in k else v) for k, v in flat.items()}
    new = nnx.state(nnx.merge(pol.graphdef, pol.state))
    new.replace_by_pure_dict(tu.unflatten_dict(zeroed))
    pol.set_state(new)
    pol.seed(5)
    after = pol.plan(obs, "p").raw
    assert np.abs(before - after).max() > 1e-4
    with pytest.raises(ValueError, match="structure"):
        pol.set_state({"not": "a state"})


def test_the_training_batch_puts_the_targets_in_the_models_own_space_and_pads_them(pol):
    obs = env_obs(2)
    target = np.random.default_rng(1).normal(size=(2, H, 8)).astype(np.float32)
    observation, actions = pol.training_batch(obs, "p", target)
    assert actions.shape == (2, H, MODEL_DIM) and observation.state.shape == (2, MODEL_DIM)
    q = obs[P.JOINT_POSITION]
    assert np.allclose(np.asarray(actions)[:, :, :7], target[:, :, :7] - q[:, None, :], atol=1e-6)      # offsets from the chunk-start state
    assert np.allclose(np.asarray(actions)[:, :, 7], target[:, :, 7]) and not np.asarray(actions)[:, :, 8:].any()
    with pytest.raises(ValueError, match="target_actions"):
        pol.training_batch(obs, "p", target[:, :-1])
    loss = pol.loss(pol.state, jax.random.key(0), observation, actions)
    assert loss.shape == () and np.isfinite(float(loss))


def test_the_trainable_filter_selects_parameters(pol):
    trainable = nnx.state(pol.graphdef and nnx.merge(pol.graphdef, pol.state), pol.trainable_filter)
    assert len(jax.tree.leaves(trainable)) > 0


def test_touch_is_off_unless_the_yaml_switches_it_on(pol):
    assert pol.touch and not any(spec.enabled for spec in pol.touch.values()) and pol.settings.required_obs_keys == ()
    assert set(pol.injections.values()) == {None}


def test_a_component_that_asks_for_touch_stops_until_an_injection_exists(pol):
    """Prepared, not silent: use_tactile on the action head parses, and the model refuses to be built without an injection for it."""
    with pytest.raises(NotImplementedError, match="action_head"):
        OpenPIPolicy(pol.policy, pol.train_cfg, {**pol.cfg, "action_head": {"use_tactile": True}})
    with pytest.raises(NotImplementedError, match="vlm"):
        OpenPIPolicy(pol.policy, pol.train_cfg, {**pol.cfg, "vlm": {"use_force": True}})


def test_with_an_injection_registered_the_touch_data_path_is_complete(pol, monkeypatch):
    """What adding a touch method will not have to change: the vectors reach the rows of `plan` and of `training_batch`, per component."""
    from src.touch import injection
    monkeypatch.setitem(injection.INJECTIONS, "action_head", lambda spec: ("stub", spec.dim))
    touched = OpenPIPolicy(pol.policy, pol.train_cfg, {**pol.cfg, "action_head": {"use_tactile": True, "use_force": True}}, seed=3)
    assert touched.injections == {"vlm": None, "action_head": ("stub", 60)} and touched.settings.required_obs_keys == (P.FT_FORCE, P.GRIPPER_EFFORT, P.TACTILE)
    obs = env_obs(2)
    with pytest.raises(KeyError, match="tactile|ft_force"):                                    # the env stream must carry the signals
        touched.plan(obs, "p")
    obs.update({P.FT_FORCE: np.ones((2, 3), np.float32), P.GRIPPER_EFFORT: np.ones(2, np.float32), P.TACTILE: np.ones((2, 2, 7, 4), np.float32)})
    from src.models.interface import rows_of
    assert rows_of(obs, "p", touched.touch)[1]["touch/action_head"].shape == (60,)
    plan = touched.plan(obs, "p")                                                              # (the stub injection changes nothing: it is not applied)
    assert plan.targets.shape == (2, 1, H, P.ACTION_DIM)
    target = np.zeros((2, H, 8), np.float32)
    observation, actions = touched.training_batch(obs, "p", target)
    assert actions.shape == (2, H, MODEL_DIM)


def test_model_info_comes_from_the_yaml(pol):
    assert pol.info == ModelInfo(name="tiny", family="pi05", action_horizon=H, action_dim=8, output_semantics="offsets_from_chunk_start_state")
