"""src/touch/features.py and src/touch/injection: the touch switches, the vector built from the observation, the refusals. numpy only."""
import numpy as np
import pytest

from src.envs import protocol as P
from src.touch.features import FORCE_DIM, TACTILE_DIM, TouchSpec, required_keys
from src.touch.injection import INJECTIONS, build_injection


def obs_of(n=3, with_touch=True):
    rng = np.random.default_rng(0)
    o = {P.JOINT_POSITION: np.zeros((n, 7), np.float32)}
    if with_touch:
        o[P.FT_FORCE] = rng.normal(size=(n, 3)).astype(np.float32)
        o[P.GRIPPER_EFFORT] = rng.normal(size=(n,)).astype(np.float32)
        o[P.TACTILE] = np.abs(rng.normal(size=(n, 2, 7, 4))).astype(np.float32)
    return o


# -- the switches ------------------------------------------------------------------------------------------------------------------
def test_everything_is_off_by_default():
    spec = TouchSpec()
    assert not spec.enabled and spec.dim == 0 and spec.keys == () and spec.describe() == "none (task only)"
    assert TouchSpec.from_mapping(None, "vlm") == spec == TouchSpec.from_mapping({}, "vlm")
    assert required_keys({"vlm": spec}) == ()


def test_dimensions_and_keys_follow_the_switches():
    assert (FORCE_DIM, TACTILE_DIM) == (4, 56)
    force, tactile = TouchSpec(use_force=True), TouchSpec(use_tactile=True)
    both = TouchSpec(use_tactile=True, use_force=True)
    assert (force.dim, tactile.dim, both.dim) == (4, 56, 60)
    assert force.keys == (P.FT_FORCE, P.GRIPPER_EFFORT) and tactile.keys == (P.TACTILE,) and both.keys == (P.FT_FORCE, P.GRIPPER_EFFORT, P.TACTILE)
    assert required_keys({"vlm": force, "action_head": both}) == (P.FT_FORCE, P.GRIPPER_EFFORT, P.TACTILE)      # each key once


def test_the_comparison_ignores_the_yaml_path():
    assert TouchSpec(use_tactile=True, where="vlm") == TouchSpec(use_tactile=True, where="residual")
    assert len({TouchSpec(use_tactile=True, where="a"), TouchSpec(use_tactile=True, where="b")}) == 1


@pytest.mark.parametrize("block", [{"use_tactile": "false"}, {"use_force": 1}, {"use_tactile": None}])
def test_a_switch_must_be_a_real_boolean(block):
    with pytest.raises(ValueError, match="true or false"):                 # the string "false" would be truthy
        TouchSpec.from_mapping(block, "action_head")


def test_force_mode_is_validated_and_the_proxy_is_routed_but_not_implemented():
    with pytest.raises(ValueError, match="force_mode"):
        TouchSpec.from_mapping({"use_force": True, "force_mode": "guess"}, "vlm")
    with pytest.raises(ValueError, match="no effect"):
        TouchSpec.from_mapping({"force_mode": "proxy"}, "vlm")
    with pytest.raises(NotImplementedError, match="proxy"):
        TouchSpec.from_mapping({"use_force": True, "force_mode": "proxy"}, "vlm")


def test_errors_name_the_yaml_block():
    with pytest.raises(ValueError, match=r"action_head\.use_tactile"):
        TouchSpec.from_mapping({"use_tactile": "yes"}, "action_head")


# -- the vector -------------------------------------------------------------------------------------------------------------------
def test_the_vector_has_the_documented_layout_and_scaling():
    o = obs_of(3)
    got = TouchSpec(use_tactile=True, use_force=True).build_batch(o)
    assert got.shape == (3, 60) and got.dtype == np.float32
    raw = np.concatenate([o[P.FT_FORCE], o[P.GRIPPER_EFFORT].reshape(3, 1), o[P.TACTILE].reshape(3, 56)], axis=1)
    assert np.allclose(got, np.sign(raw) * np.log1p(np.abs(raw)), atol=1e-6)            # [ft_force | gripper_effort | tactile], sign * log1p
    assert np.allclose(TouchSpec(use_tactile=True).build_batch(o), got[:, 4:])
    assert np.allclose(TouchSpec(use_force=True).build_batch(o), got[:, :4])


def test_the_scaling_keeps_the_sign_and_zero():
    o = obs_of(1)
    o[P.FT_FORCE] = np.array([[-3.0, 0.0, 100.0]], np.float32)
    got = TouchSpec(use_force=True).build_batch(o)
    assert got[0, 0] < 0 and got[0, 1] == 0 and got[0, 2] == pytest.approx(np.log1p(100.0), rel=1e-6)


def test_a_missing_signal_raises_instead_of_giving_a_silent_zero():
    with pytest.raises(KeyError, match="robotiq_2f85_tsf"):
        TouchSpec(use_tactile=True).build_batch(obs_of(2, with_touch=False))
    o = obs_of(2)
    del o[P.GRIPPER_EFFORT]
    with pytest.raises(KeyError, match="gripper_effort"):
        TouchSpec(use_force=True).build_batch(o)


def test_a_wrong_shape_or_a_non_finite_value_raises():
    o = obs_of(2)
    o[P.TACTILE] = o[P.TACTILE][:, :1]
    with pytest.raises(ValueError, match="shape"):
        TouchSpec(use_tactile=True).build_batch(o)
    o = obs_of(2)
    o[P.FT_FORCE][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        TouchSpec(use_force=True).build_batch(o)


def test_building_a_disabled_spec_is_a_bug():
    with pytest.raises(RuntimeError):
        TouchSpec().build_batch(obs_of(1))


# -- the injection ----------------------------------------------------------------------------------------------------------------
def test_no_injection_is_needed_while_touch_is_off():
    assert build_injection("vlm", TouchSpec()) is None


def test_asking_for_touch_where_no_injection_exists_stops_with_an_error_that_says_so(monkeypatch):
    with pytest.raises(NotImplementedError, match="no injection"):
        build_injection("action_head", TouchSpec(use_tactile=True))
    monkeypatch.setitem(INJECTIONS, "action_head", lambda spec: ("injection", spec.dim))         # what registering a method looks like
    assert build_injection("action_head", TouchSpec(use_tactile=True)) == ("injection", 56)
    with pytest.raises(NotImplementedError):
        build_injection("vlm", TouchSpec(use_force=True))                                     # per component
