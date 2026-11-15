"""src/algos/rl_critic.py on CPU with small images (the 128 x 128 network of the YAML is the same code with a larger `image_size`): shapes, the action at every layer, N chunks per state sharing one
encoding, the random shift (training only), the target network."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
import jax.numpy as jnp  # noqa: E402

from src.algos import rl_critic as rc  # noqa: E402
from src.algos.rl_config import RLConfig  # noqa: E402

H, B, N = 4, 3, 5


@pytest.fixture(scope="module")
def setup():
    net = rc.CriticNet(image_size=32, hidden=(16, 16, 16), shift_px=2)
    state = rc.init_critic(net, jax.random.key(0), horizon=H)
    rng = np.random.default_rng(0)
    obs = {c: jnp.asarray(rng.integers(0, 255, (B, 224, 224, 3), dtype=np.uint8)) for c in net.cameras}
    obs["proprio"] = jnp.asarray(rng.normal(size=(B, 8)).astype(np.float32))
    actions = jnp.asarray(rng.normal(size=(B, N, H, 8)).astype(np.float32))
    return net, state, obs, actions


def test_q_has_one_value_per_state_and_chunk_and_the_target_starts_equal(setup):
    net, state, obs, actions = setup
    q = rc.q_values(net, state.params, obs, actions)
    assert q.shape == (B, N) and bool(jnp.isfinite(q).all())
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(state.target_params)))
    assert rc.num_parameters(state.params) > 0


def test_the_action_is_concatenated_to_the_input_of_every_layer(setup):
    net, state, *_ = setup
    state_features = len(net.cameras) * rc.ENCODER_FEATURES + rc.PROPRIO_DIM
    a = H * 8
    p = state.params
    assert p["dense_0"]["kernel"].shape == (state_features + a, 16) and p["dense_1"]["kernel"].shape == (16 + a, 16) and p["dense_2"]["kernel"].shape == (16 + a, 16)
    assert p["out"]["kernel"].shape == (16 + a, 1)


def test_n_chunks_of_a_state_share_its_encoding_and_give_the_same_q_as_one_at_a_time(setup):
    net, state, obs, actions = setup
    together = rc.q_values(net, state.params, obs, actions)
    for n in (0, 3):
        alone = rc.q_values(net, state.params, obs, actions[:, n:n + 1])
        assert np.allclose(together[:, n], alone[:, 0], atol=1e-4)
    flat = rc.q_values(net, state.params, obs, actions.reshape(B, N, -1))
    assert np.allclose(flat, together)                                                           # (B, N, H * 8) and (B, N, H, 8) are the same input


def test_q_depends_on_the_action_and_its_gradient_is_finite(setup):
    net, state, obs, actions = setup
    grad = jax.grad(lambda a: rc.q_values(net, state.params, obs, a).sum())(actions)
    assert grad.shape == actions.shape and bool(jnp.isfinite(grad).all()) and float(jnp.abs(grad).max()) > 0
    other = rc.q_values(net, state.params, obs, actions + 1.0)
    assert not np.allclose(other, rc.q_values(net, state.params, obs, actions))


def test_the_random_shift_acts_in_training_only(setup):
    net, state, obs, actions = setup
    eval_q = rc.q_values(net, state.params, obs, actions)
    assert np.allclose(eval_q, rc.q_values(net, state.params, obs, actions))              # no randomness at evaluation
    t1 = rc.q_values(net, state.params, obs, actions, train=True, rng=jax.random.key(1))
    t2 = rc.q_values(net, state.params, obs, actions, train=True, rng=jax.random.key(2))
    assert not np.allclose(t1, t2) and not np.allclose(t1, eval_q)
    assert np.allclose(t1, rc.q_values(net, state.params, obs, actions, train=True, rng=jax.random.key(1)))     # the same key, the same shift
    with pytest.raises(ValueError, match="rng"):
        rc.q_values(net, state.params, obs, actions, train=True)
    still = rc.CriticNet(image_size=32, hidden=(16, 16, 16), shift_px=0)
    p0 = rc.init_critic(still, jax.random.key(0), horizon=H).params
    assert np.allclose(rc.q_values(still, p0, obs, actions, train=True), rc.q_values(still, p0, obs, actions))      # no shift configured: training = evaluation, no rng needed


def test_random_shift_moves_each_image_by_at_most_pad_pixels_with_replicated_edges():
    ramp = jnp.broadcast_to(jnp.arange(10, dtype=jnp.float32)[None, None, :, None], (6, 10, 10, 3))       # the value is the column
    out = np.asarray(rc.random_shift(jax.random.key(3), ramp, 2))
    assert out.shape == ramp.shape
    allowed = [np.clip(np.arange(10) + o - 2, 0, 9).tolist() for o in range(5)]                   # offset o of the crop in the padded image: column j shows original column clip(j + o - 2)
    for img in out:
        assert img[0, :, 0].tolist() in allowed
    assert np.array_equal(np.asarray(rc.random_shift(jax.random.key(3), ramp, 0)), np.asarray(ramp))
    assert len({tuple(np.asarray(img[:, 0, 0])) for img in out}) > 1                                  # not the same shift for every image of the batch


def test_preprocess_scales_to_minus_one_one_and_resizes():
    x = jnp.asarray(np.stack([np.zeros((224, 224, 3)), np.full((224, 224, 3), 255)]).astype(np.uint8))
    out = rc.preprocess(x, 32)
    assert out.shape == (2, 32, 32, 3) and np.allclose(out[0], -1.0, atol=1e-5) and np.allclose(out[1], 1.0, atol=1e-5)
    assert rc.preprocess(x, 224).shape == (2, 224, 224, 3)


def test_soft_update_moves_the_target_towards_the_parameters_and_leaves_them_alone(setup):
    net, state, *_ = setup
    moved = rc.CriticState(params=jax.tree.map(lambda x: x + 1.0, state.params), target_params=state.target_params)
    half = rc.soft_update(moved, 0.5)
    assert all(np.allclose(t, p + 0.5) for t, p in zip(jax.tree.leaves(half.target_params), jax.tree.leaves(state.params)))
    assert all(np.allclose(a, b) for a, b in zip(jax.tree.leaves(half.params), jax.tree.leaves(moved.params)))
    full = rc.soft_update(moved, 1.0)
    assert all(np.allclose(t, p) for t, p in zip(jax.tree.leaves(full.target_params), jax.tree.leaves(moved.params)))
    with pytest.raises(ValueError, match="tau"):
        rc.soft_update(moved, 0.0)


def test_one_camera_and_the_config_builds_the_network():
    net = rc.CriticNet(cameras=("wrist",), image_size=32, hidden=(8,), shift_px=0)
    state = rc.init_critic(net, jax.random.key(0), horizon=H)
    assert "encoder_wrist" in state.params and "encoder_exterior" not in state.params
    assert state.params["dense_0"]["kernel"].shape == (rc.ENCODER_FEATURES + rc.PROPRIO_DIM + H * 8, 8)
    cfg = RLConfig(critic_cameras=("exterior",), critic_hidden=(64, 64), critic_image_size=64, random_shift_px=3)
    built = rc.CriticNet.from_config(cfg)
    assert (built.cameras, built.hidden, built.image_size, built.shift_px) == (("exterior",), (64, 64), 64, 3)


def test_the_network_runs_under_jit(setup):
    net, state, obs, actions = setup
    f = jax.jit(lambda p, o, a: rc.q_values(net, p, o, a))
    assert np.allclose(f(state.params, obs, actions), rc.q_values(net, state.params, obs, actions), atol=1e-4)


def test_with_touch_the_vector_joins_the_state_feature_and_changes_q_and_a_mismatch_is_refused(setup):
    _, _, obs, actions = setup
    net = rc.CriticNet(image_size=32, hidden=(16, 16), shift_px=0, touch_dim=60)
    state = rc.init_critic(net, jax.random.key(0), horizon=H)
    assert state.params["dense_0"]["kernel"].shape == (2 * rc.ENCODER_FEATURES + rc.PROPRIO_DIM + 60 + H * 8, 16)
    touched = {**obs, "touch": jnp.asarray(np.random.default_rng(1).normal(size=(B, 60)).astype(np.float32))}
    q = rc.q_values(net, state.params, touched, actions)
    assert q.shape == (B, N) and not np.allclose(q, rc.q_values(net, state.params, {**touched, "touch": touched["touch"] + 1.0}, actions))
    with pytest.raises(ValueError, match="sees touch"):
        rc.q_values(net, state.params, obs, actions)                                                  # the critic needs touch and the state has none
    with pytest.raises(ValueError, match="sees touch"):
        rc.q_values(net, state.params, {**obs, "touch": jnp.zeros((B, 59))}, actions)
    plain = rc.CriticNet(image_size=32, hidden=(16,), shift_px=0)
    with pytest.raises(ValueError, match="without it"):
        rc.q_values(plain, rc.init_critic(plain, jax.random.key(0), horizon=H).params, touched, actions)
    from_cfg = rc.CriticNet.from_config(RLConfig(critic_use_force=True, critic_use_tactile=True))
    assert from_cfg.touch_dim == 60


def test_the_encoder_half_and_the_mlp_half_run_apart_and_give_q_values(setup):
    net, state, obs, actions = setup
    z = rc.encode_state(net, state.params, obs)
    assert z.shape == (B, len(net.cameras) * rc.ENCODER_FEATURES + rc.PROPRIO_DIM)
    assert np.array_equal(rc.score(net, state.params, z, actions), rc.q_values(net, state.params, obs, actions))
