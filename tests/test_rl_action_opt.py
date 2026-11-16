"""src/algos/rl_action_opt.py on CPU: first on a toy critic whose answers are known in closed form (global top-m, the local ascent and its acceptance rule, pi^Opt), then on the real network at small size."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
import jax.numpy as jnp  # noqa: E402

from src.algos import rl_action_opt as ao  # noqa: E402
from src.algos import rl_critic as rc  # noqa: E402
from src.algos.rl_config import RLConfig  # noqa: E402

H = 2


def toy(keep_top, local_steps, local_lr):
    """Q(s, a) = -c * ||a - t||^2 with the target t and the curvature c of the state: grad = 2 c (t - a), so a step maps a - t to (1 - 2 alpha c) (a - t)."""
    return ao.ActionOptimiser(lambda params, state: state, lambda params, z, a: -z["c"][:, None] * ((a - z["t"][:, None]) ** 2).sum(axis=(-2, -1)), keep_top, local_steps, local_lr)


def toy_state(c, t=0.0):
    c = np.asarray(c, np.float32)
    return {"c": jnp.asarray(c), "t": jnp.full((len(c), H, 8), t, jnp.float32)}


def candidates_at(distances, b=1):
    """(b, k, H, 8) chunks that are all equal to `d` (so ||a - 0||^2 = 16 d^2): Q is -c * 16 * d^2."""
    d = np.asarray(distances, np.float32)
    return jnp.asarray(np.broadcast_to(d[None, :, None, None], (b, len(d), H, 8)).copy())


def test_global_keeps_the_m_best_candidates_by_q_and_remembers_where_they_came_from():
    cands = candidates_at([3.0, 0.5, 2.0, 0.1, 1.0])                                            # Q order: 0.1 > 0.5 > 1.0 > 2.0 > 3.0
    result = toy(3, 0, 1e-3)(None, toy_state([1.0]), cands)
    assert result.source.tolist() == [[3, 1, 4]] and np.allclose(result.actions[0, :, 0, 0], [0.1, 0.5, 1.0])
    assert np.allclose(result.q, -16 * np.array([[0.01, 0.25, 1.0]])) and np.array_equal(result.q, result.q_start) and result.accepted.sum() == 0 and float(result.displacement.sum()) == 0
    assert np.allclose(result.candidate_q[0], -16 * np.array([9, 0.25, 4, 0.01, 1]))


def test_best_of_n_is_the_argmax_of_the_k_candidates():
    cands = candidates_at([3.0, 0.5, 2.0, 0.1, 1.0], b=2)
    result = toy(1, 0, 1e-3)(None, toy_state([1.0, 2.0]), cands)
    chosen = ao.pick(result, "argmax")
    assert chosen.source.tolist() == [3, 3] and np.allclose(chosen.actions[:, 0, 0], 0.1) and chosen.actions.shape == (2, H, 8) and np.allclose(chosen.q, [-0.16, -0.32])


def test_the_local_ascent_follows_the_gradient_step_by_step():
    alpha, steps, d0 = 0.05, 10, 1.0
    result = toy(1, steps, alpha)(None, toy_state([1.0]), candidates_at([d0]))
    expected = d0 * (1 - 2 * alpha) ** steps                                                     # every step is accepted: Q rises while the chunk moves towards the target
    assert np.allclose(result.actions[0, 0], expected, atol=1e-5) and int(result.accepted[0, 0]) == steps
    assert np.allclose(result.q[0, 0], -16 * expected ** 2, rtol=1e-4) and np.allclose(result.q_start[0, 0], -16 * d0 ** 2)
    assert np.allclose(result.displacement[0, 0], 16 * (d0 - expected), rtol=1e-4)                # L1 over the 2 x 8 entries


def test_a_step_that_does_not_raise_q_is_refused_and_the_chunk_stays():
    cands = candidates_at([1.0])
    for alpha in (1.0, 1.5):                                                                     # factor (1 - 2 alpha) = -1 (same Q: not strictly higher) and -2 (lower Q)
        result = toy(1, 5, alpha)(None, toy_state([1.0]), cands)
        assert int(result.accepted[0, 0]) == 0 and np.allclose(result.actions, 1.0) and np.allclose(result.q, result.q_start)
    overshoot = toy(1, 5, 0.75)(None, toy_state([1.0]), cands)                                    # factor -0.5: the chunk jumps over the target but Q rises: kept, and again
    assert int(overshoot.accepted[0, 0]) == 5 and np.allclose(overshoot.actions, (-0.5) ** 5, atol=1e-5)


def test_each_chunk_is_judged_on_its_own_in_a_batch():
    c = [0.25, 2.0]                                                                              # alpha = 1: factor 0.5 (accepted) for the first state, -3 (refused) for the second
    result = toy(2, 4, 1.0)(None, toy_state(c, 0.0), candidates_at([1.0, 2.0], b=2)[:, :2])
    assert result.accepted.tolist() == [[4, 4], [0, 0]]                                          # (B, m): the batch does not couple its states
    assert np.allclose(result.actions[0, :, 0, 0], np.array([1.0, 2.0]) * 0.5 ** 4, atol=1e-5) and np.allclose(result.actions[1, :, 0, 0], [1.0, 2.0])


def test_a_non_finite_q_is_never_accepted():
    opt = ao.ActionOptimiser(lambda p, s: s, lambda p, z, a: -((a - 1.0) ** 2).sum(axis=(-2, -1)) + jnp.where(a.mean(axis=(-2, -1)) > 0.5, jnp.nan, 0.0), 1, 3, 0.05)
    result = opt(None, {"x": jnp.zeros((1, 1))}, candidates_at([0.45]))                          # the first step would cross 0.5 where Q is NaN
    assert int(result.accepted[0, 0]) == 0 and bool(jnp.isfinite(result.actions).all()) and np.allclose(result.actions, 0.45)


def test_softmax_samples_in_proportion_to_exp_q_and_argmax_does_not_depend_on_the_rng():
    q = jnp.log(jnp.asarray([[0.1, 0.2, 0.7]]))                                                  # softmax(q) = [0.1, 0.2, 0.7]
    fake = ao.Optimised(actions=jnp.zeros((1, 3, H, 8)), q=q, q_start=q, source=jnp.asarray([[4, 5, 6]]), candidate_q=q, accepted=jnp.zeros((1, 3), jnp.int32), displacement=jnp.zeros((1, 3)))
    draws = np.array([int(ao.pick(fake, "softmax", jax.random.key(i)).index[0]) for i in range(1500)])
    assert np.allclose(np.bincount(draws, minlength=3) / 1500, [0.1, 0.2, 0.7], atol=0.04)
    cold = np.array([int(ao.pick(fake, "softmax", jax.random.key(i), 0.01).index[0]) for i in range(50)])
    assert set(cold) == {2}                                                                       # temperature -> 0: the argmax
    assert {int(ao.pick(fake, "argmax", jax.random.key(i)).source[0]) for i in range(5)} == {6}
    with pytest.raises(ValueError, match="rng"):
        ao.pick(fake, "softmax")
    with pytest.raises(ValueError, match="selection"):
        ao.pick(fake, "greedy")


def test_wrong_inputs_are_refused():
    with pytest.raises(ValueError, match="keep_top"):
        toy(0, 0, 1e-3)
    with pytest.raises(ValueError, match="exceed"):
        toy(6, 0, 1e-3)(None, toy_state([1.0]), candidates_at([1.0, 2.0]))
    with pytest.raises(ValueError, match="B, k, H, 8"):
        toy(1, 0, 1e-3)(None, toy_state([1.0]), jnp.zeros((2, H, 8)))


def test_the_summary_reports_the_spread_of_q_the_local_gain_and_whether_the_vla_s_first_sample_was_replaced():
    opt = toy(2, 3, 0.05)
    cands = candidates_at([0.1, 3.0, 1.0, 2.0])                                                  # the first sample is the best one
    result = opt(None, toy_state([1.0]), cands)
    logs = opt.summarise(result, ao.pick(result, "argmax"), cands)
    assert np.isclose(logs["candidate_q_std_mean"], float(np.std(-16 * np.array([0.01, 9, 1, 4]))), rtol=1e-4)
    assert logs["local_accepted_share"] == 1.0 and logs["local_q_gain_mean"] > 0 and logs["local_displacement_l1_mean"] > 0
    assert logs["chosen_differs_from_first_sample_share"] == 1.0                                  # the best candidate is the first one, but the local steps moved it
    none = toy(1, 0, 0.05)
    plain = none(None, toy_state([1.0]), cands)
    plain_logs = none.summarise(plain, ao.pick(plain, "argmax"), cands)
    assert plain_logs["chosen_differs_from_first_sample_share"] == 0.0 and "local_q_gain_mean" not in plain_logs
    assert plain_logs["chosen_shift_l1_mean"] == 0.0 and logs["chosen_shift_l1_mean"] > 0                      # the first sample itself / moved by the local steps
    q = -16 * np.array([0.01, 9, 1, 4])
    assert np.isclose(plain_logs["chosen_q_gain_mean"], q.max() - q.mean(), rtol=1e-4)                       # best candidate minus the mean one


# ---- the real critic (small images), the same code path as the runs ----

B, K, HR = 3, 6, 4


@pytest.fixture(scope="module")
def real():
    cfg = RLConfig(num_candidates=K, keep_top=3, local_steps=5, local_lr=1e-2)
    net = rc.CriticNet(image_size=32, hidden=(16, 16, 16), shift_px=2)
    critic = rc.init_critic(net, jax.random.key(0), horizon=HR)
    rng = np.random.default_rng(0)
    state = {c: jnp.asarray(rng.integers(0, 255, (B, 224, 224, 3), dtype=np.uint8)) for c in net.cameras}
    state["proprio"] = jnp.asarray(rng.normal(size=(B, 8)).astype(np.float32))
    cands = jnp.asarray(rng.normal(size=(B, K, HR, 8)).astype(np.float32))
    return cfg, net, critic, state, cands


def test_the_real_critic_global_ranking_is_the_sort_of_its_q_values(real):
    cfg, net, critic, state, cands = real
    result = ao.ActionOptimiser.for_critic(net, cfg)(critic.params, state, cands)
    q = np.asarray(rc.q_values(net, critic.params, state, cands))
    assert np.allclose(result.candidate_q, q, atol=1e-4) and np.array_equal(np.asarray(result.source), np.argsort(-q, axis=1)[:, :3])
    assert result.actions.shape == (B, 3, HR, 8) and result.q.shape == (B, 3)


def test_the_real_critic_local_ascent_never_lowers_q_and_q_of_the_result_is_the_critic_s_q(real):
    cfg, net, critic, state, cands = real
    result = ao.ActionOptimiser.for_critic(net, cfg)(critic.params, state, cands)
    assert bool((result.q >= result.q_start).all()) and float((result.q - result.q_start).sum()) > 0 and int(result.accepted.sum()) > 0
    assert np.allclose(result.q, rc.q_values(net, critic.params, state, result.actions), atol=1e-4)  # the stored Q is what the critic says of the optimised chunks
    moved = np.abs(np.asarray(result.actions) - np.take_along_axis(np.asarray(cands), np.asarray(result.source)[:, :, None, None], axis=1)).sum(axis=(-2, -1))
    assert np.allclose(moved, result.displacement, rtol=1e-4, atol=1e-5)


def test_a_batch_of_one_state_gives_the_same_answer_as_inside_a_batch(real):
    cfg, net, critic, state, cands = real
    opt = ao.ActionOptimiser.for_critic(net, cfg)
    full = opt(critic.params, state, cands)
    alone = opt(critic.params, {k: v[1:2] for k, v in state.items()}, cands[1:2])
    assert np.array_equal(alone.source, full.source[1:2]) and np.allclose(alone.actions, full.actions[1:2], atol=1e-4) and np.allclose(alone.q, full.q[1:2], atol=1e-4)


def test_the_state_is_encoded_once_whatever_the_number_of_local_steps():
    calls = []

    def encode(params, state):
        calls.append(1)
        return state

    opt = ao.ActionOptimiser(encode, lambda params, z, a: -z["c"][:, None] * ((a - z["t"][:, None]) ** 2).sum(axis=(-2, -1)), 2, 10, 0.05)
    opt(None, toy_state([1.0]), candidates_at([1.0, 2.0, 3.0]))
    assert len(calls) == 1                                                                         # traced once: the ascent's 11 evaluations only run `score` (the MLP)
