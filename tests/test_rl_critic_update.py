"""src/algos/rl_critic_update.py on CPU with a tiny critic: the batch (candidates of s and s', terminal / time-out), the loss and the target computed by hand, the soft update of the target
copy, and a critic that learns the values of a small chain of states."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
import jax.numpy as jnp  # noqa: E402

from src.algos import rl_critic as rc  # noqa: E402
from src.algos import rl_critic_update as cu  # noqa: E402
from src.algos.rl_action_opt import ActionOptimiser, pick  # noqa: E402
from src.algos.rl_buffer import Episode, ReplayBuffer  # noqa: E402
from src.algos.rl_candidates import CandidateCache  # noqa: E402
from src.algos.rl_config import load_rl_config  # noqa: E402
from src.models.interface import Candidates  # noqa: E402

H, K = 2, 4
SMALL = {"num_candidates": K, "critic_hidden": [16, 16, 16], "critic_image_size": 32, "random_shift_px": 0, "critic_batch_size": 8, "mixing_ratio": 0.5, "gamma": 0.9, "tau": 0.1,
         "critic_lr": 1e-3, "cal_ql_alpha": 0.01}
BEST_OF_N = {"method": "best_of_n", "keep_top": 1, "local_steps": 0, "rollout_selection": "argmax"}
PARL = {"method": "parl", "keep_top": 2, "local_steps": 3, "local_lr": 1e-2}


def cfg_of(extra, **more):
    return load_rl_config(overrides={**SMALL, **extra, **more})


def episode(plans, success, tag=0.0, progress=False, seed=0):
    rng = np.random.default_rng(seed)
    joints = np.full((plans + 1, 7), tag, np.float32)
    if progress:
        joints[:, 0] = np.arange(plans + 1)                      # the chain's clock: the state tells which plan it is
    else:
        joints += rng.normal(size=joints.shape).astype(np.float32)
    zeros = np.zeros((plans + 1, 224, 224, 3), np.uint8)
    return Episode(exterior=zeros, wrist=zeros, joints=joints, gripper=np.zeros((plans + 1, 1), np.float32), actions=np.zeros((plans, H, 8), np.float32) if progress else rng.normal(size=(plans, H, 8)).astype(np.float32), success=success, prompt="p")


def filled(episodes, warmups=None, spread=1.0):
    buf = ReplayBuffer(200, H, 0.9)
    for i, ep in enumerate(episodes):
        buf.add_episode(ep, warmup=bool(warmups[i]) if warmups else False)
    cache = CandidateCache(K, H, 3, capacity=8)
    ids = buf.candidate_ids()
    rng = np.random.default_rng(1)
    cands = (spread * rng.normal(size=(len(ids), K, H, 8))).astype(np.float32)
    cache.put(ids, Candidates(normalized=cands, states=np.zeros((len(ids), 3), np.float32), targets=cands, infer_ms=0.0), version=0)
    return buf, cache


def trainer_of(extra, cameras=("exterior", "wrist")):
    cfg = cfg_of(extra)
    net = rc.CriticNet(cameras=cameras, hidden=tuple(cfg.critic_hidden), image_size=cfg.critic_image_size, shift_px=0)
    return cu.CriticTrainer(net, cfg, horizon=H, seed=0), cfg


def test_a_batch_carries_the_candidates_of_s_and_of_s_prime_and_borrows_s_for_a_terminal_transition():
    buf, cache = filled([episode(3, True), episode(2, False)])                      # rows 0-3 (success), rows 4-6 (time-out: row 6 has candidates)
    cfg = cfg_of(BEST_OF_N)
    batch = cu.make_batch(buf, cache, np.random.default_rng(0), cfg, batch_size=64)
    ids, nxt = batch["ids"], batch["next_cand"]
    assert batch["cand"].shape == (64, K, H, 8) and np.array_equal(batch["cand"], cache.normalized[ids])
    terminal = batch["terminal"]
    assert terminal.any() and (~terminal).any()
    assert np.array_equal(nxt[terminal], cache.normalized[ids[terminal]])             # a success: no candidates at s', s' ones are borrowed (the value is masked)
    assert np.array_equal(nxt[~terminal], cache.normalized[ids[~terminal] + 1])      # includes the last state of the time-out (row 6)
    assert 5 in ids.tolist() and 6 not in ids.tolist()
    cache.version[6] = -1
    with pytest.raises(KeyError):
        cu.make_batch(buf, cache, np.random.default_rng(0), cfg, batch_size=64)       # the time-out's last state really needs its candidates


def test_the_target_and_the_loss_are_what_the_equations_say():
    buf, cache = filled([episode(3, True, seed=1), episode(3, False, seed=2)])
    trainer, cfg = trainer_of(BEST_OF_N)
    batch = cu.make_batch(buf, cache, np.random.default_rng(3), cfg)
    params, target = trainer.critic.params, trainer.critic.target_params
    # by hand, with the network's own functions: the action of pi^Opt at s' (argmax of the current Q over the candidates), valued by the target copy
    q_cand_next = rc.q_values(trainer.net, params, batch["next_state"], batch["next_cand"])
    best = jnp.take_along_axis(jnp.asarray(batch["next_cand"]), jnp.argmax(q_cand_next, axis=1)[:, None, None, None], axis=1)
    q_next = rc.q_values(trainer.net, target, batch["next_state"], best)[:, 0]
    y = batch["reward"] + cfg.gamma * (1.0 - batch["terminal"]) * np.asarray(q_next)
    q_data = np.asarray(rc.q_values(trainer.net, params, batch["state"], batch["action"][:, None])[:, 0])
    q_cand = np.asarray(rc.q_values(trainer.net, params, batch["state"], batch["cand"]))
    td = np.mean((q_data - y) ** 2)
    regulariser = np.mean(np.maximum(q_cand, batch["vmu"][:, None]).mean(axis=1) - q_data)
    metrics = cu.to_floats(trainer.update(batch))
    assert np.isclose(metrics["td_loss"], td, rtol=1e-4, atol=1e-6) and np.isclose(metrics["cal_ql_term"], regulariser, rtol=1e-4, atol=1e-6)
    assert np.isclose(metrics["loss"], td + cfg.cal_ql_alpha * regulariser, rtol=1e-4, atol=1e-6) and np.isclose(metrics["target_mean"], y.mean(), rtol=1e-4, atol=1e-6)
    assert np.isclose(metrics["q_minus_return"], (q_data - batch["vmu"]).mean(), rtol=1e-4, atol=1e-5) and np.isclose(metrics["candidate_q_std"], q_cand.std(axis=1).mean(), rtol=1e-4)


def rows_of(batch, rows):
    """The same batch restricted to `rows` (with repetition)."""
    out = {k: v[rows] for k, v in batch.items() if isinstance(v, np.ndarray)}
    for key in ("state", "next_state"):
        out[key] = {name: arr[rows] for name, arr in batch[key].items()}
    return out


def test_a_success_is_terminal_its_target_is_the_reward_alone_and_a_time_out_bootstraps():
    buf, cache = filled([episode(2, True, seed=1), episode(2, False, seed=2)])       # rows 0-2: plans -1 then 0 (terminal); rows 3-5: two time-out plans, -1 each
    trainer, cfg = trainer_of(BEST_OF_N)
    batch = cu.make_batch(buf, cache, np.random.default_rng(0), cfg, batch_size=32)
    terminal = np.flatnonzero(batch["terminal"])
    assert len(terminal) and not batch["terminal"][batch["ids"] >= 3].any()           # no time-out transition is terminal
    assert cu.to_floats(trainer.update(rows_of(batch, np.resize(terminal, 8))))["target_mean"] == 0.0           # the success: y = r = 0 exactly, whatever the target network says
    timed_out = np.flatnonzero(batch["ids"] >= 3)
    last_plan = timed_out[batch["ids"][timed_out] == 4]                                # the time-out's last plan: its next state is the cut state, which is bootstrapped from
    assert len(last_plan)
    y = cu.to_floats(trainer.update(rows_of(batch, np.resize(last_plan, 8))))["target_mean"]
    assert y != -1.0 and np.isfinite(y)                                               # -1 + gamma * Qbar(cut state, a'), not the bare reward


def test_the_target_copy_follows_the_critic_by_tau_after_each_update():
    buf, cache = filled([episode(3, True, seed=1), episode(3, False, seed=2)])
    trainer, cfg = trainer_of(BEST_OF_N)
    before_target = jax.tree.map(np.array, trainer.critic.target_params)
    trainer.update(cu.make_batch(buf, cache, np.random.default_rng(0), cfg))
    assert trainer.updates == 1
    expected = jax.tree.map(lambda t, p: (1 - cfg.tau) * t + cfg.tau * np.array(p), before_target, trainer.critic.params)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(trainer.critic.target_params)))
    assert not all(np.allclose(a, b) for a, b in zip(jax.tree.leaves(before_target), jax.tree.leaves(trainer.critic.params)))        # the critic itself moved


def test_parl_with_local_steps_runs_and_reports_the_optimisation_metrics_best_of_n_does_not():
    buf, cache = filled([episode(3, True, seed=1), episode(3, False, seed=2)])
    for extra, has_local in ((PARL, True), (BEST_OF_N, False)):
        trainer, cfg = trainer_of(extra)
        metrics = cu.to_floats(trainer.update(cu.make_batch(buf, cache, np.random.default_rng(0), cfg)))
        assert all(np.isfinite(v) for v in metrics.values())
        assert ("next_local_q_gain" in metrics) == has_local
        if has_local:
            assert metrics["next_local_q_gain"] >= 0 and 0 <= metrics["next_local_accepted_share"] <= 1


def test_the_critic_learns_the_values_of_a_chain_of_states():
    """Four plans per episode, reward -1, -1, -1, 0: the true Q at plan i is -(1 - g^(3 - i)) / (1 - g) (here gamma 0.9: -2.71, -1.9, -1.0, 0); the state tells the plan through joint 0 and the executed chunks are all zero (the value depends on the state only)."""
    eps = [episode(4, True, progress=True, seed=i) for i in range(6)]
    buf, cache = filled(eps, spread=0.01)                                              # the VLA's samples lie close to what was executed (here: all the chunks are about zero)
    trainer, cfg = trainer_of({**BEST_OF_N, "critic_lr": 3e-3, "tau": 0.05, "critic_batch_size": 16}, cameras=())                       # proprioception only: fast on CPU
    rng = np.random.default_rng(0)
    result = cu.run_updates(trainer, buf, cache, rng, steps=1500)
    assert result["seconds_per_update"] > 0 and trainer.updates == 1500
    ids = np.array([0, 1, 2, 3])                                                      # the first episode's states s_0 .. s_3 (plans 1..4: s_3 is the last plan)
    state = buf.critic_state(ids)
    q = np.asarray(rc.q_values(trainer.net, trainer.critic.params, state, buf.action[ids][:, None]))[:, 0]
    g = cfg.gamma
    truth = np.array([-(1 - g ** (3 - i)) / (1 - g) for i in range(4)])
    assert np.all(np.diff(q) > 0)                                                     # closer to the success = more valuable
    assert np.allclose(q, truth, atol=0.5), (q, truth)


def test_run_updates_logs_every_window_and_returns_the_mean():
    buf, cache = filled([episode(3, True, seed=1), episode(3, False, seed=2)])
    trainer, cfg = trainer_of(BEST_OF_N)
    seen = []
    result = cu.run_updates(trainer, buf, cache, np.random.default_rng(0), steps=6, log_every=2, log=lambda n, m: seen.append((n, m["td_loss"])))
    assert [n for n, _ in seen] == [2, 4, 6] and "td_loss" in result and trainer.updates == 6
    seen.clear()
    cu.run_updates(trainer, buf, cache, np.random.default_rng(0), steps=5, log_every=2, log=lambda n, m: seen.append((n, m["td_loss"])))      # 5 = 2 + 2 + the 1 left over
    assert [n for n, _ in seen] == [8, 10, 11]
    with pytest.raises(ValueError):
        cu.run_updates(trainer, buf, cache, np.random.default_rng(0), steps=0)
