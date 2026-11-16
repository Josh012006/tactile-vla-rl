"""src/algos/rl_rollout.py on CPU: a round of K fake environments with a fake VLA and a tiny critic. The episodes' lengths, states, chunks and candidates; the modes (VLA alone, argmax, softmax);
what is stored; the evaluation through src/eval/episodes.run_batch."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
import jax.numpy as jnp  # noqa: E402

from src.algos import rl_critic as rc  # noqa: E402
from src.algos import rl_rollout as RR  # noqa: E402
from src.algos.rl_action_opt import ActionOptimiser  # noqa: E402
from src.algos.rl_buffer import ReplayBuffer  # noqa: E402
from src.algos.rl_candidates import CandidateCache  # noqa: E402
from src.algos.rl_config import load_rl_config  # noqa: E402
from rl_fakes import CFG, H, K, MAX_STEPS, REPLAN, FakeEnv, FakePolicy, FakeSpec, fake_targets  # noqa: E402

def toy_critic(cfg, seed=0):
    net = rc.CriticNet(cameras=(), hidden=tuple(cfg.critic_hidden), image_size=32, shift_px=0, touch_dim=cfg.critic_touch.dim)
    return net, rc.init_critic(net, jax.random.key(seed), horizon=H)


def actor_of(**extra):
    cfg = load_rl_config(overrides={**CFG, **extra})
    net, critic = toy_critic(cfg)
    return RR.RLActor(FakePolicy(), cfg, ActionOptimiser.for_critic(net, cfg)), cfg, net, critic


def test_rollout_seeds_are_a_function_of_the_four_numbers_alone():
    assert RR.rollout_seed(1, 2, 3, 4) == RR.rollout_seed(1, 2, 3, 4)
    seeds = {RR.rollout_seed(1, r, e, k) for r in range(3) for e in range(3) for k in range(3)}
    assert len(seeds) == 27 and all(0 <= s < 2 ** 32 for s in seeds)


def test_a_round_of_the_vla_alone_gives_one_episode_per_environment_with_the_right_lengths():
    actor, cfg, net, critic = actor_of()
    env = FakeEnv([10, MAX_STEPS + 5, 8, 3])                       # success mid-chunk, time-out, success exactly at a chunk's end, success in the first chunk
    collected, stats = RR.collect_round(env, actor, mode="vla", seed=7, round_id=2, replan=REPLAN)
    assert env.resets == [(7, 2)] and [c.env for c in collected] == [0, 1, 2, 3]
    plans = [c.episode.plans for c in collected]
    assert plans == [3, 5, 2, 1]                                    # plans at steps 0 4 8 12 16: the time-out has 5, the success at step 8 only 2 (it ended where the third plan would start)
    assert [c.episode.success for c in collected] == [True, False, True, True] and stats["success_rate"] == 0.75 and stats["episodes"] == 4
    assert [c.steps for c in collected] == [10, MAX_STEPS, 8, 3] and stats["env_steps"] == 41
    for c in collected:
        T = c.episode.plans
        assert c.episode.exterior.shape == (T + 1, 224, 224, 3) and c.episode.actions.shape == (T, H, 8) and c.episode.touch is None
        assert len(c.candidates.normalized) == (T if c.episode.success else T + 1)          # a time-out also has the candidates of its last state


def test_the_executed_actions_are_the_chosen_chunks_targets_and_the_states_are_those_seen_at_the_plans():
    actor, cfg, net, critic = actor_of()
    env = FakeEnv([MAX_STEPS + 5] * 2)
    collected, _ = RR.collect_round(env, actor, mode="vla", seed=0, round_id=0, replan=REPLAN)
    c = collected[0]
    assert len(env.received) == MAX_STEPS and c.episode.plans == 5
    for j in range(c.episode.plans):                                # chunk j: the VLA's first candidate, executed REPLAN steps from the joints seen at the plan
        joints_at_plan = c.episode.joints[j]
        expected = fake_targets(c.episode.actions[j][None, None], joints_at_plan[None])[0, 0]
        steps = env.received[j * REPLAN:(j + 1) * REPLAN]
        assert np.allclose(np.stack([s[0] for s in steps]), expected[:len(steps)], atol=1e-6)
    assert np.array_equal(c.episode.actions, c.candidates.normalized[:5, 0])
    assert [int(c.episode.exterior[i, 0, 0, 0]) for i in range(4)] == [0, 4, 8, 12]       # the images at the plans' steps
    assert int(c.episode.exterior[5, 0, 0, 0]) == 20 and np.allclose(c.episode.joints[5], env.received[-1][0, :7])                         # the last state: where the last action left the arm


def test_a_critic_mode_executes_the_argmax_of_the_critic_over_the_candidates():
    actor, cfg, net, critic = actor_of(method="best_of_n", keep_top=1, local_steps=0, rollout_selection="argmax")
    env = FakeEnv([MAX_STEPS + 5] * 3)
    collected, stats = RR.collect_round(env, actor, mode="argmax", params=critic.params, seed=0, round_id=0, replan=REPLAN)
    c = collected[0]
    T = c.episode.plans
    state = {"exterior": c.episode.exterior[:T], "wrist": c.episode.wrist[:T], "proprio": np.concatenate([c.episode.joints[:T], c.episode.gripper[:T]], axis=1)}
    q = np.asarray(rc.q_values(net, critic.params, state, c.candidates.normalized[:T]))
    best = c.candidates.normalized[:T][np.arange(T), q.argmax(axis=1)]
    assert np.allclose(c.episode.actions, best, atol=1e-6) and any(q.argmax(axis=1) != 0)        # not just the VLA's first sample
    assert "opt_candidate_q_std_mean" in stats and stats["opt_chosen_differs_from_first_sample_share"] > 0


def test_softmax_with_local_steps_moves_the_chunk_and_what_is_executed_is_that_chunk():
    actor, cfg, net, critic = actor_of()                                                           # parl with 2 local steps
    env = FakeEnv([MAX_STEPS + 5] * 3)
    collected, stats = RR.collect_round(env, actor, mode="softmax", params=critic.params, seed=0, round_id=0, key=jax.random.key(0), replan=REPLAN)
    c = collected[0]
    expected = fake_targets(c.episode.actions[0][None, None], c.episode.joints[0][None])[0, 0]
    assert np.allclose(np.stack([s[0] for s in env.received[:REPLAN]]), expected[:REPLAN], atol=1e-5)
    assert "opt_local_q_gain_mean" in stats and stats["opt_local_q_gain_mean"] >= 0
    with pytest.raises(ValueError, match="PRNG key"):
        RR.collect_round(FakeEnv([MAX_STEPS + 5]), actor, mode="softmax", params=critic.params, seed=0, round_id=0, replan=REPLAN)


def test_the_candidates_of_a_plan_depend_on_seed_round_env_and_plan_only():
    actor, cfg, net, critic = actor_of()
    a, _ = RR.collect_round(FakeEnv([MAX_STEPS + 5] * 4), actor, mode="vla", seed=3, round_id=1, replan=REPLAN)
    b, _ = RR.collect_round(FakeEnv([MAX_STEPS + 5, 5, MAX_STEPS + 5, 5]), actor, mode="vla", seed=3, round_id=1, replan=REPLAN)       # other environments end early: fewer rows per call
    assert np.array_equal(a[0].candidates.normalized, b[0].candidates.normalized) and np.array_equal(a[2].candidates.normalized, b[2].candidates.normalized)
    c, _ = RR.collect_round(FakeEnv([MAX_STEPS + 5] * 4), actor, mode="vla", seed=3, round_id=2, replan=REPLAN)
    assert not np.array_equal(a[0].candidates.normalized, c[0].candidates.normalized) and not np.array_equal(a[0].candidates.normalized, a[1].candidates.normalized)


def test_the_critic_batch_is_padded_so_that_the_padding_never_changes_a_chosen_chunk():
    actor, cfg, net, critic = actor_of(method="best_of_n", keep_top=1, local_steps=0, rollout_selection="argmax")
    env = FakeEnv([MAX_STEPS + 5] * 3)
    obs = env.reset(seed=0).obs
    cands_seeds = [11, 12, 13]
    short = actor.act(obs, FakeSpec.prompt, cands_seeds, "argmax", critic.params)
    padded = actor.act(obs, FakeSpec.prompt, cands_seeds, "argmax", critic.params, pad_to=8)
    assert np.array_equal(short.normalized, padded.normalized) and np.allclose(short.q, padded.q, atol=1e-5) and padded.normalized.shape == (3, H, 8)


def test_the_touch_of_the_critic_is_stored_with_the_states_when_the_critic_sees_it():
    actor, cfg, net, critic = actor_of(critic_use_force=True)
    assert cfg.critic_touch.dim == 4
    collected, _ = RR.collect_round(FakeEnv([MAX_STEPS + 5, 10]), actor, mode="argmax", params=critic.params, seed=0, round_id=0, replan=REPLAN)
    for c in collected:
        assert c.episode.touch.shape == (c.episode.plans + 1, 4)


class LaggingEnv(FakeEnv):
    """A FakeEnv whose joints only go half way to the commanded target each step (q <- q + 0.5 (a - q)): a non-zero tracking residual. Keeps what was commanded and the joints it saw."""

    def __init__(self, success_at):
        super().__init__(success_at)
        self.commanded, self.q_seen = [], []

    def reset(self, seed=0, round=0):
        out = super().reset(seed, round)
        self.commanded, self.q_seen = [], [self.q.copy()]
        return out

    def step(self, actions, auto_reset=False):
        self.commanded.append(np.array(actions, np.float32))
        lagged = np.array(actions, np.float32)
        lagged[:, :7] = self.q + 0.5 * (lagged[:, :7] - self.q)
        out = super().step(lagged, auto_reset)
        self.q_seen.append(self.q.copy())
        return out


def _proxy(prev, q):
    raw = (prev - q) / 0.02
    return np.sign(raw) * np.log1p(np.abs(raw))


def test_the_proxy_critic_sees_the_residual_between_the_last_command_and_the_joints_at_every_stored_state():
    actor, cfg, net, critic = actor_of(critic_use_force=True, critic_force_mode="proxy")
    assert cfg.critic_touch.dim == 7 and actor.touch.needs_previous_target
    env = LaggingEnv([MAX_STEPS + 5, 10, 3])                                         # a time-out, a success after 10 steps (plans at 0, 4, 8), one after 3
    collected, _ = RR.collect_round(env, actor, mode="argmax", params=critic.params, seed=0, round_id=0, replan=REPLAN)
    for c in collected:
        e, t_end = c.env, c.steps
        times = [k * REPLAN for k in range(c.episode.plans)] + [t_end]                 # the plans, then the last state
        assert c.episode.touch.shape == (len(times), 7)
        for row, t in zip(c.episode.touch, times):
            prev = env.q_seen[0][e] if t == 0 else env.commanded[t - 1][e, :7]         # nothing commanded yet at step 0: the joints themselves
            assert np.allclose(row, _proxy(prev, env.q_seen[t][e]), atol=1e-5), (e, t)
        assert not np.any(c.episode.touch[0]) and np.any(c.episode.touch[1])           # zero at the start, a real residual afterwards (the arm lags)


def test_the_proxy_is_not_asked_of_a_critic_that_does_not_use_it():
    actor, cfg, net, critic = actor_of(critic_use_force=True)
    assert not actor.touch.needs_previous_target and not RR.OptimisingPolicy(actor, critic.params).uses_previous_target
    assert RR.OptimisingPolicy(actor_of(critic_use_force=True, critic_force_mode="proxy")[0], critic.params).uses_previous_target


def test_the_evaluation_gives_the_proxy_critic_the_last_command_and_leaves_the_others_alone():
    actor, cfg, net, critic = actor_of(method="best_of_n", keep_top=1, local_steps=0, rollout_selection="argmax", critic_use_force=True, critic_force_mode="proxy")
    seen = []
    original = actor.act

    def spy(obs, *args, **kwargs):
        seen.append({k: np.array(v) for k, v in obs.items() if k in ("client/previous_joint_target", "observation/joint_position")})
        return original(obs, *args, **kwargs)

    actor.act = spy
    env = LaggingEnv([MAX_STEPS + 5, MAX_STEPS + 5])
    RR.evaluate(actor, critic.params, env, [1, 2], replan=REPLAN, max_steps=MAX_STEPS, master_seed=42)
    assert len(seen) == MAX_STEPS // REPLAN and all("client/previous_joint_target" in s for s in seen)
    assert np.array_equal(seen[0]["client/previous_joint_target"], seen[0]["observation/joint_position"])          # the first plan: zero residual
    assert np.allclose(seen[1]["client/previous_joint_target"], env.commanded[REPLAN - 1][:, :7])                  # the next: what was commanded last
    plain, _, _, plain_critic = actor_of(method="best_of_n", keep_top=1, local_steps=0, rollout_selection="argmax", critic_use_force=True)
    seen.clear()
    original_plain = plain.act
    plain.act = lambda obs, *a, **k: (seen.append(set(obs)), original_plain(obs, *a, **k))[1]
    RR.evaluate(plain, plain_critic.params, LaggingEnv([MAX_STEPS + 5, MAX_STEPS + 5]), [1, 2], replan=REPLAN, max_steps=MAX_STEPS, master_seed=42)
    assert seen and all("client/previous_joint_target" not in keys for keys in seen)


def test_store_round_fills_the_buffer_and_the_cache_for_exactly_the_states_that_need_candidates():
    actor, cfg, net, critic = actor_of()
    collected, _ = RR.collect_round(FakeEnv([10, MAX_STEPS + 5, 3]), actor, mode="vla", seed=0, round_id=0, replan=REPLAN)
    buf = ReplayBuffer(200, H, 0.99)
    cache = CandidateCache(K, H, 3, capacity=16)
    ids = RR.store_round(buf, cache, collected, warmup=True, version=0)
    assert [len(i) for i in ids] == [4, 6, 2] and buf.warmup[:len(buf)].all()
    assert cache.has(buf.candidate_ids()).all() and len(buf.candidate_ids()) == 3 + 6 + 1          # success: T rows, time-out: T + 1 rows
    assert np.array_equal(cache.get(ids[0][:3]), collected[0].candidates.normalized) and (cache.versions_of(buf.candidate_ids()) == 0).all()
    assert buf.success_rate() == pytest.approx(2 / 3)


def test_the_evaluation_runs_the_fixed_episodes_with_the_critics_argmax_and_summarises_them():
    actor, cfg, net, critic = actor_of(method="best_of_n", keep_top=1, local_steps=0, rollout_selection="argmax")
    env = FakeEnv([10, MAX_STEPS + 5, 8, 3, 5])                                                     # 5 environments, 4 episodes play: the fifth is held
    seeds = [101, 102, 103, 104]
    result = RR.evaluate(actor, critic.params, env, seeds, replan=REPLAN, max_steps=MAX_STEPS, master_seed=42, label="eval", step=3)
    assert env.resets == [(42, 0)] and result["n_episodes"] == 4 and result["n_success"] == 3 and result["success_rate"] == 0.75 and result["step"] == 3
    assert "optimisation" in result and "candidate_q_std_mean" in result["optimisation"]
    with pytest.raises(ValueError, match="start from the same reset"):
        RR.evaluate(actor, critic.params, FakeEnv([5, 5]), [1, 2, 3], replan=REPLAN, max_steps=MAX_STEPS, master_seed=42)
    with pytest.raises(ValueError, match="argmax"):
        RR.OptimisingPolicy(actor, critic.params, mode="softmax")
