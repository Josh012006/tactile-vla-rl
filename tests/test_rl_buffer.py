"""src/algos/rl_buffer.py: rows, rewards, V^mu, the time-out rule, the mixing of warm-up and online data, what the candidate cache and the critic read."""
import numpy as np
import pytest

from src.algos.rl_buffer import Episode, ReplayBuffer
from src.envs import protocol as P

H = 4


def episode(plans, success, tag=1.0, prompt="insert", seed=0):
    """T plans; every array says what it is: image pixels = row number within the episode, joints = tag, chunk value = tag * 10 + plan."""
    rng = np.random.default_rng(seed)
    ext = np.broadcast_to(np.arange(plans + 1, dtype=np.uint8)[:, None, None, None], (plans + 1, 224, 224, 3)).copy()
    wrist = np.broadcast_to((100 + np.arange(plans + 1, dtype=np.uint8))[:, None, None, None], (plans + 1, 224, 224, 3)).copy()
    actions = np.zeros((plans, H, 8), np.float32) + (tag * 10 + np.arange(plans))[:, None, None]
    return Episode(exterior=ext, wrist=wrist, joints=np.full((plans + 1, 7), tag, np.float32), gripper=rng.random((plans + 1, 1)).astype(np.float32), actions=actions, success=success, prompt=prompt)


def buffer(capacity=40, gamma=0.5):
    return ReplayBuffer(capacity, H, gamma, step_reward=-1.0)


def test_an_episode_takes_t_plus_one_contiguous_rows_and_the_last_has_no_action():
    buf = buffer()
    ids = buf.add_episode(episode(3, True), warmup=False)
    assert ids.tolist() == [0, 1, 2, 3] and len(buf) == 4
    assert buf.has_action[:4].tolist() == [True, True, True, False]
    assert buf.transition_ids().tolist() == [0, 1, 2]
    second = buf.add_episode(episode(2, False), warmup=True)
    assert second.tolist() == [4, 5, 6] and buf.episode[:7].tolist() == [0, 0, 0, 0, 1, 1, 1] and buf.plan_index[4:7].tolist() == [0, 1, 2]


def test_reward_is_minus_one_per_plan_and_zero_at_the_success_which_is_the_only_terminal():
    buf = buffer()
    buf.add_episode(episode(3, True), warmup=False)
    buf.add_episode(episode(3, False), warmup=False)
    assert buf.reward[:3].tolist() == [-1, -1, 0] and buf.terminal[:4].tolist() == [False, False, True, False]
    assert buf.reward[4:7].tolist() == [-1, -1, -1] and not buf.terminal[4:8].any()                  # a time-out is not terminal: its last plan bootstraps


def test_v_mu_is_the_discounted_return_to_go_of_the_plans():
    buf = buffer(gamma=0.5)
    buf.add_episode(episode(3, True), warmup=False)          # rewards -1, -1, 0
    buf.add_episode(episode(3, False), warmup=True)          # rewards -1, -1, -1 (truncated: the return of the plans taken)
    assert np.allclose(buf.vmu[:3], [-1.5, -1.0, 0.0])        # -1 + .5 * (-1 + .5 * 0), -1 + .5 * 0, 0
    assert np.allclose(buf.vmu[4:7], [-1.75, -1.5, -1.0])
    assert buf.episodes[0]["return"] == -1.5 and buf.episodes[1]["return"] == -1.75


def test_a_time_outs_last_state_needs_candidates_a_successs_does_not():
    buf = buffer()
    buf.add_episode(episode(2, True), warmup=False)          # rows 0 1 2
    buf.add_episode(episode(2, False), warmup=False)         # rows 3 4 5
    assert buf.candidate_ids().tolist() == [0, 1, 3, 4, 5]


def test_the_buffer_refuses_what_does_not_fit():
    buf = buffer(capacity=5)
    buf.add_episode(episode(3, True), warmup=False)
    with pytest.raises(ValueError, match="full"):
        buf.add_episode(episode(2, True), warmup=False)
    bad = episode(2, True)
    bad.actions = bad.actions[:, :, :7]
    with pytest.raises(ValueError, match="actions"):
        buffer().add_episode(bad, warmup=False)
    with pytest.raises(ValueError):
        buffer().add_episode(Episode(**{**episode(2, True).__dict__, "actions": np.zeros((0, H, 8), np.float32)}), warmup=False)


def test_the_batches_mix_warm_up_and_online_rows_by_the_ratio_and_fall_back_to_the_one_that_exists():
    buf = buffer(capacity=80)
    rng = np.random.default_rng(0)
    buf.add_episode(episode(5, True), warmup=True)
    assert set(buf.warmup[buf.sample_transitions(rng, 32, 0.5)]) == {True}                       # nothing online yet: all warm-up
    for _ in range(3):
        buf.add_episode(episode(5, False), warmup=False)
    ids = buf.sample_transitions(rng, 32, 0.5)
    assert buf.warmup[ids].sum() == 16 and buf.has_action[ids].all()                              # half and half
    assert buf.warmup[buf.sample_transitions(rng, 32, 0.25)].sum() == 8
    only_online = buffer()
    only_online.add_episode(episode(3, False), warmup=False)
    assert not only_online.warmup[only_online.sample_transitions(rng, 8, 0.5)].any()               # no warm-up data (warmup_episodes = 0): all online
    with pytest.raises(ValueError, match="no transition"):
        buffer().sample_transitions(rng, 4, 0.5)


def test_a_transition_batch_goes_to_the_next_row_and_carries_what_the_critic_reads():
    buf = buffer()
    buf.add_episode(episode(3, True, tag=1.0), warmup=False)
    buf.add_episode(episode(2, False, tag=2.0), warmup=False)
    b = buf.transition_batch([1, 5])                                                                # row 1: plan 1 of episode 0; row 5: plan 1 of episode 1 (rows 4 5 6)
    assert b["next_ids"].tolist() == [2, 6] and b["state"]["exterior"][:, 0, 0, 0].tolist() == [1, 1] and b["next_state"]["exterior"][:, 0, 0, 0].tolist() == [2, 2]
    assert b["state"]["wrist"][:, 0, 0, 0].tolist() == [101, 101] and b["state"]["proprio"].shape == (2, 8) and np.allclose(b["state"]["proprio"][:, 0], [1.0, 2.0])
    assert np.allclose(b["action"][:, 0, 0], [11, 21]) and b["reward"].tolist() == [-1, -1] and b["terminal"].tolist() == [False, False]
    assert buf.transition_batch([2])["terminal"].tolist() == [True]
    with pytest.raises(ValueError, match="no transition"):
        buf.transition_batch([3])                                                                    # the last state of an episode


def test_observations_have_the_env_format_and_their_prompts():
    buf = buffer()
    buf.add_episode(episode(2, True, prompt="insert"), warmup=False)
    buf.add_episode(episode(2, True, prompt="insert the peg"), warmup=False)
    obs, prompts = buf.observations([0, 4, 2])
    assert set(obs) == {P.EXTERIOR_IMAGE, P.WRIST_IMAGE, P.JOINT_POSITION, P.GRIPPER_POSITION} and obs[P.EXTERIOR_IMAGE].shape == (3, 224, 224, 3)
    assert prompts == ["insert", "insert the peg", "insert"] and obs[P.JOINT_POSITION].shape == (3, 7) and obs[P.GRIPPER_POSITION].shape == (3, 1)


def test_success_rate_by_kind_and_over_the_last_episodes():
    buf = buffer(capacity=60)
    for success, warm in [(False, True), (True, True), (False, False), (False, False), (True, False)]:
        buf.add_episode(episode(2, success), warmup=warm)
    assert buf.success_rate() == 0.4 and buf.success_rate(warmup=True) == 0.5 and buf.success_rate(warmup=False) == pytest.approx(1 / 3)
    assert buf.success_rate(warmup=False, last=2) == 0.5 and np.isnan(buffer().success_rate())


def test_touch_is_stored_with_the_states_and_given_to_the_critic_only_when_the_buffer_has_it():
    buf = ReplayBuffer(40, H, 0.5, touch_dim=5)
    ep = episode(3, True)
    with pytest.raises(ValueError, match="episode.touch"):
        buf.add_episode(ep, warmup=False)                                                            # the critic sees touch and the episode has none
    ep.touch = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    buf.add_episode(ep, warmup=False)
    b = buf.transition_batch([1, 2])
    assert np.array_equal(b["state"]["touch"][0], ep.touch[1]) and np.array_equal(b["next_state"]["touch"][0], ep.touch[2]) and np.array_equal(b["state"]["touch"][1], ep.touch[2])
    plain = buffer()
    with pytest.raises(ValueError, match="no touch"):
        plain.add_episode(ep, warmup=False)
    plain.add_episode(episode(2, True), warmup=False)
    assert "touch" not in plain.critic_state([0]) and buf.nbytes > 0
