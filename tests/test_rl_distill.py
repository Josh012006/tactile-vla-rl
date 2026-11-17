"""src/algos/rl_distill.py on CPU: the labels of the critic, the epochs of SFT on them, the state that is saved and resumed, and the checkpoint, with a fake trainable VLA (tests/rl_fakes.py) and a tiny critic."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("orbax.checkpoint")
import flax.nnx as nnx  # noqa: E402

from rl_fakes import FakeEnv, FakeTrainablePolicy, MAX_STEPS  # noqa: E402

from src.algos import rl_critic as rc  # noqa: E402
from src.algos import rl_distill as D  # noqa: E402
from src.algos import rl_rollout as RR  # noqa: E402
from src.algos import sft  # noqa: E402
from src.algos.rl_action_opt import ActionOptimiser  # noqa: E402
from src.algos.rl_buffer import ReplayBuffer  # noqa: E402
from src.algos.rl_candidates import CandidateCache  # noqa: E402
from src.algos.rl_config import load_rl_config  # noqa: E402

CFG = {"num_candidates": 4, "keep_top": 2, "local_steps": 2, "local_lr": 1e-2, "critic_cameras": ["wrist"], "critic_image_size": 16, "random_shift_px": 0, "critic_hidden": [16, 16],
       "rows_per_sample_call": 3, "distill_batch_size": 8, "distill_micro_batch_size": 4, "distill_lr": 1e-1, "distill_selection": "argmax", "seed": 3, "wandb_mode": "disabled",
       "method": "parl"}


@pytest.fixture(scope="module")
def world():
    """A buffer and a cache filled by the fake VLA (episodes of 4 fake environments, two rounds), a critic, and the optimiser."""
    cfg = load_rl_config(overrides=CFG)
    net = rc.CriticNet.from_config(cfg)
    critic = rc.init_critic(net, jax.random.key(0), horizon=8)
    optimiser = ActionOptimiser.for_critic(net, cfg)
    policy = FakeTrainablePolicy()
    actor = RR.RLActor(policy, cfg, optimiser)
    buffer, cache = ReplayBuffer(300, 8, 0.99), CandidateCache(4, 8, 3, capacity=300)
    for r in range(2):
        collected, _ = RR.collect_round(FakeEnv([10, MAX_STEPS + 5, 8, 3]), actor, mode="vla", seed=1, round_id=r, replan=4)
        RR.store_round(buffer, cache, collected, warmup=r == 0, version=0)
    return cfg, net, critic, optimiser, buffer, cache


def make(world, policy=None):
    cfg, net, critic, optimiser, buffer, cache = world
    policy = policy or FakeTrainablePolicy()
    return D.Distiller(policy, optimiser, cfg, model_cfg=None), policy


def test_the_label_of_a_state_is_the_chunk_the_critic_ranks_first_after_its_local_steps(world):
    cfg, net, critic, optimiser, buffer, cache = world
    distiller, _ = make(world)
    ids = buffer.candidate_ids()
    labels, metrics = distiller.annotate(critic.params, jax.random.key(0), ids, buffer, cache)
    assert labels.shape == (len(ids), 8, 8) and len(ids) > 8
    part = ids[:5]
    result = optimiser(critic.params, buffer.critic_state(part), cache.get(part))
    assert np.allclose(labels[:5], np.asarray(result.actions)[np.arange(5), np.asarray(result.q).argmax(axis=1)], atol=1e-5)          # argmax of the optimised set
    assert "label_candidate_q_std_mean" not in metrics and "candidate_q_std_mean" in metrics and metrics["local_q_gain_mean"] >= 0


def test_the_control_labels_are_the_vlas_own_first_candidates_whatever_the_critic_says(world):
    import dataclasses
    cfg, net, critic, optimiser, buffer, cache = world
    control = dataclasses.replace(cfg, distill_selection="first")
    distiller = D.Distiller(FakeTrainablePolicy(), optimiser, control, model_cfg=None)
    ids = buffer.candidate_ids()
    labels, metrics = distiller.annotate(critic.params, jax.random.key(0), ids, buffer, cache)
    assert np.array_equal(labels, cache.get(ids)[:, 0])                                        # exactly the first sample of every state
    assert metrics["chosen_shift_l1_mean"] == 0 and metrics["chosen_differs_from_first_sample_share"] == 0 and "chosen_q_gain_mean" in metrics
    other, _ = make(world)
    assert not np.array_equal(other.annotate(critic.params, jax.random.key(0), ids, buffer, cache)[0], labels)      # the critic's argmax labels are not the first samples
    out = distiller.distill(critic.params, jax.random.key(1), np.random.default_rng(0), buffer, cache)
    assert out["label_chosen_shift_l1_mean"] == 0 and out["steps"] == len(ids) // 8 and distiller.updates == out["steps"]


def test_a_distillation_moves_the_trainable_parameters_and_not_the_frozen_ones_and_the_policy_runs_the_new_weights(world):
    cfg, net, critic, optimiser, buffer, cache = world
    distiller, policy = make(world)
    before_frozen = np.asarray(policy.state.to_pure_dict()["frozen"]).copy()
    assert not policy.bias().any()
    metrics = distiller.distill(critic.params, jax.random.key(1), np.random.default_rng(0), buffer, cache)
    n = len(buffer.candidate_ids())
    assert metrics["states"] == n and metrics["steps"] == n // 8 and distiller.updates == n // 8 and policy.trained_on == [8] * (n // 8)
    assert policy.bias().any()                                                                                # the policy got the trained weights (bfloat16)
    assert np.allclose(policy.bias(), np.asarray(distiller.trainer.trainable.to_pure_dict()["bias"]), atol=1e-2)
    assert np.array_equal(np.asarray(policy.state.to_pure_dict()["frozen"]), before_frozen)
    assert {"label_candidate_q_std_mean", "loss_first", "loss_last", "grad_norm_mean", "annotate_s", "train_s"} <= set(metrics)


def test_the_vla_goes_to_the_mean_of_the_labels_over_the_rounds_and_adam_is_kept_from_one_round_to_the_next(world):
    cfg, net, critic, optimiser, buffer, cache = world
    distiller, policy = make(world)
    labels, _ = distiller.annotate(critic.params, jax.random.key(0), buffer.candidate_ids(), buffer, cache)
    target = labels.mean(axis=0)                                              # what the squared distance to the labels is minimised by
    start = float(np.abs(target).mean())
    losses = [distiller.distill(critic.params, jax.random.key(r), np.random.default_rng(r), buffer, cache)["loss_mean"] for r in range(6)]
    assert float(np.abs(policy.bias() - target).mean()) < start / 2 and losses[-1] < losses[0]
    assert any(np.asarray(x).any() for x in jax.tree.leaves(distiller.trainer.opt_state))


def test_a_saved_state_resumes_exactly(world, tmp_path):
    cfg, net, critic, optimiser, buffer, cache = world
    a, _ = make(world)
    a.distill(critic.params, jax.random.key(1), np.random.default_rng(0), buffer, cache)
    a.save_state(tmp_path / "vla_001")
    b, policy_b = make(world)
    b.load_state(tmp_path / "vla_001", a.updates)
    assert b.updates == a.updates and np.allclose(policy_b.bias(), np.asarray(a.trainer.trainable.to_pure_dict()["bias"]), atol=1e-2)
    for x, y in zip(jax.tree.leaves(a.trainer.opt_state), jax.tree.leaves(b.trainer.opt_state)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    ma = a.distill(critic.params, jax.random.key(2), np.random.default_rng(1), buffer, cache)
    mb = b.distill(critic.params, jax.random.key(2), np.random.default_rng(1), buffer, cache)
    assert ma["loss_mean"] == mb["loss_mean"] and np.array_equal(np.asarray(a.trainer.trainable.to_pure_dict()["bias"]), np.asarray(b.trainer.trainable.to_pure_dict()["bias"]))


def test_a_checkpoint_holds_the_trainable_parameters_in_the_sft_layout(world, tmp_path):
    """`sft.load_checkpoint` reads it on the cluster (its own tests, the orbax of the `vla` group); here the files are checked: openpi's layout, the trainable parameters only, float32."""
    import orbax.checkpoint as ocp
    cfg, net, critic, optimiser, buffer, cache = world
    a, _ = make(world)
    a.distill(critic.params, jax.random.key(1), np.random.default_rng(0), buffer, cache)
    a.save_checkpoint(tmp_path / "ckpt", {"step": 1})
    assert (tmp_path / "ckpt" / sft.META_FILE).exists() and (tmp_path / "ckpt" / sft.PARAMS_DIR).is_dir()
    saved = ocp.PyTreeCheckpointer().restore((tmp_path / "ckpt" / sft.PARAMS_DIR).resolve())
    leaves = jax.tree.leaves(saved)
    assert len(leaves) == 1 and leaves[0].dtype == np.float32 and np.array_equal(leaves[0], np.asarray(a.trainer.trainable.to_pure_dict()["bias"]))        # `bias` only: not the frozen parameter


def test_fewer_states_than_a_batch_is_refused(world):
    cfg, net, critic, optimiser, buffer, cache = world
    distiller, _ = make(world)
    with pytest.raises(ValueError, match="fewer than a batch"):
        distiller.distill(critic.params, jax.random.key(0), np.random.default_rng(0), ReplayBuffer(300, 8, 0.99), cache)
