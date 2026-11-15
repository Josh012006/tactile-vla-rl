"""src/algos/rl_run.py on CPU: the warm-up, the rounds, the evaluations and the resume of a Best-of-N run, with fake environments and a fake VLA (tests/rl_fakes.py) and a tiny critic."""
import contextlib
import json
import threading

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from rl_fakes import H, MAX_STEPS, FakeEnv, FakePolicy  # noqa: E402

from src.algos import rl_run as RUN  # noqa: E402
from src.algos.rl_config import load_rl_config  # noqa: E402
from src.eval import episodes as E  # noqa: E402

SMALL = {"method": "best_of_n", "keep_top": 1, "local_steps": 0, "rollout_selection": "argmax", "num_candidates": 4, "envs_per_batch": 4, "total_episodes": 8, "warmup_episodes": 4,
         "warmup_critic_steps": 3, "critic_cameras": ["wrist"], "critic_image_size": 16, "random_shift_px": 0, "critic_hidden": [16, 16], "critic_batch_size": 8, "eval_episodes": 4,
         "eval_every_rounds": 1, "rows_per_sample_call": 3, "seed": 5, "wandb_mode": "disabled"}
PATTERN = [10, MAX_STEPS + 5, 8, 3]


class ListLog:
    def __init__(self, on_event=None):
        self.events, self.on_event = [], on_event

    def write(self, **fields):
        self.events.append(fields)
        if self.on_event:
            self.on_event(fields)

    def of(self, name):
        return [e for e in self.events if e["event"] == name]


class Opener:
    """open_env(num_envs, label) -> a context manager giving a FakeEnv; remembers what was opened."""

    def __init__(self):
        self.opened, self.envs = [], []

    def __call__(self, num_envs, label):
        env = FakeEnv([PATTERN[i % 4] for i in range(num_envs)])
        self.opened.append((num_envs, label))
        self.envs.append((label, env))
        return contextlib.nullcontext(env)


def make_run(tmp_path, log=None, opener=None, stop=None, **overrides):
    cfg = load_rl_config(overrides={**SMALL, **overrides})
    return RUN.RLRun(cfg, FakePolicy(), tmp_path / "run", opener or Opener(), log or ListLog(), max_steps=MAX_STEPS, stop=stop), cfg


def leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)]


def test_a_run_goes_through_the_warm_up_the_rounds_and_the_evaluations_in_order(tmp_path):
    log, opener = ListLog(), Opener()
    run, cfg = make_run(tmp_path, log, opener)
    summary = run.run()
    assert summary == {"stage": 2, "stopped": False, "rounds": 2, "updates": summary["updates"], "evaluated": [0, 1, 2]}
    assert [e["event"] for e in log.events if e["event"] != "critic"] == ["start", "warmup", "warmup", "eval", "round", "eval", "round", "eval", "done"]
    assert opener.opened == [(4, "collect"), (4, "eval_0"), (4, "collect"), (4, "eval_1"), (4, "collect"), (4, "eval_2")]        # one server at a time: the collection one is stopped for an evaluation
    collect_resets = [env.resets for label, env in opener.envs if label == "collect"]
    assert collect_resets == [[(5, 0)], [(5, 1)], [(5, 2)]]                                                                       # warm-up = round 0 of the sequence, the online rounds 1 and 2: no start twice
    assert all(env.resets == [(E.DEFAULT_SEED, 0)] for label, env in opener.envs if label.startswith("eval"))                     # the starts of the SFT evaluations
    steps_per_episode = [10, MAX_STEPS, 8, 3]
    assert log.of("round")[0]["critic_updates"] == cfg.utd * sum(steps_per_episode) == summary["updates"] - 3 - cfg.utd * sum(steps_per_episode)
    assert len(run.buffer.episodes) == 12 and sum(e["warmup"] for e in run.buffer.episodes) == 4 and len(run.cache.normalized) >= len(run.buffer)
    for r in (0, 1, 2):
        result = json.loads((run.result_path(r)).read_text())
        assert result["round"] == r and result["n_episodes"] == 4 and result["success_rate"] == 0.75 and result["step"] == r
    assert [p.name for p in sorted((tmp_path / "run" / "critics").glob("*"))] == ["round_000.msgpack", "round_001.msgpack", "round_002.msgpack"]
    assert [p.name for p in sorted((tmp_path / "run" / "episodes").glob("*"))] == ["stage_000.npz", "stage_001.npz", "stage_002.npz"]


def test_the_warm_up_data_is_the_vlas_and_the_online_data_is_the_critics_choice(tmp_path):
    run, cfg = make_run(tmp_path)
    run.run()
    warm = run.buffer.warmup[:len(run.buffer)]
    ids = np.flatnonzero(warm & run.buffer.has_action[:len(run.buffer)])
    assert len(ids) == 3 + 5 + 2 + 1                                                    # the plans of the four warm-up episodes (the replan is half the chunk: 4 steps)
    assert np.array_equal(run.buffer.action[ids], run.cache.normalized[ids][:, 0])      # the VLA alone executes its first candidate
    online = np.flatnonzero(~warm & run.buffer.has_action[:len(run.buffer)])
    assert len(online) == 2 * len(ids) and any(not np.array_equal(run.buffer.action[i], run.cache.normalized[i][0]) for i in online)      # the critic picks other candidates


def test_a_run_that_is_cut_resumes_where_it_stopped_and_ends_exactly_as_an_uninterrupted_one(tmp_path):
    stop = threading.Event()
    run_a, _ = make_run(tmp_path / "a", ListLog(on_event=lambda e: stop.set() if e["event"] == "round" and e["step"] == 1 else None), stop=stop)
    summary = run_a.run()
    assert summary["stopped"] and summary["stage"] == 1 and summary["evaluated"] == [0]                  # round 1 is committed, its evaluation not yet done
    log_b, opener_b = ListLog(), Opener()
    run_b, cfg = make_run(tmp_path / "a", log_b, opener_b)                                                  # a new process: nothing in memory
    assert run_b.resume() and run_b.stage == 1 and len(run_b.buffer.episodes) == 8
    summary_b = run_b.run()
    assert summary_b["stage"] == 2 and not summary_b["stopped"] and summary_b["evaluated"] == [0, 1, 2]
    assert [label for _, label in opener_b.opened] == ["eval_1", "collect", "eval_2"] and log_b.of("start")[0]["resumed_after_stage"] == 1
    reference, _ = make_run(tmp_path / "b")
    reference.run()
    assert all(np.array_equal(x, y) for x, y in zip(leaves(run_b.trainer.critic.params), leaves(reference.trainer.critic.params)))
    assert all(np.array_equal(x, y) for x, y in zip(leaves(run_b.trainer.opt_state), leaves(reference.trainer.opt_state)))
    assert run_b.trainer.updates == reference.trainer.updates and np.array_equal(run_b.buffer.action[:len(run_b.buffer)], reference.buffer.action[:len(reference.buffer)])
    for r in (0, 1, 2):
        assert json.loads(run_b.result_path(r).read_text())["eval_sha"] == json.loads(reference.result_path(r).read_text())["eval_sha"]


def test_a_stop_asked_during_a_stage_waits_for_the_end_of_that_stage(tmp_path):
    stop = threading.Event()
    run, _ = make_run(tmp_path, ListLog(on_event=lambda e: stop.set() if e["event"] == "warmup" else None), stop=stop)
    summary = run.run()
    assert summary["stopped"] and summary["stage"] == 0 and (tmp_path / "run" / RUN.STATE_FILE).exists()


def test_the_episodes_of_a_stage_come_back_from_their_file_with_their_candidates_and_touch(tmp_path):
    run, cfg = make_run(tmp_path, critic_use_force=True)
    collected, _ = RUN.collect_round(FakeEnv(PATTERN), run.actor, mode="vla", seed=1, round_id=0, replan=run.replan)
    RUN.save_episodes(tmp_path / "x.npz", collected)
    back = RUN.load_episodes(tmp_path / "x.npz")
    assert len(back) == 4
    for a, b in zip(collected, back):
        for name in ("exterior", "wrist", "joints", "gripper", "actions", "touch"):
            assert np.array_equal(getattr(a.episode, name), getattr(b.episode, name))
        assert a.episode.success == b.episode.success and a.episode.prompt == b.episode.prompt and (a.env, a.round_id, a.steps) == (b.env, b.round_id, b.steps)
        for name in ("normalized", "states", "targets"):
            assert np.array_equal(getattr(a.candidates, name), getattr(b.candidates, name))


def test_what_this_step_does_not_do_is_refused(tmp_path):
    with pytest.raises(NotImplementedError, match="distillation"):
        make_run(tmp_path, method="parl", keep_top=2, local_steps=0, rollout_selection="softmax")
    with pytest.raises(ValueError, match="warmup_episodes"):
        make_run(tmp_path, warmup_episodes=0)
