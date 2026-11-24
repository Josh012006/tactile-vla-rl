"""src/algos/rl_run.py on CPU: the warm-up, the rounds, the evaluations, the distillation and the resume of a Best-of-N run and of a PA-RL run, with fake environments and a fake VLA (tests/rl_fakes.py) and a tiny critic."""
import contextlib
import json
import threading

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from rl_fakes import H, MAX_STEPS, FakeEnv, FakePolicy, FakeTrainablePolicy  # noqa: E402

from src.algos import rl_run as RUN  # noqa: E402
from src.algos.rl_config import load_rl_config  # noqa: E402
from src.eval import episodes as E  # noqa: E402

SMALL = {"method": "best_of_n", "keep_top": 1, "local_steps": 0, "rollout_selection": "argmax", "num_candidates": 4, "envs_per_batch": 4, "total_episodes": 8, "warmup_episodes": 4,
         "warmup_critic_steps": 3, "critic_cameras": ["wrist"], "critic_image_size": 16, "random_shift_px": 0, "critic_hidden": [16, 16], "critic_batch_size": 8, "eval_episodes": 4,
         "eval_every_rounds": 1, "rows_per_sample_call": 3, "seed": 5, "wandb_mode": "disabled"}
PARL = {"method": "parl", "keep_top": 2, "local_steps": 2, "local_lr": 1e-2, "rollout_selection": "softmax", "distill_batch_size": 4, "distill_micro_batch_size": 2, "distill_lr": 0.05}
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


def make_run(tmp_path, log=None, opener=None, stop=None, policy=None, **overrides):
    cfg = load_rl_config(overrides={**SMALL, **overrides})
    return RUN.RLRun(cfg, policy or FakePolicy(), tmp_path / "run", opener or Opener(), log or ListLog(), max_steps=MAX_STEPS, stop=stop), cfg


def make_parl(tmp_path, log=None, opener=None, stop=None, **overrides):
    return make_run(tmp_path, log, opener, stop, policy=FakeTrainablePolicy(), **{**PARL, **overrides})


def leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)]


@pytest.mark.parametrize("extra", [{}, {"method": "parl", "keep_top": 2, "local_steps": 2, "local_lr": 1e-2, "rollout_selection": "softmax", "distill_batch_size": 4, "distill_micro_batch_size": 2,
                                          "distill_lr": 0.05}])
def test_a_run_whose_critic_sees_the_force_proxy_goes_through_warm_up_rounds_and_evaluations(tmp_path, extra):
    """The proxy needs the last command in the rollout loop AND in the evaluation's: both run (warm-up with the VLA alone, online rounds, evaluations), with the 7-value vectors in the buffer."""
    run, cfg = make_run(tmp_path, policy=FakeTrainablePolicy() if extra else None, critic_use_force=True, critic_force_mode="proxy", **extra)
    summary = run.run()
    assert summary["stage"] == 2 and not summary["stopped"] and summary["evaluated"] == [0, 1, 2]
    assert cfg.critic_touch.dim == 7 and run.buffer.touch.shape[1] == 7 and np.all(np.isfinite(run.buffer.touch[:len(run.buffer)]))


@pytest.mark.parametrize("parl", [False, True])
def test_what_a_run_sends_to_wandb_is_finite_grouped_and_has_the_curves_one_looks_for(tmp_path, parl):
    from src.algos import tracking

    class Run:
        def __init__(self):
            self.logs, self.summary = [], {}

        def log(self, data):
            self.logs.append(data)

    run_out = Run()
    sink = tracking.WandbSink(run_out)
    log = ListLog(on_event=sink)
    run, _ = (make_parl if parl else make_run)(tmp_path, log=log)
    run.run()
    keys = set().union(*run_out.logs)
    assert all(np.isfinite(v) for d in run_out.logs for v in d.values())
    windows = [d for d in run_out.logs if "critic_update" in d]
    assert windows and all("step" not in d and all(k.startswith("critic") or k.startswith("time/") for k in d) for d in windows)       # the critic's curve, on its own axis
    assert [d["critic_update"] for d in windows] == sorted(d["critic_update"] for d in windows)
    must = {"eval/success_rate", "eval/success_se", "eval/opt/chosen_q_gain_mean", "eval/opt/chosen_shift_l1_mean", "rollout/success_rate", "rollout/opt/candidate_q_std_mean",
            "warmup/success_rate", "critic/td_loss", "critic/q_minus_return", "critic/grad_norm", "buffer/rows"}
    assert must <= keys, must - keys
    assert not {"round/round_id", "round/steps", "eval/n_episodes", "eval/round", "eval/wall_s_total", "warmup/critic/loss"} & keys
    assert any(k.startswith("distill/") for k in keys) == parl and any(k.startswith("refresh/") for k in keys) == parl and ("eval/opt/local_q_gain_mean" in keys) == parl
    assert [d["step"] for d in run_out.logs if "eval/success_rate" in d] == [0, 1, 2]                     # the evaluations, on the round axis


def test_a_run_goes_through_the_warm_up_the_rounds_and_the_evaluations_in_order(tmp_path):
    log, opener = ListLog(), Opener()
    run, cfg = make_run(tmp_path, log, opener)
    summary = run.run()
    assert summary == {"stage": 2, "stopped": False, "rounds": 2, "updates": summary["updates"], "vla_version": 0, "evaluated": [0, 1, 2]}
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


@pytest.mark.parametrize("mode", ["sparse_01", "sparse_pm1"])
def test_a_run_with_sparse_terminal_rewards_goes_through_warm_up_rounds_and_evaluations(tmp_path, mode):
    """`reward_mode` sparse_01 / sparse_pm1: the whole loop (collection, critic updates on terminal time-outs, evaluations) runs, the critic stays finite, and the rewards in the buffer are the sparse ones."""
    run, cfg = make_run(tmp_path, reward_mode=mode)
    summary = run.run()
    assert summary["evaluated"] == [0, 1, 2] and not summary["stopped"]
    assert set(np.unique(run.buffer.reward[:run.buffer.size][run.buffer.has_action[:run.buffer.size]])) <= {-1.0, 0.0, 1.0}
    assert run.buffer.terminal[:run.buffer.size].sum() == len(run.buffer.episodes)             # one terminal per episode, success or time-out
    assert all(np.isfinite(x).all() for x in leaves(run.trainer.critic.params))


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


def test_the_episodes_of_a_file_share_one_copy_of_each_array(tmp_path):
    """The resume of a stage of many episodes was OOM-killed (64 GB, 150 warm-up episodes of the scripted run of 18 Nov. 2026): `z[key]` inside the loop read the whole array again at every episode and each
    slice kept its own copy alive, so the memory was episodes x the file. All the episodes must be views of ONE array per key."""
    run, _ = make_run(tmp_path)
    collected, _ = RUN.collect_round(FakeEnv(PATTERN), run.actor, mode="vla", seed=1, round_id=0, replan=run.replan)
    RUN.save_episodes(tmp_path / "x.npz", collected)
    back = RUN.load_episodes(tmp_path / "x.npz")
    assert len(back) > 1
    for name in ("exterior", "wrist", "joints", "actions"):
        assert all(getattr(c.episode, name).base is not None for c in back), name
        assert len({id(getattr(c.episode, name).base) for c in back}) == 1, name
    for name in ("normalized", "states", "targets"):
        assert len({id(getattr(c.candidates, name).base) for c in back}) == 1, name


def test_a_run_without_a_warm_up_is_refused(tmp_path):
    with pytest.raises(ValueError, match="warmup_episodes"):
        make_run(tmp_path, warmup_episodes=0)


def test_the_critic_of_every_round_is_kept_even_when_only_some_rounds_are_evaluated(tmp_path):
    run, _ = make_run(tmp_path, eval_every_rounds=2)
    summary = run.run()
    assert summary["evaluated"] == [0, 2]
    assert [p.name for p in sorted((tmp_path / "run" / "critics").glob("*"))] == ["round_000.msgpack", "round_001.msgpack", "round_002.msgpack"]


def test_only_the_files_of_the_last_stage_are_kept_for_the_resume_and_a_stage_cut_before_its_state_file_is_ignored(tmp_path):
    run, _ = make_run(tmp_path)
    run.run()
    folder = tmp_path / "run" / "resume"
    assert sorted(p.name for p in folder.iterdir()) == ["critic_002.msgpack", "state.json"]
    (folder / "critic_003.msgpack").write_bytes(b"cut while being written")                  # a stage that did not reach its state file
    (folder / "vla_003").mkdir()
    again, _ = make_run(tmp_path)
    assert again.resume() and again.stage == 2
    assert sorted(p.name for p in folder.iterdir()) == ["critic_002.msgpack", "state.json"]


# -- PA-RL --------------------------------------------------------------------------------------------------------------------------------------------
def parl_state(run):
    return {"bias": run.policy.bias(), "trainable": leaves(run.distiller.trainer.trainable), "adam": leaves(run.distiller.trainer.opt_state), "cache": run.cache.snapshot(),
            "critic": leaves(run.trainer.critic.params), "critic_adam": leaves(run.trainer.opt_state), "distill_rng": run.distill_rng.bit_generator.state, "rng": run.rng.bit_generator.state}


def same_state(a, b):
    assert np.array_equal(a["bias"], b["bias"])
    assert a["distill_rng"] == b["distill_rng"] and a["rng"] == b["rng"]
    for key in ("trainable", "adam", "critic", "critic_adam"):
        assert len(a[key]) == len(b[key]) and all(np.array_equal(x, y) for x, y in zip(a[key], b[key])), key
    assert all(np.array_equal(a["cache"][k], b["cache"][k]) for k in a["cache"])


def test_a_parl_run_distils_after_every_round_and_draws_the_candidates_of_every_state_again_except_after_the_last(tmp_path):
    log, opener = ListLog(), Opener()
    run, cfg = make_parl(tmp_path, log, opener)
    assert cfg.variant == "parl_global_local"
    summary = run.run()
    assert summary["stage"] == 2 and summary["vla_version"] == 2 and run.distiller.updates > 0
    assert [e["event"] for e in log.events if e["event"] not in ("critic", "refresh_progress")] == ["start", "warmup", "warmup", "eval", "distill", "refresh", "round", "eval", "distill", "round", "eval", "done"]
    assert [e["version"] for e in log.of("refresh")] == [1] and [e["vla_version"] for e in log.of("round")] == [1, 2]           # round 2 is the last: its refresh would feed nothing
    ids = run.buffer.candidate_ids()
    assert np.all(run.cache.versions_of(ids) == 1) and log.of("refresh")[0]["states"] < len(ids)                      # the cache stays as round 1 left it
    assert np.abs(run.policy.bias()).max() > 0
    assert [d["states"] for d in log.of("distill")] == [log.of("refresh")[0]["states"], len(ids)]                      # the VLA learns from every state, the warm-up's too
    assert [json.loads(run.result_path(r).read_text())["vla_version"] for r in (0, 1, 2)] == [0, 1, 2]


def test_the_cache_holds_the_vla_that_drew_it_the_new_episodes_under_the_current_version_and_the_old_ones_refreshed(tmp_path):
    seen = []
    run, _ = make_parl(tmp_path, ListLog(on_event=lambda e: seen.append(np.unique(run.cache.versions_of(run.buffer.candidate_ids()))) if e["event"] == "round" else None))
    run.run()
    assert [v.tolist() for v in seen] == [[1], [1]]                                     # after round 1 every state was drawn by its VLA; after the last round nothing is redrawn (the cache is not read any more)


def test_each_round_keeps_the_vlas_trainable_parameters_as_a_checkpoint_and_the_last_stage_as_the_resume_state(tmp_path):
    run, _ = make_parl(tmp_path)
    run.run()
    base = tmp_path / "run"
    assert sorted(p.name for p in (base / "vlas").iterdir()) == ["round_001", "round_002"]
    meta = json.loads((base / "vlas" / "round_002" / "meta.json").read_text())
    assert meta["round"] == 2 and meta["step"] == run.distiller.updates and meta["variant"] == "parl_global_local" and (base / "vlas" / "round_002" / "params").is_dir()
    assert sorted(p.name for p in (base / "resume").iterdir()) == ["cache_002.npz", "critic_002.msgpack", "state.json", "vla_002"]
    state = json.loads((base / RUN.STATE_FILE).read_text())
    assert state["stage"] == 2 and state["version"] == 2 and state["vla_updates"] == run.distiller.updates


def test_a_parl_run_that_is_cut_resumes_with_the_same_vla_cache_critic_and_generators_as_an_uninterrupted_one(tmp_path):
    stop = threading.Event()
    run_a, _ = make_parl(tmp_path / "a", ListLog(on_event=lambda e: stop.set() if e["event"] == "round" and e["step"] == 1 else None), stop=stop)
    assert run_a.run()["stage"] == 1
    mid = parl_state(run_a)
    run_b, _ = make_parl(tmp_path / "a")                                                  # a new process: a fresh policy, nothing in memory
    assert run_b.resume() and run_b.stage == 1 and run_b.version == 1
    same_state(parl_state(run_b), mid)                                                    # the resume rebuilds exactly what was committed (cache with its versions, VLA, Adam, generators)
    run_b.run()
    reference, _ = make_parl(tmp_path / "b")
    reference.run()
    same_state(parl_state(run_b), parl_state(reference))
    assert run_b.distiller.updates == reference.distiller.updates and run_b.version == reference.version == 2


def test_a_parl_resume_after_the_warm_up_alone_starts_the_distillation_from_the_base_vla(tmp_path):
    stop = threading.Event()
    run_a, _ = make_parl(tmp_path / "a", ListLog(on_event=lambda e: stop.set() if e["event"] == "warmup" else None), stop=stop)
    assert run_a.run()["stage"] == 0
    run_b, _ = make_parl(tmp_path / "a")
    run_b.run()
    reference, _ = make_parl(tmp_path / "b")
    reference.run()
    same_state(parl_state(run_b), parl_state(reference))


def test_a_parl_run_that_is_done_continues_past_its_last_round_as_if_it_had_been_planned_with_more(tmp_path):
    """`--resume --extend-to`: the old last round did not redraw the cache (nothing read it); the continuation draws it again first, then plays on exactly as a longer run."""
    first, _ = make_parl(tmp_path / "a")
    assert first.run()["stage"] == 2
    log, opener = ListLog(), Opener()
    longer, cfg = make_parl(tmp_path / "a", log, opener, total_episodes=12)                  # a new process with the extended settings: 3 rounds
    assert cfg.rounds == 3 and longer.buffer.capacity > first.buffer.capacity
    summary = longer.run()
    assert summary["stage"] == 3 and not summary["stopped"] and summary["evaluated"] == [0, 1, 2, 3]
    assert [(e["step"], e.get("continued_past_the_old_last_round", False)) for e in log.of("refresh")] == [(2, True)]          # the refresh the old last round skipped, once, before round 3
    assert [label for _, label in opener.opened] == ["collect", "eval_3"]                      # round 3 and its evaluation only: nothing is played again
    reference, _ = make_parl(tmp_path / "b", total_episodes=12)
    reference.run()
    same_state(parl_state(longer), parl_state(reference))
    assert longer.distiller.updates == reference.distiller.updates and longer.version == reference.version == 3
    assert json.loads(longer.result_path(3).read_text())["eval_sha"] == json.loads(reference.result_path(3).read_text())["eval_sha"]


def test_the_refresh_of_an_extension_is_not_repeated_when_the_continuation_is_cut_and_resumed(tmp_path):
    first, _ = make_parl(tmp_path / "a")
    first.run()
    stop = threading.Event()
    log = ListLog(on_event=lambda e: stop.set() if e["event"] == "refresh" else None)
    cut, _ = make_parl(tmp_path / "a", log, stop=stop, total_episodes=12)
    assert cut.run()["stage"] == 2                                                              # cut right after the refresh, before round 3
    log_again = ListLog()
    again, _ = make_parl(tmp_path / "a", log_again, total_episodes=12)
    assert again.run()["stage"] == 3 and log_again.of("refresh") == []                          # the cache file written after the refresh holds version 2: nothing to draw again
    reference, _ = make_parl(tmp_path / "b", total_episodes=12)
    reference.run()
    same_state(parl_state(again), parl_state(reference))


def test_a_best_of_n_run_that_is_done_continues_past_its_last_round_with_nothing_to_redraw(tmp_path):
    first, _ = make_run(tmp_path / "a")
    first.run()
    log = ListLog()
    longer, _ = make_run(tmp_path / "a", log, total_episodes=12)
    assert longer.run()["stage"] == 3 and log.of("refresh") == []
    reference, _ = make_run(tmp_path / "b", total_episodes=12)
    reference.run()
    assert all(np.array_equal(x, y) for x, y in zip(leaves(longer.trainer.critic.params), leaves(reference.trainer.critic.params)))
    assert longer.trainer.updates == reference.trainer.updates and np.array_equal(longer.buffer.action[:len(longer.buffer)], reference.buffer.action[:len(reference.buffer)])
