"""scripts/eval_rl.py on CPU, without a model or a server: the run and its rounds (read from what a real RLRun writes, with the fakes of tests/rl_fakes.py), the protocol, --dry-run, --plot-only,
the critic read back."""
import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

from rl_fakes import H  # noqa: E402
from test_eval_episodes import write_results  # noqa: E402
from test_rl_run import make_parl, make_run  # noqa: E402

from scripts import eval_rl as ER  # noqa: E402
from src.algos import rl_config  # noqa: E402
from src.eval import episodes as E  # noqa: E402

MODEL_CFG = {"name": "pi05", "family": "pi05", "action_horizon": H}
TASK = {"max_steps_per_episode": 20, "num_envs": 4, "physics": "x"}


def finish_run(tmp_path, parl):
    """A run written by the real RLRun (fake VLA and environments), its config.json as scripts/rl.py writes it, and the SFT checkpoint it started from."""
    run, cfg = (make_parl if parl else make_run)(tmp_path)
    run.run()
    start = tmp_path / "sft_run" / "checkpoints" / "4000"
    start.mkdir(parents=True)
    (start / "meta.json").write_text(json.dumps({"step": 4000}))
    saved = {"name": "r", "rl": dataclasses.asdict(cfg), "start": {"run": str(tmp_path / "sft_run"), "step": 4000, "checkpoint": str(start)}, "model_cfg": MODEL_CFG, "model_name": "pi05",
             "sft_seed": 3, "task": TASK, "env_set": []}
    (tmp_path / "run" / "config.json").write_text(json.dumps(saved))
    return run, cfg, start, tmp_path / "run"


# -- the run and its rounds ---------------------------------------------------------------------------------------------------------------------------
def test_a_run_is_found_by_path_or_by_name_under_the_rl_folders(tmp_path, monkeypatch):
    _, _, _, run_dir = finish_run(tmp_path, parl=False)
    assert ER.find_rl_run_dir(str(run_dir)) == run_dir.resolve()
    monkeypatch.setattr(ER, "ROOT", tmp_path)
    (tmp_path / "runs" / "rl" / "probing").mkdir(parents=True)
    (tmp_path / "runs" / "rl" / "probing" / "p").symlink_to(run_dir)
    assert ER.find_rl_run_dir("p").name == "run"
    with pytest.raises(SystemExit, match="no RL run"):
        ER.find_rl_run_dir("nope")


def test_every_round_of_a_best_of_n_run_is_evaluated_with_the_sft_checkpoint_the_run_started_from(tmp_path):
    _, cfg, start, run_dir = finish_run(tmp_path, parl=False)
    found = ER.discover_rounds(run_dir, cfg, str(start))
    assert found == {0: start, 1: start, 2: start}
    assert [(j.label, j.step, j.checkpoint) for j in ER.select_jobs(found, None)] == [("0", 0, start), ("1", 1, start), ("2", 2, start)]
    assert [j.step for j in ER.select_jobs(found, [2, 0, 0])] == [0, 2]
    with pytest.raises(SystemExit, match=r"\[7\] not found"):
        ER.select_jobs(found, [7])


def test_a_parl_round_is_evaluated_with_its_own_vla_checkpoint_and_round_0_with_the_sft_one(tmp_path):
    run, cfg, start, run_dir = finish_run(tmp_path, parl=True)
    found = ER.discover_rounds(run_dir, cfg, str(start))
    assert found == {0: start, 1: run_dir / "vlas" / "round_001", 2: run_dir / "vlas" / "round_002"}
    assert json.loads((found[2] / "meta.json").read_text())["round"] == 2                       # what `prepare` checks
    assert ER.vla_checkpoint(run_dir, cfg, str(start), 0) == start and ER.vla_checkpoint(run_dir, cfg, str(start), 2) == found[2]


def test_a_round_that_is_not_finished_or_whose_vla_is_missing_is_not_offered(tmp_path):
    run, cfg, start, run_dir = finish_run(tmp_path, parl=True)
    (run_dir / "vlas" / "round_002" / "meta.json").unlink()                                     # the checkpoint of round 2 was being written
    assert sorted(ER.discover_rounds(run_dir, cfg, str(start))) == [0, 1]
    (run_dir / "vlas" / "round_002" / "meta.json").write_text("{}")
    state = json.loads((run_dir / "resume" / "state.json").read_text())
    (run_dir / "resume" / "state.json").write_text(json.dumps({**state, "stage": 1}))           # a stage whose state file was not written is not finished
    assert sorted(ER.discover_rounds(run_dir, cfg, str(start))) == [0, 1]


def test_the_critic_is_read_back_exactly_as_the_run_saved_it(tmp_path):
    import jax
    run, cfg, start, run_dir = finish_run(tmp_path, parl=False)
    params = ER.load_critic(ER.critic_file(run_dir, 2), run.trainer.net, H)
    saved = [np.asarray(x) for x in jax.tree.leaves(run.trainer.critic.params)]
    assert all(np.array_equal(a, np.asarray(b)) for a, b in zip(saved, jax.tree.leaves(params))) and len(saved) == len(jax.tree.leaves(params))
    first = ER.load_critic(ER.critic_file(run_dir, 0), run.trainer.net, H)
    assert any(not np.array_equal(a, np.asarray(b)) for a, b in zip(saved, jax.tree.leaves(first)))        # round 0's critic is not round 2's


# -- protocol ---------------------------------------------------------------------------------------------------------------------------------------------
def args_with(**kw):
    import argparse
    return argparse.Namespace(**{"seed": 42, "replan": None, "envs_per_server": 100, **kw})


def test_the_protocol_is_the_sft_ones_plus_the_settings_of_the_candidates_and_the_critic(tmp_path):
    _, cfg, start, run_dir = finish_run(tmp_path, parl=True)
    saved = json.loads((run_dir / "config.json").read_text())
    task = {"max_steps_per_episode": 20, "num_envs": 100}
    p = ER.build_protocol(args_with(), run_dir, saved, cfg, task)
    assert p["recipe"] == ER.RECIPE and p["master_seed"] == 42 and p["num_envs"] == 100 and p["replan"] == H // 2 and p["max_steps"] == 20 and p["model_batch"] is None
    assert p["rl"]["variant"] == "parl_global_local" and p["rl"]["num_candidates"] == cfg.num_candidates and p["rl"]["eval_selection"] == "argmax" and p["norm_stats_sha"] is None
    assert ER.build_protocol(args_with(replan=2), run_dir, saved, cfg, task)["replan"] == 2
    with pytest.raises(SystemExit, match="between 1"):
        ER.build_protocol(args_with(replan=H + 1), run_dir, saved, cfg, task)


def test_an_output_folder_refuses_other_rl_settings_but_not_other_rounds(tmp_path):
    _, cfg, _, run_dir = finish_run(tmp_path, parl=True)
    saved = json.loads((run_dir / "config.json").read_text())
    task = {"max_steps_per_episode": 20, "num_envs": 100}
    before = ER.build_protocol(args_with(), run_dir, saved, cfg, task)
    ER.check_rl_block(before, before)
    ER.check_rl_block(None, before)
    other = ER.build_protocol(args_with(), run_dir, saved, dataclasses.replace(cfg, local_steps=0), task)
    with pytest.raises(ValueError, match="local_steps"):
        ER.check_rl_block(before, other)
    assert E.FROZEN_PROTOCOL and "rl" not in E.FROZEN_PROTOCOL                                    # the SFT's own check does not see it: this one is needed


def test_the_task_of_the_evaluation_is_compared_with_the_runs_without_its_number_of_environments():
    assert ER.task_differences({"a": 1, "num_envs": 50}, {"a": 1, "num_envs": 100}) == []
    assert ER.task_differences({"a": 1, "num_envs": 50}, {"a": 2, "b": 3, "num_envs": 100}) == ["a: 1 in the run, 2 now", "b: None in the run, 3 now"]


# -- the command line ----------------------------------------------------------------------------------------------------------------------------------------
def test_the_dry_run_says_what_each_round_uses_and_writes_nothing(tmp_path, capsys):
    run, cfg, start, run_dir = finish_run(tmp_path, parl=True)
    assert ER.main(["--run", str(run_dir), "--dry-run", "--n-episodes", "20"]) == 0
    out = capsys.readouterr().out
    assert "parl_global_local" in out and "20 episodes per round" in out and "round   0:    0/20 episodes done" in out and "vlas/round_002" in out and "critics/round_001.msgpack" in out
    assert "the SFT start" in out and "the base checkpoint's own" in out
    assert not (run_dir / "eval_rl").exists()                                                       # the run's own eval/ and this folder are never mixed
    assert ER.main(["--run", str(run_dir), "--dry-run", "--rounds", "1"]) == 0
    assert "round   0" not in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not found"):
        ER.main(["--run", str(run_dir), "--dry-run", "--rounds", "9"])


def test_the_dry_run_shows_the_episodes_already_done_and_warns_of_another_task(tmp_path, capsys):
    run, cfg, start, run_dir = finish_run(tmp_path, parl=False)
    done = E.EpisodeLog(run_dir / "eval_rl" / "results" / "1.jsonl")
    for i in range(3):
        done.append({"episode": i, "seed": E.make_seeds(42, 20)[i], "success": True})
    ER.main(["--run", str(run_dir), "--dry-run", "--n-episodes", "20"])
    out = capsys.readouterr().out
    assert "round   1:    3/20 episodes done" in out and "WARNING: the task of this evaluation is not the one of the run" in out    # the fake run's task is not the repository's YAML


def test_the_plot_only_rebuilds_the_curve_of_the_results_on_disk_with_the_collection_round_as_x(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    _, _, _, run_dir = finish_run(tmp_path, parl=False)
    with pytest.raises(SystemExit, match="no complete result"):
        ER.main(["--run", str(run_dir), "--plot-only"])
    write_results(run_dir / "eval_rl" / "results", [(0, 0.1), (1, 0.4), (2, 0.7)], n=20)
    assert ER.main(["--run", str(run_dir), "--plot-only"]) == 0
    assert (run_dir / "eval_rl" / "curve.png").exists() and "70.0%" in capsys.readouterr().out


def test_a_run_without_a_finished_round_has_nothing_to_evaluate(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    cfg = rl_config.load_rl_config(overrides={"method": "best_of_n", "keep_top": 1, "local_steps": 0, "rollout_selection": "argmax"})
    (run_dir / "config.json").write_text(json.dumps({"rl": dataclasses.asdict(cfg), "model_cfg": MODEL_CFG, "start": {"run": "x", "step": 1, "checkpoint": str(tmp_path / "none")}, "task": TASK}))
    with pytest.raises(SystemExit, match="no finished round"):
        ER.main(["--run", str(run_dir), "--dry-run"])


# -- the whole command, with the model, the sweep and the servers replaced ------------------------------------------------------------------------------------
class ForkablePolicy:
    """The fake VLA with what main asks of an OpenPIPolicy: `state` / `set_state`, `fork`, `info`; the checkpoints it is given are recorded."""

    def __init__(self):
        from rl_fakes import FakeTrainablePolicy
        self.inner = FakeTrainablePolicy()
        self.info, self.loaded = self.inner.info, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def fork(self, seed=0):
        other = ForkablePolicy()
        other.loaded = self.loaded
        return other


def run_main(tmp_path, monkeypatch, parl, argv=(), meta_of=None):
    import src.algos.sft as sft_mod
    import src.models as models
    from src.eval.sweep import SweepResult
    run, cfg, start, run_dir = finish_run(tmp_path, parl=parl)
    seen = {"prepared": [], "kwargs": None}

    class FakeSweep:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

        def run(self):
            kw = seen["kwargs"]
            policy = kw["policy_factory"]()
            for job in kw["jobs"]:
                kw["prepare"](policy, job)
                seen["prepared"].append((job.step, policy.params, list(policy.actor.policy.loaded)))
            return SweepResult(complete=[j.label for j in kw["jobs"]], incomplete=[], failed={}, stopped=False)

    def fake_load_checkpoint(policy, directory):
        policy.loaded.append(str(directory))
        meta = json.loads((__import__("pathlib").Path(directory) / "meta.json").read_text())
        return meta_of(directory, meta) if meta_of else meta

    monkeypatch.setattr(ER, "Sweep", FakeSweep)
    monkeypatch.setattr(ER, "gpu_name", lambda: "fake gpu")
    monkeypatch.setattr(ER, "enable_compilation_cache", lambda: None)
    monkeypatch.setattr(ER, "make_server_factory", lambda *a, **k: ("factory", a, k))
    monkeypatch.setattr(models, "load_model", lambda cfg_, seed: ForkablePolicy())
    monkeypatch.setattr(sft_mod, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    code = ER.main(["--run", str(run_dir), "--n-episodes", "10", "--env-set", "max_steps_per_episode=20", *argv])
    return code, seen, run, run_dir


def test_the_command_evaluates_every_round_with_its_own_vla_and_critic_under_deterministic_xla(tmp_path, monkeypatch):
    import jax
    code, seen, run, run_dir = run_main(tmp_path, monkeypatch, parl=True)
    assert code == 0 and __import__("os").environ["XLA_FLAGS"] == "--xla_gpu_deterministic_ops=true"
    kw = seen["kwargs"]
    assert kw["workers"] == 1 and kw["master_seed"] == 42 and kw["envs_per_server"] == 100 and kw["seeds"] == E.make_seeds(42, 10) and kw["replan"] == H // 2 and kw["max_steps"] == 20
    assert kw["output_dir"] == run_dir / "eval_rl" and kw["server_factory"][2]["num_envs"] == 100
    steps = [s for s, _, _ in seen["prepared"]]
    assert steps == [0, 1, 2]
    loaded = [dirs[-1] for _, _, dirs in seen["prepared"]]
    assert [l.rsplit("/", 2)[-2:] for l in loaded[1:]] == [["vlas", "round_001"], ["vlas", "round_002"]] and "sft_run" in loaded[0]
    final = [np.asarray(x) for x in jax.tree.leaves(run.trainer.critic.params)]
    last = seen["prepared"][-1][1]
    assert all(np.array_equal(a, np.asarray(b)) for a, b in zip(final, jax.tree.leaves(last)))        # round 2 is evaluated with round 2's critic
    protocol = json.loads((run_dir / "eval_rl" / E.PROTOCOL_FILE).read_text())
    assert protocol["recipe"] == ER.RECIPE and protocol["attempts"][0]["gpu"] == "fake gpu" and protocol["attempts"][0]["labels"] == ["0", "1", "2"] and protocol["rl"]["local_steps"] == cfg_local(run)
    assert json.loads((run_dir / "eval_rl" / E.SEEDS_FILE).read_text())["seeds"] == E.make_seeds(42, 10)


def cfg_local(run):
    return run.cfg.local_steps


def test_the_policy_of_a_worker_is_the_runs_own_argmax_over_the_optimised_candidates(tmp_path, monkeypatch):
    code, seen, run, _ = run_main(tmp_path, monkeypatch, parl=True)
    policy = seen["kwargs"]["policy_factory"]()
    assert policy.mode == "argmax" and policy.actor.cfg.keep_top == run.cfg.keep_top == 2 and policy.actor.cfg.local_steps == run.cfg.local_steps == 2
    assert policy.actor.sampler.envs_per_call == run.cfg.rows_per_sample_call


def test_the_vla_alone_can_be_evaluated_without_the_critic_in_its_own_output_folder(tmp_path, monkeypatch):
    code, seen, run, run_dir = run_main(tmp_path, monkeypatch, parl=True, argv=["--selection", "vla", "--output-dir", str(tmp_path / "run" / "eval_rl_vla")])
    assert code == 0
    policy = seen["kwargs"]["policy_factory"]()
    assert policy.mode == "vla"
    out = run_dir / "eval_rl_vla"
    protocol = json.loads((out / E.PROTOCOL_FILE).read_text())
    assert protocol["rl"]["eval_selection"] == "vla" and seen["kwargs"]["output_dir"] == out
    assert [s for s, _, _ in seen["prepared"]] == [0, 1, 2]                                     # the same rounds, each with its own VLA
    with pytest.raises(SystemExit, match="eval_selection"):                                      # the folder of the VLA alone does not take the critic's evaluations
        ER.main(["--run", str(run_dir), "--n-episodes", "10", "--env-set", "max_steps_per_episode=20", "--output-dir", str(out)])


def test_the_selection_is_part_of_the_protocol_and_defaults_to_the_runs_own(tmp_path):
    _, cfg, _, run_dir = finish_run(tmp_path, parl=True)
    saved = json.loads((run_dir / "config.json").read_text())
    task = {"max_steps_per_episode": 20, "num_envs": 100}
    own = ER.build_protocol(args_with(), run_dir, saved, cfg, task)
    alone = ER.build_protocol(args_with(selection="vla"), run_dir, saved, cfg, task)
    assert own["rl"]["eval_selection"] == "argmax" and alone["rl"]["eval_selection"] == "vla" and ER.build_protocol(args_with(selection=None), run_dir, saved, cfg, task) == own
    with pytest.raises(ValueError, match="eval_selection"):
        ER.check_rl_block(own, alone)


def test_a_vla_checkpoint_of_another_round_or_another_start_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="says it is round 5, not 1"):
        run_main(tmp_path / "a", monkeypatch, parl=True, meta_of=lambda d, m: {**m, "round": 5} if "vlas" in str(d) else m)
    with pytest.raises(ValueError, match="started from step 4000"):
        run_main(tmp_path / "b", monkeypatch, parl=False, meta_of=lambda d, m: {**m, "step": 2000})


def test_the_same_output_folder_cannot_be_used_with_other_settings_and_a_second_attempt_is_recorded(tmp_path, monkeypatch):
    code, _, run, run_dir = run_main(tmp_path, monkeypatch, parl=False)
    assert ER.main(["--run", str(run_dir), "--n-episodes", "10", "--env-set", "max_steps_per_episode=20", "--rounds", "2"]) == 0
    protocol = json.loads((run_dir / "eval_rl" / E.PROTOCOL_FILE).read_text())
    assert [a["attempt"] for a in protocol["attempts"]] == [1, 2] and protocol["attempts"][1]["labels"] == ["2"]
    with pytest.raises(SystemExit, match="protocol"):
        ER.main(["--run", str(run_dir), "--n-episodes", "10", "--env-set", "max_steps_per_episode=20", "--replan", "2"])
    with pytest.raises(SystemExit, match="master_seed"):
        ER.main(["--run", str(run_dir), "--n-episodes", "10", "--env-set", "max_steps_per_episode=20", "--seed", "7"])
