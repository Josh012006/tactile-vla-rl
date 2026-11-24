"""scripts/rl.py without a GPU, a model or a server: the SFT checkpoint the run starts from, what is refused, the plan printed by --dry-run, the settings of a resumed run."""
import json

import pytest

pytest.importorskip("jax")

from scripts import rl as S  # noqa: E402
from src.algos import rl_config  # noqa: E402

EXP = {"name": "probe", "model": "pi05", "set": {"method": "best_of_n", "keep_top": 1, "local_steps": 0, "rollout_selection": "argmax"}}
EXPERIMENT = "configs/experiment/rl/pi05_best_of_n.yaml"


def make_sft_run(tmp_path, family="pi05", model_name="pi05", steps=(4000,)):
    run = tmp_path / "sft_run"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"model_name": model_name, "model_cfg": {"name": "pi05", "family": family, "action_horizon": 16}, "sft": {"seed": 3}}))
    for step in steps:
        (run / "checkpoints" / str(step)).mkdir(parents=True)
        (run / "checkpoints" / str(step) / "meta.json").write_text(json.dumps({"step": step}))
    return run


def test_the_start_is_the_command_lines_or_the_experiments_and_must_exist(tmp_path):
    run = make_sft_run(tmp_path)
    start = S.resolve_start(EXP, (str(run), "4000"))
    assert start["step"] == 4000 and start["checkpoint"].endswith("checkpoints/4000") and start["run_config"]["sft"]["seed"] == 3
    assert S.resolve_start({**EXP, "start": {"run": str(run), "step": 4000}}, None)["step"] == 4000
    assert S.resolve_start({**EXP, "start": {"run": "nothing", "step": 1}}, (str(run), "4000"))["run"] == str(run.resolve())          # the command line wins
    with pytest.raises(SystemExit, match="no SFT checkpoint to start from"):
        S.resolve_start(EXP, None)
    with pytest.raises(SystemExit, match="no complete checkpoint at step 500"):
        S.resolve_start(EXP, (str(run), "500"))


def test_the_model_is_the_sft_runs_and_pi0_fast_or_another_model_is_refused(tmp_path):
    ok = S.resolve_start(EXP, (str(make_sft_run(tmp_path)), "4000"))
    assert S.check_start(EXP, ok)["family"] == "pi05"
    with pytest.raises(SystemExit, match="not part of the RL methods"):
        S.check_start({**EXP, "model": "pi0_fast"}, S.resolve_start(EXP, (str(make_sft_run(tmp_path / "f", "pi0_fast", "pi0_fast")), "4000")))
    with pytest.raises(SystemExit, match="the experiment is for pi05"):
        S.check_start(EXP, S.resolve_start(EXP, (str(make_sft_run(tmp_path / "g", "pi0", "pi0")), "4000")))
    with pytest.raises(SystemExit, match="model_set"):
        S.check_start({**EXP, "model_set": {"vlm.lora_rank": 8}}, ok)


def test_a_resumed_run_keeps_its_settings_and_only_the_logging_may_change(tmp_path):
    cfg = rl_config.load_rl_config(overrides=EXP["set"])
    saved = {"rl": __import__("dataclasses").asdict(cfg)}
    assert S.resume_config(saved, {"wandb_mode": "offline"}).wandb_mode == "offline"
    assert S.resume_config(saved, {}) == cfg
    with pytest.raises(SystemExit, match="cannot be changed"):
        S.resume_config(saved, {"total_episodes": 100})


def test_the_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    run = make_sft_run(tmp_path)
    assert S.main(["--experiment", EXPERIMENT, "--start", str(run), "4000", "--dry-run", "--run-name", "never_created"]) == 0
    out = capsys.readouterr().out
    assert "method best_of_n" in out and "after the rounds [0, 5, 10, 15, 20]" in out and "warm-up: 600 episodes" in out and "20 rounds of 50 episodes" in out
    assert not (S.ROOT / "runs/rl/peg_insert/never_created").exists()
    with pytest.raises(SystemExit):
        S.main(["--experiment", EXPERIMENT, "--dry-run"])                                                                            # no start
    with pytest.raises(SystemExit):
        S.main(["--dry-run"])                                                                                                       # neither an experiment nor a run to resume


def test_the_dry_run_of_a_parl_experiment_says_it_distils_and_keeps_a_vla_checkpoint_per_round(tmp_path, capsys):
    run = make_sft_run(tmp_path)
    assert S.main(["--experiment", "configs/experiment/rl/pi05_parl_global_local.yaml", "--start", str(run), "4000", "--dry-run", "--run-name", "never_created"]) == 0
    out = capsys.readouterr().out
    assert "method parl_global_local" in out and "distillation after every round" in out and "a VLA checkpoint" in out


def saved_config(total=500):
    cfg = rl_config.load_rl_config(overrides={**EXP["set"], "total_episodes": total})
    return {"rl": __import__("dataclasses").asdict(cfg), "name": "run"}


def test_extending_a_run_changes_only_its_total_and_writes_down_what_was_done(tmp_path):
    saved = saved_config(500)
    out = S.extend_config(saved, 1000, last_stage=10)
    assert saved["rl"]["total_episodes"] == 500                                                                            # the input is not touched
    assert out["rl"]["total_episodes"] == 1000 and {k: v for k, v in out["rl"].items() if k != "total_episodes"} == {k: v for k, v in saved["rl"].items() if k != "total_episodes"}
    assert S.resume_config(out, {}).rounds == 20
    (note,) = out["extensions"]
    assert note["total_episodes_from"] == 500 and note["total_episodes_to"] == 1000 and note["after_stage"] == 10 and note["at"].endswith("+00:00")
    again = S.extend_config(out, 1000, last_stage=10)                                                                       # the same command sent again after a stop: nothing more
    assert again == out
    twice = S.extend_config(out, 1500, last_stage=20)
    assert [e["total_episodes_to"] for e in twice["extensions"]] == [1000, 1500]


def test_an_extension_cannot_cut_a_run_or_break_the_batches(tmp_path):
    saved = saved_config(500)
    with pytest.raises(SystemExit, match="can be continued further, not cut"):
        S.extend_config(saved, 400, None)
    with pytest.raises(SystemExit, match="multiple of envs_per_batch"):
        S.extend_config(saved, 520, None)


def test_extend_to_needs_a_resume_and_the_dry_run_extends_nothing(tmp_path, capsys):
    with pytest.raises(SystemExit):
        S.main(["--experiment", EXPERIMENT, "--extend-to", "1000", "--dry-run"])
    sft = make_sft_run(tmp_path)
    run = tmp_path / "rl_run"
    run.mkdir()
    saved = {**saved_config(500), "model_cfg": {"name": "pi05", "family": "pi05", "action_horizon": 16}, "task": {"max_steps_per_episode": 150}, "start": {"run": str(sft), "step": 4000, "checkpoint": str(sft / "checkpoints" / "4000")}}
    (run / "config.json").write_text(json.dumps(saved))
    (run / "resume").mkdir()
    (run / "resume" / "state.json").write_text(json.dumps({"stage": 10}))
    assert S.main(["--resume", str(run), "--extend-to", "1000", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "EXTENDED: total_episodes 500 -> 1000 after stage 10" in out and "20 rounds of 50 episodes" in out and "the VLA stays frozen" in out
    assert json.loads((run / "config.json").read_text())["rl"]["total_episodes"] == 500                                       # a dry run writes nothing
