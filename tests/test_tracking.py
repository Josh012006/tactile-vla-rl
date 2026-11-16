"""The wandb logging of a run (src/algos/tracking.py), against a fake `wandb` module: no network, no package needed."""
import json
import sys
import types

import numpy as np
import pytest

from src.algos import sft, tracking


class FakeRun:
    def __init__(self, run_id):
        self.id = run_id
        self.config_updates, self.metric_defs, self.logs, self.finished = [], [], [], []
        self.summary = {}
        self.config = types.SimpleNamespace(update=lambda values, allow_val_change=False: self.config_updates.append((values, allow_val_change)))

    def define_metric(self, name, **kwargs):
        self.metric_defs.append((name, kwargs))

    def log(self, data):
        self.logs.append(data)

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class FakeWandb(types.ModuleType):
    """`init` records its arguments; `fail` lists the modes whose init raises."""

    def __init__(self, fail=()):
        super().__init__("wandb")
        self.calls, self.fail, self.runs, self.finish_calls = [], set(fail), [], []
        self.Image = lambda array: ("image", array.shape, str(array.dtype))

    def init(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["mode"] in self.fail:
            raise RuntimeError(f"{kwargs['mode']} is not possible here")
        run = FakeRun(kwargs.get("id", "newrun01"))
        self.runs.append(run)
        return run

    def finish(self, exit_code=0):
        self.finish_calls.append(exit_code)


@pytest.fixture
def fake(monkeypatch):
    module = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_a_new_run_gets_its_id_written_next_to_the_metrics(fake, tmp_path):
    run = tracking.init_wandb(tmp_path, False, "proj", "name", "online", {"sft": {"lr": 1}}, ["pi05", "lora"])
    call = fake.calls[0]
    assert (call["project"], call["name"], call["dir"], call["mode"], call["tags"]) == ("proj", "name", str(tmp_path), "online", ["pi05", "lora"])
    assert "id" not in call and "resume" not in call
    assert (tmp_path / "wandb_id.txt").read_text() == run.id == "newrun01"
    assert run.config_updates == [({"sft": {"lr": 1}}, False)]
    assert ("step", {}) in run.metric_defs and ("*", {"step_metric": "step"}) in run.metric_defs     # our `step` is the x axis of everything


def test_the_tags_wandb_would_refuse_are_dropped(fake, tmp_path):
    tracking.init_wandb(tmp_path, False, "proj", "name", "online", tags=["pi05", "", "x" * 80])
    assert fake.calls[0]["tags"] == ["pi05", "x" * 64]


def test_a_resume_continues_the_run_of_the_first_attempt(fake, tmp_path):
    (tmp_path / "wandb_id.txt").write_text("abc123\n")
    run = tracking.init_wandb(tmp_path, True, "proj", "name", "online", {"sft": {"steps": 8}})
    call = fake.calls[0]
    assert call["id"] == "abc123" and call["resume"] == "must"                          # the same run, an error rather than a new one
    assert (tmp_path / "wandb_id.txt").read_text() == "abc123\n"
    assert run.config_updates[0][1] is True                                             # `steps` may have changed: allow_val_change


def test_a_resume_of_a_run_that_was_never_logged_starts_a_new_wandb_run(fake, tmp_path, capsys):
    run = tracking.init_wandb(tmp_path, True, "proj", "name", "online")
    assert "id" not in fake.calls[0] and (tmp_path / "wandb_id.txt").read_text() == run.id
    assert "not found" in capsys.readouterr().err


def test_disabled_logs_nothing_and_does_not_even_import_wandb(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", None)                                     # an import would raise
    assert tracking.init_wandb(tmp_path, False, "proj", "name", "disabled") is None
    assert not (tmp_path / "wandb_id.txt").exists()
    with pytest.raises(ValueError, match="wandb_mode"):
        tracking.init_wandb(tmp_path, False, "proj", "name", "sometimes")


def test_a_missing_package_means_metrics_jsonl_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert tracking.init_wandb(tmp_path, False, "proj", "name", "online") is None
    assert "not installed" in capsys.readouterr().err


def test_an_online_run_that_cannot_start_falls_back_to_offline_with_the_same_id(monkeypatch, tmp_path, capsys):
    module = FakeWandb(fail={"online"})
    monkeypatch.setitem(sys.modules, "wandb", module)
    (tmp_path / "wandb_id.txt").write_text("abc123")
    run = tracking.init_wandb(tmp_path, True, "proj", "name", "online")
    assert [c["mode"] for c in module.calls] == ["online", "offline"] and module.calls[1]["id"] == "abc123"
    assert run is module.runs[0] and "falling back to offline" in capsys.readouterr().err


def test_when_nothing_works_the_run_goes_on_without_wandb(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(fail={"online", "offline"}))
    assert tracking.init_wandb(tmp_path, False, "proj", "name", "online") is None
    assert tracking.start_tracking(tmp_path, False, "proj", "name", "offline") is None


def test_flatten_keeps_the_numbers_of_nested_dictionaries():
    out = tracking.flatten("eval", {"train": {"loss": 1.5, "decode": {"mean_len": 22.4, "samples": ["a b"]}}, "ok": True, "n": 3})
    assert out == {"eval/train/loss": 1.5, "eval/train/decode/mean_len": 22.4, "eval/n": 3}


def test_the_events_of_metrics_jsonl_become_wandb_points_on_the_step_axis(fake, tmp_path):
    sink = tracking.start_tracking(tmp_path, False, "proj", "name", "online")
    run = fake.runs[0]
    sink({"event": "start", "trainable_m": 442.7, "frozen_m": 2508.5, "train_steps": 300, "val_steps": 0, "run_dir": "x"})
    sink({"event": "resume", "from_step": 4, "snapshot": "x", "target_steps": 8})
    sink({"event": "first_step", "seconds": 61.0, "note": "includes the compilation", "memory_mib": {"in_use": 100, "peak": 200}})
    sink({"event": "train", "step": 6, "loss": 0.5, "grad_norm": 1.2, "lr": 2.5e-5, "step_s": 0.8, "memory_mib": None})
    sink({"event": "eval", "step": 6, "eval_s": 3.0, "train": {"loss": 0.4, "joint_abs_err": 0.05, "decode": {"failures": 0, "strings": ["x"]}}})
    sink({"event": "checkpoint", "step": 6, "path": "p", "save_s": 1.5, "size_mib": 900})
    sink({"event": "resume_snapshot", "step": 6, "save_s": 4.0, "size_mib": 5600})
    sink({"event": "stopped", "steps": 8, "final_loss": 0.5, "seconds": 90.0, "stopped": True, "last_step": 6, "checkpoints": []})
    assert run.summary == {"trainable_m": 442.7, "frozen_m": 2508.5, "train_steps": 300, "val_steps": 0,
                           "final_loss": 0.5, "seconds": 90.0, "stopped": True, "last_step": 6}
    assert run.logs[0] == {"step": 4, "resume/from_step": 4}
    assert run.logs[1] == {"step": 5, "perf/seconds": 61.0, "perf/memory_mib/in_use": 100, "perf/memory_mib/peak": 200}      # the first step after the resume
    assert run.logs[2] == {"step": 6, "train/loss": 0.5, "train/grad_norm": 1.2, "train/lr": 2.5e-5, "train/step_s": 0.8}
    assert run.logs[3] == {"step": 6, "eval/eval_s": 3.0, "eval/train/loss": 0.4, "eval/train/joint_abs_err": 0.05, "eval/train/decode/failures": 0}
    assert run.logs[4] == {"step": 6, "checkpoint/save_s": 1.5, "checkpoint/size_mib": 900}
    assert run.logs[5] == {"step": 6, "snapshot/save_s": 4.0, "snapshot/size_mib": 5600}
    assert len(run.logs) == 6
    sink.close()
    assert run.finished == [0]


def test_camera_views_put_the_cameras_side_by_side_one_picture_per_row(fake, tmp_path):
    sink = tracking.start_tracking(tmp_path, False, "proj", "name", "online")
    images = {"base": np.full((7, 4, 6, 3), -1.0, np.float32), "wrist": np.ones((7, 4, 6, 3), np.float32)}      # the model's input: floats in [-1, 1]
    sink.log_camera_views(0, images)
    point = fake.runs[0].logs[-1]
    assert point["step"] == 0 and len(point["camera_views"]) == 5                           # openpi's five rows
    assert point["camera_views"][0] == ("image", (4, 12, 3), "uint8")                       # two cameras of 6 pixels, side by side, as 8-bit pictures


def test_a_failing_sink_is_dropped_and_the_log_goes_on(tmp_path, capsys):
    class Broken:
        calls = 0

        def __call__(self, fields):
            Broken.calls += 1
            raise ConnectionError("wandb is gone")

        def close(self):
            raise AssertionError("a dropped sink is not closed")

    log = sft.MetricsLog(tmp_path / "metrics.jsonl", Broken())
    log.write(event="train", step=1, loss=1.0)
    log.write(event="train", step=2, loss=0.9)
    log.close()
    assert Broken.calls == 1 and "logging stopped" in capsys.readouterr().err
    assert [json.loads(line)["step"] for line in (tmp_path / "metrics.jsonl").read_text().splitlines()] == [1, 2]


def test_the_config_knows_the_wandb_modes():
    cfg = sft.load_config()
    assert cfg.wandb_mode in tracking.MODES and cfg.wandb_project
    with pytest.raises(ValueError, match="wandb_mode"):
        sft.load_config(overrides={"wandb_mode": "sometimes"})


# -- the wandb view of the RL runs -----------------------------------------------------------------------------------------------------------------------------------------------------

ROUND = {"event": "round", "step": 3, "round_id": 5, "success_rate": 0.5, "episodes": 50, "plans_mean": 7.5, "steps": 150, "env_steps": 900, "model_s": 12.0, "wall_s": 80.0,
         "opt_candidate_q_std_mean": 0.3, "opt_chosen_q_gain_mean": 0.2, "critic_updates": 900, "critic": {"loss": 1.0}, "buffer_rows": 400, "online_episodes": 150,
         "online_success_rate": 0.4, "vla_version": 3, "round_wall_s": 400.0}


def test_the_rl_events_are_sent_grouped_by_what_they_measure_without_constants_or_copies(fake, tmp_path):
    sink = tracking.start_tracking(tmp_path, False, "proj", "name", "online", axes=tracking.RL_AXES)
    run = fake.runs[0]
    sink({"event": "warmup", "step": 0, "round_id": 0, "success_rate": 0.3, "episodes": 50, "plans_mean": 7.0, "steps": 150, "env_steps": 800, "model_s": 1.0, "wall_s": 10.0})     # a collection: not sent
    sink({"event": "warmup", "step": 0, "summary": True, "episodes": 50, "success_rate": 0.3, "env_steps": 800, "buffer_rows": 350, "critic_updates": 5000, "critic": {"loss": 2.0}, "wall_s": 900.0})
    sink({"event": "critic", "stage": 0, "update": 500, "loss": 1.5, "q_data_mean": 0.1})
    sink(ROUND)
    sink({"event": "distill", "step": 3, "label_candidate_q_mean": 0.1, "label_chosen_q_mean": 0.4, "states": 400, "steps": 20, "loss_first": 1.0, "loss_last": 0.5, "grad_norm_mean": 0.3,
          "annotate_s": 30.0, "train_s": 200.0, "step_s": 10.0, "vla_updates": 60})
    sink({"event": "refresh_progress", "round": 3, "done": 100, "total": 400})
    sink({"event": "refresh", "step": 3, "version": 3, "states": 400, "seconds": 600.0, "model_ms": 500000.0})
    sink({"event": "eval", "step": 3, "n_episodes": 100, "n_success": 60, "success_rate": 0.6, "success_se": 0.049, "steps_mean": 80.0, "steps_mean_success": 60.0, "wall_s_total": 400.0, "wall_s_batch": 399.0,
          "optimisation": {"candidate_q_mean": 0.1, "chosen_q_gain_mean": 0.3}, "round": 3, "critic_updates": 2700, "buffer_rows": 400, "vla_version": 3, "eval_wall_s": 450.0})
    logs = run.logs
    assert len(logs) == 6                                                           # the warm-up summary, a critic window, round, distill, refresh, eval
    assert logs[0] == {"step": 0, "warmup/success_rate": 0.3, "warmup/env_steps": 800, "warmup/buffer_rows": 350, "time/warmup_s": 900.0, "warmup/success_se": pytest.approx(np.sqrt(0.3 * 0.7 / 50))}
    assert logs[1] == {"critic/loss": 1.5, "critic/q_data_mean": 0.1, "critic_update": 500}                  # its own axis: no `step`
    assert logs[2]["step"] == 3 and logs[2]["rollout/success_rate"] == 0.5 and logs[2]["rollout/success_se"] == pytest.approx(np.sqrt(0.25 / 50))
    assert logs[2]["rollout/opt/chosen_q_gain_mean"] == 0.2 and logs[2]["rollout/online_success_rate_cumulative"] == 0.4 and logs[2]["time/round_s"] == 400.0
    assert not {"rollout/steps", "rollout/episodes", "rollout/round_id", "vla/version"} & set(logs[2]) and not any(k.startswith("critic") for k in logs[2])
    assert logs[3]["distill/labels/chosen_q_mean"] == 0.4 and logs[3]["time/distill_train_s"] == 200.0 and logs[3]["distill/loss_last"] == 0.5 and logs[3]["distill/vla_updates"] == 60
    assert logs[4] == {"step": 3, "refresh/states": 400, "time/refresh_s": 600.0, "time/refresh_model_s": 500.0}
    assert logs[5] == {"step": 3, "eval/success_rate": 0.6, "eval/success_se": 0.049, "eval/steps_mean": 80.0, "eval/steps_mean_success": 60.0, "eval/opt/candidate_q_mean": 0.1,
                       "eval/opt/chosen_q_gain_mean": 0.3, "time/eval_s": 450.0}
    assert sink.step == 3


def test_the_critic_axis_is_defined_before_the_star_because_wandb_applies_the_first_pattern_that_matches(fake, tmp_path):
    tracking.start_tracking(tmp_path, False, "proj", "name", "online", axes=tracking.RL_AXES)
    names = [n for n, _ in fake.runs[0].metric_defs]
    assert names.index("critic/*") < names.index("*") and ("critic/*", {"step_metric": "critic_update"}) in fake.runs[0].metric_defs
    (tmp_path / "sft").mkdir()
    assert tracking.start_tracking(tmp_path / "sft", False, "proj", "name", "online") and [n for n, _ in fake.runs[1].metric_defs] == ["step", "*"]      # the SFT is as it was


def test_an_sft_evaluation_keeps_the_generic_view(fake, tmp_path):
    sink = tracking.start_tracking(tmp_path, False, "proj", "name", "online")
    sink({"event": "eval", "step": 6, "eval_s": 3.0, "train": {"loss": 0.4}})
    assert fake.runs[0].logs == [{"step": 6, "eval/eval_s": 3.0, "eval/train/loss": 0.4}]
