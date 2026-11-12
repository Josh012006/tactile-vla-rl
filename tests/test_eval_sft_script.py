"""scripts/eval_sft.py (the run, its checkpoints, the protocol, --dry-run, --plot-only, the lock) and scripts/probe_eval_parallel.py (the analysis), on CPU, without a model or a server."""
import json

import pytest

from scripts import eval_sft as ES
from scripts import probe_eval_parallel as PE
from src.eval import episodes as E
from test_eval_episodes import record, write_results


def make_run(tmp_path, steps=(500, 1000, 1500), horizon=16, complete=True):
    run = tmp_path / "runs" / "sft" / "peg_insert" / "peg_insert_pi05_lora_teleop"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"model_name": "pi05", "mode": "lora", "model_cfg": {"name": "pi05", "family": "pi05", "action_horizon": horizon, "output_semantics": "x"},
                                                 "sft": {"seed": 0}}))
    for s in steps:
        (run / "checkpoints" / str(s)).mkdir(parents=True)
        if complete:
            (run / "checkpoints" / str(s) / "meta.json").write_text("{}")
    return run


# -- the run and its checkpoints --------------------------------------------------------------------------------------------------------
def test_a_run_is_found_by_path_or_by_name_and_only_complete_checkpoints_count(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    (run / "checkpoints" / "2000").mkdir()                                            # being written: no meta.json yet
    (run / "checkpoints" / "notes").mkdir()
    assert ES.find_run_dir(str(run)) == run.resolve()
    monkeypatch.setattr(ES, "ROOT", tmp_path)
    assert ES.find_run_dir("peg_insert_pi05_lora_teleop") == run.resolve() and ES.find_run_dir("runs/sft/peg_insert/peg_insert_pi05_lora_teleop") == run.resolve()
    with pytest.raises(SystemExit, match="no SFT run"):
        ES.find_run_dir("nope")
    assert list(ES.discover_checkpoints(run)) == [500, 1000, 1500]


def test_the_default_list_is_the_initial_state_then_every_checkpoint_and_steps_picks_some(tmp_path):
    found = ES.discover_checkpoints(make_run(tmp_path))
    jobs = ES.select_jobs(found, None, no_initial=False)
    assert [(j.label, j.step, j.checkpoint is None) for j in jobs] == [("0", 0, True), ("500", 500, False), ("1000", 1000, False), ("1500", 1500, False)]
    assert [j.step for j in ES.select_jobs(found, None, no_initial=True)] == [500, 1000, 1500]
    assert [j.step for j in ES.select_jobs(found, [1500, 0, 500, 500], no_initial=True)] == [0, 500, 1500]       # --steps is exact: 0 is there because it was asked for
    with pytest.raises(SystemExit, match=r"\[700\] not found"):
        ES.select_jobs(found, [700], False)


# -- protocol ---------------------------------------------------------------------------------------------------------------------------------------
def args_with(**kw):
    import argparse
    return argparse.Namespace(**{"seed": 42, "replan": None, **kw})


def test_the_protocol_has_one_environment_the_episode_length_of_the_task_and_half_the_chunk_as_replan(tmp_path):
    task = ES.task_settings(ES.ROOT / ES.DEFAULT_TASK, ["max_steps_per_episode=100", "success_hold_steps=2"])
    assert task["num_envs"] == 1 and task["max_steps_per_episode"] == 100 and task["success_hold_steps"] == 2 and "server_port" not in task
    for bad in ("num_envs=4", "server_port=9"):
        with pytest.raises(SystemExit, match="set by this script"):
            ES.task_settings(ES.ROOT / ES.DEFAULT_TASK, [bad])
    cfg = {"name": "pi0_fast", "action_horizon": 10}
    p = ES.build_protocol(args_with(), tmp_path, cfg, task)
    assert (p["replan"], p["max_steps"], p["num_envs"], p["master_seed"], p["recipe"]) == (5, 100, 1, 42, ES.RECIPE)
    assert ES.build_protocol(args_with(), tmp_path, {**cfg, "action_horizon": 16}, task)["replan"] == 8           # expo-ft's 8 of 16
    assert ES.build_protocol(args_with(replan=10), tmp_path, cfg, task)["replan"] == 10
    with pytest.raises(SystemExit, match="between 1 and"):
        ES.build_protocol(args_with(replan=11), tmp_path, cfg, task)


def test_xla_gets_deterministic_ops_once():
    assert ES.with_deterministic_xla("") == ES.XLA_DETERMINISTIC
    assert ES.with_deterministic_xla("--a=1") == "--a=1 " + ES.XLA_DETERMINISTIC
    assert ES.with_deterministic_xla("--a=1 " + ES.XLA_DETERMINISTIC) == "--a=1 " + ES.XLA_DETERMINISTIC


def test_force_sets_results_aside_and_never_overwrites(tmp_path):
    path = tmp_path / "0.jsonl"
    assert ES.set_aside(path) is None
    for k in (1, 2, 3):
        path.write_text(f"v{k}")
        moved = ES.set_aside(path)
        assert moved.name == f"0.jsonl.prev{k}" and not path.exists()
    assert [(tmp_path / f"0.jsonl.prev{k}").read_text() for k in (1, 2, 3)] == ["v1", "v2", "v3"]


def test_one_evaluation_at_a_time_per_output_folder(tmp_path):
    first = ES.acquire_run_lock(tmp_path / "eval")
    with pytest.raises(SystemExit, match="another evaluation is running"):
        ES.acquire_run_lock(tmp_path / "eval")
    first.close()
    ES.acquire_run_lock(tmp_path / "eval").close()                                  # released with the file


# -- the commands that need no GPU ---------------------------------------------------------------------------------------------------------------
def test_dry_run_lists_the_checkpoints_and_what_is_done_and_writes_nothing(tmp_path, capsys):
    run = make_run(tmp_path)
    log = E.EpisodeLog(run / "eval" / "results" / "500.jsonl")
    for i in range(3):
        log.append(record(i, seed=E.make_seeds(42, 10)[i]))
    assert ES.main(["--run", str(run), "--n-episodes", "10", "--dry-run", "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "step      0:    0/10" in out and "step    500:    3/10" in out and "step   1500:    0/10" in out and "initial state of the run" in out
    assert "replan 8" in out and "10 episodes per checkpoint, master seed 42, 2 worker(s)" in out and f"data in {run / 'eval'}" in out
    assert not (run / "eval" / E.SEEDS_FILE).exists() and not (run / "eval" / E.PROTOCOL_FILE).exists()


def test_a_folder_made_with_another_protocol_is_refused_before_anything_starts(tmp_path):
    run = make_run(tmp_path)
    task = ES.task_settings(ES.ROOT / ES.DEFAULT_TASK, [])
    protocol = ES.build_protocol(args_with(seed=7), run, json.loads((run / "config.json").read_text())["model_cfg"], task)
    E.atomic_write_json(run / "eval" / E.PROTOCOL_FILE, {**protocol, "attempts": []})
    with pytest.raises(SystemExit, match="master_seed"):
        ES.main(["--run", str(run), "--dry-run"])
    with pytest.raises(SystemExit, match="master_seed"):                                           # a real run too, and the seed file it did not write
        ES.main(["--run", str(run), "--n-episodes", "5"])
    ES.main(["--run", str(run), "--seed", "7", "--dry-run"])                                      # the protocol it was made with is accepted


def test_a_run_without_checkpoints_and_the_curve_only_mode(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    empty = make_run(tmp_path / "a", steps=(), complete=True)
    with pytest.raises(SystemExit, match="no checkpoint yet"):
        ES.main(["--run", str(empty), "--no-initial", "--dry-run"])
    run = make_run(tmp_path / "b")
    with pytest.raises(SystemExit, match="no complete result"):
        ES.main(["--run", str(run), "--plot-only"])
    write_results(run / "eval" / "results", [(0, 0.1), (500, 0.4), (1000, 0.7)], n=20)
    curve_dir = tmp_path / "curves"
    assert ES.main(["--run", str(run), "--plot-only", "--curve-dir", str(curve_dir), "--title", "Teleop demos"]) == 0
    out = capsys.readouterr().out
    assert (curve_dir / "curve.png").exists() and (curve_dir / "curve.pdf").exists() and (curve_dir / "curve.json").exists() and "70.0%" in out and str(curve_dir / "curve.png") in out
    assert not (run / "eval" / "curve.png").exists()                                                # the curve went where it was asked, the data stayed in the output folder


# -- the probe's analysis -----------------------------------------------------------------------------------------------------------------------
def recs(shas, wall=10.0):
    return {i: {"traj_sha": s, "wall_s": wall + (100 if i == 0 else 0)} for i, s in enumerate(shas)}


def test_the_probe_compares_every_replica_and_run_with_the_reference_and_computes_the_throughput():
    ref = ["a", "b", "c", "d"]
    runs = {1: {"r0": recs(ref, wall=10.0)},
            2: {"r0": recs(ref, wall=12.5), "r1": recs(ref, wall=12.5)},
            3: {"r0": recs(ref, wall=15.0), "r1": recs(["a", "b", "X", "d"], wall=15.0), "r2": recs(ref, wall=15.0)}}
    a = PE.analyse(runs, resume={"cut_after": 2, "records": recs(["a", "b", "c", "Y"])})
    assert a["reference"] == "W=1 r0"
    w1, w2, w3 = (a["workers"][k] for k in ("1", "2", "3"))
    assert w1["episode_wall_s_median"] == 10.0 and w1["episodes_per_hour"] == 360.0 and w1["speedup_vs_first"] == 1.0         # the first episode (compile) is left out
    assert w2["episodes_per_hour"] == 576.0 and w2["speedup_vs_first"] == 1.6 and w2["identical_to_reference"]
    assert w3["episodes_per_hour"] == 720.0 and not w3["identical_to_reference"] and w3["differing_episodes"] == {"r0": [], "r1": [2], "r2": []}
    assert a["all_workers_identical"] is False
    assert a["resume"] == {"cut_after": 2, "episodes_after_the_cut": 2, "differing_after_the_cut": [3], "identical_after_the_cut": False}
    same = PE.analyse({1: {"r0": recs(ref)}, 2: {"r0": recs(ref), "r1": recs(ref)}}, resume={"cut_after": 1, "records": recs(ref)})
    assert same["all_workers_identical"] and same["resume"]["identical_after_the_cut"]
