"""The three FORGE tasks (PegInsert, GearMesh, NutThread) in the pipeline: the task YAMLs, what FORGE calls a success, the SFT experiments, how a run carries its task to the evaluation
and the RL, and the teleop client (its default folder, the yaw keys). CPU only: nothing here starts Isaac Sim (the cluster tests are tests/test_isaac_*.py --isaac)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.envs.task_geometry import SUCCESS_XY, TASKS, success_z, task_geometry

REPO = Path(__file__).resolve().parents[1]
NEW_TASKS = ("gear_mesh", "nut_thread")
PROMPTS = {"gear_mesh": "Mesh the gear with the other gears.", "nut_thread": "Screw the nut onto the bolt."}
FORGE_IDS = {"peg_insert": "Isaac-Forge-PegInsert-Direct-v0", "gear_mesh": "Isaac-Forge-GearMesh-Direct-v0", "nut_thread": "Isaac-Forge-NutThread-Direct-v0"}


def task_yaml(task):
    return yaml.safe_load((REPO / "configs" / "task" / f"{task}_forge.yaml").read_text())


# -- the task YAMLs ------------------------------------------------------------------------------------------------------------------------
def test_the_three_task_yamls_differ_only_in_what_belongs_to_the_task():
    peg = task_yaml("peg_insert")
    for task in NEW_TASKS:
        cfg = task_yaml(task)
        assert cfg["name"] == f"{task}_forge" and cfg["task_name"] == FORGE_IDS[task]
        assert cfg["language_instruction"] == PROMPTS[task] != peg["language_instruction"]
        different = {k for k in set(cfg) | set(peg) if cfg.get(k) != peg.get(k)}
        assert different == {"name", "task_name", "language_instruction", "max_steps_per_episode"}          # robot, gains, cameras, rate, port: the same, the scripts are the same
    assert task_yaml("gear_mesh")["max_steps_per_episode"] == 300 and task_yaml("nut_thread")["max_steps_per_episode"] == 450      # FORGE's 20 s and 30 s at 15 Hz
    assert peg["max_steps_per_episode"] == 150
    assert len({task_yaml(t)["language_instruction"] for t in TASKS}) == 3


# -- what FORGE calls a success -------------------------------------------------------------------------------------------------------------
def test_the_success_height_is_the_hole_depth_for_peg_and_gear_and_the_thread_pitch_for_the_nut():
    fixed = SimpleNamespace(height=0.02, thread_pitch=0.0015)
    assert success_z("peg_insert", fixed, 0.04) == pytest.approx(0.0008)
    assert success_z("gear_mesh", fixed, 0.04) == pytest.approx(0.0008)
    assert success_z("nut_thread", fixed, 0.375) == pytest.approx(0.0015 * 0.375)                          # NOT height * threshold: the nut only goes down by about a pitch
    assert task_geometry("nut_thread", fixed, 0.375) == {"hole_depth": 0.02, "success_xy": SUCCESS_XY, "success_z": pytest.approx(0.0015 * 0.375)}
    with pytest.raises(NotImplementedError, match="known tasks"):
        success_z("other", fixed, 1.0)
    assert set(TASKS) == set(FORGE_IDS)


# -- the SFT experiments of the new tasks ---------------------------------------------------------------------------------------------------
def new_experiments():
    return sorted((REPO / "configs" / "experiment" / "sft").glob("gear_mesh_*.yaml")) + sorted((REPO / "configs" / "experiment" / "sft").glob("nut_thread_*.yaml"))


def test_each_new_task_has_a_pi05_and_a_pi0_fast_experiment_on_its_own_demos_and_its_own_runs_folder():
    from scripts import sft as cli
    from src.algos import sft
    from src.models.interface import ROOT, apply_overrides, load_model_cfg, norm_stats_source
    from src.models.settings import parse_model_settings
    files = new_experiments()
    assert [f.stem for f in files] == sorted(f"{t}_{m}_lora_teleop" for t in NEW_TASKS for m in ("pi05", "pi0_fast"))
    for f in files:
        exp = cli.load_experiment(f)
        task = next(t for t in NEW_TASKS if f.stem.startswith(t))
        model = f.stem[len(task) + 1:].removesuffix("_lora_teleop")
        camel = "".join(p.capitalize() for p in task.split("_"))
        assert exp["name"] == f.stem and exp["model"] == model and exp["full"] is False
        assert exp["task"] == f"configs/task/{task}_forge.yaml" and (REPO / exp["task"]).is_file()
        assert exp["demos"] == [f"demos/isaaclab/Forge{camel}_Robotiq_teleop"]
        assert exp["set"]["output_root"] == f"runs/sft/{task}" and exp["set"]["seed"] == 42
        assert sft.load_config(overrides=exp["set"]).output_root == f"runs/sft/{task}"
        cfg = load_model_cfg(model)
        merged = apply_overrides(cfg, cli.model_overrides(cfg, exp["full"], exp["model_set"]))
        settings = parse_model_settings(merged)
        assert settings.any_lora and not settings.active_touch
        stats = norm_stats_source(merged)
        if model == "pi05":
            assert stats == (ROOT / "assets" / "forge_demo_norm", f"forge_demos_robotiq_{task}_teleop")      # THIS task's statistics, not PegInsert's
        else:
            assert stats is None                                                                           # the checkpoint's own, as PegInsert's pi0-FAST


def test_the_prompt_of_the_demonstrations_has_to_be_the_one_of_the_task_yaml(tmp_path):
    from scripts import sft as cli
    assert cli.check_task(None, ["anything"]) is None                                                       # no task: PegInsert as before, nothing checked
    assert cli.check_task("configs/task/gear_mesh_forge.yaml", [PROMPTS["gear_mesh"]])["task_name"] == FORGE_IDS["gear_mesh"]
    with pytest.raises(SystemExit, match="language_instruction"):
        cli.check_task("configs/task/gear_mesh_forge.yaml", ["Insert the peg into the socket."])           # the demonstrations of another task
    with pytest.raises(SystemExit, match="language_instruction"):
        cli.check_task("configs/task/gear_mesh_forge.yaml", [PROMPTS["gear_mesh"], "another wording"])
    with pytest.raises(SystemExit, match="no task YAML"):
        cli.check_task("configs/task/nope.yaml", [PROMPTS["gear_mesh"]])
    bare = tmp_path / "t.yaml"
    bare.write_text("name: x\n")
    with pytest.raises(SystemExit, match="no `language_instruction`"):
        cli.check_task(str(bare), ["x"])


def test_the_dry_run_of_a_new_task_checks_the_prompt_and_says_the_task_and_the_folder(tmp_path, capsys):
    from scripts import sft as cli
    from test_sft_demos import write_demo
    good, bad = tmp_path / "ForgeGearMesh_Robotiq_teleop", tmp_path / "ForgePegInsert_Robotiq_teleop"
    for folder, prompt in ((good, PROMPTS["gear_mesh"]), (bad, "Insert the peg into the socket.")):
        for i, steps in enumerate((6, 5)):
            write_demo(folder, f"ep_{i:04d}_seed{i}.npz", steps, seed=i, prompt=prompt)
    experiment = str(REPO / "configs/experiment/sft/gear_mesh_pi0_fast_lora_teleop.yaml")
    small = ["--set", "steps=20", "--set", "batch_size=4", "--set", "micro_batch_size=2"]
    assert cli.main(["--experiment", experiment, "--demos", str(good), "--dry-run", *small]) == 0
    out = capsys.readouterr().out
    assert "task: configs/task/gear_mesh_forge.yaml" in out and "runs/sft/gear_mesh/gear_mesh_pi0_fast_lora_teleop" in out and PROMPTS["gear_mesh"] in out
    with pytest.raises(SystemExit, match="language_instruction"):
        cli.main(["--experiment", experiment, "--demos", str(bad), "--dry-run", *small])
    assert cli.main(["--model", "pi0_fast", "--demos", str(bad), "--dry-run", *small]) == 0               # a probe without a task: nothing to check, as before
    assert "none (PegInsert" in capsys.readouterr().out


def test_an_experiment_task_is_a_path_and_a_resume_takes_the_runs_own(tmp_path):
    from scripts import sft as cli
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: a\nmodel: pi05\ntask: 3\n")
    with pytest.raises(SystemExit, match="`task`"):
        cli.load_experiment(bad)
    file = new_experiments()[0]
    with pytest.raises(SystemExit, match="only --set is accepted"):
        cli.main(["--resume", "x", "--task", "configs/task/gear_mesh_forge.yaml"])
    assert cli.load_experiment(file)["task"].endswith("_forge.yaml")


def test_run_names_and_demo_tags_do_not_depend_on_the_task():
    from scripts import sft as cli
    assert [cli.demos_tag(f"demos/isaaclab/{p}") for p in ("ForgePegInsert_Robotiq_noisy", "ForgeGearMesh_Robotiq_teleop", "ForgeNutThread_Robotiq_scripted", "mine")] == ["noisy", "teleop", "scripted", "mine"]
    name = cli.run_name_of("pi05", ["demos/isaaclab/ForgeNutThread_Robotiq_teleop", "demos/isaaclab/ForgeNutThread_Robotiq_noisy"], "lora")
    assert name.startswith("pi05_teleop+noisy_lora_")


# -- a run knows its task: the evaluation and the RL use the same env YAML, without being told --------------------------------------------------
def make_runs(root):
    for kind, task in (("sft", "peg_insert"), ("sft", "gear_mesh"), ("sft", "probing"), ("rl", "nut_thread"), ("rl", "probing")):
        run = root / "runs" / kind / task / f"{task}_{kind}_run"
        run.mkdir(parents=True)
        (run / "config.json").write_text("{}")
    return root


def test_runs_are_found_by_name_under_every_task_folder_and_the_probes_come_last(tmp_path, monkeypatch):
    from scripts import eval_rl, eval_sft
    from scripts import sft as cli
    root = make_runs(tmp_path)
    for module in (eval_sft, eval_rl, cli):
        monkeypatch.setattr(module, "ROOT", root)
    assert [p.name for p in eval_sft.run_roots("sft")] == ["gear_mesh", "peg_insert", "probing"]
    assert [p.name for p in eval_sft.run_roots("rl")] == ["nut_thread", "probing"]
    assert eval_sft.find_run_dir("gear_mesh_sft_run") == (root / "runs/sft/gear_mesh/gear_mesh_sft_run").resolve()
    assert eval_sft.find_run_dir("peg_insert_sft_run").parent.name == "peg_insert" and eval_sft.find_run_dir("probing_sft_run").parent.name == "probing"
    assert eval_rl.find_rl_run_dir("nut_thread_rl_run").parent.name == "nut_thread"
    assert cli.find_run_dir("gear_mesh_sft_run", "runs/sft/probing") == (root / "runs/sft/gear_mesh/gear_mesh_sft_run").resolve()      # --resume by name: output_root first, then every task folder
    with pytest.raises(SystemExit, match="runs/sft/"):
        eval_sft.find_run_dir("nope")
    with pytest.raises(SystemExit, match="runs/rl/"):
        eval_rl.find_rl_run_dir("nope")


def test_the_env_yaml_is_the_asked_one_then_the_runs_then_peg_insert(tmp_path, monkeypatch):
    from scripts import eval_sft
    monkeypatch.setattr(eval_sft, "ROOT", REPO)
    assert eval_sft.resolve_task_config(None, None) == REPO / eval_sft.DEFAULT_TASK                         # a run that recorded nothing (all the PegInsert runs of before) is PegInsert
    assert eval_sft.resolve_task_config(None, "configs/task/nut_thread_forge.yaml") == REPO / "configs/task/nut_thread_forge.yaml"
    assert eval_sft.resolve_task_config(None, None, "configs/task/gear_mesh_forge.yaml").name == "gear_mesh_forge.yaml"       # the first recorded one
    assert eval_sft.resolve_task_config("configs/task/peg_insert_forge.yaml", "configs/task/nut_thread_forge.yaml").name == "peg_insert_forge.yaml"    # the command line wins
    other = tmp_path / "mine.yaml"
    other.write_text("name: mine\n")
    assert eval_sft.resolve_task_config(str(other)) == other
    with pytest.raises(SystemExit, match="no task YAML"):
        eval_sft.resolve_task_config("configs/task/nope.yaml")


def test_the_sft_evaluation_uses_the_task_its_run_recorded(tmp_path, monkeypatch, capsys):
    from scripts import eval_sft
    from test_eval_sft_script import make_run
    run = make_run(tmp_path)
    seen = []
    original = eval_sft.task_settings
    monkeypatch.setattr(eval_sft, "task_settings", lambda task_config, *a, **k: (seen.append(Path(task_config).name), original(task_config, *a, **k))[1])
    for recorded, expected in ((None, "peg_insert_forge.yaml"), ("configs/task/nut_thread_forge.yaml", "nut_thread_forge.yaml")):
        config = json.loads((run / "config.json").read_text())
        config["task"] = recorded
        (run / "config.json").write_text(json.dumps(config))
        assert eval_sft.main(["--run", str(run), "--dry-run", "--n-episodes", "4", "--envs-per-server", "4", "--output-dir", str(tmp_path / f"out_{expected}")]) == 0
        assert seen[-1] == expected
    assert eval_sft.main(["--run", str(run), "--dry-run", "--n-episodes", "4", "--envs-per-server", "4", "--task-config", "configs/task/gear_mesh_forge.yaml",
                          "--output-dir", str(tmp_path / "out_asked")]) == 0
    assert seen[-1] == "gear_mesh_forge.yaml"                                                               # the command line wins over the run's
    capsys.readouterr()


# -- the teleop client -----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    spec = importlib.util.spec_from_file_location("teleop_client", REPO / "scripts" / "demos" / "teleop_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_default_demo_folder_is_the_one_of_the_servers_task(client):
    assert client.default_out_dir("peg_insert") == "demos/isaaclab/ForgePegInsert_Robotiq_teleop"
    assert client.default_out_dir("gear_mesh") == "demos/isaaclab/ForgeGearMesh_Robotiq_teleop"
    assert client.default_out_dir("nut_thread") == "demos/isaaclab/ForgeNutThread_Robotiq_teleop"
    assert client.default_out_dir(None) == client.default_out_dir("peg_insert")                             # a server that does not say: the PegInsert one it always was


def test_yaw_is_a_rotation_about_the_vertical_counter_clockwise_seen_from_above(client):
    q = client.yaw_quat(np.pi / 2)
    assert np.allclose(q, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    rot = client.quat_to_matrix(q)
    assert np.allclose(rot @ [1, 0, 0], [0, 1, 0]) and np.allclose(rot @ [0, 0, 1], [0, 0, 1])           # x -> y, z stays
    hand = np.array([0.0, 1.0, 0.0, 0.0])                                                                   # pointing down (a half turn about x)
    turned = client.quat_mul(client.yaw_quat(0.3), hand)
    assert np.isclose(np.linalg.norm(turned), 1.0)
    assert np.allclose(client.quat_to_matrix(turned), client.quat_to_matrix(client.yaw_quat(0.3)) @ client.quat_to_matrix(hand))     # about the FIXED vertical, applied after the hand's own rotation
    assert np.allclose(client.quat_mul(client.yaw_quat(0.0), hand), hand)


def test_q_and_e_turn_the_hand_by_the_rate_and_fine_and_fast_scale_it(client):
    inp = client.Inputs
    assert client.build_yaw(inp(), 3.0) == 0.0 and not inp().any()
    assert client.build_yaw(inp(yaw_left=True), 3.0) == pytest.approx(np.deg2rad(3.0))
    assert client.build_yaw(inp(yaw_right=True), 3.0) == pytest.approx(-np.deg2rad(3.0))
    assert client.build_yaw(inp(yaw_left=True, yaw_right=True), 3.0) == 0.0
    assert client.build_yaw(inp(yaw_left=True, fine=True), 4.0) == pytest.approx(np.deg2rad(1.0))
    assert client.build_yaw(inp(yaw_right=True, fast=True), 4.0) == pytest.approx(-np.deg2rad(10.0))
    assert inp(yaw_left=True).any() and inp(yaw_right=True).any()                                            # a turn alone is a step


class FakeTeleopEnv:
    def __init__(self):
        self.sent = []

    def reset(self, seed):
        return self._obs()

    def _obs(self):
        return {"observation/exterior_image_1_left": np.zeros((4, 4, 3), np.uint8), "observation/wrist_image_left": np.zeros((4, 4, 3), np.uint8),
                "observation/joint_position": np.zeros(7, np.float32), "observation/gripper_position": np.zeros(1, np.float32), "extra/ft_force": np.zeros(3, np.float32),
                "extra/peg_socket_offset": np.array([0.001, 0.0, 0.002], np.float32)}

    def debug(self, name, **_):
        return {"fingertip_pose": {"quat": [0.0, 1.0, 0.0, 0.0]}, "robot_info": {"name": "fake", "task": "nut_thread"}}[name]

    def teleop_step(self, delta, quat, cmd_gain, max_joint_step):
        self.sent.append((np.asarray(delta), np.asarray(quat)))
        return {"obs": self._obs(), "action": np.zeros(8), "clipped": False, "inserted": False, "terminated": False, "truncated": False}


def test_a_step_sends_the_orientation_held_turned_by_the_yaw_so_far_and_the_demo_says_how_much(client):
    env = FakeTeleopEnv()
    args = SimpleNamespace(view="wrist", flip_x=False, flip_y=False, rotate_deg=0.0, cmd_gain=1.0, max_joint_step=0.05, max_steps=100, out_dir="x")
    session = client.Session(env, args)
    session.start(100)
    hand = np.array([0.0, 1.0, 0.0, 0.0])
    axes_before = [a.copy() for a in session.axes]
    session.step(np.zeros(3))                                                                               # no key: the orientation of the start, exactly (as before Q / E existed)
    assert np.array_equal(env.sent[-1][1], hand) and session.yaw == 0.0
    session.step(np.zeros(3), np.deg2rad(10.0))
    session.step(np.zeros(3), np.deg2rad(5.0))
    assert np.allclose(env.sent[-1][1], client.quat_mul(client.yaw_quat(np.deg2rad(15.0)), hand))           # the yaws add up
    assert not np.allclose(axes_before[0], session.axes[0])                                                 # the wrist camera turned with the hand: so did the mouse directions
    assert session.info()["yaw_deg"] == pytest.approx(15.0, abs=0.05) and session.idle_steps == 1          # a turn is not an idle step; only the first, motionless one is
    session.start(101)
    assert session.yaw == 0.0                                                                               # a new episode starts from the held orientation again
