"""The demonstration tools (scripts/demos, src/envs/demo_store.py, the teleop op of the backend, the video panels). Pure: no GPU, no Isaac Sim.

What cannot be tested here and is (the ops that read the simulator: `ik_joint_target`, `finger_width`, `finger_distance`): the first
`sbatch jobs/job_demos.sh collect --num-episodes 2` on the cluster. Everything around them is tested with a kinematic stand-in of one
environment (FakeDemoEnv: the peg moves by exactly the move that is commanded)."""
import io
import json
import threading
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from fakes import make_backend, make_manager
from scripts.demos import collect_waypoint_demos as collect
from scripts.demos import replay_demos
from scripts.demos import teleop_client as tc
from scripts.demos.single_env import SingleEnv, Status
from src.envs import protocol as P
from src.envs import wire
from src.envs.backend import ManagerBackend
from src.envs.demo_store import save_teleop_demo
from src.envs.ws_server import WsEnvServer
from src.eval import video_panels as vp


# ------------------------------------------------------------------------------------------------------------ a stand-in environment
class FakeDemoEnv:
    """What SingleEnv offers, with a kinematic peg: `ik_joint_target` returns q + the commanded move on joints 0-2, `step` moves the peg by
    what the joints moved. FORGE's geometry: offset z = 25 mm is the rim of the hole, inserted = xy < 2.5 mm and z < 12.5 mm."""
    prompt, max_steps = "Insert the peg into the socket.", 150

    def __init__(self, *args, hold=5, **kwargs):
        self.hold, self.q, self.offset, self.t, self.run = hold, np.zeros(7), np.zeros(3), 0, 0
        self.last = Status(False, False, False, False, False, 0)

    def reset(self, seed):
        rng = np.random.default_rng(int(seed))
        self.q, self.t, self.run = np.zeros(7), 0, 0
        self.offset = np.array([rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), 0.030])
        self.last = Status(False, False, False, False, False, 0)
        return self.observation()

    def observation(self):
        return {P.EXTERIOR_IMAGE: np.zeros((8, 8, 3), np.uint8), P.WRIST_IMAGE: np.zeros((8, 8, 3), np.uint8),
                P.JOINT_POSITION: self.q.astype(np.float32), P.GRIPPER_POSITION: np.array([0.9], np.float32),
                P.FT_FORCE: np.zeros(3, np.float32), P.PEG_SOCKET_OFFSET: self.offset.astype(np.float32),
                P.TACTILE: np.zeros((2, 7, 4), np.float32)}

    def step(self, action):
        action = np.asarray(action, np.float64)
        self.offset = self.offset + (action[:3] - self.q[:3])
        self.q = action[:7].copy()
        self.t += 1
        inserted = bool(np.hypot(*self.offset[:2]) < 0.0025 and self.offset[2] < 0.0125)
        self.run = self.run + 1 if inserted else 0
        terminated, truncated = self.run >= self.hold, self.t >= self.max_steps and self.run < self.hold
        self.last = Status(terminated or truncated, inserted, terminated, truncated, terminated, self.t)

    def status(self):
        return self.last

    def debug(self, kind, **kw):
        if kind == "finger_width":
            return np.float32(0.0065)
        if kind == "fingertip_pose":
            return {"pos": np.zeros(3, np.float32), "quat": np.array([0.0, 1.0, 0.0, 0.0], np.float32)}
        if kind == "task_geometry":
            return {"hole_depth": 0.025, "success_xy": 0.0025, "success_z": 0.0125}
        if kind == "ik_joint_target":
            dq = np.zeros(7)
            dq[:3] = np.asarray(kw["delta_pos"], np.float64).reshape(3)
            return {"q_now": self.q.astype(np.float32), "q_target": (self.q + dq).astype(np.float32)}
        raise AssertionError(kind)


@pytest.fixture
def fake_env(monkeypatch):
    for module in (collect, replay_demos):
        monkeypatch.setattr(module, "SingleEnv", FakeDemoEnv)


def _demos(folder):
    return sorted(Path(folder).glob("ep_*.npz"))


# ------------------------------------------------------------------------------------------------- scripted demos, noise, replay
def test_scripted_demos_are_recorded_and_replay_exactly(fake_env, tmp_path, capsys):
    out = tmp_path / "clean"
    collect.main(["--out", str(out), "--num-episodes", "3", "--seed", "10"])
    files = _demos(out)
    assert [f.name for f in files] == [f"ep_000{i}_seed{10 + i}.npz" for i in range(3)]
    z = np.load(files[0])
    T = len(z["actions"])
    assert z["actions"].shape == (T, 8) and np.all(z["actions"][:, 7] == 1.0)           # absolute targets + the gripper kept closed
    assert z["exterior_image"].shape[0] == z["joint_position"].shape[0] == z["peg_socket_offset"].shape[0] == T
    assert z["tactile"].shape == (T, 2, 7, 4) and str(z["prompt"]) == FakeDemoEnv.prompt and int(z["seed"]) == 10
    assert "noise" not in z.files                                                       # no noise option: exactly the old files
    index = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()]
    assert len(index) == 3 and all(r["success"] and not r["gripper_moved"] for r in index)
    assert index[0]["waypoint_entered_at_step"].keys() >= {"lift", "align", "descend", "insert"}

    replay_demos.main(["--demos-dir", str(out)])
    report = json.loads((out / "replay_report.json").read_text())
    assert report["counts"] == {"OK": 3, "WARN": 0, "FAIL": 0}                          # the same seed and actions give the same episode
    assert max(d["drift_max_mm"] for d in report["demos"]) == 0.0


def test_noisy_demos_keep_clean_labels_and_interleave(fake_env, tmp_path):
    out = tmp_path / "noisy"
    # slow lift (1 mm per step) so that the burst, which starts at step 1, falls in `lift` where noise acts (it is silent in descend / insert)
    collect.main(["--out", str(out), "--num-episodes", "4", "--noise-fraction", "0.5", "--action-noise-sigma", "0.01",
                  "--noise-start-max", "1", "--lift-max-z-mm", "1"])
    names = [f.name for f in _demos(out)]
    assert [n.endswith("_noisy.npz") for n in names] == [False, True, False, True]      # clean and noisy alternate
    clean, noisy = np.load(out / names[0]), np.load(out / names[1])
    assert not bool(clean["noisy"]) and bool(noisy["noisy"]) and np.all(clean["noise"] == 0)
    assert np.any(noisy["noise"] != 0)                                                  # something really was added to what was SENT ...
    assert noisy["noise"].shape == (len(noisy["actions"]), 7)                           # ... and recorded apart: the labels stay clean
    assert np.all(noisy["actions"][:, 7] == 1.0)


def test_unit_ball_and_bursts_are_reproducible():
    z = np.array([np.linalg.norm(collect.unit_ball(np.random.default_rng(i), 7)) for i in range(200)])
    assert z.max() <= 1.0 and z.min() >= 0.0
    a = collect.burst_steps(np.random.default_rng(3), 1, 4, 8, 6)
    assert a == collect.burst_steps(np.random.default_rng(3), 1, 4, 8, 6) and len(a) == 4 and min(a) >= 1


def test_waypoints_go_lift_align_descend_insert():
    wps = collect.make_waypoints(0.025, 0.0002)
    assert [w.name for w in wps] == ["lift", "align", "descend", "insert"]
    below_rim = np.array([0.003, 0.0, 0.015])
    assert not wps[0].reached(below_rim) and wps[0].reached(np.array([0.003, 0.0, 0.035]))     # lift ends 10 mm above the rim
    assert wps[1].reached(np.array([0.0, 0.0, 0.035])) and not wps[1].reached(np.array([0.001, 0.0, 0.035]))


# -------------------------------------------------------------------------------------------------------------- the teleop op, saving
def test_teleop_step_is_one_ik_step_then_one_physics_step():
    mgr, sim = make_manager(num_envs=1, max_steps=10, hold=5, auto_reset=True)
    seen = {}

    def ik(request):
        seen.update(request)
        q_now = sim.q0[:, None] * np.ones((1, 7), np.float32)
        return {"q_now": q_now, "q_target": q_now + np.array([[0.3, 0, 0, 0, 0, 0, 0]], np.float32)}   # 0.3 rad asked: 0.05 allowed

    backend = ManagerBackend(mgr, sim.obs_spec(), robot={"name": "fake"}, debug_fn=ik)
    backend.reset({"seed": 0})
    out = backend.debug({"kind": "teleop_step", "delta_pos": np.zeros((1, 3), np.float32), "target_quat": np.zeros((1, 4), np.float32),
                         "cmd_gain": 1.0, "max_joint_step": 0.05})
    assert seen["kind"] == "ik_joint_target"
    assert out["action"].shape == (1, 8) and out["action"].dtype == np.float32
    assert out["action"][0, 0] == pytest.approx(0.05) and out["action"][0, 7] == 1.0     # q_now + the move capped at max_joint_step, gripper closed
    assert out["clipped"].tolist() == [True]
    assert sim.physics_steps == 1 and out["acting"].tolist() == [True]
    # never resets by itself: the episode ends (timeout), the observation is still its last one
    for _ in range(12):
        out = backend.debug({"kind": "teleop_step", "delta_pos": np.zeros((1, 3), np.float32), "target_quat": np.zeros((1, 4), np.float32)})
    assert out["all_done"] and out["truncated"].tolist() == [False] and sim.reset_calls == [0]


def _npz(**missing):
    arrays = {k: np.zeros((3, 2)) for k in ("exterior_image", "wrist_image", "joint_position", "gripper_position", "ft_force",
                                           "peg_socket_offset", "actions")}
    arrays.update(prompt=np.array("p"), seed=np.array(1))
    for k in missing:
        arrays.pop(k)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def test_save_teleop_demo_numbers_files_and_follows_a_symlinked_folder(tmp_path):
    root, scratch = tmp_path / "project", tmp_path / "scratch" / "demos"
    root.mkdir()
    scratch.mkdir(parents=True)
    (root / "demos").symlink_to(scratch)                                                 # `demos` is a link to scratch
    req = lambda **kw: {"out_dir": "demos/teleop", "seed": 7, "npz": _npz(), "info": {"seed": 7}, "success": True, **kw}
    a, b = save_teleop_demo(root, req()), save_teleop_demo(root, req(seed=8))
    assert (a["file"], b["file"]) == ("ep_0000_seed7.npz", "ep_0001_seed8.npz") and b["count"] == 2
    assert (scratch / "teleop" / "ep_0001_seed8.npz").exists()                           # written through the link
    f = save_teleop_demo(root, req(success=False))
    assert f["file"].startswith("fail_seed7_") and f["count"] == 2                       # failures are kept apart, not counted as demos
    lines = (scratch / "teleop" / "index.jsonl").read_text().splitlines()
    assert [json.loads(x)["file"] for x in lines][:2] == ["ep_0000_seed7.npz", "ep_0001_seed8.npz"]
    assert not list((scratch / "teleop").glob(".tmp_*"))                                 # nothing half written is left


def test_save_teleop_demo_refuses_what_is_not_a_demo_or_leaves_the_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    base = {"out_dir": "demos/t", "seed": 1, "info": {}, "success": True}
    assert "lacks" in save_teleop_demo(root, {**base, "npz": _npz(actions=None)})["error"]
    assert "inside the project" in save_teleop_demo(root, {**base, "npz": _npz(), "out_dir": "../elsewhere"})["error"]
    assert not (tmp_path / "elsewhere").exists() and not (root / "demos").exists()


# ----------------------------------------------------------------------------------------------------------------- SingleEnv, on a socket
class Running:
    def __init__(self, **kw):
        self.backend, self.sim = make_backend(**kw)
        self.server = WsEnvServer(self.backend, host="127.0.0.1", port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        assert self.server.ready.wait(10)

    def stop(self):
        self.server.stop()


@pytest.fixture
def server():
    made = []

    def make(**kw):
        made.append(Running(**kw))
        return made[-1]
    yield make
    for s in made:
        s.stop()


def test_single_env_refuses_a_server_that_is_not_set_up_for_the_scripts(server):
    s = server(num_envs=4, max_steps=10, hold=1)
    with pytest.raises(SystemExit) as e:
        SingleEnv("127.0.0.1", s.server.port, hold_steps=5)
    msg = str(e.value)
    assert "num_envs=1" in msg and "success_hold_steps=5" in msg and "jobs/env_server.sh" in msg


def test_single_env_runs_one_episode_without_auto_reset(server):
    s = server(num_envs=1, max_steps=6, hold=2)
    env = SingleEnv("127.0.0.1", s.server.port, hold_steps=2)
    assert env.max_steps == 6 and env.prompt == "insert the peg"
    obs = env.reset(0)
    assert obs[P.JOINT_POSITION].shape == (7,) and obs[P.EXTERIOR_IMAGE].ndim == 3        # one environment: no leading axis
    for _ in range(6):
        env.step(np.zeros(8, np.float32))                                               # never inserted: times out
    st = env.status()
    assert st.truncated and st.done and not st.terminated and st.t == 6
    assert s.sim.reset_calls == [0]                                                      # nothing restarted behind the script's back
    env.step(np.zeros(8, np.float32))                                                    # a step after the end is a no-op, not a reset
    assert s.sim.reset_calls == [0]
    env.close()


# ----------------------------------------------------------------------------------------------------------------------- video panels
def test_taxel_panel_frames_have_the_documented_sizes():
    ext = wrist = np.zeros((224, 224, 3), np.uint8)
    tac = np.zeros((2, 7, 4), np.float32)
    tac[0, 0, :2] = 3.0
    frame = vp.compose_frame(ext, wrist, tac)
    assert frame.shape == (448, 448, 3) and frame.dtype == np.uint8
    assert not np.array_equal(frame, vp.compose_frame(ext, wrist, np.zeros_like(tac)))     # the taxels are drawn
    nothing = vp.compose_frame(ext, wrist, None)                                          # a robot without taxels: placeholder, same size
    assert nothing.shape == frame.shape
    banner = vp.text_banner([("t=0", (255, 255, 255))], 448)
    assert banner is None or banner.shape == (vp.BANNER_H, 448, 3)
    assert vp.tactile_stats(tac) == (3.0, 3.0, 0.0) and vp.tactile_stats(None) is None
    kw = vp.panel_kwargs({"extra/tactile": tac, "extra/ft_force": np.array([3.0, 4.0, 0.0]), "extra/gripper_effort": np.float32(1.5)})
    assert kw["hand_force"] == pytest.approx(5.0) and kw["gripper_effort"] == pytest.approx(1.5)
    assert vp.colormap(np.array([0.0, 1.0])).shape == (2, 3)


# ----------------------------------------------------------------------------------------------------------- the teleoperation client
def test_the_teleop_client_speaks_the_wire_format_of_the_server():
    """The client copies the wire format (it must run on a PC without the repo): both directions must stay compatible."""
    message = {"op": "debug", "kind": "teleop_step", "delta_pos": np.arange(3, dtype=np.float32).reshape(1, 3), "n": np.int64(4),
               "npz": b"\x00\x01", "ok": True}
    back = wire.unpack(tc._pack(message))
    assert np.array_equal(back["delta_pos"], message["delta_pos"]) and back["delta_pos"].dtype == np.float32
    assert back["npz"] == b"\x00\x01" and back["n"] == 4 and back["ok"] is True
    again = tc._unpackb(wire.pack({"obs": {"a": np.ones((1, 2, 2), np.uint8)}, "result": np.float32(1.5)}))
    assert again["obs"]["a"].shape == (1, 2, 2) and again["result"] == np.float32(1.5)


class FakeRemote:
    """What Session needs of RemoteEnv: a peg that gets inserted after `insert_after` steps (or never)."""
    def __init__(self, insert_after=3, truncate_at=None):
        self.insert_after, self.truncate_at, self.n, self.saved = insert_after, truncate_at, 0, []

    def _obs(self):
        return {P.EXTERIOR_IMAGE: np.zeros((8, 8, 3), np.uint8), P.WRIST_IMAGE: np.zeros((8, 8, 3), np.uint8),
                P.JOINT_POSITION: np.zeros(7, np.float32), P.GRIPPER_POSITION: np.array([0.9], np.float32),
                P.FT_FORCE: np.array([3.0, 4.0, 0.0], np.float32), P.PEG_SOCKET_OFFSET: np.array([0.0, 0.0, 0.01], np.float32)}

    def reset(self, seed):
        self.n = 0
        return self._obs()

    def debug(self, kind, **kw):
        return {"quat": np.array([0.0, 1.0, 0.0, 0.0], np.float32)} if kind == "fingertip_pose" else \
            {"name": "fake", "wrist_rot_ros_in_fingertip": [0.6916, -0.1471, -0.1471, 0.6916]}

    def teleop_step(self, delta, quat, gain, max_step):
        self.n += 1
        inserted = self.n > self.insert_after
        terminated = inserted and self.n - self.insert_after >= tc.HOLD_AFTER_SUCCESS
        return {"obs": self._obs(), "action": np.zeros(8, np.float32), "clipped": False, "inserted": inserted,
                "terminated": terminated, "truncated": self.truncate_at is not None and self.n >= self.truncate_at}

    def save_demo(self, out_dir, data, seed, info, success):
        self.saved.append((out_dir, data, seed, info, success))
        return {"file": "ep_0000_seed1.npz", "path": "/x/ep_0000_seed1.npz", "count": 1}


def _session(env, **kw):
    args = Namespace(view="wrist", flip_x=False, flip_y=False, rotate_deg=0.0, cmd_gain=1.0, max_joint_step=0.05, max_steps=150,
                     out_dir="demos/isaaclab/ForgePegInsert_Robotiq_teleop", **kw)
    session = tc.Session(env, args)
    session.prompt = "Insert the peg into the socket."
    session.start(1)
    return session


def test_teleop_episode_succeeds_after_five_inserted_steps_and_is_saved_with_the_demo_keys():
    session = _session(FakeRemote(insert_after=3))
    while not session.over:
        session.step(np.zeros(3))
    assert session.success and not session.timed_out and session.step_count == 8          # 3 steps, then 5 'inserted' in a row
    assert session.save() and session.saved
    out_dir, data, seed, info, ok = session.env.saved[0]
    z = np.load(io.BytesIO(data))
    assert set(("exterior_image", "wrist_image", "joint_position", "gripper_position", "ft_force", "peg_socket_offset", "actions",
                "prompt", "seed")) <= set(z.files)
    assert z["actions"].shape == (8, 8) and ok and seed == 1 and info["success"] and info["teleop"]
    assert out_dir == "demos/isaaclab/ForgePegInsert_Robotiq_teleop"


def test_teleop_episode_that_times_out_is_not_saved():
    session = _session(FakeRemote(insert_after=10 ** 6, truncate_at=4))
    while not session.over:
        session.step(np.zeros(3))
    assert session.timed_out and not session.success and not session.save() and not session.env.saved


def test_teleop_client_names_what_to_change_on_a_wrongly_started_server():
    env = tc.RemoteEnv.__new__(tc.RemoteEnv)
    env.spec = {"num_envs": 8, "success_hold_steps": 1, "robot": {"gripper_can_open": True}}
    with pytest.raises(SystemExit) as e:
        env.check_server()
    msg = str(e.value)
    assert "num_envs=1" in msg and "success_hold_steps=5" in msg and "gripper_can_open=false" in msg
    env.spec = {"num_envs": 1, "success_hold_steps": 5, "robot": {"gripper_can_open": False}}
    env.check_server()


def test_the_demo_norm_stats_take_the_chunk_length_of_the_model(tmp_path):
    from scripts.demos import compute_demo_norm_stats as cns
    rng = np.random.default_rng(0)
    for i in range(2):
        np.savez(tmp_path / f"ep_{i:04d}_seed{i}.npz", joint_position=rng.normal(size=(12, 7)).astype(np.float32), gripper_position=np.full(12, 0.87, np.float32),
                 actions=rng.normal(size=(12, 8)).astype(np.float32))
    files = sorted(tmp_path.glob("ep_*.npz"))
    state, chunks = cns.read_demos(files)                      # pi0.5: 16 steps
    assert state.shape == (24, 8) and chunks.shape == (24 * 16, 8)
    _, short = cns.read_demos(files, 10)                       # pi0-FAST: 10 steps, offsets from the chunk-start state too
    assert short.shape == (24 * 10, 8)
