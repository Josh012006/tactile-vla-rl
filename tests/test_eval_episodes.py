"""src/eval/episodes.py and src/eval/curve.py: the fixed episodes, the append-only log (a line cut by a kill), the numbers, the protocol check, one episode on the fake
simulator driven over a real socket, and the curve (title in a small font, inside the figure)."""
import json
import threading

import numpy as np
import pytest

from fakes import tag_of
from src.envs import protocol as P
from src.envs.client import VectorEnvClient
from src.eval import curve
from src.eval import episodes as E
from src.models.interface import VLAPolicy, ModelInfo
from test_client_server import Running, servers  # noqa: F401  (the `servers` fixture)

H = 8


# -- the fixed episodes ---------------------------------------------------------------------------------------------------------------------
def test_the_seeds_are_expo_fts_and_a_longer_list_extends_a_shorter_one():
    assert E.make_seeds(42, 50) == np.random.default_rng(42).integers(0, 2 ** 31 - 1, size=50).tolist()
    assert E.make_seeds(42, 200)[:50] == E.make_seeds(42, 50)
    assert E.make_seeds(42, 10) != E.make_seeds(43, 10)
    assert all(0 <= s < 2 ** 31 - 1 for s in E.make_seeds(7, 100))


def test_the_seed_file_is_kept_extended_and_refused_for_another_master_seed(tmp_path):
    first = E.load_or_create_seeds(tmp_path, 42, 5)
    saved = json.loads((tmp_path / E.SEEDS_FILE).read_text())
    assert saved == {"master_seed": 42, "n": 5, "seeds": first}
    assert E.load_or_create_seeds(tmp_path, 42, 5) == first
    longer = E.load_or_create_seeds(tmp_path, 42, 8)                        # more episodes later: the first five are the same
    assert longer[:5] == first and len(longer) == 8 and json.loads((tmp_path / E.SEEDS_FILE).read_text())["n"] == 8
    assert E.load_or_create_seeds(tmp_path, 42, 3) == first[:3]             # fewer: the prefix, and the longer file is kept
    assert json.loads((tmp_path / E.SEEDS_FILE).read_text())["n"] == 8
    with pytest.raises(ValueError, match="--seed 42"):
        E.load_or_create_seeds(tmp_path, 43, 5)


def test_a_file_whose_seeds_are_not_the_ones_numpy_draws_is_refused(tmp_path):
    (tmp_path / E.SEEDS_FILE).write_text(json.dumps({"master_seed": 42, "n": 3, "seeds": [1, 2, 3]}))
    with pytest.raises(ValueError, match="does not hold the seeds"):
        E.load_or_create_seeds(tmp_path, 42, 3)


# -- the log ---------------------------------------------------------------------------------------------------------------------------------------
def record(i, success=True, **extra):
    return {"episode": i, "seed": 100 + i, "success": success, "terminated": success, "truncated": not success, "steps": 10 + i, "start_sha": f"s{i}",
            "traj_sha": f"t{i}", "wall_s": 1.0, "attempt": 1, "gpu": "gpu", "decode": None, **extra}


def test_the_log_appends_reads_back_and_drops_a_line_cut_by_a_kill(tmp_path):
    log = E.EpisodeLog(tmp_path / "r" / "0.jsonl")
    assert log.load() == {}
    for i in range(3):
        log.append(record(i))
    with open(log.path, "a") as f:
        f.write('{"episode": 3, "seed": 103, "succ')                         # the process was killed in the middle of a write
    records = E.EpisodeLog(log.path).load()
    assert sorted(records) == [0, 1, 2]
    assert log.path.read_text().endswith("}\n")                              # cut back to the last whole line ...
    E.EpisodeLog(log.path).append(record(3))                                 # ... so the next append starts on a clean one
    assert sorted(E.EpisodeLog(log.path).load()) == [0, 1, 2, 3]
    assert len(log.path.read_text().splitlines()) == 4


def test_a_damaged_whole_line_is_an_error_and_a_repeated_episode_keeps_its_first_record(tmp_path):
    path = tmp_path / "0.jsonl"
    path.write_text(json.dumps(record(0)) + "\nnot json\n")
    with pytest.raises(ValueError, match="line 2"):
        E.EpisodeLog(path).load()
    log = E.EpisodeLog(tmp_path / "1.jsonl")
    log.append(record(0, steps=11))
    log.append(record(0, steps=99))
    records = log.load()
    assert records[0]["steps"] == 11 and log.duplicates == 1


def test_appends_from_several_threads_never_interleave(tmp_path):
    log = E.EpisodeLog(tmp_path / "0.jsonl")
    threads = [threading.Thread(target=lambda k=k: [log.append(record(100 * k + i)) for i in range(20)]) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log.load()) == 120


# -- numbers ---------------------------------------------------------------------------------------------------------------------------------------
def test_wilson_is_informative_at_the_extremes_and_contains_the_rate():
    lo, hi = E.wilson95(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0.05 < hi < 0.10
    lo, hi = E.wilson95(50, 50)
    assert hi == pytest.approx(1.0) and 0.90 < lo < 0.95
    lo, hi = E.wilson95(25, 50)
    assert lo < 0.5 < hi


def test_summarize_recomputes_everything_from_the_records_and_refuses_an_incomplete_or_foreign_log():
    seeds = [100 + i for i in range(4)]
    records = {i: record(i, success=i % 2 == 0) for i in range(4)}
    s = E.summarize("500", 500, records, seeds)
    assert (s["n_episodes"], s["n_success"], s["success_rate"]) == (4, 2, 0.5) and s["successes"] == [True, False, True, False]
    assert s["success_se"] == pytest.approx(0.25) and s["single_attempt"] and s["attempts"] == [1] and s["gpus"] == ["gpu"]
    assert s["steps_mean_success"] == pytest.approx(11.0)
    other = {**records, 2: record(2, traj_sha="DIFFERENT")}
    assert E.summarize("500", 500, other, seeds)["eval_sha"] != s["eval_sha"]       # one trajectory differs: the evaluation's fingerprint does
    resumed = {**records, 3: record(3, success=False, attempt=2)}
    assert E.summarize("500", 500, resumed, seeds)["single_attempt"] is False
    with pytest.raises(ValueError, match="missing"):
        E.summarize("500", 500, {0: records[0]}, seeds)
    with pytest.raises(ValueError, match="another evaluation"):
        E.summarize("500", 500, records, [7, 8, 9, 10])
    assert E.summarize("500", 500, {i: record(i) for i in range(6)}, seeds)["n_episodes"] == 4   # a log from a longer evaluation: the first seeds only


def test_the_protocol_check_names_what_changed_and_lets_the_free_settings_move():
    saved = {"master_seed": 42, "replan": 8, "max_steps": 150, "num_envs": 1, "model_cfg": {"a": (1, 2)}, "task": {"x": 1}, "recipe": "r", "attempts": [], "n_episodes": 50}
    E.check_protocol(saved, {**saved, "model_cfg": {"a": [1, 2]}, "n_episodes": 200, "attempts": [1]})          # tuple vs list, episodes, attempts: free
    with pytest.raises(ValueError) as e:
        E.check_protocol(saved, {**saved, "replan": 5, "task": {"x": 2}})
    assert "replan" in str(e.value) and "task" in str(e.value) and "master_seed" not in str(e.value)


# -- one episode on the fake simulator ---------------------------------------------------------------------------------------------------------------
class FakePolicy(VLAPolicy):
    """Pushes joint 0 to 1.0 when the start state allows it (the fake simulator's tag, read from the observation), after a noise on the holding value that comes from the
    policy's own random stream, restarted by `seed` and advanced at every call: like pi0.5's flow noise, which is what makes the trajectory depend on it."""

    def __init__(self, noise=True):
        self.info = ModelInfo.from_cfg({"name": "fake", "family": "fake", "action_horizon": H, "output_semantics": "absolute_targets"})
        self.noise, self.seeds = noise, []
        self._rng = np.random.default_rng(0)

    def seed(self, seed):
        self.seeds.append(seed)
        self._rng = np.random.default_rng(seed)

    def raw_chunks(self, rows, num_samples):
        out = np.zeros((len(rows), num_samples, H, P.ACTION_DIM), np.float32)
        for i, row in enumerate(rows):
            clock, tag = float(row["observation/joint_position"][1]), int(row["observation/joint_position"][2])
            from fakes import trigger
            for k in range(H):
                out[i, :, k, 0] = 1.0 if clock + k >= trigger(tag) else (0.01 * self._rng.standard_normal() if self.noise else 0.0)
        return out, 1.0


def will_succeed(seed, max_steps=20):
    """Outcome of the episode of `seed` on the fake simulator. The policy reads the start state's tag from a float32 observation: a large tag is rounded there, and the
    rounded one is what decides (a real seed is up to 2**31)."""
    from fakes import expected_length
    return expected_length(int(np.float32(tag_of(seed, 0))), max_steps)[1]


def play(server, seed, **kw):
    with VectorEnvClient([server.endpoint]) as env:
        return E.run_episode(env, kw.pop("policy", None) or FakePolicy(), episode=kw.pop("episode", 0), seed=seed, replan=4, max_steps=20, **kw)


def test_an_episode_gives_its_record_and_the_policy_was_seeded_with_the_episodes_seed(servers):  # noqa: F811
    seed = next(s for s in range(50) if will_succeed(s))
    policy = FakePolicy()
    rec = play(servers(num_envs=1, max_steps=20), seed, policy=policy, episode=3)
    assert (rec["episode"], rec["seed"], rec["success"], rec["terminated"], rec["truncated"]) == (3, seed, True, True, False)
    assert policy.seeds == [seed] and rec["steps"] <= 20 and len(rec["start_sha"]) == 16 and len(rec["traj_sha"]) == 16 and rec["plans"] >= 1
    assert rec["decode"] is None and rec["video"] is None and set(rec["progress"]) >= {"xy_mm_start", "xy_mm_end"}
    failing = next(s for s in range(50) if not will_succeed(s))
    bad = play(servers(num_envs=1, max_steps=20), failing)
    assert (bad["success"], bad["truncated"], bad["steps"]) == (False, True, 20)


def test_the_same_episode_on_a_fresh_server_is_bit_identical_and_another_seed_is_not(servers):  # noqa: F811
    a = play(servers(num_envs=1, max_steps=20), 11)
    b = play(servers(num_envs=1, max_steps=20), 11)
    c = play(servers(num_envs=1, max_steps=20), 12)
    assert (a["start_sha"], a["traj_sha"]) == (b["start_sha"], b["traj_sha"])
    assert c["start_sha"] != a["start_sha"] and c["traj_sha"] != a["traj_sha"]


def test_the_policys_noise_depends_on_the_episode_not_on_what_ran_before(servers):  # noqa: F811
    """Seeding per episode: episode 12 gives the same trajectory whether or not episode 11 was played before it, on the same policy object."""
    seed = next(s for s in range(50) if will_succeed(s))
    alone = play(servers(num_envs=1, max_steps=20), seed)
    policy = FakePolicy()
    play(servers(num_envs=1, max_steps=20), seed + 1, policy=policy)
    after_another = play(servers(num_envs=1, max_steps=20), seed, policy=policy)
    assert alone["traj_sha"] == after_another["traj_sha"]
    unseeded = FakePolicy()
    unseeded.seed = lambda s: None                                               # what pi0.5 would do without the per-episode key: its stream goes on
    first = play(servers(num_envs=1, max_steps=20), seed, policy=unseeded)
    second = play(servers(num_envs=1, max_steps=20), seed, policy=unseeded)
    assert first["traj_sha"] != second["traj_sha"]


def test_a_stop_during_an_episode_drops_it_and_an_episode_that_does_not_end_is_an_error(servers):  # noqa: F811
    stop = threading.Event()
    policy = FakePolicy()
    plain = policy.raw_chunks

    def raw_and_stop(rows, n):
        stop.set()
        return plain(rows, n)

    policy.raw_chunks = raw_and_stop
    with pytest.raises(E.EvalAborted):
        play(servers(num_envs=1, max_steps=20), 5, policy=policy, stop=stop)
    with VectorEnvClient([servers(num_envs=1, max_steps=20).endpoint]) as env:
        with pytest.raises(RuntimeError, match="did not end"):
            E.run_episode(env, FakePolicy(), episode=0, seed=3, replan=4, max_steps=5)          # the server's episodes are longer than the protocol's
    with VectorEnvClient([servers(num_envs=2, max_steps=20).endpoint]) as env:
        with pytest.raises(ValueError, match="ONE environment"):
            E.run_episode(env, FakePolicy(), episode=0, seed=3, replan=4, max_steps=20)


# -- the curve ------------------------------------------------------------------------------------------------------------------------------------------
def write_results(folder, steps_rates, n=50):
    folder.mkdir(parents=True, exist_ok=True)
    for step, rate in steps_rates:
        p = rate
        (folder / f"{step}.json").write_text(json.dumps({"label": str(step), "step": step, "success_rate": p, "success_se": float(np.sqrt(p * (1 - p) / n)),
                                                         "wilson95": list(E.wilson95(round(p * n), n)), "n_episodes": n, "single_attempt": True, "eval_sha": "x"}))


def test_the_curve_is_rebuilt_from_the_complete_checkpoints_only_and_in_step_order(tmp_path):
    pytest.importorskip("matplotlib")
    write_results(tmp_path / "results", [(1000, 0.6), (0, 0.0), (500, 0.3)])
    (tmp_path / "results" / "1500.jsonl").write_text(json.dumps(record(0)) + "\n")              # an unfinished checkpoint has no .json: not on the curve
    entries = curve.write_curve(tmp_path / "results", tmp_path / "out", name="run")
    assert [e["step"] for e in entries] == [0, 500, 1000]
    assert json.loads((tmp_path / "out" / "curve.json").read_text()) == entries
    assert (tmp_path / "out" / "curve.png").stat().st_size > 5000 and (tmp_path / "out" / "curve.pdf").read_bytes().startswith(b"%PDF")
    assert curve.write_curve(tmp_path / "nothing", tmp_path / "out2") == [] and not (tmp_path / "out2" / "curve.png").exists()


def test_only_the_title_is_small_and_a_long_one_is_wrapped_inside_the_figure(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    write_results(tmp_path / "results", [(0, 0.1), (500, 0.5)])
    long_title = "Teleop demonstrations, pi0.5 LoRA rank 16, normalisation statistics of the noisy demonstrations, 200 episodes per checkpoint, seed 42, replan 8 of 16"
    fig = curve.draw_curve(curve.load_entries(tmp_path / "results"), long_title)
    ax = fig.axes[0]
    assert ax.title.get_fontsize() == curve.TITLE_FONT_SIZE <= 10                                   # small
    assert ax.xaxis.label.get_fontsize() == 14 and ax.yaxis.label.get_fontsize() == 14            # the axes keep expo-ft's sizes
    assert all(len(line) <= curve.TITLE_WRAP for line in ax.get_title().splitlines()) and len(ax.get_title().splitlines()) > 1
    fig.canvas.draw()
    box, frame = ax.title.get_window_extent(fig.canvas.get_renderer()), fig.bbox
    assert box.x0 >= frame.x0 and box.x1 <= frame.x1 and box.y0 >= frame.y0 and box.y1 <= frame.y1         # the title does not leave the picture
    plt.close(fig)
