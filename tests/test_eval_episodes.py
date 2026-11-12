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


# -- one batch on the fake simulator ------------------------------------------------------------------------------------------------------------------
class FakePolicy(VLAPolicy):
    """Pushes joint 0 to 1.0 when the start state allows it (the fake simulator's tag, read from the observation), after a noise on the holding value that comes from the row's
    own seed (`row_seeds`, like pi0.5's per-episode flow noise), which is what makes the trajectory depend on it. `calls` lists (rows, row_seeds, pad_to) of every model call."""

    def __init__(self, noise=True):
        self.info = ModelInfo.from_cfg({"name": "fake", "family": "fake", "action_horizon": H, "output_semantics": "absolute_targets"})
        self.noise, self.seeds, self.calls = noise, [], []
        self._rng = np.random.default_rng(0)

    def seed(self, seed):
        self.seeds.append(seed)
        self._rng = np.random.default_rng(seed)

    def raw_chunks(self, rows, num_samples, row_seeds=None, pad_to=None):
        self.calls.append((len(rows), None if row_seeds is None else list(row_seeds), pad_to))
        out = np.zeros((len(rows), num_samples, H, P.ACTION_DIM), np.float32)
        for i, row in enumerate(rows):
            rng = np.random.default_rng(row_seeds[i]) if row_seeds is not None else self._rng
            clock, tag = float(row["observation/joint_position"][1]), int(row["observation/joint_position"][2])
            from fakes import trigger
            for k in range(H):
                out[i, :, k, 0] = 1.0 if clock + k >= trigger(tag) else (0.01 * rng.standard_normal() if self.noise else 0.0)
        return out, 1.0


def predicted(master, batch, env, max_steps=20):
    """(length, terminated) of the episode of environment `env` in round `batch` of `master` on the fake simulator. The policy reads the start state's tag from a float32
    observation: a large tag is rounded there, and the rounded one is what decides (a real seed is up to 2**31)."""
    from fakes import expected_length
    return expected_length(int(np.float32(tag_of(master + batch, env))), max_steps)


def will_succeed(master, batch, env, max_steps=20):
    return predicted(master, batch, env, max_steps)[1]


def play(server, policy=None, episodes=None, master=5, batch=0, max_steps=20, **kw):
    """Run a batch on `server`; returns (records in the order they were reported, what run_batch returned)."""
    records = []
    with VectorEnvClient([server.endpoint]) as env:
        episodes = episodes if episodes is not None else {e: (e, 100 + e) for e in range(env.num_envs)}
        info = E.run_batch(env, policy or FakePolicy(), episodes, master_seed=master, batch=batch, replan=4, max_steps=max_steps, on_episode=records.append, **kw)
    return records, info


def test_a_batch_reports_every_episode_when_it_ends_with_the_start_of_its_environment_and_round(servers):  # noqa: F811
    recs, info = play(servers(num_envs=6, max_steps=20), master=5, batch=2)
    assert sorted(r["env"] for r in recs) == list(range(6)) and info["steps_run"] == 20
    for r in recs:
        length, terminated = predicted(5, 2, r["env"])
        assert (r["episode"], r["seed"], r["batch"], r["steps"], r["success"], r["terminated"], r["truncated"]) == (r["env"], 100 + r["env"], 2, length, terminated, terminated, not terminated)
        assert len(r["start_sha"]) == 16 and len(r["traj_sha"]) == 16 and r["plans"] >= 1 and r["decode"] is None and r["video"] is None
        assert set(r["progress"]) >= {"xy_mm_start", "xy_mm_end"}
    assert [r["steps"] for r in recs] == sorted(r["steps"] for r in recs)                  # reported as they end: the short ones first
    assert [r["wall_s"] for r in recs] == sorted(r["wall_s"] for r in recs)
    assert len({r["start_sha"] for r in recs}) == 6                                       # six different start states


def test_the_same_batch_from_a_fresh_server_is_bit_identical_and_the_environments_that_do_not_play_change_nothing_for_the_others(servers):  # noqa: F811
    a, _ = play(servers(num_envs=6, max_steps=20))
    b, _ = play(servers(num_envs=6, max_steps=20))
    shas = lambda recs: {r["env"]: (r["start_sha"], r["traj_sha"]) for r in recs}               # noqa: E731
    assert shas(a) == shas(b)
    subset, _ = play(servers(num_envs=6, max_steps=20), episodes={2: (2, 102), 4: (4, 104)})     # the resume of a cut evaluation: only two episodes play
    assert shas(subset) == {e: shas(a)[e] for e in (2, 4)}
    other_seed, _ = play(servers(num_envs=6, max_steps=20), episodes={2: (2, 999)})              # same start, another noise: the start is the environment's, the noise the episode's
    assert other_seed[0]["start_sha"] == shas(a)[2][0] and other_seed[0]["traj_sha"] != shas(a)[2][1]
    later, _ = play(servers(num_envs=6, max_steps=20), batch=1)
    assert all(shas(later)[e][0] != shas(a)[e][0] for e in range(6))                              # another round: another start for every environment


def test_the_model_is_called_with_k_rows_and_the_noise_seed_of_each_row_comes_from_its_episode_and_chunk_alone(servers):  # noqa: F811
    policy = FakePolicy()
    play(servers(num_envs=4, max_steps=20), policy, episodes={1: (0, 11), 3: (1, 13)})
    rows, seeds, pad = policy.calls[0]
    assert (rows, pad) == (2, 4) and seeds == [E.plan_seed(11, 0), E.plan_seed(13, 0)]
    assert all(c[2] == 4 for c in policy.calls) and len(policy.calls) >= 2                      # every call has the same shape, K rows, whatever the number that still run
    allowed = {E.plan_seed(s, k) for s in (11, 13) for k in range(10)}
    assert all(set(c[1]) <= allowed for c in policy.calls)
    again = FakePolicy()
    play(servers(num_envs=4, max_steps=20), again, episodes={3: (1, 13)})                        # alone, episode 13 asks for the same noise at its first chunk
    assert again.calls[0][1] == [E.plan_seed(13, 0)]


def test_plan_seeds_are_32_bit_deterministic_and_differ_by_episode_and_by_chunk():
    assert E.plan_seed(7, 0) == E.plan_seed(7, 0)
    seeds = {E.plan_seed(s, k) for s in range(20) for k in range(20)}
    assert len(seeds) == 400 and all(0 <= x < 2 ** 32 for x in seeds)


def test_a_stop_drops_the_episodes_still_running_but_keeps_the_ones_that_ended(servers):  # noqa: F811
    stop, records = threading.Event(), []

    def on_episode(r):
        records.append(r)
        stop.set()

    with VectorEnvClient([servers(num_envs=4, max_steps=20).endpoint]) as env:                    # tags 15, 16, 17, 18: environment 0 never succeeds, the others do, early
        with pytest.raises(E.EvalAborted):
            E.run_batch(env, FakePolicy(), {e: (e, 100 + e) for e in range(4)}, master_seed=5, batch=0, replan=4, max_steps=20, on_episode=on_episode, stop=stop)
    assert records and 0 not in {r["env"] for r in records}


def test_an_episode_that_does_not_end_is_an_error_and_bad_environments_are_refused(servers):  # noqa: F811
    with VectorEnvClient([servers(num_envs=4, max_steps=20).endpoint]) as env:
        with pytest.raises(RuntimeError, match="did not end"):
            E.run_batch(env, FakePolicy(), {0: (0, 1)}, master_seed=5, batch=0, replan=4, max_steps=5, on_episode=lambda r: None)      # the server's episodes are longer than the protocol's
        for bad in ({9: (0, 1)}, {}):
            with pytest.raises(ValueError, match="environments of this server"):
                E.run_batch(env, FakePolicy(), bad, master_seed=5, batch=0, replan=4, max_steps=20, on_episode=lambda r: None)


def test_the_trajectory_hash_reads_its_own_environment():
    obs = {P.JOINT_POSITION: np.arange(14, dtype=np.float32).reshape(2, 7), P.GRIPPER_POSITION: np.zeros((2, 1), np.float32)}
    a, b = E.TrajectoryHash(0), E.TrajectoryHash(1)
    a.update(obs)
    b.update(obs)
    assert a.start != b.start and a.hexdigest() != b.hexdigest()
    same = E.TrajectoryHash(1)
    same.update(obs)
    assert same.hexdigest() == b.hexdigest()


def test_the_simulation_time_of_a_checkpoint_counts_a_batch_once():
    batched = [record(i, batch=0, wall_s=w) for i, w in enumerate((10.0, 20.0, 15.0))] + [record(3, batch=1, wall_s=7.0)]
    assert E.wall_total(batched) == 27.0
    assert E.wall_total([record(i, wall_s=5.0) for i in range(3)]) == 15.0                  # no batch: every episode ran alone


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
