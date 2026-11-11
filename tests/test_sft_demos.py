"""src/algos/demos.py: the demonstrations as training samples (numpy only), and the dry run of scripts/sft.py."""
import numpy as np
import pytest

from src.algos import demos
from src.envs import protocol as P

PROMPT = "Insert the peg into the socket."


def write_demo(folder, name, steps, *, seed=0, tactile=True, prompt=PROMPT, extra=None):
    """A complete .npz with the arrays of scripts/demos/collect_waypoint_demos.py (images of 224 x 224, constant per step so rows are recognisable)."""
    rng = np.random.default_rng(seed)
    arrays = {
        "exterior_image": np.broadcast_to(np.arange(steps, dtype=np.uint8)[:, None, None, None], (steps, 224, 224, 3)).copy(),
        "wrist_image": np.zeros((steps, 224, 224, 3), np.uint8),
        "joint_position": rng.normal(size=(steps, 7)).astype(np.float32),
        "gripper_position": np.full((steps, 1), 0.9, np.float32),
        "ft_force": rng.normal(size=(steps, 3)).astype(np.float32),
        "peg_socket_offset": rng.normal(size=(steps, 3)).astype(np.float32),
        "actions": np.concatenate([np.arange(steps, dtype=np.float32)[:, None] * np.ones((1, 7), np.float32), np.ones((steps, 1), np.float32)], axis=1),
        "seed": np.array(seed),
    }
    if prompt is not None:
        arrays["prompt"] = np.array(prompt)
    if tactile:
        arrays["tactile"] = rng.random((steps, 2, 7, 4)).astype(np.float32)
        arrays["gripper_effort"] = rng.random(steps).astype(np.float32)
    arrays.update(extra or {})
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / name, **arrays)


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / "ForgePegInsert_Robotiq_scripted"
    for i, steps in enumerate((6, 5, 8, 4)):
        write_demo(f, f"ep_{i:04d}_seed{i}.npz", steps, seed=i)
    write_demo(f, "fail_seed9_120000.npz", 7, seed=9)               # a failed episode: never used
    return f


def test_a_sample_is_an_env_observation_and_the_next_actions_with_the_last_one_repeated(folder):
    train, val = demos.load_demos([str(folder)], horizon=4, default_prompt=PROMPT)
    assert val is None and len(train) == 6 + 5 + 8 + 4 and train.describe()["episodes"] == 4          # fail_seed*.npz left out
    obs, prompts, target = train.sample([0, 4, 5])                      # steps 0, 4 and 5 of the first episode (6 steps)
    assert set(obs) >= {P.EXTERIOR_IMAGE, P.WRIST_IMAGE, P.JOINT_POSITION, P.GRIPPER_POSITION}
    assert obs[P.EXTERIOR_IMAGE].shape == (3, 224, 224, 3) and obs[P.EXTERIOR_IMAGE].dtype == np.uint8
    assert obs[P.GRIPPER_POSITION].shape == (3, 1) and obs[P.JOINT_POSITION].shape == (3, 7)
    assert [int(obs[P.EXTERIOR_IMAGE][i, 0, 0, 0]) for i in range(3)] == [0, 4, 5]                    # the right frames
    assert target.shape == (3, 4, 8) and prompts == [PROMPT] * 3
    assert target[0, :, 0].tolist() == [0, 1, 2, 3]                                                   # actions[0:4]
    assert target[1, :, 0].tolist() == [4, 5, 5, 5]                                                   # the end of the episode: the last action repeated
    assert target[2, :, 0].tolist() == [5, 5, 5, 5]


def test_a_chunk_never_crosses_into_the_next_episode(folder):
    train, _ = demos.load_demos([str(folder)], horizon=6, default_prompt=PROMPT)
    rows = np.arange(len(train))
    chunk_rows = train.chunk_rows(rows)
    assert (train.episode_of[chunk_rows] == train.episode_of[rows][:, None]).all()


def test_the_touch_signals_are_served_under_the_envs_keys(folder):
    train, _ = demos.load_demos([str(folder)], horizon=4, default_prompt=PROMPT)
    obs, _, _ = train.sample([0, 1])
    assert obs[P.TACTILE].shape == (2, 2, 7, 4) and obs[P.GRIPPER_EFFORT].shape == (2,)
    assert obs[P.FT_FORCE].shape == (2, 3) and obs[P.PEG_SOCKET_OFFSET].shape == (2, 3)


def test_num_demos_and_held_out_episodes_are_taken_per_folder(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for i in range(5):
        write_demo(a, f"ep_{i:04d}_seed{i}.npz", 4, seed=i)
        write_demo(b, f"ep_{i:04d}_seed{i}.npz", 3, seed=10 + i)
    train, val = demos.load_demos([str(a), str(b)], horizon=2, num_demos=4, val_episodes=1, default_prompt=PROMPT)
    assert [ep.source for ep in train.episodes] == [f"a/ep_{i:04d}_seed{i}.npz" for i in range(3)] + [f"b/ep_{i:04d}_seed{i}.npz" for i in range(3)]
    assert [ep.source for ep in val.episodes] == ["a/ep_0003_seed3.npz", "b/ep_0003_seed3.npz"]       # the last of the first 4 of each folder
    with pytest.raises(ValueError, match="no training episode"):
        demos.load_demos([str(a)], horizon=2, num_demos=2, val_episodes=2, default_prompt=PROMPT)


def test_the_prompt_comes_from_the_file_then_from_the_default(tmp_path):
    f = tmp_path / "d"
    write_demo(f, "ep_0000_seed0.npz", 4, prompt="put it in")
    write_demo(f, "ep_0001_seed1.npz", 4, prompt=None)
    train, _ = demos.load_demos([str(f)], horizon=2, default_prompt="default")
    _, prompts, _ = train.sample([0, 4])
    assert prompts == ["put it in", "default"]


def test_a_broken_demonstration_is_refused_with_its_name(tmp_path):
    f = tmp_path / "d"
    write_demo(f, "ep_0000_seed0.npz", 4, extra={"actions": np.zeros((4, 7), np.float32)})
    with pytest.raises(ValueError, match=r"ep_0000_seed0\.npz.*actions"):
        demos.load_demos([str(f)], horizon=2, default_prompt=PROMPT)
    g = tmp_path / "e"
    write_demo(g, "ep_0000_seed0.npz", 4, extra={"joint_position": np.full((4, 7), np.nan, np.float32)})
    with pytest.raises(ValueError, match="non-finite"):
        demos.load_demos([str(g)], horizon=2, default_prompt=PROMPT)
    with pytest.raises(FileNotFoundError):
        demos.load_demos([str(tmp_path / "empty")], horizon=2, default_prompt=PROMPT)


def test_demonstrations_with_and_without_taxels_are_not_mixed(tmp_path):
    f = tmp_path / "d"
    write_demo(f, "ep_0000_seed0.npz", 4, tactile=True)
    write_demo(f, "ep_0001_seed1.npz", 4, tactile=False)
    with pytest.raises(ValueError, match="with and without taxels"):
        demos.load_demos([str(f)], horizon=2, default_prompt=PROMPT)


def test_the_dry_run_reads_the_demonstrations_and_loads_no_model(folder, capsys):
    from scripts import sft as cli
    assert cli.main(["--model", "pi0_fast", "--demos", str(folder), "--dry-run", "--set", "steps=20", "--set", "batch_size=4",
                     "--set", "micro_batch_size=2", "--set", "val_episodes=1"]) == 0
    out = capsys.readouterr().out
    assert "pi0_fast" in out and "horizon 10" in out and "lora" in out and "held-out set: 1 episodes" in out
    cli.main(["--model", "pi05", "--full", "--demos", str(folder), "--dry-run"])
    out = capsys.readouterr().out
    assert "horizon 16" in out and out.splitlines()[0].endswith(", full")


def test_full_turns_lora_off_in_every_component_the_model_has():
    from scripts import sft as cli
    from src.models.interface import apply_overrides, load_model_cfg
    from src.models.settings import parse_model_settings
    for name, components in (("pi05", ("vlm", "action_head")), ("pi0_fast", ("vlm",))):
        cfg = load_model_cfg(name)
        full = apply_overrides(cfg, cli.model_overrides(cfg, True, {}))
        settings = parse_model_settings(full)
        assert not settings.any_lora, name
        assert set(cli.model_overrides(cfg, True, {})) == {f"{c}.use_lora" for c in components}
    cfg = load_model_cfg("pi05")
    assert cli.model_overrides(cfg, True, {"vlm.use_lora": True})["vlm.use_lora"] is True               # --model-set wins over --full
    assert cli.parse_assignments(["a=1", "b.c=false", "d=x"]) == {"a": 1, "b.c": False, "d": "x"}


def test_the_sft_config_refuses_what_it_does_not_know_and_what_does_not_fit():
    from src.algos import sft
    cfg = sft.load_config()
    assert cfg.batch_size % cfg.micro_batch_size == 0 and cfg.demos and cfg.lr == cfg.lr_end
    with pytest.raises(ValueError, match="unknown SFT keys"):
        sft.load_config(overrides={"learning_rate": 1})
    with pytest.raises(ValueError, match="multiple of micro_batch_size"):
        sft.load_config(overrides={"batch_size": 10, "micro_batch_size": 4})
    assert sft.load_config(overrides={"demos": "demos/x"}).demos == ("demos/x",)


def test_row_batches_cover_the_data_without_repeating_inside_an_epoch_and_drop_the_tail():
    from src.algos.sft import row_batches
    batches = list(row_batches(n_rows=10, batch_size=4, steps=5, seed=0))
    assert len(batches) == 5 and all(len(b) == 4 and len(set(b)) == 4 for b in batches)
    assert len(set(batches[0]) | set(batches[1])) == 8                       # an epoch: 2 batches of 4 out of 10 (the 2 left over are dropped)
    assert [b.tolist() for b in row_batches(10, 4, 5, seed=0)] == [b.tolist() for b in batches]             # reproducible
    with pytest.raises(ValueError, match="batch_size"):
        list(row_batches(3, 4, 1, 0))


def test_prefetch_gives_the_items_in_order_and_raises_the_error_of_its_thread():
    from src.algos.sft import prefetch
    assert list(prefetch(iter(range(7)), depth=2)) == list(range(7))

    def broken():
        yield 1
        raise RuntimeError("tokenizer exploded")

    got = []
    with pytest.raises(RuntimeError, match="tokenizer exploded"):
        for item in prefetch(broken(), depth=2):
            got.append(item)
    assert got == [1]
