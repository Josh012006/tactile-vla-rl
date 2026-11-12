"""scripts/adopt_expo_ft_checkpoint.py: an expo-ft checkpoint folder becomes a run that scripts/eval_sft.py can evaluate as its step 0."""
import json

import pytest

from scripts import adopt_expo_ft_checkpoint as AD
from scripts import eval_sft as ES


def expo_ft_checkpoint(tmp_path, ids=("forge_demos_robotiq_noisy",)):
    ckpt = tmp_path / "expo" / "4000"
    (ckpt / "params").mkdir(parents=True)
    for asset in ids:
        (ckpt / "assets" / asset).mkdir(parents=True)
        (ckpt / "assets" / asset / "norm_stats.json").write_text("{}")
    return ckpt


def test_the_adopted_config_points_at_the_weights_and_the_statistics_saved_with_the_checkpoint(tmp_path):
    ckpt = expo_ft_checkpoint(tmp_path)
    cfg = AD.adopted_config("pi05", ckpt)
    model = cfg["model_cfg"]
    assert model["checkpoint"] == str(ckpt.resolve()) and cfg["adopted_from"] == str(ckpt.resolve())
    assert model["norm_stats"] == {"dir": str((ckpt / "assets").resolve()), "asset_id": "forge_demos_robotiq_noisy"}
    assert model["vlm"]["lora_rank"] == 16 and model["action_head"]["lora_rank"] == 32          # the config expo-ft trained


def test_bad_checkpoints_are_refused_and_several_asset_ids_need_a_choice(tmp_path):
    with pytest.raises(SystemExit, match="no params"):
        AD.adopted_config("pi05", tmp_path)
    two = expo_ft_checkpoint(tmp_path / "two", ids=("a", "b"))
    with pytest.raises(SystemExit, match="--asset-id"):
        AD.adopted_config("pi05", two)
    assert AD.adopted_config("pi05", two, "b")["model_cfg"]["norm_stats"]["asset_id"] == "b"
    with pytest.raises(SystemExit, match="no norm_stats"):
        AD.adopted_config("pi05", two, "c")
    none = expo_ft_checkpoint(tmp_path / "none", ids=())
    with pytest.raises(SystemExit, match="holds no"):
        AD.adopted_config("pi05", none)


def test_the_evaluation_takes_the_adopted_run_as_its_step_0(tmp_path, monkeypatch):
    ckpt = expo_ft_checkpoint(tmp_path)
    monkeypatch.setattr(AD, "PROBING", tmp_path / "probing")
    assert AD.main(["--checkpoint", str(ckpt), "--name", "x"]) == 0
    with pytest.raises(SystemExit, match="exists"):
        AD.main(["--checkpoint", str(ckpt), "--name", "x"])
    run = ES.find_run_dir(str(tmp_path / "probing" / "_adopted_x"))
    jobs = ES.select_jobs(ES.discover_checkpoints(run), [0], False)
    assert [(j.step, j.checkpoint) for j in jobs] == [(0, None)]
    model_cfg = json.loads((run / "config.json").read_text())["model_cfg"]
    assert ES.check_norm_stats(model_cfg, jobs) is not None                                          # step 0 has no copy to compare: nothing to refuse
