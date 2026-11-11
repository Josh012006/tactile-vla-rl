"""scripts/probe_vla.py and scripts/compare_reports.py: what does not need openpi, JAX or a GPU: saved observations and their hash, the
`--set` overrides, the comparison of two reports. (The rollout is tests/test_eval_rollout.py, the model path tests/test_openpi_policy.py.)"""
import json

import numpy as np
import pytest

from src.envs import protocol as P
from src.models.interface import to_openpi_obs
from scripts import compare_reports as compare
from scripts import probe_vla as probe


def obs(n=2, size=224):
    rng = np.random.default_rng(0)
    return {P.EXTERIOR_IMAGE: rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8),
            P.WRIST_IMAGE: rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8),
            P.JOINT_POSITION: np.zeros((n, 7), np.float32), P.GRIPPER_POSITION: np.zeros((n, 1), np.float32)}


# -- saved observations ----------------------------------------------------------------------------------------------------------
def openpi_obs():
    return to_openpi_obs(obs(), 0, "Insert the peg into the socket.")


def test_a_saved_observation_comes_back_identical(tmp_path):
    o = openpi_obs()
    probe.save_obs(o, tmp_path / "o.npz")
    back = probe.load_obs(tmp_path / "o.npz")
    assert set(back) == set(o) and back["prompt"] == o["prompt"]
    for k in o:
        if k != "prompt":
            assert np.array_equal(back[k], o[k]) and back[k].dtype == o[k].dtype
    assert probe.obs_sha256(back) == probe.obs_sha256(o)


def test_the_hash_tells_observations_apart():
    a, b = openpi_obs(), openpi_obs()
    assert probe.obs_sha256(a) == probe.obs_sha256(b)
    b["observation/exterior_image_1_left"] = b["observation/exterior_image_1_left"].copy()
    b["observation/exterior_image_1_left"][0, 0, 0] ^= 1
    assert probe.obs_sha256(a) != probe.obs_sha256(b)
    c = dict(a, prompt="another prompt")
    assert probe.obs_sha256(a) != probe.obs_sha256(c)


# -- comparing two reports -----------------------------------------------------------------------------------------------------
def report(rows, sha="abc", lora=True, new_schema=True):
    r = {"model": "pi05", "load_mode": "weight_loader", "observation_sha256": sha, "first_chunk": {"rows": rows}}
    if new_schema:
        r["settings_text"] = "vlm: LoRA rank 16, touch: none (task only)" if lora else "vlm: no LoRA, touch: none (task only)"
    else:                                                    # reports written before the per-component settings
        r["lora"] = {"use_lora": lora, "vlm_rank": 16, "action_head_rank": 32}
    return r


def test_compare_chunks():
    a = np.full((16, 8), 0.1)
    b = a.copy()
    b[3, 5] += 0.02
    c = compare.compare_chunks(a, b)
    assert c["max_abs"] == pytest.approx(0.02) and c["mean_abs"] == pytest.approx(0.02 / 128) and c["rows"] == 16
    assert c["max_abs_per_column"][5] == pytest.approx(0.02) and c["max_abs_per_column"][0] == 0


def test_compare_refuses_reports_on_different_observations(tmp_path, capsys):
    rows = np.zeros((4, 8)).tolist()
    for name, sha in (("a", "x"), ("b", "y")):
        (tmp_path / f"{name}.json").write_text(json.dumps(report(rows, sha, lora=name == "a")))
    assert compare.main([str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 2
    assert "REFUSED" in capsys.readouterr().out
    assert compare.main([str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--allow-different-observations"]) == 0


def test_compare_on_the_same_observation(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps(report(np.full((4, 8), 0.1).tolist(), lora=True)))
    (tmp_path / "b.json").write_text(json.dumps(report(np.full((4, 8), 0.11).tolist(), lora=False)))
    assert compare.main([str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
    out = capsys.readouterr().out
    assert "vlm: no LoRA" in out and "vlm: LoRA rank 16" in out and "relative" in out


def test_compare_still_reads_the_reports_written_before_the_per_component_settings(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps(report(np.full((4, 8), 0.1).tolist(), lora=True, new_schema=False)))
    (tmp_path / "b.json").write_text(json.dumps(report(np.full((4, 8), 0.1).tolist(), lora=False, new_schema=False)))
    assert compare.main([str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
    out = capsys.readouterr().out
    assert "no LoRA" in out and "LoRA r16/32" in out


# -- the probe's own helpers ----------------------------------------------------------------------------------------------------
def test_the_probe_reads_its_observation_back_into_the_env_format():
    o = obs(1)
    row = to_openpi_obs(o, 0, "insert the peg")
    back = probe.env_obs_of(row)
    for k in (P.EXTERIOR_IMAGE, P.WRIST_IMAGE, P.JOINT_POSITION, P.GRIPPER_POSITION):
        assert np.array_equal(back[k], o[k]), k


def test_set_reads_component_key_value_pairs_as_yaml():
    assert probe.parse_sets(["vlm.use_lora=false", "action_head.lora_rank=8", "action_head.use_tactile=true"]) == \
        {"vlm.use_lora": False, "action_head.lora_rank": 8, "action_head.use_tactile": True}
    assert probe.parse_sets([]) == {}
    for bad in ("vlm.use_lora", "=3"):
        with pytest.raises(ValueError, match="COMPONENT.KEY=VALUE"):
            probe.parse_sets([bad])
