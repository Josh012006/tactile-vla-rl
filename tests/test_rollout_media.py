"""Progress, frames, video, saved observations and the comparison of two probe reports (src/models/{rollout_media,probe,compare}.py).
No openpi, no JAX, no GPU."""
import json

import numpy as np
import pytest

from src.envs import protocol as P
from src.models import compare, probe, rollout_media as media


def obs(n=2, size=224):
    rng = np.random.default_rng(0)
    return {P.EXTERIOR_IMAGE: rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8),
            P.WRIST_IMAGE: rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8),
            P.JOINT_POSITION: np.zeros((n, 7), np.float32), P.GRIPPER_POSITION: np.zeros((n, 1), np.float32),
            P.PEG_SOCKET_OFFSET: np.array([[0.003, 0.004, 0.012], [0.0, 0.0, -0.001]][:n], np.float32),
            P.FT_FORCE: np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]][:n], np.float32)}


# -- progress ----------------------------------------------------------------------------------------------------------------
def test_progress_sample_is_in_millimetres_and_newtons():
    s = media.progress_sample(obs(), 0)
    assert s["xy_mm"] == pytest.approx(5.0, abs=1e-4)           # norm(3 mm, 4 mm)
    assert s["z_mm"] == pytest.approx(12.0, abs=1e-4)
    assert s["force_n"] == pytest.approx(5.0, abs=1e-5)


def test_progress_summary():
    series = [{"t": t, "xy_mm": xy, "z_mm": z, "force_n": f} for t, (xy, z, f) in enumerate([(9.0, 20.0, 1.0), (3.0, 12.0, 7.0), (2.0, 5.0, 2.0), (4.0, 6.0, 1.0)])]
    g = media.progress_summary(series, success=False)
    assert (g["xy_mm_start"], g["xy_mm_end"], g["xy_mm_min"], g["step_of_xy_min"]) == (9.0, 4.0, 2.0, 2)
    assert (g["z_mm_start"], g["z_mm_end"], g["z_mm_min"], g["force_n_max"]) == (20.0, 6.0, 5.0, 7.0)
    assert g["steps_xy_within_tolerance"] == 1 and g["success"] is False and g["samples"] == 4
    assert media.progress_summary([], True) == {"samples": 0, "success": True}


# -- frames and video ----------------------------------------------------------------------------------------------------------
def test_a_frame_is_exterior_next_to_wrist_with_a_text_bar_and_macroblock_sides():
    o = obs()
    f = media.compose_frame(o, 0, t=7, plan_no=2, sample=media.progress_sample(o, 0), success=False)
    assert f.dtype == np.uint8 and f.shape == (2 * 224 + media.BAR, 4 * 224, 3) and f.shape[0] % 16 == 0 and f.shape[1] % 16 == 0
    assert np.array_equal(f[100:104, 300:304], np.repeat(np.repeat(o[P.EXTERIOR_IMAGE][0], 2, 0), 2, 1)[100:104, 300:304])   # the exterior image, enlarged x2
    assert np.array_equal(f[100:104, 448 + 300:448 + 304], np.repeat(np.repeat(o[P.WRIST_IMAGE][0], 2, 0), 2, 1)[100:104, 300:304])
    assert f[448:].any(), "the text bar is empty"
    ok = media.compose_frame(o, 1, 7, 2, media.progress_sample(o, 1), success=True)
    assert (ok[448:, -90:, 1] > 200).any(), "SUCCESS is not written"


def test_a_frame_of_an_odd_camera_size_is_padded_to_macroblocks():
    o = obs(size=100)
    f = media.compose_frame(o, 0, 0, 0, None, False)
    assert f.shape[0] % 16 == 0 and f.shape[1] % 16 == 0


def test_the_video_has_one_frame_per_step(tmp_path):
    frames = [media.compose_frame(obs(), 0, t, 1, media.progress_sample(obs(), 0), False) for t in range(5)]
    path = media.write_video(frames, tmp_path / "v.mp4", fps=10)
    assert path.endswith((".mp4", ".gif")) and (tmp_path / path.rsplit("/", 1)[-1]).stat().st_size > 0
    if path.endswith(".gif"):
        from PIL import Image
        assert Image.open(path).n_frames == 5
    else:
        imageio = pytest.importorskip("imageio.v2")
        assert sum(1 for _ in imageio.get_reader(path)) == 5      # (list() asks for len(): infinite for ffmpeg readers)


def test_the_gif_fallback_is_used_when_mp4_cannot_be_written(tmp_path, monkeypatch):
    import imageio.v2 as imageio
    monkeypatch.setattr(imageio, "get_writer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no ffmpeg")))
    path = media.write_video([media.compose_frame(obs(), 0, t, 1, None, False) for t in range(3)], tmp_path / "v.mp4")
    assert path.endswith(".gif")


# -- saved observations ----------------------------------------------------------------------------------------------------------
def openpi_obs():
    return probe.to_openpi_obs(obs(), 0, "Insert the peg into the socket.")


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
def report(rows, sha="abc", lora=True):
    return {"model": "pi05", "lora": {"use_lora": lora, "vlm_rank": 16, "action_head_rank": 32}, "load_mode": "weight_loader",
            "observation_sha256": sha, "first_chunk": {"rows": rows}}


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
    assert "no LoRA" in out and "LoRA r16/32" in out and "relative" in out
