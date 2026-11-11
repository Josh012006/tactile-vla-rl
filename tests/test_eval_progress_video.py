"""src/eval/progress.py and src/eval/video.py: progress of the peg, frames and video of a rollout. No openpi, no JAX, no GPU."""
import numpy as np
import pytest

from src.envs import protocol as P
from src.eval import progress, video


def obs(n=2, size=224):
    rng = np.random.default_rng(0)
    return {P.EXTERIOR_IMAGE: rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8),
            P.WRIST_IMAGE: rng.integers(0, 255, (n, size, size, 3), dtype=np.uint8),
            P.JOINT_POSITION: np.zeros((n, 7), np.float32), P.GRIPPER_POSITION: np.zeros((n, 1), np.float32),
            P.PEG_SOCKET_OFFSET: np.array([[0.003, 0.004, 0.012], [0.0, 0.0, -0.001]][:n], np.float32),
            P.FT_FORCE: np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]][:n], np.float32)}


# -- progress ----------------------------------------------------------------------------------------------------------------
def test_progress_sample_is_in_millimetres_and_newtons():
    s = progress.progress_sample(obs(), 0)
    assert s["xy_mm"] == pytest.approx(5.0, abs=1e-4)           # norm(3 mm, 4 mm)
    assert s["z_mm"] == pytest.approx(12.0, abs=1e-4)
    assert s["force_n"] == pytest.approx(5.0, abs=1e-5)


def test_progress_summary():
    series = [{"t": t, "xy_mm": xy, "z_mm": z, "force_n": f} for t, (xy, z, f) in enumerate([(9.0, 20.0, 1.0), (3.0, 12.0, 7.0), (2.0, 5.0, 2.0), (4.0, 6.0, 1.0)])]
    g = progress.progress_summary(series, success=False)
    assert (g["xy_mm_start"], g["xy_mm_end"], g["xy_mm_min"], g["step_of_xy_min"]) == (9.0, 4.0, 2.0, 2)
    assert (g["z_mm_start"], g["z_mm_end"], g["z_mm_min"], g["force_n_max"]) == (20.0, 6.0, 5.0, 7.0)
    assert g["steps_xy_within_tolerance"] == 1 and g["success"] is False and g["samples"] == 4
    assert progress.progress_summary([], True) == {"samples": 0, "success": True}


# -- frames and video ----------------------------------------------------------------------------------------------------------
def test_a_frame_is_exterior_next_to_wrist_with_a_text_bar_and_macroblock_sides():
    o = obs()
    f = video.compose_frame(o, 0, t=7, plan_no=2, sample=progress.progress_sample(o, 0), success=False)
    assert f.dtype == np.uint8 and f.shape == (2 * 224 + video.BAR, 4 * 224, 3) and f.shape[0] % 16 == 0 and f.shape[1] % 16 == 0
    assert np.array_equal(f[100:104, 300:304], np.repeat(np.repeat(o[P.EXTERIOR_IMAGE][0], 2, 0), 2, 1)[100:104, 300:304])   # the exterior image, enlarged x2
    assert np.array_equal(f[100:104, 448 + 300:448 + 304], np.repeat(np.repeat(o[P.WRIST_IMAGE][0], 2, 0), 2, 1)[100:104, 300:304])
    assert f[448:].any(), "the text bar is empty"
    ok = video.compose_frame(o, 1, 7, 2, progress.progress_sample(o, 1), success=True)
    assert (ok[448:, -90:, 1] > 200).any(), "SUCCESS is not written"


def test_a_frame_of_an_odd_camera_size_is_padded_to_macroblocks():
    o = obs(size=100)
    f = video.compose_frame(o, 0, 0, 0, None, False)
    assert f.shape[0] % 16 == 0 and f.shape[1] % 16 == 0


def test_the_video_has_one_frame_per_step(tmp_path):
    frames = [video.compose_frame(obs(), 0, t, 1, progress.progress_sample(obs(), 0), False) for t in range(5)]
    path = video.write_video(frames, tmp_path / "v.mp4", fps=10)
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
    path = video.write_video([video.compose_frame(obs(), 0, t, 1, None, False) for t in range(3)], tmp_path / "v.mp4")
    assert path.endswith(".gif")
