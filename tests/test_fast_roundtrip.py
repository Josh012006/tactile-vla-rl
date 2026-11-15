"""scripts/fast_roundtrip.py: what the round trip keeps of a chunk's motion (relative to the current joints), measured around the fake FAST tokenizer of tests/test_fast_tokens.py.

The fake rounds the (delta) actions to 1 / QUANT: a fine QUANT loses nothing that matters, a coarse one rounds the small motions of the demonstrations to zero, which is what the script must show
(`kept` near 0, `killed` near 100 %). The pipeline here applies a delta on the joints like openpi's DeltaActions: a chunk decoded to zeros is a chunk without motion."""
import json

import numpy as np
import pytest

import test_fast_tokens as tft
from scripts import fast_roundtrip as fr
from scripts import fast_tokens as ft


def delta_pipeline(max_len=1024):
    """to_model: joints become deltas of the current joints; from_model adds the state back (the normalised state here is the joints themselves)."""
    def to_model(item, prompt):
        acts = item.actions.copy()
        acts[:, :7] -= item.joints
        return np.concatenate([item.joints, item.gripper]), acts

    def from_model(state, chunk):
        out = np.array(chunk, copy=True)
        out[:, :7] += np.asarray(state)[:7]
        return out

    return ft.Pipeline(to_model, from_model, tft.FakeFastTokenizer(max_len))


def small_demo(n=30, still_from=25, seed=0, step=0.004):
    """Like tft.demo but with the small motions of a teleoperated insertion (about 0.01 rad per chunk), far below the coarse quantisation step."""
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.normal(0, step, (n, 7)), axis=0).astype(np.float32)
    q[still_from:] = q[still_from]
    actions = np.concatenate([q, np.ones((n, 1), np.float32)], axis=1)
    return {"joint_position": q, "gripper_position": np.full((n, 1), 0.9, np.float32), "actions": actions, "prompt": "Insert the peg into the socket."}


def folder(tmp_path, n_demos=2):
    path = tmp_path / "demos"
    path.mkdir()
    for i in range(n_demos):
        d = small_demo(seed=i)
        np.savez(path / f"ep_{i}.npz", joint_position=d["joint_position"], gripper_position=d["gripper_position"], actions=d["actions"], prompt=d["prompt"])
    return str(path)


def test_a_fine_quantisation_keeps_the_motion(tmp_path, monkeypatch):
    monkeypatch.setattr(tft, "QUANT", 1000.0)
    s = fr.measure(delta_pipeline(), [folder(tmp_path)], horizon=tft.H, max_token_len=1024)
    assert s["decode_failed"] == 0 and s["decoded_all_zeros"] == 0
    assert s["error_over_motion"]["share_above"] == 0 and s["kept"]["share_killed"] == 0
    assert s["kept"]["mean"] == pytest.approx(1.0, abs=0.05) and s["gain"]["mean"] == pytest.approx(1.0, abs=0.05)
    assert all(v["rms_error_rad"] < 0.002 for v in s["joints"].values())


def test_a_coarse_quantisation_kills_the_small_motions_and_the_script_shows_it(tmp_path, monkeypatch):
    path = folder(tmp_path)
    monkeypatch.setattr(tft, "QUANT", 1000.0)
    fine = fr.measure(delta_pipeline(), [path], horizon=tft.H, max_token_len=1024)
    monkeypatch.setattr(tft, "QUANT", 2.0)                               # steps of 0.5 rad: every motion of these demonstrations (a few hundredths of a rad) rounds to 0
    coarse = fr.measure(delta_pipeline(), [path], horizon=tft.H, max_token_len=1024)
    assert coarse["kept"]["share_killed"] > 0.9 and coarse["kept"]["mean"] < 0.1
    assert coarse["error_over_motion"]["mean"] > 0.9 and coarse["error_over_motion"]["share_above"] > 0.9
    assert coarse["error_over_motion"]["mean"] > fine["error_over_motion"]["mean"] + 0.5
    assert coarse["gain"]["mean"] < fine["gain"]["mean"]


def test_still_chunks_have_no_ratio_and_are_counted_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(tft, "QUANT", 1000.0)
    s = fr.measure(delta_pipeline(), [folder(tmp_path)], horizon=tft.H, max_token_len=1024)
    assert s["still"] > 0                                                # the demos stop moving at step 25: the last chunks repeat one position
    assert s["chunks"] == 2 * 30


def test_truncation_at_max_token_len_loses_chunks_and_is_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(tft, "QUANT", 1000.0)
    s = fr.measure(delta_pipeline(), [folder(tmp_path)], horizon=tft.H, max_token_len=tft.PREFIX + 3)       # only the prefix and the head of the postfix survive
    assert s["over_max_token_len"] > 0.9
    assert s["decode_failed"] + s["decoded_all_zeros"] > 0 or s["kept"]["mean"] < 0.5          # what is cut away is lost: the decoded chunk keeps less of the motion


def test_a_chunk_that_cannot_be_decoded_is_a_result_not_a_crash():
    class Broken(tft.FakeFastTokenizer):
        def extract_actions(self, tokens, horizon, dim):
            raise ValueError("cannot decode")

    pipe = delta_pipeline()
    pipe.tokenizer = Broken()
    d = tft.demo()
    item = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], tft.H)[0]
    r = fr.round_trip(pipe, item, d["prompt"], 1024)
    assert r["failed"] and r["decoded"] is None
    s = fr.summarize([r], 1024)
    assert s["decode_failed"] == 1.0 and s["joints"] == {}


def test_the_report_is_readable_and_serialisable(tmp_path, monkeypatch):
    monkeypatch.setattr(tft, "QUANT", 100.0)
    s = fr.measure(delta_pipeline(), [folder(tmp_path, 1)], horizon=tft.H, max_token_len=1024)
    lines = fr.format_summary("fake", s)
    assert lines[0] == "statistics: fake" and any("killed" in line for line in lines) and sum(line.strip().startswith("joint") for line in lines) == 7
    json.dumps(s)
