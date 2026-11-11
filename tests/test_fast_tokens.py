"""scripts/fast_tokens.py: chunks of a demonstration, token counting, truncation, round trip, report.

The FAST tokenizer itself (Hugging Face Hub) is replaced by a fake with the same two methods; it is lossy and variable-length like the real
one, so the logic around it is tested. One test puts openpi's real transforms (DroidInputs, DeltaActions, quantile Normalize, Unnormalize,
AbsoluteActions) around the fake, with synthetic statistics: the round trip must give back the original absolute chunk. That one needs the
`vla` dependency group and is skipped without it.
"""
import json

import numpy as np
import pytest

from scripts import fast_tokens as ft

H, D = 10, 8
PREFIX, ACTION_HEAD = 12, 3          # fake prefix length, tokens of "Action: "
QUANT = 100.0


class FakeFastTokenizer:
    """Same interface as openpi's FASTTokenizer. Postfix = 3 head tokens + the chunk quantised to 0.01, with the trailing rows that repeat
    the previous row dropped (variable length, lossy, like a compression) + 2 closing tokens. The prefix is PREFIX tokens, mask 0."""

    def __init__(self, max_len=1024):
        self.max_len = max_len

    def tokenize(self, prompt, state, actions):
        rows = np.round(np.asarray(actions) * QUANT).astype(np.int64)
        changes = [0] + [i for i in range(1, len(rows)) if np.any(rows[i] != rows[i - 1])]
        rows = rows[: changes[-1] + 1]
        body = [int(v) + 10_000 for v in rows.reshape(-1)]
        tokens = [1] * PREFIX + [7] * ACTION_HEAD + body + [8, 2]
        ar = [0] * PREFIX + [1] * (len(tokens) - PREFIX)
        n = len(tokens)
        pad = max(self.max_len - n, 0)
        mask = [True] * n + [False] * pad
        return (np.asarray((tokens + [0] * pad)[: self.max_len]), np.asarray(mask[: self.max_len]),
                np.asarray((ar + [0] * pad)[: self.max_len]), np.asarray(([False] * PREFIX + [True] * (n - PREFIX) + [False] * pad)[: self.max_len]))

    def extract_actions(self, tokens, horizon, dim):
        tokens = np.asarray(tokens)
        if len(tokens) < ACTION_HEAD or tokens[0] != 7:                 # "Action: " lost: openpi returns zeros
            return np.zeros((horizon, dim), np.float32)
        body = tokens[ACTION_HEAD:]
        body = body[(body != 8) & (body != 2)]
        rows = ((body[: len(body) // dim * dim].astype(np.float32) - 10_000) / QUANT).reshape(-1, dim)
        if len(rows) == 0:
            return np.zeros((horizon, dim), np.float32)
        out = np.repeat(rows[-1:], horizon, axis=0)                      # the dropped rows repeat the last kept one
        out[: min(len(rows), horizon)] = rows[:horizon]
        return out


def identity_pipeline(max_len=1024):
    return ft.Pipeline(to_model=lambda item, prompt: (np.concatenate([item.joints, item.gripper]), item.actions),
                       from_model=lambda state, chunk: chunk, tokenizer=FakeFastTokenizer(max_len))


def demo(n=20, still_from=14, seed=0):
    """n frames; the arm moves up to `still_from`, then the targets stay put (so late chunks have short token sequences)."""
    rng = np.random.default_rng(seed)
    still_from = min(still_from, n - 1)
    q = np.cumsum(rng.normal(0, 0.05, (n, 7)), axis=0).astype(np.float32)
    q[still_from:] = q[still_from]
    grip = np.full((n, 1), 0.9, np.float32)
    actions = np.concatenate([q, np.ones((n, 1), np.float32)], axis=1)
    return {"joint_position": q, "gripper_position": grip, "actions": actions, "prompt": "Insert the peg into the socket."}


# -- chunks ------------------------------------------------------------------------------------------------------------------------
def test_chunk_items_pad_repeats_the_last_action_and_drop_skips_the_tail():
    d = demo()
    padded = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "pad")
    dropped = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "drop")
    assert len(padded) == 20 and len(dropped) == 20 - H + 1
    last = padded[-1]
    assert last.actions.shape == (H, 8) and np.all(last.actions == d["actions"][-1])
    assert np.array_equal(padded[3].actions, d["actions"][3:3 + H])
    assert np.array_equal(padded[3].joints, d["joint_position"][3]) and padded[3].gripper.shape == (1,)
    with pytest.raises(ValueError):
        ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "other")


def test_load_demos_reads_ep_files_only(tmp_path):
    d = demo(12)
    for name in ("ep_0000_seed0.npz", "ep_0001_seed1.npz", "fail_seed2_101010.npz"):
        np.savez(tmp_path / name, exterior_image=np.zeros((12, 2, 2, 3), np.uint8), prompt=np.array(d["prompt"]), **{k: d[k] for k in ("joint_position", "gripper_position", "actions")})
    assert [n for n, _ in ft.load_demos(str(tmp_path))] == ["ep_0000_seed0.npz", "ep_0001_seed1.npz"]
    assert len(ft.load_demos(str(tmp_path), limit=1)) == 1
    assert ft.load_demos(str(tmp_path))[0][1]["prompt"] == d["prompt"]
    with pytest.raises(FileNotFoundError):
        ft.load_demos(str(tmp_path / "nothing"))


# -- measurement -------------------------------------------------------------------------------------------------------------------
def test_counts_and_a_lossless_round_trip_when_nothing_is_truncated():
    d, pipe = demo(), identity_pipeline()
    item = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "pad")[2]
    r = ft.measure_item(pipe, item, d["prompt"], max_token_len=1000)
    assert r.total == r.prefix + r.postfix and r.prefix == PREFIX
    assert r.postfix == ACTION_HEAD + 2 + H * D                         # a moving chunk: every row is kept (no trailing repeat)
    assert not r.failed and not r.zeros
    assert r.joint_err <= 0.5 / QUANT + 1e-6 and r.gripper_err <= 0.5 / QUANT + 1e-6      # the quantisation of the fake, nothing else


def test_a_still_chunk_is_shorter():
    d, pipe = demo(), identity_pipeline()
    items = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "pad")
    moving = ft.measure_item(pipe, items[2], d["prompt"], 1000)
    still = ft.measure_item(pipe, items[-1], d["prompt"], 1000)
    assert still.total < moving.total


def test_truncation_at_max_token_len_loses_the_chunk_and_is_reported():
    d, pipe = demo(), identity_pipeline()
    item = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "pad")[2]
    full = ft.measure_item(pipe, item, d["prompt"], 1000)
    cut = ft.measure_item(pipe, item, d["prompt"], max_token_len=full.total - 40)
    assert cut.total == full.total                                      # the count is the needed length, not the kept one
    assert cut.joint_err > 10 * full.joint_err                          # the last rows are gone: they decode as zeros
    gone = ft.measure_item(pipe, item, d["prompt"], max_token_len=PREFIX)         # not even "Action: " survives
    assert gone.zeros


def test_a_decoding_error_is_a_result_not_a_crash():
    class Broken(FakeFastTokenizer):
        def extract_actions(self, tokens, horizon, dim):
            raise ValueError("bad tokens")
    d = demo()
    pipe = ft.Pipeline(identity_pipeline().to_model, lambda s, c: c, Broken())
    item = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "pad")[0]
    r = ft.measure_item(pipe, item, d["prompt"], 1000)
    assert r.failed and np.isnan(r.joint_err)
    s = ft.summarize([r], 1000, 256)
    assert s["decode_failed"] == 1.0 and s["joint_err_rad"] is None


def test_a_measuring_tokenizer_that_is_too_short_is_refused():
    d = demo()
    item = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "pad")[2]
    with pytest.raises(ValueError, match="measure-len"):
        ft.measure_item(identity_pipeline(max_len=30), item, d["prompt"], 1000)


def test_summary_and_recommendation():
    recs = [ft.Record(100, 12, 88, False, False, 0.01, 0.0), ft.Record(190, 12, 178, False, False, 0.02, 0.0),
            ft.Record(300, 12, 288, False, False, 0.5, 0.0), ft.Record(100, 12, 88, False, False, 0.01, 0.0)]
    s = ft.summarize(recs, max_token_len=180, max_decoding_steps=256)
    assert s["chunks"] == 4 and s["tokens_total"]["max"] == 300
    assert s["over_max_token_len"] == 0.5 and s["over_max_decoding_steps"] == 0.25
    rec = ft.recommend(recs, margin=0.2)
    assert rec["max_token_len_covering_all"] == 300 and rec["max_token_len_with_margin"] == 360 and rec["max_decoding_steps_covering_all"] == 288


# -- the real openpi transforms around the fake tokenizer ------------------------------------------------------------------------------
def test_openpi_transforms_round_trip_to_the_absolute_chunk():
    pytest.importorskip("jax")
    pytest.importorskip("openpi")
    import openpi.models.model as _model
    import openpi.policies.droid_policy as droid
    import openpi.transforms as transforms
    from openpi.shared.normalize import NormStats

    mask = transforms.make_bool_mask(7, -1)
    group = transforms.Group(inputs=[droid.DroidInputs(model_type=_model.ModelType.PI0_FAST)], outputs=[droid.DroidOutputs(8)])
    group = group.push(inputs=[transforms.DeltaActions(mask)], outputs=[transforms.AbsoluteActions(mask)])    # as RLDSDroidDataConfig does
    stats = {"state": NormStats(mean=np.zeros(8), std=np.ones(8), q01=np.full(8, -3.0), q99=np.full(8, 3.0)),
             "actions": NormStats(mean=np.zeros(8), std=np.ones(8), q01=np.full(8, -0.5), q99=np.full(8, 0.5))}
    pipe = ft.make_pipeline(group, stats, True, FakeFastTokenizer())
    d = demo()
    items = ft.chunk_items(d["joint_position"], d["gripper_position"], d["actions"], H, "pad")
    before = items[3].actions.copy()
    state, actions = pipe.to_model(items[3], d["prompt"])
    assert state.shape == (8,) and actions.shape == (H, 8)
    assert np.array_equal(items[3].actions, before)                     # DeltaActions works in place: the item must not be touched
    assert np.allclose(actions[0, :7], 2 * (items[3].actions[0, :7] - items[3].joints), atol=1e-4)     # delta, then quantile [-0.5, 0.5] -> [-1, 1]
    for item in items:
        r = ft.measure_item(pipe, item, d["prompt"], 1000)
        assert not r.failed and r.joint_err < 0.02 and r.gripper_err < 0.02, (r.joint_err, r.gripper_err)


# -- the command line ----------------------------------------------------------------------------------------------------------------
def test_main_writes_the_report_and_prints_the_recommendation(tmp_path, capsys):
    folder = tmp_path / "demos"
    folder.mkdir()
    for i in range(2):
        d = demo(seed=i)
        np.savez(folder / f"ep_{i:04d}_seed{i}.npz", prompt=np.array(d["prompt"]), **{k: d[k] for k in ("joint_position", "gripper_position", "actions")})
    out = tmp_path / "report.json"
    code = ft.main(["--demos-dir", str(folder), "--horizon", "10", "--horizon", "5", "--out", str(out)], pipeline=identity_pipeline())
    assert code == 0
    report = json.loads(out.read_text())
    assert [r["horizon"] for r in report["results"]] == [10, 5] and all(r["chunks"] == 40 for r in report["results"])
    assert report["max_token_len"] == 180                                # the YAML's value (configs/model/pi0_fast.yaml)
    assert report["results"][1]["tokens_total"]["max"] < report["results"][0]["tokens_total"]["max"]     # a shorter chunk needs fewer tokens
    text = capsys.readouterr().out
    assert "smallest max_token_len covering every measured chunk" in text


def test_main_refuses_a_model_that_is_not_pi0_fast(capsys):
    assert ft.main(["--model", "pi05"], pipeline=identity_pipeline()) == 2
    assert "pi0_fast" in capsys.readouterr().err
