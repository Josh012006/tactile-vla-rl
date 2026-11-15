"""scripts/fast_stats_fit.py without openpi: the fit of normalised values, the state bins, the chunks a pipeline normalises, and the table of evaluation results."""
import json

import numpy as np

from scripts import fast_stats_fit as FS
from test_fast_tokens import H, identity_pipeline


def test_fit_of_a_narrow_band_and_of_values_beyond_the_quantile_range():
    v = np.linspace(-0.1, 0.1, 1000)
    f = FS.fit_of(v)
    assert f["out_pct"] == 0.0 and abs(f["use"] - 0.098) < 0.002 and abs(f["mid"]) < 1e-9          # a tenth of the range, centred
    w = np.concatenate([np.zeros(950), np.full(50, 1.5)])
    g = FS.fit_of(w)
    assert g["out_pct"] == 5.0 and g["p99"] == 1.5 and g["mid"] > 0.7                                # 5 % beyond +1: the 99th percentile is out of the range too
    assert FS.fit_of(np.zeros((4, 3, 2)))["std"] == 0.0                                             # any shape


def test_state_bins_are_those_of_openpis_discretisation_and_values_outside_fall_in_the_end_bins():
    assert FS.state_bins_used(np.full(10, 0.3)) == 1
    assert FS.state_bins_used(np.linspace(-1, 0.999, 256)) == 256                                    # one value per bin
    assert FS.state_bins_used(np.linspace(-0.05, 0.05, 1000)) <= 14                                  # a narrow band uses a few of the 256 bins (a tenth of the range: about 13)
    assert FS.state_bins_used(np.array([2.0, 3.0, 9.0])) == 1                                        # all in the last bin


def write_demos(folder, n_demos=3, n=12):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_demos):
        q = np.cumsum(rng.normal(0, 0.02, (n, 7)), axis=0).astype(np.float32)
        np.savez(folder / f"ep_{i:04d}_seed{i}.npz", joint_position=q, gripper_position=np.full((n, 1), 0.5, np.float32),
                 actions=np.concatenate([q, np.ones((n, 1), np.float32)], axis=1), prompt="Insert the peg into the socket.")


def test_every_chunk_of_the_demonstrations_goes_through_the_pipeline(tmp_path):
    write_demos(tmp_path / "demos")
    states, actions = FS.normalised_inputs(identity_pipeline(), [str(tmp_path / "demos")], horizon=H)
    assert states.shape == (36, 8) and actions.shape == (36, H, 8)                                   # one chunk per step (the tail repeats the last action)
    report = FS.fit_report(states, actions)
    assert report["chunks"] == 36 and list(report["state"]) == FS.JOINT_NAMES and list(report["actions"]) == FS.JOINT_NAMES
    assert report["chunks_out_pct"] == 0.0 and report["state"]["gripper"]["bins"] == 1
    lines = FS.format_fit(report)
    assert any(l.startswith("  state") for l in lines) and any(l.startswith("  actio") for l in lines)


def write_results(folder, success, failed, chunks=1000, steps=(0, 200, 1000, 2000)):
    folder.mkdir(parents=True)
    for s, k, f in zip(steps, success, failed):
        p = k / 100
        (folder / f"{s}.json").write_text(json.dumps({"step": s, "success_rate": p, "success_se": (p * (1 - p) / 100) ** 0.5, "decode_chunks": chunks if f is not None else None,
                                                      "decode_failed": f}))
    (folder / "0.jsonl").write_text("{}")                                                             # the episode log is not a result file


def test_the_evaluation_table_gives_success_and_the_share_of_undecodable_chunks(tmp_path):
    a, b = tmp_path / "a" / "eval" / "results", tmp_path / "b" / "eval" / "results"
    write_results(a, [0, 53, 50, 60], [0, 1, 2, 3])
    write_results(b, [0, 17, 40, 44], [0, 20, 40, 50])
    rows = FS.eval_rows(a)
    assert [r["step"] for r in rows] == [0, 200, 1000, 2000] and rows[2]["decode_failed_pct"] == 0.2 and abs(rows[1]["success_pct"] - 53) < 1e-9
    s = FS.run_summary(rows)
    assert s["checkpoints"] == 2 and s["success_pct"] == 55.0 and abs(s["decode_failed_pct"] - 0.25) < 1e-9          # steps 1000 and 2000
    lines = FS.format_runs(["run_a", "run_b"], [rows, FS.eval_rows(b)])
    assert any("2000" in l for l in lines) and any(l.strip().startswith("run_b: from step 1000") and "4.5 %" in l for l in lines)


def test_main_tabulates_runs_without_any_model(tmp_path, capsys):
    for name, success in (("run_a", [0, 53, 50, 60]), ("run_b", [0, 17, 40, 44])):
        run = tmp_path / name
        run.mkdir()
        (run / "config.json").write_text("{}")
        write_results(run / "eval" / "results", success, [None] * 4)
    out = tmp_path / "report.json"
    assert FS.main(["--runs", str(tmp_path / "run_a"), str(tmp_path / "run_b"), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "run_a" in text and "run_b" in text and "mean success 55.0 %" in text
    assert set(json.loads(out.read_text())["runs"]) == {"run_a", "run_b"}
