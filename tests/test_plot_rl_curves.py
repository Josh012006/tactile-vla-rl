"""scripts/plot_rl_curves.py: round curves of several runs from the result files on disk, paired difference, reference line."""
import json
import math
import os

import numpy as np
import pytest

from scripts import plot_rl_curves as P


def write(path, successes, seeds=None, mtime=None):
    n = len(successes)
    seeds = list(range(1000, 1000 + n)) if seeds is None else seeds
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seeds": seeds, "successes": [bool(x) for x in successes]}))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def bits(n, k, rng):
    x = np.zeros(n, dtype=bool)
    x[rng.choice(n, k, replace=False)] = True
    return x


def test_collect_reads_rounds_from_several_folders_and_ignores_other_files(tmp_path):
    write(tmp_path / "eval_rl_a/results/0.json", [1, 0, 1, 0])
    write(tmp_path / "eval_rl_b/results/12.json", [1, 1, 1, 0])
    write(tmp_path / "eval_rl_b/results/summary.json", [1])
    found = P.collect(str(tmp_path / "eval_rl_*/results/*.json"))
    assert sorted(found) == [0, 12]


def test_a_round_found_twice_keeps_the_most_episodes_then_the_newest(tmp_path):
    write(tmp_path / "a/results/3.json", [1, 0], mtime=2000)
    write(tmp_path / "b/results/3.json", [1, 1, 0, 0], mtime=1000)
    assert len(P.collect(str(tmp_path / "*/results/*.json"))[3][1]["successes"]) == 4
    write(tmp_path / "c/results/3.json", [0, 0, 0, 0], mtime=3000)
    assert P.collect(str(tmp_path / "*/results/*.json"))[3][0].parent.parent.name == "c"


def test_summary_is_rate_standard_error_and_wilson_interval():
    s = P.summarise({"successes": [1] * 163 + [0] * 337})
    assert s["episodes"] == 500 and s["successes"] == 163
    assert s["rate"] == pytest.approx(0.326) and s["se"] == pytest.approx(math.sqrt(0.326 * 0.674 / 500))
    assert s["wilson_low"] < s["rate"] < s["wilson_high"]


def test_paired_difference_uses_the_same_episodes_and_refuses_other_seeds():
    a = {"seeds": [1, 2, 3, 4], "successes": [1, 0, 0, 0]}
    b = {"seeds": [1, 2, 3, 4], "successes": [1, 1, 1, 0]}
    p = P.paired(a, b)
    assert p["difference"] == pytest.approx(0.5) and p["episodes"] == 4
    assert P.paired(a, {"seeds": [1, 2, 3, 5], "successes": [1, 1, 1, 0]}) is None


def test_build_two_curves_with_reference_and_paired_rows(tmp_path):
    rng = np.random.default_rng(0)
    for r, (k1, k2) in {0: (150, 150), 1: (160, 175), 2: (165, 190)}.items():
        write(tmp_path / f"no/eval_a/results/{r}.json", bits(500, k1, rng))
        write(tmp_path / f"yes/eval_a/results/{r}.json", bits(500, k2, rng))
    write(tmp_path / "sft/1600.json", bits(500, 230, rng))
    data = P.build([("no touch", str(tmp_path / "no/*/results/*.json")), ("touch", str(tmp_path / "yes/*/results/*.json"))], ("VLA alone", str(tmp_path / "sft/1600.json")))
    assert [c["label"] for c in data["curves"]] == ["no touch", "touch"]
    assert sorted(data["curves"][0]["rounds"], key=int) == ["0", "1", "2"]
    assert data["reference"]["rate"] == pytest.approx(0.46)
    assert data["paired"]["second_minus_first"] == "touch - no touch" and set(data["paired"]["rounds"]) == {"0", "1", "2"}
    assert data["paired"]["rounds"]["2"]["difference"] == pytest.approx(0.05)
    assert data["warnings"] == []


def test_unequal_episode_counts_and_other_seeds_are_warned(tmp_path):
    write(tmp_path / "a/results/0.json", [1, 0, 1, 0])
    write(tmp_path / "a/results/1.json", [1, 0])
    write(tmp_path / "b/results/0.json", [1, 0, 1, 0], seeds=[7, 8, 9, 10])
    data = P.build([("a", str(tmp_path / "a/results/*.json")), ("b", str(tmp_path / "b/results/*.json"))])
    text = " ".join(data["warnings"])
    assert "not evaluated on the same number of episodes" in text and "other episode seeds" in text


def test_no_match_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no <round>.json"):
        P.build([("x", str(tmp_path / "nothing/*.json"))])


def test_main_writes_figure_table_and_numbers(tmp_path, capsys):
    rng = np.random.default_rng(1)
    for r in range(3):
        write(tmp_path / f"no/results/{r}.json", bits(100, 40 + r, rng))
        write(tmp_path / f"yes/results/{r}.json", bits(100, 42 + 2 * r, rng))
    write(tmp_path / "ref.json", bits(100, 46, rng))
    out = tmp_path / "fig"
    P.main(["--curve", f"no touch={tmp_path}/no/results/*.json", "--curve", f"touch={tmp_path}/yes/results/*.json", "--reference", f"VLA alone={tmp_path}/ref.json", "--out", str(out), "--title", "t"])
    for name in ("curve.png", "curve.pdf", "curve.json", "table.md"):
        assert (out / name).stat().st_size > 0
    table = (out / "table.md").read_text()
    assert "| round | no touch | touch | paired difference |" in table and "VLA alone: 46.0" in table
    assert json.loads((out / "curve.json").read_text())["paired"] is not None


def test_one_curve_has_no_paired_panel(tmp_path):
    write(tmp_path / "a/results/0.json", [1, 0, 1, 0])
    out = tmp_path / "fig"
    P.main(["--curve", f"only={tmp_path}/a/results/*.json", "--out", str(out)])
    assert json.loads((out / "curve.json").read_text())["paired"] is None and (out / "curve.png").exists()


def test_a_label_may_contain_an_equal_sign():
    assert P.split_label("sparse 0/1, alpha = 0.01=runs/a/*/results/*.json") == ("sparse 0/1, alpha = 0.01", "runs/a/*/results/*.json")


def test_relative_curves_start_at_zero_and_count_the_change_on_the_same_episodes(tmp_path):
    # two runs that do not start at the same level (their critics differ after the warm-up) and the same +10 episodes gained at round 1
    write(tmp_path / "a/results/0.json", [1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    write(tmp_path / "a/results/1.json", [1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    write(tmp_path / "b/results/0.json", [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    write(tmp_path / "b/results/1.json", [1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    data = P.build([("a", str(tmp_path / "a/results/*.json")), ("b", str(tmp_path / "b/results/*.json"))], relative=True)
    a, b = (c["rounds"] for c in data["curves"])
    assert a["0"]["rate"] == 0 and b["0"]["rate"] == 0 and a["0"]["se"] == 0
    assert a["1"]["rate"] == pytest.approx(0.2) and b["1"]["rate"] == pytest.approx(0.1)
    assert data["paired"] is None and data["reference"] is None and data["warnings"] == []


def test_relative_is_refused_with_a_reference_and_warns_on_other_seeds_or_a_missing_round_0(tmp_path):
    write(tmp_path / "a/results/1.json", [1, 0, 1, 0])
    write(tmp_path / "a/results/2.json", [1, 1, 1, 0], seeds=[7, 8, 9, 10])
    write(tmp_path / "ref.json", [1, 0, 1, 0])
    spec = [("a", str(tmp_path / "a/results/*.json"))]
    with pytest.raises(ValueError, match="does not go with"):
        P.build(spec, ("ref", str(tmp_path / "ref.json")), relative=True)
    text = " ".join(P.build(spec, relative=True)["warnings"])
    assert "no round 0" in text and "other episode seeds" in text


def test_main_relative_writes_a_signed_table_and_the_figure(tmp_path):
    rng = np.random.default_rng(2)
    for r in range(3):
        write(tmp_path / f"no/results/{r}.json", bits(100, 40 + 3 * r, rng))
        write(tmp_path / f"yes/results/{r}.json", bits(100, 45 - 2 * r, rng))
    out = tmp_path / "fig"
    P.main(["--curve", f"no={tmp_path}/no/results/*.json", "--curve", f"yes={tmp_path}/yes/results/*.json", "--out", str(out), "--relative"])
    table = (out / "table.md").read_text()
    assert "| 0 | +0.0 +- 0.0 | +0.0 +- 0.0 |" in table and "| round | no | yes |" in table
    assert (out / "curve.png").stat().st_size > 0 and json.loads((out / "curve.json").read_text())["relative"] is True


def test_bad_spec_is_refused():
    with pytest.raises(ValueError):
        P.split_label("no equals sign")
