"""scripts/compare_eval_paired.py: pairing of two evaluations episode by episode."""
import json
import math

import numpy as np
import pytest

from scripts import compare_eval_paired as CE


def result(successes, seeds=None, sha=None):
    n = len(successes)
    seeds = list(range(100, 100 + n)) if seeds is None else seeds
    return {"seeds": seeds, "successes": list(successes),
            "episodes": [{"episode": i, "seed": s, "success": bool(x), "start_sha": (sha or "s") + str(i)} for i, (s, x) in enumerate(zip(seeds, successes))]}


def test_mcnemar_exact_matches_the_binomial_tail():
    assert CE.mcnemar_exact(0, 0) == 1.0
    assert CE.mcnemar_exact(5, 5) == 1.0
    assert CE.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert CE.mcnemar_exact(3, 9) == pytest.approx(2 * sum(math.comb(12, i) for i in range(4)) / 2 ** 12)
    assert CE.mcnemar_exact(9, 3) == CE.mcnemar_exact(3, 9)


def test_single_files_give_the_two_by_two_table():
    a = result([1, 1, 1, 0, 0, 0, 1, 0])
    b = result([1, 0, 0, 0, 1, 0, 0, 0])
    c = CE.compare([a], [b])
    assert (c["both"], c["only_a"], c["only_b"], c["neither"]) == (1, 3, 1, 3)
    assert c["rate_a"] == 0.5 and c["rate_b"] == 0.25 and c["paired_difference"] == pytest.approx(0.25)
    assert c["mcnemar_p"] == pytest.approx(CE.mcnemar_exact(3, 1))


def test_several_files_are_pooled_per_episode():
    a1, a2 = result([1, 1, 0, 0]), result([1, 0, 0, 0])
    b = result([0, 0, 0, 1])
    c = CE.compare([a1, a2], [b])
    assert c["rate_a"] == pytest.approx(0.375) and c["rate_b"] == 0.25
    assert c["episodes_a_never_b_always"] == 1 and c["hard_for_both"] == 1            # episode 3: a never, b always; episode 2: hard for both
    assert "mcnemar_p" not in c


def test_pairing_pins_down_a_gap_the_unpaired_rates_cannot():
    rng = np.random.default_rng(0)
    hard = rng.uniform(size=200)                                                      # the difficulty of the starting state, shared by both policies
    a = result((hard < 0.55).astype(int))
    b = result((hard < 0.45).astype(int))
    c = CE.compare([a], [b])
    assert c["only_b"] == 0 and c["only_a"] > 10 and c["mcnemar_p"] < 0.01


def test_results_with_other_episode_seeds_are_refused_and_other_starts_are_warned():
    a, b = result([1, 0, 1]), result([1, 1, 0], seeds=[1, 2, 3])
    with pytest.raises(ValueError, match="cannot be paired"):
        CE.check_same_episodes([a, b], ["a", "b"])
    c = result([1, 1, 0], sha="other")
    warnings = CE.check_same_episodes([a, c], ["a", "c"])
    assert len(warnings) == 1 and "not bit-identical" in warnings[0]


def test_command_line(tmp_path, capsys):
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(result([1, 1, 1, 0, 0, 0, 1, 0])))
    pb.write_text(json.dumps(result([1, 0, 0, 0, 1, 0, 0, 0])))
    CE.main(["--a", str(pa), "--b", str(pb)])
    out = capsys.readouterr().out
    assert "8 episodes" in out and "McNemar" in out and "only A 3, only B 1" in out
    with pytest.raises(ValueError, match="no `seeds`"):
        (tmp_path / "bad.json").write_text("{}")
        CE.load_result(tmp_path / "bad.json")
