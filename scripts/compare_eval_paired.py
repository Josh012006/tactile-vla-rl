"""Compare the evaluations of two policies EPISODE BY EPISODE (they start from the same states: same seeds, same `start_sha`), from the result files the evaluation writes
(`<output-dir>/results/<step>.json`, `successes` + `episodes`), without any simulation: it only reads JSON.

    python -m scripts.compare_eval_paired \
        --a runs/sft/peg_insert/peg_insert_pi05_lora_teleop/eval/results/4000.json \
        --b runs/sft/probing/_cost_expoft_teleop_4000/results/0.json

`--a` / `--b` take one file or several (the plateau of a curve: steps 2200 to 4000). With several, an episode's score is the share of the files in which it succeeded.

What it answers: are the two policies failing on the SAME starting states (hard states, a gap that is small once paired) or on different ones? The paired test is more sensitive than
the comparison of two success rates, because the difficulty of a state is shared by both policies. One file against one: exact McNemar test on the discordant episodes. Several
files: the mean paired difference of the per-episode scores with its standard error over episodes. The checkpoints of one training run are not independent draws of the training
(same seed, same data): the standard error says how well the episodes pin the gap down, NOT how much two training runs differ by chance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def load_result(path) -> dict:
    data = json.loads(Path(path).read_text())
    for key in ("seeds", "successes"):
        if key not in data:
            raise ValueError(f"{path}: no `{key}` (is it a result file of src/eval/episodes.py?)")
    if len(data["seeds"]) != len(data["successes"]):
        raise ValueError(f"{path}: {len(data['seeds'])} seeds for {len(data['successes'])} outcomes")
    return data


def check_same_episodes(results: Sequence[dict], names: Sequence[str]) -> List[str]:
    """Refuse results that do not share the episode seeds (they are not paired); warn, in the returned list, about starts that are not bit-identical."""
    warnings: List[str] = []
    first = results[0]
    for r, name in zip(results[1:], names[1:]):
        if [int(s) for s in r["seeds"]] != [int(s) for s in first["seeds"]]:
            raise ValueError(f"{name} was evaluated with other episode seeds than {names[0]}: the episodes cannot be paired")
        a = {e["episode"]: e.get("start_sha") for e in first.get("episodes", [])}
        b = {e["episode"]: e.get("start_sha") for e in r.get("episodes", [])}
        differ = sorted(i for i in a if i in b and a[i] != b[i])
        if differ:
            warnings.append(f"{name}: the starting state of {len(differ)} episodes is not bit-identical to {names[0]}'s (other GPU model or renderer?), first: {differ[:5]}")
    return warnings


def scores(results: Sequence[dict]) -> np.ndarray:
    """(episodes,) share of the results in which each episode succeeded."""
    return np.mean([[bool(x) for x in r["successes"]] for r in results], axis=0)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value (binomial, p = 1/2) of `only_a` against `only_b` discordant episodes."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a: Sequence[dict], b: Sequence[dict]) -> dict:
    sa, sb = scores(a), scores(b)
    d = sa - sb
    n = len(d)
    out: Dict[str, object] = {
        "episodes": n, "files_a": len(a), "files_b": len(b), "rate_a": float(sa.mean()), "rate_b": float(sb.mean()), "paired_difference": float(d.mean()),
        "paired_se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "correlation_of_episode_scores": float(np.corrcoef(sa, sb)[0, 1]) if sa.std() > 0 and sb.std() > 0 else None,
    }
    if len(a) == 1 and len(b) == 1:
        xa, xb = sa > 0.5, sb > 0.5
        both, only_a, only_b = int((xa & xb).sum()), int((xa & ~xb).sum()), int((~xa & xb).sum())
        out.update({"both": both, "only_a": only_a, "only_b": only_b, "neither": int((~xa & ~xb).sum()), "mcnemar_p": mcnemar_exact(only_a, only_b)})
    else:
        out["episodes_a_never_b_always"] = int(((sa == 0) & (sb == 1)).sum())
        out["episodes_b_never_a_always"] = int(((sb == 0) & (sa == 1)).sum())
        out["hard_for_both"] = int(((sa < 0.2) & (sb < 0.2)).sum())
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", nargs="+", required=True, metavar="FILE", help="result file(s) of the first policy")
    ap.add_argument("--b", nargs="+", required=True, metavar="FILE", help="result file(s) of the second policy")
    args = ap.parse_args(argv)
    ra, rb = [load_result(p) for p in args.a], [load_result(p) for p in args.b]
    for w in check_same_episodes(ra + rb, args.a + args.b):
        print("WARNING:", w)
    c = compare(ra, rb)
    print(f"{c['episodes']} episodes. A: {100 * c['rate_a']:.1f} % ({c['files_a']} file(s)), B: {100 * c['rate_b']:.1f} % ({c['files_b']} file(s)).")
    print(f"paired difference A - B: {100 * c['paired_difference']:+.1f} points, SE over episodes {100 * c['paired_se']:.1f} "
          f"({c['paired_difference'] / c['paired_se']:+.1f} SE)" if c["paired_se"] > 0 else "paired difference: no spread")
    if "mcnemar_p" in c:
        print(f"both succeed {c['both']}, only A {c['only_a']}, only B {c['only_b']}, neither {c['neither']}; exact McNemar p = {c['mcnemar_p']:.4f}")
    else:
        print(f"episodes A never / B always: {c['episodes_a_never_b_always']}; B never / A always: {c['episodes_b_never_a_always']}; hard for both (< 20 %): {c['hard_for_both']}")
    if c["correlation_of_episode_scores"] is not None:
        print(f"correlation of the per-episode scores: {c['correlation_of_episode_scores']:+.2f} (high: the same states are hard for both)")


if __name__ == "__main__":
    main()
