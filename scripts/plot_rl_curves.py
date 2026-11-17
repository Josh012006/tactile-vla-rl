"""Success rate of RL runs against the collection round, several runs on one clear figure, from the evaluation results on disk (`<output-dir>/results/<round>.json`
written by `scripts.eval_rl`): no simulation, only JSON and matplotlib.

    python -m scripts.plot_rl_curves \
        --curve "no touch=runs/rl/peg_insert/bon20_teleop_s1600/eval_rl_*/results/*.json" \
        --curve "touch=runs/rl/peg_insert/bon20_touch_teleop_s1600/eval_rl_*/results/*.json" \
        --reference "VLA alone (SFT step 1600)=runs/sft/peg_insert/peg_insert_pi05_lora_teleop/eval/results/1600.json" \
        --out runs/rl/peg_insert/bon20_curves --title "Best-of-N, teleop demos, start: SFT step 1600"

`--curve LABEL=PATH_OR_GLOB` (repeatable; quote the glob so that the script expands it): the evaluation results of one run, from as many output folders as the evaluation was cut
into (several jobs, one `--output-dir` each). The round is the file's name (`7.json` = round 7). A round found in several folders keeps the evaluation with the most episodes (then the
newest). `--reference LABEL=FILE`: the policy the runs start from (the SFT checkpoint at the same number of episodes), a dashed line with its +-1 SE band.

Error bars are +-1 standard error of a proportion (sqrt(p (1 - p) / n), what the SFT curves show). With exactly two curves a second panel gives their PAIRED difference per round (second minus
first, same episodes: it is much tighter than the two bars suggest because the difficulty of a start state is shared), +-1 SE over episodes. The SE says how well the episodes pin the value
down, NOT how much two training runs differ by chance: one run per curve is one draw of the training.

Writes <out>/curve.png, curve.pdf, curve.json (every number of the figure) and table.md (what goes on a slide).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.compare_eval_paired import load_result, scores  # noqa: E402


def split_label(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"expected LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    if not label.strip() or not path.strip():
        raise ValueError(f"expected LABEL=PATH, got {spec!r}")
    return label.strip(), path.strip()


def collect(pattern: str) -> Dict[int, Tuple[Path, dict]]:
    """{round: (file, result)} of the files matching `pattern` (a file or a glob). Files that are not `<integer>.json` are ignored; of a round found twice the evaluation with the most episodes
    is kept, then the newest."""
    files = sorted(Path(p) for p in glob.glob(pattern))
    found: Dict[int, Tuple[Path, dict]] = {}
    for path in files:
        if not re.fullmatch(r"\d+", path.stem) or path.suffix != ".json":
            continue
        result = load_result(path)
        rnd = int(path.stem)
        if rnd not in found:
            found[rnd] = (path, result)
            continue
        old_path, old = found[rnd]
        key_new, key_old = (len(result["successes"]), path.stat().st_mtime), (len(old["successes"]), old_path.stat().st_mtime)
        if key_new > key_old:
            found[rnd] = (path, result)
    return found


def summarise(result: dict) -> dict:
    s = np.array([bool(x) for x in result["successes"]], dtype=float)
    n = len(s)
    p = float(s.mean())
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    z = 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"episodes": n, "successes": int(s.sum()), "rate": p, "se": se, "wilson_low": centre - half, "wilson_high": centre + half}


def paired(first: dict, second: dict) -> Optional[dict]:
    """Second minus first on the same episodes (None when the episode seeds differ: they are not paired)."""
    if [int(x) for x in first["seeds"]] != [int(x) for x in second["seeds"]]:
        return None
    d = scores([second]) - scores([first])
    n = len(d)
    return {"episodes": n, "difference": float(d.mean()), "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0}


def build(curves: Sequence[Tuple[str, str]], reference: Optional[Tuple[str, str]] = None) -> dict:
    out: dict = {"curves": [], "reference": None, "paired": None, "warnings": []}
    results: List[Dict[int, dict]] = []
    for label, pattern in curves:
        found = collect(pattern)
        if not found:
            raise ValueError(f"{label}: no <round>.json matches {pattern!r}")
        counts = sorted({len(r["successes"]) for _, r in found.values()})
        if len(counts) > 1:
            out["warnings"].append(f"{label}: the rounds were not evaluated on the same number of episodes ({counts})")
        results.append({rnd: res for rnd, (_, res) in found.items()})
        out["curves"].append({"label": label, "rounds": {str(rnd): summarise(res) for rnd, (_, res) in sorted(found.items())}})
    if reference:
        label, path = reference
        ref = load_result(path)
        out["reference"] = {"label": label, **summarise(ref)}
    if len(curves) == 2:
        common = sorted(set(results[0]) & set(results[1]))
        rows = {}
        for rnd in common:
            p = paired(results[0][rnd], results[1][rnd])
            if p is None:
                out["warnings"].append(f"round {rnd}: the two runs were evaluated on other episode seeds, no paired difference")
            else:
                rows[str(rnd)] = p
        out["paired"] = {"second_minus_first": f"{curves[1][0]} - {curves[0][0]}", "rounds": rows}
    return out


def table_markdown(data: dict) -> str:
    labels = [c["label"] for c in data["curves"]]
    rounds = sorted({int(r) for c in data["curves"] for r in c["rounds"]})
    head = "| round | " + " | ".join(labels) + (" | paired difference |" if data["paired"] else " |")
    lines = [head, "|" + "---|" * (len(labels) + 1 + (1 if data["paired"] else 0))]
    for rnd in rounds:
        cells = []
        for c in data["curves"]:
            r = c["rounds"].get(str(rnd))
            cells.append(f"{100 * r['rate']:.1f} +- {100 * r['se']:.1f}" if r else "")
        if data["paired"]:
            p = data["paired"]["rounds"].get(str(rnd))
            cells.append(f"{100 * p['difference']:+.1f} +- {100 * p['se']:.1f}" if p else "")
        lines.append(f"| {rnd} | " + " | ".join(cells) + " |")
    if data["reference"]:
        ref = data["reference"]
        lines.append("")
        lines.append(f"{ref['label']}: {100 * ref['rate']:.1f} +- {100 * ref['se']:.1f} % ({ref['episodes']} episodes)")
    return "\n".join(lines) + "\n"


PALETTE = ["#1f6fb5", "#d9622b", "#2a9d5c", "#7b4fa3"]


def plot(data: dict, out_dir: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    two = data["paired"] is not None and data["paired"]["rounds"]
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(8.0, 6.6 if two else 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]} if two else None)
    ax = axes[0] if two else axes
    values = []
    for i, c in enumerate(data["curves"]):
        rounds = sorted(int(r) for r in c["rounds"])
        y = np.array([100 * c["rounds"][str(r)]["rate"] for r in rounds])
        e = np.array([100 * c["rounds"][str(r)]["se"] for r in rounds])
        values += list(y - e) + list(y + e)
        nudge = 0.12 * (i - (len(data["curves"]) - 1) / 2)          # side by side, so that the bars of two runs do not hide each other
        ax.errorbar(np.array(rounds) + nudge, y, yerr=e, marker="o", ms=4.5, lw=2, capsize=3, color=PALETTE[i % len(PALETTE)], label=c["label"])
    ref = data["reference"]
    if ref:
        ry, re_ = 100 * ref["rate"], 100 * ref["se"]
        values += [ry - re_, ry + re_]
        ax.axhline(ry, color="#555555", ls="--", lw=1.4, label=f"{ref['label']}: {ry:.1f} %")
        ax.axhspan(ry - re_, ry + re_, color="#555555", alpha=0.12, lw=0)
    lo, hi = min(values), max(values)
    ax.set_ylim(max(0, 5 * math.floor((lo - 6) / 5)), min(100, 5 * math.ceil((hi + 6) / 5)))
    ax.set_ylabel("Success rate (%)  ± 1 SE")
    ax.set_title(title, fontsize=10)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    if two:
        bx = axes[1]
        rows = data["paired"]["rounds"]
        rounds = sorted(int(r) for r in rows)
        d = np.array([100 * rows[str(r)]["difference"] for r in rounds])
        e = np.array([100 * rows[str(r)]["se"] for r in rounds])
        bx.axhline(0, color="#555555", lw=1)
        bx.errorbar(rounds, d, yerr=e, marker="s", ms=4, lw=1.6, capsize=3, color="#444444")
        bx.set_ylabel(f"paired difference\n({data['paired']['second_minus_first']}), points", fontsize=8)
        bx.grid(True, ls="--", alpha=0.35)
        bx.set_xlabel("Collection round (50 episodes each; 0 = after the warm-up)")
    else:
        ax.set_xlabel("Collection round (50 episodes each; 0 = after the warm-up)")
    from matplotlib.ticker import MaxNLocator
    (axes[-1] if two else ax).xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "curve.png", dpi=200)
    fig.savefig(out_dir / "curve.pdf")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curve", action="append", required=True, metavar="LABEL=PATH_OR_GLOB", help="the evaluation results of one run (repeatable; quote the glob)")
    ap.add_argument("--reference", metavar="LABEL=FILE", help="the policy the runs start from: one result file, drawn as a dashed line with its band")
    ap.add_argument("--out", required=True, metavar="DIR", help="where curve.png / curve.pdf / curve.json / table.md go")
    ap.add_argument("--title", default="Success rate against collection round")
    args = ap.parse_args(argv)
    curves = [split_label(c) for c in args.curve]
    reference = split_label(args.reference) if args.reference else None
    data = build(curves, reference)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "curve.json").write_text(json.dumps(data, indent=1))
    (out / "table.md").write_text(table_markdown(data))
    plot(data, out, args.title)
    for w in data["warnings"]:
        print("WARNING:", w)
    print((out / "table.md").read_text())
    print(f"figure: {out / 'curve.png'} ({out / 'curve.pdf'}), numbers: {out / 'curve.json'}")


if __name__ == "__main__":
    main()
