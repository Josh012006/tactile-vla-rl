"""The success-rate curve of an evaluation: curve.json, curve.png and curve.pdf rebuilt from the COMPLETE checkpoints of a results folder
(`results/<label>.json`, written by the sweep when every episode of the checkpoint is done). matplotlib is imported when the plot is drawn.

Same plot as expo-ft's scripts/eval_curve.py (success rate in %, +- 1 standard error, one point per checkpoint, the value written above each point),
except for the TITLE: it is set small (TITLE_FONT_SIZE) and wrapped to the width of the figure, so that a long title stays inside the picture and the PDF.
Only the title is small; the axes, ticks and annotations keep expo-ft's sizes.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import List, Optional

from src.eval.episodes import atomic_write_json

TITLE_FONT_SIZE = 9          # points: the title only
TITLE_WRAP = 90              # characters per line of the title at this size on the 8 inch wide figure
FIGSIZE = (8, 5)


def load_entries(results_dir: Path) -> List[dict]:
    """One entry per complete checkpoint of `results_dir`, by step."""
    entries = []
    for path in Path(results_dir).glob("*.json"):
        data = json.loads(path.read_text())
        entries.append({"label": data["label"], "step": data["step"], "success_rate": data["success_rate"], "success_se": data["success_se"],
                        "wilson95": data["wilson95"], "n_episodes": data["n_episodes"], "single_attempt": data["single_attempt"],
                        "eval_sha": data["eval_sha"]})
    entries.sort(key=lambda e: e["step"])
    return entries


def wrapped_title(title: str) -> str:
    return "\n".join(textwrap.fill(line, TITLE_WRAP) for line in title.splitlines())


def draw_curve(entries: List[dict], title: Optional[str] = None, name: str = "evaluation", x_label: Optional[str] = None):
    """The figure of `entries` (matplotlib, Agg backend). The caller saves and closes it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 14, "axes.titlesize": TITLE_FONT_SIZE, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12}):
        steps = [e["step"] for e in entries]
        rates = [100 * e["success_rate"] for e in entries]
        errs = [100 * e["success_se"] for e in entries]
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.errorbar(steps, rates, yerr=errs, marker="o", markersize=7, linewidth=3, capsize=5, elinewidth=1.5, ecolor="black", alpha=0.9, color="#1f77b4")
        ax.set_xlabel(x_label or "SFT training iteration (0 = initial state of the run)", labelpad=10)
        ax.set_ylabel("Success rate (%)  \u00b1 1 SE", labelpad=10)
        ax.set_title(wrapped_title(title or f"Eval success rate vs. checkpoint \u2014 {name}"), pad=12, weight="bold", fontsize=TITLE_FONT_SIZE)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.4, linestyle="--")
        for s, r in zip(steps, rates):
            ax.annotate(f"{r:.0f}%", (s, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, weight="bold")
        fig.tight_layout()
    return fig


def write_curve(results_dir: Path, curve_dir: Path, title: Optional[str] = None, name: str = "evaluation", x_label: Optional[str] = None) -> List[dict]:
    """Rebuild curve.json / curve.png / curve.pdf in `curve_dir` from the complete checkpoints of `results_dir`. Returns the entries. Nothing is drawn when there is none."""
    entries = load_entries(results_dir)
    curve_dir = Path(curve_dir)
    atomic_write_json(curve_dir / "curve.json", entries)
    if not entries:
        return entries
    import matplotlib.pyplot as plt
    fig = draw_curve(entries, title, name, x_label)
    fig.savefig(curve_dir / "curve.png", dpi=200)
    fig.savefig(curve_dir / "curve.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    return entries
