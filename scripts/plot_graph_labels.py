"""Plot Figure S2 from its source table, without API calls."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"crystalsystem": "crystal system", "ph": "pH", "solvent_v": "solvent v"}
# Same colors as Figure 1 (scripts/data_prep/table_stat.py).
COLOR_MAIN = "#04C9AD"
COLOR_ALT = "#5C4A9C"


def label(parameter):
    return NAMES.get(parameter, parameter.replace("_", " ").replace(" conc", " conc."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/graph_figures")
    args = parser.parse_args()
    with (ROOT / "results/source_data/figure_s2.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    before = np.array([int(r["nodes_before"]) for r in rows])
    merged = np.array([int(r["nodes_removed_by_merging"]) for r in rows])
    x = np.arange(len(rows))
    width = 0.36
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig, ax = plt.subplots(figsize=(14.7, 6.4), layout="constrained")
    ax.bar(
        x - width / 2,
        before,
        width,
        color=COLOR_MAIN,
        label=f"Before (total: {before.sum()} nodes)",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        merged,
        width,
        color=COLOR_ALT,
        label=f"Merged (removed by simplify): {merged.sum()}",
        zorder=3,
    )
    ax.set_xticks(x, [label(r["parameter"]) for r in rows], rotation=35, ha="right")
    ax.set(
        ylabel="Count", title="Label distributions: Before vs Merged", xlim=(-0.8, len(rows) - 0.2)
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Stages", loc="upper right", frameon=True, edgecolor="0.3")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "figure_s2.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Figure S2 written to {args.out}")


if __name__ == "__main__":
    main()
