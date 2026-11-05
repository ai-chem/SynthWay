"""Reproduce manuscript Figures 5 and S6 from the three recorded repeats."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synthway/regenerate_runs/scripts"))
import plot_cross_judge_paper_figures as plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/regeneration_figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plots.UNCERTAINTY = "repeats"
    plots.configure_plot_style()
    runs = [ROOT / f"synthway/regenerate_runs/repeat_{i}" for i in range(1, 4)]
    attempts, routes = plots.load_runs(runs, [f"repeat_{i}" for i in range(1, 4)])
    trajectories = plots.build_trajectories(attempts, routes, 3)
    figure = plots.figure_main_ab(trajectories, routes, attempts, 3)
    plots.save(figure, args.out, "figure_5", 300)
    figure, table = plots.figure_si6_pair_metrics(routes)
    plots.save(figure, args.out, "figure_s6", 300)
    table.to_csv(args.out / "figure_s6.csv", index=False)
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
