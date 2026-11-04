#!/usr/bin/env python3
"""Paper figures for the cross-judge regeneration experiment.

Use scripts/plot_regeneration.py from the repository root to reproduce manuscript
Figures 5 and S6 with sample SD across the three repeats. This module also exposes
an optional system-cluster bootstrap analysis. These are different uncertainty
summaries and should not be interchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


COLORS = {
    "ink": "#2C234F",
    "purple": "#674095",
    "violet": "#9B4AA7",
    "lilac": "#D18BD2",
    "pale_lilac": "#EBC9EC",
    "magenta": "#E442A6",
    "pink": "#F1A9CE",
    "teal": "#10BFAE",
    "muted_teal": "#5F8585",
    "pale_teal": "#DDF2EF",
    "gray": "#AAA8B7",
    "light_gray": "#F1EFF4",
    "grid": "#E7E2EA",
    "white": "#FFFFFF",
}
MODEL_ORDER = ["claude_opus_4_5", "llama_4_maverick", "nemotron_3_nano"]
MODEL_LABELS = {
    "claude_opus_4_5": "Claude Opus 4.5",
    "llama_4_maverick": "Llama 4 Maverick",
    "nemotron_3_nano": "Nemotron 3 Nano",
}
METRIC_ORDER = ["chemical_geval", "answer_relevancy", "faithfulness"]
METRIC_PLOT_LABELS = {
    "chemical_geval": "Chemical plausibility\nand consistency",
    "answer_relevancy": "Answer relevancy",
    "faithfulness": "New Faithfulness",
}
# spelled-out form used on the bar panel, matching manuscript Fig. 5a
STEP_LABELS_LONG = [
    "Initial generation",
    "After regeneration 1",
    "After regeneration 2",
    "After regeneration 3",
]
ROUTE_KEY = ["arm_key", "case_id", "route_name"]
THRESHOLD = 0.8
N_BOOT = 20000
SEED = 20260904

# "repeats": error bars are the SD across the three independent repeats, i.e.
#   pipeline reproducibility on the fixed route set.  This is what the claim
#   "the regeneration module works, and works consistently" needs.
# "cluster": error bars are 95% percentile cluster-bootstrap CIs over the source
#   systems, i.e. how precisely the level is estimated for new materials.  Use
#   this only if the text generalises a pass-rate number beyond this route set.
UNCERTAINTY = "repeats"

# Nature-family journals carry the title in the caption, not inside the figure.
# Set to False (--no-titles) to strip every on-figure title for that house style.
SHOW_TITLES = True


# ------------------------------------------------------------------ plumbing


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.9,
            "axes.titleweight": "normal",
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "figure.facecolor": COLORS["white"],
            "axes.facecolor": COLORS["white"],
            "savefig.facecolor": COLORS["white"],
            "savefig.bbox": "tight",
            "ps.fonttype": 42,
        }
    )


def panel_title(axis: plt.Axes, text: str, **kwargs) -> None:
    """Set an axes title, unless on-figure titles are switched off."""
    if SHOW_TITLES:
        axis.set_title(text, **kwargs)


def figure_title(fig: plt.Figure, text: str, **kwargs) -> None:
    """Set a figure-level title, unless on-figure titles are switched off."""
    if SHOW_TITLES:
        fig.suptitle(text, **kwargs)


def add_grid(axis: plt.Axes, axis_name: str = "y") -> None:
    axis.grid(axis=axis_name, color=COLORS["grid"], linewidth=0.7, alpha=0.9)
    axis.set_axisbelow(True)


def parse_bool(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    values = series.astype(str).str.strip().str.lower().map(mapping)
    if values.isna().any():
        raise ValueError(f"Cannot parse booleans: {sorted(series[values.isna()].unique())}")
    return values.astype(bool)


def load_runs(run_dirs: Sequence[Path], labels: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts, routes = [], []
    for run_dir, label in zip(run_dirs, labels):
        run_dir = run_dir.expanduser().resolve()
        a = pd.read_csv(run_dir / "tables" / "attempt_trajectories_long.csv")
        r = pd.read_csv(run_dir / "tables" / "route_effects.csv")
        a["run"], r["run"] = label, label
        a["attempt_no"] = a["attempt_no"].astype(int)
        a["score"] = pd.to_numeric(a["score"], errors="coerce")
        a["passed_all"] = parse_bool(a["passed_all"])
        r["initial_pass_all"] = parse_bool(r["initial_pass_all"])
        r["final_pass_all"] = parse_bool(r["final_pass_all"])
        attempts.append(a)
        routes.append(r)
    return pd.concat(attempts, ignore_index=True), pd.concat(routes, ignore_index=True)


# --------------------------------------------------------------- inference


def cluster_boot(values: np.ndarray, clusters: np.ndarray, seed: int = SEED) -> dict[str, float]:
    """Percentile cluster bootstrap of a mean; whole clusters are resampled."""
    ok = ~np.isnan(values)
    values, clusters = values[ok], clusters[ok]
    if values.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "n": 0, "k": 0}
    groups = [values[clusters == c] for c in pd.unique(clusters)]
    n_clusters = len(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n_clusters, n_clusters)
        draws[b] = np.concatenate([groups[i] for i in idx]).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(draws.std(ddof=1)),
        "n": int(values.size),
        "k": n_clusters,
    }


def asym_err(stats: Sequence[dict], scale: float = 1.0) -> np.ndarray:
    """Convert percentile intervals into matplotlib's (2, n) yerr layout."""
    means = np.array([s["mean"] for s in stats], dtype=float) * scale
    lo = np.array([s["lo"] for s in stats], dtype=float) * scale
    hi = np.array([s["hi"] for s in stats], dtype=float) * scale
    return np.vstack([np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)])


def route_average(frame: pd.DataFrame, value: str, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Average the repeats within a route: the repeats are measurement noise."""
    keys = ROUTE_KEY + list(extra)
    return frame.groupby(keys, as_index=False).agg(**{value: (value, "mean")})


def repeat_sd(frame: pd.DataFrame, value: str) -> dict[str, float]:
    """Mean and sample SD across the independent repeats.

    This is the reproducibility of the pipeline on a fixed route set: how much
    the number moves when the whole experiment is run again on the same routes.
    It is deliberately *not* a confidence interval for the pass rate, because it
    holds the route sample fixed.
    """
    per_run = frame.groupby("run")[value].mean()
    mean = float(per_run.mean())
    sd = float(per_run.std(ddof=1)) if len(per_run) > 1 else 0.0
    return {
        "mean": mean,
        "lo": mean - sd,
        "hi": mean + sd,
        "se": sd,
        "n": int(len(frame)),
        "k": int(len(per_run)),
    }


def summarise(frame: pd.DataFrame, value: str) -> dict[str, float]:
    """Uncertainty for one quantity, in whichever mode the run selected."""
    if UNCERTAINTY == "repeats":
        return repeat_sd(frame, value)
    rl = route_average(frame, value)
    return cluster_boot(rl[value].to_numpy(float), rl["case_id"].to_numpy())


# ------------------------------------------------------------------ derived


def build_trajectories(attempts: pd.DataFrame, routes: pd.DataFrame, max_regens: int) -> pd.DataFrame:
    status = (
        attempts.groupby(["run"] + ROUTE_KEY + ["attempt_no"], as_index=False)
        .agg(passed=("passed_all", "first"))
    )
    hits = {
        key: set(g.loc[g["passed"], "attempt_no"].astype(int))
        for key, g in status.groupby(["run"] + ROUTE_KEY, sort=False)
    }
    meta = routes[["run"] + ROUTE_KEY + ["generator_key", "judge_key"]]
    rows = []
    for row in meta.itertuples(index=False):
        got = hits.get((row.run, row.arm_key, row.case_id, row.route_name), set())
        for step in range(max_regens + 1):
            rows.append(
                {
                    "run": row.run,
                    "arm_key": row.arm_key,
                    "case_id": row.case_id,
                    "route_name": row.route_name,
                    "generator_key": row.generator_key,
                    "judge_key": row.judge_key,
                    "step": step,
                    "passed": float(any(n <= step + 1 for n in got)),
                }
            )
    return pd.DataFrame(rows)


def paired_delta(routes: pd.DataFrame, metric: str) -> dict[str, float]:
    """Paired best-minus-initial change, the quantity the Delta label reports.

    The comparison is within route, so route difficulty cancels.  A confidence
    interval built from the two level estimates would be much wider and would
    invite the "overlapping error bars mean no difference" misreading.
    """
    rl = routes.rename(columns={f"initial_{metric}": "i", f"final_{metric}": "f"}).copy()
    rl["d"] = rl["f"] - rl["i"]
    return summarise(rl, "d")


def unselected_delta(attempts: pd.DataFrame, metric: str) -> dict[str, float]:
    """Initial-to-attempt-2 delta among routes whose initial attempt failed."""
    wide = attempts.pivot_table(
        index=["run"] + ROUTE_KEY + ["metric"], columns="attempt_no", values="score"
    )
    status = (
        attempts.groupby(["run"] + ROUTE_KEY + ["attempt_no"], as_index=False)
        .agg(passed=("passed_all", "first"))
    )
    failed_first = set(
        status[(status.attempt_no == 1) & (~status.passed)].set_index(["run"] + ROUTE_KEY).index
    )
    sub = wide.xs(metric, level="metric").reset_index()
    keep = [
        (r.run, r.arm_key, r.case_id, r.route_name) in failed_first
        for r in sub.itertuples(index=False)
    ]
    sub = sub[keep].dropna(subset=[1, 2]).copy()
    sub["delta"] = sub[2] - sub[1]
    return summarise(sub, "delta")


# ------------------------------------------------------------- main figures


def draw_pass_panel(axis: plt.Axes, traj: pd.DataFrame, max_regens: int) -> list[dict]:
    """Cumulative pass rate per regeneration step, drawn onto a given axes."""
    stats, records = [], []
    for step in range(max_regens + 1):
        st = summarise(traj[traj.step == step], "passed")
        stats.append(st)
        records.append({"step": step, **st})

    means = np.array([st["mean"] for st in stats]) * 100
    steps = np.arange(max_regens + 1)
    bar_colors = [COLORS["pale_lilac"], COLORS["lilac"], COLORS["violet"], COLORS["purple"]][
        : len(steps)
    ]
    bars = axis.bar(
        steps,
        means,
        width=0.62,
        color=bar_colors,
        edgecolor=COLORS["white"],
        linewidth=1.0,
        yerr=asym_err(stats, scale=100),
        error_kw={"ecolor": COLORS["ink"], "elinewidth": 1.2, "capsize": 4, "capthick": 1.2},
        zorder=3,
    )
    axis.axhline(100, color=COLORS["teal"], linestyle="--", linewidth=2.0, zorder=2)
    axis.bar_label(
        bars,
        labels=[f"{v:.1f}%" for v in means],
        padding=14,
        fontsize=11.5,
        fontweight="semibold",
        color=COLORS["ink"],
    )
    axis.set_xticks(steps, STEP_LABELS_LONG[: max_regens + 1])
    axis.tick_params(axis="x", labelsize=11, rotation=22)
    for tick in axis.get_xticklabels():
        tick.set_horizontalalignment("right")
    axis.tick_params(axis="y", labelsize=11)
    axis.set_xlabel("Regeneration step", labelpad=10, fontsize=12)
    axis.set_ylabel("Routes passing all thresholds (%)", fontsize=12)
    axis.set_ylim(0, 108)
    axis.set_yticks(np.arange(0, 101, 20))
    add_grid(axis)
    panel_title(
        axis,
        "Number of routes passing all thresholds after each regeneration step",
        fontsize=13,
        pad=12,
    )
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=COLORS["teal"],
                linestyle="--",
                linewidth=2.0,
                label="All routes: 100% (60 routes)",
            )
        ],
        loc="upper left",
        frameon=True,
        facecolor=COLORS["white"],
        edgecolor=COLORS["grid"],
        framealpha=0.95,
        fontsize=11,
    )
    return records


def draw_metric_panel(
    axis: plt.Axes, routes: pd.DataFrame, attempts: pd.DataFrame
) -> list[dict]:
    """Initial vs selected best attempt per metric, drawn onto a given axes."""
    x = np.arange(len(METRIC_ORDER), dtype=float)
    width = 0.32
    offsets = {"initial": -width / 2, "best": width / 2}
    colors = {"initial": COLORS["lilac"], "best": COLORS["teal"]}

    records, stats_by = [], {"initial": [], "best": []}
    for metric in METRIC_ORDER:
        for stage, column in (("initial", f"initial_{metric}"), ("best", f"final_{metric}")):
            rl = routes.rename(columns={column: "v"})[ROUTE_KEY + ["run", "v"]]
            st = summarise(rl, "v")
            stats_by[stage].append(st)
            records.append({"metric": metric, "stage": stage, **st})

    for stage in ("initial", "best"):
        stats = stats_by[stage]
        means = np.array([st["mean"] for st in stats])
        axis.bar(
            x + offsets[stage],
            means,
            width=width,
            color=colors[stage],
            edgecolor=COLORS["white"],
            linewidth=0.7,
            zorder=2,
        )
        axis.errorbar(
            x + offsets[stage],
            means,
            yerr=asym_err(stats),
            fmt="none",
            ecolor=COLORS["ink"],
            elinewidth=0.9,
            capsize=3,
            zorder=4,
        )

    for i, metric in enumerate(METRIC_ORDER):
        sel = paired_delta(routes, metric)
        uns = unselected_delta(attempts, metric)
        top = max(stats_by["initial"][i]["hi"], stats_by["best"][i]["hi"])
        axis.text(
            x[i],
            min(top + 0.03, 1.045),
            (
                f"Δ {sel['mean']:+.3f} ± {sel['se']:.3f}"
                if UNCERTAINTY == "repeats"
                else f"Δ {sel['mean']:+.3f}  [{sel['lo']:+.3f}, {sel['hi']:+.3f}]"
            ),
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
            color=COLORS["ink"],
        )
        records.append({"metric": metric, "stage": "delta_paired", **sel})
        records.append({"metric": metric, "stage": "delta_unselected", **uns})

    axis.axhline(THRESHOLD, color=COLORS["muted_teal"], linestyle="--", linewidth=0.9, alpha=0.7)
    axis.set_ylim(0, 1.13)
    axis.set_xticks(x, [METRIC_PLOT_LABELS[m] for m in METRIC_ORDER], fontsize=11)
    axis.tick_params(axis="y", labelsize=11)
    axis.set_xlabel("Metric", labelpad=10, fontsize=12)
    axis.set_ylabel("Mean score", fontsize=12)
    add_grid(axis)
    panel_title(
        axis,
        "Mean metric scores: initial vs selected best attempt",
        fontsize=13,
        pad=12,
    )
    return records


def metric_panel_handles() -> list[object]:
    return [
        Patch(facecolor=COLORS["lilac"], label="Initial attempt"),
        Patch(facecolor=COLORS["teal"], label="Selected best attempt"),
        Line2D([0], [0], color=COLORS["muted_teal"], linestyle="--", linewidth=1.2, label="Threshold = 0.8"),
    ]


def figure_main_ab(
    traj: pd.DataFrame, routes: pd.DataFrame, attempts: pd.DataFrame, max_regens: int
) -> plt.Figure:
    """Panels a and b side by side, in the two-panel layout of manuscript Fig. 5."""
    fig, axes = plt.subplots(1, 2, figsize=(15.6, 6.6))
    draw_pass_panel(axes[0], traj, max_regens)
    draw_metric_panel(axes[1], routes, attempts)

    # the metric panel needs its own key; keep it inside the axes so the two
    # panels stay self-contained, the way the manuscript figure has them
    axes[1].set_ylim(0, 1.28)
    axes[1].legend(
        handles=metric_panel_handles(),
        loc="upper left",
        frameon=True,
        facecolor=COLORS["white"],
        edgecolor=COLORS["grid"],
        framealpha=0.95,
        fontsize=11,
    )
    for axis, label in zip(axes, ("a)", "b)")):
        axis.text(
            -0.11,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
            ha="left",
            color=COLORS["ink"],
        )
    fig.tight_layout(rect=(0.025, 0.045, 0.985, 0.955), w_pad=4.0)
    # one frame per panel: it separates a from b and boxes the pair,
    # matching how the manuscript figure is laid out
    for x0, x1 in ((0.008, 0.497), (0.503, 0.992)):
        fig.patches.append(
            Rectangle(
                (x0, 0.012),
                x1 - x0,
                0.976,
                transform=fig.transFigure,
                facecolor="none",
                edgecolor=COLORS["purple"],
                linewidth=2.0,
                zorder=0,
                clip_on=False,
            )
        )
    return fig


# -------------------------------------------------------------- SI figures


def figure_si6_pair_metrics(routes: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Initial vs selected best attempt, resolved per generator-judge pair.

    The per-pair version of the main metric panel.  Each panel rests on only 10
    routes, so panel-level differences are far less certain than the pooled
    figure; the caption says so rather than leaving the reader to infer it.
    """
    # one row per generator, one column per judge it was paired with: the grid
    # then mirrors the design instead of running the pairs out in a flat list
    generators = [g for g in MODEL_ORDER if g in set(routes.generator_key)]
    fig, axes = plt.subplots(
        len(generators), 2, figsize=(11.6, 11.0), sharey=True, sharex=True
    )
    x = np.arange(len(METRIC_ORDER), dtype=float)
    width = 0.32
    offsets = {"initial": -width / 2, "best": width / 2}
    colors = {"initial": COLORS["lilac"], "best": COLORS["teal"]}
    records: list[dict] = []

    for row, generator in enumerate(generators):
        judges = [
            j
            for j in MODEL_ORDER
            if j != generator
            and not routes[
                (routes.generator_key == generator) & (routes.judge_key == j)
            ].empty
        ]
        for col, axis in enumerate(np.atleast_1d(axes[row])):
            if col >= len(judges):
                axis.axis("off")
                continue
            judge = judges[col]
            subset = routes[
                (routes.generator_key == generator) & (routes.judge_key == judge)
            ]
            stats_by = {"initial": [], "best": []}
            for metric in METRIC_ORDER:
                for stage, column in (
                    ("initial", f"initial_{metric}"),
                    ("best", f"final_{metric}"),
                ):
                    frame = subset.rename(columns={column: "v"})[ROUTE_KEY + ["run", "v"]]
                    st = summarise(frame, "v")
                    stats_by[stage].append(st)
                    records.append(
                        {
                            "generator": generator,
                            "judge": judge,
                            "metric": metric,
                            "stage": stage,
                            **st,
                        }
                    )
            for stage in ("initial", "best"):
                stats = stats_by[stage]
                means = np.array([st["mean"] for st in stats])
                axis.bar(
                    x + offsets[stage],
                    means,
                    width=width,
                    color=colors[stage],
                    edgecolor=COLORS["white"],
                    linewidth=0.7,
                    zorder=2,
                )
                axis.errorbar(
                    x + offsets[stage],
                    means,
                    yerr=asym_err(stats),
                    fmt="none",
                    ecolor=COLORS["ink"],
                    elinewidth=0.9,
                    capsize=3,
                    zorder=4,
                )
            for i, metric in enumerate(METRIC_ORDER):
                frame = subset.rename(
                    columns={f"initial_{metric}": "i", f"final_{metric}": "f"}
                ).copy()
                frame["d"] = frame["f"] - frame["i"]
                delta = summarise(frame, "d")
                top = max(stats_by["initial"][i]["hi"], stats_by["best"][i]["hi"])
                axis.text(
                    x[i],
                    min(top + 0.03, 1.045),
                    f"Δ {delta['mean']:+.3f} ± {delta['se']:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    color=COLORS["ink"],
                )
                records.append(
                    {
                        "generator": generator,
                        "judge": judge,
                        "metric": metric,
                        "stage": "delta",
                        **delta,
                    }
                )
            axis.axhline(
                THRESHOLD, color=COLORS["muted_teal"], linestyle="--", linewidth=0.9, alpha=0.7
            )
            axis.set_ylim(0, 1.15)
            axis.set_xticks(x, [METRIC_PLOT_LABELS[m] for m in METRIC_ORDER], fontsize=9)
            axis.tick_params(axis="y", labelsize=10)
            panel_title(axis, f"Judged by {MODEL_LABELS[judge]}", fontsize=11)
            if col == 0:
                axis.set_ylabel(
                    f"Generated by\n{MODEL_LABELS[generator]}",
                    fontsize=11,
                    fontweight="semibold",
                    labelpad=12,
                )
            add_grid(axis)

    figure_title(
        fig,
        "Mean metric scores at the initial and selected best attempt for each generator-judge combination",
        fontsize=13,
        y=0.955,
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["lilac"], label="Initial attempt"),
            Patch(facecolor=COLORS["teal"], label="Selected best attempt"),
            Line2D([0], [0], color=COLORS["muted_teal"], linestyle="--", linewidth=1.2, label="Threshold = 0.8"),
            ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.935), h_pad=2.4, w_pad=1.4)
    return fig, pd.DataFrame(records)


# ------------------------------------------------------------------- driver


def save(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> str:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)


