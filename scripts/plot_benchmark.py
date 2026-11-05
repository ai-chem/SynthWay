"""Replot Figure 3: evaluation-only resources and pairwise wins among 12 models."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
MODEL_LABELS = {
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2",
    "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-3-flash-preview": "Gemini 3 Flash Preview",
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "nvidia/nemotron-3-nano-30b-a3b:free": "NVIDIA Nemotron 3 Nano 30B A3B",
    "openai/gpt-5.2": "GPT-5.2",
    "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B-Instruct-2507",
    "tngtech/deepseek-r1t2-chimera": "Deepseek-R1T2-Chimera",
    "xiaomi/mimo-v2-flash": "Xiaomi MiMo-V2-Flash",
}
THRESHOLD = 0.8
JUDGE_STAGES = ("chem", "ans", "faith")
FIELDS = (
    "model",
    "n_cases",
    "n_routes",
    "chem_score_route",
    "ans_score_route",
    "faith_score_route",
    "mean_score",
    "total_cost_usd_case",
    "total_time_s_case",
    "all_three_pass_rate",
    "pairwise_all_three_win_rate",
    "resource_scope",
    "faithfulness_metric",
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require_selected_models(models):
    if set(models) != set(MODEL_LABELS):
        raise ValueError("Figure 3 requires exactly the selected 12 benchmark models")


def coordinates_from_records(root=ROOT):
    """Use the same 90 route slots for scores, 11 opponents, and judge resources.

    Resource totals are averaged over 30 cases, each containing three routes.
    Generation observations are deliberately excluded. Original Faithfulness
    scores come from the frozen benchmark, not the current live SS evaluator.
    """
    scores = defaultdict(dict)
    for row in read_rows(root / "results/source_data/benchmark_selected_routes.csv"):
        slot = row["route_key"].split("/", 1)[1]
        model = row["model"]
        if slot in scores[model]:
            raise ValueError(f"Repeated benchmark slot: {model}, {slot}")
        triple = tuple(float(row[k]) for k in ("cpc", "answer", "faithfulness"))
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in triple):
            raise ValueError(f"Invalid benchmark score: {model}, {slot}")
        scores[model][slot] = triple
    require_selected_models(scores)
    slots = set(scores[next(iter(MODEL_LABELS))])
    if any(set(values) != slots for values in scores.values()):
        raise ValueError("Benchmark route slots do not align")
    cases = defaultdict(set)
    for slot in slots:
        case, route = slot.rsplit("/", 1)
        cases[case].add(route)
    if len(cases) != 30 or any(len(routes) != 3 for routes in cases.values()):
        raise ValueError("Figure 3 requires 30 cases with three routes each")

    resources = defaultdict(dict)
    for row in read_rows(root / "results/benchmark_resource_observations.csv"):
        model, stage = row["model"], row["stage"]
        if model not in MODEL_LABELS or stage not in JUDGE_STAGES:
            continue
        slot = f"{row['case_id']}/{row['route']}"
        group = resources[model, stage]
        if slot in group:
            raise ValueError(f"Repeated judge observation: {model}, {stage}, {slot}")
        values = tuple(float(row[field]) for field in ("cost_usd", "time_s"))
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"Invalid judge resources: {model}, {stage}, {slot}")
        group[slot] = values

    rows = []
    for model in MODEL_LABELS:
        values = scores[model]
        for stage in JUDGE_STAGES:
            if set(resources[model, stage]) != slots:
                raise ValueError(f"Incomplete or unexpected judge observations: {model}, {stage}")
        opponents = [scores[other] for other in MODEL_LABELS if other != model]
        wins = mean(
            (1 + int(min(ours) >= THRESHOLD) - int(min(other[slot]) >= THRESHOLD)) / 2
            for slot, ours in values.items()
            for other in opponents
        )
        row = {
            "model": model,
            "n_cases": len(cases),
            "n_routes": len(values),
            **{
                field: mean(triple[i] for triple in values.values())
                for i, field in enumerate(
                    ("chem_score_route", "ans_score_route", "faith_score_route")
                )
            },
            "mean_score": mean(mean(triple) for triple in values.values()),
            "all_three_pass_rate": mean(min(triple) >= THRESHOLD for triple in values.values()),
            "pairwise_all_three_win_rate": wins,
            "resource_scope": "evaluation_only",
            "faithfulness_metric": "original",
        }
        for i, field in enumerate(("cost_usd", "time_s")):
            # Sum all three judges for all three routes, then average cases.
            # Use unrounded observations, not the rounded display values in S7.
            row[f"total_{field}_case"] = mean(
                math.fsum(
                    resources[model, stage][f"{case}/{route}"][i]
                    for stage in JUDGE_STAGES
                    for route in sorted(routes)
                )
                for case, routes in sorted(cases.items())
            )
        rows.append(row)
    return rows


def load_coordinates(root=ROOT):
    rows = read_rows(root / "results/source_data/figure_3.csv")
    require_selected_models(row["model"] for row in rows)
    if len(rows) != len(MODEL_LABELS):
        raise ValueError("Repeated Figure 3 model")
    for row in rows:
        if (
            row.get("resource_scope") != "evaluation_only"
            or row.get("faithfulness_metric") != "original"
        ):
            raise ValueError(
                "Figure 3 must use evaluation-only resources and original Faithfulness"
            )
        for field in FIELDS:
            if field not in {"model", "resource_scope", "faithfulness_metric"}:
                row[field] = (
                    int(row[field]) if field in {"n_cases", "n_routes"} else float(row[field])
                )
    by_model = {row["model"]: row for row in rows}
    return [by_model[model] for model in MODEL_LABELS]


def write_coordinates(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_coordinates(rows, out):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12.5,
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )
    fig = plt.figure(figsize=(16, 5.3))
    axes = [fig.add_axes([left, 0.205, 0.300, 0.75]) for left in (0.045, 0.390)]
    cax = fig.add_axes([0.735, 0.205, 0.020, 0.75])
    legend_ax = fig.add_axes([0.802, 0.08, 0.193, 0.89])
    serif = ["Times New Roman", "Times", "DejaVu Serif"]
    cmap = LinearSegmentedColormap.from_list("quality", ["#8E84BA", "#4C3F86", "#24C7B8"])
    colors = [row["pairwise_all_three_win_rate"] for row in rows]
    norm = Normalize(min(colors), max(colors))
    y = [row["mean_score"] for row in rows]
    free_x = min(row["total_cost_usd_case"] for row in rows if row["total_cost_usd_case"] > 0) / 10
    offsets = {
        "cost": [
            (8, 11),
            (-18, -16),
            (6, 10),
            (8, 12),
            (8, -15),
            (5, -17),
            (6, -17),
            (-14, 11),
            (-16, 12),
            (7, 10),
            (7, 11),
            (7, -17),
        ],
        "time": [
            (8, -8),
            (-18, -17),
            (6, 10),
            (9, 8),
            (-14, -18),
            (8, -17),
            (8, 12),
            (-14, 8),
            (8, 12),
            (7, 12),
            (7, 12),
            (7, -18),
        ],
    }
    for ax, measure, label, letter in zip(
        axes, ("cost", "time"), ("Cost per case ($), log scale", "Time per case (s)"), ("a)", "b)")
    ):
        field = "total_cost_usd_case" if measure == "cost" else "total_time_s_case"
        x = [free_x if measure == "cost" and row[field] == 0 else row[field] for row in rows]
        points = ax.scatter(
            x,
            y,
            c=colors,
            cmap=cmap,
            norm=norm,
            s=95,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.96,
            zorder=3,
        )
        frontier = [
            i
            for i in range(len(rows))
            if not any(
                rows[j][field] <= rows[i][field]
                and y[j] >= y[i]
                and (rows[j][field] < rows[i][field] or y[j] > y[i])
                for j in range(len(rows))
            )
        ]
        frontier.sort(key=lambda i: x[i])
        ax.plot([x[i] for i in frontier], [y[i] for i in frontier], color="#F06292", lw=1.4)
        ax.scatter(
            [x[i] for i in frontier],
            [y[i] for i in frontier],
            facecolors="none",
            edgecolors="#F06292",
            s=175,
            linewidths=1.2,
            zorder=4,
        )
        for i, (xx, yy) in enumerate(zip(x, y)):
            ax.annotate(
                str(i + 1),
                (xx, yy),
                xytext=offsets[measure][i],
                textcoords="offset points",
                fontsize=11,
                color="#2E255C",
                bbox={"boxstyle": "round,pad=.14", "fc": "white", "ec": "#D7D9E8"},
            )
        if measure == "cost":
            ax.set_xscale("log")
            ax.set_xticks([0.001, 0.01, 0.1], ["$0.0010", "$0.01", "$0.10"])
            ax.set_xlim(free_x * 0.85, max(x) * 1.45)
            ax.tick_params(axis="x", which="minor", bottom=False)
        else:
            ax.set_xticks([150, 300, 450, 600], ["150s", "300s", "450s", "10.0m"])
            padding = 0.05 * (max(x) - min(x))
            ax.set_xlim(min(x) - 1.2 * padding, max(x) + 2.5 * padding)
        ax.set(xlabel=label, ylabel="Mean evaluation score", ylim=(0.5, 1.02))
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(axis="y", color="#D8DCEB", alpha=0.8)
        ax.spines[["right", "top"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#D7D9E8")
        ax.tick_params(colors="#444444", labelsize=11.5, width=0.6, length=3)
        for tick in ax.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")
        fig.text(
            ax.get_position().x0 + ax.get_position().width / 2,
            0.015,
            letter,
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            fontfamily=serif,
        )
    cb = fig.colorbar(points, cax=cax)
    cb.set_label("Win-rate")
    cb.set_ticks([0.40, 0.45, 0.50, 0.55, 0.60, 0.65])
    cb.ax.tick_params(labelsize=10.5, colors="#555555", length=0)
    cb.ax.grid(axis="y", color="white", alpha=0.6, linewidth=0.8)
    cb.outline.set_edgecolor("#D7D9E8")
    legend_ax.axis("off")
    legend_ax.plot(
        0.05,
        0.965,
        marker="o",
        markersize=16,
        markerfacecolor="none",
        markeredgecolor="#F06292",
        markeredgewidth=2,
        transform=legend_ax.transAxes,
    )
    legend_ax.text(
        0.13,
        0.965,
        "- Pareto frontier",
        va="center",
        fontsize=14,
        fontfamily=serif,
        transform=legend_ax.transAxes,
    )
    legend_ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.055),
            0.97,
            0.84,
            boxstyle="round,pad=.016,rounding_size=.065",
            facecolor="white",
            edgecolor="#24C7B8",
            linewidth=1.1,
            clip_on=False,
            transform=legend_ax.transAxes,
        )
    )
    legend_ax.text(
        0.12,
        0.887,
        "Models:",
        va="top",
        fontfamily=serif,
        fontsize=14.5,
        fontweight="bold",
        transform=legend_ax.transAxes,
    )
    labels = []
    for i, row in enumerate(rows, 1):
        name = MODEL_LABELS[row["model"]]
        if i == 8:
            name = "NVIDIA Nemotron 3\n   Nano 30B A3B"
        elif i == 10:
            name = "Qwen3-30B-A3B-Instruct\n   -2507"
        labels.append(f"{i}.{' ' if i < 10 else ''}{name}")
    legend_ax.text(
        0.04,
        0.827,
        "\n".join(labels),
        va="top",
        fontfamily=serif,
        fontsize=13.5,
        linespacing=1.65,
        transform=legend_ax.transAxes,
    )
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figure_3.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional export directory; default: update results/figures in place",
    )
    parser.add_argument(
        "--from-records",
        action="store_true",
        help="Recalculate from the selected route and judge-resource observations",
    )
    parser.add_argument(
        "--data-out",
        type=Path,
        help="CSV destination; default: results/source_data/figure_3.csv, or figure_3.csv in a custom --out",
    )
    args = parser.parse_args()
    out = args.out or ROOT / "results/figures"
    data_out = args.data_out or (
        args.out / "figure_3.csv" if args.out else ROOT / "results/source_data/figure_3.csv"
    )
    rows = coordinates_from_records(ROOT) if args.from_records else load_coordinates(ROOT)
    write_coordinates(rows, data_out)
    plot_coordinates(rows, out)
    print(f"Figure 3 (evaluation only, 12 models) saved in {out}")


if __name__ == "__main__":
    main()
