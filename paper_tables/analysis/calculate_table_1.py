"""Recalculate SynthWay Table 1 from the judge ratings and the existing expert workbook.

Requires numpy, pandas, scipy, scikit-learn and openpyxl. The default inputs are in ../data, or you can supply paths with --judges and --experts.
From the repository root, the existing expert workbook is found automatically.
Run: python calculate_table_1.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


MODELS = {
    "anthropic_claude-opus-4.5": "Claude Opus 4.5",
    "google_gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "meta-llama_llama-4-maverick": "Llama 4 Maverick",
    "nvidia_nemotron-3-nano-30b-a3b_free": "Nemotron 3 Nano 30B A3B",
}
EXPERT_FILE = "experts_mean_median_scores_detailed_thr_08.xlsx"
# Fixed route IDs keep bootstrap resampling independent of composition labels.
BOOTSTRAP_ROUTE_ORDER = (
    5, 4, 10, 1, 9, 8, 6, 2, 7, 3,
    15, 14, 20, 11, 18, 16, 12, 17, 13, 19,
    25, 24, 30, 21, 28, 26, 22, 27, 23, 29,
    35, 34, 40, 31, 38, 36, 32, 37, 33, 39,
)


def paired_scores(judge_path, expert_path):
    """Average the available runs per route and match to the expert mean."""
    # Accept the sheet with or without a three-row caption.
    first_column = pd.read_excel(
        judge_path, sheet_name="Judge ratings", header=None, usecols=[0], nrows=4
    ).iloc[:, 0]
    header_rows = first_column.index[first_column.eq("ID")]
    if len(header_rows) != 1:
        raise ValueError("Expected the judge sheet to start with an ID column.")
    scores = pd.read_excel(
        judge_path, sheet_name="Judge ratings", header=int(header_rows[0])
    )
    experts = pd.read_excel(expert_path, sheet_name="route_mean_median_scores")
    route = ["ID", "generator", "composition", "route"]
    key = ["judge", *route]
    if scores[["judge", *route, "run"]].isna().any().any():
        raise ValueError("Missing judge, route metadata or run label.")
    if scores.duplicated(["ID", "judge", "run"]).any():
        raise ValueError("More than one row for a judge, expert ID and run.")
    if scores[route].drop_duplicates()["ID"].duplicated().any():
        raise ValueError(
            "A route ID must identify exactly one generator, composition and route."
        )
    if (
        set(scores["run"]) != {"v1", "v2", "v3"}
        or not scores.groupby(key).size().eq(3).all()
    ):
        raise ValueError(
            "Expected one row per planned run (v1, v2 and v3), including missing scores."
        )
    if not scores["faithfulness_score"].dropna().between(0, 1).all():
        raise ValueError("Judge scores must be in [0, 1].")
    # Route means are pandas group means stored at 16 significant digits, as in
    # the reported analysis. Preserve this step and its ties.
    routes = scores.groupby(key, as_index=False)["faithfulness_score"].mean()
    routes["judge_score"] = routes.pop("faithfulness_score").map(
        lambda x: float(f"{x:.16g}")
    )
    paired = routes.merge(
        experts[["ID", "model", "composition", "mean_faithfulness_score"]],
        on="ID",
        how="left",
        suffixes=("", "_expert"),
        validate="many_to_one",
    ).rename(columns={"mean_faithfulness_score": "expert_score"})
    if paired[["judge_score", "expert_score"]].isna().any().any():
        raise ValueError("A route has no judge or expert score. Check input files.")
    if not paired["expert_score"].between(0, 1).all():
        raise ValueError("Expert scores must be in [0, 1].")
    same_composition = paired["composition"] == paired["composition_expert"]
    if (
        not (paired["generator"] == paired["model"].map(MODELS)).all()
        or not same_composition.all()
    ):
        raise ValueError(
            "The judge and expert records disagree on the generator or composition."
        )
    counts = paired.groupby(["judge", "generator"]).size()
    external_pairs = {
        (j, g) for j in MODELS.values() for g in MODELS.values() if j != g
    }
    if set(counts.index) != external_pairs or not counts.eq(10).all():
        raise ValueError("Expected all 12 external pairs, with ten routes per pair.")
    order = {route_id: i for i, route_id in enumerate(BOOTSTRAP_ROUTE_ORDER)}
    return (
        paired.assign(_bootstrap_order=paired["ID"].map(order))
        .sort_values(["judge", "generator", "_bootstrap_order", "composition", "route"])
        .drop(columns="_bootstrap_order")
    )


def absolute_agreement_icc(ratings):
    """ICC(A,1) for paired expert and judge scores; final axes are route and rater."""
    n, k = ratings.shape[-2:]
    grand = ratings.mean(axis=(-2, -1), keepdims=True)
    route_means = ratings.mean(axis=-1, keepdims=True)
    rater_means = ratings.mean(axis=-2, keepdims=True)
    ms_routes = k * ((route_means - grand) ** 2).sum(axis=(-2, -1)) / (n - 1)
    ms_raters = n * ((rater_means - grand) ** 2).sum(axis=(-2, -1)) / (k - 1)
    residual = ratings - route_means - rater_means + grand
    ms_error = (residual**2).sum(axis=(-2, -1)) / ((n - 1) * (k - 1))
    denominator = ms_routes + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    return np.divide(
        ms_routes - ms_error,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=~np.isclose(denominator, 0),
    )


def agreement(group):
    """Compute the eight reported agreement statistics across ten paired routes."""
    x = group["expert_score"].to_numpy()
    y = group["judge_score"].to_numpy()
    rho, p = spearmanr(x, y)
    human_pass, judge_pass = x >= 0.8, y >= 0.8
    if len(set(human_pass)) == len(set(judge_pass)) == 1:
        kappa = 1.0 if human_pass[0] == judge_pass[0] else float("nan")
    else:
        kappa = cohen_kappa_score(human_pass, judge_pass, labels=[False, True])
    paired = np.column_stack([x, y])
    # Each sample resamples whole routes and keeps the judge/expert pairing.
    rng = np.random.default_rng(42)
    sample = rng.integers(0, len(paired), size=(5000, len(paired)))
    limits = np.nanpercentile(absolute_agreement_icc(paired[sample]), [2.5, 97.5])
    return {
        "n": len(x),
        "spearman_rho": rho,
        "spearman_pvalue": p,
        "accuracy": np.mean(human_pass == judge_pass),
        "cohen_kappa": kappa,
        "mae": np.mean(np.abs(y - x)),
        "bias_llm_minus_human": np.mean(y - x),
        "icc_a_1": float(absolute_agreement_icc(paired)),
        "icc_a_1_ci": f"[{limits[0]:.3f}, {limits[1]:.3f}]",
    }


def main():
    here = Path(__file__).resolve().parent
    data = here.parent / "data"
    expert_paths = [
        data / EXPERT_FILE,
        here / EXPERT_FILE,
        Path("synthway") / "bench" / EXPERT_FILE,
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judges", type=Path, default=data / "Table_1_Judge_Ratings.xlsx"
    )
    parser.add_argument(
        "--experts",
        type=Path,
        default=next((p for p in expert_paths if p.is_file()), expert_paths[0]),
    )
    parser.add_argument("--out", type=Path, default=Path("Table_1_recalculated.csv"))
    args = parser.parse_args()
    paired = paired_scores(args.judges, args.experts)
    results = [
        {"judge_model": judge, "generator_model": generator, **agreement(group)}
        for (judge, generator), group in paired.groupby(["judge", "generator"])
    ]
    table = pd.DataFrame(results)
    # Table 1 lists Gemini-generated routes before Claude-generated routes.
    display_order = [
        "Gemini 2.0 Flash",
        "Claude Opus 4.5",
        "Llama 4 Maverick",
        "Nemotron 3 Nano 30B A3B",
    ]
    table["generator_model"] = pd.Categorical(
        table["generator_model"], display_order, ordered=True
    )
    table = table.sort_values(["judge_model", "generator_model"])
    table.to_csv(args.out, index=False, float_format="%.3f")
    print(f"Wrote {len(table)} judge–generator comparisons to {args.out}")


if __name__ == "__main__":
    main()
