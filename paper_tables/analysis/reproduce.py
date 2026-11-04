"""Rebuild the requested SynthWay tables from portable measurements. No API calls."""

from pathlib import Path
import argparse
import csv
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning
from calculate_table_1 import MODELS, agreement, paired_scores, absolute_agreement_icc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS_DISPLAY = list(MODELS.values())
CRITERIA = ["CPC", "Relevance", "Faithfulness"]
CROSS_FIELDS = [
    "spearman_rho",
    "spearman_pvalue",
    "accuracy",
    "cohen_kappa",
    "mae",
    "bias_llm_minus_human",
    "icc_a_1",
    "icc_a_1_ci",
]
MODELS_REPEAT = [MODELS_DISPLAY[i] for i in [1, 3, 2, 0]]
REPEAT_STATS = [
    "Mean score across runs",
    "SD of mean score across runs",
    "Mean pass rate across runs",
    "SD of pass rate across runs",
    "Range of mean score across runs",
    "Range of pass rate across runs",
    "Range of within-run score SD",
]
EXPERT = DATA / "experts_mean_median_scores_detailed_thr_08.xlsx"


def read(name):
    return pd.read_csv(DATA / name, float_precision="round_trip")


def describe(group):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        stats = agreement(group)
    scores = group["judge_score"].to_numpy()
    experts = group["expert_score"].to_numpy()
    stats.update(
        judge_mean=scores.mean(),
        expert_mean=experts.mean(),
        rmse=np.sqrt(np.mean((scores - experts) ** 2)),
    )
    return stats


def repeatability_tables():
    runs = read("pipeline_ratings.csv").replace(
        {"model": {"Nemotron 3 Nano": MODELS_DISPLAY[3]}}
    )
    key = ["model", "run", "metric", "route_id"]
    if runs.duplicated(key).any() or len(runs) != 3240 or runs.score.isna().any():
        raise ValueError(
            "Expected 3,240 unique scored route/run/criterion observations."
        )
    counts = runs.groupby(key[:-1]).score.count()
    if len(counts) != 36 or not counts.eq(90).all() or not runs.threshold.eq(0.8).all():
        raise ValueError(
            "Pipeline score coverage or pass threshold differs from the study design."
        )
    runs["pass_fraction"] = runs.score.ge(runs.threshold)
    per_run = runs.groupby(["model", "metric", "run"]).agg(
        run_mean=("score", "mean"),
        within_sd=("score", "std"),
        pass_rate=("pass_fraction", "mean"),
    )
    values = {}
    for (model, metric), subset in per_run.groupby(level=["model", "metric"]):
        if len(subset) != 3:
            raise ValueError("Expected three distinct generation runs.")
        a, b, c = subset.run_mean, subset.pass_rate, subset.within_sd
        values[model, metric] = [
            a.mean(),
            a.std(),
            b.mean(),
            b.std(),
            a.max() - a.min(),
            b.max() - b.min(),
            c.max() - c.min(),
        ]
    s8 = [["Statistic", *MODELS_REPEAT]]
    s9 = [
        ["Model", *[m for m in MODELS_REPEAT for _ in range(2)]],
        ["Statistic", *["Answer Relevancy", "Faithfulness"] * 4],
    ]
    for i, name in enumerate(REPEAT_STATS):
        s8.append([name, *[values[m, "CPC"][i] for m in MODELS_REPEAT]])
        s9.append(
            [
                name,
                *[
                    values[m, metric][i]
                    for m in MODELS_REPEAT
                    for metric in ("Answer Relevancy", "Faithfulness")
                ],
            ]
        )
    return {8: s8, 9: s9}


def repeated_icc(matrix):
    """Compute ICC(A,1) and reliability of the average of k ratings using the same route bootstrap."""
    if np.ptp(matrix) == 0:
        return ["Not estimable (constant ratings)"] * 2
    n, k = matrix.shape
    idx = np.random.default_rng(42).integers(0, n, size=(5000, n))
    single = absolute_agreement_icc(matrix)
    sampled = absolute_agreement_icc(matrix[idx])

    def average(x):
        denominator = 1 + (k - 1) * x
        return np.divide(
            k * x,
            denominator,
            out=np.full_like(x, np.nan),
            where=~np.isclose(denominator, 0),
        )

    result = []
    for estimate, reps in [(single, sampled), (average(single), average(sampled))]:
        lo, hi = np.nanpercentile(reps, [2.5, 97.5])
        result.append(f"{estimate:.3f}, [{lo:.3f}, {hi:.3f}]")
    return result


def reliability_tables():
    ratings = read("fixed_route_ratings.csv")
    s10 = [["Model", "Statistic", "CPC", "Answer Relevance", "Faithfulness"]]
    for m in MODELS_REPEAT:
        stats = []
        for metric in ("CPC", "Answer Relevancy", "Faithfulness"):
            wide = (
                ratings[(ratings.model == m) & (ratings.criterion == metric)]
                .pivot(index="route_id", columns="repeat", values="score")
                .sort_index()
            )
            if wide.shape != (90, 3) or wide.isna().any().any():
                raise ValueError("Expected three ratings for the same 90 fixed routes.")
            stats.append(repeated_icc(wide.to_numpy()))
        for i, label in enumerate(["ICC(A,1), 95% CI", "ICC(A,3), 95% CI"]):
            s10.append([m, label, *[s[i] for s in stats]])
    pairs = read("context_pairs.csv")
    model_keys = [
        "gemini-2.0-flash-001",
        "claude-opus-4.5",
        "llama-4-maverick",
        "nemotron-3-nano-30b-a3b_free",
    ]
    columns = []
    for m in model_keys:
        v = pairs[(pairs.model == m) & pairs.eligible_gt1].sort_values("route_key")
        if len(v) != 42 or not v.route_key.is_unique or not v.n_context.gt(1).all():
            raise ValueError("Expected 42 eligible paired routes per model.")
        x, y = v.score_before, v.score_after
        delta = y - x
        icc = repeated_icc(np.column_stack([x, y]))[0]
        columns.append(
            [
                x.mean(),
                y.mean(),
                delta.mean(),
                delta.median(),
                delta.std(ddof=0),
                (x >= 0.8).mean(),
                (y >= 0.8).mean(),
                (delta > 0).sum(),
                (delta < 0).sum(),
                ((x >= 0.8) & (y < 0.8)).sum(),
                ((x < 0.8) & (y >= 0.8)).sum(),
                icc,
            ]
        )
    labels = [
        "Mean before",
        "Mean after",
        "Mean difference",
        "Median difference",
        "SD of difference",
        "Pass rate before",
        "Pass rate after",
        "Improved",
        "Worsened",
        "Pass to fail",
        "Fail to pass",
        "ICC(A,1), 95% CI",
    ]
    s11 = [
        ["Metric", *[MODELS_DISPLAY[i] for i in [1, 0, 2, 3]]],
        *[[k, *v] for k, v in zip(labels, zip(*columns))],
    ]
    return {10: s10, 11: s11}


def experts():
    frame = pd.read_excel(EXPERT, sheet_name="route_mean_median_scores")
    return frame.melt(
        id_vars=["ID"],
        value_vars=[
            "mean_cpc_score",
            "mean_relevance_score",
            "mean_faithfulness_score",
        ],
        var_name="criterion",
        value_name="expert_score",
    ).replace(
        {
            "criterion": dict(
                zip(
                    [
                        "mean_cpc_score",
                        "mean_relevance_score",
                        "mean_faithfulness_score",
                    ],
                    CRITERIA,
                )
            )
        }
    )


def cell_icc(stats):
    return f"{stats['icc_a_1']:.3f} {stats['icc_a_1_ci']}"


def self_evaluation_tables():
    data = read("self_judge_ratings.csv").merge(
        experts(), on=["ID", "criterion"], validate="one_to_one"
    )
    if (
        len(data) != 120
        or data.ID.nunique() != 40
        or not np.allclose(
            data[["v1", "v2", "v3"]].mean(axis=1).round(3),
            data.judge_mean,
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "The matched expert and self-judge observations differ from the declared 40 routes."
        )
    data = data.rename(columns={"judge_mean": "judge_score"}).sort_values(
        "analysis_order"
    )
    s13 = [
        ["Model", *[m for m in CRITERIA for _ in range(2)]],
        ["Mean score", *(["Mean score"] * 6)],
        ["Model", *(["LLM judge", "Experts"] * 3)],
    ]
    s14 = [
        ["Model", *[m for m in [*CRITERIA, "All three"] for _ in range(2)]],
        ["Model", *(["LLM judge", "Experts"] * 4)],
    ]
    for model in [*MODELS_DISPLAY, "ALL MODELS"]:
        group = data if model == "ALL MODELS" else data[data.generator == model]
        s13.append(
            [
                model,
                *[
                    group[group.criterion == c][s].mean()
                    for c in CRITERIA
                    for s in ["judge_score", "expert_score"]
                ],
            ]
        )
        rates = {}
        for col in ["judge_score", "expert_score"]:
            passes = group.pivot(index="ID", columns="criterion", values=col).ge(0.8)
            passes["All three"] = passes.all(axis=1)
            for m, n in passes.sum().items():
                rates[m, col] = f"{n}/{len(passes)} ({100*n/len(passes):.1f}%)"
        s14.append(
            [
                model,
                *[
                    rates[m, s]
                    for m in [*CRITERIA, "All three"]
                    for s in ["judge_score", "expert_score"]
                ],
            ]
        )
    cols = ["ρ", "ICC(A,1), 95% CI", "Accuracy", "κ", "MAE", "RMSE", "Bias"]
    s15 = [["Criterion", *cols]]
    s16 = [["Model", "Criterion", *cols]]
    s17 = [
        [
            "Criterion",
            "Subgroup",
            "LLM",
            "Expert",
            "Bias",
            "ρ",
            "ICC(A,1), 95% CI",
            "Accuracy",
            "κ",
            "MAE",
            "RMSE",
        ]
    ]

    def common(s):
        return [
            s["spearman_rho"],
            cell_icc(s),
            s["accuracy"],
            s["cohen_kappa"],
            s["mae"],
            s["rmse"],
            s["bias_llm_minus_human"],
        ]

    for c in ("CPC", "Faithfulness", "Relevance"):
        block = data[data.criterion == c]
        s15.append([c, *common(describe(block))])
        for m in MODELS_DISPLAY:
            s16.append([m, c, *common(describe(block[block.generator == m]))])
        for subgroup in ("composition_top", "composition_rare"):
            s = describe(block[block.subgroup == subgroup])
            s17.append(
                [
                    c,
                    subgroup,
                    s["judge_mean"],
                    s["expert_mean"],
                    s["bias_llm_minus_human"],
                    s["spearman_rho"],
                    cell_icc(s),
                    s["accuracy"],
                    s["cohen_kappa"],
                    s["mae"],
                    s["rmse"],
                ]
            )
    return {13: s13, 14: s14, 15: s15, 16: s16, 17: s17}


def comparison_rows(data):
    rows = [["LLM judge", "Generator", *CROSS_FIELDS]]
    for j in MODELS_DISPLAY:
        for g in [MODELS_DISPLAY[i] for i in (1, 0, 2, 3)]:
            if j == g:
                continue
            block = data[(data.judge == j) & (data.generator == g)]
            if len(block) != 10 or not block.ID.is_unique:
                raise ValueError(f"Expected ten external routes for {j} judging {g}.")
            stats = describe(block)
            rows.append([j, g, *[stats[field] for field in CROSS_FIELDS]])
    return rows


def cross_evaluation_tables():
    t1 = paired_scores(DATA / "Table_1_Judge_Ratings.xlsx", EXPERT)
    out = {1: comparison_rows(t1)}
    ratings = read("cross_judge_ratings.csv")
    pairs = read("cross_judge_route_means.csv")
    key = ["table", "ID", "judge", "generator"]
    if ratings.duplicated(key + ["run"]).any() or len(ratings) != 720:
        raise ValueError("External judge run coverage is incomplete.")
    observed = ratings.groupby(key).score.agg(["count", "mean"]).reset_index()
    pairs = pairs.merge(observed, on=key, validate="one_to_one", how="outer")
    if (
        len(pairs) != 240
        or not pairs["count"].eq(3).all()
        or not np.allclose(pairs.judge_mean, pairs["mean"], rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "Cross-judge observations do not match the linked route means."
        )
    pairs["criterion"] = pairs.table.map({"S18": "CPC", "S19": "Faithfulness"})
    pairs = pairs.merge(
        experts(), on=["ID", "criterion"], validate="many_to_one"
    ).rename(columns={"judge_mean": "judge_score"})
    for n in [18, 19]:
        out[n] = comparison_rows(
            pairs[pairs.table == f"S{n}"].sort_values("analysis_order")
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    tables = {
        **repeatability_tables(),
        **reliability_tables(),
        **self_evaluation_tables(),
        **cross_evaluation_tables(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for n, rows in sorted(tables.items()):
        dest = args.out / ("Table_1.csv" if n == 1 else f"Table_S{n}.csv")
        with dest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            for row in rows:
                writer.writerow(
                    [
                        (
                            (
                                "—"
                                if not np.isfinite(v)
                                else (
                                    f"{v:.12g}" if n == 11 else f"{np.round(v,3):.3f}"
                                )
                            )
                            if isinstance(v, (float, np.floating))
                            else v
                        )
                        for v in row
                    ]
                )
    print(f"Wrote {len(tables)} tables to {args.out}.")


if __name__ == "__main__":
    main()
