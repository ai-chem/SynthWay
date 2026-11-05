"""Repeated-rating reliability calculations and route-level bootstrap intervals."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_CI_ALPHA = 0.05
BOOTSTRAP_N_RESAMPLES = 5000
BOOTSTRAP_RANDOM_SEED = 42
ROUTE_ID_FIELD = "path"


def normalize_route_id(path_str):
    """
    Convert an absolute path to a stable route_id shared across repeats.
    For example:
    /.../output_LLM_stratified_anthropic_claude-opus-4.5_v2/1184.json_Ag_(0.25_wt%)_1138/route_2.json
    -> 1184.json_Ag_(0.25_wt%)_1138/route_2.json
    """
    p = Path(path_str.replace("\\", "/"))
    if len(p.parts) < 2:
        return str(p)
    return "/".join(p.parts[-2:])


def load_scores(files, score_field, route_id_field=ROUTE_ID_FIELD):
    """
    Load scores from multiple JSON files.
    Each file represents one repeated evaluation.
    Normalize route_id to match the same routes across repeats.
    Skip records where the metric was not calculated (e.g., error-only entries),
    score=None, the route is missing, or scores are strings, booleans,
    infinite, or NaN. Numbers outside [0, 1] are data errors.
    No requested repeat may disappear after filtering.
    """
    rows = []
    files = list(files)
    if len(files) < 2:
        raise ValueError("Need at least two repeated-rating files.")

    for rep_idx, file_path in enumerate(files, start=1):
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as f_in:
            data = json.load(f_in)

        if not isinstance(data, list):
            raise ValueError(f"{file_path} does not contain a JSON list.")

        kept_in_repeat = 0
        skipped_missing_score = 0
        skipped_missing_route = 0
        skipped_invalid_score = 0

        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"{file_path} contains a non-object score record.")
            if not isinstance(item.get(route_id_field), str) or not item[route_id_field].strip():
                skipped_missing_route += 1
                continue

            if score_field not in item or item[score_field] is None:
                skipped_missing_score += 1
                continue

            raw_score = item[score_field]
            if type(raw_score) not in (int, float) or not np.isfinite(raw_score):
                skipped_invalid_score += 1
                continue
            score_value = float(raw_score)
            if not 0 <= score_value <= 1:
                raise ValueError(f"{file_path}: {score_field} must be between 0 and 1.")

            rows.append(
                {
                    "route_id": normalize_route_id(item[route_id_field]),
                    "repeat": rep_idx,
                    "score": score_value,
                }
            )
            kept_in_repeat += 1

        print(
            f"[load_scores] {file_path.name}: "
            f"kept={kept_in_repeat}, "
            f"skipped_missing_route={skipped_missing_route}, "
            f"skipped_missing_score={skipped_missing_score}, "
            f"skipped_invalid_score={skipped_invalid_score}"
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No valid rows found for metric '{score_field}'.")

    counts = df.groupby(["route_id", "repeat"]).size()
    duplicated = counts[counts > 1]
    if not duplicated.empty:
        raise ValueError(
            "Found duplicate route_id within the same repeat after normalization/filtering. Examples:\n"
            f"{duplicated.head()}"
        )

    wide = (
        df.pivot(index="route_id", columns="repeat", values="score")
        .sort_index()
        .sort_index(axis=1)
        .reindex(columns=range(1, len(files) + 1))
    )

    if wide.isna().any().any():
        missing_routes = wide[wide.isna().any(axis=1)]
        raise ValueError(
            "Some routes are missing in one or more repeats after filtering.\n"
            "This means not every logical route received this metric in every run.\n"
            f"Examples:\n{missing_routes.head()}"
        )

    return df, wide


def anova_components(ratings_matrix):
    """
    Return ANOVA components for a matrix where
    rows represent routes (targets) and columns represent repeated evaluations.
    """
    y = np.asarray(ratings_matrix, dtype=float)

    if y.ndim != 2:
        raise ValueError("ratings_matrix must be 2D.")

    n, k = y.shape
    if n < 2 or k < 2:
        raise ValueError("Need at least 2 targets and 2 ratings per target.")
    if not np.isfinite(y).all():
        raise ValueError("ratings_matrix must contain only finite values.")

    grand_mean = y.mean()
    row_means = y.mean(axis=1)
    col_means = y.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = np.sum((y - row_means[:, None] - col_means[None, :] + grand_mean) ** 2)
    ss_total = np.sum((y - grand_mean) ** 2)

    df_rows = n - 1
    df_cols = k - 1
    df_error = (n - 1) * (k - 1)

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error

    return {
        "n_targets": n,
        "k_repeats": k,
        "grand_mean": grand_mean,
        "SS_total": ss_total,
        "SS_rows": ss_rows,
        "SS_cols": ss_cols,
        "SS_error": ss_error,
        "df_rows": df_rows,
        "df_cols": df_cols,
        "df_error": df_error,
        "MS_rows": ms_rows,
        "MS_cols": ms_cols,
        "MS_error": ms_error,
    }


def icc_3_1(stats):
    """ICC(3,1): two-way mixed, consistency, single measurement."""
    msr = stats["MS_rows"]
    mse = stats["MS_error"]
    k = stats["k_repeats"]
    return (msr - mse) / (msr + (k - 1) * mse)


def icc_3_k(stats):
    """ICC(3,k): two-way mixed, consistency, average measurement."""
    msr = stats["MS_rows"]
    mse = stats["MS_error"]
    return (msr - mse) / msr


def icc_a_1(stats):
    """ICC(A,1): two-way mixed, absolute agreement, single measurement."""
    msr = stats["MS_rows"]
    msc = stats["MS_cols"]
    mse = stats["MS_error"]
    n = stats["n_targets"]
    k = stats["k_repeats"]
    return (msr - mse) / (msr + (k - 1) * mse + (k * (msc - mse) / n))


def icc_a_k(stats):
    """ICC(A,k): two-way mixed, absolute agreement, average measurement."""
    msr = stats["MS_rows"]
    msc = stats["MS_cols"]
    mse = stats["MS_error"]
    n = stats["n_targets"]
    return (msr - mse) / (msr + (msc - mse) / n)


def bootstrap_icc_cis(
    ratings_matrix,
    alpha=BOOTSTRAP_CI_ALPHA,
    n_resamples=BOOTSTRAP_N_RESAMPLES,
    random_seed=BOOTSTRAP_RANDOM_SEED,
):
    """
    Percentile bootstrap 95% CI for all ICC variants by resampling targets (rows).
    This is used to report CI not only for ICC(3,1)/(3,k), but also for ICC(A,1)/(A,k).
    """
    y = np.asarray(ratings_matrix, dtype=float)

    anova_components(y)  # Validate before resampling; do not hide malformed input.
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive).")
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    n, _ = y.shape

    rng = np.random.default_rng(random_seed)
    metric_names = ["ICC(A,1)", "ICC(A,k)", "ICC(3,1)", "ICC(3,k)"]
    samples = {name: [] for name in metric_names}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_boot = y[idx, :]

        try:
            stats_boot = anova_components(y_boot)
            values = {
                "ICC(A,1)": icc_a_1(stats_boot),
                "ICC(A,k)": icc_a_k(stats_boot),
                "ICC(3,1)": icc_3_1(stats_boot),
                "ICC(3,k)": icc_3_k(stats_boot),
            }
        except Exception:
            continue

        for name, value in values.items():
            if np.isfinite(value):
                samples[name].append(value)

    ci_map = {}
    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)

    for name, vals in samples.items():
        if len(vals) == 0:
            ci_map[name] = (np.nan, np.nan)
        else:
            ci_map[name] = (
                float(np.percentile(vals, lower_q)),
                float(np.percentile(vals, upper_q)),
            )

    return ci_map
