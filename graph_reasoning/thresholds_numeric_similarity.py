"""
Analyze numeric-node similarity and suggest merge thresholds.
Save histograms and threshold recommendations to --out_dir (default: graph_output/merge).
Also plot a combined chart for other labels (outside the numeric list).
"""

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Default configuration
DEFAULT_PAIRS_CSV = f"{_REPO_ROOT}/graph_output/merge_pairs_for_renaming.csv"
DEFAULT_OUT_DIR = f"{_REPO_ROOT}/graph_output/merge"

# Numeric labels
DEFAULT_NUMERIC_LABELS = [
    "precursor_mass",
    "precursor_conc",
    "additive_mass",
    "additive_conc",
    "ph_minmax",
    "seed_mass_minmax",
    "seed_conc_minmax",
    "size_length_minmax",
    "size_width_minmax",
    "size_depth_minmax",
    "temperature_minmax",
]

# Synonyms for column autodetection
LABEL_SYNONYMS = [
    "label",
    "label_source",
    "label_target",
    "attr",
    "attribute",
    "attr_name",
    "feature",
    "key",
    "type",
    "name",
    "column",
    "kind",
    "node_label",
    "edge_label",
]
SIMILARITY_SYNONYMS = [
    "similarity",
    "sim",
    "score",
    "cos_sim",
    "cosine_similarity",
    "match_score",
    "prob",
    "probability",
]
DISTANCE_SYNONYMS = ["distance", "dist", "cosine_distance", "1-cosine_similarity", "d"]

# Utilities


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _norm(s: str) -> str:
    """Normalize a column name: lowercase, retaining only letters/digits."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def find_column(df: pd.DataFrame, preferred: str | None, synonyms: list[str]) -> str | None:
    """Return the actual column name using the preferred name or synonyms, ignoring case/underscores."""
    if df.columns.size == 0:
        return None
    colmap = {_norm(c): c for c in df.columns}
    if preferred:
        key = _norm(preferred)
        if key in colmap:
            return colmap[key]
    for syn in synonyms:
        key = _norm(syn)
        if key in colmap:
            return colmap[key]
    return None


# Threshold heuristics


def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold on [0..1]. Useful for bimodal distributions."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    hist, edges = np.histogram(x, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    if hist.sum() <= 0:
        return float("nan")
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(bins))
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    idx = int(np.nanargmax(sigma_b2))
    thr = 0.5 * (edges[idx] + edges[idx + 1])
    return float(thr)


def knee_threshold(x: np.ndarray) -> float:
    """
    CDF knee: point of maximum CDF excess over the diagonal from (min,0) to (max,1).
    Works well for a shifted unimodal distribution.
    """
    x = np.sort(x[np.isfinite(x)])
    if x.size == 0:
        return float("nan")
    uniq, counts = np.unique(x, return_counts=True)
    cdf = np.cumsum(counts) / x.size
    x0, x1 = uniq[0], uniq[-1]
    if x1 <= x0:
        return float(uniq[0])
    t = (uniq - x0) / (x1 - x0)
    dist = cdf - t
    mid = np.searchsorted(cdf, 0.5)
    k_idx = int(np.argmax(dist[mid:]) + mid) if mid < len(dist) else int(np.argmax(dist))
    return float(uniq[k_idx])


def adjust_threshold(base_thr: float, strict: bool, relaxed: bool) -> float:
    """Adjust the threshold: strict -> +0.02; relaxed -> -0.02; strict takes precedence if both flags are set."""
    if math.isnan(base_thr):
        return base_thr
    if strict:
        return float(min(1.0, base_thr + 0.02))
    if relaxed:
        return float(max(0.0, base_thr - 0.02))
    return base_thr


def recommend_threshold(otsu: float, knee: float, stats: dict[str, float]) -> float:
    """Recommend a conservative threshold: max(otsu, knee); fall back to p90 if both are NaN."""
    if not math.isnan(otsu) and not math.isnan(knee):
        return float(max(otsu, knee))
    if not math.isnan(otsu):
        return float(otsu)
    if not math.isnan(knee):
        return float(knee)
    return float(stats.get("p90", 0.9))


def plot_hist(
    values: np.ndarray,
    label: str,
    otsu: float,
    knee: float,
    rec: float,
    out_path: str,
    bins: int = 50,
):
    plt.figure(figsize=(7, 4.2))
    plt.hist(values, bins=bins, range=(0.0, 1.0), alpha=0.85, color="#69b3a2", edgecolor="#2f4f4f")
    if not math.isnan(otsu):
        plt.axvline(otsu, linestyle="--", linewidth=2, color="#ff7f0e", label=f"Otsu={otsu:.3f}")
    if not math.isnan(knee):
        plt.axvline(knee, linestyle="-.", linewidth=2, color="#1f77b4", label=f"Knee={knee:.3f}")
    plt.axvline(rec, linestyle="-", linewidth=2, color="#d62728", label=f"Recommended={rec:.3f}")
    plt.title(f"Similarity histogram: {label}")
    plt.xlabel("similarity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_global_hist_and_cdf(
    all_values: np.ndarray,
    lines: list[tuple[str, float]],
    out_dir: str,
    basename: str,
    bins: int = 100,
):
    """Combined histogram and CDF with customizable line labels."""
    # Histogram
    hist_path = os.path.join(out_dir, f"hist_{basename}.png")
    plt.figure(figsize=(8, 4.6))
    plt.hist(
        all_values, bins=bins, range=(0.0, 1.0), alpha=0.85, color="#8fb9e8", edgecolor="#2f4f4f"
    )
    for name, thr in lines:
        if thr == thr:  # not NaN
            plt.axvline(thr, linestyle="--", linewidth=2, label=f"{name}={thr:.3f}")
    plt.title(f"Similarity histogram: {basename}")
    plt.xlabel("similarity")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # CDF
    cdf_path = os.path.join(out_dir, f"cdf_{basename}.png")
    xs = np.sort(all_values[np.isfinite(all_values)])
    if xs.size > 0:
        ys = np.arange(1, xs.size + 1) / xs.size
        plt.figure(figsize=(8, 4.6))
        plt.plot(xs, ys, linewidth=2)
        for name, thr in lines:
            if thr == thr:
                plt.axvline(thr, linestyle="--", linewidth=2, label=f"{name}={thr:.3f}")
        plt.title(f"Similarity CDF: {basename}")
        plt.xlabel("similarity")
        plt.ylabel("CDF")
        plt.legend()
        plt.tight_layout()
        plt.savefig(cdf_path, dpi=150)
        plt.close()
    return hist_path, cdf_path


# Publication-quality histogram: zoomed, stylized, with legend
def plot_global_hist_publication(
    all_values: np.ndarray,
    lines: list[tuple[str, float]],
    out_path: str,
    x_left: float = 0.98,  # Left cutoff (show [0.98, 1.0] by default)
    x_right: float = 1.0,
    bins: int = 60,
    dpi: int = 300,
):
    """Histogram as in the base version, with an X cutoff and colored lines.
    Y shows counts, not density. Compact legend.
    """
    vals = np.asarray(all_values)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= 0.0) & (vals <= 1.0)]
    if vals.size == 0:
        return

    left = float(max(0.0, min(1.0, x_left)))
    right = float(min(1.0, max(left + 1e-4, x_right)))

    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.0))

    # Standard count histogram
    ax.hist(
        vals,
        bins=bins,
        range=(left, right),
        alpha=0.85,
        color="#9ecae1",
        edgecolor="#2f4f4f",
        linewidth=0.5,
    )

    # Line palette
    color_map = {
        "Global Otsu": "#ff7f0e",  # Orange
        "Global Knee": "#1f77b4",  # Blue
        "Global Weighted Median": "#2ca02c",  # Green
    }

    for name, thr in lines:
        if thr == thr:  # not NaN
            ax.axvline(
                thr,
                linestyle="--",
                linewidth=2.2,
                color=color_map.get(name, None),
                label=f"{name} = {thr:.3f}",
            )

    ax.set_title("Similarity histogram: ALL numeric")
    ax.set_xlabel("similarity")
    ax.set_ylabel("count")
    ax.set_xlim(left, right)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Select similarity thresholds for numeric nodes.")
    ap.add_argument("--pairs_csv", default=DEFAULT_PAIRS_CSV, help="merge_pairs_for_renaming.csv")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Artifact directory (merge outputs)")
    ap.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    ap.add_argument(
        "--min_samples",
        type=int,
        default=20,
        help="Minimum pairs per label for a supported threshold (otherwise mark as low_support)",
    )
    # Explicit column names, if needed
    ap.add_argument(
        "--col-label",
        dest="col_label",
        default=None,
        help="Name of the label/label_source/label_target column",
    )
    ap.add_argument(
        "--col-similarity",
        dest="col_similarity",
        default=None,
        help="Name of the similarity column [0..1]",
    )
    ap.add_argument(
        "--col-distance",
        dest="col_distance",
        default=None,
        help="Name of the distance column [0..1], if similarity is unavailable",
    )
    # Manual adjustments
    ap.add_argument(
        "--strict", action="store_true", help="More conservative: increase recommended by 0.02"
    )
    ap.add_argument(
        "--relaxed", action="store_true", help="More permissive: decrease recommended by 0.02"
    )
    # Override the label set (comma-separated)
    ap.add_argument(
        "--labels",
        default=",".join(DEFAULT_NUMERIC_LABELS),
        help="Comma-separated labels; defaults to the predefined numeric label set",
    )
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.pairs_csv)

    # Autodetect columns
    label_col = find_column(df, args.col_label, LABEL_SYNONYMS)
    sim_col = find_column(df, args.col_similarity, SIMILARITY_SYNONYMS)
    dist_col = None if sim_col else find_column(df, args.col_distance, DISTANCE_SYNONYMS)

    print("[AUTO-DETECT]")
    print("  columns in CSV:", list(df.columns))
    print(f"  label column -> {repr(label_col)}")
    if sim_col:
        print(f"  similarity column -> {repr(sim_col)} (direct)")
    elif dist_col:
        print(f"  distance column -> {repr(dist_col)} (will use 1 - distance)")
    else:
        print("  similarity/distance column not found by synonyms.")

    if not label_col or (not sim_col and not dist_col):
        raise ValueError(
            "Could not identify columns. Specify --col-label and --col-similarity explicitly, "
            "or use --col-distance if distance is in [0,1]."
        )

    # Final label list
    numeric_labels = [s.strip() for s in args.labels.split(",") if s.strip()]

    # Construct similarity values in [0..1]
    if sim_col:
        sim = pd.to_numeric(df[sim_col], errors="coerce")
        sim_name = sim_col
        transform_note = "direct"
    else:
        dist = pd.to_numeric(df[dist_col], errors="coerce")
        if dist.min() < -1e-9 or dist.max() > 1.0 + 1e-9:
            raise ValueError(
                f"Distance column {dist_col!r} is outside [0,1]; automatic normalization is disabled."
            )
        sim = 1.0 - dist
        sim_name = f"1-({dist_col})"
        transform_note = "from_distance"

    labels = df[label_col].astype(str)

    # Filter by the requested labels
    mask_numeric = labels.isin(numeric_labels)
    sub_sim = sim[mask_numeric].astype(float)
    sub_labels = labels[mask_numeric]

    if sub_sim.dropna().empty:
        raise ValueError(
            "No data found for the specified numeric labels. Check the label/label_source/label_target column values."
        )

    print(
        f"[INFO] Using similarity='{sim_name}' ({transform_note}), label='{label_col}'. "
        f"Filtering by {len(numeric_labels)} numeric labels."
    )

    rows = []
    per_label_thresholds: dict[str, float] = {}
    pooled_numeric: list[np.ndarray] = []

    for lb in numeric_labels:
        d = sub_sim[sub_labels == lb].dropna().values
        if d.size == 0:
            continue
        d = d[(d >= 0.0) & (d <= 1.0)]
        if d.size == 0:
            continue
        pooled_numeric.append(d)

        stats = {
            "n": int(d.size),
            "min": float(np.min(d)),
            "max": float(np.max(d)),
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "p75": float(np.quantile(d, 0.75)),
            "p90": float(np.quantile(d, 0.90)),
            "p95": float(np.quantile(d, 0.95)),
        }
        ot = otsu_threshold(d, bins=max(args.bins, 64))
        kn = knee_threshold(d)
        rec_base = recommend_threshold(ot, kn, stats)
        rec_adj = adjust_threshold(rec_base, strict=args.strict, relaxed=args.relaxed)

        low_support = d.size < args.min_samples

        img_path = os.path.join(args.out_dir, f"hist_{lb}.png")
        plot_hist(d, lb, ot, kn, rec_adj, img_path, bins=args.bins)

        rows.append(
            {
                "label": lb,
                **stats,
                "otsu": ot,
                "knee": kn,
                "recommended_threshold_base": rec_base,
                "recommended_threshold_final": rec_adj,
                "low_support": bool(low_support),
                "hist_png": img_path,
            }
        )
        per_label_thresholds[lb] = rec_adj

    if not rows:
        print("[WARN] No valid similarity values in [0,1] for the specified labels.")
        return

    tab = pd.DataFrame(rows).sort_values(by="label")
    csv_path = os.path.join(args.out_dir, "thresholds_numeric_similarity.csv")
    tab.to_csv(csv_path, index=False)

    # Global threshold: weighted median of final recommendations (weight = n)
    weights = tab["n"].values
    recs = tab["recommended_threshold_final"].values
    order = np.argsort(recs)
    recs_sorted = recs[order]
    w_sorted = weights[order].astype(float)
    w_cum = np.cumsum(w_sorted) / max(1.0, w_sorted.sum())
    idx = np.searchsorted(w_cum, 0.5)
    global_weighted_median_numeric = float(recs_sorted[min(idx, len(recs_sorted) - 1)])

    # Combined plot for numeric labels
    if pooled_numeric:
        all_num = np.concatenate(pooled_numeric, axis=0)
        all_num = all_num[(all_num >= 0.0) & (all_num <= 1.0)]
        glob_otsu_num = otsu_threshold(all_num, bins=max(args.bins, 128))
        glob_knee_num = knee_threshold(all_num)
        hist_num_path, cdf_num_path = plot_global_hist_and_cdf(
            all_num,
            lines=[
                ("Global Otsu", glob_otsu_num),
                ("Global Knee", glob_knee_num),
                ("Global Weighted Median", global_weighted_median_numeric),
            ],
            out_dir=args.out_dir,
            basename="ALL_numeric",
            bins=max(args.bins, 100),
        )
        # Redraw the publication version to the same file
        plot_global_hist_publication(
            all_num,
            lines=[
                ("Global Otsu", glob_otsu_num),
                ("Global Knee", glob_knee_num),
                ("Global Weighted Median", global_weighted_median_numeric),
            ],
            out_path=hist_num_path,
            x_left=0.98,
            bins=max(args.bins, 100),
        )

    # Combined plot for nonnumeric labels.
    mask_other = ~labels.isin(numeric_labels)
    other_sim = pd.to_numeric(sim[mask_other], errors="coerce").dropna().values
    other_sim = other_sim[(other_sim >= 0.0) & (other_sim <= 1.0)]

    if other_sim.size > 0:
        glob_otsu_other = otsu_threshold(other_sim, bins=max(args.bins, 128))
        glob_knee_other = knee_threshold(other_sim)
        other_median = float(np.median(other_sim))
        hist_other_path, cdf_other_path = plot_global_hist_and_cdf(
            other_sim,
            lines=[
                ("Other Otsu", glob_otsu_other),
                ("Other Knee", glob_knee_other),
                ("Other Median", other_median),
            ],
            out_dir=args.out_dir,
            basename="ALL_other",
            bins=max(args.bins, 100),
        )
        other_labels_used = sorted(set(labels[mask_other].astype(str)))
    else:
        glob_otsu_other = float("nan")
        glob_knee_other = float("nan")
        other_median = float("nan")
        hist_other_path = ""
        cdf_other_path = ""
        other_labels_used = []

    meta = {
        "used_similarity_column": sim_name,
        "label_column": label_col,
        "transform": transform_note,
        "strict": args.strict,
        "relaxed": args.relaxed,
        "min_samples_flag": args.min_samples,
        # numeric
        "global_weighted_median_numeric": global_weighted_median_numeric,
        "global_otsu_on_numeric_pool": glob_otsu_num,
        "global_knee_on_numeric_pool": glob_knee_num,
        "global_hist_numeric_png": hist_num_path,
        "global_cdf_numeric_png": cdf_num_path,
        "per_label_thresholds": per_label_thresholds,
        "numeric_labels_used": numeric_labels,
        # others
        "other_labels_used": other_labels_used,
        "global_median_other_pool": other_median,
        "global_otsu_on_other_pool": glob_otsu_other,
        "global_knee_on_other_pool": glob_knee_other,
        "global_hist_other_png": hist_other_path,
        "global_cdf_other_png": cdf_other_path,
    }
    with open(
        os.path.join(args.out_dir, "thresholds_numeric_similarity.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved table: {csv_path}")
    print(f"[OK] Saved per-label histograms to: {args.out_dir}")
    if hist_num_path:
        print(f"[OK] Saved global numeric histogram: {hist_num_path}")
        print(f"[OK] Saved global numeric CDF:       {cdf_num_path}")
    if hist_other_path:
        print(f"[OK] Saved global OTHER histogram:  {hist_other_path}")
        print(f"[OK] Saved global OTHER CDF:        {cdf_other_path}")
    print(f"[OK] Global weighted median (numeric): {global_weighted_median_numeric:.3f}")


if __name__ == "__main__":
    main()
