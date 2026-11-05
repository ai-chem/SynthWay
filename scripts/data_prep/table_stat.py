"""Summarize parameter coverage by synthesis_id from IN_XLSX.

A parameter is present when a procedure has at least one known value.
Treat NaN, "unknown", -999 and "-999" as missing; trim string values.

Write present/missing counts and percentages to the
procedure_parameter_coverage Excel sheet and save a stacked PNG chart,
ordered by decreasing coverage.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Hardcoded inputs/columns
IN_XLSX = f"{_REPO_ROOT}/data/merged_all.xlsx"
SID_COL = "synthesis_id"
# None selects all parameter columns except synthesis_id and EXCLUDE_COLS.
# synthesis_id_original is metadata, not a synthesis parameter.
VALUE_COLS = None
EXCLUDE_COLS = {"synthesis_id_original"}
RENAME_COLS = {"pdfno_list": "pdfno"}

# Output paths
OUT_XLSX = os.path.splitext(IN_XLSX)[0] + "_procedure_coverage.xlsx"
OUT_PNG = os.path.splitext(IN_XLSX)[0] + "_procedure_coverage.png"

# Plot colors
COLOR_MAIN = "#04C9AD"
COLOR_ALT = "#5C4A9C"


# Unknown handling
UNKNOWN_STR = "unknown"
UNKNOWN_NUM = -999


def _to_known_or_none(x):
    """Return the value if known; otherwise None for unknown markers or NaN."""
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        if x == UNKNOWN_STR or x == "-999":
            return None
        return x
    if isinstance(x, (int, float)) and x == UNKNOWN_NUM:
        return None
    return x


def build_procedure_coverage(df: pd.DataFrame, sid_col: str, value_cols: list[str]) -> pd.DataFrame:
    """
    For each parameter column, count in how many procedures (SIDs) the parameter is present
    at least once, and in how many procedures it is missing entirely.
    """
    total_procedures = df[sid_col].nunique()
    rows = []

    for col in value_cols:
        g = df[[sid_col, col]].copy()
        g["_known"] = g[col].map(_to_known_or_none)

        present_by_sid = g.groupby(sid_col)["_known"].apply(lambda x: x.notna().any())
        present_cnt = int(present_by_sid.sum())
        missing_cnt = int((~present_by_sid).sum())

        present_pct = present_cnt / total_procedures * 100 if total_procedures else 0.0
        missing_pct = missing_cnt / total_procedures * 100 if total_procedures else 0.0

        display_name = RENAME_COLS.get(col, col)
        rows.append(
            {
                "Parameter": display_name,
                "Present procedures": present_cnt,
                "Missing procedures": missing_cnt,
                "Present (%)": present_pct,
                "Missing (%)": missing_pct,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(by=["Present (%)", "Parameter"], ascending=[False, True]).reset_index(
        drop=True
    )
    return out


def plot_procedure_coverage(coverage_df: pd.DataFrame, out_png: str) -> None:
    """Save a horizontal stacked bar chart (Present vs Missing) sorted by Present (%)."""
    if coverage_df.empty:
        raise ValueError("Coverage table is empty. Nothing to plot.")

    plot_df = coverage_df.iloc[::-1].copy()  # reverse so highest coverage appears at the top

    n_params = len(plot_df)
    fig_height = max(8, 0.58 * n_params + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    title_fontsize = 34
    axis_label_fontsize = 28
    tick_fontsize = 24
    value_label_fontsize = 23
    legend_fontsize = 23

    y = range(n_params)
    present = plot_df["Present (%)"]
    missing = plot_df["Missing (%)"]

    ax.barh(
        y, present, label="Present", color=COLOR_MAIN, height=0.82, edgecolor="white", linewidth=1.2
    )
    ax.barh(
        y,
        missing,
        left=present,
        label="Missing",
        color=COLOR_ALT,
        height=0.82,
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_df["Parameter"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Parameter coverage across procedures (%)", fontsize=axis_label_fontsize)
    ax.set_ylabel("Parameter", fontsize=axis_label_fontsize)
    ax.set_title(
        "Procedure-level coverage of synthesis parameters", fontsize=title_fontsize, pad=12
    )
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize, frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Add present % labels at the end of the present segment when there is enough space
    for i, val in enumerate(present):
        if val >= 5:
            ax.text(
                val - 1.0, i, f"{val:.1f}%", va="center", ha="right", fontsize=value_label_fontsize
            )
        else:
            ax.text(
                val + 1.0, i, f"{val:.1f}%", va="center", ha="left", fontsize=value_label_fontsize
            )

    plt.tight_layout()
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(IN_XLSX):
        raise FileNotFoundError("Input file not found at the hardcoded path:\n" + IN_XLSX)

    df = pd.read_excel(IN_XLSX)

    if SID_COL not in df.columns:
        raise KeyError(f"Column '{SID_COL}' not found. Available: {list(df.columns)}")

    value_cols = VALUE_COLS or [c for c in df.columns if c != SID_COL and c not in EXCLUDE_COLS]
    missing_excluded = [c for c in EXCLUDE_COLS if c not in df.columns]
    if missing_excluded:
        print(f"Excluded columns not found and skipped: {missing_excluded}")
    if not value_cols:
        raise ValueError("No value columns to analyze. Ensure the file has columns besides SID.")

    coverage_df = build_procedure_coverage(df, SID_COL, value_cols)

    print("Input file:", IN_XLSX)
    print("Output Excel:", OUT_XLSX)
    print("Output chart:", OUT_PNG)
    print(f"Columns analyzed: {len(value_cols)} | SID column: {SID_COL}")
    print(f"Total procedures: {df[SID_COL].nunique()}")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        coverage_df.to_excel(xw, sheet_name="procedure_parameter_coverage", index=False)

    plot_procedure_coverage(coverage_df, OUT_PNG)
    print("Done.")


if __name__ == "__main__":
    main()
