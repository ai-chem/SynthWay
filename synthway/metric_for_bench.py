from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
import json
from pathlib import Path

import pandas as pd

model_files = [
    {
        "model_label": "anthropic_claude-opus-4.5",
        "faith_path": Path(
            f"{_REPO_ROOT}/synthway/check_metrics/evaluate_faithfullness/generation_evaluate_faithfullness_anthropic_claude-opus-4.5.json"
        ),
        "chem_path": Path(
            f"{_REPO_ROOT}/synthway/check_metrics/generation_rag_eval_correct/anthropic_claude-opus-4.5.json"
        ),
    },
    {
        "model_label": "google_gemini-2.0-flash-001",
        "faith_path": Path(
            f"{_REPO_ROOT}/synthway/check_metrics/evaluate_faithfullness/generation_evaluate_faithfullness_google_gemini-2.0-flash-001.json"
        ),
        "chem_path": Path(
            f"{_REPO_ROOT}/synthway/check_metrics/generation_rag_eval_correct/google_gemini-2.0-flash-001.json"
        ),
    },
    {
        "model_label": "meta-llama_llama-4-maverick",
        "faith_path": Path(
            f"{_REPO_ROOT}/synthway/check_metrics/evaluate_faithfullness/generation_evaluate_faithfullness_meta-llama_llama-4-maverick.json"
        ),
        "chem_path": Path(
            f"{_REPO_ROOT}/synthway/check_metrics/generation_rag_eval_correct/llama-4-maverick.json"
        ),
    },
    {
        "model_label": "nvidia_nemotron-3-nano-30b-a3b_free",
        "faith_path": Path(
            f"{_REPO_ROOT}/synthway/check_metrics/evaluate_faithfullness/generation_evaluate_faithfullness_nvidia_nemotron-3-nano-30b-a3b_free.json"
        ),
        "chem_path": Path(
            f"{_REPO_ROOT}/synthway/check_metrics/generation_rag_eval_correct/nvidia_nemotron-3-nano-30b-a3b_free.json"
        ),
    },
]
selected_routes_path = Path(
    f"{_REPO_ROOT}/synthway/check_metrics/final_slot_selection/final_selected_40_routes.csv"
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_selected_routes(path: Path) -> set[str]:
    df = pd.read_csv(path)

    candidate_cols = [
        "path",
        "route_path",
        "selected_route_path",
        "file_path",
    ]

    for col in candidate_cols:
        if col in df.columns:
            return set(df[col].dropna().astype(str).str.strip())

    raise ValueError(
        f"Could not find a route path column in {path}. "
        f"Expected one of: {candidate_cols}. Available columns: {list(df.columns)}"
    )


def extract_model_from_path(path_str: str) -> str:
    """
    Extract the model from a path of the form:
    .../output_LLM_stratified_anthropic_claude-opus-4.5/.../route_1.json
    """
    marker = "output_LLM_stratified_"
    if marker in path_str:
        tail = path_str.split(marker, 1)[1]
        return tail.split("/", 1)[0]
    return "unknown_model"


def deduplicate_faith_records(records: list[dict]) -> pd.DataFrame:
    """
    The faithfulness file may contain duplicate paths:
    - error entry
    - a normal entry with answer_score / faithfulness_score

    Logic:
    1. Prefer the entry with actual metrics for each path.
    2. If no entry has metrics, retain the first.
    """
    grouped = {}

    for rec in records:
        route_path = rec.get("path")
        if route_path is None:
            continue
        grouped.setdefault(route_path, []).append(rec)

    selected = []
    for route_path, recs in grouped.items():
        scored = [r for r in recs if ("answer_score" in r) or ("faithfulness_score" in r)]
        chosen = scored[-1] if scored else recs[0]
        selected.append(chosen)

    return pd.DataFrame(selected)


def prepare_chem_df(records: list[dict], selected_paths: set[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(records).copy()

    if selected_paths is not None:
        df = df[df["path"].astype(str).isin(selected_paths)].copy()

    df = df.rename(
        columns={
            "score": "chemistry_score",
            "threshold": "chemistry_threshold",
            "passed": "chemistry_passed",
            "reason": "chemistry_reason",
        }
    )

    return df[
        [
            "path",
            "composition",
            "method",
            "chemistry_score",
            "chemistry_threshold",
            "chemistry_passed",
            "chemistry_reason",
        ]
    ]


def prepare_faith_df(records: list[dict], selected_paths: set[str] | None = None) -> pd.DataFrame:
    df = deduplicate_faith_records(records).copy()

    if selected_paths is not None:
        df = df[df["path"].astype(str).isin(selected_paths)].copy()

    # Select only columns present in the records.
    keep_cols = [
        "path",
        "composition",
        "answer_score",
        "answer_threshold",
        "answer_passed",
        "answer_reason",
        "faithfulness_score",
        "faithfulness_threshold",
        "faithfulness_passed",
        "faithfulness_reason",
    ]

    existing_cols = [c for c in keep_cols if c in df.columns]
    return df[existing_cols]


def build_model_table(
    model_label: str,
    faith_path: Path,
    chem_path: Path,
    selected_paths: set[str],
) -> pd.DataFrame:
    faith_records = load_json(faith_path)
    chem_records = load_json(chem_path)

    faith_df = prepare_faith_df(faith_records, selected_paths=selected_paths)
    chem_df = prepare_chem_df(chem_records, selected_paths=selected_paths)

    result = chem_df.merge(faith_df, on="path", how="outer", suffixes=("_chem", "_faith"))

    if "composition_chem" in result.columns or "composition_faith" in result.columns:
        result["composition"] = result.get("composition_chem").combine_first(
            result.get("composition_faith")
        )

    result["model"] = model_label
    result["model_from_path"] = result["path"].apply(extract_model_from_path)

    result["all_three_passed"] = (
        result["chemistry_passed"].fillna(False)
        & result["answer_passed"].fillna(False)
        & result["faithfulness_passed"].fillna(False)
    )

    final_cols = [
        "model",
        "model_from_path",
        "composition",
        "path",
        "chemistry_score",
        "chemistry_passed",
        "answer_score",
        "answer_passed",
        "faithfulness_score",
        "faithfulness_passed",
        "all_three_passed",
    ]

    return result[final_cols].sort_values(["model", "composition", "path"]).reset_index(drop=True)


# load selected routes
selected_paths = load_selected_routes(selected_routes_path)

# build tables for all models
all_tables = []
for cfg in model_files:
    model_df = build_model_table(
        model_label=cfg["model_label"],
        faith_path=cfg["faith_path"],
        chem_path=cfg["chem_path"],
        selected_paths=selected_paths,
    )
    all_tables.append(model_df)

final_df = pd.concat(all_tables, ignore_index=True)
final_df = final_df.sort_values(["model", "composition", "path"]).reset_index(drop=True)

print("Selected routes in CSV:", len(selected_paths))
print("Matched routes in the final table:", final_df["path"].nunique())
print(final_df.head(20))
print()
print("Total rows:", len(final_df))
print("Unique models:", final_df["model"].nunique())
print("Passed all three:", int(final_df["all_three_passed"].sum()))
print()
print("Pass count by model:")
print(final_df.groupby("model")["all_three_passed"].sum())
print()
print("Row count by model:")
print(final_df.groupby("model").size())

# save
out_csv = Path(f"{_REPO_ROOT}/synthway/check_metrics/all_models_merged_metrics_selected_40.csv")
out_xlsx = Path(f"{_REPO_ROOT}/synthway/check_metrics/all_models_merged_metrics_selected_40.xlsx")

final_df.to_csv(out_csv, index=False, encoding="utf-8")
final_df.to_excel(out_xlsx, index=False)

print(f"Saved CSV:  {out_csv}")
print(f"Saved XLSX: {out_xlsx}")
