#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select one common proposal per slot across 4 models.

Pipeline:
1) For each model and each proposal in a slot, take the best route among 3 routes.
2) For each slot+proposal, average that best-route score across models.
3) Choose the proposal with the best average score.
4) Fix this proposal for all models.
5) For each model, keep its best route inside the chosen proposal.

Score:
    mean_score = (chem_score + faithfulness_score + answer_score) / 3

Expected input schemas:
- chemical files: list[dict] with at least:
    path, score
- faithfulness files: list[dict] with at least:
    path, answer_score, faithfulness_score

The same route is matched across files by normalized route path.
"""

from __future__ import annotations

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
import json
import re
from pathlib import Path
from typing import Any

# Configuration

SLOT_ASSIGNMENTS: dict[str, list[str]] = {
    "composition_top": [
        f"{_REPO_ROOT}/synthway/json/2288.json_NiO_2872.json",
        f"{_REPO_ROOT}/synthway/json/1561.json_Ag_1749.json",
        f"{_REPO_ROOT}/synthway/json/1712.json_GO_3644.json",
    ],
    "composition_rare": [
        f"{_REPO_ROOT}/synthway/json/893.json_Fe7C3@C_1724.json",
        f"{_REPO_ROOT}/synthway/json/2240.json_dodecanethiol-coated_Au_3616.json",
        f"{_REPO_ROOT}/synthway/json/1164.json_CD-MOF_2732.json",
    ],
    "formula_simple": [
        f"{_REPO_ROOT}/synthway/json/2016.json_Co80Ni20_3532.json",
        f"{_REPO_ROOT}/synthway/json/1184.json_Ag_(0.25_wt%)_1138.json",
        f"{_REPO_ROOT}/synthway/json/1303.json_Se_3591.json",
    ],
    "formula_complex": [
        f"{_REPO_ROOT}/synthway/json/1385.json_Zn0.1Ni0.9_1299.json",
        f"{_REPO_ROOT}/synthway/json/1838.json_Pd@Pt_683.json",
        f"{_REPO_ROOT}/synthway/json/324.json_Zn1-xErxO_(0_1_5_10_at.%)_1120.json",
    ],
    "method_frequent": [
        f"{_REPO_ROOT}/synthway/json/2306.json_S_739.json",
        f"{_REPO_ROOT}/synthway/json/1194.json_HA-Alg-ZnO_297.json",
        f"{_REPO_ROOT}/synthway/json/1284.json_Lf-CCAu_3822.json",
    ],
    "method_rare": [
        f"{_REPO_ROOT}/synthway/json/864.json_Au_3126.json",
        f"{_REPO_ROOT}/synthway/json/2160.json_Co3O4@NiO_1834.json",
        f"{_REPO_ROOT}/synthway/json/938.json_HfO2@Au_1516.json",
    ],
    "recipe_minimal": [
        f"{_REPO_ROOT}/synthway/json/2084.json_Pt_CdS_2287.json",
        f"{_REPO_ROOT}/synthway/json/2365.json_Cu2O_4138.json",
        f"{_REPO_ROOT}/synthway/json/557.json_ZnO_2509.json",
    ],
    "recipe_normal": [
        f"{_REPO_ROOT}/synthway/json/710.json_Сu_4119.json",
        f"{_REPO_ROOT}/synthway/json/1234.json_Mat_PEG-CuO_1710.json",
        f"{_REPO_ROOT}/synthway/json/1838.json_Pd_682.json",
    ],
    "recipe_hardcore": [
        f"{_REPO_ROOT}/synthway/json/2108.json_PEG_coated_Fe3O4_3226.json",
        f"{_REPO_ROOT}/synthway/json/1542.json_SiO2-ZnO_2216.json",
        f"{_REPO_ROOT}/synthway/json/1815.json_H2TCPP-Fe2O3_427.json",
    ],
    "recipe_seed_mediated": [
        f"{_REPO_ROOT}/synthway/json/580.json_HAp_-_Ag_1589.json",
        f"{_REPO_ROOT}/synthway/json/920.json_Pt14Ag86_2147.json",
        f"{_REPO_ROOT}/synthway/json/1799.json_10Co3O4@CeO2_1730.json",
    ],
}

MODEL_FILES: dict[str, dict[str, str]] = {
    "google_gemini-2.0-flash-001": {
        "faith": f"{_REPO_ROOT}/synthway/check_metrics/evaluate_faithfullness/generation_evaluate_faithfullness_google_gemini-2.0-flash-001.json",
        "chem": f"{_REPO_ROOT}/synthway/check_metrics/generation_rag_eval_correct/google_gemini-2.0-flash-001.json",
    },
    "claude-opus-4.5": {
        "faith": f"{_REPO_ROOT}/synthway/check_metrics/evaluate_faithfullness/generation_evaluate_faithfullness_anthropic_claude-opus-4.5.json",
        "chem": f"{_REPO_ROOT}/synthway/check_metrics/generation_rag_eval_correct/anthropic_claude-opus-4.5.json",
    },
    "meta-llama_llama-4-maverick": {
        "faith": f"{_REPO_ROOT}/synthway/check_metrics/evaluate_faithfullness/generation_evaluate_faithfullness_meta-llama_llama-4-maverick.json",
        "chem": f"{_REPO_ROOT}/synthway/check_metrics/generation_rag_eval_correct/llama-4-maverick.json",
    },
    "nvidia_nemotron-3-nano-30b-a3b_free": {
        "faith": f"{_REPO_ROOT}/synthway/check_metrics/evaluate_faithfullness/generation_evaluate_faithfullness_nvidia_nemotron-3-nano-30b-a3b_free.json",
        "chem": f"{_REPO_ROOT}/synthway/check_metrics/generation_rag_eval_correct/nvidia_nemotron-3-nano-30b-a3b_free.json",
    },
}

OUT_DIR = Path(f"{_REPO_ROOT}/synthway/check_metrics/final_slot_selection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If True, fail when any route is missing in any model.
STRICT_MODE = True


# Helpers


def load_json_file(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_path(path: str) -> str:
    return str(Path(path)).replace("\\", "/")


def strip_json_suffix(filename: str) -> str:
    """
    Examples:
        1164.json_CD-MOF_2732.json -> 1164.json_CD-MOF_2732
        route_2.json -> route_2
    """
    if filename.endswith(".json"):
        return filename[:-5]
    return filename


def extract_route_idx(route_path: str) -> int:
    m = re.search(r"/route_(\d+)\.json$", normalize_path(route_path))
    if not m:
        raise ValueError(f"Cannot parse route index from path: {route_path}")
    return int(m.group(1))


def extract_proposal_key_from_route_path(route_path: str) -> str:
    """
    From:
      .../1164.json_CD-MOF_2732/route_2.json
    return:
      1164.json_CD-MOF_2732.json

    This makes it comparable to slot_assignments entries.
    """
    parent = Path(route_path).parent.name
    return f"{parent}.json"


def proposal_to_stem(proposal_path: str) -> str:
    """
    From:
      /.../json/1164.json_CD-MOF_2732.json
    return:
      1164.json_CD-MOF_2732
    """
    return strip_json_suffix(Path(proposal_path).name)


def is_close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


# Data preparation


def build_slot_lookup(slot_assignments: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """
    Returns:
        {
          proposal_filename.json: {
              "slot": "...",
              "proposal_index": 0/1/2,
              "proposal_path": "...",
              "proposal_stem": "...",
          },
          ...
        }
    """
    lookup: dict[str, dict[str, Any]] = {}
    for slot_name, proposals in slot_assignments.items():
        if len(proposals) != 3:
            raise ValueError(
                f"Slot {slot_name} must contain exactly 3 proposals, got {len(proposals)}"
            )
        for idx, proposal_path in enumerate(proposals, start=1):
            proposal_name = Path(proposal_path).name
            if proposal_name in lookup:
                raise ValueError(f"Duplicate proposal in slot_assignments: {proposal_name}")
            lookup[proposal_name] = {
                "slot": slot_name,
                "proposal_index": idx,
                "proposal_path": Path(proposal_path).relative_to(_REPO_ROOT).as_posix(),
                "proposal_stem": proposal_to_stem(proposal_path),
            }
    return lookup


def index_faith_file(faith_data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in faith_data:
        path = row.get("path")
        if not path:
            continue
        indexed[normalize_path(path)] = row
    return indexed


def index_chem_file(chem_data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in chem_data:
        path = row.get("path")
        if not path:
            continue
        indexed[normalize_path(path)] = row
    return indexed


def merge_model_records(
    model_name: str,
    faith_path: str,
    chem_path: str,
    slot_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    faith_data = load_json_file(faith_path)
    chem_data = load_json_file(chem_path)

    if not isinstance(faith_data, list):
        raise TypeError(f"{faith_path} must contain a JSON list")
    if not isinstance(chem_data, list):
        raise TypeError(f"{chem_path} must contain a JSON list")

    faith_idx = index_faith_file(faith_data)
    chem_idx = index_chem_file(chem_data)

    all_paths = sorted(set(faith_idx) | set(chem_idx))
    records: list[dict[str, Any]] = []
    missing_pairs: list[str] = []

    for route_path in all_paths:
        faith_row = faith_idx.get(route_path)
        chem_row = chem_idx.get(route_path)

        if faith_row is None or chem_row is None:
            missing_pairs.append(route_path)
            continue

        proposal_name = extract_proposal_key_from_route_path(route_path)
        if proposal_name not in slot_lookup:
            # Route not in the 10 selected slots x 3 proposals, ignore it.
            continue

        slot_meta = slot_lookup[proposal_name]

        chem_score = chem_row.get("score")
        faith_score = faith_row.get("faithfulness_score")
        answer_score = faith_row.get("answer_score")

        if chem_score is None or faith_score is None or answer_score is None:
            raise ValueError(
                f"Missing score field for {model_name} route {route_path}: "
                f"chem={chem_score}, faith={faith_score}, answer={answer_score}"
            )

        mean_score = (float(chem_score) + float(faith_score) + float(answer_score)) / 3.0

        records.append(
            {
                "model": model_name,
                "slot": slot_meta["slot"],
                "proposal_index": slot_meta["proposal_index"],
                "proposal_path": slot_meta["proposal_path"],
                "proposal_name": proposal_name,
                "proposal_stem": slot_meta["proposal_stem"],
                "route_path": route_path,
                "route_idx": extract_route_idx(route_path),
                "chem_score": float(chem_score),
                "faith_score": float(faith_score),
                "answer_score": float(answer_score),
                "mean_score": mean_score,
                "chem_row": chem_row,
                "faith_row": faith_row,
            }
        )

    if missing_pairs and STRICT_MODE:
        preview = "\n".join(missing_pairs[:10])
        raise ValueError(
            f"[{model_name}] Some routes exist only in one of the two files.\n"
            f"Examples:\n{preview}\n"
            f"Total mismatched routes: {len(missing_pairs)}"
        )

    return records


# Selection logic


def choose_best_route_per_model_and_proposal(
    merged_records: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """
    Key:
      (model, slot, proposal_index)

    Value:
      best route record among 3 routes of this proposal for this model.
    """
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    for rec in merged_records:
        key = (rec["model"], rec["slot"], rec["proposal_index"])
        grouped.setdefault(key, []).append(rec)

    best: dict[tuple[str, str, int], dict[str, Any]] = {}

    for key, items in grouped.items():
        # tie-breakers:
        # 1) higher mean_score
        # 2) higher chem_score
        # 3) higher faith_score
        # 4) higher answer_score
        # 5) lower route_idx
        best_item = sorted(
            items,
            key=lambda x: (
                -x["mean_score"],
                -x["chem_score"],
                -x["faith_score"],
                -x["answer_score"],
                x["route_idx"],
            ),
        )[0]
        best[key] = best_item

    return best


def choose_common_proposal_per_slot(
    best_routes: dict[tuple[str, str, int], dict[str, Any]],
    model_names: list[str],
    slot_assignments: dict[str, list[str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns:
      slot_winners:
        {
          slot_name: {
            "slot": ...,
            "winning_proposal_index": 1/2/3,
            "winning_proposal_path": ...,
            "winning_proposal_name": ...,
            "proposal_avg_score": ...,
            "per_model_best_routes": {model_name: best_route_record}
          }
        }

      proposal_summary_rows:
        one row per slot x proposal, useful for csv/json export.
    """
    slot_winners: dict[str, dict[str, Any]] = {}
    proposal_summary_rows: list[dict[str, Any]] = []

    for slot_name, proposals in slot_assignments.items():
        candidate_rows: list[dict[str, Any]] = []

        for proposal_index, proposal_path in enumerate(proposals, start=1):
            per_model_best: dict[str, dict[str, Any]] = {}
            missing_models: list[str] = []

            for model_name in model_names:
                key = (model_name, slot_name, proposal_index)
                item = best_routes.get(key)
                if item is None:
                    missing_models.append(model_name)
                else:
                    per_model_best[model_name] = item

            if missing_models:
                msg = (
                    f"Missing best route(s) for slot={slot_name}, proposal_index={proposal_index}, "
                    f"proposal={proposal_path}, models={missing_models}"
                )
                if STRICT_MODE:
                    raise ValueError(msg)
                else:
                    print(f"WARNING: {msg}")
                    continue

            avg_score = sum(x["mean_score"] for x in per_model_best.values()) / len(per_model_best)

            row = {
                "slot": slot_name,
                "proposal_index": proposal_index,
                "proposal_path": Path(proposal_path).relative_to(_REPO_ROOT).as_posix(),
                "proposal_name": Path(proposal_path).name,
                "proposal_avg_score": avg_score,
                "model_scores": {m: per_model_best[m]["mean_score"] for m in model_names},
                "per_model_best_routes": per_model_best,
            }
            candidate_rows.append(row)
            proposal_summary_rows.append(row)

        if not candidate_rows:
            raise ValueError(f"No valid proposal candidates for slot {slot_name}")

        winner = sorted(
            candidate_rows,
            key=lambda x: (
                -x["proposal_avg_score"],
                # tie-breakers:
                -sum(x["model_scores"].values()),
                x["proposal_index"],
            ),
        )[0]

        slot_winners[slot_name] = {
            "slot": slot_name,
            "winning_proposal_index": winner["proposal_index"],
            "winning_proposal_path": winner["proposal_path"],
            "winning_proposal_name": winner["proposal_name"],
            "proposal_avg_score": winner["proposal_avg_score"],
            "model_scores": winner["model_scores"],
            "per_model_best_routes": winner["per_model_best_routes"],
        }

    return slot_winners, proposal_summary_rows


def build_final_rows(
    slot_winners: dict[str, dict[str, Any]],
    model_names: list[str],
) -> list[dict[str, Any]]:
    """
    Final output: 10 slots x 4 models = 40 rows
    """
    rows: list[dict[str, Any]] = []

    for slot_name, info in slot_winners.items():
        for model_name in model_names:
            rec = info["per_model_best_routes"][model_name]
            rows.append(
                {
                    "slot": slot_name,
                    "model": model_name,
                    "selected_proposal_index": info["winning_proposal_index"],
                    "selected_proposal_name": info["winning_proposal_name"],
                    "selected_proposal_path": info["winning_proposal_path"],
                    "proposal_avg_score_across_models": round(info["proposal_avg_score"], 6),
                    "route_idx": rec["route_idx"],
                    "route_path": rec["route_path"],
                    "mean_score": round(rec["mean_score"], 6),
                    "chem_score": rec["chem_score"],
                    "faith_score": rec["faith_score"],
                    "answer_score": rec["answer_score"],
                }
            )

    rows.sort(key=lambda x: (x["slot"], x["model"]))
    return rows


# Export


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write into {path}")

    import csv

    # flatten nested dicts if needed
    flat_rows: list[dict[str, Any]] = []
    all_keys = set()

    for row in rows:
        flat: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flat[f"{k}.{subk}"] = subv
            else:
                flat[k] = v
        flat_rows.append(flat)
        all_keys.update(flat.keys())

    fieldnames = sorted(all_keys)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    slot_lookup = build_slot_lookup(SLOT_ASSIGNMENTS)
    model_names = list(MODEL_FILES.keys())

    all_records: list[dict[str, Any]] = []

    for model_name, paths in MODEL_FILES.items():
        merged = merge_model_records(
            model_name=model_name,
            faith_path=paths["faith"],
            chem_path=paths["chem"],
            slot_lookup=slot_lookup,
        )
        all_records.extend(merged)

    # Sanity check:
    # expected = 10 slots * 3 proposals * 3 routes * 4 models = 360 merged route records
    expected_records = len(SLOT_ASSIGNMENTS) * 3 * 3 * len(model_names)
    if len(all_records) != expected_records:
        msg = f"Expected {expected_records} merged records, got {len(all_records)}"
        if STRICT_MODE:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")

    best_routes = choose_best_route_per_model_and_proposal(all_records)

    # Sanity check:
    # expected = 10 slots * 3 proposals * 4 models = 120 best-of-3 records
    expected_best = len(SLOT_ASSIGNMENTS) * 3 * len(model_names)
    if len(best_routes) != expected_best:
        msg = f"Expected {expected_best} best-route records, got {len(best_routes)}"
        if STRICT_MODE:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")

    slot_winners, proposal_summary_rows = choose_common_proposal_per_slot(
        best_routes=best_routes,
        model_names=model_names,
        slot_assignments=SLOT_ASSIGNMENTS,
    )

    final_rows = build_final_rows(slot_winners, model_names)

    summary = {
        "n_slots": len(SLOT_ASSIGNMENTS),
        "n_models": len(model_names),
        "n_final_rows": len(final_rows),
        "score_formula": "mean_score = (chem_score + faithfulness_score + answer_score) / 3",
        "selection_logic": {
            "step_1": "For each model and each proposal, choose best route among 3 by max mean_score",
            "step_2": "For each slot and each proposal, average best-route mean_score across models",
            "step_3": "For each slot, choose proposal with max averaged score",
            "step_4": "Fix chosen proposal for all models",
            "step_5": "For each model, keep its best route inside the chosen proposal",
        },
        "slot_winners": {
            slot: {
                "winning_proposal_index": info["winning_proposal_index"],
                "winning_proposal_name": info["winning_proposal_name"],
                "winning_proposal_path": info["winning_proposal_path"],
                "proposal_avg_score": info["proposal_avg_score"],
                "model_scores": info["model_scores"],
            }
            for slot, info in slot_winners.items()
        },
    }

    write_json(OUT_DIR / "all_merged_route_records.json", all_records)
    write_csv(OUT_DIR / "all_merged_route_records.csv", all_records)

    write_json(
        OUT_DIR / "best_route_per_model_per_proposal.json",
        {
            f"{model}__{slot}__proposal_{proposal_idx}": rec
            for (model, slot, proposal_idx), rec in best_routes.items()
        },
    )

    proposal_rows_export = []
    for row in proposal_summary_rows:
        proposal_rows_export.append(
            {
                "slot": row["slot"],
                "proposal_index": row["proposal_index"],
                "proposal_name": row["proposal_name"],
                "proposal_path": row["proposal_path"],
                "proposal_avg_score": round(row["proposal_avg_score"], 6),
                **{f"model_score__{m}": round(row["model_scores"][m], 6) for m in model_names},
            }
        )
    write_json(OUT_DIR / "proposal_scores_by_slot.json", proposal_rows_export)
    write_csv(OUT_DIR / "proposal_scores_by_slot.csv", proposal_rows_export)

    write_json(OUT_DIR / "selected_slot_winners_summary.json", summary)
    write_json(OUT_DIR / "final_selected_40_routes.json", final_rows)
    write_csv(OUT_DIR / "final_selected_40_routes.csv", final_rows)

    print("=" * 80)
    print("Done.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Final selected routes: {len(final_rows)}")
    print("=" * 80)

    for slot_name, info in sorted(slot_winners.items()):
        print(
            f"{slot_name}: proposal #{info['winning_proposal_index']} | "
            f"{info['winning_proposal_name']} | "
            f"avg_score={info['proposal_avg_score']:.4f}"
        )


if __name__ == "__main__":
    main()
