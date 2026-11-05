r"""Summarize generation and judge usage by model or route.

--table-mode models writes one row per model; routes writes one row per route.
Usage fields are input/output tokens, seconds and USD cost.
Generation usage is averaged per case (n_cases). Judge scores, pass rates and
usage are averaged per route over evaluations with a score and no error.
Derived judge usage per case equals per_route * routes_per_case, where
routes_per_case = n_routes / n_cases. Judge totals are reported at both levels.

--routes-root accepts a run directory or its parent and discovers cases recursively.
--chem and --answer-faith accept files or directories. A --chem directory supplies
all *.json; an --answer-faith directory supplies generation_evaluate_faithfullness_*.json.
In routes mode, --target accepts a route file, llm_usage.json or a case directory.

Examples from the repository root:
    python synthway/summarize_usage_for_table.py --table-mode models \
        --routes-root synthway/generation_routes \
        --answer-faith path/to/faithfulness.json --chem path/to/chemical.json \
        --csv-out outputs/usage_models.csv

    python synthway/summarize_usage_for_table.py --table-mode routes \
        --target path/to/case/route_1.json \
        --answer-faith path/to/faithfulness.json --chem path/to/chemical.json \
        --csv-out outputs/usage_routes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

# Utilities


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        value = float(x)
        return value if math.isfinite(value) else None
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _sum_optional(*vals: float | None) -> float | None:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return float(sum(nums)) if nums else None


def _sum_optional_int(*vals: int | None) -> int | None:
    nums = [v for v in vals if isinstance(v, int)]
    return int(sum(nums)) if nums else None


def _mean_optional(vals: list[float | None]) -> float | None:
    nums = [v for v in vals if isinstance(v, (int, float))]
    return float(sum(nums)) / len(nums) if nums else None


def _mean_optional_int(vals: list[int | None]) -> float | None:
    nums = [v for v in vals if isinstance(v, int)]
    return float(sum(nums)) / len(nums) if nums else None


def _norm_tail_key(p: str, keep: int = 2) -> str:
    """Normalize paths for fuzzy matching across machines."""
    parts = [x for x in p.replace("\\", "/").split("/") if x]
    if len(parts) >= keep:
        return "/".join(parts[-keep:])
    return "/".join(parts)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# Helper to round floats for table output
def _round_for_table(x: Any, ndigits: int = 3) -> Any:
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return round(x, ndigits)
    return x


def load_eval_records(path: Path, pattern: str) -> list[dict[str, Any]]:
    """Load eval records from a JSON list file OR from a directory of JSON list files."""
    path = path.expanduser().resolve()

    if path.is_file():
        recs = _read_json(path)
        if not isinstance(recs, list):
            raise ValueError(f"{path} must be a JSON array of records")
        return recs

    if path.is_dir():
        all_recs: list[dict[str, Any]] = []
        for p in sorted(path.glob(pattern)):
            if not p.is_file() or p.suffix.lower() != ".json":
                continue
            recs = _read_json(p)
            if not isinstance(recs, list):
                raise ValueError(f"{p} must be a JSON array of records")
            all_recs.extend(recs)
        return all_recs

    raise SystemExit(f"Not found: {path}")


# Data model


@dataclass
class Row:
    # Identity
    route_file: str
    route_dir: str
    composition: str | None
    model: str | None

    # Generation timing (from timing.json if present)
    gen_llm_call_time_s: float | None  # pure LLM call time
    gen_case_total_time_s: float | None  # end-to-end case time (includes parsing, retries, etc.)
    gen_parsed_routes: int | None

    # Generation (main LLM)
    gen_prompt_tokens: int | None
    gen_completion_tokens: int | None
    gen_total_tokens: int | None
    gen_cost_usd: float | None

    # AnswerRelevancy judge
    ans_judge_prompt_tokens: int | None
    ans_judge_completion_tokens: int | None
    ans_judge_total_tokens: int | None
    ans_judge_cost_usd: float | None
    ans_llm_time_s: float | None

    # Faithfulness judge
    faith_judge_prompt_tokens: int | None
    faith_judge_completion_tokens: int | None
    faith_judge_total_tokens: int | None
    faith_judge_cost_usd: float | None
    faith_llm_time_s: float | None

    # Chemical / RAG judge
    chem_judge_prompt_tokens: int | None
    chem_judge_completion_tokens: int | None
    chem_judge_total_tokens: int | None
    chem_judge_cost_usd: float | None
    chem_judge_llm_time_s: float | None

    # Times
    eval_total_time_s: float | None  # judge eval time (as logged)
    gen_time_s: float | None  # generation time (timing.json preferred)
    total_time_s: float | None

    # Totals
    total_tokens_all: int | None
    total_cost_usd: float | None

    # Diagnostics
    has_error_answer_faith: bool
    has_error_chem: bool


@dataclass
class ModelAggregateRow:
    model: str

    # Counts
    n_cases: int
    n_routes: int
    routes_per_case: float | None

    # Generation (per CASE)
    gen_input_tokens_avg: float | None
    gen_output_tokens_avg: float | None
    gen_time_s_avg: float | None
    gen_llm_call_time_s_avg: float | None
    gen_cost_usd_avg: float | None

    # Chemical plausibility (per ROUTE)
    chem_input_tokens_avg: float | None
    chem_output_tokens_avg: float | None
    chem_time_s_avg: float | None
    chem_cost_usd_avg: float | None

    # Answer relevancy (per ROUTE)
    ans_input_tokens_avg: float | None
    ans_output_tokens_avg: float | None
    ans_time_s_avg: float | None
    ans_cost_usd_avg: float | None

    # Faithfulness (per ROUTE)
    faith_input_tokens_avg: float | None
    faith_output_tokens_avg: float | None
    faith_time_s_avg: float | None
    faith_cost_usd_avg: float | None

    # Judges (per CASE, rescaled from per-ROUTE using routes_per_case)
    chem_input_tokens_per_case: float | None
    chem_output_tokens_per_case: float | None
    chem_time_s_per_case: float | None
    chem_cost_usd_per_case: float | None

    ans_input_tokens_per_case: float | None
    ans_output_tokens_per_case: float | None
    ans_time_s_per_case: float | None
    ans_cost_usd_per_case: float | None

    faith_input_tokens_per_case: float | None
    faith_output_tokens_per_case: float | None
    faith_time_s_per_case: float | None
    faith_cost_usd_per_case: float | None

    # Totals per CASE (generation + all judges)
    judge_cost_usd_per_case: float | None
    judge_time_s_per_case: float | None
    total_cost_usd_per_case: float | None
    total_time_s_per_case: float | None

    # Totals (sanity-check)
    total_cost_usd_sum: float | None
    total_time_s_sum: float | None


def build_eval_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Build an index to locate records by:
    - full path
    - tail key ".../<folder>/<file>"
    """
    idx: dict[str, dict[str, Any]] = {}
    canonical = set()
    short_aliases: dict[str, dict[str, Any] | None] = {}
    for r in records:
        p = r.get("path")
        if not isinstance(p, str):
            continue
        key = _norm_tail_key(p, keep=3)
        if key in canonical:
            raise ValueError(
                f"Duplicate evaluation for {key}; select one source record explicitly. "
                "Do not combine reruns or different judges by directory order."
            )
        canonical.add(key)
        idx[p] = r
        idx[key] = r
        short = _norm_tail_key(p, keep=2)
        short_aliases[short] = None if short in short_aliases else r
    # A two-component suffix is unsafe when several generators share a case.
    idx.update({key: value for key, value in short_aliases.items() if value is not None})
    return idx


# Route discovery


def resolve_route_context(target: Path) -> tuple[Path, Path | None, Path | None, list[Path]]:
    """
    Returns:
      route_dir, llm_usage.json if found, timing.json if found, and list of route_*.json files
    target may be:
      - route_1.json
      - llm_usage.json
      - a case directory
    """
    if target.is_dir():
        route_dir = target
    else:
        route_dir = target.parent

    route_files = sorted(route_dir.glob("route_*.json"))

    usage_candidates = [
        route_dir / "llm_usage.json",
        route_dir / "llm_usage_route.json",
    ]
    usage_candidates += sorted(route_dir.glob("llm_usage*.json"))
    llm_usage = _first_existing(usage_candidates)

    timing_path = _first_existing([route_dir / "timing.json"])

    return route_dir, llm_usage, timing_path, route_files


def iter_case_dirs(routes_root: Path) -> Iterable[Path]:
    """Recursive case discovery (works for generation_routes parent folder too)."""
    routes_root = routes_root.expanduser().resolve()
    seen = set()
    for p in routes_root.rglob("*"):
        if not p.is_dir():
            continue
        if (
            (p / "llm_usage.json").exists()
            or (p / "timing.json").exists()
            or any(p.glob("route_*.json"))
        ):
            if p not in seen:
                seen.add(p)
                yield p


# Extraction logic


def extract_generation(llm_usage_path: Path | None) -> dict[str, Any]:
    if not llm_usage_path or not llm_usage_path.exists():
        return {
            "model": None,
            "gen_prompt_tokens": None,
            "gen_completion_tokens": None,
            "gen_total_tokens": None,
            "gen_cost_usd": None,
            "gen_time_s": None,
        }

    u = _read_json(llm_usage_path)

    model = u.get("model")
    gen_prompt_tokens = _safe_int(u.get("prompt_tokens"))
    gen_completion_tokens = _safe_int(u.get("completion_tokens"))
    gen_total_tokens = _safe_int(u.get("total_tokens"))
    gen_cost_usd = _safe_float(u.get("cost_usd"))

    gen_time_s = _safe_float(
        _first_not_none(u.get("time_s"), u.get("total_time_s"), u.get("elapsed_s"))
    )

    return {
        "model": model,
        "gen_prompt_tokens": gen_prompt_tokens,
        "gen_completion_tokens": gen_completion_tokens,
        "gen_total_tokens": gen_total_tokens,
        "gen_cost_usd": gen_cost_usd,
        "gen_time_s": gen_time_s,
    }


def extract_timing(timing_path: Path | None) -> dict[str, Any]:
    if not timing_path or not timing_path.exists():
        return {
            "gen_llm_call_time_s": None,
            "gen_case_total_time_s": None,
            "gen_parsed_routes": None,
        }

    t = _read_json(timing_path)
    return {
        "gen_llm_call_time_s": _safe_float(t.get("llm_call_time_s")),
        "gen_case_total_time_s": _safe_float(t.get("case_total_time_s")),
        "gen_parsed_routes": _safe_int(t.get("parsed_routes")),
    }


def pick_eval_record(
    eval_idx: dict[str, dict[str, Any]], route_file: Path
) -> dict[str, Any] | None:
    candidates = [
        str(route_file),
        _norm_tail_key(str(route_file), keep=2),
        _norm_tail_key(str(route_file), keep=3),
    ]
    for key in candidates:
        r = eval_idx.get(key)
        if r is not None:
            return r
    return None


def extract_answer_faith(rec: dict[str, Any] | None) -> dict[str, Any]:
    if not rec:
        return {
            "composition": None,
            "has_error_answer_faith": False,
            "ans_judge_prompt_tokens": None,
            "ans_judge_completion_tokens": None,
            "ans_judge_total_tokens": None,
            "ans_judge_cost_usd": None,
            "ans_llm_time_s": None,
            "ans_score": None,
            "ans_passed": None,
            "faith_judge_prompt_tokens": None,
            "faith_judge_completion_tokens": None,
            "faith_judge_total_tokens": None,
            "faith_judge_cost_usd": None,
            "faith_llm_time_s": None,
            "faith_score": None,
            "faith_passed": None,
            "eval_total_time_s": None,
        }

    has_error = bool(rec.get("error"))
    composition = rec.get("composition")

    ans_score = _safe_float(rec.get("answer_score"))
    ans_passed = rec.get("answer_passed") if isinstance(rec.get("answer_passed"), bool) else None

    faith_score = _safe_float(rec.get("faithfulness_score"))
    faith_passed = (
        rec.get("faithfulness_passed") if isinstance(rec.get("faithfulness_passed"), bool) else None
    )

    return {
        "composition": composition,
        "has_error_answer_faith": has_error,
        # Prefer totals across judge calls; fall back to per-attempt fields.
        "ans_judge_prompt_tokens": _safe_int(
            _first_not_none(
                rec.get("answer_judge_prompt_tokens_total"), rec.get("answer_judge_prompt_tokens")
            )
        ),
        "ans_judge_completion_tokens": _safe_int(
            _first_not_none(
                rec.get("answer_judge_completion_tokens_total"),
                rec.get("answer_judge_completion_tokens"),
            )
        ),
        "ans_judge_total_tokens": _safe_int(
            _first_not_none(
                rec.get("answer_judge_total_tokens_total"), rec.get("answer_judge_total_tokens")
            )
        ),
        "ans_judge_cost_usd": _safe_float(
            _first_not_none(
                rec.get("answer_judge_cost_usd_total"), rec.get("answer_judge_cost_usd")
            )
        ),
        "ans_llm_time_s": _safe_float(rec.get("answer_llm_time_s")),
        "ans_score": ans_score,
        "ans_passed": ans_passed,
        "faith_judge_prompt_tokens": _safe_int(
            _first_not_none(
                rec.get("faith_judge_prompt_tokens_total"), rec.get("faith_judge_prompt_tokens")
            )
        ),
        "faith_judge_completion_tokens": _safe_int(
            _first_not_none(
                rec.get("faith_judge_completion_tokens_total"),
                rec.get("faith_judge_completion_tokens"),
            )
        ),
        "faith_judge_total_tokens": _safe_int(
            _first_not_none(
                rec.get("faith_judge_total_tokens_total"), rec.get("faith_judge_total_tokens")
            )
        ),
        "faith_judge_cost_usd": _safe_float(
            _first_not_none(rec.get("faith_judge_cost_usd_total"), rec.get("faith_judge_cost_usd"))
        ),
        "faith_llm_time_s": _safe_float(rec.get("faith_llm_time_s")),
        "faith_score": faith_score,
        "faith_passed": faith_passed,
        "eval_total_time_s": _safe_float(rec.get("route_total_time_s")),
    }


def extract_chem(rec: dict[str, Any] | None) -> dict[str, Any]:
    if not rec:
        return {
            "has_error_chem": False,
            "chem_judge_prompt_tokens": None,
            "chem_judge_completion_tokens": None,
            "chem_judge_total_tokens": None,
            "chem_judge_cost_usd": None,
            "chem_judge_llm_time_s": None,
            "chem_eval_time_s": None,
            "chem_score": None,
            "chem_passed": None,
        }

    has_error = bool(rec.get("error"))

    chem_score = _safe_float(rec.get("score"))
    chem_passed = rec.get("passed") if isinstance(rec.get("passed"), bool) else None

    return {
        "has_error_chem": has_error,
        "chem_judge_prompt_tokens": _safe_int(
            _first_not_none(rec.get("judge_prompt_tokens_total"), rec.get("judge_prompt_tokens"))
        ),
        "chem_judge_completion_tokens": _safe_int(
            _first_not_none(
                rec.get("judge_completion_tokens_total"), rec.get("judge_completion_tokens")
            )
        ),
        "chem_judge_total_tokens": _safe_int(
            _first_not_none(rec.get("judge_total_tokens_total"), rec.get("judge_total_tokens"))
        ),
        "chem_judge_cost_usd": _safe_float(
            _first_not_none(rec.get("judge_cost_usd_total"), rec.get("judge_cost_usd"))
        ),
        "chem_judge_llm_time_s": _safe_float(rec.get("judge_llm_time_s")),
        "chem_eval_time_s": _safe_float(rec.get("route_total_time_s")),
        "chem_score": chem_score,
        "chem_passed": chem_passed,
    }


def build_row_for_route(
    route_file: Path,
    llm_usage_path: Path | None,
    timing_path: Path | None,
    ans_faith_idx: dict[str, dict[str, Any]] | None,
    chem_idx: dict[str, dict[str, Any]] | None,
    gen_time_override: float | None = None,
) -> Row:
    gen = extract_generation(llm_usage_path)
    t = extract_timing(timing_path)

    af_rec = pick_eval_record(ans_faith_idx, route_file) if ans_faith_idx else None
    af = extract_answer_faith(af_rec)

    chem_rec = pick_eval_record(chem_idx, route_file) if chem_idx else None
    ch = extract_chem(chem_rec)

    eval_total_time_s = _sum_optional(af.get("eval_total_time_s"), ch.get("chem_eval_time_s"))

    if gen_time_override is not None:
        gen_time_s = gen_time_override
    else:
        gen_time_s = _first_not_none(
            t.get("gen_case_total_time_s"), t.get("gen_llm_call_time_s"), gen.get("gen_time_s")
        )

    total_time_s = _sum_optional(gen_time_s, eval_total_time_s)

    total_tokens_all = _sum_optional_int(
        gen.get("gen_total_tokens"),
        af.get("ans_judge_total_tokens"),
        af.get("faith_judge_total_tokens"),
        ch.get("chem_judge_total_tokens"),
    )
    total_cost_usd = _sum_optional(
        gen.get("gen_cost_usd"),
        af.get("ans_judge_cost_usd"),
        af.get("faith_judge_cost_usd"),
        ch.get("chem_judge_cost_usd"),
    )

    return Row(
        route_file=str(route_file),
        route_dir=str(route_file.parent),
        composition=af.get("composition"),
        model=gen.get("model"),
        gen_llm_call_time_s=t.get("gen_llm_call_time_s"),
        gen_case_total_time_s=t.get("gen_case_total_time_s"),
        gen_parsed_routes=t.get("gen_parsed_routes"),
        gen_prompt_tokens=gen.get("gen_prompt_tokens"),
        gen_completion_tokens=gen.get("gen_completion_tokens"),
        gen_total_tokens=gen.get("gen_total_tokens"),
        gen_cost_usd=gen.get("gen_cost_usd"),
        ans_judge_prompt_tokens=af.get("ans_judge_prompt_tokens"),
        ans_judge_completion_tokens=af.get("ans_judge_completion_tokens"),
        ans_judge_total_tokens=af.get("ans_judge_total_tokens"),
        ans_judge_cost_usd=af.get("ans_judge_cost_usd"),
        ans_llm_time_s=af.get("ans_llm_time_s"),
        faith_judge_prompt_tokens=af.get("faith_judge_prompt_tokens"),
        faith_judge_completion_tokens=af.get("faith_judge_completion_tokens"),
        faith_judge_total_tokens=af.get("faith_judge_total_tokens"),
        faith_judge_cost_usd=af.get("faith_judge_cost_usd"),
        faith_llm_time_s=af.get("faith_llm_time_s"),
        chem_judge_prompt_tokens=ch.get("chem_judge_prompt_tokens"),
        chem_judge_completion_tokens=ch.get("chem_judge_completion_tokens"),
        chem_judge_total_tokens=ch.get("chem_judge_total_tokens"),
        chem_judge_cost_usd=ch.get("chem_judge_cost_usd"),
        chem_judge_llm_time_s=ch.get("chem_judge_llm_time_s"),
        eval_total_time_s=eval_total_time_s,
        gen_time_s=gen_time_s,
        total_time_s=total_time_s,
        total_tokens_all=total_tokens_all,
        total_cost_usd=total_cost_usd,
        has_error_answer_faith=bool(af.get("has_error_answer_faith")),
        has_error_chem=bool(ch.get("has_error_chem")),
    )


def aggregate_models_paper(
    routes_root: Path,
    ans_faith_idx: dict[str, dict[str, Any]] | None,
    chem_idx: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return paper-table rows as dicts (only required columns)."""

    def _is_valid_chem_record(ch: dict[str, Any]) -> bool:
        return (not bool(ch.get("has_error_chem"))) and (ch.get("chem_score") is not None)

    def _is_valid_answer_record(af: dict[str, Any]) -> bool:
        return (not bool(af.get("has_error_answer_faith"))) and (af.get("ans_score") is not None)

    def _is_valid_faith_record(af: dict[str, Any]) -> bool:
        return (not bool(af.get("has_error_answer_faith"))) and (af.get("faith_score") is not None)

    def _mean_numeric(vals: list[Any]) -> float | None:
        nums: list[float] = []
        for v in vals:
            vv = _safe_float(v)
            if isinstance(vv, (int, float)):
                nums.append(float(vv))
        if not nums:
            return None
        return float(sum(nums)) / float(len(nums))

    buf: dict[str, dict[str, Any]] = {}

    def _bucket(model_name: str) -> dict[str, Any]:
        if model_name not in buf:
            buf[model_name] = {
                "n_cases": 0,
                "n_routes": 0,
                # Generation per CASE
                "gen_prompt": [],
                "gen_comp": [],
                "gen_time": [],
                "gen_cost": [],
                # Judges per ROUTE (usage)
                "chem_prompt": [],
                "chem_comp": [],
                "chem_time": [],
                "chem_cost": [],
                "ans_prompt": [],
                "ans_comp": [],
                "ans_time": [],
                "ans_cost": [],
                "faith_prompt": [],
                "faith_comp": [],
                "faith_time": [],
                "faith_cost": [],
                # Judges per ROUTE (pass/score)
                "chem_pass": [],
                "chem_score": [],
                "ans_pass": [],
                "ans_score": [],
                "faith_pass": [],
                "faith_score": [],
                # Valid judge counts (after filtering out errored / score-missing evaluations)
                "chem_valid_routes": 0,
                "ans_valid_routes": 0,
                "faith_valid_routes": 0,
            }
        return buf[model_name]

    # Walk cases
    for case_dir in iter_case_dirs(routes_root):
        _, llm_usage_path, timing_path, route_files = resolve_route_context(case_dir)
        if not route_files:
            # Skip directories without route files.
            continue

        gen = extract_generation(llm_usage_path)
        t = extract_timing(timing_path)

        model_name = gen.get("model") or "unknown"
        b = _bucket(model_name)

        # Generation per CASE
        b["n_cases"] += 1
        b["gen_prompt"].append(gen.get("gen_prompt_tokens"))
        b["gen_comp"].append(gen.get("gen_completion_tokens"))
        b["gen_cost"].append(gen.get("gen_cost_usd"))

        gen_time_s = _first_not_none(
            t.get("gen_case_total_time_s"), t.get("gen_llm_call_time_s"), gen.get("gen_time_s")
        )
        b["gen_time"].append(gen_time_s)

        # Judges per ROUTE
        for rf in route_files:
            b["n_routes"] += 1

            if chem_idx is not None:
                ch_rec = pick_eval_record(chem_idx, rf)
                ch = extract_chem(ch_rec)
                if _is_valid_chem_record(ch):
                    b["chem_valid_routes"] += 1
                    b["chem_prompt"].append(ch.get("chem_judge_prompt_tokens"))
                    b["chem_comp"].append(ch.get("chem_judge_completion_tokens"))
                    chem_time_value = ch.get("chem_judge_llm_time_s")
                    if chem_time_value is None:
                        chem_time_value = ch.get("chem_eval_time_s")
                    b["chem_time"].append(chem_time_value)
                    b["chem_cost"].append(ch.get("chem_judge_cost_usd"))
                    if ch.get("chem_passed") is not None:
                        b["chem_pass"].append(1.0 if ch.get("chem_passed") else 0.0)
                    b["chem_score"].append(float(ch.get("chem_score")))

            if ans_faith_idx is not None:
                af_rec = pick_eval_record(ans_faith_idx, rf)
                af = extract_answer_faith(af_rec)

                if _is_valid_answer_record(af):
                    b["ans_valid_routes"] += 1
                    b["ans_prompt"].append(af.get("ans_judge_prompt_tokens"))
                    b["ans_comp"].append(af.get("ans_judge_completion_tokens"))
                    ans_time_value = af.get("ans_llm_time_s")
                    if ans_time_value is None:
                        ans_time_value = af.get("eval_total_time_s")
                    b["ans_time"].append(ans_time_value)
                    b["ans_cost"].append(af.get("ans_judge_cost_usd"))
                    if af.get("ans_passed") is not None:
                        b["ans_pass"].append(1.0 if af.get("ans_passed") else 0.0)
                    b["ans_score"].append(float(af.get("ans_score")))

                if _is_valid_faith_record(af):
                    b["faith_valid_routes"] += 1
                    b["faith_prompt"].append(af.get("faith_judge_prompt_tokens"))
                    b["faith_comp"].append(af.get("faith_judge_completion_tokens"))
                    faith_time_value = af.get("faith_llm_time_s")
                    if faith_time_value is None:
                        faith_time_value = af.get("eval_total_time_s")
                    b["faith_time"].append(faith_time_value)
                    b["faith_cost"].append(af.get("faith_judge_cost_usd"))
                    if af.get("faith_passed") is not None:
                        b["faith_pass"].append(1.0 if af.get("faith_passed") else 0.0)
                    b["faith_score"].append(float(af.get("faith_score")))

    # Build output rows
    rows: list[dict[str, Any]] = []

    for model_name, b in sorted(buf.items(), key=lambda kv: kv[0]):
        n_cases = int(b["n_cases"])
        n_routes = int(b["n_routes"])
        routes_per_case = (float(n_routes) / float(n_cases)) if n_cases else None

        # Generation per CASE
        gen_input_tokens_case = _mean_optional_int(b["gen_prompt"])
        gen_output_tokens_case = _mean_optional_int(b["gen_comp"])
        gen_time_s_case = _mean_optional([_safe_float(x) for x in b["gen_time"]])
        gen_cost_usd_case = _mean_optional([_safe_float(x) for x in b["gen_cost"]])

        # Judges per ROUTE: pass/score
        chem_pass_rate_route = _mean_optional([_safe_float(x) for x in b["chem_pass"]])
        chem_score_route = _mean_optional([_safe_float(x) for x in b["chem_score"]])
        ans_pass_rate_route = _mean_optional([_safe_float(x) for x in b["ans_pass"]])
        ans_score_route = _mean_optional([_safe_float(x) for x in b["ans_score"]])
        faith_pass_rate_route = _mean_optional([_safe_float(x) for x in b["faith_pass"]])
        faith_score_route = _mean_optional([_safe_float(x) for x in b["faith_score"]])

        # Judges per ROUTE: usage (only valid evaluations: no error and score present)
        chem_input_tokens_route = _mean_optional_int(b["chem_prompt"])
        chem_output_tokens_route = _mean_optional_int(b["chem_comp"])
        chem_time_s_route = _mean_numeric(b["chem_time"])
        chem_cost_usd_route = _mean_numeric(b["chem_cost"])

        ans_input_tokens_route = _mean_optional_int(b["ans_prompt"])
        ans_output_tokens_route = _mean_optional_int(b["ans_comp"])
        ans_time_s_route = _mean_numeric(b["ans_time"])
        ans_cost_usd_route = _mean_numeric(b["ans_cost"])

        faith_input_tokens_route = _mean_optional_int(b["faith_prompt"])
        faith_output_tokens_route = _mean_optional_int(b["faith_comp"])
        faith_time_s_route = _mean_numeric(b["faith_time"])
        faith_cost_usd_route = _mean_numeric(b["faith_cost"])
        judges_cost_usd_route_total = _sum_optional(
            chem_cost_usd_route, ans_cost_usd_route, faith_cost_usd_route
        )
        judges_time_s_route_total = _sum_optional(
            chem_time_s_route, ans_time_s_route, faith_time_s_route
        )

        # Derived judges per CASE usage/time/cost from per-ROUTE means
        chem_input_tokens_case = (
            (chem_input_tokens_route * routes_per_case)
            if (chem_input_tokens_route is not None and routes_per_case is not None)
            else None
        )
        chem_output_tokens_case = (
            (chem_output_tokens_route * routes_per_case)
            if (chem_output_tokens_route is not None and routes_per_case is not None)
            else None
        )
        chem_time_s_case = (
            (chem_time_s_route * routes_per_case)
            if (chem_time_s_route is not None and routes_per_case is not None)
            else None
        )
        chem_cost_usd_case = (
            (chem_cost_usd_route * routes_per_case)
            if (chem_cost_usd_route is not None and routes_per_case is not None)
            else None
        )

        ans_input_tokens_case = (
            (ans_input_tokens_route * routes_per_case)
            if (ans_input_tokens_route is not None and routes_per_case is not None)
            else None
        )
        ans_output_tokens_case = (
            (ans_output_tokens_route * routes_per_case)
            if (ans_output_tokens_route is not None and routes_per_case is not None)
            else None
        )
        ans_time_s_case = (
            (ans_time_s_route * routes_per_case)
            if (ans_time_s_route is not None and routes_per_case is not None)
            else None
        )
        ans_cost_usd_case = (
            (ans_cost_usd_route * routes_per_case)
            if (ans_cost_usd_route is not None and routes_per_case is not None)
            else None
        )

        faith_input_tokens_case = (
            (faith_input_tokens_route * routes_per_case)
            if (faith_input_tokens_route is not None and routes_per_case is not None)
            else None
        )
        faith_output_tokens_case = (
            (faith_output_tokens_route * routes_per_case)
            if (faith_output_tokens_route is not None and routes_per_case is not None)
            else None
        )
        faith_time_s_case = (
            (faith_time_s_route * routes_per_case)
            if (faith_time_s_route is not None and routes_per_case is not None)
            else None
        )
        faith_cost_usd_case = (
            (faith_cost_usd_route * routes_per_case)
            if (faith_cost_usd_route is not None and routes_per_case is not None)
            else None
        )

        judges_cost_usd_case_total = (
            (judges_cost_usd_route_total * routes_per_case)
            if (judges_cost_usd_route_total is not None and routes_per_case is not None)
            else None
        )
        judges_time_s_case_total = (
            (judges_time_s_route_total * routes_per_case)
            if (judges_time_s_route_total is not None and routes_per_case is not None)
            else None
        )

        rows.append(
            {
                "model": model_name,
                "n_cases": n_cases,
                "n_routes": n_routes,
                "routes_per_case": routes_per_case,
                "chem_valid_routes": int(b["chem_valid_routes"]),
                "ans_valid_routes": int(b["ans_valid_routes"]),
                "faith_valid_routes": int(b["faith_valid_routes"]),
                "chem_valid_route_share": (float(b["chem_valid_routes"]) / float(n_routes))
                if n_routes
                else None,
                "ans_valid_route_share": (float(b["ans_valid_routes"]) / float(n_routes))
                if n_routes
                else None,
                "faith_valid_route_share": (float(b["faith_valid_routes"]) / float(n_routes))
                if n_routes
                else None,
                # Generation (per CASE)
                "gen_input_tokens_case": gen_input_tokens_case,
                "gen_output_tokens_case": gen_output_tokens_case,
                "gen_time_s_case": gen_time_s_case,
                "gen_cost_usd_case": gen_cost_usd_case,
                # Chemical (per ROUTE)
                "chem_pass_rate_route": chem_pass_rate_route,
                "chem_score_route": chem_score_route,
                "chem_input_tokens_route": chem_input_tokens_route,
                "chem_output_tokens_route": chem_output_tokens_route,
                "chem_time_s_route": chem_time_s_route,
                "chem_cost_usd_route": chem_cost_usd_route,
                # Chemical (derived per CASE)
                "chem_input_tokens_case": chem_input_tokens_case,
                "chem_output_tokens_case": chem_output_tokens_case,
                "chem_time_s_case": chem_time_s_case,
                "chem_cost_usd_case": chem_cost_usd_case,
                # Answer relevancy (per ROUTE)
                "ans_pass_rate_route": ans_pass_rate_route,
                "ans_score_route": ans_score_route,
                "ans_input_tokens_route": ans_input_tokens_route,
                "ans_output_tokens_route": ans_output_tokens_route,
                "ans_time_s_route": ans_time_s_route,
                "ans_cost_usd_route": ans_cost_usd_route,
                # Answer relevancy (derived per CASE)
                "ans_input_tokens_case": ans_input_tokens_case,
                "ans_output_tokens_case": ans_output_tokens_case,
                "ans_time_s_case": ans_time_s_case,
                "ans_cost_usd_case": ans_cost_usd_case,
                # Faithfulness (per ROUTE)
                "faith_pass_rate_route": faith_pass_rate_route,
                "faith_score_route": faith_score_route,
                "faith_input_tokens_route": faith_input_tokens_route,
                "faith_output_tokens_route": faith_output_tokens_route,
                "faith_time_s_route": faith_time_s_route,
                "faith_cost_usd_route": faith_cost_usd_route,
                # Faithfulness (derived per CASE)
                "faith_input_tokens_case": faith_input_tokens_case,
                "faith_output_tokens_case": faith_output_tokens_case,
                "faith_time_s_case": faith_time_s_case,
                "faith_cost_usd_case": faith_cost_usd_case,
                # Totals across judges (per ROUTE)
                "judges_cost_usd_route_total": judges_cost_usd_route_total,
                "judges_time_s_route_total": judges_time_s_route_total,
                # Totals across judges (derived per CASE)
                "judges_cost_usd_case_total": judges_cost_usd_case_total,
                "judges_time_s_case_total": judges_time_s_case_total,
            }
        )

    return rows


# CSV writers


def write_csv_dataclasses(rows: list[Any], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _round_for_table(v) for k, v in asdict(r).items()})


def write_csv_dicts(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: _round_for_table(v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--table-mode",
        type=str,
        default="models",
        choices=["models", "routes"],
        help="'models' => one row per model (paper table); 'routes' => one row per route file",
    )

    ap.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to route_*.json, llm_usage.json, or a case directory (one route; table-mode routes only)",
    )
    ap.add_argument(
        "--routes-root",
        type=str,
        default=None,
        help="Path to output_LLM_... or its parent generation_routes directory (batch; required for table-mode models)",
    )
    ap.add_argument(
        "--answer-faith",
        type=str,
        default=None,
        help="Path to generation_evaluate_faithfullness_*.json or a directory containing these files",
    )
    ap.add_argument(
        "--chem",
        type=str,
        default=None,
        help="Path to generation_rag_eval_*.json or the generation_rag_eval_correct directory",
    )
    ap.add_argument("--csv-out", type=str, default=None, help="Write CSV if specified")
    ap.add_argument(
        "--gen-time-override",
        type=float,
        default=None,
        help="Manually specify seconds if generation time was not logged (table-mode routes)",
    )

    args = ap.parse_args()

    if args.table_mode == "models" and not args.routes_root:
        raise SystemExit("--table-mode models requires --routes-root")

    if args.table_mode == "routes" and not args.target and not args.routes_root:
        raise SystemExit("--table-mode routes requires --target or --routes-root")

    # Build indexes
    ans_faith_idx = None
    if args.answer_faith:
        af_records = load_eval_records(
            Path(args.answer_faith), "generation_evaluate_faithfullness_*.json"
        )
        ans_faith_idx = build_eval_index(af_records)

    chem_idx = None
    if args.chem:
        ch_records = load_eval_records(Path(args.chem), "*.json")
        chem_idx = build_eval_index(ch_records)

    # MODELS (paper table)
    if args.table_mode == "models":
        routes_root = Path(args.routes_root).expanduser().resolve()
        if not routes_root.exists():
            raise SystemExit(f"--routes-root does not exist: {routes_root}")

        paper_rows = aggregate_models_paper(routes_root, ans_faith_idx, chem_idx)
        if not paper_rows:
            raise SystemExit("Nothing to write: paper_rows is empty")

        if args.csv_out:
            write_csv_dicts(paper_rows, Path(args.csv_out))
            print(f"[OK] CSV written: {args.csv_out}")

        rounded_paper_rows = [
            {k: _round_for_table(v) for k, v in row.items()} for row in paper_rows
        ]
        print(json.dumps(rounded_paper_rows, ensure_ascii=False, indent=2))
        return

    # ROUTES (detailed)
    rows: list[Row] = []

    if args.target:
        target = Path(args.target).expanduser().resolve()
        route_dir, llm_usage_path, timing_path, route_files = resolve_route_context(target)

        if target.is_file() and target.name.startswith("route_") and target.suffix == ".json":
            route_files = [target]

        if not route_files:
            raise SystemExit(f"No route_*.json files found in {route_dir}")

        for rf in route_files:
            rows.append(
                build_row_for_route(
                    route_file=rf,
                    llm_usage_path=llm_usage_path,
                    timing_path=timing_path,
                    ans_faith_idx=ans_faith_idx,
                    chem_idx=chem_idx,
                    gen_time_override=args.gen_time_override,
                )
            )

    else:
        routes_root = Path(args.routes_root).expanduser().resolve()
        if not routes_root.exists():
            raise SystemExit(f"--routes-root does not exist: {routes_root}")

        for case_dir in iter_case_dirs(routes_root):
            _, llm_usage_path, timing_path, route_files = resolve_route_context(case_dir)
            for rf in route_files:
                rows.append(
                    build_row_for_route(
                        route_file=rf,
                        llm_usage_path=llm_usage_path,
                        timing_path=timing_path,
                        ans_faith_idx=ans_faith_idx,
                        chem_idx=chem_idx,
                        gen_time_override=args.gen_time_override,
                    )
                )

    if not rows:
        raise SystemExit("Nothing to write: rows is empty")

    if args.csv_out:
        write_csv_dataclasses(rows, Path(args.csv_out))
        print(f"[OK] CSV written: {args.csv_out}")

    rounded_rows = [{k: _round_for_table(v) for k, v in asdict(r).items()} for r in rows]
    print(json.dumps(rounded_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
