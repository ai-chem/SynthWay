from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
import json
import os
import re
import sys
import time
import shutil
import subprocess
import io
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import FreeSimpleGUI as sg

from deepeval.metrics import AnswerRelevancyMetric
from ss_faithfulness import SSFaithfulnessMetric, build_ss_judge_model
from deepeval.test_case import LLMTestCase

from gui_models import GUI_MODELS, DEFAULT_GENERATOR, DEFAULT_JUDGE, resolve_model_pair
from gui_settings import parse_run_settings, new_run_directory
from credentials import credential_guidance
from gui_results import (
    build_run_summary as _build_run_summary,
    route_overall_status as _route_overall_status,
    metric_line, score_text, validate_metric_result,
)

from predictions import (
    generate_synthesis_routes,
    AVAILABLE_MODELS,
    DEFAULT_RAG_INDEX_DIR,
    DEFAULT_LLM_OUTPUT_DIR,
    regenerate_single_route_with_feedback,
)

from evaluate_generation_deepeval import (
    build_test_case,
    make_chemical_geval_metric,
)


def _load_retrieval_context_json(case_dir: str) -> Tuple[List[str], Optional[str]]:
    """Load retrieval_context.json from a case directory.

    Returns: (retrieval_context_list, composition_from_meta)
    """
    rc_path = os.path.join(case_dir, "retrieval_context.json")
    if not os.path.isfile(rc_path):
        return [], None

    try:
        with open(rc_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return [], None

    rc = meta.get("retrieval_context") or []
    if isinstance(rc, list):
        retrieval_context = [str(x) for x in rc]
    elif rc:
        retrieval_context = [str(rc)]
    else:
        retrieval_context = []

    composition = meta.get("composition")
    return retrieval_context, composition


# --- Robust route file loading helpers ---
_ROUTE_FILE_RE = re.compile(r"^route_(\d+)\.json$", re.IGNORECASE)


def _split_combined_routes(raw_text: str) -> List[str]:
    """Some generators sometimes dump multiple routes into a single JSON.

    If we detect that format, we split it into per-route chunks so each chunk
    gets its own metric line + separator in the GUI.
    """
    txt = (raw_text or "").strip()
    if not txt:
        return []

    # Common formats we have seen:
    #   **Route 1: ...**
    #   Route 1: ...
    markers = [m.start() for m in re.finditer(r"(?m)^\*\*Route\s+\d+\s*:", txt)]
    if len(markers) <= 1:
        markers = [m.start() for m in re.finditer(r"(?m)^Route\s+\d+\s*:", txt)]

    if len(markers) <= 1:
        return [txt]

    parts: List[str] = []
    for idx, start in enumerate(markers):
        end = markers[idx + 1] if idx + 1 < len(markers) else len(txt)
        chunk = txt[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


def _load_routes_recursive(output_dir: str) -> List[Dict[str, Any]]:
    """Recursively find route_*.json under output_dir and load them.

    This is intentionally more robust than the non-recursive loader, because
    generation code may place routes inside nested case folders.
    """
    found: List[Dict[str, Any]] = []

    for root, _, files in os.walk(output_dir):
        for fn in files:
            m = _ROUTE_FILE_RE.match(fn)
            if not m:
                continue

            path = os.path.join(root, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                # Skip unreadable route files (don’t kill the whole run)
                continue

            raw_text = (
                data.get("raw_text")
                or data.get("route_text")
                or data.get("text")
                or ""
            )
            retrieval_text = data.get("retrieval_text") or ""

            found.append(
                {
                    "file": fn,
                    "path": path,
                    "composition": data.get("composition") or data.get("target_composition"),
                    "method": data.get("method") or data.get("synthesis_method"),
                    "raw_text": raw_text,
                    "retrieval_text": retrieval_text,
                    "_route_num": int(m.group(1)),
                }
            )

    found.sort(key=lambda r: (r.get("_route_num", 10**9), r.get("path", "")))
    for r in found:
        r.pop("_route_num", None)
    return found


def _build_rag_test_case(
    composition: str,
    raw_text: str,
    retrieval_text: str,
    retrieval_context: List[str],
) -> LLMTestCase:
    """Builds an LLMTestCase for AnswerRelevancy/Faithfulness metrics."""
    comp = (composition or "UNKNOWN").strip() or "UNKNOWN"
    actual_output = (raw_text or "").strip() or (retrieval_text or "").strip()
    return LLMTestCase(
        input=comp,
        actual_output=actual_output,
        retrieval_context=retrieval_context or [],
    )


def _shorten(text: str, max_chars: int = 320) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


def _fmt_bool(flag: bool) -> str:
    return "PASS" if flag else "FAIL"



def _open_folder(path: Optional[str]) -> None:
    """Open a folder in the OS file manager."""
    if not path:
        return
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        # Non-fatal UX helper
        return


# --------------------------
# Regeneration feedback and route-selection helpers
# --------------------------

@dataclass
class _MetricResult:
    score: Optional[float]
    threshold: float
    passed: bool
    reason: str


def _is_bad_parent_text(raw_text: str) -> bool:
    """Heuristic: reject empty/obviously broken parent inputs for regeneration."""
    t = (raw_text or "").strip()
    if not t:
        return True
    # Very short outputs are often truncated / formatting failures.
    if len(t) < 120:
        return True
    low = t.lower().strip()
    if low in {"none", "null", "n/a"}:
        return True
    return False


def _build_reviewer_feedback(
    *,
    chem: _MetricResult,
    ans: _MetricResult,
    faith: _MetricResult,
    composition: str,
    method: Optional[str],
    history: Optional[Dict[str, Any]] = None,
) -> str:
    """Turn metric failures into an actionable reviewer-style feedback string."""

    def _fmt_metric(name: str, m: _MetricResult) -> str:
        status = "PASS" if m.passed else "FAIL"
        score_txt = "None" if m.score is None else f"{m.score:.4f}"
        thr_txt = f"{m.threshold:.4f}"
        reason_txt = (m.reason or "").strip() or "(no reason provided)"
        return (
            f"[{name} {status}] score={score_txt} | thr={thr_txt}\n"
            f"Reason:\n{reason_txt}"
        )

    scorecard_parts: List[str] = [
        _fmt_metric("Chemical GEval", chem),
        _fmt_metric("AnswerRelevancy", ans),
        _fmt_metric("SS-Faithfulness", faith),
    ]

    history_txt = ""
    if history:
        def _fmt_best(label: str, d: Dict[str, Any]) -> str:
            a_no = d.get("attempt_no")
            sc = d.get("score")
            sc_txt = "None" if sc is None else f"{float(sc):.4f}"
            return f"- {label}: attempt {a_no} | score={sc_txt}"

        best_faith = history.get("best_faith", {})
        best_chem = history.get("best_chem", {})
        best_combo = history.get("best_combo", {})
        last_pass = history.get("last_pass", {})
        last_pass_txt = ", ".join([f"{k}={'PASS' if v else 'FAIL'}" for k, v in last_pass.items()]) if last_pass else ""

        history_txt = (
            "[Best-so-far across attempts]\n"
            + _fmt_best("SS-Faithfulness", best_faith) + "\n"
            + _fmt_best("Chemical GEval", best_chem) + "\n"
            + _fmt_best("Min(Chem,Faith)", best_combo) + "\n"
        )
        if last_pass_txt:
            history_txt += f"\n[Last attempt PASS/FAIL] {last_pass_txt}\n"
        history_txt += "\n"

    return (
        "You are improving ONE inorganic synthesis route.\n"
        f"Target composition: {composition}\n"
        f"Target method (if any): {method or 'N/A'}\n\n"
        "Below is the FULL evaluation scorecard (including PASS metrics).\n"
        "Use it to preserve strengths and fix weaknesses.\n\n"
        + (history_txt or "")
        + "[Evaluation scorecard]\n"
        + "\n\n---\n\n".join(scorecard_parts)
        + "\n\n"
        "[Hard constraints]\n"
        "0) Do NOT degrade any metric that is currently PASS. Preserve what already passes.\n"
        "1) Stay faithful to the retrieved context. Do NOT invent contradictory conditions.\n"
        "2) Keep the JSON structure exactly as requested (method/precursors/additives/seeds/steps etc.).\n"
        "3) If a step requires atmosphere/temperature/time/pressure, specify them explicitly.\n"
        "4) Avoid unsafe instructions; keep it lab-safe and realistic.\n"
    )


def _save_report_text(*, out_dir: str, text: str) -> Optional[str]:
    """Save a text report into the output directory and return the file path."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, "report.txt")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(text)
        return fp
    except Exception:
        return None


def _load_logo_data(path: Optional[str], max_size: Tuple[int, int] = (96, 96)) -> Optional[bytes]:
    """Load and downscale a logo image to fit the header without cropping.

    Uses Pillow if available; falls back to None if not.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        img = Image.open(path)
        img = img.convert("RGBA")
        img.thumbnail(max_size, Image.LANCZOS)
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        return None


def _print_route_result(window: sg.Window, i: int, r: Dict[str, Any], verbose: bool) -> None:
    overall_ok, overall_label = _route_overall_status(r)

    overall_symbol = "✅" if overall_ok else "⚠️"
    status_symbol = "🏁" if overall_ok else "🚩"

    window["-LOG-"].print("\n" + "─" * 56)
    window["-LOG-"].print(f"Route #{i} {status_symbol}  {overall_symbol}  {overall_label}")
    if int(r.get("attempt", 1) or 1) > 1:
        max_a = int(r.get("max_attempts", 3) or 3)
        window["-LOG-"].print(f"🔁 Regeneration attempt: {int(r.get('attempt', 1))}/{max_a}")
    window["-LOG-"].print(
        "File: {} | composition={} | method={}".format(
            r.get("file"),
            r.get("composition"),
            r.get("method"),
        )
    )

    # Compact KPI line
    window["-LOG-"].print(metric_line(r))

    if not r.get("retrieval_context_found", True):
        window["-LOG-"].print("⚠️  retrieval_context.json not found beside route_*.json → SS-Faithfulness requires retrieved literature evidence")

    # Always show the route text, but keep it visually separated
    window["-LOG-"].print("\nSYNTHESIS ROUTE:")
    window["-LOG-"].print(_shorten(r.get("raw_text", ""), 2500))

    # Verbose block (optional)
    if verbose:
        window["-LOG-"].print("\nDETAILS:")
        window["-LOG-"].print(
            "CHEMICAL G-EVAL  →  {} (thr={}) | {}".format(
                score_text(r.get("score")),
                score_text(r.get("threshold"), 2),
                _fmt_bool(bool(r.get("passed"))),
            )
        )
        window["-LOG-"].print("G-Eval reason:")
        window["-LOG-"].print(_shorten(r.get("reason", ""), 900))
        window["-LOG-"].print(
            "RAG METRICS:" +
            "\nAnswerRelevancy → {} (thr={}) | {}".format(
                score_text(r.get("answer_score")),
                score_text(r.get("answer_threshold"), 2),
                _fmt_bool(bool(r.get("answer_passed"))),
            )
        )
        window["-LOG-"].print("AnswerRelevancy reason:")
        window["-LOG-"].print(_shorten(r.get("answer_reason", ""), 650))

        window["-LOG-"].print(
            "\nSS-Faithfulness: → {} (thr={}) | {}".format(
                score_text(r.get("faithfulness_score")),
                score_text(r.get("faithfulness_threshold"), 2),
                _fmt_bool(bool(r.get("faithfulness_passed"))),
            )
        )
        window["-LOG-"].print("SS-Faithfulness reason:")
        window["-LOG-"].print(_shorten(r.get("faithfulness_reason", ""), 900))



# --- Helper: render results with explanations toggle ---
def _render_results(
    window: sg.Window,
    *,
    out_dir: str,
    results: List[Dict[str, Any]],
    requested_num_routes: int,
    verbose: bool,
) -> None:
    """Render the latest evaluation results to the multiline.

    This is used both right after a run and when the user toggles
    the 'Metric explanations' checkbox after generation.
    """
    window["-LOG-"].update("")

    window["-LOG-"].print(f"\n✅ Generation complete. Routes saved under:\n  {out_dir}\n")
    if len(results) != requested_num_routes:
        window["-LOG-"].print(
            f"⚠️  Requested {requested_num_routes} routes, but found/evaluated {len(results)}. "
            f"(If routes were dumped into a single file, they were auto-split.)\n"
        )

    window["-LOG-"].print("Evaluation results")
    if verbose:
        window["-LOG-"].print("Mode: Metric explanations = ON (full details)")
    else:
        window["-LOG-"].print("Mode: Metric explanations = OFF (compact)")
    window["-LOG-"].print(
        "Tip: you can toggle 'Metric explanations' after generation — the output will refresh instantly."
    )

    for i, r in enumerate(results, start=1):
        _print_route_result(window, i, r, verbose=verbose)


def run_generation_and_eval(
    composition: str,
    method: Optional[str],
    gen_model: Optional[str],
    eval_model: Optional[str],
    num_routes: int,
    threshold: float,
    rpm_limit: int,
    improve_failed: bool = False,
    max_attempts: int = 3,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    High-level wrapper:
    1) generate N routes for composition;
    2) immediately evaluate them using:
       - Chemical G-Eval
       - AnswerRelevancy + SS-Faithfulness (RAG metrics)

    Returns: (output_dir, results).
    """
    if not composition:
        raise ValueError("Composition must not be empty.")

    # 1) Create a new directory for this run
    output_dir = new_run_directory(DEFAULT_LLM_OUTPUT_DIR, composition)

    # 2) Run synthway generation
    generate_synthesis_routes(
        composition=composition,
        method=method or None,
        num_routes=num_routes,
        output_dir=output_dir,
        rag_idx_dir=DEFAULT_RAG_INDEX_DIR,
        rag_topk=5,
        rag_min_lex_score=None,
        rag_filter_method=None,
        rag_elements=None,
        rag_req_precursors=None,
        rag_req_additives=None,
        rag_req_seeds=None,
        rag_preselect=10,
        model_name=gen_model,
        rag_shape=None,
        rag_solvent=None,
        rag_temperature=None,
        rag_time=None,
        rag_pH=None,
        rag_precursors=None,
        rag_additives=None,
        rag_seeds=None,
    )

    # 3) Load the newly generated routes
    # The generator may save route_*.json in subdirectories; load recursively.
    routes = _load_routes_recursive(output_dir)

    # Fallback: several routes may occur within one route_1.json (as "Route 1/2/3").
    # Split the text and calculate metrics for each part separately.
    if routes and len(routes) < num_routes:
        # Try to expand the single/partial route dump into per-route chunks.
        expanded: List[Dict[str, Any]] = []
        for r in routes:
            parts = _split_combined_routes(r.get("raw_text", ""))
            if len(parts) <= 1:
                expanded.append(r)
                continue
            for j, chunk in enumerate(parts, start=1):
                rr = dict(r)
                rr["raw_text"] = chunk
                rr["file"] = f"{r.get('file')} (part {j})"
                expanded.append(rr)
        routes = expanded

    if not routes:
        raise RuntimeError(f"No routes were generated in {output_dir!r}.")

    # If we expanded combined routes, multiple entries may share the same source path.
    # Materialize them into distinct route_*.json so downstream (incl. regeneration) can safely overwrite per-route.
    if len({r.get("path") for r in routes}) != len(routes):
        # Backup any existing route_*.json that might be a combined dump.
        try:
            for r in routes:
                p = r.get("path")
                if p and os.path.isfile(p):
                    base = os.path.splitext(os.path.basename(p))[0]
                    backup = os.path.join(os.path.dirname(p), f"{base}_combined_backup.json")
                    if not os.path.isfile(backup):
                        shutil.copy2(p, backup)
        except Exception:
            pass

        # Rewrite normalized per-route files.
        normalized: List[Dict[str, Any]] = []
        for idx, r in enumerate(routes, start=1):
            new_path = os.path.join(output_dir, f"route_{idx}.json")
            payload = {
                "retrieval_text": r.get("retrieval_text", "") or "",
                "raw_text": r.get("raw_text", "") or "",
            }
            try:
                with open(new_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            rr = dict(r)
            rr["file"] = f"route_{idx}.json"
            rr["path"] = new_path
            normalized.append(rr)
        routes = normalized

    # 4) Prepare the judge model and calculate metrics per route
    judge_model_obj = build_ss_judge_model(eval_model)

    # Create metrics once per run for efficiency and stability
    chemical_metric = make_chemical_geval_metric(
        threshold=threshold,
        judge_model=judge_model_obj,
    )
    answer_metric = AnswerRelevancyMetric(
        threshold=threshold,
        model=judge_model_obj,
        include_reason=True,
    )
    faithfulness_metric = SSFaithfulnessMetric(
        threshold=threshold,
        judge_model=judge_model_obj,
        before_call=lambda: _before_ss_call(),
    )

    results: List[Dict[str, Any]] = []
    request_timestamps: List[float] = []

    def _wait_if_needed() -> None:
        """RPM limiter: at most rpm_limit requests per minute."""
        nonlocal request_timestamps
        if not rpm_limit or rpm_limit <= 0:
            return
        now = time.time()
        request_timestamps = [t for t in request_timestamps if now - t < 60.0]
        if len(request_timestamps) >= rpm_limit:
            wait_for = 60.0 - (now - request_timestamps[0])
            if wait_for > 0:
                time.sleep(wait_for)

    def _before_ss_call():
        _wait_if_needed()
        request_timestamps.append(time.time())

    gen_model_name = (gen_model or "").strip() or next(iter(AVAILABLE_MODELS.keys()), "")

    for route in routes:
        # Attempt loop: attempt 1 is the original generation; attempts 2..max_attempts are guided regenerations.
        attempt = 1

        # Load retrieval_context.json once per route dir (same for all attempts).
        case_dir = os.path.dirname(route["path"])
        retrieval_context, comp_from_meta = _load_retrieval_context_json(case_dir)
        comp = (route.get("composition") or comp_from_meta or "UNKNOWN")

        # Start from the current on-disk payload (so regen overwrites remain consistent).
        current_raw_text = route.get("raw_text", "")
        current_retrieval_text = route.get("retrieval_text", "")

        # Keep attempt #1 texts as a safe fallback parent.
        attempt1_raw_text = current_raw_text
        attempt1_retrieval_text = current_retrieval_text

        # Feedback used for the NEXT attempt (attempt #1 has empty feedback).
        feedback_for_next = ""

        # Track which parent input produced the NEXT regen attempt.
        parent_source_for_next = "current"
        parent_attempt_no_for_next: Optional[int] = 1

        # Track best candidates across attempts so we can choose a better parent than "last attempt".
        best_faith = {"score": None, "attempt_no": 1, "raw_text": current_raw_text, "retrieval_text": current_retrieval_text}
        best_chem = {"score": None, "attempt_no": 1, "raw_text": current_raw_text, "retrieval_text": current_retrieval_text}
        best_combo = {"score": None, "attempt_no": 1, "raw_text": current_raw_text, "retrieval_text": current_retrieval_text}

        # Conditional bests to avoid selecting a parent that breaks a metric that is already PASS.
        best_faith_with_chem_pass = {"score": None, "attempt_no": None, "raw_text": "", "retrieval_text": ""}
        best_chem_with_faith_pass = {"score": None, "attempt_no": None, "raw_text": "", "retrieval_text": ""}

        last_metric_payload: Dict[str, Any] = {}

        while True:
            # --- (A) Chemical G-Eval ---
            tc_geval = build_test_case({"raw_text": current_raw_text, "retrieval_text": current_retrieval_text})
            _wait_if_needed()
            chemical_metric.measure(tc_geval)
            request_timestamps.append(time.time())

            # --- (B) RAG AnswerRelevancy + SS-Faithfulness ---
            tc_rag = _build_rag_test_case(
                composition=comp,
                raw_text=current_raw_text,
                retrieval_text=current_retrieval_text,
                retrieval_context=retrieval_context,
            )

            _wait_if_needed()
            answer_metric.measure(tc_rag)
            request_timestamps.append(time.time())

            faithfulness_metric.measure(tc_rag)

            metric_payload = {
                "file": route["file"],
                "path": route["path"],
                "composition": comp,
                "method": route.get("method"),
                "raw_text": current_raw_text,
                "generator_model": gen_model,
                "judge_model": eval_model,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "parent_source": ("original" if attempt == 1 else parent_source_for_next),
                "parent_attempt_no": (None if attempt == 1 else parent_attempt_no_for_next),
                "feedback_used": ("" if attempt == 1 else feedback_for_next),

                # Chemical G-Eval
                "score": chemical_metric.score,
                "threshold": chemical_metric.threshold,
                "passed": chemical_metric.is_successful(),
                "reason": chemical_metric.reason,

                # RAG metrics
                "answer_score": answer_metric.score,
                "answer_threshold": answer_metric.threshold,
                "answer_passed": answer_metric.is_successful(),
                "answer_reason": answer_metric.reason,
                "faithfulness_score": faithfulness_metric.score,
                "faithfulness_threshold": faithfulness_metric.threshold,
                "faithfulness_passed": faithfulness_metric.is_successful(),
                "faithfulness_reason": faithfulness_metric.reason,
                **faithfulness_metric.result_fields(),
                "retrieval_context_found": bool(retrieval_context),
            }

            validate_metric_result(metric_payload)
            last_metric_payload = metric_payload
            evaluation_path = _Path(output_dir) / f"evaluation_{_Path(route['path']).stem}_attempt{attempt}.json"
            evaluation_path.write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            # Regen trigger: ANY metric fails.
            needs_regen = (
                (not bool(metric_payload.get("passed")))
                or (not bool(metric_payload.get("answer_passed")))
                or (not bool(metric_payload.get("faithfulness_passed")))
            )

            if (not improve_failed) or (not needs_regen) or (attempt >= max_attempts):
                break

            # Build MetricResult objects (for feedback + parent selection).
            chem_res = _MetricResult(
                score=float(metric_payload.get("score", 0.0) or 0.0),
                threshold=float(metric_payload.get("threshold", 0.0) or 0.0),
                passed=bool(metric_payload.get("passed")),
                reason=str(metric_payload.get("reason") or ""),
            )
            ans_res = _MetricResult(
                score=float(metric_payload.get("answer_score", 0.0) or 0.0),
                threshold=float(metric_payload.get("answer_threshold", 0.0) or 0.0),
                passed=bool(metric_payload.get("answer_passed")),
                reason=str(metric_payload.get("answer_reason") or ""),
            )
            faith_res = _MetricResult(
                score=float(metric_payload.get("faithfulness_score", 0.0) or 0.0),
                threshold=float(metric_payload.get("faithfulness_threshold", 0.0) or 0.0),
                passed=bool(metric_payload.get("faithfulness_passed")),
                reason=str(metric_payload.get("faithfulness_reason") or ""),
            )

            # Update best candidates (by metric and by min(Chem,Faith)).
            if faith_res.score is not None:
                if best_faith["score"] is None or float(faith_res.score) > float(best_faith["score"]):
                    best_faith = {"score": float(faith_res.score), "attempt_no": attempt, "raw_text": current_raw_text, "retrieval_text": current_retrieval_text}
            if chem_res.score is not None:
                if best_chem["score"] is None or float(chem_res.score) > float(best_chem["score"]):
                    best_chem = {"score": float(chem_res.score), "attempt_no": attempt, "raw_text": current_raw_text, "retrieval_text": current_retrieval_text}
            if chem_res.score is not None and faith_res.score is not None:
                combo_val = float(min(float(chem_res.score), float(faith_res.score)))
                if best_combo["score"] is None or combo_val > float(best_combo["score"]):
                    best_combo = {"score": combo_val, "attempt_no": attempt, "raw_text": current_raw_text, "retrieval_text": current_retrieval_text}

            # Update conditional bests (only consider attempts that keep the other metric PASS).
            if chem_res.passed and (faith_res.score is not None):
                if best_faith_with_chem_pass["score"] is None or float(faith_res.score) > float(best_faith_with_chem_pass["score"]):
                    best_faith_with_chem_pass = {
                        "score": float(faith_res.score),
                        "attempt_no": attempt,
                        "raw_text": current_raw_text,
                        "retrieval_text": current_retrieval_text,
                    }
            if faith_res.passed and (chem_res.score is not None):
                if best_chem_with_faith_pass["score"] is None or float(chem_res.score) > float(best_chem_with_faith_pass["score"]):
                    best_chem_with_faith_pass = {
                        "score": float(chem_res.score),
                        "attempt_no": attempt,
                        "raw_text": current_raw_text,
                        "retrieval_text": current_retrieval_text,
                    }

            # Build feedback for the next regen attempt, including best-so-far summary.
            feedback_for_next = _build_reviewer_feedback(
                chem=chem_res,
                ans=ans_res,
                faith=faith_res,
                composition=str(comp),
                method=route.get("method"),
                history={
                    "best_faith": {"attempt_no": best_faith.get("attempt_no"), "score": best_faith.get("score")},
                    "best_chem": {"attempt_no": best_chem.get("attempt_no"), "score": best_chem.get("score")},
                    "best_combo": {"attempt_no": best_combo.get("attempt_no"), "score": best_combo.get("score")},
                    "best_faith_with_chem_pass": {"attempt_no": best_faith_with_chem_pass.get("attempt_no"), "score": best_faith_with_chem_pass.get("score")},
                    "best_chem_with_faith_pass": {"attempt_no": best_chem_with_faith_pass.get("attempt_no"), "score": best_chem_with_faith_pass.get("score")},
                    "last_pass": {"chemical": chem_res.passed, "answer": ans_res.passed, "faithfulness": faith_res.passed},
                },
            )

            # Choose parent for the NEXT regeneration attempt.
            # Strengthened: NEVER choose a parent that fails a metric that is already PASS in the current attempt.
            if (not faith_res.passed) and chem_res.passed:
                # Improve faithfulness while keeping chemical PASS.
                if best_faith_with_chem_pass.get("score") is not None:
                    parent = best_faith_with_chem_pass
                    parent_source_for_next = "best_faith_with_chem_pass"
                    parent_attempt_no_for_next = int(best_faith_with_chem_pass.get("attempt_no") or attempt)
                else:
                    parent = {"raw_text": current_raw_text, "retrieval_text": current_retrieval_text}
                    parent_source_for_next = "current"
                    parent_attempt_no_for_next = attempt
            elif (not chem_res.passed) and faith_res.passed:
                # Improve chemical while keeping faith PASS.
                if best_chem_with_faith_pass.get("score") is not None:
                    parent = best_chem_with_faith_pass
                    parent_source_for_next = "best_chem_with_faith_pass"
                    parent_attempt_no_for_next = int(best_chem_with_faith_pass.get("attempt_no") or attempt)
                else:
                    parent = {"raw_text": current_raw_text, "retrieval_text": current_retrieval_text}
                    parent_source_for_next = "current"
                    parent_attempt_no_for_next = attempt
            elif (not chem_res.passed) and (not faith_res.passed):
                # Both fail: pick the most balanced attempt.
                if best_combo.get("score") is not None:
                    parent = best_combo
                    parent_source_for_next = "best_combo"
                    parent_attempt_no_for_next = int(best_combo.get("attempt_no") or attempt)
                else:
                    parent = best_faith
                    parent_source_for_next = "best_faith"
                    parent_attempt_no_for_next = int(best_faith.get("attempt_no") or attempt)
            else:
                # If only AnswerRelevancy fails, keep the current attempt as parent.
                parent = {"raw_text": current_raw_text, "retrieval_text": current_retrieval_text}
                parent_source_for_next = "current"
                parent_attempt_no_for_next = attempt

            # Fallback: if the chosen best_* parent has no validated score (None),
            # use the CURRENT attempt as parent, unless the current attempt is a bad input;
            # in that case, fall back to attempt #1.
            if isinstance(parent, dict) and ("score" in parent) and (parent.get("score") is None):
                if _is_bad_parent_text(current_raw_text):
                    parent = {"raw_text": attempt1_raw_text, "retrieval_text": attempt1_retrieval_text}
                    parent_source_for_next = "attempt1_fallback"
                    parent_attempt_no_for_next = 1
                else:
                    parent = {"raw_text": current_raw_text, "retrieval_text": current_retrieval_text}
                    parent_source_for_next = "current"
                    parent_attempt_no_for_next = attempt

            parent_raw_text = str(parent.get("raw_text") or "")
            parent_retrieval_text = str(parent.get("retrieval_text") or "")

            # Respect RPM limiter for the generation call too.
            _wait_if_needed()
            new_route, raw_model_out = regenerate_single_route_with_feedback(
                composition=str(comp),
                method=route.get("method"),
                contexts=retrieval_context,
                model_name=gen_model_name,
                previous_raw_text=parent_raw_text,
                previous_retrieval_text=parent_retrieval_text,
                reviewer_feedback=feedback_for_next,
                attempt_no=attempt + 1,
            )
            request_timestamps.append(time.time())

            # Save some provenance into the payload for UX/debugging.
            last_metric_payload["parent_source"] = parent_source_for_next
            last_metric_payload["parent_attempt_no"] = parent_attempt_no_for_next
            last_metric_payload["feedback_used"] = feedback_for_next

            if not new_route:
                # If we couldn't parse a new route, stop retrying to avoid infinite loops.
                last_metric_payload["regen_error"] = "Regeneration output was not parseable as a route."
                last_metric_payload["regen_raw_output"] = _shorten(raw_model_out or "", 1200)
                last_metric_payload["parent_source"] = parent_source_for_next
                last_metric_payload["parent_attempt_no"] = parent_attempt_no_for_next
                last_metric_payload["feedback_used"] = feedback_for_next
                break

            # Backup current on-disk route and overwrite with the regenerated payload.
            try:
                if os.path.isfile(route["path"]):
                    backup_path = route["path"].replace(".json", f"_attempt{attempt}.json")
                    if not os.path.isfile(backup_path):
                        shutil.copy2(route["path"], backup_path)
                with open(route["path"], "w", encoding="utf-8") as f:
                    json.dump(new_route, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            current_raw_text = new_route.get("raw_text", "")
            current_retrieval_text = new_route.get("retrieval_text", "")
            # Safety: if the model forgot to fill raw_text, keep the previous retrieval_text as fallback.
            if not (current_raw_text or "").strip() and (current_retrieval_text or "").strip():
                current_raw_text = ""
            attempt += 1

        results.append(last_metric_payload)

    (_Path(output_dir) / "evaluations.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_dir, results


def main():
    # --- UI / Layout ---
    # Try to resolve a logo path (prefer the user-provided absolute path, fallback to local file).
    logo_candidates = [
        f"{_REPO_ROOT}/synthway/IMG_5265.PNG",
        os.path.join(os.path.dirname(__file__), "IMG_5265.PNG"),
    ]
    logo_path = next((p for p in logo_candidates if os.path.isfile(p)), None)
    logo_data = _load_logo_data(logo_path, max_size=(96, 96))

    # Header (brand)
    if logo_data:
        logo_el = sg.Image(data=logo_data, key="-LOGO-")
    elif logo_path:
        # Fallback (no Pillow): avoid cropping by not forcing a fixed size.
        # subsample reduces the displayed size on Tk; tweak if needed.
        logo_el = sg.Image(filename=logo_path, key="-LOGO-", subsample=2)
    else:
        logo_el = sg.Text("")

    header_row = [
        sg.Column([[logo_el]], pad=((0, 10), (0, 0))),
        sg.Text("SynthWay", font=("Helvetica", 16, "bold")),
        sg.Push(),
        sg.Button("Help", key="-HELP-", button_color=("white", "#3b4252")),
        sg.Button("Paths", key="-PATHS-", button_color=("white", "#3b4252")),
    ]

    # Left panel — Quick inputs
    inputs_frame = sg.Frame(
        "Inputs",
        [
            [sg.Text("Target composition", size=(16, 1)), sg.Input(key="-COMP-", size=(22, 1))],
            [sg.Text("Method (optional)", size=(16, 1)), sg.Input(key="-METHOD-", size=(22, 1))],
        ],
        pad=(0, (0, 8)),
    )

    models_frame = sg.Frame(
        "Models",
        [
            [
                sg.Text("Generation", size=(10, 1)),
                sg.Combo(
                    values=list(GUI_MODELS),
                    default_value=DEFAULT_GENERATOR,
                    key="-GEN-",
                    size=(28, 1),
                    readonly=True,
                ),
            ],
            [
                sg.Text("Judge", size=(10, 1)),
                sg.Combo(
                    values=list(GUI_MODELS),
                    default_value=DEFAULT_JUDGE,
                    key="-EVAL-",
                    size=(28, 1),
                    readonly=True,
                ),
            ],
        ],
        pad=(0, (0, 8)),
    )

    run_setup_frame = sg.Frame(
        "Run setup",
        [
            [
                sg.Text("Num routes", size=(10, 1)),
                sg.Spin(values=[1, 2, 3, 4, 5], initial_value=3, key="-NUM-", size=(5, 1)),
                sg.Push(),
                sg.Button("Advanced ▸", key="-ADV_TOGGLE-", button_color=("white", "#3b4252")),
            ],
        ],
        pad=(0, (0, 8)),
    )

    # Advanced (collapsed by default)
    advanced_frame = sg.Frame(
        "Advanced",
        [
            [
                sg.Text("Threshold", size=(10, 1)),
                sg.Input("0.80", key="-THR-", size=(6, 1)),
                sg.Text("RPM limit", size=(8, 1)),
                sg.Input("10", key="-RPM-", size=(6, 1)),
            ],
            [
                sg.Checkbox(
                    "Auto-improve failed routes",
                    default=False,
                    key="-IMPROVE-",
                ),
                sg.Push(),
                sg.Text("Max tries"),
                sg.Spin(values=[1, 2, 3], initial_value=3, key="-MAXTRY-", size=(4, 1)),
            ],
        ],
        visible=False,
        key="-ADV_FRAME-",
        pad=(0, (0, 8)),
    )

    left_panel = sg.Column(
        [
            header_row,
            [inputs_frame],
            [models_frame],
            [run_setup_frame],
            [advanced_frame],
        ],
        pad=(8, 8),
        vertical_alignment="top",
    )

    # Right panel — Output + toolbar
    output_toolbar = [
        sg.Button("📂 Open folder", key="-OPENF-", button_color=("white", "#3b4252")),
        sg.Button("📋 Copy summary", key="-COPY-", button_color=("white", "#3b4252")),
        sg.Button("💾 Save report", key="-SAVEREPORT-", button_color=("white", "#3b4252")),
        sg.Button("🧹 Clear", key="-CLEAR-", button_color=("white", "#3b4252")),
        sg.Push(),
        sg.Text("View:"),
        sg.Radio("Compact", "VIEW", key="-VIEW_COMPACT-", default=True, enable_events=True),
        sg.Radio("Detailed", "VIEW", key="-VIEW_DETAILED-", default=False, enable_events=True),
    ]

    right_panel = sg.Column(
        [
            [sg.Column([output_toolbar], expand_x=True)],
            [
                sg.Multiline(
                    size=(100, 25),
                    key="-LOG-",
                    autoscroll=True,
                    reroute_stdout=False,
                    expand_x=True,
                    expand_y=True,
                )
            ],
        ],
        pad=(8, 8),
        expand_x=True,
        expand_y=True,
        vertical_alignment="top",
    )

    # Footer action bar
    footer = [
        sg.Text("Status: Idle", key="-STATUS-"),
        sg.Push(),
        sg.Button("Exit", key="Exit"),
        sg.Button("▶ Run", key="Run", button_color=("white", "#2f855a")),
    ]

    layout = [
        [left_panel, right_panel],
        [sg.HorizontalSeparator()],
        footer,
    ]

    window = sg.Window(
        "SynthWay",
        layout,
        resizable=True,
        finalize=True,
    )

    last_out_dir: Optional[str] = None
    last_results: Optional[List[Dict[str, Any]]] = None
    last_requested_num_routes: int = 0

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "-HELP-":
            sg.popup(
                "1) Enter a target composition and optional method.\n"
                "2) Choose a generator and judge available to your account.\n"
                "3) Start with one route and automatic improvement off.\n"
                "4) Run generates and evaluates routes using your API credentials.\n\n"
                "Run makes billable provider requests. Generated protocols and scores\n"
                "do not establish experimental feasibility or safety.\n\n"
                "Compact shows the route and scores; Detailed adds the judge's reasons.\n"
                "Changing the view does not call a model again.\n"
                "Save report retains the displayed assessment. Paths shows where files are stored.\n\n"
                "See README.md for installation and demo instructions.",
                title="Help",
            )
            continue

        if event == "-PATHS-":
            sg.popup(
                f"Search index:\n{DEFAULT_RAG_INDEX_DIR}\n\n"
                f"New runs:\n{DEFAULT_LLM_OUTPUT_DIR}\n\n"
                + credential_guidance(),
                title="Paths",
            )
            continue

        if event == "-ADV_TOGGLE-":
            adv_visible = window["-ADV_FRAME-"].visible
            window["-ADV_FRAME-"].update(visible=not adv_visible)
            window["-ADV_TOGGLE-"].update("Advanced ▾" if not adv_visible else "Advanced ▸")
            window.refresh()
            continue

        if event in ("-VIEW_COMPACT-", "-VIEW_DETAILED-"):
            # View mode is a display toggle, not a compute toggle.
            if last_results is not None and last_out_dir is not None:
                verbose_now = bool(values.get("-VIEW_DETAILED-"))
                _render_results(
                    window,
                    out_dir=last_out_dir,
                    results=last_results,
                    requested_num_routes=last_requested_num_routes,
                    verbose=verbose_now,
                )
            continue

        if event == "-CLEAR-":
            window["-LOG-"].update("")
            continue

        if event == "-OPENF-":
            if last_out_dir:
                _open_folder(last_out_dir)
            else:
                sg.popup("No output folder yet — run generation first.", title="Open folder")
            continue

        if event == "-COPY-":
            summary = _build_run_summary(
                out_dir=last_out_dir,
                results=last_results,
                requested_num_routes=last_requested_num_routes,
            )
            try:
                sg.clipboard_set(summary)
                sg.popup("Copied summary to clipboard.", title="Copy")
            except Exception:
                sg.popup("Could not copy to clipboard.", title="Copy")
            continue

        if event == "-SAVEREPORT-":
            if not last_out_dir or not last_results:
                sg.popup("No results to save yet — run generation first.", title="Save report")
                continue
            text = _build_run_summary(
                out_dir=last_out_dir,
                results=last_results,
                requested_num_routes=last_requested_num_routes,
            )
            fp = _save_report_text(out_dir=last_out_dir, text=text)
            if fp:
                sg.popup(f"Saved report:\n{fp}", title="Save report")
            else:
                sg.popup("Could not save report.", title="Save report")
            continue

        if event == "Run":
            comp = (values.get("-COMP-") or "").strip()
            method = (values.get("-METHOD-") or "").strip() or None
            try:
                gen_model, eval_model = resolve_model_pair(
                    values.get("-GEN-") or DEFAULT_GENERATOR,
                    values.get("-EVAL-") or DEFAULT_JUDGE,
                )
            except ValueError as exc:
                window["-STATUS-"].update("Status: Idle")
                window["-LOG-"].print(str(exc))
                continue

            try:
                num_routes, threshold, rpm_limit, max_attempts = parse_run_settings(values)
            except ValueError as exc:
                window["-STATUS-"].update("Status: Idle")
                window["-LOG-"].print(f"Invalid settings: {exc}")
                continue

            verbose = bool(values.get("-VIEW_DETAILED-"))
            improve_failed = bool(values.get("-IMPROVE-"))

            window["-LOG-"].update("")
            if not comp:
                window["-STATUS-"].update("Status: Idle")
                window["-LOG-"].print(
                    "❌ Please specify target composition (e.g. Ag, Co80Ni20, SnxNi1-xO)."
                )
                continue

            window["-STATUS-"].update("Status: Running…")
            window.refresh()

            window["-LOG-"].print(
                f"▶ Launching generation + evaluation for composition={comp!r}, "
                f"model={gen_model!r}, eval_model={eval_model!r}, "
                f"num_routes={num_routes}"
            )

            try:
                out_dir, results = run_generation_and_eval(
                    composition=comp,
                    method=method,
                    gen_model=gen_model,
                    eval_model=eval_model,
                    num_routes=num_routes,
                    threshold=threshold,
                    rpm_limit=rpm_limit,
                    improve_failed=improve_failed,
                    max_attempts=max_attempts,
                )
            except Exception as e:
                window["-STATUS-"].update("Status: Error")
                window["-LOG-"].print(f"\n[ERROR] {type(e).__name__}: {e}")
                continue

            # Store last results for possible re-rendering.
            last_out_dir = out_dir
            last_results = results
            last_requested_num_routes = num_routes

            _render_results(
                window,
                out_dir=out_dir,
                results=results,
                requested_num_routes=num_routes,
                verbose=verbose,
            )
            window["-STATUS-"].update("Status: Done")

    window.close()


if __name__ == "__main__":
    main()
