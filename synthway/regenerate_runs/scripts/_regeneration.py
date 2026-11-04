"""Shared route iteration and feedback for the SS-Faithfulness experiment."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple



@dataclass
class MetricResult:
    score: Optional[float]
    threshold: float
    passed: bool
    reason: str


@dataclass
class AttemptRecord:
    run_id: str
    ts_utc: str
    case_id: str
    route_name: str
    attempt_no: int                 # 1..max_attempts_total
    attempt_kind: str               # "original" or "regen"
    regen_model: str
    judge_model: str

    # Parent used to generate this attempt (attempt #1 is the original file).
    parent_source: str              # best_faith_with_chem_pass | best_chem_with_faith_pass | best_combo | best_faith | current | attempt1_fallback | original
    parent_attempt_no: Optional[int]

    route_json_path: str
    raw_text: str                   # keep full raw_text for later analysis (can be big)
    retrieval_text: str

    composition: Optional[str]
    method: Optional[str]

    chemical_geval: MetricResult
    answer_relevancy: MetricResult
    faithfulness: MetricResult

    passed_all: bool
    feedback_used: str              # feedback that was injected for this attempt (empty for original)


@dataclass
class RouteSummary:
    run_id: str
    case_id: str
    route_name: str
    attempts_used: int
    final_status: str               # "PASS" or "FAIL"
    initial_scores: Dict[str, Optional[float]]
    final_scores: Dict[str, Optional[float]]
    deltas: Dict[str, Optional[float]]
    best_attempt_no: int
    best_scores: Dict[str, Optional[float]]


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Evaluator(ABC):
    """Evaluator interface and call pacing for SS-Faithfulness regeneration."""

    def __init__(
        self,
        judge_model_id: str,
        chemical_threshold: float,
        answer_threshold: float,
        faithfulness_threshold: float,
        rpm_limit: Optional[float] = None,
        stop_on_rate_limit: bool = False,
    ) -> None:
        self.judge_model_id = judge_model_id
        self.chemical_threshold = chemical_threshold
        self.answer_threshold = answer_threshold
        self.faithfulness_threshold = faithfulness_threshold
        self.rpm_limit = rpm_limit
        self.stop_on_rate_limit = stop_on_rate_limit

        # Built lazily in init_models()
        self._judge_model: Optional[Any] = None
        self._last_judge_call_ts: float = 0.0

    def _rate_limit_sleep_if_needed(self) -> None:
        if not self.rpm_limit or self.rpm_limit <= 0:
            return
        min_interval = 60.0 / float(self.rpm_limit)
        now = time.time()
        dt = now - self._last_judge_call_ts
        if dt < min_interval:
            time.sleep(min_interval - dt)
        self._last_judge_call_ts = time.time()

    @abstractmethod
    def evaluate(
        self,
        *,
        composition: str,
        method: Optional[str],
        retrieval_context: List[str],
        retrieval_text: str,
        raw_text: str,
    ) -> Tuple[MetricResult, MetricResult, MetricResult, bool]:
        """Return CPC, Answer Relevancy, SS-Faithfulness, and joint acceptance."""
        raise NotImplementedError


def build_feedback(
    *,
    chem: MetricResult,
    ans: MetricResult,
    faith: MetricResult,
    composition: str,
    method: Optional[str],
    history: Optional[Dict[str, Any]] = None,
) -> str:
    """Turn metric failures into an actionable reviewer-style feedback string."""
    # Always include ALL metric summaries (PASS and FAIL) so regeneration sees the full scorecard.
    def _fmt_metric(name: str, m: MetricResult) -> str:
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
        _fmt_metric("Faithfulness", faith),
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
            + _fmt_best("Faithfulness", best_faith) + "\n"
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


def list_case_dirs(routes_dir: str) -> List[Path]:
    p = Path(routes_dir)
    if not p.exists():
        raise FileNotFoundError(f"--routes-dir not found: {routes_dir}")
    return sorted([x for x in p.iterdir() if x.is_dir()])


def list_route_files(case_dir: Path) -> List[Path]:
    return sorted([x for x in case_dir.iterdir() if x.is_file() and x.name.startswith("route_") and x.suffix == ".json"])


def ensure_route_fields(route: Dict[str, Any]) -> Tuple[str, str]:
    """Return (retrieval_text, raw_text) from a route dict.

    Parity note:
    - For RAG metrics we want the same behavior as evaluate_answer_faithfulness_deepeval.py:
      if raw_text is empty, the metric test-case builder will fall back to retrieval_text.
    - Therefore we DO NOT replace empty raw_text with json.dumps(route).
    """
    retrieval_text = str(route.get("retrieval_text") or "")
    raw_text = str(route.get("raw_text") or "")
    return retrieval_text, raw_text


def is_bad_parent_text(raw_text: str) -> bool:
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


def pick_best_attempt(attempts: List[AttemptRecord]) -> AttemptRecord:
    """
    Best attempt heuristic: maximize (passed_all, chemical, faithfulness, answer).
    """
    def key(a: AttemptRecord):
        return (
            1 if a.passed_all else 0,
            -1 if a.chemical_geval.score is None else a.chemical_geval.score,
            -1 if a.faithfulness.score is None else a.faithfulness.score,
            -1 if a.answer_relevancy.score is None else a.answer_relevancy.score,
        )
    return sorted(attempts, key=key, reverse=True)[0]


def process_one_route(
    *,
    run_id: str,
    evaluator: Evaluator,
    regen_model: str,
    case_id: str,
    case_dir: Path,
    route_path: Path,
    out_attempts_jsonl: str,
    out_routes_subdir: Path,
    max_attempts_total: int,
    dry_run: bool = False,
) -> RouteSummary:
    """
    Attempt #1: evaluate original route file.
    Attempts #2..max_attempts_total: regenerate + evaluate.
    """
    meta_path = case_dir / "retrieval_context.json"
    meta = read_json(str(meta_path)) if meta_path.exists() else {}

    composition = str(meta.get("composition") or "").strip() or "UNKNOWN"
    method = meta.get("method", None)
    retrieval_context = meta.get("retrieval_context", []) or []

    route_name = route_path.name
    route_obj = read_json(str(route_path))
    retrieval_text, raw_text = ensure_route_fields(route_obj)

    attempts: List[AttemptRecord] = []

    # Keep attempt #1 texts as a safe fallback parent.
    attempt1_raw_text = raw_text
    attempt1_retrieval_text = retrieval_text

    feedback = ""  # empty for attempt #1
    prev_raw_text = raw_text
    prev_retrieval_text = retrieval_text

    # Track which parent input produced the CURRENT regen attempt.
    # Attempt #2 is generated from the original attempt #1 (current/original).
    prev_parent_source = "current"
    prev_parent_attempt_no: Optional[int] = 1

    # Track best candidates across attempts so we can choose a better parent than "last attempt".
    best_faith = {"score": None, "attempt_no": 1, "raw_text": prev_raw_text, "retrieval_text": prev_retrieval_text}
    best_chem = {"score": None, "attempt_no": 1, "raw_text": prev_raw_text, "retrieval_text": prev_retrieval_text}
    best_combo = {"score": None, "attempt_no": 1, "raw_text": prev_raw_text, "retrieval_text": prev_retrieval_text}
    # Conditional bests to avoid selecting a parent that breaks a metric that is already PASS.
    best_faith_with_chem_pass = {"score": None, "attempt_no": None, "raw_text": "", "retrieval_text": ""}
    best_chem_with_faith_pass = {"score": None, "attempt_no": None, "raw_text": "", "retrieval_text": ""}

    for attempt_no in range(1, max_attempts_total + 1):
        attempt_kind = "original" if attempt_no == 1 else "regen"

        # On regen attempts: create new route
        if attempt_no > 1:
            if dry_run:
                new_route = None
                raw_model_out = ""
            else:
                from predictions import regenerate_single_route_with_feedback  # type: ignore

                new_route, raw_model_out = regenerate_single_route_with_feedback(
                    composition=composition,
                    method=method,
                    contexts=retrieval_context,
                    model_name=regen_model,
                    previous_raw_text=prev_raw_text,
                    previous_retrieval_text=prev_retrieval_text,
                    reviewer_feedback=feedback,
                    attempt_no=attempt_no,
                )

            # If model returned nothing parseable, stop further attempts and log as skipped (predictions.py returns None in this case)
            if not new_route:
                # Regeneration failed to produce a parseable route (predictions.py returns None in this case).
                # Log the failed attempt and stop further attempts for this route.
                rec = AttemptRecord(
                    run_id=run_id,
                    ts_utc=utc_now_iso(),
                    case_id=case_id,
                    route_name=route_name,
                    attempt_no=attempt_no,
                    attempt_kind=attempt_kind,
                    regen_model=regen_model,
                    judge_model=evaluator.judge_model_id,
                    parent_source=prev_parent_source,
                    parent_attempt_no=prev_parent_attempt_no,
                    route_json_path=str(out_routes_subdir / f"{route_name.replace('.json','')}_attempt{attempt_no}_regen_raw.txt"),
                    raw_text=(raw_model_out or ""),
                    retrieval_text=prev_retrieval_text,
                    composition=composition,
                    method=method if method is None or isinstance(method, str) else str(method),
                    chemical_geval=MetricResult(score=None, threshold=evaluator.chemical_threshold, passed=False, reason="SKIPPED: regeneration output not parseable"),
                    answer_relevancy=MetricResult(score=None, threshold=evaluator.answer_threshold, passed=False, reason="SKIPPED: regeneration output not parseable"),
                    faithfulness=MetricResult(score=None, threshold=evaluator.faithfulness_threshold, passed=False, reason="SKIPPED: regeneration output not parseable"),
                    passed_all=False,
                    feedback_used=feedback,
                )
                out_routes_subdir.mkdir(parents=True, exist_ok=True)
                raw_txt_path = out_routes_subdir / f"{route_name.replace('.json','')}_attempt{attempt_no}_regen_raw.txt"
                with open(raw_txt_path, "w", encoding="utf-8") as f:
                    f.write(raw_model_out or "")
                rec.route_json_path = str(raw_txt_path)
                attempts.append(rec)
                append_jsonl(out_attempts_jsonl, asdict(rec))
                break

            # Persist regenerated artifact
            out_routes_subdir.mkdir(parents=True, exist_ok=True)
            out_route_path = out_routes_subdir / f"{route_name.replace('.json','')}_attempt{attempt_no}.json"
            write_json(str(out_route_path), new_route)

            route_obj = new_route
            retrieval_text, raw_text = ensure_route_fields(route_obj)

        # Evaluate
        chem, ans, faith, passed_all = evaluator.evaluate(
            composition=composition,
            method=method,
            retrieval_context=retrieval_context,
            retrieval_text=retrieval_text,
            raw_text=raw_text,
        )
        # Update best candidates (by metric and by min(Chem,Faith)).
        if faith.score is not None:
            if best_faith["score"] is None or float(faith.score) > float(best_faith["score"]):
                best_faith = {"score": float(faith.score), "attempt_no": attempt_no, "raw_text": raw_text, "retrieval_text": retrieval_text}
        if chem.score is not None:
            if best_chem["score"] is None or float(chem.score) > float(best_chem["score"]):
                best_chem = {"score": float(chem.score), "attempt_no": attempt_no, "raw_text": raw_text, "retrieval_text": retrieval_text}
        if chem.score is not None and faith.score is not None:
            combo_val = float(min(float(chem.score), float(faith.score)))
            if best_combo["score"] is None or combo_val > float(best_combo["score"]):
                best_combo = {"score": combo_val, "attempt_no": attempt_no, "raw_text": raw_text, "retrieval_text": retrieval_text}

        # Update conditional bests (only consider attempts that keep the other metric PASS).
        if chem.passed and (faith.score is not None):
            if best_faith_with_chem_pass["score"] is None or float(faith.score) > float(best_faith_with_chem_pass["score"]):
                best_faith_with_chem_pass = {
                    "score": float(faith.score),
                    "attempt_no": attempt_no,
                    "raw_text": raw_text,
                    "retrieval_text": retrieval_text,
                }

        if faith.passed and (chem.score is not None):
            if best_chem_with_faith_pass["score"] is None or float(chem.score) > float(best_chem_with_faith_pass["score"]):
                best_chem_with_faith_pass = {
                    "score": float(chem.score),
                    "attempt_no": attempt_no,
                    "raw_text": raw_text,
                    "retrieval_text": retrieval_text,
                }

        rec = AttemptRecord(
            run_id=run_id,
            ts_utc=utc_now_iso(),
            case_id=case_id,
            route_name=route_name,
            attempt_no=attempt_no,
            attempt_kind=attempt_kind,
            regen_model=regen_model,
            judge_model=evaluator.judge_model_id,
            parent_source=("original" if attempt_no == 1 else prev_parent_source),
            parent_attempt_no=(None if attempt_no == 1 else prev_parent_attempt_no),
            route_json_path=str(route_path) if attempt_no == 1 else str(out_routes_subdir / f"{route_name.replace('.json','')}_attempt{attempt_no}.json"),
            raw_text=raw_text,
            retrieval_text=retrieval_text,
            composition=composition,
            method=method if method is None or isinstance(method, str) else str(method),
            chemical_geval=chem,
            answer_relevancy=ans,
            faithfulness=faith,
            passed_all=passed_all,
            feedback_used=feedback,
        )
        attempts.append(rec)
        append_jsonl(out_attempts_jsonl, asdict(rec))

        if passed_all:
            break

        # Build feedback for the next regen attempt, including best-so-far summary.
        feedback = build_feedback(
            chem=chem,
            ans=ans,
            faith=faith,
            composition=composition,
            method=method if method is None or isinstance(method, str) else str(method),
            history={
                "best_faith": {"attempt_no": best_faith.get("attempt_no"), "score": best_faith.get("score")},
                "best_chem": {"attempt_no": best_chem.get("attempt_no"), "score": best_chem.get("score")},
                "best_combo": {"attempt_no": best_combo.get("attempt_no"), "score": best_combo.get("score")},
                "best_faith_with_chem_pass": {"attempt_no": best_faith_with_chem_pass.get("attempt_no"), "score": best_faith_with_chem_pass.get("score")},
                "best_chem_with_faith_pass": {"attempt_no": best_chem_with_faith_pass.get("attempt_no"), "score": best_chem_with_faith_pass.get("score")},
                "last_pass": {"chemical": chem.passed, "answer": ans.passed, "faithfulness": faith.passed},
            },
        )

        # Choose parent for the NEXT regeneration attempt.
        # Strengthened: NEVER choose a parent that fails a metric that is already PASS in the current attempt.
        if (not faith.passed) and chem.passed:
            # Improve faithfulness while keeping chemical PASS: choose the best faith attempt among those with Chem PASS.
            if best_faith_with_chem_pass.get("score") is not None:
                parent = best_faith_with_chem_pass
                next_parent_source = "best_faith_with_chem_pass"
                next_parent_attempt_no = best_faith_with_chem_pass.get("attempt_no")
            else:
                parent = {"raw_text": raw_text, "retrieval_text": retrieval_text}
                next_parent_source = "current"
                next_parent_attempt_no = attempt_no
        elif (not chem.passed) and faith.passed:
            # Improve chemical while keeping faith PASS: choose the best chem attempt among those with Faith PASS.
            if best_chem_with_faith_pass.get("score") is not None:
                parent = best_chem_with_faith_pass
                next_parent_source = "best_chem_with_faith_pass"
                next_parent_attempt_no = best_chem_with_faith_pass.get("attempt_no")
            else:
                parent = {"raw_text": raw_text, "retrieval_text": retrieval_text}
                next_parent_source = "current"
                next_parent_attempt_no = attempt_no
        elif (not chem.passed) and (not faith.passed):
            # Both fail: pick the most balanced attempt (maximize the weaker of the two scores).
            if best_combo.get("score") is not None:
                parent = best_combo
                next_parent_source = "best_combo"
                next_parent_attempt_no = best_combo.get("attempt_no")
            else:
                parent = best_faith
                next_parent_source = "best_faith"
                next_parent_attempt_no = best_faith.get("attempt_no")
        else:
            # If only AnswerRelevancy fails, keep the current attempt as parent.
            parent = {"raw_text": raw_text, "retrieval_text": retrieval_text}
            next_parent_source = "current"
            next_parent_attempt_no = attempt_no

        # Fallback: if the chosen best_* parent has no validated score (None),
        # use the CURRENT attempt as parent, unless the current attempt is a bad input;
        # in that case, fall back to attempt #1.
        if isinstance(parent, dict) and ("score" in parent) and (parent.get("score") is None):
            if is_bad_parent_text(raw_text):
                parent = {"raw_text": attempt1_raw_text, "retrieval_text": attempt1_retrieval_text}
                next_parent_source = "attempt1_fallback"
                next_parent_attempt_no = 1
            else:
                parent = {"raw_text": raw_text, "retrieval_text": retrieval_text}
                next_parent_source = "current"
                next_parent_attempt_no = attempt_no

        prev_raw_text = str(parent.get("raw_text") or "")
        prev_retrieval_text = str(parent.get("retrieval_text") or "")
        prev_parent_source = str(next_parent_source)
        prev_parent_attempt_no = int(next_parent_attempt_no) if next_parent_attempt_no is not None else None

    best = pick_best_attempt(attempts)
    init = attempts[0]
    final = best

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(b) - float(a)

    initial_scores = {
        "chemical_geval": init.chemical_geval.score,
        "answer_relevancy": init.answer_relevancy.score,
        "faithfulness": init.faithfulness.score,
    }
    best_scores = {
        "chemical_geval": final.chemical_geval.score,
        "answer_relevancy": final.answer_relevancy.score,
        "faithfulness": final.faithfulness.score,
    }
    deltas = {
        k: _delta(initial_scores[k], best_scores[k]) for k in initial_scores
    }

    final_status = "PASS" if final.passed_all else "FAIL"
    return RouteSummary(
        run_id=run_id,
        case_id=case_id,
        route_name=route_name,
        attempts_used=len(attempts),
        final_status=final_status,
        initial_scores=initial_scores,
        final_scores=best_scores,
        deltas=deltas,
        best_attempt_no=final.attempt_no,
        best_scores=best_scores,
    )


def aggregate_summaries(route_summaries: List[RouteSummary]) -> Dict[str, Any]:
    total = len(route_summaries)
    pass_final = sum(1 for r in route_summaries if r.final_status == "PASS")
    attempt_dist: Dict[str, int] = {}
    for r in route_summaries:
        key = str(r.attempts_used) if r.final_status == "PASS" else f"FAIL_{r.attempts_used}"
        attempt_dist[key] = attempt_dist.get(key, 0) + 1

    def avg(vals: List[Optional[float]]) -> Optional[float]:
        xs = [float(x) for x in vals if x is not None]
        if not xs:
            return None
        return sum(xs) / len(xs)

    initial_geval = avg([r.initial_scores["chemical_geval"] for r in route_summaries])
    final_geval = avg([r.final_scores["chemical_geval"] for r in route_summaries])
    initial_ans = avg([r.initial_scores["answer_relevancy"] for r in route_summaries])
    final_ans = avg([r.final_scores["answer_relevancy"] for r in route_summaries])
    initial_faith = avg([r.initial_scores["faithfulness"] for r in route_summaries])
    final_faith = avg([r.final_scores["faithfulness"] for r in route_summaries])

    delta_geval = avg([r.deltas["chemical_geval"] for r in route_summaries])
    delta_ans = avg([r.deltas["answer_relevancy"] for r in route_summaries])
    delta_faith = avg([r.deltas["faithfulness"] for r in route_summaries])

    return {
        "total_routes": total,
        "pass_final": pass_final,
        "pass_final_rate": (pass_final / total) if total else 0.0,
        "attempt_distribution": attempt_dist,
        "avg_scores": {
            "initial": {
                "chemical_geval": initial_geval,
                "answer_relevancy": initial_ans,
                "faithfulness": initial_faith,
            },
            "final_best": {
                "chemical_geval": final_geval,
                "answer_relevancy": final_ans,
                "faithfulness": final_faith,
            },
            "delta_best_minus_initial": {
                "chemical_geval": delta_geval,
                "answer_relevancy": delta_ans,
                "faithfulness": delta_faith,
            },
        },
    }


def write_summary_csv(path: str, summary: Dict[str, Any]) -> None:
    # Minimal CSV without pandas (portable)
    lines: List[str] = []
    lines.append("metric,value")
    lines.append(f"total_routes,{summary.get('total_routes')}")
    lines.append(f"pass_final,{summary.get('pass_final')}")
    lines.append(f"pass_final_rate,{summary.get('pass_final_rate')}")
    for k, v in (summary.get("attempt_distribution") or {}).items():
        lines.append(f"attempt_dist_{k},{v}")

    avg_scores = summary.get("avg_scores", {})
    for stage in ["initial", "final_best", "delta_best_minus_initial"]:
        stage_obj = avg_scores.get(stage, {})
        for m in ["chemical_geval", "answer_relevancy", "faithfulness"]:
            lines.append(f"avg_{stage}_{m},{stage_obj.get(m)}")

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
