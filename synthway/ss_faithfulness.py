"""Shared runtime adapter for the study's synthesis-specific Faithfulness formula."""
from dataclasses import asdict
import math
from typing import Any, Callable, Dict, List, Optional

SS_FAITHFULNESS_ID = "ss_faithfulness"
SS_FAITHFULNESS_FORMULA = "(3*Q1 + 2*Q2 + 2*Q3 + Q4) / 8"
ORIGINAL_FAITHFULNESS_ID = "original_faithfulness"


def build_ss_judge_model(model_id):
    from bench.new_faith_same_as_expert_logic import build_judge_model
    judge = build_judge_model(model_id)
    if judge is None or not callable(getattr(judge, "generate", None)):
        raise ValueError("SS-Faithfulness requires an explicit JSON-capable judge model.")
    return judge


def validate_resume_metric(records, metric_id):
    if not isinstance(records, list):
        raise ValueError("Expected a list of saved evaluations.")
    for record in records:
        if not isinstance(record, dict) or record.get("faithfulness_metric", ORIGINAL_FAITHFULNESS_ID) != metric_id:
            raise ValueError("Cannot resume evaluations with a different Faithfulness metric; choose a new output file.")


def validate_ss_record(record):
    if record.get("faithfulness_metric") != SS_FAITHFULNESS_ID:
        raise ValueError("Expected SS-Faithfulness, not an original or unlabelled Faithfulness result.")
    if record.get("faithfulness_formula") != SS_FAITHFULNESS_FORMULA:
        raise ValueError("Unexpected SS-Faithfulness formula.")
    details = record.get("faithfulness_details")
    if not isinstance(details, dict):
        raise ValueError("Missing SS-Faithfulness details.")
    q = details.get("q_scores", {})
    values = [q.get(key) for key in ("q1", "q2", "q3", "q4")]
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Missing or invalid SS-Faithfulness Q1-Q4 scores.")
    expected = sum(w*v for w, v in zip((3, 2, 2, 1), values)) / 8
    for value in (record.get("faithfulness_score"), details.get("score")):
        if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(value, expected, abs_tol=1e-12):
            raise ValueError("SS-Faithfulness score differs from its Q1-Q4 weighted formula.")


class RateLimitedJudgeProxy:
    """Apply the arm RPM limit to every internal expert-faithfulness call."""

    def __init__(self, judge: Any, before_call: Callable[[], None]) -> None:
        self._judge = judge
        self._before_call = before_call

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        self._before_call()
        return self._judge.generate(prompt, **kwargs)

    async def a_generate(self, prompt: str, **kwargs: Any) -> Any:
        self._before_call()
        async_generate = getattr(self._judge, "a_generate", None)
        if callable(async_generate):
            return await async_generate(prompt, **kwargs)
        return self._judge.generate(prompt, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._judge, name)


class SSFaithfulnessMetric:
    """DeepEval-compatible interface around the unchanged study implementation."""

    def __init__(self, threshold, judge_model, before_call=None):
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("SS-Faithfulness threshold must be between 0 and 1.")
        if judge_model is None or not callable(getattr(judge_model, "generate", None)):
            raise ValueError("SS-Faithfulness requires an explicit JSON-capable judge model.")
        self.threshold = float(threshold)
        self.judge_model = judge_model
        self.before_call = before_call
        self.score = None
        self.reason = ""
        self.details = None

    def measure(self, test_case):
        from bench.new_faith_same_as_expert_logic import measure_expert_faithfulness

        self.score, self.reason, self.details = None, "", None
        if not (test_case.actual_output or "").strip():
            raise ValueError("SS-Faithfulness requires a nonempty generated route.")
        if not test_case.retrieval_context or not all(isinstance(x, str) and x.strip() for x in test_case.retrieval_context):
            raise ValueError("SS-Faithfulness requires retrieved literature evidence.")
        judge = (RateLimitedJudgeProxy(self.judge_model, self.before_call)
                 if self.before_call else self.judge_model)
        result = measure_expert_faithfulness(test_case, judge)
        if not result.claims or not result.truths:
            raise ValueError("SS-Faithfulness judge returned no claims or no literature truths.")
        details = asdict(result)
        details.update(metric=SS_FAITHFULNESS_ID, formula=SS_FAITHFULNESS_FORMULA)
        record = {"faithfulness_metric": SS_FAITHFULNESS_ID,
                  "faithfulness_formula": SS_FAITHFULNESS_FORMULA,
                  "faithfulness_score": result.score, "faithfulness_details": details}
        validate_ss_record(record)
        self.score = float(result.score)
        self.details = details
        self.reason = _feedback_ready_faithfulness_reason(result)
        return self.score

    def is_successful(self):
        return self.score is not None and self.score >= self.threshold

    def result_fields(self):
        return {"faithfulness_metric": SS_FAITHFULNESS_ID,
                "faithfulness_formula": SS_FAITHFULNESS_FORMULA,
                "faithfulness_details": self.details}


def _feedback_ready_faithfulness_reason(result: Any) -> str:
    """Keep the exact formula explanation and surface unsupported claims."""

    categories = {
        int(item.get("claim_index")): str(item.get("category") or "minor")
        for item in (result.claim_categories or [])
        if isinstance(item, dict) and item.get("claim_index") is not None
    }
    failures: List[str] = []
    for verdict in result.verdicts or []:
        if not isinstance(verdict, dict) or verdict.get("verdict") != "no":
            continue
        claim_index = int(verdict.get("claim_index", -1))
        category = categories.get(claim_index, "minor").upper()
        claim = str(verdict.get("claim") or "").strip()
        reason = str(verdict.get("reason") or "unsupported by retrieval context").strip()
        failures.append(f"- [{category}] {claim} — {reason}")

    if not failures:
        return str(result.reason)
    visible = failures[:12]
    suffix = ""
    if len(failures) > len(visible):
        suffix = f"\n- ... and {len(failures) - len(visible)} more unsupported claim(s)."
    return (
        f"{result.reason}\n"
        "Unsupported synthesis claims to correct in the next attempt:\n"
        + "\n".join(visible)
        + suffix
    )


