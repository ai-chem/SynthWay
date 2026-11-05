"""Evaluate synthesis-specific Faithfulness against retrieved literature procedures.

The metric extracts claims and reference facts, assigns support verdicts and
Q1–Q4 categories, and combines question scores as
(3*Q1 + 2*Q2 + 2*Q3 + Q4) / 8.

The CLI evaluates Faithfulness only. It reads route_*.json and sibling
retrieval_context.json files, writes results to --out-json, and supports --resume.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any


def _usage_to_dict(usage_obj) -> dict[str, Any] | None:
    """Convert OpenAI/OpenRouter usage object into a plain dict (best-effort)."""
    if usage_obj is None:
        return None

    if isinstance(usage_obj, dict):
        return usage_obj

    # Pydantic / OpenAI objects often expose model_dump
    try:
        md = getattr(usage_obj, "model_dump", None)
        if callable(md):
            return md()
    except Exception:
        pass

    out: dict[str, Any] = {}
    for k in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "cached_tokens",
        "cache_write_tokens",
        "cost",
    ):
        try:
            v = getattr(usage_obj, k, None)
        except Exception:
            v = None
        if v is not None:
            out[k] = v

    if not out:
        try:
            return dict(usage_obj)
        except Exception:
            return None

    return out


# DeepEval imports
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

try:
    from deepeval.metrics.faithfulness.template import FaithfulnessTemplate
except Exception:
    FaithfulnessTemplate = object


class SynthesisFaithfulnessTemplate(FaithfulnessTemplate):
    """Faithfulness template adapted for inorganic synthesis routes."""

    @staticmethod
    def generate_claims(actual_output: str, *args, **kwargs):
        return f"""
Extract factual claims from the generated inorganic synthesis route below.

Split the route into concrete, checkable claims. Prioritize claims that correspond to the slot-based retrieval_text schema used in this project:
- target composition and product identity;
- synthesis method;
- precursors, additives, seeds;
- solvent and solvent volume;
- pH and atmosphere;
- temperature, time, heating/cooling/aging/calcination/reduction conditions;
- crystal system, phase, morphology/shape;
- size.length, size.width, size.depth or particle size;
- PDF numbers / pdfno_list;
- washing, drying, separation and post-treatment steps only as secondary workup claims.

Rules:
1. Each claim must be specific enough to be verified against retrieval context.
2. Do not merge unrelated synthesis details into one claim.
3. Do not invent claims that are not stated in the output.
4. If a sentence contains several independent synthesis details, split it into several claims.
5. Extract key-slot claims before minor workup claims.
6. Return only valid JSON with exactly this structure:
   {{"claims": ["claim 1", "claim 2"]}}

Generated synthesis route:
{actual_output}
"""

    @staticmethod
    def generate_truths(
        retrieval_context: str, extraction_limit: int | None = None, *args, **kwargs
    ):
        return f"""
Extract factual statements from the retrieved literature context below.

The retrieved context uses a slot-based retrieval_text schema. Extract support by preserving these slot families whenever they are present:
- composition / target material identity;
- method;
- precursor, additive, seed;
- solvent and solvent.volume;
- pH and atmosphere;
- temperature and time-related process conditions;
- crystalsystem / phase;
- shape / morphology;
- size.length, size.width, size.depth;
- pdfno_list;
- workup steps such as washing, drying, centrifugation, filtration only if explicitly present.

Rules:
1. Extract only statements explicitly supported by the retrieved context.
2. Preserve numerical conditions and units when present.
3. Do not generalize beyond the context.
4. Keep different retrieved routes conceptually separate when the context contains several compositions.
5. If an extraction limit is provided, prioritize the most synthesis-relevant key-slot statements first.
6. Return only valid JSON with exactly this structure:
   {{"truths": ["truth 1", "truth 2"]}}

Extraction limit:
{extraction_limit}

Retrieved context:
{retrieval_context}
"""

    @staticmethod
    def generate_verdicts(claims: str, truths: str | None = None, *args, **kwargs):
        if truths is None:
            truths = (
                kwargs.get("truths") or kwargs.get("truth") or kwargs.get("retrieval_context") or ""
            )
        return f"""
For each generated synthesis claim, decide whether it is faithful to the retrieved literature truths.

This project uses a slot-based retrieval_text representation. Evaluate claims primarily against these key slot families:
- composition / target material identity;
- method;
- precursor, additive, seed;
- solvent and solvent.volume;
- pH and atmosphere;
- temperature and time-related process conditions;
- crystalsystem / phase;
- shape / morphology;
- size.length, size.width, size.depth;
- pdfno_list.

Before assigning verdicts:
1. If the truths contain several retrieved routes or several compositions, first identify the route(s) most relevant to the target composition in the generated claims.
2. Prefer same-composition truths when evaluating support.
3. Do not mark a claim as contradicted only because a different-composition retrieved route has a different slot value, if a same-composition route supports the claim.

Use the following verdict labels exactly:
- "yes": the claim is directly stated or strongly implied by the relevant retrieved truth(s), with the same underlying slot/value.
- "no": use this for any claim about a key slot that is contradicted by the relevant retrieved truth(s), OR for any claim about a key slot that introduces a specific value not supported by the relevant retrieved truth(s).
- "idk": use this only for minor secondary workup/procedural details that are absent from the context and do not change the synthesis route, such as generic collection, centrifugation, filtration, washing, drying, or cooling, unless those details contradict the context.

Important scoring rules for this project:
1. Do not treat chemical plausibility as faithfulness. A plausible synthesis detail is faithful only if it is supported by the relevant retrieved context.
2. Unsupported key-slot substitutions must be "no", not "idk". This includes unsupported or mismatched composition, method, precursor, additive, seed, solvent, pH, temperature, time, crystal system, morphology/shape, particle size, and PDF number.
3. Minor wording differences are acceptable only when the underlying synthesis slot and value are the same.
4. Generic workup claims may be "idk" if absent, but should be "no" if they specify a key unsupported value or conflict with the context.
5. Generate exactly one verdict object per claim, in the same order as the claims.
6. Return only valid JSON with exactly this structure:
   {{"verdicts": [{{"verdict": "yes"}}, {{"verdict": "idk", "reason": "minor workup detail not specified in the retrieved context"}}, {{"verdict": "no", "reason": "unsupported key-slot value in the retrieved context"}}]}}
7. Include "reason" only for "no" or "idk" verdicts.

Claims:
{claims}

Retrieved truths:
{truths}
"""


EXPERT_FAITHFULNESS_WEIGHTS: dict[str, float] = {
    "q1": 3.0,
    "q2": 2.0,
    "q3": 2.0,
    "q4": 1.0,
}

EXPERT_FAITHFULNESS_DENOMINATOR = sum(EXPERT_FAITHFULNESS_WEIGHTS.values())

EXPERT_Q_TEXT: dict[str, str] = {
    "q1": "Is the generated route consistent with the retrieval context with respect to the target material?",
    "q2": "Is the generated route consistent with the retrieval context in terms of key reagents and/or reaction medium?",
    "q3": "Is the generated route consistent with the retrieval context in terms of main process conditions?",
    "q4": "Is the generated route consistent with the retrieval context in terms of product characteristics or related output descriptors?",
}


@dataclass
class ExpertFaithfulnessResult:
    score: float
    q_scores: dict[str, float]
    q_counts: dict[str, dict[str, int]]
    claims: list[str]
    truths: list[str]
    verdicts: list[dict[str, Any]]
    claim_categories: list[dict[str, Any]]
    reason: str


def _strip_code_fences_from_json_text(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_object(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_json_object(text: Any, expected_key: str | None = None) -> dict[str, Any]:
    if isinstance(text, dict):
        data = text
    else:
        raw = str(text or "")
        candidates = [raw, _strip_code_fences_from_json_text(raw)]
        extracted = _extract_first_json_object(raw)
        if extracted:
            candidates.append(extracted)

        data = None
        errors: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    data = parsed
                    break
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
        if data is None:
            preview = raw[:500].replace("\n", " ")
            raise ValueError(
                "Judge returned invalid JSON. "
                f"Errors: {' | '.join(errors) if errors else 'no object found'}. "
                f"Response preview: {preview!r}"
            )

    if expected_key and expected_key not in data:
        raise ValueError(f"Judge JSON does not contain expected key {expected_key!r}: {data!r}")
    return data


def _call_judge_json(judge_model: Any, prompt: str, expected_key: str) -> dict[str, Any]:
    if judge_model is None:
        raise RuntimeError(
            "Expert-style faithfulness requires an explicit judge model. "
            "Pass --eval-model, for example google/gemini-2.0-flash-001 or openrouter:<provider/model>."
        )

    generate_fn = getattr(judge_model, "generate", None)
    if not callable(generate_fn):
        raise RuntimeError(
            "Expert-style faithfulness requires a custom judge object with .generate(). "
            "Use an OpenRouter/Gemini model through build_judge_model instead of a plain string model id."
        )

    response = generate_fn(prompt)
    return _parse_json_object(response, expected_key=expected_key)


def _normalize_verdict_label(value: Any) -> str:
    v = str(value or "").strip().lower()
    if v in {"yes", "supported", "true", "1"}:
        return "yes"
    if v in {"no", "unsupported", "contradicted", "false", "0"}:
        return "no"
    if v in {"idk", "unknown", "unclear", "not enough information", "n/a", "na"}:
        return "idk"
    return "idk"


def _normalize_q_category(value: Any) -> str:
    v = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "1": "q1",
        "q_1": "q1",
        "question_1": "q1",
        "target": "q1",
        "target_material": "q1",
        "material": "q1",
        "composition": "q1",
        "2": "q2",
        "q_2": "q2",
        "question_2": "q2",
        "reagents": "q2",
        "reaction_medium": "q2",
        "medium": "q2",
        "solvent": "q2",
        "3": "q3",
        "q_3": "q3",
        "question_3": "q3",
        "conditions": "q3",
        "process_conditions": "q3",
        "main_process_conditions": "q3",
        "4": "q4",
        "q_4": "q4",
        "question_4": "q4",
        "product": "q4",
        "product_characteristics": "q4",
        "output_descriptors": "q4",
        "minor_detail": "minor",
        "workup": "minor",
        "secondary": "minor",
    }
    v = aliases.get(v, v)
    if v not in {"q1", "q2", "q3", "q4", "minor"}:
        return "minor"
    return v


def build_claim_category_prompt(claims: list[str], verdicts: list[dict[str, Any]]) -> str:
    payload = []
    for i, claim in enumerate(claims):
        verdict = verdicts[i] if i < len(verdicts) else {}
        payload.append(
            {
                "claim_index": i,
                "claim": claim,
                "verdict": _normalize_verdict_label(verdict.get("verdict")),
                "verdict_reason": verdict.get("reason"),
            }
        )

    return f"""
Classify each generated synthesis claim into exactly one expert faithfulness question category.

Categories:
- q1: target material / composition / generated product identity.
- q2: key reagents and/or reaction medium, including precursors, salts, additives, seeds, solvents, solvent mixtures, ligand/reducing/capping agents.
- q3: main process conditions, including pH, atmosphere, temperature, time, heating, cooling, aging, calcination, reduction, hydrothermal/solvothermal conditions, stirring, irradiation, pressure.
- q4: product characteristics or output descriptors, including phase, crystal system, morphology, shape, size, particle dimensions, PDF numbers, crystallinity and related descriptors.
- minor: secondary generic procedural/workup details that should not affect the expert faithfulness formula directly, such as generic collection, centrifugation, filtration, washing, drying, storage, or wording-only details.

Rules:
1. Assign exactly one category per claim.
2. Use q1-q4 for synthesis-relevant claims whenever possible.
3. Use minor only for generic secondary workup/procedural details that do not correspond to q1-q4.
4. Keep the same order and number of items as the input claims.
5. Return only valid JSON with exactly this structure:
   {{"claim_categories": [{{"claim_index": 0, "category": "q1", "reason": "target composition/product identity"}}]}}

Claims with verdicts:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def compute_expert_q_scores(
    verdicts: list[dict[str, Any]],
    claim_categories: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, int]], str]:
    q_counts: dict[str, dict[str, int]] = {
        q: {"yes": 0, "no": 0, "idk": 0, "total": 0} for q in EXPERT_FAITHFULNESS_WEIGHTS
    }
    q_counts["minor"] = {"yes": 0, "no": 0, "idk": 0, "total": 0}

    for i, verdict_obj in enumerate(verdicts):
        label = _normalize_verdict_label(verdict_obj.get("verdict"))
        category = "minor"
        if i < len(claim_categories):
            category = _normalize_q_category(claim_categories[i].get("category"))
        q_counts[category][label] += 1
        q_counts[category]["total"] += 1

    q_scores: dict[str, float] = {}
    reason_parts: list[str] = []
    for q in ("q1", "q2", "q3", "q4"):
        counts = q_counts[q]
        # DeepEval-style ambiguous-claim logic with penalize_ambiguous_claims=False:
        # - "yes" claims support the question;
        # - "no" claims count against the question;
        # - "idk" claims are treated as ambiguous and excluded from the denominator.
        # In this project, the prompt above restricts "idk" to minor/secondary workup details;
        # unsupported key-slot values should be labelled as "no", not "idk".
        scored_total = counts["yes"] + counts["no"]
        if scored_total > 0:
            q_scores[q] = counts["yes"] / scored_total
            reason_parts.append(
                f"{q.upper()}={q_scores[q]:.3f} "
                f"(yes/(yes+no); yes={counts['yes']}, no={counts['no']}, "
                f"idk={counts['idk']} excluded)"
            )
        else:
            q_scores[q] = 1.0
            reason_parts.append(
                f"{q.upper()}=1.000 (no scored yes/no claims; idk={counts['idk']} excluded)"
            )

    reason = "; ".join(reason_parts)
    return q_scores, q_counts, reason


def compute_expert_formula_score(q_scores: dict[str, float]) -> float:
    weighted_sum = sum(
        EXPERT_FAITHFULNESS_WEIGHTS[q] * float(q_scores[q]) for q in EXPERT_FAITHFULNESS_WEIGHTS
    )
    return weighted_sum / EXPERT_FAITHFULNESS_DENOMINATOR


def measure_expert_faithfulness(tc: LLMTestCase, judge_model: Any) -> ExpertFaithfulnessResult:
    retrieval_context_text = "\n\n--- RETRIEVED ROUTE SEPARATOR ---\n\n".join(
        tc.retrieval_context or []
    )

    claims_prompt = SynthesisFaithfulnessTemplate.generate_claims(tc.actual_output)
    claims_data = _call_judge_json(judge_model, claims_prompt, expected_key="claims")
    claims = [str(x).strip() for x in (claims_data.get("claims") or []) if str(x).strip()]

    truths_prompt = SynthesisFaithfulnessTemplate.generate_truths(retrieval_context_text)
    truths_data = _call_judge_json(judge_model, truths_prompt, expected_key="truths")
    truths = [str(x).strip() for x in (truths_data.get("truths") or []) if str(x).strip()]

    verdicts_prompt = SynthesisFaithfulnessTemplate.generate_verdicts(
        claims=json.dumps({"claims": claims}, ensure_ascii=False, indent=2),
        truths=json.dumps({"truths": truths}, ensure_ascii=False, indent=2),
    )
    verdicts_data = _call_judge_json(judge_model, verdicts_prompt, expected_key="verdicts")
    verdicts_raw = verdicts_data.get("verdicts") or []
    verdicts: list[dict[str, Any]] = []
    for i, claim in enumerate(claims):
        raw = verdicts_raw[i] if i < len(verdicts_raw) and isinstance(verdicts_raw[i], dict) else {}
        verdicts.append(
            {
                "claim_index": i,
                "claim": claim,
                "verdict": _normalize_verdict_label(raw.get("verdict")),
                "reason": raw.get("reason"),
            }
        )

    category_prompt = build_claim_category_prompt(claims, verdicts)
    category_data = _call_judge_json(judge_model, category_prompt, expected_key="claim_categories")
    categories_raw = category_data.get("claim_categories") or []
    categories_by_index: dict[int, dict[str, Any]] = {}
    for item in categories_raw:
        if not isinstance(item, dict):
            continue
        try:
            ci = int(item.get("claim_index"))
        except Exception:
            continue
        categories_by_index[ci] = item

    claim_categories: list[dict[str, Any]] = []
    for i, claim in enumerate(claims):
        raw = categories_by_index.get(i, {})
        claim_categories.append(
            {
                "claim_index": i,
                "claim": claim,
                "category": _normalize_q_category(raw.get("category")),
                "reason": raw.get("reason"),
            }
        )

    q_scores, q_counts, q_reason = compute_expert_q_scores(verdicts, claim_categories)
    score = compute_expert_formula_score(q_scores)
    reason = (
        "Expert-style Faithfulness = (3*Q1 + 2*Q2 + 2*Q3 + Q4) / 8. "
        "Within each Q, idk verdicts follow DeepEval logic with penalize_ambiguous_claims=False: "
        "they are excluded from the yes/no denominator, while unsupported key-slot values must be labelled as no. "
        f"{q_reason}. Minor claims are excluded from the weighted formula."
    )

    return ExpertFaithfulnessResult(
        score=score,
        q_scores=q_scores,
        q_counts=q_counts,
        claims=claims,
        truths=truths,
        verdicts=verdicts,
        claim_categories=claim_categories,
        reason=reason,
    )


from dotenv import load_dotenv

load_dotenv()


def load_retrieval_context_json(case_dir: str) -> tuple[list[str], str | None, str | None]:
    """Load retrieval_context.json from a case directory.

    Returns:
        retrieval_context_list, composition_from_meta, method_from_meta
    """
    rc_path = os.path.join(case_dir, "retrieval_context.json")
    if not os.path.isfile(rc_path):
        return [], None, None

    try:
        with open(rc_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return [], None, None

    rc = meta.get("retrieval_context") or []
    if isinstance(rc, list):
        retrieval_context = [str(x) for x in rc]
    elif rc:
        retrieval_context = [str(rc)]
    else:
        retrieval_context = []

    composition = meta.get("composition")
    method = meta.get("method")
    return retrieval_context, composition, method


def build_test_case_from_parts(
    composition: str,
    method: str | None,
    raw_text: str,
    retrieval_text: str,
    retrieval_context: list[str],
) -> LLMTestCase:
    """Builds an LLMTestCase for RAG metrics from primitive fields."""
    comp = (composition or "UNKNOWN").strip() or "UNKNOWN"
    mth = (method or "").strip()
    input_text = comp if not mth else f"{comp} | method={mth}"

    actual_output = (raw_text or "").strip() or (retrieval_text or "").strip()
    return LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context or [],
    )


def make_answer_relevancy_metric(
    threshold: float,
    judge_model: Any | None = None,
) -> AnswerRelevancyMetric:
    """Build AnswerRelevancyMetric with the threshold and optional judge model."""
    kwargs: dict[str, Any] = {
        "threshold": threshold,
        "include_reason": True,
    }
    if judge_model is not None:
        kwargs["model"] = judge_model
    return AnswerRelevancyMetric(**kwargs)


def make_faithfulness_metric(
    threshold: float,
    judge_model: Any | None = None,
) -> FaithfulnessMetric:
    """Build the synthesis-specific Faithfulness metric with SynthesisFaithfulnessTemplate."""
    kwargs: dict[str, Any] = {
        "threshold": threshold,
        "include_reason": True,
        "verbose_mode": True,
        "penalize_ambiguous_claims": False,
        "evaluation_template": SynthesisFaithfulnessTemplate,
    }
    if judge_model is not None:
        kwargs["model"] = judge_model
    return FaithfulnessMetric(**kwargs)


@dataclass
class RouteRecord:
    file: str
    path: str
    composition: str | None
    raw_text: str
    retrieval_text: str
    retrieval_context: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated routes with DeepEval RAG metrics "
            "(AnswerRelevancyMetric + FaithfulnessMetric)."
        )
    )
    parser.add_argument(
        "--routes-dir",
        required=True,
        help=(
            "Root directory of generated routes (predictions.py output), "
            "containing case_id subdirectories with route_*.json and retrieval_context.json."
        ),
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help=(
            "Judge model for DeepEval metrics. "
            "Provider-prefixed identifiers use OpenRouter and OPENROUTER_API_KEY. "
            "Gemini without a provider prefix uses GeminiModel and GEMINI_API_KEY/GOOGLE_API_KEY. "
            "If omitted, use the DeepEval default configuration."
        ),
    )
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=0.7,
        help="AnswerRelevancyMetric pass threshold (0–1). Default: 0.7.",
    )
    parser.add_argument(
        "--faithfulness-threshold",
        type=float,
        default=0.8,
        help="FaithfulnessMetric pass threshold (0–1). Default: 0.8.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum routes to evaluate (for debugging or limiting API usage).",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=15,
        help=("Limit metric calls per minute (requests per minute). No limit is applied if <= 0."),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If set and --out-json exists, skip routes already evaluated successfully "
            "(identified by file path)."
        ),
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help=(
            "Path to the JSON file for per-route results. If omitted, print results to stdout only."
        ),
    )
    parser.add_argument(
        "--per-attempt-timeout",
        type=int,
        default=240,
        help=(
            "Timeout in seconds per metric-call attempt (DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE). "
            "Default: 240 seconds."
        ),
    )
    return parser.parse_args()


def parse_composition_from_retrieval(retrieval_text: str) -> str | None:
    if not retrieval_text:
        return None
    m = re.search(r"composition\s*[:=]\s*([^\|,\n]+)", retrieval_text, flags=re.IGNORECASE)
    if not m:
        return None
    comp = m.group(1).strip()
    comp = re.sub(r"\s+", " ", comp)
    return comp or None


# Load routes and retrieval_context


import glob


def load_route_records(routes_dir: str) -> list[RouteRecord]:
    """
    Recursively find all route_*.json files in routes_dir and load
    retrieval_context.json from the same directory for each route.
    """
    pattern = os.path.join(routes_dir, "**", "route_*.json")
    paths = sorted([p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)])

    if not paths:
        raise RuntimeError(f"No route_*.json files found in {routes_dir!r}.")

    records: list[RouteRecord] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        retrieval_text = (data.get("retrieval_text") or "").strip()
        raw_text = (data.get("raw_text") or "").strip()

        case_dir = os.path.dirname(path)
        rc_path = os.path.join(case_dir, "retrieval_context.json")
        retrieval_context: list[str] = []
        composition: str | None = None
        # First try obtaining composition from retrieval_context.json
        if os.path.isfile(rc_path):
            try:
                with open(rc_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                rc = meta.get("retrieval_context") or []
                if isinstance(rc, list):
                    retrieval_context = [str(x) for x in rc]
                else:
                    retrieval_context = [str(rc)]
                composition = meta.get("composition")
            except Exception as e:
                print(f"[WARN] Failed to read {rc_path!r}: {e}")
        else:
            print(
                f"[WARN] No retrieval_context.json found for route {os.path.basename(path)!r} "
                f"in {case_dir!r}. FaithfulnessMetric will be less informative for this route."
            )

        # If metadata lacks composition, try extracting it from retrieval_text
        if not composition:
            composition = parse_composition_from_retrieval(retrieval_text)

        records.append(
            RouteRecord(
                file=os.path.basename(path),
                path=path,
                composition=composition,
                raw_text=raw_text,
                retrieval_text=retrieval_text,
                retrieval_context=retrieval_context,
            )
        )

    return records


# Judge model


class OpenRouterChatModel(DeepEvalBaseLLM):
    """Minimal OpenAI-compatible chat model wrapper for DeepEval.

    Implements sync and async interfaces using the `openai` client
    with OpenRouter as base_url.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str | None = None,
        app_title: str | None = None,
        temperature: float = 0.0,
        force_json: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.force_json = force_json
        self.last_usage: dict[str, Any] | None = None
        self.last_cost_usd: float | None = None
        # Per-call accounting history (covers cases where a metric performs multiple LLM calls).
        self.usage_log: list[dict[str, Any]] = []

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "Failed to import 'openai'. Install/update it: pip install -U openai"
            ) from e

        headers: dict[str, str] = {}
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_title:
            headers["X-Title"] = app_title

        # `default_headers` is supported by the OpenAI python SDK and is passed on each request.
        self._client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)

    def load_model(self):
        """Return cached OpenAI client (required by DeepEvalBaseLLM in some versions)."""
        return self._client

    def get_model_name(self) -> str:
        return f"openrouter:{self.model}"

    def generate(self, prompt: str, schema: Any | None = None, **kwargs: Any) -> Any:
        client = self.load_model()

        # A strict system guard reduces "extra text" after JSON.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluation assistant. "
                    "Return ONLY a single valid JSON object as the entire response. "
                    "Do not include markdown, code fences, explanations, or any extra keys beyond what is requested."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if getattr(self, "force_json", False):
            req_kwargs["response_format"] = {"type": "json_object"}
        req_kwargs.update(kwargs)

        try:
            resp = client.chat.completions.create(**req_kwargs)
        except Exception:
            # Fallback: retry without response_format for compatibility.
            req_kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**req_kwargs)

        # OpenRouter usage accounting (judge)
        usage_dict = _usage_to_dict(getattr(resp, "usage", None)) or {}
        if "cost" not in usage_dict:
            try:
                extra_cost = getattr(resp, "cost", None)
                if extra_cost is not None:
                    usage_dict["cost"] = extra_cost
            except Exception:
                pass

        self.last_usage = usage_dict if usage_dict else None
        try:
            c = usage_dict.get("cost")
            self.last_cost_usd = float(c) if c is not None else None
        except Exception:
            self.last_cost_usd = None

        # Persist per-call usage so callers can sum *all* judge calls inside one measure().
        try:
            rec = dict(usage_dict) if isinstance(usage_dict, dict) else {}
        except Exception:
            rec = {}
        rec["ts"] = time.time()
        rec["model"] = self.model
        self.usage_log.append(rec)

        content = (resp.choices[0].message.content or "").strip()

        if schema is not None:

            def _strip_code_fences(s: str) -> str:
                s = (s or "").strip()
                if s.startswith("```"):
                    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
                    s = re.sub(r"\s*```$", "", s)
                return s.strip()

            def _extract_first_json_object(s: str) -> str | None:
                """Extract the first balanced JSON object from a string."""
                if not s:
                    return None
                start = s.find("{")
                if start < 0:
                    return None
                depth = 0
                in_string = False
                escape = False
                for i in range(start, len(s)):
                    ch = s[i]
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            return s[start : i + 1]
                return None

            json_candidates = [content, _strip_code_fences(content)]
            extracted = _extract_first_json_object(content)
            if extracted:
                json_candidates.append(extracted)

            data = None
            parse_errors: list[str] = []
            for candidate in json_candidates:
                if not candidate:
                    continue
                try:
                    data = json.loads(candidate)
                    break
                except Exception as pe:
                    parse_errors.append(f"{type(pe).__name__}: {pe}")

            if data is None:
                preview = content[:500].replace("\n", " ")
                raise ValueError(
                    "Invalid JSON returned by judge for schema-based DeepEval call. "
                    f"Errors: {' | '.join(parse_errors) if parse_errors else 'no JSON candidate found'}. "
                    f"Response preview: {preview!r}"
                )

            try:
                model_validate = getattr(schema, "model_validate", None)
                if callable(model_validate):
                    return model_validate(data)
            except Exception as ve:
                raise ValueError(
                    "Invalid JSON schema returned by judge for DeepEval schema-based call: "
                    f"{type(ve).__name__}: {ve}"
                ) from ve

            try:
                parse_obj = getattr(schema, "parse_obj", None)
                if callable(parse_obj):
                    return parse_obj(data)
            except Exception as ve:
                raise ValueError(
                    "Invalid JSON schema returned by judge for DeepEval schema-based call: "
                    f"{type(ve).__name__}: {ve}"
                ) from ve

            try:
                return schema(**data)
            except Exception as ve:
                raise ValueError(
                    "Invalid JSON schema returned by judge for DeepEval schema-based call: "
                    f"{type(ve).__name__}: {ve}"
                ) from ve

        return content

    async def a_generate(self, prompt: str, schema: Any | None = None, **kwargs: Any) -> Any:
        import asyncio
        from functools import partial

        loop = asyncio.get_event_loop()
        call = partial(self.generate, prompt, schema=schema, **kwargs)
        return await loop.run_in_executor(None, call)


def build_judge_model(eval_model: str | None) -> Any | None:
    """
    Prepare the judge model for DeepEval RAG metrics.

    Routing follows evaluate_generation_deepeval.py:
      0) openrouter:<vendor/model> -> OpenRouterChatModel
      1) google/gemini-...         -> OpenRouterChatModel (Gemini routed through OpenRouter)
      2) Any OpenRouter-like ID   -> OpenRouterChatModel (contains '/' or ':free')
      3) 'gemini'                 -> deepeval.models.GeminiModel (Google GenAI SDK)
      3) Gemini without a provider prefix -> GeminiModel; otherwise pass the string to DeepEval
    """
    if not eval_model:
        return None

    eval_model_str = str(eval_model).strip()
    eval_model_lc = eval_model_str.lower()

    # 0) Explicit OpenRouter prefix: openrouter:<vendor/model>
    if eval_model_lc.startswith("openrouter:"):
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY in the environment to use models through OpenRouter."
            )

        openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        app_title = os.getenv("OPENROUTER_APP_TITLE")

        model_id = eval_model_str.split(":", 1)[1].strip()

        print(
            "Using OpenRouterChatModel (openrouter: prefix) as the judge model: "
            f"model={model_id!r}, base_url={openrouter_base_url!r}"
        )

        return OpenRouterChatModel(
            model=model_id,
            api_key=openrouter_key,
            base_url=openrouter_base_url,
            http_referer=http_referer,
            app_title=app_title,
            temperature=0,
            force_json=True,
        )

    # 1) SPECIAL CASE: treat 'google/gemini-...' identifiers as OpenRouter models.
    #    This routes Gemini judging through OpenRouter rather than the Google GenAI SDK.
    if eval_model_lc.startswith("google/") and "gemini" in eval_model_lc:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise RuntimeError(
                "The 'google/gemini-...' format selects OpenRouter. "
                "Set OPENROUTER_API_KEY in the environment (or omit 'google/' to use the Gemini SDK)."
            )

        openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        app_title = os.getenv("OPENROUTER_APP_TITLE")

        print(
            "Using OpenRouterChatModel (google/gemini treated as OpenRouter) as the judge model: "
            f"model={eval_model_str!r}, base_url={openrouter_base_url!r}"
        )

        return OpenRouterChatModel(
            model=eval_model_str,
            api_key=openrouter_key,
            base_url=openrouter_base_url,
            http_referer=http_referer,
            app_title=app_title,
            temperature=0,
            force_json=True,
        )

    # 2) Any model through OpenRouter (OpenAI-compatible provider)
    #    Support both paid and free variants, e.g.:
    #    'nvidia/nemotron-3-nano-30b-a3b:free', 'qwen/qwen3-4b:free', 'deepseek/deepseek-r1'
    openrouter_like = ("/" in eval_model_str) or (":free" in eval_model_lc)
    if openrouter_like:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY in the environment to use models through OpenRouter."
            )

        openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        app_title = os.getenv("OPENROUTER_APP_TITLE")

        print(
            "Using OpenRouterChatModel through OpenRouter as the judge model: "
            f"model={eval_model_str!r}, base_url={openrouter_base_url!r}"
        )

        return OpenRouterChatModel(
            model=eval_model_str,
            api_key=openrouter_key,
            base_url=openrouter_base_url,
            http_referer=http_referer,
            app_title=app_title,
            temperature=0,
            force_json=True,
        )

    # 3) Gemini through the official GeminiModel (Google GenAI SDK)
    #    Expect 'gemini-2.x-...' without the 'google/' prefix here.
    if "gemini" in eval_model_lc:
        try:
            from deepeval.models import GeminiModel
        except Exception as e:
            raise RuntimeError(
                "Failed to import deepeval.models.GeminiModel. "
                "Ensure an up-to-date version of deepeval is installed."
            ) from e

        gem_api = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gem_api:
            raise RuntimeError(
                "To use Gemini in deepeval, set GEMINI_API_KEY "
                "or GOOGLE_API_KEY in the environment."
            )

        # Normalize common prefixes
        model_name = eval_model_str
        for prefix in (
            "publishers/google/models/",
            "models/",
            "google/",
        ):
            if model_name.lower().startswith(prefix):
                model_name = model_name[len(prefix) :]
                break

        print(
            f"Using GeminiModel for eval-model={eval_model_str!r} "
            f"(normalized model_name={model_name!r})"
        )

        try:
            return GeminiModel(model_name=model_name, api_key=gem_api, temperature=0)
        except TypeError:
            # Handle alternative deepeval signatures
            return GeminiModel(model=model_name, api_key=gem_api, temperature=0)

    # 4) Fallback: retain the string identifier for deepeval to resolve
    print(f"Using string judge model identifier: {eval_model_str!r}")
    return eval_model_str


# LLMTestCase from RouteRecord

# Sanitize judge input without rewriting the saved route.

_LATEX_MATH_RE = re.compile(r"\$[^$]*\$")


def sanitize_for_judge(text: str) -> str:
    """Remove inline $...$ blocks, escape backslashes, and normalize blank lines
    for judge input.
    """
    if not text:
        return text

    s = str(text)

    # Remove inline math blocks to reduce LaTeX backslash sequences.
    s = _LATEX_MATH_RE.sub("", s)

    # Escape backslashes (single -> double)
    s = s.replace("\\", "\\\\")

    # Trim excessive whitespace
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def build_test_case(route: RouteRecord) -> LLMTestCase:
    """
    Build an LLMTestCase:
    - input: the user's composition request,
    - actual_output: generated route,
    - retrieval_context: RAG context.
    """
    composition = route.composition or "UNKNOWN"

    input_text = composition

    actual_output = route.raw_text or route.retrieval_text
    actual_output = sanitize_for_judge(actual_output)

    return LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=route.retrieval_context,
    )


def main() -> None:
    args = parse_args()
    run_t0 = time.perf_counter()

    # Set per-attempt timeout env var if specified
    if getattr(args, "per_attempt_timeout", None) is not None:
        os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = str(args.per_attempt_timeout)
        # DeepEval caches settings at import time; reset so runtime env overrides take effect
        try:
            from deepeval.config.settings import reset_settings

            reset_settings(reload_dotenv=True)
        except Exception:
            pass

    judge_model = None
    try:
        print(f"Loading routes from: {args.routes_dir}")
        records = load_route_records(args.routes_dir)
        print(f"Total routes found: {len(records)}")

        if args.max_cases is not None:
            records = records[: args.max_cases]
            print(f"Limiting routes to {len(records)} using --max-cases.")

        judge_model = build_judge_model(args.eval_model)

        # Evaluate SS-Faithfulness through claims, reference facts, verdicts and Q1–Q4 categories.
        answer_metric = None
        faithfulness_metric = None

        existing_results: list[dict[str, Any]] = []
        processed_paths: set[str] = set()
        if args.out_json and args.resume and os.path.isfile(args.out_json):
            try:
                with open(args.out_json, "r", encoding="utf-8") as f:
                    existing_results = json.load(f) or []
                # Recompute pass flags for legacy/new result files so SUMMARY matches score>=threshold.
                for r in existing_results:
                    if isinstance(r, dict) and isinstance(
                        r.get("faithfulness_score"), (int, float)
                    ):
                        r["faithfulness_passed"] = float(r["faithfulness_score"]) >= float(
                            args.faithfulness_threshold
                        )

                # In resume mode, skip ONLY successfully evaluated routes (not errors).
                processed_paths = {
                    r["path"]
                    for r in existing_results
                    if (
                        isinstance(r, dict)
                        and "path" in r
                        and isinstance(r.get("faithfulness_score"), (int, float))
                        and "error" not in r
                    )
                }
                print(f"Resume mode: previously successful routes found: {len(processed_paths)}")
            except Exception as e:
                print(f"[WARN] Failed to load the existing results file {args.out_json!r}: {e}")
                existing_results = []
                processed_paths = set()

        if processed_paths:
            records = [r for r in records if r.path not in processed_paths]
            print(
                f"Routes remaining for evaluation after excluding previously evaluated routes: "
                f"{len(records)}"
            )

        rpm_limit: int = getattr(args, "rpm_limit", 15)
        request_timestamps: list[float] = []

        results: list[dict[str, Any]] = list(existing_results)

        # run health counters (errors / retries)
        n_to_process = len(records)
        n_success = 0
        n_errors = 0
        n_rate_limit = 0
        n_transient_errors = 0

        # consecutive error circuit-breaker
        consecutive_errors = 0
        max_consecutive_errors = 10

        # Retry counters
        n_retries_total = 0
        n_retries_invalid_json = 0
        n_retries_transient = 0

        # Error type histogram
        error_type_counts: dict[str, int] = {}

        def _flush_results() -> None:
            if not args.out_json:
                return
            out_path = args.out_json
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(
            "Running expert-style Faithfulness only: "
            "claims -> truths -> verdicts -> q1/q2/q3/q4/minor -> "
            "(3*Q1 + 2*Q2 + 2*Q3 + Q4)/8 "
            f"(faith_thr={args.faithfulness_threshold}, "
            f"model={'deepeval default' if judge_model is None else repr(args.eval_model)})"
        )

        def _wait_if_needed():
            nonlocal request_timestamps, rpm_limit
            # Fixed 7-second pause before each LLM call
            time.sleep(15)
            if rpm_limit and rpm_limit > 0:
                now = time.time()
                request_timestamps = [t for t in request_timestamps if now - t < 60.0]
                if len(request_timestamps) >= rpm_limit:
                    wait_for = 60.0 - (now - request_timestamps[0])
                    if wait_for > 0:
                        print(
                            f"[RATE LIMIT] Local limit of {rpm_limit} req/min reached; "
                            f"pausing for {wait_for:.1f} seconds."
                        )
                        time.sleep(wait_for)
                        now = time.time()
                        request_timestamps = [t for t in request_timestamps if now - t < 60.0]
                request_timestamps.append(time.time())

        for idx, route in enumerate(records, start=1):
            tc = build_test_case(route)
            used_fallback = not bool(route.raw_text)
            route_t0 = time.perf_counter()
            answer_total_time_s = None
            faith_total_time_s = None
            route_total_time_s = None

            # LLM-call time excludes fixed pauses and local RPM waits.
            answer_llm_time_s = 0.0
            faith_llm_time_s = 0.0

            # per-route counters
            route_retries_invalid_json = 0
            route_retries_transient = 0

            # per-route judge usage accumulation (tokens/cost across ALL judge calls, incl. retries)
            answer_attempts = 0
            faith_attempts = 0
            answer_judge_calls_total = 0
            faith_judge_calls_total = 0

            answer_usage_sum = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }
            faith_usage_sum = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }

            def _accumulate_usage(target: dict[str, Any], usage: dict[str, Any] | None) -> None:
                """Accumulate OpenRouter usage dict into target (best-effort)."""
                if not usage or not isinstance(usage, dict):
                    return

                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = usage.get(k)
                    if isinstance(v, (int, float)):
                        target[k] = int(target.get(k, 0)) + int(v)

                c = usage.get("cost")
                if isinstance(c, (int, float)):
                    target["cost"] = float(target.get("cost", 0.0)) + float(c)
                else:
                    # Fall back to cost_usd, total_cost, or price when cost is not numeric.
                    for alt in ("cost_usd", "total_cost", "price"):
                        ac = usage.get(alt)
                        if isinstance(ac, (int, float)):
                            target["cost"] = float(target.get("cost", 0.0)) + float(ac)
                            break

            def _accumulate_usage_from_log(
                target: dict[str, Any], entries: list[dict[str, Any]]
            ) -> None:
                """Accumulate usage across multiple judge calls (best-effort)."""
                if not entries:
                    return
                for ent in entries:
                    if isinstance(ent, dict) and ent:
                        _accumulate_usage(target, ent)

            try:

                def _is_invalid_json_exc(exc: Exception) -> bool:
                    msg = str(exc).lower()
                    invalid_markers = (
                        "invalid json",
                        "jsondecodeerror",
                        "invalid json schema",
                        "has no attribute 'verdicts'",
                        'has no attribute "verdicts"',
                    )
                    if any(marker in msg for marker in invalid_markers):
                        return True
                    for nested in (
                        getattr(exc, "__cause__", None),
                        getattr(exc, "__context__", None),
                    ):
                        if nested is None:
                            continue
                        nmsg = str(nested).lower()
                        if any(marker in nmsg for marker in invalid_markers):
                            return True
                    return False

                def _is_rate_limit_exc(exc: Exception) -> bool:
                    """Best-effort detection of HTTP 429 / rate limit errors from various SDKs."""
                    # Direct message match
                    msg = str(exc).lower()
                    if (
                        "429" in msg
                        or "rate limit" in msg
                        or "ratelimit" in msg
                        or "too many requests" in msg
                    ):
                        return True

                    # Some SDKs expose a status_code attribute
                    status = getattr(exc, "status_code", None)
                    if status == 429:
                        return True

                    # Walk cause/context chain
                    for nested in (
                        getattr(exc, "__cause__", None),
                        getattr(exc, "__context__", None),
                    ):
                        if nested is None:
                            continue
                        nmsg = str(nested).lower()
                        if (
                            "429" in nmsg
                            or "rate limit" in nmsg
                            or "ratelimit" in nmsg
                            or "too many requests" in nmsg
                        ):
                            return True
                        nstatus = getattr(nested, "status_code", None)
                        if nstatus == 429:
                            return True

                    return False

                def _is_transient_exc(exc: Exception) -> bool:
                    """Best-effort detection of transient transport / backend errors worth retrying."""
                    msg = str(exc).lower()

                    transient_markers = [
                        "server disconnected",
                        "connection reset",
                        "connection aborted",
                        "broken pipe",
                        "bad decrypt",
                        "bad_record_mac",
                        "decryption_failed",
                        "decryption failed",
                        "tsi_data_corrupted",
                        "corruption detected",
                        "ssl_transport_security_utils",
                        "ssl routines",
                        "transport is closing",
                        "temporarily unavailable",
                        "unavailable",
                        "deadline exceeded",
                        "read timeout",
                        "timeout",
                        "cannot connect",
                        "connection refused",
                        "connection error",
                        "network",
                    ]
                    if any(m in msg for m in transient_markers):
                        return True

                    if (
                        isinstance(exc, AttributeError)
                        and "statements" in msg
                        and "nonetype" in msg
                    ):
                        return True

                    for nested in (
                        getattr(exc, "__cause__", None),
                        getattr(exc, "__context__", None),
                    ):
                        if nested is None:
                            continue
                        nmsg = str(nested).lower()
                        if any(m in nmsg for m in transient_markers):
                            return True
                        if (
                            isinstance(nested, AttributeError)
                            and "statements" in nmsg
                            and "nonetype" in nmsg
                        ):
                            return True

                    try:
                        import aiohttp  # type: ignore

                        if isinstance(exc, aiohttp.ClientError):
                            return True
                    except Exception:
                        pass

                    try:
                        import grpc  # type: ignore

                        if isinstance(exc, grpc.RpcError):
                            return True
                    except Exception:
                        pass

                    return False

                max_json_retries = 2  # total attempts = 1 + retries
                max_transient_retries = 3  # extra retries for flaky transport/backend issues

                # AnswerRelevancy disabled: this script version evaluates only Faithfulness.
                answer_total_time_s = None
                answer_llm_time_s = 0.0
                answer_attempts = 0
                answer_judge_calls_total = 0

                # Faithfulness
                attempt = 0
                while True:
                    try:
                        usage_log_before = (
                            len(getattr(judge_model, "usage_log", []))
                            if judge_model is not None and hasattr(judge_model, "usage_log")
                            else 0
                        )
                        _wait_if_needed()
                        _t_call = time.perf_counter()
                        expert_faithfulness = measure_expert_faithfulness(tc, judge_model)
                        faith_llm_time_s += time.perf_counter() - _t_call

                        # Count attempt and accumulate usage for this attempt.
                        # Use per-call usage_log to capture *all* judge calls that may happen inside measure().
                        faith_attempts += 1
                        faith_usage = None
                        if judge_model is not None and hasattr(judge_model, "usage_log"):
                            try:
                                entries = list(getattr(judge_model, "usage_log"))[usage_log_before:]
                                faith_judge_calls_total += len(entries)
                                _accumulate_usage_from_log(faith_usage_sum, entries)
                                if entries:
                                    faith_usage = dict(entries[-1])
                            except Exception:
                                pass
                        elif judge_model is not None and hasattr(judge_model, "last_usage"):
                            try:
                                ju = getattr(judge_model, "last_usage")
                                if isinstance(ju, dict) and ju:
                                    faith_usage = dict(ju)
                                    _accumulate_usage(faith_usage_sum, faith_usage)
                                    faith_judge_calls_total += 1
                            except Exception:
                                pass

                        faith_total_time_s = time.perf_counter() - route_t0
                        break
                    except Exception as e:
                        # Count time spent in the judge call for failed attempts as well
                        try:
                            faith_llm_time_s += time.perf_counter() - _t_call
                        except Exception:
                            pass
                        attempt += 1
                        # Even on failure, count this attempt and accumulate any usage exposed by the judge.
                        faith_attempts += 1
                        if judge_model is not None and hasattr(judge_model, "usage_log"):
                            try:
                                entries = list(getattr(judge_model, "usage_log"))[usage_log_before:]
                                faith_judge_calls_total += len(entries)
                                _accumulate_usage_from_log(faith_usage_sum, entries)
                            except Exception:
                                pass
                        elif judge_model is not None and hasattr(judge_model, "last_usage"):
                            try:
                                ju = getattr(judge_model, "last_usage")
                                if isinstance(ju, dict) and ju:
                                    _accumulate_usage(faith_usage_sum, dict(ju))
                                    faith_judge_calls_total += 1
                            except Exception:
                                pass
                        if attempt <= max_json_retries and _is_invalid_json_exc(e):
                            n_retries_total += 1
                            n_retries_invalid_json += 1
                            route_retries_invalid_json += 1
                            print(
                                f"[WARN] Faithfulness judge returned invalid JSON; retry {attempt}/{max_json_retries}..."
                            )
                            time.sleep(2)
                            continue
                        if attempt <= (
                            max_json_retries + max_transient_retries
                        ) and _is_transient_exc(e):
                            n_retries_total += 1
                            n_retries_transient += 1
                            route_retries_transient += 1
                            backoff = min(30, 2 * attempt)
                            print(
                                f"[WARN] Faithfulness transient error; retry {attempt}/{max_json_retries + max_transient_retries} "
                                f"after {backoff}s... ({type(e).__name__}: {e})"
                            )
                            time.sleep(backoff)
                            continue
                        raise

                route_total_time_s = time.perf_counter() - route_t0
                result = {
                    "file": route.file,
                    "path": route.path,
                    "composition": route.composition,
                    "used_retrieval_text_fallback": used_fallback,
                    "answer_score": None,
                    "answer_threshold": None,
                    "answer_passed": None,
                    "answer_reason": None,
                    "faithfulness_score": expert_faithfulness.score,
                    "faithfulness_threshold": args.faithfulness_threshold,
                    "faithfulness_passed": (expert_faithfulness.score is not None)
                    and (float(expert_faithfulness.score) >= float(args.faithfulness_threshold)),
                    "faithfulness_reason": expert_faithfulness.reason,
                    "expert_faithfulness_formula": "(3*Q1 + 2*Q2 + 2*Q3 + Q4) / 8",
                    "expert_q_scores": expert_faithfulness.q_scores,
                    "expert_q_counts": expert_faithfulness.q_counts,
                    "expert_claims": expert_faithfulness.claims,
                    "expert_truths": expert_faithfulness.truths,
                    "expert_verdicts": expert_faithfulness.verdicts,
                    "expert_claim_categories": expert_faithfulness.claim_categories,
                    "route_retries_invalid_json": route_retries_invalid_json,
                    "route_retries_transient": route_retries_transient,
                    "answer_attempts": answer_attempts,
                    "faith_attempts": faith_attempts,
                    "answer_judge_calls_total": int(answer_judge_calls_total),
                    "faith_judge_calls_total": int(faith_judge_calls_total),
                    "answer_judge_prompt_tokens_total": int(
                        answer_usage_sum.get("prompt_tokens", 0)
                    ),
                    "answer_judge_completion_tokens_total": int(
                        answer_usage_sum.get("completion_tokens", 0)
                    ),
                    "answer_judge_total_tokens_total": int(answer_usage_sum.get("total_tokens", 0)),
                    "answer_judge_cost_usd_total": float(answer_usage_sum.get("cost", 0.0)),
                    "faith_judge_prompt_tokens_total": int(faith_usage_sum.get("prompt_tokens", 0)),
                    "faith_judge_completion_tokens_total": int(
                        faith_usage_sum.get("completion_tokens", 0)
                    ),
                    "faith_judge_total_tokens_total": int(faith_usage_sum.get("total_tokens", 0)),
                    "faith_judge_cost_usd_total": float(faith_usage_sum.get("cost", 0.0)),
                    "answer_total_time_s": answer_total_time_s,
                    "faith_total_time_s": faith_total_time_s,
                    "route_total_time_s": route_total_time_s,
                    "answer_llm_time_s": round(float(answer_llm_time_s), 6),
                    "faith_llm_time_s": round(float(faith_llm_time_s), 6),
                }
                # Attach per-metric judge usage (tokens + cost) when available
                if isinstance(locals().get("answer_usage"), dict) and answer_usage:
                    result["answer_judge_prompt_tokens"] = answer_usage.get("prompt_tokens")
                    result["answer_judge_completion_tokens"] = answer_usage.get("completion_tokens")
                    result["answer_judge_total_tokens"] = answer_usage.get("total_tokens")
                    result["answer_judge_cost_usd"] = answer_usage.get("cost")
                    result["answer_judge_usage_raw"] = answer_usage

                if isinstance(locals().get("faith_usage"), dict) and faith_usage:
                    result["faith_judge_prompt_tokens"] = faith_usage.get("prompt_tokens")
                    result["faith_judge_completion_tokens"] = faith_usage.get("completion_tokens")
                    result["faith_judge_total_tokens"] = faith_usage.get("total_tokens")
                    result["faith_judge_cost_usd"] = faith_usage.get("cost")
                    result["faith_judge_usage_raw"] = faith_usage
                results.append(result)

                n_success += 1
                consecutive_errors = 0

                _flush_results()

                print(
                    f"\n=== [{idx}/{len(records)}] {route.file} "
                    f"(composition={route.composition}) ===\n"
                    f"Expert Faithfulness: {expert_faithfulness.score:.3f} "
                    f"({'PASS' if result['faithfulness_passed'] else 'FAIL'}; "
                    f"thr={args.faithfulness_threshold})\n"
                    f"  formula: (3*Q1 + 2*Q2 + 2*Q3 + Q4) / 8\n"
                    f"  Q scores: {expert_faithfulness.q_scores}\n"
                    f"  reason: {expert_faithfulness.reason}\n"
                )

            except Exception as e:
                err_type = type(e).__name__
                base_msg = f"{err_type}: {e}"
                print(f"\n[ERROR] Evaluation of {route.file} failed: {base_msg}")

                # classify error
                is_rate_limited = _is_rate_limit_exc(e)
                is_transient = _is_transient_exc(e)
                category = (
                    "rate_limit" if is_rate_limited else ("network" if is_transient else "other")
                )

                n_errors += 1
                error_type_counts[err_type] = error_type_counts.get(err_type, 0) + 1
                consecutive_errors += 1
                if is_transient:
                    n_transient_errors += 1

                # Try extracting a more specific root cause (e.g., ClientError inside RetryError)
                cause = getattr(e, "__cause__", None)
                context_exc = getattr(e, "__context__", None)
                error_details: dict[str, Any] = {"base": base_msg}

                if cause is not None:
                    cause_type = type(cause).__name__
                    cause_msg = f"{cause_type}: {cause}"
                    error_details["cause"] = cause_msg
                    print(f"[ERROR-DETAIL] Root cause (cause): {cause_msg}")
                elif context_exc is not None:
                    ctx_type = type(context_exc).__name__
                    ctx_msg = f"{ctx_type}: {context_exc}"
                    error_details["context"] = ctx_msg
                    print(f"[ERROR-DETAIL] Exception context (context): {ctx_msg}")

                results.append(
                    {
                        "file": route.file,
                        "path": route.path,
                        "composition": route.composition,
                        "used_retrieval_text_fallback": used_fallback,
                        "error": base_msg,
                        "error_details": error_details,
                        "error_category": category,
                        "is_network_error": bool(is_transient),
                        "is_rate_limit": bool(is_rate_limited),
                        "route_retries_invalid_json": route_retries_invalid_json,
                        "route_retries_transient": route_retries_transient,
                        "answer_attempts": answer_attempts,
                        "faith_attempts": faith_attempts,
                        "answer_judge_calls_total": int(answer_judge_calls_total),
                        "faith_judge_calls_total": int(faith_judge_calls_total),
                        "answer_judge_prompt_tokens_total": int(
                            answer_usage_sum.get("prompt_tokens", 0)
                        ),
                        "answer_judge_completion_tokens_total": int(
                            answer_usage_sum.get("completion_tokens", 0)
                        ),
                        "answer_judge_total_tokens_total": int(
                            answer_usage_sum.get("total_tokens", 0)
                        ),
                        "answer_judge_cost_usd_total": float(answer_usage_sum.get("cost", 0.0)),
                        "faith_judge_prompt_tokens_total": int(
                            faith_usage_sum.get("prompt_tokens", 0)
                        ),
                        "faith_judge_completion_tokens_total": int(
                            faith_usage_sum.get("completion_tokens", 0)
                        ),
                        "faith_judge_total_tokens_total": int(
                            faith_usage_sum.get("total_tokens", 0)
                        ),
                        "faith_judge_cost_usd_total": float(faith_usage_sum.get("cost", 0.0)),
                        "route_total_time_s": round(float(time.perf_counter() - route_t0), 6),
                        "answer_llm_time_s": round(float(answer_llm_time_s), 6),
                        "faith_llm_time_s": round(float(faith_llm_time_s), 6),
                        "consecutive_errors": consecutive_errors,
                    }
                )

                _flush_results()

                # Stop the run on a provider rate-limit response.
                if is_rate_limited:
                    n_rate_limit += 1
                    print(
                        "[FATAL] Rate limit (429) received. Stopping the run "
                        "to avoid repeated failed requests. "
                        "Restart later, increase pauses, or reduce rpm-limit."
                    )
                    raise SystemExit(2)

                if consecutive_errors > max_consecutive_errors:
                    print(
                        f"[FATAL] Too many consecutive errors: {consecutive_errors} > {max_consecutive_errors}. "
                        "Stopping the run to avoid wasting API quota. "
                        "Check the network, credentials, and judge-model availability; restart with --resume."
                    )
                    raise SystemExit(3)

                print(
                    "[INFO] Route evaluation error recorded; continuing with the next route. "
                    "(If many errors occur, rerun later with --resume.)"
                )
                continue

        # Aggregated metric summary
        eval_items = [
            r
            for r in results
            if (isinstance(r, dict) and isinstance(r.get("faithfulness_score"), (int, float)))
        ]

        print("\n=== RUN HEALTH (errors/retries) ===")
        print(f"Routes to process in this run: {n_to_process}")
        print(f"Successfully evaluated: {n_success}")
        print(
            f"Errors: {n_errors} (network/transient: {n_transient_errors}, rate limit: {n_rate_limit})"
        )
        print(
            f"Total retries: {n_retries_total} "
            f"(invalid JSON: {n_retries_invalid_json}, transient: {n_retries_transient})"
        )
        print(f"Consecutive error threshold: {max_consecutive_errors}")
        # Total judge cost (best-effort, only when OpenRouter returns cost)
        # Prefer per-route aggregated totals (incl. retries) when available.
        judge_costs = []
        for r in results:
            if not isinstance(r, dict):
                continue

            # Usage totals across judge calls.
            for k in ("answer_judge_cost_usd_total", "faith_judge_cost_usd_total"):
                v = r.get(k)
                if isinstance(v, (int, float)):
                    judge_costs.append(float(v))

            # Backward-compatible fallback (last-attempt only)
            if not any(
                k in r for k in ("answer_judge_cost_usd_total", "faith_judge_cost_usd_total")
            ):
                for k in ("answer_judge_cost_usd", "faith_judge_cost_usd"):
                    v = r.get(k)
                    if isinstance(v, (int, float)):
                        judge_costs.append(float(v))

        if judge_costs:
            print(f"Judge cost total (USD): {sum(judge_costs):.6f}")
        if error_type_counts:
            top = sorted(error_type_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            top_str = ", ".join([f"{k}={v}" for k, v in top])
            print(f"Most frequent error types: {top_str}")

        # timing summary
        run_total_time_s = time.perf_counter() - run_t0
        print(f"Run total time (s): {run_total_time_s:.1f}")

        timing_items = [
            r
            for r in results
            if isinstance(r, dict) and isinstance(r.get("route_total_time_s"), (int, float))
        ]
        if timing_items:
            avg_route = sum(float(r["route_total_time_s"]) for r in timing_items) / len(
                timing_items
            )
            avg_answer_llm = sum(
                float(r.get("answer_llm_time_s", 0.0)) for r in timing_items
            ) / len(timing_items)
            avg_faith_llm = sum(float(r.get("faith_llm_time_s", 0.0)) for r in timing_items) / len(
                timing_items
            )
            print(f"Avg route time (s): {avg_route:.1f}")
            print(f"Avg AnswerRelevancy LLM time (s): {avg_answer_llm:.2f}")
            print(f"Avg Faithfulness LLM time (s): {avg_faith_llm:.2f}")

        if eval_items:
            n_eval = len(eval_items)
            avg_faith = sum(r["faithfulness_score"] for r in eval_items) / n_eval
            pass_faith = (
                sum(
                    1
                    for r in eval_items
                    if float(r["faithfulness_score"]) >= float(args.faithfulness_threshold)
                )
                / n_eval
            )

            print("\n=== SUMMARY (successfully evaluated routes) ===")
            print(f"Routes evaluated: {n_eval}")
            print(f"Avg Expert Faithfulness: {avg_faith:.3f} (pass rate {pass_faith * 100:.1f}%)")

        if args.out_json:
            _flush_results()
            print(f"Results saved to: {args.out_json}")
        else:
            print("\nDone. Complete results (in memory; not saved to disk):")
            print(json.dumps(results, ensure_ascii=False, indent=2))
    finally:
        # best-effort cleanup to reduce Unclosed client session warnings
        try:
            # Some models expose async cleanup
            aclose_fn = getattr(judge_model, "aclose", None)
            if callable(aclose_fn):
                import asyncio

                try:
                    asyncio.run(aclose_fn())
                except RuntimeError:
                    # If an event loop is already running, fallback to sync close
                    pass
        except Exception:
            pass
        try:
            close_fn = getattr(judge_model, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass


if __name__ == "__main__":
    main()
