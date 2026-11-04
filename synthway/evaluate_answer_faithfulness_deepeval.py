"""
evaluate_rag_answer_faithfulness.py

Evaluate generated routes (predictions.py outputs)
using DeepEval RAG metrics:
  * AnswerRelevancyMetric  — relevance of the answer to the original request;
  * FaithfulnessMetric     — grounding in the retrieved context and consistency with it.

Expected data layout:

For each task (case_id) under --routes-dir:
  case_dir/
    retrieval_context.json        # Saved by predictions.py
    route_1.json                  # One JSON file per route
    route_2.json
    ...

retrieval_context.json format:
  {
    "composition": "Ag80Fe20",
    "method": "polyol reduction" | null,
    "rag_topk": 3,
    "query": "...",
    "retrieval_context": [
        "original RAG hit 1 ...",
        "original RAG hit 2 ...",
        ...
    ]
  }

route_*.json format (saved by utils.save_routes_to_directory):
  {
    "raw_text": "human-readable, step-by-step protocol...",
    "retrieval_text": "slot-based summary: composition=..., method=..., solvent=..., ..."
  }

Workflow:
  - Recursively traverse all route_*.json files under --routes-dir.
  - Find retrieval_context.json beside each route.
  - Build an LLMTestCase with:
        input             ~ target composition (from retrieval_context.json) + brief task description;
        actual_output     ~ raw_text if available, otherwise retrieval_text;
        retrieval_context ~ list of texts from retrieval_context.json.
  - Compute two DeepEval metrics separately for each route.
  - Write a summary JSON containing results for all routes.

  Supports resuming: --out-json saves results incrementally during evaluation,
  and --resume skips processed routes to continue after a failure or interruption.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass

from typing import Any, Dict, List, Optional
def _usage_to_dict(usage_obj) -> Optional[Dict[str, Any]]:
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

    out: Dict[str, Any] = {}
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

from credentials import load_credentials

# DeepEval imports
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from ss_faithfulness import (
    SSFaithfulnessMetric, SS_FAITHFULNESS_ID, ORIGINAL_FAITHFULNESS_ID,
    build_ss_judge_model, validate_resume_metric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

load_credentials()


def load_retrieval_context_json(case_dir: str) -> tuple[List[str], Optional[str], Optional[str]]:
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
    method: Optional[str],
    raw_text: str,
    retrieval_text: str,
    retrieval_context: List[str],
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
    judge_model: Optional[Any] = None,
) -> AnswerRelevancyMetric:
    """Factory analogous to make_chemical_geval_metric but for AnswerRelevancyMetric."""
    kwargs: Dict[str, Any] = {
        "threshold": threshold,
        "include_reason": True,
    }
    if judge_model is not None:
        kwargs["model"] = judge_model
    return AnswerRelevancyMetric(**kwargs)


def make_faithfulness_metric(
    threshold: float,
    judge_model: Optional[Any] = None,
    *, mode: str = "original", before_call=None,
):
    """Select SS-Faithfulness explicitly; preserve original evaluations by default."""
    if mode == "ss":
        return SSFaithfulnessMetric(threshold, judge_model, before_call=before_call)
    if mode != "original":
        raise ValueError(f"Unknown Faithfulness mode: {mode}")
    kwargs: Dict[str, Any] = {
        "threshold": threshold,
        "include_reason": True,
    }
    if judge_model is not None:
        kwargs["model"] = judge_model
    return FaithfulnessMetric(**kwargs)


@dataclass
class RouteRecord:
    file: str
    path: str
    composition: Optional[str]
    raw_text: str
    retrieval_text: str
    retrieval_context: List[str]


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
            "vendor/model identifiers (including google/gemini-...) and openrouter:vendor/model "
            "use OpenRouter and OPENROUTER_API_KEY. Gemini without a provider prefix "
            "uses GeminiModel and GEMINI_API_KEY/GOOGLE_API_KEY. "
            "Other identifiers are passed to DeepEval; if omitted, its default configuration is used."
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
        default=0.7,
        help="FaithfulnessMetric pass threshold (0–1). Default: 0.7.",
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
        help=(
            "Limit metric calls per minute (requests per minute). "
            "No limit is applied if <= 0."
        ),
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
            "Path to the JSON file for per-route results. "
            "If omitted, print results to stdout only."
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
    parser.add_argument(
        "--faithfulness-mode", choices=("original", "ss"), default="original",
        help="original: DeepEval Faithfulness metric used in the initial screening; ss: synthesis-specific expert formula. The GUI and live demo use ss.",
    )
    return parser.parse_args()




def parse_composition_from_retrieval(retrieval_text: str) -> Optional[str]:
    if not retrieval_text:
        return None
    m = re.search(r"composition\s*[:=]\s*([^\|,\n]+)", retrieval_text, flags=re.IGNORECASE)
    if not m:
        return None
    comp = m.group(1).strip()
    comp = re.sub(r"\s+", " ", comp)
    return comp or None




# Load routes and retrieval_context


import glob  # Import after function definitions to keep the file header concise


def load_route_records(routes_dir: str) -> List[RouteRecord]:
    """
    Recursively find all route_*.json files in routes_dir and load
    retrieval_context.json from the same directory for each route.
    """
    pattern = os.path.join(routes_dir, "**", "route_*.json")
    paths = sorted([p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)])

    if not paths:
        raise RuntimeError(f"No route_*.json files found in {routes_dir!r}.")

    records: List[RouteRecord] = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        retrieval_text = (data.get("retrieval_text") or "").strip()
        raw_text = (data.get("raw_text") or "").strip()

        case_dir = os.path.dirname(path)
        rc_path = os.path.join(case_dir, "retrieval_context.json")
        retrieval_context: List[str] = []
        composition: Optional[str] = None
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

    DeepEval's metrics accept a custom model object as long as it can be called to
    produce a completion. We implement both sync and async interfaces.

    Uses the official `openai` python client but points it to OpenRouter via base_url.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
        temperature: float = 0.0,
        force_json: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.force_json = force_json
        self.last_usage: Optional[Dict[str, Any]] = None
        self.last_cost_usd: Optional[float] = None
        # Per-call accounting history (covers cases where a metric performs multiple LLM calls).
        self.usage_log: List[Dict[str, Any]] = []

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(
                "Failed to import 'openai'. Install/update it: pip install -U openai"
            ) from e

        headers: Dict[str, str] = {}
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

    def generate(self, prompt: str, **_: Any) -> str:
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

        # Prefer JSON mode when supported (OpenRouter supports response_format for many models).
        # If the backend/model rejects it, fall back to a normal call.
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if getattr(self, "force_json", False):
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            # Fallback: retry without response_format for compatibility.
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)

        # --- OpenRouter usage accounting (judge) ---
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

        return (resp.choices[0].message.content or "").strip()

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, **kwargs)


def build_judge_model(eval_model: Optional[str]) -> Optional[Any]:
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
                model_name = model_name[len(prefix):]
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



# --- LLMTestCase from RouteRecord ---

# NOTE: DeepEval's judge LLM is expected to output valid JSON.
# Some route texts include LaTeX fragments like "\\text{...}" or "\\cdot" or file paths
# that can sometimes leak into judge outputs and break JSON parsing (Invalid \\escape).
# We keep the original generation output intact, but feed a slightly sanitized version
# to the judge to reduce flaky JSONDecodeError.

_LATEX_MATH_RE = re.compile(r"\$[^$]*\$")

def sanitize_for_judge(text: str) -> str:
    r"""Lightweight sanitization for judge-facing text.

    - Removes inline LaTeX math blocks `$...$` (common source of backslashes like \text, \cdot).
    - Escapes backslashes so that if the judge copies snippets into its JSON, it stays parseable.

    This is intentionally minimal to avoid changing the semantics of the protocol.
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

    # Keep this close to the actual user input
    input_text = composition

    actual_output = route.raw_text or route.retrieval_text
    actual_output = sanitize_for_judge(actual_output)

    return LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=route.retrieval_context,
    )


# --- Main workflow ---


def main() -> None:
    args = parse_args()
    run_t0 = time.perf_counter()
    metric_id = SS_FAITHFULNESS_ID if args.faithfulness_mode == "ss" else ORIGINAL_FAITHFULNESS_ID
    if args.out_json and args.resume and os.path.isfile(args.out_json):
        with open(args.out_json, encoding="utf-8") as stream:
            validate_resume_metric(json.load(stream), metric_id)

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

        judge_model = (build_ss_judge_model(args.eval_model) if args.faithfulness_mode == "ss"
                       else build_judge_model(args.eval_model))

        answer_metric = make_answer_relevancy_metric(
            threshold=args.answer_threshold,
            judge_model=judge_model,
        )
        faithfulness_metric = make_faithfulness_metric(
            threshold=args.faithfulness_threshold,
            judge_model=judge_model,
            mode=args.faithfulness_mode,
            before_call=lambda: _wait_if_needed(),
        )
        print(f"Faithfulness metric: {metric_id}")

        existing_results: List[Dict[str, Any]] = []
        processed_paths: set[str] = set()
        if args.out_json and args.resume and os.path.isfile(args.out_json):
            try:
                with open(args.out_json, "r", encoding="utf-8") as f:
                    existing_results = json.load(f) or []
                # Recompute pass flags for legacy result files so SUMMARY matches score>=threshold.
                for r in existing_results:
                    if (
                        isinstance(r, dict)
                        and isinstance(r.get("answer_score"), (int, float))
                        and isinstance(r.get("faithfulness_score"), (int, float))
                    ):
                        r["answer_passed"] = float(r["answer_score"]) >= float(args.answer_threshold)
                        r["faithfulness_passed"] = float(r["faithfulness_score"]) >= float(args.faithfulness_threshold)

                # In resume mode, skip ONLY successfully evaluated routes (not errors).
                processed_paths = {
                    r["path"]
                    for r in existing_results
                    if (
                        isinstance(r, dict)
                        and "path" in r
                        and isinstance(r.get("answer_score"), (int, float))
                        and isinstance(r.get("faithfulness_score"), (int, float))
                        and "error" not in r
                    )
                }
                print(f"Resume mode: previously successful routes found: {len(processed_paths)}")
            except Exception as e:
                print(
                    f"[WARN] Failed to load the existing results file "
                    f"{args.out_json!r}: {e}"
                )
                existing_results = []
                processed_paths = set()

        if processed_paths:
            records = [r for r in records if r.path not in processed_paths]
            print(
                f"Routes remaining for evaluation after excluding previously evaluated routes: "
                f"{len(records)}"
            )

        rpm_limit: int = getattr(args, "rpm_limit", 15)
        request_timestamps: List[float] = []

        results: List[Dict[str, Any]] = list(existing_results)

        # --- run health counters (errors / retries) ---
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
        error_type_counts: Dict[str, int] = {}

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
            "Running DeepEval RAG metrics "
            f"(answer_thr={args.answer_threshold}, faith_thr={args.faithfulness_threshold}, "
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

            # LLM-call time excludes our fixed sleeps / local rpm waits.
            answer_llm_time_s = 0.0
            faith_llm_time_s = 0.0

            # per-route counters
            route_retries_invalid_json = 0
            route_retries_transient = 0

            # --- per-route judge usage accumulation (tokens/cost across ALL judge calls, incl. retries) ---
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

            def _accumulate_usage(target: Dict[str, Any], usage: Optional[Dict[str, Any]]) -> None:
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
                    # Some OpenRouter responses may expose cost under alternate keys; be defensive.
                    for alt in ("cost_usd", "total_cost", "price"):
                        ac = usage.get(alt)
                        if isinstance(ac, (int, float)):
                            target["cost"] = float(target.get("cost", 0.0)) + float(ac)
                            break

            def _accumulate_usage_from_log(target: Dict[str, Any], entries: List[Dict[str, Any]]) -> None:
                """Accumulate usage across multiple judge calls (best-effort)."""
                if not entries:
                    return
                for ent in entries:
                    if isinstance(ent, dict) and ent:
                        _accumulate_usage(target, ent)

            try:
                def _is_invalid_json_exc(exc: Exception) -> bool:
                    msg = str(exc).lower()
                    if "invalid json" in msg or "jsondecodeerror" in msg:
                        return True
                    ctx = getattr(exc, "__context__", None)
                    if ctx is not None and ("json" in str(ctx).lower() or "jsondecodeerror" in str(ctx).lower()):
                        return True
                    return False

                def _is_rate_limit_exc(exc: Exception) -> bool:
                    """Best-effort detection of HTTP 429 / rate limit errors from various SDKs."""
                    # Direct message match
                    msg = str(exc).lower()
                    if "429" in msg or "rate limit" in msg or "ratelimit" in msg or "too many requests" in msg:
                        return True

                    # Some SDKs expose a status_code attribute
                    status = getattr(exc, "status_code", None)
                    if status == 429:
                        return True

                    # Walk cause/context chain
                    for nested in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
                        if nested is None:
                            continue
                        nmsg = str(nested).lower()
                        if "429" in nmsg or "rate limit" in nmsg or "ratelimit" in nmsg or "too many requests" in nmsg:
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

                    if isinstance(exc, AttributeError) and "statements" in msg and "nonetype" in msg:
                        return True

                    for nested in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
                        if nested is None:
                            continue
                        nmsg = str(nested).lower()
                        if any(m in nmsg for m in transient_markers):
                            return True
                        if isinstance(nested, AttributeError) and "statements" in nmsg and "nonetype" in nmsg:
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

                # Answer relevancy
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
                        answer_metric.measure(tc)
                        answer_llm_time_s += (time.perf_counter() - _t_call)

                        # Count attempt and accumulate usage for this attempt.
                        # Use per-call usage_log to capture *all* judge calls that may happen inside measure().
                        answer_attempts += 1
                        answer_usage = None
                        if judge_model is not None and hasattr(judge_model, "usage_log"):
                            try:
                                entries = list(getattr(judge_model, "usage_log"))[usage_log_before:]
                                answer_judge_calls_total += len(entries)
                                _accumulate_usage_from_log(answer_usage_sum, entries)
                                if entries:
                                    answer_usage = dict(entries[-1])
                            except Exception:
                                pass
                        elif judge_model is not None and hasattr(judge_model, "last_usage"):
                            try:
                                ju = getattr(judge_model, "last_usage")
                                if isinstance(ju, dict) and ju:
                                    answer_usage = dict(ju)
                                    _accumulate_usage(answer_usage_sum, answer_usage)
                                    answer_judge_calls_total += 1
                            except Exception:
                                pass

                        answer_total_time_s = time.perf_counter() - route_t0
                        break
                    except Exception as e:
                        # Count time spent in the judge call for failed attempts as well
                        try:
                            answer_llm_time_s += (time.perf_counter() - _t_call)
                        except Exception:
                            pass
                        attempt += 1
                        # Even on failure, count this attempt and accumulate any usage exposed by the judge.
                        answer_attempts += 1
                        if judge_model is not None and hasattr(judge_model, "usage_log"):
                            try:
                                entries = list(getattr(judge_model, "usage_log"))[usage_log_before:]
                                answer_judge_calls_total += len(entries)
                                _accumulate_usage_from_log(answer_usage_sum, entries)
                            except Exception:
                                pass
                        elif judge_model is not None and hasattr(judge_model, "last_usage"):
                            try:
                                ju = getattr(judge_model, "last_usage")
                                if isinstance(ju, dict) and ju:
                                    _accumulate_usage(answer_usage_sum, dict(ju))
                                    answer_judge_calls_total += 1
                            except Exception:
                                pass
                        if attempt <= max_json_retries and _is_invalid_json_exc(e):
                            n_retries_total += 1
                            n_retries_invalid_json += 1
                            route_retries_invalid_json += 1
                            print(f"[WARN] AnswerRelevancy judge returned invalid JSON; retry {attempt}/{max_json_retries}...")
                            time.sleep(2)
                            continue
                        if attempt <= (max_json_retries + max_transient_retries) and _is_transient_exc(e):
                            n_retries_total += 1
                            n_retries_transient += 1
                            route_retries_transient += 1
                            backoff = min(30, 2 * attempt)
                            print(
                                f"[WARN] AnswerRelevancy transient error; retry {attempt}/{max_json_retries + max_transient_retries} "
                                f"after {backoff}s... ({type(e).__name__}: {e})"
                            )
                            time.sleep(backoff)
                            continue
                        raise

                # Faithfulness
                attempt = 0
                while True:
                    try:
                        usage_log_before = (
                            len(getattr(judge_model, "usage_log", []))
                            if judge_model is not None and hasattr(judge_model, "usage_log")
                            else 0
                        )
                        if args.faithfulness_mode != "ss":
                            _wait_if_needed()
                        _t_call = time.perf_counter()
                        faithfulness_metric.measure(tc)
                        faith_llm_time_s += (time.perf_counter() - _t_call)

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
                            faith_llm_time_s += (time.perf_counter() - _t_call)
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
                            print(f"[WARN] Faithfulness judge returned invalid JSON; retry {attempt}/{max_json_retries}...")
                            time.sleep(2)
                            continue
                        if attempt <= (max_json_retries + max_transient_retries) and _is_transient_exc(e):
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
                    "answer_score": answer_metric.score,
                    "answer_threshold": answer_metric.threshold,
                    "answer_passed": (answer_metric.score is not None) and (float(answer_metric.score) >= float(args.answer_threshold)),
                    "answer_reason": answer_metric.reason,
                    "faithfulness_score": faithfulness_metric.score,
                    "faithfulness_threshold": faithfulness_metric.threshold,
                    "faithfulness_passed": (faithfulness_metric.score is not None) and (float(faithfulness_metric.score) >= float(args.faithfulness_threshold)),
                    "faithfulness_reason": faithfulness_metric.reason,
                    **(faithfulness_metric.result_fields() if args.faithfulness_mode == "ss"
                       else {"faithfulness_metric": ORIGINAL_FAITHFULNESS_ID}),
                    "route_retries_invalid_json": route_retries_invalid_json,
                    "route_retries_transient": route_retries_transient,
                    "answer_attempts": answer_attempts,
                    "faith_attempts": faith_attempts,
                    "answer_judge_calls_total": int(answer_judge_calls_total),
                    "faith_judge_calls_total": int(faith_judge_calls_total),
                    "answer_judge_prompt_tokens_total": int(answer_usage_sum.get("prompt_tokens", 0)),
                    "answer_judge_completion_tokens_total": int(answer_usage_sum.get("completion_tokens", 0)),
                    "answer_judge_total_tokens_total": int(answer_usage_sum.get("total_tokens", 0)),
                    "answer_judge_cost_usd_total": float(answer_usage_sum.get("cost", 0.0)),
                    "faith_judge_prompt_tokens_total": int(faith_usage_sum.get("prompt_tokens", 0)),
                    "faith_judge_completion_tokens_total": int(faith_usage_sum.get("completion_tokens", 0)),
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
                    f"AnswerRelevancy:  {answer_metric.score:.3f} "
                    f"({'PASS' if result['answer_passed'] else 'FAIL'}; "
                    f"thr={answer_metric.threshold})\n"
                    f"  reason: {answer_metric.reason}\n"
                    f"{metric_id}: {faithfulness_metric.score:.3f} "
                    f"({'PASS' if result['faithfulness_passed'] else 'FAIL'}; "
                    f"thr={faithfulness_metric.threshold})\n"
                    f"  reason: {faithfulness_metric.reason}\n"
                )

            except Exception as e:
                err_type = type(e).__name__
                base_msg = f"{err_type}: {e}"
                print(
                    f"\n[ERROR] Evaluation of {route.file} failed: {base_msg}"
                )

                # classify error
                is_rate_limited = _is_rate_limit_exc(e)
                is_transient = _is_transient_exc(e)
                category = "rate_limit" if is_rate_limited else ("network" if is_transient else "other")

                n_errors += 1
                error_type_counts[err_type] = error_type_counts.get(err_type, 0) + 1
                consecutive_errors += 1
                if is_transient:
                    n_transient_errors += 1

                # Try extracting a more specific root cause (e.g., ClientError inside RetryError)
                cause = getattr(e, "__cause__", None)
                context_exc = getattr(e, "__context__", None)
                error_details: Dict[str, Any] = {"base": base_msg}

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
                        "faithfulness_metric": metric_id,
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
                        "answer_judge_prompt_tokens_total": int(answer_usage_sum.get("prompt_tokens", 0)),
                        "answer_judge_completion_tokens_total": int(answer_usage_sum.get("completion_tokens", 0)),
                        "answer_judge_total_tokens_total": int(answer_usage_sum.get("total_tokens", 0)),
                        "answer_judge_cost_usd_total": float(answer_usage_sum.get("cost", 0.0)),
                        "faith_judge_prompt_tokens_total": int(faith_usage_sum.get("prompt_tokens", 0)),
                        "faith_judge_completion_tokens_total": int(faith_usage_sum.get("completion_tokens", 0)),
                        "faith_judge_total_tokens_total": int(faith_usage_sum.get("total_tokens", 0)),
                        "faith_judge_cost_usd_total": float(faith_usage_sum.get("cost", 0.0)),
                        "route_total_time_s": round(float(time.perf_counter() - route_t0), 6),
                        "answer_llm_time_s": round(float(answer_llm_time_s), 6),
                        "faith_llm_time_s": round(float(faith_llm_time_s), 6),
                        "consecutive_errors": consecutive_errors,
                    }
                )

                _flush_results()

                # If we're rate-limited (HTTP 429), stop the run early.
                # No point in burning CPU on the remaining routes when the provider is throttling.
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

        # --- Aggregated metric summary ---
        eval_items = [
            r for r in results
            if (
                isinstance(r, dict)
                and isinstance(r.get("answer_score"), (int, float))
                and isinstance(r.get("faithfulness_score"), (int, float))
            )
        ]

        print("\n=== RUN HEALTH (errors/retries) ===")
        print(f"Routes to process in this run: {n_to_process}")
        print(f"Successfully evaluated: {n_success}")
        print(f"Errors: {n_errors} (network/transient: {n_transient_errors}, rate limit: {n_rate_limit})")
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

            # New aggregated fields (preferred)
            for k in ("answer_judge_cost_usd_total", "faith_judge_cost_usd_total"):
                v = r.get(k)
                if isinstance(v, (int, float)):
                    judge_costs.append(float(v))

            # Backward-compatible fallback (last-attempt only)
            if not any(k in r for k in ("answer_judge_cost_usd_total", "faith_judge_cost_usd_total")):
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

        # --- timing summary ---
        run_total_time_s = time.perf_counter() - run_t0
        print(f"Run total time (s): {run_total_time_s:.1f}")

        timing_items = [r for r in results if isinstance(r, dict) and isinstance(r.get("route_total_time_s"), (int, float))]
        if timing_items:
            avg_route = sum(float(r["route_total_time_s"]) for r in timing_items) / len(timing_items)
            avg_answer_llm = sum(float(r.get("answer_llm_time_s", 0.0)) for r in timing_items) / len(timing_items)
            avg_faith_llm = sum(float(r.get("faith_llm_time_s", 0.0)) for r in timing_items) / len(timing_items)
            print(f"Avg route time (s): {avg_route:.1f}")
            print(f"Avg AnswerRelevancy LLM time (s): {avg_answer_llm:.2f}")
            print(f"Avg Faithfulness LLM time (s): {avg_faith_llm:.2f}")

        if eval_items:
            n_eval = len(eval_items)
            avg_answer = sum(r["answer_score"] for r in eval_items) / n_eval
            avg_faith = sum(r["faithfulness_score"] for r in eval_items) / n_eval
            pass_answer = sum(1 for r in eval_items if float(r["answer_score"]) >= float(args.answer_threshold)) / n_eval
            pass_faith = sum(1 for r in eval_items if float(r["faithfulness_score"]) >= float(args.faithfulness_threshold)) / n_eval

            print("\n=== SUMMARY (successfully evaluated routes) ===")
            print(f"Routes evaluated: {n_eval}")
            print(f"Avg AnswerRelevancy:  {avg_answer:.3f} "
                  f"(pass rate {pass_answer * 100:.1f}%)")
            print(f"Avg Faithfulness:     {avg_faith:.3f} "
                  f"(pass rate {pass_faith * 100:.1f}%)")

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
