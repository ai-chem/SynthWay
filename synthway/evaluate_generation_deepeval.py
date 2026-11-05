r"""Evaluate generated routes with Chemical Plausibility & Consistency G-Eval.

Read route_*.json recursively under --routes-dir. Each route contains retrieval_text
and raw_text. Target composition and method come from the sibling retrieval_context.json,
with retrieval_text as a fallback when metadata is missing.
The judge receives the target request and both route representations.

The metric checks chemical plausibility, consistency between retrieval_text and raw_text,
and completeness of the main synthesis stages.

Example after generation, from the repository root:
    python synthway/evaluate_generation_deepeval.py --routes-dir outputs/generation \
        --eval-model gemini-2.0-flash-001 --threshold 0.8 \
        --out-json outputs/generation_eval_Ag.json
"""

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
import argparse
import glob
import json
import os
import re
import time
import traceback

# OpenRouter usage/cost accounting helper
from typing import Any

from openai import OpenAI


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


from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM


class OpenRouterChatModel(DeepEvalBaseLLM):
    """Universal DeepEval LLM wrapper for any OpenRouter chat model.

    OpenRouter exposes many vendors/models via an OpenAI-compatible Chat Completions API.
    This wrapper lets GEval use *any* such model as the judge.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str | None = None,
        x_title: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.last_response_text: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.last_cost_usd: float | None = None
        # Per-call accounting history (covers cases where a metric performs multiple LLM calls).
        self.usage_log: list[dict[str, Any]] = []

        # OpenRouter recommends HTTP-Referer and X-Title headers to identify the client
        referer = http_referer or os.getenv("OPENROUTER_HTTP_REFERER") or "https://localhost"
        title = x_title or os.getenv("OPENROUTER_X_TITLE") or "SynthWay-RAG-eval"
        headers = {
            "HTTP-Referer": referer,
            "X-Title": title,
        }

        # OpenAI-compatible client that talks to OpenRouter
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=headers,
        )

    def load_model(self):
        return self._client

    def generate(self, prompt: str, **_: Any) -> str:
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        # OpenRouter usage accounting (judge)
        usage_dict = _usage_to_dict(getattr(response, "usage", None)) or {}
        if "cost" not in usage_dict:
            try:
                extra_cost = getattr(response, "cost", None)
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
        content = response.choices[0].message.content
        text = "" if content is None else str(content)
        self.last_response_text = text
        if not text.strip():
            raise RuntimeError("Empty response from OpenRouter judge")
        return text

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, **kwargs)

    def get_model_name(self) -> str:
        return f"openrouter:{self.model}"


from credentials import load_credentials
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_credentials()

# Parse arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated synthesis routes using DeepEval (G-Eval)."
    )
    parser.add_argument(
        "--routes-dir",
        required=True,
        help="Root directory of routes (predictions.py / save_routes_to_directory output): may contain route JSON files directly or in subdirectories.",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help=(
            "Judge model for G-Eval. "
            "vendor/model and openrouter:vendor/model identifiers use OpenRouter "
            "and OPENROUTER_API_KEY; Gemini without a provider prefix uses "
            "GEMINI_API_KEY/GOOGLE_API_KEY. If omitted, use the DeepEval default configuration."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Metric pass threshold (0–1). Default: 0.5.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help=(
            "Path to the JSON file for per-route results. If omitted, print results to stdout only."
        ),
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
            "Maximum judge-model requests per minute (requests per minute). "
            "No limit is applied if <= 0."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If set and --out-json points to an existing file, "
            "skip routes already evaluated successfully (resume mode)."
        ),
    )
    return parser.parse_args()


# Helper functions


def parse_composition_from_retrieval(retrieval_text: str) -> str | None:
    """
    Extract composition from retrieval_text of the form:
      'composition: Ag | method: ... | solvent: ... | ...'
    """
    if not retrieval_text:
        return None

    m = re.search(r"composition:\s*([^|]+)", retrieval_text)
    if not m:
        return None

    comp = m.group(1).strip()
    comp = re.sub(r"\s+", " ", comp)
    return comp or None


def parse_method_from_retrieval(retrieval_text: str) -> str | None:
    if not retrieval_text:
        return None
    m = re.search(r"method:\s*([^|]+)", retrieval_text)
    if not m:
        return None
    method = m.group(1).strip()
    method = re.sub(r"\s+", " ", method)
    return method or None


def load_retrieval_context_for_route(route_path: str) -> dict[str, Any]:
    """
    Load retrieval_context.json from the same directory as route_*.json.
    Return the context dictionary, or an empty dictionary if the file is missing/unreadable.
    """
    case_dir = os.path.dirname(route_path)
    context_path = os.path.join(case_dir, "retrieval_context.json")

    if not os.path.exists(context_path):
        return {}

    try:
        with open(context_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_routes_from_directory(routes_dir: str) -> list[dict[str, Any]]:
    """
    Recursively load all *.json files from a directory and its subdirectories; return a list of dictionaries:
      {
        "file": "route_1.json",
        "path": "/abs/path/route_1.json",
        "retrieval_text": "...",
        "raw_text": "...",
        "composition": "Ag",              # From the sibling retrieval_context.json
        "method": "sol-gel",              # From the sibling retrieval_context.json
        "retrieval_context_path": "/abs/path/retrieval_context.json"
      }
    """
    pattern = os.path.join(routes_dir, "**", "route_*.json")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        raise RuntimeError(f"No route_*.json files found in {routes_dir!r}.")

    routes: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        retrieval_text = data.get("retrieval_text", "")
        raw_text = data.get("raw_text", "")

        retrieval_context = load_retrieval_context_for_route(path)

        # Prefer retrieval_context.json for the judge INPUT,
        # rather than the route/retrieval_text itself.
        comp = retrieval_context.get("composition")
        method = retrieval_context.get("method")

        # Fallback for older/incomplete directories without retrieval_context.json
        if not comp:
            comp = parse_composition_from_retrieval(retrieval_text)
        if not method:
            method = parse_method_from_retrieval(retrieval_text)

        routes.append(
            {
                "file": os.path.basename(path),
                "path": path,
                "retrieval_text": retrieval_text,
                "raw_text": raw_text,
                "composition": comp,
                "method": method,
                "retrieval_context_path": os.path.join(
                    os.path.dirname(path), "retrieval_context.json"
                ),
            }
        )

    return routes


def build_judge_model(eval_model: str | None) -> Any | None:
    """
    Prepare the judge model for G-Eval.
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
        model_id = eval_model_str.split(":", 1)[1].strip()

        print(
            "Using OpenRouterChatModel (openrouter: prefix) as the judge model: "
            f"model={model_id!r}, base_url={openrouter_base_url!r}"
        )

        return OpenRouterChatModel(
            model=model_id,
            api_key=openrouter_key,
            base_url=openrouter_base_url,
            temperature=0,
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

        print(
            "Using OpenRouterChatModel (google/gemini treated as OpenRouter) as the judge model: "
            f"model={eval_model_str!r}, base_url={openrouter_base_url!r}"
        )

        return OpenRouterChatModel(
            model=eval_model_str,
            api_key=openrouter_key,
            base_url=openrouter_base_url,
            temperature=0,
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

        print(
            "Using OpenRouterChatModel through OpenRouter as the judge model: "
            f"model={eval_model_str!r}, base_url={openrouter_base_url!r}"
        )

        return OpenRouterChatModel(
            model=eval_model_str,
            api_key=openrouter_key,
            base_url=openrouter_base_url,
            temperature=0,
        )

    # 3) Gemini through the official GeminiModel (Google GenAI SDK)
    #    Expect a model such as 'gemini-2.0-flash-001' without the 'google/' prefix.
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

        return GeminiModel(model_name=model_name, api_key=gem_api, temperature=0)

    # 4) Fallback: retain the string identifier for deepeval to resolve
    print(f"Using string judge model identifier: {eval_model_str!r}")
    return eval_model_str


def make_chemical_geval_metric(
    threshold: float,
    judge_model: Any | None = None,
) -> GEval:
    """
    Build the G-Eval metric for this task:
      - chemical plausibility,
      - internal consistency of retrieval_text / raw_text,
      - protocol completeness.
    """
    criteria = (
        "You are an expert in inorganic and materials chemistry. "
        "Evaluate the proposed synthesis route for the given target composition and method.\n\n"
        "The route is provided in two synchronized views:\n"
        "  1) 'retrieval_text' — a slot-based, machine-readable summary (composition, method, solvent, temperature, pH, additives, precursors, seeds, crystal_system, particle shape and size, etc.).\n"
        "  2) 'raw_text' — a human-readable, step-by-step synthesis description.\n\n"
        "IMPORTANT EVIDENCE RULE: Do NOT trust confidence, jargon, or PDF reference numbers. PDF reference numbers are NOT evidence that a phase exists or was obtained.\n\n"
        "Score from 0.0 to 1.0 how GOOD this route is along the following axes:\n"
        "  1) Chemical plausibility — reagents, solvents, temperatures, pH and operations are realistic for laboratory synthesis of the stated inorganic material within the stated method. "
        "     IMPORTANT: If there is at least ONE serious chemical/conditions mistake that would realistically prevent the claimed material/phase from forming (or makes the route chemically impossible), you MUST penalize heavily (typically score <= 0.25), even if the rest of the text is consistent and complete. Treat a single serious chemistry mistake as near-fatal, not as a minor issue. "
        "     Examples of serious chemistry mistakes (any ONE of these is near-fatal):\n"
        "     - Redox feasibility failure: the route claims an ELEMENTAL METAL or METALLIC ALLOY starting from OXIDES/SALTS, but provides NO plausible reducing agent and NO reducing atmosphere/step (e.g., only Ar/air with oxides). Inert gas alone is not a reducing agent.\n"
        "     - Implausible product class/stoichiometry: the claimed phase/product looks chemically implausible for the stated elements/precursors (e.g., exotic/rare compounds asserted without a plausible mechanism). If the product seems hallucinated or unsupported by the described chemistry, penalize heavily.\n"
        "     - Thermal/chemical stability mismatch: conditions likely decompose/volatilize/destroy key precursors or the claimed matrix (e.g., volatile sulfides/halides at extreme temperatures without containment/pressure control), making the claimed composite/phase unlikely to form or survive.\n"
        "     - Method/setup contradiction: conditions clearly do not fit the stated synthesis regime (e.g., hydrothermal claimed but no solvent/autoclave; CVD claimed but no gas delivery/flow/pressure control; sealed reactor + highly reactive metal powder at very high temperature with no pressure/safety controls).\n"
        "     - Conditions that consume/remove essential reagents needed for the target (e.g., oxidizing atmosphere when reduction is required; aggressive etching/acid that would dissolve the claimed product without a protection step).\n"
        "  2) Internal consistency — 'raw_text' does not contradict 'retrieval_text'; key slots (especially composition and method, as well as solvent, temperature, precursors, additives, seeds, crystal system, particle shape and size) match between the two views or are at least compatible. "
        "     IMPORTANT: A hard mismatch in core identifiers is near-fatal and should force score <= 0.25 (e.g., different composition, different method, different core precursor set, or incompatible core temperature/solvent/atmosphere such that both views cannot be true).\n"
        "  3) Completeness — the protocol describes all major stages: preparation of initial solutions or mixtures, main synthesis step (heating, aging, hydrothermal treatment, calcination, deposition, etc.), and product isolation (washing, drying, calcining/annealing/collection) in a way that a trained chemist could follow. "
        "     Where applicable, it clearly specifies enabling parameters (atmosphere composition/flow, containment, pressure for CVD/hydrothermal, critical ramp/hold times) and expected product characteristics (crystal system/phase, particle shape and size).\n"
        "  4) Safety and best practices — there should be no obviously unsafe or nonsense instructions. "
        "     IMPORTANT: Any clearly unsafe instruction with high accident risk should strongly lower the score and can be near-fatal (<= 0.25), for example: heating a sealed vessel with gas-generating reagents, running H2 at high temperature without adequate controls, or handling toxic/pyrophoric reagents without containment/ventilation.\n\n"
        "Interpret the score as follows:\n"
        "  - 1.0 — route is chemically plausible, internally consistent and reasonably complete.\n"
        "  - ~0.5 — partially plausible, but with noticeable gaps, ambiguities or mild inconsistencies.\n"
        "  - <=0.25 — at least one serious chemical/conditions mistake OR a hard internal contradiction OR a clearly unsafe instruction makes the route chemically implausible or likely to fail.\n"
        "  - 0.0 — chemically impossible, self-contradictory, or so incomplete and confusing that it is unusable as a protocol.\n\n"
        "Return ONLY the final numeric score between 0.0 and 1.0."
    )

    metric_kwargs: dict[str, Any] = dict(
        name="Chemical Plausibility & Consistency",
        criteria=criteria,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )

    if judge_model is not None:
        metric_kwargs["model"] = judge_model

    return GEval(**metric_kwargs)


def build_test_case(route: dict[str, Any]) -> LLMTestCase:
    """
    Build an LLMTestCase from a route dictionary.
    INPUT: target composition and method from retrieval_context.json
            + an explanation of retrieval_text/raw_text.
    OUTPUT: combined text containing both route representations.
    """
    composition = route.get("composition") or "UNKNOWN"
    method = route.get("method") or "UNKNOWN"
    retrieval_text = (route.get("retrieval_text") or "").strip()
    raw_text = (route.get("raw_text") or "").strip()

    input_text = (
        f"Target composition: {composition}.\n"
        f"Target method: {method}.\n"
        "Below is one candidate synthesis route for this composition and method, represented in "
        "two synchronized fields: 'retrieval_text' (slot-style summary) and "
        "'raw_text' (step-by-step description)."
    )

    actual_output = f"retrieval_text:\n{retrieval_text}\n\nraw_text:\n{raw_text}"

    return LLMTestCase(
        input=input_text,
        actual_output=actual_output,
    )


# Main logic


def main() -> None:
    args = parse_args()
    run_t0 = time.perf_counter()

    # Resume mode: load previous results if available
    existing_results: list[dict[str, Any]] = []
    processed_paths = set()

    if args.out_json and args.resume and os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r", encoding="utf-8") as f:
                existing_results = json.load(f)

            for r in existing_results:
                path = r.get("path")
                score = r.get("score")
                # Only cases with numeric scores count as completed
                if path and isinstance(score, (int, float)):
                    processed_paths.add(path)

            print(f"Resume mode: previously successful routes found: {len(processed_paths)}")
        except Exception as e:
            print(f"[WARN] Failed to load the existing results file {args.out_json!r}: {e}")
            existing_results = []
            processed_paths = set()

    print(f"Loading routes from: {args.routes_dir}")
    routes = load_routes_from_directory(args.routes_dir)
    print(f"Total routes found: {len(routes)}")

    # Exclude previously successful routes when resuming
    if processed_paths:
        routes = [r for r in routes if r["path"] not in processed_paths]
        print(
            f"Routes remaining for evaluation after excluding previously evaluated routes: "
            f"{len(routes)}"
        )

    if args.max_cases is not None:
        routes = routes[: args.max_cases]
        print(f"Limiting routes to {len(routes)} using --max-cases.")

    # Resolve the judge model (string / GeminiModel / None)
    judge_model = build_judge_model(args.eval_model)

    # run-level judge usage/cost accumulators (OpenRouter only; best-effort)
    run_judge_usage_sum = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }
    run_judge_calls_total = 0

    # RPM limiter state
    rpm_limit: int = getattr(args, "rpm_limit", 15)
    request_timestamps: list[float] = []

    # Build test cases only for routes to be evaluated now
    test_cases: list[LLMTestCase] = [build_test_case(r) for r in routes]

    # Combine existing and new results
    results: list[dict[str, Any]] = list(existing_results)

    # run health counters (errors / retries)
    n_to_process = len(routes)
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

    print(
        "Running the G-Eval metric "
        f"(threshold={args.threshold}, "
        f"model={'deepeval default' if judge_model is None else repr(args.eval_model)})"
    )

    def _is_invalid_json_exc(exc: Exception) -> bool:
        msg = str(exc).lower()
        if "invalid json" in msg or "jsondecodeerror" in msg:
            return True
        ctx = getattr(exc, "__context__", None)
        if ctx is not None:
            cmsg = str(ctx).lower()
            if "invalid json" in cmsg or "jsondecodeerror" in cmsg:
                return True
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            cmsg = str(cause).lower()
            if "invalid json" in cmsg or "jsondecodeerror" in cmsg:
                return True
        return False

    def _is_rate_limit_exc(exc: Exception) -> bool:
        msg = str(exc).lower()
        if "429" in msg or "rate limit" in msg or "ratelimit" in msg or "too many requests" in msg:
            return True
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        for nested in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
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
            "empty response",
        ]
        if any(m in msg for m in transient_markers):
            return True
        for nested in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
            if nested is None:
                continue
            nmsg = str(nested).lower()
            if any(m in nmsg for m in transient_markers):
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
            for alt in ("cost_usd", "total_cost", "price"):
                ac = usage.get(alt)
                if isinstance(ac, (int, float)):
                    target["cost"] = float(target.get("cost", 0.0)) + float(ac)
                    break

    def _accumulate_usage_from_log(target: dict[str, Any], entries: list[dict[str, Any]]) -> None:
        """Accumulate usage across multiple judge calls (best-effort)."""
        if not entries:
            return
        for ent in entries:
            if isinstance(ent, dict) and ent:
                _accumulate_usage(target, ent)

    for idx, (route, tc) in enumerate(zip(routes, test_cases), start=1):
        # Simple RPM limiter: at most rpm_limit requests per minute
        if rpm_limit and rpm_limit > 0:
            now = time.time()
            # Retain only requests from the last minute
            request_timestamps = [t for t in request_timestamps if now - t < 60.0]

            if len(request_timestamps) >= rpm_limit:
                wait_for = 60.0 - (now - request_timestamps[0])
                if wait_for > 0:
                    print(
                        f"[RATE LIMIT] Local limit of {rpm_limit} req/min reached; "
                        f"pausing for {wait_for:.1f} seconds."
                    )
                    time.sleep(wait_for)

            # Update time after any pause and register the new request
            now = time.time()
            request_timestamps.append(now)

        # Explicit pause before each judge-model request
        print("[RATE LIMIT] Pausing 10 seconds before the next judge-model request.")
        time.sleep(10)

        route_retries_invalid_json = 0
        route_retries_transient = 0
        max_json_retries = 2
        max_transient_retries = 3

        route_t0 = time.perf_counter()
        route_total_time_s = None

        # LLM-call time excludes fixed pauses and local RPM waits.
        judge_llm_time_s = 0.0
        # Per-route judge accounting (OpenRouter only; best-effort)
        judge_attempts = 0
        judge_calls_total = 0
        judge_usage_sum = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
        }

        try:
            metric = make_chemical_geval_metric(
                threshold=args.threshold,
                judge_model=judge_model,
            )

            attempt = 0
            while True:
                try:
                    usage_log_before = (
                        len(getattr(judge_model, "usage_log", []))
                        if judge_model is not None and hasattr(judge_model, "usage_log")
                        else 0
                    )
                    _t_call = time.perf_counter()
                    metric.measure(tc)
                    judge_llm_time_s += time.perf_counter() - _t_call

                    # Count attempt and accumulate all judge calls that happened inside measure().
                    judge_attempts += 1
                    if judge_model is not None and hasattr(judge_model, "usage_log"):
                        try:
                            entries = list(getattr(judge_model, "usage_log"))[usage_log_before:]
                            judge_calls_total += len(entries)
                            _accumulate_usage_from_log(judge_usage_sum, entries)
                        except Exception:
                            pass
                    elif judge_model is not None and hasattr(judge_model, "last_usage"):
                        # Fallback for non-logging judge models
                        try:
                            ju = getattr(judge_model, "last_usage")
                            if isinstance(ju, dict) and ju:
                                judge_calls_total += 1
                                _accumulate_usage(judge_usage_sum, dict(ju))
                        except Exception:
                            pass

                    break
                except Exception as e:
                    # Count time spent in the judge call for failed attempts as well
                    try:
                        judge_llm_time_s += time.perf_counter() - _t_call
                    except Exception:
                        pass
                    attempt += 1
                    # Even on failure, count this attempt and accumulate any usage exposed by the judge.
                    judge_attempts += 1
                    if judge_model is not None and hasattr(judge_model, "usage_log"):
                        try:
                            entries = list(getattr(judge_model, "usage_log"))[usage_log_before:]
                            judge_calls_total += len(entries)
                            _accumulate_usage_from_log(judge_usage_sum, entries)
                        except Exception:
                            pass
                    elif judge_model is not None and hasattr(judge_model, "last_usage"):
                        try:
                            ju = getattr(judge_model, "last_usage")
                            if isinstance(ju, dict) and ju:
                                judge_calls_total += 1
                                _accumulate_usage(judge_usage_sum, dict(ju))
                        except Exception:
                            pass
                    if attempt <= max_json_retries and _is_invalid_json_exc(e):
                        n_retries_total += 1
                        n_retries_invalid_json += 1
                        route_retries_invalid_json += 1
                        print(
                            f"[WARN] GEval judge returned invalid JSON; retry {attempt}/{max_json_retries}... "
                            f"({type(e).__name__}: {e})"
                        )
                        time.sleep(2)
                        continue
                    if attempt <= (max_json_retries + max_transient_retries) and _is_transient_exc(
                        e
                    ):
                        n_retries_total += 1
                        n_retries_transient += 1
                        route_retries_transient += 1
                        backoff = min(30, 2 * attempt)
                        print(
                            f"[WARN] GEval transient error; retry {attempt}/{max_json_retries + max_transient_retries} "
                            f"after {backoff}s... ({type(e).__name__}: {e})"
                        )
                        time.sleep(backoff)
                        continue
                    raise

            route_total_time_s = time.perf_counter() - route_t0
            result = {
                "file": route["file"],
                "path": route["path"],
                "composition": route.get("composition"),
                "method": route.get("method"),
                "score": metric.score,
                "threshold": metric.threshold,
                "passed": metric.is_successful(),
                "reason": metric.reason,
                "route_retries_invalid_json": route_retries_invalid_json,
                "route_retries_transient": route_retries_transient,
                "judge_attempts": int(judge_attempts),
                "judge_calls_total": int(judge_calls_total),
                "judge_prompt_tokens_total": int(judge_usage_sum.get("prompt_tokens", 0)),
                "judge_completion_tokens_total": int(judge_usage_sum.get("completion_tokens", 0)),
                "judge_total_tokens_total": int(judge_usage_sum.get("total_tokens", 0)),
                "judge_cost_usd_total": float(judge_usage_sum.get("cost", 0.0)),
                "route_total_time_s": round(float(route_total_time_s), 6)
                if route_total_time_s is not None
                else None,
                "judge_llm_time_s": round(float(judge_llm_time_s), 6),
            }
            results.append(result)
            # Run-level accounting
            run_judge_calls_total += int(judge_calls_total)
            _accumulate_usage(run_judge_usage_sum, dict(judge_usage_sum))

            n_success += 1
            consecutive_errors = 0

            print(
                f"\n=== [{idx}/{len(routes)}] {route['file']} "
                f"(composition={route.get('composition')}, method={route.get('method')}) ===\n"
                f"Score:   {metric.score:.3f} "
                f"({'PASS' if metric.is_successful() else 'FAIL'}; thr={metric.threshold})\n"
                f"Reason:  {metric.reason}"
            )

        except Exception as e:
            err_type = type(e).__name__
            is_rate_limited = _is_rate_limit_exc(e)
            is_transient = _is_transient_exc(e)
            category = "rate_limit" if is_rate_limited else ("network" if is_transient else "other")

            n_errors += 1
            error_type_counts[err_type] = error_type_counts.get(err_type, 0) + 1
            consecutive_errors += 1
            if is_transient:
                n_transient_errors += 1

            print(f"\n[ERROR] Evaluation of {route['file']} failed with {err_type}: {e}")

            # Try extracting the root cause, including ClientError inside RetryError
            underlying = None
            try:
                from tenacity import (
                    RetryError as TenacityRetryError,
                )  # Used by deepeval for retries

                if isinstance(e, TenacityRetryError):
                    try:
                        underlying = e.last_attempt.exception()
                    except Exception:
                        underlying = None
            except Exception:
                # Ignore if tenacity is not installed separately
                underlying = None

            if underlying is not None:
                print(
                    "[DEBUG] Inner exception inside RetryError: "
                    f"{type(underlying).__name__}: {repr(underlying)}"
                )

            if getattr(e, "__cause__", None):
                print(f"[DEBUG] __cause__: {type(e.__cause__).__name__}: {repr(e.__cause__)}")

            if getattr(e, "__context__", None):
                print(f"[DEBUG] __context__: {type(e.__context__).__name__}: {repr(e.__context__)}")

            # Full traceback to inspect the deepeval / aiohttp stack
            traceback.print_exc()

            judge_preview = None
            if judge_model is not None and hasattr(judge_model, "last_response_text"):
                try:
                    txt = getattr(judge_model, "last_response_text")
                    if isinstance(txt, str) and txt:
                        judge_preview = txt[:500]
                except Exception:
                    judge_preview = None

            reason_msg = f"ERROR during evaluation: {err_type}: {e}"
            if underlying is not None:
                reason_msg += f" | inner: {type(underlying).__name__}: {underlying}"
            if judge_preview:
                reason_msg += f" | judge_preview: {judge_preview}"

            error_result = {
                "file": route["file"],
                "path": route["path"],
                "composition": route.get("composition"),
                "method": route.get("method"),
                "score": None,
                "threshold": args.threshold,
                "passed": False,
                "reason": reason_msg,
                "error_category": category,
                "is_network_error": bool(is_transient),
                "is_rate_limit": bool(is_rate_limited),
                "route_retries_invalid_json": route_retries_invalid_json,
                "route_retries_transient": route_retries_transient,
                "judge_attempts": int(judge_attempts),
                "judge_calls_total": int(judge_calls_total),
                "judge_prompt_tokens_total": int(judge_usage_sum.get("prompt_tokens", 0)),
                "judge_completion_tokens_total": int(judge_usage_sum.get("completion_tokens", 0)),
                "judge_total_tokens_total": int(judge_usage_sum.get("total_tokens", 0)),
                "judge_cost_usd_total": float(judge_usage_sum.get("cost", 0.0)),
                "route_total_time_s": round(float(time.perf_counter() - route_t0), 6),
                "judge_llm_time_s": round(float(judge_llm_time_s), 6),
                "consecutive_errors": consecutive_errors,
            }
            results.append(error_result)
            # Run-level accounting
            run_judge_calls_total += int(judge_calls_total)
            _accumulate_usage(run_judge_usage_sum, dict(judge_usage_sum))

            if is_rate_limited:
                n_rate_limit += 1
                print(
                    "[FATAL] Rate limit (429) received. Stopping the run "
                    "to avoid wasting API quota. Restart later, reduce rpm-limit, or increase pauses."
                )
                break

            if consecutive_errors > max_consecutive_errors:
                print(
                    f"[FATAL] Too many consecutive errors: {consecutive_errors} > {max_consecutive_errors}. "
                    "Stopping the run to avoid wasting API quota. "
                    "Check the network, credentials, and judge-model availability; restart with --resume."
                )
                break

            print(
                "[INFO] Error recorded; continuing with the next route. "
                "(If many errors occur, rerun later with --resume.)"
            )
            continue

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
    # Total judge usage/cost (best-effort; OpenRouter only)
    if run_judge_calls_total:
        print(
            "Judge calls total: "
            f"{int(run_judge_calls_total)} | "
            f"prompt_tokens={int(run_judge_usage_sum.get('prompt_tokens', 0))} | "
            f"completion_tokens={int(run_judge_usage_sum.get('completion_tokens', 0))} | "
            f"total_tokens={int(run_judge_usage_sum.get('total_tokens', 0))}"
        )
    if float(run_judge_usage_sum.get("cost", 0.0)) > 0:
        print(f"Judge cost total (USD): {float(run_judge_usage_sum.get('cost', 0.0)):.6f}")
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
        avg_route = sum(float(r["route_total_time_s"]) for r in timing_items) / len(timing_items)
        avg_judge_llm = sum(float(r.get("judge_llm_time_s", 0.0)) for r in timing_items) / len(
            timing_items
        )
        print(f"Avg route time (s): {avg_route:.1f}")
        print(f"Avg CPC judge LLM time (s): {avg_judge_llm:.2f}")

    # Aggregate statistics over all results (existing and new)
    scores = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    passes = [r["passed"] for r in results]

    if scores:
        avg_score = sum(scores) / len(scores)
        pass_rate = sum(1 for p in passes if p) / len(passes)
        print("\n" + "=" * 72)
        print("[SUMMARY] Route sample results")
        print(f"Mean score: {avg_score:.3f}")
        print(f"PASSED share: {pass_rate * 100:.1f}% ({sum(1 for p in passes if p)}/{len(passes)})")
        print("=" * 72)

    # Save the JSON report (update/create the combined results file)
    if args.out_json:
        out_path = args.out_json
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
