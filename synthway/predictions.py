"""
Synthesis route generation (synthway, direct mode).

Workflow:
- Load the RAG index and build a query from the target composition and optional hints/filters.
- Retrieve top-k contexts and assemble the seven-part composite prompt (see prompt.py).
- Call the selected LLM (Gemini by default) and obtain a response.
- Parse multiple JSON routes robustly and save each in a separate file.

"""

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
import argparse
import logging
import os
import sys
import json
import time
import threading
from pathlib import Path
from types import SimpleNamespace
from credentials import load_credentials
from google import genai
import requests

import utils
from utils import parse_model_routes_output, save_routes_to_directory

# RAG and Prompt Imports
from build_rag_index import RagIndex, format_query, apply_post_filters, rerank_by_slot_coverage
from prompt import (
    role_information_component,
    task_description_component,
    rag_integration_component,
    planning_requirement_component,
    explanation_requirement_component,
    structured_output_format_component,
    detailed_requirements_component,
)


# --- Setup ---
# Shared with both evaluators and the demo; independent of the launch directory.
load_credentials()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Read keys (do NOT hard-fail globally — only fail when a provider is actually used)
GEMINI_KEY = (os.getenv("GEMINI_API_KEY") or "").strip() or None
OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip() or None

# Optional: app attribution headers for OpenRouter (recommended but not required)
OPENROUTER_APP_URL = (os.getenv("OPENROUTER_APP_URL") or "http://localhost").strip()
OPENROUTER_APP_NAME = (os.getenv("OPENROUTER_APP_NAME") or "SynthWay").strip()

# ITMO Ollama endpoint (no API key required)
OLLAMA_ENDPOINT = (os.getenv("OLLAMA_ENDPOINT") or "https://aicltr.itmo.ru/ollama/api/generate").strip()
OLLAMA_DEFAULT_MODEL = (os.getenv("OLLAMA_MODEL") or "deepseek-v2:16b").strip()

# Google (Gemini) client
google_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

# OpenRouter client (OpenAI-compatible)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

openrouter_client = (
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": OPENROUTER_APP_URL,
            "X-Title": OPENROUTER_APP_NAME,
        },
    )
    if (OPENROUTER_API_KEY and OpenAI)
    else None
)

# --- OpenRouter usage accounting (tokens + cost) ---
# OpenRouter adds usage fields (prompt/completion/total tokens) and often includes `cost`.
# We store the last response usage in-process so callers can persist/log it.
_OPENROUTER_LAST_USAGE: dict | None = None

# Per-call usage log (one record per OpenRouter API call in this process)
_OPENROUTER_USAGE_LOG: list[dict] = []

# --- OpenRouter last call timing accounting ---
_OPENROUTER_LAST_CALL_TIME_S: float | None = None

def _set_openrouter_last_call_time_s(dt_s: float | None) -> None:
    global _OPENROUTER_LAST_CALL_TIME_S
    _OPENROUTER_LAST_CALL_TIME_S = dt_s

def get_openrouter_last_call_time_s() -> float | None:
    """Return duration (seconds) of the last OpenRouter chat completion call in this process."""
    return _OPENROUTER_LAST_CALL_TIME_S


def _usage_to_dict(usage_obj) -> dict | None:
    """Convert OpenAI/OpenRouter usage object into a plain dict (best-effort)."""
    if usage_obj is None:
        return None


    # If it already looks like a dict
    if isinstance(usage_obj, dict):
        return usage_obj

    # Pydantic / OpenAI objects often expose model_dump
    try:
        md = getattr(usage_obj, "model_dump", None)
        if callable(md):
            return md()
    except Exception:
        pass

    # Generic attribute -> dict
    out = {}
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

    # If we got nothing, try a last-resort dict() cast
    if not out:
        try:
            return dict(usage_obj)
        except Exception:
            return None

    return out


def _accumulate_usage(target: dict, usage: dict | None) -> None:
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


def _set_openrouter_last_usage(usage: dict | None) -> None:
    global _OPENROUTER_LAST_USAGE
    _OPENROUTER_LAST_USAGE = usage


def get_openrouter_last_usage() -> dict | None:
    """Return the last OpenRouter usage dict captured in this process."""
    return _OPENROUTER_LAST_USAGE

def _append_openrouter_usage_log(rec: dict) -> None:
    """Append one OpenRouter call accounting record to the in-process log."""
    global _OPENROUTER_USAGE_LOG
    try:
        if isinstance(rec, dict) and rec:
            _OPENROUTER_USAGE_LOG.append(rec)
    except Exception:
        pass


def get_openrouter_usage_log() -> list[dict]:
    """Return the full in-process OpenRouter per-call usage log."""
    return _OPENROUTER_USAGE_LOG


def reset_openrouter_usage_log() -> None:
    """Reset the in-process OpenRouter per-call usage log."""
    global _OPENROUTER_USAGE_LOG
    _OPENROUTER_USAGE_LOG = []

def list_available_models() -> None:
    """Print available models and whether they are callable in the current environment."""
    print("Available models:")
    for name, cfg in AVAILABLE_MODELS.items():
        provider = cfg.get("provider", "?")
        ok = "YES" if cfg.get("client") else "NO"
        print(f"- {name}  | provider={provider} | client_ready={ok}")


# Constants
AVAILABLE_MODELS = {
    # Google models (Gemini)
    "gemini-2.5-pro-exp-03-25": {"provider": "google", "client": google_client},
    "gemini-2.0-flash-thinking-exp-01-21": {"provider": "google", "client": google_client},
    "gemini-2.0-flash-001": {"provider": "google", "client": google_client},
    "gemini-3-flash-preview": {"provider": "google", "client": google_client},
    "gemini-3-pro-preview": {"provider": "google", "client": google_client},

    # OpenRouter models (OpenAI-compatible endpoint)
    "tngtech/deepseek-r1t2-chimera": {"provider": "openrouter", "client": openrouter_client},
    "qwen/qwen3-4b:free": {"provider": "openrouter", "client": openrouter_client},
    "nvidia/nemotron-3-nano-30b-a3b:free": {"provider": "openrouter", "client": openrouter_client},
    "nvidia/nemotron-3-nano-30b-a3b": {"provider": "openrouter", "client": openrouter_client},
    "xiaomi/mimo-v2-flash:free": {"provider": "openrouter", "client": openrouter_client},
    "xiaomi/mimo-v2-flash": {"provider": "openrouter", "client": openrouter_client},
    "openai/gpt-5.2": {"provider": "openrouter", "client": openrouter_client},
    "google/gemini-2.0-flash-001": {"provider": "openrouter", "client": openrouter_client},
    "google/gemini-2.5-flash": {"provider": "openrouter", "client": openrouter_client},
    "qwen/qwen3-30b-a3b-instruct-2507": {"provider": "openrouter", "client": openrouter_client},
    "meta-llama/llama-4-maverick": {"provider": "openrouter", "client": openrouter_client},
    "anthropic/claude-opus-4.5": {"provider": "openrouter", "client": openrouter_client},
    "deepseek/deepseek-v3.2": {"provider": "openrouter", "client": openrouter_client},
    "google/gemini-3-pro-preview": {"provider": "openrouter", "client": openrouter_client},
    "google/gemini-3-flash-preview": {"provider": "openrouter", "client": openrouter_client},
    "nvidia/nemotron-3-super-120b-a12b:free": {"provider": "openrouter", "client": openrouter_client},
    "google/gemini-3.1-pro-preview": {"provider": "openrouter", "client": openrouter_client},


    # ITMO Ollama proxy (no key)
    "qwen3:8b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "mistral-small3.1:24b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "mixtral:8x7b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "gpt-oss:20b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "gpt-oss:120b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "llama4:16x17b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "granite3.2-vision:2b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "gemma3:27b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "mistral-small3.1-24b-128k:latest": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "mistral-small:24b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "deepseek-v2:16b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "qwen3:32b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "llama3:70b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "deepseek-r1:70b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "qwen2.5vl:72b": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "mistral-small3.1:latest": {"provider": "ollama", "client": OLLAMA_ENDPOINT},
    "llama4:latest": {"provider": "ollama", "client": OLLAMA_ENDPOINT},

    # # OpenRouter models (examples)
    # "deepseek/deepseek-chat-v3-0324": {"provider": "openrouter", "client": openrouter_client},
    # "deepseek/deepseek-r1": {"provider": "openrouter", "client": openrouter_client},
    # "qwen/qwen2.5-vl-72b-instruct": {"provider": "openrouter", "client": openrouter_client},
    # "meta-llama/llama-3.3-70b-instruct": {"provider": "openrouter", "client": openrouter_client},
    # "meta-llama/llama-4-maverick": {"provider": "openrouter", "client": openrouter_client},
    # "meta-llama/llama-4-scout": {"provider": "openrouter", "client": openrouter_client},
    # "openai/o3-mini": {"provider": "openrouter", "client": openrouter_client},
    # "mistralai/mistral-small-3.1-24b-instruct": {"provider": "openrouter", "client": openrouter_client},
    # "openrouter/quasar-alpha": {"provider": "openrouter", "client": openrouter_client},
    # "openrouter/optimus-alpha": {"provider": "openrouter", "client": openrouter_client},
    # "google/gemini-2.5-pro-preview-03-25": {"provider": "openrouter", "client": openrouter_client},
    # "google/gemini-2.0-flash-001": {"provider": "openrouter", "client": openrouter_client},
    # "meta-llama/llama-4-maverick:free": {"provider": "openrouter", "client": openrouter_client},
    # "openai/gpt-4.1": {"provider": "openrouter", "client": openrouter_client},
    # "x-ai/grok-3-mini-beta": {"provider": "openrouter", "client": openrouter_client},
}

# Default paths for RAG and output
DEFAULT_RAG_INDEX_DIR = f"{_REPO_ROOT}/synthway/index"
DEFAULT_LLM_OUTPUT_DIR = f"{_REPO_ROOT}/outputs/generation"

# --- RAG index cache (avoid re-loading SentenceTransformer & index per case) ---
_RAG_INDEX_CACHE: dict[str, "RagIndex"] = {}

def get_cached_rag_index(rag_idx_dir: str) -> "RagIndex":
    """Load RagIndex once per process and reuse it across cases.

    This avoids repeatedly loading heavy components (e.g., SentenceTransformer)
    inside RagIndex on every case in batch mode.
    """
    if rag_idx_dir in _RAG_INDEX_CACHE:
        return _RAG_INDEX_CACHE[rag_idx_dir]

    ri = RagIndex()
    ri.load(rag_idx_dir)
    logging.info("RAG index loaded from %s (cached)", rag_idx_dir)
    _RAG_INDEX_CACHE[rag_idx_dir] = ri
    return ri



class _SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _call_openrouter(client, model: str, prompt: str) -> str:
    """Call an OpenRouter model via OpenAI-compatible chat completions.

    Captures OpenRouter usage (prompt/completion/total tokens and often `cost`) and stores it
    in-process for later persistence.
    """
    _t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    _dt = time.perf_counter() - _t0
    _set_openrouter_last_call_time_s(_dt)

    usage_dict = _usage_to_dict(getattr(resp, "usage", None))

    # OpenRouter sometimes exposes additional fields at the top level; best-effort capture.
    # If `cost` is not present in usage, try to look it up on the response object.
    if usage_dict is None:
        usage_dict = {}
    if "cost" not in usage_dict:
        try:
            extra_cost = getattr(resp, "cost", None)
            if extra_cost is not None:
                usage_dict["cost"] = extra_cost
        except Exception:
            pass

    _set_openrouter_last_usage(usage_dict if usage_dict else None)
    # Persist per-call accounting record (enables run-level aggregation and multi-call workflows).
    try:
        rec = dict(usage_dict) if isinstance(usage_dict, dict) else {}
    except Exception:
        rec = {}
    rec["ts"] = time.time()
    rec["model"] = model
    rec["prompt_chars"] = len(prompt or "")
    rec["time_s"] = float(_dt) if isinstance(_dt, (int, float)) else None

    # Lightweight log for immediate visibility
    if usage_dict:
        pt = usage_dict.get("prompt_tokens")
        ct = usage_dict.get("completion_tokens")
        tt = usage_dict.get("total_tokens")
        cost = usage_dict.get("cost")
        logging.info(
            "[OpenRouter usage] model=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s | cost=$%s | time_s=%.3f",
            model,
            pt,
            ct,
            tt,
            f"{cost:.6f}" if isinstance(cost, (int, float)) else (str(cost) if cost is not None else "n/a"),
            float(_dt) if isinstance(_dt, (int, float)) else -1.0,
        )

    content = resp.choices[0].message.content or ""
    try:
        rec["output_chars"] = len(content or "")
    except Exception:
        rec["output_chars"] = None
    _append_openrouter_usage_log(rec)
    return content


# --- Ollama rate limiting (per-process) ---
_OLLAMA_MIN_INTERVAL_SEC = float(
    (os.getenv("OLLAMA_MIN_INTERVAL_SEC") or os.getenv("LLM_MIN_INTERVAL_SEC") or "0").strip()
)
_OLLAMA_RATE_LOCK = threading.Lock()
_OLLAMA_LAST_CALL_TS = 0.0


# --- Gemini rate limiting (per-process) ---
# Set GEMINI_MIN_INTERVAL_SEC (or LLM_MIN_INTERVAL_SEC) to enforce a minimum delay
# between consecutive Gemini requests within this process.
# Example: RPM=5 -> interval=12.0 sec.
_GEMINI_MIN_INTERVAL_SEC = float(
    (os.getenv("GEMINI_MIN_INTERVAL_SEC") or os.getenv("LLM_MIN_INTERVAL_SEC") or "0").strip()
)
_GEMINI_RATE_LOCK = threading.Lock()
_GEMINI_LAST_CALL_TS = 0.0


def _gemini_rate_limit() -> None:
    global _GEMINI_LAST_CALL_TS
    if _GEMINI_MIN_INTERVAL_SEC and _GEMINI_MIN_INTERVAL_SEC > 0:
        with _GEMINI_RATE_LOCK:
            now = time.time()
            wait = _GEMINI_MIN_INTERVAL_SEC - (now - _GEMINI_LAST_CALL_TS)
            if wait > 0:
                time.sleep(wait)
            _GEMINI_LAST_CALL_TS = time.time()


def _call_ollama(endpoint: str, model: str, prompt: str) -> str:
    """Call an Ollama-compatible /api/generate endpoint (ITMO proxy).

    Rate limiting:
      - Set env OLLAMA_MIN_INTERVAL_SEC (or LLM_MIN_INTERVAL_SEC) to enforce a minimum
        delay between consecutive Ollama requests within this process.
        Example: OLLAMA_MIN_INTERVAL_SEC=1.0  -> ~1 request/second.
    """
    global _OLLAMA_LAST_CALL_TS

    if _OLLAMA_MIN_INTERVAL_SEC and _OLLAMA_MIN_INTERVAL_SEC > 0:
        with _OLLAMA_RATE_LOCK:
            now = time.time()
            wait = _OLLAMA_MIN_INTERVAL_SEC - (now - _OLLAMA_LAST_CALL_TS)
            if wait > 0:
                time.sleep(wait)
            _OLLAMA_LAST_CALL_TS = time.time()

    # Allow both plain Ollama names (e.g. "gpt-oss:120b") and legacy names like "itmo-ollama/gpt-oss:120b"
    if isinstance(model, str) and model.startswith("itmo-ollama/"):
        model = model.split("/", 1)[1]

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(endpoint, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json() or {}
    return (data.get("response") or "").strip()


def call_llm(provider: str, client, model: str, prompt: str) -> str:
    """Unified LLM call.

    Providers:
    - google: Gemini via utils.get_precursors
    - openrouter: OpenAI-compatible chat completions
    - ollama: Ollama-compatible /api/generate (ITMO proxy)
    """
    if provider == "google":
        _gemini_rate_limit()
        return utils.get_precursors(client, model, prompt)
    if provider == "openrouter":
        return _call_openrouter(client, model, prompt)
    if provider == "ollama":
        return _call_ollama(client, model, prompt)
    raise ValueError(f"Unknown provider: {provider}")

def _get_provider_client_for_model(model_name: str):
    """Resolve provider+client for a model name with the same readiness checks as generation."""
    chosen_model = (model_name or "").strip()
    if not chosen_model:
        raise ValueError("Model name must not be empty.")

    if chosen_model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{chosen_model}'. Use --list-models to see available options."
        )

    cfg = AVAILABLE_MODELS[chosen_model]
    provider = cfg.get("provider", "google")
    client = cfg.get("client")

    # Provider-specific readiness checks
    if provider == "google" and not client:
        raise RuntimeError(
            "A Gemini model was selected, but GEMINI_API_KEY was not found. "
            "Set the key in .env or as an environment variable."
        )

    if provider == "openrouter" and not client:
        raise RuntimeError(
            "An OpenRouter model was selected, but OPENROUTER_API_KEY was not found. "
            "Set the key in .env or as an environment variable."
        )

    if provider == "ollama" and not client:
        raise RuntimeError(
            "An Ollama model was selected, but OLLAMA_ENDPOINT is unset. "
            "Set OLLAMA_ENDPOINT in .env."
        )

    return provider, client


def regenerate_single_route_with_feedback(
    *,
    composition: str,
    method: str | None,
    contexts: list[str],
    model_name: str,
    previous_raw_text: str,
    previous_retrieval_text: str | None,
    reviewer_feedback: str,
    attempt_no: int,
) -> tuple[dict | None, str]:
    """Regenerate ONE route using reviewer feedback injected into the prompt.

    Returns: (route_dict_or_None, raw_model_output_text)

    Notes:
    - We DO NOT re-run retrieval here. Caller is expected to pass the same contexts.
    """

    comp = (composition or "").strip() or "UNKNOWN"
    base_prompt = assemble_composite_prompt(comp, 1, contexts or [], method)

    keep_rt_hint = ""
    prev_rt = previous_retrieval_text or ""
    if prev_rt:
        keep_rt_hint = (
            "\n\nPrevious retrieval_text (reference only; fix issues, do not copy blindly):\n"
            f"{prev_rt}\n"
        )

    revision_block = (
        "\n\n[REVISION TASK]"
        "\nYou are revising a previously generated inorganic synthesis route. "
        "The previous attempt did NOT pass automated evaluation."
        f"\nAttempt: {attempt_no}"
        "\n\nReviewer feedback (address ALL points):\n"
        f"{(reviewer_feedback or '').strip()}"
        "\n\nPrevious route (reference only; fix issues, do not copy blindly):\n"
        f"{(previous_raw_text or '').strip()}"
        + keep_rt_hint
        + "\n\nNow regenerate ONE improved route that addresses the feedback while staying faithful to the retrieved context. "
        "Return the output in the EXACT structured format requested above. "
        "Do not add extra commentary outside the required fields."
    )

    prompt = base_prompt + revision_block

    provider, client = _get_provider_client_for_model(model_name)
    _t0 = time.perf_counter()
    predicted_text = call_llm(provider, client, model_name, prompt)
    _dt = time.perf_counter() - _t0

    if provider == "openrouter":
        usage = get_openrouter_last_usage() or {}
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        tt = usage.get("total_tokens")
        cost = usage.get("cost")
        logging.info(
            "[OpenRouter regen usage] attempt=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s | cost=$%s | time_s=%.3f",
            attempt_no,
            pt,
            ct,
            tt,
            f"{cost:.6f}" if isinstance(cost, (int, float)) else (str(cost) if cost is not None else "n/a"),
            float(_dt) if isinstance(_dt, (int, float)) else -1.0,
        )

    routes = parse_model_routes_output(predicted_text, expected_num_routes=1)
    if not routes:
        return None, (predicted_text or "")

    return routes[0], (predicted_text or "")

# synthway helper functions
def assemble_composite_prompt(composition: str, num_routes: int, contexts, method: str | None = None) -> str:
    """
    Assemble a seven-part composite prompt with the top-k RAG contexts.
    """
    # Build the context block as a bulleted list
    rag_block = "".join(f"- {ctx}\n" for ctx in contexts if ctx)

    # If requested, add the method as a preference rather than a strict constraint
    method_hint = ""
    if method:
        method_hint = f"\n\n[Note] When possible, prefer the **{method}** method.\n"

    fmt_block = structured_output_format_component
    fmt_block = fmt_block.replace("{composition}", composition).replace("{num_routes}", str(num_routes))

    prompt = (
        role_information_component
        + task_description_component.format(num_routes=num_routes, composition=composition)
        + rag_integration_component.format(composition=composition, rag_evidence_block=rag_block, num_routes=num_routes)
        + planning_requirement_component.format(composition=composition)
        + explanation_requirement_component
        + fmt_block
        + detailed_requirements_component
        + method_hint
    )
    return prompt

def parse_arguments():
    """Parse command line arguments."""
    # Parse CLI arguments, including direct generation (--composition) and RAG settings.
    parser = argparse.ArgumentParser(
        description="Direct synthway route generation (retrosynthesis only)"
    )
    parser.add_argument(
        "--model", dest="models", action="append", help="Specific model(s) to test."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List all available models."
    )
    # Direct synthway generation mode
    parser.add_argument("--composition", type=str, help="Target composition formula for direct generation (enables synthway mode).")
    parser.add_argument("--method", type=str, help="Desired synthesis method (optional).")
    parser.add_argument("--num-routes", type=int, default=3, help="Number of routes to generate in direct mode.")
    parser.add_argument("--llm-output-dir", default=DEFAULT_LLM_OUTPUT_DIR, help="Where to save per-route JSONs in direct mode.")

    # Batch mode via stratified dataset (JSONL with fields id, composition, method, retrieval_text)
    parser.add_argument(
        "--stratified-dataset",
        type=str,
        help=(
            "Path to stratified dataset JSONL (e.g., final_stratified_dataset.jsonl). "
            "If set (and --composition is not provided), runs synthway generation "
            "for every line using its composition."
        ),
    )

    # RAG knobs (query, filters, index paths)
    parser.add_argument("--rag-idx", default=DEFAULT_RAG_INDEX_DIR, help="Path to RAG index directory.")
    parser.add_argument("--rag-topk", type=int, default=5, help="How many contexts to inject into the prompt.")
    parser.add_argument("--rag-min-lex-score", type=float, default=None, help="Min cosine threshold for TF-IDF retrieval (0..1).")
    parser.add_argument("--rag-filter-method", type=str, help="Post-filter by method (case-insensitive contains).")
    parser.add_argument("--rag-element", action="append", help="Require element(s) to be present in the hit (repeatable).")
    parser.add_argument("--rag-require-precursor", action="append", help="Require precursor name(s) to occur (repeatable).")
    parser.add_argument("--rag-require-additive", action="append", help="Require additive name(s) to occur (repeatable).")
    parser.add_argument("--rag-require-seed", action="append", help="Require seed name(s) to occur (repeatable).")
    parser.add_argument("--rag-preselect", type=int, default=30, help="Preselect multiplier before post-filtering (k*preselect).")

    # RAG query fields (used to build the query string; optional)
    parser.add_argument("--rag-shape", type=str, help="Optional shape hint for query (e.g., spherical).")
    parser.add_argument("--rag-solvent", type=str, help="Optional solvent name for query.")
    parser.add_argument("--rag-temperature", type=float, help="Optional temperature for query.")
    parser.add_argument("--rag-time", type=float, help="Optional time (hours) for query.")
    parser.add_argument("--rag-pH", type=float, help="Optional pH for query.")
    parser.add_argument("--rag-precursor", action="append", help="Optional precursor(s) to include in the query (repeatable).")
    parser.add_argument("--rag-additive", action="append", help="Optional additive(s) to include in the query (repeatable).")
    parser.add_argument("--rag-seed", action="append", help="Optional seed(s) to include in the query (repeatable).")

    args = parser.parse_args()
    return args
def generate_synthesis_routes(composition: str,
                              method: str | None,
                              num_routes: int,
                              output_dir: str,
                              rag_idx_dir: str,
                              rag_topk: int,
                              rag_min_lex_score: float | None,
                              rag_filter_method: str | None,
                              rag_elements: list[str] | None,
                              rag_req_precursors: list[str] | None,
                              rag_req_additives: list[str] | None,
                              rag_req_seeds: list[str] | None,
                              rag_preselect: int,
                              model_name: str | None,
                              rag_shape: str | None = None,
                              rag_solvent: str | None = None,
                              rag_temperature: float | None = None,
                              rag_time: float | None = None,
                              rag_pH: float | None = None,
                              rag_precursors: list[str] | None = None,
                              rag_additives: list[str] | None = None,
                              rag_seeds: list[str] | None = None):
    """
    Full workflow: load index, retrieve top-k examples, assemble the seven-part prompt, call Gemini, parse JSON, and save separate files.
    """
    case_t0 = time.perf_counter()
    # 1) RAG: load the index (cached per process)
    ri = get_cached_rag_index(rag_idx_dir)

    # 2) Build the query string: composition is required; include other fields only if specified
    q_str = format_query(
        composition=composition,
        method=method if method else None,
        shape=rag_shape,
        temperature=rag_temperature,
        solvent=rag_solvent,
        additives=rag_additives,
        precursors=rag_precursors,
        seeds=rag_seeds,
        time=rag_time,
        pH=rag_pH
    )
    # Semantic query expansion without an additional composition boost
    q_str = ri.expand_query_with_denselex(q_str)

    # 3) Retrieve and filter candidates
    n_pre = max(rag_topk * (rag_preselect if rag_preselect else 10), rag_topk)
    raw_hits = ri.retrieve_lexical(q_str, k=n_pre, min_score=rag_min_lex_score)
    hits = apply_post_filters(
        ri, raw_hits, rag_topk,
        filter_method=rag_filter_method,
        elements=rag_elements,
        req_precursors=rag_req_precursors,
        req_additives=rag_req_additives,
        req_seeds=rag_req_seeds,
        composition_filter=composition,
    )
    # Rerank by slot coverage (composition, method, shape, solvent, etc.)
    slot_args = SimpleNamespace(
        composition=composition,
        method=method,
        shape=rag_shape,
        solvent=rag_solvent,
        crystalsystem=None,
        temperature=rag_temperature,
        time=rag_time,
        pH=rag_pH,
        size_length=None,
        size_width=None,
        size_depth=None,
        pdfno=None,
        precursor=rag_precursors,
        additive=rag_additives,
        seed=rag_seeds,
    )
    hits = rerank_by_slot_coverage(hits, slot_args)
    # Prefer retrieval_text for context examples
    contexts = [h.retrieval_text for h in hits if h.retrieval_text][:rag_topk]
    logging.info("RAG retrieved %d/%d contexts.", len(contexts), rag_topk)

    # --- Save retrieval context to JSON ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        rc_path = os.path.join(output_dir, "retrieval_context.json")
        meta = {
            "composition": composition,
            "method": method,
            "rag_topk": rag_topk,
            "query": q_str,
            "retrieval_context": contexts,
        }
        with open(rc_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logging.info("Saved retrieval context to %s", rc_path)
    except Exception as e:
        logging.error("Failed to save retrieval context: %s", e)

    # 4) Assemble the final seven-part prompt
    prompt = assemble_composite_prompt(composition, num_routes, contexts, method)

    # 5) Select the model and client
    chosen_model = model_name or "gemini-2.5-pro-exp-03-25"

    # Normalize legacy model names like "itmo-ollama/<name>" to plain Ollama names "<name>"
    if chosen_model.startswith("itmo-ollama/"):
        chosen_model = chosen_model.split("/", 1)[1]

    if chosen_model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{chosen_model}'. Use --list-models to see available options."
        )

    cfg = AVAILABLE_MODELS[chosen_model]
    provider = cfg.get("provider", "google")
    client = cfg.get("client")

    # Provider-specific readiness checks
    if provider == "google" and not client:
        raise RuntimeError(
            "A Gemini model was selected, but GEMINI_API_KEY was not found. "
            "Set the key in .env or as an environment variable."
        )

    if provider == "openrouter" and not client:
        if OpenAI is None:
            raise RuntimeError(
                "OpenRouter requires 'openai'. Install it: pip install openai"
            )
        raise RuntimeError(
            "An OpenRouter model was selected, but OPENROUTER_API_KEY was not found. "
            "Set the key in .env or as an environment variable."
        )

    if provider == "ollama" and not client:
        raise RuntimeError(
            "An Ollama model was selected, but OLLAMA_ENDPOINT is unset. "
            "Set OLLAMA_ENDPOINT in .env."
        )

    # 6) Call the model
    # Reset per-call OpenRouter accounting for this case so we can attribute calls to the case.
    if provider == "openrouter":
        reset_openrouter_usage_log()
    llm_t0 = time.perf_counter()
    predicted_text = call_llm(provider, client, chosen_model, prompt)
    llm_call_time_s = time.perf_counter() - llm_t0

    # --- OpenRouter: persist accounting (input+output tokens and $ cost) ---
    if provider == "openrouter":
        usage = get_openrouter_last_usage() or {}
        try:
            os.makedirs(output_dir, exist_ok=True)
            usage_path = os.path.join(output_dir, "llm_usage.json")
            # Aggregate across all OpenRouter calls in this case (supports multi-call workflows).
            log_entries = get_openrouter_usage_log() or []
            totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}
            for ent in log_entries:
                _accumulate_usage(totals, ent)
            # "Pure" OpenRouter LLM time: sum of per-call API durations (excludes local waits outside the API call).
            openrouter_llm_time_s_total = 0.0
            openrouter_llm_time_s_last = None
            try:
                for ent in log_entries:
                    ts = ent.get("time_s") if isinstance(ent, dict) else None
                    if isinstance(ts, (int, float)):
                        openrouter_llm_time_s_total += float(ts)
                        openrouter_llm_time_s_last = float(ts)
            except Exception:
                pass
            payload = {
                "provider": provider,
                "model": chosen_model,
                "prompt_chars": len(prompt or ""),
                "output_chars": len(predicted_text or ""),
                "llm_call_time_s": round(float(llm_call_time_s), 6),
                "case_total_time_s": None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cost_usd": usage.get("cost"),
                "usage_raw": usage,
                # New: per-case totals across all calls
                "openrouter_calls_total": int(len(log_entries)),
                "openrouter_prompt_tokens_total": int(totals.get("prompt_tokens", 0)),
                "openrouter_completion_tokens_total": int(totals.get("completion_tokens", 0)),
                "openrouter_total_tokens_total": int(totals.get("total_tokens", 0)),
                "openrouter_cost_usd_total": float(totals.get("cost", 0.0)),
                # New: OpenRouter pure LLM time (sum across all API calls)
                "openrouter_llm_time_s_total": round(float(openrouter_llm_time_s_total), 6),
                "openrouter_llm_time_s_last": (round(float(openrouter_llm_time_s_last), 6) if isinstance(openrouter_llm_time_s_last, (int, float)) else None),
            }
            with open(usage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logging.info("Saved LLM usage accounting to %s", usage_path)
            # Persist full per-call log for audit/debugging
            try:
                calls_path = os.path.join(output_dir, "llm_usage_calls.json")
                with open(calls_path, "w", encoding="utf-8") as cf:
                    json.dump(log_entries, cf, ensure_ascii=False, indent=2)
                logging.info("Saved OpenRouter per-call usage log to %s", calls_path)
            except Exception as e:
                logging.error("Failed to persist OpenRouter per-call usage log: %s", e)

            # Extra friendly log line (input+output combined)
            if payload.get("total_tokens") is not None:
                logging.info(
                    "[OpenRouter cost] input+output=%s tokens | cost=$%s",
                    payload.get("total_tokens"),
                    f"{payload['cost_usd']:.6f}" if isinstance(payload.get("cost_usd"), (int, float)) else (str(payload.get("cost_usd")) if payload.get("cost_usd") is not None else "n/a"),
                )
        except Exception as e:
            logging.error("Failed to persist OpenRouter usage accounting: %s", e)

    # 7) Parse JSON routes and save one per file
    routes = parse_model_routes_output(predicted_text, expected_num_routes=num_routes)
    if not routes:
        logging.warning("Model returned no parseable routes. Saving raw text into a single file.")
        os.makedirs(output_dir, exist_ok=True)
        raw_fp = os.path.join(output_dir, "route_raw.txt")
        with open(raw_fp, "w", encoding="utf-8") as f:
            f.write(predicted_text or "")
        # --- Persist timing even on parse failure ---
        try:
            case_total_time_s = time.perf_counter() - case_t0
            timing_path = os.path.join(output_dir, "timing.json")
            timing_payload = {
                "model": chosen_model,
                "provider": provider,
                "llm_call_time_s": round(float(llm_call_time_s), 6),
                "case_total_time_s": round(float(case_total_time_s), 6),
                "parsed_routes": 0,
            }
            # If OpenRouter usage file exists, copy pure LLM time + call totals into timing.json for easier aggregation.
            if provider == "openrouter":
                try:
                    usage_path = os.path.join(output_dir, "llm_usage.json")
                    if os.path.exists(usage_path):
                        with open(usage_path, "r", encoding="utf-8") as uf:
                            up = json.load(uf) or {}
                        if "openrouter_llm_time_s_total" in up:
                            timing_payload["openrouter_llm_time_s_total"] = up.get("openrouter_llm_time_s_total")
                        if "openrouter_calls_total" in up:
                            timing_payload["openrouter_calls_total"] = up.get("openrouter_calls_total")
                        if "openrouter_total_tokens_total" in up:
                            timing_payload["openrouter_total_tokens_total"] = up.get("openrouter_total_tokens_total")
                        if "openrouter_cost_usd_total" in up:
                            timing_payload["openrouter_cost_usd_total"] = up.get("openrouter_cost_usd_total")
                except Exception:
                    pass
            with open(timing_path, "w", encoding="utf-8") as tf:
                json.dump(timing_payload, tf, ensure_ascii=False, indent=2)

            # Update llm_usage.json if it exists (OpenRouter case)
            usage_path = os.path.join(output_dir, "llm_usage.json")
            if os.path.exists(usage_path):
                with open(usage_path, "r", encoding="utf-8") as uf:
                    up = json.load(uf) or {}
                up["case_total_time_s"] = round(float(case_total_time_s), 6)
                with open(usage_path, "w", encoding="utf-8") as uf:
                    json.dump(up, uf, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error("Failed to persist timing on parse failure: %s", e)
        return
    save_routes_to_directory(routes, output_dir)
    # --- Persist timing on success ---
    try:
        case_total_time_s = time.perf_counter() - case_t0

        # Update llm_usage.json if it exists (OpenRouter case)
        usage_path = os.path.join(output_dir, "llm_usage.json")
        if os.path.exists(usage_path):
            with open(usage_path, "r", encoding="utf-8") as uf:
                up = json.load(uf) or {}
            up["case_total_time_s"] = round(float(case_total_time_s), 6)
            with open(usage_path, "w", encoding="utf-8") as uf:
                json.dump(up, uf, ensure_ascii=False, indent=2)

        timing_path = os.path.join(output_dir, "timing.json")
        timing_payload = {
            "model": chosen_model,
            "provider": provider,
            "llm_call_time_s": round(float(llm_call_time_s), 6),
            "case_total_time_s": round(float(case_total_time_s), 6),
            "parsed_routes": len(routes) if routes else 0,
        }
        # If OpenRouter usage file exists, copy pure LLM time + call totals into timing.json for easier aggregation.
        if provider == "openrouter":
            try:
                usage_path = os.path.join(output_dir, "llm_usage.json")
                if os.path.exists(usage_path):
                    with open(usage_path, "r", encoding="utf-8") as uf:
                        up = json.load(uf) or {}
                    if "openrouter_llm_time_s_total" in up:
                        timing_payload["openrouter_llm_time_s_total"] = up.get("openrouter_llm_time_s_total")
                    if "openrouter_calls_total" in up:
                        timing_payload["openrouter_calls_total"] = up.get("openrouter_calls_total")
                    if "openrouter_total_tokens_total" in up:
                        timing_payload["openrouter_total_tokens_total"] = up.get("openrouter_total_tokens_total")
                    if "openrouter_cost_usd_total" in up:
                        timing_payload["openrouter_cost_usd_total"] = up.get("openrouter_cost_usd_total")
            except Exception:
                pass
        with open(timing_path, "w", encoding="utf-8") as tf:
            json.dump(timing_payload, tf, ensure_ascii=False, indent=2)
        logging.info("Saved timing to %s", timing_path)
    except Exception as e:
        logging.error("Failed to persist timing: %s", e)


# --- Batch mode: run synthway for all (composition, method) pairs in stratified dataset ---
def run_batch_from_stratified_dataset(args) -> None:
    """Run synthway route generation for each row in the stratified JSONL.

    Expected JSONL fields: id, composition, method, retrieval_text.
    Create a separate case_id subdirectory under `--llm-output-dir` for each row
    and save the generated routes there.
    """
    jsonl_path = args.stratified_dataset
    if not jsonl_path:
        raise ValueError("run_batch_from_stratified_dataset called without --stratified-dataset.")

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Stratified dataset not found: {jsonl_path}")

    os.makedirs(args.llm_output_dir, exist_ok=True)

    # Select the first --model value if supplied; otherwise use the default
    model_name = None
    if args.models:
        model_name = args.models[0]

    logging.info("[BATCH] Starting batch generation from %s", jsonl_path)
    batch_t0 = time.perf_counter()
    batch_cases_done = 0
    batch_case_times: list[float] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logging.error("[BATCH] Line %d: invalid JSON (%s). Skipping.", line_no, e)
                continue

            composition = rec.get("composition")
            method = None  # batch mode uses only composition by default
            case_id = rec.get("id") or f"case_{line_no:04d}"

            if not composition:
                logging.warning(
                    "[BATCH] Line %d (id=%s): missing composition, skipping.",
                    line_no,
                    case_id,
                )
                continue

            # Guard against slashes in IDs
            safe_case_id = str(case_id).replace(os.sep, "_")
            case_out_dir = os.path.join(args.llm_output_dir, safe_case_id)

            logging.info(
                "[BATCH] Case %s (line %d): composition=%s, method=%s",
                safe_case_id,
                line_no,
                composition,
                method,
            )

            # Resume-friendly behavior: if this case already has generated routes, skip it
            try:
                if os.path.isdir(case_out_dir):
                    existing_files = os.listdir(case_out_dir)
                    has_routes = any(
                        name.startswith("route_") and name.endswith(".json")
                        for name in existing_files
                    ) or any(
                        name == "route_raw.txt"
                        for name in existing_files
                    )
                    if has_routes:
                        logging.info(
                            "[BATCH] Case %s already has generated routes, skipping.",
                            safe_case_id,
                        )
                        continue
            except OSError as e:
                logging.warning(
                    "[BATCH] Could not inspect output directory for case %s: %s",
                    safe_case_id,
                    e,
                )

            _case_t0 = time.perf_counter()
            generate_synthesis_routes(
                composition=composition,
                method=method,
                num_routes=args.num_routes,
                output_dir=case_out_dir,
                rag_idx_dir=args.rag_idx,
                rag_topk=args.rag_topk,
                rag_min_lex_score=args.rag_min_lex_score,
                # Batch mode: only use CLI filter, never dataset method
                rag_filter_method=args.rag_filter_method,
                rag_elements=(args.rag_element or []),
                rag_req_precursors=(args.rag_require_precursor or []),
                rag_req_additives=(args.rag_require_additive or []),
                rag_req_seeds=(args.rag_require_seed or []),
                rag_preselect=args.rag_preselect,
                model_name=model_name,
                rag_shape=args.rag_shape,
                rag_solvent=args.rag_solvent,
                rag_temperature=args.rag_temperature,
                rag_time=args.rag_time,
                rag_pH=getattr(args, "rag_pH", None),
                rag_precursors=(args.rag_precursor or []),
                rag_additives=(args.rag_additive or []),
                rag_seeds=(args.rag_seed or []),
            )
            _case_dt = time.perf_counter() - _case_t0
            batch_cases_done += 1
            batch_case_times.append(float(_case_dt))

    try:
        batch_total_time_s = time.perf_counter() - batch_t0
        avg_case_s = (sum(batch_case_times) / len(batch_case_times)) if batch_case_times else 0.0
        logging.info(
            "[BATCH] Timing: cases_done=%s | total_time_s=%.1f | avg_case_time_s=%.2f",
            batch_cases_done,
            float(batch_total_time_s),
            float(avg_case_s),
        )
    except Exception:
        pass
    logging.info("[BATCH] Finished batch generation from %s", jsonl_path)


if __name__ == "__main__":
    args = parse_arguments()
    if args.list_models:
        list_available_models()
        sys.exit(0)

    # Batch mode: take composition/method requests from the stratified JSONL
    if getattr(args, "stratified_dataset", None):
        run_batch_from_stratified_dataset(args)
        sys.exit(0)

    # Direct synthway generation (single request via --composition)
    if args.composition:
        # Select the first --model value if supplied; otherwise use the default
        model_name = None
        if args.models:
            model_name = args.models[0]
        generate_synthesis_routes(
            composition=args.composition,
            method=args.method,
            num_routes=args.num_routes,
            output_dir=args.llm_output_dir,
            rag_idx_dir=args.rag_idx,
            rag_topk=args.rag_topk,
            rag_min_lex_score=args.rag_min_lex_score,
            rag_filter_method=(args.rag_filter_method or args.method),
            rag_elements=(args.rag_element or []),
            rag_req_precursors=(args.rag_require_precursor or []),
            rag_req_additives=(args.rag_require_additive or []),
            rag_req_seeds=(args.rag_require_seed or []),
            rag_preselect=args.rag_preselect,
            model_name=model_name,
            rag_shape=args.rag_shape,
            rag_solvent=args.rag_solvent,
            rag_temperature=args.rag_temperature,
            rag_time=args.rag_time,
            rag_pH=getattr(args, "rag_pH", None),
            rag_precursors=(args.rag_precursor or []),
            rag_additives=(args.rag_additive or []),
            rag_seeds=(args.rag_seed or []),
        )
        sys.exit(0)

    print(
        "No --composition or --stratified-dataset provided. "
        "This build supports direct synthway generation for a single query (via --composition) "
        "or batch mode from a stratified dataset (via --stratified-dataset). See --help."
    )
    sys.exit(0)
