"""
Utilities for the SynthWay pipeline.

Module responsibilities:
- Unified LLM adapter for Google Gemini through `google-genai`, with rate limiting.
- I/O helpers: filename sanitization, logging setup, and saving/loading results.
- Parse text model responses:
  • extract routes from narrative responses using `Route N: ...` headings;
  • find each route's `retrieval_text: ...` index line, strip outer quotes if present, and preserve its contents;
  • return dictionaries of the form {"raw_text": "...without index line...", "retrieval_text": "...without outer quotes..."}.

RAG indexing and prompt assembly are implemented in build_rag_index.py and prompt.py.
"""

import json
import logging
import os
import re
import time
from datetime import datetime

from google.genai import types


# FS / Logging
def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for cross-platform writing.
    Replace characters invalid on Windows (`<>:"/\\|?*`) with underscores.
    Preserve case and the extension.
    """
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, "_", filename)


def setup_logging(
    log_filename: str | None = None,
    model_name: str | None = None,
    start_index: int = 0,
    end_index: int | None = None,
) -> str:
    """Initialize file logging in `./logs`.
    Automatically append a timestamp and batch suffix (`_start{n}[_end{m}]`).
    Return the created log file path.
    """
    os.makedirs("logs", exist_ok=True)

    if model_name:
        model_name = sanitize_filename(model_name)

    timestamp = None
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = (
            f"logs/retrosynthesis_{model_name}_{timestamp}.log"
            if model_name
            else f"logs/retrosynthesis_test_{timestamp}.log"
        )
    else:
        if not log_filename.startswith("logs/"):
            log_filename = os.path.join("logs", log_filename)

    batch_suffix = ""
    if start_index > 0 or end_index is not None:
        batch_suffix += f"_start{start_index}"
        batch_suffix += f"_end{end_index}" if end_index is not None else "_endFull"

    if timestamp is not None:
        base_name = log_filename.replace(f"_{timestamp}.log", "")
        log_filename = f"{base_name}{batch_suffix}_{timestamp}.log"
    else:
        base_name, ext = os.path.splitext(log_filename)
        log_filename = f"{base_name}{batch_suffix}{ext}"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging initialized. Start Index: %s, End Index: %s", start_index, end_index)
    return log_filename


# LLM call (Gemini)
def get_precursors(client, model_name: str, prompt: str, delay: float = 2) -> str | None:
    """Unified LLM call (Google Gemini through the `google-genai` SDK).
    Respect rate limits; return response text, or `None` on failure.

    Despite its name, the function returns the full LLM response text.
    """
    try:
        # Rate-limit hints
        if model_name.startswith("gemini"):
            gemini_rate_limits = {
                "gemini-2.5-pro": 2,
                "gemini-2.0-flash-thinking": 10,
                "gemini-2.0-flash-001": 15,
                "gemini-2.0-flash-lite": 30,
                "gemini-2.0-flash-exp": 10,
                "gemini-1.5-flash": 15,
                "gemini-1.5-flash-8b": 15,
                "gemini-1.5-pro": 2,
                "gemini-embedding": 5,
            }
            model_delay = delay
            for model_family, rpm in gemini_rate_limits.items():
                if model_name.startswith(model_family):
                    model_delay = 60.0 / rpm
                    logging.info(
                        "Using rate limit for %s: %s RPM (%.2f s delay)",
                        model_family,
                        rpm,
                        model_delay,
                    )
                    break
            logging.info("Applying delay of %.2f seconds for %s", model_delay, model_name)
            time.sleep(model_delay)

        if model_name.startswith("gemini"):
            # Build contents compatible with SDK variants
            try:
                user_contents = [types.Content(role="user", parts=[types.Part.from_text(prompt)])]
            except Exception:
                try:
                    user_contents = [types.Part(text=prompt)]
                except Exception:
                    user_contents = prompt

            response = client.models.generate_content(
                model=model_name,
                contents=user_contents,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            out_text = getattr(response, "text", None)
            if not out_text:
                try:
                    out_text = response.candidates[0].content.parts[0].text
                except Exception:
                    pass
            if not out_text:
                out_text = str(response)
            return out_text.strip()

        logging.error(
            "Unsupported model provider: %s. Only Google Gemini models are allowed.", model_name
        )
        return None

    except Exception as e:
        logging.error("Error with API for model %s: %s", model_name, e, exc_info=True)
        print(f"Error with API for model {model_name}: {e}")
        return None


# synthway: multi-text routes parsing & saving
def parse_model_routes_output(
    response_text: str, expected_num_routes: int | None = None
) -> list[dict]:
    """Parse narrative routes from a model response.
    Require a `retrieval_text:` line in every block.
    """
    if not response_text or not response_text.strip():
        logging.error("Model returned empty output.")
        return []

    routes = _extract_text_routes(response_text)
    logging.info("Extracted %d text route(s) with retrieval_text.", len(routes))

    if not routes:
        logging.warning("No text routes with retrieval_text found.")
        return []

    if expected_num_routes is not None and len(routes) != expected_num_routes:
        logging.warning("Extracted %d route(s), expected %d.", len(routes), expected_num_routes)

    return routes


def save_routes_to_directory(
    routes: list[dict], output_dir: str, prefix: str = "route_"
) -> list[str]:
    """Save each route to `{output_dir}/{prefix}{i}.json` (UTF-8, indent=2)."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: list[str] = []
    for idx, route in enumerate(routes, start=1):
        file_path = os.path.join(output_dir, f"{prefix}{idx}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(route, f, ensure_ascii=False, indent=2)
            logging.info("Saved route %d to %s", idx, file_path)
            saved_paths.append(file_path)
        except (IOError, OSError) as e:
            logging.error("Failed to save route %d to %s: %s", idx, file_path, e)
    return saved_paths


# Text parsing helpers
# Supports:
# - "Route N: ..." blocks split by headings,
# - a "retrieval_text: ..." index line whose value may be quoted,
# - removal of only the retrieval_text line from raw_text, preserving everything else.

_ROUTE_HDR_RE = re.compile(r"(?im)^\s*(?:[#]{1,6}\s*)?(?:[-*]\s*)?Route\s+\d+\s*(?::|-)?\s*.*$")
_RT_LINE_RE = re.compile(r"(?im)^\s*retrieval_text\s*:\s*(.+)$")


def _split_routes_by_headers(text: str) -> list[str]:
    """Split a response into blocks by 'Route N: ...' headings.
    If there are no headings, return one block.
    """
    lines = text.splitlines()
    idxs = [i for i, ln in enumerate(lines) if _ROUTE_HDR_RE.match(ln)]
    if not idxs:
        return [text.strip()] if text.strip() else []
    idxs.append(len(lines))
    chunks: list[str] = []
    for a, b in zip(idxs, idxs[1:]):
        chunk = "\n".join(lines[a:b]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _unwrap_outer_quotes(s: str) -> str:
    """Strip exactly one pair of outer quotes if they enclose the entire string.
    Supported pairs: " … ", ' … ', “ … ”, « … ».
    Preserve the inner contents.
    """
    s = s.strip()
    pairs = [('"', '"'), ("'", "'"), ("“", "”"), ("«", "»")]
    for lq, rq in pairs:
        if s.startswith(lq) and s.endswith(rq):
            return s[len(lq) : -len(rq)].strip()
    return s


def _cleanup_retrieval_text(rt: str) -> str:
    """Normalize `retrieval_text` minimally:
    - remove a trailing comma if present;
    - strip one enclosing pair of outer quotes (ASCII or typographic).
    Make no other formatting changes.
    """
    rt = rt.strip()
    if rt.endswith(","):
        rt = rt[:-1].strip()
    rt = _unwrap_outer_quotes(rt)
    return rt


def _extract_text_routes(response_text: str) -> list[dict]:
    """Extract routes from a text response. Return a list of dictionaries:
    {"retrieval_text": <string without outer quotes>, "raw_text": <procedure without index line>}.
    """
    if not isinstance(response_text, str):
        return []

    cleaned = response_text.strip()

    blocks = _split_routes_by_headers(cleaned)
    if not blocks and cleaned:
        blocks = [cleaned]

    routes: list[dict] = []
    for blk in blocks:
        m = _RT_LINE_RE.search(blk)
        if not m:
            # Skip blocks without an index line
            continue

        # Preserve contents, then remove the outer formatting
        rt_value_as_is = m.group(1).strip()
        rt_clean = _cleanup_retrieval_text(rt_value_as_is)

        # Remove all retrieval_text lines from the raw route text
        raw_no_rt = re.sub(_RT_LINE_RE, "", blk).strip()

        routes.append({"retrieval_text": rt_clean, "raw_text": raw_no_rt})
    return routes
