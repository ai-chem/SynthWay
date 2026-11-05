#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Prepare retrieval cases and optionally evaluate them with DeepEval.

Dataset mode uses --stratified-dataset JSONL or all JSON files in --corpus-dir.
--dataset-query-fields selects the fields used to form each request.
Query mode is enabled by CLI filters such as --composition or --method;
it selects matching corpus records and builds one request from those flags.

Each case contains:
- input: request text;
- expected_output: selected retrieval_text slots (--expected slots) or the
  complete retrieval_text (--expected pass_through);
- retrieval_context: up to --k retrieved source texts.
In slots mode, an empty --expected-slots includes all slots; JSON fields are
used if retrieval_text is unavailable.
Write cases as JSON to --out-dataset, with companion JSONL and metrics CSV files.
--evaluate enables DeepEval and requires credentials for --judge-model.

Examples from the repository root:
    python synthway/prepare_retrieval_eval.py --corpus-dir synthway/json \
        --index-dir synthway/index --out-dataset outputs/retrieval_cases.json \
        --stratified-dataset \
            synthway/output_analysis/stratified_dataset/stratified_dataset.jsonl \
        --dataset-query-fields composition,method --expected slots --expected-slots ""

    python synthway/prepare_retrieval_eval.py --corpus-dir synthway/json \
        --index-dir synthway/index --out-dataset outputs/retrieval_cases_Au.json \
        --composition Au --method "Chemical Reduction" --max-cases 5
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Avoid HuggingFace tokenizers fork deadlock warnings in DeepEval runs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import retriever
from build_rag_index import RagIndex, apply_post_filters, format_query, rerank_by_slot_coverage

# Optional DeepEval support
try:
    from deepeval.metrics import (
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase

    _DEEPEVAL = True
except Exception:
    _DEEPEVAL = False


# JSON extraction utilities
def _safe_get(d: dict[str, Any], path: str, default=None):
    """Safely obtain a nested dictionary field using an 'a.b.c' path."""
    cur = d
    for k in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return default if cur is None else cur


def _names(list_of_dicts: list[dict[str, Any]] | None) -> list[str]:
    """Extract names from a list of dictionaries with a 'name' key, e.g., precursors/additives."""
    out: list[str] = []
    for x in list_of_dicts or []:
        if isinstance(x, dict):
            n = x.get("name")
            if n:
                out.append(str(n))
    return out


def _normalize(s: str | None) -> str | None:
    """Normalize a string for comparison by collapsing whitespace and standardizing its format."""
    if s is None:
        return None
    return re.sub(r"\s+", " ", str(s)).strip()


def _parse_fields(csv_text: str) -> set:
    """Parse comma-separated fields into a set of lowercase names."""
    return {x.strip().lower() for x in (csv_text or "").split(",") if x.strip()}


# Slot-name normalization
def _canonical_slot_name(name: str) -> str:
    """
    Convert a slot name to its canonical form for matching.
    For example:
      'precursors' -> 'precursor'
      'additives'  -> 'additive'
      'seeds'      -> 'seed'
      'ph'         -> 'pH' (as used in retrieval_text)
    Lowercase all other names.
    """
    if name is None:
        return ""
    n = name.strip()
    if not n:
        return ""
    low = n.lower()
    if low == "precursors":
        return "precursor"
    if low == "additives":
        return "additive"
    if low == "seeds":
        return "seed"
    if low in ("ph", "pH".lower()):
        return "pH"
    # Keep size.length / size.width / size.depth, etc., in lowercase
    return low


def _canonical_slot_set(slots: set) -> set:
    """Convert a set of CLI slots to canonical names."""
    return {_canonical_slot_name(s) for s in (slots or set()) if s}


# Build INPUT
def _build_cli_input(args) -> str:
    """
    Build `input` from CLI flags as a compact query string, e.g.:
      "composition: Au | method: Chemical Reduction | solvent: water | pH: 6 | precursor: HAuCl4"
    """
    parts = []
    if args.composition:
        parts.append(f"composition: {args.composition}")
    if args.method:
        parts.append(f"method: {args.method}")
    if args.solvent:
        parts.append(f"solvent: {args.solvent}")
    if args.temperature is not None:
        parts.append(f"temperature: {args.temperature}")
    if args.ph is not None:
        parts.append(f"pH: {args.ph}")
    for p in args.precursor or []:
        parts.append(f"precursor: {p}")
    for a in args.additive or []:
        parts.append(f"additive: {a}")
    for s in args.seed or []:
        parts.append(f"seed: {s}")
    return " | ".join(parts) if parts else "composition: ?"


def _build_record_input(rec: dict[str, Any], fields_csv: str) -> str:
    """
    Build `input` in DATASET mode for each corpus record using the specified fields.
    For example, fields_csv="composition,method" produces "composition: X | method: Y".
    """
    comp = rec.get("composition") or "material"
    fields = _parse_fields(fields_csv)
    parts = []
    if "composition" in fields:
        parts.append(f"composition: {comp}")
    if "method" in fields and rec.get("method"):
        parts.append(f"method: {rec.get('method')}")
    return " | ".join(parts) if parts else f"composition: {comp}"


def _parse_retrieval_text(rt: str | None) -> dict[str, list[str]]:
    """
    Parse retrieval_text into {label -> [values]}.
    Example:
      "composition: Co1.5Mn1.5O4 | temperature: 80.0 | precursor: Co(NO3)2⋅6H2O (mass 1.0)"
    ->
      {
        "composition": ["Co1.5Mn1.5O4"],
        "temperature": ["80.0"],
        "precursor": ["Co(NO3)2⋅6H2O (mass 1.0)"]
      }
    """
    slots: dict[str, list[str]] = {}
    if not isinstance(rt, str):
        return slots
    for chunk in rt.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            continue
        label, val = chunk.split(":", 1)
        label = label.strip()
        val = val.strip()
        if not label or not val:
            continue
        slots.setdefault(label, []).append(val)
    return slots


def _build_expected_from_retrieval_text(rec: dict[str, Any], slots: set) -> str | None:
    """
    Build expected_output from a subset of rec['retrieval_text'] slots.
    If slots is empty, include all slots.
    """
    rt = rec.get("retrieval_text")
    if not isinstance(rt, str) or not rt.strip():
        return None

    parsed = _parse_retrieval_text(rt)
    if not parsed:
        return None

    # Canonical slots to retain
    canon_slots = _canonical_slot_set(slots)
    use_all = not canon_slots  # No slots explicitly specified by the user

    selected_chunks: list[str] = []

    for raw_label, values in parsed.items():
        # raw_label as in retrieval_text (e.g., "temperature", "pH", "precursor")
        canon_label = _canonical_slot_name(raw_label)
        if (not use_all) and (canon_label not in canon_slots):
            continue
        for v in values:
            # Reassemble in the same format so DeepEval sees the same facts
            selected_chunks.append(f"{raw_label}: {v}")

    if not selected_chunks:
        return None

    return " | ".join(selected_chunks)


# Build EXPECTED (reference output)
def _build_expected_from_json(rec: dict[str, Any], canon_slots: set) -> str:
    """
    Fallback when retrieval_text is unavailable.
    Build expected_output from JSON fields using English slot names
    consistent with retrieval_text.
    """
    # Main synthesis conditions
    method = rec.get("method")
    temp = _safe_get(rec, "conditions.temperature")
    ph = _safe_get(rec, "conditions.ph") or _safe_get(rec, "conditions.pH")
    solv_name = _safe_get(rec, "solvent.name") or rec.get("solvent")
    solv_vol = _safe_get(rec, "solvent.volume")
    time_val = _safe_get(rec, "conditions.time")

    # Starting substances
    prec = _names(rec.get("precursors"))
    add = _names(rec.get("additives"))
    seed = _names(rec.get("seeds"))

    # Product parameters
    shape = _safe_get(rec, "product.shape") or rec.get("shape")
    crys = (
        _safe_get(rec, "product.crystalsystem")
        or rec.get("crystalsystem")
        or _safe_get(rec, "product.crystal_system")
    )
    size_len = _safe_get(rec, "product.size.length")
    size_wid = _safe_get(rec, "product.size.width")
    size_dep = _safe_get(rec, "product.size.depth")

    parts: list[str] = []

    if "method" in canon_slots and method:
        parts.append(f"method: {method}")
    if "temperature" in canon_slots and temp is not None:
        parts.append(f"temperature: {temp}")
    if "pH" in canon_slots and ph is not None:
        parts.append(f"pH: {ph}")
    if "solvent" in canon_slots and solv_name:
        parts.append(f"solvent: {solv_name}")
    if "solvent.volume" in canon_slots and solv_vol is not None:
        parts.append(f"solvent.volume: {solv_vol}")
    if "time" in canon_slots and time_val is not None:
        parts.append(f"time: {time_val}")

    if "precursor" in canon_slots and prec:
        for p in prec:
            parts.append(f"precursor: {p}")
    if "additive" in canon_slots and add:
        for a in add:
            parts.append(f"additive: {a}")
    if "seed" in canon_slots and seed:
        for s in seed:
            parts.append(f"seed: {s}")

    if "shape" in canon_slots and shape:
        parts.append(f"shape: {shape}")
    if "crystalsystem" in canon_slots and crys:
        parts.append(f"crystalsystem: {crys}")

    if "size.length" in canon_slots and size_len is not None:
        parts.append(f"size.length: {size_len}")
    if "size.width" in canon_slots and size_wid is not None:
        parts.append(f"size.width: {size_wid}")
    if "size.depth" in canon_slots and size_dep is not None:
        parts.append(f"size.depth: {size_dep}")

    return " | ".join(parts).strip()


def _build_expected(rec: dict[str, Any], mode: str, slots: set) -> str:
    """
    Build expected_output:

      - "slots" mode:
          * parse rec['retrieval_text'],
          * retain only the slots specified by --expected-slots,
          * assemble a string such as "temperature: 80.0 | solvent: water | precursor: ... | additive: ...".
        If retrieval_text is unavailable, use the JSON fallback with the same slots.

      - "pass_through" mode:
          * use the original retrieval_text unchanged (full document context).

    Supported CLI slot names (--expected-slots):
      composition, method, temperature, pH, solvent, solvent.volume,
      precursor / precursors, additive / additives, seed / seeds,
      time, shape, crystalsystem, size.length, size.width, size.depth, etc.
    """
    if mode == "pass_through":
        return (rec.get("retrieval_text") or "").strip()

    # Canonicalize the requested slots
    canon_slots = _canonical_slot_set(slots)

    # 1) Prefer building the reference directly from retrieval_text for consistency with the context
    exp_rt = _build_expected_from_retrieval_text(rec, slots)
    if exp_rt:
        return exp_rt

    # 2) Fallback: use JSON fields with the same slots as retrieval_text
    exp = _build_expected_from_json(rec, canon_slots)
    if exp:
        return exp

    # 3) Last resort: return the complete retrieval_text
    return (rec.get("retrieval_text") or "").strip()


def _check_topk_hit(
    expected_output: str, retrieval_context: list[str] | None
) -> (bool, int | None):
    """
    Check whether expected_output occurs in retrieval_context.

    Logic:
      - normalize expected_output by collapsing whitespace,
      - for each retrieval_context element:
          * normalize the string,
          * check whether expected_output is a substring.
    Returns:
      hit (bool), rank (0-based retrieval_context index, or None if not found).
    """
    if not expected_output or not retrieval_context:
        return False, None

    exp_norm = _normalize(expected_output)
    if not exp_norm:
        return False, None

    for idx, ctx in enumerate(retrieval_context):
        if not isinstance(ctx, str):
            continue
        ctx_norm = _normalize(ctx)
        if exp_norm and ctx_norm and exp_norm in ctx_norm:
            return True, idx
    return False, None


def _compute_pair_doc_fraction(
    input_str: str, retrieval_context: list[str] | None
) -> (float | None, bool | None):
    """
    Calculate the fraction of top-k documents whose (composition, method) pair
    matches the query.

    Logic:
      - parse input into slots and extract composition and method,
      - parse each retrieval_context document,
      - count the fraction matching both composition and method.

    Returns:
      pair_doc_fraction (0..1), or None if input lacks composition/method,
      pair_hit (bool): True if at least one matching document exists.
    """
    if not input_str or not retrieval_context:
        return None, None

    input_slots = _parse_retrieval_text(input_str)
    if not input_slots:
        return None, None

    comp_vals = input_slots.get("composition") or []
    meth_vals = input_slots.get("method") or []
    if not comp_vals or not meth_vals:
        return None, None

    # Use the first composition and method values from the query.
    q_comp = _normalize(comp_vals[0])
    q_meth = _normalize(meth_vals[0])
    if not q_comp or not q_meth:
        return None, None

    total_docs = 0
    matched_docs = 0

    for ctx in retrieval_context:
        if not isinstance(ctx, str):
            continue
        ctx_slots = _parse_retrieval_text(ctx)
        ctx_comp_vals = [(_normalize(v) or "") for v in (ctx_slots.get("composition") or [])]
        ctx_meth_vals = [(_normalize(v) or "") for v in (ctx_slots.get("method") or [])]

        # Include documents missing composition/method in the denominator, without counting a match
        total_docs += 1
        if not ctx_comp_vals or not ctx_meth_vals:
            continue

        if (q_comp in ctx_comp_vals) and (q_meth in ctx_meth_vals):
            matched_docs += 1

    if total_docs == 0:
        return None, None

    frac = matched_docs / float(total_docs)
    return frac, (matched_docs > 0)


# Fraction of retrieved documents matching one slot.
def _compute_single_slot_fraction(
    slot_name: str, input_str: str, retrieval_context: list[str] | None
) -> (float | None, bool | None):
    """
    Computes the fraction of documents in retrieval_context whose slot value matches that in input_str for the given slot_name.
    Returns (fraction, hit) where fraction is None if not computable, hit is True if at least one doc matches.
    """
    if not input_str or not retrieval_context:
        return None, None
    input_slots = _parse_retrieval_text(input_str)
    if not input_slots:
        return None, None
    vals = input_slots.get(slot_name) or []
    if not vals:
        return None, None
    q_val = _normalize(vals[0])
    if not q_val:
        return None, None
    total_docs = 0
    matched_docs = 0
    for ctx in retrieval_context:
        if not isinstance(ctx, str):
            continue
        ctx_slots = _parse_retrieval_text(ctx)
        ctx_vals = [(_normalize(v) or "") for v in (ctx_slots.get(slot_name) or [])]
        total_docs += 1
        if not ctx_vals:
            continue
        if q_val in ctx_vals:
            matched_docs += 1
    if total_docs == 0:
        return None, None
    frac = matched_docs / float(total_docs)
    return frac, (matched_docs > 0)


def _match_filters(rec: dict[str, Any], args) -> bool:
    """
    Check whether a corpus record matches the specified CLI filters.
    (composition, method, solvent, temperature, pH, precursor, additive, seed).
    """

    def eq(field_val, arg_val):
        if arg_val is None:
            return True
        return _normalize(field_val) == _normalize(arg_val)

    # Simple scalar fields
    if args.composition is not None and not eq(rec.get("composition"), args.composition):
        return False
    if args.method is not None and not eq(rec.get("method"), args.method):
        return False
    if args.solvent is not None:
        solv = rec.get("solvent") or _safe_get(rec, "solvent.name")
        if not eq(solv, args.solvent):
            return False
    if args.temperature is not None:
        t = _safe_get(rec, "conditions.temperature")
        if str(t) != str(args.temperature):
            return False
    if args.ph is not None:
        ph = _safe_get(rec, "conditions.pH")
        if str(ph) != str(args.ph):
            return False

    # List-valued fields
    def contains_all(names_list, arg_list):
        if not arg_list:
            return True
        names_lower = {str(x).lower() for x in (names_list or [])}
        for a in arg_list:
            if str(a).lower() not in names_lower:
                return False
        return True

    prec_names = _names(rec.get("precursors"))
    add_names = _names(rec.get("additives"))
    seed_names = _names(rec.get("seeds"))
    if not contains_all(prec_names, args.precursor):
        return False
    if not contains_all(add_names, args.additive):
        return False
    if not contains_all(seed_names, args.seed):
        return False
    return True


from types import SimpleNamespace


def _make_slot_args_for_rerank(
    query_comp: str | None,
    query_method: str | None,
    query_shape: str | None,
    query_temperature: str | None,
    query_solvent: str | None,
    query_time: str | None,
    query_ph: str | None,
    query_prec_names: list[str] | None,
    query_add_names: list[str] | None,
    query_seed_names: list[str] | None,
):
    """
    Build an argument object resembling build_rag_index.py CLI arguments
    so rerank_by_slot_coverage can identify the specified slots.
    """
    return SimpleNamespace(
        composition=query_comp,
        method=query_method,
        shape=query_shape,
        temperature=query_temperature,
        solvent=query_solvent,
        time=query_time,
        pH=query_ph,
        precursor=query_prec_names,
        additive=query_add_names,
        seed=query_seed_names,
        # Additional slots are not yet used in evaluation
        crystalsystem=None,
        size_length=None,
        size_width=None,
        size_depth=None,
        pdfno=None,
    )


def _retrieve_context(
    ri: RagIndex,
    query_comp: str | None,
    query_method: str | None,
    k: int,
    preselect: int = 30,
    min_lex_score: float | None = None,
    *,
    elements: list[str] | None = None,
    req_precursors: list[str] | None = None,
    req_additives: list[str] | None = None,
    req_seeds: list[str] | None = None,
    filter_method: str | None = None,
    query_shape: str | None = None,
    query_temperature: str | None = None,
    query_solvent: str | None = None,
    query_time: str | None = None,
    query_ph: str | None = None,
    query_prec_names: list[str] | None = None,
    query_add_names: list[str] | None = None,
    query_seed_names: list[str] | None = None,
) -> list[str]:
    """Search the index for contexts.

    Steps:
      1. Build a text query with format_query using the same fields
         as predictions.py (composition, method, shape, temperature,
         solvent, time, pH, precursors, additives, seeds).
      2. Expand the query with expand_query_with_denselex.
      3. Run TF-IDF search over the index.
      4. Apply post-filters (method, elements, required precursors/additives/seeds).
    Return up to k retrieval_text strings.
    """
    q = format_query(
        composition=query_comp,
        method=query_method,
        shape=query_shape,
        temperature=query_temperature,
        solvent=query_solvent,
        time=query_time,
        pH=query_ph,
        precursors=query_prec_names,
        additives=query_add_names,
        seeds=query_seed_names,
    )
    q = ri.expand_query_with_denselex(q)

    n_pre = max(k * preselect, k)
    raw_hits = ri.retrieve_lexical(q, k=n_pre, min_score=min_lex_score)

    if query_comp == "Co80Ni20":
        print("[DEBUG] Co80Ni20 query:", q)
        print("[DEBUG] Co80Ni20 raw_hits:", len(raw_hits))
        for h in raw_hits[:5]:
            print("   raw_hit:", h.sid, repr(h.composition), repr(h.method), h.score)

    hits = apply_post_filters(
        ri,
        raw_hits,
        k,
        filter_method=filter_method,
        elements=(elements or None),
        req_precursors=(req_precursors or None),
        req_additives=(req_additives or None),
        req_seeds=(req_seeds or None),
        composition_filter=query_comp,
    )

    if query_comp == "Co80Ni20":
        print("[DEBUG] Co80Ni20 hits after filters:", len(hits))
        for h in hits[:5]:
            print("   hit:", h.sid, repr(h.composition), repr(h.method), h.score)

    # Slot-aware reranking to match the application retriever
    slot_args = _make_slot_args_for_rerank(
        query_comp=query_comp,
        query_method=query_method,
        query_shape=query_shape,
        query_temperature=query_temperature,
        query_solvent=query_solvent,
        query_time=query_time,
        query_ph=query_ph,
        query_prec_names=query_prec_names,
        query_add_names=query_add_names,
        query_seed_names=query_seed_names,
    )
    hits = rerank_by_slot_coverage(hits, slot_args)

    return [h.retrieval_text for h in hits if getattr(h, "retrieval_text", None)]


def main():
    ap = argparse.ArgumentParser(
        description="Generate a retrieval evaluation dataset and optionally compute DeepEval metrics."
    )
    # Required arguments
    ap.add_argument(
        "--corpus-dir", required=True, help="Directory of corpus JSON files (syntheses)"
    )
    ap.add_argument(
        "--index-dir",
        required=True,
        help="Directory of an existing RagIndex built with build_rag_index.py",
    )
    ap.add_argument("--out-dataset", required=True, help="JSON output path for generated cases")
    ap.add_argument(
        "--stratified-dataset",
        type=str,
        default=None,
        help="Path to the stratified sample JSONL (one JSON record per line). "
        "If supplied, DATASET mode uses only these records instead of all JSON files in --corpus-dir.",
    )

    # Retrieval (RAG) settings
    ap.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-K retrieval_context size (number of context strings to save)",
    )
    ap.add_argument(
        "--preselect",
        type=int,
        default=30,
        help="Candidate multiplier before filtering (take k*preselect candidates before post-filters)",
    )
    ap.add_argument(
        "--min-lex-score",
        type=float,
        default=None,
        help="Minimum TF-IDF candidate score (optional)",
    )

    # expected_output format
    ap.add_argument(
        "--expected",
        choices=["slots", "pass_through"],
        default="slots",
        help="expected_output format: "
        "slots — subset of retrieval_text slots; "
        "pass_through — complete original retrieval_text.",
    )
    ap.add_argument(
        "--expected-slots",
        type=str,
        # Check physically important synthesis parameters by default
        default="method,temperature,pH,solvent,precursors,additives",
        help=(
            "Comma-separated slots to include in expected_output. "
            "Use retrieval_text-style names: composition,method,temperature,pH,solvent,solvent.volume,"
            "precursor/precursors,additive/additives,seed/seeds,time,shape,crystalsystem,size.length,size.width,size.depth"
        ),
    )

    # DATASET mode settings
    ap.add_argument(
        "--dataset-query-fields",
        default="composition",
        help="Record fields used to build DATASET queries (CSV), e.g., composition | composition,method",
    )

    # QUERY mode filters (any specified filter activates QUERY mode)
    ap.add_argument("--composition", type=str, default=None, help="Filter by composition")
    ap.add_argument("--method", type=str, default=None, help="Filter by synthesis method")
    ap.add_argument("--solvent", type=str, default=None, help="Filter by solvent")
    ap.add_argument(
        "--temperature",
        type=str,
        default=None,
        help="Filter by temperature (conditions.temperature)",
    )
    ap.add_argument("--ph", type=str, default=None, help="Filter by pH (conditions.pH)")
    ap.add_argument(
        "--precursor", action="append", help="Filter by precursor name (repeatable)", default=None
    )
    ap.add_argument(
        "--additive", action="append", help="Filter by additive name (repeatable)", default=None
    )
    ap.add_argument(
        "--seed", action="append", help="Filter by seed name (repeatable)", default=None
    )
    ap.add_argument(
        "--max-cases", type=int, default=None, help="Maximum cases to process (limit sample size)"
    )

    # DeepEval evaluation
    ap.add_argument(
        "--evaluate", action="store_true", help="Enable retrieval evaluation with DeepEval metrics"
    )
    ap.add_argument(
        "--judge-model",
        default="gemini-2.0-flash-001",
        help="LLM judge for DeepEval (e.g., gemini-2.0-flash-001, gpt-4, anthropic-claude-2, etc.)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Pass/fail threshold (0–1) for metric evaluation",
    )

    # Parameter grid and summary
    ap.add_argument(
        "--k-list",
        type=str,
        default=None,
        help="Comma-separated k values for a grid run, e.g., 3,5,7",
    )
    ap.add_argument(
        "--field-sets",
        type=str,
        default=None,
        help="Semicolon-separated DATASET field sets, e.g., 'composition;composition,method'",
    )
    ap.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="CSV output path for grid-run summaries. Defaults to a file beside out-dataset with the _summary.csv suffix.",
    )
    ap.add_argument(
        "--eval-concurrency",
        type=int,
        default=1,
        help="DeepEval evaluation concurrency (1 = sequential).",
    )
    ap.add_argument(
        "--eval-rpm",
        type=int,
        default=None,
        help="Judge-model requests per minute. Sequential evaluation inserts pauses according to this limit.",
    )
    ap.add_argument(
        "--resume-eval",
        action="store_true",
        help="When rerunning with --evaluate, try reusing calculated metrics from an existing *_metrics.csv instead of recomputing them.",
    )

    args = ap.parse_args()

    # Parse expected_output slots from a comma-separated string into a name set
    expected_slots = _parse_fields(getattr(args, "expected_slots", ""))

    # Initialize the retriever index once
    ri = RagIndex()
    ri.load(args.index_dir)

    # Load corpus JSON records or the stratified sample
    records: list[dict[str, Any]] = []

    if args.stratified_dataset:
        # Stratified sample in JSONL format: one record per line.
        with open(args.stratified_dataset, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict) or not rec:
                    continue
                # Extract composition and method from retrieval_text if present.
                rt = rec.get("retrieval_text")
                if isinstance(rt, str):
                    slots = _parse_retrieval_text(rt)
                    comp_vals = slots.get("composition") or []
                    meth_vals = slots.get("method") or []
                    comp = comp_vals[0] if comp_vals else None
                    meth = meth_vals[0] if meth_vals else None
                    if comp is not None:
                        rec.setdefault("composition", comp)
                    if meth is not None:
                        rec.setdefault("method", meth)
                records.append(rec)
    else:
        # Default workflow: use all corpus JSON files.
        files = sorted(glob.glob(os.path.join(args.corpus_dir, "*.json")))
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    rec = json.load(f)
            except Exception:
                continue  # Skip invalid JSON files
            if isinstance(rec, dict) and rec:
                records.append(rec)

    # Determine the mode (DATASET vs QUERY)
    any_query_flag = any(
        [
            args.composition,
            args.method,
            args.solvent,
            args.temperature,
            args.ph,
            args.precursor,
            args.additive,
            args.seed,
        ]
    )

    def _build_cases_for(fields_csv: str, k_val: int) -> list[dict[str, Any]]:
        cases_local: list[dict[str, Any]] = []
        if any_query_flag:
            # QUERY mode: filter records by the specified parameters and build one input from CLI flags
            filtered = [r for r in records if _match_filters(r, args)]
            if not filtered:
                print(
                    "[WARN] No corpus records match the specified filters; no cases were created."
                )
                return cases_local

            # QUERY mode: one shared input from CLI flags; reference output from each filtered record
            input_str = _build_cli_input(args)
            subset = filtered[: args.max_cases] if args.max_cases else filtered
            for rec in subset:
                expected = _build_expected(rec, args.expected, expected_slots)
                ctx = _retrieve_context(
                    ri,
                    args.composition or rec.get("composition"),
                    args.method or rec.get("method"),
                    k_val,
                    args.preselect,
                    args.min_lex_score,
                    elements=None,
                    req_precursors=(args.precursor or None),
                    req_additives=(args.additive or None),
                    req_seeds=(args.seed or None),
                    filter_method=(args.method or None),
                    query_shape=None,  # shape is not yet exposed through the CLI
                    query_temperature=args.temperature,
                    query_solvent=args.solvent,
                    query_time=None,  # time is also absent from the CLI
                    query_ph=args.ph,
                    query_prec_names=args.precursor,
                    query_add_names=args.additive,
                    query_seed_names=args.seed,
                )
                rid = (
                    rec.get("synthesis_id")
                    or _safe_get(rec, "source.synthesis_id_original")
                    or rec.get("composition")
                    or "case"
                )
                cases_local.append(
                    {
                        "id": rid,
                        "input": input_str,
                        "expected_output": expected,
                        "retrieval_context": ctx,
                    }
                )
            return cases_local
        # DATASET mode: build one case per corpus record, limited by max_cases if specified
        _fields_raw = _parse_fields(fields_csv)
        _fields = {_canonical_slot_name(f) for f in _fields_raw}  # For precursor/additive/seed

        for rec in records[: args.max_cases] if args.max_cases else records:
            input_q = _build_record_input(rec, fields_csv)
            expected = _build_expected(rec, args.expected, expected_slots)
            query_comp = rec.get("composition") if "composition" in _fields_raw else None
            query_method = rec.get("method") if "method" in _fields_raw else None
            # Build post-filters from fields a user could specify, only if included in fields_csv
            use_prec = "precursor" in _fields
            use_add = "additive" in _fields
            use_seed = "seed" in _fields
            req_prec = _names(rec.get("precursors")) if use_prec else None
            req_add = _names(rec.get("additives")) if use_add else None
            req_seed = _names(rec.get("seeds")) if use_seed else None
            # Collect additional format_query fields if included in fields
            query_temp = (
                _safe_get(rec, "conditions.temperature") if "temperature" in _fields_raw else None
            )
            query_solvent = None
            if "solvent" in _fields_raw:
                query_solvent = rec.get("solvent") or _safe_get(rec, "solvent.name")
            query_time = _safe_get(rec, "conditions.time") if "time" in _fields_raw else None
            query_ph = None
            if "pH" in _fields_raw or "ph" in _fields_raw:
                query_ph = _safe_get(rec, "conditions.ph") or _safe_get(rec, "conditions.pH")
            query_shape = None
            if "shape" in _fields_raw:
                query_shape = _safe_get(rec, "product.shape") or rec.get("shape")
            # For term-level fields, use names already computed for req_* if included in fields
            query_prec_names = req_prec if use_prec else None
            query_add_names = req_add if use_add else None
            query_seed_names = req_seed if use_seed else None
            ctx = _retrieve_context(
                ri,
                query_comp,
                query_method,
                k_val,
                args.preselect,
                args.min_lex_score,
                elements=None,
                req_precursors=req_prec,
                req_additives=req_add,
                req_seeds=req_seed,
                filter_method=query_method,
                query_shape=query_shape,
                query_temperature=query_temp,
                query_solvent=query_solvent,
                query_time=query_time,
                query_ph=query_ph,
                query_prec_names=query_prec_names,
                query_add_names=query_add_names,
                query_seed_names=query_seed_names,
            )
            rid = (
                rec.get("synthesis_id")
                or _safe_get(rec, "source.synthesis_id_original")
                or rec.get("composition")
                or "case"
            )
            cases_local.append(
                {
                    "id": rid,
                    "input": input_q,
                    "expected_output": expected,
                    "retrieval_context": ctx,
                }
            )
        return cases_local

    # If evaluation is enabled, check API keys for the selected model
    if args.evaluate:
        jm = (args.judge_model or "").lower()

        # OpenRouter: OPENROUTER_API_KEY, even if the identifier contains "gemini"
        if jm.startswith("openrouter/"):
            if not os.getenv("OPENROUTER_API_KEY"):
                raise RuntimeError(
                    "OPENROUTER_API_KEY is missing for OpenRouter (required for --judge-model openrouter/*)."
                )
        else:
            # Native Gemini: GEMINI_API_KEY
            if "gemini" in jm and not os.getenv("GEMINI_API_KEY"):
                raise RuntimeError(
                    "GEMINI_API_KEY is missing for Gemini (required for --judge-model gemini*)."
                )

    def _save_and_eval(
        cases_local: list[dict[str, Any]], out_path: str, k_for_stats: int | None = None
    ):
        # Save generated cases as JSON
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cases_local, f, ensure_ascii=False, indent=2)
        print(f"[OK] saved {len(cases_local)} test cases -> {out_path}")

        # Also save flat JSONL containing the main case fields
        base_no_ext, _ = os.path.splitext(out_path)
        flat_cases_path = base_no_ext + "_cases.jsonl"
        with open(flat_cases_path, "w", encoding="utf-8") as cf:
            for c in cases_local:
                row = {
                    "id": c.get("id", ""),
                    "input": c.get("input", ""),
                    "expected_output": c.get("expected_output", ""),
                    "retrieval_context": c.get("retrieval_context", []),
                }
                cf.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[OK] saved flat cases -> {flat_cases_path}")
        # Metrics CSV path, also usable for resume logic
        metrics_csv_path = base_no_ext + "_metrics.csv"

        # Basic offline retriever metrics (Top@k and pair_doc_fraction@k)
        base_results: list[dict[str, Any]] = []

        for c in cases_local:
            inp = c.get("input", "") or ""
            exp = c.get("expected_output", "") or ""
            ctx = c.get("retrieval_context") or []

            hit, rank = _check_topk_hit(exp, ctx)
            pair_frac, pair_hit = _compute_pair_doc_fraction(inp, ctx)
            comp_frac, comp_hit = _compute_single_slot_fraction("composition", inp, ctx)

            base_results.append(
                {
                    "id": c.get("id", ""),
                    "hit_in_topk": bool(hit),
                    "hit_rank": rank,
                    "pair_doc_fraction": float(pair_frac) if pair_frac is not None else None,
                    "pair_hit_in_topk": bool(pair_hit) if pair_hit is not None else None,
                    "comp_doc_fraction": float(comp_frac) if comp_frac is not None else None,
                    "comp_hit_in_topk": bool(comp_hit) if comp_hit is not None else None,
                    "retrieved_k": len(ctx),
                }
            )

        # Aggregate statistics for these metrics
        total_cases = len(base_results)
        if total_cases > 0:
            hit_count = sum(1 for r in base_results if r["hit_in_topk"])
            hit_rate = hit_count / float(total_cases)
            pair_values = [
                r["pair_doc_fraction"] for r in base_results if r["pair_doc_fraction"] is not None
            ]
            pair_hit_count = sum(1 for r in base_results if r["pair_hit_in_topk"])
            # Actual k used to build cases, relevant in GRID mode
            k_used = k_for_stats if k_for_stats is not None else args.k

            print(
                f"[STATS] Top@{k_used} hit-rate (expected_output ∈ retrieval_context): "
                f"{hit_count}/{total_cases} = {hit_rate:.3f}"
            )
            if pair_values:
                avg_pair = sum(pair_values) / float(len(pair_values))
                print(
                    f"[STATS] pair_doc_fraction@{k_used} (docs with matching composition+method): "
                    f"avg={avg_pair:.3f}, any_pair_hit={pair_hit_count}/{total_cases}"
                )

            # Aggregated stats for comp_doc_fraction@k
            comp_cases = [r for r in base_results if r["comp_doc_fraction"] is not None]
            if comp_cases:
                comp_values = [r["comp_doc_fraction"] for r in comp_cases]
                comp_hit_count = sum(1 for r in comp_cases if r["comp_hit_in_topk"])
                n_comp = len(comp_cases)
                avg_comp = sum(comp_values) / float(n_comp)
                print(
                    f"[STATS] comp_doc_fraction@{k_used} (docs with matching composition): "
                    f"avg={avg_comp:.3f}, any_comp_hit={comp_hit_count}/{n_comp}"
                )

        # Reuse existing *_metrics.csv values with --resume-eval
        if (
            args.evaluate
            and getattr(args, "resume_eval", False)
            and os.path.exists(metrics_csv_path)
        ):
            print(
                f"[RESUME] Existing metrics file found -> {metrics_csv_path}; reusing calculated values."
            )
            metrics_by_id = {}
            try:
                with open(metrics_csv_path, "r", encoding="utf-8") as mf:
                    reader = csv.DictReader(mf)
                    for row in reader:
                        rid = row.get("id")
                        if not rid:
                            continue
                        metrics_by_id[rid] = row
            except Exception as e:
                print(f"[RESUME] Failed to load existing metrics: {e}")
                metrics_by_id = {}

            # Update base_results with previously calculated values
            for res in base_results:
                rid = res.get("id")
                if not rid or rid not in metrics_by_id:
                    continue
                row = metrics_by_id[rid]
                # Restore numeric and boolean metric values if available
                for key in ["precision_score", "recall_score", "relevancy_score"]:
                    val = row.get(key)
                    if val not in (
                        None,
                        "",
                    ):
                        try:
                            res[key] = float(val)
                        except Exception:
                            pass
                for key in ["precision_pass", "recall_pass", "relevancy_pass"]:
                    val = row.get(key)
                    if val in ["True", "true", "1", "yes"]:
                        res[key] = True
                    elif val in ["False", "false", "0", "no"]:
                        res[key] = False

        # DeepEval (optional)
        if args.evaluate:
            if not _DEEPEVAL:
                raise RuntimeError("DeepEval is not installed. Install it: pip install -U deepeval")
            # Resolve the judge model object or string identifier
            judge_obj = args.judge_model

            # 1) OpenRouter through an OpenAI-compatible endpoint
            if isinstance(judge_obj, str) and judge_obj.lower().startswith("openrouter/"):
                # Use an OpenAI-compatible client for OpenRouter, as in predictions.py.
                # Strip `openrouter/`: OpenRouter expects names such as `google/gemini-...` or `anthropic/claude-...`.
                or_key = os.getenv("OPENROUTER_API_KEY")
                if not or_key:
                    raise RuntimeError(
                        "OPENROUTER_API_KEY is unset; set the OpenRouter API key in the environment."
                    )

                try:
                    from openai import OpenAI
                except Exception as e:
                    raise RuntimeError(
                        "Package `openai` not found. Install it: pip install -U openai"
                    ) from e

                try:
                    # DeepEval requires a model compatible with its base interface
                    from deepeval.models.base_model import DeepEvalBaseLLM
                except Exception as e:
                    raise RuntimeError(
                        "Failed to import DeepEvalBaseLLM from deepeval. Check the deepeval installation."
                    ) from e

                base_url = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

                # Optional OpenRouter attribution headers, as in predictions.py
                extra_headers = {}
                if os.getenv("OPENROUTER_HTTP_REFERER"):
                    extra_headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER")
                if os.getenv("OPENROUTER_X_TITLE"):
                    extra_headers["X-Title"] = os.getenv("OPENROUTER_X_TITLE")

                # Normalize model name: `openrouter/google/gemini-2.0-flash-001` -> `google/gemini-2.0-flash-001`
                or_model = args.judge_model.split("/", 1)[1]

                class OpenRouterDeepEvalModel(DeepEvalBaseLLM):
                    def __init__(
                        self,
                        model_name: str,
                        api_key: str,
                        base_url: str,
                        extra_headers: dict | None = None,
                        temperature: float = 0.0,
                    ):
                        self.model_name = model_name
                        self.temperature = temperature
                        self.extra_headers = extra_headers or {}
                        # Increase timeout to reduce OpenRouter "Request timed out" failures during DeepEval.
                        # Can be overridden via env var OPENROUTER_TIMEOUT (seconds).
                        try:
                            timeout_s = float(os.getenv("OPENROUTER_TIMEOUT", "120"))
                        except Exception:
                            timeout_s = 120.0
                        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

                    def get_model_name(self):
                        return f"openrouter/{self.model_name}"

                    def load_model(self):
                        # DeepEvalBaseLLM requires this method. OpenRouter uses an OpenAI-compatible client,
                        # so return the initialized client.
                        return self.client

                    def _extract_json_obj(self, text: str) -> dict | None:
                        """Parse a JSON object directly or from the span between the first opening
                        brace and the last closing brace. Return None if parsing fails.
                        """
                        if not text:
                            return None
                        t = str(text).strip()

                        # 1) direct JSON
                        try:
                            obj = json.loads(t)
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            pass

                        # 2) Parse the span between the first opening and last closing brace.
                        try:
                            m = re.search(r"\{.*\}", t, flags=re.DOTALL)
                            if not m:
                                return None
                            obj = json.loads(m.group(0))
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            return None

                    def _maybe_cast_to_schema(self, text: str, schema):
                        """Use parsed JSON for schema conversion, or stripped text when no schema is given.

                        If no nonempty JSON object is parsed, populate the schema's supported
                        reason, verdicts and score fallback fields.
                        """
                        if schema is None:
                            return (text or "").strip()

                        data = self._extract_json_obj(text)

                        # Build fallback fields when parsed data is missing or empty.
                        if not isinstance(data, dict) or not data:
                            # Try to infer required keys from pydantic schema
                            keys = []
                            try:
                                # pydantic v2
                                keys = list(getattr(schema, "model_fields", {}).keys())
                            except Exception:
                                keys = []

                            # Minimal fallbacks for common DeepEval schemas
                            fallback = {}
                            if "reason" in keys:
                                fallback["reason"] = (text or "").strip()[:2000]
                            if "verdicts" in keys:
                                fallback["verdicts"] = []
                            if "score" in keys and "score" not in fallback:
                                fallback["score"] = 0.0
                            data = fallback

                        # pydantic v2
                        if hasattr(schema, "model_validate"):
                            return schema.model_validate(data)

                        # pydantic v1 / dataclass-like
                        try:
                            return schema(**data)
                        except Exception:
                            return data

                    def generate(self, prompt: str, **kwargs):
                        # DeepEval may pass `schema` (or `response_model`) for structured verdicts.
                        schema = kwargs.get("schema") or kwargs.get("response_model")

                        # DeepEval may pass a list of messages instead of a plain string.
                        if isinstance(prompt, list):
                            messages = prompt
                        else:
                            messages = [{"role": "user", "content": str(prompt)}]

                        # If schema is requested, add a strict JSON-only instruction with required keys.
                        if schema is not None:
                            try:
                                keys = list(getattr(schema, "model_fields", {}).keys())
                            except Exception:
                                keys = []
                            if keys:
                                instr = (
                                    "Return ONLY a valid JSON object and nothing else. "
                                    f"The JSON MUST contain these top-level keys: {keys}. "
                                    "Do not wrap in markdown. Do not add commentary."
                                )
                            else:
                                instr = "Return ONLY a valid JSON object and nothing else. Do not wrap in markdown."
                            messages = [{"role": "system", "content": instr}] + messages

                        def _call():
                            resp = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                temperature=self.temperature,
                                extra_headers=self.extra_headers or None,
                            )
                            return (resp.choices[0].message.content or "").strip()

                        text = _call()

                        # First attempt to cast to schema (or return raw)
                        try:
                            return self._maybe_cast_to_schema(text, schema)
                        except Exception:
                            # Retry once with an additional JSON-only system instruction.
                            if schema is not None:
                                retry_instr = (
                                    "IMPORTANT: Respond with STRICT JSON only. "
                                    "No extra keys beyond the schema. No markdown. No prose."
                                )
                                messages_retry = [
                                    {"role": "system", "content": retry_instr}
                                ] + messages
                                resp = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=messages_retry,
                                    temperature=self.temperature,
                                    extra_headers=self.extra_headers or None,
                                )
                                text2 = (resp.choices[0].message.content or "").strip()
                                return self._maybe_cast_to_schema(text2, schema)
                            raise

                    async def a_generate(self, prompt: str, **kwargs):
                        # This script runs metrics in sync mode, but keep async compatibility.
                        return self.generate(prompt, **kwargs)

                judge_obj = OpenRouterDeepEvalModel(
                    model_name=or_model,
                    api_key=or_key,
                    base_url=base_url,
                    extra_headers=extra_headers or None,
                    temperature=0.0,
                )

            # 2) Gemini through native DeepEval integration, only if not openrouter/*
            elif isinstance(judge_obj, str) and "gemini" in judge_obj.lower():
                from deepeval.models import GeminiModel

                gem_api = os.getenv("GEMINI_API_KEY")
                if not gem_api:
                    raise RuntimeError(
                        "GEMINI_API_KEY is unset; set the Gemini API key in the environment."
                    )
                judge_obj = GeminiModel(model_name=args.judge_model, api_key=gem_api, temperature=0)

            # helpers: throttling & safe measure
            def _is_quota_error(ex: Exception) -> bool:
                txt = " | ".join(
                    [
                        str(ex),
                        str(getattr(ex, "__cause__", "")),
                        str(getattr(ex, "__context__", "")),
                    ]
                )
                return ("RESOURCE_EXHAUSTED" in txt) or ("429" in txt) or ("quota" in txt.lower())

            def _is_timeout_error(ex: Exception) -> bool:
                txt = " | ".join(
                    [
                        str(ex),
                        str(getattr(ex, "__cause__", "")),
                        str(getattr(ex, "__context__", "")),
                    ]
                ).lower()
                return ("timed out" in txt) or ("timeout" in txt) or ("request timed out" in txt)

            def _throttle_sleep(estimated_calls: int):
                """Simple RPM throttling (default: 10 RPM); provider limits must be checked separately."""
                rpm = args.eval_rpm or 10
                delay = max(0.0, (estimated_calls * 60.0 / float(rpm)) * 1.10)  # 10% margin
                if delay > 0:
                    time.sleep(delay)

            def _safe_measure(metric, test_case) -> bool:
                """Run metric.measure with up to three attempts for rate-limit or timeout errors.

                Return True on success and False after retry exhaustion; propagate other errors.
                """
                max_tries = 3
                for attempt in range(1, max_tries + 1):
                    try:
                        print("[RATE LIMIT] Pausing 10 seconds before the judge-model call.")
                        time.sleep(10)
                        metric.measure(test_case)
                        return True
                    except Exception as e:
                        # Quota/rate-limit handling
                        if _is_quota_error(e):
                            wait_s = 45 if attempt < max_tries else 0
                            print(f"[RATE LIMIT] Quota/429 (attempt {attempt}/{max_tries}).")
                            if wait_s:
                                print(f"[RATE LIMIT] Waiting {wait_s} seconds before retrying.")
                                time.sleep(wait_s)
                            continue

                        # Timeout handling
                        if _is_timeout_error(e):
                            wait_s = 20 * attempt
                            print(
                                f"[TIMEOUT] Request timed out (attempt {attempt}/{max_tries}). Waiting {wait_s} seconds before retrying…"
                            )
                            time.sleep(wait_s)
                            continue

                        # Non-recoverable error
                        raise

                return False

            # Evaluate cases sequentially for reliability and API rate limiting
            for idx, c in enumerate(cases_local):
                # With --resume-eval, skip cases already containing precision_score
                if getattr(base_results[idx], "get", None) is not None:
                    if base_results[idx].get("precision_score") is not None:
                        continue

                tc = LLMTestCase(
                    input=c["input"],
                    actual_output="",  # Retrieval metrics do not require actual_output
                    expected_output=c["expected_output"],
                    retrieval_context=c["retrieval_context"],
                )
                # Create metric objects for this case
                prec_metric = ContextualPrecisionMetric(
                    model=judge_obj, threshold=args.threshold, include_reason=True, async_mode=False
                )
                recall_metric = ContextualRecallMetric(
                    model=judge_obj, threshold=args.threshold, include_reason=True, async_mode=False
                )
                relev_metric = ContextualRelevancyMetric(
                    model=judge_obj, threshold=args.threshold, include_reason=True, async_mode=False
                )

                try:
                    ok_prec = _safe_measure(prec_metric, tc)
                    ok_rec = _safe_measure(recall_metric, tc)
                    ok_rel = _safe_measure(relev_metric, tc)
                except Exception as e:
                    # Non-recoverable error (e.g., schema/validation bug): log and continue.
                    print(
                        f"[ERROR] DeepEval non-recoverable on case {idx} (id={c.get('id', '')}): {e}"
                    )
                    ok_prec = ok_rec = ok_rel = False

                # Extract results if the metric succeeded; otherwise keep None
                prec_score = getattr(prec_metric, "score", None) if ok_prec else None
                rec_score = getattr(recall_metric, "score", None) if ok_rec else None
                relev_score = getattr(relev_metric, "score", None) if ok_rel else None

                prec_pass = prec_score is not None and float(prec_score) >= args.threshold
                rec_pass = rec_score is not None and float(rec_score) >= args.threshold
                relev_pass = relev_score is not None and float(relev_score) >= args.threshold

                base_results[idx].update(
                    {
                        "precision_score": float(prec_score) if prec_score is not None else None,
                        "precision_pass": bool(prec_pass) if prec_score is not None else False,
                        "recall_score": float(rec_score) if rec_score is not None else None,
                        "recall_pass": bool(rec_pass) if rec_score is not None else False,
                        "relevancy_score": float(relev_score) if relev_score is not None else None,
                        "relevancy_pass": bool(relev_pass) if relev_score is not None else False,
                    }
                )

                # Throttle before the next case
                # Estimate LLM calls per case: ~K for relevancy and ~2 for precision/recall
                k_used = len(c.get("retrieval_context") or [])
                est_calls = max(k_used, 0 if args.k is None else args.k) + 2
                _throttle_sleep(est_calls)

            # Print the metric summary
            total = len(base_results)
            if total > 0:
                prec_vals = [
                    res.get("precision_score")
                    for res in base_results
                    if res.get("precision_score") is not None
                ]
                rec_vals = [
                    res.get("recall_score")
                    for res in base_results
                    if res.get("recall_score") is not None
                ]
                rel_vals = [
                    res.get("relevancy_score")
                    for res in base_results
                    if res.get("relevancy_score") is not None
                ]

                avg_prec = (sum(prec_vals) / float(len(prec_vals))) if prec_vals else 0.0
                avg_rec = (sum(rec_vals) / float(len(rec_vals))) if rec_vals else 0.0
                avg_relev = (sum(rel_vals) / float(len(rel_vals))) if rel_vals else 0.0

                pass_prec = sum(1 for res in base_results if res.get("precision_pass"))
                pass_rec = sum(1 for res in base_results if res.get("recall_pass"))
                pass_relev = sum(1 for res in base_results if res.get("relevancy_pass"))

                scored_any = sum(
                    1
                    for res in base_results
                    if (res.get("precision_score") is not None)
                    or (res.get("recall_score") is not None)
                    or (res.get("relevancy_score") is not None)
                )

                print(
                    f"Avg Precision: {avg_prec:.3f} (passed {pass_prec}/{total} cases; scored {len(prec_vals)}/{total})"
                )
                print(
                    f"Avg Recall:    {avg_rec:.3f} (passed {pass_rec}/{total} cases; scored {len(rec_vals)}/{total})"
                )
                print(
                    f"Avg Relevancy: {avg_relev:.3f} (passed {pass_relev}/{total} cases; scored {len(rel_vals)}/{total})"
                )
                if scored_any < total:
                    print(
                        f"[WARN] DeepEval did not score all cases ({scored_any}/{total}). See timeouts above."
                    )

        # Save detailed offline and DeepEval metric results to CSV
        with open(metrics_csv_path, "w", encoding="utf-8", newline="") as cf:
            fieldnames = [
                "id",
                "hit_in_topk",
                "hit_rank",
                "pair_doc_fraction",
                "pair_hit_in_topk",
                "comp_doc_fraction",
                "comp_hit_in_topk",
                "retrieved_k",
            ]
            if args.evaluate:
                fieldnames += [
                    "precision_score",
                    "precision_pass",
                    "recall_score",
                    "recall_pass",
                    "relevancy_score",
                    "relevancy_pass",
                ]
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for res in base_results:
                # Round numeric values to 3 decimal places for readability
                if res.get("pair_doc_fraction") is not None:
                    res["pair_doc_fraction"] = round(float(res["pair_doc_fraction"]), 3)
                if res.get("comp_doc_fraction") is not None:
                    res["comp_doc_fraction"] = round(float(res["comp_doc_fraction"]), 3)
                if args.evaluate:
                    for key in ["precision_score", "recall_score", "relevancy_score"]:
                        if res.get(key) is not None:
                            res[key] = round(float(res[key]), 3)
                writer.writerow(res)
        print(f"[OK] saved metrics results -> {metrics_csv_path}")

    # Main process: single run or grid
    do_grid = bool(args.k_list or args.field_sets)
    if not do_grid:
        # Single run: one k value and one field set
        cases = _build_cases_for(args.dataset_query_fields, args.k)
        _save_and_eval(cases, args.out_dataset, k_for_stats=args.k)
    else:
        # Grid run: iterate over k and field_sets lists
        k_values = [
            int(x.strip())
            for x in (args.k_list.split(",") if args.k_list else [args.k])
            if x.strip()
        ]
        if any_query_flag:
            # QUERY mode varies only k; ignore field_sets and use default/specified dataset_query_fields
            field_sets = [args.dataset_query_fields]
        else:
            field_sets = [
                s.strip()
                for s in (
                    args.field_sets.split(";") if args.field_sets else [args.dataset_query_fields]
                )
                if s.strip()
            ]
        base, ext = os.path.splitext(args.out_dataset)
        summary_csv_path = args.summary_csv or (base + "_summary.csv")
        # Create the summary_csv parent directory if needed
        _sum_dir = os.path.dirname(summary_csv_path)
        if _sum_dir:
            os.makedirs(_sum_dir, exist_ok=True)
        with open(summary_csv_path, "w", encoding="utf-8", newline="") as cf:
            headers = [
                "mode",
                "fields",
                "k",
                "num_cases",
                "out_dataset",
                # Coarse retrieval evaluation
                "hit_rate_exact",
                "avg_comp_precision",
                "comp_hit_rate",
                "comp_all_k_rate",
                "avg_pair_precision",
                "pair_hit_rate",
            ]
            if args.evaluate:
                headers += [
                    "avg_precision",
                    "avg_recall",
                    "avg_relevancy",
                    "passed_precision",
                    "passed_recall",
                    "passed_relevancy",
                ]

            writer = csv.DictWriter(cf, fieldnames=headers)
            writer.writeheader()

            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            def _is_true(x) -> bool:
                if x is None:
                    return False
                if isinstance(x, bool):
                    return bool(x)
                return str(x).strip().lower() in ("true", "1", "yes")

            for fields_csv in field_sets:
                for kv in k_values:
                    suffix = f"__fields={fields_csv.replace(',', '+')}__k={kv}"
                    outp = f"{base}{suffix}{ext}"
                    print(f"[GRID] fields={fields_csv} k={kv} -> {outp}")

                    cases_local = _build_cases_for(fields_csv, kv)
                    _save_and_eval(cases_local, outp, k_for_stats=kv)

                    row_data = {
                        "mode": "QUERY" if any_query_flag else "DATASET",
                        "fields": fields_csv,
                        "k": kv,
                        "num_cases": len(cases_local),
                        "out_dataset": outp,
                        "hit_rate_exact": None,
                        "avg_comp_precision": None,
                        "comp_hit_rate": None,
                        "comp_all_k_rate": None,
                        "avg_pair_precision": None,
                        "pair_hit_rate": None,
                    }

                    metrics_csv_path = os.path.splitext(outp)[0] + "_metrics.csv"
                    try:
                        with open(metrics_csv_path, "r", encoding="utf-8") as mf:
                            rows = list(csv.DictReader(mf))
                        n = len(rows)

                        if n > 0:
                            hit_exact = sum(1 for r in rows if _is_true(r.get("hit_in_topk")))
                            row_data["hit_rate_exact"] = round(hit_exact / float(n), 3)

                            comp_fracs = [_to_float(r.get("comp_doc_fraction")) for r in rows]
                            comp_fracs = [v for v in comp_fracs if v is not None]
                            if comp_fracs:
                                row_data["avg_comp_precision"] = round(
                                    sum(comp_fracs) / float(len(comp_fracs)), 3
                                )
                            comp_hits = sum(1 for r in rows if _is_true(r.get("comp_hit_in_topk")))
                            row_data["comp_hit_rate"] = round(comp_hits / float(n), 3)
                            comp_all = sum(
                                1 for r in rows if (_to_float(r.get("comp_doc_fraction")) == 1.0)
                            )
                            row_data["comp_all_k_rate"] = round(comp_all / float(n), 3)

                            pair_fracs = [_to_float(r.get("pair_doc_fraction")) for r in rows]
                            pair_fracs = [v for v in pair_fracs if v is not None]
                            if pair_fracs:
                                row_data["avg_pair_precision"] = round(
                                    sum(pair_fracs) / float(len(pair_fracs)), 3
                                )
                            pair_hits = sum(1 for r in rows if _is_true(r.get("pair_hit_in_topk")))
                            row_data["pair_hit_rate"] = round(pair_hits / float(n), 3)

                            if args.evaluate:
                                scores_prec, scores_rec, scores_relev = [], [], []
                                pass_prec = pass_rec = pass_relev = 0

                                for r in rows:
                                    v = _to_float(r.get("precision_score"))
                                    if v is not None:
                                        scores_prec.append(v)
                                    v = _to_float(r.get("recall_score"))
                                    if v is not None:
                                        scores_rec.append(v)
                                    v = _to_float(r.get("relevancy_score"))
                                    if v is not None:
                                        scores_relev.append(v)

                                    if _is_true(r.get("precision_pass")):
                                        pass_prec += 1
                                    if _is_true(r.get("recall_pass")):
                                        pass_rec += 1
                                    if _is_true(r.get("relevancy_pass")):
                                        pass_relev += 1

                                if scores_prec:
                                    row_data["avg_precision"] = round(
                                        sum(scores_prec) / float(len(scores_prec)), 3
                                    )
                                if scores_rec:
                                    row_data["avg_recall"] = round(
                                        sum(scores_rec) / float(len(scores_rec)), 3
                                    )
                                if scores_relev:
                                    row_data["avg_relevancy"] = round(
                                        sum(scores_relev) / float(len(scores_relev)), 3
                                    )

                                row_data["passed_precision"] = f"{pass_prec}/{n}"
                                row_data["passed_recall"] = f"{pass_rec}/{n}"
                                row_data["passed_relevancy"] = f"{pass_relev}/{n}"

                    except Exception as e:
                        print(f"[WARN] Failed to read {metrics_csv_path} for the summary: {e}")

                    writer.writerow(row_data)


if __name__ == "__main__":
    main()
