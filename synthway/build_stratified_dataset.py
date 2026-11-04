"""
build_stratified_dataset.py prepares a stratified sample with N cases per checklist slot (default: one), controlled by --per-slot.

Workflow:
1) Load all JSON files from input_dir.
2) Count composition and method frequencies for top/rare tags.
3) Tag each synthesis along the checklist axes (see TAGS_NEEDED).
4) Select EXACTLY N unique cases for EACH tag/slot where possible (N = --per-slot, default 1):
   - Pass 1: singleton-first; prioritize slots with exactly one corpus candidate (fill 1 of N positions).
   - Pass 2: round-robin across slots until N/slot, preferring new composition/method values for diversity.
   - Within each slot, rank by gain in new tags and then total tag count.
   - Break ties randomly, controlled by --random-seed.
5) Save:
   - stratified_dataset.jsonl: selected records with tags (one case per line);
   - final_stratified_dataset.jsonl: sample for RAG evaluation and generation (one retrieval_text per line);
   - coverage_summary.json: coverage status, slots without candidates, underfilled slots, and slot → files mapping;
   - coverage_report.md: human-readable report.

Run (using the configured default paths):
    python synthway/build_stratified_dataset.py \
      --input-dir "synthway/json" \
      --output-dir "synthway/output_analysis/stratified_dataset" \
      --per-slot 3 \
      --random-seed 42

Notes:
- Tags/slots absent from the data are listed in "not_present_in_corpus_slots".
- Top composition/method sets are determined dynamically from K and the unique-value count (see flags).
- Edit slot traversal order in ORDERED_SLOTS.
"""

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
import argparse
import json
import os
import re
import random
from glob import glob
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

# --------- Default constants (overridable by flags) ---------

DEFAULT_INPUT_DIR = f"{_REPO_ROOT}/synthway/json"
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "outputs" / "benchmark")
DEFAULT_TOP_COMPOSITIONS_K = 10  # Minimum number of top compositions
DEFAULT_TOP_METHODS_K = 8        # Minimum number of frequent methods

# Target tags, if present in the data
TAGS_NEEDED = {
    # 1) Composition frequency
    "composition_top", "composition_rare",
    # 2) Formula type
    "formula_simple", "formula_complex",
    # 3) Synthesis method
    "method_frequent", "method_rare",
    # 4) Recipe complexity
    "recipe_minimal", "recipe_normal", "recipe_hardcore", "recipe_seed_mediated",
}

# Checklist slot traversal order (one case per slot in the default mode)
ORDERED_SLOTS = [
    # 1) Composition frequency
    "composition_top", "composition_rare",
    # 2) Formula type
    "formula_simple", "formula_complex",
    # 3) Synthesis method
    "method_frequent", "method_rare",
    # 4) Recipe complexity
    "recipe_minimal", "recipe_normal", "recipe_hardcore", "recipe_seed_mediated",
]

# Tagging reference sets
WATER_ALIASES = {"water", "deionized water", "distilled water", "h2o", "aqueous"}
UNKNOWN_TOKENS = {"unknown", "none", "null", ""}

# --------- Utilities ---------

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if cur is None:
            return default
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
    return default if cur is None else cur

def normalize_str(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

def composition_elements(comp: str) -> List[str]:
    """Extract elements from a formula using a simple parser."""
    if not comp:
        return []
    tokens = re.findall(r"[A-Z][a-z]?", comp)
    return tokens

def is_core_shell_or_composite(comp: str) -> bool:
    s = comp.replace(" ", "")
    return ("@" in s) or ("-" in s and not s.startswith("-")) or ("+" in s)

def is_complex_formula(comp: str) -> bool:
    if not comp:
        return False
    if is_core_shell_or_composite(comp):
        return True
    elems = composition_elements(comp)
    if len(set(elems)) >= 3:
        return True
    if re.search(r"\d+\.\d+", comp):  # Decimal stoichiometries (doping)
        return True
    return False

def is_simple_formula(comp: str) -> bool:
    if not comp:
        return False
    if is_core_shell_or_composite(comp):
        return False
    elems = composition_elements(comp)
    return len(set(elems)) <= 2  # Metals and binary compounds

# --------- Main logic ---------

def load_records(input_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob(os.path.join(input_dir, "*.json")))
    records = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                rec = json.load(f)
            rec["_file_path"] = fp
            records.append(rec)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    return records

def compute_frequencies(records: List[Dict[str, Any]]) -> Tuple[Counter, Counter]:
    comp_counter = Counter()
    method_counter = Counter()
    for r in records:
        comp = r.get("composition") or ""
        if comp:
            comp_counter[comp] += 1
        method = normalize_str(r.get("method"))
        if method and method not in UNKNOWN_TOKENS:
            method_counter[method] += 1
    return comp_counter, method_counter

def determine_top_sets(counter: Counter, top_k_min: int) -> Set[str]:
    if not counter:
        return set()
    items = counter.most_common()
    unique = len(items)
    top_k = max(top_k_min, max(3, unique // 10))  # At least 3 or floor(10% of unique values)
    return {name for name, _ in items[:top_k]}

def tag_record(rec: Dict[str, Any],
               comp_counter: Counter,
               top_comps: Set[str],
               method_counter: Counter,
               top_methods: Set[str]) -> Set[str]:
    tags = set()

    # 1) Composition frequency
    comp = rec.get("composition") or ""
    if comp:
        cnt = comp_counter.get(comp, 0)
        if cnt == 1:
            tags.add("composition_rare")
        if comp in top_comps:
            tags.add("composition_top")

    # 2) Formula type
    if is_complex_formula(comp):
        tags.add("formula_complex")
    if is_simple_formula(comp):
        tags.add("formula_simple")

    # 3) Synthesis method
    method = normalize_str(rec.get("method"))
    if method and method not in UNKNOWN_TOKENS:
        mcnt = method_counter.get(method, 0)
        if mcnt == 1:
            tags.add("method_rare")
        if method in top_methods:
            tags.add("method_frequent")

    # 4) Recipe complexity
    precs = rec.get("precursors") or []
    adds = rec.get("additives") or []
    seeds = rec.get("seeds") or []
    n_prec = len(precs) if isinstance(precs, list) else 0
    n_add = len(adds) if isinstance(adds, list) else 0
    n_seed = len(seeds) if isinstance(seeds, list) else 0

    if n_prec == 1 and n_add == 0:
        tags.add("recipe_minimal")
    if (1 <= n_prec <= 2) and (n_add >= 1):
        tags.add("recipe_normal")
    if (n_prec >= 4) or (n_add >= 3):
        tags.add("recipe_hardcore")
    if n_seed >= 1:
        tags.add("recipe_seed_mediated")


    return tags

# --------- Selection helpers ---------

def _pick_max_with_random_tiebreak(pool, score_fn):
    """
    Return the highest-scoring item; randomly choose among tied items.
    Randomness is controlled by --random-seed.
    """
    best_items = []
    best_score = None
    for it in pool:
        sc = score_fn(it)
        if (best_score is None) or (sc > best_score):
            best_score = sc
            best_items = [it]
        elif sc == best_score:
            best_items.append(it)
    return random.choice(best_items) if len(best_items) > 1 else best_items[0]

def select_k_per_slot(items: List[Dict[str, Any]],
                      slots: List[str],
                      k_per_slot: int) -> Tuple[List[Dict[str, Any]], Set[str], List[str], Dict[str, List[str]]]:
    """
    N-per-slot mode with the following priorities:
      1) Singleton-first: assign slots with exactly one corpus candidate (1 of N if capacity remains).
      2) Round-robin: cycle through slots until k_per_slot, diversifying composition/method values.
      3) Coverage: gain in new tags → total tag count; break ties randomly.

    Returns:
      selected — list of selected cases,
      covered  — set of all covered tags,
      missing  — slots absent from the corpus or underfilled,
      slot_map — mapping from slot to selected file paths.
    """
    selected: List[Dict[str, Any]] = []
    covered: Set[str] = set()
    used_files: Set[str] = set()
    used_comps: Set[str] = set()
    used_methods: Set[str] = set()
    slot_map: Dict[str, List[str]] = {s: [] for s in slots}

    # Tags present anywhere in the corpus
    available_tags = set().union(*[it["tags"] for it in items]) if items else set()

    # Candidate pool per slot
    slot_to_all_cands: Dict[str, List[Dict[str, Any]]] = {
        slot: [it for it in items if slot in it["tags"]]
        for slot in slots
    }

    # Slots entirely absent from the corpus
    not_present = [slot for slot in slots if slot not in available_tags]

    # Pass 1: singleton-first (one each)
    for slot in slots:
        if slot in not_present:
            continue
        cands_all = slot_to_all_cands.get(slot, [])
        cands_now = [it for it in cands_all if it["file_path"] not in used_files]
        if len(cands_all) == 1 and len(cands_now) == 1 and len(slot_map[slot]) < k_per_slot:
            best = cands_now[0]
            selected.append(best)
            used_files.add(best["file_path"])
            covered |= best["tags"]
            comp = (best.get("composition") or "").strip()
            if comp:
                used_comps.add(comp)
            meth = normalize_str(best.get("method"))
            if meth:
                used_methods.add(meth)
            slot_map[slot].append(best["file_path"])

    # Pass 2: round-robin until k_per_slot
    def mnorm(it):
        return normalize_str(it.get("method"))

    progressed = True
    while progressed:
        progressed = False
        for slot in slots:
            if slot in not_present:
                continue
            if len(slot_map[slot]) >= k_per_slot:
                continue

            cands = [it for it in slot_to_all_cands.get(slot, []) if it["file_path"] not in used_files]
            if not cands:
                continue

            # Diversification: new composition first, then new method
            new_comp_pool = [it for it in cands if (it.get("composition") or "").strip() not in used_comps]
            pool = new_comp_pool if new_comp_pool else cands

            new_meth_pool = [it for it in pool if mnorm(it) and mnorm(it) not in used_methods]
            pool = new_meth_pool if new_meth_pool else pool

            # Coverage: gain in new tags → total tag count; break ties randomly
            def score(it):
                gain = len(it["tags"] - covered)
                return (gain, len(it["tags"]))

            best = _pick_max_with_random_tiebreak(pool, score)

            selected.append(best)
            used_files.add(best["file_path"])
            covered |= best["tags"]
            comp = (best.get("composition") or "").strip()
            if comp:
                used_comps.add(comp)
            meth = mnorm(best)
            if meth:
                used_methods.add(meth)
            slot_map[slot].append(best["file_path"])
            progressed = True

    # Underfilled: present in the corpus but selected < k_per_slot
    underfilled = [slot for slot in slots if (slot not in not_present) and (len(slot_map[slot]) < k_per_slot)]

    # missing = absent + underfilled slots
    missing = list(dict.fromkeys(not_present + underfilled))

    return selected, covered, missing, slot_map

def main():
    parser = argparse.ArgumentParser(description="Build stratified dataset per-slot (N) from JSON corpus.")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-compositions-k", type=int, default=DEFAULT_TOP_COMPOSITIONS_K)
    parser.add_argument("--top-methods-k", type=int, default=DEFAULT_TOP_METHODS_K)
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Seed for random tie-breaking during candidate selection (does not affect singleton-first).")
    parser.add_argument("--per-slot", type=int, default=1,
                        help="Cases to select per slot (default: 1)")
    args = parser.parse_args()

    # Seed random tie-breaking for reproducibility
    if args.random_seed is not None:
        random.seed(args.random_seed)
        try:
            import numpy as np  # noqa
            np.random.seed(args.random_seed)
        except Exception:
            pass

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_records(args.input_dir)
    if not records:
        raise SystemExit(f"No JSON files found in {args.input_dir}")

    # Frequencies and head sets for top/rare tags
    comp_counter, method_counter = compute_frequencies(records)
    top_comps = determine_top_sets(comp_counter, args.top_compositions_k)
    top_methods = determine_top_sets(method_counter, args.top_methods_k)

    # Tag all records
    tagged_items = []
    for r in records:
        tags = tag_record(r, comp_counter, top_comps, method_counter, top_methods)
        info = {
            "file_path": r.get("_file_path"),
            "synthesis_id": r.get("synthesis_id"),
            "composition": r.get("composition"),
            "method": r.get("method"),
            "n_precursors": len(r.get("precursors") or []),
            "n_additives": len(r.get("additives") or []),
            "n_seeds": len(r.get("seeds") or []),
            "tags": tags,
        }
        tagged_items.append(info)

    # Select cases per slot
    selected, covered, missing_slots, slot_map = select_k_per_slot(tagged_items, ORDERED_SLOTS, args.per_slot)
    # Save the sample
    out_jsonl = os.path.join(args.output_dir, "stratified_dataset.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it in selected:
            it_dump = dict(it)
            it_dump["tags"] = sorted(list(it_dump["tags"]))  # JSON-friendly
            f.write(json.dumps(it_dump, ensure_ascii=False) + "\n")

    # Additional dataset for RAG evaluation/generation (one retrieval_text per line)
    path_to_record = {
        r["_file_path"]: r
        for r in records
        if isinstance(r, dict) and r.get("_file_path")
    }

    out_final_jsonl = os.path.join(args.output_dir, "final_stratified_dataset.jsonl")
    with open(out_final_jsonl, "w", encoding="utf-8") as f:
        for it in selected:
            full_rec = path_to_record.get(it["file_path"], {})

            # Write an extended RAG evaluation record: id, composition, method, and full retrieval_text
            row = {
                "id": full_rec.get("synthesis_id") or it.get("synthesis_id"),
                "composition": full_rec.get("composition") or it.get("composition"),
                "method": full_rec.get("method") or it.get("method"),
                "retrieval_text": full_rec.get("retrieval_text"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Coverage report
    available = set().union(*[it["tags"] for it in tagged_items]) if tagged_items else set()
    ordered_slots_set = set(ORDERED_SLOTS)
    goals = ordered_slots_set & available
    not_present = ordered_slots_set - available
    # Recompute statuses for multiple cases per slot
    underfilled = [s for s in ORDERED_SLOTS if (s in goals) and (len(slot_map.get(s, [])) < args.per_slot)]
    not_assigned = sorted(underfilled)

    coverage = {
        "mode": "multi-per-slot",
        "per_slot_target": args.per_slot,
        "random_seed": args.random_seed,
        "total_records": len(records),
        "unique_compositions": len(comp_counter),
        "unique_methods": len(method_counter),
        "selected_count": len(selected),
        "covered_slots_count": len(covered & ordered_slots_set),
        "present_slots_count": len(goals),
        "slots_fully_assigned_count": sum(1 for s in goals if len(slot_map.get(s, [])) >= args.per_slot),
        "not_present_in_corpus_slots": sorted(list(not_present)),
        "not_assigned_slots": not_assigned,
        "underfilled_slots_detail": {s: {"assigned": len(slot_map.get(s, [])), "target": args.per_slot} for s in
                                     underfilled},
        "top_compositions_cutoff": sorted(list(top_comps)),
        "top_methods_cutoff": sorted(list(top_methods)),
    }

    # Show the top 5 candidates for each tag
    by_tag = defaultdict(list)
    for it in tagged_items:
        for t in it["tags"]:
            by_tag[t].append(it["file_path"])
    top_candidates = {t: sorted(list(dict.fromkeys(by_tag[t])))[:5] for t in by_tag}

    with open(os.path.join(args.output_dir, "coverage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "coverage": coverage,
                "slot_assignments": slot_map,
                "top_candidates_by_tag": top_candidates
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    # Markdown report
    md_lines = []
    md_lines.append(f"# Stratified Selection Report (per-slot={args.per_slot})\n")

    md_lines.append(f"- Input dir: `{args.input_dir}`")
    md_lines.append(f"- Output: `{out_jsonl}`")
    md_lines.append(f"- Total records: **{len(records)}**")
    md_lines.append(f"- Unique compositions: **{len(comp_counter)}**")
    md_lines.append(f"- Unique methods: **{len(method_counter)}**")
    md_lines.append(f"- Selected count: **{len(selected)}**")
    md_lines.append(f"- Covered slots: **{len(covered & ordered_slots_set)}/{len(goals)}**")
    if not_present:
        md_lines.append(f"- Slots not present in corpus: {', '.join(sorted(not_present))}")
    if not_assigned:
        md_lines.append(f"- Slots present but not assigned (unique-candidate conflict): {', '.join(sorted(not_assigned))}")
    md_lines.append("\n## Slot → Selected files\n")
    for slot in ORDERED_SLOTS:
        fps = slot_map.get(slot, [])
        if fps:
            md_lines.append(f"- **{slot}** → " + ", ".join('`' + p + '`' for p in fps))
        else:
            md_lines.append(f"- **{slot}** → _no candidate_")
    md_lines.append("\n## Selected items\n")
    for i, it in enumerate(selected, 1):
        md_lines.append(
            f"**{i:02d}.** `{it['file_path']}` — {it['composition']} | method: {it['method']} | "
            f"prec:{it['n_precursors']} add:{it['n_additives']} seed:{it['n_seeds']} | "
            f"tags: {', '.join(sorted(it['tags']))}"
        )
    with open(os.path.join(args.output_dir, "coverage_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[OK] Saved: {out_jsonl}")
    print(f"[OK] RAG dataset: {out_final_jsonl}")
    print(f"[OK] Report: {os.path.join(args.output_dir, 'coverage_report.md')}")
    print(f"[OK] Coverage JSON: {os.path.join(args.output_dir, 'coverage_summary.json')}")

if __name__ == "__main__":
    main()
