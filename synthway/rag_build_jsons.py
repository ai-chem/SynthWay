# rag_build_jsons.py
# Build a RAG JSON corpus from local .graphml files only (one graph per synthesis), adding retrieval_text.

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
import argparse
import sys
sys.path.insert(0, str(_REPO_ROOT / "graph_reasoning"))
from local_graph_files import graph_paths
import json
import os
import re
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
from tqdm import tqdm



# Directories
PROJECT_ROOT = f"{_REPO_ROOT}"
DATA_DIR = os.environ.get("SYNTHWAY_GRAPH_OUTPUT", os.path.join(PROJECT_ROOT, "outputs", "graph"))
LOCAL_GRAPHS_DIR  = os.path.join(DATA_DIR, "local_graphs_renamed")
OUTPUT_DIR        = os.path.join(PROJECT_ROOT, "synthway")


# Utilities
def safe_filename(name: Optional[str], fallback: str = "doc") -> str:
    s = (name or fallback).strip()
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    return (s[:120] or fallback)


def coalesce_str(x) -> Optional[str]:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return None
    # Treat placeholders as missing
    if s.lower() == "unknown" or s == "-999":
        return None
    return s


def to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass
        if isinstance(x, str):
            s = x.strip().replace(",", ".")
            if not s:
                return None
            if s.lower() == "unknown" or s == "-999":
                return None
            val = float(s)
            return None if val == -999 else val
        # non-string numeric
        val = float(x)
        return None if val == -999 else val
    except Exception:
        return None


def parse_elements(formula: Optional[str]) -> Optional[List[str]]:
    """Extract elements from formulas using a simple parser."""
    if not formula:
        return None
    # Discard hydrates (*6H2O)
    f = formula.replace("·", "+").replace("*", "+")
    tokens = re.findall(r"[A-Z][a-z]?", f)
    if not tokens:
        return None
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out




def prune_nones(obj: Any) -> Any:
    """Recursively remove missing values."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            pv = prune_nones(v)
            if pv is None:
                continue
            # Also discard empty containers
            if isinstance(pv, (list, dict)) and len(pv) == 0:
                continue
            out[k] = pv
        return out
    elif isinstance(obj, list):
        out_list = [prune_nones(v) for v in obj]
        out_list = [v for v in out_list if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)]
        return out_list
    else:
        return obj


def build_retrieval_text(doc: Dict[str, Any]) -> str:
    """Flat document summary for embedding/search."""
    parts: List[str] = []

    def add(label: str, val: Any):
        if val is None:
            return
        if isinstance(val, (list, tuple)):
            if not val:
                return
            parts.append(f"{label}: " + "; ".join(map(str, val)))
        elif isinstance(val, dict):
            for k, v in val.items():
                if v is None:
                    continue
                parts.append(f"{label}.{k}: {v}")
        else:
            s = str(val).strip()
            if s:
                parts.append(f"{label}: {s}")

    add("composition", doc.get("composition"))
    add("method", doc.get("method"))
    solvent = doc.get("solvent")
    if isinstance(solvent, dict):
        add("solvent", solvent.get("name"))
        add("solvent.volume", solvent.get("volume"))
    else:
        add("solvent", solvent)

    cond = doc.get("conditions") or {}
    add("temperature", cond.get("temperature"))
    add("pH", cond.get("ph"))

    prod = doc.get("product") or {}
    add("crystalsystem", prod.get("crystalsystem"))
    add("shape", prod.get("shape"))
    size = prod.get("size") or {}
    if isinstance(size, dict):
        for k, v in size.items():
            add(f"size.{k}", v)

    def _fmt_reagent(entry: Dict[str, Any]) -> Optional[str]:
        if not isinstance(entry, dict):
            return None
        nm = entry.get("name")
        if not nm:
            return None
        details = []
        m = entry.get("mass")
        c = entry.get("conc")
        if m is not None:
            details.append(f"mass {m}")
        if c is not None:
            details.append(f"conc {c}")
        return f"{nm} ({'; '.join(details)})" if details else nm

    for p in (doc.get("precursors") or []):
        s = _fmt_reagent(p)
        if s:
            add("precursor", s)
    for a in (doc.get("additives") or []):
        s = _fmt_reagent(a)
        if s:
            add("additive", s)
    for s in (doc.get("seeds") or []):
        s2 = _fmt_reagent(s)
        if s2:
            add("seed", s2)

    add("pdfno_list", doc.get("pdfno_list"))
    return " | ".join(parts)




# GraphML parsing
def _label_value_from_node(n: Any, data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    label = data.get("label")
    value = data.get("name")
    nid = str(n)
    if label is None and ":" in nid:
        label = nid.split(":", 1)[0]
    if value is None and ":" in nid:
        value = nid.split(":", 1)[1].strip()
    return label, value


def graph_to_doc(G: nx.DiGraph, synthesis_id_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Extract a document from one local graph (one graph per synthesis)."""
    # 1) composition
    comp_vals: List[str] = []
    for n, data in G.nodes(data=True):
        lab, val = _label_value_from_node(n, data)
        if lab == "composition" and val:
            comp_vals.append(val)
    composition = comp_vals[0] if comp_vals else None
    if not composition:
        return None

    # synthesis_id: from edge attributes, a hint, or the filename
    synth_candidates = set()
    for u, v, edata in G.edges(data=True):
        s = coalesce_str(edata.get("synthesis"))
        if s:
            synth_candidates.add(s)
    synthesis_id = (
        next(iter(synth_candidates))
        if synth_candidates
        else coalesce_str(synthesis_id_hint)
    )

    # 2) Simple fields
    method = None
    temperature = None
    ph = None
    shape = None
    crystalsystem = None
    size_length = None
    size_width = None
    size_depth = None
    pdfno_list: List[str] = []

    precursors: List[Dict[str, Any]] = []
    additives: List[Dict[str, Any]] = []
    seeds: List[Dict[str, Any]] = []
    solvent_name = None
    solvent_vol = None

    out_map: Dict[Any, List[Any]] = {}
    for u, v in G.edges():
        out_map.setdefault(u, []).append(v)

    # Traverse all nodes
    for n, data in G.nodes(data=True):
        label, value = _label_value_from_node(n, data)
        if not label:
            continue
        if label == "method" and method is None:
            method = coalesce_str(value)
        elif label == "temperature" and temperature is None:
            temperature = to_float(value)
        elif label == "ph" and ph is None:
            ph = to_float(value)
        elif label == "shape" and shape is None:
            shape = coalesce_str(value)
        elif label in ("crystalsystem", "crystal_system") and crystalsystem is None:
            crystalsystem = coalesce_str(value)
        elif label == "size_length" and size_length is None:
            size_length = to_float(value)
        elif label == "size_width" and size_width is None:
            size_width = to_float(value)
        elif label == "size_depth" and size_depth is None:
            size_depth = to_float(value)
        elif label == "pdfno" and value:
            vv = coalesce_str(value)
            if vv:
                pdfno_list.append(vv)
        elif label in ("precursor", "additive", "seeds", "solvent"):
            base_name = coalesce_str(value)
            if not base_name:
                continue
            # Collect parameters from outgoing edges
            pmass = None
            pconc = None
            pvol = None
            for child in out_map.get(n, []) or []:
                c_lab, c_val = _label_value_from_node(child, G.nodes[child])
                if c_lab in ("precursor_mass", "additive_mass", "seed_mass"):
                    pmass = pmass or to_float(c_val)
                elif c_lab in ("precursor_conc", "additive_conc", "seed_conc"):
                    pconc = pconc or to_float(c_val)
                elif c_lab == "solvent_v":
                    pvol = pvol or to_float(c_val)
            if label == "precursor":
                precursors.append({"name": base_name, "mass": pmass, "conc": pconc})
            elif label == "additive":
                additives.append({"name": base_name, "mass": pmass, "conc": pconc})
            elif label == "seeds":
                seeds.append({"name": base_name, "mass": pmass, "conc": pconc})
            elif label == "solvent":
                solvent_name = solvent_name or base_name
                solvent_vol = solvent_vol or pvol

    # Assemble product/conditions/solvent
    conditions = prune_nones({
        "temperature": temperature,
        "ph": ph,
    })

    size = prune_nones({
        "length": size_length,
        "width": size_width,
        "depth": size_depth,
    }) or None

    product = prune_nones({
        "crystalsystem": crystalsystem,
        "shape": shape,
        "size": size,
    })

    solvent = None
    if solvent_name or solvent_vol is not None:
        solvent = {"name": solvent_name} if solvent_name else {}
        if solvent_vol is not None:
            solvent["volume"] = solvent_vol

    # method stays as-is

    # Document
    doc: Dict[str, Any] = {
        "synthesis_id": synthesis_id,
        "composition": composition,
        "elements": parse_elements(composition),
        "method": method,
        "precursors": precursors or None,
        "additives": additives or None,
        "seeds": seeds or None,
        "solvent": solvent,
        "conditions": conditions if conditions else None,
        "product": product if product else None,
        "pdfno_list": pdfno_list or None,
        "source": prune_nones({
            "synthesis_id_original": synthesis_id,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }),
    }

    doc = prune_nones(doc)
    doc["retrieval_text"] = build_retrieval_text(doc)
    return doc


def build_docs_from_graphs(graphs_dir: str, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    json_dir = os.path.join(out_dir, "json")
    os.makedirs(json_dir, exist_ok=True)

    files = graph_paths(graphs_dir)
    docs: List[Dict[str, Any]] = []
    skipped = 0

    for p in tqdm(files, desc="Building docs from graphs"):
        base = os.path.basename(p)
        synth_hint = None
        m = re.match(r"synthesis_(.+)\.graphml$", base)
        if m:
            synth_hint = m.group(1)
        try:
            G = nx.read_graphml(p)
        except Exception as e:
            print(f"Failed to read {base}: {e}")
            skipped += 1
            continue
        doc = graph_to_doc(G, synthesis_id_hint=synth_hint)
        if not doc:
            print(f"Skipped {base}: no composition (or unparsable composition node)")
            skipped += 1
            continue

        fname = os.path.join(json_dir, f"{safe_filename(doc.get('synthesis_id') or 'NA')}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        docs.append(doc)

    jsonl_path = os.path.join(out_dir, "routes.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return {
        "count": len(docs),
        "skipped": skipped,
        "out_dir": out_dir,
        "json_dir": json_dir,
        "jsonl": jsonl_path,
    }



def main():
    ap = argparse.ArgumentParser(description="Build RAG-ready JSON corpus from local GraphML graphs (1 file = 1 synthesis).")
    ap.add_argument("--graphs", dest="graphs", default=LOCAL_GRAPHS_DIR, help="Directory with *.graphml local graphs (1 file = 1 synthesis)")
    ap.add_argument("--out", dest="out", default=OUTPUT_DIR, help="Output dir for corpus")
    args = ap.parse_args()

    report = build_docs_from_graphs(args.graphs, args.out)


    print(f"Documents: {report['count']}")
    print(f"JSON dir: {report['json_dir']}")
    print(f"JSONL:    {report['jsonl']}")
    print(f"Skipped:  {report['skipped']}")


if __name__ == "__main__":
    main()
