"""
Profile merged_all.xlsx:
- summary.csv: per-column summary (NaN, '', unknown, -999, distinct values, etc.)
- distribution.csv: complete per-column value distributions (excluding NaN/''/unknown/-999)
- index.html: interactive report (DataTables)
- NEW: deduplicate by synthesis_id; count each value at most once per synthesis; add *_by_sid and count_by_sid/pct_by_sid.
- NEW: local graphs; add two summary columns: local_param_count (total nodes of this type across local graphs) and local_param_unique (unique node values of this type).
- NEW: top_value_local; most frequent value among nodes of the corresponding type in local graphs.
"""

import os
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "graph_reasoning"))
from local_graph_files import graph_paths
import ast
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


import networkx as nx

# --- Path defaults (portable) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
GRAPH_DIR = os.environ.get('SYNTHWAY_GRAPH_OUTPUT', os.path.join(REPO_DIR, 'outputs', 'graph'))

DEFAULT_INPUT_XLSX = os.path.join(REPO_DIR, "data", "merged_all.xlsx")
DEFAULT_GRAPH_DIR = GRAPH_DIR

def _default_input_xlsx() -> str:
    return DEFAULT_INPUT_XLSX

def _default_graph_dir() -> str:
    return DEFAULT_GRAPH_DIR

def _default_local_graphs_dir() -> str:
    return os.path.join(_default_graph_dir(), "local_graphs_renamed")

def _default_global_graph_path() -> str:
    return os.path.join(_default_graph_dir(), "graph.graphml")


UNKNOWN_TOKENS_DEFAULT = ["unknown"]  # Case-insensitive matching; ignore surrounding whitespace

# Sanitize synthesis_id to the same "file key" used in local graph filenames
def _sanitize_sid_for_filekey(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', str(s))

def _safe_name(s: str) -> str:
    import re
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s[:80]

def _is_valid_value(val, unknown_tokens):
    if pd.isna(val):
        return False
    s = str(val).strip()
    if s == "":
        return False
    if s.lower() in [u.lower() for u in unknown_tokens]:
        return False
    if s == "-999" or (isinstance(val, (int, float)) and val == -999):
        return False
    return True


# Map table column names to node labels in local graphs
def _col_to_graph_label(col: str) -> str | None:
    """
    Map a table column name to a node label in local graphs.
    Names usually match; the exception is pdfno_list -> pdfno.
    Return None for unsupported columns.
    """
    if col == "pdfno_list":
        return "pdfno"
    if col == "synthesis_id":
        return None
    # Leave unchanged by default
    return col


# Scan local graphs
def _scan_local_graphs(local_dir: str, allowed_filekeys: set[str] | None = None):
    """
    Scan *.graphml in local_dir and return:
      - total_by_label: dict[label] = total node occurrences with this label across all local graphs
      - unique_values_by_label: dict[label] = number of unique node names with this label
      - top_value_by_label: dict[label] = most frequent node name (value) for this label
      - values_by_label: dict[label] = set(unique node names) for each label
      - counts_by_label: dict[label] = dict[name] = occurrences of individual node names
    """
    total_by_label: dict[str, int] = {}
    values_by_label: dict[str, set] = {}
    counts_by_label: dict[str, dict] = {}

    if not local_dir or not os.path.isdir(local_dir):
        # Return empty structures for all five return values
        return total_by_label, {k: 0 for k in total_by_label}, {}, {}, {}

    files = graph_paths(local_dir)

    # Optionally filter by a set of filekeys (basename without prefix/suffix)
    if allowed_filekeys is not None:
        filtered = []
        for gm in files:
            base = os.path.basename(gm)
            if base.endswith(".graphml"):
                base = base[:-8]
            if base.startswith("synthesis_"):
                base = base[len("synthesis_"):]
            if base in allowed_filekeys:
                filtered.append(gm)
        files = filtered

    for gm in files:
        try:
            G = nx.read_graphml(gm)
        except Exception as e:
            print(f"[WARN] Skipping {gm}: {e}")
            continue
        for n, data in G.nodes(data=True):
            lbl = (data or {}).get("label")
            if not isinstance(lbl, str):
                continue
            name = str(n)
            total_by_label[lbl] = total_by_label.get(lbl, 0) + 1
            if lbl not in values_by_label:
                values_by_label[lbl] = set()
            values_by_label[lbl].add(name)
            if lbl not in counts_by_label:
                counts_by_label[lbl] = {}
            counts_by_label[lbl][name] = counts_by_label[lbl].get(name, 0) + 1

    unique_by_label = {lbl: len(vals) for lbl, vals in values_by_label.items()}

    # Select the most frequent node name for each label
    top_value_by_label: dict[str, str] = {}
    for lbl, cdict in counts_by_label.items():
        if not cdict:
            continue
        top_name, top_cnt = None, -1
        for name, cnt in cdict.items():
            if cnt > top_cnt or (cnt == top_cnt and (top_name is None or name < top_name)):
                top_name, top_cnt = name, cnt
        if top_name is not None:
            top_value_by_label[lbl] = top_name

    # Also return values_by_label and counts_by_label
    return total_by_label, unique_by_label, top_value_by_label, values_by_label, counts_by_label

# Scan the global graph (one .graphml file)
def _scan_global_graph(graph_path: str):
    """
    Read graph.graphml and return:
      - total_by_label: dict[label] = total nodes with this label
      - unique_values_by_label: dict[label] = number of unique node names with this label
      - top_value_by_label: dict[label] = most frequent node name (value) for this label
      - values_by_label: dict[label] = set(unique node names) for each label
    Print basic diagnostics: node/edge counts.
    """
    total_by_label: dict[str, int] = {}
    values_by_label: dict[str, set] = {}
    counts_by_label: dict[str, dict] = {}

    if not graph_path or not os.path.isfile(graph_path):
        print(f"[INFO] Global graph not found or path is empty: {graph_path}")
        return total_by_label, {k: 0 for k in total_by_label}, {}, {}

    try:
        G = nx.read_graphml(graph_path)
    except Exception as e:
        print(f"[WARN] Failed to read global graph '{graph_path}': {e}")
        return total_by_label, {k: 0 for k in total_by_label}, {}, {}

    try:
        print(f"[INFO] Global graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()} • {graph_path}")
    except Exception:
        pass

    for n, data in G.nodes(data=True):
        lbl = (data or {}).get("label")
        if not isinstance(lbl, str):
            continue
        name = str(n)
        total_by_label[lbl] = total_by_label.get(lbl, 0) + 1
        if lbl not in values_by_label:
            values_by_label[lbl] = set()
        values_by_label[lbl].add(name)
        if lbl not in counts_by_label:
            counts_by_label[lbl] = {}
        counts_by_label[lbl][name] = counts_by_label[lbl].get(name, 0) + 1

    unique_by_label = {lbl: len(vals) for lbl, vals in values_by_label.items()}

    top_value_by_label: dict[str, str] = {}
    for lbl, cdict in counts_by_label.items():
        if not cdict:
            continue
        top_name, top_cnt = None, -1
        for name, cnt in cdict.items():
            if cnt > top_cnt or (cnt == top_cnt and (top_name is None or name < top_name)):
                top_name, top_cnt = name, cnt
        if top_name is not None:
            top_value_by_label[lbl] = top_name

    return total_by_label, unique_by_label, top_value_by_label, values_by_label


# Count table parameters using the global-graph rules
def _scan_table_as_global_nodes(df: pd.DataFrame, unknown_tokens):
    """
    Iterate over merged_all.xlsx rows and count nodes/values using the same rules
    as the global graph construction:
      - composition (as a string)
      - normalized_* (as strings) and their parameters as "key:value"
      - solvent_v_minmax
      - temperature_minmax / ph_minmax / method
      - shape / crystalsystem
      - size_*_minmax
      - pdfno_list (each list element) -> label 'pdfno'
    Returns:
      - total_by_label
      - unique_values_by_label
      - top_value_by_label
    """
    import ast
    total_by_label: dict[str, int] = {}
    values_by_label: dict[str, set] = {}
    counts_by_label: dict[str, dict] = {}

    def _add(name: str, label: str):
        if not isinstance(label, str) or not isinstance(name, str):
            return
        total_by_label[label] = total_by_label.get(label, 0) + 1
        if label not in values_by_label:
            values_by_label[label] = set()
        values_by_label[label].add(name)
        if label not in counts_by_label:
            counts_by_label[label] = {}
        counts_by_label[label][name] = counts_by_label[label].get(name, 0) + 1

    # Same rules as in Global_graph_generation_without_synthesisID.py
    FIELD_PARAMS = {
        "precursor": ["precursor_mass", "precursor_conc"],
        "additive": ["additive_mass", "additive_conc"],
        "seeds": ["seed_mass", "seed_conc"],
        "solvent": ["solvent_v"],
    }

    for _, row in df.iterrows():
        # composition must be a string other than 'unknown'
        comp = row.get("composition")
        if isinstance(comp, str) and comp.strip() and comp.strip().lower() != "unknown":
            _add(comp.strip(), "composition")

        # Normalized reagents and their parameters
        for field, param_keys in FIELD_PARAMS.items():
            val = row.get(field)
            if isinstance(val, str) and val.strip() and val.strip().lower() != "unknown":
                v = val.strip()
                _add(v, field)
                for param in param_keys:
                    pval = row.get(param)
                    if pd.notna(pval) and str(pval).lower() != "unknown" and pval != -999:
                        node = f"{param}:{pval}"
                        _add(node, param)

        # Conditions: temperature/ph/method
        for fld in ["temperature", "ph", "method"]:
            val = row.get(fld)
            if pd.notna(val) and str(val).strip() and str(val).lower() != "unknown" and val != -999:
                if "minmax" in fld:
                    node = f"{fld}:{val}"
                else:
                    node = str(val).strip()
                _add(node, fld)

        # Product properties
        for fld in ["shape", "crystalsystem"]:
            val = row.get(fld)
            if isinstance(val, str) and val.strip() and val.strip().lower() != "unknown":
                _add(val.strip(), fld)

        # Dimensions
        for size_fld in ["size_length", "size_width", "size_depth"]:
            val = row.get(size_fld)
            if pd.notna(val) and val != -999:
                node = f"{size_fld}:{val}"
                _add(node, size_fld)

        # pdfno (list or string)
        pdf_values = row.get("pdfno_list")
        if isinstance(pdf_values, str) and pdf_values.strip() and pdf_values.lower() != "unknown":
            try:
                if "[" in pdf_values:
                    try:
                        items = ast.literal_eval(pdf_values)
                        if not isinstance(items, (list, tuple)):
                            items = [pdf_values]
                    except Exception:
                        items = [pdf_values]
                else:
                    items = [pdf_values]
                for pdf in items:
                    s = str(pdf).strip()
                    if s:
                        _add(s, "pdfno")
            except Exception:
                pass

    unique_by_label = {lbl: len(vals) for lbl, vals in values_by_label.items()}
    top_value_by_label: dict[str, str] = {}
    for lbl, cdict in counts_by_label.items():
        if not cdict:
            continue
        top_name, top_cnt = None, -1
        for name, cnt in cdict.items():
            if cnt > top_cnt or (cnt == top_cnt and (top_name is None or name < top_name)):
                top_name, top_cnt = name, cnt
        if top_name is not None:
            top_value_by_label[lbl] = top_name
    return total_by_label, unique_by_label, top_value_by_label

def analyze_column(ser: pd.Series, unknown_tokens, sid: pd.Series | None = None, sid_key: pd.Series | None = None, sid_in_scope_mask: pd.Series | None = None, sid_base_keys: set[str] | None = None):
    total = len(ser)
    is_nan = ser.isna()

    # String representation for whitespace/case normalization
    ser_str = ser.astype(str)
    ser_str = ser_str.where(~is_nan, other="")  # Prevent the literal string 'nan' from appearing

    s_trim = ser_str.str.strip()
    s_low  = s_trim.str.lower()

    # masks
    is_empty   = (~is_nan) & (s_trim == "")
    is_unknown = (~is_nan) & (s_low.isin([u.lower() for u in unknown_tokens]))
    is_minus999 = (~is_nan) & ( (ser == -999) | (s_trim == "-999") )

    valid_mask = (~is_nan) & (~is_empty) & (~is_unknown) & (~is_minus999)

    # counts
    nan_count      = int(is_nan.sum())
    empty_count    = int(is_empty.sum())
    unknown_count  = int(is_unknown.sum())
    minus999_count = int(is_minus999.sum())
    valid_count    = int(valid_mask.sum())

    # Distinct valid values after trimming whitespace
    distinct_valid = int(s_trim[valid_mask].nunique(dropna=True))

    # Most frequent valid value
    vc = s_trim[valid_mask].value_counts(dropna=False)
    if not vc.empty:
        top_value = str(vc.index[0])
        top_count = int(vc.iloc[0])
        top_pct   = float(100.0 * top_count / total) if total else 0.0
    else:
        top_value, top_count, top_pct = "", 0, 0.0

    summary = {
        "total_rows": total,
        "non_null": int((~is_nan).sum()),
        "nan_count": nan_count,
        "empty_string_count": empty_count,
        "unknown_count": unknown_count,
        "minus999_count": minus999_count,
        "valid_count": valid_count,
        "distinct_valid": distinct_valid,
        "top_value": top_value,
        "top_value_count": top_count,
        "top_value_pct": round(top_pct, 4),
    }

    # Detailed distribution of valid values
    # Explicitly name the value column 'value' to ensure a stable merge
    dist = (
        vc.rename("count")
          .rename_axis("value")
          .reset_index()
          .assign(pct=lambda d: d["count"] / total * 100.0 if total else 0.0)
    )

    # --- Deduplicate by synthesis_id using uniq(sanitize(synthesis_id)): count at most once per filekey ---
    if sid is not None and len(sid) == len(ser) and sid_key is not None and len(sid_key) == len(ser):
        sid_series = pd.Series(sid_key).reset_index(drop=True)  # Use sanitized filekeys
        # Base scope: exactly the set of unique sanitized SIDs (≈4248)
        if sid_base_keys is not None:
            base_keys = set(sid_base_keys)
            base_mask = sid_series.isin(list(base_keys))
            sid_total = len(base_keys)
        else:
            # Fallback: all valid rows
            base_mask = pd.Series([True] * len(ser))
            sid_total = int(sid_series.nunique())
        # Additional row-validity mask (NaN/unknown/-999 already handled by valid_mask)
        if sid_in_scope_mask is not None and len(sid_in_scope_mask) == len(ser):
            in_scope = pd.Series(sid_in_scope_mask).reset_index(drop=True).astype(bool)
        else:
            in_scope = pd.Series([True] * len(ser))
        # Valid records within the base scope
        mask = valid_mask & in_scope & base_mask
        df_pairs = (
            pd.DataFrame({"sid": sid_series, "value": s_trim})
              .loc[mask, ["sid", "value"]]
              .drop_duplicates()
        )
        # Number of syntheses with AT LEAST ONE valid value for this parameter
        filled_sids = int(df_pairs["sid"].nunique()) if not df_pairs.empty else 0
        missing_sids = int(max(0, sid_total - filled_sids))
        coverage_pct_by_sid = round((filled_sids / sid_total * 100.0) if sid_total else 0.0, 4)
        vc_sid = df_pairs["value"].value_counts(dropna=False)
        # Summary over the base of 4248 keys
        if not vc_sid.empty:
            top_value_by_sid = str(vc_sid.index[0])
            top_count_by_sid = int(vc_sid.iloc[0])
        else:
            top_value_by_sid, top_count_by_sid = "", 0
        summary.update({
            # Unique (sid_filekey, value) pairs among valid values
            "valid_count_by_sid": int(len(df_pairs)),
            # Number of syntheses in uniq(sanitize(synthesis_id)) with at least one valid value
            "filled_sids_by_sid": filled_sids,
            "missing_sids_by_sid": missing_sids,
            "coverage_pct_by_sid": coverage_pct_by_sid,
            # Number of distinct values present in at least one synthesis
            "distinct_valid_by_sid": int(vc_sid.index.size),
            # Most frequent value by synthesis (number of syntheses containing the value)
            "top_value_by_sid": top_value_by_sid,
            "top_value_by_sid_count": top_count_by_sid,
            "top_value_by_sid_pct": round((top_count_by_sid / sid_total * 100.0) if sid_total else 0.0, 4),
        })
        # Distribution: add count_by_sid / pct_by_sid columns
        if not dist.empty:
            sid_counts_df = (
                vc_sid.rename("count_by_sid")
                      .rename_axis("value")
                      .reset_index()
            )
            dist = dist.merge(sid_counts_df, on="value", how="left")
            dist["count_by_sid"] = dist["count_by_sid"].fillna(0).astype(int)
            dist["pct_by_sid"] = (dist["count_by_sid"].astype(float) / sid_total * 100.0) if sid_total else 0.0

    return summary, dist


def build_report(input_xlsx, sheet=None, out_dir=None, unknown_tokens=None):
    global args  # Access args.local_graphs_dir inside build_report
    unknown_tokens = unknown_tokens or UNKNOWN_TOKENS_DEFAULT

    # Prepare the output directory
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(input_xlsx), f"dq_report_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Read Excel
    # Use the first sheet by default (sheet_name=0), or the specified name/index.
    sheet_to_use = 0 if sheet is None else sheet
    df = pd.read_excel(input_xlsx, sheet_name=sheet_to_use)  # Preserve native NaN handling; do not force dtype=str
    # If a dict of multiple sheets is returned, use the first sheet
    if isinstance(df, dict):
        first_key = next(iter(df))
        print(f"[WARN] Multiple sheets returned; using first: {first_key}")
        df = df[first_key]
    print(f"[INFO] Using input: {input_xlsx}")
    if 'args' in globals():
        print(f"[INFO] Using local_graphs_dir: {args.local_graphs_dir}")
        if hasattr(args, 'global_graph_path'):
            print(f"[INFO] Using global_graph_path: {args.global_graph_path}")

    # Use synthesis_id, if available, to deduplicate by synthesis
    sid_series = df["synthesis_id"] if "synthesis_id" in df.columns else None

    # ---- Source size summary: number of syntheses and local graphs ----
    if sid_series is not None:
        try:
            sid_unique = int(pd.Series(sid_series).nunique(dropna=True))
        except Exception:
            sid_unique = int(pd.Series(sid_series).dropna().nunique())
        print(f"[INFO] synthesis_id: unique syntheses in the table: {sid_unique}")
    else:
        print("[INFO] Table has no 'synthesis_id' column; skipping synthesis count.")

    local_dir = args.local_graphs_dir if 'args' in globals() else None
    if local_dir and os.path.isdir(local_dir):
        local_graphs_count = len(graph_paths(local_dir))
        print(f"[INFO] Local graphs: found {local_graphs_count} *.graphml files in: {local_dir}")
    else:
        print(f"[INFO] Local graph directory is unspecified or does not exist: {local_dir}")

    # Collect filekeys from local graph filenames (basename without prefix/suffix)
    local_filekeys: set[str] = set()
    if local_dir and os.path.isdir(local_dir):
        for p in graph_paths(local_dir):
            base = os.path.basename(p)
            if base.endswith('.graphml'):
                base = base[:-8]
            if base.startswith('synthesis_'):
                base = base[len('synthesis_'):]
            local_filekeys.add(base)
        print(f"[INFO] Filekeys found in local graphs (for reference): {len(local_filekeys)}")

    # Build keys from Excel synthesis_id values after sanitization; include all valid sanitized SIDs
    sid_key_series = None
    sid_in_scope_mask = None
    sanitized_filekeys = set()
    if sid_series is not None:
        sid_raw = pd.Series(sid_series)
        sid_str = sid_raw.astype(str)
        sid_trim = sid_str.str.strip()
        sid_low  = sid_trim.str.lower()
        bad_tokens = {"", "unknown", "-999", "nan", "none", "na", "null"}
        bad_mask = sid_raw.isna() | sid_low.isin(bad_tokens)
        sid_key_series = sid_trim.map(_sanitize_sid_for_filekey)
        sid_in_scope_mask = ~bad_mask  # Count all valid sanitized SIDs (base ≈ 4248)
        # Base-scope diagnostics
        raw_unique = int(sid_raw.nunique(dropna=True))
        sanitized_unique = int(sid_key_series[sid_in_scope_mask].nunique())
        print(f"[INFO] synthesis_id: raw_unique={raw_unique}; sanitized_unique(base)={sanitized_unique} • scope=uniq(sanitize(synthesis_id))")
        sanitized_filekeys = set(sid_key_series[sid_in_scope_mask].unique()) if (sid_key_series is not None and sid_in_scope_mask is not None) else set()

    # Load local graph statistics (nodes by label)
    total_by_label_local, unique_by_label_local, top_value_by_label_local, values_by_label_local, counts_by_label_local = _scan_local_graphs(
        args.local_graphs_dir if 'args' in globals() else None,
        allowed_filekeys=sanitized_filekeys
    )

    # Global graph statistics
    global_graph_path = args.global_graph_path if 'args' in globals() and hasattr(args, 'global_graph_path') else _default_global_graph_path()
    total_by_label_global, unique_by_label_global, top_value_by_label_global, values_by_label_global = _scan_global_graph(global_graph_path)

    # Table parameters as used to build the global graph
    table_total_by_label, table_unique_by_label, table_top_value_by_label = _scan_table_as_global_nodes(df, unknown_tokens)
    print(f"[INFO] Table (using global-graph rules): total nodes={sum(table_total_by_label.values())}, label types={len(table_total_by_label)}")

    # === Nodes present in local graphs but absent from the global graph ===
    try:
        missing_rows = []
        for lbl, local_names in (values_by_label_local or {}).items():
            global_names = (values_by_label_global or {}).get(lbl, set())
            only_local = sorted(local_names - global_names)
            if not only_local:
                continue
            for name in only_local:
                occ = 0
                if counts_by_label_local and lbl in counts_by_label_local:
                    occ = int(counts_by_label_local[lbl].get(name, 0))
                missing_rows.append({
                    "label": lbl,
                    "node_name": str(name),
                    "local_occurrences": occ,
                })
        if missing_rows:
            df_missing = pd.DataFrame(missing_rows)
            out_missing = os.path.join(out_dir, "local_nodes_NOT_in_global.csv")
            df_missing.to_csv(out_missing, index=False)
            print(f"[OK] Missing nodes (local → not in global): {out_missing} • rows={len(df_missing)}")
        else:
            print("[OK] Missing nodes: none (all local nodes are present in the global graph)")
    except Exception as e:
        print(f"[WARN] Failed to compute/write missing local nodes: {e}")

    all_summ_rows = []
    all_dists = []
    top_img_files = []

    for col in df.columns:
        summ, dist = analyze_column(
            df[col],
            unknown_tokens,
            sid=sid_series,
            sid_key=sid_key_series,
            sid_in_scope_mask=sid_in_scope_mask,
            sid_base_keys=sanitized_filekeys,
        )
        summ_row = {"column": col, **summ}
        # Synthesis base size: uniq(sanitize(synthesis_id))
        if sid_series is not None:
            summ_row["sid_base_sanitized_unique"] = int(sanitized_unique)
            summ_row["sid_scope_note"] = "scope=uniq(sanitize(synthesis_id))"
        else:
            summ_row["sid_base_sanitized_unique"] = 0
            summ_row["sid_scope_note"] = ""
        g_label = _col_to_graph_label(col)
        if g_label:
            # IMPORTANT: count local nodes by UNIQUE names for a valid comparison with the global graph
            # After renaming, local nodes should form a subset of the global node set.
            summ_row["local_param_count"] = int(unique_by_label_local.get(g_label, 0))          # unique local nodes
            summ_row["local_param_occurrences"] = int(total_by_label_local.get(g_label, 0))      # total occurrences across locals
            summ_row["local_param_unique"] = int(unique_by_label_local.get(g_label, 0))          # kept for backward compatibility
            summ_row["top_value_local"] = str(top_value_by_label_local.get(g_label, ""))
            # Each node occurs once in the global graph, so total == unique.
            summ_row["global_param_count"] = int(unique_by_label_global.get(g_label, 0))         # unique global nodes
            summ_row["global_param_occurrences"] = int(total_by_label_global.get(g_label, 0))    # Equals count; retained for symmetry
            summ_row["global_param_unique"] = int(unique_by_label_global.get(g_label, 0))
            summ_row["top_value_global"] = str(top_value_by_label_global.get(g_label, ""))
            # Table using global-graph rules
            summ_row["table_param_count"] = int(table_total_by_label.get(g_label, 0))
            summ_row["table_param_unique"] = int(table_unique_by_label.get(g_label, 0))
            summ_row["top_value_table"] = str(table_top_value_by_label.get(g_label, ""))
        else:
            summ_row["local_param_count"] = 0
            summ_row["local_param_unique"] = 0
            summ_row["top_value_local"] = ""
            summ_row["local_param_occurrences"] = 0
            summ_row["global_param_count"] = 0
            summ_row["global_param_unique"] = 0
            summ_row["top_value_global"] = ""
            summ_row["global_param_occurrences"] = 0
            summ_row["table_param_count"] = 0
            summ_row["table_param_unique"] = 0
            summ_row["top_value_table"] = ""
        all_summ_rows.append(summ_row)

        if not dist.empty:
            dist.insert(0, "column", col)
            # Sort by count descending for readability
            dist = dist.sort_values(["column", "count"], ascending=[True, False])
            all_dists.append(dist)

            # === Plot Top-15 for this column ===
            try:
                metric = "count_by_sid" if "count_by_sid" in dist.columns else "count"
                top = dist.nlargest(15, metric)
                if not top.empty:
                    plt.figure(figsize=(max(8, 0.55 * len(top)), 5))
                    plt.barh(range(len(top)), top[metric].values)
                    plt.yticks(range(len(top)), top["value"].astype(str).tolist())
                    plt.gca().invert_yaxis()
                    plt.title(f"Top-15: {col} ({metric})")
                    plt.xlabel(metric)
                    plt.tight_layout()
                    img_name = f"top15_{_safe_name(col)}.png"
                    plt.savefig(os.path.join(out_dir, img_name), dpi=200, bbox_inches="tight")
                    plt.close()
                    top_img_files.append(img_name)
            except Exception as e:
                print(f"[WARN] Failed to render Top-15 for '{col}': {e}")
        else:
            print(f"[INFO] Skip Top-15 for '{col}': no valid values")

    # Full diagnostic summary retained for debugging
    df_summary_full = pd.DataFrame(all_summ_rows).sort_values("column")
    df_summary_full.to_csv(os.path.join(out_dir, "summary_full.csv"), index=False)

    # === Compact manuscript table (essential columns only) ===
    # Include:
    # - Synthesis base (number of syntheses in scope=uniq(sanitize(synthesis_id)))
    # - Parameter completeness by synthesis: filled/missing/coverage
    # - Value diversity/top value by synthesis: distinct/top (+count)
    # - Local graphs: occurrences/unique/top
    paper_cols = [
        "column",
        "sid_base_sanitized_unique",
        # By synthesis
        "filled_sids_by_sid", "missing_sids_by_sid", "coverage_pct_by_sid",
        "valid_count_by_sid",
        "distinct_valid_by_sid", "top_value_by_sid", "top_value_by_sid_count",
        # Local graphs
        "local_param_occurrences", "local_param_count", "top_value_local",
    ]
    existing_paper_cols = [c for c in paper_cols if c in df_summary_full.columns]
    df_paper = df_summary_full[existing_paper_cols].copy()

    # By default, summary.csv is the compact manuscript table
    df_paper.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    df_paper.to_csv(os.path.join(out_dir, "table_for_paper.csv"), index=False)

    # Use the compact version as the main table in the HTML below
    df_summary = df_paper

    # Distribution may be large; save only if available
    if all_dists:
        df_dist = pd.concat(all_dists, ignore_index=True)
        # Round percentages
        df_dist["pct"] = df_dist["pct"].round(4)
        if "pct_by_sid" in df_dist.columns:
            df_dist["pct_by_sid"] = df_dist["pct_by_sid"].round(4)
        df_dist.to_csv(os.path.join(out_dir, "distribution.csv"), index=False)
    else:
        df_dist = pd.DataFrame(columns=["column", "value", "count", "pct"])

    # === Top-3 charts for key parameters ===
    key_cols = [
        "composition",
        "precursor", "precursor_mass", "precursor_conc",
        "additive", "additive_mass", "additive_conc",
        "seeds", "seed_mass", "seed_conc",
        "solvent", "temperature", "ph", "method",
        "shape", "crystalsystem",
        "size_length", "size_width", "size_depth",
        "pdfno_list",
    ]
    key_img_files = []

    def _plot_topk_from_dist(col_name, topk=3):
        nonlocal key_img_files
        if df_dist is None or df_dist.empty:
            return
        sub = df_dist[df_dist["column"] == col_name]
        if sub.empty:
            return
        metric = "count_by_sid" if "count_by_sid" in sub.columns and sub["count_by_sid"].notna().any() else "count"
        top = sub.nlargest(topk, metric)
        if top.empty:
            return
        plt.figure(figsize=(8, 3.8))
        plt.barh(range(len(top)), top[metric].values)
        plt.yticks(range(len(top)), top["value"].astype(str).tolist())
        plt.gca().invert_yaxis()
        plt.title(f"Top-{topk}: {col_name} ({metric})")
        plt.xlabel(metric)
        plt.tight_layout()
        img_name = f"top3_{_safe_name(col_name)}.png"
        plt.savefig(os.path.join(out_dir, img_name), dpi=200, bbox_inches="tight")
        plt.close()
        key_img_files.append(img_name)

    # 1) Plot the top 3 values for all key columns except pdf, which needs special handling
    for kc in key_cols:
        if kc == "pdfno_list":
            continue
        if kc in df.columns:
            _plot_topk_from_dist(kc, topk=3)

    # 2) Handle pdfno_list separately: expand lists and count by sid, if available, within the filekey scope
    if "pdfno_list" in df.columns:
        try:
            sid_series_local = sid_key_series if sid_key_series is not None else pd.Series(range(len(df)))
            scope_mask_local = sid_in_scope_mask if (sid_key_series is not None and sid_in_scope_mask is not None) else pd.Series([True] * len(df))
            rows = []
            for idx, (sid_val, cell) in enumerate(zip(sid_series_local, df["pdfno_list"])):
                if not bool(scope_mask_local.iloc[idx]):
                    continue
                if not _is_valid_value(cell, unknown_tokens):
                    continue
                items = []
                if isinstance(cell, str):
                    val = cell.strip()
                    parsed = None
                    if val.startswith("[") and val.endswith("]"):
                        try:
                            parsed = ast.literal_eval(val)
                        except Exception:
                            parsed = None
                    if parsed is None:
                        # Simple split on commas/semicolons
                        items = [x.strip() for x in re.split(r"[;,]", val) if x.strip()]
                    else:
                        # Expect a list of strings
                        items = [str(x).strip() for x in parsed if str(x).strip()]
                elif isinstance(cell, (list, tuple)):
                    items = [str(x).strip() for x in cell if str(x).strip()]
                for it in items:
                    rows.append({"sid": sid_val, "value": it})
            if rows:
                # Count unique (sid_filekey, value) pairs within scope
                pdf_df = pd.DataFrame(rows).drop_duplicates()
                if sid_key_series is not None:
                    vc_pdf = pdf_df["value"].value_counts()
                    metric_label = "count_by_sid"
                else:
                    vc_pdf = pdf_df["value"].value_counts()
                    metric_label = "count"
                top_pdf = vc_pdf.nlargest(3)
                if not top_pdf.empty:
                    plt.figure(figsize=(8, 3.8))
                    plt.barh(range(len(top_pdf)), top_pdf.values)
                    plt.yticks(range(len(top_pdf)), top_pdf.index.astype(str).tolist())
                    plt.gca().invert_yaxis()
                    plt.title(f"Top-3: pdfno_list ({metric_label})")
                    plt.xlabel(metric_label)
                    plt.tight_layout()
                    img_name = f"top3_{_safe_name('pdfno_list')}.png"
                    plt.savefig(os.path.join(out_dir, img_name), dpi=200, bbox_inches="tight")
                    plt.close()
                    key_img_files.append(img_name)
        except Exception as e:
            print(f"[WARN] Failed Top-3 for pdfno_list: {e}")

    # 3) Summary plot: top-3 grid for all key columns
    combined_img = None
    try:
        tops = []
        # Collect the top 3 values for each column from df_dist (and separately for pdf)
        for kc in key_cols:
            if kc == "pdfno_list":
                # Recalculate the top 3 pdf values as above, returning a DataFrame for plotting
                try:
                    sid_series_local = sid_key_series if sid_key_series is not None else pd.Series(range(len(df)))
                    scope_mask_local = sid_in_scope_mask if (sid_key_series is not None and sid_in_scope_mask is not None) else pd.Series([True] * len(df))
                    rows = []
                    for idx, (sid_val, cell) in enumerate(zip(sid_series_local, df["pdfno_list"])):
                        if not bool(scope_mask_local.iloc[idx]):
                            continue
                        if not _is_valid_value(cell, unknown_tokens):
                            continue
                        items = []
                        if isinstance(cell, str):
                            val = cell.strip()
                            parsed = None
                            if val.startswith("[") and val.endswith("]"):
                                try:
                                    parsed = ast.literal_eval(val)
                                except Exception:
                                    parsed = None
                            if parsed is None:
                                items = [x.strip() for x in re.split(r"[;,]", val) if x.strip()]
                            else:
                                items = [str(x).strip() for x in parsed if str(x).strip()]
                        elif isinstance(cell, (list, tuple)):
                            items = [str(x).strip() for x in cell if str(x).strip()]
                        for it in items:
                            rows.append({"sid": sid_val, "value": it})
                    metric = "count_by_sid" if sid_key_series is not None else "count"
                    if rows:
                        pdf_df = pd.DataFrame(rows).drop_duplicates()
                        vc_pdf = pdf_df["value"].value_counts()
                        top_pdf = vc_pdf.nlargest(3)
                        if not top_pdf.empty:
                            tdf = pd.DataFrame({"value": top_pdf.index.astype(str), metric: top_pdf.values})
                            tops.append((kc, tdf.reset_index(drop=True), metric))
                except Exception as e:
                    print(f"[WARN] Combined grid pdf parse failed: {e}")
            else:
                if kc in df.columns and df_dist is not None and not df_dist.empty:
                    sub = df_dist[df_dist["column"] == kc]
                    if not sub.empty:
                        metric = "count_by_sid" if ("count_by_sid" in sub.columns and sub["count_by_sid"].notna().any()) else "count"
                        top = sub.nlargest(3, metric)[["value", metric]]
                        if not top.empty:
                            tops.append((kc, top.reset_index(drop=True), metric))
        if tops:
            cols = 3
            rows = math.ceil(len(tops) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3.8))
            # Normalize axes, which may be one-dimensional
            if isinstance(axes, np.ndarray) and axes.ndim == 1:
                axes = axes.reshape(rows, cols)
            for i, (name, tdf, metric) in enumerate(tops):
                r, c = divmod(i, cols)
                ax = axes[r, c]
                ax.barh(range(len(tdf)), tdf[metric].values)
                ax.set_yticks(range(len(tdf)))
                ax.set_yticklabels(tdf["value"].astype(str).tolist(), fontsize=8)
                ax.invert_yaxis()
                ax.set_title(f"{name} (Top-3 {metric})", fontsize=10)
                ax.set_xlabel(metric)
            # Hide empty subplots
            for j in range(len(tops), rows * cols):
                r, c = divmod(j, cols)
                axes[r, c].axis("off")
            plt.tight_layout()
            combined_name = "top3__combined_grid.png"
            plt.savefig(os.path.join(out_dir, combined_name), dpi=200, bbox_inches="tight")
            plt.close(fig)
            combined_img = combined_name
    except Exception as e:
        print(f"[WARN] Failed to render combined Top-3 grid: {e}")

    # HTML report (DataTables)
    index_html = os.path.join(out_dir, "index.html")
    html = []
    html.append("""
<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8" />
<title>Data Quality — merged_all.xlsx</title>
<link rel="stylesheet" href="https://cdn.datatables.net/2.0.7/css/dataTables.dataTables.min.css">
<style>
 body {font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 24px;}
 h1 {margin:0 0 8px 0;}
 h2 {margin-top: 24px;}
 .muted{color:#6b7280;font-size:12px;margin-bottom:16px}
 .wrap{overflow-x:auto}
 table.dataTable thead th {white-space: nowrap;}
</style></head><body>
""")
    html.append(f"<h1>Data Quality Report</h1><div class='muted'>File: {os.path.basename(input_xlsx)} • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>")

    # Summary
    html.append("<h2>Summary table (for the manuscript)</h2>")
    html.append(f"<div class='muted'>Base for by_sid: <code>uniq(sanitize(synthesis_id))</code> = {int(sanitized_unique) if sid_series is not None else 0}</div><div class='wrap'>")
    html.append(pd.DataFrame(df_summary).to_html(index=False, classes="dt", table_id="tbl_summary"))
    html.append("</div>")

    # Compact manuscript table
    try:
        html.append("<h2>Manuscript table (CSV copy)</h2>")
        html.append("<div class='muted'>Completeness is calculated by synthesis: a synthesis is filled if it has at least one valid parameter value. The <code>filled_sids_by_sid</code>/<code>missing_sids_by_sid</code> metrics use <code>uniq(sanitize(synthesis_id))</code> as the base. Local graph statistics include total nodes (occurrences), distinct values (unique), and the most frequent value.</div><div class='wrap'>")
        html.append(pd.DataFrame(df_paper).to_html(index=False, classes="dt", table_id="tbl_paper"))
        html.append("</div>")
    except Exception as e:
        html.append(f"<div class='muted'>[WARN] Failed to build the manuscript table: {e}</div>")

    # Distributions
    html.append("<h2>Valid value distributions</h2><div class='muted'>NaN, empty strings, 'unknown', and '-999' are excluded. <code>count_by_sid</code>/<code>pct_by_sid</code> are also shown: each value is counted at most once per <code>sanitize(synthesis_id)</code>; the base contains ≈ 4248 syntheses.</div><div class='wrap'>")
    html.append(pd.DataFrame(df_dist.head(100000)).to_html(index=False, classes="dt", table_id="tbl_dist"))
    html.append("</div>")

    # Top-15 plots for each column
    if top_img_files:
        html.append("<h2>Top 15 values per column</h2>")
        html.append("<div class='muted'>Metric: <code>count_by_sid</code> if available, otherwise <code>count</code>. NaN/empty/'unknown'/'-999' values are excluded.</div>")
        # Local grid styling
        html.append("""<style>.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px} figure{margin:0} figure img{width:100%;height:auto;border:1px solid #e5e7eb;border-radius:6px;background:#fff}</style>""")
        html.append("<div class='grid'>")
        for fname in top_img_files:
            html.append(f"<figure><img src='{fname}' loading='lazy'/><figcaption class='muted'>{fname}</figcaption></figure>")
        html.append("</div>")

    # Top 3 values for key parameters (summary block)
    if key_img_files:
        html.append("<h2>Top 3 values for key parameters</h2>")
        html.append("<div class='muted'>Each column uses <code>count_by_sid</code> if available, otherwise <code>count</code>. For <code>pdfno_list</code>, values are extracted from lists.</div>")
        html.append("<div class='grid'>")
        for fname in key_img_files:
            html.append(f"<figure><img src='{fname}' loading='lazy'/><figcaption class='muted'>{fname}</figcaption></figure>")
        html.append("</div>")

    # Combined summary plot (one image)
    if combined_img:
        html.append("<h2>Summary plot: top 3 values for key parameters (one image)</h2>")
        html.append("<div class='wrap'>")
        html.append(f"<img src='{combined_img}' loading='lazy' style='max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:6px;background:#fff'/>")
        html.append("</div>")

    html.append("""
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/2.0.7/js/dataTables.min.js"></script>
<script>
  new DataTable('#tbl_summary', {pageLength:50, deferRender:true, scrollX:true});
  new DataTable('#tbl_paper', {pageLength:50, deferRender:true, scrollX:true});
  new DataTable('#tbl_dist', {pageLength:50, deferRender:true, scrollX:true});
</script>
</body></html>
""")
    with open(index_html, "w", encoding="utf-8") as f:
        f.write("".join(html))

    print(f"[OK] Summary:       {os.path.join(out_dir, 'summary.csv')}")
    print(f"[OK] Summary(full):  {os.path.join(out_dir, 'summary_full.csv')}")
    print(f"[OK] Distribution:  {os.path.join(out_dir, 'distribution.csv') if not df_dist.empty else '(no valid values)'}")
    print(f"[OK] HTML report:   {index_html}")
    return out_dir


def parse_args():
    ap = argparse.ArgumentParser(description="Profile merged_all.xlsx: missing values and per-column distributions")
    ap.add_argument(
        "--input", "-i", default=_default_input_xlsx(),
        help="Path to the Excel file. Default: <script_dir>/merged_all.xlsx",
    )
    ap.add_argument("--sheet", "-s", default=None, help="Sheet name (first sheet by default)")
    ap.add_argument("--out-dir", "-o", default=None, help="Report directory (default: dq_report_YYYYmmdd_HHMMSS beside the input file)")
    ap.add_argument("--unknown", nargs="*", default=UNKNOWN_TOKENS_DEFAULT,
                    help="Case-insensitive tokens treated as 'unknown'. Default: %(default)s")
    ap.add_argument(
        "--local-graphs-dir", default=_default_local_graphs_dir(),
        help="Directory containing local graphs (*.graphml). Default: outputs/graph/local_graphs_renamed",
    )
    ap.add_argument(
        "--global-graph-path", default=_default_global_graph_path(),
        help="Path to the global graph (graph.graphml). Default: outputs/graph/graph.graphml",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(args.input, sheet=args.sheet, out_dir=args.out_dir, unknown_tokens=args.unknown)