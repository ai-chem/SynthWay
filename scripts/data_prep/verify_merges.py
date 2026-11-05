#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Checks only:
  1) Numeric merges outside tolerances (labels ending in *_mass, *_conc)
  2) Whether similarity >= the label-specific or default threshold

Sources: merges_detailed.csv and/or merge_pairs_for_renaming.csv.
Outputs: violations_numeric_*.csv, violations_similarity_*.csv
"""

import argparse
import json
import os
import re

import pandas as pd

NUMERIC_LABEL_PAT = re.compile(r".*_(mass|conc)$", re.IGNORECASE)


def read_csv_safe(path: str) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False) if os.path.exists(path) else pd.DataFrame()


def extract_label_value(node_id: str):
    s = str(node_id)
    if ":" in s:
        a, b = s.split(":", 1)
        return a.strip(), b.strip()
    return "", s.strip()


def normalize_number_text(v: str) -> str:
    if not isinstance(v, str):
        v = str(v)
    s = v.strip()
    # Replace comma with decimal point; normalize minus signs/whitespace
    s = s.replace(",", ".").replace("\u2212", "-").replace("\u2013", "-").replace("\u00a0", " ")
    # Convert "×10^k" / "x10^k" / "*10^k" forms to exponential notation
    s = re.sub(r"([0-9.]+)\s*[x×*]\s*10\s*\^\s*([+-]?\d+)", r"\1e\2", s, flags=re.IGNORECASE)
    s = re.sub(r"([0-9.]+)\s*[x×*]\s*10\s*([+-]?\d+)", r"\1e\2", s, flags=re.IGNORECASE)
    m = re.search(r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", s, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def to_float_maybe(v: str):
    s = normalize_number_text(v)
    try:
        return float(s) if s else None
    except Exception:
        return None


def within_tolerance(x: float, y: float, rtol: float, atol: float) -> bool:
    return abs(x - y) <= max(atol, rtol * max(abs(x), abs(y)))


def load_label_thresholds(path: str | None, default_thr: float) -> dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        # Convert threshold labels to strings and values to floats.
        return {str(k): float(v) for k, v in d.items()}
    return {"__default__": float(default_thr)}


def get_thr(thr_map: dict, label: str, default_key="__default__") -> float:
    return float(thr_map.get(label, thr_map.get(default_key, 0.95)))


def check_source(
    df: pd.DataFrame, source_name: str, outdir: str, rtol: float, atol: float, thr_map: dict
):
    if df.empty:
        return {"checked": 0, "viol_num": 0, "viol_sim": 0, "exports": {}}

    # Standardize column names
    # merges_detailed: removed_node, representative, similarity, label_removed, label_representative
    # merge_pairs_for_renaming: source_node, target_node, label_source, label_target, similarity, mapping_rule
    if "removed_node" in df.columns:
        src_col, dst_col = "removed_node", "representative"
        lsrc_col, ldst_col = "label_removed", "label_representative"
        sim_col, rule_col = "similarity", None
    else:
        src_col, dst_col = "source_node", "target_node"
        lsrc_col, ldst_col = "label_source", "label_target"
        sim_col = "similarity" if "similarity" in df.columns else None
        rule_col = "mapping_rule" if "mapping_rule" in df.columns else None

    work = df.copy()
    # Use only rows with a target node (actual pairs)
    work = work[work[dst_col].astype(str).str.len() > 0]
    if rule_col:
        work = work[work[rule_col] == "merge"]

    numeric_viol = []
    sim_viol = []

    for _, r in work.iterrows():
        node_src = str(r[src_col])
        node_dst = str(r[dst_col])
        lbl_src = str(r.get(lsrc_col, ""))
        lbl_dst = str(r.get(ldst_col, ""))
        lbl = lbl_src or lbl_dst

        # 1) Check the similarity threshold if the column exists and the number is valid
        sim = None
        if sim_col and str(r.get(sim_col, "")).strip() not in {"", "NaN", "nan"}:
            try:
                sim = float(r[sim_col])
            except Exception:
                sim = None
        thr = get_thr(thr_map, lbl)
        if sim is not None and sim < thr:
            sim_viol.append(
                {
                    "src": node_src,
                    "dst": node_dst,
                    "label": lbl,
                    "similarity": sim,
                    "threshold": thr,
                }
            )

        # 2) Check numeric labels *_mass / *_conc for tolerance violations
        #   Use the label column, or fall back to node_id
        if not lbl:
            lbl, _ = extract_label_value(node_src)
        if NUMERIC_LABEL_PAT.fullmatch(lbl or ""):
            _, vsrc = extract_label_value(node_src)
            _, vdst = extract_label_value(node_dst)
            xs, ys = to_float_maybe(vsrc), to_float_maybe(vdst)
            if xs is not None and ys is not None:
                if not within_tolerance(xs, ys, rtol=rtol, atol=atol):
                    numeric_viol.append(
                        {
                            "src": node_src,
                            "dst": node_dst,
                            "label": lbl,
                            "x": xs,
                            "y": ys,
                            "abs_diff": abs(xs - ys),
                            "rtol": rtol,
                            "atol": atol,
                            "similarity": sim,
                            "threshold": thr,
                        }
                    )

    exports = {}
    if numeric_viol:
        p = os.path.join(outdir, f"violations_numeric_{source_name}.csv")
        pd.DataFrame(numeric_viol).to_csv(p, index=False)
        exports["numeric"] = p
    if sim_viol:
        p = os.path.join(outdir, f"violations_similarity_{source_name}.csv")
        pd.DataFrame(sim_viol).to_csv(p, index=False)
        exports["similarity"] = p

    return {
        "checked": int(work.shape[0]),
        "viol_num": len(numeric_viol),
        "viol_sim": len(sim_viol),
        "exports": exports,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Validate ONLY numeric merges and similarity thresholds."
    )
    ap.add_argument(
        "--outdir", required=True, help="Artifact directory (graph_output_without_id_95)"
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=0.05,
        help="Relative tolerance for *_mass/*_conc (default: 5%%)",
    )
    ap.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    ap.add_argument(
        "--sim-thr-default", type=float, default=0.95, help="Default similarity threshold"
    )
    ap.add_argument(
        "--sim-thr-per-label",
        type=str,
        default="",
        help='JSON of label-specific thresholds {"precursor_mass": 0.9999, ...}',
    )
    args = ap.parse_args()

    outdir = args.outdir
    thr_map = load_label_thresholds(args.sim_thr_per_label, args.sim_thr_default)

    print("\n=== Numeric & Similarity Checks (lean mode) ===")
    print(f"Outdir: {outdir}")
    print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")
    print(
        f"Similarity thresholds: default={args.sim_thr_default}, per-label keys={list(thr_map.keys())}"
    )

    md = read_csv_safe(os.path.join(outdir, "merges_detailed.csv"))
    pr = read_csv_safe(os.path.join(outdir, "merge_pairs_for_renaming.csv"))

    res_md = (
        check_source(md, "merges_detailed", outdir, args.rtol, args.atol, thr_map)
        if not md.empty
        else None
    )
    res_pr = (
        check_source(pr, "pairs_for_renaming", outdir, args.rtol, args.atol, thr_map)
        if not pr.empty
        else None
    )

    def pretty(res, name):
        if not res:
            print(f"- {name}: file missing or empty.")
            return
        print(
            f"- {name}: checked {res['checked']} | numeric_viol={res['viol_num']} | sim_viol={res['viol_sim']}"
        )
        for k, p in (res["exports"] or {}).items():
            print(f"    ↳ {k}: {p}")

    pretty(res_md, "merges_detailed.csv")
    pretty(res_pr, "merge_pairs_for_renaming.csv")
    print("Done.")


if __name__ == "__main__":
    main()
