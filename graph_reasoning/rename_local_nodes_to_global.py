"""
rename_local_nodes_to_global.py:
- Rename nodes in local .graphml files using merge_map_by_rep.csv (exact matches only)
- Remove nodes listed in removed_nodes_cleaning.csv, if available, or sample_nodes from removed_components.csv
- Save the result and write a report
- Check how many renaming targets exist in the global graph.graphml
"""

import argparse
import ast
import json
import os
import re

import networkx as nx
import pandas as pd


def read_global_graph(path: str) -> nx.Graph:
    """Read global graph from .graphml or .gexf based on extension."""
    p = str(path)
    ext = os.path.splitext(p)[1].lower()
    if ext == ".gexf":
        return nx.read_gexf(p)
    return nx.read_graphml(p)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
GRAPH_DIR = os.environ.get("SYNTHWAY_GRAPH_OUTPUT", os.path.join(REPO_DIR, "outputs", "graph"))
DEFAULT_MERGE_MAP = os.path.join(GRAPH_DIR, "merge_map_by_rep.csv")
DEFAULT_REMOVED_COMPONENTS = os.path.join(GRAPH_DIR, "removed_components.csv")
DEFAULT_REMOVED_NODES = os.path.join(GRAPH_DIR, "removed_nodes_cleaning.csv")
DEFAULT_INPUT_DIR = os.path.join(GRAPH_DIR, "local_graphs_without_synthesis")
DEFAULT_OUTPUT_DIR = os.path.join(GRAPH_DIR, "local_graphs_renamed")
# Prefer the final cleaned global graph if present; fallback to graph.graphml
_DEFAULT_GEXF = os.path.join(GRAPH_DIR, "cleaned_synthesis_graph.gexf")
_DEFAULT_GRAPHML = os.path.join(GRAPH_DIR, "graph.graphml")
DEFAULT_GLOBAL_GRAPH = _DEFAULT_GEXF if os.path.isfile(_DEFAULT_GEXF) else _DEFAULT_GRAPHML

# Nodes that must NEVER be removed during rename/prune (explicit exceptions)
KEEP_NODES: set[str] = {
    "composition:HA-Alg-ZnO MPs",
}

NBSP = "\u00a0"
DASH_CHARS = [
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2015",
    "\u2212",
    "\u2043",
    "\ufe58",
    "\ufe63",
    "\uff0d",
]
_dash_re = re.compile("|".join(map(re.escape, DASH_CHARS)))
_ws_re = re.compile(r"\s+")


def _canon(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace(NBSP, " ")
    s = _dash_re.sub("-", s)
    s = _ws_re.sub(" ", s).strip()
    return s


def _split_label_value(node_id: str) -> tuple[str, str]:
    s = str(node_id)
    if ":" not in s:
        return "", _canon(s)
    lbl, val = s.split(":", 1)
    return _canon(lbl), _canon(val)


def _add_bidirectional_keys(mapping: dict, src: str, dst: str):
    # Preserve destinations (dst) exactly as stored in CSV; canonicalize source keys only
    dst_raw = str(dst)
    c_src = _canon(src)
    mapping[c_src] = dst_raw
    # Add a source alias in value##label format
    lbl_s, val_s = _split_label_value(c_src)
    if lbl_s and val_s:
        mapping[f"{val_s}##{lbl_s}"] = dst_raw
    # Allow the target node itself and its value##label alias as identity keys
    mapping[dst_raw] = dst_raw
    lbl_d, val_d = _split_label_value(dst_raw)
    if lbl_d and val_d:
        mapping[f"{val_d}##{lbl_d}"] = dst_raw


# Read merge_map_by_rep.csv and build the source -> representative mapping
def load_rename_map(merge_csv: str) -> dict[str, str]:
    df = pd.read_csv(merge_csv)
    # If kept_after_cleaning is available, retain only representatives present in the final global graph
    if "kept_after_cleaning" in df.columns:
        df = df[df["kept_after_cleaning"] == True].copy()
    need = {"representative", "merged_from_json"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{merge_csv}: missing columns {miss}")

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        rep = str(row["representative"])
        _add_bidirectional_keys(mapping, rep, rep)
        merged_json = row.get("merged_from_json")
        if pd.isna(merged_json) or str(merged_json).strip() in ("", "[]"):
            continue
        items = None
        try:
            items = ast.literal_eval(str(merged_json))
            if not isinstance(items, (list, tuple)):
                items = None
        except Exception:
            items = None
        if items is None:
            s = str(merged_json).strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            items = [x.strip().strip("'\"") for x in re.split(r"[;,]", s) if x.strip()]
        for src in items:
            _add_bidirectional_keys(mapping, str(src), rep)
    return mapping


# Return a set of node names to remove
def load_removed_nodes(removed_components_csv: str, removed_nodes_csv: str = "") -> set[str]:
    nodes: set[str] = set()
    if removed_nodes_csv and os.path.isfile(removed_nodes_csv):
        df = pd.read_csv(removed_nodes_csv)
        if "node" not in df.columns:
            for cand in ("nodes", "label"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "node"})
                    break
        if "node" in df.columns:
            for val in df["node"].dropna().astype(str):
                if val != "":
                    nodes.add(val)
    if not nodes and removed_components_csv and os.path.isfile(removed_components_csv):
        dfc = pd.read_csv(removed_components_csv)
        if "sample_nodes" in dfc.columns:
            for sample in dfc["sample_nodes"].dropna().astype(str):
                parts = [p for p in sample.split(";") if p != ""]
                nodes.update(parts)
    return nodes


# Read merge_pairs_for_renaming.csv, if available, and return the complete source->target mapping
def load_pairs_map(pairs_csv: str) -> dict[str, str]:
    if not pairs_csv or not os.path.isfile(pairs_csv):
        return {}
    df = pd.read_csv(pairs_csv)
    need = {"source_node", "target_node"}
    miss = need - set(df.columns)
    if miss:
        return {}
    m: dict[str, str] = {}
    for _, r in df.iterrows():
        src = str(r.get("source_node", "")).strip()
        dst = str(r.get("target_node", "")).strip()
        if src and dst:
            _add_bidirectional_keys(m, src, dst)
    return m


# Rename nodes and remove specified nodes. Return (new_graph, n_renamed, n_removed, [(old,new),...])
def relabel_and_prune(
    G: nx.Graph,
    rename_map: dict[str, str],
    to_remove: set[str],
    keep_nodes: set[str] | None = None,
) -> tuple[nx.Graph, int, int, list[tuple]]:
    relabel: dict[str, str] = {}
    renamed_pairs: list[tuple] = []
    for n in list(G.nodes()):
        try:
            lbl = (G.nodes[n] or {}).get("label", "")
        except Exception:
            lbl = ""
        candidates = []
        if isinstance(lbl, str) and lbl:
            candidates.append(_canon(f"{n}##{lbl}"))
        candidates.append(_canon(str(n)))

        new_name = None
        for key in candidates:
            if key in rename_map:
                new_name = rename_map[key]
                break
        if new_name is not None and new_name != n:
            relabel[n] = new_name
            renamed_pairs.append((n, new_name))

    G2 = nx.relabel_nodes(G, relabel, copy=True)
    n_renamed = len(relabel)

    # Remove specified nodes, except those in keep_nodes
    keep_nodes = set(keep_nodes or set())
    effective_remove = set(to_remove)
    for nm in list(to_remove):
        if nm in rename_map:
            effective_remove.add(rename_map[nm])

    # Apply explicit keep-list: never drop these nodes (even if they appear in to_remove)
    if keep_nodes:
        effective_remove.difference_update(keep_nodes)

    to_drop: set[str] = set()
    for n in list(G2.nodes()):
        if keep_nodes and n in keep_nodes:
            continue
        if n in effective_remove:
            to_drop.add(n)
            continue
        try:
            lbl = (G2.nodes[n] or {}).get("label", "")
        except Exception:
            lbl = ""
        if isinstance(lbl, str) and lbl and f"{n}##{lbl}" in effective_remove:
            # respect keep-list
            if not (keep_nodes and n in keep_nodes):
                to_drop.add(n)

    if to_drop:
        G2.remove_nodes_from(to_drop)
    n_removed = len(to_drop)
    return G2, n_renamed, n_removed, renamed_pairs


def sanitize_for_graphml(G: nx.Graph) -> None:
    for _, data in list(G.nodes(data=True)):
        for k, v in list(data.items()):
            if isinstance(v, (list, tuple, dict)):
                data[k] = json.dumps(v, ensure_ascii=False)
    for _, _, data in list(G.edges(data=True)):
        for k, v in list(data.items()):
            if isinstance(v, (list, tuple, dict)):
                data[k] = json.dumps(v, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Rename local graph nodes using merge_map, remove nodes, and compare with the global graph"
    )
    parser.add_argument("--merge_map", default=DEFAULT_MERGE_MAP)
    parser.add_argument("--removed_components", default=DEFAULT_REMOVED_COMPONENTS)
    parser.add_argument("--removed_nodes_cleaning", default=DEFAULT_REMOVED_NODES)
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--global_graph", default=DEFAULT_GLOBAL_GRAPH)
    parser.add_argument(
        "--pairs_csv", default="", help="Path to merge_pairs_for_renaming.csv (optional)"
    )
    parser.add_argument(
        "--strict_align",
        action="store_true",
        help="Strictly align local graphs with the final global graph: remove all nodes absent from the global graph",
    )
    parser.add_argument(
        "--audit_not_in_global_csv",
        default="",
        help="Output path for nodes removed by strict alignment",
    )
    parser.add_argument(
        "--audit_local_vs_global_csv",
        default="",
        help="Output path for nodes remaining in local graphs but absent from the global graph after all operations",
    )
    args = parser.parse_args()

    if not args.pairs_csv:
        args.pairs_csv = os.path.join(
            os.path.dirname(args.merge_map), "merge_pairs_for_renaming.csv"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the global graph immediately to obtain the final node set
    try:
        G_global = read_global_graph(args.global_graph)
        global_nodes = set(G_global.nodes())
    except Exception as e:
        print(f"Failed to load global graph {args.global_graph}: {e}")
        global_nodes = set()

    rename_map = load_rename_map(args.merge_map)
    pairs_map = load_pairs_map(args.pairs_csv)
    if pairs_map:
        rename_map.update(pairs_map)

    to_remove = load_removed_nodes(args.removed_components, args.removed_nodes_cleaning)

    # Apply explicit keep-list early
    if KEEP_NODES:
        to_remove -= KEEP_NODES

    # Add renaming targets absent from the final global graph to the removal set
    if global_nodes:
        missing_targets = {dst for dst in set(rename_map.values()) if dst not in global_nodes}
        if missing_targets:
            print(
                f"{len(missing_targets)} target names are absent from the final global graph; adding them to the removal set"
            )
            to_remove |= missing_targets

    rows = []
    renamed_pairs_all: list[tuple] = []
    # Count all nodes before and after processing for the removal audit
    pre_nodes_all: set[str] = set()
    post_nodes_all: set[str] = set()

    audit_not_in_global: list[str] = []
    audit_local_vs_global: list[dict[str, str]] = []

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".graphml")]
    if not files:
        print(f"No .graphml files in input directory: {args.input_dir}")
    for fn in sorted(files):
        src_path = os.path.join(args.input_dir, fn)
        dst_path = os.path.join(args.output_dir, fn)
        try:
            G_loc = nx.read_graphml(src_path)
        except Exception as e:
            print(f"{fn}: failed to read GraphML: {e}")
            continue

        # Audit: before transformations
        pre_nodes_all.update(G_loc.nodes())

        G_out, n_ren, n_del, pairs = relabel_and_prune(
            G_loc, rename_map, to_remove, keep_nodes=KEEP_NODES
        )
        renamed_pairs_all.extend(pairs)

        # Count nodes absent from the global graph before strict alignment
        not_in_global_before = 0
        if global_nodes:
            try:
                not_in_global_before = sum(1 for n in list(G_out.nodes()) if n not in global_nodes)
            except Exception:
                not_in_global_before = 0
        else:
            not_in_global_before = 0

        removed_not_in_global = 0
        if args.strict_align and global_nodes:
            extra = [n for n in list(G_out.nodes()) if n not in global_nodes]
            if extra:
                G_out.remove_nodes_from(extra)
                removed_not_in_global = len(extra)
                audit_not_in_global.extend(extra)

        # Nodes still absent from the global graph after all transformations
        not_in_global_after = 0
        if global_nodes:
            try:
                leftovers = [n for n in list(G_out.nodes()) if n not in global_nodes]
                not_in_global_after = len(leftovers)
                if leftovers:
                    for n in leftovers:
                        try:
                            lbl = (G_out.nodes[n] or {}).get("label", "")
                        except Exception:
                            lbl = ""
                        audit_local_vs_global.append(
                            {"file": fn, "node": str(n), "label": str(lbl)}
                        )
            except Exception:
                not_in_global_after = 0
        else:
            not_in_global_after = 0

        # Audit: after transformations
        post_nodes_all.update(G_out.nodes())

        sanitize_for_graphml(G_out)
        try:
            nx.write_graphml(G_out, dst_path)
        except Exception as e:
            print(f"{fn}: error writing GraphML: {e}")
            continue

        rows.append(
            {
                "file": fn,
                "renamed": n_ren,
                "removed": n_del,
                "removed_not_in_global": removed_not_in_global,
                "not_in_global_before": not_in_global_before,
                "not_in_global_after": not_in_global_after,
                "nodes_out": G_out.number_of_nodes(),
                "edges_out": G_out.number_of_edges(),
            }
        )
        print(
            f"{fn}: renamed={n_ren}, removed={n_del}, nodes={G_out.number_of_nodes()}, edges={G_out.number_of_edges()}"
        )

    # Per-file report
    report_path = os.path.join(args.output_dir, "_rename_prune_report.csv")
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"Done. Report: {report_path}")

    # Compare all local nodes with the global graph after all operations
    if global_nodes:
        try:
            mismatch_total = int(sum(int(r.get("not_in_global_after", 0)) for r in rows))
        except Exception:
            mismatch_total = 0
        if mismatch_total == 0:
            print("\n Local/global check: all local graph nodes are present in the global graph")
        else:
            print(
                f"\n Local/global check: found {mismatch_total} nodes absent from the global graph"
            )
        if args.audit_local_vs_global_csv:
            try:
                pd.DataFrame(audit_local_vs_global).to_csv(
                    args.audit_local_vs_global_csv, index=False
                )
                print(
                    f" Local nodes absent from the global graph after all transformations: {args.audit_local_vs_global_csv} ({len(audit_local_vs_global)} entries)"
                )
            except Exception as e:
                print(f"Failed to save local-vs-global audit: {e}")
    else:
        print("Local/global node check skipped: global graph not loaded.")

    if args.audit_not_in_global_csv:
        try:
            pd.DataFrame({"node": sorted(set(audit_not_in_global))}).to_csv(
                args.audit_not_in_global_csv, index=False
            )
            print(
                f"Nodes removed by strict alignment (absent from the global graph): {args.audit_not_in_global_csv} ({len(set(audit_not_in_global))})"
            )
        except Exception as e:
            print(f"Failed to save not-in-global audit: {e}")

    # Compare with the global graph, accounting for explicit removals
    intended_remove = set(to_remove)  # Nodes explicitly selected for removal

    if not global_nodes:
        try:
            G_global = read_global_graph(args.global_graph)
            global_nodes = set(G_global.nodes())
        except Exception as e:
            print(f"Failed to load global graph {args.global_graph}: {e}")
            global_nodes = set()

    present_actions = 0
    missing_actions_unexpected = 0
    missing_actions_expected = 0
    missing_unexpected_pairs = []
    target_unique = set()

    for old_name, new_name in renamed_pairs_all:
        target_unique.add(new_name)
        if new_name in global_nodes:
            present_actions += 1
        elif new_name in intended_remove:
            missing_actions_expected += 1
        else:
            missing_actions_unexpected += 1
            missing_unexpected_pairs.append((old_name, new_name))

    unique_in_global = sum(1 for t in target_unique if t in global_nodes)
    unique_not_in_global_expected = sum(
        1 for t in target_unique if (t not in global_nodes and t in intended_remove)
    )
    unique_not_in_global_unexpected = sum(
        1 for t in target_unique if (t not in global_nodes and t not in intended_remove)
    )

    print("\n Comparison with the global graph:")
    print(f"  renamed_actions_total:                        {len(renamed_pairs_all)}")
    print(f"  renamed_actions_in_global:                    {present_actions}")
    print(
        f"  renamed_actions_not_in_global_expected:       {missing_actions_expected}  # in the removal set"
    )
    print(f"  renamed_actions_not_in_global_UNEXPECTED:     {missing_actions_unexpected}")
    print(f"  renamed_targets_unique_total:                 {len(target_unique)}")
    print(f"  renamed_targets_unique_in_global:             {unique_in_global}")
    print(f"  renamed_targets_unique_not_in_global_expected:{unique_not_in_global_expected}")
    print(f"  renamed_targets_unique_not_in_global_UNEXPECTED:{unique_not_in_global_unexpected}")

    # Removal audit: confirm all selected nodes were removed
    # Which removal-set nodes existed before processing, and which remain afterward
    deleted_seen_in_local = sorted(intended_remove & pre_nodes_all)
    deleted_still_present_after = sorted(intended_remove & post_nodes_all)  # Should be empty

    if len(deleted_still_present_after) == 0:
        print(
            f"\n Confirmation: ALL {len(intended_remove)} nodes in the removal set are absent from the output local graphs (KEEP_NODES exceptions were retained)"
        )
        if len(intended_remove) <= 10:
            print("   Removed nodes:", ", ".join(sorted(intended_remove)))
    else:
        print(
            f"\n Nodes from the removal set still present after cleaning: "
            f"{len(deleted_still_present_after)} entries."
        )
        for name in deleted_still_present_after[:20]:
            print("   •", name)
        if len(deleted_still_present_after) > 20:
            print("   … and more:", len(deleted_still_present_after) - 20)

    # Final CSV summary
    summary = {
        "renamed_actions_total": len(renamed_pairs_all),
        "renamed_actions_in_global": present_actions,
        "renamed_actions_not_in_global_expected": missing_actions_expected,
        "renamed_actions_not_in_global_UNEXPECTED": missing_actions_unexpected,
        "renamed_targets_unique_total": len(target_unique),
        "renamed_targets_unique_in_global": unique_in_global,
        "renamed_targets_unique_not_in_global_expected": unique_not_in_global_expected,
        "renamed_targets_unique_not_in_global_UNEXPECTED": unique_not_in_global_unexpected,
        "deleted_seen_in_local_before": len(deleted_seen_in_local),
        "deleted_still_present_after": len(deleted_still_present_after),
        "global_graph_path": args.global_graph,
    }
    summary_path = os.path.join(args.output_dir, "_renamed_vs_global_summary.csv")
    pd.Series(summary).to_csv(summary_path)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
