"""
Global_graph_generation_without_synthesisID.py
Build the global synthesis graph from merged_all.xlsx, add frequencies, compute embeddings, merge similar nodes, remove small components, and save artifacts (CSV/PKL/graphs/HTML).
"""

import os
import re
import sys
import json
import ast
import hashlib
import pickle

import numpy as np
import pandas as pd
import networkx as nx

# Optional deps: keep the pipeline running even if visualization libs are missing
try:
    import seaborn as sns
    sns.set(style="whitegrid", font_scale=1.1)
except Exception:
    sns = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from pyvis.network import Network
except Exception:
    Network = None

try:
    from networkx.drawing.nx_agraph import write_dot
except Exception:
    write_dot = None

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = AutoModel = None

# Run everything from this folder (no sys.path hacking)
import graph_tools



# Thresholds and settings from a single source (graph_tools)
DEFAULT_SIM_THRESHOLD = graph_tools.DEFAULT_SIMILARITY_THRESHOLD
NUMERIC_THRESHOLD = graph_tools.NUMERIC_SIMILARITY_THRESHOLD
PDFNO_THRESHOLD = graph_tools.PDFNO_SIMILARITY_THRESHOLD
THRESHOLDS_BY_LABEL = graph_tools.THRESHOLDS_BY_LABEL.copy()

# Seed for reproducibility
RANDOM_SEED = int(os.environ.get("SEED", "42"))

# Merge strategy: 'id' (label:value) vs 'name' (value only)
MERGE_BY = os.environ.get("MERGE_BY", "name").lower()  # valid: 'id' or 'name'
if MERGE_BY not in {"id", "name"}:
    MERGE_BY = "id"
print(f"[MERGE] strategy = {MERGE_BY} — id: full 'label:value'; name: value of the node's 'name' attribute")


def _get(row, *candidates):
    for c in candidates:
        if c is None:
            continue
        val = row.get(c)
        if isinstance(val, str):
            s = val.strip()
            if s and s.lower() != 'unknown':
                return s
        elif pd.notna(val) and val != -999:
            return val
    return None

# Paths and output directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
GRAPH_DIR = os.environ.get('SYNTHWAY_GRAPH_OUTPUT', os.path.join(REPO_DIR, 'outputs', 'graph'))
GRAPH_OUTPUT_DIR = GRAPH_DIR
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)
print(f"[RUN] script={os.path.abspath(__file__)}, pid={os.getpid()}, out_dir={GRAPH_OUTPUT_DIR}")


DO_NOT_MERGE_FILE = os.path.join(BASE_DIR, "do_not_merge_pairs.txt")
if not os.path.isfile(DO_NOT_MERGE_FILE):
    raise FileNotFoundError(f"Required graph merge exclusions are missing: {DO_NOT_MERGE_FILE}")
DO_NOT_MERGE_PAIRS = graph_tools.load_do_not_merge_pairs(DO_NOT_MERGE_FILE)
if not DO_NOT_MERGE_PAIRS:
    raise ValueError(f"No valid graph merge exclusions found in {DO_NOT_MERGE_FILE}")

print(f"[DONT-MERGE-PAIRS] loaded pairs: {len(DO_NOT_MERGE_PAIRS)}")


MERGED_XLSX_PATH = os.path.join(REPO_DIR, 'data', 'merged_all.xlsx')
# Expected key Excel fields:
#   synthesis_id, composition,
#   precursor / additive / seeds (+ masses/concentrations),
#   solvent (+solvent_v), temperature, ph, method,
#   shape, crystalsystem, size_*,
#   pdfno_list


# Paths to count JSON files (output of count_local_graphs.py)
COUNTS_DIR = os.path.join(GRAPH_DIR, 'analytics', 'local_counts')
NODE_COUNTS_FILE = os.path.join(COUNTS_DIR, 'node_counts.json')
EDGE_COUNTS_FILE = os.path.join(COUNTS_DIR, 'edge_counts.json')

if not os.path.exists(NODE_COUNTS_FILE) or not os.path.exists(EDGE_COUNTS_FILE):
    raise FileNotFoundError(
        "Counts files not found. Run count_local_graphs.py first. Expected: "
        f"{NODE_COUNTS_FILE} and {EDGE_COUNTS_FILE}"
    )

# Load precomputed node and edge frequencies from local graphs:
# node_counts: {node: count}; edge_counts_raw: {"{u}>>> {v}": count}
with open(NODE_COUNTS_FILE, 'r', encoding='utf-8') as f:
    node_counts = json.load(f)
with open(EDGE_COUNTS_FILE, 'r', encoding='utf-8') as f:
    edge_counts_raw = json.load(f)

# Step 1. Load data from the Excel table
# Next: group by synthesis_id → build local graphs → aggregate into a global DiGraph
df = pd.read_excel(MERGED_XLSX_PATH)
# Step 2. Group by synthesis_id
grouped = df.groupby("synthesis_id")

# Each group represents one synthesis (a local mini-graph)
# Step 3. Initialize the directed global graph
G = nx.DiGraph()

# Field names and associated parameter columns
FIELD_PARAMS = {
    "precursor": ["precursor_mass", "precursor_conc"],
    "additive": ["additive_mass", "additive_conc"],
    "seeds": ["seed_mass", "seed_conc"],
    "solvent": ["solvent_v"],  # Solvent volume
}

# Iterate over syntheses and add nodes/edges to the global graph
for synthesis_id, group in grouped:
    for _, row in group.iterrows():
        # Composition is the central synthesis node; skip empty/unknown values
        comp_raw = row.get("composition")
        if not isinstance(comp_raw, str) or comp_raw.strip().lower() == 'unknown':
            continue
        comp_clean = comp_raw.strip()
        comp_id = f"composition:{comp_clean}"
        G.add_node(comp_id, label="composition")

        for field, param_keys in FIELD_PARAMS.items():
            val = _get(row, field)
            if isinstance(val, str):
                node_id = f"{field}:{val}"
                G.add_node(node_id, label=field)
                G.add_edge(node_id, comp_id, synthesis=synthesis_id)

                for param in param_keys:
                    pval = _get(row, param)
                    if pval is not None:
                        param_node = f"{param}:{pval}"
                        G.add_node(param_node, label=param)
                        G.add_edge(node_id, param_node, synthesis=synthesis_id)

        # Additional process conditions: temperature, pH, method
        temp_val = _get(row, "temperature")
        if temp_val is not None:
            n = f"temperature:{temp_val}"
            G.add_node(n, label="temperature")
            G.add_edge(n, comp_id, synthesis=synthesis_id)

        ph_val = _get(row, "ph")
        if ph_val is not None:
            n = f"ph:{ph_val}"
            G.add_node(n, label="ph")
            G.add_edge(n, comp_id, synthesis=synthesis_id)

        method_val = _get(row, "method")
        if method_val is not None:
            n = f"method:{method_val}"
            G.add_node(n, label="method")
            G.add_edge(n, comp_id, synthesis=synthesis_id)

        # Product properties (shape/crystal system): direct edges from comp_id to each property
        for field in ["shape", "crystalsystem"]:
            val = _get(row, field)
            if isinstance(val, str):
                node_id = f"{field}:{val}"
                G.add_node(node_id, label=field)
                G.add_edge(comp_id, node_id, synthesis=synthesis_id)

        # Particle dimensions are separate nodes of the form 'size_*:value'
        for size_field in ["size_length", "size_width", "size_depth"]:
            val = _get(row, size_field)
            if val is not None:
                node = f"{size_field}:{val}"
                G.add_node(node, label=size_field)
                G.add_edge(comp_id, node, synthesis=synthesis_id)

        # PDF: 'pdfno_list' may contain a string ("05-0661") or a serialized list (e.g., '["05-0661", "01-2345"]')
        pdf_values = _get(row, "pdfno_list")
        if isinstance(pdf_values, str):
            try:
                pdf_list = ast.literal_eval(pdf_values) if pdf_values.strip().startswith("[") else [pdf_values]
                for pdf in pdf_list:
                    pdfs = str(pdf).strip()
                    if pdfs:
                        node_pdf = f"pdfno:{pdfs}"
                        G.add_node(node_pdf, label="pdfno")
                        G.add_edge(comp_id, node_pdf, synthesis=synthesis_id)
            except Exception as e:
                print(f"Error reading the PDF list for {synthesis_id}: {e}")

# Brief graph size summary before simplification
print(f"Before simplification: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Step 4. Assign weights based on local graphs
# 1. Node frequencies
 # Assign node/edge frequencies from the loaded statistics
nx.set_node_attributes(G, node_counts, name="freq")

# 2. Edge frequencies
edge_freq = {}
for u, v in G.edges():
    key = f"{u}>>> {v}"
    edge_freq[(u, v)] = edge_counts_raw.get(key, 0)
nx.set_edge_attributes(G, edge_freq, name="freq")

# 3. Probability normalization (nodes: sum=1; edges: p_cond=P(v|u) with Laplace smoothing)
graph_tools.normalize_graph_frequencies(G, alpha=1.0)
print("'freq' attributes assigned; 'freq_norm' and 'p_cond' computed as probabilities")
# Assign the 'name' attribute to every node in G
graph_tools.attach_name_attr(G)


# Step 5. Load the tokenizer/embedding model (BGE-Large); generate_node_embeddings() then computes the vectors
tokenizer_model = "BAAI/bge-large-en-v1.5"
if AutoTokenizer is None or AutoModel is None:
    raise ImportError("transformers is required for embeddings. Install transformers (and torch) or run a version that skips embeddings.")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
model = AutoModel.from_pretrained(tokenizer_model)
model.eval()


# Step 6. Embeddings for merging (in graph_tools)
# Merge ONLY within the same label; use THRESHOLDS_BY_LABEL or fall back to DEFAULT_SIM_THRESHOLD.
# merge_by='id' embeds 'label:value'; merge_by='name' embeds only 'value'
# For GUARDED_LABELS, add [HYDNUM=…] to the embedding text to prevent merging different hydrates of the same salt based on similar names
GUARDED_LABELS = {"composition", "precursor", "additive", "seeds"}

# Build embeddings for all graph nodes (some are frozen later); add [HYDNUM=…] for GUARDED_LABELS.
embeddings_for_merge = graph_tools.build_merge_embeddings(
    G, tokenizer, model,
    merge_by=MERGE_BY,
    add_hydnum_labels=GUARDED_LABELS  # Add [HYDNUM=…] to the embedding text for these labels
)

# Make a copy before filtering
embeddings_for_all = dict(embeddings_for_merge)

# Pairs that must not be merged:
if DO_NOT_MERGE_PAIRS:
    # Collect all nodes appearing in at least one prohibited pair
    freeze_nodes = set()
    for p in DO_NOT_MERGE_PAIRS:
        for node in p:
            if node in G.nodes:
                freeze_nodes.add(node)
    if freeze_nodes:
        before = len(embeddings_for_merge)
        # Exclude frozen nodes from the embeddings used for merging
        embeddings_for_merge = {k: v for k, v in embeddings_for_merge.items() if k not in freeze_nodes}
        print(f"[DONT-MERGE-PAIRS] loaded {len(DO_NOT_MERGE_PAIRS)} pairs; frozen {len(freeze_nodes)} nodes (embeds {before}->{len(embeddings_for_merge)}).")
else:
    print("[DONT-MERGE-PAIRS] no pairs loaded")

# Step 7. Merge similar nodes
# Rules:
# Merge ONLY within the same label; use THRESHOLDS_BY_LABEL or DEFAULT_SIM_THRESHOLD
# GUARDED_LABELS (composition/precursor/additive/seeds): account for hydration by grouping by [HYDNUM]
# DO_NOT_MERGE_PAIRS: freeze nodes in prohibited pairs
G_simple, updated_embeddings = graph_tools.simplify_graph(
    G,
    embeddings_for_merge,
    tokenizer,
    model,
    similarity_threshold=DEFAULT_SIM_THRESHOLD,
    thresholds_by_label=THRESHOLDS_BY_LABEL,
    guarded_labels=GUARDED_LABELS,
    guard_policy="relaxed",                 # Do not merge different HYDNUM values; unspecified values share a bucket
    bucketize_by_hydrate=True,               # Group GUARDED_LABELS by hydration number
    do_not_merge_pairs=DO_NOT_MERGE_PAIRS,   # Explicitly prohibited pairs
    data_dir_output=GRAPH_OUTPUT_DIR,
    graph_root="cleaned_graph",
    verbatim=False,
)
print(f" After merging: {G_simple.number_of_nodes()} nodes, {G_simple.number_of_edges()} edges")

# Readable names for simplified graph nodes (name = right-hand part of 'label:value')
graph_tools.attach_name_attr(G_simple)

# Value embeddings (without the hydration tag)
name_embeddings_simple = graph_tools.build_merge_embeddings(
    G_simple, tokenizer, model,
    merge_by="name",
    add_hydnum_labels=frozenset(),
)

# MERGE ANALYSIS
# Node lists before/after merging; removed nodes are candidates for mapping to representatives
_orig_nodes = list(G.nodes())
_kept_nodes = list(G_simple.nodes())
_removed_after_simplify = [n for n in _orig_nodes if n not in _kept_nodes]

merge_rows = []  # Rows for merges_detailed.csv

if len(_removed_after_simplify) > 0 and len(_kept_nodes) > 0:
    # Prepare embeddings and cosine similarities
    def _to_row_vec(x):
        """Reshape a vector to (D,), or return None if this fails."""
        a = np.asarray(x)
        a = np.squeeze(a)
        if a.ndim == 0:
            return None
        return a.reshape(-1)

    def _stack_embeddings(edict, keys):
        """Build a matrix of equal-length vectors from an embedding dictionary and a list of keys."""
        vecs, good_keys, dims = [], [], []
        for k in keys:
            v = _to_row_vec(edict.get(k))
            if v is None:
                continue
            vecs.append(v)
            good_keys.append(k)
            dims.append(len(v))
        if not vecs:
            return None, []
        # Choose the mode (most frequent dimension) as the target dimension
        from collections import Counter
        target_dim = Counter(dims).most_common(1)[0][0]
        vecs2 = [v for v in vecs if len(v) == target_dim]
        keys2 = [k for k, d in zip(good_keys, dims) if d == target_dim]
        if len(vecs2) < len(vecs):
            print(f"[merge-analytics] Filtered out {len(vecs)-len(vecs2)} nodes with mismatched dimensions (target_dim={target_dim}).")
        return np.stack(vecs2), keys2

    nodes_removed_eff = [n for n in _removed_after_simplify if n in embeddings_for_merge]
    nodes_kept_eff    = [n for n in _kept_nodes if n in embeddings_for_merge]

    E_removed, nodes_removed_eff = _stack_embeddings(embeddings_for_merge, nodes_removed_eff)
    E_kept,    nodes_kept_eff    = _stack_embeddings(embeddings_for_merge, nodes_kept_eff)

    if E_removed is None or E_kept is None or E_removed.shape[1] != E_kept.shape[1]:
        print("[merge-analytics] Skipping similarity calculations: embeddings are missing or dimensions do not match.")
    else:
        # Normalize rows and calculate cosine similarities
        def _row_norm(a):
            return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

        S = _row_norm(E_removed) @ _row_norm(E_kept).T

        # Label-matching mask (prevent mappings between different node types)
        labels_removed = [G.nodes[n].get("label") for n in nodes_removed_eff]
        labels_kept    = [G_simple.nodes[n].get("label") for n in nodes_kept_eff]
        mask_same_label = np.array([[lr == lk for lk in labels_kept] for lr in labels_removed], dtype=bool)
        S = np.where(mask_same_label, S, -1.0)

        # Additional mask for GUARDED_LABELS: allow only matching hydration buckets
        def _node_text(Gx, n):
            try:
                return (Gx.nodes[n].get("name") or graph_tools.extract_value_from_node_id(n))
            except Exception:
                return graph_tools.extract_value_from_node_id(n)

        keys_removed = [
            graph_tools.guard_bucket_key_from_name(_node_text(G, n)) if (lr in GUARDED_LABELS) else ""
            for n, lr in zip(nodes_removed_eff, labels_removed)
        ]
        keys_kept = [
            graph_tools.guard_bucket_key_from_name(_node_text(G_simple, n)) if (lk in GUARDED_LABELS) else ""
            for n, lk in zip(nodes_kept_eff, labels_kept)
        ]
        mask_bucket = np.array([
            [ (lr not in GUARDED_LABELS) or (keys_removed[i] == keys_kept[j]) for j, _ in enumerate(labels_kept) ]
            for i, lr in enumerate(labels_removed)
        ], dtype=bool)
        S = np.where(mask_bucket, S, -1.0)

        # Best representative passing the label-specific threshold
        best_idx = S.argmax(axis=1)
        best_val = S.max(axis=1)
        for i, n_removed in enumerate(nodes_removed_eff):
            lr = G.nodes[n_removed].get("label")
            thr = THRESHOLDS_BY_LABEL.get(lr, DEFAULT_SIM_THRESHOLD)
            rep = nodes_kept_eff[int(best_idx[i])] if best_val[i] >= thr else None
            if rep is not None:
                merge_rows.append({
                    "removed_node": n_removed,
                    "representative": rep,
                    "similarity": float(best_val[i]),
                    "label_removed": lr,
                    "label_representative": G_simple.nodes[rep].get("label"),
                })
            else:
                merge_rows.append({
                    "removed_node": n_removed,
                    "representative": "",
                    "similarity": float(best_val[i]),
                    "label_removed": lr,
                    "label_representative": "",
                })

    # Save details and filter prohibited pairs
    df_merges = pd.DataFrame(merge_rows)
    try:
        if 'DO_NOT_MERGE_PAIRS' in globals() and DO_NOT_MERGE_PAIRS and not df_merges.empty:
            def _is_forbidden(a, b):
                if not a or not b:
                    return False
                return frozenset((str(a).strip(), str(b).strip())) in DO_NOT_MERGE_PAIRS
            bad_mask = df_merges.apply(lambda r: _is_forbidden(r.get("removed_node", ""), r.get("representative", "")), axis=1)
            bad_cnt = int(bad_mask.sum())
            if bad_cnt:
                df_merges = df_merges.loc[~bad_mask].reset_index(drop=True)
                print(f"[DONT-MERGE-PAIRS] Removed {bad_cnt} prohibited merges from the analysis")
    except Exception as e:
        print(f"[DONT-MERGE-PAIRS] Error filtering the analysis: {e}")

    df_merges.to_csv(os.path.join(GRAPH_OUTPUT_DIR, "merges_detailed.csv"), index=False)

    # Aggregated summary by representative
    agg = (
        df_merges[df_merges["representative"] != ""]
        .groupby(["representative", "label_representative"], as_index=False)
        .size().rename(columns={"size": "merged_count"})
        .sort_values("merged_count", ascending=False)
    )
    agg.to_csv(os.path.join(GRAPH_OUTPUT_DIR, "merges_aggregated.csv"), index=False)

    # Complete source -> target mapping for renaming local graphs
    def _canon_key(s):
        """Normalize a key: trim and collapse whitespace without changing case or the 'label:value' format."""
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        return s

    pairs_rows = []
    if not df_merges.empty:
        for _, r in df_merges.iterrows():
            src = r.get('removed_node', '')
            dst = r.get('representative', '') or r.get('removed_node', '')
            sim = float(r.get('similarity', np.nan)) if pd.notna(r.get('similarity', np.nan)) else np.nan
            pairs_rows.append({
                'source_node': src,
                'target_node': dst,
                'label_source': r.get('label_removed', ''),
                'label_target': r.get('label_representative', ''),
                'similarity': sim,
                'mapping_rule': 'merge' if dst != src else 'identity',
            })
    for rep in _kept_nodes:  # Identity mapping for all retained nodes
        pairs_rows.append({
            'source_node': rep,
            'target_node': rep,
            'label_source': G_simple.nodes[rep].get('label'),
            'label_target': G_simple.nodes[rep].get('label'),
            'similarity': 1.0,
            'mapping_rule': 'identity',
        })

    df_pairs = pd.DataFrame(pairs_rows).drop_duplicates(subset=['source_node', 'target_node'], keep='first').reset_index(drop=True)
    df_pairs['source_node_norm'] = df_pairs['source_node'].map(_canon_key)
    df_pairs['target_node_norm'] = df_pairs['target_node'].map(_canon_key)

    pairs_path = os.path.join(GRAPH_OUTPUT_DIR, 'merge_pairs_for_renaming.csv')
    df_pairs.to_csv(pairs_path, index=False)

    # Validate the pair mapping
    try:
        kept_set = set(_kept_nodes)
        sources_in_pairs = set(df_pairs['source_node'].tolist())
        targets_in_pairs = set(df_pairs['target_node'].tolist())
        orig_nodes_set = set(_orig_nodes)

        missing_sources = sorted(list(orig_nodes_set - sources_in_pairs))
        bad_target_rows = [(r['source_node'], r['target_node']) for _, r in df_pairs.iterrows() if r['source_node'] != r['target_node'] and r['target_node'] not in kept_set]
        dup_sources = (
            df_pairs['source_node']
            .value_counts()
            .loc[lambda s: s > 1]
            .index.tolist()
        )

        validation = {
            'merge_by': MERGE_BY,
            'total_original_nodes': len(orig_nodes_set),
            'total_pairs_rows': int(len(df_pairs)),
            'unique_sources_in_pairs': int(len(sources_in_pairs)),
            'missing_sources_count': int(len(missing_sources)),
            'missing_sources_sample': missing_sources[:20],
            'bad_targets_count': int(len(bad_target_rows)),
            'bad_targets_sample': bad_target_rows[:20],
            'duplicate_sources_count': int(len(dup_sources)),
            'duplicate_sources_sample': dup_sources[:20],
        }
        val_path = os.path.join(GRAPH_OUTPUT_DIR, 'merge_pairs_validation.json')
        with open(val_path, 'w', encoding='utf-8') as vf:
            json.dump(validation, vf, ensure_ascii=False, indent=2)
        print(
            "[pairs-check] ok: "
            f"sources {validation['unique_sources_in_pairs']}/{validation['total_original_nodes']}; "
            f"bad_targets={validation['bad_targets_count']}; "
            f"dup_sources={validation['duplicate_sources_count']}. "
            f"Details: {val_path}"
        )
    except Exception as e:
        print(f"[pairs-check] validation failed: {e}")

    # Representative mapping and binary files containing representative embeddings
    merged_groups = {}
    if not df_merges.empty:
        merged_groups = (
            df_merges[df_merges['representative'] != '']
            .groupby('representative')['removed_node']
            .apply(list)
            .to_dict()
        )

    rows_rep, rep_embeddings = [], {}
    for rep in _kept_nodes:
        emb = updated_embeddings.get(rep, None)
        emb_dim, emb_sha = 0, ''
        if emb is not None:
            a = np.asarray(emb).reshape(-1)
            emb_dim = int(a.shape[0])
            emb_sha = hashlib.sha1(a.tobytes()).hexdigest()
            rep_embeddings[rep] = a.astype(np.float32)  # compact
        rows_rep.append({
            'representative': rep,
            'label': G_simple.nodes[rep].get('label'),
            'merged_from_count': int(len(merged_groups.get(rep, []))),
            'merged_from_json': json.dumps(merged_groups.get(rep, []), ensure_ascii=False),
            'kept_after_cleaning': '',  # Populated after the cleaning step
            'embedding_dim': emb_dim,
            'embedding_sha1': emb_sha,
        })

    df_rep = pd.DataFrame(rows_rep).sort_values(['merged_from_count', 'representative'], ascending=[False, True])
    rep_map_path = os.path.join(GRAPH_OUTPUT_DIR, 'merge_map_by_rep.csv')
    df_rep.to_csv(rep_map_path, index=False)

    rep_emb_path = os.path.join(GRAPH_OUTPUT_DIR, 'representative_embeddings.pkl')
    with open(rep_emb_path, 'wb') as f:
        pickle.dump(rep_embeddings, f)

    # Name embeddings of representatives (shape=(1, D) for consistency)
    rep_name_emb_path = os.path.join(GRAPH_OUTPUT_DIR, 'representative_name_embeddings.pkl')
    rep_name_embeddings = {}
    for rep in _kept_nodes:
        embn = name_embeddings_simple.get(rep, None)
        if embn is not None:
            a = np.asarray(embn)
            if a.ndim == 1:
                a = a[None, :]
            else:
                a = np.squeeze(a)
                if a.ndim == 1:
                    a = a[None, :]
                elif a.ndim != 2:
                    a = a.reshape(1, -1)
            rep_name_embeddings[rep] = a.astype(np.float32)
    with open(rep_name_emb_path, 'wb') as f:
        pickle.dump(rep_name_embeddings, f)

    print(
        "[merge-map] Saved:",
        f"\n - {os.path.relpath(pairs_path)}",
        f"\n - {os.path.relpath(rep_map_path)}",
        f"\n - {os.path.relpath(rep_emb_path)}",
        f"\n - {os.path.relpath(rep_name_emb_path)}",
    )

    print(f"Merge summary saved to {GRAPH_OUTPUT_DIR}/merges_*.csv.")
else:
    print("No nodes/embeddings available for merge analysis; skipping.")

# Before export: convert list attributes to strings for GraphML/GEXF
graph_tools.convert_list_attrs_to_strings(G_simple)

# Step 8. Remove small components and synchronize embeddings (returns the cleaned graph and updated vectors)
G_cleaned, updated_embeddings = graph_tools.remove_small_fragments_and_clean_embeddings(G_simple, updated_embeddings, size_threshold=10)

print(f"After cleaning: {G_cleaned.number_of_nodes()} nodes, {G_cleaned.number_of_edges()} edges")

graph_tools.attach_name_attr(G_cleaned)

# SMALL-COMPONENT REMOVAL ANALYSIS
nodes_simple = set(G_simple.nodes())
nodes_cleaned = set(G_cleaned.nodes())
removed_clean = nodes_simple - nodes_cleaned
print(f"Nodes removed during small-component cleaning: {len(removed_clean)}")

if removed_clean:
    # Find components in G_simple and select those that were removed
    comps = list(nx.connected_components(G_simple.to_undirected()))
    rows_comp = []
    removed_nodes_rows = []
    comp_id = 0
    for comp in comps:
        comp = set(comp)
        if comp.issubset(removed_clean):
            comp_id += 1
            labels = [G_simple.nodes[n].get("label") for n in comp]
            vc = pd.Series(labels, dtype="object").value_counts()
            label_counts = {str(k): int(v) for k, v in vc.items()}
            rows_comp.append({
                "component_id": comp_id,
                "size": len(comp),
                "label_counts_json": json.dumps(label_counts, ensure_ascii=False),
                "sample_nodes": "; ".join(list(comp)[:10])
            })
            for n in comp:
                removed_nodes_rows.append({
                    "component_id": comp_id,
                    "node": n,
                    "label": G_simple.nodes[n].get("label")
                })

    df_removed_comps = pd.DataFrame(rows_comp)
    df_removed_nodes = pd.DataFrame(removed_nodes_rows)

    df_removed_comps.to_csv(os.path.join(GRAPH_OUTPUT_DIR, "removed_components.csv"), index=False)
    df_removed_nodes.to_csv(os.path.join(GRAPH_OUTPUT_DIR, "removed_nodes_cleaning.csv"), index=False)

    # Plots
    if not df_removed_comps.empty:
        if plt is not None:
            plt.figure(figsize=(9, 4))
            plt.hist(df_removed_comps["size"], bins=30)
            plt.title("Sizes of removed components (after cleaning)")
            plt.xlabel("Number of nodes per component")
            plt.ylabel("Number of components")
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, "removed_components_size_hist.png"))
            plt.close()

    if not df_removed_nodes.empty:
        if plt is not None:
            counts_by_label = df_removed_nodes["label"].value_counts().sort_values(ascending=False).head(20)
            plt.figure(figsize=(10, 5))
            plt.barh(counts_by_label.index, counts_by_label.values)
            plt.gca().invert_yaxis()
            plt.title("Most frequent labels among removed nodes (cleaning)")
            plt.xlabel("Number of nodes")
            plt.tight_layout()
            plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, "removed_nodes_by_label.png"))
            plt.close()

    print(f"Cleaning analysis saved to {GRAPH_OUTPUT_DIR}/removed_*.csv and PNG files.")
else:
    print("No nodes were removed during cleaning or the list is empty; skipping cleaning analysis.")


# Update representative records: set kept_after_cleaning based on G_cleaned
try:
    rep_map_path = os.path.join(GRAPH_OUTPUT_DIR, 'merge_map_by_rep.csv')
    if os.path.exists(rep_map_path):
        df_rep = pd.read_csv(rep_map_path)
        kept_set = set(G_cleaned.nodes())
        df_rep['kept_after_cleaning'] = df_rep['representative'].map(lambda n: n in kept_set)
        df_rep.to_csv(rep_map_path, index=False)
        print(f"[merge-map] Updated kept_after_cleaning in {rep_map_path}")
except Exception as e:
    print(f"[merge-map] Failed to update kept_after_cleaning: {e}")

# Check that every node has an embedding and no extra embeddings remain
graph_nodes = set(G_cleaned.nodes())
embedding_nodes = set(updated_embeddings.keys())

missing_embeddings = graph_nodes - embedding_nodes
extra_embeddings = embedding_nodes - graph_nodes

print(f"Nodes without embeddings: {len(missing_embeddings)}")
if len(missing_embeddings) > 0:
    print(f"Found {len(missing_embeddings)} nodes without embeddings. Computing missing embeddings...")
    new_embeds = graph_tools.regenerate_node_embeddings(G_cleaned, missing_embeddings, tokenizer, model)
    updated_embeddings.update(new_embeds)
    print(f"Added embeddings for {len(missing_embeddings)} nodes.")
if len(extra_embeddings) > 0:
    print(f"ℹRemoving {len(extra_embeddings)} extra embeddings not represented in the graph.")
    for node in extra_embeddings:
        updated_embeddings.pop(node, None)

# Save the final cleaned-graph embeddings
with open(os.path.join(GRAPH_OUTPUT_DIR, "cleaned_embeddings.pkl"), "wb") as f:
    pickle.dump(updated_embeddings, f)
# Save value-only embeddings for the cleaned graph
name_embeddings_cleaned = graph_tools.build_merge_embeddings(G_cleaned, tokenizer, model, merge_by="name", add_hydnum_labels=frozenset())
with open(os.path.join(GRAPH_OUTPUT_DIR, "cleaned_name_embeddings.pkl"), "wb") as f:
    pickle.dump(name_embeddings_cleaned, f)


graph_tools.convert_list_attrs_to_strings(G_cleaned)

# Normalize probabilities again on the cleaned graph
graph_tools.normalize_graph_frequencies(G_cleaned, alpha=1.0)


# Step 9. Export the graph in multiple formats (GraphML/GEXF/DOT)
output_dir = GRAPH_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

nx.write_graphml(G_cleaned, os.path.join(output_dir, "graph.graphml"))

print(f"\n Graph built. Number of nodes after cleaning: {len(G_cleaned.nodes)}")
print(f" Graph saved to: {os.path.join(GRAPH_OUTPUT_DIR, 'cleaned_synthesis_graph.gexf')}")
print(f" Graph visualization saved to: {os.path.join(GRAPH_OUTPUT_DIR, 'cleaned_graph_visualization.svg')}")

# GEXF (already present)
nx.write_gexf(G_cleaned, os.path.join(output_dir, "cleaned_synthesis_graph.gexf"))

# DOT (Graphviz)
if write_dot is not None:
    write_dot(G_cleaned, os.path.join(output_dir, "cleaned_graph.dot"))
else:
    print("[WARN] pygraphviz is not installed; skipping DOT export")

# Step 10. Visualize the cleaned graph and save HTML (pyvis)
if Network is not None:
    graph_tools.build_pyvis_interactive_louvain(
        G_cleaned,
        outfile=os.path.join(GRAPH_OUTPUT_DIR, "cleaned_graph_louvain_interactive.html"),
        norm_attr="freq_norm",
        edge_norm_attr="p_cond",
        size_min=10.0,
        size_max=40.0,
    )
else:
    print("[WARN] pyvis is not installed; skipping interactive HTML export")