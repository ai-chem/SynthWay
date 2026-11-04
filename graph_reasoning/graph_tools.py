"""
graph_tools.py — utilities for building and simplifying the graph
 - Parse chemical notation (hydrates) and prepare node text
 - Compute node embeddings
 - Merge similar nodes (cosine similarity + domain-specific checks) and rebuild edges
 - Normalize frequencies and export results

Usage:
1) Build the graph (Global_graph_generation_without_synthesisID.py)
2) Compute embeddings: build_merge_embeddings()
3) Simplify the graph using rules: simplify_graph()
4) Save GraphML/metrics: save_graph(), analyze_directed_network()
"""
import os
import re
import pickle
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings("ignore")

def load_do_not_merge_pairs(path_or_dir: str):
    """
    Load node pairs that must not be merged.
    """
    path = path_or_dir
    if os.path.isdir(path):
        path = os.path.join(path, "do_not_merge_pairs.txt")

    pairs = set()
    if not os.path.exists(path):
        return pairs

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Use exactly one TAB as the separator
            if line.count("\t") != 1:
                continue
            a, b = line.split("\t", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                pairs.add(frozenset((a, b)))
    return pairs


# Regular expressions for hydrates:
_HYD_PAT = re.compile(r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)\s*H\s*2\s*O", re.IGNORECASE) # Find the number before H2O in the text
_HYD_SUFFIX_RE = re.compile(r"[·\*\u2219\u22C5\.\u00B7]\s*\d+(?:\.\d+)?\s*H\s*2\s*O$", re.IGNORECASE) # Match a trailing suffix such as "·6H2O" or "*2.5H2O"
NUMERIC_LABEL_PAT = re.compile(r".*_(mass|conc)$", re.IGNORECASE) # Recognize numeric labels such as *_mass and *_conc

# Extract the value from a "label:value" node ID. Example: "composition:FeCl3·6H2O" -> "FeCl3·6H2O"
def extract_value_from_node_id(node_id: Union[str, int, float]) -> str:
    try:
        if isinstance(node_id, str) and ":" in node_id:
            return node_id.split(":", 1)[1].strip()
        return str(node_id)
    except Exception:
        return str(node_id)


# Ensure all nodes have a name attribute; if absent, use the part after the colon in the node ID
def attach_name_attr(GX: nx.Graph) -> None:
    name_map = {n: (d.get("name") if d.get("name") else extract_value_from_node_id(n))
                for n, d in GX.nodes(data=True)}
    nx.set_node_attributes(GX, name_map, name="name")


# Find the hydration number before H2O (e.g., "FeCl3·6H2O" -> "6"). Return an empty string if no number is present
def hydrate_key(text: str) -> str:
    s = str(text).replace("\u2009", " ").replace("\u202F", " ").replace(" ", " ").replace(",", ".")
    m = _HYD_PAT.search(s)
    return m.group(1).strip() if m else ""


# Remove a trailing suffix such as "·6H2O", leaving the base without hydrate notation
def strip_hydrate_suffix(text: str) -> str:
    s = str(text).replace("\u2009", " ").replace("\u202F", " ").replace(" ", " ").replace(",", ".")
    return _HYD_SUFFIX_RE.sub("", s).strip()


# Generate a key for grouping nodes by hydration
def guard_bucket_key_from_name(name: str, policy: str = "relaxed") -> str:
    """Hydration grouping key:
  relaxed — ignore the base; group only by hydration number (or its absence)
  strict  — use both the base (without ·nH2O) and the hydration number"""
    hyd = hydrate_key(name) or ""
    if policy.lower() == "strict":
        base = re.sub(r"\s+", " ", strip_hydrate_suffix(name)).strip()
        return f"{base}||HYD={hyd}"
    return f"HYDONLY={hyd}" if hyd else "NOHYD"


def _node_name(G: nx.Graph, n) -> str:
    return (G.nodes[n].get("name") or extract_value_from_node_id(n))



# Embeddings: compute, save, load:

def _hf_embed(text: str, tokenizer, model) -> np.ndarray:
    if torch is None:
        raise RuntimeError("Torch is not available")
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(str(text), return_tensors="pt")
        outputs = model(**inputs)
        # mean-pool over sequence dimension -> shape (hidden,)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().numpy()
        return emb



# Compute embeddings for merging:
def build_merge_embeddings(
    GX: nx.Graph,
    tokenizer,
    model,
    merge_by: str = "name",
    add_hydnum_labels: Set[str] = frozenset({"composition", "precursor", "additive", "seeds"}),
) -> Dict[Union[str, int], np.ndarray]:
    """Embeddings for merging:
    - merge_by='name': compare values only ('name' attribute); otherwise, 'id': full 'label:value'
    - for labels in add_hydnum_labels, append '[HYDNUM=...]' if a number before H2O is found
    """
    attach_name_attr(GX)
    embs: Dict[Union[str, int], np.ndarray] = {}
    for n, data in GX.nodes(data=True):
        base_text = _node_name(GX, n) if merge_by == "name" else str(n)
        if data.get("label") in (add_hydnum_labels or set()):
            hk = hydrate_key(base_text)
            if hk:
                base_text = f"{base_text} [HYDNUM={hk}]"
        embs[n] = _hf_embed(base_text, tokenizer, model)
    return embs


# Legacy version: compute embeddings from the full node ID string
#def generate_node_embeddings(graph: nx.Graph, tokenizer, model) -> Dict[Union[str, int], np.ndarray]:
#    embeddings: Dict[Union[str, int], np.ndarray] = {}
#    for node in tqdm(graph.nodes()):
#        embeddings[node] = _hf_embed(str(node), tokenizer, model)
#    return embeddings


# Save the {node_id -> embedding} dictionary to pickle. Create the directory if needed
def save_embeddings(embeddings: Dict[Union[str, int], np.ndarray], file_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)


# Load an embedding dictionary from pickle
def load_embeddings(file_path: str) -> Dict[Union[str, int], np.ndarray]:
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


# Similarity thresholds: default 0.95; 0.99 for pdfno
NUMERIC_LABELS_STRICT: Set[str] = {
    "precursor_mass", "precursor_conc",
    "additive_mass", "additive_conc",
    "ph", "seed_mass", "seed_conc",
    "size_length", "size_width", "size_depth",
    "temperature", "solvent_v",
}
DEFAULT_SIMILARITY_THRESHOLD = float(os.environ.get("SIM_THR_DEFAULT", "0.95"))
NUMERIC_SIMILARITY_THRESHOLD = float(os.environ.get("SIM_THR_NUMERIC", "0.95"))
PDFNO_SIMILARITY_THRESHOLD = float(os.environ.get("SIM_THR_PDFNO", "0.99"))


def build_thresholds_by_label() -> Dict[str, float]:
    d = {lbl: NUMERIC_SIMILARITY_THRESHOLD for lbl in NUMERIC_LABELS_STRICT}
    d["pdfno"] = PDFNO_SIMILARITY_THRESHOLD
    return d


THRESHOLDS_BY_LABEL: Dict[str, float] = build_thresholds_by_label()


def _threshold_for_label(lbl: Optional[str]) -> float:
    """Return the threshold for the specified label, or the default threshold if none is defined."""
    if isinstance(lbl, str) and lbl in THRESHOLDS_BY_LABEL:
        return THRESHOLDS_BY_LABEL[lbl]
    return DEFAULT_SIMILARITY_THRESHOLD


def _is_numeric_label(lbl: Optional[str]) -> bool:
    return bool(NUMERIC_LABEL_PAT.fullmatch(lbl or "")) or (lbl in NUMERIC_LABELS_STRICT)


# Numeric safeguards for merging:
def _parse_numeric_scalar(name: Union[str, int, float]) -> Optional[float]:
    """Extract a scalar from a node name of the form 'label:value'.
    """
    if not isinstance(name, str):
        name = str(name)
    if ":" not in name:
        return None
    val = name.split(":", 1)[1].strip()
    # Normalization
    s = (val.replace(",", ".")
             .replace("\u2212", "-")  # minus
             .replace("\u2013", "-")  # en dash
             .replace("\u00A0", " "))  # nbsp
    # Convert "×10^k", "x10^k", "*10^k" forms to e{k}
    s = re.sub(r'([0-9.]+)\s*[x×*]\s*10\s*\^\s*([+-]?\d+)', r'\1e\2', s, flags=re.IGNORECASE)
    s = re.sub(r'([0-9.]+)\s*[x×*]\s*10\s*([+-]?\d+)', r'\1e\2', s, flags=re.IGNORECASE)
    # Extract the first valid number
    m = re.search(r'([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# Block numeric-node merges if parsing fails, |x−y| exceeds the atol/rtol tolerance, or orders of magnitude differ too much
def _should_block_numeric_merge(u: str, v: str, label: Optional[str],
                                atol: float = 1e-6, rtol: float = 0.05, log_gap: float = 0.15) -> bool:
    if not _is_numeric_label(label):
        return False
    xu = _parse_numeric_scalar(u)
    xv = _parse_numeric_scalar(v)
    # Reject the merge if parsing fails
    if xu is None or xv is None:
        return True
    # Absolute/relative difference
    rel_tol = rtol * max(abs(xu), abs(xv))
    if abs(xu - xv) > max(atol, rel_tol):
        return True
    # Difference in orders of magnitude
    try:
        lgap = abs(np.log10(max(abs(xu), 1e-12)) - np.log10(max(abs(xv), 1e-12)))
        if lgap > log_gap:
            return True
    except Exception:
        return True
    return False


# Export the graph to GraphML
def save_graph(G: nx.Graph, graph_GraphML: Optional[str] = None) -> None:
    if graph_GraphML is None:
        print("Error: no file path specified")
        return
    os.makedirs(os.path.dirname(os.path.abspath(graph_GraphML)), exist_ok=True)
    nx.write_graphml(G, graph_GraphML)
    print("GraphML saved")



# Remove small components (< size_threshold) and discard embeddings for removed nodes
def remove_small_fragments_and_clean_embeddings(
    G: Union[nx.Graph, nx.DiGraph],
    embeddings: Dict[Union[str, int], np.ndarray],
    size_threshold: int = 10,
) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[Union[str, int], np.ndarray]]:
    G_new = G.copy()
    if isinstance(G_new, nx.DiGraph):
        components = list(nx.weakly_connected_components(G_new))
    else:
        components = list(nx.connected_components(G_new))
    for component in components:
        if len(component) < size_threshold:
            G_new.remove_nodes_from(component)
    embeddings_cleaned = {n: e for n, e in embeddings.items() if n in G_new.nodes}
    return G_new, embeddings_cleaned


# Recompute embeddings only for selected nodes (representatives after merging)
def regenerate_node_embeddings(
    graph: nx.Graph,
    nodes_to_recalculate: Iterable[Union[str, int]],
    tokenizer,
    model,
) -> Dict[Union[str, int], np.ndarray]:
    new_embeddings: Dict[Union[str, int], np.ndarray] = {}
    for node in tqdm(list(nodes_to_recalculate)):
        new_embeddings[node] = _hf_embed(str(node), tokenizer, model)
    return new_embeddings



# Frequency normalization:
# Nodes: freq as a fraction of the sum across all nodes (node_attr_out)
# Edges: Laplace-smoothed conditional probability p(v|u) (edge_attr_out), also stored in edge_attr_compat

def normalize_graph_frequencies(
    G: Union[nx.Graph, nx.DiGraph],
    alpha: float = 1.0,
    edge_attr_out: str = "p_cond",
    edge_attr_compat: str = "freq_norm",
    node_attr_out: str = "freq_norm",
) -> None:
    # Nodes
    total_nodes = 0.0
    for _, d in G.nodes(data=True):
        total_nodes += float(d.get("freq", 0.0))
    if total_nodes <= 0:
        total_nodes = 1.0
    for _, d in G.nodes(data=True):
        f = float(d.get("freq", 0.0))
        d[node_attr_out] = f / total_nodes

    # Outgoing edge frequency sums
    out_sum: Dict[Union[str, int], float] = {}
    out_deg: Dict[Union[str, int], int] = {}
    for u, v, ed in G.edges(data=True):
        out_sum[u] = out_sum.get(u, 0.0) + float(ed.get("freq", 0.0))
        out_deg[u] = out_deg.get(u, 0) + 1

    # Conditional probabilities p(v|u)
    for u, v, ed in G.edges(data=True):
        num = float(ed.get("freq", 0.0)) + alpha
        den = out_sum.get(u, 0.0) + alpha * max(1, out_deg.get(u, 0))
        p = num / den if den > 0 else 0.0
        ed[edge_attr_out] = p
        ed[edge_attr_compat] = p



# Graph simplification:

def convert_list_attrs_to_strings(G: Union[nx.Graph, nx.DiGraph]) -> None:
    for _, data in G.nodes(data=True):
        for key, value in list(data.items()):
            if isinstance(value, (list, set, tuple)):
                data[key] = ", ".join(map(str, value))

    for _, _, data in G.edges(data=True):
        for key, value in list(data.items()):
            if isinstance(value, (list, set, tuple)):
                data[key] = ", ".join(map(str, value))


def simplify_graph(
    graph_: Union[nx.Graph, nx.DiGraph],
    node_embeddings: Dict[Union[str, int], np.ndarray],
    tokenizer,
    model,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    thresholds_by_label: Optional[Dict[str, float]] = None,
    data_dir_output: str = './',
    graph_root: str = 'simple_graph',
    verbatim: bool = False,
    guarded_labels: Optional[Set[str]] = None,
    guard_policy: str = "relaxed",
    bucketize_by_hydrate: bool = False,
    do_not_merge_pairs: Optional[Iterable[Union[Set[str], Tuple[str, str]]]] = None,
) -> Tuple[Union[nx.Graph, nx.DiGraph], Dict[Union[str, int], np.ndarray]]:
    """
    Merge nodes using cosine similarity of embeddings and domain-specific validity checks:
    1) Form groups of nodes to compare, optionally by hydration (relaxed/strict).
    2) Compute pairwise embedding similarities within each group and select pairs above the minimum threshold.
    3) Filter pairs:
    - labels must match; skip synthesis_step nodes
    - similarity ≥ the label-specific threshold
    - numeric guard: numeric values must not conflict
    - chemistry guard: hydration rules (relaxed/strict policy)
    4) Merge each pair, choosing the more connected node as the representative:
    - sum freq,
    - combine synthesis_ids,
    - collect unique merged_from entries,
    - redirect edges from drop to keep (sum freq for duplicates)
    5) After the pass, relabel node IDs, recompute embeddings only for representatives, normalize frequencies, and save the simplified graph as GraphML.
    """
    from collections import defaultdict

    graph = graph_.copy()
    attach_name_attr(graph)

    if thresholds_by_label is None:
        thresholds_by_label = THRESHOLDS_BY_LABEL
    if guarded_labels is None:
        guarded_labels = {"composition", "precursor", "additive", "seeds"}

    # Do-not-merge pairs
    DNM: Set[frozenset] = set()
    if do_not_merge_pairs:
        for p in do_not_merge_pairs:
            DNM.add(p if isinstance(p, frozenset) else frozenset(p))

    # 1) Form comparison groups (by hydration if enabled)
    nodes_all = [n for n in node_embeddings.keys() if n in graph.nodes]
    if bucketize_by_hydrate:
        buckets: Dict[Tuple[str, str], List] = defaultdict(list)
        for n in nodes_all:
            lbl = graph.nodes[n].get("label")
            if lbl in guarded_labels:
                key = guard_bucket_key_from_name(_node_name(graph, n), policy=guard_policy)
                buckets[(lbl, key)].append(n)
            else:
                buckets[("other", lbl)].append(n)
    else:
        buckets = {"__all__": nodes_all}

    node_mapping: Dict[Union[str, int], Union[str, int]] = {}
    nodes_to_recalc: Set[Union[str, int]] = set()
    merged_nodes: Set[Union[str, int]] = set()
    updated_embeddings: Dict[Union[str, int], np.ndarray] = dict(node_embeddings)
    # Log actual merges (drop -> keep) for a deterministic report
    merge_log: List[Dict[str, Union[str, float]]] = []

    if verbatim:
        total = sum(len(v) for v in buckets.values())
        print(f" simplify_graph: {total} nodes / {len(buckets)} bucket(s)")

    def _to_id_set(val) -> Set[str]:
        if val is None:
            return set()
        if isinstance(val, str):
            return {p.strip() for p in val.split(',') if p.strip()}
        if isinstance(val, (list, tuple, set)):
            return {str(x).strip() for x in val if str(x).strip()}
        return {str(val).strip()}

    for _, bucket_nodes in buckets.items():
        bucket_nodes = [n for n in bucket_nodes if n in updated_embeddings]
        if len(bucket_nodes) < 2:
            continue

        # 2) Compute the similarity matrix for the current group
        X = np.array([updated_embeddings[n].ravel() for n in bucket_nodes], dtype=np.float32)
        if X.ndim != 2 or X.shape[1] == 0:
            continue
        S = cosine_similarity(X)

        # Minimum threshold for candidate selection
        min_thr = min([similarity_threshold] + list(thresholds_by_label.values()))
        ii, jj = np.where(np.triu(S, k=1) > min_thr)
        order = np.argsort(S[ii, jj])[::-1]
        ii, jj = ii[order], jj[order]

        for i, j in zip(ii, jj):
            ni, nj = bucket_nodes[i], bucket_nodes[j]
            if ni in merged_nodes or nj in merged_nodes:
                continue
            if frozenset((ni, nj)) in DNM:
                if verbatim:
                    print(f"[DNM] {ni} ✕ {nj}")
                continue

            li = graph.nodes.get(ni, {}).get('label')
            lj = graph.nodes.get(nj, {}).get('label')

            # 3) Filter by label and similarity threshold
            # Skip synthesis_step
            if (isinstance(ni, str) and "synthesis_step:" in ni) or (isinstance(nj, str) and "synthesis_step:" in nj):
                continue
            # Different labels cannot be merged
            if li != lj or li is None:
                continue

            sim_ij = float(S[i, j])
            thr = thresholds_by_label.get(li, similarity_threshold)
            if sim_ij < thr:
                continue

            # 4) Domain-specific checks: numeric and chemical
            # Numeric check
            if _should_block_numeric_merge(str(ni), str(nj), li):
                if verbatim:
                    print(f"[NUMERIC-BLOCK] {ni} ✕ {nj}")
                continue

            # Chemical check (hydrates)
            if li in guarded_labels:
                name_i, name_j = _node_name(graph, ni), _node_name(graph, nj)
                hyd_i, hyd_j = hydrate_key(name_i), hydrate_key(name_j)
                if guard_policy == "strict":
                    base_i, base_j = strip_hydrate_suffix(name_i), strip_hydrate_suffix(name_j)
                    if (hyd_i and hyd_j and hyd_i != hyd_j) or (base_i != base_j):
                        if verbatim:
                            print(f"[CHEM-STRICT] {ni} ✕ {nj}")
                        continue
                else:  # relaxed
                    if hyd_i and hyd_j and hyd_i != hyd_j:
                        if verbatim:
                            print(f"[CHEM-RELAXED] {ni} ✕ {nj}")
                        continue

            # 5) Choose the representative node (higher degree)
            keep, drop = (ni, nj) if graph.degree(ni) >= graph.degree(nj) else (nj, ni)

            # Record the actual merge decision
            merge_log.append({
                "removed_node": str(drop),
                "representative": str(keep),
                "similarity": float(sim_ij),
                "label_removed": li,
                "label_representative": lj
            })

            # 6) Merge frequencies, synthesis_ids, and merged_from
            # Node frequencies
            graph.nodes[keep]["freq"] = graph.nodes[keep].get("freq", 0) + graph.nodes[drop].get("freq", 0)

            # Collect synthesis_ids from the edges of both nodes
            sids: Set[str] = set()
            for u, v, ed in list(graph.in_edges(keep, data=True)) + list(graph.out_edges(keep, data=True)):
                if 'synthesis' in ed:
                    sids.add(str(ed['synthesis']))
            for u, v, ed in list(graph.in_edges(drop, data=True)) + list(graph.out_edges(drop, data=True)):
                if 'synthesis' in ed:
                    sids.add(str(ed['synthesis']))
            ids = _to_id_set(graph.nodes[keep].get("synthesis_ids"))
            ids |= {s for s in sids if s}
            graph.nodes[keep]["synthesis_ids"] = ",".join(sorted(ids))

            # merged_from
            mfk = graph.nodes[keep].get('merged_from', [])
            if isinstance(mfk, str):
                mfk = [p.strip() for p in mfk.split(',') if p.strip()]
            mfd = graph.nodes[drop].get('merged_from', [])
            if isinstance(mfd, str):
                mfd = [p.strip() for p in mfd.split(',') if p.strip()]
            seen, mf_final = set(), []
            for x in list(mfk) + [drop] + list(mfd):
                sx = str(x).strip()
                if sx and sx not in seen:
                    seen.add(sx)
                    mf_final.append(sx)
            graph.nodes[keep]['merged_from'] = mf_final
            graph.nodes[keep]['label'] = li

            # 7) Redirect edges from drop to keep (summing freq)
            for pred, _, data in list(graph.in_edges(drop, data=True)):
                pred_new = node_mapping.get(pred, pred)
                if pred_new != keep:
                    if graph.has_edge(pred_new, keep):
                        graph[pred_new][keep]["freq"] = graph[pred_new][keep].get("freq", 0) + data.get("freq", 0)
                    else:
                        graph.add_edge(pred_new, keep, **data)
                if graph.has_edge(pred, drop):
                    graph.remove_edge(pred, drop)

            for _, succ, data in list(graph.out_edges(drop, data=True)):
                succ_new = node_mapping.get(succ, succ)
                if succ_new != keep:
                    if graph.has_edge(keep, succ_new):
                        graph[keep][succ_new]["freq"] = graph[keep][succ_new].get("freq", 0) + data.get("freq", 0)
                    else:
                        graph.add_edge(keep, succ_new, **data)
                if graph.has_edge(drop, succ):
                    graph.remove_edge(drop, succ)

            if graph.has_node(drop):
                graph.remove_node(drop)

            for old_node, new_node in list(node_mapping.items()):
                if new_node == drop:
                    node_mapping[old_node] = keep
            node_mapping[drop] = keep
            nodes_to_recalc.add(keep)
            merged_nodes.add(drop)

    # 8) Final relabeling, partial embedding recomputation, and normalization
    final_map: Dict[Union[str, int], Union[str, int]] = {}
    for old, new in node_mapping.items():
        rep = new
        while rep in node_mapping:
            rep = node_mapping[rep]
        final_map[old] = rep

    new_graph = nx.relabel_nodes(graph, final_map, copy=True) if final_map else graph

    recalculated = regenerate_node_embeddings(new_graph, nodes_to_recalc, tokenizer, model) if nodes_to_recalc else {}
    updated_embeddings.update(recalculated)
    for rem in merged_nodes:
        updated_embeddings.pop(rem, None)

    if verbatim:
        print("Nodes relabeled and embeddings (for reps) regenerated.")

    # Save a deterministic merge log
    try:
        import pandas as pd
        os.makedirs(os.path.abspath(data_dir_output), exist_ok=True)
        df_merges = pd.DataFrame(merge_log, columns=["removed_node","representative","similarity","label_removed","label_representative"])
        merges_path = os.path.join(os.path.abspath(data_dir_output), "merges_detailed.csv")
        df_merges.to_csv(merges_path, index=False)
        if not df_merges.empty:
            agg = (
                df_merges
                .groupby(["representative","label_representative"], as_index=False)
                .size()
                .rename(columns={"size":"merged_count"})
                .sort_values("merged_count", ascending=False)
            )
            agg.to_csv(os.path.join(os.path.abspath(data_dir_output), "merges_aggregated.csv"), index=False)
        print(f"merges_detailed.csv written: {merges_path} (rows={len(df_merges)})")
    except Exception as e:
        print(f"failed to write merges_detailed.csv: {e}")

    normalize_graph_frequencies(new_graph, alpha=1.0)
    convert_list_attrs_to_strings(new_graph)

    os.makedirs(os.path.abspath(data_dir_output), exist_ok=True)
    nx.write_graphml(new_graph, os.path.join(data_dir_output, f'{graph_root}_graphML_simplified.graphml'))
    return new_graph, updated_embeddings



# Visualization (Louvain):
def assign_louvain_groups(G: nx.Graph, attr_name: str = "louvain", random_state: int = 42) -> str:
    import community.community_louvain as community_louvain  # noqa: F401
    undirected = G.to_undirected()
    # Same giant-component order and seed as the Figure S1 statistics (analysis_digraph.py).
    giant = undirected.subgraph(max(nx.connected_components(undirected), key=len)).copy()
    part = community_louvain.best_partition(giant, random_state=random_state)
    nx.set_node_attributes(G, part, attr_name)
    return attr_name


def build_pyvis_interactive_louvain(
    G: nx.Graph,
    outfile: str,
    norm_attr: str = "freq_norm",
    max_nodes: Optional[int] = None,
    edge_norm_attr: str = "p_cond",
    size_min: float = 10.0,
    size_max: float = 40.0,
) -> None:
    """
    Interactive visualization (PyVis) colored by Louvain community.
    """
    from pyvis.network import Network  # type: ignore

    # Assign Louvain clusters
    group_attr = assign_louvain_groups(G, attr_name="louvain")

    # Select nodes (by norm_attr)
    all_nodes = list(G.nodes())

    def _norm_val(n):
        v = G.nodes[n].get(norm_attr, None)
        try:
            vv = float(v)
        except Exception:
            raise ValueError(f"Node '{n}' has no valid '{norm_attr}' attribute.")
        if not (0.0 <= vv <= 1.0):
            raise ValueError(f"Node '{n}' has '{norm_attr}'={vv}; expected a value in [0, 1].")
        return vv

    if max_nodes is not None and len(all_nodes) > max_nodes:
        all_nodes = sorted(all_nodes, key=_norm_val, reverse=True)[:max_nodes]

    H = G.subgraph(all_nodes).copy()

    # Initialize PyVis
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#000000",
        notebook=False,
        cdn_resources="in_line",
    )
    net.set_options("""
    {
      "layout": { "randomSeed": 42 },
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "stabilization": { "enabled": true, "iterations": 1000, "updateInterval": 25 },
        "minVelocity": 0.75
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 120,
        "hideEdgesOnDrag": true,
        "hideEdgesOnZoom": true
      },
      "configure": {
        "enabled": true,
        "filter": ["physics", "interaction"]
      },
      "edges": {
        "color": { "inherit": "from", "opacity": 0.7 },
        "smooth": { "type": "dynamic" }
      }
    }
    """)

    # Nodes: label/title/group/size (size derived only from norm_attr)
    for n, attrs in H.nodes(data=True):
        grp = attrs.get(group_attr)
        H.nodes[n]["group"] = str(grp)

        norm = _norm_val(n)
        size = float(size_min) + (float(size_max) - float(size_min)) * norm

        freq = attrs.get("freq", 1)
        try:
            freq = int(freq)
        except Exception:
            try:
                freq = int(float(freq))
            except Exception:
                freq = 1

        label = attrs.get("name", str(n))
        title = f"{label}&lt;br&gt;Group: {H.nodes[n].get('group','-')}&lt;br&gt;Frequency: {freq}&lt;br&gt;Normalized: {norm:.3f}"

        H.nodes[n]["label"] = label
        H.nodes[n]["title"] = title
        H.nodes[n]["size"]  = size

    # Edge width
    for u, v, ed in H.edges(data=True):
        e_norm = ed.get(edge_norm_attr, None)
        if e_norm is None:
            raise ValueError(f"Edge ({u}->{v}) has no '{edge_norm_attr}' attribute. Run normalize_graph_frequencies().")
        try:
            e_norm_f = float(e_norm)
        except Exception:
            raise ValueError(f"Edge ({u}->{v}) has a nonnumeric '{edge_norm_attr}'={e_norm}.")
        if not (0.0 <= e_norm_f <= 1.0):
            raise ValueError(f"Edge ({u}->{v}) has '{edge_norm_attr}'={e_norm_f}; expected a value in [0,1].")
        width = 1.0 + 4.0 * e_norm_f
        ed["value"] = width
        ed["width"] = width

    # Export HTML
    net.from_nx(H)
    net.save_graph(outfile)
    print(f" PyVis (Louvain) saved to: {outfile}")


