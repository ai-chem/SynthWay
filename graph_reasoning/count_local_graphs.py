"""
count_local_graphs.py counts node and edge frequencies in several representations (by node ID, by (name,label), by field:value, and by label->label roles) and saves JSON reports for building the global graph.
- Main artifacts for the global graph: node_counts.json and edge_counts.json (ID-based representations)
"""

import glob
import json
import os

import networkx as nx


def _node_key_str(name, label):
    return f"{name}|||{label}"


def _edge_key_str(s_name, s_label, t_name, t_label):
    return f"{s_name}|||{s_label}>>> {t_name}|||{t_label}"


def _split_field_value(name: str, label: str):
    name = (str(name) if name is not None else "").strip()
    label = (str(label) if label is not None else "").strip()

    if label:
        return label, name
    if ":" in name:
        f, v = name.split(":", 1)
        return f.strip(), v.strip()
    return "", name


# Frequency aggregators
# (name, label) -> frequency across all local graphs
node_counts_by_name_label = {}
# value-only (node 'name attribute, e.g., 'Fe(NO3)3') -> frequency
node_counts_by_value = {}
# node-id (GraphML id, e.g., 'precursor:Fe(NO3)3') -> frequency
node_counts_by_id = {}
# (src.name, src.label -> dst.name, dst.label) -> frequency
edge_counts_by_name_label = {}

# Additional representations
# (field:value) -> frequency
node_counts_by_field_value = {}
# label only -> frequency (how often any value of a field occurs)
node_counts_by_field = {}
# (src.label -> dst.label) -> frequency (number of edges between label pairs)
edge_counts_by_label = {}

# (src_id -> dst_id) (node-ID representation) -> frequency
edge_counts_by_node = {}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
GRAPH_DIR = os.environ.get("SYNTHWAY_GRAPH_OUTPUT", os.path.join(REPO_DIR, "outputs", "graph"))
LOCAL_DIR = os.path.join(GRAPH_DIR, "local_graphs_without_synthesis")
PATTERN = os.path.join(LOCAL_DIR, "*.graphml")
OUTPUT_DIR = os.path.join(GRAPH_DIR, "analytics", "local_counts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


files = glob.glob(PATTERN)
if not files:
    print(f"No .graphml files found in {LOCAL_DIR}")

for gm in files:
    try:
        G = nx.read_graphml(gm)
    except Exception as e:
        print(f"Failed to read {gm}: {e}")
        continue

    def _attrs(n):
        d = G.nodes[n]
        name = str(d.get("name", n))
        label = str(d.get("label", ""))
        return name, label

    # Nodes
    for n in G.nodes():
        # Count by node ID (GraphML node id)
        n_id = str(n)
        node_counts_by_id[n_id] = node_counts_by_id.get(n_id, 0) + 1
        name, label = _attrs(n)
        # Diagnostic representation
        nk = _node_key_str(name, label)
        node_counts_by_name_label[nk] = node_counts_by_name_label.get(nk, 0) + 1
        node_counts_by_value[name] = node_counts_by_value.get(name, 0) + 1
        # (field:value) representation
        field, value = _split_field_value(name, label)
        fv_key = f"{field}:{value}" if field else value
        node_counts_by_field_value[fv_key] = node_counts_by_field_value.get(fv_key, 0) + 1
        node_counts_by_field[field] = node_counts_by_field.get(field, 0) + 1

    # Edges
    for u, v in G.edges():
        s_name, s_label = _attrs(u)
        t_name, t_label = _attrs(v)
        # Diagnostic representation
        ek = _edge_key_str(s_name, s_label, t_name, t_label)
        edge_counts_by_name_label[ek] = edge_counts_by_name_label.get(ek, 0) + 1
        # Edge key based on node IDs (u_id >>> v_id)
        edge_k_id = f"{str(u)}>>> {str(v)}"
        edge_counts_by_node[edge_k_id] = edge_counts_by_node.get(edge_k_id, 0) + 1
        # Aggregate by role (label -> label)
        el = f"{s_label}>>> {t_label}"
        edge_counts_by_label[el] = edge_counts_by_label.get(el, 0) + 1


# Save artifacts
node_counts_by_field_clean = {k: v for k, v in node_counts_by_field.items() if k}

with open(os.path.join(OUTPUT_DIR, "node_counts_by_name_label.json"), "w", encoding="utf-8") as f:
    json.dump(node_counts_by_name_label, f, indent=2, ensure_ascii=False)

# "name": save the aggregate by value
with open(os.path.join(OUTPUT_DIR, "node_counts_by_value.json"), "w", encoding="utf-8") as f:
    json.dump(node_counts_by_value, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "edge_counts_by_name_label.json"), "w", encoding="utf-8") as f:
    json.dump(edge_counts_by_name_label, f, indent=2, ensure_ascii=False)

# Main artifacts (by node/edge IDs)
with open(os.path.join(OUTPUT_DIR, "node_counts.json"), "w", encoding="utf-8") as f:
    json.dump(node_counts_by_id, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "edge_counts.json"), "w", encoding="utf-8") as f:
    json.dump(edge_counts_by_node, f, indent=2, ensure_ascii=False)

# Additional artifacts
with open(os.path.join(OUTPUT_DIR, "node_counts_by_field_value.json"), "w", encoding="utf-8") as f:
    json.dump(node_counts_by_field_value, f, indent=2, ensure_ascii=False)

with open(os.path.join(OUTPUT_DIR, "node_counts_by_field.json"), "w", encoding="utf-8") as f:
    json.dump(node_counts_by_field_clean, f, indent=2, ensure_ascii=False)

_edge_counts_by_label_clean = {
    k: v
    for k, v in edge_counts_by_label.items()
    if ">>>" in k and k.split(">>>")[0].strip() and k.split(">>>")[1].strip()
}

with open(os.path.join(OUTPUT_DIR, "edge_counts_by_label.json"), "w", encoding="utf-8") as f:
    json.dump(_edge_counts_by_label_clean, f, indent=2, ensure_ascii=False)


# summary.json
print(f"Processed {len(files)} graphs from: {LOCAL_DIR}")
print(
    f"Unique nodes (by id): {len(node_counts_by_id)}; "
    f"(name,label): {len(node_counts_by_name_label)}; "
    f"(value-only): {len(node_counts_by_value)}; "
    f"(field:value): {len(node_counts_by_field_value)}; "
    f"unique fields: {len([k for k in node_counts_by_field if k])}"
)
print(
    f"Unique edges (by name+label): {len(edge_counts_by_name_label)}; "
    f"unique edges (by label->label): {len(edge_counts_by_label)}"
)
print(f"Unique edges (by node-id): {len(edge_counts_by_node)}")

# Integrity check: both endpoints of every edge must exist in node_ids
dangling = [
    e
    for e in edge_counts_by_node
    if e.split(">>>", 1)[0].strip() not in node_counts_by_id
    or e.split(">>>", 1)[1].strip() not in node_counts_by_id
]
print(f"Dangling edges: {len(dangling)}")
print(f"Saved analytics to: {OUTPUT_DIR}")

summary = {
    "graphs": len(files),
    "unique_nodes_by_id": len(node_counts_by_id),
    "unique_nodes_by_name_label": len(node_counts_by_name_label),
    "unique_nodes_by_value": len(node_counts_by_value),
    "unique_nodes_by_field_value": len(node_counts_by_field_value),
    "unique_fields": len([k for k in node_counts_by_field if k]),
    "unique_edges_by_name_label": len(edge_counts_by_name_label),
    "unique_edges_by_label": len(edge_counts_by_label),
    "unique_edges_by_node": len(edge_counts_by_node),
    "dangling_edges": len(dangling),
    "local_dir": str(LOCAL_DIR),
    "out_dir": str(OUTPUT_DIR),
}
with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
