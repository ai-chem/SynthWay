"""
local_graph_without_synthesisID.py builds local graphs corresponding to synthesis routes.
"""
import pandas as pd
import networkx as nx
import os
import re
import ast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
GRAPH_DIR = os.environ.get('SYNTHWAY_GRAPH_OUTPUT', os.path.join(REPO_DIR, 'outputs', 'graph'))
INPUT_FILE = os.path.join(REPO_DIR, 'data', 'merged_all.xlsx')
OUTPUT_DIR = os.path.join(GRAPH_DIR, 'local_graphs_without_synthesis')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', s)

def convert_attrs_to_strings(G):
    for _, data in G.nodes(data=True):
        for k, v in data.items():
            if isinstance(v, (list, set)):
                data[k] = ", ".join(map(str, v))
    for _, _, data in G.edges(data=True):
        for k, v in data.items():
            if isinstance(v, (list, set)):
                data[k] = ", ".join(map(str, v))

# Set 'name' to the value part of 'label:value'
def _extract_value_from_node_id(node_id: str) -> str:
    try:
        if isinstance(node_id, str) and ":" in node_id:
            return node_id.split(":", 1)[1].strip()
        return str(node_id)
    except Exception:
        return str(node_id)

def _attach_name_attr(GX: nx.Graph):
    """
    Set each node's 'name' attribute to the value part of 'label:value'.
    """
    try:
        nx.set_node_attributes(GX, {n: _extract_value_from_node_id(n) for n in GX.nodes()}, name="name")
    except Exception as e:
        print(f"[WARN] _attach_name_attr failed: {e}")

# Step 1. Load the table
df = pd.read_excel(INPUT_FILE)

# Step 2. Group by synthesis_id
grouped = df.groupby("synthesis_id")

# Step 3. Build and save graphs
for synthesis_id, group in grouped:
    G_local = nx.DiGraph()
    nodes = {}
    edges = []

    for _, row in group.iterrows():
        # Get composition directly as field:value
        comp_val = row.get("composition")
        if not isinstance(comp_val, str) or comp_val.strip().lower() == 'unknown':
            continue  # Skip if composition is missing

        comp_clean = comp_val.strip()
        comp_id = f"composition:{comp_clean}"
        nodes[comp_id] = {"label": "composition"}

        # Reagents and solvent -> composition, with parameters for each
        FIELD_PARAMS = {
            "precursor": ["precursor_mass", "precursor_conc"],
            "additive": ["additive_mass", "additive_conc"],
            "seeds": ["seed_mass", "seed_conc"],
            "solvent": ["solvent_v"],
        }

        for field, param_keys in FIELD_PARAMS.items():
            val = row.get(field)
            if isinstance(val, str) and val.strip().lower() != 'unknown':
                val_clean = val.strip()
                node_id = f"{field}:{val_clean}"
                nodes[node_id] = {"label": field}
                edges.append((node_id, comp_id))

                for param in param_keys:
                    pval = row.get(param)
                    if pd.notna(pval) and str(pval).lower() != "unknown" and pval != -999:
                        param_node = f"{param}:{pval}"
                        nodes[param_node] = {"label": param}
                        edges.append((node_id, param_node))

        # Other parameters -> composition
        for field in ["temperature", "ph", "method"]:
            val = row.get(field)
            if pd.notna(val) and str(val).lower() != "unknown" and val != -999:
                node_id = f"{field}:{str(val).strip()}"
                nodes[node_id] = {"label": field}
                edges.append((node_id, comp_id))

        # composition -> properties
        for field in ["shape", "crystalsystem"]:
            val = row.get(field)
            if isinstance(val, str) and val.strip().lower() != 'unknown':
                val_clean = val.strip()
                node_id = f"{field}:{val_clean}"
                nodes[node_id] = {"label": field}
                edges.append((comp_id, node_id))

        for size_field in ["size_length", "size_width", "size_depth"]:
            val = row.get(size_field)
            if pd.notna(val) and val != -999:
                node = f"{size_field}:{val}"
                nodes[node] = {"label": size_field}
                edges.append((comp_id, node))

        pdf_values = row.get("pdfno_list")
        if isinstance(pdf_values, str) and pdf_values.lower() != "unknown":
            try:
                pdf_list = ast.literal_eval(pdf_values) if "[" in pdf_values else [pdf_values]
                for pdf in pdf_list:
                    pdf = str(pdf).strip()
                    if pdf:
                        node_id = f"pdfno:{pdf}"
                        nodes[node_id] = {"label": "pdfno"}
                        edges.append((comp_id, node_id))
            except Exception as e:
                print(f"Error reading pdfno_list for {synthesis_id}: {e}")

    # Add to graph
    for node, attrs in nodes.items():
        G_local.add_node(node, **attrs)
    for u, v in edges:
        G_local.add_edge(u, v, synthesis=synthesis_id)

    _attach_name_attr(G_local)
    convert_attrs_to_strings(G_local)
    safe_id = sanitize_filename(synthesis_id)
    output_path = os.path.join(OUTPUT_DIR, f"synthesis_{safe_id}.graphml")

    try:
        nx.write_graphml(G_local, output_path)
    except Exception as e:
        print(f"Error saving {synthesis_id}: {e}")

print(f"Local graphs saved to: {OUTPUT_DIR}")
