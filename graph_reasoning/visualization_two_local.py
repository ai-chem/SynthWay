"""Write the combined local graph of Figure S1 (c) as a self-contained HTML file."""
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
import json
import networkx as nx
from pyvis.network import Network
from visualization_local_example import node_label, options, input_dir, output_dir

# === Paths ===
G1_path = str(_Path(input_dir) / "synthesis_1001.json_Ag_3005.graphml")
G2_path = str(_Path(input_dir) / "synthesis_1001.json_Ag_3006.graphml")
output_html = str(_Path(output_dir) / "figure_s1c_combined_3005_3006.html")

# === Colors ===
common_color = "#16a3a3"
blue_color = "#3277b3"

# === Load and combine ===
G1 = nx.read_graphml(G1_path)
G2 = nx.read_graphml(G2_path)
combined = nx.compose(G1, G2)

# === PyVis ===
nt = Network(height="900px", width="100%", notebook=False, directed=True, cdn_resources="in_line")

for node in combined.nodes():
    if node in G1 and node in G2:
        color = common_color
    elif node in G1:
        color = "orange"
    else:
        color = blue_color

    nt.add_node(
        node,
        label=node_label(node),
        color=color,
        shape='dot',
        size=15,
        font={'size': 15, 'color': 'black'}
    )

for u, v in combined.edges():
    nt.add_edge(u, v)

nt.set_options(json.dumps(options))
nt.save_graph(output_html)
print(f"Graph ready: {output_html}")
