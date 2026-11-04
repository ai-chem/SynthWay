"""Write the two local-graph examples of Figure S1 (a, b) as self-contained HTML files."""
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
import json
import os
import networkx as nx
from pyvis.network import Network

# === Parameters ===
output_dir = os.environ.get("SYNTHWAY_GRAPH_OUTPUT", str(_REPO_ROOT / "outputs" / "graph"))
input_dir = os.path.join(output_dir, "local_graphs_without_synthesis")
panels = [("figure_s1a_local_graph_3005.html", "synthesis_1001.json_Ag_3005.graphml", "orange"),
          ("figure_s1b_local_graph_3006.html", "synthesis_1001.json_Ag_3006.graphml", "#3277b3")]
keep_prefix = ("additive_conc", "precursor_conc", "seed_conc", "additive_mass", "precursor_mass", "seed_mass",
               "solvent_v", "temperature", "ph", "size_length", "size_width", "size_depth")
options = {"layout": {"randomSeed": 42},
           "physics": {"solver": "forceAtlas2Based", "forceAtlas2Based": {"centralGravity": 0.015, "gravitationalConstant": -31}}}


def node_label(node):
    prefix, _, value = str(node).partition(":")
    return node if prefix in keep_prefix or not value else value


if __name__ == "__main__":
    for output_name, input_name, color in panels:
        G = nx.read_graphml(os.path.join(input_dir, input_name))
        print(f"{input_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        nt = Network(height="900px", width="100%", notebook=False, directed=True, cdn_resources="in_line")
        for node in G.nodes():
            nt.add_node(node, label=node_label(node), color=color, shape='dot', size=15, font={'size': 15, 'color': 'black'})
        for source, target in G.edges():
            nt.add_edge(source, target)
        nt.set_options(json.dumps(options))
        output_file = os.path.join(output_dir, output_name)
        nt.save_graph(output_file)
        print(f"Saved: {output_file}")
