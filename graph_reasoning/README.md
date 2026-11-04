# Graph construction

This directory builds and inspects the synthesis knowledge graph used by SynthWay. Each local graph represents one synthesis procedure; the global graph combines procedures and merges similar nodes. `synthesis_id` identifies procedures and is stored on edges rather than represented as a separate node.

## Where this fits

```text
data/merged_all.xlsx       Curated synthesis records: input to graph construction
graph_reasoning/           Construction, node matching, analysis and visualization
graph_output/              Saved graph artifacts supplied with the study
outputs/graph/             Default destination for newly generated graph artifacts
synthway/                  Conversion to retrieval records, indexing and generation
```

The publication includes the final graph and retrieval index. Intermediate local graphs are generated with the commands below and are not distributed. The application and table calculations do not require rebuilding them. See the [main README](../README.md) for installation and application usage.

## Build order

Run the following from the repository root, in the Python environment described in the main README:

```sh
python graph_reasoning/local_graph_without_synthesisID.py
python graph_reasoning/count_local_graphs.py
python graph_reasoning/Global_graph_generation_without_synthesisID.py
python graph_reasoning/rename_local_nodes_to_global.py
```

| Step | Script | Reads and writes |
|---|---|---|
| 1. Local graphs | [local_graph_without_synthesisID.py](local_graph_without_synthesisID.py) | Reads `data/merged_all.xlsx`, groups rows by `synthesis_id`, and writes one GraphML file per procedure to `local_graphs_without_synthesis/`. |
| 2. Frequencies | [count_local_graphs.py](count_local_graphs.py) | Counts nodes and edges across the local graphs; writes `analytics/local_counts/node_counts.json`, `edge_counts.json`, and diagnostic reports. |
| 3. Global graph | [Global_graph_generation_without_synthesisID.py](Global_graph_generation_without_synthesisID.py) | Reads the source table and frequency files; embeds node names, merges eligible nodes within the same field, and removes small components. Writes `graph.graphml`, merge maps, cleaning reports, embeddings and visualizations. |
| 4. Local node alignment | [rename_local_nodes_to_global.py](rename_local_nodes_to_global.py) | Applies the global merge maps and removal records to local graphs. Writes `local_graphs_renamed/`, `_rename_prune_report.csv`, and `_renamed_vs_global_summary.csv` inside that directory. |

Output paths in the table are relative to `outputs/graph/`. All four steps use `SYNTHWAY_GRAPH_OUTPUT` if set; use the same destination throughout a run. These scripts resolve the repository root from their own location, so the outer project folder can have any name.

Step 4 reports any remaining differences from the global graph. Its optional `--strict_align` flag also removes nodes absent from the final global graph, subject to the script's explicit keep-list.

## Supporting files

| File | When to use it |
|---|---|
| [graph_tools.py](graph_tools.py) | Shared embedding, node matching, cleaning, frequency normalization and interactive visualization functions; imported by the global builder. |
| [local_graph_files.py](local_graph_files.py) | Reads a local graph collection, including unchanged graphs referenced by `shared_from_raw.csv`; verifies their checksums. Also supports newly generated local graph collections. |
| [stat_for_merged.py](stat_for_merged.py) | Reports field coverage, missing values and distributions, with local/global graph comparisons. Use `--out-dir outputs/graph/statistics` to choose the report destination. |
| [thresholds_numeric_similarity.py](thresholds_numeric_similarity.py) | Analyses saved merge-pair similarities and writes suggested thresholds and plots. Suggestions do not automatically change the graph builder. |
| [visualization_local_example.py](visualization_local_example.py) | Regenerates the two individual local-graph HTML examples for Figure S1a,b. |
| [visualization_two_local.py](visualization_two_local.py) | Regenerates the combined local-graph HTML example for Figure S1c. |

The visualization scripts use `outputs/graph/` (or `SYNTHWAY_GRAPH_OUTPUT`); first generate the local graphs with step 1. Saved Figure S1 HTML examples remain in `graph_output/`. The threshold-analysis script defaults to `graph_output/`; use `--pairs_csv outputs/graph/merge_pairs_for_renaming.csv --out_dir outputs/graph/thresholds` to analyse a new build.

## Next: retrieval records

To convert newly aligned local graphs into JSON records for retrieval:

```sh
python synthway/rag_build_jsons.py --graphs outputs/graph/local_graphs_renamed --out outputs/rebuilt_corpus
```

This creates `outputs/rebuilt_corpus/json/` and `routes.jsonl`. Index construction is a separate step handled by [build_rag_index.py](../synthway/build_rag_index.py); graph construction alone does not update the application's retrieval index.
