"""Inspect a recorded silver synthesis example without installing packages or calling a model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = Path("synthway/generation_routes/output_LLM_stratified_anthropic_claude-opus-4.5/1561.json_Ag_1749")
SCORES = Path("synthway/bench/all_models_merged_metrics_selected_40_median3.csv")
JUDGMENT = Path("paper_tables/examples/Claude_Llama_evaluations.json")
LAB_SCORES = Path("results/source_data/table_s20.csv")
LAB_OUTCOMES = Path("results/source_data/figure_4_outcomes.csv")


def main(out):
    out = out.resolve()
    # An example must never overwrite the publication's supplied data.
    protected = [ROOT / name for name in ("data", "graph_output", "synthway", "results")]
    if any(out == path or path in out.parents for path in protected):
        raise ValueError("Choose a new output directory, for example outputs/demo.")
    route_path = CASE / "route_3.json"
    route = json.loads((ROOT / route_path).read_text(encoding="utf-8"))
    if not route.get("raw_text") or not route.get("retrieval_text"):
        raise ValueError("Incomplete example route.")
    context = json.loads((ROOT / CASE / "retrieval_context.json").read_text(encoding="utf-8"))
    with (ROOT / SCORES).open(encoding="utf-8-sig", newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row["path"] == route_path.as_posix()]
    if len(matches) != 1:
        raise ValueError("Expected exactly one recorded score row for the example route.")
    row = matches[0]
    scores = {"chemical_plausibility": float(row["chemistry_score_mean"]),
              "answer_relevancy": float(row["answer_score_mean"]),
              "original_faithfulness": float(row["faithfulness_score_mean"])}
    summary = {"mode": "recorded_example", "composition": context["composition"],
               "generator": row["model"], "retrieved_contexts": len(context["retrieval_context"]),
               "scores": scores, "score_aggregation": "stored mean of three ratings",
               "sources": [route_path.as_posix(), (CASE / "retrieval_context.json").as_posix(), SCORES.as_posix()]}
    bundle = json.loads((ROOT / JUDGMENT).read_text(encoding="utf-8"))
    judgment = bundle["worked_example"]
    repeated_scores = []
    for run in bundle["evaluation_runs"]:
        if run["table"] != "1":
            continue
        item = next(row for row in run["records"] if row["path"].endswith(
            "/".join(route_path.parts[-2:])))
        repeated_scores.append({
            "run": run["run"], "score": item["faithfulness_score"],
            "input_tokens": item.get("faith_judge_prompt_tokens_total"),
            "output_tokens": item.get("faith_judge_completion_tokens_total"),
            "time_s": item.get("faith_total_time_s"), "cost_usd": item.get("faith_judge_cost_usd_total"),
        })
    chosen = next(row for row in repeated_scores if row["run"] == judgment["run"])
    if judgment["route_source"] != route_path.as_posix():
        raise ValueError("The worked judgment belongs to a different route.")
    with (ROOT / LAB_SCORES).open(encoding="utf-8", newline="") as stream:
        lab = next(row for row in csv.DictReader(stream) if row["path"] == route_path.as_posix())
    with (ROOT / LAB_OUTCOMES).open(encoding="utf-8", newline="") as stream:
        outcome = next(row for row in csv.DictReader(stream)
                       if row["route_label"] == judgment["laboratory_route_label"])
    if lab["raw_text"] != route["raw_text"]:
        raise ValueError("The laboratory record refers to a different generated protocol.")
    summary["worked_judgment"] = {
        "expert_route_ID": judgment["expert_ID"], "judge": bundle["judge"], "run": judgment["run"],
        "synthesis_specific_faithfulness": chosen["score"],
        "repeats": repeated_scores,
    }
    summary["laboratory_outcome"] = outcome
    summary["sources"].extend(path.as_posix() for path in (JUDGMENT, LAB_SCORES, LAB_OUTCOMES))
    files = {name: (ROOT / CASE / name).read_bytes()
             for name in ("route_3.json", "retrieval_context.json")}
    files["summary.json"] = (json.dumps(summary, indent=2) + "\n").encode("utf-8")
    # Validate all existing outputs before writing anything. Repeated runs can
    # reuse identical files, while edited outputs are never silently replaced.
    existing = []
    for name, content in files.items():
        target = out / name
        if target.is_symlink() or (target.exists() and
                                  (not target.is_file() or target.read_bytes() != content)):
            raise FileExistsError(
                f"Existing output differs from this recorded example: {target}. "
                "No files were overwritten; choose a different --out directory.")
        if target.exists():
            existing.append(name)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if name not in existing:
            with (out / name).open("xb") as stream:
                stream.write(content)
    if existing:
        print(f"Reused {len(existing)} matching output file(s) in {out}.")
    print("Recorded example — no generation, API calls or new evaluations.")
    print(f"Target: {summary['composition']}; retrieved contexts: {summary['retrieved_contexts']}")
    for name, score in scores.items():
        print(f"{name}: {score:.3f} (stored mean of three ratings)")
    print(f"Synthesis-specific Faithfulness judged by {bundle['judge']}:")
    for observation in repeated_scores:
        print(f"  {observation['run']}: score={observation['score']:.5f}, "
              f"input/output tokens={observation['input_tokens']}/{observation['output_tokens']}, "
              f"time_s={observation['time_s']}, cost_usd={observation['cost_usd']}")
    print(f"Laboratory product / phase / morphology / size: "
          f"{outcome['solid_product']} / {outcome['phase_XRD']} / "
          f"{outcome['morphology_SEM']} / {outcome['size_SEM']}")
    print(f"Output: {out}")
    print("Open route_3.json for the generated protocol and retrieval_context.json for its evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/recorded_example")
    args = parser.parse_args()
    try:
        main(args.out)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Demo failed: {exc}\n")
