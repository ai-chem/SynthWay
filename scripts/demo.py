"""Run the real BGE retrieval, LLM generation and three-metric evaluation on one small example."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/demo"
sys.path.insert(0, str(ROOT / "synthway"))
from credentials import credential_environment
from gui_models import GUI_MODELS
from ss_faithfulness import SS_FAITHFULNESS_ID, validate_ss_record


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def credentials(root=ROOT, environ=None):
    """Match the generator's .env priority, then pass the same key to both evaluators."""
    env, selected = credential_environment(root, environ)
    key = env.get("OPENROUTER_API_KEY", "").strip()
    require(
        key and key != "paste_your_openrouter_key_here" and not key.startswith("Bearer "),
        "Set OPENROUTER_API_KEY in synthway/.env (if it exists) or the root .env. See README.md.",
    )
    env["OPENROUTER_API_KEY"] = key
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
    env["DEEPEVAL_UPDATE_WARNING_OPT_OUT"] = "YES"
    return env, selected


def preflight(root=ROOT):
    require(sys.version_info[:2] in {(3, 11), (3, 12)}, "Use Python 3.11 or 3.12.")
    needed = (
        "numpy",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
        "transformers",
        "torch",
        "faiss-cpu",
        "rich",
        "python-dotenv",
        "google-genai",
        "openai",
        "requests",
        "deepeval",
    )
    missing = []
    for package in needed:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    require(
        not missing,
        "Missing dependencies: "
        + ", ".join(missing)
        + ". Run python -m pip install -r requirements.txt",
    )
    env, selected = credentials(root)
    source = str(selected.relative_to(root)) if selected else "environment variable"
    print(
        f"Local prerequisites found; credential source: {source}. Key value is not displayed.",
        flush=True,
    )
    print(
        "This local check does not verify API credits, model access or the BGE download/cache.",
        flush=True,
    )
    return env


def commands(out, config, root=ROOT):
    """Use the existing public entry points, with the normal BGE and metric implementations."""

    def script(name):
        return [sys.executable, str(root / "synthway" / name)]

    generated = str(out / "generation")
    model = config["generator"]
    judge = config["judge"]
    threshold = str(config["threshold"])
    pacing = ["--rpm-limit", str(config["rpm_limit"])]
    generation = script("predictions.py") + [
        "--composition",
        config["composition"],
        "--num-routes",
        str(config["num_routes"]),
        "--model",
        model,
        "--rag-idx",
        str(out / "index"),
        "--rag-topk",
        str(config["rag_topk"]),
        "--llm-output-dir",
        generated,
    ]
    if config["method"]:
        generation += ["--method", config["method"]]
    return [
        (
            "index",
            script("build_rag_index.py")
            + [
                "build",
                "--json_dir",
                str(root / "data/demo/corpus"),
                "--out_idx",
                str(out / "index"),
            ],
        ),
        ("generation", generation),
        (
            "chemical",
            script("evaluate_generation_deepeval.py")
            + [
                "--routes-dir",
                generated,
                "--eval-model",
                judge,
                "--threshold",
                threshold,
                "--max-cases",
                str(config["num_routes"]),
                "--out-json",
                str(out / "chemical_scores.json"),
            ]
            + pacing,
        ),
        (
            "rag_metrics",
            script("evaluate_answer_faithfulness_deepeval.py")
            + [
                "--routes-dir",
                generated,
                "--eval-model",
                judge,
                "--answer-threshold",
                threshold,
                "--faithfulness-threshold",
                threshold,
                "--faithfulness-mode",
                "ss",
                "--max-cases",
                str(config["num_routes"]),
                "--out-json",
                str(out / "rag_scores.json"),
            ]
            + pacing,
        ),
    ]


def validate_index(out, data=DATA):
    index = out / "index"
    for name in (
        "meta.jsonl",
        "model.json",
        "tfidf.pkl",
        "tfidf.npz",
        "tfidf_row_norms.npy",
        "vocab_terms.json",
        "vocab_index.faiss",
    ):
        require(
            (index / name).is_file() and (index / name).stat().st_size > 0,
            f"Missing index artifact: {name}",
        )
    require(
        read_json(index / "model.json")["model"] == "BAAI/bge-large-en-v1.5",
        "Unexpected embedding model",
    )
    require(
        bool(read_json(index / "vocab_terms.json")),
        "BGE vocabulary is empty; semantic expansion was not built",
    )
    with (index / "vocab_index.faiss").open("rb") as stream:
        require(
            not stream.read(64).startswith(b"version https://git-lfs"),
            "FAISS file is a Git LFS pointer",
        )
    records = [
        json.loads(line)
        for line in (index / "meta.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    corpus = [read_json(path) for path in sorted((data / "corpus").glob("*.json"))]
    require(len(records) == len(corpus) == 3, "The demo must index its three supplied records")
    require(
        {r["synthesis_id"] for r in records} == {r["synthesis_id"] for r in corpus},
        "Demo index IDs differ from input",
    )


def validate_generation(out, config, contract, data=DATA):
    paths = sorted((out / "generation").rglob("route_*.json"))
    require(
        len(paths) == contract["route_count"],
        "Expected one parsed route; inspect generation.log and route_raw.txt",
    )
    route_path = paths[0]
    route = read_json(route_path)
    for field in contract["route_required_text_fields"]:
        require(nonempty_text(route.get(field)), f"Generated route has no usable {field}")
    context = read_json(route_path.parent / "retrieval_context.json")
    require(
        context.get("composition") == config["composition"],
        "Retrieved context has the wrong target",
    )
    require(
        context.get("method") == config["method"] and context.get("rag_topk") == config["rag_topk"],
        "Retrieved context has the wrong request settings",
    )
    texts = context.get("retrieval_context")
    supplied = {read_json(path)["retrieval_text"] for path in (data / "corpus").glob("*.json")}
    require(
        isinstance(texts, list) and all(nonempty_text(text) for text in texts),
        "Missing retrieval evidence",
    )
    require(
        len(texts) == contract["context_count"]
        and len(set(texts)) == len(texts)
        and set(texts) == supplied,
        "Expected the three distinct demo contexts",
    )
    require(
        nonempty_text(context.get("query"))
        and context["query"] != f"composition: {config['composition']}",
        "Expected a BGE-expanded query for this example; check model loading in generation.log",
    )
    return route_path


def validate_evaluation(out, filename, fields, route_path, config, contract):
    records = read_json(out / filename)
    require(
        isinstance(records, list) and len(records) == 1, f"Expected one evaluation in {filename}"
    )
    record = records[0]
    require(isinstance(record, dict) and not record.get("error"), f"Evaluation error in {filename}")
    evaluated = Path(record.get("path", ""))
    if not evaluated.is_absolute():
        evaluated = ROOT / evaluated
    require(evaluated.resolve() == route_path.resolve(), f"{filename} evaluates a different route")
    scores = {}
    for field in fields:
        score = record.get(field["score"])
        require(
            type(score) in (int, float)
            and math.isfinite(score)
            and contract["score_min"] <= score <= contract["score_max"],
            f"Missing/invalid {field['score']} in {filename}",
        )
        require(
            record.get(field["threshold"]) == config["threshold"], f"Wrong threshold in {filename}"
        )
        passed = record.get(field["passed"])
        require(
            type(passed) is bool and passed == (score >= config["threshold"]),
            f"Inconsistent pass flag in {filename}",
        )
        require(
            nonempty_text(record.get(field["reason"])), f"Missing {field['reason']} in {filename}"
        )
        scores[field["score"]] = {"score": score, "passed": passed}
    if filename == "rag_scores.json":
        validate_ss_record(record)
    return scores


def validate_output(out, config, data=DATA):
    contract = read_json(data / "expected_output.json")
    validate_index(out, data)
    route_path = validate_generation(out, config, contract, data)
    scores = {}
    for filename, fields in contract["evaluations"].items():
        scores.update(validate_evaluation(out, filename, fields, route_path, config, contract))
    return {
        "status": "completed",
        "route": route_path.relative_to(out).as_posix(),
        "retrieved_contexts": contract["context_count"],
        "faithfulness_metric": SS_FAITHFULNESS_ID,
        "scores": scores,
        "all_metrics_passed": all(item["passed"] for item in scores.values()),
    }


def run_demo(out, config, env, *, timeout=1800, root=ROOT):
    out = out.resolve()
    # A failed or previous run is retained; choose a new folder to run again.
    for name in ("data", "synthway", "graph_output", "results", "docs", "scripts", "tests"):
        protected = (root / name).resolve()
        require(
            out != protected and protected not in out.parents,
            "Choose a new output folder, e.g. outputs/demo",
        )
    data = root / "data/demo"
    contract = read_json(data / "expected_output.json")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "input.json", config)
    metadata = {
        "input_sha256": {
            path.relative_to(data).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(data.rglob("*.json"))
        },
        "stages_seconds": {},
    }
    write_json(out / "run.json", metadata)
    started = time.perf_counter()
    for stage, command in commands(out, config, root):
        print(f"Running {stage}; log: {out / (stage + '.log')}", flush=True)
        stage_started = time.perf_counter()
        with (out / f"{stage}.log").open("x", encoding="utf-8") as log:
            try:
                subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    f"Stage {stage} failed or timed out. Inspect {out / (stage + '.log')}; partial outputs were retained."
                ) from exc
        metadata["stages_seconds"][stage] = round(time.perf_counter() - stage_started, 3)
        write_json(out / "run.json", metadata)
        if stage == "index":
            validate_index(out, data)
        elif stage == "generation":
            validate_generation(out, config, contract, data)
        elif stage == "chemical":
            route = validate_generation(out, config, contract, data)
            validate_evaluation(
                out,
                "chemical_scores.json",
                contract["evaluations"]["chemical_scores.json"],
                route,
                config,
                contract,
            )
    summary = validate_output(out, config, data)
    summary["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check",
        action="store_true",
        help="Check local dependencies and key presence; no model/API calls",
    )
    modes.add_argument(
        "--validate-output", type=Path, help="Validate an existing run without model/API calls"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/demo",
        help="New output folder; existing folders are not overwritten",
    )
    parser.add_argument(
        "--generator",
        choices=list(GUI_MODELS.values()),
        help="Override the default OpenRouter generator",
    )
    parser.add_argument(
        "--judge", choices=list(GUI_MODELS.values()), help="Override the default OpenRouter judge"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Maximum seconds per stage (default: 1800)"
    )
    args = parser.parse_args()
    try:
        if args.validate_output:
            out = args.validate_output.resolve()
            summary = validate_output(out, read_json(out / "input.json"))
        else:
            require(args.timeout > 0, "--timeout must be positive")
            if not args.check:
                require(
                    not args.out.exists(),
                    f"Output folder already exists: {args.out}. Choose another --out.",
                )
            env = preflight()
            if args.check:
                return
            config = read_json(DATA / "input.json")
            config["generator"] = args.generator or config["generator"]
            config["judge"] = args.judge or config["judge"]
            print(
                f"Live demo: BGE retrieval -> {config['generator']} -> {config['judge']}. API calls are billable.",
                flush=True,
            )
            summary = run_demo(args.out, config, env, timeout=args.timeout)
        print("PASS: live-demo output meets data/demo/expected_output.json")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Demo failed: {exc}\nSee README.md for setup and expected output.\n")


if __name__ == "__main__":
    main()
