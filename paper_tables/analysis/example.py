"""Verify the Claude–Llama example against the released ratings and regeneration records."""

from collections import Counter
from pathlib import Path
import csv
import json
import math

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def route_key(path):
    return "/".join(path.replace("\\", "/").split("/")[-2:])


def recompute_score(item):
    categories = {row["claim_index"]: row["category"] for row in item["expert_claim_categories"]}
    counts = Counter((categories[row["claim_index"]], row["verdict"]) for row in item["expert_verdicts"])
    parts = []
    for number in range(1, 5):
        name = f"q{number}"
        decisive = counts[name, "yes"] + counts[name, "no"]
        score = counts[name, "yes"] / decisive if decisive else 1.0
        if not math.isclose(score, item["expert_q_scores"][name], abs_tol=1e-12):
            raise ValueError(f"Criterion differs: {name}")
        parts.append(score)
    total = sum(weight * value for weight, value in zip((3, 2, 2, 1), parts)) / 8
    if not math.isclose(total, item["faithfulness_score"], abs_tol=1e-12):
        raise ValueError("The weighted score differs from the recorded rating.")
    return total


def verify(bundle):
    routes = {route_key(row["route_source"]): row["expert_ID"] for row in bundle["routes"]}
    if len(routes) != 10 or len(set(routes.values())) != 10:
        raise ValueError("Expected ten distinct routes with expert IDs.")

    book = load_workbook(ROOT / "data/Table_1_Judge_Ratings.xlsx", read_only=True, data_only=True)
    values = book["Judge ratings"].values
    header = next(values)
    ratings = [dict(zip(header, row)) for row in values]
    book.close()
    expected = {
        ("1", row["ID"], row["run"]): row["faithfulness_score"]
        for row in ratings if row["generator"] == bundle["generator"] and row["judge"] == bundle["judge"]
    }
    with (ROOT / "data/cross_judge_ratings.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["generator"] == bundle["generator"] and row["judge"] == bundle["judge"]:
                expected[row["table"], int(row["ID"]), row["run"]] = float(row["score"])

    seen = set()
    groups = Counter()
    for run in bundle["evaluation_runs"]:
        if len(run["records"]) != 10:
            raise ValueError("Each evaluation run must cover ten routes.")
        for item in run["records"]:
            identity = (run["table"], routes[route_key(item["path"])], run["source_run_label"])
            if identity in seen or identity not in expected:
                raise ValueError(f"Repeated or unexpected evaluation: {identity}")
            seen.add(identity)
            value = recompute_score(item) if run["table"] == "1" else item[
                "score" if run["table"] == "S18" else "faithfulness_score"
            ]
            if not math.isclose(value, expected[identity], abs_tol=1e-12):
                raise ValueError(f"Evaluation differs from the table input: {identity}")
            groups[run["table"], run["run"]] += 1
    if seen != expected.keys() or groups != Counter(
        {(table, f"v{run}"): 10 for table in ("1", "S18", "S19") for run in range(1, 4)}
    ):
        raise ValueError("The example does not cover the complete pair.")

    for run in bundle["regeneration_runs"]:
        source = PROJECT / f"synthway/regenerate_runs/repeat_{run['repeat']}/tables/attempt_trajectories_long.csv"
        with source.open(encoding="utf-8", newline="") as stream:
            expected_attempts = {
                (row["case_id"], row["route_name"], int(row["attempt_no"]), row["metric"]): float(row["score"])
                for row in csv.DictReader(stream) if row["arm_key"] == run["arm_key"]
            }
        observed_attempts = {}
        for item in run["records"]:
            for metric in ("chemical_geval", "answer_relevancy", "faithfulness"):
                key = item["case_id"], item["route_name"], item["attempt_no"], metric
                if key in observed_attempts:
                    raise ValueError(f"Repeated regeneration score: {key}")
                observed_attempts[key] = item[metric]["score"]
        if observed_attempts.keys() != expected_attempts.keys():
            raise ValueError("Regeneration attempts do not match the supplied trajectory table.")
        for key, score in observed_attempts.items():
            if not math.isclose(score, expected_attempts[key], abs_tol=1e-12):
                raise ValueError(f"Regeneration score differs: {key}")
    print(f"Verified {len(seen)} evaluations across Table 1, S18 and S19.")
    print(f"Verified {sum(len(run['records']) for run in bundle['regeneration_runs'])} regeneration attempts.")

    example = bundle["worked_example"]
    for run in bundle["evaluation_runs"]:
        if run["table"] != "1":
            continue
        item = next(row for row in run["records"]
                    if route_key(row["path"]) == route_key(example["route_source"]))
        print(f"{bundle['judge']} judging {bundle['generator']} / Ag / route_3 / "
              f"{run['run']}: {item['faithfulness_score']:.5f}")


if __name__ == "__main__":
    verify(json.loads((ROOT / "examples/Claude_Llama_evaluations.json").read_text()))
