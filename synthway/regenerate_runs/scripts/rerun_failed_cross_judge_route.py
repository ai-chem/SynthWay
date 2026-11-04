#!/usr/bin/env python3
"""Safely rerun one cross-judge route and replace only its stored trajectory.

The replacement is transactional:

1. The route is rerun in a staging directory with the original experiment
   generator, judge, thresholds, RPM limit, and attempt budget.
2. Every staged attempt must contain all three metric scores.  A new parsing or
   evaluation failure aborts without touching the existing experiment.
3. The current route records and derived experiment outputs are backed up.
4. Only the target route is replaced in the arm JSONL files, then all combined
   tables are rebuilt from the six arms.
5. For the default v2 experiment, the three-run figures are refreshed.

Running without --yes performs validation only and makes no model calls or
filesystem changes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent
for _local_path in (SCRIPT_DIR,):
    if str(_local_path) not in sys.path:
        sys.path.insert(0, str(_local_path))
from _paths import locate_rag_llm_dir  # noqa: E402

RAG_LLM_DIR = locate_rag_llm_dir(SCRIPT_DIR)
PROJECT_ROOT = RAG_LLM_DIR.parent
if str(RAG_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_LLM_DIR))

import regenerate_cross_judge_expert_faithfulness as experiment  # noqa: E402
import _regeneration as base_regen  # noqa: E402


DEFAULT_OUT_ROOT = (
    RUNS_DIR / "repeat_2"
)
DEFAULT_ARM_KEY = "gen_nemotron_3_nano__judge_claude_opus_4_5"
DEFAULT_CASE_ID = "2108.json_PEG_coated_Fe3O4_3226"
DEFAULT_ROUTE_NAME = "route_1.json"
METRICS = ("chemical_geval", "answer_relevancy", "faithfulness")
REQUIRED_RUNTIME_MODULES = (
    "deepeval",
    "openai",
    "sentence_transformers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun one route in an existing cross-judge experiment and safely "
            "replace its attempts, route summary, combined tables, and plots."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--arm-key", default=DEFAULT_ARM_KEY)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--route-name", default=DEFAULT_ROUTE_NAME)
    parser.add_argument(
        "--rpm-limit",
        type=float,
        default=None,
        help="Override the arm RPM limit; by default arm_config.json is used.",
    )
    parser.add_argument(
        "--recovery-attempts",
        type=int,
        default=1,
        help=(
            "Maximum number of complete staged reruns if a rerun has missing "
            "metric scores or an unparseable regeneration (default: 1)."
        ),
    )
    parser.add_argument(
        "--continue-on-rate-limit",
        action="store_true",
        help="Do not abort immediately on an API rate-limit response.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Authorize model calls and replacement. Without this flag, validate only.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def missing_runtime_modules() -> list[str]:
    return [
        module_name
        for module_name in REQUIRED_RUNTIME_MODULES
        if importlib.util.find_spec(module_name) is None
    ]


def require_runtime_dependencies() -> None:
    missing = missing_runtime_modules()
    if not missing:
        return
    project_python = PROJECT_ROOT / "all_env" / "bin" / "python"
    if project_python.exists() and Path(sys.executable) != project_python:
        print(
            "Current Python is missing regeneration dependencies; restarting with:\n"
            f"  {project_python}",
            flush=True,
        )
        os.execv(
            str(project_python),
            [
                str(project_python),
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
        )
    suggestion = (
        f"Use this interpreter:\n  {project_python} rerun_failed_cross_judge_route.py "
        "--yes --recovery-attempts 3"
        if project_python.exists()
        else "Activate the same Python environment that was used for the original experiment."
    )
    raise RuntimeError(
        "The current Python environment cannot run regeneration. "
        f"Interpreter: {sys.executable}. Missing modules: {missing}.\n{suggestion}"
    )


def target_matches(row: dict[str, Any], case_id: str, route_name: str) -> bool:
    return str(row.get("case_id")) == case_id and str(row.get("route_name")) == route_name


def replace_target_rows(
    existing: Sequence[dict[str, Any]],
    replacement: Sequence[dict[str, Any]],
    *,
    case_id: str,
    route_name: str,
) -> list[dict[str, Any]]:
    target_indices = [
        index
        for index, row in enumerate(existing)
        if target_matches(row, case_id, route_name)
    ]
    if not target_indices:
        raise ValueError(f"Target {case_id}/{route_name} is absent from the existing JSONL file.")
    first_target = min(target_indices)
    insert_at = sum(
        1
        for row in existing[:first_target]
        if not target_matches(row, case_id, route_name)
    )
    retained = [
        row for row in existing if not target_matches(row, case_id, route_name)
    ]
    return retained[:insert_at] + list(replacement) + retained[insert_at:]


def atomic_write_json(path: Path, value: Any) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    experiment.write_json(temp_path, value)
    os.replace(temp_path, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    experiment.write_jsonl(temp_path, rows)
    os.replace(temp_path, path)


def validate_target(
    *,
    out_root: Path,
    arm_key: str,
    case_id: str,
    route_name: str,
) -> dict[str, Any]:
    manifest_path = out_root / "experiment_manifest.json"
    arm_dir = out_root / "arms" / arm_key
    arm_config_path = arm_dir / "arm_config.json"
    attempts_path = arm_dir / "attempts.jsonl"
    summaries_path = arm_dir / "per_route_summary.jsonl"
    route_path = arm_dir / "input_snapshot" / case_id / route_name
    context_path = arm_dir / "input_snapshot" / case_id / "retrieval_context.json"

    required = [
        manifest_path,
        arm_config_path,
        attempts_path,
        summaries_path,
        route_path,
        context_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required recovery inputs are missing: {missing}")

    manifest = experiment.read_json(manifest_path)
    arm_config = experiment.read_json(arm_config_path)
    if str(arm_config.get("arm_key")) != arm_key:
        raise ValueError("arm_config.json does not match --arm-key.")

    attempts = experiment.read_jsonl(attempts_path)
    summaries = experiment.read_jsonl(summaries_path)
    old_attempts = [
        row for row in attempts if target_matches(row, case_id, route_name)
    ]
    old_summaries = [
        row for row in summaries if target_matches(row, case_id, route_name)
    ]
    if not old_attempts:
        raise ValueError(f"No attempts found for {case_id}/{route_name}.")
    if len(old_summaries) != 1:
        raise ValueError(
            f"Expected exactly one route summary for {case_id}/{route_name}; "
            f"found {len(old_summaries)}."
        )

    generator = experiment.GeneratorSpec(**arm_config["generator"])
    judge = experiment.JudgeSpec(**arm_config["judge"])
    expected_arm_key = experiment._arm_key(generator, judge)
    if expected_arm_key != arm_key:
        raise ValueError(
            f"Generator/judge config resolves to {expected_arm_key}, not {arm_key}."
        )

    return {
        "manifest": manifest,
        "arm_config": arm_config,
        "arm_dir": arm_dir,
        "attempts_path": attempts_path,
        "summaries_path": summaries_path,
        "route_path": route_path,
        "case_dir": route_path.parent,
        "attempts": attempts,
        "summaries": summaries,
        "old_attempts": old_attempts,
        "old_summary": old_summaries[0],
        "generator": generator,
        "judge": judge,
    }


def attempt_missing_scores(attempt: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for metric in METRICS:
        metric_result = attempt.get(metric)
        score = metric_result.get("score") if isinstance(metric_result, dict) else None
        if not isinstance(score, (int, float)):
            missing.append(metric)
    return missing


def run_staged_trajectory(
    *,
    target: dict[str, Any],
    staging_root: Path,
    rpm_limit: float | None,
    stop_on_rate_limit: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    arm_config = target["arm_config"]
    generator: experiment.GeneratorSpec = target["generator"]
    judge: experiment.JudgeSpec = target["judge"]
    thresholds = arm_config["thresholds"]
    staged_attempts_path = staging_root / "attempts.jsonl"
    staged_regenerated_dir = staging_root / "regenerated" / target["case_dir"].name

    evaluator = experiment.CrossJudgeEvaluator(
        judge_model_id=judge.model_id,
        chemical_threshold=float(thresholds["chemical_geval"]),
        answer_threshold=float(thresholds["answer_relevancy"]),
        faithfulness_threshold=float(thresholds["faithfulness"]),
        rpm_limit=rpm_limit,
        stop_on_rate_limit=stop_on_rate_limit,
    )
    run_id = f"{arm_config['experiment_run_id']}:{arm_config['arm_key']}"
    try:
        summary = base_regen.process_one_route(
            run_id=run_id,
            evaluator=evaluator,
            regen_model=generator.model_id,
            case_id=target["case_dir"].name,
            case_dir=target["case_dir"],
            route_path=target["route_path"],
            out_attempts_jsonl=str(staged_attempts_path),
            out_routes_subdir=staged_regenerated_dir,
            max_attempts_total=int(arm_config["max_attempts_total"]),
            dry_run=False,
        )
    finally:
        experiment._close_judge(evaluator)

    attempts = experiment.read_jsonl(staged_attempts_path)
    if not attempts:
        raise RuntimeError("The staged rerun produced no attempt records.")
    wrong_routes = [
        row
        for row in attempts
        if not target_matches(row, target["case_dir"].name, target["route_path"].name)
    ]
    if wrong_routes:
        raise RuntimeError("The staged rerun wrote records for an unexpected route.")

    incomplete = [
        {
            "attempt_no": row.get("attempt_no"),
            "missing_metrics": attempt_missing_scores(row),
        }
        for row in attempts
        if attempt_missing_scores(row)
    ]
    if incomplete:
        raise RuntimeError(
            "The staged rerun still contains a technical failure and will not be committed: "
            f"{incomplete}"
        )
    return attempts, asdict(summary), staged_regenerated_dir


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def create_backup(
    *,
    out_root: Path,
    target: dict[str, Any],
    case_id: str,
    route_name: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = (
        out_root
        / "recovery_backups"
        / f"{timestamp}_{safe_slug(target['arm_config']['arm_key'])}_{safe_slug(case_id)}_{safe_slug(route_name)}"
    )
    if backup_root.exists():
        raise FileExistsError(f"Backup path already exists: {backup_root}")
    backup_root.mkdir(parents=True)

    relative_files = [
        target["attempts_path"].relative_to(out_root),
        target["summaries_path"].relative_to(out_root),
        (target["arm_dir"] / "summary.json").relative_to(out_root),
        (target["arm_dir"] / "summary.csv").relative_to(out_root),
        Path("experiment_manifest.json"),
        Path("experiment_summary.json"),
    ]
    for relative_path in relative_files:
        copy_if_exists(out_root / relative_path, backup_root / relative_path)
    for directory in ("combined", "tables", "plots"):
        copy_if_exists(out_root / directory, backup_root / directory)

    generated_dir = target["arm_dir"] / "regenerated" / case_id
    artifact_backup_dir = backup_root / "target_generated_artifacts"
    artifact_backup_dir.mkdir(parents=True)
    route_stem = Path(route_name).stem
    for artifact in sorted(generated_dir.glob(f"{route_stem}_attempt*")):
        if artifact.is_file():
            shutil.copy2(artifact, artifact_backup_dir / artifact.name)

    backup_manifest = {
        "created_utc": utc_now(),
        "out_root": str(out_root),
        "arm_key": target["arm_config"]["arm_key"],
        "case_id": case_id,
        "route_name": route_name,
        "old_attempt_count": len(target["old_attempts"]),
        "old_summary": target["old_summary"],
        "checksums": {
            str(relative_path): sha256(out_root / relative_path)
            for relative_path in relative_files
        },
    }
    experiment.write_json(backup_root / "backup_manifest.json", backup_manifest)
    return backup_root


def restore_backup(
    *,
    out_root: Path,
    target: dict[str, Any],
    backup_root: Path,
    case_id: str,
    route_name: str,
) -> None:
    relative_files = [
        target["attempts_path"].relative_to(out_root),
        target["summaries_path"].relative_to(out_root),
        (target["arm_dir"] / "summary.json").relative_to(out_root),
        (target["arm_dir"] / "summary.csv").relative_to(out_root),
        Path("experiment_manifest.json"),
        Path("experiment_summary.json"),
    ]
    for relative_path in relative_files:
        backup_file = backup_root / relative_path
        if backup_file.is_file():
            (out_root / relative_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_file, out_root / relative_path)
    for directory in ("combined", "tables", "plots"):
        backup_dir = backup_root / directory
        destination = out_root / directory
        if backup_dir.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(backup_dir, destination)

    generated_dir = target["arm_dir"] / "regenerated" / case_id
    generated_dir.mkdir(parents=True, exist_ok=True)
    route_stem = Path(route_name).stem
    for artifact in generated_dir.glob(f"{route_stem}_attempt*"):
        if artifact.is_file():
            artifact.unlink()
    artifact_backup_dir = backup_root / "target_generated_artifacts"
    if artifact_backup_dir.is_dir():
        for artifact in artifact_backup_dir.iterdir():
            if artifact.is_file():
                shutil.copy2(artifact, generated_dir / artifact.name)


def experiment_arms_from_manifest(
    manifest: dict[str, Any],
) -> list[tuple[experiment.GeneratorSpec, experiment.JudgeSpec]]:
    generators = {
        item["key"]: experiment.GeneratorSpec(**item)
        for item in manifest.get("generators", [])
    }
    judges = {
        item["key"]: experiment.JudgeSpec(**item)
        for item in manifest.get("judges", [])
    }
    arms: list[tuple[experiment.GeneratorSpec, experiment.JudgeSpec]] = []
    for arm in manifest.get("arms", []):
        generator_key = arm.get("generator_key")
        judge_key = arm.get("judge_key")
        if generator_key not in generators or judge_key not in judges:
            raise ValueError(f"Manifest arm references unknown models: {arm}")
        pair = (generators[generator_key], judges[judge_key])
        if experiment._arm_key(*pair) != arm.get("arm_key"):
            raise ValueError(f"Manifest arm key is inconsistent: {arm}")
        arms.append(pair)
    if not arms:
        raise ValueError("No experiment arms found in experiment_manifest.json.")
    return arms


def rebuild_arm_summary(target: dict[str, Any]) -> dict[str, Any]:
    arm_config = target["arm_config"]
    summaries = experiment._route_summary_objects(target["summaries_path"])
    summary = base_regen.aggregate_summaries(summaries)
    summary.update(
        {
            "experiment_run_id": arm_config["experiment_run_id"],
            "arm_key": arm_config["arm_key"],
            "generator": arm_config["generator"],
            "judge": arm_config["judge"],
            "ts_utc": experiment.utc_now_iso(),
            "config": arm_config,
        }
    )
    experiment.write_json(target["arm_dir"] / "summary.json", summary)
    base_regen.write_summary_csv(str(target["arm_dir"] / "summary.csv"), summary)
    return summary


def rewrite_staged_artifact_paths(
    attempts: Sequence[dict[str, Any]],
    *,
    production_generated_dir: Path,
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for row in attempts:
        updated = dict(row)
        attempt_no = int(updated.get("attempt_no", 0))
        if attempt_no > 1:
            staged_path = Path(str(updated.get("route_json_path") or ""))
            updated["route_json_path"] = str(production_generated_dir / staged_path.name)
        rewritten.append(updated)
    return rewritten


def commit_replacement(
    *,
    out_root: Path,
    target: dict[str, Any],
    staged_attempts: Sequence[dict[str, Any]],
    staged_summary: dict[str, Any],
    staged_regenerated_dir: Path,
    backup_root: Path,
    case_id: str,
    route_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    production_generated_dir = target["arm_dir"] / "regenerated" / case_id
    production_generated_dir.mkdir(parents=True, exist_ok=True)
    route_stem = Path(route_name).stem
    for artifact in production_generated_dir.glob(f"{route_stem}_attempt*"):
        if artifact.is_file():
            artifact.unlink()
    if staged_regenerated_dir.is_dir():
        for artifact in staged_regenerated_dir.iterdir():
            if artifact.is_file():
                shutil.copy2(artifact, production_generated_dir / artifact.name)

    rewritten_attempts = rewrite_staged_artifact_paths(
        staged_attempts,
        production_generated_dir=production_generated_dir,
    )
    merged_attempts = replace_target_rows(
        target["attempts"],
        rewritten_attempts,
        case_id=case_id,
        route_name=route_name,
    )
    merged_summaries = replace_target_rows(
        target["summaries"],
        [staged_summary],
        case_id=case_id,
        route_name=route_name,
    )
    atomic_write_jsonl(target["attempts_path"], merged_attempts)
    atomic_write_jsonl(target["summaries_path"], merged_summaries)

    arm_summary = rebuild_arm_summary(target)
    manifest = experiment.read_json(out_root / "experiment_manifest.json")
    arms = experiment_arms_from_manifest(manifest)
    combined = experiment.collect_combined_results(
        out_root=out_root,
        arms=arms,
        thresholds={key: float(value) for key, value in manifest["thresholds"].items()},
    )
    atomic_write_json(out_root / "experiment_summary.json", combined)

    recovery_record = {
        "completed_utc": utc_now(),
        "arm_key": target["arm_config"]["arm_key"],
        "case_id": case_id,
        "route_name": route_name,
        "backup_dir": str(backup_root),
        "old_attempt_count": len(target["old_attempts"]),
        "new_attempt_count": len(staged_attempts),
        "old_summary": target["old_summary"],
        "new_summary": staged_summary,
    }
    recoveries = manifest.get("route_recoveries")
    if not isinstance(recoveries, list):
        recoveries = []
    recoveries.append(recovery_record)
    manifest["route_recoveries"] = recoveries
    manifest["status"] = "completed"
    manifest["last_route_recovery_utc"] = recovery_record["completed_utc"]
    manifest["combined_summary"] = combined
    arm_summaries = manifest.get("arm_summaries")
    if isinstance(arm_summaries, list):
        manifest["arm_summaries"] = [
            arm_summary if row.get("arm_key") == target["arm_config"]["arm_key"] else row
            for row in arm_summaries
        ]
    manifest.pop("error", None)
    manifest.pop("failed_utc", None)
    atomic_write_json(out_root / "experiment_manifest.json", manifest)
    return arm_summary, combined


def main() -> None:
    args = parse_args()
    if args.recovery_attempts < 1:
        raise ValueError("--recovery-attempts must be >= 1")
    out_root = args.out_dir.expanduser().resolve()
    target = validate_target(
        out_root=out_root,
        arm_key=args.arm_key,
        case_id=args.case_id,
        route_name=args.route_name,
    )
    arm_config = target["arm_config"]
    rpm_limit = (
        float(args.rpm_limit)
        if args.rpm_limit is not None
        else (
            float(arm_config["rpm_limit"])
            if arm_config.get("rpm_limit") is not None
            else None
        )
    )

    print("Validated targeted recovery:")
    print(f"  experiment: {out_root}")
    print(f"  arm:        {args.arm_key}")
    print(f"  route:      {args.case_id}/{args.route_name}")
    print(f"  generator:  {target['generator'].model_id}")
    print(f"  judge:      {target['judge'].model_id}")
    print(f"  thresholds: {arm_config['thresholds']}")
    print(f"  max regens: {arm_config['max_regens']}")
    print(f"  RPM limit:  {rpm_limit}")
    print(f"  old attempts: {len(target['old_attempts'])}")
    print(f"  old result:   {target['old_summary']['final_status']}")

    if not args.yes:
        missing = missing_runtime_modules()
        if missing:
            print(
                "Runtime warning: the current Python cannot perform the rerun; "
                f"missing modules: {missing}."
            )
        print("Validation only: no model calls were made and no files were changed.")
        print("Run again with --yes to stage and commit the replacement.")
        return

    # Fail before initializing a judge or making any API/model call.  This also
    # prevents --recovery-attempts from repeating the same environment error.
    require_runtime_dependencies()

    accepted: tuple[list[dict[str, Any]], dict[str, Any], Path] | None = None
    staging_parent = target["arm_dir"] / "route_recovery_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rerun_", dir=staging_parent) as temp_dir:
        session_root = Path(temp_dir)
        failures: list[str] = []
        for recovery_index in range(1, args.recovery_attempts + 1):
            attempt_root = session_root / f"recovery_attempt_{recovery_index}"
            attempt_root.mkdir(parents=True)
            print(f"Staged rerun {recovery_index}/{args.recovery_attempts}...")
            try:
                accepted = run_staged_trajectory(
                    target=target,
                    staging_root=attempt_root,
                    rpm_limit=rpm_limit,
                    stop_on_rate_limit=not args.continue_on_rate_limit,
                )
                break
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                print(f"Staged rerun rejected: {failures[-1]}")

        if accepted is None:
            raise RuntimeError(
                "No technically complete rerun was produced. Existing v2 results were not changed. "
                f"Failures: {failures}"
            )

        staged_attempts, staged_summary, staged_regenerated_dir = accepted
        backup_root = create_backup(
            out_root=out_root,
            target=target,
            case_id=args.case_id,
            route_name=args.route_name,
        )
        try:
            arm_summary, combined = commit_replacement(
                out_root=out_root,
                target=target,
                staged_attempts=staged_attempts,
                staged_summary=staged_summary,
                staged_regenerated_dir=staged_regenerated_dir,
                backup_root=backup_root,
                case_id=args.case_id,
                route_name=args.route_name,
            )
        except Exception:
            restore_backup(
                out_root=out_root,
                target=target,
                backup_root=backup_root,
                case_id=args.case_id,
                route_name=args.route_name,
            )
            raise

    print("Targeted recovery complete.")
    print(f"Backup: {backup_root}")
    print(f"New attempts: {staged_summary['attempts_used']}")
    print(f"New status: {staged_summary['final_status']}")
    print(f"Arm final pass rate: {arm_summary['pass_final_rate']:.3f}")
    print(f"Experiment routes: {combined['total_route_trajectories']}")


if __name__ == "__main__":
    main()
