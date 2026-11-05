"""Cross-judge regeneration experiment with synthesis-specific Faithfulness.

Evaluates routes with Chemical Plausibility & Consistency, Answer Relevancy,
and the study's synthesis-specific Faithfulness implementation.

The default experiment uses the 10 expert-selected routes for each surviving
generator (Claude Opus 4.5, Llama 4 Maverick, Nemotron 3 Nano) and evaluates
each cohort with the other two models. Gemini 2.0 is intentionally absent.

Outputs are written both per generator/judge arm and as combined long/wide
tables suitable for plotting and mixed-effects/statistical analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _paths import locate_rag_llm_dir  # noqa: E402

RAG_LLM_DIR = locate_rag_llm_dir(SCRIPT_DIR)
BENCH_DIR = RAG_LLM_DIR / "bench"
for _local_path in (SCRIPT_DIR, RAG_LLM_DIR, BENCH_DIR):
    if str(_local_path) not in sys.path:
        sys.path.insert(0, str(_local_path))

import _regeneration as base_regen  # noqa: E402

DEFAULT_CONFIG_PATH = RAG_LLM_DIR.parent / "config" / "regeneration.json"
DEFAULT_SELECTED_ROUTES_ROOT = (
    BENCH_DIR / "diff_judges_for_exp_bench" / "selected_40_routes_for_expert_eval"
)
DEFAULT_OUT_DIR = RAG_LLM_DIR.parent / "outputs" / "regeneration"
METRICS = ("chemical_geval", "answer_relevancy", "faithfulness")


@dataclass(frozen=True)
class GeneratorSpec:
    key: str
    label: str
    model_id: str
    routes_subdir: str
    expected_routes: int = 10


@dataclass(frozen=True)
class JudgeSpec:
    key: str
    label: str
    model_id: str


@dataclass
class DetailedMetricResult(base_regen.MetricResult):
    """Metric result with auditable expert-style Faithfulness internals."""

    details: dict[str, Any] | None = None


from ss_faithfulness import (
    RateLimitedJudgeProxy as _RateLimitedJudgeProxy,
)
from ss_faithfulness import (  # shared with the desktop application and live demo
    SSFaithfulnessMetric as ExpertFaithfulnessMetricAdapter,
)


class CrossJudgeEvaluator(base_regen.Evaluator):
    """Evaluate CPC, Answer Relevancy, and synthesis-specific Faithfulness."""

    def init_models(self) -> None:
        if self._judge_model is None:
            # Reuse the SS-Faithfulness judge builder.
            from bench.new_faith_same_as_expert_logic import build_judge_model

            self._judge_model = build_judge_model(self.judge_model_id)
            if self._judge_model is None:
                raise RuntimeError(
                    "An explicit judge model is required for expert-style Faithfulness."
                )

    def _measure_metric(
        self,
        metric: Any,
        test_case: Any,
        metric_name: str,
        *,
        external_rate_limit: bool = True,
    ) -> base_regen.MetricResult:
        max_attempts = 3
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                if external_rate_limit:
                    self._rate_limit_sleep_if_needed()
                metric.measure(test_case)
                score = getattr(metric, "score", None)
                reason = str(getattr(metric, "reason", "") or "")
                threshold = float(getattr(metric, "threshold", 0.0) or 0.0)
                details = getattr(metric, "details", None)
                return DetailedMetricResult(
                    score=float(score) if score is not None else None,
                    threshold=threshold,
                    passed=score is not None and float(score) >= threshold,
                    reason=reason,
                    details=details if isinstance(details, dict) else None,
                )
            except Exception as exc:  # Retry failed metric calls up to three times.
                last_error = exc
                message = str(exc).lower()
                is_rate_limit = any(
                    x in message for x in ("429", "rate limit", "too many requests")
                )
                if is_rate_limit and self.stop_on_rate_limit:
                    raise
                time.sleep(min(8.0, 2.0 * attempt))

        threshold = float(getattr(metric, "threshold", 0.0) or 0.0)
        return DetailedMetricResult(
            score=None,
            threshold=threshold,
            passed=False,
            reason=f"ERROR during {metric_name}: {type(last_error).__name__}: {last_error}",
            details=None,
        )

    def evaluate(
        self,
        *,
        composition: str,
        method: str | None,
        retrieval_context: list[str],
        retrieval_text: str,
        raw_text: str,
    ) -> tuple[base_regen.MetricResult, base_regen.MetricResult, base_regen.MetricResult, bool]:
        self.init_models()
        judge = self._judge_model

        from bench.new_faith_same_as_expert_logic import (
            build_test_case_from_parts as build_expert_faithfulness_test_case,
        )
        from evaluate_answer_faithfulness_deepeval import (
            build_test_case_from_parts as build_answer_test_case,
        )
        from evaluate_answer_faithfulness_deepeval import (
            make_answer_relevancy_metric,
        )
        from evaluate_generation_deepeval import build_test_case, make_chemical_geval_metric

        route_obj = {
            "composition": composition,
            "method": method,
            "retrieval_text": retrieval_text or "",
            "raw_text": raw_text or "",
        }

        chemical_metric = make_chemical_geval_metric(
            threshold=self.chemical_threshold,
            judge_model=judge,
        )
        chemical = self._measure_metric(
            chemical_metric,
            build_test_case(route_obj),
            "Chemical G-Eval (CPC)",
        )

        answer_metric = make_answer_relevancy_metric(
            threshold=self.answer_threshold,
            judge_model=judge,
        )
        answer_case = build_answer_test_case(
            composition=composition,
            method=method,
            raw_text=raw_text or "",
            retrieval_text=retrieval_text or "",
            retrieval_context=retrieval_context or [],
        )
        answer = self._measure_metric(answer_metric, answer_case, "Answer Relevancy")

        # The expert-style metric performs four judge calls: claims, truths,
        # verdicts, and Q1-Q4 claim classification. Rate-limit every one.
        limited_judge = _RateLimitedJudgeProxy(judge, self._rate_limit_sleep_if_needed)
        faith_metric = ExpertFaithfulnessMetricAdapter(
            threshold=self.faithfulness_threshold,
            judge_model=limited_judge,
        )
        faith_case = build_expert_faithfulness_test_case(
            composition=composition,
            method=method,
            raw_text=raw_text or "",
            retrieval_text=retrieval_text or "",
            retrieval_context=retrieval_context or [],
        )
        faith = self._measure_metric(
            faith_metric,
            faith_case,
            "Synthesis-specific expert-style Faithfulness",
            external_rate_limit=False,
        )

        passed_all = bool(chemical.passed and answer.passed and faith.passed)
        return chemical, answer, faith, passed_all


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if not isinstance(config.get("generators"), list) or not config["generators"]:
        raise ValueError("Config must contain a non-empty 'generators' list.")
    if not isinstance(config.get("judges"), list) or not config["judges"]:
        raise ValueError("Config must contain a non-empty 'judges' list.")
    return config


def parse_specs(config: dict[str, Any]) -> tuple[list[GeneratorSpec], list[JudgeSpec]]:
    generators = [GeneratorSpec(**item) for item in config["generators"]]
    judges = [JudgeSpec(**item) for item in config["judges"]]
    _require_unique_keys([x.key for x in generators], "generator")
    _require_unique_keys([x.key for x in judges], "judge")
    return generators, judges


def _require_unique_keys(keys: Sequence[str], role: str) -> None:
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate {role} keys: {duplicates}")


def select_specs(items: Sequence[Any], selected_keys: Sequence[str], role: str) -> list[Any]:
    if not selected_keys:
        return list(items)
    wanted = set(selected_keys)
    known = {item.key for item in items}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown {role} key(s): {unknown}; available: {sorted(known)}")
    return [item for item in items if item.key in wanted]


def build_arms(
    generators: Sequence[GeneratorSpec],
    judges: Sequence[JudgeSpec],
    *,
    include_self_judge: bool,
) -> list[tuple[GeneratorSpec, JudgeSpec]]:
    return [
        (generator, judge)
        for generator in generators
        for judge in judges
        if include_self_judge or generator.key != judge.key
    ]


def discover_source_routes(routes_dir: Path) -> list[tuple[Path, Path]]:
    if not routes_dir.is_dir():
        raise FileNotFoundError(f"Generator routes directory not found: {routes_dir}")
    pairs: list[tuple[Path, Path]] = []
    for case_dir in base_regen.list_case_dirs(str(routes_dir)):
        for route_path in base_regen.list_route_files(case_dir):
            pairs.append((case_dir, route_path))
    return sorted(pairs, key=lambda item: (item[0].name, item[1].name))


def prepare_input_snapshot(
    *,
    source_routes_dir: Path,
    snapshot_root: Path,
    expected_routes: int,
    max_routes: int | None,
) -> list[dict[str, Any]]:
    """Copy inputs exactly; validate them without changing experimental fields."""

    source_pairs = discover_source_routes(source_routes_dir)
    if len(source_pairs) != expected_routes and max_routes is None:
        raise ValueError(
            f"Expected exactly {expected_routes} selected routes in {source_routes_dir}, "
            f"found {len(source_pairs)}. Refusing to silently change the design."
        )
    if max_routes is not None:
        if max_routes < 1:
            raise ValueError("--max-routes-per-arm must be >= 1")
        source_pairs = source_pairs[:max_routes]

    manifest: list[dict[str, Any]] = []
    for source_case_dir, source_route_path in source_pairs:
        route = read_json(source_route_path)
        retrieval_text = str(route.get("retrieval_text") or "")
        source_meta_path = source_case_dir / "retrieval_context.json"
        if not source_meta_path.exists():
            raise FileNotFoundError(f"Missing retrieval_context.json for {source_route_path}")
        meta = read_json(source_meta_path)
        retrieval_context = meta.get("retrieval_context") or []
        if not isinstance(retrieval_context, list) or not retrieval_context:
            raise ValueError(f"Empty retrieval_context for {source_route_path}")

        method_source = "retrieval_context.json (preserved as-is; null allowed)"
        composition_source = "retrieval_context.json (preserved as-is)"
        if not str(meta.get("composition") or "").strip():
            raise ValueError(f"Missing composition for {source_route_path}")

        if not str(route.get("raw_text") or "").strip():
            raise ValueError(f"Empty raw_text in selected route {source_route_path}")

        snapshot_case_dir = snapshot_root / source_case_dir.name
        snapshot_route_path = snapshot_case_dir / source_route_path.name
        write_json(snapshot_case_dir / "retrieval_context.json", meta)
        write_json(snapshot_route_path, route)

        manifest.append(
            {
                "case_id": source_case_dir.name,
                "route_name": source_route_path.name,
                "source_route_path": str(source_route_path.resolve()),
                "source_retrieval_context_path": str(source_meta_path.resolve()),
                "snapshot_route_path": str(snapshot_route_path.resolve()),
                "composition": meta.get("composition"),
                "composition_source": composition_source,
                "method": meta.get("method"),
                "method_source": method_source,
                "retrieval_context_count": len(retrieval_context),
            }
        )
    return manifest


def _arm_key(generator: GeneratorSpec, judge: JudgeSpec) -> str:
    return f"gen_{generator.key}__judge_{judge.key}"


def _close_judge(evaluator: CrossJudgeEvaluator) -> None:
    judge = getattr(evaluator, "_judge_model", None)
    if judge is None:
        return
    close_fn = getattr(judge, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def _route_summary_objects(path: Path) -> list[base_regen.RouteSummary]:
    return [base_regen.RouteSummary(**row) for row in read_jsonl(path)]


def run_arm(
    *,
    experiment_run_id: str,
    generator: GeneratorSpec,
    judge: JudgeSpec,
    selected_root: Path,
    out_root: Path,
    thresholds: dict[str, float],
    max_regens: int,
    rpm_limit: float | None,
    stop_on_rate_limit: bool,
    resume: bool,
    max_routes: int | None,
) -> dict[str, Any]:
    arm_key = _arm_key(generator, judge)
    arm_dir = out_root / "arms" / arm_key
    attempts_path = arm_dir / "attempts.jsonl"
    route_summaries_path = arm_dir / "per_route_summary.jsonl"
    source_routes_dir = selected_root / generator.routes_subdir

    existing_arm_config = arm_dir / "arm_config.json"
    if not resume and (attempts_path.exists() or route_summaries_path.exists()):
        raise FileExistsError(
            f"Arm already has results: {arm_dir}. Use --resume or choose a new --out-dir."
        )

    snapshot_manifest = prepare_input_snapshot(
        source_routes_dir=source_routes_dir,
        snapshot_root=arm_dir / "input_snapshot",
        expected_routes=generator.expected_routes,
        max_routes=max_routes,
    )
    write_json(arm_dir / "source_manifest.json", snapshot_manifest)

    arm_config = {
        "experiment_run_id": experiment_run_id,
        "arm_key": arm_key,
        "generator": asdict(generator),
        "judge": asdict(judge),
        "source_routes_dir": str(source_routes_dir.resolve()),
        "thresholds": thresholds,
        "max_regens": max_regens,
        "max_attempts_total": 1 + max_regens,
        "rpm_limit": rpm_limit,
        "route_count": len(snapshot_manifest),
        "metric_implementations": {
            "chemical_geval": "evaluate_generation_deepeval.make_chemical_geval_metric",
            "answer_relevancy": "evaluate_answer_faithfulness_deepeval.make_answer_relevancy_metric",
            "faithfulness": "bench.new_faith_same_as_expert_logic.measure_expert_faithfulness",
        },
    }
    if resume and existing_arm_config.exists():
        previous = read_json(existing_arm_config)
        comparable_keys = ("generator", "judge", "source_routes_dir", "thresholds", "max_regens")
        mismatches = [key for key in comparable_keys if previous.get(key) != arm_config.get(key)]
        if mismatches:
            raise ValueError(f"Cannot resume {arm_key}; configuration changed: {mismatches}")
    write_json(existing_arm_config, arm_config)

    done: set[tuple[str, str]] = set()
    if resume:
        for row in read_jsonl(route_summaries_path):
            done.add((str(row.get("case_id")), str(row.get("route_name"))))

    evaluator = CrossJudgeEvaluator(
        judge_model_id=judge.model_id,
        chemical_threshold=float(thresholds["chemical_geval"]),
        answer_threshold=float(thresholds["answer_relevancy"]),
        faithfulness_threshold=float(thresholds["faithfulness"]),
        rpm_limit=rpm_limit,
        stop_on_rate_limit=stop_on_rate_limit,
    )

    arm_run_id = f"{experiment_run_id}:{arm_key}"
    try:
        for case_dir in base_regen.list_case_dirs(str(arm_dir / "input_snapshot")):
            for route_path in base_regen.list_route_files(case_dir):
                if (case_dir.name, route_path.name) in done:
                    continue
                route_summary = base_regen.process_one_route(
                    run_id=arm_run_id,
                    evaluator=evaluator,
                    regen_model=generator.model_id,
                    case_id=case_dir.name,
                    case_dir=case_dir,
                    route_path=route_path,
                    out_attempts_jsonl=str(attempts_path),
                    out_routes_subdir=arm_dir / "regenerated" / case_dir.name,
                    max_attempts_total=1 + max_regens,
                    dry_run=False,
                )
                base_regen.append_jsonl(str(route_summaries_path), asdict(route_summary))
    finally:
        _close_judge(evaluator)

    all_summaries = _route_summary_objects(route_summaries_path)
    summary = base_regen.aggregate_summaries(all_summaries)
    summary.update(
        {
            "experiment_run_id": experiment_run_id,
            "arm_key": arm_key,
            "generator": asdict(generator),
            "judge": asdict(judge),
            "ts_utc": utc_now_iso(),
            "config": arm_config,
        }
    )
    write_json(arm_dir / "summary.json", summary)
    base_regen.write_summary_csv(str(arm_dir / "summary.csv"), summary)
    return summary


def _score(row: dict[str, Any], stage: str, metric: str) -> float | None:
    scores = row.get(stage) or {}
    value = scores.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _rate(flags: Iterable[bool]) -> float | None:
    values = list(flags)
    return sum(bool(value) for value in values) / len(values) if values else None


def _all_metrics_pass(row: dict[str, Any], stage: str, thresholds: dict[str, float]) -> bool:
    return all(
        (_score(row, stage, metric) is not None)
        and float(_score(row, stage, metric)) >= float(thresholds[metric])
        for metric in METRICS
    )


def _metric_pass(score: float | None, threshold: float) -> bool:
    return score is not None and float(score) >= float(threshold)


def collect_combined_results(
    *,
    out_root: Path,
    arms: Sequence[tuple[GeneratorSpec, JudgeSpec]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    all_attempts: list[dict[str, Any]] = []
    all_routes: list[dict[str, Any]] = []

    for generator, judge in arms:
        arm_key = _arm_key(generator, judge)
        arm_dir = out_root / "arms" / arm_key
        # source_manifest.json is a list; read it without the object-only helper.
        source_manifest_path = arm_dir / "source_manifest.json"
        source_rows: list[dict[str, Any]] = []
        if source_manifest_path.exists():
            with source_manifest_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                source_rows = [row for row in loaded if isinstance(row, dict)]
        source_manifest = {
            (str(row.get("case_id")), str(row.get("route_name"))): row for row in source_rows
        }

        shared = {
            "arm_key": arm_key,
            "generator_key": generator.key,
            "generator_label": generator.label,
            "generator_model_id": generator.model_id,
            "judge_key": judge.key,
            "judge_label": judge.label,
            "judge_model_id": judge.model_id,
        }
        for attempt in read_jsonl(arm_dir / "attempts.jsonl"):
            key = (str(attempt.get("case_id")), str(attempt.get("route_name")))
            all_attempts.append(
                {
                    **shared,
                    "source_route_path": source_manifest.get(key, {}).get("source_route_path"),
                    **attempt,
                }
            )
        for route in read_jsonl(arm_dir / "per_route_summary.jsonl"):
            key = (str(route.get("case_id")), str(route.get("route_name")))
            all_routes.append(
                {
                    **shared,
                    "source_route_path": source_manifest.get(key, {}).get("source_route_path"),
                    **route,
                }
            )

    write_jsonl(out_root / "combined" / "all_attempts.jsonl", all_attempts)
    write_jsonl(out_root / "combined" / "all_route_summaries.jsonl", all_routes)

    route_effects = build_route_effect_rows(all_routes, thresholds)
    metric_effects = build_metric_effect_rows(all_routes, thresholds)
    attempts_long = build_attempt_long_rows(all_attempts)
    pair_summary = build_pair_summary_rows(all_routes, arms, thresholds)
    generator_effects = build_role_effect_rows(metric_effects, "generator")
    judge_effects = build_role_effect_rows(metric_effects, "judge")
    disagreement = build_judge_disagreement_rows(metric_effects)

    write_csv(
        out_root / "tables" / "route_effects.csv",
        route_effects,
        route_effect_fieldnames(),
    )
    write_csv(
        out_root / "tables" / "metric_effects_long.csv",
        metric_effects,
        metric_effect_fieldnames(),
    )
    write_csv(
        out_root / "tables" / "attempt_trajectories_long.csv",
        attempts_long,
        attempt_long_fieldnames(),
    )
    write_csv(
        out_root / "tables" / "pair_summary.csv",
        pair_summary,
        pair_summary_fieldnames(),
    )
    write_csv(
        out_root / "tables" / "generator_effects.csv",
        generator_effects,
        role_effect_fieldnames("generator"),
    )
    write_csv(
        out_root / "tables" / "judge_effects.csv",
        judge_effects,
        role_effect_fieldnames("judge"),
    )
    write_csv(
        out_root / "tables" / "judge_disagreement.csv",
        disagreement,
        disagreement_fieldnames(),
    )

    return {
        "total_arms": len(arms),
        "total_route_trajectories": len(all_routes),
        "total_attempts": len(all_attempts),
        "pair_summary": pair_summary,
        "files": {
            "all_attempts_jsonl": "combined/all_attempts.jsonl",
            "all_route_summaries_jsonl": "combined/all_route_summaries.jsonl",
            "route_effects_csv": "tables/route_effects.csv",
            "metric_effects_long_csv": "tables/metric_effects_long.csv",
            "attempt_trajectories_long_csv": "tables/attempt_trajectories_long.csv",
            "pair_summary_csv": "tables/pair_summary.csv",
            "generator_effects_csv": "tables/generator_effects.csv",
            "judge_effects_csv": "tables/judge_effects.csv",
            "judge_disagreement_csv": "tables/judge_disagreement.csv",
        },
    }


def build_route_effect_rows(
    routes: Sequence[dict[str, Any]], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in routes:
        row: dict[str, Any] = {
            key: route.get(key)
            for key in (
                "arm_key",
                "generator_key",
                "generator_label",
                "generator_model_id",
                "judge_key",
                "judge_label",
                "judge_model_id",
                "case_id",
                "route_name",
                "source_route_path",
                "attempts_used",
                "best_attempt_no",
                "final_status",
            )
        }
        row["initial_pass_all"] = _all_metrics_pass(route, "initial_scores", thresholds)
        row["final_pass_all"] = _all_metrics_pass(route, "final_scores", thresholds)
        row["resolved_by_regeneration"] = bool(
            row["final_pass_all"] and not row["initial_pass_all"]
        )
        for metric in METRICS:
            initial = _score(route, "initial_scores", metric)
            final = _score(route, "final_scores", metric)
            delta = None if initial is None or final is None else final - initial
            row[f"initial_{metric}"] = initial
            row[f"final_{metric}"] = final
            row[f"delta_{metric}"] = delta
            row[f"initial_pass_{metric}"] = _metric_pass(initial, thresholds[metric])
            row[f"final_pass_{metric}"] = _metric_pass(final, thresholds[metric])
        rows.append(row)
    return rows


def route_effect_fieldnames() -> list[str]:
    fields = [
        "arm_key",
        "generator_key",
        "generator_label",
        "generator_model_id",
        "judge_key",
        "judge_label",
        "judge_model_id",
        "case_id",
        "route_name",
        "source_route_path",
        "attempts_used",
        "best_attempt_no",
        "final_status",
        "initial_pass_all",
        "final_pass_all",
        "resolved_by_regeneration",
    ]
    for metric in METRICS:
        fields.extend(
            [
                f"initial_{metric}",
                f"final_{metric}",
                f"delta_{metric}",
                f"initial_pass_{metric}",
                f"final_pass_{metric}",
            ]
        )
    return fields


def build_metric_effect_rows(
    routes: Sequence[dict[str, Any]], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in routes:
        for metric in METRICS:
            initial = _score(route, "initial_scores", metric)
            final = _score(route, "final_scores", metric)
            delta = None if initial is None or final is None else final - initial
            rows.append(
                {
                    "arm_key": route.get("arm_key"),
                    "generator_key": route.get("generator_key"),
                    "generator_label": route.get("generator_label"),
                    "generator_model_id": route.get("generator_model_id"),
                    "judge_key": route.get("judge_key"),
                    "judge_label": route.get("judge_label"),
                    "judge_model_id": route.get("judge_model_id"),
                    "case_id": route.get("case_id"),
                    "route_name": route.get("route_name"),
                    "source_route_path": route.get("source_route_path"),
                    "metric": metric,
                    "threshold": thresholds[metric],
                    "initial_score": initial,
                    "final_score": final,
                    "delta": delta,
                    "initial_pass": _metric_pass(initial, thresholds[metric]),
                    "final_pass": _metric_pass(final, thresholds[metric]),
                    "improved": delta is not None and delta > 0,
                    "regressed": delta is not None and delta < 0,
                    "attempts_used": route.get("attempts_used"),
                    "best_attempt_no": route.get("best_attempt_no"),
                }
            )
    return rows


def metric_effect_fieldnames() -> list[str]:
    return [
        "arm_key",
        "generator_key",
        "generator_label",
        "generator_model_id",
        "judge_key",
        "judge_label",
        "judge_model_id",
        "case_id",
        "route_name",
        "source_route_path",
        "metric",
        "threshold",
        "initial_score",
        "final_score",
        "delta",
        "initial_pass",
        "final_pass",
        "improved",
        "regressed",
        "attempts_used",
        "best_attempt_no",
    ]


def build_attempt_long_rows(attempts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        for metric in METRICS:
            metric_obj = attempt.get(metric) or {}
            rows.append(
                {
                    "arm_key": attempt.get("arm_key"),
                    "generator_key": attempt.get("generator_key"),
                    "generator_label": attempt.get("generator_label"),
                    "generator_model_id": attempt.get("generator_model_id"),
                    "judge_key": attempt.get("judge_key"),
                    "judge_label": attempt.get("judge_label"),
                    "judge_model_id": attempt.get("judge_model_id"),
                    "case_id": attempt.get("case_id"),
                    "route_name": attempt.get("route_name"),
                    "source_route_path": attempt.get("source_route_path"),
                    "attempt_no": attempt.get("attempt_no"),
                    "attempt_kind": attempt.get("attempt_kind"),
                    "parent_source": attempt.get("parent_source"),
                    "parent_attempt_no": attempt.get("parent_attempt_no"),
                    "metric": metric,
                    "score": metric_obj.get("score"),
                    "threshold": metric_obj.get("threshold"),
                    "metric_passed": metric_obj.get("passed"),
                    "passed_all": attempt.get("passed_all"),
                }
            )
    return rows


def attempt_long_fieldnames() -> list[str]:
    return [
        "arm_key",
        "generator_key",
        "generator_label",
        "generator_model_id",
        "judge_key",
        "judge_label",
        "judge_model_id",
        "case_id",
        "route_name",
        "source_route_path",
        "attempt_no",
        "attempt_kind",
        "parent_source",
        "parent_attempt_no",
        "metric",
        "score",
        "threshold",
        "metric_passed",
        "passed_all",
    ]


def build_pair_summary_rows(
    routes: Sequence[dict[str, Any]],
    arms: Sequence[tuple[GeneratorSpec, JudgeSpec]],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generator, judge in arms:
        arm_key = _arm_key(generator, judge)
        arm_routes = [route for route in routes if route.get("arm_key") == arm_key]
        row: dict[str, Any] = {
            "arm_key": arm_key,
            "generator_key": generator.key,
            "generator_label": generator.label,
            "generator_model_id": generator.model_id,
            "judge_key": judge.key,
            "judge_label": judge.label,
            "judge_model_id": judge.model_id,
            "n_routes": len(arm_routes),
            "initial_pass_all_rate": _rate(
                _all_metrics_pass(route, "initial_scores", thresholds) for route in arm_routes
            ),
            "final_pass_all_rate": _rate(
                _all_metrics_pass(route, "final_scores", thresholds) for route in arm_routes
            ),
            "resolved_by_regeneration_rate": _rate(
                _all_metrics_pass(route, "final_scores", thresholds)
                and not _all_metrics_pass(route, "initial_scores", thresholds)
                for route in arm_routes
            ),
            "avg_attempts_used": _mean(
                float(route["attempts_used"])
                if isinstance(route.get("attempts_used"), (int, float))
                else None
                for route in arm_routes
            ),
        }
        for metric in METRICS:
            initial_values = [_score(route, "initial_scores", metric) for route in arm_routes]
            final_values = [_score(route, "final_scores", metric) for route in arm_routes]
            deltas = [
                None if initial is None or final is None else final - initial
                for initial, final in zip(initial_values, final_values)
            ]
            row[f"avg_initial_{metric}"] = _mean(initial_values)
            row[f"avg_final_{metric}"] = _mean(final_values)
            row[f"avg_delta_{metric}"] = _mean(deltas)
            row[f"initial_pass_rate_{metric}"] = _rate(
                _metric_pass(value, thresholds[metric]) for value in initial_values
            )
            row[f"final_pass_rate_{metric}"] = _rate(
                _metric_pass(value, thresholds[metric]) for value in final_values
            )
            row[f"improved_rate_{metric}"] = _rate(
                delta is not None and delta > 0 for delta in deltas
            )
            row[f"regressed_rate_{metric}"] = _rate(
                delta is not None and delta < 0 for delta in deltas
            )
        rows.append(row)
    return rows


def pair_summary_fieldnames() -> list[str]:
    fields = [
        "arm_key",
        "generator_key",
        "generator_label",
        "generator_model_id",
        "judge_key",
        "judge_label",
        "judge_model_id",
        "n_routes",
        "initial_pass_all_rate",
        "final_pass_all_rate",
        "resolved_by_regeneration_rate",
        "avg_attempts_used",
    ]
    for metric in METRICS:
        fields.extend(
            [
                f"avg_initial_{metric}",
                f"avg_final_{metric}",
                f"avg_delta_{metric}",
                f"initial_pass_rate_{metric}",
                f"final_pass_rate_{metric}",
                f"improved_rate_{metric}",
                f"regressed_rate_{metric}",
            ]
        )
    return fields


def build_role_effect_rows(
    metric_rows: Sequence[dict[str, Any]], role: str
) -> list[dict[str, Any]]:
    key_field = f"{role}_key"
    label_field = f"{role}_label"
    model_field = f"{role}_model_id"
    keys = sorted({str(row[key_field]) for row in metric_rows if row.get(key_field) is not None})
    result: list[dict[str, Any]] = []
    for role_key in keys:
        for metric in METRICS:
            group = [
                row
                for row in metric_rows
                if row.get(key_field) == role_key and row.get("metric") == metric
            ]
            if not group:
                continue
            result.append(
                {
                    key_field: role_key,
                    label_field: group[0].get(label_field),
                    model_field: group[0].get(model_field),
                    "metric": metric,
                    "n_observations": len(group),
                    "avg_initial_score": _mean(
                        _optional_float(row.get("initial_score")) for row in group
                    ),
                    "avg_final_score": _mean(
                        _optional_float(row.get("final_score")) for row in group
                    ),
                    "avg_delta": _mean(_optional_float(row.get("delta")) for row in group),
                    "initial_pass_rate": _rate(bool(row.get("initial_pass")) for row in group),
                    "final_pass_rate": _rate(bool(row.get("final_pass")) for row in group),
                    "improved_rate": _rate(bool(row.get("improved")) for row in group),
                    "regressed_rate": _rate(bool(row.get("regressed")) for row in group),
                }
            )
    return result


def role_effect_fieldnames(role: str) -> list[str]:
    return [
        f"{role}_key",
        f"{role}_label",
        f"{role}_model_id",
        "metric",
        "n_observations",
        "avg_initial_score",
        "avg_final_score",
        "avg_delta",
        "initial_pass_rate",
        "final_pass_rate",
        "improved_rate",
        "regressed_rate",
    ]


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def build_judge_disagreement_rows(metric_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        key = (
            str(row.get("generator_key")),
            str(row.get("case_id")),
            str(row.get("route_name")),
            str(row.get("metric")),
        )
        grouped.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for (generator_key, case_id, route_name, metric), group in sorted(grouped.items()):
        initial = [
            float(row["initial_score"])
            for row in group
            if isinstance(row.get("initial_score"), (int, float))
        ]
        final = [
            float(row["final_score"])
            for row in group
            if isinstance(row.get("final_score"), (int, float))
        ]
        result.append(
            {
                "generator_key": generator_key,
                "generator_label": group[0].get("generator_label"),
                "case_id": case_id,
                "route_name": route_name,
                "metric": metric,
                "n_judges": len(group),
                "judge_keys": "|".join(sorted(str(row.get("judge_key")) for row in group)),
                "initial_mean": _mean(initial),
                "initial_min": min(initial) if initial else None,
                "initial_max": max(initial) if initial else None,
                "initial_range": max(initial) - min(initial) if initial else None,
                "initial_population_sd": statistics.pstdev(initial)
                if len(initial) > 1
                else 0.0
                if initial
                else None,
                "final_mean": _mean(final),
                "final_min": min(final) if final else None,
                "final_max": max(final) if final else None,
                "final_range": max(final) - min(final) if final else None,
                "final_population_sd": statistics.pstdev(final)
                if len(final) > 1
                else 0.0
                if final
                else None,
            }
        )
    return result


def disagreement_fieldnames() -> list[str]:
    return [
        "generator_key",
        "generator_label",
        "case_id",
        "route_name",
        "metric",
        "n_judges",
        "judge_keys",
        "initial_mean",
        "initial_min",
        "initial_max",
        "initial_range",
        "initial_population_sd",
        "final_mean",
        "final_min",
        "final_max",
        "final_range",
        "final_population_sd",
    ]


def validate_design(
    *,
    selected_root: Path,
    out_root: Path,
    arms: Sequence[tuple[GeneratorSpec, JudgeSpec]],
    max_routes: int | None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    arm_plans: list[dict[str, Any]] = []
    arm_table_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    total = 0
    for generator, judge in arms:
        arm_dir = out_root / "arms" / _arm_key(generator, judge)
        snapshot = prepare_input_snapshot(
            source_routes_dir=selected_root / generator.routes_subdir,
            snapshot_root=arm_dir / "input_snapshot",
            expected_routes=generator.expected_routes,
            max_routes=max_routes,
        )
        write_json(arm_dir / "source_manifest.json", snapshot)
        arm_plan = {
            "arm_key": _arm_key(generator, judge),
            "generator": asdict(generator),
            "judge": asdict(judge),
            "route_count": len(snapshot),
            "status": "validated_not_run",
        }
        write_json(arm_dir / "arm_plan.json", arm_plan)
        arm_plans.append(arm_plan)
        arm_table_rows.append(
            {
                "arm_key": arm_plan["arm_key"],
                "generator_key": generator.key,
                "generator_label": generator.label,
                "generator_model_id": generator.model_id,
                "judge_key": judge.key,
                "judge_label": judge.label,
                "judge_model_id": judge.model_id,
                "route_count": len(snapshot),
                "status": "validated_not_run",
            }
        )
        for source_row in snapshot:
            input_rows.append(
                {
                    "arm_key": arm_plan["arm_key"],
                    "generator_key": generator.key,
                    "generator_label": generator.label,
                    "generator_model_id": generator.model_id,
                    "judge_key": judge.key,
                    "judge_label": judge.label,
                    "judge_model_id": judge.model_id,
                    **source_row,
                }
            )
        total += len(snapshot)

    out_root.joinpath("combined").mkdir(parents=True, exist_ok=True)
    out_root.joinpath("plots").mkdir(parents=True, exist_ok=True)
    write_csv(
        out_root / "tables" / "experiment_arms.csv",
        arm_table_rows,
        [
            "arm_key",
            "generator_key",
            "generator_label",
            "generator_model_id",
            "judge_key",
            "judge_label",
            "judge_model_id",
            "route_count",
            "status",
        ],
    )
    write_csv(
        out_root / "tables" / "input_routes.csv",
        input_rows,
        [
            "arm_key",
            "generator_key",
            "generator_label",
            "generator_model_id",
            "judge_key",
            "judge_label",
            "judge_model_id",
            "case_id",
            "route_name",
            "source_route_path",
            "source_retrieval_context_path",
            "snapshot_route_path",
            "composition",
            "composition_source",
            "method",
            "method_source",
            "retrieval_context_count",
        ],
    )

    validation = {
        "status": "validated_not_run",
        "ts_utc": utc_now_iso(),
        "total_arms": len(arm_plans),
        "total_route_trajectories": total,
        "arm_plans": arm_plans,
        "design_tables": ["tables/experiment_arms.csv", "tables/input_routes.csv"],
    }
    manifest["validation"] = validation
    write_json(out_root / "experiment_manifest.json", manifest)
    return validation


def reuse_compatible_results(
    *,
    source_root: Path,
    destination_root: Path,
    arms: Sequence[tuple[GeneratorSpec, JudgeSpec]],
    experiment_run_id: str,
) -> dict[str, Any]:
    """Reuse only completed, error-free trajectories from model-compatible arms.

    Reuse requirements:
    - generator and judge model IDs must match the new design exactly;
    - a route must have a completed per-route summary;
    - its latest completed attempt sequence must be 1..attempts_used;
    - every metric in every reused attempt must have a numeric score.

    Duplicate attempt rows left by route-level restarts are normalized by taking
    the last complete sequence represented by the per-route summary.
    """

    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    if source_root == destination_root:
        raise ValueError("--reuse-compatible-from must point to a different result directory.")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Reuse source does not exist: {source_root}")

    reuse_manifest_path = destination_root / "reuse_manifest.json"
    if reuse_manifest_path.exists():
        previous = read_json(reuse_manifest_path)
        if previous.get("source_root") != str(source_root):
            raise ValueError(
                "The destination was already prepared from a different reuse source. "
                "Choose another --out-dir."
            )
        return previous

    report: dict[str, Any] = {
        "source_root": str(source_root),
        "destination_root": str(destination_root),
        "created_utc": utc_now_iso(),
        "policy": (
            "exact generator/judge model match; completed route; latest complete attempt "
            "sequence; no metric score=None"
        ),
        "arms": [],
        "total_reused_routes": 0,
        "total_excluded_routes": 0,
    }

    for generator, judge in arms:
        arm_key = _arm_key(generator, judge)
        source_arm = source_root / "arms" / arm_key
        destination_arm = destination_root / "arms" / arm_key
        arm_report: dict[str, Any] = {
            "arm_key": arm_key,
            "generator_model_id": generator.model_id,
            "judge_model_id": judge.model_id,
            "status": "not_available",
            "reused_routes": [],
            "excluded_routes": [],
        }

        source_arm_config_path = source_arm / "arm_config.json"
        source_attempts_path = source_arm / "attempts.jsonl"
        source_summaries_path = source_arm / "per_route_summary.jsonl"
        if not (
            source_arm_config_path.exists()
            and source_attempts_path.exists()
            and source_summaries_path.exists()
        ):
            report["arms"].append(arm_report)
            continue

        source_arm_config = read_json(source_arm_config_path)
        source_generator_id = str((source_arm_config.get("generator") or {}).get("model_id") or "")
        source_judge_id = str((source_arm_config.get("judge") or {}).get("model_id") or "")
        if source_generator_id != generator.model_id or source_judge_id != judge.model_id:
            arm_report["status"] = "incompatible_model_ids"
            arm_report["source_generator_model_id"] = source_generator_id
            arm_report["source_judge_model_id"] = source_judge_id
            report["arms"].append(arm_report)
            continue

        source_attempts = read_jsonl(source_attempts_path)
        source_summaries = read_jsonl(source_summaries_path)
        attempts_by_route: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for attempt in source_attempts:
            key = (str(attempt.get("case_id")), str(attempt.get("route_name")))
            attempts_by_route.setdefault(key, []).append(attempt)

        selected_attempts: list[dict[str, Any]] = []
        selected_summaries: list[dict[str, Any]] = []
        selected_case_ids: set[str] = set()
        new_arm_run_id = f"{experiment_run_id}:{arm_key}"

        for summary_row in source_summaries:
            case_id = str(summary_row.get("case_id"))
            route_name = str(summary_row.get("route_name"))
            key = (case_id, route_name)
            try:
                attempts_used = int(summary_row.get("attempts_used"))
            except (TypeError, ValueError):
                attempts_used = 0
            all_route_attempts = attempts_by_route.get(key, [])
            candidate_attempts = all_route_attempts[-attempts_used:] if attempts_used > 0 else []
            expected_numbers = list(range(1, attempts_used + 1))
            actual_numbers = [int(row.get("attempt_no", -1)) for row in candidate_attempts]

            exclusion_reason: str | None = None
            if not candidate_attempts or actual_numbers != expected_numbers:
                exclusion_reason = (
                    f"no complete latest attempt sequence: expected={expected_numbers}, "
                    f"actual={actual_numbers}"
                )
            else:
                metric_errors = [
                    {
                        "attempt_no": attempt.get("attempt_no"),
                        "metric": metric,
                    }
                    for attempt in candidate_attempts
                    for metric in METRICS
                    if not isinstance((attempt.get(metric) or {}).get("score"), (int, float))
                ]
                if metric_errors:
                    exclusion_reason = f"technical metric errors: {metric_errors}"

            if exclusion_reason:
                arm_report["excluded_routes"].append(
                    {
                        "case_id": case_id,
                        "route_name": route_name,
                        "reason": exclusion_reason,
                    }
                )
                continue

            rewritten_attempts: list[dict[str, Any]] = []
            source_prefix = str(source_arm.resolve())
            destination_prefix = str(destination_arm.resolve())
            for source_attempt in candidate_attempts:
                attempt = dict(source_attempt)
                attempt["reused_from_run_id"] = source_attempt.get("run_id")
                attempt["run_id"] = new_arm_run_id
                route_json_path = str(attempt.get("route_json_path") or "")
                if route_json_path.startswith(source_prefix):
                    attempt["route_json_path"] = (
                        destination_prefix + route_json_path[len(source_prefix) :]
                    )
                rewritten_attempts.append(attempt)
            selected_attempts.extend(rewritten_attempts)

            summary = dict(summary_row)
            summary["run_id"] = new_arm_run_id
            selected_summaries.append(summary)
            selected_case_ids.add(case_id)
            arm_report["reused_routes"].append(
                {
                    "case_id": case_id,
                    "route_name": route_name,
                    "attempts_used": attempts_used,
                }
            )

        if selected_summaries:
            destination_arm.mkdir(parents=True, exist_ok=True)
            write_jsonl(destination_arm / "attempts.jsonl", selected_attempts)
            write_jsonl(destination_arm / "per_route_summary.jsonl", selected_summaries)
            for case_id in sorted(selected_case_ids):
                source_case_artifacts = source_arm / "regenerated" / case_id
                if source_case_artifacts.is_dir():
                    shutil.copytree(
                        source_case_artifacts,
                        destination_arm / "regenerated" / case_id,
                        dirs_exist_ok=True,
                    )
            arm_report["status"] = "partially_or_fully_reused"
        else:
            arm_report["status"] = "no_valid_routes_to_reuse"

        report["total_reused_routes"] += len(arm_report["reused_routes"])
        report["total_excluded_routes"] += len(arm_report["excluded_routes"])
        report["arms"].append(arm_report)

    write_json(reuse_manifest_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-judge feedback regeneration with existing CPC/Answer Relevancy and "
            "expert-style synthesis Faithfulness."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--selected-routes-root", type=Path, default=DEFAULT_SELECTED_ROUTES_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only-generator", action="append", default=[])
    parser.add_argument("--only-judge", action="append", default=[])
    parser.add_argument(
        "--include-self-judge",
        action="store_true",
        help="Also evaluate a generator with the same model; disabled by default.",
    )
    parser.add_argument("--max-regens", type=int, default=None)
    parser.add_argument("--chemical-threshold", type=float, default=None)
    parser.add_argument("--answer-threshold", type=float, default=None)
    parser.add_argument("--faithfulness-threshold", type=float, default=None)
    parser.add_argument("--rpm-limit", type=float, default=None)
    parser.add_argument("--stop-on-rate-limit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-compatible-from",
        type=Path,
        default=None,
        help=(
            "For a new result directory, reuse only completed error-free routes from arms "
            "whose generator and judge model IDs exactly match the current design. Requires --resume."
        ),
    )
    parser.add_argument(
        "--max-routes-per-arm",
        type=int,
        default=None,
        help="Debug-only deterministic prefix; omit for the required 10 routes per generator.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate/copy inputs and create the result layout without any model/API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_compatible_from is not None and not args.resume:
        raise ValueError("--reuse-compatible-from requires --resume.")
    config = load_config(args.config.resolve())
    generators, judges = parse_specs(config)
    generators = select_specs(generators, args.only_generator, "generator")
    judges = select_specs(judges, args.only_judge, "judge")

    include_self = bool(args.include_self_judge or config.get("include_self_judge", False))
    arms = build_arms(generators, judges, include_self_judge=include_self)
    if not arms:
        raise ValueError("The selected model matrix contains no generator/judge arms.")

    config_thresholds = config.get("thresholds") or {}
    thresholds = {
        "chemical_geval": float(
            args.chemical_threshold
            if args.chemical_threshold is not None
            else config_thresholds.get("chemical_geval", 0.8)
        ),
        "answer_relevancy": float(
            args.answer_threshold
            if args.answer_threshold is not None
            else config_thresholds.get("answer_relevancy", 0.8)
        ),
        "faithfulness": float(
            args.faithfulness_threshold
            if args.faithfulness_threshold is not None
            else config_thresholds.get("faithfulness", 0.8)
        ),
    }
    max_regens = int(
        args.max_regens if args.max_regens is not None else config.get("max_regens", 3)
    )
    if max_regens < 0:
        raise ValueError("max_regens must be >= 0")

    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    existing_attempt_files = sorted(out_root.glob("arms/*/attempts.jsonl"))
    if existing_attempt_files and args.validate_only:
        raise FileExistsError(
            f"{out_root} already contains model results. Validation was not allowed to overwrite "
            "the completed/partial experiment manifest; use a new --out-dir."
        )
    if existing_attempt_files and not args.resume:
        raise FileExistsError(
            f"{out_root} already contains model results. Use --resume or a new --out-dir; "
            "the existing experiment was not modified."
        )

    existing_manifest_path = out_root / "experiment_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if args.resume and existing_manifest_path.exists():
        existing_manifest = read_json(existing_manifest_path)
    experiment_run_id = str(existing_manifest.get("experiment_run_id") or uuid.uuid4().hex[:12])
    manifest: dict[str, Any] = {
        "experiment_run_id": experiment_run_id,
        "created_utc": utc_now_iso(),
        "status": "validated_not_run" if args.validate_only else "running",
        "config_path": str(args.config.resolve()),
        "selected_routes_root": str(args.selected_routes_root.resolve()),
        "out_dir": str(out_root),
        "gemini_2_0_excluded": True,
        "include_self_judge": include_self,
        "generators": [asdict(item) for item in generators],
        "judges": [asdict(item) for item in judges],
        "arms": [
            {
                "arm_key": _arm_key(generator, judge),
                "generator_key": generator.key,
                "judge_key": judge.key,
            }
            for generator, judge in arms
        ],
        "thresholds": thresholds,
        "max_regens": max_regens,
        "max_attempts_total": 1 + max_regens,
        "max_routes_per_arm": args.max_routes_per_arm,
        "metric_contract": {
            "chemical_geval": "unchanged existing CPC/Chemical G-Eval",
            "answer_relevancy": "unchanged existing DeepEval Answer Relevancy",
            "faithfulness": {
                "implementation": "bench/new_faith_same_as_expert_logic.py::measure_expert_faithfulness",
                "formula": "(3*Q1 + 2*Q2 + 2*Q3 + Q4) / 8",
                "within_q": "yes / (yes + no); idk excluded",
            },
        },
    }
    if args.resume and existing_manifest:
        comparable_keys = (
            "selected_routes_root",
            "include_self_judge",
            "generators",
            "judges",
            "arms",
            "thresholds",
            "max_regens",
            "max_routes_per_arm",
        )
        mismatches = [
            key for key in comparable_keys if existing_manifest.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"Cannot resume because the experiment design changed: {mismatches}. "
                "Use a new --out-dir for the new design."
            )
    write_json(out_root / "experiment_manifest.json", manifest)

    if args.reuse_compatible_from is not None:
        reuse_report = reuse_compatible_results(
            source_root=args.reuse_compatible_from,
            destination_root=out_root,
            arms=arms,
            experiment_run_id=experiment_run_id,
        )
        manifest["reuse_report"] = reuse_report
        write_json(out_root / "experiment_manifest.json", manifest)

    if args.validate_only:
        validation = validate_design(
            selected_root=args.selected_routes_root.resolve(),
            out_root=out_root,
            arms=arms,
            max_routes=args.max_routes_per_arm,
            manifest=manifest,
        )
        print(
            f"Validated {validation['total_arms']} arms / "
            f"{validation['total_route_trajectories']} route trajectories. No model calls were made."
        )
        print(f"Manifest: {out_root / 'experiment_manifest.json'}")
        return

    arm_summaries: list[dict[str, Any]] = []
    try:
        for index, (generator, judge) in enumerate(arms, start=1):
            print(f"[{index}/{len(arms)}] generator={generator.label} | judge={judge.label}")
            arm_summaries.append(
                run_arm(
                    experiment_run_id=experiment_run_id,
                    generator=generator,
                    judge=judge,
                    selected_root=args.selected_routes_root.resolve(),
                    out_root=out_root,
                    thresholds=thresholds,
                    max_regens=max_regens,
                    rpm_limit=args.rpm_limit,
                    stop_on_rate_limit=bool(args.stop_on_rate_limit),
                    resume=bool(args.resume),
                    max_routes=args.max_routes_per_arm,
                )
            )

        combined = collect_combined_results(
            out_root=out_root,
            arms=arms,
            thresholds=thresholds,
        )
        manifest["status"] = "completed"
        manifest["completed_utc"] = utc_now_iso()
        manifest["arm_summaries"] = arm_summaries
        manifest["combined_summary"] = combined
        write_json(out_root / "experiment_manifest.json", manifest)
        write_json(out_root / "experiment_summary.json", combined)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_utc"] = utc_now_iso()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out_root / "experiment_manifest.json", manifest)
        raise

    print("Experiment complete.")
    print(f"Summary: {out_root / 'experiment_summary.json'}")
    print(f"Tables: {out_root / 'tables'}")


if __name__ == "__main__":
    main()
