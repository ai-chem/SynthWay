"""Pure result validation and reporting, without GUI or provider dependencies."""

import math

from ss_faithfulness import SS_FAITHFULNESS_ID, validate_ss_record


def metrics_for(record):
    for metric in METRICS:
        if (
            metric[1] == "faithfulness_score"
            and record.get("faithfulness_metric") == SS_FAITHFULNESS_ID
        ):
            yield ("SS-Faithfulness", *metric[1:])
        else:
            yield metric


METRICS = (
    ("Chemical Plausibility & Consistency", "score", "threshold", "passed", "reason"),
    ("Answer Relevancy", "answer_score", "answer_threshold", "answer_passed", "answer_reason"),
    (
        "Original Faithfulness",
        "faithfulness_score",
        "faithfulness_threshold",
        "faithfulness_passed",
        "faithfulness_reason",
    ),
)


def valid_score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def score_text(value, digits=3):
    return f"{value:.{digits}f}" if valid_score(value) else "N/A"


def validate_metric_result(record):
    if record.get("error"):
        raise ValueError("Evaluation returned an error; no quality status is available.")
    if record.get("faithfulness_metric") == SS_FAITHFULNESS_ID:
        validate_ss_record(record)
    for label, score, threshold, passed, _ in metrics_for(record):
        if not valid_score(record.get(score)) or not valid_score(record.get(threshold)):
            raise ValueError(f"{label}: missing or invalid score/threshold; not a zero score.")
        flag = record.get(passed)
        if type(flag) is not bool or flag != (record[score] >= record[threshold]):
            raise ValueError(f"{label}: pass flag is inconsistent with the score and threshold.")


def route_overall_status(record):
    try:
        validate_metric_result(record)
    except ValueError:
        return False, "Evaluation incomplete or invalid"
    passed = all(record[metric[3]] for metric in METRICS)
    return passed, "All three metrics passed" if passed else "One or more metrics failed"


def metric_line(record):
    entries = []
    for label, score, threshold, passed, _ in metrics_for(record):
        numeric = valid_score(record.get(score)) and valid_score(record.get(threshold))
        flag = record.get(passed)
        consistent = numeric and type(flag) is bool and flag == (record[score] >= record[threshold])
        status = (
            ("PASS" if flag else "FAIL")
            if consistent and not record.get("error")
            else "INVALID / MISSING"
        )
        entries.append(f"{label}: {score_text(record.get(score))} ({status})")
    return " | ".join(entries)


def build_run_summary(*, out_dir, results, requested_num_routes):
    if not out_dir or not results:
        return "(no results yet)"
    lines = [
        "SynthWay — run report",
        f"Output: {out_dir}",
        f"Routes requested: {requested_num_routes}",
        f"Routes evaluated: {len(results)}",
        "Scores are model judgements, not experimental or safety validation.",
        "",
    ]
    for index, record in enumerate(results, start=1):
        _, label = route_overall_status(record)
        lines.extend(
            [
                "─" * 56,
                f"Route #{index}: {label}",
                f"File: {record.get('path') or record.get('file') or 'not recorded'}",
                f"composition={record.get('composition')} | method={record.get('method')}",
                f"Generator: {record.get('generator_model') or 'not recorded'}; judge: {record.get('judge_model') or 'not recorded'}",
                f"Attempt: {record.get('attempt', 1)}/{record.get('max_attempts', 1)}",
                metric_line(record),
                "",
                "SYNTHESIS ROUTE:",
                record.get("raw_text") or "(not recorded)",
            ]
        )
        for title, score, threshold, _, reason in metrics_for(record):
            lines.extend(
                [
                    "",
                    f"{title}: {score_text(record.get(score))}; threshold={score_text(record.get(threshold))}",
                    str(record.get(reason) or "(no explanation recorded)"),
                ]
            )
        if not record.get("retrieval_context_found", True):
            lines.append("WARNING: retrieval evidence was not found.")
        if record.get("error"):
            lines.append(f"Evaluation error: {record['error']}")
        lines.append("")
    return "\n".join(lines)
