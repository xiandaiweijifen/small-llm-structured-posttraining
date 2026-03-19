"""Dataset profiling helpers for mapped and processed samples."""

from __future__ import annotations

from collections import Counter
from typing import Any


def profile_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_records = len(records)
    complexity_counts = Counter()
    source_type_counts = Counter()
    raw_source_counts = Counter()
    category_counts = Counter()
    priority_counts = Counter()
    environment_counts = Counter()
    blocking_counts = Counter()
    missing_reporter_name = 0
    missing_reporter_team = 0
    empty_affected_systems = 0
    empty_actions_requested = 0
    summary_lengths: list[int] = []
    input_lengths: list[int] = []

    for record in records:
        complexity_counts[record.get("complexity_bucket", "unknown")] += 1

        metadata = record.get("metadata", {})
        source_type_counts[metadata.get("source_type", "unknown")] += 1
        raw_source_counts[metadata.get("raw_source", "unknown")] += 1

        target = record.get("target_json", {})
        category_counts[target.get("category", "unknown")] += 1
        priority_counts[target.get("priority", "unknown")] += 1

        constraints = target.get("constraints", {})
        environment_counts[str(constraints.get("environment"))] += 1
        blocking_counts[str(constraints.get("blocking"))] += 1

        reporter = target.get("reporter", {})
        if reporter.get("name") is None:
            missing_reporter_name += 1
        if reporter.get("team") is None:
            missing_reporter_team += 1

        affected_systems = target.get("affected_systems", [])
        actions_requested = target.get("actions_requested", [])
        if not affected_systems:
            empty_affected_systems += 1
        if not actions_requested:
            empty_actions_requested += 1

        summary = target.get("summary", "")
        input_text = record.get("input_text", "")
        summary_lengths.append(len(summary))
        input_lengths.append(len(input_text))

    return {
        "num_records": num_records,
        "complexity_counts": dict(complexity_counts),
        "source_type_counts": dict(source_type_counts),
        "raw_source_counts": dict(raw_source_counts),
        "category_counts": dict(category_counts),
        "priority_counts": dict(priority_counts),
        "environment_counts": dict(environment_counts),
        "blocking_counts": dict(blocking_counts),
        "missing_reporter_name_rate": ratio(missing_reporter_name, num_records),
        "missing_reporter_team_rate": ratio(missing_reporter_team, num_records),
        "empty_affected_systems_rate": ratio(empty_affected_systems, num_records),
        "empty_actions_requested_rate": ratio(empty_actions_requested, num_records),
        "summary_length": summarize_numeric(summary_lengths),
        "input_length": summarize_numeric(input_lengths),
    }


def summarize_numeric(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def ratio(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total
