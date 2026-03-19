"""Filtering and sampling helpers for phase-1 candidate dataset construction."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


VALID_CATEGORIES = {"bug", "feature", "question", "incident", "task"}
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}


def is_candidate_record(record: dict[str, Any]) -> bool:
    input_text = record.get("input_text", "")
    target = record.get("target_json", {})
    reporter = target.get("reporter", {})
    summary = target.get("summary", "")
    affected_systems = target.get("affected_systems", [])
    actions_requested = target.get("actions_requested", [])

    if not isinstance(input_text, str) or len(input_text) < 80 or len(input_text) > 1600:
        return False
    if not isinstance(summary, str) or len(summary) < 10 or len(summary) > 160:
        return False
    if target.get("category") not in VALID_CATEGORIES:
        return False
    if target.get("priority") not in VALID_PRIORITIES:
        return False
    if reporter.get("name") is None:
        return False
    if len(affected_systems) != 1:
        return False
    if len(actions_requested) != 1:
        return False
    return True


def filter_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if is_candidate_record(record)]


def sample_balanced_candidates(
    records: list[dict[str, Any]],
    max_per_source: dict[str, int],
    max_per_category_per_source: int,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        source = record.get("metadata", {}).get("raw_source", "unknown")
        category = record.get("target_json", {}).get("category", "unknown")
        grouped[source][category].append(record)

    rng = random.Random(shuffle_seed)
    sampled: list[dict[str, Any]] = []

    for source, category_groups in grouped.items():
        source_records: list[dict[str, Any]] = []
        for category_records in category_groups.values():
            shuffled = list(category_records)
            rng.shuffle(shuffled)
            source_records.extend(shuffled[:max_per_category_per_source])

        rng.shuffle(source_records)
        source_limit = max_per_source.get(source, len(source_records))
        sampled.extend(source_records[:source_limit])

    rng.shuffle(sampled)
    return sampled


def summarize_candidate_build(
    input_counts: dict[str, int],
    filtered_records: list[dict[str, Any]],
    sampled_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "input_counts": input_counts,
        "num_filtered_records": len(filtered_records),
        "num_sampled_records": len(sampled_records),
        "sampled_source_counts": count_metadata_field(sampled_records, "raw_source"),
        "sampled_complexity_counts": count_field(sampled_records, "complexity_bucket"),
        "sampled_category_counts": count_target_field(sampled_records, "category"),
    }


def count_field(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get(field_name, "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_metadata_field(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("metadata", {}).get(field_name, "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_target_field(records: list[dict[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("target_json", {}).get(field_name, "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts
