"""Split helpers for phase-1 candidate datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def assign_stratified_splits(
    records: list[dict[str, Any]],
    split_config: dict[str, float],
    shuffle_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("complexity_bucket", "unknown")),
            str(record.get("metadata", {}).get("raw_source", "unknown")),
        )
        grouped[key].append(record)

    rng = random.Random(shuffle_seed)
    assigned = {"train": [], "val": [], "test": []}

    for group_records in grouped.values():
        shuffled = [clone_record(record) for record in group_records]
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_count = int(total * split_config["train_ratio"])
        val_count = int(total * split_config["val_ratio"])

        for index, record in enumerate(shuffled):
            if index < train_count:
                assigned["train"].append(set_split(record, "train"))
            elif index < train_count + val_count:
                assigned["val"].append(set_split(record, "val"))
            else:
                assigned["test"].append(set_split(record, "test"))

    return assigned


def clone_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    updated["metadata"] = dict(record.get("metadata", {}))
    return updated


def set_split(record: dict[str, Any], split: str) -> dict[str, Any]:
    record["metadata"]["split"] = split
    return record


def summarize_phase1_splits(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, records in split_records.items():
        summary[split_name] = {
            "num_samples": len(records),
            "complexity_counts": count_field(records, "complexity_bucket"),
            "raw_source_counts": count_metadata_field(records, "raw_source"),
            "category_counts": count_target_field(records, "category"),
        }
    return summary


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
