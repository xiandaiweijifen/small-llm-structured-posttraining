"""Dataset building helpers for phase-1 experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from src.data.io import dump_jsonl
from src.data.validation import ensure_unique_sample_ids, validate_dataset_record


def validate_records(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    task_name: str,
    schema_name: str,
) -> None:
    ensure_unique_sample_ids(records)
    for record in records:
        validate_dataset_record(
            record,
            schema=schema,
            expected_task_name=task_name,
            expected_schema_name=schema_name,
        )


def assign_splits(
    records: list[dict[str, Any]],
    split_config: dict[str, float],
    shuffle_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    preset_records = {"train": [], "val": [], "test": []}
    unassigned_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        split = record["metadata"].get("split")
        if split in preset_records:
            preset_records[split].append(record)
        else:
            unassigned_by_bucket[record["complexity_bucket"]].append(record)

    rng = random.Random(shuffle_seed)
    for bucket_records in unassigned_by_bucket.values():
        rng.shuffle(bucket_records)

    assigned = {
        "train": list(preset_records["train"]),
        "val": list(preset_records["val"]),
        "test": list(preset_records["test"]),
    }

    for bucket_records in unassigned_by_bucket.values():
        total = len(bucket_records)
        train_count = int(total * split_config["train_ratio"])
        val_count = int(total * split_config["val_ratio"])

        for index, record in enumerate(bucket_records):
            copied_record = clone_record(record)
            if index < train_count:
                assigned["train"].append(set_split(copied_record, "train"))
            elif index < train_count + val_count:
                assigned["val"].append(set_split(copied_record, "val"))
            else:
                assigned["test"].append(set_split(copied_record, "test"))

    return assigned


def clone_record(record: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(record)
    cloned["metadata"] = dict(record["metadata"])
    return cloned


def set_split(record: dict[str, Any], split: str) -> dict[str, Any]:
    record["metadata"]["split"] = split
    return record


def build_dataset(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    task_name: str,
    schema_name: str,
    split_config: dict[str, float],
    shuffle_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    validate_records(records, schema=schema, task_name=task_name, schema_name=schema_name)
    return assign_splits(records, split_config=split_config, shuffle_seed=shuffle_seed)


def write_dataset_splits(output_dir: str, split_records: dict[str, list[dict[str, Any]]]) -> None:
    for split_name, records in split_records.items():
        dump_jsonl(f"{output_dir}/phase1_{split_name}.jsonl", records)


def summarize_split_counts(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, records in split_records.items():
        bucket_counts: dict[str, int] = defaultdict(int)
        synthetic_count = 0
        for record in records:
            bucket_counts[record["complexity_bucket"]] += 1
            if record["metadata"]["is_synthetic"]:
                synthetic_count += 1
        summary[split_name] = {
            "num_samples": len(records),
            "synthetic_samples": synthetic_count,
            "bucket_counts": dict(bucket_counts),
        }
    return summary
