"""Dataset validation utilities."""

from __future__ import annotations

from typing import Any

from jsonschema import ValidationError, validate

from src.data.schema_buckets import SCHEMA_COMPLEXITY_BUCKETS

REQUIRED_SAMPLE_FIELDS = (
    "sample_id",
    "task_name",
    "schema_name",
    "complexity_bucket",
    "input_text",
    "target_json",
    "metadata",
)

REQUIRED_METADATA_FIELDS = ("source_type", "is_synthetic")
ALLOWED_SOURCE_TYPES = {"email", "issue", "notification", "task"}
ALLOWED_SPLITS = {"train", "val", "test"}


def validate_dataset_record(
    record: dict[str, Any],
    schema: dict[str, Any],
    expected_task_name: str,
    expected_schema_name: str,
) -> None:
    missing_fields = [field for field in REQUIRED_SAMPLE_FIELDS if field not in record]
    if missing_fields:
        raise ValueError(
            f"Sample {record.get('sample_id', '<unknown>')} missing fields: {missing_fields}"
        )

    if record["task_name"] != expected_task_name:
        raise ValueError(
            f"Sample {record['sample_id']} has unexpected task_name: {record['task_name']}"
        )
    if record["schema_name"] != expected_schema_name:
        raise ValueError(
            f"Sample {record['sample_id']} has unexpected schema_name: {record['schema_name']}"
        )

    if record["complexity_bucket"] not in SCHEMA_COMPLEXITY_BUCKETS:
        raise ValueError(
            f"Sample {record['sample_id']} has invalid complexity_bucket: "
            f"{record['complexity_bucket']}"
        )

    input_text = record["input_text"]
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError(f"Sample {record['sample_id']} has empty input_text")

    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"Sample {record['sample_id']} metadata must be an object")

    missing_metadata = [field for field in REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing_metadata:
        raise ValueError(
            f"Sample {record['sample_id']} missing metadata fields: {missing_metadata}"
        )

    if metadata["source_type"] not in ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"Sample {record['sample_id']} has invalid source_type: {metadata['source_type']}"
        )
    if not isinstance(metadata["is_synthetic"], bool):
        raise ValueError(
            f"Sample {record['sample_id']} metadata.is_synthetic must be boolean"
        )

    split = metadata.get("split")
    if split is not None and split not in ALLOWED_SPLITS:
        raise ValueError(f"Sample {record['sample_id']} has invalid split: {split}")

    target_json = record["target_json"]
    if not isinstance(target_json, dict):
        raise ValueError(f"Sample {record['sample_id']} target_json must be an object")

    try:
        validate(instance=target_json, schema=schema)
    except ValidationError as exc:
        raise ValueError(
            f"Sample {record['sample_id']} target_json failed schema validation: {exc.message}"
        ) from exc


def ensure_unique_sample_ids(records: list[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for record in records:
        sample_id = record.get("sample_id")
        if sample_id in seen_ids:
            duplicate_ids.add(sample_id)
        else:
            seen_ids.add(sample_id)
    if duplicate_ids:
        raise ValueError(f"Duplicate sample_id values found: {sorted(duplicate_ids)}")
