"""Helpers for schema-variant and generalization datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def convert_reduced_record_to_variant(record: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(record)
    updated["schema_name"] = "ticket_schema_v1_reduced_1_1"
    updated["target_json"]["customer_impact"] = None
    metadata = dict(updated.get("metadata", {}))
    metadata["schema_seen_status"] = "unseen"
    metadata["base_schema_name"] = record.get("schema_name")
    updated["metadata"] = metadata
    updated["schema_seen_status"] = "unseen"
    return updated


def mark_record_as_seen(record: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(record)
    metadata = dict(updated.get("metadata", {}))
    metadata["schema_seen_status"] = "seen"
    updated["metadata"] = metadata
    updated["schema_seen_status"] = "seen"
    return updated


def build_seen_unseen_reduced_sets(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen_records = [mark_record_as_seen(record) for record in records]
    unseen_records = [convert_reduced_record_to_variant(record) for record in records]
    return seen_records, unseen_records
