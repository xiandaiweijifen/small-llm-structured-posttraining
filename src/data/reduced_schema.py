"""Helpers for converting records to the reduced phase-1 schema."""

from __future__ import annotations

from typing import Any


def to_reduced_target(target_json: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": target_json["summary"],
        "category": target_json["category"],
        "priority": target_json["priority"],
        "requires_followup": target_json["requires_followup"],
        "affected_systems": target_json["affected_systems"],
        "actions_requested": target_json["actions_requested"],
        "constraints": target_json["constraints"],
    }


def convert_record_to_reduced_schema(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    updated["metadata"] = dict(record.get("metadata", {}))
    updated["target_json"] = to_reduced_target(record["target_json"])
    updated["schema_name"] = "ticket_schema_v1_reduced"
    return updated


def convert_records_to_reduced_schema(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [convert_record_to_reduced_schema(record) for record in records]
