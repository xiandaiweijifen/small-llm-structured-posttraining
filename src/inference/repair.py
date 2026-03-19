"""Schema-aware repair helpers for model outputs."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from src.evaluation.metrics import validate_schema


def repair_prediction(
    prediction: dict[str, Any] | None,
    schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    if prediction is None or not isinstance(prediction, dict):
        return prediction, False

    repaired = deepcopy(prediction)
    changed = apply_alias_repairs(repaired, schema)
    if not isinstance(schema.get("properties"), dict):
        return repaired, changed

    changed = apply_object_repairs(repaired, schema) or changed
    is_valid, _ = validate_schema(repaired, schema)
    return repaired, changed and is_valid


def apply_alias_repairs(instance: dict[str, Any], schema: dict[str, Any]) -> bool:
    changed = False

    summary = first_non_empty(
        instance.get("summary"),
        instance.get("subject"),
        instance.get("title"),
        instance.get("short_description"),
        extract_first_sentence(instance.get("description")),
        extract_first_sentence(instance.get("content")),
    )
    if summary is not None and instance.get("summary") != summary:
        instance["summary"] = summary
        changed = True

    category = infer_category(instance)
    if category is not None and instance.get("category") != category:
        instance["category"] = category
        changed = True

    priority = infer_priority(instance)
    if priority is not None and instance.get("priority") != priority:
        instance["priority"] = priority
        changed = True

    requires_followup = infer_requires_followup(instance)
    if requires_followup is not None and instance.get("requires_followup") != requires_followup:
        instance["requires_followup"] = requires_followup
        changed = True

    if "reporter" in schema.get("properties", {}):
        reporter = instance.get("reporter")
        if not isinstance(reporter, dict):
            reporter = {}
        reporter_name = first_non_empty(
            reporter.get("name"),
            instance.get("created_by"),
            instance.get("requester"),
            instance.get("requester_name"),
            instance.get("customer"),
        )
        reporter_team = first_non_empty(
            reporter.get("team"),
            instance.get("department"),
            instance.get("assignment_group"),
            instance.get("team"),
            instance.get("group"),
        )
        next_reporter = {"name": reporter_name, "team": reporter_team}
        if instance.get("reporter") != next_reporter:
            instance["reporter"] = next_reporter
            changed = True

    return changed


def apply_object_repairs(instance: dict[str, Any], schema: dict[str, Any]) -> bool:
    changed = False
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    if schema.get("additionalProperties") is False:
        unknown_keys = [key for key in instance.keys() if key not in properties]
        for key in unknown_keys:
            instance.pop(key, None)
            changed = True

    for field in required_fields:
        if field not in instance and field in properties:
            default_value, can_fill = build_default_value(properties[field])
            if can_fill:
                instance[field] = default_value
                changed = True

    for field_name, field_schema in properties.items():
        if field_name not in instance:
            continue

        field_value = instance[field_name]
        repaired_value, field_changed = repair_value(field_value, field_schema)
        if field_changed:
            instance[field_name] = repaired_value
            changed = True

    return changed


def repair_value(value: Any, schema: dict[str, Any]) -> tuple[Any, bool]:
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        allowed_types = set(schema_type)
        if value is None and "null" in allowed_types:
            return None, False
        non_null_types = [item for item in schema_type if item != "null"]
        if len(non_null_types) == 1:
            repaired_value, changed = coerce_value(value, non_null_types[0], schema)
            return repaired_value, changed
        return value, False

    if schema_type == "object" and isinstance(value, dict):
        nested = deepcopy(value)
        changed = apply_object_repairs(nested, schema)
        return nested, changed

    if schema_type == "array" and isinstance(value, list):
        items_schema = schema.get("items", {})
        repaired_items = []
        changed = False
        for item in value:
            repaired_item, item_changed = repair_value(item, items_schema)
            repaired_items.append(repaired_item)
            changed = changed or item_changed
        return repaired_items, changed

    return coerce_value(value, schema_type, schema)


def coerce_value(value: Any, expected_type: str | None, schema: dict[str, Any]) -> tuple[Any, bool]:
    if "enum" in schema and value in schema["enum"]:
        return value, False

    if expected_type == "string":
        if value is None:
            return value, False
        if isinstance(value, (bool, int, float)):
            return str(value), True
        return value, False

    if expected_type == "boolean":
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True, True
            if lowered == "false":
                return False, True
        return value, False

    if expected_type == "array":
        if value is None:
            return [], True
        return value, False

    if expected_type == "object":
        if value is None:
            return {}, True
        return value, False

    return value, False


def build_default_value(schema: dict[str, Any]) -> tuple[Any, bool]:
    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        if "null" in schema_type:
            return None, True
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "object":
        value: dict[str, Any] = {}
        apply_object_repairs(value, schema)
        is_valid, _ = validate_schema(value, schema)
        return (value, True) if is_valid else (None, False)
    if schema_type == "array":
        return [], True
    if schema_type == "string":
        enum_values = schema.get("enum")
        if isinstance(enum_values, list):
            non_null_values = [value for value in enum_values if value is not None]
            if len(non_null_values) == 1:
                return non_null_values[0], True
            return None, False
        return "", True
    if schema_type == "boolean":
        return False, True

    return None, False


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        elif value is not None:
            return value
    return None


def extract_first_sentence(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return None
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0][:160].strip()


def infer_category(instance: dict[str, Any]) -> str | None:
    explicit = normalize_category(instance.get("category"))
    if explicit is not None:
        return explicit

    category_sources = " ".join(
        str(value)
        for value in [
            instance.get("type"),
            instance.get("status"),
            instance.get("subject"),
            instance.get("title"),
            instance.get("short_description"),
            instance.get("description"),
            instance.get("content"),
        ]
        if value is not None
    ).lower()

    if any(token in category_sources for token in ["feature", "enhancement", "add ", "new capability"]):
        return "feature"
    if any(token in category_sources for token in ["how to", "instruction", "instructions", "where can i", "can you explain", "question"]):
        return "question"
    if any(token in category_sources for token in ["outage", "down", "unavailable", "sev", "critical incident", "incident"]):
        return "incident"
    if any(token in category_sources for token in ["error", "bug", "broken", "failure", "failing", "cannot", "can't", "unable", "issue"]):
        return "bug"
    if "request" in category_sources or "task" in category_sources:
        return "task"
    return "task" if category_sources else None


def normalize_category(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    mapping = {
        "bug": "bug",
        "incident": "incident",
        "feature": "feature",
        "question": "question",
        "task": "task",
        "request": "task",
        "issue": "bug",
    }
    return mapping.get(lowered)


def infer_priority(instance: dict[str, Any]) -> str | None:
    explicit = normalize_priority(instance.get("priority"))
    if explicit is not None:
        return explicit

    urgency = normalize_int(instance.get("urgency"))
    if urgency is not None:
        return map_numeric_priority(urgency)

    text = " ".join(
        str(value)
        for value in [
            instance.get("subject"),
            instance.get("title"),
            instance.get("short_description"),
            instance.get("description"),
            instance.get("content"),
        ]
        if value is not None
    ).lower()

    if any(token in text for token in ["urgent", "asap", "immediately", "blocker", "critical"]):
        return "urgent"
    if any(token in text for token in ["high priority", "soon", "cannot access", "can't access"]):
        return "high"
    if any(token in text for token in ["whenever possible", "low priority", "minor"]):
        return "low"
    return "medium" if text else None


def normalize_priority(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return map_numeric_priority(int(value))
    if not isinstance(value, str):
        return None

    cleaned = value.strip().lower()
    direct = {
        "low": "low",
        "medium": "medium",
        "med": "medium",
        "high": "high",
        "urgent": "urgent",
        "critical": "urgent",
        "p1": "urgent",
        "p2": "high",
        "p3": "medium",
        "p4": "low",
    }
    if cleaned in direct:
        return direct[cleaned]
    if cleaned.isdigit():
        return map_numeric_priority(int(cleaned))
    return None


def map_numeric_priority(value: int) -> str:
    if value >= 5:
        return "urgent"
    if value == 4:
        return "high"
    if value == 3:
        return "medium"
    return "low"


def normalize_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def infer_requires_followup(instance: dict[str, Any]) -> bool | None:
    if isinstance(instance.get("requires_followup"), bool):
        return instance["requires_followup"]

    text = " ".join(
        str(value)
        for value in [
            instance.get("subject"),
            instance.get("title"),
            instance.get("description"),
            instance.get("content"),
            instance.get("status"),
        ]
        if value is not None
    ).lower()
    if not text:
        return None
    if any(token in text for token in ["please", "need", "can you", "follow up", "pending", "waiting"]):
        return True
    return False
