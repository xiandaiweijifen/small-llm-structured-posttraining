"""Schema-aware repair helpers for model outputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.evaluation.metrics import validate_schema


def repair_prediction(
    prediction: dict[str, Any] | None,
    schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    if prediction is None or not isinstance(prediction, dict):
        return prediction, False

    repaired = deepcopy(prediction)
    if not isinstance(schema.get("properties"), dict):
        return repaired, False

    changed = apply_object_repairs(repaired, schema)
    is_valid, _ = validate_schema(repaired, schema)
    return repaired, changed and is_valid


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
