"""Evaluation utilities for structured-output experiments."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from jsonschema import ValidationError, validate

from src.evaluation.error_types import ERROR_TYPES


@dataclass
class SampleEvaluation:
    sample_id: str
    valid_json: bool
    schema_compliant: bool
    field_exact_match: float
    exact_match: bool
    primary_error: str | None


def try_parse_prediction_text(prediction_text: str) -> tuple[bool, dict[str, Any] | None]:
    try:
        parsed = json.loads(prediction_text)
    except json.JSONDecodeError:
        return False, None
    if not isinstance(parsed, dict):
        return False, None
    return True, parsed


def validate_schema(instance: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        validate(instance=instance, schema=schema)
    except ValidationError as exc:
        return False, map_validation_error(exc)
    return True, None


def map_validation_error(error: ValidationError) -> str:
    validator = error.validator
    if validator == "required":
        return "missing_required_field"
    if validator == "enum":
        return "enum_error"
    if validator == "type":
        return "type_error"
    return "schema_violation"


def flatten_structure(data: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(flatten_structure(value, next_prefix))
        return flattened
    if isinstance(data, list):
        for index, value in enumerate(data):
            next_prefix = f"{prefix}[{index}]"
            flattened.update(flatten_structure(value, next_prefix))
        return flattened
    flattened[prefix] = data
    return flattened


def compute_field_exact_match(
    prediction: dict[str, Any], target: dict[str, Any]
) -> float:
    prediction_flat = flatten_structure(prediction)
    target_flat = flatten_structure(target)

    if not target_flat:
        return 1.0

    matched = sum(
        1 for key, target_value in target_flat.items() if prediction_flat.get(key) == target_value
    )
    return matched / len(target_flat)


def classify_semantic_error(
    prediction: dict[str, Any], target: dict[str, Any]
) -> str:
    prediction_flat = flatten_structure(prediction)
    target_flat = flatten_structure(target)

    missing_keys = [key for key in target_flat if key not in prediction_flat]
    if missing_keys:
        return "extraction_omission"

    mismatched_values = [
        key for key, target_value in target_flat.items() if prediction_flat.get(key) != target_value
    ]
    if mismatched_values:
        return "value_hallucination"

    return "cross_field_inconsistency"


def evaluate_sample(
    sample_id: str,
    prediction_text: str | None,
    prediction_json: dict[str, Any] | None,
    target_json: dict[str, Any],
    schema: dict[str, Any],
) -> SampleEvaluation:
    parsed_from_text = False
    parsed_prediction = prediction_json

    if parsed_prediction is None and prediction_text is not None:
        parsed_from_text, parsed_prediction = try_parse_prediction_text(prediction_text)
        if not parsed_from_text:
            return SampleEvaluation(
                sample_id=sample_id,
                valid_json=False,
                schema_compliant=False,
                field_exact_match=0.0,
                exact_match=False,
                primary_error="invalid_json",
            )

    if not isinstance(parsed_prediction, dict):
        return SampleEvaluation(
            sample_id=sample_id,
            valid_json=False,
            schema_compliant=False,
            field_exact_match=0.0,
            exact_match=False,
            primary_error="invalid_json",
        )

    schema_ok, schema_error = validate_schema(parsed_prediction, schema)
    field_exact_match = compute_field_exact_match(parsed_prediction, target_json)
    exact_match = parsed_prediction == target_json

    if not schema_ok:
        return SampleEvaluation(
            sample_id=sample_id,
            valid_json=True,
            schema_compliant=False,
            field_exact_match=field_exact_match,
            exact_match=False,
            primary_error=schema_error,
        )

    if exact_match:
        return SampleEvaluation(
            sample_id=sample_id,
            valid_json=True,
            schema_compliant=True,
            field_exact_match=1.0,
            exact_match=True,
            primary_error=None,
        )

    return SampleEvaluation(
        sample_id=sample_id,
        valid_json=True,
        schema_compliant=True,
        field_exact_match=field_exact_match,
        exact_match=False,
        primary_error=classify_semantic_error(parsed_prediction, target_json),
    )


def summarize_results(results: list[SampleEvaluation]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "num_samples": 0,
            "valid_json_rate": 0.0,
            "schema_compliance_rate": 0.0,
            "field_exact_match": 0.0,
            "end_to_end_exact_match": 0.0,
            "error_counts": {error: 0 for error in ERROR_TYPES},
        }

    error_counts = Counter(
        result.primary_error for result in results if result.primary_error is not None
    )
    return {
        "num_samples": total,
        "valid_json_rate": sum(result.valid_json for result in results) / total,
        "schema_compliance_rate": sum(result.schema_compliant for result in results) / total,
        "field_exact_match": sum(result.field_exact_match for result in results) / total,
        "end_to_end_exact_match": sum(result.exact_match for result in results) / total,
        "error_counts": {error: error_counts.get(error, 0) for error in ERROR_TYPES},
    }
