"""Canonical error taxonomy for structured-output evaluation."""

ERROR_TYPES = (
    "invalid_json",
    "schema_violation",
    "missing_required_field",
    "enum_error",
    "type_error",
    "value_hallucination",
    "extraction_omission",
    "cross_field_inconsistency",
)
